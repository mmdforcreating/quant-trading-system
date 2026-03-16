"""
A股 Alpha158 因子扫描与动量排序工具
基于 Qlib Alpha158 因子集，对指定股票进行 WVMA20 动量排序
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)

QLIB_DATA_PATH = os.path.expanduser("~/.qlib/qlib_data/cn_data")

STOCKS = [
    "SH600519",  # 贵州茅台
    "SZ000001",  # 平安银行
    "SH600036",  # 招商银行
    "SZ000858",  # 五粮液
    "SZ002594",  # 比亚迪
]

STOCK_NAMES = {
    "SH600519": "贵州茅台",
    "SZ000001": "平安银行",
    "SH600036": "招商银行",
    "SZ000858": "五粮液",
    "SZ002594": "比亚迪",
}

START_TIME = "2020-06-01"
END_TIME = "2020-09-25"


def check_data_exists(data_path: str) -> bool:
    """检查 Qlib 二进制数据是否存在"""
    p = Path(data_path)
    if not p.exists():
        return False
    instruments_dir = p / "instruments"
    features_dir = p / "features"
    calendars_dir = p / "calendars"
    if not (instruments_dir.exists() and features_dir.exists() and calendars_dir.exists()):
        return False
    feature_subdirs = list(features_dir.iterdir())
    return len(feature_subdirs) > 0


def check_openmp() -> str:
    """检测 OpenMP 多核并行支持状态"""
    omp_num = os.environ.get("OMP_NUM_THREADS")
    cpu_count = os.cpu_count() or 1
    if omp_num:
        return "已设置 OMP_NUM_THREADS=%s (CPU 核心数: %d)" % (omp_num, cpu_count)
    return "未设置 OMP_NUM_THREADS, 默认使用全部 %d 核心" % cpu_count


def main():
    print("=" * 60)
    print("  A股 Alpha158 因子扫描器 (Qlib)")
    print("=" * 60)
    print()

    # 1. OpenMP 检测
    omp_status = check_openmp()
    print("[环境] %s" % omp_status)
    print()

    # 2. 数据检测
    print("[数据] 检查路径: %s" % QLIB_DATA_PATH)
    if not check_data_exists(QLIB_DATA_PATH):
        print()
        print("  !! 未找到 A 股二进制数据 !!")
        print("  请运行以下命令下载 cn_data:")
        print()
        print("    cd ~/Quant/qlib && python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn")
        print()
        sys.exit(1)
    print("[数据] A 股二进制数据已就绪")
    print()

    # 3. 初始化 Qlib
    print("[初始化] 正在初始化 Qlib ...")
    import qlib
    from qlib.config import REG_CN

    qlib.init(provider_uri=QLIB_DATA_PATH, region=REG_CN)
    print("[初始化] Qlib 初始化完成")
    print()

    # 4. 计算 Alpha158 因子
    print("[因子] 正在计算 Alpha158 因子 ...")
    print("  股票: %s" % ", ".join(STOCKS))
    print("  时间: %s ~ %s" % (START_TIME, END_TIME))
    print()

    from qlib.contrib.data.handler import Alpha158

    t0 = time.time()
    handler = Alpha158(
        instruments=STOCKS,  # type: ignore[arg-type]
        start_time=START_TIME,
        end_time=END_TIME,
        infer_processors=[
            {"class": "ProcessInf", "kwargs": {}},
            {"class": "Fillna", "kwargs": {}},
        ],
    )
    df = handler.fetch()
    elapsed = time.time() - t0

    print("[因子] 计算完成, 耗时 %.2f 秒" % elapsed)
    print("[因子] 数据形状: %s" % str(df.shape))

    cpu_count = os.cpu_count() or 1
    if cpu_count > 1 and elapsed < 30:
        print("[并行] M4 Mac 多核加速可能已生效 (%d 核, 耗时 %.2fs)" % (cpu_count, elapsed))
    print()

    # 5. 提取 WVMA20 并排序
    wvma_cols = [c for c in df.columns if "WVMA" in c and "20" in c]
    if not wvma_cols:
        wvma_cols = [c for c in df.columns if "WVMA" in c]
    if not wvma_cols:
        print("[警告] 未找到 WVMA 相关因子，可用列:")
        print("  %s" % ", ".join(df.columns[:20].tolist()))
        sys.exit(1)

    target_col = wvma_cols[0]
    print("[排序] 使用因子: %s" % target_col)

    latest_data = df.groupby(level="instrument").tail(1).copy()
    latest_data = latest_data.reset_index()

    latest_data["stock_name"] = latest_data["instrument"].map(STOCK_NAMES)  # type: ignore[arg-type]
    latest_data = latest_data.sort_values(target_col, ascending=False)
    latest_data["rank"] = range(1, len(latest_data) + 1)

    # 6. 输出结果
    print()
    print("=" * 60)
    print("  动量排序结果 (按 %s 从大到小)" % target_col)
    print("=" * 60)

    result = latest_data[["rank", "instrument", "stock_name", "datetime", target_col]].copy()
    result.columns = ["优先级", "代码", "名称", "日期", target_col]
    result = result.reset_index(drop=True)

    pd.set_option("display.unicode.ambiguous_as_wide", True)
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.float_format", lambda x: "%.6f" % x)

    print()
    print(result.to_string(index=False))
    print()

    print("[提示] WVMA20 = 成交量加权价格变化波动率 (20日窗口)")
    print("       值越大，表示近期成交量加权的价格波动越剧烈")
    print()

    # 7. 显示所有因子概览
    print("=" * 60)
    print("  Alpha158 因子概览 (最新一天)")
    print("=" * 60)
    summary = latest_data.set_index("instrument")[df.columns].describe()
    print()
    print(summary.to_string())
    print()


if __name__ == "__main__":
    main()
