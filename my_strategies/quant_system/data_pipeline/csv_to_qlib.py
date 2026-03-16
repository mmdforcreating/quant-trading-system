"""
CSV → Qlib 二进制格式转换器

调用 qlib 仓库自带的 dump_bin.py 脚本，将标准化 CSV 转换为
Qlib 所需的 .bin 文件、instruments 列表和 calendars 日历。

前置条件：
- qlib 源码位于 ~/Quant/qlib
- CSV 文件格式: 每只股票一个文件 {SYMBOL}.csv，包含 date,open,close,high,low,volume,amount 列
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

def _resolve_qlib_repo() -> Path:
    """优先级: 环境变量 QLIB_REPO → ~/Quant/qlib → ~/qlib"""
    import os as _os
    env = _os.environ.get("QLIB_REPO")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
    for candidate in [Path.home() / "Quant" / "qlib", Path.home() / "qlib"]:
        if candidate.exists():
            return candidate
    return Path.home() / "Quant" / "qlib"


QLIB_REPO = _resolve_qlib_repo()
DUMP_SCRIPT = QLIB_REPO / "scripts" / "dump_bin.py"


def csv_to_qlib_bin(
    csv_dir: str | Path,
    qlib_dir: str | Path,
    include_fields: str = "open,close,high,low,volume,amount",
    date_field: str = "date",
    symbol_field: str = "symbol",
    freq: str = "day",
) -> bool:
    """
    将 CSV 目录转换为 Qlib .bin 格式。

    Parameters
    ----------
    csv_dir : str | Path
        包含 {SYMBOL}.csv 文件的目录
    qlib_dir : str | Path
        Qlib 数据输出目录
    include_fields : str
        需要转换的字段列表（逗号分隔）
    date_field : str
        CSV 中日期列名
    symbol_field : str
        CSV 中股票代码列名
    freq : str
        数据频率，day / 1min / 5min

    Returns
    -------
    bool
        成功返回 True
    """
    csv_dir = Path(csv_dir).resolve()
    qlib_dir = Path(qlib_dir).resolve()

    if not DUMP_SCRIPT.exists():
        logger.error("dump_bin.py 不存在: %s", DUMP_SCRIPT)
        return False

    if not csv_dir.exists() or not list(csv_dir.glob("*.csv")):
        logger.error("CSV 目录为空或不存在: %s", csv_dir)
        return False

    qlib_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(DUMP_SCRIPT),
        "dump_all",
        "--data_path", str(csv_dir),
        "--qlib_dir", str(qlib_dir),
        "--include_fields", include_fields,
        "--date_field_name", date_field,
        "--symbol_field_name", symbol_field,
        "--freq", freq,
    ]

    logger.info("执行: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("dump_bin stdout:\n%s", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("dump_bin 失败:\nstdout: %s\nstderr: %s", e.stdout[-1000:], e.stderr[-1000:])
        return False
