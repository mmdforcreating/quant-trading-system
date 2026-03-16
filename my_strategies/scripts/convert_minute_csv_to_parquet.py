#!/usr/bin/env python3
"""
将「股票分钟数据」目录下的按日 CSV 转为 GRU 所需的按股 Parquet。

输入: 股票分钟数据/2026-02/*.csv, 2026-03/*.csv（列名中文，每日每股一个 CSV）
输出: data/minute/1min/{六位代码}.parquet（列名英文，每股多日合并）

用法（在 my_strategies 或项目根目录）:
  python scripts/convert_minute_csv_to_parquet.py
  python scripts/convert_minute_csv_to_parquet.py --input /path/to/股票分钟数据 --output /path/to/data/minute
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_minute")

COL_MAP = {
    "时间": "datetime",
    "开盘价": "open",
    "收盘价": "close",
    "最高价": "high",
    "最低价": "low",
    "成交量": "volume",
    "成交额": "amount",
    "代码": "symbol",
}


def csv_code_from_path(path: Path) -> str | None:
    """从文件名提取 6 位代码，如 sh600000.csv -> 600000, sz000001.csv -> 000001"""
    name = path.stem.lower()
    if name.startswith("sh") and len(name) == 8:
        return name[2:]  # 600000
    if name.startswith("sz") and len(name) == 8:
        return name[2:]  # 000001
    return None


def main():
    parser = argparse.ArgumentParser(description="股票分钟 CSV 转 GRU 用 Parquet")
    parser.add_argument(
        "--input",
        type=str,
        default="/Users/mengmingdong/股票分钟数据",
        help="股票分钟数据根目录（含 2026-02、2026-03 等子目录）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出根目录，默认 my_strategies/data/minute",
    )
    args = parser.parse_args()

    input_root = Path(args.input).expanduser()
    if not input_root.exists():
        logger.error("输入目录不存在: %s", input_root)
        return 1

    # 默认输出到 my_strategies/data/minute
    if args.output:
        output_root = Path(args.output).expanduser()
    else:
        script_dir = Path(__file__).resolve().parent
        base = script_dir.parent  # my_strategies
        output_root = base / "data" / "minute"
    out_1min = output_root / "1min"
    out_1min.mkdir(parents=True, exist_ok=True)
    logger.info("输入: %s  输出: %s", input_root, out_1min)

    # 收集所有 CSV：按 (子目录, 文件名) 分组得到 (code -> [csv paths])
    code_to_files: dict[str, list[Path]] = {}
    for sub in sorted(input_root.iterdir()):
        if not sub.is_dir():
            continue
        for f in sub.glob("*.csv"):
            code = csv_code_from_path(f)
            if code is None:
                continue
            code_to_files.setdefault(code, []).append(f)

    logger.info("共 %d 只股票", len(code_to_files))

    required_cols = ["时间", "开盘价", "收盘价", "最高价", "最低价", "成交量", "成交额"]
    written = 0
    for code, files in sorted(code_to_files.items()):
        dfs = []
        for p in sorted(files):
            try:
                df = pd.read_csv(p, encoding="utf-8")
            except Exception as e:
                try:
                    df = pd.read_csv(p, encoding="gbk")
                except Exception as e2:
                    logger.warning("跳过 %s: %s", p.name, e2)
                    continue
            # 列名兼容
            if "时间" not in df.columns and "datetime" in df.columns:
                dfs.append(df)
                continue
            if not all(c in df.columns for c in required_cols):
                logger.warning("缺少列 %s: %s", p.name, list(df.columns))
                continue
            df = df.rename(columns=COL_MAP)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["symbol"] = code
            df = df[["datetime", "open", "close", "high", "low", "volume", "amount", "symbol"]]
            dfs.append(df)
        if not dfs:
            continue
        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)
        out_path = out_1min / f"{code}.parquet"
        merged.to_parquet(out_path, index=False)
        written += 1
        if written <= 3 or written % 500 == 0:
            logger.info("已写 %s: %d 行", out_path.name, len(merged))

    logger.info("完成: 共写入 %d 个 Parquet 到 %s", written, out_1min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
