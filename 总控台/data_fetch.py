#!/usr/bin/env python3
"""
数据原料车间 (data_fetch)

职责：彻底分离低频与高频数据，各司其职。

日频分支 (--freq daily):
  1. TuShare 拉取 daily/daily_basic/adj_factor/moneyflow/fina_indicator
  2. 合并为 panel.parquet
  3. 可选：导出 per-stock CSV + 转 Qlib .bin（为 Alpha158 准备）

分钟频分支 (--freq minute):
  1. akshare 拉取 5min（东方财富 + 新浪兜底，断点续拉）
  2. 可选拉取 1min
  3. 存入 data/minute/{period}min/*.parquet
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent / "my_strategies"
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

logger = logging.getLogger("data_fetch")


# ── 日频 ─────────────────────────────────────────────

def fetch_daily(cfg: dict, skip_fina: bool = False, export_qlib: bool = False):
    """
    日频数据全流程：股票池 → TuShare 拉取 → 合并 panel.parquet。

    Parameters
    ----------
    cfg : dict        全局配置（ConfigManager 实例）
    skip_fina : bool  是否跳过 fina_indicator（较慢）
    export_qlib : bool 是否额外导出 CSV + 转 Qlib .bin
    """
    from quant_system.data_pipeline.stock_pool import get_stock_pool
    from quant_system.data_pipeline.tushare_fetcher import TuShareFetcher
    from quant_system.data_pipeline.panel_builder import build_panel

    dp = cfg.get("data_pipeline", {})
    start_date = dp.get("start_date", "20200101")
    end_date = dp.get("end_date") or datetime.now().strftime("%Y%m%d")
    panel_path = str(Path(dp.get("panel_path", "data/panel.parquet")).expanduser())

    ts_codes, name_map = get_stock_pool(cfg)
    logger.info("股票池: %d 只", len(ts_codes))

    fetcher = TuShareFetcher(cfg)
    raw_data = fetcher.fetch_all(
        ts_codes=ts_codes,
        start_date=start_date,
        end_date=end_date,
        skip_fina=skip_fina,
    )
    for name, df in raw_data.items():
        logger.info("  %s: %d 行", name, len(df))

    panel = build_panel(raw_data, panel_path, start_date)
    logger.info("Panel 构建完成: %s (%d 行, %d 列)", panel_path, len(panel), panel.shape[1])

    if export_qlib:
        _export_to_qlib(panel, cfg)

    return panel


def _export_to_qlib(panel: pd.DataFrame, cfg: dict):
    """导出 per-stock CSV 并转 Qlib .bin（为 Alpha158 准备）。"""
    try:
        from quant_system.data_pipeline.qlib_alpha158 import _export_per_stock_csv
        from quant_system.data_pipeline.csv_to_qlib import csv_to_qlib_bin

        dp = cfg.get("data_pipeline", {})
        csv_storage = str(Path(dp.get("csv_storage", "~/.qlib/csv_data")).expanduser())
        qlib_dir = str(Path(dp.get("qlib_dir", "~/.qlib/qlib_data/cn_data")).expanduser())

        _export_per_stock_csv(panel, csv_storage)
        ok = csv_to_qlib_bin(csv_storage, qlib_dir)
        if ok:
            logger.info("Qlib .bin 导出完成: %s", qlib_dir)
        else:
            logger.warning("Qlib .bin 导出失败")
    except Exception as e:
        logger.warning("Qlib 导出失败: %s", e)


# ── 分钟频 ───────────────────────────────────────────

def fetch_minute(cfg: dict):
    """
    分钟级数据拉取（akshare），断点续拉。

    支持 5min（东方财富 + 新浪兜底）和 1min（新浪）。
    若 config 中 data_pipeline.minute.enable=false 则跳过拉取（使用外部购买的分钟数据时关闭）。
    """
    dp = cfg.get("data_pipeline", {})
    minute_cfg = dp.get("minute", {})
    if not minute_cfg.get("enable", True):
        logger.info(
            "分钟数据拉取已关闭（data_pipeline.minute.enable=false），"
            "请将外部下载数据放入 data/minute/1min 或 5min"
        )
        return

    from quant_system.data_pipeline.stock_pool import get_stock_pool
    from quant_system.data_pipeline.minute_accumulator import backfill_5min, accumulate_minutes
    start_date = dp.get("start_date", "20200101")
    end_date = dp.get("end_date") or datetime.now().strftime("%Y%m%d")

    ts_codes, _ = get_stock_pool(cfg)
    symbols_6 = [c.split(".")[0].zfill(6) for c in ts_codes]
    storage = str(Path(minute_cfg.get("storage_path", "data/minute")).expanduser())
    periods = minute_cfg.get("periods", ["5"])
    sleep_sec = float(minute_cfg.get("sleep_sec", 0.5))
    use_fallback = minute_cfg.get("use_fallback", True)

    rd = minute_cfg.get("recent_days")
    if rd is not None and isinstance(rd, int) and rd > 0:
        end_dt = datetime.strptime(end_date, "%Y%m%d")
        start_minute = (end_dt - timedelta(days=int(rd))).strftime("%Y%m%d")
    else:
        start_minute = start_date

    if "5" in periods:
        backfill_5min(
            symbols_6,
            start_date=start_minute,
            end_date=end_date,
            storage_path=storage,
            sleep_sec=sleep_sec,
            use_fallback=use_fallback,
        )

    if "1" in periods:
        accumulate_minutes(
            symbols=symbols_6,
            periods=["1"],
            storage_path=storage,
            sleep_sec=sleep_sec,
            use_fallback=use_fallback,
        )

    logger.info("分钟数据拉取完成（见 %s）", storage)


# ── 统一入口 ─────────────────────────────────────────

def run(cfg: dict, freq: str = "daily", **kwargs):
    """由 quant_cli.py 调用的统一入口。"""
    if freq == "daily":
        return fetch_daily(cfg, skip_fina=kwargs.get("skip_fina", False),
                           export_qlib=kwargs.get("export_qlib", False))
    elif freq == "minute":
        return fetch_minute(cfg)
    else:
        raise ValueError(f"不支持的频率: {freq}，请使用 daily 或 minute")
