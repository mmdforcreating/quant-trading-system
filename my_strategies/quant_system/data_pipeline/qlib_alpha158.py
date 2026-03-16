"""
Alpha158 因子桥接模块

将 panel 中的 OHLCV 数据转为 Qlib 二进制格式，
计算 Alpha158 因子，返回 DataFrame 供 merge 到 panel。
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_alpha158(
    panel: pd.DataFrame,
    cfg: dict,
) -> Optional[pd.DataFrame]:
    """
    从 panel 中提取 OHLCV → 写 per-stock CSV → dump Qlib bin → 算 Alpha158。

    Returns
    -------
    DataFrame with (date, ts_code) + Alpha158 columns，或 None 失败时
    """
    dp = cfg.get("data_pipeline", {})
    csv_storage = str(Path(dp.get("csv_storage", "~/.qlib/csv_data")).expanduser())
    qlib_dir = str(Path(dp.get("qlib_dir", "~/.qlib/qlib_data/cn_data")).expanduser())
    fe = cfg.get("factor_engine", {})
    label_horizon = fe.get("label_horizon", 5)

    qlib_cfg = cfg.get("qlib_init", {})
    provider_uri = str(Path(qlib_cfg.get("provider_uri", qlib_dir)).expanduser())

    logger.info("Alpha158: 准备 per-stock CSV → %s", csv_storage)
    _export_per_stock_csv(panel, csv_storage)

    logger.info("Alpha158: CSV → Qlib .bin → %s", qlib_dir)
    from quant_system.data_pipeline.csv_to_qlib import csv_to_qlib_bin
    ok = csv_to_qlib_bin(csv_storage, qlib_dir)
    if not ok:
        logger.error("Alpha158: Qlib bin 转换失败")
        return None

    logger.info("Alpha158: 初始化 Qlib & 计算因子")
    try:
        import qlib
        qlib.init(provider_uri=provider_uri, region="cn")
    except Exception as e:
        logger.error("Alpha158: Qlib 初始化失败: %s", e)
        return None

    instruments = _get_qlib_instruments(panel)
    start_time = panel["date"].min().strftime("%Y-%m-%d")
    end_time = panel["date"].max().strftime("%Y-%m-%d")

    try:
        from quant_system.data_handlers.factor_handler import FactorHandler
        handler = FactorHandler(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            label_horizon=label_horizon,
        )
        df_feat = handler.fetch(col_set="feature", data_key="infer")
        logger.info("Alpha158: %d 行, %d 个因子", len(df_feat), df_feat.shape[1])
    except Exception as e:
        logger.error("Alpha158: 因子计算失败: %s", e)
        return None

    alpha_df = _qlib_index_to_panel_format(df_feat)
    return alpha_df


def _export_per_stock_csv(panel: pd.DataFrame, csv_dir: str):
    """将 panel 拆成 Qlib 要求的 per-stock CSV（{INSTRUMENT}.csv）。
    
    导出前清理目录下的旧 CSV，防止残留已不在 panel 中的股票导致不同步。
    """
    os.makedirs(csv_dir, exist_ok=True)
    for old_csv in Path(csv_dir).glob("*.csv"):
        old_csv.unlink()
    logger.debug("已清理旧 CSV: %s", csv_dir)

    close_col = "close_adj" if "close_adj" in panel.columns else "close"
    vol_col = "vol" if "vol" in panel.columns else "volume"

    needed = ["ts_code", "date", "open", close_col, "high", "low", vol_col, "amount"]
    avail = [c for c in needed if c in panel.columns]
    sub = panel[avail].copy()

    if close_col != "close" and "close" not in sub.columns:
        sub = sub.rename(columns={close_col: "close"})
    if vol_col != "volume" and "volume" not in sub.columns:
        sub = sub.rename(columns={vol_col: "volume"})

    sub["date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d")

    count = 0
    for code, grp in sub.groupby("ts_code"):
        instrument = _ts_code_to_qlib(code)
        fpath = os.path.join(csv_dir, f"{instrument}.csv")
        out = grp.drop(columns=["ts_code"]).sort_values("date")
        out.to_csv(fpath, index=False)
        count += 1

    logger.info("Alpha158: 导出 %d 只股票 CSV", count)


def _ts_code_to_qlib(ts_code: str) -> str:
    """600519.SH → SH600519"""
    parts = ts_code.split(".")
    if len(parts) == 2:
        return f"{parts[1]}{parts[0]}"
    return ts_code


def _qlib_code_to_ts(qlib_code: str) -> str:
    """SH600519 → 600519.SH"""
    m = re.match(r"^(SH|SZ|BJ)(\d{6})$", qlib_code, re.IGNORECASE)
    if m:
        return f"{m.group(2)}.{m.group(1).upper()}"
    return qlib_code


def _get_qlib_instruments(panel: pd.DataFrame):
    """从 panel 的 ts_code 列表构造 Qlib instruments 列表。"""
    codes = panel["ts_code"].unique()
    return [_ts_code_to_qlib(c) for c in codes]


def _qlib_index_to_panel_format(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    将 Qlib 输出的 (datetime, instrument) MultiIndex DataFrame
    转换为 panel 的 (date, ts_code) 格式。
    """
    df = df_feat.reset_index()

    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})
    if "instrument" in df.columns:
        df["ts_code"] = df["instrument"].apply(_qlib_code_to_ts)
        df = df.drop(columns=["instrument"])

    df["date"] = pd.to_datetime(df["date"])

    prefix = "alpha158_"
    feat_cols = [c for c in df.columns if c not in ("date", "ts_code")]
    rename_map = {c: f"{prefix}{c}" for c in feat_cols}
    df = df.rename(columns=rename_map)

    return df


def merge_alpha158_to_panel(
    panel: pd.DataFrame,
    alpha_df: pd.DataFrame,
) -> pd.DataFrame:
    """将 Alpha158 因子 merge 进 panel。"""
    alpha_cols = [c for c in alpha_df.columns if c.startswith("alpha158_")]

    for col in alpha_cols:
        if col in panel.columns:
            panel = panel.drop(columns=[col])

    panel = panel.merge(alpha_df, on=["date", "ts_code"], how="left")
    logger.info("Alpha158 merge 完成: panel 现有 %d 列", panel.shape[1])
    return panel
