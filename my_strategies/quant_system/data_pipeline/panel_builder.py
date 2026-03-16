"""
面板构建器

将 TuShare 拉取的 5 类 raw 数据合并为统一的 panel.parquet：
daily + daily_basic + adj_factor + moneyflow + fina_indicator → 合并 → 复权 → 输出
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_panel(
    raw_data: Dict[str, pd.DataFrame],
    output_path: str,
    start_date: str = "20200101",
) -> pd.DataFrame:
    """
    合并 5 类数据，计算复权价，输出 panel.parquet。

    Parameters
    ----------
    raw_data : dict
        {"daily": df, "daily_basic": df, "adj_factor": df, "moneyflow": df, "fina_indicator": df}
    output_path : str
        panel 输出路径
    start_date : str
        过滤起始日期
    """
    df_daily = raw_data.get("daily", pd.DataFrame())
    if df_daily.empty:
        raise ValueError("daily 数据为空，无法构建 panel")

    df_daily = _normalize_date(df_daily, "trade_date")

    for name in ["daily_basic", "adj_factor", "moneyflow"]:
        other = raw_data.get(name, pd.DataFrame())
        if other.empty:
            logger.warning("%s 为空，跳过合并", name)
            continue
        other = _normalize_date(other, "trade_date")
        overlap = [c for c in other.columns if c in df_daily.columns and c not in ("ts_code", "trade_date")]
        if overlap:
            other = other.drop(columns=overlap, errors="ignore")
        df_daily = df_daily.merge(other, on=["ts_code", "trade_date"], how="left")

    fina = raw_data.get("fina_indicator", pd.DataFrame())
    if not fina.empty and "ann_date" in fina.columns:
        df_daily = _merge_fina(df_daily, fina)

    if "date" not in df_daily.columns:
        df_daily["date"] = pd.to_datetime(df_daily["trade_date"], format="%Y%m%d", errors="coerce")
    else:
        df_daily["date"] = pd.to_datetime(df_daily["date"], errors="coerce")

    df_daily = df_daily.dropna(subset=["date"])
    df_daily = df_daily[df_daily["date"] >= pd.Timestamp(start_date)]
    df_daily = df_daily.drop_duplicates(subset=["ts_code", "date"])
    df_daily = df_daily.sort_values(["ts_code", "date"]).reset_index(drop=True)

    df_daily = _compute_adj_prices(df_daily)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_daily.to_parquet(output_path, index=False)
    logger.info("Panel 已保存: %s | %d 行, %d 列", output_path, len(df_daily), df_daily.shape[1])

    return df_daily


def _normalize_date(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col in df.columns:
        df[date_col] = df[date_col].astype(str).str.replace("-", "")
        df["date"] = pd.to_datetime(df[date_col], format="%Y%m%d", errors="coerce")
    return df


def _merge_fina(daily: pd.DataFrame, fina: pd.DataFrame) -> pd.DataFrame:
    """用 merge_asof 将 fina_indicator 按 ann_date 对齐到 daily。"""
    fina = fina.copy()
    fina["ann_date"] = fina["ann_date"].astype(str).str.replace("-", "")
    fina["ann_date_dt"] = pd.to_datetime(fina["ann_date"], format="%Y%m%d", errors="coerce")
    fina = fina.dropna(subset=["ann_date_dt"])

    fina_cols = ["ts_code", "ann_date_dt", "roe", "roa", "netprofit_yoy", "tr_yoy",
                 "grossprofit_margin", "netprofit_margin", "debt_to_assets",
                 "current_ratio", "quick_ratio"]
    fina_cols = [c for c in fina_cols if c in fina.columns]
    fina = fina[fina_cols].drop_duplicates(subset=["ts_code", "ann_date_dt"])

    daily = daily.sort_values(["ts_code", "date"]).reset_index(drop=True)
    fina = fina.sort_values(["ts_code", "ann_date_dt"]).reset_index(drop=True)

    merged_parts = []
    for code, grp in daily.groupby("ts_code"):
        fina_sub = fina[fina["ts_code"] == code].copy()
        if fina_sub.empty:
            merged_parts.append(grp)
            continue
        m = pd.merge_asof(
            grp.sort_values("date"),
            fina_sub.drop(columns=["ts_code"]).sort_values("ann_date_dt"),
            left_on="date", right_on="ann_date_dt",
            direction="backward",
        )
        merged_parts.append(m)

    result = pd.concat(merged_parts, ignore_index=True)
    result = result.drop(columns=["ann_date_dt"], errors="ignore")
    return result


def _compute_adj_prices(df: pd.DataFrame) -> pd.DataFrame:
    """计算复权价格（后复权）。"""
    if "adj_factor" not in df.columns:
        if "close" in df.columns:
            df["close_adj"] = df["close"]
        return df

    df["adj_factor"] = df.groupby("ts_code")["adj_factor"].ffill()

    for col in ["close", "open", "high", "low"]:
        if col in df.columns:
            df[f"{col}_adj"] = df[col] * df["adj_factor"].fillna(1.0)

    return df
