"""
收益回填模块

扫描 output/daily_* 目录的历史推荐，计算持有期收益并写入汇总文件。
可在 run_daily 每日结束时调用做增量回填。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def backfill_returns(
    output_dir: str,
    panel: pd.DataFrame,
    hold_days: int = 5,
    returns_filename: str = "recs_returns.csv",
):
    """
    扫描 output/daily_* 下的 recs_best.csv，计算持有期收益并汇总。

    Parameters
    ----------
    output_dir : str
        输出根目录（含 daily_YYYYMMDD 子目录）
    panel : DataFrame
        面板数据（含 date, ts_code, open, close_adj）
    hold_days : int
        持仓天数
    returns_filename : str
        汇总输出文件名
    """
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    summary_path = os.path.join(reports_dir, returns_filename)

    existing = pd.DataFrame()
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        existing["signal_date"] = pd.to_datetime(existing["signal_date"], errors="coerce")

    already_filled = set()
    if not existing.empty and "signal_date" in existing.columns and "ts_code" in existing.columns:
        for _, row in existing.iterrows():
            if pd.notna(row.get("return_pct")):
                already_filled.add((str(row["signal_date"].date()), row["ts_code"]))

    daily_dirs = sorted(Path(output_dir).glob("daily_*"))
    if not daily_dirs:
        return

    close_col = "close_adj" if "close_adj" in panel.columns else "close"
    trade_dates = sorted(panel["date"].unique())

    new_records = []
    for ddir in daily_dirs:
        recs_path = ddir / "recs_best.csv"
        if not recs_path.exists():
            continue

        date_str = ddir.name.replace("daily_", "")
        try:
            signal_date = pd.Timestamp(date_str)
        except Exception:
            continue

        recs = pd.read_csv(recs_path)
        if recs.empty or "ts_code" not in recs.columns:
            continue

        for _, row in recs.iterrows():
            code = row["ts_code"]
            key = (str(signal_date.date()), code)
            if key in already_filled:
                continue

            ret = _compute_single_return(
                panel, trade_dates, signal_date, code,
                hold_days, close_col,
            )
            new_records.append({
                "signal_date": signal_date,
                "ts_code": code,
                "ensemble_score": row.get("ensemble_score", np.nan),
                "rank": row.get("rank", np.nan),
                "return_pct": ret,
                "hold_days": hold_days,
            })

    if not new_records:
        return

    new_df = pd.DataFrame(new_records)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["signal_date", "ts_code"], keep="last")
    combined = combined.sort_values(["signal_date", "rank"]).reset_index(drop=True)
    combined.to_csv(summary_path, index=False)

    filled = combined["return_pct"].notna().sum()
    total = len(combined)
    logger.info(
        "收益回填: %d/%d 条已填充, 保存 → %s",
        filled, total, summary_path,
    )

    if filled > 0:
        valid = combined.dropna(subset=["return_pct"])
        wins = (valid["return_pct"] > 0).sum()
        avg = valid["return_pct"].mean()
        logger.info(
            "  胜率: %.1f%% (%d/%d), 平均收益: %.2f%%",
            wins / len(valid) * 100, wins, len(valid), avg * 100,
        )


def _compute_single_return(
    panel: pd.DataFrame,
    trade_dates,
    signal_date: pd.Timestamp,
    ts_code: str,
    hold_days: int,
    close_col: str,
) -> Optional[float]:
    """计算单条推荐的持有期收益（T+1 open 买，T+hold_days close 卖）。"""
    buy_date = None
    sell_date = None
    count = 0
    for d in trade_dates:
        if d > signal_date:
            if buy_date is None:
                buy_date = d
            count += 1
            if count >= hold_days:
                sell_date = d
                break

    if buy_date is None or sell_date is None:
        return np.nan

    buy_row = panel[(panel["date"] == buy_date) & (panel["ts_code"] == ts_code)]
    sell_row = panel[(panel["date"] == sell_date) & (panel["ts_code"] == ts_code)]

    if buy_row.empty or sell_row.empty:
        return np.nan

    buy_price = float(buy_row["open"].iloc[0]) if "open" in buy_row.columns else float(buy_row[close_col].iloc[0])
    sell_price = float(sell_row[close_col].iloc[0])

    if buy_price > 0:
        return sell_price / buy_price - 1
    return np.nan
