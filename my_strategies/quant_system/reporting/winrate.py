"""
胜率统计与收益回填

移植自 quant_core 的 core/model_winrate.py 和 core/reporting.py。
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_forward_returns(
    recs: pd.DataFrame,
    panel: pd.DataFrame,
    hold_days: int = 5,
) -> pd.DataFrame:
    """
    计算推荐的前向收益（T+1 open 买入，T+hold_days close 卖出）。

    Parameters
    ----------
    recs : DataFrame with signal_date, ts_code
    panel : DataFrame with date, ts_code, open, close_adj
    """
    if recs.empty or panel.empty:
        return recs

    close_col = "close_adj" if "close_adj" in panel.columns else "close"
    trade_dates = sorted(panel["date"].unique())

    returns = []
    for _, row in recs.iterrows():
        sig = pd.Timestamp(row.get("signal_date", row.get("date")))
        code = row["ts_code"]

        buy_date = None
        sell_date = None
        count = 0
        for d in trade_dates:
            if d > sig:
                if buy_date is None:
                    buy_date = d
                count += 1
                if count >= hold_days:
                    sell_date = d
                    break

        if buy_date is None or sell_date is None:
            returns.append(np.nan)
            continue

        buy_row = panel[(panel["date"] == buy_date) & (panel["ts_code"] == code)]
        sell_row = panel[(panel["date"] == sell_date) & (panel["ts_code"] == code)]

        if buy_row.empty or sell_row.empty:
            returns.append(np.nan)
            continue

        buy_price = float(buy_row["open"].iloc[0]) if "open" in buy_row.columns else float(buy_row[close_col].iloc[0])
        sell_price = float(sell_row[close_col].iloc[0])

        if buy_price > 0:
            returns.append(sell_price / buy_price - 1)
        else:
            returns.append(np.nan)

    recs = recs.copy()
    recs["return_pct"] = returns
    return recs


def compute_model_winrates(
    recs_with_returns: pd.DataFrame,
    cost_threshold: float = 0.002,
    model_col: str = "model_family",
    min_samples: int = 20,
) -> tuple:
    """
    按模型计算胜率。

    Returns
    -------
    (winrates, sample_counts) : (Dict[str, float], Dict[str, int])
        样本不足 min_samples 的模型仍会返回胜率值，
        但 sample_counts 可供下游（如 adjust_weights_by_winrate）决定是否采纳。
    """
    if recs_with_returns.empty or "return_pct" not in recs_with_returns.columns:
        return {}, {}

    if model_col not in recs_with_returns.columns:
        valid = recs_with_returns["return_pct"].dropna()
        if len(valid) == 0:
            return {}, {}
        return {"all": float((valid > cost_threshold).mean())}, {"all": len(valid)}

    winrates: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for model, grp in recs_with_returns.groupby(model_col):
        valid = grp["return_pct"].dropna()
        n = len(valid)
        if n == 0:
            continue
        counts[model] = n
        if n < min_samples:
            logger.info("模型 %s 样本数 %d < %d，胜率结果可能不可靠", model, n, min_samples)
        winrates[model] = float((valid > cost_threshold).mean())
    return winrates, counts


def print_winrate_summary(
    recs_with_returns: pd.DataFrame,
    model_col: str = "model_family",
):
    """打印胜率汇总。"""
    if recs_with_returns.empty or "return_pct" not in recs_with_returns.columns:
        logger.info("无收益数据")
        return

    valid = recs_with_returns.dropna(subset=["return_pct"])
    if valid.empty:
        logger.info("无有效收益数据")
        return

    total = len(valid)
    wins = (valid["return_pct"] > 0).sum()
    avg_ret = valid["return_pct"].mean()
    avg_win = valid.loc[valid["return_pct"] > 0, "return_pct"].mean() if wins > 0 else 0
    avg_loss = valid.loc[valid["return_pct"] <= 0, "return_pct"].mean() if (total - wins) > 0 else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    logger.info("=" * 50)
    logger.info("胜率统计: %d 笔推荐", total)
    logger.info("  胜率: %.1f%% (%d/%d)", wins / total * 100, wins, total)
    logger.info("  平均收益: %.2f%%", avg_ret * 100)
    logger.info("  平均盈利: %.2f%% | 平均亏损: %.2f%%", avg_win * 100, avg_loss * 100)
    logger.info("  盈亏比: %.2f", profit_factor)
    logger.info("=" * 50)
