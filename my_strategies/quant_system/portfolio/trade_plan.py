"""
持仓管理 + 交易计划生成

移植自 quant_core 的 core/portfolio.py，提供：
- 持仓跟踪（含天数、盈亏、到期状态）
- 交易计划生成（BUY/SELL + ATR 止损止盈 + 仓位）
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..risk.atr_risk import (
    ATRRiskConfig,
    check_trailing_stop,
    compute_atr,
    compute_position_size,
    compute_stop_levels,
)

logger = logging.getLogger(__name__)


def generate_positions(
    recs_history: pd.DataFrame,
    panel: pd.DataFrame,
    hold_days: int = 5,
    atr_config: Optional[ATRRiskConfig] = None,
) -> pd.DataFrame:
    """
    从推荐历史生成当前持仓表。

    Parameters
    ----------
    recs_history : DataFrame
        历史推荐记录，包含 signal_date, ts_code
    panel : DataFrame
        面板数据（含 close_adj, date, ts_code）
    hold_days : int
        持仓天数
    atr_config : ATRRiskConfig
        ATR 配置

    Returns
    -------
    DataFrame: ts_code, entry_date, entry_price, days_held, is_expired, unrealized_pnl, ...
    """
    if recs_history.empty:
        return pd.DataFrame()

    close_col = "close_adj" if "close_adj" in panel.columns else "close"
    price_col_display = "close" if "close" in panel.columns else close_col
    latest_date = panel["date"].max()
    trade_dates = sorted(panel["date"].unique())

    positions = []
    for _, rec in recs_history.iterrows():
        sig_date = pd.Timestamp(rec.get("signal_date", rec.get("date")))
        code = rec["ts_code"]

        entry_date = _next_trade_date(trade_dates, sig_date)
        if entry_date is None:
            continue

        exit_date = _nth_trade_date_after(trade_dates, sig_date, hold_days)
        days_held = _count_trade_days_between(trade_dates, entry_date, latest_date)
        is_expired = days_held >= hold_days

        entry_row = panel[(panel["date"] == entry_date) & (panel["ts_code"] == code)]
        entry_price = float(entry_row["open"].iloc[0]) if (not entry_row.empty and "open" in entry_row.columns) else np.nan
        if np.isnan(entry_price) and not entry_row.empty:
            entry_price = float(entry_row[price_col_display].iloc[0])

        latest_row = panel[(panel["date"] == latest_date) & (panel["ts_code"] == code)]
        current_price = float(latest_row[price_col_display].iloc[0]) if not latest_row.empty else np.nan

        pnl = (current_price / entry_price - 1) if (entry_price > 0 and np.isfinite(current_price)) else np.nan

        high_col = "high" if "high" in panel.columns else close_col
        high_since_entry = _get_high_since_entry(panel, code, entry_date, latest_date, high_col)

        pos = {
            "ts_code": code,
            "signal_date": sig_date,
            "entry_date": entry_date,
            "entry_price": round(entry_price, 4) if np.isfinite(entry_price) else np.nan,
            "current_price": round(current_price, 4) if np.isfinite(current_price) else np.nan,
            "high_since_entry": round(high_since_entry, 4) if np.isfinite(high_since_entry) else np.nan,
            "days_held": days_held,
            "expected_exit": exit_date,
            "is_expired": is_expired,
            "unrealized_pnl": round(pnl, 6) if np.isfinite(pnl) else np.nan,
        }

        if atr_config and np.isfinite(entry_price):
            atr_val = _get_atr_for_code(panel, code, entry_date, atr_config.atr_window)
            if atr_val > 0:
                levels = compute_stop_levels(entry_price, atr_val, atr_config)
                pos.update(levels)

        positions.append(pos)

    return pd.DataFrame(positions) if positions else pd.DataFrame()


def build_trade_plan(
    positions: pd.DataFrame,
    candidates: pd.DataFrame,
    panel: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    生成交易计划：到期卖 + ATR 止损/追踪止盈卖 + 换手控制补买/替换。

    Returns
    -------
    DataFrame: ts_code, action, reason, suggested_price, stop_loss, take_profit, position_size
    """
    top_k = cfg.get("walk_forward", {}).get("top_k", 5)
    risk = cfg.get("risk", {})
    capital = risk.get("initial_capital", 1_000_000)
    risk_pct = risk.get("risk_pct_per_trade", 0.01)
    atr_cfg = ATRRiskConfig.from_cfg(cfg) if risk.get("atr", {}).get("enabled", False) else None

    tc = cfg.get("turnover_control", {})
    tc_enabled = tc.get("enabled", False)
    buffer_mult = tc.get("buffer_mult", 2)
    min_hold = tc.get("min_hold_days_before_replace", 2)
    replace_delta = tc.get("replace_score_delta", 0.0)
    max_turnover = tc.get("max_turnover_names_per_day")

    close_col = "close_adj" if "close_adj" in panel.columns else "close"
    price_col_display = "close" if "close" in panel.columns else close_col
    latest_date = panel["date"].max()

    plan = []

    # --- Phase 1: 生成强制 SELL（到期/止损/追踪止盈）---
    if not positions.empty:
        for _, pos in positions.iterrows():
            if pos.get("is_expired", False):
                plan.append({
                    "ts_code": pos["ts_code"],
                    "action": "SELL",
                    "reason": "到期卖出",
                    "priority": 0,
                    "suggested_price": pos.get("current_price", np.nan),
                })
            elif atr_cfg and np.isfinite(pos.get("current_price", np.nan)):
                cur = pos["current_price"]
                sl = pos.get("stop_loss")
                if sl and np.isfinite(sl) and cur <= sl:
                    plan.append({
                        "ts_code": pos["ts_code"],
                        "action": "SELL",
                        "reason": "ATR 止损",
                        "priority": 0,
                        "suggested_price": cur,
                    })
                elif pos.get("atr") and np.isfinite(pos.get("atr", 0)):
                    high_since = pos.get("high_since_entry", cur)
                    if not np.isfinite(high_since):
                        high_since = cur
                    if check_trailing_stop(
                        entry_price=pos["entry_price"],
                        current_price=cur,
                        high_since_entry=high_since,
                        atr=pos["atr"],
                        config=atr_cfg,
                    ):
                        plan.append({
                            "ts_code": pos["ts_code"],
                            "action": "SELL",
                            "reason": "追踪止盈",
                            "priority": 0,
                            "suggested_price": cur,
                        })

    forced_sell_codes = {p["ts_code"] for p in plan if p["action"] == "SELL"}

    # --- Phase 2: 确定当前活跃持仓 ---
    active_codes = set()
    active_days = {}
    if not positions.empty:
        active = positions[~positions.get("is_expired", pd.Series(False, index=positions.index))]
        for _, row in active.iterrows():
            code = row["ts_code"]
            if code not in forced_sell_codes:
                active_codes.add(code)
                active_days[code] = int(row.get("days_held", 0))

    # --- Phase 3: 换手控制（缓冲池 + 替换规则） ---
    score_map = {}
    if not candidates.empty and "ensemble_score" in candidates.columns:
        score_map = candidates.set_index("ts_code")["ensemble_score"].to_dict()

    if tc_enabled and not candidates.empty:
        buffer_n = int(buffer_mult * top_k)
        buffer_set = set(candidates.head(buffer_n)["ts_code"].tolist())

        keep = []
        keep_soft = []
        candidates_for_replace = []

        for code in active_codes:
            dh = active_days.get(code, 0)
            if code in buffer_set:
                keep.append(code)
            elif dh < min_hold:
                keep_soft.append(code)
            else:
                candidates_for_replace.append(code)

        target = list(dict.fromkeys(keep + keep_soft))

        buys_needed = []
        for _, row in candidates.iterrows():
            if len(target) >= top_k:
                break
            code = row["ts_code"]
            if code not in target and code not in forced_sell_codes:
                target.append(code)
                if code not in active_codes:
                    buys_needed.append(code)

        sell_replace = []
        if replace_delta > 0 and candidates_for_replace and buys_needed:
            for old_code in candidates_for_replace:
                old_score = score_map.get(old_code, -1e18)
                better = any(score_map.get(nc, -1e18) >= old_score + replace_delta for nc in buys_needed)
                if better:
                    sell_replace.append(old_code)
        else:
            sell_replace = [c for c in candidates_for_replace if c not in set(target)]

        if max_turnover is not None:
            cap = int(max_turnover)
            sell_replace = sell_replace[:cap]
            buys_needed = buys_needed[:cap]

        for code in sell_replace:
            plan.append({"ts_code": code, "action": "SELL", "reason": "换手替换", "priority": 1, "suggested_price": np.nan})
            if code in active_codes:
                active_codes.discard(code)

        remaining_active = active_codes - forced_sell_codes - set(sell_replace)
        need_buy = max(0, top_k - len(remaining_active))
        buy_list = buys_needed[:need_buy]

    else:
        remaining_active = active_codes
        need_buy = max(0, top_k - len(remaining_active))
        buy_list = []
        if need_buy > 0 and not candidates.empty:
            for _, row in candidates.iterrows():
                if len(buy_list) >= need_buy:
                    break
                code = row["ts_code"]
                if code not in remaining_active and code not in forced_sell_codes:
                    buy_list.append(code)

    # --- Phase 4: 生成 BUY ---
    for code in buy_list:
        price_row = panel[(panel["date"] == latest_date) & (panel["ts_code"] == code)]
        price = float(price_row[price_col_display].iloc[0]) if not price_row.empty else np.nan

        entry = {
            "ts_code": code,
            "action": "BUY",
            "reason": "推荐补仓",
            "priority": 2,
            "suggested_price": round(price, 4) if np.isfinite(price) else np.nan,
        }

        if atr_cfg and np.isfinite(price):
            atr_val = _get_atr_for_code(panel, code, latest_date, atr_cfg.atr_window)
            if atr_val > 0:
                levels = compute_stop_levels(price, atr_val, atr_cfg)
                entry.update(levels)
                entry["position_size"] = compute_position_size(
                    capital, risk_pct, price, levels["stop_loss"],
                )

        plan.append(entry)

    # --- Phase 5: 生成 HOLD（方便审计与运营追踪）---
    hold_codes = remaining_active - set(buy_list)
    for code in sorted(hold_codes):
        price_row = panel[(panel["date"] == latest_date) & (panel["ts_code"] == code)]
        price = float(price_row[price_col_display].iloc[0]) if not price_row.empty else np.nan
        plan.append({
            "ts_code": code,
            "action": "HOLD",
            "reason": "继续持有",
            "priority": 3,
            "suggested_price": round(price, 4) if np.isfinite(price) else np.nan,
        })

    if not plan:
        return pd.DataFrame()

    result = pd.DataFrame(plan)
    priority_order = {"SELL": 0, "BUY": 1, "HOLD": 2}
    result["_sort"] = result["action"].map(priority_order).fillna(9)
    result = result.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return result


def _next_trade_date(trade_dates, after_date):
    for d in trade_dates:
        if d > after_date:
            return d
    return None


def _nth_trade_date_after(trade_dates, after_date, n):
    count = 0
    for d in trade_dates:
        if d > after_date:
            count += 1
            if count >= n:
                return d
    return None


def _count_trade_days_between(trade_dates, start, end):
    return sum(1 for d in trade_dates if start <= d <= end)


def _get_high_since_entry(panel, code, entry_date, latest_date, high_col="high"):
    """获取从 entry_date 到 latest_date 之间该股票的最高价。"""
    sub = panel[(panel["ts_code"] == code) &
                (panel["date"] >= entry_date) &
                (panel["date"] <= latest_date)]
    if sub.empty or high_col not in sub.columns:
        return np.nan
    val = sub[high_col].max()
    return float(val) if np.isfinite(val) else np.nan


def _get_atr_for_code(panel, code, date, window):
    sub = panel[panel["ts_code"] == code].copy()
    if sub.empty or "high" not in sub.columns or "low" not in sub.columns:
        return 0.0
    close_col = "close" if "close" in sub.columns else ("close_adj" if "close_adj" in sub.columns else "close")
    sub = sub.sort_values("date").reset_index(drop=True)
    prev = sub[close_col].shift(1)
    hl = (sub["high"] - sub["low"]).abs()
    hc = (sub["high"] - prev).abs()
    lc = (sub["low"] - prev).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=window).mean()
    mask = sub["date"] <= date
    if not mask.any():
        return 0.0
    last_idx = mask[mask].index[-1]
    val = float(atr.iloc[last_idx])
    return val if np.isfinite(val) else 0.0
