"""
Backtrader 策略模块

提供两种策略：
1. SignalCsvStrategy  —— 基于预测分数 CSV 的历史回测
2. TradePlanStrategy  —— 基于 trade_plan.csv 的每日信号执行
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import backtrader as bt
except ImportError:
    raise ImportError(
        "请先安装 backtrader: pip install backtrader 或从源码安装"
    )

logger = logging.getLogger(__name__)


# ====================================================================
#  SignalCsvStrategy: 基于预测分数做历史回测
# ====================================================================

class SignalCsvStrategy(bt.Strategy):
    """
    基于 predictions CSV 的多股票择时策略。

    每个 walk-forward 步长内，选取 top_k 股票等权买入，
    持有 hold_days 后卖出。支持 ATR 止损和追踪止盈。

    Parameters (通过 params 传入)
    ----------
    predictions : pd.DataFrame
        列: date(datetime), ts_code(str), score(float)
    top_k : int
    hold_days : int
    atr_window : int           ATR 计算窗口（0=禁用 ATR 风控）
    atr_k_stop : float         止损倍数
    atr_k_trail_start : float  追踪止盈启动倍数
    atr_k_trail : float        追踪止盈宽度
    """

    params = dict(
        predictions=None,
        top_k=5,
        hold_days=5,
        atr_window=14,
        atr_k_stop=1.5,
        atr_k_trail_start=2.0,
        atr_k_trail=1.0,
    )

    def __init__(self):
        self._data_map: Dict[str, bt.AbstractDataBase] = {}
        for d in self.datas:
            self._data_map[d._name] = d

        self._holdings: Dict[str, dict] = {}
        self._pred_by_date: Dict = {}

        if self.p.predictions is not None:
            preds = self.p.predictions.copy()
            preds["date"] = pd.to_datetime(preds["date"])
            for dt, grp in preds.groupby("date"):
                self._pred_by_date[dt.date()] = grp

        self._last_signal_date = None

        self._atr_enabled = self.p.atr_window > 0

    def _compute_atr(self, data) -> float:
        """从 backtrader data feed 计算 ATR。"""
        w = self.p.atr_window
        if len(data) < w + 1:
            return 0.0
        trs = []
        for i in range(-w, 0):
            h = data.high[i]
            l = data.low[i]
            c_prev = data.close[i - 1] if abs(i - 1) <= len(data) else data.close[i]
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 0.0

    def next(self):
        today = self.datetime.date()

        # --- 到期 / ATR 止损 / 追踪止盈 卖出 ---
        for code in list(self._holdings.keys()):
            h = self._holdings[code]
            h["days_held"] += 1

            data = self._data_map.get(code)
            if data is None or self.getposition(data).size <= 0:
                self._holdings.pop(code, None)
                continue

            cur_price = data.close[0]

            # 更新最高价
            if cur_price > h.get("high_since_entry", 0):
                h["high_since_entry"] = cur_price

            sell_reason = None

            # 到期卖出
            if h["days_held"] >= self.p.hold_days:
                sell_reason = "expired"

            # ATR 止损
            if sell_reason is None and self._atr_enabled:
                sl = h.get("stop_loss")
                if sl is not None and cur_price <= sl:
                    sell_reason = "atr_stop"

            # 追踪止盈
            if sell_reason is None and self._atr_enabled:
                entry = h.get("entry_price", 0)
                atr = h.get("atr", 0)
                high = h.get("high_since_entry", cur_price)
                if entry > 0 and atr > 0:
                    profit_atr = (high - entry) / atr
                    if profit_atr >= self.p.atr_k_trail_start:
                        trail_level = high - self.p.atr_k_trail * atr
                        if cur_price <= trail_level:
                            sell_reason = "trailing_stop"

            if sell_reason:
                self.close(data=data)
                del self._holdings[code]

        # --- 检查买入信号 ---
        grp = self._pred_by_date.get(today)
        if grp is None or today == self._last_signal_date:
            return
        self._last_signal_date = today

        top = grp.nlargest(self.p.top_k, "score")
        buy_codes = [
            row["ts_code"]
            for _, row in top.iterrows()
            if row["ts_code"] not in self._holdings
            and row["ts_code"] in self._data_map
        ]

        if not buy_codes:
            return

        available_cash = self.broker.getcash()
        per_stock = available_cash * 0.95 / len(buy_codes)

        for code in buy_codes:
            data = self._data_map[code]
            price = data.close[0]
            if price <= 0:
                continue
            shares = int(per_stock / (price * 100)) * 100
            if shares >= 100:
                self.buy(data=data, size=shares)
                entry_info = {"days_held": 0, "entry_price": price, "high_since_entry": price}
                if self._atr_enabled:
                    atr = self._compute_atr(data)
                    entry_info["atr"] = atr
                    if atr > 0:
                        entry_info["stop_loss"] = price - self.p.atr_k_stop * atr
                self._holdings[code] = entry_info


# ====================================================================
#  TradePlanStrategy: 基于 trade_plan.csv 执行
# ====================================================================

class TradePlanStrategy(bt.Strategy):
    """
    逐日读取交易计划并执行 BUY/SELL。

    Parameters
    ----------
    trade_plans : dict
        {date -> list[dict(ts_code, action, suggested_price, position_size, ...)]}
    """

    params = dict(
        trade_plans=None,
    )

    def __init__(self):
        self._data_map: Dict[str, bt.AbstractDataBase] = {}
        for d in self.datas:
            self._data_map[d._name] = d

    def next(self):
        today = self.datetime.date()
        plans = (self.p.trade_plans or {}).get(today, [])

        for plan in plans:
            code = plan.get("ts_code")
            action = plan.get("action", "").upper()
            data = self._data_map.get(code)
            if data is None:
                continue

            if action == "BUY":
                size = int(plan.get("position_size", 0))
                if size <= 0:
                    price = data.close[0]
                    if price > 0:
                        cash = self.broker.getcash()
                        size = int(cash * 0.18 / (price * 100)) * 100
                if size >= 100:
                    self.buy(data=data, size=size)

            elif action == "SELL":
                pos = self.getposition(data)
                if pos.size > 0:
                    self.close(data=data)


# ====================================================================
#  辅助：从 panel 创建 PandasData feeds
# ====================================================================

def create_data_feeds(
    panel: pd.DataFrame,
    stock_codes: Optional[List[str]] = None,
) -> List[bt.feeds.PandasData]:
    """
    从 panel.parquet 创建 backtrader PandasData feed 列表。

    每只股票一个 feed，使用复权价（close_adj 等）。
    """
    close_col = "close_adj" if "close_adj" in panel.columns else "close"
    open_col = "open_adj" if "open_adj" in panel.columns else "open"
    high_col = "high_adj" if "high_adj" in panel.columns else "high"
    low_col = "low_adj" if "low_adj" in panel.columns else "low"
    vol_col = "vol" if "vol" in panel.columns else "volume"

    if stock_codes is None:
        stock_codes = sorted(panel["ts_code"].unique())

    feeds = []
    for code in stock_codes:
        sub = panel[panel["ts_code"] == code].copy()
        if sub.empty or len(sub) < 5:
            continue

        df = pd.DataFrame({
            "open": sub[open_col].values,
            "high": sub[high_col].values,
            "low": sub[low_col].values,
            "close": sub[close_col].values,
            "volume": sub[vol_col].values if vol_col in sub.columns else 0,
        }, index=pd.to_datetime(sub["date"]))
        df = df.sort_index().dropna(subset=["close"])

        if df.empty:
            continue

        feed = bt.feeds.PandasData(
            dataname=df,
            name=code,
            openinterest=None,
        )
        feeds.append(feed)

    logger.info("创建 %d 个 PandasData feeds", len(feeds))
    return feeds
