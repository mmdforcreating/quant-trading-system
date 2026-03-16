"""
ATR 风控模块

移植自 quant_core 的 core/atr_risk_incremental.py，提供：
- ATR 计算
- 止损位 / 追踪止盈位
- 风险平价仓位计算
- 回撤控制
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ATRRiskConfig:
    atr_window: int = 14
    k_stop: float = 1.5
    k_trail: float = 1.0
    k_trail_start: float = 2.0
    min_stop_pct: float = 0.03
    max_stop_pct: float = 0.12

    @classmethod
    def from_cfg(cls, cfg: dict) -> "ATRRiskConfig":
        atr = cfg.get("risk", {}).get("atr", {})
        return cls(
            atr_window=atr.get("window", 14),
            k_stop=atr.get("k_stop", 1.5),
            k_trail=atr.get("k_trail", 1.0),
            k_trail_start=atr.get("k_trail_start", 2.0),
            min_stop_pct=atr.get("min_stop_pct", 0.03),
            max_stop_pct=atr.get("max_stop_pct", 0.12),
        )


def compute_atr(
    df: pd.DataFrame,
    window: int = 14,
    code_col: str = "ts_code",
    date_col: str = "date",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close_adj",
) -> pd.DataFrame:
    """计算 ATR 并添加 atr{window} 列。"""
    df = df.copy()
    if close_col not in df.columns:
        close_col = "close"

    df = df.sort_values([code_col, date_col]).reset_index(drop=True)
    prev_close = df.groupby(code_col)[close_col].shift(1)
    hl = (df[high_col] - df[low_col]).abs()
    hc = (df[high_col] - prev_close).abs()
    lc = (df[low_col] - prev_close).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)

    col_name = f"atr{window}"
    df[col_name] = tr.groupby(df[code_col]).transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    return df


def compute_stop_levels(
    entry_price: float,
    atr: float,
    config: ATRRiskConfig,
) -> dict:
    """计算止损/止盈位。"""
    stop_width = config.k_stop * atr
    stop_pct = stop_width / entry_price if entry_price > 0 else 0
    stop_pct = np.clip(stop_pct, config.min_stop_pct, config.max_stop_pct)
    stop_width = entry_price * stop_pct

    stop_loss = entry_price - stop_width
    take_profit = entry_price + config.k_trail_start * atr

    return {
        "stop_loss": round(stop_loss, 4),
        "take_profit": round(take_profit, 4),
        "stop_width": round(stop_width, 4),
        "atr": round(atr, 4),
    }


def compute_position_size(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    lot_size: int = 100,
) -> int:
    """风险平价仓位：每笔亏损不超过 capital * risk_pct。"""
    risk_per_trade = capital * risk_pct
    risk_per_unit = entry_price - stop_loss
    if risk_per_unit <= 0:
        return 0
    raw_shares = risk_per_trade / risk_per_unit
    return int(raw_shares // lot_size) * lot_size


def check_trailing_stop(
    entry_price: float,
    current_price: float,
    high_since_entry: float,
    atr: float,
    config: ATRRiskConfig,
) -> bool:
    """检查是否触发追踪止盈。"""
    profit = current_price - entry_price
    if profit < config.k_trail_start * atr:
        return False
    trail_line = high_since_entry - config.k_trail * atr
    return current_price <= trail_line


def check_stop_loss(
    current_price: float,
    stop_loss: float,
) -> bool:
    return current_price <= stop_loss


class DrawdownGuard:
    """回撤控制：触发最大回撤后进入冷静期。"""

    def __init__(self, cfg: dict):
        risk = cfg.get("risk", {})
        self.max_dd = risk.get("max_drawdown_stop", 0.08)
        self.cooldown = risk.get("cooldown_days", 3)
        self._peak = 0.0
        self._cooldown_remaining = 0

    def reset(self):
        """重置内部状态，供多模型独立回测时在每轮开始前调用。"""
        self._peak = 0.0
        self._cooldown_remaining = 0

    def update(self, equity: float) -> bool:
        """更新权益并返回是否允许开仓。"""
        if equity > self._peak:
            self._peak = equity
            self._cooldown_remaining = 0

        if self._peak > 0:
            dd = (self._peak - equity) / self._peak
            if dd >= self.max_dd:
                self._cooldown_remaining = self.cooldown
                logger.warning("回撤 %.2f%% >= %.2f%%，进入冷静期 %d 天",
                               dd * 100, self.max_dd * 100, self.cooldown)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return False
        return True
