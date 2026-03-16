#!/usr/bin/env python3
"""
终极沙盘推演 (benefit_backtest)

职责：调用 Backtrader 引擎做带摩擦成本的 T+1 事件驱动回测。

两种模式：
  historical: 加载 ensemble_predictions / predictions_*.csv → SignalCsvStrategy
  daily:      加载历史 trade_plan.csv → TradePlanStrategy

摩擦成本：万二佣金 + 千一印花税 + 滑点
T+1 限制由 Backtrader 策略内 cheat-on-open 模式体现
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent / "my_strategies"
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

logger = logging.getLogger("benefit_backtest")


def backtest(
    cfg: dict,
    mode: str = "historical",
    predictions_path: str | None = None,
    trade_plan_path: str | None = None,
    do_plot: bool = False,
):
    """
    Backtrader 回测入口。

    Parameters
    ----------
    cfg : dict           全局配置
    mode : str           historical / daily
    predictions_path : str  指定预测 CSV（historical 模式）
    trade_plan_path : str   指定交易计划 CSV（daily 模式）
    do_plot : bool       是否绘制资金曲线
    """
    dp = cfg.get("data_pipeline", {})
    panel_path = str(Path(dp.get("panel_path", "data/panel.parquet")).expanduser())

    if not os.path.exists(panel_path):
        raise FileNotFoundError(
            f"Panel 不存在: {panel_path}，请先运行 `python quant_cli.py fetch --freq daily`"
        )

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])
    logger.info("Panel: %d 行, %s → %s",
                len(panel), panel["date"].min().date(), panel["date"].max().date())

    if mode == "historical":
        return _run_historical(cfg, panel, predictions_path, do_plot)
    elif mode == "daily":
        return _run_daily(cfg, panel, trade_plan_path, do_plot)
    else:
        raise ValueError(f"不支持的模式: {mode}，请使用 historical 或 daily")


# ── historical 模式 ──────────────────────────────────

def _run_historical(cfg: dict, panel: pd.DataFrame, predictions_path: str | None, do_plot: bool):
    """使用 SignalCsvStrategy 做完整历史回测。"""
    from quant_system.run_backtest import run_historical_backtest

    predictions = None
    if predictions_path:
        predictions = pd.read_csv(predictions_path)
        predictions["date"] = pd.to_datetime(predictions["date"])
        logger.info("加载指定预测: %s (%d 行)", predictions_path, len(predictions))
    else:
        predictions = _load_predictions(cfg)

    if predictions is None or predictions.empty:
        raise RuntimeError("无可用预测数据，请先运行 `python quant_cli.py weekly`")

    results = run_historical_backtest(cfg, panel, predictions)

    if do_plot:
        logger.info("do_plot=True 已传递给 run_historical_backtest (需 backtrader.plot 配置)")

    return results


# ── daily 模式 ──────────────────────────────────────

def _run_daily(cfg: dict, panel: pd.DataFrame, trade_plan_path: str | None, do_plot: bool):
    """使用 TradePlanStrategy 模拟执行 trade_plan。"""
    from quant_system.run_backtest import run_daily_backtest

    if trade_plan_path is None:
        trade_plan_path = _find_latest_trade_plan(cfg)

    if trade_plan_path is None or not os.path.exists(trade_plan_path):
        raise FileNotFoundError("未找到 trade_plan.csv，请先运行 `python quant_cli.py daily`")

    logger.info("加载交易计划: %s", trade_plan_path)
    results = run_daily_backtest(cfg, panel, trade_plan_path)
    return results


# ── 辅助 ────────────────────────────────────────────

def _load_predictions(cfg: dict) -> pd.DataFrame | None:
    """从缓存加载预测数据（优先集成预测，其次合并各模型）。"""
    cache_dir = cfg.get("output", {}).get("cache_dir", "output/cache")

    ensemble_path = os.path.join(cache_dir, "ensemble_predictions.csv")
    if os.path.exists(ensemble_path):
        df = pd.read_csv(ensemble_path)
        df["date"] = pd.to_datetime(df["date"])
        logger.info("加载集成预测: %s (%d 行)", ensemble_path, len(df))
        return df

    all_preds = []
    for f in Path(cache_dir).glob("predictions_*.csv"):
        df = pd.read_csv(f)
        df["date"] = pd.to_datetime(df["date"])
        all_preds.append(df)
        logger.info("加载模型预测: %s (%d 行)", f.name, len(df))

    if not all_preds:
        return None

    combined = pd.concat(all_preds, ignore_index=True)
    ensemble = (
        combined.groupby(["date", "ts_code"])["pred_score"]
        .mean()
        .reset_index()
        .rename(columns={"pred_score": "score"})
    )
    return ensemble


def _find_latest_trade_plan(cfg: dict) -> str | None:
    """查找最新的 trade_plan.csv。"""
    output_dir = cfg.get("output", {}).get("dir", "output")
    candidates = sorted(
        Path(output_dir).glob("daily_*/trade_plan.csv"),
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


# ── 统一入口 ─────────────────────────────────────────

def run(cfg: dict, **kwargs):
    """由 quant_cli.py 调用的统一入口。"""
    return backtest(
        cfg,
        mode=kwargs.get("mode", "historical"),
        predictions_path=kwargs.get("predictions"),
        trade_plan_path=kwargs.get("trade_plan"),
        do_plot=kwargs.get("plot", False),
    )
