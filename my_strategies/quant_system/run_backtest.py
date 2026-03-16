#!/usr/bin/env python3
"""
Backtrader 回测入口

=== 用法 ===

# 历史回测（使用 walk-forward 集成预测信号）
python -m quant_system.run_backtest --config quant_system/configs/config.yaml --mode historical

# 每日信号执行（使用 trade_plan.csv）
python -m quant_system.run_backtest --config quant_system/configs/config.yaml --mode daily

=== 流程 ===

historical 模式:
  1. 加载 panel.parquet → 创建 PandasData feeds
  2. 加载 ensemble_predictions 或 predictions_*.csv
  3. 运行 SignalCsvStrategy → 资金曲线 + Sharpe + 最大回撤

daily 模式:
  1. 加载 panel.parquet → 创建 PandasData feeds
  2. 加载最新 trade_plan.csv
  3. 运行 TradePlanStrategy → 模拟执行
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import backtrader as bt
except ImportError:
    raise ImportError("backtrader 未安装，请运行: pip install backtrader")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_backtest")


def parse_args():
    p = argparse.ArgumentParser(description="Backtrader 回测")
    p.add_argument("--config", type=str, default="quant_system/configs/config.yaml")
    p.add_argument(
        "--mode", type=str, choices=["historical", "daily"], default="historical",
    )
    p.add_argument("--predictions", type=str, default=None,
                    help="指定 predictions CSV 路径（默认自动查找）")
    p.add_argument("--trade-plan", type=str, default=None,
                    help="指定 trade_plan CSV 路径（daily 模式）")
    p.add_argument("--plot", action="store_true", help="绘制资金曲线图")
    return p.parse_args()


# ------------------------------------------------------------------
#  历史回测
# ------------------------------------------------------------------

def run_historical_backtest(
    cfg,
    panel: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
):
    """
    用 SignalCsvStrategy 做完整历史回测。

    Parameters
    ----------
    cfg : ConfigManager
    panel : 面板数据
    predictions : 预测数据 (date, ts_code, score)。若为 None 则从缓存加载。
    """
    from quant_system.portfolio.bt_strategy import (
        SignalCsvStrategy,
        create_data_feeds,
    )

    bt_cfg = cfg.get("backtrader", {})
    wf_cfg = cfg.get("walk_forward", {})
    risk_cfg = cfg.get("risk", {})

    initial_capital = bt_cfg.get(
        "initial_capital", risk_cfg.get("initial_capital", 1_000_000)
    )
    commission = bt_cfg.get("commission_pct", 0.0007)
    stamp_tax = bt_cfg.get("stamp_tax_pct", 0.0005)
    slippage = bt_cfg.get("slippage_pct", 0.001)
    top_k = wf_cfg.get("top_k", 5)
    hold_days = wf_cfg.get("hold_days", 5)
    do_plot = bt_cfg.get("plot", False)

    # 加载预测
    if predictions is None:
        predictions = _load_predictions(cfg)
    if predictions is None or predictions.empty:
        logger.error("无可用预测数据")
        return

    predictions = _normalize_pred_columns(predictions)

    # 确定需要数据的股票
    pred_codes = set(predictions["ts_code"].unique())
    panel_codes = set(panel["ts_code"].unique())
    needed_codes = sorted(pred_codes & panel_codes)
    logger.info("预测涉及 %d 只股票，panel 中可用 %d 只",
                len(pred_codes), len(needed_codes))

    if not needed_codes:
        logger.error("预测股票与 panel 无交集")
        return

    # 创建 Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(
        commission=commission + stamp_tax,
        mult=1.0,
        margin=None,
    )
    cerebro.broker.set_slippage_perc(slippage)

    # 添加数据 feeds
    feeds = create_data_feeds(panel, needed_codes)
    for feed in feeds:
        cerebro.adddata(feed)

    if not feeds:
        logger.error("未创建任何数据 feed")
        return

    # 添加策略（含 ATR 风控参数）
    atr_cfg = risk_cfg.get("atr", {})
    cerebro.addstrategy(
        SignalCsvStrategy,
        predictions=predictions,
        top_k=top_k,
        hold_days=hold_days,
        atr_window=atr_cfg.get("window", 14) if atr_cfg.get("enabled", False) else 0,
        atr_k_stop=atr_cfg.get("k_stop", 1.5),
        atr_k_trail_start=atr_cfg.get("k_trail_start", 2.0),
        atr_k_trail=atr_cfg.get("k_trail", 1.0),
    )

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe",
                        riskfreerate=0.03, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    # 运行
    logger.info("Backtrader 回测启动 (资金=%.0f, top_k=%d, hold=%d天)",
                initial_capital, top_k, hold_days)
    results = cerebro.run()
    strat = results[0]

    # 输出结果
    final_value = cerebro.broker.getvalue()
    total_return = (final_value / initial_capital - 1) * 100

    logger.info("=" * 50)
    logger.info("Backtrader 回测结果")
    logger.info("=" * 50)
    logger.info("  初始资金:   %.2f", initial_capital)
    logger.info("  最终净值:   %.2f", final_value)
    logger.info("  总收益率:   %.2f%%", total_return)

    try:
        sharpe = strat.analyzers.sharpe.get_analysis()
        sr = sharpe.get("sharperatio")
        logger.info("  年化 Sharpe: %s", f"{sr:.4f}" if sr else "N/A")
    except Exception:
        pass

    try:
        dd = strat.analyzers.drawdown.get_analysis()
        logger.info("  最大回撤:   %.2f%%", dd.get("max", {}).get("drawdown", 0))
    except Exception:
        pass

    try:
        trades = strat.analyzers.trades.get_analysis()
        total_trades = trades.get("total", {}).get("total", 0)
        won = trades.get("won", {}).get("total", 0)
        lost = trades.get("lost", {}).get("total", 0)
        logger.info("  总交易次数: %d (赢 %d / 亏 %d)", total_trades, won, lost)
    except Exception:
        pass

    # 保存报告
    report_dir = bt_cfg.get("report_dir", "output/backtest_reports")
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": ts,
        "initial_capital": initial_capital,
        "final_value": final_value,
        "total_return_pct": total_return,
        "top_k": top_k,
        "hold_days": hold_days,
    }
    report_df = pd.DataFrame([report])
    report_path = os.path.join(report_dir, f"bt_report_{ts}.csv")
    report_df.to_csv(report_path, index=False)
    logger.info("报告已保存: %s", report_path)

    if do_plot:
        try:
            cerebro.plot(style="candle", volume=False)
        except Exception as e:
            logger.warning("绘图失败: %s", e)

    return results


# ------------------------------------------------------------------
#  每日信号执行
# ------------------------------------------------------------------

def run_daily_backtest(cfg, panel: pd.DataFrame, trade_plan_path: str):
    """用 TradePlanStrategy 模拟执行 trade_plan。"""
    from quant_system.portfolio.bt_strategy import (
        TradePlanStrategy,
        create_data_feeds,
    )

    bt_cfg = cfg.get("backtrader", {})
    risk_cfg = cfg.get("risk", {})
    initial_capital = bt_cfg.get(
        "initial_capital", risk_cfg.get("initial_capital", 1_000_000)
    )

    # 加载交易计划
    plan_df = pd.read_csv(trade_plan_path)
    if plan_df.empty:
        logger.error("交易计划为空")
        return

    if "date" not in plan_df.columns:
        plan_df["date"] = pd.Timestamp.now().date()
    plan_df["date"] = pd.to_datetime(plan_df["date"]).dt.date

    trade_plans = {}
    for dt, grp in plan_df.groupby("date"):
        trade_plans[dt] = grp.to_dict("records")

    needed_codes = sorted(plan_df["ts_code"].unique())
    feeds = create_data_feeds(panel, needed_codes)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_capital)

    commission = bt_cfg.get("commission_pct", 0.0007)
    stamp_tax = bt_cfg.get("stamp_tax_pct", 0.0005)
    cerebro.broker.setcommission(commission=commission + stamp_tax)

    for feed in feeds:
        cerebro.adddata(feed)

    cerebro.addstrategy(TradePlanStrategy, trade_plans=trade_plans)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    logger.info("每日信号回测启动 (%d 条计划)", len(plan_df))
    results = cerebro.run()

    final_value = cerebro.broker.getvalue()
    logger.info("最终净值: %.2f (收益率: %.2f%%)",
                final_value, (final_value / initial_capital - 1) * 100)

    return results


# ------------------------------------------------------------------
#  辅助
# ------------------------------------------------------------------

def _normalize_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一预测 DataFrame 的列名：instrument→ts_code, datetime→date, pred_score→score。"""
    if "instrument" in df.columns and "ts_code" not in df.columns:
        df = df.rename(columns={"instrument": "ts_code"})
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    if "pred_score" in df.columns and "score" not in df.columns:
        df = df.rename(columns={"pred_score": "score"})
    df["date"] = pd.to_datetime(df["date"])
    return df


def _load_predictions(cfg) -> pd.DataFrame | None:
    """从缓存目录加载预测数据。"""
    cache_dir = cfg.get("output", {}).get("cache_dir", "output/cache")

    ensemble_path = os.path.join(cache_dir, "ensemble_predictions.csv")
    if os.path.exists(ensemble_path):
        df = _normalize_pred_columns(pd.read_csv(ensemble_path))
        logger.info("加载集成预测: %s (%d 行)", ensemble_path, len(df))
        return df

    all_preds = []
    for f in Path(cache_dir).glob("predictions_*.csv"):
        df = _normalize_pred_columns(pd.read_csv(f))
        all_preds.append(df)
        logger.info("加载模型预测: %s (%d 行)", f.name, len(df))

    if not all_preds:
        return None

    combined = pd.concat(all_preds, ignore_index=True)
    score_col = "score" if "score" in combined.columns else "pred_score"
    ensemble = (
        combined.groupby(["date", "ts_code"])[score_col]
        .mean()
        .reset_index()
        .rename(columns={score_col: "score"})
    )
    return ensemble


# ------------------------------------------------------------------
#  主入口
# ------------------------------------------------------------------

def main():
    args = parse_args()

    from quant_system.utils.config_manager import ConfigManager
    cfg = ConfigManager(args.config)

    # 加载 panel
    dp = cfg.get("data_pipeline", {})
    panel_path = str(Path(dp.get("panel_path", "data/panel.parquet")).expanduser())
    if not os.path.exists(panel_path):
        logger.error("Panel 不存在: %s，请先运行 run_data.py", panel_path)
        return

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])

    if args.mode == "historical":
        predictions = None
        if args.predictions:
            predictions = pd.read_csv(args.predictions)
            predictions["date"] = pd.to_datetime(predictions["date"])
        run_historical_backtest(cfg, panel, predictions)

    elif args.mode == "daily":
        if args.trade_plan:
            plan_path = args.trade_plan
        else:
            output_dir = cfg.get("output", {}).get("dir", "output")
            latest = _find_latest_trade_plan(output_dir)
            if not latest:
                logger.error("未找到 trade_plan.csv")
                return
            plan_path = latest
        run_daily_backtest(cfg, panel, plan_path)


def _find_latest_trade_plan(output_dir: str) -> str | None:
    """查找最新的 trade_plan.csv。"""
    from pathlib import Path
    candidates = sorted(
        Path(output_dir).glob("daily_*/trade_plan.csv"),
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


if __name__ == "__main__":
    main()
