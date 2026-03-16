#!/usr/bin/env python3
"""
端到端流水线验证脚本

从有分钟数据的股票中抽取 10 只, top_k=3, 依次跑完:
  阶段 1: 加载 panel + selected_features
  阶段 2: Walk-Forward 回测 (4 模型)
  阶段 3: 模型评分 + 集成预测
  阶段 4: 输出 scores CSV
  阶段 5: 交易计划生成
  阶段 6: Backtrader 历史回测
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_pipeline")

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

CONFIG_PATH = BASE_DIR / "quant_system" / "configs" / "config.yaml"
MINUTE_DIR = BASE_DIR / "data" / "minute" / "1min"
OUTPUT_DIR = BASE_DIR / "output" / "test_run"
N_STOCKS = 10


def load_cfg():
    from quant_system.utils.config_manager import ConfigManager
    cfg = ConfigManager(CONFIG_PATH)

    cfg["walk_forward"]["top_k"] = 3
    cfg["walk_forward"]["train_window_days"] = 180
    cfg["walk_forward"]["step_days"] = 10
    cfg["walk_forward"]["hold_days"] = 5

    cfg["output"]["dir"] = str(OUTPUT_DIR)
    cfg["output"]["cache_dir"] = str(OUTPUT_DIR / "cache")
    cfg["output"]["saved_models_dir"] = str(OUTPUT_DIR / "saved_models")
    cfg["output"]["reports_dir"] = str(OUTPUT_DIR / "reports")
    cfg["backtrader"]["report_dir"] = str(OUTPUT_DIR / "backtest_reports")
    cfg["backtrader"]["plot"] = False

    return cfg


def pick_test_stocks():
    """从有分钟数据的股票中选 N_STOCKS 只与 panel 匹配的。"""
    panel = pd.read_parquet(BASE_DIR / "data" / "panel.parquet")
    panel_codes = set(panel["ts_code"].unique())

    minute_files = sorted(f.stem for f in MINUTE_DIR.glob("*.parquet"))
    matched = []
    for s in minute_files:
        if s.startswith(("0", "3")):
            tc = s + ".SZ"
        elif s.startswith("6"):
            tc = s + ".SH"
        else:
            continue
        if tc in panel_codes:
            matched.append(tc)
        if len(matched) >= N_STOCKS:
            break

    logger.info("选取 %d 只测试股票: %s", len(matched), matched)
    return matched, panel


def banner(stage: int, title: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info("  阶段 %d: %s", stage, title)
    logger.info("=" * 60)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = load_cfg()

    test_codes, panel_full = pick_test_stocks()
    panel = panel_full[panel_full["ts_code"].isin(test_codes)].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])
    logger.info("过滤后 Panel: %d 行, %d 只股票, %s → %s",
                len(panel), panel["ts_code"].nunique(),
                panel["date"].min().date(), panel["date"].max().date())

    results = {}
    all_passed = True

    # ── 阶段 1: 加载因子 ──────────────────────────────────
    banner(1, "加载 selected_features")
    try:
        feat_path = BASE_DIR / "output" / "selected_features.txt"
        if not feat_path.exists():
            raise FileNotFoundError(f"因子列表不存在: {feat_path}")

        with open(feat_path) as f:
            features = [line.strip() for line in f if line.strip()]
        features = [ft for ft in features if ft in panel.columns]
        logger.info("可用因子: %d 个 (总 %d 个)",
                     len(features), sum(1 for _ in open(feat_path)))

        if not features:
            raise RuntimeError("无可用因子列")

        fe_cfg = cfg.get("factor_engine", {})
        label_col = fe_cfg.get("label_col", "y_ret_5d_adj")
        if label_col not in panel.columns:
            logger.warning("标签列 %s 不在 panel 中，尝试构造...", label_col)
            close_col = "close_adj" if "close_adj" in panel.columns else "close"
            panel = panel.sort_values(["ts_code", "date"]).reset_index(drop=True)
            panel[label_col] = panel.groupby("ts_code")[close_col].transform(
                lambda x: x.shift(-5) / x - 1
            )
            logger.info("已构造 %s (非空: %d)", label_col, panel[label_col].notna().sum())

        results["features"] = features
        logger.info("[阶段 1] PASS — %d 个因子", len(features))
    except Exception:
        traceback.print_exc()
        logger.error("[阶段 1] FAIL")
        all_passed = False
        return

    # ── 阶段 2: Walk-Forward 回测 ─────────────────────────
    banner(2, "Walk-Forward 回测 (4 模型)")
    try:
        from quant_system.trainer.walk_forward import WalkForwardEngine

        engine = WalkForwardEngine(cfg)
        active_models = cfg.get("active_models", ["catboost", "ridge"])
        all_results = {}

        for model_key in active_models:
            model_cfg = cfg.get("models", {}).get(model_key)
            if not model_cfg:
                logger.warning("模型 %s 无配置，跳过", model_key)
                continue

            module_path = model_cfg.get("module_path")
            class_name = model_cfg.get("class")
            kwargs = dict(model_cfg.get("kwargs", {}))

            try:
                mod = importlib.import_module(module_path)
                model_cls = getattr(mod, class_name)
            except Exception as e:
                logger.error("导入模型 %s 失败: %s", model_key, e)
                continue

            logger.info("----- Walk-Forward: %s -----", model_key)
            result = engine.run(
                df=panel,
                features=features,
                label_col=label_col,
                model_type=model_key,
                model_cls=model_cls,
                model_kwargs=kwargs,
                full_cross_section=True,
            )
            all_results[model_key] = result

            stats = result.get("stats_test", {}) or result.get("stats_val", {})
            logger.info("  %s: Sharpe=%.3f, Return=%.2f%%, MaxDD=%.2f%%, Turnover=%.4f",
                        model_key,
                        stats.get("sharpe", 0),
                        stats.get("cum_return", 0) * 100,
                        stats.get("max_dd", 0) * 100,
                        result.get("turnover", 0))

        if not all_results:
            raise RuntimeError("所有模型回测均失败")

        results["wf"] = all_results
        logger.info("[阶段 2] PASS — %d 个模型完成", len(all_results))
    except Exception:
        traceback.print_exc()
        logger.error("[阶段 2] FAIL")
        all_passed = False
        return

    # ── 阶段 3: 模型评分 + 集成预测 ───────────────────────
    banner(3, "模型评分 + 集成预测")
    try:
        from quant_system.trainer.walk_forward import score_models

        scores_df = score_models(all_results, cfg)
        logger.info("===== 模型评分 =====")
        model_weights = {}
        for _, row in scores_df.iterrows():
            mk = row["model_key"]
            w = float(row["model_weight"])
            model_weights[mk] = w
            logger.info("  %s: score=%.4f, weight=%.4f, icir=%.4f",
                        mk, row["score_total"], w, row.get("icir", 0))

        all_preds = []
        for mk, result in all_results.items():
            preds = result.get("predictions")
            if preds is None or (isinstance(preds, pd.DataFrame) and preds.empty):
                continue
            w = model_weights.get(mk, 0)
            if w <= 0:
                continue
            df = preds.copy()
            df = df.rename(columns={"pred_score": "raw_score"})
            df["zscore"] = df.groupby("date")["raw_score"].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            )
            df["weighted_score"] = df["zscore"] * w
            df["model"] = mk
            all_preds.append(df[["date", "ts_code", "weighted_score", "model"]])

        if not all_preds:
            raise RuntimeError("无有效预测结果")

        combined = pd.concat(all_preds, ignore_index=True)
        ensemble_df = (
            combined.groupby(["date", "ts_code"])["weighted_score"]
            .sum()
            .reset_index()
            .rename(columns={"weighted_score": "score"})
        )
        ensemble_df = ensemble_df.sort_values(["date", "score"], ascending=[True, False])
        ensemble_df["rank"] = ensemble_df.groupby("date")["score"].rank(
            ascending=False, method="min"
        )

        cache_dir = OUTPUT_DIR / "cache"
        os.makedirs(cache_dir, exist_ok=True)
        scores_df.to_csv(cache_dir / "model_scores_cache.csv", index=False)
        for mk, result in all_results.items():
            preds = result.get("predictions")
            if isinstance(preds, pd.DataFrame) and not preds.empty:
                preds.to_csv(cache_dir / f"predictions_{mk}.csv", index=False)

        results["ensemble"] = ensemble_df
        results["scores"] = scores_df
        logger.info("[阶段 3] PASS — %d 条集成预测, 覆盖 %d 个交易日",
                     len(ensemble_df), ensemble_df["date"].nunique())
    except Exception:
        traceback.print_exc()
        logger.error("[阶段 3] FAIL")
        all_passed = False
        return

    # ── 阶段 4: 输出 scores CSV ───────────────────────────
    banner(4, "输出 scores CSV")
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = OUTPUT_DIR / f"scores_{timestamp}.csv"

        out_df = ensemble_df.copy()
        out_df = out_df.rename(columns={"ts_code": "instrument", "date": "datetime"})
        cols = ["datetime", "instrument", "score", "rank"]
        out_df = out_df[[c for c in cols if c in out_df.columns]]
        out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        last_date = ensemble_df["date"].max()
        top = ensemble_df[ensemble_df["date"] == last_date].head(10)
        logger.info("最新截面 Top:\n%s", top.to_string(index=False))
        logger.info("[阶段 4] PASS — 已保存 %s", csv_path)
    except Exception:
        traceback.print_exc()
        logger.error("[阶段 4] FAIL")
        all_passed = False

    # ── 阶段 5: 交易计划生成 ──────────────────────────────
    banner(5, "交易计划生成")
    try:
        from quant_system.portfolio.trade_plan import generate_positions, build_trade_plan
        from quant_system.risk.atr_risk import ATRRiskConfig

        risk_cfg = cfg.get("risk", {})
        atr_enabled = risk_cfg.get("atr", {}).get("enabled", False)
        atr_config = ATRRiskConfig.from_cfg(cfg) if atr_enabled else None
        wf_cfg = cfg.get("walk_forward", {})
        hold_days = wf_cfg.get("hold_days", 5)
        top_k = wf_cfg.get("top_k", 3)

        latest_date = ensemble_df["date"].max()
        latest_recs = ensemble_df[ensemble_df["date"] == latest_date].copy()
        latest_recs = latest_recs.rename(columns={"score": "ensemble_score"})

        recs_top = latest_recs.head(top_k).copy()
        recs_top["signal_date"] = latest_date
        recs_top["model"] = "ensemble"

        positions = generate_positions(recs_top, panel, hold_days, atr_config)
        logger.info("持仓数: %d", len(positions))

        trade_plan = build_trade_plan(positions, latest_recs, panel, cfg)
        logger.info("交易计划:\n%s", trade_plan.to_string(index=False) if not trade_plan.empty else "(空)")

        daily_dir = OUTPUT_DIR / f"daily_{latest_date.strftime('%Y%m%d')}"
        os.makedirs(daily_dir, exist_ok=True)
        if not trade_plan.empty:
            trade_plan.to_csv(daily_dir / "trade_plan.csv", index=False, encoding="utf-8-sig")
        if not positions.empty:
            positions.to_csv(daily_dir / "positions.csv", index=False, encoding="utf-8-sig")

        results["trade_plan"] = trade_plan
        logger.info("[阶段 5] PASS — %d 条交易计划", len(trade_plan))
    except Exception:
        traceback.print_exc()
        logger.error("[阶段 5] FAIL")
        all_passed = False

    # ── 阶段 6: Backtrader 历史回测 ───────────────────────
    banner(6, "Backtrader 历史回测")
    try:
        from quant_system.run_backtest import run_historical_backtest

        run_historical_backtest(cfg, panel, ensemble_df)
        logger.info("[阶段 6] PASS")
    except Exception:
        traceback.print_exc()
        logger.error("[阶段 6] FAIL")
        all_passed = False

    # ── 汇总 ─────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    if all_passed:
        logger.info("  全部 6 个阶段通过!")
    else:
        logger.info("  存在失败阶段，请查看上方日志")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
