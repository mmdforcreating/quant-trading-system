#!/usr/bin/env python3
"""
每日推荐：增量数据 → 加载模型 → 预测 → 集成(含胜率微调) → 推荐 + 交易计划

=== 用法 ===
python -m quant_system.run_daily --config quant_system/configs/config.yaml
python -m quant_system.run_daily --config quant_system/configs/config.yaml --skip-update

=== 流程 ===
0. 增量拉取当日数据，更新 panel（可 --skip-update 跳过）
1. 加载 panel + selected_features + 模型权重缓存
2. 加载已保存的模型（零训练）
3. 各模型对最新截面预测
4. 胜率微调模型权重
5. 集成投票 → 推荐列表
6. 生成持仓 + 交易计划（含 ATR 止损/止盈）
7. 输出 recs_best.csv, positions.csv, trade_plan.csv
8. 增量回填历史收益
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_daily")


def parse_args():
    p = argparse.ArgumentParser(description="每日推荐")
    p.add_argument("--config", type=str, default="quant_system/configs/config.yaml")
    p.add_argument("--date", type=str, default=None, help="指定日期 (YYYYMMDD)，默认用 panel 最新日")
    p.add_argument("--skip-update", action="store_true", help="跳过增量数据更新")
    return p.parse_args()


def _incremental_update(cfg, panel_path: str) -> pd.DataFrame:
    """增量拉取最新数据，更新 panel 并返回。"""
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])

    last_date = panel["date"].max()
    today = datetime.now().strftime("%Y%m%d")
    next_day = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")

    if next_day > today:
        logger.info("Panel 已是最新 (%s)，无需增量更新", last_date.date())
        return panel

    logger.info("增量更新: %s → %s", next_day, today)

    dp = cfg.get("data_pipeline", {})

    from quant_system.data_pipeline.stock_pool import get_stock_pool
    ts_codes, _ = get_stock_pool(cfg)

    from quant_system.data_pipeline.tushare_fetcher import TuShareFetcher
    try:
        fetcher = TuShareFetcher(cfg)
        raw_data = fetcher.fetch_all(
            ts_codes=ts_codes, start_date=next_day, end_date=today,
            skip_fina=True,
        )
    except Exception as e:
        logger.warning("增量拉取失败: %s，使用已有 panel", e)
        return panel

    has_new = any(not df.empty for df in raw_data.values())
    if not has_new:
        logger.info("无新数据")
        return panel

    from quant_system.data_pipeline.panel_builder import build_panel
    start_date = dp.get("start_date", "20200101")
    new_panel = build_panel(raw_data, panel_path + ".tmp", start_date)

    if not new_panel.empty:
        new_panel["date"] = pd.to_datetime(new_panel["date"], errors="coerce")
        combined = pd.concat([panel, new_panel], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ts_code", "date"])
        combined = combined.sort_values(["ts_code", "date"]).reset_index(drop=True)

        from quant_system.factor_engine.custom_factors import compute_all_factors
        combined = compute_all_factors(combined, cfg)
        combined.to_parquet(panel_path, index=False)
        logger.info("Panel 增量更新完成: %d 行", len(combined))

        tmp = panel_path + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        return combined

    return panel


def main():
    args = parse_args()

    from quant_system.utils.config_manager import ConfigManager
    cfg = ConfigManager(args.config)

    logger.info("=" * 60)
    logger.info("每日推荐启动")
    logger.info("=" * 60)

    # Step 0: 增量数据更新
    dp = cfg.get("data_pipeline", {})
    panel_path = str(Path(dp.get("panel_path", "data/panel.parquet")).expanduser())
    if not os.path.exists(panel_path):
        logger.error("Panel 不存在: %s，请先运行 run_data.py", panel_path)
        return

    if not args.skip_update:
        try:
            panel = _incremental_update(cfg, panel_path)
        except Exception as e:
            logger.warning("增量更新异常: %s，使用已有 panel", e)
            panel = pd.read_parquet(panel_path)
            panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
            panel = panel.dropna(subset=["date"])
    else:
        panel = pd.read_parquet(panel_path)
        panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
        panel = panel.dropna(subset=["date"])

    if args.date:
        target_date = pd.Timestamp(args.date)
    else:
        target_date = panel["date"].max()
    logger.info("目标日期: %s", target_date.date())

    # Step 1: 加载因子列表
    output_dir = cfg.get("output", {}).get("dir", "output")
    feat_path = os.path.join(output_dir, "selected_features.txt")
    if not os.path.exists(feat_path):
        logger.error("selected_features.txt 不存在，请先运行 run_data.py")
        return

    with open(feat_path) as f:
        features = [line.strip() for line in f if line.strip()]
    features = [f for f in features if f in panel.columns]
    logger.info("加载 %d 个选定因子", len(features))

    # Step 2: 加载模型权重缓存
    cache_dir = cfg.get("output", {}).get("cache_dir", "output/cache")
    models_dir = cfg.get("output", {}).get("saved_models_dir", "output/saved_models")

    from quant_system.ensemble.model_ensemble import ModelEnsemble, adjust_weights_by_winrate
    ensemble = ModelEnsemble(cfg)

    cache_path = os.path.join(cache_dir, "model_scores_cache.csv")
    model_weights = ensemble.load_weights_from_cache(cache_path)
    if model_weights:
        logger.info("加载模型权重: %s", {k: f"{v:.4f}" for k, v in model_weights.items()})
    else:
        logger.info("无权重缓存，使用等权")

    # Step 3: 加载模型并预测
    active_models = cfg.get("active_models", ["catboost", "ridge"])
    df_today = panel[panel["date"] == target_date].copy()
    if df_today.empty:
        logger.error("目标日期 %s 无数据", target_date.date())
        return
    logger.info("当日截面: %d 只股票", len(df_today))

    model_preds = {}
    for model_key in active_models:
        model_path = os.path.join(models_dir, f"{model_key}_latest.joblib")
        if not os.path.exists(model_path):
            logger.warning("模型 %s 不存在: %s", model_key, model_path)
            continue

        try:
            model = joblib.load(model_path)
            avail = [f for f in features if f in df_today.columns]
            X = df_today[avail].fillna(0).replace([np.inf, -np.inf], 0)
            scores = model.predict(X)
            pred_df = df_today[["ts_code"]].copy()
            pred_df["pred_score"] = scores.values if hasattr(scores, "values") else scores
            model_preds[model_key] = pred_df
            logger.info("  %s: 预测 %d 只", model_key, len(pred_df))
        except Exception as e:
            logger.error("模型 %s 预测失败: %s", model_key, e)

    if not model_preds:
        logger.error("无模型预测结果")
        return

    # Step 3.5: 构建各模型 Top-N 明细（供胜率按模型生效）
    per_model_topn = cfg.get("ensemble", {}).get("per_model_topn", 10)
    top_k = cfg.get("walk_forward", {}).get("top_k", 5)
    hold_days = cfg.get("walk_forward", {}).get("hold_days", 5)

    models_detail_parts = []
    models_history_parts = []
    for mk, pred_df in model_preds.items():
        topn = pred_df.nlargest(per_model_topn, "pred_score").copy()
        topn["model"] = mk
        topn["signal_date"] = target_date
        topn["rank_in_model"] = range(1, len(topn) + 1)
        models_detail_parts.append(topn)

        topk = pred_df.nlargest(top_k, "pred_score").copy()
        topk["model"] = mk
        topk["signal_date"] = target_date
        models_history_parts.append(topk[["ts_code", "pred_score", "model", "signal_date"]])

    recs_models_detail = pd.concat(models_detail_parts, ignore_index=True) if models_detail_parts else pd.DataFrame()

    # Step 4: 胜率微调模型权重
    if not model_weights:
        model_weights = {k: 1.0 / len(model_preds) for k in model_preds}

    recs_history_path = os.path.join(output_dir, "recs_history.csv")
    wr_cfg = cfg.get("winrate", {})
    if wr_cfg.get("enable", False) and os.path.exists(recs_history_path):
        try:
            from quant_system.reporting.winrate import (
                compute_forward_returns,
                compute_model_winrates,
            )
            rh = pd.read_csv(recs_history_path)
            rh["signal_date"] = pd.to_datetime(rh["signal_date"], errors="coerce")

            lookback = wr_cfg.get("lookback_days", 90)
            cutoff = target_date - pd.Timedelta(days=lookback)
            rh_recent = rh[rh["signal_date"] >= cutoff]
            rh_models = rh_recent[rh_recent["model"] != "ensemble"]

            if not rh_models.empty:
                rh_with_ret = compute_forward_returns(rh_models, panel, hold_days)
                winrate_stats, sample_counts = compute_model_winrates(
                    rh_with_ret,
                    cost_threshold=wr_cfg.get("cost_threshold", 0.002),
                    model_col="model",
                    min_samples=wr_cfg.get("min_samples", 20),
                )
                if winrate_stats:
                    model_weights = adjust_weights_by_winrate(
                        model_weights, winrate_stats,
                        sample_counts=sample_counts, cfg=cfg,
                    )
                    logger.info("胜率微调后权重: %s", {k: f"{v:.4f}" for k, v in model_weights.items()})
        except Exception as e:
            logger.warning("胜率微调失败: %s，使用原始权重", e)

    # Step 5: 集成
    recs = ensemble.ensemble_predictions(model_preds, model_weights)
    logger.info("集成推荐: %d 只", len(recs))

    # Step 6: 生成持仓与交易计划
    risk_cfg = cfg.get("risk", {})
    atr_enabled = risk_cfg.get("atr", {}).get("enabled", False)

    from quant_system.risk.atr_risk import ATRRiskConfig
    atr_config = ATRRiskConfig.from_cfg(cfg) if atr_enabled else None

    recs_history_path = os.path.join(output_dir, "recs_history.csv")
    recs_history = pd.DataFrame()
    if os.path.exists(recs_history_path):
        recs_history = pd.read_csv(recs_history_path)
        recs_history["signal_date"] = pd.to_datetime(recs_history["signal_date"], errors="coerce")

    recs_top = recs.head(top_k).copy()
    recs_top["signal_date"] = target_date
    recs_top["model"] = "ensemble"

    history_parts = [recs_history, recs_top]
    if models_history_parts:
        history_parts.extend(models_history_parts)
    new_history = pd.concat(history_parts, ignore_index=True)
    new_history = new_history.drop_duplicates(subset=["signal_date", "ts_code", "model"], keep="last")
    recent_cutoff = target_date - pd.Timedelta(days=hold_days * 3)
    new_history = new_history[new_history["signal_date"] >= recent_cutoff]

    from quant_system.portfolio.trade_plan import generate_positions, build_trade_plan
    positions = generate_positions(new_history, panel, hold_days, atr_config)
    trade_plan = build_trade_plan(positions, recs, panel, cfg)

    # Step 7: 保存
    date_str = target_date.strftime("%Y%m%d")
    daily_dir = os.path.join(output_dir, f"daily_{date_str}")
    os.makedirs(daily_dir, exist_ok=True)

    recs.to_csv(os.path.join(daily_dir, "recs_best.csv"), index=False, encoding="utf-8-sig")
    if not recs_models_detail.empty:
        recs_models_detail.to_csv(os.path.join(daily_dir, "recs_models_detail.csv"), index=False, encoding="utf-8-sig")
    if not positions.empty:
        positions.to_csv(os.path.join(daily_dir, "positions.csv"), index=False, encoding="utf-8-sig")
    if not trade_plan.empty:
        trade_plan.to_csv(os.path.join(daily_dir, "trade_plan.csv"), index=False, encoding="utf-8-sig")

    new_history.to_csv(recs_history_path, index=False)

    # 打印推荐
    logger.info("\n===== 推荐 Top %d (%s) =====", top_k, date_str)
    for _, row in recs_top.iterrows():
        logger.info("  %s  score=%.4f", row["ts_code"], row["ensemble_score"])

    if not trade_plan.empty:
        logger.info("\n===== 交易计划 =====")
        for _, row in trade_plan.iterrows():
            parts = [f"{row['action']:4s} {row['ts_code']}"]
            if pd.notna(row.get("suggested_price")):
                parts.append(f"price={row['suggested_price']:.2f}")
            if pd.notna(row.get("stop_loss")):
                parts.append(f"SL={row['stop_loss']:.2f}")
            if pd.notna(row.get("take_profit")):
                parts.append(f"TP={row['take_profit']:.2f}")
            if pd.notna(row.get("position_size")) and row.get("position_size", 0) > 0:
                parts.append(f"qty={int(row['position_size'])}")
            parts.append(f"({row.get('reason', '')})")
            logger.info("  %s", " | ".join(parts))

    logger.info("\n输出目录: %s", daily_dir)

    # Step 8: 增量回填历史收益
    try:
        from quant_system.reporting.backfill import backfill_returns
        backfill_returns(output_dir, panel, hold_days)
    except Exception as e:
        logger.warning("收益回填失败: %s", e)

    logger.info("=" * 60)
    logger.info("每日推荐完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
