#!/usr/bin/env python3
"""
前线狙击手 / 日频执行 (run_daily)

职责：零训练极速推断 + 刚性风控 + 生成 trade_plan.csv。

流程：
  1. 可选增量更新 panel（仅拉当日）
  2. 加载 selected_features + model_weights.json + 各模型 joblib
  3. 各模型 predict → 当日预测分
  4. 投票融合（rank_decay 加权投票，借鉴 quant_core-main _ensemble_recommendations）
  5. 胜率微调
  6. 刚性风控：ATR 止损 / 追踪止盈 / 换手控制
  7. 产出：recs_best.csv, positions.csv, trade_plan.csv
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent / "my_strategies"
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

logger = logging.getLogger("run_daily")


# ── 从 quant_core-main 移植的投票融合 ────────────────

def _rank_decay(rank0: int) -> float:
    """排名衰减: 1/log2(2 + rank0)"""
    return float(1.0 / np.log2(2.0 + rank0))


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    x = x / max(float(temp), 1e-9)
    x = x - np.nanmax(x)
    ex = np.exp(np.clip(x, -50, 50))
    s = np.nansum(ex)
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(ex) / len(ex)
    return ex / s


def _compute_model_weights_from_json_or_fallback(
    model_preds: dict,
    weights_path: str,
    score_totals: dict | None = None,
    scheme: str = "rank_decay",
) -> dict:
    """
    优先读取 model_weights.json（周末产出），否则 fallback 到 softmax 或 rank_decay。
    移植自 quant_core-main/core/strategy._compute_model_weights。
    """
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            loaded = json.load(f)
        available = {k: loaded[k] for k in model_preds if k in loaded}
        if available:
            total = sum(available.values())
            if total > 0:
                return {k: v / total for k, v in available.items()}

    keys = list(model_preds.keys())
    n = len(keys)
    if n == 0:
        return {}

    if scheme == "rank_decay":
        weights_arr = np.array([1.0 / (i + 1) for i in range(n)])
        weights_arr = weights_arr / weights_arr.sum()
        return dict(zip(keys, weights_arr))

    if score_totals:
        scores = np.array([score_totals.get(k, 0) for k in keys], dtype=float)
        w = _softmax(scores, temp=1.0)
        return dict(zip(keys, w))

    return {k: 1.0 / n for k in keys}


def _ensemble_recommendations(
    model_preds: dict,
    model_weights: dict,
    per_model_topn: int = 10,
    final_topn: int = 5,
) -> pd.DataFrame:
    """
    Rank-decay 投票融合。
    移植自 quant_core-main/core/strategy._ensemble_recommendations。

    每个模型取 Top per_model_topn，按 model_weight * rank_decay(rank_in_model) 投票，
    汇总后按 ensemble_score 排序取 Top final_topn。
    """
    vote_scores: dict[str, float] = {}

    for mk, df in model_preds.items():
        w = model_weights.get(mk, 0)
        if w <= 0 or df.empty:
            continue
        top = df.nlargest(per_model_topn, "pred_score")
        for rank0, (_, row) in enumerate(top.iterrows()):
            code = row["ts_code"]
            contrib = w * _rank_decay(rank0)
            vote_scores[code] = vote_scores.get(code, 0) + contrib

    if not vote_scores:
        return pd.DataFrame()

    result = pd.DataFrame([
        {"ts_code": c, "ensemble_score": s}
        for c, s in vote_scores.items()
    ])
    result = result.sort_values("ensemble_score", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


# ── 主流程 ───────────────────────────────────────────

def daily(cfg: dict, skip_update: bool = False, target_date_str: str | None = None):
    """
    日频推断 + 风控 + trade_plan 生成。

    Parameters
    ----------
    cfg : dict
    skip_update : bool   跳过增量数据更新
    target_date_str : str  YYYYMMDD，默认用 panel 最新日
    """
    from quant_system.ensemble.model_ensemble import adjust_weights_by_winrate
    from quant_system.portfolio.trade_plan import generate_positions, build_trade_plan
    from quant_system.risk.atr_risk import ATRRiskConfig

    dp = cfg.get("data_pipeline", {})
    panel_path = str(Path(dp.get("panel_path", "data/panel.parquet")).expanduser())

    if not os.path.exists(panel_path):
        raise FileNotFoundError(
            f"Panel 不存在: {panel_path}，请先运行 `python quant_cli.py fetch --freq daily`"
        )

    # ── Step 0: 增量更新 ───────────────────────────
    if not skip_update:
        try:
            panel = _incremental_update(cfg, panel_path)
        except Exception as e:
            logger.warning("增量更新异常 (%s)，使用已有 panel", e)
            panel = _load_panel(panel_path)
    else:
        panel = _load_panel(panel_path)

    if target_date_str:
        target_date = pd.Timestamp(target_date_str)
    else:
        target_date = panel["date"].max()
    logger.info("目标日期: %s", target_date.date())

    # ── Step 1: 加载因子 / 权重 / 模型 ────────────
    output_dir = cfg.get("output", {}).get("dir", "output")
    cache_dir = cfg.get("output", {}).get("cache_dir", "output/cache")
    models_dir = cfg.get("output", {}).get("saved_models_dir", "output/saved_models")

    feat_path = os.path.join(output_dir, "selected_features.txt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"因子列表不存在: {feat_path}")

    with open(feat_path) as f:
        features = [line.strip() for line in f if line.strip()]
    features = [ft for ft in features if ft in panel.columns]
    logger.info("因子: %d 个", len(features))

    # ── Step 2: 各模型推断 ─────────────────────────
    active_models = cfg.get("active_models", ["catboost", "ridge"])
    df_today = panel[panel["date"] == target_date].copy()
    if df_today.empty:
        raise RuntimeError(f"目标日期 {target_date.date()} 无数据")
    logger.info("当日截面: %d 只", len(df_today))

    model_preds = {}
    for model_key in active_models:
        model_path = os.path.join(models_dir, f"{model_key}_latest.joblib")
        if not os.path.exists(model_path):
            logger.warning("模型文件不存在: %s", model_path)
            continue
        try:
            model = joblib.load(model_path)
            avail = [f for f in features if f in df_today.columns]
            X = df_today[avail].fillna(0).replace([np.inf, -np.inf], 0)
            scores = model.predict(X)
            pred_df = df_today[["ts_code"]].copy()
            pred_df["pred_score"] = scores.values if hasattr(scores, "values") else scores
            model_preds[model_key] = pred_df
            logger.info("  %s → %d 只", model_key, len(pred_df))
        except Exception as e:
            logger.error("模型 %s 推断失败: %s", model_key, e)

    if not model_preds:
        raise RuntimeError("所有模型推断均失败")

    # ── Step 3: 投票融合 ──────────────────────────
    ens_cfg = cfg.get("ensemble", {})
    per_model_topn = ens_cfg.get("per_model_topn", 10)
    final_topn = ens_cfg.get("final_topn", 5)
    weight_scheme = ens_cfg.get("weight_scheme", "rank_decay")

    weights_path = os.path.join(cache_dir, "model_weights.json")
    model_weights = _compute_model_weights_from_json_or_fallback(
        model_preds, weights_path, scheme=weight_scheme,
    )
    logger.info("模型权重: %s", {k: f"{v:.4f}" for k, v in model_weights.items()})

    # ── Step 4: 胜率微调 ──────────────────────────
    top_k = cfg.get("walk_forward", {}).get("top_k", 5)
    hold_days = cfg.get("walk_forward", {}).get("hold_days", 5)
    wr_cfg = cfg.get("winrate", {})

    recs_history_path = os.path.join(output_dir, "recs_history.csv")
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
                    logger.info("胜率微调后: %s", {k: f"{v:.4f}" for k, v in model_weights.items()})
        except Exception as e:
            logger.warning("胜率微调失败: %s", e)

    recs = _ensemble_recommendations(model_preds, model_weights, per_model_topn, final_topn)
    logger.info("集成推荐: %d 只", len(recs))

    # ── Step 5: 刚性风控 + trade_plan ────────────
    risk_cfg = cfg.get("risk", {})
    atr_enabled = risk_cfg.get("atr", {}).get("enabled", False)
    atr_config = ATRRiskConfig.from_cfg(cfg) if atr_enabled else None

    recs_history = pd.DataFrame()
    if os.path.exists(recs_history_path):
        recs_history = pd.read_csv(recs_history_path)
        recs_history["signal_date"] = pd.to_datetime(recs_history["signal_date"], errors="coerce")

    recs_top = recs.head(top_k).copy()
    recs_top["signal_date"] = target_date
    recs_top["model"] = "ensemble"

    models_history_parts = []
    for mk, pred_df in model_preds.items():
        topk = pred_df.nlargest(top_k, "pred_score").copy()
        topk["model"] = mk
        topk["signal_date"] = target_date
        models_history_parts.append(topk[["ts_code", "pred_score", "model", "signal_date"]])

    history_parts = [recs_history, recs_top] + models_history_parts
    new_history = pd.concat(history_parts, ignore_index=True)
    new_history = new_history.drop_duplicates(subset=["signal_date", "ts_code", "model"], keep="last")
    recent_cutoff = target_date - pd.Timedelta(days=hold_days * 3)
    new_history = new_history[new_history["signal_date"] >= recent_cutoff]

    positions = generate_positions(new_history, panel, hold_days, atr_config)
    trade_plan = build_trade_plan(positions, recs, panel, cfg)

    # ── Step 6: 保存产出 ─────────────────────────
    date_str = target_date.strftime("%Y%m%d")
    daily_dir = os.path.join(output_dir, f"daily_{date_str}")
    os.makedirs(daily_dir, exist_ok=True)

    recs.to_csv(os.path.join(daily_dir, "recs_best.csv"), index=False, encoding="utf-8-sig")
    if not positions.empty:
        positions.to_csv(os.path.join(daily_dir, "positions.csv"), index=False, encoding="utf-8-sig")
    if not trade_plan.empty:
        trade_plan.to_csv(os.path.join(daily_dir, "trade_plan.csv"), index=False, encoding="utf-8-sig")
    new_history.to_csv(recs_history_path, index=False)

    logger.info("\n===== 推荐 Top %d (%s) =====", top_k, date_str)
    for _, row in recs_top.iterrows():
        logger.info("  %s  score=%.4f", row["ts_code"], row.get("ensemble_score", 0))

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
            parts.append(f"({row.get('reason', '')})")
            logger.info("  %s", " | ".join(parts))

    logger.info("产出目录: %s", daily_dir)

    # 增量回填历史收益
    try:
        from quant_system.reporting.backfill import backfill_returns
        backfill_returns(output_dir, panel, hold_days)
    except Exception as e:
        logger.warning("收益回填跳过: %s", e)

    return recs, trade_plan


# ── 辅助函数 ─────────────────────────────────────────

def _load_panel(panel_path: str) -> pd.DataFrame:
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    return panel.dropna(subset=["date"])


def _incremental_update(cfg: dict, panel_path: str) -> pd.DataFrame:
    """增量拉取当日数据并合并到 panel。"""
    panel = _load_panel(panel_path)
    last_date = panel["date"].max()
    today = datetime.now().strftime("%Y%m%d")
    next_day = (last_date + pd.Timedelta(days=1)).strftime("%Y%m%d")

    if next_day > today:
        logger.info("Panel 已是最新 (%s)", last_date.date())
        return panel

    logger.info("增量更新: %s → %s", next_day, today)

    from quant_system.data_pipeline.stock_pool import get_stock_pool
    from quant_system.data_pipeline.tushare_fetcher import TuShareFetcher
    from quant_system.data_pipeline.panel_builder import build_panel

    ts_codes, _ = get_stock_pool(cfg)
    dp = cfg.get("data_pipeline", {})

    try:
        fetcher = TuShareFetcher(cfg)
        raw_data = fetcher.fetch_all(ts_codes=ts_codes, start_date=next_day, end_date=today, skip_fina=True)
    except Exception as e:
        logger.warning("增量拉取失败: %s", e)
        return panel

    if not any(not df.empty for df in raw_data.values()):
        logger.info("无新数据")
        return panel

    new_panel = build_panel(raw_data, panel_path + ".tmp", dp.get("start_date", "20200101"))
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


# ── 统一入口 ─────────────────────────────────────────

def run(cfg: dict, **kwargs):
    """由 quant_cli.py 调用的统一入口。"""
    return daily(
        cfg,
        skip_update=kwargs.get("skip_update", False),
        target_date_str=kwargs.get("date"),
    )
