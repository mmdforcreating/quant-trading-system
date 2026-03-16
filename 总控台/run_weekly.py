#!/usr/bin/env python3
"""
指挥官军训 / 周频调仓 (run_weekly)

职责：Walk-Forward 模型训练 → 模型评分 → 权重分配 → 保存模型 + 权重。

流程：
  1. 加载 panel.parquet + selected_features.txt
  2. 对每个 active_model 运行 WalkForwardEngine（全截面预测）
  3. score_models 综合评分
  4. _compute_rank_ic_weights 动态权重分配（负熔断 + cap + 兜底）
  5. 保存各模型 joblib + model_weights.json + model_scores_cache.csv
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent / "my_strategies"
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

logger = logging.getLogger("run_weekly")


# ── 从 quant_core-main/core/engine.py 移植 ──────────────

def _compute_rank_ic_weights(
    mean_rank_ic_per_model: dict,
    weight_cap: float = 0.40,
) -> dict:
    """
    滚动 RankIC 动态加权：负向熔断、等比归一、上限保护、兜底等权。

    Parameters
    ----------
    mean_rank_ic_per_model : {model_key: mean_rank_ic}
    weight_cap : float  单模型最高权重

    Returns
    -------
    {model_key: weight}，sum ≈ 1.0
    """
    keys = list(mean_rank_ic_per_model.keys())
    if not keys:
        return {}

    w = {}
    for k in keys:
        s = float(mean_rank_ic_per_model.get(k, np.nan))
        w[k] = max(0.0, s) if np.isfinite(s) else 0.0

    total = sum(w.values())
    if total <= 0:
        n = len(keys)
        return {k: 1.0 / n for k in keys}

    for k in keys:
        w[k] = w[k] / total

    cap = min(max(float(weight_cap), 0.0), 1.0)
    overflow = 0.0
    capped = {}
    for k in keys:
        if w[k] > cap:
            overflow += w[k] - cap
            capped[k] = cap
        else:
            capped[k] = w[k]

    if overflow <= 0:
        return w

    recipients = [k for k in keys if 0 < capped[k] < cap]
    if not recipients:
        s = sum(capped.values())
        if s > 1e-9:
            return {k: capped[k] / s for k in keys}
        return w
    per_recipient = overflow / len(recipients)
    for k in recipients:
        capped[k] = min(cap, capped[k] + per_recipient)

    s = sum(capped.values())
    if abs(s - 1.0) > 1e-6 and s > 1e-9:
        return {k: capped[k] / s for k in keys}
    return capped


# ── 主流程 ───────────────────────────────────────────

def weekly(cfg: dict, start_date: str | None = None, end_date: str | None = None):
    """
    周频 Walk-Forward 训练 + 模型权重分配。

    Parameters
    ----------
    cfg : dict        全局配置
    start_date : str  可选，覆盖 panel 起始日
    end_date : str    可选，覆盖 panel 截止日
    """
    from quant_system.trainer.walk_forward import WalkForwardEngine, score_models

    dp = cfg.get("data_pipeline", {})
    fe_cfg = cfg.get("factor_engine", {})
    scoring_cfg = cfg.get("scoring", {})
    label_col = fe_cfg.get("label_col", "y_ret_5d_adj")

    panel_path = str(Path(dp.get("panel_path", "data/panel.parquet")).expanduser())
    if not os.path.exists(panel_path):
        raise FileNotFoundError(
            f"Panel 不存在: {panel_path}，请先运行 `python quant_cli.py fetch --freq daily`"
        )

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])

    if start_date:
        panel = panel[panel["date"] >= pd.Timestamp(start_date)]
    if end_date:
        panel = panel[panel["date"] <= pd.Timestamp(end_date)]

    logger.info("Panel: %d 行, %s → %s",
                len(panel), panel["date"].min().date(), panel["date"].max().date())

    # ── 加载精选因子 ────────────────────────────────
    output_dir = cfg.get("output", {}).get("dir", "output")
    feat_path = os.path.join(output_dir, "selected_features.txt")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"因子列表不存在: {feat_path}，请先运行 `python quant_cli.py diagnose`"
        )

    with open(feat_path) as f:
        features = [line.strip() for line in f if line.strip()]
    features = [ft for ft in features if ft in panel.columns]
    logger.info("加载 %d 个因子", len(features))

    # ── Walk-Forward 回测 ───────────────────────────
    engine = WalkForwardEngine(cfg)
    active_models = cfg.get("active_models", ["catboost", "ridge"])
    all_results = {}
    last_models = {}

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

        logger.info("===== Walk-Forward: %s =====", model_key)
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
        logger.info("  %s: Sharpe=%.3f, Return=%.2f%%, MaxDD=%.2f%%",
                     model_key,
                     stats.get("sharpe", 0),
                     stats.get("cum_return", 0) * 100,
                     stats.get("max_dd", 0) * 100)

        # 用全量数据训练最终模型用于日频推断
        try:
            final_model = model_cls(**kwargs)
            avail = [ft for ft in features if ft in panel.columns]
            X_all = panel[avail].fillna(0).replace([np.inf, -np.inf], 0)
            y_all = panel[label_col].fillna(0)
            if model_key == "lambdarank":
                groups = panel.groupby("date").size().tolist()
                final_model.fit(X_all, y_all, groups=groups)
            else:
                final_model.fit(X_all, y_all)
            last_models[model_key] = final_model
        except Exception as e:
            logger.warning("最终模型 %s 训练失败: %s", model_key, e)

    if not all_results:
        raise RuntimeError("所有模型训练均失败")

    # ── 模型评分 ────────────────────────────────────
    scores_df = score_models(all_results, cfg)
    logger.info("===== 模型评分 =====")
    for _, row in scores_df.iterrows():
        logger.info("  %s: score=%.4f, rank_ic=%.4f",
                     row["model_key"], row["score_total"], row.get("mean_rank_ic", 0))

    # ── 权重分配（使用 ICIR + cap + 兜底）──────────
    weight_cap = scoring_cfg.get("weight_cap", 0.40)

    # score_models 已用 ICIR 计算 model_weight，此处直接读取并应用 cap 保护
    model_weights = {}
    for _, row in scores_df.iterrows():
        model_weights[row["model_key"]] = float(row["model_weight"])
    model_weights = _compute_rank_ic_weights(
        {mk: float(scores_df.loc[scores_df["model_key"] == mk, "icir"].iloc[0])
         for mk in model_weights if mk in scores_df["model_key"].values},
        weight_cap,
    )
    logger.info("===== 动态权重 (ICIR 加权) =====")
    for mk, w in model_weights.items():
        icir_val = float(scores_df.loc[scores_df["model_key"] == mk, "icir"].iloc[0]) if mk in scores_df["model_key"].values else 0
        logger.info("  %s: weight=%.4f (icir=%.4f)", mk, w, icir_val)

    scores_df["model_weight"] = scores_df["model_key"].map(model_weights).fillna(0)

    # ── 保存 ────────────────────────────────────────
    models_dir = cfg.get("output", {}).get("saved_models_dir", "output/saved_models")
    cache_dir = cfg.get("output", {}).get("cache_dir", "output/cache")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    for mk, model in last_models.items():
        path = os.path.join(models_dir, f"{mk}_latest.joblib")
        joblib.dump(model, path)
        logger.info("模型已保存: %s", path)

    scores_df.to_csv(os.path.join(cache_dir, "model_scores_cache.csv"), index=False)
    logger.info("评分缓存 → %s", os.path.join(cache_dir, "model_scores_cache.csv"))

    weights_path = os.path.join(cache_dir, "model_weights.json")
    with open(weights_path, "w") as f:
        json.dump(model_weights, f, indent=2)
    logger.info("权重 JSON → %s", weights_path)

    for mk, res in all_results.items():
        preds = res.get("predictions")
        if isinstance(preds, pd.DataFrame) and not preds.empty:
            preds.to_csv(os.path.join(cache_dir, f"predictions_{mk}.csv"), index=False)

    return scores_df, model_weights


# ── 统一入口 ─────────────────────────────────────────

def run(cfg: dict, **kwargs):
    """由 quant_cli.py 调用的统一入口。"""
    return weekly(
        cfg,
        start_date=kwargs.get("start_date"),
        end_date=kwargs.get("end_date"),
    )
