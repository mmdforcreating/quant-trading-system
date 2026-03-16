#!/usr/bin/env python3
"""
周频训练：Walk-Forward 回测 → 模型评分 → 保存模型 + 缓存

=== 用法 ===
python -m quant_system.run_train --config quant_system/configs/config.yaml

=== 流程 ===
1. 加载 panel + selected_features
2. 对每个激活模型做 walk-forward 回测
3. 多维评分（Sharpe/Return/DD/IC）
4. 输出 model_scores_cache.csv + 保存最佳模型
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
logger = logging.getLogger("run_train")


def parse_args():
    p = argparse.ArgumentParser(description="Walk-Forward 回测 + 模型评分")
    p.add_argument("--config", type=str, default="quant_system/configs/config.yaml")
    return p.parse_args()


def main():
    args = parse_args()

    from quant_system.utils.config_manager import ConfigManager
    cfg = ConfigManager(args.config)

    logger.info("=" * 60)
    logger.info("Walk-Forward 训练启动")
    logger.info("=" * 60)

    # Step 1: 加载 panel
    dp = cfg.get("data_pipeline", {})
    panel_path = str(Path(dp.get("panel_path", "data/panel.parquet")).expanduser())
    if not os.path.exists(panel_path):
        logger.error("Panel 不存在: %s，请先运行 run_data.py", panel_path)
        return

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])
    logger.info("Panel 加载: %d 行, %d 列", len(panel), panel.shape[1])

    # Step 2: 加载 selected_features
    output_dir = cfg.get("output", {}).get("dir", "output")
    feat_path = os.path.join(output_dir, "selected_features.txt")
    if not os.path.exists(feat_path):
        logger.error("selected_features.txt 不存在，请先运行 run_data.py")
        return

    with open(feat_path) as f:
        features = [line.strip() for line in f if line.strip()]
    features = [f for f in features if f in panel.columns]
    logger.info("加载 %d 个选定因子", len(features))

    if not features:
        logger.error("无可用因子")
        return

    fe_cfg = cfg.get("factor_engine", {})
    label_col = fe_cfg.get("label_col", "y_ret_5d_adj")
    if label_col not in panel.columns:
        logger.error("标签列 %s 不在 panel 中", label_col)
        return

    # Step 3: 对每个模型做 walk-forward
    from quant_system.trainer.walk_forward import WalkForwardEngine, score_models

    engine = WalkForwardEngine(cfg)
    active_models = cfg.get("active_models", ["catboost", "ridge"])

    all_results = {}
    trained_models = {}

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

        logger.info("===== 回测模型: %s =====", model_key)
        result = engine.run(
            df=panel,
            features=features,
            label_col=label_col,
            model_type=model_key,
            model_cls=model_cls,
            model_kwargs=kwargs,
        )

        all_results[model_key] = result

        stats = result.get("stats_test", {}) or result.get("stats_val", {})
        logger.info("  Sharpe=%.3f, Return=%.2f%%, MaxDD=%.2f%%",
                     stats.get("sharpe", 0),
                     stats.get("cum_return", 0) * 100,
                     stats.get("max_dd", 0) * 100)

        # 用全量数据训练最终模型
        try:
            final_model = model_cls(**kwargs)
            train_df = panel.dropna(subset=[label_col])
            avail = [f for f in features if f in train_df.columns]
            if model_key == "lambdarank":
                groups = train_df.groupby("date").size().tolist()
                final_model.fit(train_df[avail], train_df[label_col], groups=groups)
            else:
                final_model.fit(train_df[avail], train_df[label_col])
            trained_models[model_key] = final_model
        except Exception as e:
            logger.warning("最终模型 %s 训练失败: %s", model_key, e)

    # Step 4: 模型评分
    scores_df = score_models(all_results, cfg)
    logger.info("\n===== 模型评分 =====")
    if not scores_df.empty:
        for _, row in scores_df.iterrows():
            logger.info("  %s: score=%.4f, weight=%.4f, rank_ic=%.4f",
                         row["model_key"], row["score_total"],
                         row["model_weight"], row.get("mean_rank_ic", 0))

    # Step 5: 保存
    cache_dir = cfg.get("output", {}).get("cache_dir", "output/cache")
    models_dir = cfg.get("output", {}).get("saved_models_dir", "output/saved_models")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, "model_scores_cache.csv")
    if not scores_df.empty:
        scores_df["cache_generated_at"] = datetime.now().isoformat()
        scores_df.to_csv(cache_path, index=False)
        logger.info("模型评分缓存: %s", cache_path)

    for mk, model in trained_models.items():
        model_path = os.path.join(models_dir, f"{mk}_latest.joblib")
        try:
            joblib.dump(model, model_path)
            logger.info("模型保存: %s", model_path)
        except Exception as e:
            logger.warning("模型 %s 保存失败: %s", mk, e)

    # 保存预测记录
    for mk, res in all_results.items():
        preds = res.get("predictions")
        if isinstance(preds, pd.DataFrame) and not preds.empty:
            pred_path = os.path.join(cache_dir, f"predictions_{mk}.csv")
            preds.to_csv(pred_path, index=False)

    logger.info("=" * 60)
    logger.info("训练完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
