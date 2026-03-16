#!/usr/bin/env python3
"""
多因子量化选股系统 - 全流程回测入口

=== 用法 ===

# 完整流程（加载数据 → walk-forward 回测 → 集成预测 → 输出）
python -m quant_system.run_strategy --config quant_system/configs/config.yaml

# 仅使用已有 panel（跳过数据更新）
python -m quant_system.run_strategy --config quant_system/configs/config.yaml --mode backtest_only

=== 全流程 ===

1. 加载 panel.parquet（需先运行 run_data.py 生成）
2. 加载 selected_features.txt（需先运行 run_data.py 生成，或自动执行因子选择）
3. Walk-Forward 回测（每个激活模型独立回测，输出全截面预测分）
4. 模型评分 + 加权集成预测
5. 输出 scores_*.csv（含股票名称）
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_strategy")


def parse_args():
    parser = argparse.ArgumentParser(description="多因子量化选股系统 - 全流程回测")
    parser.add_argument(
        "--config", type=str,
        default="quant_system/configs/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--mode", type=str,
        choices=["full", "backtest_only"],
        default="full",
        help="full=含数据更新, backtest_only=直接加载已有 panel",
    )
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    return parser.parse_args()


# ------------------------------------------------------------------
#  Step 1: 加载/准备数据
# ------------------------------------------------------------------

def step_1_load_data(cfg):
    """加载 panel.parquet + selected_features.txt。"""
    dp = cfg.get("data_pipeline", {})
    panel_path = str(Path(dp.get("panel_path", "data/panel.parquet")).expanduser())
    if not os.path.exists(panel_path):
        raise FileNotFoundError(
            f"Panel 不存在: {panel_path}，请先运行 run_data.py 生成"
        )

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])
    logger.info("Panel 加载: %d 行, %d 列", len(panel), panel.shape[1])

    output_dir = cfg.get("output", {}).get("dir", "output")
    feat_path = os.path.join(output_dir, "selected_features.txt")

    if os.path.exists(feat_path):
        with open(feat_path) as f:
            features = [line.strip() for line in f if line.strip()]
        features = [ft for ft in features if ft in panel.columns]
        logger.info("加载 %d 个选定因子 (来自 %s)", len(features), feat_path)
    else:
        logger.info("selected_features.txt 不存在，自动执行因子选择")
        features = _run_factor_selection(panel, cfg, output_dir)

    if not features:
        raise RuntimeError("无可用因子，请先运行 run_data.py")

    return panel, features


def _run_factor_selection(panel, cfg, output_dir):
    """在 panel 上执行因子选择并保存。"""
    from quant_system.factor_engine.factor_selector import select_factors

    fe_cfg = cfg.get("factor_engine", {})
    label_col = fe_cfg.get("label_col", "y_ret_5d_adj")
    if label_col not in panel.columns:
        logger.error("标签列 %s 不在 panel 中", label_col)
        return []

    selected, ic_table = select_factors(panel, label_col, cfg)

    os.makedirs(output_dir, exist_ok=True)
    feat_path = os.path.join(output_dir, "selected_features.txt")
    with open(feat_path, "w") as f:
        f.write("\n".join(selected))
    logger.info("因子选择完成: %d 个，已保存到 %s", len(selected), feat_path)

    if not ic_table.empty:
        reports_dir = cfg.get("output", {}).get("reports_dir", "output/reports")
        os.makedirs(reports_dir, exist_ok=True)
        ic_table.to_csv(os.path.join(reports_dir, "ic_table.csv"), index=False)

    return selected


# ------------------------------------------------------------------
#  Step 2: Walk-Forward 回测
# ------------------------------------------------------------------

def step_2_walk_forward(panel, features, cfg):
    """对每个激活模型做 walk-forward 回测（全截面预测）。"""
    from quant_system.trainer.walk_forward import WalkForwardEngine

    fe_cfg = cfg.get("factor_engine", {})
    label_col = fe_cfg.get("label_col", "y_ret_5d_adj")

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

        logger.info("===== Walk-Forward 回测: %s =====", model_key)
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
        logger.info(
            "  %s: Sharpe=%.3f, Return=%.2f%%, MaxDD=%.2f%%",
            model_key,
            stats.get("sharpe", 0),
            stats.get("cum_return", 0) * 100,
            stats.get("max_dd", 0) * 100,
        )

    if not all_results:
        raise RuntimeError("所有模型回测均失败")

    return all_results


# ------------------------------------------------------------------
#  Step 3: 模型评分 + 集成预测
# ------------------------------------------------------------------

def step_3_ensemble(all_results, cfg):
    """模型评分 → 加权 → 合并各模型全截面预测 → 每日排序。"""
    from quant_system.trainer.walk_forward import score_models

    scores_df = score_models(all_results, cfg)
    logger.info("===== 模型评分 =====")
    model_weights = {}
    for _, row in scores_df.iterrows():
        mk = row["model_key"]
        w = float(row["model_weight"])
        model_weights[mk] = w
        logger.info(
            "  %s: score=%.4f, weight=%.4f, icir=%.4f, rank_ic=%.4f",
            mk, row["score_total"], w,
            row.get("icir", 0), row.get("mean_rank_ic", 0),
        )

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

        # 截面 Z-Score 标准化后加权
        df["zscore"] = df.groupby("date")["raw_score"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )
        df["weighted_score"] = df["zscore"] * w
        df["model"] = mk
        all_preds.append(df[["date", "ts_code", "weighted_score", "model"]])

    if not all_preds:
        raise RuntimeError("无有效预测结果")

    combined = pd.concat(all_preds, ignore_index=True)
    ensemble = (
        combined.groupby(["date", "ts_code"])["weighted_score"]
        .sum()
        .reset_index()
        .rename(columns={"weighted_score": "score"})
    )
    ensemble = ensemble.sort_values(["date", "score"], ascending=[True, False])
    ensemble["rank"] = ensemble.groupby("date")["score"].rank(
        ascending=False, method="min"
    )

    logger.info("集成预测: %d 条记录, 覆盖 %d 个交易日",
                len(ensemble), ensemble["date"].nunique())

    # 保存模型评分和预测
    cache_dir = cfg.get("output", {}).get("cache_dir", "output/cache")
    os.makedirs(cache_dir, exist_ok=True)
    if not scores_df.empty:
        scores_df.to_csv(
            os.path.join(cache_dir, "model_scores_cache.csv"), index=False
        )
    for mk, result in all_results.items():
        preds = result.get("predictions")
        if isinstance(preds, pd.DataFrame) and not preds.empty:
            preds.to_csv(
                os.path.join(cache_dir, f"predictions_{mk}.csv"), index=False
            )

    return ensemble, scores_df


# ------------------------------------------------------------------
#  Step 4: 输出
# ------------------------------------------------------------------

def _get_name_map(ts_codes: list) -> dict:
    """获取 ts_code → 股票名称 映射。"""
    codes = [c for c in ts_codes if c and str(c) != "unknown"]
    if not codes:
        return {}

    def to_6digit(code):
        s = str(code).strip()
        if "." in s:
            return s.split(".")[0].zfill(6)
        if len(s) >= 8 and s[:2] in ("SH", "SZ", "BJ"):
            return s[2:8]
        return s.zfill(6)[:6] if len(s) <= 6 else s[:6]

    try:
        import akshare as ak
        info = ak.stock_info_a_code_name()
        if info is None or info.empty:
            return {}
        code_to_name = {}
        for _, row in info.iterrows():
            c = str(row["code"]).strip()
            c6 = c.split(".")[0].zfill(6) if "." in c else c.zfill(6)[:6]
            code_to_name[c6] = str(row["name"]).strip()
        return {tc: code_to_name.get(to_6digit(tc), "") for tc in codes}
    except Exception as e:
        logger.warning("获取股票名称失败: %s", e)
        return {}


def step_4_output(ensemble_df: pd.DataFrame, cfg):
    """输出 scores_*.csv。"""
    output_dir = cfg.get("output", {}).get("dir", "output")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out / f"scores_{timestamp}.csv"

    result = ensemble_df.copy()
    result = result.rename(columns={"ts_code": "instrument"})

    unique_insts = result["instrument"].dropna().unique().tolist()
    name_map = _get_name_map(unique_insts)
    result["name"] = result["instrument"].map(
        lambda x: name_map.get(x, "") if pd.notna(x) else ""
    )

    cols = ["date", "instrument", "name", "score", "rank"]
    result = result[[c for c in cols if c in result.columns]]
    result = result.rename(columns={"date": "datetime"})

    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("结果已保存: %s", csv_path)

    last_date = result["datetime"].max()
    top = result[result["datetime"] == last_date].head(20)
    logger.info("\n最新截面 Top 20 (%s):\n%s", last_date, top.to_string(index=False))

    return result


# ------------------------------------------------------------------
#  主流程
# ------------------------------------------------------------------

def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("多因子量化选股系统 - 全流程回测")
    logger.info("配置: %s | 模式: %s", args.config, args.mode)
    logger.info("=" * 60)

    from quant_system.utils.config_manager import ConfigManager
    cfg = ConfigManager(args.config)

    if args.mode == "full":
        logger.info("===== 数据管道 (run_data) =====")
        try:
            from quant_system.run_data import main as run_data_main
            run_data_main()
        except Exception as e:
            logger.warning("数据管道执行失败: %s，尝试使用已有 panel", e)

    # Step 1
    panel, features = step_1_load_data(cfg)

    # 可选日期过滤
    if args.start_date:
        sd = pd.Timestamp(args.start_date)
        panel = panel[panel["date"] >= sd]
    if args.end_date:
        ed = pd.Timestamp(args.end_date)
        panel = panel[panel["date"] <= ed]

    logger.info("回测区间: %s → %s (%d 行)",
                panel["date"].min().date(), panel["date"].max().date(), len(panel))

    # Step 2
    all_results = step_2_walk_forward(panel, features, cfg)

    # Step 3
    ensemble_df, scores_df = step_3_ensemble(all_results, cfg)

    # Step 4
    step_4_output(ensemble_df, cfg)

    # 可选: 生成交易计划（基于集成预测的最新截面）
    if cfg.get("turnover_control", {}).get("enabled", False):
        logger.info("===== 生成交易计划 =====")
        try:
            from quant_system.portfolio.trade_plan import generate_positions, build_trade_plan
            from quant_system.risk.atr_risk import ATRRiskConfig

            risk_cfg = cfg.get("risk", {})
            atr_enabled = risk_cfg.get("atr", {}).get("enabled", False)
            atr_config = ATRRiskConfig.from_cfg(cfg) if atr_enabled else None
            wf_cfg = cfg.get("walk_forward", {})
            hold_days = wf_cfg.get("hold_days", 5)
            top_k = wf_cfg.get("top_k", 5)

            latest_date = ensemble_df["date"].max()
            latest_recs = ensemble_df[ensemble_df["date"] == latest_date].copy()
            latest_recs = latest_recs.rename(columns={"score": "ensemble_score"})

            recs_top = latest_recs.head(top_k).copy()
            recs_top["signal_date"] = latest_date
            recs_top["model"] = "ensemble"

            positions = generate_positions(recs_top, panel, hold_days, atr_config)
            trade_plan = build_trade_plan(positions, latest_recs, panel, cfg)

            output_dir = cfg.get("output", {}).get("dir", "output")
            date_str = latest_date.strftime("%Y%m%d") if hasattr(latest_date, "strftime") else str(latest_date)[:10].replace("-", "")
            daily_dir = os.path.join(output_dir, f"daily_{date_str}")
            os.makedirs(daily_dir, exist_ok=True)

            if not trade_plan.empty:
                tp_path = os.path.join(daily_dir, "trade_plan.csv")
                trade_plan.to_csv(tp_path, index=False, encoding="utf-8-sig")
                logger.info("交易计划 (%d 条) → %s", len(trade_plan), tp_path)
            if not positions.empty:
                positions.to_csv(os.path.join(daily_dir, "positions.csv"), index=False, encoding="utf-8-sig")
        except Exception as e:
            logger.warning("交易计划生成失败: %s", e)

    # 可选: 自动触发 backtrader 回测
    bt_cfg = cfg.get("backtrader", {})
    if bt_cfg.get("enabled", False):
        logger.info("===== 触发 Backtrader 回测 =====")
        try:
            from quant_system.run_backtest import run_historical_backtest
            cache_dir = cfg.get("output", {}).get("cache_dir", "output/cache")
            run_historical_backtest(cfg, panel, ensemble_df)
        except Exception as e:
            logger.warning("Backtrader 回测失败: %s", e)

    logger.info("=" * 60)
    logger.info("全流程执行完毕")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
