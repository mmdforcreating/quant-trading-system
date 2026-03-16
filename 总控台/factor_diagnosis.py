#!/usr/bin/env python3
"""
因子检验与 X 光室 (factor_diagnosis)

职责：只看数学有效性，绝对不碰机器学习。

流程：
  1. 加载 panel.parquet
  2. 计算自研因子（资金流/估值/质量/裂变/量价/截面）
  3. 可选计算 Alpha158（Qlib）
  4. 对每列因子做 IC / ICIR 体检、VIF 共线性剔除
  5. 产出 selected_features.txt（精简黄金特征表）+ ic_table.csv
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent / "my_strategies"
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

logger = logging.getLogger("factor_diagnosis")


def diagnose(cfg: dict, skip_alpha158: bool = False, method: str | None = None):
    """
    因子计算 + 数学诊断 + 筛选。

    Parameters
    ----------
    cfg : dict           全局配置
    skip_alpha158 : bool 跳过 Alpha158 计算
    method : str | None  因子筛选方法覆盖（greedy / ic_vif），None 则用配置
    """
    from quant_system.factor_engine.custom_factors import compute_all_factors
    from quant_system.factor_engine.factor_selector import select_factors

    dp = cfg.get("data_pipeline", {})
    fe_cfg = cfg.get("factor_engine", {})
    panel_path = str(Path(dp.get("panel_path", "data/panel.parquet")).expanduser())

    if not os.path.exists(panel_path):
        raise FileNotFoundError(
            f"Panel 不存在: {panel_path}，请先运行 `python quant_cli.py fetch --freq daily`"
        )

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel = panel.dropna(subset=["date"])
    logger.info("Panel 加载: %d 行, %d 列", len(panel), panel.shape[1])

    # ── Step 1: 自研因子 ────────────────────────────
    if fe_cfg.get("use_custom", True):
        panel = compute_all_factors(panel, cfg)
        panel.to_parquet(panel_path, index=False)
        logger.info("自研因子计算完成: %d 列", panel.shape[1])

    # ── Step 2: Alpha158（可选）──────────────────────
    if fe_cfg.get("use_alpha158", False) and not skip_alpha158:
        try:
            from quant_system.data_pipeline.qlib_alpha158 import (
                compute_alpha158,
                merge_alpha158_to_panel,
            )
            alpha_df = compute_alpha158(panel, cfg)
            if alpha_df is not None and not alpha_df.empty:
                panel = merge_alpha158_to_panel(panel, alpha_df)
                panel.to_parquet(panel_path, index=False)
                logger.info("Alpha158 因子已合并: %d 列", panel.shape[1])
            else:
                logger.warning("Alpha158 计算返回空，跳过")
        except Exception as e:
            logger.warning("Alpha158 计算失败 (%s)，继续使用自研因子", e)

    # ── Step 3: IC / ICIR / VIF 筛选 ────────────────
    label_col = fe_cfg.get("label_col", "y_ret_5d_adj")
    if label_col not in panel.columns:
        logger.error("标签列 %s 不在 panel 中，无法做因子诊断", label_col)
        return [], pd.DataFrame()

    if method:
        cfg_override = dict(cfg)
        fe_override = dict(cfg_override.get("factor_engine", {}))
        sel_override = dict(fe_override.get("selection", {}))
        sel_override["method"] = method
        fe_override["selection"] = sel_override
        cfg_override["factor_engine"] = fe_override
        selected, ic_table = select_factors(panel, label_col, cfg_override)
    else:
        selected, ic_table = select_factors(panel, label_col, cfg)

    # ── Step 4: 保存产出 ────────────────────────────
    output_dir = cfg.get("output", {}).get("dir", "output")
    reports_dir = cfg.get("output", {}).get("reports_dir", "output/reports")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    features_path = os.path.join(output_dir, "selected_features.txt")
    with open(features_path, "w") as f:
        f.write("\n".join(selected))
    logger.info("选定因子 (%d 个) → %s", len(selected), features_path)

    if not ic_table.empty:
        ic_path = os.path.join(reports_dir, "ic_table.csv")
        ic_table.to_csv(ic_path, index=False)
        logger.info("IC 统计表 → %s", ic_path)

    if selected:
        logger.info("前 10 个因子: %s", selected[:10])

    return selected, ic_table


# ── 统一入口 ─────────────────────────────────────────

def run(cfg: dict, **kwargs):
    """由 quant_cli.py 调用的统一入口。"""
    return diagnose(
        cfg,
        skip_alpha158=kwargs.get("skip_alpha158", False),
        method=kwargs.get("method"),
    )
