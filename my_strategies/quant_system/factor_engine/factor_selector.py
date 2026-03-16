"""
因子选择器

支持两种模式：
- greedy: 贪心选择（IC/ICIR + VIF + 相关性阈值 + 族限制），移植自 quant_core
- ic_vif: 简单 IC/ICIR + VIF 筛选（原 quant_system 逻辑）
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

FAMILY_MAP = {
    "mom": ["mom_5d_adj", "mom_10d_adj", "mom_20d_adj", "mom_60d_adj", "rev_1d", "reversal_spread_20d_5d"],
    "mf": ["mf_to_amount", "mf_5d_strength", "mf_20d_strength", "net_mf_amount"],
    "big": ["big_net_to_amount", "big_5d_strength", "big_20d_strength"],
}
MAX_PER_FAMILY = 2


def select_factors(
    df: pd.DataFrame,
    label_col: str,
    cfg: dict,
    date_col: str = "date",
) -> Tuple[List[str], pd.DataFrame]:
    """
    主入口：根据配置选择因子。

    Returns
    -------
    (selected_features, ic_table)
    """
    sel_cfg = cfg.get("factor_engine", {}).get("selection", {})
    method = sel_cfg.get("method", "greedy")

    feature_cols = [
        c for c in df.columns
        if c not in (
            "ts_code", "date", "trade_date", label_col, "ret_1d_fwd",
            "adj_factor", "name", "ann_date", "end_date",
        )
        and not c.startswith("y_ret_")
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]

    if not feature_cols:
        logger.warning("无候选因子列")
        return [], pd.DataFrame()

    ic_table = make_ic_table(df, feature_cols, label_col, date_col)

    if method == "greedy":
        selected = greedy_select(
            df=df,
            candidates=feature_cols,
            ic_table=ic_table,
            label_col=label_col,
            date_col=date_col,
            min_abs_icir=sel_cfg.get("min_abs_icir", 0.02),
            max_vif=sel_cfg.get("max_vif", 100.0),
            corr_threshold=sel_cfg.get("corr_threshold", 0.95),
            min_coverage=sel_cfg.get("min_coverage", 0.45),
            min_keep=sel_cfg.get("min_keep", 5),
            max_keep=sel_cfg.get("max_keep", 80),
        )
    else:
        selected = ic_vif_select(
            df=df,
            candidates=feature_cols,
            ic_table=ic_table,
            ic_threshold=sel_cfg.get("min_abs_icir", 0.02),
            vif_threshold=sel_cfg.get("max_vif", 10.0),
        )

    logger.info("因子选择完成: %d / %d (method=%s)", len(selected), len(feature_cols), method)
    return selected, ic_table


def make_ic_table(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """计算每个因子的 IC/ICIR 统计。"""
    records = []
    dates = df[date_col].unique()

    for col in feature_cols:
        ics = []
        valid_dates = 0
        for dt in dates:
            sub = df[df[date_col] == dt][[col, label_col]].dropna()
            if len(sub) < 5:
                continue
            valid_dates += 1
            corr, _ = spearmanr(sub[col], sub[label_col])
            if np.isfinite(corr):
                ics.append(corr)

        if not ics:
            records.append({
                "factor": col, "IC_mean": 0.0, "IC_std": 0.0,
                "ICIR": 0.0, "absICIR": 0.0, "coverage": 0.0,
            })
            continue

        ic_mean = np.mean(ics)
        ic_std = np.std(ics) if len(ics) > 1 else 1e-10
        icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
        coverage = valid_dates / max(len(dates), 1)
        records.append({
            "factor": col,
            "IC_mean": ic_mean,
            "IC_std": ic_std,
            "ICIR": icir,
            "absICIR": abs(icir),
            "coverage": coverage,
        })

    return pd.DataFrame(records)


def greedy_select(
    df: pd.DataFrame,
    candidates: List[str],
    ic_table: pd.DataFrame,
    label_col: str,
    date_col: str = "date",
    min_abs_icir: float = 0.02,
    max_vif: float = 100.0,
    corr_threshold: float = 0.95,
    min_coverage: float = 0.45,
    min_keep: int = 5,
    max_keep: int = 80,
) -> List[str]:
    """贪心因子选择：按 absICIR 降序，依次加入不高相关的因子。"""
    qualified = ic_table[
        (ic_table["absICIR"] >= min_abs_icir) &
        (ic_table["coverage"] >= min_coverage)
    ].sort_values("absICIR", ascending=False)

    if qualified.empty:
        logger.warning("无因子通过 IC/coverage 阈值")
        return candidates[:min_keep] if len(candidates) >= min_keep else candidates

    qualified_names = qualified["factor"].tolist()

    sample = df[qualified_names].dropna()
    if len(sample) > 50000:
        sample = sample.sample(50000, random_state=42)
    if len(sample) < 10:
        return qualified_names[:max_keep]

    corr_matrix = sample.rank().corr(method="spearman")

    selected = []
    family_count: Dict[str, int] = {k: 0 for k in FAMILY_MAP}

    for factor in qualified_names:
        if len(selected) >= max_keep:
            break

        fam = _get_family(factor)
        if fam and family_count.get(fam, 0) >= MAX_PER_FAMILY:
            continue

        if selected:
            max_corr = corr_matrix.loc[factor, selected].abs().max()
            if max_corr >= corr_threshold:
                continue

        selected.append(factor)
        if fam:
            family_count[fam] = family_count.get(fam, 0) + 1

    if len(selected) < min_keep:
        for factor in qualified_names:
            if factor not in selected:
                selected.append(factor)
            if len(selected) >= min_keep:
                break

    return selected


def ic_vif_select(
    df: pd.DataFrame,
    candidates: List[str],
    ic_table: pd.DataFrame,
    ic_threshold: float = 0.02,
    vif_threshold: float = 10.0,
) -> List[str]:
    """简单 IC + VIF 筛选。"""
    passed = ic_table[ic_table["absICIR"] >= ic_threshold]["factor"].tolist()
    if len(passed) <= 1:
        return passed

    from statsmodels.stats.outliers_influence import variance_inflation_factor

    sample = df[passed].dropna()
    if len(sample) > 50000:
        sample = sample.sample(50000, random_state=42)
    if len(sample) < 10:
        return passed

    X = sample.values.astype(float)
    const_mask = X.std(axis=0) > 1e-10
    passed = [p for p, ok in zip(passed, const_mask) if ok]
    X = X[:, const_mask]

    while len(passed) > 1:
        vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        max_vif = max(vifs)
        if max_vif <= vif_threshold:
            break
        drop_idx = vifs.index(max_vif)
        logger.info("VIF 剔除: %s (VIF=%.1f)", passed[drop_idx], max_vif)
        passed.pop(drop_idx)
        X = np.delete(X, drop_idx, axis=1)

    return passed


def _get_family(factor: str) -> Optional[str]:
    for fam, members in FAMILY_MAP.items():
        if factor in members or any(factor.startswith(m) for m in members):
            return fam
    return None
