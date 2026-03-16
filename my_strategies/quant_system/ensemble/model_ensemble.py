"""
模型集成预测器

支持：
- RankIC 动态加权（从 model_scores_cache 读取）
- Softmax / rank_decay 权重方案
- 投票融合（含 vote_count 统计）
- 胜率微调
- 行业暴露控制
- 情绪否决
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODEL_FAMILY_ALIAS: Dict[str, str] = {
    "lightgbm": "lgbm",
    "light_gbm": "lgbm",
    "lambdarank": "lgbm_rank",
}


def rank_decay(rank0: int) -> float:
    return float(1.0 / np.log2(2.0 + rank0))


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
    x = x / max(float(temp), 1e-9)
    x = x - np.nanmax(x)
    ex = np.exp(np.clip(x, -50, 50))
    s = np.nansum(ex)
    if not np.isfinite(s) or s <= 0:
        out = np.ones_like(ex, dtype=float)
        return out / out.sum()
    return ex / s


def normalize_model_family(name: str) -> str:
    """统一模型族名称（lightgbm → lgbm 等）。"""
    return _MODEL_FAMILY_ALIAS.get(name.lower(), name.lower())


class ModelEnsemble:
    """模型集成预测器。"""

    def __init__(self, cfg: dict):
        ens_cfg = cfg.get("ensemble", {})
        self.method = ens_cfg.get("method", "rank_ic_weighted")
        self.weight_scheme = ens_cfg.get("weight_scheme", "softmax")
        self.per_model_topn = ens_cfg.get("per_model_topn", 10)
        self.final_topn = ens_cfg.get("final_topn", 5)

    def ensemble_predictions(
        self,
        model_preds: Dict[str, pd.DataFrame],
        model_weights: Optional[Dict[str, float]] = None,
        industry_map: Optional[Dict[str, str]] = None,
        max_per_industry: int = 2,
        sentiment_veto: Optional[set] = None,
        recs_price_max: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        融合多模型预测。

        Parameters
        ----------
        model_preds : {model_key: DataFrame with columns [ts_code, pred_score]}
        model_weights : {model_key: weight}
        industry_map : {ts_code: industry_name}，若提供则启用行业暴露控制
        max_per_industry : 每个行业最多选入的股票数
        sentiment_veto : 需要否决的股票集合（例如重大利空）
        recs_price_max : 推荐股票的最高价格限制

        Returns
        -------
        DataFrame with [ts_code, ensemble_score, rank, vote_count]
        """
        if not model_preds:
            return pd.DataFrame()

        if model_weights is None or self.method == "average":
            model_weights = {k: 1.0 / len(model_preds) for k in model_preds}

        vote_scores: Dict[str, float] = {}
        vote_counts: Dict[str, int] = {}

        for mk, df in model_preds.items():
            w = model_weights.get(mk, 0)
            if w <= 0 or df.empty:
                continue

            top = df.nlargest(self.per_model_topn, "pred_score")
            for rank0, (_, row) in enumerate(top.iterrows()):
                code = row["ts_code"]
                contrib = w * rank_decay(rank0)
                vote_scores[code] = vote_scores.get(code, 0) + contrib
                vote_counts[code] = vote_counts.get(code, 0) + 1

        if not vote_scores:
            return pd.DataFrame()

        result = pd.DataFrame([
            {"ts_code": code, "ensemble_score": score, "vote_count": vote_counts.get(code, 0)}
            for code, score in vote_scores.items()
        ])

        if sentiment_veto:
            before = len(result)
            result = result[~result["ts_code"].isin(sentiment_veto)]
            vetoed = before - len(result)
            if vetoed > 0:
                logger.info("情绪否决: 排除 %d 只股票", vetoed)

        if recs_price_max is not None and recs_price_max > 0:
            result = result[result.get("price", recs_price_max + 1) <= recs_price_max] if "price" in result.columns else result

        result = result.sort_values("ensemble_score", ascending=False).reset_index(drop=True)

        if industry_map:
            result = apply_industry_exposure_control(result, industry_map, max_per_industry)

        result["rank"] = range(1, len(result) + 1)
        return result

    def load_weights_from_cache(self, cache_path: str) -> Dict[str, float]:
        """从 model_scores_cache.csv 加载模型权重。"""
        try:
            df = pd.read_csv(cache_path)
            if df.empty or "model_weight" not in df.columns:
                return {}
            weights = {}
            for _, row in df.iterrows():
                key = row.get("model_key", row.get("model_family", ""))
                w = float(row["model_weight"])
                if key and w > 0:
                    weights[key] = w
            total = sum(weights.values())
            if total > 0:
                weights = {k: v / total for k, v in weights.items()}
            return weights
        except Exception as e:
            logger.warning("加载权重缓存失败: %s", e)
            return {}


def apply_industry_exposure_control(
    recs: pd.DataFrame,
    industry_map: Dict[str, str],
    max_per_industry: int = 2,
) -> pd.DataFrame:
    """限制每个行业的最大推荐数量，防止行业集中风险。"""
    if recs.empty or not industry_map:
        return recs

    recs = recs.copy()
    recs["_industry"] = recs["ts_code"].map(industry_map).fillna("未知")

    kept = []
    industry_count: Dict[str, int] = {}
    for _, row in recs.iterrows():
        ind = row["_industry"]
        cnt = industry_count.get(ind, 0)
        if cnt < max_per_industry:
            kept.append(row)
            industry_count[ind] = cnt + 1

    if not kept:
        return recs.drop(columns=["_industry"])

    result = pd.DataFrame(kept).drop(columns=["_industry"])
    return result.reset_index(drop=True)


def adjust_weights_by_winrate(
    weights: Dict[str, float],
    winrate_stats: Dict[str, float],
    sample_counts: Optional[Dict[str, int]] = None,
    cfg: Optional[dict] = None,
) -> Dict[str, float]:
    """
    用历史胜率微调模型权重。

    multiplier = clip(1 + beta * (win_rate - 0.5), min_mult, max_mult)

    当某模型的历史样本数不足 min_samples 时，不做微调以避免噪声。
    """
    if cfg is None:
        cfg = {}
    wr_cfg = cfg.get("winrate", {})
    if not wr_cfg.get("enable", False):
        return weights

    beta = wr_cfg.get("beta", 1.0)
    min_mult = wr_cfg.get("min_multiplier", 0.75)
    max_mult = wr_cfg.get("max_multiplier", 1.25)
    min_samples = wr_cfg.get("min_samples", 20)

    adjusted = {}
    for mk, w in weights.items():
        wr = winrate_stats.get(mk)
        if wr is None or not isinstance(wr, (int, float)):
            adjusted[mk] = w
            continue
        if sample_counts and sample_counts.get(mk, 0) < min_samples:
            logger.debug("模型 %s 样本数 %d < %d，跳过胜率微调",
                         mk, sample_counts.get(mk, 0), min_samples)
            adjusted[mk] = w
            continue
        mult = np.clip(1 + beta * (wr - 0.5), min_mult, max_mult)
        adjusted[mk] = w * mult

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: v / total for k, v in adjusted.items()}
    return adjusted
