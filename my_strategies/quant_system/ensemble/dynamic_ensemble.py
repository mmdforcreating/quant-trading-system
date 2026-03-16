"""
动态集成预测器

融合多个异构模型的截面预测分数，生成最终排序 Score。

支持两种融合方式:
1. dynamic_weighted: 基于各模型最近 N 天的 Rank IC 动态分配权重
   - 权重 = softmax(各模型最近 N 天 Rank IC 均值)
   - IC 高的模型获得更大权重，自动适应市场风格变化
2. average: 简单等权平均（基线方法）

=== 数据流 ===

输入: Dict[model_key, pd.Series]
  - 每个 Series 有 MultiIndex (datetime, instrument)，值为模型预测分

输出: pd.Series
  - MultiIndex (datetime, instrument)，值为融合后的 Final Score
  - 可直接按分数排序选股
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class DynamicEnsemblePredictor:
    """
    动态集成预测器。

    Parameters
    ----------
    method : str
        "dynamic_weighted" 或 "average"
    lookback_eval_days : int
        动态加权时回看的评估天数
    """

    def __init__(
        self,
        method: str = "dynamic_weighted",
        lookback_eval_days: int = 20,
    ):
        if method not in ("dynamic_weighted", "average"):
            raise ValueError(f"不支持的集成方法: {method}")
        self.method = method
        self.lookback_eval_days = lookback_eval_days
        self._weights_history: list[dict[str, float]] = []

    def compute_weights(
        self,
        model_preds_history: dict[str, pd.Series],
        labels_history: pd.Series,
    ) -> dict[str, float]:
        """
        基于历史预测表现计算各模型的动态权重。

        对每个模型，计算最近 lookback_eval_days 天的平均 Rank IC，
        然后对所有模型的 IC 取 softmax 得到权重。

        Parameters
        ----------
        model_preds_history : dict[str, pd.Series]
            {model_key: 历史预测 Series (MultiIndex)}
        labels_history : pd.Series
            历史实际 label (MultiIndex)

        Returns
        -------
        dict[str, float]
            {model_key: weight}，权重之和为 1
        """
        model_ics = {}
        for mk, preds in model_preds_history.items():
            common = preds.index.intersection(labels_history.index)
            if len(common) == 0:
                model_ics[mk] = 0.0
                continue

            p = preds.loc[common]
            l = labels_history.loc[common]

            dates = p.index.get_level_values(0).unique()
            # 取最近 N 天
            recent_dates = sorted(dates)[-self.lookback_eval_days:]

            ics = []
            for dt in recent_dates:
                try:
                    p_day = p.loc[dt]
                    l_day = l.loc[dt]
                except KeyError:
                    continue
                mask = p_day.notna() & l_day.notna()
                if mask.sum() < 5:
                    continue
                corr, _ = spearmanr(p_day[mask], l_day[mask])
                ics.append(corr)

            model_ics[mk] = np.nanmean(ics) if ics else 0.0

        # Softmax 权重: exp(IC) / sum(exp(IC))
        ic_vals = np.array(list(model_ics.values()))
        # 温度缩放，避免 softmax 输出过于集中
        temperature = 5.0
        exp_vals = np.exp(ic_vals * temperature)
        weights_vals = exp_vals / exp_vals.sum()

        weights = dict(zip(model_ics.keys(), weights_vals))
        self._weights_history.append(weights)

        logger.info("动态权重: %s", {k: f"{v:.4f}" for k, v in weights.items()})
        return weights

    def predict(
        self,
        model_preds: dict[str, pd.Series],
        weights: dict[str, float] | None = None,
    ) -> pd.Series:
        """
        融合多模型预测生成最终 Score。

        Parameters
        ----------
        model_preds : dict[str, pd.Series]
            {model_key: 当期预测 Series}
        weights : dict[str, float] | None
            模型权重；为 None 时使用等权

        Returns
        -------
        pd.Series
            最终融合分数
        """
        if not model_preds:
            return pd.Series(dtype=float)

        all_keys = list(model_preds.keys())

        if self.method == "average" or weights is None:
            weights = {k: 1.0 / len(all_keys) for k in all_keys}

        # 收集所有模型预测到一个 DataFrame
        pred_df = pd.DataFrame({k: v for k, v in model_preds.items()})

        # 对每个模型的预测做截面 Z-Score 标准化，消除量纲差异
        for col in pred_df.columns:
            pred_df[col] = pred_df.groupby(level=0)[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-10)
            )

        # 加权求和
        final_score = pd.Series(0.0, index=pred_df.index)
        for mk in all_keys:
            if mk in pred_df.columns and mk in weights:
                final_score += weights[mk] * pred_df[mk].fillna(0)

        logger.info("集成预测: %d 条记录, %d 个模型", len(final_score), len(all_keys))
        return final_score

    def ensemble_rolling_results(
        self,
        rolling_results: list[dict],
        labels: pd.Series,
    ) -> pd.Series:
        """
        对滚动训练的全部结果做集成。

        遍历每轮的 predictions，累积历史预测，
        动态计算权重（或等权），融合后拼接所有 test 段。

        Parameters
        ----------
        rolling_results : list[dict]
            RollingTrainer.run() 的输出
        labels : pd.Series
            全量日频 label

        Returns
        -------
        pd.Series
            全部 test 段拼接后的最终 Score
        """
        accumulated_preds: dict[str, list[pd.Series]] = {}
        all_scores = []

        for round_data in rolling_results:
            preds = round_data["predictions"]

            for mk, pred_s in preds.items():
                if mk not in accumulated_preds:
                    accumulated_preds[mk] = []
                accumulated_preds[mk].append(pred_s)

            # 计算动态权重
            weights = None
            if self.method == "dynamic_weighted":
                history = {
                    mk: pd.concat(v) for mk, v in accumulated_preds.items()
                }
                weights = self.compute_weights(history, labels)

            # 融合当轮预测
            score = self.predict(preds, weights=weights)
            all_scores.append(score)

        if not all_scores:
            return pd.Series(dtype=float)

        final = pd.concat(all_scores)
        # 去重（同一日期+股票可能出现在多轮的边界）
        final = final[~final.index.duplicated(keep="last")]
        return final.sort_index()
