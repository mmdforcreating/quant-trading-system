"""
VIF (方差膨胀因子) 多重共线性筛选模块

迭代式剔除: 每轮计算所有因子的 VIF，移除最大 VIF 的因子，
直到所有因子的 VIF 都低于阈值。

VIF_i = 1 / (1 - R²_i)
其中 R²_i 是因子 i 对其余因子的线性回归拟合度。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


class VIFFilter:
    """
    VIF 共线性剔除器。

    Parameters
    ----------
    vif_threshold : float
        VIF 上限，大于此值的因子将被逐步剔除（通常设 5~10）
    """

    def __init__(self, vif_threshold: float = 10.0):
        self.vif_threshold = vif_threshold

    def filter(self, factor_df: pd.DataFrame) -> list[str]:
        """
        迭代剔除高 VIF 因子，直到所有 VIF < threshold。

        Parameters
        ----------
        factor_df : pd.DataFrame
            MultiIndex (datetime, instrument) × factors
            或普通 DataFrame (rows × factors)

        Returns
        -------
        list[str]
            筛选后的因子名列表 (Active_Features_List)
        """
        # 先用截面均值将 MultiIndex 降维到 (factors,) 的相关矩阵，
        # 或直接在一个大 DataFrame 上计算。
        # 为了效率，采样部分截面日的数据来估计 VIF
        if isinstance(factor_df.index, pd.MultiIndex):
            df = self._sample_cross_sections(factor_df, max_rows=50000)
        else:
            df = factor_df.copy()

        df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
        if df.shape[1] < 2:
            return list(df.columns)

        remaining = list(df.columns)
        removed = []

        while True:
            X = df[remaining].values
            if X.shape[1] < 2:
                break

            # 添加常数项偏移，防止 VIF 计算异常
            X_with_const = np.column_stack([np.ones(X.shape[0]), X])
            vifs = {}
            for i, col in enumerate(remaining):
                try:
                    v = variance_inflation_factor(X_with_const, i + 1)  # +1 跳过常数列
                except Exception:
                    v = np.inf
                vifs[col] = v

            max_col = max(vifs, key=vifs.get)
            max_vif = vifs[max_col]

            if max_vif <= self.vif_threshold:
                break

            remaining.remove(max_col)
            removed.append((max_col, max_vif))
            logger.debug("剔除因子 %s (VIF=%.2f)", max_col, max_vif)

        logger.info(
            "VIF 筛选: 保留 %d / %d 因子 (阈值=%.1f, 剔除 %d 个)",
            len(remaining),
            len(remaining) + len(removed),
            self.vif_threshold,
            len(removed),
        )
        return remaining

    @staticmethod
    def _sample_cross_sections(
        factor_df: pd.DataFrame,
        max_rows: int = 50000,
    ) -> pd.DataFrame:
        """
        从 MultiIndex DataFrame 中均匀采样截面数据，控制计算量。

        当总行数 > max_rows 时随机采样，否则使用全部。
        """
        if len(factor_df) <= max_rows:
            return factor_df.reset_index(drop=True)

        rng = np.random.default_rng(42)
        idx = rng.choice(len(factor_df), size=max_rows, replace=False)
        return factor_df.iloc[idx].reset_index(drop=True)
