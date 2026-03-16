"""
Rank IC / ICIR 因子筛选模块

对每个候选因子，逐日计算与 label 的 Spearman Rank IC，
然后根据 |mean(IC)| 和 |ICIR| 阈值筛选有效因子。

ICIR = mean(IC) / std(IC)  （信息比率，越高越好）
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


class ICFilter:
    """
    基于 Rank IC 和 ICIR 的因子筛选器。

    Parameters
    ----------
    ic_threshold : float
        |mean Rank IC| 最低阈值
    icir_threshold : float
        |ICIR| 最低阈值
    """

    def __init__(self, ic_threshold: float = 0.02, icir_threshold: float = 0.5):
        self.ic_threshold = ic_threshold
        self.icir_threshold = icir_threshold

    @staticmethod
    def compute_rank_ic(
        factor_df: pd.DataFrame,
        label_series: pd.Series,
    ) -> pd.DataFrame:
        """
        逐截面日计算每个因子与 label 的 Spearman Rank IC。

        Parameters
        ----------
        factor_df : pd.DataFrame
            MultiIndex (datetime, instrument) × factors
        label_series : pd.Series
            同样的 MultiIndex

        Returns
        -------
        pd.DataFrame
            index=日期, columns=因子名, values=当日 IC
        """
        common_idx = factor_df.index.intersection(label_series.index)
        factor_df = factor_df.loc[common_idx]
        label_series = label_series.loc[common_idx]

        dates = factor_df.index.get_level_values(0).unique()
        factors = factor_df.columns.tolist()

        records = []
        for dt in dates:
            try:
                f_slice = factor_df.loc[dt]
                l_slice = label_series.loc[dt]
            except KeyError:
                continue

            row = {}
            for col in factors:
                mask = f_slice[col].notna() & l_slice.notna()
                if mask.sum() < 10:
                    row[col] = np.nan
                    continue
                corr, _ = spearmanr(f_slice[col][mask], l_slice[mask])
                row[col] = corr
            records.append({"date": dt, **row})

        return pd.DataFrame(records).set_index("date")

    def filter(
        self,
        factor_df: pd.DataFrame,
        label_series: pd.Series,
    ) -> list[str]:
        """
        执行 IC/ICIR 筛选，返回通过阈值的因子名列表。

        Parameters
        ----------
        factor_df : pd.DataFrame
            MultiIndex (datetime, instrument) × factors
        label_series : pd.Series
            同样的 MultiIndex，未来 N 日收益

        Returns
        -------
        list[str]
            通过筛选的因子名列表（Active_Features_List 的前半步）
        """
        ic_df = self.compute_rank_ic(factor_df, label_series)

        mean_ic = ic_df.mean()
        std_ic = ic_df.std()
        icir = mean_ic / std_ic.replace(0, np.nan)

        passed = []
        for col in factor_df.columns:
            abs_mic = abs(mean_ic.get(col, 0))
            abs_icir = abs(icir.get(col, 0))
            if abs_mic >= self.ic_threshold and abs_icir >= self.icir_threshold:
                passed.append(col)
                logger.debug(
                    "因子 %s 通过: |IC|=%.4f, |ICIR|=%.4f",
                    col, abs_mic, abs_icir,
                )
            else:
                logger.debug(
                    "因子 %s 被剔除: |IC|=%.4f, |ICIR|=%.4f",
                    col, abs_mic, abs_icir,
                )

        logger.info(
            "IC/ICIR 筛选: %d / %d 因子通过 (ic>%.3f, icir>%.3f)",
            len(passed), len(factor_df.columns),
            self.ic_threshold, self.icir_threshold,
        )
        return passed
