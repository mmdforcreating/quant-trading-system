"""
ExtraTreesQuantModel - 极端随机树模型

增加高方差截面特征的泛化能力，相比随机森林引入更多随机性。
继承 Qlib 的 Model 基类。
"""
from __future__ import annotations

import logging
from typing import Text, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

logger = logging.getLogger(__name__)


class ExtraTreesQuantModel(Model):
    """
    ExtraTrees 截面预测模型。

    Parameters
    ----------
    n_estimators : int
        树的数量
    max_depth : int | None
        最大深度
    n_jobs : int
        并行数，-1 使用所有核心
    random_state : int | None
        随机种子
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = 12,
        n_jobs: int = -1,
        random_state: int | None = 42,
        **kwargs,
    ):
        self.model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )
        self._fitted = False

    def fit(self, dataset: DatasetH, reweighter=None):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty:
            raise ValueError("训练集为空，请检查 Dataset 配置")

        x_train = df_train["feature"].values
        y_train = np.squeeze(df_train["label"].values)

        # 处理 NaN：ExtraTrees 不原生支持缺失值
        mask = ~(np.isnan(x_train).any(axis=1) | np.isnan(y_train))
        x_train = x_train[mask]
        y_train = y_train[mask]

        self.model.fit(x_train, y_train)
        self._fitted = True

        logger.info(
            "ExtraTrees 训练完成: n_estimators=%d, 样本数=%d",
            self.model.n_estimators,
            len(y_train),
        )

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        if not self._fitted:
            raise ValueError("模型尚未训练")
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        x_vals = x_test.values
        # 将 NaN 替换为 0（推理阶段容错）
        x_vals = np.nan_to_num(x_vals, nan=0.0)
        preds = self.model.predict(x_vals)
        return pd.Series(preds, index=x_test.index)
