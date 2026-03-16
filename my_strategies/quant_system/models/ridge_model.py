"""
RidgeQuantModel - 岭回归线性模型

支持双接口：
- Qlib DatasetH（兼容旧管线）
- 普通 DataFrame（walk-forward 新管线）
"""
from __future__ import annotations

import logging
from typing import Optional, Text, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RidgeQuantModel:
    """岭回归截面预测模型，支持 DataFrame 和 Qlib DatasetH 双接口。"""

    def __init__(self, alpha: float = 1.0, **kwargs):
        kwargs.pop("verbose", None)
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, **kwargs)
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X_train, y_train=None,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None,
            **kwargs):
        try:
            from qlib.data.dataset import DatasetH
            if isinstance(X_train, DatasetH):
                return self._fit_qlib(X_train)
        except ImportError:
            pass

        if isinstance(X_train, pd.DataFrame):
            x = X_train.values
        else:
            x = np.asarray(X_train, dtype=float)
        if isinstance(y_train, pd.Series):
            y = y_train.values
        else:
            y = np.asarray(y_train, dtype=float)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.isfinite(y)
        x, y = x[mask], y[mask]

        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y)
        self._fitted = True
        logger.info("Ridge 训练完成: alpha=%.4f, 样本数=%d", self.alpha, len(y))

    def predict(self, X, segment=None, **kwargs) -> pd.Series:
        if not self._fitted:
            raise ValueError("模型尚未训练")

        try:
            from qlib.data.dataset import DatasetH
            if isinstance(X, DatasetH):
                return self._predict_qlib(X, segment or "test")
        except ImportError:
            pass

        idx = X.index if hasattr(X, "index") else None
        if isinstance(X, pd.DataFrame):
            x = X.values
        else:
            x = np.asarray(X, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_scaled = self.scaler.transform(x)
        preds = self.model.predict(x_scaled)
        return pd.Series(preds, index=idx)

    def _fit_qlib(self, dataset):
        from qlib.data.dataset.handler import DataHandlerLP
        df_train, _ = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x = df_train["feature"].values
        y = np.squeeze(df_train["label"].values)
        mask = ~(np.isnan(x).any(axis=1) | np.isnan(y))
        x_scaled = self.scaler.fit_transform(x[mask])
        self.model.fit(x_scaled, y[mask])
        self._fitted = True

    def _predict_qlib(self, dataset, segment):
        from qlib.data.dataset.handler import DataHandlerLP
        x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        x = np.nan_to_num(x_test.values, nan=0.0)
        x_scaled = self.scaler.transform(x)
        preds = self.model.predict(x_scaled)
        return pd.Series(preds, index=x_test.index)
