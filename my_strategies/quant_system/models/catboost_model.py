"""
CatBoostQuantModel - 基于 CatBoost 的梯度提升树模型

支持双接口：
- Qlib DatasetH（兼容旧管线）
- 普通 DataFrame（walk-forward 新管线）
"""
from __future__ import annotations

import logging
from typing import Optional, Text, Union

import numpy as np
import pandas as pd
from catboost import CatBoost, Pool

logger = logging.getLogger(__name__)


class CatBoostQuantModel:
    """CatBoost 截面预测模型，支持 DataFrame 和 Qlib DatasetH 双接口。"""

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        depth: int = 8,
        loss_function: str = "RMSE",
        verbose: int = 0,
        early_stopping_rounds: int = 50,
        **kwargs,
    ):
        self.params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "depth": depth,
            "loss_function": loss_function,
            "verbose": verbose,
            "task_type": "CPU",
            **kwargs,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None

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

        X_train = self._clean(X_train)
        if isinstance(y_train, pd.DataFrame):
            y_train = np.squeeze(y_train.values)
        elif isinstance(y_train, pd.Series):
            y_train = y_train.values

        mask = np.isfinite(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]

        train_pool = Pool(data=X_train, label=y_train)
        valid_pool = None
        if X_valid is not None and y_valid is not None:
            X_valid = self._clean(X_valid)
            y_v = y_valid.values if isinstance(y_valid, pd.Series) else np.asarray(y_valid)
            valid_pool = Pool(data=X_valid, label=y_v)

        params = {**self.params, "early_stopping_rounds": self.early_stopping_rounds}
        self.model = CatBoost(params)
        self.model.fit(train_pool, eval_set=valid_pool, use_best_model=valid_pool is not None)
        best_iter = self.model.get_best_iteration()
        if best_iter is None:
            best_iter = getattr(self.model, "tree_count_", 0) or 0
        logger.info("CatBoost 训练完成: best_iter=%d", best_iter)

    def predict(self, X, segment=None, **kwargs) -> pd.Series:
        if self.model is None:
            raise ValueError("模型尚未训练")

        try:
            from qlib.data.dataset import DatasetH
            if isinstance(X, DatasetH):
                return self._predict_qlib(X, segment or "test")
        except ImportError:
            pass

        idx = X.index if hasattr(X, "index") else None
        X = self._clean(X)
        preds = self.model.predict(X.values if hasattr(X, "values") else X)
        return pd.Series(preds, index=idx)

    def _fit_qlib(self, dataset):
        from qlib.data.dataset.handler import DataHandlerLP
        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x_train, y_train = df_train["feature"], np.squeeze(df_train["label"].values)
        x_valid, y_valid = df_valid["feature"], np.squeeze(df_valid["label"].values)
        train_pool = Pool(data=x_train, label=y_train)
        valid_pool = Pool(data=x_valid, label=y_valid)
        params = {**self.params, "early_stopping_rounds": self.early_stopping_rounds}
        self.model = CatBoost(params)
        self.model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    def _predict_qlib(self, dataset, segment):
        from qlib.data.dataset.handler import DataHandlerLP
        x = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        preds = self.model.predict(x.values)
        return pd.Series(preds, index=x.index)

    @staticmethod
    def _clean(X):
        if isinstance(X, pd.DataFrame):
            return X.fillna(0).replace([np.inf, -np.inf], 0)
        return np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
