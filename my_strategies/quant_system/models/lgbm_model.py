"""
LightGBM 模型（含 LambdaRank 排序学习）

提供两个模型类：
- LightGBMQuantModel: 标准回归模型
- LambdaRankQuantModel: 排序学习模型（适合选股排序问题）
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


class LightGBMQuantModel:
    """LightGBM 回归模型，兼容 quant_system 的 fit/predict 接口。"""

    def __init__(self, **kwargs):
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm 未安装")
        self.params = {
            "n_estimators": kwargs.get("n_estimators", 400),
            "learning_rate": kwargs.get("learning_rate", 0.03),
            "num_leaves": kwargs.get("num_leaves", 31),
            "subsample": kwargs.get("subsample", 0.9),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.9),
            "min_child_samples": kwargs.get("min_child_samples", 50),
            "random_state": kwargs.get("random_state", 42),
            "verbose": -1,
        }
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None):
        X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
        callbacks = [lgb.log_evaluation(0)]
        if X_valid is not None:
            X_valid = X_valid.fillna(0).replace([np.inf, -np.inf], 0)
            callbacks.append(lgb.early_stopping(50, verbose=False))

        self.model = lgb.LGBMRegressor(**self.params)
        eval_set = [(X_valid, y_valid)] if X_valid is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        logger.info("LightGBM 训练完成: best_iteration=%s",
                     getattr(self.model, "best_iteration_", "N/A"))

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def save(self, path: str):
        self.model.booster_.save_model(path)

    def load(self, path: str):
        self.model = lgb.Booster(model_file=path)


class LambdaRankQuantModel:
    """LambdaRank 排序学习模型。"""

    def __init__(self, **kwargs):
        if not LGBM_AVAILABLE:
            raise ImportError("lightgbm 未安装")
        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [5, 10],
            "n_estimators": kwargs.get("n_estimators", 400),
            "learning_rate": kwargs.get("learning_rate", 0.03),
            "num_leaves": kwargs.get("num_leaves", 31),
            "subsample": kwargs.get("subsample", 0.9),
            "colsample_bytree": kwargs.get("colsample_bytree", 0.9),
            "min_child_samples": kwargs.get("min_child_samples", 50),
            "random_state": kwargs.get("random_state", 42),
            "verbose": -1,
        }
        self.model = None
        self._n_bins = 10

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            groups: Optional[List[int]] = None,
            X_valid: Optional[pd.DataFrame] = None,
            y_valid: Optional[pd.Series] = None,
            groups_valid: Optional[List[int]] = None):
        X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)

        relevance = self._to_relevance(y_train)

        if groups is None:
            groups = [len(X_train)]

        try:
            self.model = lgb.LGBMRanker(**self.params)
            eval_set = None
            eval_group = None
            if X_valid is not None:
                X_valid = X_valid.fillna(0).replace([np.inf, -np.inf], 0)
                eval_set = [(X_valid, self._to_relevance(y_valid))]
                eval_group = [groups_valid or [len(X_valid)]]

            self.model.fit(
                X_train, relevance,
                group=groups,
                eval_set=eval_set,
                eval_group=eval_group,
                callbacks=[lgb.log_evaluation(0)],
            )
            logger.info("LambdaRank 训练完成")
        except Exception as e:
            logger.warning("LambdaRank 失败 (%s)，回退到 LGBMRegressor", e)
            fallback = lgb.LGBMRegressor(
                n_estimators=self.params["n_estimators"],
                learning_rate=self.params["learning_rate"],
                num_leaves=self.params["num_leaves"],
                verbose=-1,
            )
            fallback.fit(X_train, y_train)
            self.model = fallback

    def predict(self, X: pd.DataFrame) -> pd.Series:
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        preds = self.model.predict(X)
        return pd.Series(preds, index=X.index)

    def _to_relevance(self, y: pd.Series) -> np.ndarray:
        try:
            return pd.qcut(y.rank(method="first"), self._n_bins, labels=False).astype(int).values
        except Exception:
            return np.zeros(len(y), dtype=int)

    def save(self, path: str):
        if hasattr(self.model, "booster_"):
            self.model.booster_.save_model(path)
        else:
            self.model.save_model(path)

    def load(self, path: str):
        self.model = lgb.Booster(model_file=path)
