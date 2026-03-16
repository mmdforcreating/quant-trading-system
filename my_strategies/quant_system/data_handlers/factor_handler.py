"""
自定义因子 DataHandler

继承 Qlib 的 DataHandlerLP，基于 Alpha158 因子集生成候选因子，
并附带未来 N 日收益率 label。

本模块不改动 qlib 源码，仅通过配置和继承扩展因子定义。
"""
from __future__ import annotations

import copy

from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.handler import DataHandlerLP


class FactorHandler(DataHandlerLP):
    """
    继承 Alpha158 的因子定义，并允许通过 active_features 参数
    动态裁剪最终输出的特征列。

    Parameters
    ----------
    instruments : str | list[str]
        股票池
    start_time : str
        开始日期
    end_time : str
        结束日期
    label_horizon : int
        预测收益的天数（默认 5 天）
    active_features : list[str] | None
        经 IC/VIF 筛选后的活跃因子列表；为 None 时使用全部 Alpha158 因子
    """

    def __init__(
        self,
        instruments,
        start_time,
        end_time,
        label_horizon: int = 5,
        active_features: list[str] | None = None,
        infer_processors=None,
        learn_processors=None,
        fit_start_time=None,
        fit_end_time=None,
        **kwargs,
    ):
        self._active_features = active_features
        self._label_horizon = label_horizon

        feature_cfg = Alpha158DL.get_feature_config()
        label_cfg = self._get_label_config(label_horizon)

        _fit_start = fit_start_time or start_time
        _fit_end = fit_end_time or end_time

        if infer_processors is None:
            infer_processors = [
                {"class": "RobustZScoreNorm", "kwargs": {
                    "fields_group": "feature",
                    "clip_outlier": True,
                    "fit_start_time": _fit_start,
                    "fit_end_time": _fit_end,
                }},
                {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
            ]
        if learn_processors is None:
            learn_processors = [
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
            ]

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": feature_cfg,
                    "label": label_cfg,
                },
                "freq": "day",
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            **kwargs,
        )

    @staticmethod
    def _get_label_config(horizon: int) -> tuple[list, list]:
        """
        构建 N 日前瞻收益率的 label 配置。

        label = Ref($close, -{horizon}) / $close - 1
        """
        fields = [f"Ref($close, -{horizon}) / $close - 1"]
        names = [f"LABEL_{horizon}d"]
        return fields, names

    def get_feature_names(self) -> list[str]:
        """返回当前所有特征列名。"""
        df = self.fetch(col_set="feature")
        return list(df.columns)

    def fetch_filtered(self, segment="train"):
        """
        获取筛选后的特征数据。

        若 active_features 不为 None，则只保留这些列。
        """
        df = self.fetch(col_set="feature", data_key="infer")
        if self._active_features is not None:
            available = [f for f in self._active_features if f in df.columns]
            df = df[available]
        return df
