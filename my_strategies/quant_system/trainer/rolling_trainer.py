"""
周频滚动训练器

以配置的步长（默认 5 个交易日 = 一周）滚动推进，
每轮重新训练所有激活模型，生成滚动预测。

=== 滚动窗口示意 ===

假设 step=5, train_window=252, valid_window=63:

Round 0:
  train: [day_0, day_251]
  valid: [day_252, day_314]
  test:  [day_315, day_319]

Round 1: (前进 5 天)
  train: [day_5, day_256]
  valid: [day_257, day_319]
  test:  [day_320, day_324]

Round N:
  train: [day_5N, day_5N+251]
  valid: [day_5N+252, day_5N+314]
  test:  [day_5N+315, day_5N+319]
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

import pandas as pd

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R

from ..utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class RollingTrainer:
    """
    周频滚动训练器。

    Parameters
    ----------
    cfg : ConfigManager
        全局配置
    handler : DataHandlerLP
        已初始化的因子 handler（含 active_features 筛选）
    """

    def __init__(
        self,
        cfg: ConfigManager,
        handler: DataHandlerLP,
        start_time: str | None = None,
        end_time: str | None = None,
    ):
        self.cfg = cfg
        self.handler = handler
        self.step = cfg.rolling["step"]
        self.train_window = cfg.rolling["train_window"]
        self.valid_window = cfg.rolling["valid_window"]
        self.start_time = pd.Timestamp(start_time) if start_time else None
        self.end_time = pd.Timestamp(end_time) if end_time else None

    def _get_calendar(self) -> list[pd.Timestamp]:
        """获取交易日历，裁剪到实际数据范围。"""
        from qlib.data import D
        cal = [pd.Timestamp(d) for d in D.calendar(freq="day")]
        if self.start_time:
            cal = [d for d in cal if d >= self.start_time]
        if self.end_time:
            cal = [d for d in cal if d <= self.end_time]
        return cal

    def generate_rolling_segments(
        self,
        calendar: list[pd.Timestamp],
        start_idx: int | None = None,
        end_idx: int | None = None,
    ) -> list[dict[str, tuple[str, str]]]:
        """
        生成所有滚动窗口的 train/valid/test 时间段。

        Returns
        -------
        list[dict]
            每个 dict 包含 "train", "valid", "test" 键，
            值为 (start_date_str, end_date_str) 元组
        """
        min_start = self.train_window + self.valid_window
        if start_idx is None:
            start_idx = min_start
        if end_idx is None:
            end_idx = len(calendar) - self.step

        segments = []
        idx = start_idx
        while idx + self.step <= len(calendar):
            test_end = min(idx + self.step - 1, len(calendar) - 1)
            test_start = idx
            valid_end = idx - 1
            valid_start = valid_end - self.valid_window + 1
            train_end = valid_start - 1
            train_start = train_end - self.train_window + 1

            if train_start < 0:
                idx += self.step
                continue

            seg = {
                "train": (
                    calendar[train_start].strftime("%Y-%m-%d"),
                    calendar[train_end].strftime("%Y-%m-%d"),
                ),
                "valid": (
                    calendar[valid_start].strftime("%Y-%m-%d"),
                    calendar[valid_end].strftime("%Y-%m-%d"),
                ),
                "test": (
                    calendar[test_start].strftime("%Y-%m-%d"),
                    calendar[test_end].strftime("%Y-%m-%d"),
                ),
            }
            segments.append(seg)
            idx += self.step

        logger.info("生成 %d 个滚动窗口", len(segments))
        return segments

    def _instantiate_model(self, model_key: str) -> Any:
        """根据 config 实例化模型。"""
        model_cfg = self.cfg.models[model_key]
        module_path = model_cfg["module_path"]
        class_name = model_cfg["class"]
        kwargs = dict(model_cfg.get("kwargs", {}))

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(**kwargs)

    def _build_dataset(self, segment: dict[str, tuple[str, str]]) -> DatasetH:
        """为一个滚动窗口构建 DatasetH。"""
        dataset = DatasetH(
            handler=self.handler,
            segments={
                "train": segment["train"],
                "valid": segment["valid"],
                "test": segment["test"],
            },
        )
        return dataset

    def train_one_round(
        self,
        segment: dict[str, tuple[str, str]],
        active_models: list[str],
    ) -> dict[str, tuple[Any, pd.Series]]:
        """
        在单个滚动窗口上训练所有激活模型并生成 test 预测。

        Returns
        -------
        dict[str, tuple[Model, pd.Series]]
            {model_key: (fitted_model, test_predictions)}
        """
        dataset = self._build_dataset(segment)
        results = {}

        for model_key in active_models:
            logger.info("训练模型 %s (test=%s)", model_key, segment["test"])
            try:
                model = self._instantiate_model(model_key)
                model.fit(dataset)
                preds = model.predict(dataset, segment="test")
                results[model_key] = (model, preds)
                logger.info("模型 %s 预测 %d 条记录", model_key, len(preds))
            except Exception as e:
                logger.error("模型 %s 训练失败: %s", model_key, e, exc_info=True)

        return results

    def run(self) -> list[dict]:
        """
        执行完整的滚动训练流程。

        Returns
        -------
        list[dict]
            每轮的结果列表，每个元素包含:
            - "segment": 时间段信息
            - "predictions": {model_key: pd.Series}
        """
        calendar = self._get_calendar()
        segments = self.generate_rolling_segments(calendar)
        active_models = list(self.cfg.active_models)

        all_results = []

        for i, seg in enumerate(segments):
            logger.info(
                "===== 滚动轮次 %d/%d: test=%s =====",
                i + 1, len(segments), seg["test"],
            )

            round_results = self.train_one_round(seg, active_models)

            preds_dict = {k: v[1] for k, v in round_results.items()}
            all_results.append({
                "segment": seg,
                "predictions": preds_dict,
            })

            # 记录到 Qlib Recorder
            try:
                with R.start(experiment_name=self.cfg.strategy_name):
                    R.log_params(segment=seg, round_idx=i)
                    for mk, pred_s in preds_dict.items():
                        R.log_metrics(**{f"{mk}_pred_count": len(pred_s)})
            except Exception as e:
                logger.warning("Recorder 记录失败: %s", e)

        logger.info("滚动训练完成: %d 轮", len(all_results))
        return all_results
