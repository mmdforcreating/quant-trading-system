"""
分钟级时序 Dataset

专为 HighFreqGRUModel 设计，从 parquet 文件读取分钟 K 线数据，
构建 (batch_size, sequence_length, features) 形状的张量。

=== 数据对齐逻辑（关键）===

分钟数据和日频 label 的对齐是本模块的核心难点：

1. 日频 label 的含义:
   - 对于交易日 T，label = close(T+5) / close(T) - 1
   - 即从 T 日收盘开始，持有 5 天后的收益率

2. 分钟数据的组织:
   - 对于交易日 T，我们取 [T-lookback_days+1, T] 共 lookback_days 天的分钟 K 线
   - 例如 lookback_days=5，则取过去 5 天（含当天 T）的所有分钟 bar

3. 对齐方式:
   - 一个样本 = (X, y)
   - X: 过去 lookback_days 天的分钟序列 → shape (seq_len, n_features)
     - seq_len = bars_per_day × lookback_days
     - 5min 线: bars_per_day = 48 (9:30-11:30 有 24 根 + 13:00-15:00 有 24 根)
     - 1min 线: bars_per_day = 240
   - y: 交易日 T 对应的日频 label (未来 5 日收益)

4. 时间序列结束在 T 日收盘(15:00)最后一根 bar，确保不使用未来信息

=== MPS 加速说明 ===

本模块只负责数据准备（CPU 上完成）。
MPS 加速在 HighFreqGRUModel 的 fit/predict 中通过 device.py 实现:
  model.to(device), x_batch.to(device)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)

# A 股交易时间: 9:30-11:30 (2h) + 13:00-15:00 (2h) = 4h
BARS_PER_DAY = {"1": 240, "5": 48, "15": 16, "30": 8, "60": 4}


class MinuteSequenceDataset(TorchDataset):
    """
    PyTorch Dataset: 将分钟 K 线切成固定长度序列并对齐到日频 label。

    每个样本:
    - X: (seq_len, n_features) 的张量
    - y: scalar (未来 N 日收益)

    Parameters
    ----------
    minute_dir : str | Path
        分钟数据目录，如 ~/.qlib/minute_data/5min/
    label_df : pd.DataFrame | pd.Series
        MultiIndex (datetime, instrument) 的日频 label
    calendar : list[pd.Timestamp]
        交易日列表，用于精确定位「过去 N 天」
    lookback_days : int
        回看天数
    minute_period : str
        "1" / "5" / "15" 等
    feature_cols : list[str]
        使用的列名，默认 ["open", "close", "high", "low", "volume", "amount"]
    normalize : bool
        是否对每个序列做归一化
    """

    def __init__(
        self,
        minute_dir: str | Path,
        label_df: pd.DataFrame | pd.Series,
        calendar: list[pd.Timestamp],
        lookback_days: int = 5,
        minute_period: str = "5",
        feature_cols: list[str] | None = None,
        normalize: bool = True,
    ):
        self.minute_dir = Path(minute_dir)
        self.lookback_days = lookback_days
        self.minute_period = minute_period
        self.bars_per_day = BARS_PER_DAY.get(minute_period, 48)
        self.seq_len = self.bars_per_day * lookback_days
        self.feature_cols = feature_cols or ["open", "close", "high", "low", "volume", "amount"]
        self.normalize = normalize

        # 将 label 转换为统一格式: (datetime, instrument) → float
        if isinstance(label_df, pd.DataFrame):
            self.label_series = label_df.iloc[:, 0]
        else:
            self.label_series = label_df

        self.calendar = sorted(calendar)
        self._cal_set = set(self.calendar)

        # 预构建样本索引: (trade_date, instrument) 列表
        self.samples = self._build_sample_index()
        logger.info(
            "MinuteSequenceDataset: %d 个样本, seq_len=%d, features=%d",
            len(self.samples), self.seq_len, len(self.feature_cols),
        )

    def _build_sample_index(self) -> list[tuple[pd.Timestamp, str]]:
        """
        遍历 label 的所有 (date, instrument)，
        过滤掉回看窗口不足 lookback_days 的样本。
        """
        samples = []
        for (dt, inst) in self.label_series.index:
            dt = pd.Timestamp(dt)
            if dt not in self._cal_set:
                continue
            cal_idx = self.calendar.index(dt)
            if cal_idx < self.lookback_days - 1:
                continue
            parquet_path = self.minute_dir / f"{inst}.parquet"
            if not parquet_path.exists():
                continue
            samples.append((dt, inst))
        return samples

    def _load_minute_data(self, instrument: str) -> pd.DataFrame | None:
        """从 parquet 加载某只股票的全部分钟数据。"""
        path = self.minute_dir / f"{instrument}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        return df

    def _extract_sequence(
        self,
        minute_df: pd.DataFrame,
        trade_date: pd.Timestamp,
    ) -> np.ndarray | None:
        """
        从分钟数据中截取以 trade_date 结尾的 lookback_days 天的序列。

        === 对齐逻辑详解 ===

        trade_date = T
        需要的日期范围: [T - (lookback_days-1) 个交易日, T]

        步骤:
        1. 在交易日历中找到 T 的位置 cal_idx
        2. 取 calendar[cal_idx - lookback_days + 1 : cal_idx + 1] 作为目标日期
        3. 从分钟数据中筛选属于这些日期的 bar
        4. 如果 bar 数不足 seq_len，则前补零 (padding)

        返回 shape: (seq_len, n_features)
        """
        cal_idx = self.calendar.index(trade_date)
        start_idx = cal_idx - self.lookback_days + 1
        target_dates = set(self.calendar[start_idx: cal_idx + 1])

        # 按日期过滤分钟 bar
        minute_df = minute_df.copy()
        minute_df["trade_date"] = minute_df["datetime"].dt.normalize()
        mask = minute_df["trade_date"].isin(target_dates)
        selected = minute_df.loc[mask, self.feature_cols].values

        if selected.shape[0] == 0:
            return None

        # 序列长度对齐
        if selected.shape[0] >= self.seq_len:
            # 取最后 seq_len 根 bar（确保以 T 日收盘结尾）
            selected = selected[-self.seq_len:]
        else:
            # 不足则前补零
            pad = np.zeros((self.seq_len - selected.shape[0], selected.shape[1]))
            selected = np.vstack([pad, selected])

        return selected.astype(np.float32)

    def _normalize_sequence(self, seq: np.ndarray) -> np.ndarray:
        """
        对单个序列做归一化:
        - 价格列 (OHLC): 除以序列中第一个非零收盘价，变为比率
        - 量额列: 除以序列均值 (避免除零)

        这样不同股票、不同时间段的数据具有可比性。
        """
        result = seq.copy()
        n_features = result.shape[1]

        # 价格列: 假设前 4 列是 OHLC
        price_cols = min(4, n_features)
        first_nonzero = result[:, 1]  # close 列
        nz_mask = first_nonzero != 0
        if nz_mask.any():
            base_price = first_nonzero[nz_mask][0]
            result[:, :price_cols] = result[:, :price_cols] / (base_price + 1e-10)

        # 量额列: 后续列
        for j in range(price_cols, n_features):
            col = result[:, j]
            m = col.mean()
            if m > 0:
                result[:, j] = col / m

        return result

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        trade_date, inst = self.samples[idx]

        minute_df = self._load_minute_data(inst)
        if minute_df is None:
            return torch.zeros(self.seq_len, len(self.feature_cols)), torch.tensor(0.0)

        seq = self._extract_sequence(minute_df, trade_date)
        if seq is None:
            return torch.zeros(self.seq_len, len(self.feature_cols)), torch.tensor(0.0)

        if self.normalize:
            seq = self._normalize_sequence(seq)

        y_val = self.label_series.loc[(trade_date, inst)]
        if isinstance(y_val, pd.Series):
            y_val = y_val.iloc[0]

        x = torch.tensor(seq, dtype=torch.float32)
        y = torch.tensor(float(y_val), dtype=torch.float32)
        return x, y


def build_minute_dataset_from_config(
    cfg: dict,
    label_df: pd.DataFrame | pd.Series,
    calendar: list[pd.Timestamp],
) -> MinuteSequenceDataset:
    """
    从配置字典构建 MinuteSequenceDataset 的便捷函数。

    Parameters
    ----------
    cfg : dict
        包含 minute_period, lookback_days, minute_data_path 等
    label_df : pd.DataFrame | pd.Series
        日频 label
    calendar : list[pd.Timestamp]
        交易日列表
    """
    minute_period = cfg.get("minute_period", "5")
    storage = Path(cfg.get("minute_data_path", "~/.qlib/minute_data")).expanduser()
    minute_dir = storage / f"{minute_period}min"

    return MinuteSequenceDataset(
        minute_dir=minute_dir,
        label_df=label_df,
        calendar=calendar,
        lookback_days=cfg.get("lookback_days", 5),
        minute_period=minute_period,
    )
