"""
HighFreqGRUModel - 高频分钟数据 GRU 时序模型

=== 核心设计思路 ===

传统日频 GRU 仅用日 OHLCV 作为序列输入，信息密度极低，效果差。
本模块直接消费分钟级 K 线数据，将过去 N 天的分钟 bar 拼成一个长序列
喂入 GRU，捕捉日内模式和跨日趋势的微观结构信息。

=== MPS 加速说明 ===

本模块使用 utils/device.py 统一管理设备分配：
1. 模型参数: model.to(device) — 将 GRU 权重放到 MPS/CPU
2. 输入张量: x_batch.to(device) — 训练和推理时将数据搬到对应设备
3. MPS 对 GRU 的支持:
   - PyTorch 2.0+ 的 MPS 后端已支持 GRU forward/backward
   - 如果遇到个别算子不兼容，device.py 的 try/except 会自动回退 CPU
4. 若 MPS 训练出现精度问题(float16 溢出等)，可在 config 中设 force_cpu: true

=== 双轨输入兼容 ===

当前实现为「分钟级高频特征聚合」单轨路线:
  input = 5min K 线序列 → GRU → 日频预测

未来可扩展为双轨:
  Track A: 分钟序列 → GRU → hidden
  Track B: 日频因子 → FC → embedding
  Concat(hidden, embedding) → FC → 预测
"""
from __future__ import annotations

import logging
from typing import Text, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from ..utils.device import get_device
from ..data_handlers.minute_dataset import MinuteSequenceDataset

logger = logging.getLogger(__name__)


class GRUNet(nn.Module):
    """
    GRU 网络定义。

    输入: (batch_size, sequence_length, input_size)
    输出: (batch_size,) — 标量预测值

    === 网络结构 ===

    分钟 K 线序列
         ↓
    [多层 GRU] (捕捉时序依赖)
         ↓
    取最后时间步隐藏状态 h[-1]
         ↓
    [Dropout] (防过拟合)
         ↓
    [全连接层] hidden_size → 1
         ↓
    预测值 (未来5日收益率)

    Parameters
    ----------
    input_size : int
        每个时间步的特征数（如 OHLCV+amount = 6）
    hidden_size : int
        GRU 隐藏层维度
    num_layers : int
        GRU 层数
    dropout : float
        Dropout 比率（仅在 num_layers > 1 时生效于层间）
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            shape (batch, seq_len, input_size)

        Returns
        -------
        Tensor
            shape (batch,) — 预测值
        """
        # GRU 输出: output (batch, seq_len, hidden), h_n (num_layers, batch, hidden)
        output, h_n = self.gru(x)

        # 取最后一个时间步的输出（等价于 h_n[-1]）
        last_hidden = output[:, -1, :]

        out = self.dropout(last_hidden)
        out = self.fc(out).squeeze(-1)
        return out


class HighFreqGRUModel(Model):
    """
    高频 GRU 预测模型。

    与其他 3 个日频模型不同，此模型不从 DatasetH 取日频因子，
    而是直接读取分钟级 parquet 数据构建序列输入。

    === fit() 流程 ===

    1. 从 DatasetH 获取 train/valid 的 label (日频未来 N 日收益)
    2. 用 MinuteSequenceDataset 将 label 和分钟数据配对
    3. 构建 DataLoader → 在 MPS/CPU 上训练 GRU
    4. 基于验证集 loss 做 Early Stopping

    === predict() 流程 ===

    1. 从 DatasetH 获取 test 段的 label index（用于确定预测范围）
    2. 构建 MinuteSequenceDataset → DataLoader
    3. 前向推理 → 返回 pd.Series(predictions, index=...)

    Parameters
    ----------
    input_size : int
        特征数
    hidden_size : int
        GRU 隐层维度
    num_layers : int
        GRU 层数
    dropout : float
        Dropout 比率
    lr : float
        学习率
    batch_size : int
        批大小
    max_epochs : int
        最大训练轮数
    patience : int
        Early Stopping 耐心
    lookback_days : int
        回看天数
    minute_period : str
        分钟周期 "1" / "5"
    minute_data_path : str
        分钟数据存储根目录
    force_cpu : bool
        强制使用 CPU
    """

    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 0.001,
        batch_size: int = 256,
        max_epochs: int = 100,
        patience: int = 10,
        lookback_days: int = 5,
        minute_period: str = "5",
        minute_data_path: str = "~/.qlib/minute_data",
        force_cpu: bool = False,
        **kwargs,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lookback_days = lookback_days
        self.minute_period = minute_period
        self.minute_data_path = str(
            pd.io.common.stringify_path(minute_data_path)
        ).replace("~", str(pd.io.common.stringify_path("~")))
        from pathlib import Path
        self.minute_data_path = str(Path(minute_data_path).expanduser())

        # === MPS 加速: 通过 device.py 自动选择设备 ===
        # 如果当前是 Apple Silicon Mac 且 MPS 可用，模型和数据会被放到 MPS 设备上
        # 在 forward/backward 中利用 Metal GPU 加速矩阵运算
        self.device = get_device(force_cpu=force_cpu)
        logger.info("HighFreqGRUModel 设备: %s", self.device)

        self.net = None
        self._fitted = False

    def _build_net(self) -> GRUNet:
        net = GRUNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        # === MPS 加速: 将模型参数搬到 MPS 设备 ===
        net = net.to(self.device)
        return net

    def _get_calendar(self, dataset: DatasetH) -> list[pd.Timestamp]:
        """从 dataset 的 handler 中提取交易日历。"""
        try:
            from qlib.data import D
            calendar = D.calendar(freq="day")
            return [pd.Timestamp(d) for d in calendar]
        except Exception:
            df_label = dataset.prepare("train", col_set="label", data_key=DataHandlerLP.DK_L)
            dates = df_label.index.get_level_values(0).unique()
            return sorted([pd.Timestamp(d) for d in dates])

    def _make_minute_dataset(
        self,
        dataset: DatasetH,
        segment: str,
        calendar: list[pd.Timestamp],
    ) -> MinuteSequenceDataset:
        """构建分钟级 Dataset。"""
        from pathlib import Path

        label_df = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)
        if isinstance(label_df, pd.DataFrame) and "label" in label_df.columns.get_level_values(0):
            label_series = label_df["label"].iloc[:, 0]
        elif isinstance(label_df, pd.DataFrame):
            label_series = label_df.iloc[:, 0]
        else:
            label_series = label_df

        minute_dir = Path(self.minute_data_path) / f"{self.minute_period}min"

        return MinuteSequenceDataset(
            minute_dir=minute_dir,
            label_df=label_series,
            calendar=calendar,
            lookback_days=self.lookback_days,
            minute_period=self.minute_period,
        )

    def fit(self, dataset: DatasetH, reweighter=None):
        """
        训练 GRU 模型。

        流程:
        1. 从 dataset 获取 train/valid 的 label
        2. 构建 MinuteSequenceDataset + DataLoader
        3. 在 MPS/CPU 上迭代训练
        4. Early Stopping: 验证集 loss 连续 patience 轮不改善则停止
        """
        calendar = self._get_calendar(dataset)

        train_ds = self._make_minute_dataset(dataset, "train", calendar)
        valid_ds = self._make_minute_dataset(dataset, "valid", calendar)

        if len(train_ds) == 0:
            raise ValueError("GRU 训练集为空，请检查分钟数据是否已下载")

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=0,
        )

        self.net = self._build_net()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            # --- 训练阶段 ---
            self.net.train()
            train_loss_sum = 0.0
            train_count = 0

            for x_batch, y_batch in train_loader:
                # === MPS 加速: 将输入数据搬到与模型相同的设备 ===
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.net(x_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()

                train_loss_sum += loss.item() * len(y_batch)
                train_count += len(y_batch)

            avg_train = train_loss_sum / max(train_count, 1)

            # --- 验证阶段 ---
            self.net.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for x_batch, y_batch in valid_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds = self.net(x_batch)
                    loss = criterion(preds, y_batch)
                    val_loss_sum += loss.item() * len(y_batch)
                    val_count += len(y_batch)

            avg_val = val_loss_sum / max(val_count, 1)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d — train_loss=%.6f, val_loss=%.6f (device=%s)",
                    epoch + 1, self.max_epochs, avg_train, avg_val, self.device,
                )

            # --- Early Stopping ---
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net = self.net.to(self.device)

        self._fitted = True
        logger.info("GRU 训练完成: best_val_loss=%.6f", best_val_loss)

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        """
        GRU 推理，返回 pd.Series。

        流程:
        1. 构建 test 段的 MinuteSequenceDataset
        2. DataLoader 遍历 → 模型前向 → 收集预测值
        3. 对齐回 (datetime, instrument) 的 MultiIndex
        """
        if not self._fitted or self.net is None:
            raise ValueError("GRU 模型尚未训练")

        calendar = self._get_calendar(dataset)
        test_ds = self._make_minute_dataset(dataset, segment if isinstance(segment, str) else "test", calendar)

        if len(test_ds) == 0:
            logger.warning("GRU 测试集为空")
            return pd.Series(dtype=float)

        test_loader = DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0,
        )

        self.net.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch, _ in test_loader:
                # === MPS 加速: 推理时也将数据搬到 MPS ===
                x_batch = x_batch.to(self.device)
                preds = self.net(x_batch)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        index_tuples = test_ds.samples
        index = pd.MultiIndex.from_tuples(index_tuples, names=["datetime", "instrument"])
        return pd.Series(all_preds, index=index)
