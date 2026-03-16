"""
GRU Walk-Forward 适配器

将分钟级 GRU 模型包装为与 WalkForwardEngine 兼容的 fit(df, y)/predict(df) 接口。
不依赖 Qlib，直接读取 parquet 分钟数据。

运行逻辑:
  1. fit(): 从训练集的 (date, ts_code) 对中, 查找有分钟数据的股票,
     将过去 lookback_days 天的 5min K 线拼成序列, 训练 GRU
  2. predict(): 对推断日的每只股票加载分钟序列, 前向推理得到分数
  3. 没有分钟数据的股票返回 0 分 (中性), 不影响排名
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.device import get_device

logger = logging.getLogger(__name__)

BARS_PER_DAY = {"1": 240, "5": 48, "15": 16, "30": 8}


class _GRUNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.1):
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
        output, _ = self.gru(x)
        out = self.dropout(output[:, -1, :])
        return self.fc(out).squeeze(-1)


def _ts_code_to_filename(ts_code: str) -> str:
    """000001.SZ → 000001"""
    return ts_code.split(".")[0] if "." in ts_code else ts_code


class GRUQuantModel:
    """
    WalkForwardEngine 兼容的分钟级 GRU 模型。

    与 CatBoost/LightGBM 等日频模型平行参与集成:
    - 输入: 分钟级 OHLCV 序列 (而非日频因子)
    - 输出: 与日频模型相同的 per-stock 预测分数
    - 评判标准: 与其他模型统一 — Sharpe / Rank IC / ICIR / cost_robust
    """

    def __init__(
        self,
        minute_data_path: str = "data/minute",
        minute_period: str = "5",
        lookback_days: int = 3,
        input_size: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        lr: float = 0.001,
        batch_size: int = 64,
        max_epochs: int = 30,
        patience: int = 5,
        min_train_samples: int = 30,
        force_cpu: bool = False,
        **kwargs,
    ):
        self.minute_data_path = Path(minute_data_path).expanduser()
        self.minute_period = minute_period
        self.lookback_days = lookback_days
        self.bars_per_day = BARS_PER_DAY.get(minute_period, 48)
        self.seq_len = self.bars_per_day * lookback_days
        self.feature_cols = ["open", "close", "high", "low", "volume", "amount"]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_train_samples = min_train_samples

        self.device = get_device(force_cpu=force_cpu)
        self.net: Optional[_GRUNet] = None
        self._fitted = False
        self._minute_cache: dict[str, pd.DataFrame] = {}

    def _minute_dir(self) -> Path:
        return self.minute_data_path / f"{self.minute_period}min"

    def _load_minute(self, ts_code: str) -> Optional[pd.DataFrame]:
        fname = _ts_code_to_filename(ts_code)
        if fname in self._minute_cache:
            return self._minute_cache[fname]
        path = self._minute_dir() / f"{fname}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["trade_date"] = df["datetime"].dt.normalize()
        df = df.sort_values("datetime").reset_index(drop=True)
        self._minute_cache[fname] = df
        return df

    def _build_sequence(
        self, minute_df: pd.DataFrame, target_date: pd.Timestamp, calendar: list,
    ) -> Optional[np.ndarray]:
        """截取以 target_date 结尾的 lookback_days 天分钟 K 线序列。"""
        td = pd.Timestamp(target_date).normalize()
        cal_dates = [pd.Timestamp(d).normalize() for d in calendar]
        if td not in cal_dates:
            return None
        idx = cal_dates.index(td)
        if idx < self.lookback_days - 1:
            return None

        target_dates = set(cal_dates[idx - self.lookback_days + 1: idx + 1])
        mask = minute_df["trade_date"].isin(target_dates)
        selected = minute_df.loc[mask, self.feature_cols].values

        if selected.shape[0] == 0:
            return None

        if selected.shape[0] >= self.seq_len:
            selected = selected[-self.seq_len:]
        else:
            pad = np.zeros((self.seq_len - selected.shape[0], selected.shape[1]))
            selected = np.vstack([pad, selected])

        return self._normalize(selected.astype(np.float32))

    @staticmethod
    def _normalize(seq: np.ndarray) -> np.ndarray:
        result = seq.copy()
        n_feat = result.shape[1]
        price_cols = min(4, n_feat)
        close_col = result[:, 1]
        nz = close_col[close_col != 0]
        if len(nz) > 0:
            base = nz[0]
            result[:, :price_cols] /= (base + 1e-10)
        for j in range(price_cols, n_feat):
            m = result[:, j].mean()
            if m > 0:
                result[:, j] /= m
        return result

    def _build_samples(
        self, df: pd.DataFrame, label_col: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """
        从 df 的 (date, ts_code) 对中构建分钟序列样本。

        Returns: (X, y, valid_indices)
          X: (n_valid, seq_len, n_features)
          y: (n_valid,)
          valid_indices: df 中有效行的位置索引
        """
        dates_sorted = sorted(df["date"].unique())
        X_list, y_list, idx_list = [], [], []

        for i, row in df.iterrows():
            ts_code = row["ts_code"]
            date = row["date"]
            mdf = self._load_minute(ts_code)
            if mdf is None:
                continue
            seq = self._build_sequence(mdf, date, dates_sorted)
            if seq is None:
                continue
            X_list.append(seq)
            y_val = float(row[label_col]) if label_col and label_col in row.index else 0.0
            y_list.append(y_val)
            idx_list.append(i)

        if not X_list:
            return np.empty((0, self.seq_len, len(self.feature_cols))), np.empty(0), []

        return np.stack(X_list), np.array(y_list), idx_list

    def fit(self, X_train, y_train=None, **kwargs):
        """
        训练 GRU。

        Parameters
        ----------
        X_train : pd.DataFrame
            必须包含 date, ts_code 列 (由 WF 引擎传入完整 df_train)
        y_train : label 列名或 Series (不直接使用, 从 X_train 中取)
        """
        label_col = kwargs.get("label_col")
        if label_col is None and isinstance(y_train, pd.Series):
            label_col = y_train.name

        if label_col and label_col in X_train.columns:
            df = X_train
        elif isinstance(y_train, pd.Series):
            df = X_train.copy()
            df[y_train.name or "_label"] = y_train.values
            label_col = y_train.name or "_label"
        else:
            logger.warning("GRU: 无法确定标签列, 跳过训练")
            return

        X_arr, y_arr, _ = self._build_samples(df, label_col)
        n = len(X_arr)
        if n < self.min_train_samples:
            logger.info("GRU: 分钟数据样本 %d < %d, 跳过训练 (预测将返回 0)", n, self.min_train_samples)
            return

        valid_mask = np.isfinite(y_arr)
        X_arr, y_arr = X_arr[valid_mask], y_arr[valid_mask]
        n = len(X_arr)
        if n < self.min_train_samples:
            logger.info("GRU: 有效样本 %d < %d, 跳过训练", n, self.min_train_samples)
            return

        split = max(1, int(n * 0.8))
        X_tr, X_va = X_arr[:split], X_arr[split:]
        y_tr, y_va = y_arr[:split], y_arr[split:]

        self.net = _GRUNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float("inf")
        patience_cnt = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.net.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                optimizer.step()

            if len(X_va) > 0:
                self.net.eval()
                with torch.no_grad():
                    xv = torch.tensor(X_va, dtype=torch.float32).to(self.device)
                    yv = torch.tensor(y_va, dtype=torch.float32).to(self.device)
                    val_loss = criterion(self.net(xv), yv).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_cnt = 0
                    best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                else:
                    patience_cnt += 1
                    if patience_cnt >= self.patience:
                        break

        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net = self.net.to(self.device)

        self._fitted = True
        logger.info("GRU 训练完成: %d 样本, best_val_loss=%.6f, device=%s",
                     n, best_val_loss, self.device)

    def predict(self, X, **kwargs) -> np.ndarray:
        """
        GRU 推断。

        Parameters
        ----------
        X : pd.DataFrame
            必须包含 date, ts_code 列

        Returns
        -------
        np.ndarray: 每行一个预测分数, 无分钟数据的行返回 0
        """
        n_rows = len(X)
        scores = np.zeros(n_rows, dtype=np.float64)

        if not self._fitted or self.net is None:
            return scores

        X_arr, _, valid_indices = self._build_samples(X)
        if len(X_arr) == 0:
            return scores

        self.net.eval()
        with torch.no_grad():
            xt = torch.tensor(X_arr, dtype=torch.float32).to(self.device)
            preds = self.net(xt).cpu().numpy()

        row_positions = list(X.index)
        for vi, pred_val in zip(valid_indices, preds):
            pos = row_positions.index(vi) if vi in row_positions else -1
            if pos >= 0:
                scores[pos] = float(pred_val)

        return scores
