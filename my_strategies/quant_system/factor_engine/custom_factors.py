"""
自研因子计算模块

移植自 quant_core 的 data_processor/features.py，覆盖：
- 基础因子：换手/市值/资金流/复权收益/动量/波动
- 扩展因子：RSI/MACD/ATR/影线/缺口/Amihud/OBV
- 估值与质量：BP/EP/SP/quality_q
- 资金流因子：大单净流/5/20 日强度
- 时序裂变：多窗口 mean/std/range
- 高阶量价：偏度/量价趋势/反转价差
- 截面处理：winsorize + Z-score + 市值中性化
- 标签：未来 N 日复权收益率
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_all_factors(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """计算全部自研因子的总入口。"""
    fe_cfg = cfg.get("factor_engine", {})
    label_horizon = fe_cfg.get("label_horizon", 5)
    cs_cfg = fe_cfg.get("cross_sectional", {})

    df = compute_base_factors(df)
    df = add_more_factors(df)
    df = add_fission_factors(df)
    df = add_advanced_pv_factors(df)

    if cs_cfg.get("enable", True):
        cs_cols = _get_cs_candidate_cols(df)
        if cs_cols:
            df = add_cross_sectional_features(
                df, cs_cols,
                winsor_q=cs_cfg.get("winsor_q", 0.01),
            )
        if cs_cfg.get("enable_neutralize", False) and "log_circ_mv" in df.columns:
            neu_cols = [c for c in df.columns if c.endswith("_csz")]
            if neu_cols:
                df = neutralize_factors(df, neu_cols, mv_col="log_circ_mv")

    df = compute_labels(df, horizon=label_horizon)

    logger.info("因子计算完成: %d 行, %d 列", len(df), df.shape[1])
    return df


def compute_base_factors(df: pd.DataFrame) -> pd.DataFrame:
    """基础因子：换手率、市值、复权价、收益率、动量、波动。"""
    if "turnover_rate" in df.columns and "turnover_rate_f" in df.columns:
        df["turnover_used"] = df["turnover_rate_f"].fillna(df["turnover_rate"])
    elif "turnover_rate" in df.columns:
        df["turnover_used"] = df["turnover_rate"]

    if "circ_mv" in df.columns:
        df["log_circ_mv"] = np.log1p(df["circ_mv"].clip(lower=1))

    if "amount" in df.columns and "circ_mv" in df.columns:
        df["amount_to_circ_mv"] = df["amount"] / df["circ_mv"].replace(0, np.nan)

    if "net_mf_amount" in df.columns and "amount" in df.columns:
        df["mf_to_amount"] = df["net_mf_amount"] / df["amount"].replace(0, np.nan)

    if "close_adj" not in df.columns and "adj_factor" in df.columns and "close" in df.columns:
        df["adj_factor"] = df.groupby("ts_code")["adj_factor"].ffill()
        df["close_adj"] = df["close"] * df["adj_factor"].fillna(1)

    if "close_adj" in df.columns:
        g = df.groupby("ts_code")["close_adj"]
        df["ret_1d_adj"] = g.pct_change()
        df["mom_20d_adj"] = g.transform(lambda x: x / x.shift(20) - 1)
        df["vol_20d_adj"] = g.transform(lambda x: x.pct_change().rolling(20).std())

    return df


def add_more_factors(df: pd.DataFrame) -> pd.DataFrame:
    """扩展因子：动量/波动/技术/量价/估值/资金流。"""
    if "close_adj" in df.columns:
        g = df.groupby("ts_code")["close_adj"]
        for w in [5, 10, 60]:
            df[f"mom_{w}d_adj"] = g.transform(lambda x: x / x.shift(w) - 1)
        df["rev_1d"] = -df.get("ret_1d_adj", 0)

        for w in [5, 60]:
            df[f"vol_{w}d_adj"] = g.transform(lambda x: x.pct_change().rolling(w).std())

        ret = g.transform(lambda x: x.pct_change())
        df["downvol_20d"] = ret.where(ret < 0).groupby(df["ts_code"]).transform(
            lambda x: x.rolling(20, min_periods=5).std()
        )
        df["mdd_20d"] = g.transform(
            lambda x: (x / x.rolling(20).max() - 1).clip(upper=0)
        )

    if "close" in df.columns and "open" in df.columns:
        df["gap_1d"] = df["open"] / df.groupby("ts_code")["close"].shift(1) - 1
        df["intraday_ret"] = df["close"] / df["open"] - 1

    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        hl = df["high"] - df["low"]
        body = (df["close"] - df["open"]).abs()
        df["range_pct"] = hl / df["close"].replace(0, np.nan)
        df["body_pct"] = body / hl.replace(0, np.nan)
        df["upper_shadow_pct"] = (df["high"] - df[["close", "open"]].max(axis=1)) / hl.replace(0, np.nan)
        df["lower_shadow_pct"] = (df[["close", "open"]].min(axis=1) - df["low"]) / hl.replace(0, np.nan)

        prev_close = df.groupby("ts_code")["close_adj" if "close_adj" in df.columns else "close"].shift(1)
        hc = (df["high"] - prev_close).abs()
        lc = (df["low"] - prev_close).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df["tr"] = tr
        df["atr_14"] = tr.groupby(df["ts_code"]).transform(lambda x: x.rolling(14, min_periods=7).mean())
        close_col = "close_adj" if "close_adj" in df.columns else "close"
        df["atr_14_pct"] = df["atr_14"] / df[close_col].replace(0, np.nan)

    if "close_adj" in df.columns:
        g = df.groupby("ts_code")["close_adj"]
        delta = g.diff()
        gain = delta.clip(lower=0).groupby(df["ts_code"]).transform(lambda x: x.rolling(14).mean())
        loss = (-delta.clip(upper=0)).groupby(df["ts_code"]).transform(lambda x: x.rolling(14).mean())
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - 100 / (1 + rs)

        ema12 = g.transform(lambda x: x.ewm(span=12).mean())
        ema26 = g.transform(lambda x: x.ewm(span=26).mean())
        df["macd_line"] = ema12 - ema26
        df["macd_signal"] = df.groupby("ts_code")["macd_line"].transform(lambda x: x.ewm(span=9).mean())
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    if "ret_1d_adj" in df.columns and "amount" in df.columns:
        df["amihud_20d"] = (
            df["ret_1d_adj"].abs() / df["amount"].replace(0, np.nan)
        ).groupby(df["ts_code"]).transform(lambda x: x.rolling(20).mean())

    if "vol" in df.columns:
        vol_mean = df.groupby("ts_code")["vol"].transform(lambda x: x.rolling(20).mean())
        vol_std = df.groupby("ts_code")["vol"].transform(lambda x: x.rolling(20).std())
        df["volz_20"] = (df["vol"] - vol_mean) / vol_std.replace(0, np.nan)

    if "vol" in df.columns and "close_adj" in df.columns:
        signed_vol = df["vol"] * np.sign(df.get("ret_1d_adj", 0))
        df["obv"] = signed_vol.groupby(df["ts_code"]).cumsum()
        df["obv_20d_chg"] = df.groupby("ts_code")["obv"].pct_change(20)

    if "ret_1d_adj" in df.columns and "vol" in df.columns:
        df["ret_vol_corr_20d"] = df.groupby("ts_code").apply(
            lambda g: g["ret_1d_adj"].rolling(20).corr(g["vol"])
        ).reset_index(level=0, drop=True)

    if "turnover_used" in df.columns:
        for w in [5, 20]:
            df[f"turn_{w}d_mean"] = df.groupby("ts_code")["turnover_used"].transform(
                lambda x: x.rolling(w).mean()
            )

    if "volume_ratio" in df.columns:
        df["vol_ratio_5d_chg"] = df.groupby("ts_code")["volume_ratio"].pct_change(5)

    for val_col, raw_col in [("bp", "pb"), ("ep", "pe"), ("sp", "ps")]:
        if raw_col in df.columns:
            df[val_col] = 1.0 / df[raw_col].replace(0, np.nan)

    if "roe" in df.columns and "debt_to_assets" in df.columns:
        df["quality_q"] = df["roe"].fillna(0) - df["debt_to_assets"].fillna(50)

    for prefix, net_col, amt_col in [
        ("mf", "net_mf_amount", "amount"),
    ]:
        if net_col in df.columns and amt_col in df.columns:
            ratio = df[net_col] / df[amt_col].replace(0, np.nan)
            for w in [5, 20]:
                df[f"{prefix}_{w}d_strength"] = ratio.groupby(df["ts_code"]).transform(
                    lambda x: x.rolling(w).mean()
                )

    for bc in ["buy_elg_amount", "buy_lg_amount"]:
        sc = bc.replace("buy_", "sell_")
        if bc in df.columns and sc in df.columns and "amount" in df.columns:
            big_net = df[bc] - df[sc]
            df["big_net_to_amount"] = big_net / df["amount"].replace(0, np.nan)
            for w in [5, 20]:
                df[f"big_{w}d_strength"] = df["big_net_to_amount"].groupby(df["ts_code"]).transform(
                    lambda x: x.rolling(w).mean()
                )
            break

    return df


def add_fission_factors(df: pd.DataFrame) -> pd.DataFrame:
    """时序裂变因子：对核心因子做多窗口 mean/std/range。"""
    base_cols = ["turnover_used", "ret_1d_adj", "intraday_ret", "amount_to_circ_mv", "volume_ratio"]
    base_cols = [c for c in base_cols if c in df.columns]

    for col in base_cols:
        for w in [5, 10, 20]:
            g = df.groupby("ts_code")[col]
            df[f"{col}_{w}d_mean"] = g.transform(lambda x: x.rolling(w).mean())
            df[f"{col}_{w}d_std"] = g.transform(lambda x: x.rolling(w).std())
            df[f"{col}_{w}d_range"] = g.transform(
                lambda x: x.rolling(w).max() - x.rolling(w).min()
            )
    return df


def add_advanced_pv_factors(df: pd.DataFrame) -> pd.DataFrame:
    """高阶量价因子。"""
    if "high" in df.columns and "low" in df.columns and "close" in df.columns:
        hl = df["high"] - df["low"]
        df["hl_skew_1d"] = (2 * df["close"] - df["high"] - df["low"]) / hl.replace(0, np.nan)
        df["hl_skew_20d_mean"] = df.groupby("ts_code")["hl_skew_1d"].transform(
            lambda x: x.rolling(20).mean()
        )

    if "ret_1d_adj" in df.columns and "turnover_used" in df.columns:
        vp = df["ret_1d_adj"] * df["turnover_used"]
        df["vp_trend_20d"] = vp.groupby(df["ts_code"]).transform(
            lambda x: x.rolling(20).sum()
        )

    if "mom_20d_adj" in df.columns and "mom_5d_adj" in df.columns:
        df["reversal_spread_20d_5d"] = df["mom_20d_adj"] - df["mom_5d_adj"]

    return df


def add_cross_sectional_features(
    df: pd.DataFrame,
    cols: List[str],
    date_col: str = "date",
    winsor_q: float = 0.01,
) -> pd.DataFrame:
    """截面 winsorize + Z-score，生成 {col}_csz 列。"""
    for col in cols:
        if col not in df.columns:
            continue

        def _csz(x):
            lo = x.quantile(winsor_q)
            hi = x.quantile(1 - winsor_q)
            x = x.clip(lo, hi)
            mu, sd = x.mean(), x.std()
            if sd == 0 or pd.isna(sd):
                return x * 0
            return (x - mu) / sd

        df[f"{col}_csz"] = df.groupby(date_col)[col].transform(_csz)
    return df


def neutralize_factors(
    df: pd.DataFrame,
    cols: List[str],
    mv_col: str = "log_circ_mv",
    date_col: str = "date",
) -> pd.DataFrame:
    """市值中性化：对每个截面做 OLS 残差。"""
    from numpy.linalg import lstsq

    for col in cols:
        if col not in df.columns or mv_col not in df.columns:
            continue

        def _neu(grp):
            y = grp[col].values
            x = grp[mv_col].values
            mask = np.isfinite(y) & np.isfinite(x)
            if mask.sum() < 5:
                return pd.Series(y, index=grp.index)
            X = np.column_stack([x[mask], np.ones(mask.sum())])
            beta, _, _, _ = lstsq(X, y[mask], rcond=None)
            residuals = np.full(len(y), np.nan)
            residuals[mask] = y[mask] - X @ beta
            return pd.Series(residuals, index=grp.index)

        df[col.replace("_csz", "_neu")] = df.groupby(date_col).apply(_neu).reset_index(level=0, drop=True)
    return df


def compute_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """计算未来 N 日复权收益率标签。"""
    close_col = "close_adj" if "close_adj" in df.columns else "close"
    if close_col not in df.columns:
        logger.warning("无 %s 列，跳过标签计算", close_col)
        return df

    fwd = df.groupby("ts_code")[close_col].shift(-horizon)
    df[f"y_ret_{horizon}d_adj"] = fwd / df[close_col] - 1

    fwd_1 = df.groupby("ts_code")[close_col].shift(-1)
    df["ret_1d_fwd"] = fwd_1 / df[close_col] - 1

    label_col = f"y_ret_{horizon}d_adj"
    if "amount" in df.columns:
        df.loc[df["amount"] <= 0, label_col] = np.nan
    ret = df[label_col]
    df.loc[(ret < -0.5) | (ret > 1.5), label_col] = np.nan

    return df


def _get_cs_candidate_cols(df: pd.DataFrame) -> List[str]:
    """自动收集适合做截面标准化的因子列。"""
    candidates = [
        "turnover_used", "ret_1d_adj", "intraday_ret", "amount_to_circ_mv",
        "volume_ratio", "mom_5d_adj", "mom_10d_adj", "mom_20d_adj", "mom_60d_adj",
        "vol_5d_adj", "vol_20d_adj", "vol_60d_adj", "rsi_14", "atr_14_pct",
        "amihud_20d", "volz_20", "obv_20d_chg", "ret_vol_corr_20d",
        "mf_to_amount", "mf_5d_strength", "mf_20d_strength",
        "big_net_to_amount", "big_5d_strength", "big_20d_strength",
        "bp", "ep", "sp", "quality_q",
        "hl_skew_1d", "hl_skew_20d_mean", "vp_trend_20d", "reversal_spread_20d_5d",
    ]
    for base in ["turnover_used", "ret_1d_adj", "intraday_ret", "amount_to_circ_mv", "volume_ratio"]:
        for w in [5, 10, 20]:
            for stat in ["mean", "std", "range"]:
                candidates.append(f"{base}_{w}d_{stat}")
    return [c for c in candidates if c in df.columns]
