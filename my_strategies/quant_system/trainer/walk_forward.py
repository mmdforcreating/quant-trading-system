"""
Walk-Forward 回测引擎

移植自 quant_core 的 core/engine.py，支持：
- 周频滚动步长（step_days）
- T+1 执行滞后
- 多模型并行训练
- 等权 / 风险平价（反 ATR）建仓
- 持有 N 天到期卖出（含停牌/跌停顺延）
- 涨跌停 / 停牌 / 大幅跳空买入过滤
- DrawdownGuard 回撤保护
- Purging gap 防标签泄漏
- 性能统计与模型评分（ICIR 加权、cost_robust、overfit_gap）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..risk.atr_risk import DrawdownGuard

logger = logging.getLogger(__name__)


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) > 0 else 0.0


def perf_stats(equity: pd.Series) -> Dict[str, float]:
    if len(equity) < 2:
        return {"cum_return": 0, "ann_return": 0, "ann_vol": 0, "sharpe": 0, "max_dd": 0}
    daily_ret = equity.pct_change().dropna()
    cum = float(equity.iloc[-1] / equity.iloc[0] - 1)
    n_days = len(daily_ret)
    ann_ret = float((1 + cum) ** (252 / max(n_days, 1)) - 1)
    ann_vol = float(daily_ret.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-10 else 0.0
    mdd = max_drawdown(equity)
    return {"cum_return": cum, "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_dd": mdd}


def robust_zscore(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad < 1e-10:
        return np.zeros_like(x)
    return (x - med) / (mad * 1.4826)


# ---------------------------------------------------------------------------
#  交易可行性检查
# ---------------------------------------------------------------------------

def _can_buy(row: pd.Series) -> bool:
    """排除停牌、涨停封板、跌停封板、大幅跳空（>4%）的股票。"""
    vol = row.get("vol", row.get("volume", None))
    if vol is not None and float(vol) <= 0:
        return False
    pre_close = row.get("pre_close")
    open_price = row.get("open")
    if pre_close is not None and open_price is not None and float(pre_close) > 0:
        ratio = float(open_price) / float(pre_close)
        if ratio > 1.095 or ratio < 0.905:
            return False
        if abs(ratio - 1.0) > 0.04:
            return False
    return True


def _can_sell(row: pd.Series) -> bool:
    """排除停牌、跌停封板的股票（不可卖出，顺延至下一交易日）。"""
    vol = row.get("vol", row.get("volume", None))
    if vol is not None and float(vol) <= 0:
        return False
    pre_close = row.get("pre_close")
    close_price = row.get("close")
    if pre_close is not None and close_price is not None and float(pre_close) > 0:
        ratio = float(close_price) / float(pre_close)
        if ratio < 0.905:
            return False
    return True


def _adj_open(row: pd.Series, close_col: str, use_adj: bool) -> float:
    """买入价：若使用复权 close，则将 open 也同比例复权，保证价格基准一致。"""
    if use_adj and "open" in row.index:
        raw_open = float(row["open"])
        raw_close = float(row.get("close", 0))
        adj_close = float(row.get(close_col, 0))
        if raw_close > 0 and adj_close > 0:
            return raw_open * (adj_close / raw_close)
        return adj_close if adj_close > 0 else raw_open
    if "open" in row.index:
        return float(row["open"])
    return float(row.get(close_col, 0))


class WalkForwardEngine:
    """Walk-Forward 回测引擎。"""

    def __init__(self, cfg: dict):
        wf = cfg.get("walk_forward", {})
        self.step_days = wf.get("step_days", 5)
        self.train_window = wf.get("train_window_days", 360)
        self.min_train_rows = wf.get("min_train_rows", 300)
        self.execution_lag = wf.get("execution_lag", 1)
        self.top_k = wf.get("top_k", 5)
        self.hold_days = wf.get("hold_days", 5)
        self.purge_days = wf.get("purge_days", 2)
        self.risk_parity = wf.get("risk_parity", False)

        costs = cfg.get("costs", {})
        comm = costs.get("commission_bps", 2)
        slip = costs.get("slippage_bps", 5)
        stamp = costs.get("stamp_tax_bps", 5)
        self.c_buy = (comm + slip) / 10000
        self.c_sell = (comm + slip + stamp) / 10000

        risk = cfg.get("risk", {})
        self.initial_capital = risk.get("initial_capital", 1_000_000)
        self.target_exposure = risk.get("target_gross_exposure", 0.85)
        self.max_position_per_name = risk.get("max_position_per_name", 0.20)
        self.max_gross_exposure = risk.get("max_gross_exposure", 0.95)

        self.dd_guard = DrawdownGuard(cfg)

    # ------------------------------------------------------------------
    #  主循环
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        features: List[str],
        label_col: str,
        model_type: str,
        model_cls: Any,
        model_kwargs: dict,
        full_cross_section: bool = False,
    ) -> Dict[str, Any]:
        """
        执行单模型的 walk-forward 回测。

        Returns
        -------
        dict with keys:
            equity, predictions, stats_train, stats_val, stats_test,
            rank_ics, model_type, turnover
        """
        self.dd_guard.reset()

        df = df.sort_values(["date", "ts_code"]).reset_index(drop=True)
        dates = sorted(df["date"].unique())
        n_dates = len(dates)

        min_start = self.train_window + self.execution_lag
        if n_dates < min_start + self.step_days:
            logger.warning("日期不足: %d < %d", n_dates, min_start + self.step_days)
            return {
                "equity": pd.Series(dtype=float), "predictions": [],
                "rank_ics": [], "turnover": 0.0,
            }

        close_col = "close_adj" if "close_adj" in df.columns else "close"
        use_adj = close_col == "close_adj" and "close" in df.columns

        equity = [self.initial_capital]
        cash = self.initial_capital
        holdings: List[Dict] = []
        all_predictions: List[pd.DataFrame] = []
        rank_ics: List[Dict] = []

        total_buy_value = 0.0
        total_sell_value = 0.0
        n_steps = 0

        idx = min_start
        while idx + self.step_days <= n_dates:
            t_feat_idx = idx - self.execution_lag
            t_feat = dates[t_feat_idx]

            # --- Purging gap: 训练集末尾截断 purge_days 防止标签泄漏 ---
            train_end_idx = max(0, t_feat_idx - self.purge_days)
            train_start_idx = max(0, train_end_idx - self.train_window)
            train_dates_range = dates[train_start_idx:train_end_idx]
            if not train_dates_range:
                idx += self.step_days
                continue

            df_train = df[df["date"].isin(train_dates_range)].copy()
            df_train = df_train.dropna(subset=[label_col])
            if len(df_train) < self.min_train_rows:
                idx += self.step_days
                continue

            df_today = df[df["date"] == t_feat].copy()
            if df_today.empty:
                idx += self.step_days
                continue

            avail_feats = [f for f in features if f in df_train.columns and f in df_today.columns]
            if not avail_feats:
                idx += self.step_days
                continue

            # ---- 训练 & 推断 ----
            try:
                model = model_cls(**model_kwargs)
                X_train = df_train[avail_feats]
                y_train = df_train[label_col]

                if model_type == "gru":
                    model.fit(df_train, y_train, label_col=label_col)
                    scores = model.predict(df_today)
                elif model_type == "lambdarank":
                    groups = df_train.groupby("date").size().tolist()
                    model.fit(X_train, y_train, groups=groups)
                    scores = model.predict(df_today[avail_feats])
                else:
                    model.fit(X_train, y_train)
                    scores = model.predict(df_today[avail_feats])
                df_today = df_today.copy()
                df_today["pred_score"] = scores.values if hasattr(scores, "values") else scores

            except Exception as e:
                logger.warning("模型 %s 训练失败 (date=%s): %s", model_type, t_feat, e)
                idx += self.step_days
                continue

            top = df_today.nlargest(self.top_k, "pred_score")
            if full_cross_section:
                all_predictions.append(df_today[["date", "ts_code", "pred_score"]].copy())
            else:
                all_predictions.append(top[["date", "ts_code", "pred_score"]].copy())

            if label_col in df_today.columns:
                valid = df_today[["pred_score", label_col]].dropna()
                if len(valid) >= 5:
                    ic, _ = spearmanr(valid["pred_score"], valid[label_col])
                    if np.isfinite(ic):
                        rank_ics.append({"date": t_feat, "rank_ic": ic})

            # ---- 逐日执行交易 ----
            exec_dates = dates[idx: idx + self.step_days]
            for exec_date in exec_dates:
                day_data = df[df["date"] == exec_date]
                if day_data.empty:
                    continue

                expired: List[Dict] = []
                port_value = cash
                for h in holdings:
                    h["days_held"] += 1
                    row = day_data[day_data["ts_code"] == h["ts_code"]]
                    if not row.empty:
                        price = float(row[close_col].iloc[0])
                        h["current_price"] = price
                        port_value += h["shares"] * price
                    else:
                        port_value += h["shares"] * h["current_price"]

                    if h["days_held"] >= self.hold_days:
                        if not row.empty and _can_sell(row.iloc[0]):
                            expired.append(h)
                        elif h["days_held"] >= self.hold_days * 3:
                            expired.append(h)  # 安全阀：超过 3× 持有期仍无法卖出则强制清仓

                for h in expired:
                    sell_value = h["shares"] * h["current_price"] * (1 - self.c_sell)
                    cash += sell_value
                    total_sell_value += h["shares"] * h["current_price"]
                    holdings.remove(h)

                equity.append(port_value)

            # ---- DrawdownGuard: 回撤过大禁止新买入 ----
            allow_buy = self.dd_guard.update(equity[-1])

            if not top.empty and exec_dates and allow_buy:
                buy_date = exec_dates[0]
                buy_data = df[df["date"] == buy_date]
                if not buy_data.empty:
                    target_alloc = (self.target_exposure * equity[-1]) / max(self.hold_days, 1)
                    buy_cash = min(cash, target_alloc)

                    holdings_value = sum(h["shares"] * h["current_price"] for h in holdings)
                    max_allow = max(0.0, self.max_gross_exposure * equity[-1] - holdings_value)
                    buy_cash = min(buy_cash, max_allow)

                    if buy_cash > 1000:
                        buy_codes = top["ts_code"].tolist()
                        prices: Dict[str, float] = {}
                        for code in buy_codes:
                            row = buy_data[buy_data["ts_code"] == code]
                            if not row.empty and _can_buy(row.iloc[0]):
                                p = _adj_open(row.iloc[0], close_col, use_adj)
                                if p > 0:
                                    prices[code] = p

                        if len(prices) < self.top_k:
                            extras = df_today.nlargest(self.top_k * 3, "pred_score")
                            for _, erow in extras.iterrows():
                                if len(prices) >= self.top_k:
                                    break
                                c = erow["ts_code"]
                                if c in prices:
                                    continue
                                rr = buy_data[buy_data["ts_code"] == c]
                                if not rr.empty and _can_buy(rr.iloc[0]):
                                    pp = _adj_open(rr.iloc[0], close_col, use_adj)
                                    if pp > 0:
                                        prices[c] = pp

                        if prices:
                            if self.risk_parity:
                                alloc_weights = self._risk_parity_weights(df, buy_date, prices, close_col)
                            else:
                                alloc_weights = {c: 1.0 / len(prices) for c in prices}

                            for code, price in prices.items():
                                alloc = buy_cash * alloc_weights.get(code, 1.0 / len(prices))
                                max_pos = self.max_position_per_name * equity[-1]
                                alloc = min(alloc, max_pos)

                                shares = int(alloc / (price * 100)) * 100
                                if shares > 0:
                                    cost = shares * price * (1 + self.c_buy)
                                    if cost <= cash:
                                        cash -= cost
                                        total_buy_value += shares * price
                                        holdings.append({
                                            "ts_code": code, "entry_price": price,
                                            "current_price": price, "shares": shares,
                                            "days_held": 0, "entry_date": buy_date,
                                        })

            n_steps += 1
            idx += self.step_days

        # ---- 汇总 ----
        equity_series = pd.Series(equity)
        n_eq = len(equity_series)
        split_60 = int(n_eq * 0.6)
        split_80 = int(n_eq * 0.8)

        avg_eq = equity_series.mean() if len(equity_series) > 0 else 1.0
        turnover = (
            (total_buy_value + total_sell_value) / (2.0 * max(avg_eq, 1.0)) / max(n_steps, 1)
            if n_steps > 0 else 0.0
        )

        return {
            "equity": equity_series,
            "predictions": pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame(),
            "stats_train": perf_stats(equity_series.iloc[:split_60]) if split_60 > 1 else {},
            "stats_val": perf_stats(equity_series.iloc[split_60:split_80]) if split_80 > split_60 + 1 else {},
            "stats_test": perf_stats(equity_series.iloc[split_80:]) if n_eq > split_80 + 1 else {},
            "rank_ics": rank_ics,
            "model_type": model_type,
            "turnover": turnover,
        }

    # ------------------------------------------------------------------
    #  风险平价仓位权重
    # ------------------------------------------------------------------

    def _risk_parity_weights(
        self, df: pd.DataFrame, date, prices: dict, close_col: str,
    ) -> Dict[str, float]:
        """反 ATR 加权：波动率低的股票分配更多资金。"""
        inv_vol: Dict[str, float] = {}
        for code, price in prices.items():
            sub = df[(df["ts_code"] == code) & (df["date"] <= date)].tail(15)
            if len(sub) >= 2 and "high" in sub.columns and "low" in sub.columns:
                prev = sub[close_col].shift(1)
                hl = (sub["high"] - sub["low"]).abs()
                hc = (sub["high"] - prev).abs()
                lc = (sub["low"] - prev).abs()
                tr = pd.concat([hl, hc, lc], axis=1).max(axis=1).dropna()
                atr_val = float(tr.mean()) if len(tr) > 0 else price * 0.02
            else:
                atr_val = price * 0.02
            inv_vol[code] = 1.0 / max(atr_val, 1e-6)

        total = sum(inv_vol.values())
        if total <= 0:
            return {c: 1.0 / len(prices) for c in prices}
        return {c: v / total for c, v in inv_vol.items()}


# ======================================================================
#  模型评分
# ======================================================================

def score_models(results: Dict[str, Dict], cfg: dict) -> pd.DataFrame:
    """
    对多个模型的回测结果做综合评分。

    改进：
    - 修复 overfit_gap 在 Sharpe=0 时误判为无效的 bug
    - 使用 ICIR（IC / std(IC)）代替 mean IC 做权重分配
    - 加入 cost_robust 评分维度
    """
    scoring_cfg = cfg.get("scoring", {})
    weights = scoring_cfg.get("weights", {
        "sharpe_val": 0.40, "cum_return_val": 0.15,
        "max_drawdown_val": 0.15, "turnover": -0.10,
        "cost_robust": 0.10, "overfit_gap": -0.10,
    })
    oos_days = scoring_cfg.get("oos_rank_ic_days", 60)
    weight_cap = scoring_cfg.get("weight_cap", 0.40)

    records = []
    for mk, res in results.items():
        sv = res.get("stats_val", {})
        s_train = res.get("stats_train", {})

        sharpe_val = float(sv.get("sharpe", 0))
        sharpe_train = float(s_train.get("sharpe", 0))
        cum_return_val = float(sv.get("cum_return", 0))
        max_dd_val = float(sv.get("max_dd", 0))
        turnover = float(res.get("turnover", 0.0))

        overfit_gap = sharpe_train - sharpe_val

        if abs(sharpe_train) > 1e-9:
            cost_robust = sharpe_val / sharpe_train
        else:
            cost_robust = 1.0

        ics = res.get("rank_ics", [])
        recent_ics = [r["rank_ic"] for r in ics[-oos_days:]] if ics else []
        mean_rank_ic = float(np.mean(recent_ics)) if recent_ics else 0.0

        if len(recent_ics) >= 3:
            ic_std = float(np.std(recent_ics))
            icir = mean_rank_ic / ic_std if ic_std > 1e-9 else 0.0
        else:
            icir = 0.0

        records.append({
            "model_key": mk,
            "model_family": res.get("model_type", mk),
            "sharpe_val": sharpe_val,
            "cum_return_val": cum_return_val,
            "max_drawdown_val": max_dd_val,
            "overfit_gap": overfit_gap,
            "mean_rank_ic": mean_rank_ic,
            "icir": icir,
            "turnover": turnover,
            "cost_robust": cost_robust,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    score = np.zeros(len(df))
    for col, w in weights.items():
        if col in df.columns:
            z = robust_zscore(df[col].values)
            score += w * z

    df["score_total"] = score
    df = df.sort_values("score_total", ascending=False).reset_index(drop=True)

    icir_vals = df["icir"].values.copy()
    icir_vals[icir_vals <= 0] = 0
    total_icir = icir_vals.sum()
    if total_icir > 0:
        model_weights = icir_vals / total_icir
        model_weights = np.clip(model_weights, 0, weight_cap)
        s = model_weights.sum()
        if s > 0:
            model_weights = model_weights / s
    else:
        model_weights = np.ones(len(df)) / len(df)

    df["model_weight"] = model_weights
    return df
