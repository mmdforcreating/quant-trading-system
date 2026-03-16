"""
TuShare 数据拉取引擎

支持增量抓取 5 类数据：daily, daily_basic, adj_factor, moneyflow, fina_indicator。
移植自 quant_core 的 data_supply/fetcher.py，增加重试与 checkpoint 机制。
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def yyyymmdd_today() -> str:
    return datetime.today().strftime("%Y%m%d")


def next_day(d: str) -> str:
    return (datetime.strptime(d, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")


def prev_day(d: str) -> str:
    return (datetime.strptime(d, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")


def date_range(start: str, end: str) -> List[str]:
    out, d = [], start
    while d <= end:
        out.append(d)
        d = next_day(d)
    return out


def _get_min_max(hist: pd.DataFrame, code: str, col: str):
    if hist.empty or "ts_code" not in hist.columns or col not in hist.columns:
        return None, None
    m = hist.loc[hist["ts_code"] == code, col]
    if len(m) == 0:
        return None, None
    return str(m.min()), str(m.max())


def safe_read_csv(path: str, **kw) -> pd.DataFrame:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_df(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def request_with_retry(
    fn: Callable,
    max_retry: int = 5,
    backoff_base: float = 2.0,
    sleep_s: float = 0.3,
    **kwargs,
) -> pd.DataFrame:
    last_exc = None
    for i in range(max_retry):
        try:
            df = fn(**kwargs)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
            if i < max_retry - 1:
                time.sleep(backoff_base ** i * sleep_s)
            continue
        except Exception as e:
            last_exc = e
            time.sleep(backoff_base ** i * sleep_s)
    if last_exc:
        logger.warning("请求失败 (重试 %d 次): %s", max_retry, last_exc)
    return pd.DataFrame()


class TuShareFetcher:
    """TuShare 数据抓取器，管理 5 类数据的增量拉取。"""

    def __init__(self, cfg: dict):
        import tushare as ts

        dp = cfg.get("data_pipeline", {})
        token = dp.get("tushare_token") or os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            raise ValueError(
                "未配置 TuShare Token。请设置环境变量 TUSHARE_TOKEN 或在配置中设置 data_pipeline.tushare_token"
            )
        self.pro = ts.pro_api(token)

        fetch_cfg = dp.get("fetch", {})
        self.sleep_s = fetch_cfg.get("sleep_s", 0.3)
        self.max_retry = fetch_cfg.get("max_retry", 5)
        self.backoff = fetch_cfg.get("backoff_base", 2)
        self.batch_by_date = fetch_cfg.get("batch_by_trade_date", True)

        raw_dir = dp.get("raw_data_dir", "data/raw")
        self.raw_dir = str(Path(raw_dir).expanduser())
        os.makedirs(self.raw_dir, exist_ok=True)

    def _retry(self, fn, **kw) -> pd.DataFrame:
        return request_with_retry(
            fn, max_retry=self.max_retry, backoff_base=self.backoff,
            sleep_s=self.sleep_s, **kw,
        )

    def fetch_all(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        skip_fina: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """拉取全部 5 类数据，返回 {name: DataFrame}。"""
        results = {}
        for name, fn, fields, date_col in [
            ("daily", self.pro.daily, None, "trade_date"),
            ("daily_basic", self.pro.daily_basic, None, "trade_date"),
            ("adj_factor", self.pro.adj_factor, None, "trade_date"),
            ("moneyflow", self.pro.moneyflow, None, "trade_date"),
        ]:
            logger.info("拉取 %s ...", name)
            df = self._incremental_fetch(
                name=name, fn=fn, ts_codes=ts_codes,
                start_date=start_date, end_date=end_date,
                date_col=date_col, fields=fields,
            )
            results[name] = df

        if not skip_fina:
            logger.info("拉取 fina_indicator ...")
            results["fina_indicator"] = self._fetch_fina(
                ts_codes=ts_codes, start_date=start_date, end_date=end_date,
            )

        return results

    def _incremental_fetch(
        self,
        name: str,
        fn: Callable,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        date_col: str = "trade_date",
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        fpath = os.path.join(self.raw_dir, f"{name}.csv")
        hist = safe_read_csv(fpath, dtype={date_col: str})
        if not hist.empty and date_col in hist.columns:
            hist[date_col] = hist[date_col].astype(str)

        new_rows = []

        if self.batch_by_date:
            dates_needed = set()
            for code in ts_codes:
                mn, mx = _get_min_max(hist, code, date_col)
                if mn is None:
                    for d in date_range(start_date, end_date):
                        dates_needed.add(d)
                else:
                    if start_date < mn:
                        for d in date_range(start_date, prev_day(mn)):
                            dates_needed.add(d)
                    if mx and end_date > mx:
                        for d in date_range(next_day(mx), end_date):
                            dates_needed.add(d)

            for d in sorted(dates_needed):
                kw = {"trade_date": d}
                if fields:
                    kw["fields"] = fields
                df = self._retry(fn, **kw)
                if not df.empty and "ts_code" in df.columns:
                    df = df[df["ts_code"].isin(ts_codes)].copy()
                    if date_col in df.columns:
                        df[date_col] = df[date_col].astype(str)
                    if not df.empty:
                        new_rows.append(df)
                time.sleep(self.sleep_s)
        else:
            for code in ts_codes:
                mn, mx = _get_min_max(hist, code, date_col)
                fetch_start = start_date if mn is None else (next_day(mx) if mx else start_date)
                if fetch_start > end_date:
                    continue
                kw = {"ts_code": code, "start_date": fetch_start, "end_date": end_date}
                if fields:
                    kw["fields"] = fields
                df = self._retry(fn, **kw)
                if not df.empty and date_col in df.columns:
                    df[date_col] = df[date_col].astype(str)
                    new_rows.append(df)
                time.sleep(self.sleep_s)

        inc = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()
        out = pd.concat([hist, inc], ignore_index=True) if not hist.empty else inc.copy()

        if not out.empty:
            dedup_cols = ["ts_code", date_col] if "ts_code" in out.columns and date_col in out.columns else None
            if dedup_cols:
                out = out.drop_duplicates(subset=dedup_cols)
                out = out.sort_values(dedup_cols).reset_index(drop=True)

        if not out.empty:
            save_df(out, fpath)
            logger.info("[%s] 保存 -> %s | %d 行", name, fpath, len(out))
        return out

    def _fetch_fina(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        fpath = os.path.join(self.raw_dir, "fina_indicator.csv")
        date_col = "ann_date"
        fields = "ts_code,ann_date,end_date,roe,roa,netprofit_yoy,tr_yoy,grossprofit_margin,netprofit_margin,debt_to_assets,current_ratio,quick_ratio"

        hist = safe_read_csv(fpath, dtype={date_col: str})
        if not hist.empty and date_col in hist.columns:
            hist[date_col] = hist[date_col].astype(str)

        ckpt_path = os.path.join(self.raw_dir, "fina_checkpoint.json")
        verified = None
        if os.path.exists(ckpt_path):
            try:
                with open(ckpt_path) as f:
                    verified = json.load(f).get("verified_through")
            except Exception:
                pass

        new_rows = []

        if verified and not hist.empty:
            inc_start = next_day(str(verified))
            if inc_start > end_date:
                logger.info("[fina] 已验证至 %s，跳过", verified)
                return hist
            for code in ts_codes:
                try:
                    df = self._retry(
                        self.pro.fina_indicator,
                        ts_code=code, start_date=inc_start, end_date=end_date, fields=fields,
                    )
                except Exception:
                    df = pd.DataFrame()
                if not df.empty and date_col in df.columns:
                    df[date_col] = df[date_col].astype(str)
                    if not df.empty:
                        new_rows.append(df)
                time.sleep(self.sleep_s)
        else:
            for code in ts_codes:
                try:
                    df = self._retry(
                        self.pro.fina_indicator,
                        ts_code=code, start_date=start_date, end_date=end_date,
                        fields=fields,
                    )
                except TypeError:
                    df = self._retry(self.pro.fina_indicator, ts_code=code, fields=fields)
                if not df.empty and date_col in df.columns:
                    df[date_col] = df[date_col].astype(str)
                    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
                    if not df.empty:
                        new_rows.append(df)
                time.sleep(self.sleep_s)

        inc = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()
        out = pd.concat([hist, inc], ignore_index=True) if not hist.empty else inc.copy()

        if not out.empty:
            dedup = ["ts_code", "ann_date", "end_date"] if all(c in out.columns for c in ["ts_code", "ann_date", "end_date"]) else ["ts_code", "ann_date"]
            out = out.drop_duplicates(subset=[c for c in dedup if c in out.columns])
            out = out.sort_values(["ts_code", date_col]).reset_index(drop=True)
            save_df(out, fpath)

        try:
            with open(ckpt_path, "w") as f:
                json.dump({"verified_through": end_date}, f)
        except Exception:
            pass

        logger.info("[fina] 保存 -> %s | %d 行", fpath, len(out))
        return out
