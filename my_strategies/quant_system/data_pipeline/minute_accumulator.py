"""
分钟级数据（仅 akshare），多数据源缓冲：主源失败则自动换备用源。

数据源顺序（可配置）：
- 5 分钟：① 东方财富 stock_zh_a_hist_min_em（支持日期范围）→ ② 新浪 stock_zh_a_minute(1min) 聚合为 5min（仅最近几天）
- 1 分钟：新浪 stock_zh_a_minute

存储结构：
    {storage_path}/5min/{SYMBOL}.parquet
    {storage_path}/1min/{SYMBOL}.parquet
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

MINUTE_COLUMN_MAP_EM = {
    "时间": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}

MINUTE_COLUMN_MAP_SINA = {
    "day": "datetime",
    "open": "open",
    "close": "close",
    "high": "high",
    "low": "low",
    "volume": "volume",
}


# 统一 5 分钟输出列
FIVE_MIN_COLS = ["datetime", "open", "close", "high", "low", "volume", "amount"]


def _fetch_5min_em(symbol: str, start_date: str, end_date: str, max_retry: int = 3) -> pd.DataFrame | None:
    """数据源①：东方财富 5 分钟，支持日期范围。"""
    last_e = None
    for attempt in range(max_retry):
        try:
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period="5",
                start_date=f"{start_date} 09:30:00",
                end_date=f"{end_date} 15:00:00",
                adjust="hfq",
            )
        except Exception as e:
            last_e = e
            if attempt < max_retry - 1:
                time.sleep(2.0 * (attempt + 1))
            continue
        if df is None or df.empty:
            return None
        break
    else:
        logger.debug("5min [东方财富] %s 失败(%d次): %s", symbol, max_retry, last_e)
        return None

    df = df.rename(columns=MINUTE_COLUMN_MAP_EM)
    keep = [c for c in FIVE_MIN_COLS if c in df.columns]
    df = df[keep]
    if "amount" not in df.columns:
        df["amount"] = 0.0
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["symbol"] = symbol.upper()
    return df


def _normalize_sina_tz(dt_series: pd.Series) -> pd.Series:
    """新浪接口可能返回 UTC 或无时区时间戳，统一转为北京时间(无时区标记)。"""
    dt = pd.to_datetime(dt_series)
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return dt


def _fetch_5min_sina_fallback(symbol: str) -> pd.DataFrame | None:
    """
    数据源②：新浪 1 分钟，聚合为 5 分钟（仅最近约 5 个交易日）。
    主源被限/封时用作缓冲。
    """
    prefix = "sh" if symbol.startswith("6") else "sz"
    sina_symbol = f"{prefix}{symbol}"
    try:
        df = ak.stock_zh_a_minute(symbol=sina_symbol, period="1")
    except Exception as e:
        logger.debug("5min [新浪兜底] %s 失败: %s", symbol, e)
        return None
    if df is None or df.empty:
        return None

    df = df.rename(columns=MINUTE_COLUMN_MAP_SINA)
    df["datetime"] = _normalize_sina_tz(df["datetime"])
    df = df.set_index("datetime")
    # 1min -> 5min: open=first, high=max, low=min, close=last, volume=sum
    res = df.resample("5min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(how="all")
    res["amount"] = 0.0
    res = res.reset_index()
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        res[col] = pd.to_numeric(res[col], errors="coerce")
    res = res.dropna(subset=["open", "close"], how="all")
    res["symbol"] = symbol.upper()
    res = res[["datetime", "open", "close", "high", "low", "volume", "amount", "symbol"]]
    return res


def _fetch_5min(symbol: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """单日单股票：仅用东方财富（按日回填时用）。"""
    return _fetch_5min_em(symbol, start_date, end_date)


def _fetch_1min(symbol: str) -> pd.DataFrame | None:
    """
    通过新浪拉取 1 分钟 K 线（最近 5 个交易日左右的数据）。

    新浪接口不支持自定义日期范围，返回最近可用数据。
    """
    # 新浪接口需要带市场前缀：沪市 sh，深市 sz
    prefix = "sh" if symbol.startswith("6") else "sz"
    sina_symbol = f"{prefix}{symbol}"
    try:
        df = ak.stock_zh_a_minute(symbol=sina_symbol, period="1")
    except Exception as e:
        logger.warning("1min %s 拉取失败: %s", symbol, e)
        return None
    if df is None or df.empty:
        return None

    df = df.rename(columns=MINUTE_COLUMN_MAP_SINA)
    keep = [c for c in ["datetime", "open", "close", "high", "low", "volume"] if c in df.columns]
    df = df[keep]
    df["datetime"] = _normalize_sina_tz(df["datetime"])
    df["symbol"] = symbol.upper()
    return df


def _date_range(start_yyyymmdd: str, end_yyyymmdd: str):
    """生成 start 到 end 的日期列表（含首尾），格式 YYYY-MM-DD。"""
    start = datetime.strptime(start_yyyymmdd, "%Y%m%d")
    end = datetime.strptime(end_yyyymmdd, "%Y%m%d")
    out = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def backfill_5min(
    symbols_6: list[str],
    start_date: str,
    end_date: str,
    storage_path: str | Path,
    sleep_sec: float = 0.5,
    use_fallback: bool = True,
) -> int:
    """
    按日期范围回填 5 分钟数据（与 run_data 的 start_date/end_date 一致时可一起拉全量）。

    断点续拉：若某只股票已有 parquet 且其最新 bar 日期 >= end_date，则跳过该只，不再请求。
    symbols_6: 6 位代码，如 ["600519", "000858"]
    start_date / end_date: YYYYMMDD
    写入 {storage_path}/5min/{SYMBOL}.parquet，增量追加、按时间去重。
    """
    storage = Path(storage_path).expanduser().resolve()
    period_dir = storage / "5min"
    period_dir.mkdir(parents=True, exist_ok=True)
    dates = _date_range(start_date, end_date)
    end_yyyymmdd = str(end_date).replace("-", "")[:8]
    success = 0
    skipped = 0
    for sym in symbols_6:
        sym = str(sym).zfill(6)
        parquet_path = period_dir / f"{sym.upper()}.parquet"
        if parquet_path.exists():
            try:
                old = pd.read_parquet(parquet_path)
                if not old.empty and "datetime" in old.columns:
                    max_ts = pd.to_datetime(old["datetime"]).max()
                    file_max_yyyymmdd = max_ts.strftime("%Y%m%d")
                    if file_max_yyyymmdd >= end_yyyymmdd:
                        skipped += 1
                        continue
            except Exception as e:
                logger.debug("读取 %s 跳过检查失败: %s", parquet_path.name, e)
        all_dfs = []
        for d in dates:
            df = _fetch_5min(sym, d, d)
            if df is not None and not df.empty:
                all_dfs.append(df)
            time.sleep(sleep_sec)
        # 缓冲：主源(东方财富)整只股票都失败时，用新浪兜底（仅最近几天）
        if not all_dfs and use_fallback:
            fallback = _fetch_5min_sina_fallback(sym)
            if fallback is not None and not fallback.empty:
                d_min = start_date if len(start_date) == 8 else start_date.replace("-", "")
                d_max = end_date if len(end_date) == 8 else end_date.replace("-", "")
                fallback["_d"] = fallback["datetime"].dt.strftime("%Y%m%d")
                fallback = fallback[(fallback["_d"] >= d_min) & (fallback["_d"] <= d_max)].drop(columns=["_d"])
                if not fallback.empty:
                    all_dfs = [fallback]
                    logger.info("5min [新浪兜底] %s 拉取 %d 条", sym, len(fallback))
            time.sleep(sleep_sec)
        if not all_dfs:
            continue
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["datetime"], keep="last")
        combined = combined.sort_values("datetime").reset_index(drop=True)
        parquet_path = period_dir / f"{sym.upper()}.parquet"
        if parquet_path.exists():
            old = pd.read_parquet(parquet_path)
            combined = pd.concat([old, combined], ignore_index=True)
            combined = combined.drop_duplicates(subset=["datetime"], keep="last")
            combined = combined.sort_values("datetime").reset_index(drop=True)
        combined.to_parquet(parquet_path, index=False)
        success += 1
    logger.info(
        "5min 回填完成: 成功 %d / %d 只 (跳过已覆盖至 %s 的 %d 只), 日期 %s → %s",
        success, len(symbols_6), end_yyyymmdd, skipped, start_date, end_date,
    )
    return success


def accumulate_minutes(
    symbols: list[str],
    periods: list[str],
    storage_path: str | Path,
    sleep_sec: float = 0.5,
    use_fallback: bool = True,
) -> dict[str, int]:
    """
    增量累积分钟数据到 parquet 文件。

    断点续拉：若某只股票已有 parquet 且其最新 bar 日期 >= 今日，则跳过该只。
    否则读取已有 parquet → 追加新数据 → 按时间去重后写回。

    Parameters
    ----------
    symbols : list[str]
        股票代码列表
    periods : list[str]
        分钟周期列表，如 ["1", "5"]
    storage_path : str | Path
        存储根目录
    sleep_sec : float
        请求间隔

    Returns
    -------
    dict[str, int]
        {period: 成功数}
    """
    storage = Path(storage_path)
    today = datetime.now().strftime("%Y-%m-%d")
    today_yyyymmdd = datetime.now().strftime("%Y%m%d")
    result = {}

    for period in periods:
        period_dir = storage / f"{period}min"
        period_dir.mkdir(parents=True, exist_ok=True)
        success = 0
        skipped = 0

        for sym in symbols:
            sym_6 = str(sym).split(".")[0].zfill(6)
            parquet_path = period_dir / f"{sym_6.upper()}.parquet"
            if parquet_path.exists():
                try:
                    old = pd.read_parquet(parquet_path)
                    if not old.empty and "datetime" in old.columns:
                        max_ts = pd.to_datetime(old["datetime"]).max()
                        file_max_yyyymmdd = max_ts.strftime("%Y%m%d")
                        if file_max_yyyymmdd >= today_yyyymmdd:
                            skipped += 1
                            continue
                except Exception as e:
                    logger.debug("读取 %s 跳过检查失败: %s", parquet_path.name, e)
            if period == "5":
                new_df = _fetch_5min(sym_6, today, today)
                if (new_df is None or new_df.empty) and use_fallback:
                    fb = _fetch_5min_sina_fallback(sym_6)
                    if fb is not None and not fb.empty:
                        today_8 = datetime.now().strftime("%Y%m%d")
                        fb["_d"] = fb["datetime"].dt.strftime("%Y%m%d")
                        fb = fb[fb["_d"] == today_8].drop(columns=["_d"])
                        if not fb.empty:
                            new_df = fb
                            logger.info("5min [新浪兜底] %s 当日 %d 条", sym_6, len(fb))
            elif period == "1":
                new_df = _fetch_1min(sym_6)
            else:
                logger.warning("不支持的周期: %s", period)
                continue

            if new_df is None or new_df.empty:
                time.sleep(sleep_sec)
                continue

            if parquet_path.exists():
                old_df = pd.read_parquet(parquet_path)
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["datetime", "symbol"], keep="last")
                combined = combined.sort_values("datetime").reset_index(drop=True)
            else:
                combined = new_df.sort_values("datetime").reset_index(drop=True)

            combined.to_parquet(parquet_path, index=False)
            success += 1
            time.sleep(sleep_sec)

        result[period] = success
        logger.info(
            "%smin 累积完成: 成功 %d / 总计 %d (跳过已有今日数据的 %d 只)",
            period, success, len(symbols), skipped,
        )

    return result
