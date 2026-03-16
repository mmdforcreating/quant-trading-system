"""
Akshare 数据抓取模块

负责从 akshare 拉取 A 股日线数据并存为 Qlib 可消费的 CSV 格式。
使用东方财富数据源 (ak.stock_zh_a_hist)。

限速策略：每只股票之间 sleep 0.3s，避免触发反爬。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import akshare as ak
import pandas as pd

logger = logging.getLogger(__name__)

COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def get_all_a_stock_codes() -> list[str]:
    """获取全部 A 股代码列表（不含退市、ST 等筛选，由下游处理）。"""
    info = ak.stock_info_a_code_name()
    return info["code"].tolist()


def fetch_daily_single(
    symbol: str,
    start_date: str,
    end_date: str,
    adjust: str = "hfq",
) -> pd.DataFrame | None:
    """
    拉取单只股票的日线数据。

    Parameters
    ----------
    symbol : str
        6 位股票代码，如 "000001"
    start_date, end_date : str
        "YYYYMMDD" 格式
    adjust : str
        复权方式: hfq(后复权) / qfq(前复权) / 空字符串(不复权)

    Returns
    -------
    pd.DataFrame | None
        标准化后的 DataFrame，列为 date/open/close/high/low/volume/amount；
        拉取失败则返回 None。
    """
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    except Exception as e:
        logger.warning("拉取 %s 日线失败: %s", symbol, e)
        return None

    if df is None or df.empty:
        return None

    df = df.rename(columns=COLUMN_MAP)
    keep = [c for c in ["date", "open", "close", "high", "low", "volume", "amount"] if c in df.columns]
    df = df[keep]

    df["symbol"] = symbol.upper()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_daily_all(
    start_date: str,
    end_date: str,
    csv_dir: str | Path,
    stocks: str | list[str] = "all",
    sleep_sec: float = 0.3,
) -> int:
    """
    批量拉取日线数据并按股票代码存为独立 CSV。

    每只股票存为 ``{csv_dir}/{SYMBOL}.csv``，可直接被 dump_bin.py 消费。

    Parameters
    ----------
    start_date, end_date : str
        "YYYYMMDD"
    csv_dir : str | Path
        CSV 输出目录
    stocks : str | list[str]
        "all" 拉全市场，或提供代码列表
    sleep_sec : float
        每只股票请求间隔秒数

    Returns
    -------
    int
        成功下载的股票数量
    """
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    if stocks == "all":
        symbols = get_all_a_stock_codes()
        logger.info("全市场股票数: %d", len(symbols))
    else:
        symbols = list(stocks)

    success = 0
    for i, sym in enumerate(symbols):
        df = fetch_daily_single(sym, start_date, end_date)
        if df is not None and not df.empty:
            prefix = "SH" if sym.startswith("6") else "SZ" if sym.startswith(("0", "3")) else "BJ"
            out = csv_dir / f"{prefix}{sym}.csv"
            df.to_csv(out, index=False)
            success += 1

        if (i + 1) % 100 == 0:
            logger.info("进度: %d/%d (成功 %d)", i + 1, len(symbols), success)

        time.sleep(sleep_sec)

    logger.info("日线下载完成: 成功 %d / 总计 %d", success, len(symbols))
    return success
