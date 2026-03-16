"""
股票池管理

支持三种模式：
- builtin: 预设的约 1100 只大中盘 A 股
- tushare_top: 按流通市值动态筛选 Top N
- custom: 用户自定义代码列表
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def infer_exchange(symbol6: str) -> str:
    s = str(symbol6).zfill(6)
    if s.startswith("92"):
        return "BJ"
    if s.startswith(("60", "68")):
        return "SH"
    if s.startswith(("00", "30", "20", "03", "01")):
        return "SZ"
    return "SH"


def to_ts_code(symbol6: str) -> str:
    s = str(symbol6).zfill(6)
    return f"{s}.{infer_exchange(s)}"


def to_qlib_code(symbol6: str) -> str:
    s = str(symbol6).zfill(6)
    prefix = infer_exchange(s)
    return f"{prefix}{s}"


_BUILTIN_POOL: List[str] = [
    "601288", "601857", "601398", "600519", "300750", "601988", "601138",
    "601628", "600036", "601899", "601088", "600028", "601318", "601166",
    "600900", "601668", "600276", "601012", "000858", "000333", "600030",
    "601888", "000568", "002415", "600585", "603259", "600809", "002594",
    "000001", "601601", "600887", "601009", "002304", "600050", "601919",
    "002142", "600196", "000063", "002475", "601225", "600690", "600309",
    "601816", "002352", "000776", "603501", "600048", "601800", "002714",
    "300059", "600000", "601328", "002230", "600104", "601088", "000002",
    "002049", "601211", "600406", "300015", "300124", "601766", "600745",
    "002460", "000725", "600436", "601390", "600426", "601985", "002027",
    "601186", "600031", "000651", "601360", "600016", "601658", "300274",
    "000538", "002371", "300496", "603986", "600089", "600741", "601117",
    "002841", "600143", "600570", "002601", "600600", "002032", "601100",
    "300033", "600176", "600346", "601155", "002120", "601669", "002916",
]


def get_stock_pool(cfg: dict) -> Tuple[List[str], Dict[str, str]]:
    """
    返回 (ts_codes, name_map)。

    ts_codes: ["600519.SH", ...] 格式
    name_map: {"600519.SH": "贵州茅台", ...}（从 TuShare 获取或空 dict）
    """
    pool_cfg = cfg.get("data_pipeline", {}).get("stock_pool", {})
    source = pool_cfg.get("source", "builtin")
    exclude_bj = pool_cfg.get("exclude_bj", True)
    exclude_cyb = pool_cfg.get("exclude_cyb", False)
    exclude_kcb = pool_cfg.get("exclude_kcb", False)

    if source == "custom":
        codes_6 = pool_cfg.get("custom_codes", [])
    elif source == "tushare_top":
        codes_6 = _fetch_top_by_circ_mv(cfg, pool_cfg)
    else:
        codes_6 = list(_BUILTIN_POOL)

    codes_6 = [str(c).zfill(6) for c in codes_6]
    codes_6 = list(dict.fromkeys(codes_6))

    if exclude_bj:
        codes_6 = [c for c in codes_6 if not c.startswith("92")]
    if exclude_cyb:
        codes_6 = [c for c in codes_6 if not c.startswith(("300", "301"))]
    if exclude_kcb:
        codes_6 = [c for c in codes_6 if not c.startswith("688")]

    ts_codes = [to_ts_code(c) for c in codes_6]
    name_map = _try_load_names(ts_codes, cfg)

    logger.info("股票池: %d 只 (source=%s)", len(ts_codes), source)
    return ts_codes, name_map


def _fetch_top_by_circ_mv(cfg: dict, pool_cfg: dict) -> List[str]:
    """按 A 股流通市值从高到低取 Top N，已应用 exclude_bj/cyb/kcb。"""
    try:
        import tushare as ts
        from datetime import datetime, timedelta

        dp = cfg.get("data_pipeline", {})
        token = dp.get("tushare_token") or os.environ.get("TUSHARE_TOKEN", "")
        if not token:
            logger.warning("未配置 tushare_token，回退到 builtin 池")
            return list(_BUILTIN_POOL)
        pro = ts.pro_api(token)

        # 取最近一个交易日：end_date 或今天
        end_date = dp.get("end_date")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")
        trade_date = str(end_date).replace("-", "")[:8]

        df = pro.daily_basic(trade_date=trade_date, fields="ts_code,circ_mv")
        if df is None or df.empty:
            # 若当天无数据（非交易日），尝试前几日
            for d in range(1, 6):
                try:
                    d0 = datetime.strptime(trade_date, "%Y%m%d") - timedelta(days=d)
                    trade_date = d0.strftime("%Y%m%d")
                    df = pro.daily_basic(trade_date=trade_date, fields="ts_code,circ_mv")
                    if df is not None and not df.empty:
                        break
                except Exception:
                    continue
        if df is None or df.empty:
            logger.warning("TuShare daily_basic 返回空，回退到 builtin 池")
            return list(_BUILTIN_POOL)

        df = df.dropna(subset=["circ_mv"])
        df["code_6"] = df["ts_code"].str.split(".").str[0].str.zfill(6)

        exclude_bj = pool_cfg.get("exclude_bj", True)
        exclude_cyb = pool_cfg.get("exclude_cyb", False)
        exclude_kcb = pool_cfg.get("exclude_kcb", False)
        if exclude_bj:
            df = df[~df["code_6"].str.startswith("92")]
        if exclude_cyb:
            df = df[~df["code_6"].str.startswith(("300", "301"))]
        if exclude_kcb:
            df = df[~df["code_6"].str.startswith("688")]

        n = int(pool_cfg.get("size", 500))
        df = df.sort_values("circ_mv", ascending=False).head(n)
        return df["code_6"].tolist()
    except Exception as e:
        logger.warning("按流通市值取 Top 失败: %s，回退到 builtin 池", e)
        return list(_BUILTIN_POOL)


def _try_load_names(ts_codes: List[str], cfg: dict) -> Dict[str, str]:
    try:
        import tushare as ts
        token = (
            cfg.get("data_pipeline", {}).get("tushare_token")
            or os.environ.get("TUSHARE_TOKEN", "")
        )
        if not token:
            return {}
        pro = ts.pro_api(token)
        df = pro.stock_basic(fields="ts_code,name")
        if df is not None and not df.empty:
            return dict(zip(df["ts_code"], df["name"]))
    except Exception:
        pass
    return {}
