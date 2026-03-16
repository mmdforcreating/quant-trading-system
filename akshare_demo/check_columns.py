"""
用 1～2 支股票检查 akshare 各接口实际返回的列变量。
运行: python check_columns.py
若网络异常会打印「预期列」（来自源码），不影响查看有哪些列可拉。
"""
import sys
from datetime import datetime, timedelta

try:
    import akshare as ak
except ImportError:
    print("请先安装: pip install akshare")
    sys.exit(1)

SYMBOLS = ["000001", "600519"]  # 平安银行、贵州茅台


def get_recent_trading_days():
    """返回 (start_ymd, end_ymd, last_day_ymd, last_day_str)。
    日线用 start_ymd~end_ymd；分钟线用 last_day 的 09:30~15:00。
    """
    d = datetime.now().date()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    end_ymd = d.strftime("%Y%m%d")
    end_str = d.strftime("%Y-%m-%d")
    n = 5
    while n > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    start_ymd = d.strftime("%Y%m%d")
    return start_ymd, end_ymd, end_ymd, end_str


START_YMD, END_YMD, LAST_YMD, LAST_STR = get_recent_trading_days()


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def show_df(name, df, max_rows=2):
    if df is None or df.empty:
        print(f"[{name}] 无数据（可能网络或日期问题）")
        return
    print(f"[{name}] 列({len(df.columns)}个):", list(df.columns))
    print(df.head(max_rows).to_string())
    print()


# ---------- 日线（最近约 5 个交易日）----------
section("1. 日线 stock_zh_a_hist(period='daily')")
print(f"  日期范围: {START_YMD} ~ {END_YMD}")
try:
    df = ak.stock_zh_a_hist(symbol=SYMBOLS[0], period="daily", start_date=START_YMD, end_date=END_YMD)
    show_df("日线-000001", df)
except Exception as e:
    print("请求失败:", e)
    print("预期列: 日期, 股票代码, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率")

# ---------- 周线、月线（用近期区间）----------
for period, label in [("weekly", "周线"), ("monthly", "月线")]:
    section(f"2. {label} stock_zh_a_hist(period='{period}')")
    try:
        df = ak.stock_zh_a_hist(symbol=SYMBOLS[0], period=period, start_date=START_YMD, end_date=END_YMD)
        show_df(f"{label}-000001", df)
    except Exception as e:
        print("请求失败:", e)
        print("预期列: 与日线相同（日期, 股票代码, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率）")

# ---------- 分钟线（最近一个交易日 09:30~15:00）----------
section("3. 1分钟线 stock_zh_a_hist_min_em(period='1')")
print(f"  日期: {LAST_STR} 09:30~11:30")
try:
    df = ak.stock_zh_a_hist_min_em(
        symbol=SYMBOLS[0],
        start_date=f"{LAST_STR} 09:30:00",
        end_date=f"{LAST_STR} 11:30:00",
        period="1",
    )
    show_df("1分钟-000001", df)
except Exception as e:
    print("请求失败:", e)
    print("预期列: 时间, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 均价")

section("4. 5分钟线 stock_zh_a_hist_min_em(period='5')")
print(f"  日期: {LAST_STR} 09:30~15:00")
try:
    df = ak.stock_zh_a_hist_min_em(
        symbol=SYMBOLS[0],
        start_date=f"{LAST_STR} 09:30:00",
        end_date=f"{LAST_STR} 15:00:00",
        period="5",
    )
    show_df("5分钟-000001", df)
except Exception as e:
    print("请求失败:", e)
    print("预期列: 时间, 开盘, 收盘, 最高, 最低, 涨跌幅, 涨跌额, 成交量, 成交额, 振幅, 换手率")

# ---------- 第二只股票日线 ----------
section("5. 日线 600519（列与 000001 一致）")
try:
    df = ak.stock_zh_a_hist(symbol=SYMBOLS[1], period="daily", start_date=START_YMD, end_date=END_YMD)
    show_df("日线-600519", df)
except Exception as e:
    print("请求失败:", e)

print("\n" + "=" * 60)
print("  完整接口与列说明见: A股数据接口与列变量说明.md")
print("=" * 60)
