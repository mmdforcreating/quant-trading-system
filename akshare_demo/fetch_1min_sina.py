"""
用新浪数据源拉 A 股 1 分钟线（东财 1 分钟易断连时用此脚本）。
返回最近约 1970 根 K 线，包含 11 号、12 号等近期交易日。
运行: python fetch_1min_sina.py
"""
import akshare as ak

SYMBOL = "sz000001"  # 平安银行，沪市用 sh600519
TARGET_DATE = "2026-03-11"  # 只看这一天的 1 分钟

df = ak.stock_zh_a_minute(symbol=SYMBOL, period="1", adjust="")
print("新浪 1 分钟线 列:", list(df.columns))
print("总行数:", len(df))
print("时间范围:", df["day"].iloc[0], "~", df["day"].iloc[-1])

# 筛出 11 号
df["date"] = df["day"].astype(str).str[:10]
sub = df[df["date"] == TARGET_DATE]
print(f"\n{TARGET_DATE} 当天 1 分钟线 行数:", len(sub))
if not sub.empty:
    print(sub.to_string())
else:
    print("该日无数据（可能非交易日或不在最近 1970 根内）")
