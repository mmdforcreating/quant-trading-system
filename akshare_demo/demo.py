"""
Akshare 快速体验：看看接口返回的数据长什么样
运行: python demo.py
"""
import akshare as ak

print("=" * 60)
print("  Akshare 数据示例")
print("=" * 60)

# 1. 平安银行(000001) 最近 5 天日线
print("\n[1] A股日线 stock_zh_a_hist (平安银行 000001, 最近5天)")
print("-" * 40)
df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20240301", end_date="20240315")
print(df.head().to_string())
print(f"列名: {list(df.columns)}")

# 2. 全市场实时行情（只取前 3 条）
print("\n[2] 全市场实时行情 stock_zh_a_spot_em (前3条)")
print("-" * 40)
spot = ak.stock_zh_a_spot_em()
print(spot.head(3).to_string())
print(f"列名: {list(spot.columns)}")

# 3. 上证指数日线
print("\n[3] 指数日线 stock_zh_index_daily (上证指数)")
print("-" * 40)
index_df = ak.stock_zh_index_daily(symbol="sh000001")
print(index_df.tail(5).to_string())
print(f"列名: {list(index_df.columns)}")

print("\n" + "=" * 60)
print("  运行完成。若要对接 Qlib，需将列名映射为 date/open/high/low/close/volume 等")
print("=" * 60)
