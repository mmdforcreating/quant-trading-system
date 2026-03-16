# Akshare 示例

Akshare 是一个开源的财经数据接口库，可以获取 A 股、指数、基金、期货等数据。

## 安装

```bash
pip install akshare
```

## 常用接口示例

| 功能 | 接口示例 |
|------|----------|
| A 股日线 | `ak.stock_zh_a_hist(symbol="000001", period="daily")` |
| 实时行情 | `ak.stock_zh_a_spot_em()` |
| 指数日线 | `ak.stock_zh_index_daily(symbol="sh000001")` |
| 北向资金 | `ak.stock_hsgt_north_net_flow_in_em()` |

运行 `demo.py` 可查看实际返回的数据结构。
