# 量化交易系统

日频 + 分钟频多模型 Walk-Forward 回测与交易计划生成（CatBoost / LightGBM / LambdaRank / Ridge / GRU）。

## 目录结构

- **总控台/** — 命令行入口：`quant_cli.py`（fetch / diagnose / weekly / daily / backtest）
- **my_strategies/** — 核心逻辑：因子、模型、WF 引擎、风控、交易计划、回测
- **qlib/** — 需单独克隆，见下方「另一台设备运行」

## 本机运行（已有环境）

```bash
cd /Users/mengmingdong/Quant
# 使用虚拟环境
source .venv/bin/activate   # 或 conda activate qlib_lab

# 数据拉取（日频）
python 总控台/quant_cli.py fetch --freq daily

# 因子诊断
python 总控台/quant_cli.py diagnose

# 周频训练 + 权重
python 总控台/quant_cli.py weekly

# 每日推断 + 交易计划
python 总控台/quant_cli.py daily

# 回测
python 总控台/quant_cli.py backtest --mode historical
```

## 另一台设备运行（从 GitHub 克隆后）

1. **克隆本仓库**
   ```bash
   git clone https://github.com/<你的用户名>/<仓库名>.git Quant
   cd Quant
   ```

2. **创建虚拟环境并安装依赖**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r my_strategies/requirements.txt
   pip install pyarrow   # 分钟数据 parquet
   ```

3. **（可选）Qlib（用于 Alpha158 因子）**
   ```bash
   git clone https://github.com/microsoft/qlib.git qlib
   pip install -e qlib/
   ```

4. **配置**
   - 在 `my_strategies/quant_system/configs/config.yaml` 中填写数据源等配置。
   - 日频数据需自行拉取或放置到 `my_strategies/data/`；分钟数据若使用外部文件，放到 `my_strategies/data/minute/1min/*.parquet`。

5. **跑通测试（10 只股票）**
   ```bash
   python my_strategies/test_pipeline.py
   ```

6. **使用总控台**
   ```bash
   python 总控台/quant_cli.py fetch --freq daily
   python 总控台/quant_cli.py diagnose
   python 总控台/quant_cli.py weekly
   python 总控台/quant_cli.py daily
   python 总控台/quant_cli.py backtest --mode historical
   ```

## 依赖概要

- Python 3.8+
- pandas, numpy, scipy, pyyaml, tqdm
- catboost, lightgbm, scikit-learn, torch（GRU）
- pyarrow（分钟 parquet）
- 可选：qlib、akshare、tushare（按 config 使用）

若 `my_strategies/requirements.txt` 不存在，可根据上述依赖自行创建。
