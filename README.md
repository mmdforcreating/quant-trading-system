# 量化交易系统 (quant-trading-system)

多模型 Walk-Forward 回测与交易计划生成系统。  
日频模型（CatBoost / LightGBM / LambdaRank / Ridge）+ 分钟频模型（GRU）集成。

## 目录结构

```
Quant/
├── 总控台/                    # 命令行入口
│   ├── quant_cli.py           #   统一 CLI (fetch/diagnose/weekly/daily/backtest)
│   ├── data_fetch.py          #   数据拉取
│   ├── factor_diagnosis.py    #   因子诊断
│   ├── run_weekly.py          #   周频训练
│   ├── run_daily.py           #   每日推断
│   └── benefit_backtest.py    #   回测
├── my_strategies/             # 核心逻辑
│   ├── quant_system/          #   引擎代码
│   │   ├── configs/config.yaml#     全局配置
│   │   ├── models/            #     CatBoost/LGBM/Ridge/GRU 等
│   │   ├── trainer/           #     Walk-Forward 引擎
│   │   ├── ensemble/          #     模型集成
│   │   ├── risk/              #     ATR 风控
│   │   ├── portfolio/         #     交易计划 + Backtrader 策略
│   │   ├── factor_engine/     #     因子计算与筛选
│   │   ├── data_pipeline/     #     数据拉取/转换
│   │   └── utils/             #     配置管理/设备检测
│   ├── scripts/               #   工具脚本
│   ├── requirements.txt       #   Python 依赖
│   ├── test_pipeline.py       #   端到端测试
│   └── data/                  #   数据目录 (不在 Git 中，需本地准备)
│       ├── panel.parquet      #     日频面板
│       └── minute/1min/       #     分钟 K 线 (每只股票一个 parquet)
└── README.md
```

---

## Windows 新电脑完整安装指南

### 第 1 步：安装 Python

1. 下载 Python 3.9+：https://www.python.org/downloads/
2. 安装时 **勾选「Add Python to PATH」**
3. 打开 CMD/PowerShell 验证：`python --version`

### 第 2 步：安装 Git

1. 下载 Git：https://git-scm.com/download/win
2. 安装后打开 Git Bash 或 CMD 验证：`git --version`

### 第 3 步：克隆仓库

```bash
git clone https://github.com/mmdforcreating/quant-trading-system.git Quant
cd Quant
```

### 第 4 步：创建虚拟环境 + 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell 用: .venv\Scripts\Activate.ps1

pip install -r my_strategies/requirements.txt
```

> **PyTorch 注意**：上面会安装 CPU 版 torch。若 Windows 有 NVIDIA GPU 想加速 GRU，改用：
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 第 5 步：安装 Qlib（可选，用于 Alpha158 因子）

```bash
pip install pyqlib
```

或从源码安装（更灵活）：
```bash
git clone https://github.com/microsoft/qlib.git qlib
pip install -e qlib/
```

### 第 6 步：准备数据

数据文件不在 Git 中，需要从本机拷贝或重新生成：

**方式 A — 从 Mac 拷贝（推荐，最快）：**
将以下文件/文件夹从 Mac 复制到 Windows 的 `Quant/my_strategies/` 下：
- `data/panel.parquet`（日频面板）
- `data/minute/1min/`（分钟 parquet，5000+ 文件）
- `data/raw/`（原始 CSV，可选）
- `output/selected_features.txt`（因子列表）

**方式 B — 在 Windows 上重新拉取：**
```bash
python 总控台/quant_cli.py fetch --freq daily
python 总控台/quant_cli.py diagnose
```
分钟数据若为外部购买，手动放到 `my_strategies/data/minute/1min/` 即可。

### 第 7 步：验证系统

```bash
python my_strategies/test_pipeline.py
```

看到「全部 6 个阶段通过!」即说明系统正常运行。

### 日常使用

```bash
cd Quant
.venv\Scripts\activate

# 数据拉取（日频）
python 总控台/quant_cli.py fetch --freq daily

# 因子诊断
python 总控台/quant_cli.py diagnose

# 周频训练 + 模型权重
python 总控台/quant_cli.py weekly

# 每日推断 + 交易计划
python 总控台/quant_cli.py daily

# 历史回测
python 总控台/quant_cli.py backtest --mode historical
```

---

## Mac 本机运行

```bash
cd /Users/mengmingdong/Quant
source .venv/bin/activate   # 或 conda activate qlib_lab
python 总控台/quant_cli.py fetch --freq daily
python 总控台/quant_cli.py weekly
python 总控台/quant_cli.py daily
python 总控台/quant_cli.py backtest --mode historical
```

## 依赖总览

| 包 | 用途 |
|---|---|
| pandas / numpy / scipy | 数据处理 |
| catboost / lightgbm / scikit-learn | 日频 ML 模型 |
| torch | GRU 分钟频模型 |
| backtrader | 历史回测引擎 |
| pyarrow | Parquet 读写 |
| tushare / akshare | 数据源 API |
| pyqlib (可选) | Alpha158 因子 |
| PyYAML / tqdm | 配置与进度条 |
