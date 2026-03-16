#!/usr/bin/env python3
"""
量化交易系统 - 统一控制台 (quant_cli)

五大子命令:

  fetch     数据原料车间          日频(TuShare) / 分钟频(Akshare)
  diagnose  因子 X 光室            IC/ICIR/VIF 诊断 → selected_features.txt
  weekly    指挥官军训             Walk-Forward 训练 + 权重分配 → 模型 + 权重
  daily     前线狙击手             极速推断 + ATR 风控 → trade_plan.csv
  backtest  终极沙盘推演          Backtrader 带摩擦成本 T+1 回测

用法示例:

  python quant_cli.py fetch --freq daily
  python quant_cli.py fetch --freq minute
  python quant_cli.py diagnose
  python quant_cli.py weekly
  python quant_cli.py daily
  python quant_cli.py daily --skip-update --date 20250310
  python quant_cli.py backtest --mode historical
  python quant_cli.py backtest --mode daily --trade-plan output/daily_20250310/trade_plan.csv
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# 确保 quant_system 可被 import
_PROJECT = Path(__file__).resolve().parent.parent / "my_strategies"
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quant_cli")

DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent.parent
    / "my_strategies" / "quant_system" / "configs" / "config.yaml"
)


def _load_cfg(config_path: str):
    """加载 YAML 配置为 ConfigManager 实例。"""
    from quant_system.utils.config_manager import ConfigManager
    return ConfigManager(config_path)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  子命令入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_fetch(args):
    """数据原料车间: 日频(TuShare+panel) / 分钟线(Akshare)。"""
    from data_fetch import run as run_fetch

    cfg = _load_cfg(args.config)
    logger.info("=" * 60)
    logger.info("📦  数据拉取  freq=%s", args.freq)
    logger.info("=" * 60)

    t0 = time.time()
    run_fetch(cfg, freq=args.freq, skip_fina=args.skip_fina, export_qlib=args.export_qlib)
    logger.info("耗时 %.1f 秒", time.time() - t0)


def cmd_diagnose(args):
    """因子 X 光室: 计算因子 → IC/ICIR/VIF → selected_features.txt。"""
    from factor_diagnosis import run as run_diag

    cfg = _load_cfg(args.config)
    logger.info("=" * 60)
    logger.info("🔬  因子诊断")
    logger.info("=" * 60)

    t0 = time.time()
    selected, ic_table = run_diag(cfg, skip_alpha158=args.skip_alpha158, method=args.method)
    logger.info("诊断完成: 选定 %d 个因子, 耗时 %.1f 秒", len(selected), time.time() - t0)


def cmd_weekly(args):
    """指挥官军训: Walk-Forward 训练 + RankIC 权重分配。"""
    from run_weekly import run as run_wk

    cfg = _load_cfg(args.config)
    logger.info("=" * 60)
    logger.info("🎯  周频训练 (Walk-Forward)")
    logger.info("=" * 60)

    t0 = time.time()
    scores_df, weights = run_wk(cfg, start_date=args.start_date, end_date=args.end_date)
    logger.info("训练完成, 耗时 %.1f 秒", time.time() - t0)
    logger.info("权重: %s", {k: f"{v:.4f}" for k, v in weights.items()})


def cmd_daily(args):
    """前线狙击手: 极速推断 + ATR/追踪止盈/换手控制 → trade_plan.csv。"""
    from run_daily import run as run_dy

    cfg = _load_cfg(args.config)
    logger.info("=" * 60)
    logger.info("⚡  日频推断")
    logger.info("=" * 60)

    t0 = time.time()
    recs, trade_plan = run_dy(cfg, skip_update=args.skip_update, date=args.date)
    logger.info("完成, 耗时 %.1f 秒", time.time() - t0)


def cmd_backtest(args):
    """终极沙盘推演: Backtrader 带摩擦成本 T+1 回测。"""
    from benefit_backtest import run as run_bt

    cfg = _load_cfg(args.config)
    logger.info("=" * 60)
    logger.info("📊  Backtrader 回测  mode=%s", args.mode)
    logger.info("=" * 60)

    t0 = time.time()
    run_bt(
        cfg,
        mode=args.mode,
        predictions=args.predictions,
        trade_plan=args.trade_plan,
        plot=args.plot,
    )
    logger.info("回测完成, 耗时 %.1f 秒", time.time() - t0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  argparse 构建
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quant_cli",
        description="量化交易系统统一控制台",
    )
    parser.add_argument(
        "--config", type=str, default=DEFAULT_CONFIG,
        help="配置文件路径 (默认 quant_system/configs/config.yaml)",
    )
    sub = parser.add_subparsers(dest="command", help="可用子命令")

    # ── fetch ────────────────────────────────────
    p_fetch = sub.add_parser("fetch", help="数据原料车间: 拉取日频/分钟线数据")
    p_fetch.add_argument(
        "--freq", choices=["daily", "minute"], default="daily",
        help="daily=日频(TuShare+panel), minute=分钟线(Akshare)",
    )
    p_fetch.add_argument("--skip-fina", action="store_true", help="跳过 fina_indicator")
    p_fetch.add_argument("--export-qlib", action="store_true", help="额外导出 Qlib .bin")
    p_fetch.set_defaults(func=cmd_fetch)

    # ── diagnose ─────────────────────────────────
    p_diag = sub.add_parser("diagnose", help="因子 X 光室: IC/ICIR/VIF 诊断筛选")
    p_diag.add_argument("--skip-alpha158", action="store_true", help="跳过 Alpha158")
    p_diag.add_argument(
        "--method", choices=["greedy", "ic_vif"], default=None,
        help="因子筛选方法 (默认用配置)",
    )
    p_diag.set_defaults(func=cmd_diagnose)

    # ── weekly ───────────────────────────────────
    p_wk = sub.add_parser("weekly", help="指挥官军训: Walk-Forward 训练 + 权重分配")
    p_wk.add_argument("--start-date", type=str, default=None, help="起始日期 YYYYMMDD")
    p_wk.add_argument("--end-date", type=str, default=None, help="截止日期 YYYYMMDD")
    p_wk.set_defaults(func=cmd_weekly)

    # ── daily ────────────────────────────────────
    p_dy = sub.add_parser("daily", help="前线狙击手: 极速推断 + 风控 → trade_plan")
    p_dy.add_argument("--skip-update", action="store_true", help="跳过增量数据更新")
    p_dy.add_argument("--date", type=str, default=None, help="目标日期 YYYYMMDD")
    p_dy.set_defaults(func=cmd_daily)

    # ── backtest ─────────────────────────────────
    p_bt = sub.add_parser("backtest", help="终极沙盘推演: Backtrader T+1 回测")
    p_bt.add_argument(
        "--mode", choices=["historical", "daily"], default="historical",
        help="historical=历史信号回测, daily=交易计划回测",
    )
    p_bt.add_argument("--predictions", type=str, default=None, help="预测 CSV 路径")
    p_bt.add_argument("--trade-plan", type=str, default=None, help="交易计划 CSV 路径")
    p_bt.add_argument("--plot", action="store_true", help="绘制资金曲线图")
    p_bt.set_defaults(func=cmd_backtest)

    return parser


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  主入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
