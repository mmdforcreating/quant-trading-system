#!/usr/bin/env python3
"""
数据获取与 panel 构建（因子可后续再做）

=== 用法 ===
# 只做数据：股票池 → 拉取/加载 raw → 合并 panel（不跑因子）
python -m quant_system.run_data --config quant_system/configs/config.yaml --data-only

# 完整流程：数据 + 因子计算 + 因子选择
python -m quant_system.run_data --config quant_system/configs/config.yaml

=== 数据流程（--data-only 时仅执行此处）===
1. 加载配置、股票池
2. TuShare 拉取 daily/daily_basic/adj_factor/moneyflow/fina（或 --skip-fetch 用已有 raw）
3. 可选：分钟数据（akshare）写入 data/minute/
4. 合并 raw → panel.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_data")


def parse_args():
    p = argparse.ArgumentParser(description="数据获取与 panel 构建")
    p.add_argument("--config", type=str, default="quant_system/configs/config.yaml")
    p.add_argument("--data-only", action="store_true", help="只做数据：拉取/加载 raw + 合并 panel，不做因子计算与选择")
    p.add_argument("--skip-fetch", action="store_true", help="跳过数据拉取，直接用已有 raw 数据构建")
    p.add_argument("--skip-fina", action="store_true", help="跳过 fina_indicator（较慢）")
    return p.parse_args()


def main():
    args = parse_args()

    from quant_system.utils.config_manager import ConfigManager
    cfg = ConfigManager(args.config)

    dp = cfg.get("data_pipeline", {})
    start_date = dp.get("start_date", "20200101")
    end_date = dp.get("end_date") or datetime.now().strftime("%Y%m%d")
    panel_path = str(Path(dp.get("panel_path", "data/panel.parquet")).expanduser())

    logger.info("=" * 60)
    logger.info("数据管道启动")
    logger.info("  配置: %s", args.config)
    logger.info("  日期: %s → %s", start_date, end_date)
    logger.info("=" * 60)

    # Step 1: 股票池
    from quant_system.data_pipeline.stock_pool import get_stock_pool
    ts_codes, name_map = get_stock_pool(cfg)
    logger.info("股票池: %d 只", len(ts_codes))

    # Step 2: TuShare 数据拉取
    if not args.skip_fetch:
        from quant_system.data_pipeline.tushare_fetcher import TuShareFetcher
        fetcher = TuShareFetcher(cfg)
        raw_data = fetcher.fetch_all(
            ts_codes=ts_codes,
            start_date=start_date,
            end_date=end_date,
            skip_fina=args.skip_fina,
        )
        logger.info("数据拉取完成:")
        for name, df in raw_data.items():
            logger.info("  %s: %d 行", name, len(df))
    else:
        raw_data = _load_existing_raw(dp.get("raw_data_dir", "data/raw"))
        logger.info("跳过拉取，使用已有 raw 数据")

    # Step 2.5: 分钟级数据（可选，akshare；与 skip_fetch 无关，enable 即拉）
    minute_cfg = dp.get("minute", {})
    if minute_cfg.get("enable", False):
        try:
            from quant_system.data_pipeline.minute_accumulator import backfill_5min
            symbols_6 = [c.split(".")[0].zfill(6) for c in ts_codes]
            storage = str(Path(minute_cfg.get("storage_path", "data/minute")).expanduser())
            periods = minute_cfg.get("periods", ["5"])
            # recent_days 存在则只拉最近 N 天，否则全区间
            end_d = end_date
            rd = minute_cfg.get("recent_days")
            if rd is not None and isinstance(rd, int) and rd > 0:
                end_dt = datetime.strptime(end_date, "%Y%m%d")
                start_dt = end_dt - timedelta(days=int(rd))
                start_minute = start_dt.strftime("%Y%m%d")
            else:
                start_minute = start_date
            if "5" in periods:
                backfill_5min(
                    symbols_6,
                    start_date=start_minute,
                    end_date=end_d,
                    storage_path=storage,
                    sleep_sec=float(minute_cfg.get("sleep_sec", 0.5)),
                    use_fallback=minute_cfg.get("use_fallback", True),
                )
            logger.info("分钟数据拉取完成（见 %s/5min/）", storage)
        except Exception as e:
            logger.warning("分钟数据拉取失败: %s", e)

    # Step 3: 构建 panel
    from quant_system.data_pipeline.panel_builder import build_panel
    panel = build_panel(raw_data, panel_path, start_date)
    logger.info("Panel 构建完成: %d 行, %d 列", len(panel), panel.shape[1])

    if args.data_only:
        logger.info("=" * 60)
        logger.info("数据管道完成（仅数据，未做因子）")
        logger.info("  Panel: %s (%d 行, %d 列)", panel_path, len(panel), panel.shape[1])
        logger.info("=" * 60)
        return

    # Step 3.5: Alpha158 因子（可选）
    fe_cfg = cfg.get("factor_engine", {})
    if fe_cfg.get("use_alpha158", False):
        try:
            from quant_system.data_pipeline.qlib_alpha158 import (
                compute_alpha158,
                merge_alpha158_to_panel,
            )
            alpha_df = compute_alpha158(panel, cfg)
            if alpha_df is not None and not alpha_df.empty:
                panel = merge_alpha158_to_panel(panel, alpha_df)
                panel.to_parquet(panel_path, index=False)
                logger.info("Alpha158 因子已合并: %d 列", panel.shape[1])
            else:
                logger.warning("Alpha158 计算返回空，跳过")
        except Exception as e:
            logger.warning("Alpha158 计算失败 (%s)，继续使用自研因子", e)

    # Step 4: 计算自研因子
    if fe_cfg.get("use_custom", True):
        from quant_system.factor_engine.custom_factors import compute_all_factors
        panel = compute_all_factors(panel, cfg)
        panel.to_parquet(panel_path, index=False)
        logger.info("自研因子计算完成: %d 列", panel.shape[1])

    # Step 5: 因子选择
    label_col = fe_cfg.get("label_col", "y_ret_5d_adj")
    if label_col not in panel.columns:
        logger.error("标签列 %s 不在 panel 中", label_col)
        return

    from quant_system.factor_engine.factor_selector import select_factors
    selected, ic_table = select_factors(panel, label_col, cfg)

    # 保存
    output_dir = cfg.get("output", {}).get("dir", "output")
    reports_dir = cfg.get("output", {}).get("reports_dir", "output/reports")
    os.makedirs(reports_dir, exist_ok=True)

    features_path = os.path.join(output_dir, "selected_features.txt")
    os.makedirs(output_dir, exist_ok=True)
    with open(features_path, "w") as f:
        f.write("\n".join(selected))
    logger.info("选定因子 (%d 个) 保存到: %s", len(selected), features_path)

    if not ic_table.empty:
        ic_path = os.path.join(reports_dir, "ic_table.csv")
        ic_table.to_csv(ic_path, index=False)
        logger.info("IC 统计表保存到: %s", ic_path)

    logger.info("=" * 60)
    logger.info("数据管道完成")
    logger.info("  Panel: %s (%d 行, %d 列)", panel_path, len(panel), panel.shape[1])
    logger.info("  选定因子: %d 个", len(selected))
    if selected:
        logger.info("  前 10 个: %s", selected[:10])
    logger.info("=" * 60)


def _load_existing_raw(raw_dir: str):
    """从已有 CSV 加载 raw 数据。"""
    from quant_system.data_pipeline.tushare_fetcher import safe_read_csv
    raw_dir = str(Path(raw_dir).expanduser())
    data = {}
    for name in ["daily", "daily_basic", "adj_factor", "moneyflow", "fina_indicator"]:
        path = os.path.join(raw_dir, f"{name}.csv")
        data[name] = safe_read_csv(path)
        if not data[name].empty:
            logger.info("  已加载 %s: %d 行", name, len(data[name]))
    return data


if __name__ == "__main__":
    main()
