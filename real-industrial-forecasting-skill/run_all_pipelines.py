#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实工业时序预测一键完整流程。

封装：
1. 可选预处理（调用 preprocess_for_tsa.py）
2. 训练 ITransformerForecaster
3. 可选测试集推理评估（调用 inference_tsa.py）

用法：
    python run_all_pipelines.py --run_preprocessing --data_file <csv> --run_inference
    python run_all_pipelines.py --run_inference
"""

import argparse
import logging
import sys
from pathlib import Path

from config import setup_paths
from run_train_pipeline import RealForecastPipeline, resolve_config, setup_logging

PATHS = setup_paths()
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "real_forecast_config.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="真实工业时序预测一键完整流程",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径，默认 {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--run_preprocessing",
        action="store_true",
        help="是否先调用 preprocess_for_tsa.py 进行预处理。",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="原始 CSV 路径（仅在 --run_preprocessing 时使用）。",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default=None,
        help="预处理输出目录。",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default=None,
        help="模型保存目录。",
    )
    parser.add_argument(
        "--run_inference",
        action="store_true",
        help="训练结束后是否调用 inference_tsa.py 对测试集进行评估。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="固定随机种子以获得可复现结果。",
    )

    # 常用超参数快捷覆盖
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda", "npu"])

    args = parser.parse_args()

    logger = setup_logging()
    cfg = resolve_config(args.config, vars(args))

    pipeline = RealForecastPipeline(
        logger=logger,
        cfg=cfg,
        run_preprocessing=args.run_preprocessing,
        data_file=args.data_file,
        run_inference=args.run_inference,
        seed=args.seed,
    )
    result = pipeline.run()
    logger.info("输出汇总: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
