#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实工业时序预测 HPO 参数化入口。

接收 JSON 超参数，训练 ITransformerForecaster，并返回统一格式的指标 JSON。
主要用于被 hebo-forecasting-hpo 等 orchestrator skill 在超参数寻优循环中调用。

用法：
    python run_train_with_params.py \
      --params '{"d_model": 256, "nhead": 4, "num_layers": 2, "epochs": 30}' \
      --data_file data/total_final_0430/total_final_0430_balanced.csv \
      --model_tag trial_001 \
      --output -
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from config import setup_paths
from run_train_pipeline import RealForecastPipeline, resolve_config, setup_logging

PATHS = setup_paths()
REPO_ROOT = PATHS["REPO_ROOT"]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "real_forecast_config.yaml"

# 允许被 JSON 覆盖的模型超参数
MODEL_PARAM_KEYS = {
    "seq_len", "pred_len", "d_model", "nhead", "num_layers",
    "dim_feedforward", "dropout", "kan_grid_size", "target_idx",
    "epochs", "batch_size", "lr", "weight_decay", "early_stop_patience",
    "train_ratio", "val_ratio", "device", "lag_aware", "step_cond_head",
    "trend_weight",
}


def parse_params(params_json: str) -> dict[str, Any]:
    """解析 JSON 参数字符串。"""
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"--params 必须是合法 JSON: {e}")

    if not isinstance(params, dict):
        raise ValueError("--params 必须解析为 JSON 对象")

    return params


def clip_params(params: dict[str, Any]) -> dict[str, Any]:
    """对常见超参数做边界裁剪，避免无效训练。"""
    clipped = params.copy()

    # 整数参数下限
    for key in ["seq_len", "pred_len", "d_model", "nhead", "num_layers",
                "dim_feedforward", "kan_grid_size", "epochs", "batch_size",
                "early_stop_patience"]:
        if key in clipped:
            clipped[key] = max(1, int(clipped[key]))

    # d_model 必须能被 nhead 整除（Transformer 要求）
    if "d_model" in clipped and "nhead" in clipped:
        d_model = clipped["d_model"]
        nhead = clipped["nhead"]
        if d_model % nhead != 0:
            # 向下取整到能被 nhead 整除的最大值
            clipped["d_model"] = (d_model // nhead) * nhead
            if clipped["d_model"] == 0:
                clipped["d_model"] = nhead

    # 比例参数
    for key in ["dropout", "train_ratio", "val_ratio"]:
        if key in clipped:
            clipped[key] = float(clipped[key])

    if "dropout" in clipped:
        clipped["dropout"] = max(0.0, min(1.0, clipped["dropout"]))
    if "train_ratio" in clipped and "val_ratio" in clipped:
        if clipped["train_ratio"] + clipped["val_ratio"] >= 1.0:
            total = clipped["train_ratio"] + clipped["val_ratio"]
            clipped["train_ratio"] = 0.7
            clipped["val_ratio"] = min(0.25, max(0.05, 0.9 - total))

    return clipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="真实工业时序预测 HPO 参数化入口",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"基础配置文件路径，默认 {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--params",
        type=str,
        required=True,
        help='JSON 格式的超参数字符串，例如：\'{"d_model": 256, "epochs": 30}\'',
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="原始 CSV 路径。",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help="模型标签，用于区分不同 trial 的输出目录。",
    )
    parser.add_argument(
        "--run_preprocessing",
        action="store_true",
        help="是否先调用 preprocess_for_tsa.py 进行预处理。",
    )
    parser.add_argument(
        "--run_inference",
        action="store_true",
        default=True,
        help="训练结束后是否执行测试集评估（默认开启）。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help='输出文件路径，"-" 表示输出到 stdout。',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="固定随机种子。",
    )

    args = parser.parse_args()

    logger = setup_logging()

    # 解析并裁剪参数
    params = parse_params(args.params)
    params = clip_params(params)

    # 加载基础配置
    base_cfg = resolve_config(args.config)

    # 用 JSON 参数覆盖模型配置
    for key, value in params.items():
        if key in MODEL_PARAM_KEYS:
            base_cfg["operator"][key] = value

    # 确定模型保存目录
    if args.model_tag:
        base_cfg["paths"]["model_save_dir"] = str(
            REPO_ROOT / "saved_models" / f"tsa_itransformer_{args.model_tag}"
        )

    pipeline = RealForecastPipeline(
        logger=logger,
        cfg=base_cfg,
        run_preprocessing=args.run_preprocessing,
        data_file=args.data_file,
        run_inference=args.run_inference,
        seed=args.seed,
    )

    result = pipeline.run()

    # 构造输出 JSON
    metrics = result.get("metrics", {})
    overall = metrics.get("overall", {}) if isinstance(metrics, dict) else {}

    output = {
        "params": params,
        "metrics": overall,
        "model_dir": result.get("model_dir"),
        "elapsed_seconds": result.get("elapsed_seconds"),
    }

    output_json = json.dumps(output, ensure_ascii=False, indent=2, default=str)

    if args.output == "-":
        print(output_json)
    else:
        Path(args.output).write_text(output_json, encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
