#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端 TSA-Suite iTransformer 训练脚本（放在 TSA-Suite 工程内使用）。

用法：
    # 1) 先运行预处理（CSV -> x.npy / y.npy / chunk_ids.npy）
    cd /Users/panhui/Desktop/ph_hw/lanqu_code/TSA-Suite
    python scripts/preprocess_for_tsa.py \
        --data_file /path/to/your/data.csv \
        --output_dir preprocessed_tsa

    # 2) 再运行本脚本训练
    python scripts/run_tsa_forecast.py \
        --preprocessed_dir preprocessed_tsa \
        --model_save_dir saved_models/tsa_itransformer

或者一步完成：
    python scripts/run_tsa_forecast.py \
        --run_preprocessing \
        --data_file /path/to/your/data.csv \
        --output_dir preprocessed_tsa \
        --model_save_dir saved_models/tsa_itransformer
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# 本脚本位于 TSA-Suite/scripts/，把 src/ 目录加入路径即可直接导入 tsas
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsas.engine.operator.forecasting.itransformer import (
    ITransformerForecaster,
    ITransformerForecasterConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TSA-Suite ITransformerForecaster end-to-end."
    )

    # 预处理相关
    parser.add_argument(
        "--run_preprocessing",
        action="store_true",
        help="是否先调用 scripts/preprocess_for_tsa.py 进行预处理。",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/total_final_0430/total_final_0430_balanced.csv",
        help="原始 CSV 路径（仅在 --run_preprocessing 时使用）。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="preprocessed_tsa",
        help="预处理输出目录。",
    )

    # 模型保存路径
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="saved_models/tsa_itransformer",
        help="训练好的模型保存目录。",
    )

    # 模型超参数（与 train_new.py 默认保持一致）
    parser.add_argument("--seq_len", type=int, default=600)
    parser.add_argument("--pred_len", type=int, default=20)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--target_idx", type=int, default=-1)

    # 训练参数
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "npu"],
        help="计算设备。",
    )

    # 随机种子（可选）
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="固定随机种子以获得可复现结果。",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # 固定随机种子（可选）
    if args.seed is not None:
        import random

        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import torch

            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except ImportError:
            pass

    # 1. 预处理（可选）
    if args.run_preprocessing:
        # 动态导入，避免路径问题
        import importlib.util

        preprocess_path = Path(__file__).parent / "preprocess_for_tsa.py"
        spec = importlib.util.spec_from_file_location("preprocess_for_tsa", preprocess_path)
        preprocess_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(preprocess_module)

        cfg = preprocess_module.PreprocessConfig(
            data_file=args.data_file,
            output_dir=args.output_dir,
        )
        pipeline = preprocess_module.PreprocessPipeline(cfg)
        pipeline.run_and_save()

    # 2. 加载预处理结果
    output_dir = Path(args.output_dir)
    x_path = output_dir / "x.npy"
    y_path = output_dir / "y.npy"
    chunk_ids_path = output_dir / "chunk_ids.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"预处理文件不存在。请先运行：\n"
            f"  python scripts/preprocess_for_tsa.py --data_file <your_csv> --output_dir {args.output_dir}\n"
            f"或带上 --run_preprocessing 参数运行本脚本。"
        )

    x = np.load(x_path)
    y = np.load(y_path)
    chunk_ids = np.load(chunk_ids_path) if chunk_ids_path.exists() else None

    print(
        f"Loaded preprocessed data: x={x.shape}, y={y.shape}, "
        f"chunk_ids={chunk_ids.shape if chunk_ids is not None else None}"
    )

    # 3. 构造算子
    dim_feedforward = (
        args.dim_feedforward if args.dim_feedforward is not None else args.d_model * 2
    )

    cfg = ITransformerForecasterConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=dim_feedforward,
        dropout=args.dropout,
        step_cond_head=False,
        lag_aware=True,
        kan_grid_size=args.kan_grid_size,
        target_idx=args.target_idx,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        device=args.device,
    )

    forecaster = ITransformerForecaster(config=cfg)
    if chunk_ids is not None:
        forecaster.set_chunk_ids(chunk_ids)
        print("chunk_ids loaded; windows will stay within each continuous segment.")

    # 4. 训练
    print("\n--- Training TSA-Suite ITransformerForecaster ---")
    forecaster.fit(x, y)

    # 5. 保存
    save_dir = Path(args.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    forecaster.save(save_dir)
    print(f"\n✅ Model saved to: {save_dir}")


if __name__ == "__main__":
    main()
