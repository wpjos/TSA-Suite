#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端 TSA-Suite 时序预测训练脚本。

支持模型：itransformer、lightgbm、xgboost。

用法：
    # 1) 先运行预处理
    python preprocess_for_tsa.py --data_file data/total_final_0430/total_final_0430_balanced.csv

    # 2) 再运行本脚本训练（默认 itransformer）
    python run_tsa_forecast.py

    # 3) 指定其它模型
    python run_tsa_forecast.py --model lightgbm
    python run_tsa_forecast.py --model xgboost

或者一步完成：
    python run_tsa_forecast.py --run_preprocessing --data_file data/total_final_0430/total_final_0430_balanced.csv --model lightgbm

本脚本默认从 preprocessed_tsa/ 目录读取：
    - x.npy
    - y.npy
    - chunk_ids.npy

训练完成后，模型保存到 saved_models/tsa_<model>/。
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 0. 让当前环境能找到 TSA-Suite
# ---------------------------------------------------------------------------
DEFAULT_TSA_SUITE_ROOT = "/Users/panhui/Desktop/ph_hw/lanqu_code/TSA-Suite"


def _ensure_tsa_suite_import(tsa_suite_root: str) -> None:
    src_dir = Path(tsa_suite_root).expanduser().resolve() / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"TSA-Suite src 目录不存在: {src_dir}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


# ---------------------------------------------------------------------------
# 1. 命令行参数
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TSA-Suite forecaster on preprocessed_tsa outputs."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="itransformer",
        choices=["itransformer", "lightgbm", "xgboost"],
        help="选择预测模型，默认 itransformer。",
    )

    # 预处理相关
    parser.add_argument(
        "--run_preprocessing",
        action="store_true",
        help="是否先调用 preprocess_for_tsa.py 进行预处理。",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/total_final_0430/total_final_0430_balanced.csv",
        help="原始 CSV 路径（仅在 --run_preprocessing 时使用）。",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default="preprocessed_tsa",
        help="预处理输出目录。",
    )

    # TSA-Suite 路径
    parser.add_argument(
        "--tsa_suite_root",
        type=str,
        default=DEFAULT_TSA_SUITE_ROOT,
        help="TSA-Suite 仓库根目录。",
    )

    # 模型保存路径
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default=None,
        help="训练好的模型保存目录。默认 saved_models/tsa_<model>。",
    )
    parser.add_argument(
        "--run_inference",
        action="store_true",
        help="训练结束后是否调用 inference_tsa.py 对测试集进行评估。",
    )

    # 公共窗口参数
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--pred_len", type=int, default=30)

    # iTransformer 超参数
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--kan_grid_size", type=int, default=5)
    parser.add_argument("--target_idx", type=int, default=-1)

    # 训练参数（itransformer 与树模型通用部分）
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda", "npu"],
        help="计算设备。树模型会自动把 'auto/cuda/npu' 映射为 'gpu'。",
    )

    # 树模型超参数（LightGBM / XGBoost）
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=200,
        help="树模型 boosting 轮数。",
    )
    parser.add_argument(
        "--tree_learning_rate",
        type=float,
        default=0.05,
        help="树模型学习率。",
    )
    parser.add_argument(
        "--num_leaves",
        type=int,
        default=31,
        help="LightGBM 每棵树最大叶子数。",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="XGBoost 树最大深度。",
    )
    parser.add_argument(
        "--reg_alpha",
        type=float,
        default=0.1,
        help="树模型 L1 正则化。",
    )
    parser.add_argument(
        "--reg_lambda",
        type=float,
        default=0.1,
        help="树模型 L2 正则化。",
    )
    parser.add_argument(
        "--min_child_samples",
        type=int,
        default=20,
        help="LightGBM 叶节点最小样本数。",
    )
    parser.add_argument(
        "--min_child_weight",
        type=float,
        default=1.0,
        help="XGBoost 子节点最小权重和。",
    )
    parser.add_argument(
        "--tree_strategy",
        type=str,
        default="direct",
        choices=["direct", "mimo"],
        help="树模型多步预测策略：direct（每个步长独立建模）或 mimo（同时输出多步）。",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="树模型线程数，-1 表示使用全部 CPU。",
    )

    # 随机种子（可选）
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="固定随机种子以获得可复现结果。",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. 根据模型名构造 forecaster
# ---------------------------------------------------------------------------
def _build_forecaster(args: argparse.Namespace):
    """根据 --model 构造对应的 forecaster 与配置。"""
    model = args.model

    if model == "itransformer":
        from tsas.engine.operator.forecasting.itransformer import (
            ITransformerForecaster,
            ITransformerForecasterConfig,
        )

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
        return ITransformerForecaster(config=cfg)

    # 树模型设备映射：itransformer 的 'auto/cuda/npu' 对树模型统一视为 'gpu'
    tree_device = "cpu" if args.device == "cpu" else "gpu"
    strategy = None if args.tree_strategy == "direct" else "MIMO"

    if model == "lightgbm":
        from tsas.engine.operator.forecasting.lightgbm import (
            LightGBMForecaster,
            LightGBMForecasterConfig,
        )

        cfg = LightGBMForecasterConfig(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            strategy=strategy,
            n_estimators=args.n_estimators,
            learning_rate=args.tree_learning_rate,
            num_leaves=args.num_leaves,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            min_child_samples=args.min_child_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            device=tree_device,
            n_jobs=args.n_jobs,
        )
        return LightGBMForecaster(config=cfg)

    if model == "xgboost":
        from tsas.engine.operator.forecasting.xgboost import (
            XGBoostForecaster,
            XGBoostForecasterConfig,
        )

        cfg = XGBoostForecasterConfig(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            strategy=strategy,
            n_estimators=args.n_estimators,
            learning_rate=args.tree_learning_rate,
            max_depth=args.max_depth,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            min_child_weight=args.min_child_weight,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            device=tree_device,
            n_jobs=args.n_jobs,
        )
        return XGBoostForecaster(config=cfg)

    raise ValueError(f"不支持的模型: {model}")


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    # 默认保存目录随模型变化
    if args.model_save_dir is None:
        args.model_save_dir = f"saved_models/tsa_{args.model}"

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
        from preprocess_for_tsa import PreprocessConfig, PreprocessPipeline

        cfg = PreprocessConfig(
            data_file=args.data_file,
            output_dir=args.preprocessed_dir,
        )
        pipeline = PreprocessPipeline(cfg)
        pipeline.run_and_save()

    # 2. 加载预处理结果
    preprocessed_dir = Path(args.preprocessed_dir)
    x_path = preprocessed_dir / "x.npy"
    y_path = preprocessed_dir / "y.npy"
    chunk_ids_path = preprocessed_dir / "chunk_ids.npy"

    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"预处理文件不存在。请先运行：\n"
            f"  python preprocess_for_tsa.py --data_file <your_csv> --output_dir {args.preprocessed_dir}\n"
            f"或带上 --run_preprocessing 参数运行本脚本。"
        )

    x = np.load(x_path)
    y = np.load(y_path)
    chunk_ids = np.load(chunk_ids_path) if chunk_ids_path.exists() else None

    print(f"Loaded preprocessed data: x={x.shape}, y={y.shape}, chunk_ids={chunk_ids.shape if chunk_ids is not None else None}")

    # 3. 导入并构造 TSA-Suite 算子
    _ensure_tsa_suite_import(args.tsa_suite_root)
    forecaster = _build_forecaster(args)

    # itransformer 支持 chunk_ids；树模型当前内部自己按时间顺序切窗，不处理 chunk_ids
    if chunk_ids is not None and hasattr(forecaster, "set_chunk_ids"):
        forecaster.set_chunk_ids(chunk_ids)
        print("chunk_ids loaded; windows will stay within each continuous segment.")

    # 4. 训练
    print(f"\n--- Training TSA-Suite {args.model} forecaster ---")
    forecaster.fit(x, y)

    # 5. 保存
    save_dir = Path(args.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    forecaster.save(save_dir)
    print(f"\n✅ Model saved to: {save_dir}")

    # 6. 推理评估（可选）
    if args.run_inference:
        from inference_tsa import run_inference

        print("\n--- Running inference on test set ---")
        run_inference(
            model_save_dir=args.model_save_dir,
            preprocessed_dir=args.preprocessed_dir,
            tsa_suite_root=args.tsa_suite_root,
            batch_size=args.batch_size,
            model=args.model,
        )


if __name__ == "__main__":
    main()
