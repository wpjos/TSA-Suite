#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone sliding-window evaluation for TSAS-CLI forecasting operators.

Usage:
    python run_sliding_window_eval.py \
        --data_file data/total_final_0430/total_final_0430_balanced.csv \
        --target_col diya_qibao_shuiwei_youxuanzhi \
        --output_dir sliding_eval_output \
        --operator itransformer_forecaster \
        --device cpu
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _setup_tsa_suite_paths(tsa_suite_root: Path) -> Path:
    """把 TSA-Suite src 目录加入当前进程和子进程 PYTHONPATH。"""
    src_dir = tsa_suite_root / "src"
    if not src_dir.is_dir():
        raise FileNotFoundError(
            f"找不到 TSA-Suite src 目录: {src_dir}\n"
            f"请通过 --tsa_suite_root 或环境变量 TSA_SUITE_DIR 指定正确的仓库根目录。"
        )
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    os.environ["PYTHONPATH"] = src_str + os.pathsep + os.environ.get("PYTHONPATH", "")
    return src_dir


def _default_tsa_suite_root() -> Path:
    """默认 TSA-Suite 根目录：优先环境变量 TSA_SUITE_DIR，否则取脚本所在目录。"""
    if env := os.environ.get("TSA_SUITE_DIR"):
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parent


def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in ("true", "1", "yes", "y")


def _load_yaml_config(path: Path) -> dict:
    """Load YAML config if it exists, otherwise return empty dict."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _config_defaults(config_path: Path) -> dict:
    """Extract argument defaults from a TSAS forecasting config YAML."""
    cfg = _load_yaml_config(config_path)
    op_cfg = cfg.get("operator", {}).get("config", {})
    prep_cfg = cfg.get("preprocessing", {})
    op_meta = cfg.get("operator", {})

    mapping = {
        "operator": (op_meta, "name"),
        "time_col": (prep_cfg, "time_col"),
        "max_gap": (prep_cfg, "max_gap"),
        "target_col": (prep_cfg, "target_col"),
        "seq_len": (op_cfg, "seq_len"),
        "pred_len": (op_cfg, "pred_len"),
        "d_model": (op_cfg, "d_model"),
        "nhead": (op_cfg, "nhead"),
        "num_layers": (op_cfg, "num_layers"),
        "dim_feedforward": (op_cfg, "dim_feedforward"),
        "dropout": (op_cfg, "dropout"),
        "lag_aware": (op_cfg, "lag_aware"),
        "lag_max": (op_cfg, "lag_max"),
        "lag_bias_scale": (op_cfg, "lag_bias_scale"),
        "lag_dropout": (op_cfg, "lag_dropout"),
        "kan_grid_size": (op_cfg, "kan_grid_size"),
        "epochs": (op_cfg, "epochs"),
        "batch_size": (op_cfg, "batch_size"),
        "lr": (op_cfg, "lr"),
        "weight_decay": (op_cfg, "weight_decay"),
        "early_stop_patience": (op_cfg, "early_stop_patience"),
        "train_ratio": (op_cfg, "train_ratio"),
        "val_ratio": (op_cfg, "val_ratio"),
        "device": (op_cfg, "device"),
    }

    defaults = {}
    for arg_name, (src, key) in mapping.items():
        if key in src and src[key] is not None:
            defaults[arg_name] = src[key]
    return defaults


def _load_forecaster_class(operator: str):
    """根据算子名称动态导入 forecaster 类。"""
    if operator == "itransformer_forecaster":
        from tsas.engine.operator.forecasting.itransformer import ITransformerForecaster
        return ITransformerForecaster
    if operator == "lightgbm_forecaster":
        from tsas.engine.operator.forecasting.lightgbm import LightGBMForecaster
        return LightGBMForecaster
    if operator == "xgboost_forecaster":
        from tsas.engine.operator.forecasting.xgboost import XGBoostForecaster
        return XGBoostForecaster
    raise ValueError(f"不支持的算子: {operator}")


def _load_forecaster(operator: str, model_dir: Path):
    """加载已保存的模型。"""
    ForecasterClass = _load_forecaster_class(operator)
    return ForecasterClass.load(model_dir)


def run_cli(cmd: list[str]) -> None:
    """Run a TSAS-CLI command and raise on failure."""
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def detect_time_gaps(timestamps: pd.Series, max_gap: float) -> np.ndarray:
    """Return chunk_ids based on time gaps > max_gap seconds."""
    ts = pd.to_datetime(timestamps, errors="coerce")
    diff = ts.diff().dt.total_seconds().fillna(0).values
    diff[0] = 0.0
    chunk_ids = (diff > max_gap).cumsum()
    return chunk_ids.astype(int)


def impute_missing_values(data: np.ndarray, chunk_ids: np.ndarray) -> np.ndarray:
    """Linear interpolation per chunk, fill remaining with 0."""
    df = pd.DataFrame(data)
    df.replace(0, np.nan, inplace=True)
    df = df.groupby(chunk_ids, group_keys=False).apply(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    df.fillna(0, inplace=True)
    return df.values.astype(np.float32)


def double_ema_smooth(
    data: np.ndarray, chunk_ids: np.ndarray, target_idx: int, alpha: float
) -> np.ndarray:
    """Brown's double EMA on target column per chunk."""
    for cid in np.unique(chunk_ids):
        mask = chunk_ids == cid
        series = data[mask, target_idx]
        if len(series) == 0:
            continue
        s1 = pd.Series(series).ewm(alpha=alpha, adjust=False).mean().values
        s2 = pd.Series(s1).ewm(alpha=alpha, adjust=False).mean().values
        data[mask, target_idx] = 2 * s1 - s2
    return data


def temporal_split(
    n_total: int, train_ratio: float, val_ratio: float
) -> tuple[int, int, int]:
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"数据量不足以按 {train_ratio}/{val_ratio} 划分: "
            f"n_total={n_total}, n_train={n_train}, n_val={n_val}, n_test={n_test}"
        )
    return n_train, n_val, n_test


def preprocess(args: argparse.Namespace) -> dict:
    """Prepare CSV files for TSAS-CLI."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine columns
    cols_to_read = [args.time_col]
    if args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    else:
        header = pd.read_csv(args.data_file, nrows=0).columns.tolist()
        feature_cols = [c for c in header if c not in (args.time_col, args.target_col)]

    for col in feature_cols + [args.target_col]:
        cols_to_read.append(col)

    df = pd.read_csv(args.data_file, usecols=cols_to_read)
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")
    df = df.dropna(subset=[args.time_col])

    for col in feature_cols + [args.target_col]:
        if col not in df.columns:
            raise ValueError(f"列不存在: {col}")

    numeric_cols = feature_cols + [args.target_col]
    target_idx = len(feature_cols)

    # Gap detection
    chunk_ids = detect_time_gaps(df[args.time_col], args.max_gap)

    # Numeric matrix
    data = df[numeric_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float32)

    # Imputation
    data = impute_missing_values(data, chunk_ids)

    # Smoothing
    if args.smooth_target:
        data = double_ema_smooth(data, chunk_ids, target_idx, args.smooth_alpha)

    df[numeric_cols] = data

    # Temporal split
    n_total = len(df)
    n_train, n_val, n_test = temporal_split(n_total, args.train_ratio, args.val_ratio)
    df_train = df.iloc[: n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val :].copy()
    test_chunk_ids = chunk_ids[n_train + n_val :]

    # Save outputs
    train_path = output_dir / "train.csv"
    chunk_ids_path = output_dir / "chunk_ids.csv"
    test_path = output_dir / "test.csv"
    test_chunk_ids_path = output_dir / "test_chunk_ids.csv"
    test_window_path = output_dir / "test_window.csv"
    test_truth_path = output_dir / "test_truth.csv"
    meta_path = output_dir / "meta.yaml"

    df_train[[args.time_col] + numeric_cols].to_csv(train_path, index=False)
    pd.DataFrame(chunk_ids[: n_train + n_val]).to_csv(
        chunk_ids_path, index=False, header=False
    )
    df_test[[args.time_col] + numeric_cols].to_csv(test_path, index=False)
    pd.DataFrame(test_chunk_ids).to_csv(test_chunk_ids_path, index=False, header=False)

    if len(df_test) < args.seq_len + args.pred_len:
        raise ValueError(
            f"测试集长度 {len(df_test)} 不足以构造 "
            f"seq_len={args.seq_len} + pred_len={args.pred_len}"
        )
    df_test.iloc[: args.seq_len][[args.time_col] + numeric_cols].to_csv(
        test_window_path, index=False
    )
    df_test.iloc[args.seq_len : args.seq_len + args.pred_len][[args.target_col]].to_csv(
        test_truth_path, index=False
    )

    meta = {
        "dataset": {
            "time_col": args.time_col,
            "feature_cols": feature_cols,
            "target_col": args.target_col,
            "input_columns": numeric_cols,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": len(df_test),
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "n_train_chunks": int(len(np.unique(chunk_ids[: n_train + n_val]))),
            "n_test_chunks": int(len(np.unique(test_chunk_ids))),
        },
        "output_files": {
            "train": str(train_path),
            "chunk_ids": str(chunk_ids_path),
            "test": str(test_path),
            "test_chunk_ids": str(test_chunk_ids_path),
            "test_window": str(test_window_path),
            "test_truth": str(test_truth_path),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)

    return {"feature_cols": feature_cols, "numeric_cols": numeric_cols}


def build_forecasting_config(args: argparse.Namespace, feature_cols: list[str]) -> dict:
    """Build forecasting_config.yaml content."""
    config = {
        "operator": {
            "name": args.operator,
            "input_columns": feature_cols + [args.target_col],
            "target_column": args.target_col,
            "config": {
                "seq_len": args.seq_len,
                "pred_len": args.pred_len,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "early_stop_patience": args.early_stop_patience,
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "device": args.device,
            },
        }
    }

    if args.operator == "itransformer_forecaster":
        config["operator"]["config"].update(
            {
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dim_feedforward": args.dim_feedforward,
                "dropout": args.dropout,
                "lag_aware": args.lag_aware,
                "lag_max": args.lag_max,
                "lag_bias_scale": args.lag_bias_scale,
                "lag_dropout": args.lag_dropout,
                "kan_grid_size": args.kan_grid_size,
                "target_idx": -1,
            }
        )
    elif args.operator in ("lightgbm_forecaster", "xgboost_forecaster"):
        config["operator"]["config"].update(
            {
                "n_estimators": args.n_estimators,
                "learning_rate": args.tree_learning_rate,
                "reg_alpha": args.reg_alpha,
                "reg_lambda": args.reg_lambda,
                "n_jobs": args.n_jobs,
            }
        )
        if args.operator == "lightgbm_forecaster":
            config["operator"]["config"]["num_leaves"] = args.num_leaves
            config["operator"]["config"]["min_child_samples"] = args.min_child_samples
        else:
            config["operator"]["config"]["max_depth"] = args.max_depth
            config["operator"]["config"]["min_child_weight"] = args.min_child_weight

    return config


def train(args: argparse.Namespace) -> None:
    """Run forecasting fit via CLI."""
    output_dir = Path(args.output_dir)
    config_path = output_dir / "forecasting_config.yaml"
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "tsas.engine.operator.cli",
        "forecasting",
        "fit",
        "--input",
        str(output_dir / "train.csv"),
        "--target",
        args.target_col,
        "--config",
        str(config_path),
        "--save",
        str(model_dir),
    ]
    if (output_dir / "chunk_ids.csv").exists():
        cmd.extend(["--chunk-ids", str(output_dir / "chunk_ids.csv")])
    run_cli(cmd)


def evaluate(args: argparse.Namespace) -> dict:
    """Run sliding-window evaluation with batch inference (Python API) + CLI metrics."""
    output_dir = Path(args.output_dir)
    test_df = pd.read_csv(output_dir / "test.csv")
    test_chunk_ids = (
        pd.read_csv(output_dir / "test_chunk_ids.csv", header=None).iloc[:, 0].values
    )
    numeric_cols = [c for c in test_df.columns if c != args.time_col]
    target_idx = numeric_cols.index(args.target_col)
    num_targets = 1

    # Generate valid window start indices per chunk
    valid_indices = []
    for cid in np.unique(test_chunk_ids):
        mask = test_chunk_ids == cid
        chunk_row_indices = np.where(mask)[0]
        num_samples = len(chunk_row_indices) - args.seq_len - args.pred_len + 1
        if num_samples > 0:
            valid_indices.extend(chunk_row_indices[:num_samples].tolist())
    valid_indices = np.array(valid_indices, dtype=int)

    n_windows = len(valid_indices)
    n_test_chunks = int(len(np.unique(test_chunk_ids)))
    print(f"测试集 chunk 数量: {n_test_chunks}")
    print(f"测试集窗口数量: {n_windows}")

    if n_windows == 0:
        raise ValueError(
            f"测试集无法构造窗口: 所有 chunk 长度均 < "
            f"seq_len({args.seq_len}) + pred_len({args.pred_len})"
        )

    test_values = test_df[numeric_cols].values.astype(np.float32)

    # Construct batched windows
    x_test = np.stack([test_values[i : i + args.seq_len] for i in valid_indices])
    y_true = np.stack([
        test_values[i + args.seq_len : i + args.seq_len + args.pred_len, target_idx : target_idx + 1]
        for i in valid_indices
    ])

    # Load model and run batch inference
    print("Loading model for batch inference...")
    forecaster = _load_forecaster(args.operator, output_dir / "model")

    print(
        f"Running batch inference (eval_batch_size={args.eval_batch_size}, "
        f"total_windows={n_windows})..."
    )
    preds = []
    n_batches = (n_windows + args.eval_batch_size - 1) // args.eval_batch_size
    for i in range(0, n_windows, args.eval_batch_size):
        batch = x_test[i : i + args.eval_batch_size]
        pred = forecaster.run(batch)
        preds.append(pred)
        print(f"  已推理 batch {(i // args.eval_batch_size) + 1}/{n_batches}")
    y_pred = np.concatenate(preds, axis=0)

    # Ensure shapes
    if y_pred.shape[0] != n_windows:
        raise ValueError(f"预测窗口数不匹配: {y_pred.shape[0]} != {n_windows}")
    if y_pred.ndim == 2:
        # (N, pred_len) -> (N, pred_len, 1)
        y_pred = y_pred[:, :, np.newaxis]

    # Evaluation config
    eval_config_path = output_dir / "eval_config.yaml"
    eval_config = {
        "operators": [
            {
                "name": "forecasting_metrics",
                "alias": "forecast_metrics",
                "truth_columns": ["y_true"],
                "predict_columns": ["y_pred"],
                "config": {"epsilon": 1e-8, "max_dtw_len": 2000},
            }
        ]
    }
    with open(eval_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(eval_config, f, sort_keys=False, allow_unicode=True)

    # Per-step metrics
    per_step = []
    for h in range(args.pred_len):
        step_df = pd.DataFrame(
            {"y_true": y_true[:, h, 0], "y_pred": y_pred[:, h, 0]}
        )
        step_csv = output_dir / f"eval_step_{h}.csv"
        step_json = output_dir / f"eval_step_{h}.json"
        step_df.to_csv(step_csv, index=False)
        run_cli(
            [
                sys.executable,
                "-m",
                "tsas.engine.operator.cli",
                "evaluation",
                "run",
                "--input",
                str(step_csv),
                "--config",
                str(eval_config_path),
                "--output",
                str(step_json),
            ]
        )
        with open(step_json, encoding="utf-8") as f:
            result = json.load(f)["results"]["forecast_metrics"]["result"]
        per_step.append({"step": h + 1, **result})

    # Overall metrics
    all_df = pd.DataFrame({"y_true": y_true.reshape(-1), "y_pred": y_pred.reshape(-1)})
    all_csv = output_dir / "eval_all.csv"
    all_json = output_dir / "eval_all.json"
    all_df.to_csv(all_csv, index=False)
    run_cli(
        [
            sys.executable,
            "-m",
            "tsas.engine.operator.cli",
            "evaluation",
            "run",
            "--input",
            str(all_csv),
            "--config",
            str(eval_config_path),
            "--output",
            str(all_json),
        ]
    )
    with open(all_json, encoding="utf-8") as f:
        overall = json.load(f)["results"]["forecast_metrics"]["result"]

    # Save summary
    per_step_df = pd.DataFrame(per_step)
    per_step_path = output_dir / "per_step_metrics.csv"
    per_step_df.to_csv(per_step_path, index=False)

    summary = {
        "n_test_chunks": n_test_chunks,
        "n_windows": n_windows,
        "per_step": per_step,
        "overall": overall,
    }
    summary_path = output_dir / "overall_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # Print
    print("\n=== Per-step metrics ===")
    print(per_step_df.to_string(index=False))
    print("\n=== Overall metrics ===")
    for k, v in overall.items():
        print(f"{k}: {v:.4f}")

    return summary


def main() -> int:
    # Pre-parse --tsa_suite_root and --config so we can locate TSA-Suite and load defaults.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--tsa_suite_root",
        type=Path,
        default=_default_tsa_suite_root(),
        help="TSA-Suite 仓库根目录（默认取环境变量 TSA_SUITE_DIR 或脚本所在目录）",
    )
    pre_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML 配置文件路径（默认取 <tsa_suite_root>/configs/real_forecast_config.yaml）",
    )
    pre_args, remaining_argv = pre_parser.parse_known_args()

    tsa_suite_root = pre_args.tsa_suite_root.expanduser().resolve()
    _setup_tsa_suite_paths(tsa_suite_root)

    config_path = pre_args.config
    if config_path is None:
        config_path = tsa_suite_root / "configs" / "real_forecast_config.yaml"
    config_path = config_path.expanduser().resolve()
    config_defaults = _config_defaults(config_path)

    parser = argparse.ArgumentParser(
        description="TSAS-CLI sliding-window evaluation (no BO)",
        parents=[pre_parser],
    )
    parser.set_defaults(**config_defaults)

    # Data
    parser.add_argument("--data_file", required=True, help="原始 CSV 路径")
    parser.add_argument("--time_col", help="时间列名")
    parser.add_argument("--target_col", help="目标列名")
    parser.add_argument(
        "--feature_cols",
        default="",
        help="逗号分隔的特征列名，为空则使用除时间/目标外的所有列",
    )
    parser.add_argument(
        "--output_dir", default="sliding_window_eval_output", help="输出目录"
    )

    # Preprocessing
    parser.add_argument(
        "--max_gap", type=float, help="时间断点阈值（秒）"
    )
    parser.add_argument("--smooth_target", action="store_true", help="目标列 EMA 平滑")
    parser.add_argument("--smooth_alpha", type=float, default=0.3, help="EMA 系数")
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--val_ratio", type=float)

    # Window
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--pred_len", type=int)

    # Operator common
    parser.add_argument(
        "--operator",
        choices=["itransformer_forecaster", "lightgbm_forecaster", "xgboost_forecaster"],
        help="预测算子",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=2000,
        help="推理时的批量大小，默认 2000；设为大于等于窗口数即一次性推理",
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--early_stop_patience", type=int)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "npu"],
        help="计算设备",
    )

    # iTransformer-specific
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--nhead", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--dim_feedforward", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument(
        "--lag_aware",
        type=_str_to_bool,
        help="是否启用 Lag-Aware Refiner",
    )
    parser.add_argument("--lag_max", type=int)
    parser.add_argument("--lag_bias_scale", type=float)
    parser.add_argument("--lag_dropout", type=float)
    parser.add_argument("--kan_grid_size", type=int)

    # Tree-model-specific
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--tree_learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--max_depth", type=int, default=4)
    parser.add_argument("--reg_alpha", type=float, default=0.1)
    parser.add_argument("--reg_lambda", type=float, default=0.1)
    parser.add_argument("--min_child_samples", type=int, default=20)
    parser.add_argument("--min_child_weight", type=float, default=1.0)
    parser.add_argument("--n_jobs", type=int, default=-1)

    args = parser.parse_args(remaining_argv)

    # Required fields that may come from config
    if not args.target_col:
        parser.error("--target_col 必填（可通过 --config 配置文件指定）")
    if args.time_col is None:
        args.time_col = "datatime"
    if args.max_gap is None:
        args.max_gap = 5.0

    # 1. Preprocess
    info = preprocess(args)

    # 2. Build forecasting config
    config = build_forecasting_config(args, info["feature_cols"])
    with open(Path(args.output_dir) / "forecasting_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    # 3. Train
    train(args)

    # 4. Sliding-window evaluation
    evaluate(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
