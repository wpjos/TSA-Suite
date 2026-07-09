#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TSA-Suite 时序预测推理与评估脚本。

支持模型：itransformer、lightgbm、xgboost。

单独执行：
    python inference_tsa.py \\
        --model itransformer \\
        --model_save_dir saved_models/tsa_itransformer \\
        --preprocessed_dir preprocessed_tsa_v2 \\
        --batch_size 256

也可被 run_tsa_forecast.py 在训练后调用：
    python run_tsa_forecast.py --model lightgbm --run_inference ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error

DEFAULT_TSA_SUITE_ROOT = "/Users/panhui/Desktop/ph_hw/lanqu_code/TSA-Suite"

try:
    from fastdtw import fastdtw

    HAS_FASTDTW = True
except ImportError:
    HAS_FASTDTW = False
    print("⚠️  未检测到 fastdtw，DTW 将用 MAE 占位")


def _ensure_tsa_suite_import(tsa_suite_root: str) -> None:
    src_dir = Path(tsa_suite_root).expanduser().resolve() / "src"
    if not src_dir.exists():
        raise FileNotFoundError(f"TSA-Suite src 目录不存在: {src_dir}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _load_forecaster_class(model: str):
    """根据模型名动态导入对应的 forecaster 类。"""
    if model == "itransformer":
        from tsas.engine.operator.forecasting.itransformer import ITransformerForecaster
        return ITransformerForecaster
    if model == "lightgbm":
        from tsas.engine.operator.forecasting.lightgbm import LightGBMForecaster
        return LightGBMForecaster
    if model == "xgboost":
        from tsas.engine.operator.forecasting.xgboost import XGBoostForecaster
        return XGBoostForecaster
    raise ValueError(f"不支持的模型: {model}")


def _align_to_chunk_end(chunk_ids: np.ndarray, target: int, n_total: int) -> int:
    """将目标行索引对齐到所在 chunk 的末尾。"""
    if target >= n_total:
        return n_total
    cid = chunk_ids[target]
    return int(np.where(chunk_ids == cid)[0][-1]) + 1


def _get_valid_indices(chunk_ids: np.ndarray, input_len: int, output_len: int) -> np.ndarray:
    """返回在每个连续 chunk 内可完整构成窗口的起始索引。"""
    valid_indices = []
    for cid in np.unique(chunk_ids):
        mask = chunk_ids == cid
        chunk_row_indices = np.where(mask)[0]
        num_samples = len(chunk_row_indices) - input_len - output_len + 1
        if num_samples > 0:
            valid_indices.extend(chunk_row_indices[:num_samples].tolist())
    return np.array(valid_indices, dtype=int)


def _compute_dtw(y_true: np.ndarray, y_pred: np.ndarray, max_len: int = 2000) -> float:
    """计算归一化 DTW 距离；数据过长时均匀采样，防止卡住。

    未安装 fastdtw 时退化为 MAE。
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    if len(y_true) > max_len:
        idx = np.linspace(0, len(y_true) - 1, max_len).astype(int)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    if HAS_FASTDTW:
        dist, _ = fastdtw(y_true, y_pred, dist=lambda a, b: abs(a - b))
        return dist / len(y_true)
    return float(np.mean(np.abs(y_true - y_pred)))


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    naive_error: float,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """计算 RMSE、MAE、MAPE(%)、MASE、DTW。"""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)
    mase = float(mae / naive_error if naive_error > epsilon else mae / epsilon)
    dtw = float(_compute_dtw(y_true, y_pred))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "MASE": mase,
        "DTW": dtw,
    }


def run_inference(
    model_save_dir: str,
    preprocessed_dir: str,
    tsa_suite_root: str = DEFAULT_TSA_SUITE_ROOT,
    batch_size: int = 256,
    model: str = "itransformer",
) -> dict[str, dict[str, float]]:
    """加载已保存模型并对测试集执行推理与评估。

    Parameters
    ----------
    model_save_dir : str
        模型保存目录。
    preprocessed_dir : str
        预处理输出目录。
    tsa_suite_root : str, optional
        TSA-Suite 仓库根目录。
    batch_size : int, optional
        推理批次大小。
    model : str, optional
        模型类型，支持 itransformer / lightgbm / xgboost。

    Returns
    -------
    dict
        {
            "per_step": {step: {metric: value, ...}, ...},
            "overall": {metric: value, ...},
        }
    """
    _ensure_tsa_suite_import(tsa_suite_root)

    ForecasterClass = _load_forecaster_class(model)

    save_dir = Path(model_save_dir)
    prep_dir = Path(preprocessed_dir)

    # 加载模型
    print(f"Loading {model} model from {save_dir}...")
    forecaster = ForecasterClass.load(save_dir)
    cfg = forecaster.config

    # 加载预处理数据
    x = np.load(prep_dir / "x.npy")
    y = np.load(prep_dir / "y.npy")
    chunk_ids_path = prep_dir / "chunk_ids.npy"
    chunk_ids = np.load(chunk_ids_path) if chunk_ids_path.exists() else None

    # itransformer 支持 chunk_ids，树模型当前不依赖 chunk_ids
    if chunk_ids is not None and hasattr(forecaster, "set_chunk_ids"):
        forecaster.set_chunk_ids(chunk_ids)

    n_total = len(x)

    # 按时间边界确定测试集（训练 + 验证之后的数据）
    if chunk_ids is not None and hasattr(forecaster, "set_chunk_ids"):
        train_end = _align_to_chunk_end(chunk_ids, int(n_total * cfg.train_ratio), n_total)
        val_end = _align_to_chunk_end(
            chunk_ids, train_end + int(n_total * cfg.val_ratio), n_total
        )
        test_chunk_ids = chunk_ids[val_end:]
        test_indices = _get_valid_indices(test_chunk_ids, cfg.seq_len, cfg.pred_len) + val_end
    else:
        test_start = int(n_total * (cfg.train_ratio + cfg.val_ratio))
        test_indices = np.arange(
            test_start, n_total - cfg.seq_len - cfg.pred_len + 1
        )

    if len(test_indices) == 0:
        raise ValueError("测试集无法构造窗口，请检查数据长度与 train_ratio/val_ratio")

    print(f"Test samples: {len(test_indices)}")

    # 构造测试窗口
    x_test = np.stack([x[i : i + cfg.seq_len] for i in test_indices])
    y_test = np.stack(
        [y[i + cfg.seq_len : i + cfg.seq_len + cfg.pred_len] for i in test_indices]
    )

    # naive error：用最后一个已知值重复预测未来 horizon 步的 MAE
    base_values = y_test[:, 0, :]  # (N, num_targets)
    naive_pred = np.repeat(base_values[:, np.newaxis, :], cfg.pred_len, axis=1)
    naive_error = float(np.mean(np.abs(y_test - naive_pred)))

    # 批量推理
    print("Running inference...")
    preds = []
    for i in range(0, len(x_test), batch_size):
        batch = x_test[i : i + batch_size]
        pred = forecaster.run(batch)
        preds.append(pred)
    y_pred = np.concatenate(preds, axis=0)

    # 逐 step 指标
    per_step = {}
    print("\n=== Per-step metrics ===")
    header = f"{'Step':>6} | {'RMSE':>10} | {'MAE':>10} | {'MAPE(%)':>10} | {'MASE':>10} | {'DTW':>10}"
    print(header)
    print("-" * len(header))
    for h in range(cfg.pred_len):
        metrics = _compute_metrics(y_test[:, h, :], y_pred[:, h, :], naive_error)
        per_step[h + 1] = metrics
        print(
            f"{h + 1:>6} | {metrics['RMSE']:>10.4f} | {metrics['MAE']:>10.4f} | "
            f"{metrics['MAPE(%)']:>10.2f} | {metrics['MASE']:>10.4f} | {metrics['DTW']:>10.4f}"
        )

    # 整体指标
    overall = _compute_metrics(y_test, y_pred, naive_error)
    print("\n=== Overall metrics ===")
    for name, value in overall.items():
        print(f"{name}: {value:.4f}")

    return {"per_step": per_step, "overall": overall}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TSA-Suite inference and evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="itransformer",
        choices=["itransformer", "lightgbm", "xgboost"],
        help="模型类型，默认 itransformer。",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="saved_models/tsa_itransformer",
        help="训练好的模型保存目录。",
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        default="preprocessed_tsa",
        help="预处理输出目录。",
    )
    parser.add_argument(
        "--tsa_suite_root",
        type=str,
        default=DEFAULT_TSA_SUITE_ROOT,
        help="TSA-Suite 仓库根目录。",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="推理批次大小。",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_inference(
        model_save_dir=args.model_save_dir,
        preprocessed_dir=args.preprocessed_dir,
        tsa_suite_root=args.tsa_suite_root,
        batch_size=args.batch_size,
        model=args.model,
    )


if __name__ == "__main__":
    main()
