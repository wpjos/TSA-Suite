#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实工业时序预测训练与验证流程。

本脚本封装 run_tsa_forecast.py 的核心逻辑，提供结构化的 Pipeline 类：
1. 可选：调用 preprocess_for_tsa.py 生成 x.npy / y.npy / chunk_ids.npy
2. 加载预处理后的 NumPy 数组
3. 构建并训练 ITransformerForecaster
4. 保存模型
5. 可选：调用 inference_tsa.py 在测试集上评估

用法：
    python run_train_pipeline.py --run_inference
    python run_train_pipeline.py --run_preprocessing --data_file <csv> --run_inference
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# 0. 环境准备
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
from config import setup_paths

PATHS = setup_paths()
REPO_ROOT = PATHS["REPO_ROOT"]
SRC_DIR = PATHS["SRC_DIR"]

# 为了能够从项目根目录导入 preprocess_for_tsa 和 inference_tsa
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")

# 延迟导入 torch，便于在环境检查阶段给出友好提示
try:
    import torch
except ImportError as e:
    print("错误：当前环境缺少 torch，请先安装 torch。")
    print(str(e))
    sys.exit(1)

from tsas.engine.operator.forecasting.itransformer import (
    ITransformerForecaster,
    ITransformerForecasterConfig,
)

DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "real_forecast_config.yaml"

# ---------------------------------------------------------------------------
# 1. 配置解析
# ---------------------------------------------------------------------------

def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """从 YAML 配置文件加载完整配置。"""
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"配置文件格式错误: {config_path}")

    return cfg


def resolve_config(
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """解析并合并配置文件与命令行覆盖参数。"""
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    cfg = load_yaml_config(config_path)

    # 默认值
    operator_cfg = cfg.get("operator", {}).get("config", {}).copy()
    preprocessing_cfg = cfg.get("preprocessing", {}).copy()
    paths_cfg = cfg.get("paths", {}).copy()

    # 应用覆盖
    if overrides:
        # 模型相关覆盖
        model_keys = {
            "seq_len", "pred_len", "d_model", "nhead", "num_layers",
            "dim_feedforward", "dropout", "kan_grid_size", "target_idx",
            "epochs", "batch_size", "lr", "weight_decay", "early_stop_patience",
            "train_ratio", "val_ratio", "device", "lag_aware", "step_cond_head",
            "trend_weight",
        }
        for key in model_keys:
            if overrides.get(key) is not None:
                operator_cfg[key] = overrides[key]

        # 预处理相关覆盖
        prep_keys = {"time_col", "max_gap", "target_col", "chunk_size", "ema_alpha"}
        for key in prep_keys:
            if overrides.get(key) is not None:
                preprocessing_cfg[key] = overrides[key]

        # 路径相关覆盖
        if overrides.get("preprocessed_dir") is not None:
            paths_cfg["preprocessed_dir"] = overrides["preprocessed_dir"]
        if overrides.get("model_save_dir") is not None:
            paths_cfg["model_save_dir"] = overrides["model_save_dir"]

    #  dim_feedforward 默认值
    if operator_cfg.get("dim_feedforward") is None:
        operator_cfg["dim_feedforward"] = operator_cfg.get("d_model", 512) * 2

    # 确保路径是相对于 REPO_ROOT 的绝对路径
    paths_cfg["preprocessed_dir"] = str(REPO_ROOT / paths_cfg["preprocessed_dir"])
    paths_cfg["model_save_dir"] = str(REPO_ROOT / paths_cfg["model_save_dir"])

    return {
        "operator": operator_cfg,
        "preprocessing": preprocessing_cfg,
        "paths": paths_cfg,
    }


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("real_forecast_pipeline")


# ---------------------------------------------------------------------------
# 2. Pipeline
# ---------------------------------------------------------------------------

class RealForecastPipeline:
    """真实工业时序预测训练与验证 Pipeline。"""

    def __init__(
        self,
        logger: logging.Logger,
        cfg: dict[str, Any],
        run_preprocessing: bool = False,
        data_file: str | None = None,
        run_inference: bool = False,
        seed: int | None = None,
    ) -> None:
        self.logger = logger
        self.cfg = cfg
        self.run_preprocessing = run_preprocessing
        self.data_file = data_file
        self.run_inference = run_inference
        self.seed = seed
        self.forecaster: ITransformerForecaster | None = None

    def set_seed(self) -> None:
        """固定随机种子（可选）。"""
        if self.seed is None:
            return
        import random
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.logger.info("随机种子已固定: %d", self.seed)

    def preprocess(self) -> None:
        """可选：调用 preprocess_for_tsa.py 进行预处理。"""
        if not self.run_preprocessing:
            return

        if not self.data_file:
            raise ValueError("--run_preprocessing 需要提供 --data_file")

        from preprocess_for_tsa import PreprocessConfig, PreprocessPipeline

        prep_cfg = self.cfg["preprocessing"]
        cfg = PreprocessConfig(
            data_file=self.data_file,
            output_dir=self.cfg["paths"]["preprocessed_dir"],
            time_col=prep_cfg.get("time_col", "datatime"),
            max_gap=prep_cfg.get("max_gap", 5.0),
            target_col=prep_cfg.get("target_col", "diya_qibao_shuiwei_youxuanzhi"),
            chunk_size=prep_cfg.get("chunk_size", 100000),
            ema_alpha=prep_cfg.get("ema_alpha", 0.3),
        )

        self.logger.info("开始预处理: %s", self.data_file)
        pipeline = PreprocessPipeline(cfg)
        pipeline.run_and_save()
        self.logger.info("预处理完成。")

    def load_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载预处理后的 NumPy 数组。"""
        preprocessed_dir = Path(self.cfg["paths"]["preprocessed_dir"])
        x_path = preprocessed_dir / "x.npy"
        y_path = preprocessed_dir / "y.npy"
        chunk_ids_path = preprocessed_dir / "chunk_ids.npy"

        if not x_path.exists() or not y_path.exists():
            raise FileNotFoundError(
                f"预处理文件不存在。请先运行：\n"
                f"  python preprocess_for_tsa.py --data_file <your_csv> --output_dir {preprocessed_dir}\n"
                f"或带上 --run_preprocessing 参数运行本脚本。"
            )

        x = np.load(x_path)
        y = np.load(y_path)
        chunk_ids = np.load(chunk_ids_path) if chunk_ids_path.exists() else np.zeros(len(x), dtype=int)

        self.logger.info(
            "加载预处理数据: x=%s, y=%s, chunk_ids=%s",
            x.shape, y.shape, chunk_ids.shape,
        )
        return x, y, chunk_ids

    def build_forecaster(self) -> ITransformerForecaster:
        """根据配置构造 ITransformerForecaster。"""
        model_cfg = self.cfg["operator"].copy()
        # 移除 ITransformerForecasterConfig 不支持的字段（防御性）
        unsupported = {"input_columns", "target_column", "name"}
        for key in unsupported:
            model_cfg.pop(key, None)

        config = ITransformerForecasterConfig(**model_cfg)
        return ITransformerForecaster(config=config)

    def train(self, x: np.ndarray, y: np.ndarray, chunk_ids: np.ndarray) -> ITransformerForecaster:
        """训练模型并保存。"""
        self.forecaster = self.build_forecaster()
        self.forecaster.set_chunk_ids(chunk_ids)

        self.logger.info(
            "开始训练: x=%s, y=%s, device=%s",
            x.shape, y.shape, self.forecaster._device,
        )
        self.forecaster.fit(x, y)
        self.logger.info("训练完成。")

        model_save_dir = Path(self.cfg["paths"]["model_save_dir"])
        model_save_dir.mkdir(parents=True, exist_ok=True)
        self.forecaster.save(model_save_dir)
        self.logger.info("模型已保存到: %s", model_save_dir)

        return self.forecaster

    def infer(self) -> dict[str, Any]:
        """可选：在测试集上执行推理评估。"""
        if not self.run_inference:
            return {}

        from inference_tsa import run_inference

        self.logger.info("开始在测试集上推理评估...")
        metrics = run_inference(
            model_save_dir=self.cfg["paths"]["model_save_dir"],
            preprocessed_dir=self.cfg["paths"]["preprocessed_dir"],
            tsa_suite_root=str(REPO_ROOT),
            batch_size=self.cfg["operator"].get("batch_size", 128),
        )
        self.logger.info("推理评估完成。")
        return metrics

    def run(self) -> dict[str, Any]:
        """执行完整流程。"""
        start = datetime.now()
        self.logger.info("启动真实工业时序预测训练流程: %s", start.strftime("%Y-%m-%d %H:%M:%S"))

        self.set_seed()
        self.preprocess()
        x, y, chunk_ids = self.load_arrays()
        self.train(x, y, chunk_ids)
        metrics = self.infer()

        end = datetime.now()
        elapsed = (end - start).total_seconds()
        self.logger.info("流程结束，总耗时: %.2f 秒", elapsed)

        result = {
            "model_dir": self.cfg["paths"]["model_save_dir"],
            "metrics": metrics,
            "elapsed_seconds": elapsed,
        }
        self.logger.info("输出汇总: %s", json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return result


# ---------------------------------------------------------------------------
# 3. CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="真实工业时序预测训练与验证流程",
    )

    # 配置
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径，默认 {DEFAULT_CONFIG_PATH}",
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
        default=None,
        help="原始 CSV 路径（仅在 --run_preprocessing 时使用）。",
    )

    # 路径
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

    # 推理
    parser.add_argument(
        "--run_inference",
        action="store_true",
        help="训练结束后是否调用 inference_tsa.py 对测试集进行评估。",
    )

    # 随机种子
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="固定随机种子以获得可复现结果。",
    )

    # 模型超参数覆盖
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dim_feedforward", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--kan_grid_size", type=int, default=None)
    parser.add_argument("--target_idx", type=int, default=None)

    # 训练参数覆盖
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "npu"],
        help="计算设备。",
    )

    # 预处理参数覆盖
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--max_gap", type=float, default=None)
    parser.add_argument("--target_col", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--ema_alpha", type=float, default=None)

    return parser


def main() -> int:
    parser = _build_parser()
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
