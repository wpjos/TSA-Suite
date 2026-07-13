# -*- coding: utf-8 -*-

"""
时序预测评价指标算子

提供工业时序预测中常用的 4 项评价指标：
MAE、RMSE、MAPE、DTW。

输入为 ``(y_true, y_pred)`` 元组，支持 ndarray 与 DataFrame，
内部统一拉平后计算（保持与 HBHD_predict_v1.5 的 ``calculate_metrics`` 一致）。
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import mean_squared_error

from tsas.engine.operator.evaluation.base import BaseMetricConfig, BaseMetricOperator

__all__ = [
    'ForecastingMetricResult',
    'ForecastingMetricConfig',
    'ForecastingMetrics',
]


def _to_ndarray(x):
    """统一转换为 ndarray 并拉平。"""
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    return np.asarray(x).ravel()


class ForecastingMetricResult(BaseModel):
    """时序预测评价指标结果。"""

    model_config = ConfigDict(frozen=True)

    mae: float
    rmse: float
    mape: float
    dtw: float


class ForecastingMetricConfig(BaseMetricConfig):
    """时序预测指标算子配置。

    ``main_scores`` 默认暴露全部 4 项指标，用户可覆写以选择 HPO 优化目标。
    """

    main_scores: Optional[Dict[str, str]] = {
        "mae": "mae",
        "rmse": "rmse",
        "mape": "mape",
        "dtw": "dtw",
    }
    epsilon: float = Field(default=1e-8, ge=1e-12, le=1.0, description="零值保护常数")
    max_dtw_len: int = Field(default=2000, ge=100, le=100000, description="DTW 最大采样长度")


class ForecastingMetrics(
    BaseMetricOperator[
        Tuple[np.ndarray, np.ndarray],
        ForecastingMetricResult,
        ForecastingMetricConfig,
        None,
    ],
):
    """时序预测多指标算子。

    输入 ``x`` 为 ``(y_true, y_pred)``：

        y_true, y_pred: ndarray 或 DataFrame，形状任意但元素数必须相同。

    输出 ``ForecastingMetricResult`` 包含全部指标；
    通过 ``scores()`` 可按 ``main_scores`` 提取子集供 HPO 使用。
    """

    @classmethod
    def name(cls) -> str:
        return "forecasting_metrics"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号。

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def _run(self, x: tuple[np.ndarray, np.ndarray], *, params: None) -> ForecastingMetricResult:
        y_true_raw, y_pred_raw = x
        y_true = _to_ndarray(y_true_raw)
        y_pred = _to_ndarray(y_pred_raw)

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"y_true 与 y_pred 长度不一致: {len(y_true)} != {len(y_pred)}"
            )
        if len(y_true) == 0:
            raise ValueError("输入数组为空")

        cfg = self.config
        epsilon = cfg.epsilon

        # 基础误差
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

        # DTW（归一化），fastdtw 为可选依赖
        dtw = self._compute_dtw(y_true, y_pred, cfg.max_dtw_len)

        return ForecastingMetricResult(
            mae=mae,
            rmse=rmse,
            mape=mape,
            dtw=dtw,
        )

    def _compute_dtw(self, y_true: np.ndarray, y_pred: np.ndarray, max_len: int) -> float:
        """计算归一化 DTW，大样本时均匀降采样。"""
        try:
            from fastdtw import fastdtw
        except ImportError:
            warnings.warn("fastdtw 未安装，DTW 指标回退为 MAE")
            return float(np.mean(np.abs(y_pred - y_true)))

        n = len(y_true)
        if n > max_len:
            idx = np.linspace(0, n - 1, max_len).astype(int)
            dist, _ = fastdtw(y_true[idx], y_pred[idx], dist=lambda a, b: abs(a - b))
            return float(dist / max_len)
        else:
            dist, _ = fastdtw(y_true, y_pred, dist=lambda a, b: abs(a - b))
            return float(dist / n)
