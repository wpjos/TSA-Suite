# -*- coding: utf-8 -*-

"""
均值预测器

重构型预测器，学习训练数据的列均值，推理时将均值广播为预测值。
常用于 CompositeScorer 路径中作为基线预测器，残差即为样本到均值的偏差。

示例用法::

    predictor = MeanPredictor()
    predictor.fit(train_data)            # 学习列均值
    pred, eo = predictor.run(test_data)  # 广播均值作为预测值
"""

import numpy as np
import pandas as pd

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import BasePredictorMixin

__all__ = ['MeanPredictor']


class MeanPredictor(UnsupervisedNumericOperatorMixin[None],
                    BasePredictorMixin[None, None, None],
                    NumericOperator[None, None, None]):
    """
    均值预测器 — 重构型预测器

    训练阶段学习训练数据的列均值（各特征的均值向量），
    推理阶段将均值向量广播为与输入同维度的预测值。

    核心逻辑:
        - ``_fit``: 计算 ``x.mean(axis=0)`` 并保存为 ``_mean``
        - ``_run``: 将 ``_mean`` 广播为与输入同形状的预测值

    典型应用:
        作为 CompositeScorer 中的基线预测器。当与 ResidualComparator 组合时，
        比较结果等价于每个样本到均值的逐元素偏差的聚合度量。

    Input:
        x: 特征矩阵，形状 (n_samples, n_features)

    Output:
        重构值矩阵，与输入同形状（每行均为训练数据的列均值）

    泛型参数:
        - EO: None（无附加输出）
        - C: MeanPredictorConfig
        - RP: None（无运行参数）
        - FP: None（无训练参数）
    """

    @classmethod
    def name(cls) -> str:
        return "mean_predictor"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None, config: None = None, **kwargs):
        """
        初始化均值预测器

        Args:
            **kwargs: 透传给基类的参数，支持:
                - n_components (int): 兼容参数，默认 1
                - oid (str): 算子标识
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._mean: np.ndarray | None = None
        """训练数据的列均值向量"""

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """
        学习列均值

        计算训练数据沿样本轴（axis=0）的均值，保存为内部状态。
        训练完成后设置 ``_fitted = True``。

        Args:
            x(np.ndarray): 训练数据，形状 (n_samples, n_features)
            params(None): 无训练参数
        """
        # 沿样本轴求均值，得到 (n_features,) 向量
        self._mean = x.mean(axis=0)

    def _run_data(self, x: np.ndarray, params: None, idx: pd.Index | None = None) -> np.ndarray:
        """
        广播均值作为预测值

        将训练阶段学习的均值向量广播为与输入同形状的预测矩阵。
        输出形状与输入完全相同: (n_samples, n_features)。

        Args:
            x(np.ndarray): 输入数据，形状 (n_samples, n_features)
            params(None): 无运行参数

        Returns:
            np.ndarray: 预测值 ndarray，形状 (n_samples, n_features)，每行均为 _mean
        """
        pred = np.broadcast_to(self._mean, x.shape).copy()  # broadcast_to 不复制数据，需要 copy 以确保可写
        return pred
