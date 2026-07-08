# -*- coding: utf-8 -*-

"""
均值合并评分器

将多列异常分数通过多种平均方法合并为单列异常分数，支持加权合并和后验权重矩阵。

支持四种平均计算方法:
    - ARITHMETIC: 算术平均，无输入约束
    - GEOMETRIC: 几何平均，要求输入全为正数
    - HARMONIC: 调和平均，要求输入全为正数
    - QUADRATIC: 平方平均（RMS），无输入约束

后验权重矩阵表示各列在对应方法计算空间中对合并分数的贡献比例，行和始终为 1
（合并分数为零时除零保护，对应行设为 0）。

示例用法::

    # 等权算术平均
    scorer = MeanMergeScorer()
    scores, eo = scorer.run(data)

    # 加权几何平均
    scorer = MeanMergeScorer(config=MeanMergeScorerConfig(
        method=ScoreMergeMethod.GEOMETRIC,
        weights=[0.3, 0.5, 0.2],
    ))
    scores, eo = scorer.run(data)
"""

from enum import Enum

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.base import NumericOperator
from tsas.engine.operator.detection.base import SingleScorerMixin

__all__ = [
    'ScoreMergeMethod',
    'MeanMergeScorerExtraOutput',
    'MeanMergeScorerConfig',
    'MeanMergeScorer',
]


# ============================================================================
# 枚举定义
# ============================================================================


class ScoreMergeMethod(str, Enum):
    """分数合并平均方法枚举

    定义多列异常分数合并时使用的平均计算方法。

    Attributes:
        ARITHMETIC: 算术平均，适用于任意实数输入
        GEOMETRIC: 几何平均，要求输入全为正数，适用于乘积型数据
        HARMONIC: 调和平均，要求输入全为正数，适用于比率型数据
        QUADRATIC: 平方平均（RMS），适用于任意实数输入，对大值更敏感
    """
    ARITHMETIC = "arithmetic"
    """算术平均"""
    GEOMETRIC = "geometric"
    """几何平均"""
    HARMONIC = "harmonic"
    """调和平均"""
    QUADRATIC = "quadratic"
    """平方平均（RMS）"""


# ============================================================================
# 附加输出
# ============================================================================


class MeanMergeScorerExtraOutput(BaseModel):
    """均值合并评分器附加输出

    Attributes:
        posterior_weights (np.ndarray): 后验权重矩阵，形状 (n_samples, n_features)，
            各列在对应方法计算空间中对合并分数的贡献比例，行和为 1
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    posterior_weights: np.ndarray = Field(
        description="后验权重矩阵 (n_samples, n_features)，各列在对应方法计算空间中对合并分数的贡献比例"
    )


# ============================================================================
# Config
# ============================================================================


class MeanMergeScorerConfig(BaseModel):
    """均值合并评分器配置

    Attributes:
        method (ScoreMergeMethod): 平均计算方法，默认 ARITHMETIC
        weights (list[float] | None): 各列权重向量，None 时等权合并
    """
    model_config = ConfigDict(frozen=True)

    method: ScoreMergeMethod = Field(
        default=ScoreMergeMethod.ARITHMETIC,
        description="平均计算方法: 'arithmetic', 'geometric', 'harmonic', 'quadratic'",
    )
    weights: list[float] | None = Field(
        default=None,
        description="各列权重向量，None 时等权合并；非 None 时长度须与输入列数一致",
    )


# ============================================================================
# 均值合并评分器
# ============================================================================


class MeanMergeScorer(SingleScorerMixin[None],
                      NumericOperator[MeanMergeScorerExtraOutput, MeanMergeScorerConfig, None]):
    """均值合并评分器

    将多列异常分数通过多种平均方法合并为单列异常分数。

    支持四种平均计算方法:
        - ARITHMETIC: 算术平均 ``Σ(w_i·x_i) / Σw_i``
        - GEOMETRIC: 几何平均 ``∏(x_i^w_i)^(1/Σw_i)``
        - HARMONIC: 调和平均 ``Σw_i / Σ(w_i/x_i)``
        - QUADRATIC: 平方平均 ``√(Σ(w_i·x_i²) / Σw_i)``

    后验权重矩阵在各方法的计算空间中计算:
        - ARITHMETIC: ``w_i·x_i / Σ(w_j·x_j)`` — 原始空间
        - GEOMETRIC: ``w_i·log(x_i) / Σ(w_j·log(x_j))`` — 对数空间
        - HARMONIC: ``(w_i/x_i) / Σ(w_j/x_j)`` — 倒数空间
        - QUADRATIC: ``w_i·x_i² / Σ(w_j·x_j²)`` — 平方空间

    特点:
        - **无需训练**: 直接推理，无需调用 ``fit``
        - **单输出列**: 输出列名固定为 ``"score"``
        - **加权支持**: 通过 Config 指定各列权重

    Input:
        x: 多列异常分数矩阵，形状 (n_samples, n_features)

    Output:
        合并后异常分数，形状 (n_samples,)，值越大越异常。
        后验权重矩阵由附加输出 ``MeanMergeScorerExtraOutput`` 提供

    泛型参数:
        - EO: MeanMergeScorerExtraOutput（附加输出含后验权重矩阵）
        - C: MeanMergeScorerConfig
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        """返回算子名称

        Returns:
            str: ``"mean_merge_scorer"``
        """
        return "mean_merge_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def _validate_ndarray_input(self, x: np.ndarray, params: None) -> None:
        """校验 ndarray 输入

        GEOMETRIC 和 HARMONIC 方法要求输入全为正数。

        Args:
            x (np.ndarray): 输入 ndarray
            params (None): 无运行参数

        Raises:
            ValueError: GEOMETRIC 或 HARMONIC 模式下输入包含非正数值时
        """
        method = self.config.method
        if method in (ScoreMergeMethod.GEOMETRIC, ScoreMergeMethod.HARMONIC):
            if np.any(x <= 0):
                min_val = float(np.min(x))
                raise ValueError(
                    f"{method.value} 模式要求输入数据全为正数，"
                    f"但检测到非正数值（最小值: {min_val}）"
                )

    def _run_data(self, x: np.ndarray, params: None,
                  idx: pd.Index | None = None) -> tuple[np.ndarray, MeanMergeScorerExtraOutput]:
        """计算多列异常分数的均值合并

        根据配置的平均方法和权重，将多列分数合并为单列分数，
        并计算后验权重矩阵。

        Args:
            x (np.ndarray): 输入数据（多列异常分数），形状 (n_samples, n_features)
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, MeanMergeScorerExtraOutput]:
                - merged_scores: 合并后异常分数，形状 (n_samples,)
                - eo: 附加输出，含后验权重矩阵
        """
        method = self.config.method
        # 解析权重
        weights = np.array(self.config.weights, dtype=float) if self.config.weights else None
        n_features = x.shape[1]

        # 归一化权重（等权时使用均匀权重）
        if weights is not None:
            normalized_weights = weights / np.sum(weights)
        else:
            normalized_weights = np.ones(n_features) / n_features

        # 根据方法计算合并分数和后验权重
        if method == ScoreMergeMethod.ARITHMETIC:
            # 算术平均: Σ(w_i·x_i) / Σw_i，归一化后 Σw_i=1，即 Σ(w_i·x_i)
            weighted_values = x * normalized_weights  # (n_samples, n_features)
            merged = np.sum(weighted_values, axis=1)  # (n_samples,)
            # 后验权重: w_i·x_i / Σ(w_j·x_j) — 原始空间
            contribution = weighted_values

        elif method == ScoreMergeMethod.GEOMETRIC:
            # 几何平均: ∏(x_i^w_i)^(1/Σw_i)，归一化后即 ∏(x_i^w_i)
            # 在对数空间计算: exp(Σ(w_i·log(x_i)))
            log_x = np.log(x)  # (n_samples, n_features)
            weighted_log = log_x * normalized_weights  # (n_samples, n_features)
            merged_log = np.sum(weighted_log, axis=1)  # (n_samples,)
            merged = np.exp(merged_log)  # (n_samples,)
            # 后验权重: w_i·log(x_i) / Σ(w_j·log(x_j)) — 对数空间
            contribution = weighted_log

        elif method == ScoreMergeMethod.HARMONIC:
            # 调和平均: Σw_i / Σ(w_i/x_i)，归一化后即 1 / Σ(w_i/x_i)
            reciprocal_x = 1.0 / x  # (n_samples, n_features)
            weighted_reciprocal = reciprocal_x * normalized_weights  # (n_samples, n_features)
            sum_weighted_reciprocal = np.sum(weighted_reciprocal, axis=1)  # (n_samples,)
            merged = 1.0 / sum_weighted_reciprocal  # (n_samples,)
            # 后验权重: (w_i/x_i) / Σ(w_j/x_j) — 倒数空间
            contribution = weighted_reciprocal

        else:  # ScoreMergeMethod.QUADRATIC
            # 平方平均: √(Σ(w_i·x_i²) / Σw_i)，归一化后即 √(Σ(w_i·x_i²))
            squared_x = x ** 2  # (n_samples, n_features)
            weighted_squared = squared_x * normalized_weights  # (n_samples, n_features)
            merged = np.sqrt(np.sum(weighted_squared, axis=1))  # (n_samples,)
            # 后验权重: w_i·x_i² / Σ(w_j·x_j²) — 平方空间
            contribution = weighted_squared

        # 计算后验权重矩阵: contribution_i / Σ(contribution_j)
        # 除零保护: 当分母为 0 时后验权重设为 0
        contribution_sum = np.sum(contribution, axis=1)  # (n_samples,)
        safe_sum = np.where(contribution_sum == 0, 1.0, contribution_sum)
        posterior_weights = contribution / safe_sum[:, np.newaxis]  # (n_samples, n_features)
        posterior_weights[contribution_sum == 0] = 0.0

        eo = MeanMergeScorerExtraOutput(posterior_weights=posterior_weights)
        return merged.ravel(), eo
