# -*- coding: utf-8 -*-

"""
无标签自评估指标算子

在无真实标签的场景下，通过分析异常分数分布特性来评估检测器质量。
核心方法：计算分数的变异系数（CV）并通过 sigmoid 映射到 [0, 1] 区间。

核心组件:
    - SelfEvaluationConfig: 配置类（继承 BaseMetricConfig）
    - SelfEvaluation: 无标签自评估指标算子

使用示例::

    from tsas.engine.operator.evaluation import SelfEvaluation

    # 基本用法
    op = SelfEvaluation()
    score = op.run(scores)  # -> float

    # HPO 集成
    op = SelfEvaluation()
    scores_dict = op.scores(scores)  # -> {"self_eval": float}

算法说明::

    1. 计算分数的均值（绝对值）和标准差
    2. 变异系数 CV = std / |mean|
    3. 使用 sigmoid 映射: result = 1 / (1 + exp(-CV))
    4. CV=0 → ~0.5, CV=1 → ~0.73, CV=2 → ~0.88

原理:
    好的异常检测器应该产生较大的分数差异（正常和异常的分数差距大），
    变异系数越大表示分数分布越不均匀，检测效果越好。
"""

from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'SelfEvaluationConfig',
    'SelfEvaluation',
]


# ============================================================================
# Config 类定义
# ============================================================================

class SelfEvaluationConfig(BaseMetricConfig):
    """无标签自评估指标配置类

    继承自 :class:`BaseMetricConfig`，用于配置无标签自评估算子的行为。
    本配置类采用 frozen 模式（Pydantic ConfigDict），实例创建后不可修改，
    保证了配置的不可变性和线程安全性。

    默认配置下，``main_scores`` 指定了一个名为 ``"self_eval"`` 的评分提取路径，
    使用 ``"_"`` 占位符表示直接从 ``_run`` 返回的 float 结果中取值。

    Attributes:
        main_scores (dict[str, str] | None): 主评分路径映射字典，用于
            :meth:`BaseMetricOperator.scores` 方法按名称提取标量指标值。
            - 键 (str): 指标名称，如 ``"self_eval"``
            - 值 (str): 从 _run 返回值中提取标量的属性路径。
              对于 float 类型返回值，统一使用 ``"_"`` 占位符，表示直接取值。
            - 默认值为 ``{"self_eval": "_"}``
        model_config (ConfigDict): Pydantic 模型配置，``frozen=True`` 表示
            实例不可变
    """

    model_config = ConfigDict(frozen=True)

    main_scores: dict[str, str] | None = Field(
        default={"self_eval": "_"},
        description="主评分路径映射，键为指标名称、值为结果属性路径；float 类型 MR 使用 '_' 占位符表示直接取值",
    )


# ============================================================================
# 算子类定义
# ============================================================================

class SelfEvaluation(
    BaseMetricOperator[
        np.ndarray,
        float,
        SelfEvaluationConfig,
        None,
    ],
):
    """无标签自评估指标算子

    输入为异常分数数组，输出为一个 [0, 1) 范围的自评估标量分数。
    核心思想是：好的异常检测器应该产生较大的分数差异（正常样本与异常样本
    的分数差距大），因此通过计算分数的变异系数（CV = std / |mean|）来衡量
    检测器的区分能力，并使用 sigmoid 函数将 CV 映射到 [0, 1) 区间。

    本算子是 :class:`BaseMetricOperator` 的子类，遵循无状态纯函数设计，
    可通过 :meth:`run` 直接获取标量结果，或通过 :meth:`scores` 按
    ``main_scores`` 配置获取命名标量字典供 HPO 使用。

    Input:
        x: 异常分数数组，形状任意（多维会被展平为一维）

    Output:
        自评估分数，标量，范围约 [0, 1)，值越大表示检测器区分能力越强。
        可通过 Config 的 ``main_scores`` 配置（默认 ``{"self_eval": "_"}``）
        提取命名标量用于 HPO。

    泛型参数:
        I (np.ndarray): 输入类型 — 异常分数数组，支持一维或多维（多维会被展平）
        MR (float): 指标结果类型 — 自评估标量分数，范围约 [0, 1)
        MC (SelfEvaluationConfig): 实例参数类型 — 配置类
        RP (None): 运行参数类型 — 无运行参数

    Attributes:
        _run_params_type (ClassVar[type | None]): 运行参数的类型声明，
            固定为 None，表示本算子不需要运行时参数
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "self_evaluation"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def _run(
        self,
        x: np.ndarray,
        *,
        params: None,
    ) -> float:
        """计算无标签自评估分数

        通过分析异常分数的变异系数（Coefficient of Variation, CV）来评估
        检测器的区分质量，无需真实标签。核心流程如下：

        1. **输入预处理**: 将输入转为 numpy 数组并展平为一维向量
        2. **边界检查**: 处理空数组和零均值等退化情况
        3. **变异系数计算**: CV = std(scores) / |mean(|scores|)|
        4. **Sigmoid 映射**: result = 1 / (1 + exp(-CV))，将 CV 映射到 [0, 1)

        算法原理：
            变异系数（CV）衡量的是数据的离散程度相对于均值的比例。
            在异常检测场景中，好的检测器应该让正常样本和异常样本产生
            较大的分数差异（即分数分布更不均匀），CV 值越大说明分数
            的区分度越好。通过 sigmoid 映射将 CV 转换为 [0, 1) 的标量
            评分，便于与其他指标统一比较。

        Args:
            x (np.ndarray): 异常分数数组，支持一维或多维。
                多维数组会被展平（ravel）为一维进行处理。
                典型输入包括检测器输出的 anomaly score 向量。
            params (None): 运行时参数，本算子不使用，始终为 None

        Returns:
            float: 自评估标量分数，范围约 [0, 1)，值越大表示检测器
                的区分能力越强。
                - CV = 0 时，result ≈ 0.5（分数无差异）
                - CV = 1 时，result ≈ 0.73（中等区分度）
                - CV = 2 时，result ≈ 0.88（较强区分度）
                - CV → ∞ 时，result → 1.0（极强区分度）
                - 空数组或零均值时，result = 0.0（退化情况）

        Raises:
            TypeError: 当输入 x 无法转换为 numpy 数组时抛出
                （由 np.asarray 内部抛出）
        """
        # ========== 阶段 1: 输入预处理 ==========
        # 将输入统一转换为 numpy ndarray，支持 list、tuple 等可转换类型；
        # ravel() 将多维数组展平为一维，确保后续统计计算的一致性
        scores = np.asarray(x).ravel()

        # ========== 阶段 2: 退化情况处理 ==========
        # 空数组无法计算有意义的统计量，直接返回 0.0 表示无区分能力
        if len(scores) == 0:
            return 0.0

        # ========== 阶段 3: 计算均值与标准差 ==========
        # 计算分数绝对值的均值作为变异系数的分母，
        # 使用绝对值是为了避免正负分数相互抵消导致的均值趋近于零
        mean_val = np.mean(np.abs(scores))

        # ========== 阶段 4: 零均值保护 ==========
        # 当所有分数的绝对值均值极小（< 1e-10）时，
        # 说明检测器几乎未产生有效区分，此时返回 0.0 避免除零异常
        if mean_val < 1e-10:
            return 0.0

        # ========== 阶段 5: 变异系数计算 ==========
        # CV（Coefficient of Variation）= 标准差 / 绝对均值
        # CV 越大说明分数分布越不均匀，正常/异常样本的区分度越好
        cv = float(np.std(scores) / mean_val)

        # ========== 阶段 6: Sigmoid 映射 ==========
        # 使用 sigmoid 函数将 CV（范围 [0, +inf)）映射到 (0.5, 1) 区间
        # 特殊值参考：
        #   cv=0 → sigmoid(0) = 0.5  （无区分）
        #   cv=1 → sigmoid(1) ≈ 0.731 （中等区分）
        #   cv=2 → sigmoid(2) ≈ 0.881 （较强区分）
        result = 1.0 / (1.0 + np.exp(-cv))

        # 确保 float 标量返回，避免返回 numpy 浮点类型
        return float(result)
