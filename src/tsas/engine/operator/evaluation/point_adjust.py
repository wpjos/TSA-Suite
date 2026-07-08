# -*- coding: utf-8 -*-

"""
点调整评价指标算子（Point-Adjust, PA-F1）

用于时序异常检测的段级评价指标。核心思想：
如果一段连续异常区间中有任何一个点被正确检出，则将该区间内所有点都视为正确检出。

核心组件:
    - PointAdjustResult: PA 指标结果（Pydantic BaseModel）
    - PointAdjustConfig: 配置类（继承 BaseMetricConfig）
    - PointAdjust: 点调整指标算子

使用示例::

    from tsas.engine.operator.evaluation import PointAdjust

    # 基本用法（输入为一维离散标签对）
    op = PointAdjust()
    result = op.run((y_truth, y_predict))
    print(result.pa_f1, result.pa_recall)

    # HPO 集成
    op = PointAdjust(main_scores={"pa_f1": "pa_f1"})
    scores = op.scores((y_truth, y_predict))  # -> {"pa_f1": 0.85}

算法说明::

    1. 从 y_truth 中提取连续异常段（值为 positive_label 的连续区间）
    2. 对每个异常段：如果 y_predict 在该段内有至少一个点为 positive_label → 整段标记为 TP
    3. 如果该段内 y_predict 全部不是 positive_label → 整段标记为 FN
    4. y_predict 中不在任何异常段内的 positive_label 点 → 标记为 FP
    5. 计算 PA-Precision / PA-Recall / PA-F1

参考文献::

    Xu et al., "Unsupervised Anomaly Detection via Variational Auto-Encoder
    for Seasonal KPIs in Web Applications" (WWW 2018)
"""

from typing import ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.evaluation.base import (
    BaseMetricConfig,
    BaseMetricOperator,
)

__all__ = [
    'PointAdjustResult',
    'PointAdjustConfig',
    'PointAdjust',
]


# ============================================================================
# Result 类定义
# ============================================================================

class PointAdjustResult(BaseModel):
    """
    点调整评价指标结果。

    封装 Point-Adjust（PA）算法的全部计算产出，包括原始样本统计信息、
    PA 调整后的 TP / FP / FN 计数以及由此派生的 Precision / Recall / F1 指标。
    该模型为不可变对象（frozen=True），一旦创建便不可修改。

    Attributes:
        n_samples (int): 总样本数，即 y_truth 与 y_predict 的公共长度。
        n_anomaly_segments (int): 真实异常段数量，由 y_truth 中连续的
            positive_label 区间决定。
        pa_tp (int): PA 调整后的真正例数量。每个真实异常段中，只要至少有一个点
            被预测为 positive_label，则该整段均计为一个 TP。
        pa_fp (int): PA 调整后的假正例数量。在所有真实异常段之外的索引中，
            预测值为 positive_label 的点数之和。
        pa_fn (int): PA 调整后的假反例数量。每个真实异常段中，若没有任何一个点
            被预测为 positive_label，则该整段均计为一个 FN。
        pa_precision (float): PA-Precision，计算公式为 pa_tp / (pa_tp + pa_fp)，
            当分母为零时取 0.0。
        pa_recall (float): PA-Recall，计算公式为 pa_tp / (pa_tp + pa_fn)，
            当分母为零时取 0.0。
        pa_f1 (float): PA-F1，计算公式为 2 * pa_precision * pa_recall /
            (pa_precision + pa_recall)，当分母为零时取 0.0。
    """
    model_config = ConfigDict(frozen=True)

    n_samples: int = Field(description="总样本数，即 y_truth 与 y_predict 的公共长度")
    n_anomaly_segments: int = Field(description="真实异常段数量，由 y_truth 中连续的 positive_label 区间决定")
    pa_tp: int = Field(
        description="PA 调整后的真正例数量。每个真实异常段中，只要至少有一个点被预测为 positive_label，则该整段均计为一个 TP")
    pa_fp: int = Field(
        description="PA 调整后的假正例数量。在所有真实异常段之外的索引中，预测值为 positive_label 的点数之和")
    pa_fn: int = Field(
        description="PA 调整后的假反例数量。每个真实异常段中，若没有任何一个点被预测为 positive_label，则该整段均计为一个 FN")
    pa_precision: float = Field(description="PA-Precision，计算公式为 pa_tp / (pa_tp + pa_fp)，当分母为零时取 0.0")
    pa_recall: float = Field(description="PA-Recall，计算公式为 pa_tp / (pa_tp + pa_fn)，当分母为零时取 0.0")
    pa_f1: float = Field(
        description="PA-F1，计算公式为 2 * pa_precision * pa_recall / (pa_precision + pa_recall)，当分母为零时取 0.0")


# ============================================================================
# Config 类定义
# ============================================================================

class PointAdjustConfig(BaseMetricConfig):
    """
    点调整评价指标配置。

    继承自 :class:`BaseMetricConfig`，用于控制 PA 算子的行为参数，
    包括异常标签的判定值以及 HPO 场景下的主评分路径映射。

    Attributes:
        positive_label (int): 异常标签值，y_truth 和 y_predict 中等于该值的
            位置被判定为"异常"点，默认为 1。
        main_scores (dict[str, str] | None): 主评分路径映射。键为对外暴露的
            评分名称，值为 :class:`PointAdjustResult` 中对应的字段名；
            设为 None 时表示不注册任何主评分。默认为
            ``{"pa_f1": "pa_f1", "pa_recall": "pa_recall"}``。
    """
    positive_label: int = Field(default=1,
                                description="异常标签值，y_truth 和 y_predict 中等于该值的位置被判定为异常点，默认为 1")
    main_scores: dict[str, str] | None = Field(
        default={"pa_f1": "pa_f1", "pa_recall": "pa_recall"},
        description="主评分路径映射，键为指标名称、值为结果属性路径",
    )


# ============================================================================
# 算子类定义
# ============================================================================

class PointAdjust(
    BaseMetricOperator[
        tuple[np.ndarray, np.ndarray],
        PointAdjustResult,
        PointAdjustConfig,
        None,
    ],
):
    """
    点调整评价指标算子。

    输入为一维离散标签对 ``(y_truth, y_predict)``，输出段级 PA 指标结果。
    核心思想：一段连续异常区间内，只要有一个点被正确检出，则将该区间内所有点
    都视为正确检出（True Positive），从而在段级别衡量检测性能。

    Input:
        y_truth: 真实离散标签，一维数组（正值表示异常点）
        y_predict: 预测离散标签，一维数组，与 y_truth 等长

    Output:
        点调整（Point-Adjust）段级评价指标集（含 PA-TP/PA-FP/PA-FN 计数 + PA-Precision/PA-Recall/PA-F1）。
        可通过 Config 的 ``main_scores`` 配置提取 pa_f1/pa_recall 等命名标量用于 HPO。
        字段结构详见下方的"主输出结构"表格。

    泛型参数:
        I: ``tuple[np.ndarray, np.ndarray]`` -- 输入类型，
            ``(y_truth, y_predict)`` 一维离散标签对。
        MR: :class:`PointAdjustResult` -- 度量结果类型。
        MC: :class:`PointAdjustConfig` -- 配置类型。
        RP: ``None`` -- 运行时参数类型，本算子不需要额外运行参数。
    """

    _run_params_type: ClassVar[type | None] = None

    @classmethod
    def name(cls) -> str:
        return "point_adjust"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def _run(
        self,
        x: tuple[np.ndarray, np.ndarray],
        *,
        params: None,
    ) -> PointAdjustResult:
        """计算点调整（Point-Adjust）评价指标。

        执行流程:
            1. **输入校验** -- 将 y_truth / y_predict 转为 NumPy 数组，并验证
               两者均为一维且长度一致。
            2. **提取异常段** -- 从 y_truth 中找出所有连续 positive_label 区间。
            3. **计算 PA-TP / PA-FN** -- 遍历每个异常段：段内至少一个预测命中则
               整段计为 TP，否则计为 FN。
            4. **计算 PA-FP** -- 在所有异常段之外的索引中，统计预测值为
               positive_label 的点数。
            5. **汇总指标** -- 根据 PA-TP / PA-FP / PA-FN 计算 Precision、
               Recall 及 F1，其中除零情况安全退回 0.0。
            6. **封装返回** -- 将所有统计数据打包为 :class:`PointAdjustResult`。

        Args:
            x (tuple[np.ndarray, np.ndarray]): 二元组 ``(y_truth, y_predict)``，
                其中 y_truth 为真实标签、y_predict 为预测标签，两者均为一维离散
                标签数组（例如 0/1）。
            params (None): 运行时参数，本算子不使用，始终传入 None。

        Returns:
            PointAdjustResult: 包含以下字段的 PA 指标结果：
                - ``n_samples`` -- 总样本数
                - ``n_anomaly_segments`` -- 真实异常段数量
                - ``pa_tp`` / ``pa_fp`` / ``pa_fn`` -- PA 调整后的混淆矩阵计数
                - ``pa_precision`` / ``pa_recall`` / ``pa_f1`` -- PA 派生指标

        Raises:
            ValueError: 当 y_truth 或 y_predict 不是一维数组时抛出，异常消息中
                包含实际维数信息。
            ValueError: 当 y_truth 与 y_predict 长度不一致时抛出，异常消息中
                包含两者实际长度。
        """
        y_truth, y_predict = x

        # ======================================================================
        # 第一步：输入校验
        # ======================================================================

        # 将输入统一转换为 NumPy ndarray，确保后续操作可使用向量化方法
        y_truth = np.asarray(y_truth)
        y_predict = np.asarray(y_predict)

        # 校验维度：两个数组必须都是一维的
        if y_truth.ndim != 1 or y_predict.ndim != 1:
            raise ValueError(
                f"输入必须为一维数组，当前 y_truth.ndim={y_truth.ndim}, "
                f"y_predict.ndim={y_predict.ndim}"
            )

        # 校验长度：两个数组必须具有相同的样本数
        if len(y_truth) != len(y_predict):
            raise ValueError(
                f"y_truth 和 y_predict 长度不一致: {len(y_truth)} vs {len(y_predict)}"
            )

        n_samples = len(y_truth)

        # 获取配置中的异常标签值；若配置缺失则回退到默认值 1
        config = self.config
        positive_label = config.positive_label if config else 1

        # ======================================================================
        # 第二步：从 y_truth 中提取所有连续异常段
        # ======================================================================

        anomaly_segments = self._extract_anomaly_segments(y_truth, positive_label)
        n_anomaly_segments = len(anomaly_segments)

        # ======================================================================
        # 第三步：遍历每个异常段，计算 PA-TP 和 PA-FN
        # ======================================================================
        #
        # PA 调整规则（Point-Adjust）：
        #   - 若异常段 [start, end] 内 y_predict 存在至少一个 positive_label，
        #     则该段内所有点视为 TP（整段被"调整"为正确检出）。
        #   - 若异常段内 y_predict 完全不包含 positive_label，
        #     则该段内所有点视为 FN。

        pa_tp = 0
        pa_fn = 0
        anomaly_segment_indices = set()  # 记录所有属于真实异常段的索引位置

        for start, end in anomaly_segments:
            # 记录当前段的索引集合，供后续计算 PA-FP 时使用
            segment_indices = set(range(start, end + 1))
            anomaly_segment_indices.update(segment_indices)

            # 检查该段内是否有任何点被预测为异常
            if np.any(y_predict[start:end + 1] == positive_label):
                # 整段视为 TP（至少一个点命中）
                pa_tp += 1
            else:
                # 整段视为 FN（没有任何点命中）
                pa_fn += 1

        # ======================================================================
        # 第四步：计算 PA-FP
        # ======================================================================
        #
        # PA-FP 定义：在所有真实异常段之外（即 y_truth != positive_label 的区域），
        # 那些被 y_predict 错误预测为 positive_label 的点。

        non_anomaly_indices = set(range(n_samples)) - anomaly_segment_indices
        pa_fp = int(sum(
            y_predict[i] == positive_label
            for i in non_anomaly_indices
        ))

        # ======================================================================
        # 第五步：汇总派生指标（Precision / Recall / F1）
        # ======================================================================
        #
        # 使用 _safe_divide 进行安全除法，分母为零时统一回退到 default 值 0.0，
        # 避免触发 ZeroDivisionError。

        pa_precision = self._safe_divide(pa_tp, pa_tp + pa_fp, 0.0)
        pa_recall = self._safe_divide(pa_tp, pa_tp + pa_fn, 0.0)
        pa_f1 = self._safe_divide(
            2 * pa_precision * pa_recall,
            pa_precision + pa_recall,
            0.0,
        )

        # ======================================================================
        # 第六步：封装并返回结果
        # ======================================================================

        return PointAdjustResult(
            n_samples=n_samples,
            n_anomaly_segments=n_anomaly_segments,
            pa_tp=pa_tp,
            pa_fp=pa_fp,
            pa_fn=pa_fn,
            pa_precision=pa_precision,
            pa_recall=pa_recall,
            pa_f1=pa_f1,
        )

    def _extract_anomaly_segments(
        self,
        y_truth: np.ndarray,
        positive_label: int,
    ) -> list[tuple[int, int]]:
        """从真实标签中提取所有连续异常段。

        采用单次线性扫描算法遍历 ``y_truth``，将所有值等于
        ``positive_label`` 的连续索引区间提取为 ``[start, end]`` 半开半闭
        区间（两端均包含在内）。

        算法说明:
            维护一个状态标志 ``in_segment`` 表示当前是否正处于一段异常区间中。

            - 当 ``in_segment`` 为 False 且当前值等于 ``positive_label`` 时，
              记录段的起始索引并将 ``in_segment`` 置为 True。
            - 当 ``in_segment`` 为 True 且当前值不等于 ``positive_label`` 时，
              将 ``[start, i - 1]`` 追加到结果列表并将 ``in_segment`` 置为 False。
            - 遍历结束后，若 ``in_segment`` 仍为 True，说明数组末尾存在一段
              未闭合的异常区间，需补入结果。

        示例::

            >>> y = np.array([0, 1, 1, 0, 0, 1, 1, 1, 0])
            >>> _extract_anomaly_segments(y, 1)
            [(1, 2), (5, 7)]

        Args:
            y_truth (np.ndarray): 真实标签，一维离散标签数组，例如 0/1 序列。
            positive_label (int): 异常标签值，等于该值的样本被视为异常点。

        Returns:
            list[tuple[int, int]]: 异常段列表，每个元素为一个二元组
                ``(start_index, end_index)``，表示一段从 start_index 到
                end_index（两端均包含）的连续异常区间。列表按起始索引升序排列。
        """
        segments = []
        in_segment = False  # 当前是否处于一段异常区间内
        start = 0  # 当前异常段的起始索引

        for i, val in enumerate(y_truth):
            if val == positive_label:
                if not in_segment:
                    # 进入新的异常段：记录起始索引，切换状态
                    in_segment = True
                    start = i
            else:
                if in_segment:
                    # 异常段结束：将 [start, i-1] 追加到结果列表，重置状态
                    in_segment = False
                    segments.append((start, i - 1))

        # 兜底处理：若数组末尾仍处于异常段内，需要手动闭合该段
        if in_segment:
            segments.append((start, len(y_truth) - 1))

        return segments

    @staticmethod
    def _safe_divide(
        numerator: float,
        denominator: float,
        default: float,
    ) -> float:
        """安全除法运算，避免除零异常。

        当 ``denominator`` 为零时返回 ``default`` 值，而非抛出
        ``ZeroDivisionError``。本方法在计算 PA-Precision、PA-Recall 和 PA-F1
        时被调用，确保即使没有任何 TP / FP / FN 也不会崩溃。

        Args:
            numerator (float): 被除数（分子）。
            denominator (float): 除数（分母）。当该值为零时，函数将返回
                ``default`` 而非执行除法。
            default (float): 分母为零时的回退返回值，通常为 ``0.0``。

        Returns:
            float: 若 ``denominator`` 不为零，返回 ``numerator / denominator``；
                否则返回 ``default``。
        """
        if denominator == 0:
            return default
        return numerator / denominator
