# -*- coding: utf-8 -*-

"""
分布异常分数合并评分器

在假定多列异常分数中每列服从某分布的前提下，将其合并为单列异常分数，
且合并后的异常分数仍符合标准该分布。

提供两种算子:
    - DistMergeScorer: 可训练版本，fit 时从训练数据学习分布参数 μ/σ，
      run 时使用训练参数进行合并
    - DistDirectMergeScorer: 非训练版本，直接对当前推理数据计算局部参数
      或使用 Config 预设参数进行合并

支持两种先验分布:
    - NORMAL: 正态分布标准化合并，合并后分数服从标准正态分布 N(0,1)
    - LOG_NORMAL: 对数正态分布标准化合并，合并后分数服从标准对数正态分布 LogN(0,1)

核心数学原理:
    - 正态分布: 各列标准化为 N(0,1) → 加权合并 → 再标准化使结果仍为 N(0,1)
      等权时: s = Σz_i / √n, Var(s) = n × 1 / n = 1
      加权时: s = Σ(z_i × w_i) / √(Σw_i²), Var(s) = Σw_i² / Σw_i² = 1
    - 对数正态分布: 取对数 → 正态分布合并 → 取指数回到原空间

示例用法::

    # 可训练版本
    scorer = DistMergeScorer(config=DistMergeScorerConfig(dist=ScoreDistribution.NORMAL))
    scorer.fit(train_scores)          # 从训练分数学习 μ/σ
    scores, eo = scorer.run(test_scores)  # 使用训练参数合并

    # 非训练版本（局部参数）
    scorer = DistDirectMergeScorer()  # 不需 fit
    scores, eo = scorer.run(scores)   # 从当前数据计算局部 μ/σ

    # 非训练版本（预设参数）
    scorer = DistDirectMergeScorer(config=DistDirectMergeScorerConfig(
        dist=ScoreDistribution.NORMAL,
        mus=[0.5, 1.0],
        sigmas=[0.1, 0.2],
    ))
    scores, eo = scorer.run(scores)   # 使用预设参数合并
"""

from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from tsas.engine.operator.base import NumericOperator, UnsupervisedNumericOperatorMixin
from tsas.engine.operator.detection.base import SingleScorerMixin

__all__ = [
    'ScoreDistribution',
    'DistMergeScorerExtraOutput',
    'DistMergeScorerConfig',
    'DistMergeScorer',
    'DistDirectMergeScorerConfig',
    'DistDirectMergeScorer',
]


# ============================================================================
# 枚举定义
# ============================================================================


class ScoreDistribution(str, Enum):
    """分数合并分布策略枚举

    定义分数合并时假设的分布类型。

    Attributes:
        NORMAL: 正态分布标准化合并，合并后分数服从标准正态分布 N(0,1)
        LOG_NORMAL: 对数正态分布标准化合并，合并后分数服从标准对数正态分布 LogN(0,1)
    """
    NORMAL = "normal"
    """正态分布标准化合并"""
    LOG_NORMAL = "log_normal"
    """对数正态分布标准化合并"""


# ============================================================================
# 共用附加输出
# ============================================================================


class DistMergeScorerExtraOutput(BaseModel):
    """分布合并评分器附加输出

    两个算子（DistMergeScorer 和 DistDirectMergeScorer）共用此附加输出类型。

    Attributes:
        mus (np.ndarray): 各列分布参数 μ，形状 (n_scores,)
        sigmas (np.ndarray): 各列分布参数 σ，形状 (n_scores,)
        posterior_weights (np.ndarray): 后验权重矩阵，形状 (n_samples, n_scores)，
            标准化空间中各列对合并分数的贡献比例
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mus: np.ndarray = Field(description="各列分布参数 μ，形状 (n_scores,)")
    sigmas: np.ndarray = Field(description="各列分布参数 σ，形状 (n_scores,)")
    posterior_weights: np.ndarray = Field(
        description="后验权重矩阵 (n_samples, n_scores)，标准化空间中各列对合并分数的贡献比例"
    )


# ============================================================================
# 模块级共享函数
# ============================================================================


def _validate_positive_input(x: np.ndarray) -> None:
    """校验输入数据全为正数

    LOG_NORMAL 模式下需要对输入取对数，因此要求所有元素严格大于 0。
    两个算子在 LOG_NORMAL 模式下均调用此函数。

    Args:
        x (np.ndarray): 输入数据

    Raises:
        ValueError: 当输入包含非正数值时
    """
    if np.any(x <= 0):
        min_val = float(np.min(x))
        raise ValueError(
            f"LOG_NORMAL 模式要求输入数据全为正数，"
            f"但检测到非正数值（最小值: {min_val}）"
        )


def _compute_distribution_params(
    dist: ScoreDistribution,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """从数据计算分布参数 μ 和 σ

    根据先验分布对数据做相应变换后计算各列的均值和标准差:
        - NORMAL: 直接计算 mean 和 std
        - LOG_NORMAL: 对 log(data) 计算 mean 和 std

    对零标准差的特征做保护处理（σ=0 替换为 1.0），使常数列的标准化分数为 0。

    Args:
        dist (ScoreDistribution): 先验分布
        data (np.ndarray): 输入数据，形状 (n_samples, n_scores)

    Returns:
        tuple[np.ndarray, np.ndarray]: (mus, sigmas)，各列的 μ 和 σ，形状均为 (n_scores,)
    """
    # LOG_NORMAL 模式下先取对数再计算分布参数
    if dist == ScoreDistribution.LOG_NORMAL:
        data = np.log(data)
    # 沿样本轴计算各列均值和标准差
    mus = np.mean(data, axis=0)
    sigmas = np.std(data, axis=0)
    # 零标准差保护: 常数列的标准化分数应为 0，将 σ=0 替换为 1 使 (x-μ)/σ = 0
    sigmas[sigmas == 0] = 1.0
    return mus, sigmas


def _merge_scores(
    dist: ScoreDistribution,
    scores: np.ndarray,
    mus: np.ndarray,
    sigmas: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """核心合并逻辑 — 两个算子共用

    执行流程:
        1. 变换到正态空间（LOG_NORMAL 时取 log）
        2. 标准化为标准正态 N(0,1)
        3. 加权合并
        4. 再标准化（使结果服从标准正态 N(0,1)）
        5. LOG_NORMAL 时 exp 回原空间
        6. 计算后验权重矩阵（标准化空间）

    Args:
        dist (ScoreDistribution): 先验分布
        scores (np.ndarray): 输入分数，形状 (n_samples, n_scores)
        mus (np.ndarray): 各列分布参数 μ，形状 (n_scores,)
        sigmas (np.ndarray): 各列分布参数 σ，形状 (n_scores,)
        weights (np.ndarray | None): 权重向量，形状 (n_scores,)，None 时等权

    Returns:
        tuple[np.ndarray, np.ndarray]: (merged_scores, posterior_weights)
            - merged_scores: 合并后分数，形状 (n_samples,)
            - posterior_weights: 后验权重矩阵，形状 (n_samples, n_scores)，
              标准化空间中各列对合并分数的贡献比例
    """
    # 步骤1: 变换到正态空间
    if dist == ScoreDistribution.LOG_NORMAL:
        normal_scores = np.log(scores)
    else:
        normal_scores = scores

    # 步骤2: 标准化为标准正态 N(0,1)
    std_scores = (normal_scores - mus) / sigmas  # (n_samples, n_scores)

    # 步骤3: 加权合并
    if weights is not None:
        # 归一化权重
        normalized_weights = weights / np.sum(weights)
        # 加权标准化分数
        weighted_scores = std_scores * normalized_weights  # (n_samples, n_scores)
        # 合并分数（再标准化前）
        merged = np.sum(weighted_scores, axis=1)  # (n_samples,)
        # 步骤4: 再标准化，使结果服从 N(0,1)
        # Var = Σ(w_i² × 1) / (Σw_i²) = 1，因此除以 √(Σw_i²)
        scale = np.sqrt(np.sum(normalized_weights ** 2))
    else:
        # 等权合并
        weighted_scores = std_scores
        merged = np.sum(std_scores, axis=1)
        # 等权再标准化: Var = n × 1 / n² = 1/n，除以 1/√n = 乘以 √n
        scale = np.sqrt(scores.shape[1])

    final_score = merged / scale  # 服从 N(0,1)

    # 步骤5: LOG_NORMAL 时 exp 回原空间 → 服从 LogN(0,1)
    if dist == ScoreDistribution.LOG_NORMAL:
        final_score = np.exp(final_score)

    # 步骤6: 后验权重矩阵（标准化空间，基于再标准化前的 merged）
    # posterior_weights[i, j] = weighted_scores[i, j] / merged[i]
    # 除零保护: merged 为 0 时权重设为 0
    safe_merged = np.where(merged == 0, 1.0, merged)
    posterior_weights = weighted_scores / safe_merged[:, np.newaxis]
    posterior_weights[merged == 0] = 0.0

    return final_score.ravel(), posterior_weights


# ============================================================================
# 可训练版本: DistMergeScorer
# ============================================================================


class DistMergeScorerConfig(BaseModel):
    """可训练分布合并评分器配置

    Attributes:
        dist (ScoreDistribution): 先验分布，默认 NORMAL-正态分布
        weights (list[float] | None): 各列权重向量，None 时等权合并
    """
    model_config = ConfigDict(frozen=True)

    dist: ScoreDistribution = Field(
        default=ScoreDistribution.NORMAL,
        description="先验分布: 'normal' 正态分布, 'log_normal' 对数正态分布",
    )
    weights: list[float] | None = Field(
        default=None,
        description="各列权重向量，None 时等权合并；非 None 时长度须与输入列数一致",
    )


class DistMergeScorer(SingleScorerMixin[None],
                      UnsupervisedNumericOperatorMixin[None],
                      NumericOperator[DistMergeScorerExtraOutput, DistMergeScorerConfig, None]):
    """可训练分布合并评分器

    在假定多列异常分数中每列服从某分布的前提下，将其合并为单列异常分数，
    且合并后的异常分数仍符合标准该分布。

    训练阶段从训练数据学习各列的分布参数 μ/σ，
    推理阶段使用训练得到的参数进行标准化和合并。

    核心逻辑:
        - ``_fit_data``: 调用 ``_compute_distribution_params`` 从训练数据学习 μ/σ
        - ``_run_data``: 调用 ``_merge_scores`` 使用训练参数进行合并

    Input:
        x: 多列异常分数矩阵，形状 (n_samples, n_scores)，每列为一个评分器的分数

    Output:
        合并后异常分数，形状 (n_samples,)，值越大越异常。
        NORMAL 模式下服从标准正态分布 N(0,1)，
        LOG_NORMAL 模式下服从标准对数正态分布 LogN(0,1)

    泛型参数:
        - EO: DistMergeScorerExtraOutput（附加输出含 μ/σ 和后验权重矩阵）
        - C: DistMergeScorerConfig
        - RP: None（无运行参数）
    """

    # 训练状态文件名
    _LEARNED_STATE_FILE = '_learned_state.npz'

    @classmethod
    def name(cls) -> str:
        """返回算子名称

        Returns:
            str: ``"dist_merge_scorer"``
        """
        return "dist_merge_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def __init__(self, *, oid: str | None = None,
                 config: DistMergeScorerConfig | None = None, **kwargs):
        """初始化可训练分布合并评分器

        Args:
            oid (str | None): 算子实例唯一标识后缀，缺省自动生成
            config (DistMergeScorerConfig | None): 类型化实例参数
            **kwargs: 透传给基类的参数，支持 dist、weights 等键值对
        """
        super().__init__(oid=oid, config=config, **kwargs)
        self._mus: np.ndarray | None = None
        """训练阶段学习到的各列 μ"""
        self._sigmas: np.ndarray | None = None
        """训练阶段学习到的各列 σ"""

    def _validate_ndarray_input(self, x: np.ndarray, params: None) -> None:
        """校验 ndarray 输入

        LOG_NORMAL 模式下额外检查输入全为正数。

        Args:
            x (np.ndarray): 输入 ndarray
            params (None): 无运行参数

        Raises:
            ValueError: LOG_NORMAL 模式下输入包含非正数值时
        """
        if self.config.dist == ScoreDistribution.LOG_NORMAL:
            _validate_positive_input(x)

    def _fit_data(self, x: np.ndarray, params: None) -> None:
        """训练阶段：从训练数据学习分布参数 μ/σ

        根据先验分布对训练数据做相应变换后计算各列的均值和标准差。

        Args:
            x (np.ndarray): 训练数据（多列异常分数），形状 (n_samples, n_scores)
            params (None): 无训练参数

        Raises:
            ValueError: LOG_NORMAL 模式下训练数据包含非正数值时
        """
        # LOG_NORMAL 模式下校验正数（fit 管线不调用 _validate_ndarray_input）
        if self.config.dist == ScoreDistribution.LOG_NORMAL:
            _validate_positive_input(x)
        # 从训练数据学习分布参数
        self._mus, self._sigmas = _compute_distribution_params(self.config.dist, x)

    def _run_data(self, x: np.ndarray, params: None,
                  idx: pd.Index | None = None) -> tuple[np.ndarray, DistMergeScorerExtraOutput]:
        """推理阶段：使用训练参数进行分布合并

        Args:
            x (np.ndarray): 输入数据（多列异常分数），形状 (n_samples, n_scores)
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, DistMergeScorerExtraOutput]:
                - merged_scores: 合并后异常分数，形状 (n_samples,)
                - eo: 附加输出，含 μ/σ 和后验权重矩阵
        """
        # 解析权重
        weights = np.array(self.config.weights) if self.config.weights else None
        # 调用核心合并逻辑
        merged_scores, posterior_weights = _merge_scores(
            self.config.dist, x, self._mus, self._sigmas, weights
        )
        # 构造附加输出
        eo = DistMergeScorerExtraOutput(
            mus=self._mus,
            sigmas=self._sigmas,
            posterior_weights=posterior_weights,
        )
        return merged_scores, eo

    def _save_fit_state(self, path: Path) -> None:
        """保存训练状态：训练参数 + 学习到的分布参数

        在基类保存 last_fit_params 的基础上，额外将 _mus 和 _sigmas 持久化到 npz 文件。

        Args:
            path (Path): 目标目录路径
        """
        super()._save_fit_state(path)
        np.savez(path / self._LEARNED_STATE_FILE, mus=self._mus, sigmas=self._sigmas)

    def _load_fit_state(self, path: Path) -> None:
        """恢复训练状态：训练参数 + 学习到的分布参数

        从 npz 文件恢复 _mus 和 _sigmas，并将 _fitted 标记为 True。

        Args:
            path (Path): 源目录路径
        """
        super()._load_fit_state(path)
        data = np.load(path / self._LEARNED_STATE_FILE)
        self._mus = data['mus']
        self._sigmas = data['sigmas']
        # 恢复训练完成标记
        self._fitted = True


# ============================================================================
# 非训练版本: DistDirectMergeScorer
# ============================================================================


class DistDirectMergeScorerConfig(BaseModel):
    """非训练分布合并评分器配置

    与 DistMergeScorerConfig 相比，额外支持预设分布参数 mus/sigmas。
    当 mus/sigmas 均为 None 时，从当前推理数据计算局部参数。

    Attributes:
        dist (ScoreDistribution): 先验分布，默认 NORMAL-正态分布
        weights (list[float] | None): 各列权重向量，None 时等权合并
        mus (list[float] | None): 预设各列 μ，None 时从当前数据计算
        sigmas (list[float] | None): 预设各列 σ，None 时从当前数据计算
    """
    model_config = ConfigDict(frozen=True)

    dist: ScoreDistribution = Field(
        default=ScoreDistribution.NORMAL,
        description="先验分布: 'normal' 正态分布, 'log_normal' 对数正态分布",
    )
    weights: list[float] | None = Field(
        default=None,
        description="各列权重向量，None 时等权合并",
    )
    mus: list[float] | None = Field(
        default=None,
        description="预设各列 μ，None 时从当前数据计算",
    )
    sigmas: list[float] | None = Field(
        default=None,
        description="预设各列 σ，None 时从当前数据计算",
    )


class DistDirectMergeScorer(SingleScorerMixin[None],
                            NumericOperator[DistMergeScorerExtraOutput, DistDirectMergeScorerConfig, None]):
    """非训练分布合并评分器

    在假定多列异常分数中每列服从某分布的前提下，将其合并为单列异常分数，
    且合并后的异常分数仍符合标准该分布。

    无需训练，直接在推理阶段确定分布参数:
        - Config 中 mus/sigmas 均非 None → 使用预设参数
        - 否则 → 从当前推理数据计算局部参数

    核心逻辑:
        - ``_run_data``: 确定参数来源 → 调用 ``_merge_scores`` 进行合并

    Input:
        x: 多列异常分数矩阵，形状 (n_samples, n_scores)，每列为一个评分器的分数

    Output:
        合并后异常分数，形状 (n_samples,)，值越大越异常。
        NORMAL 模式下服从标准正态分布 N(0,1)，
        LOG_NORMAL 模式下服从标准对数正态分布 LogN(0,1)

    泛型参数:
        - EO: DistMergeScorerExtraOutput（附加输出含 μ/σ 和后验权重矩阵）
        - C: DistDirectMergeScorerConfig
        - RP: None（无运行参数）
    """

    @classmethod
    def name(cls) -> str:
        """返回算子名称

        Returns:
            str: ``"dist_direct_merge_scorer"``
        """
        return "dist_direct_merge_scorer"

    @classmethod
    def version(cls) -> tuple[int, ...]:
        """返回算子版本号

        Returns:
            tuple[int, ...]: 版本号三元组 ``(1, 0, 0)``
        """
        return (1, 0, 0)

    def _validate_ndarray_input(self, x: np.ndarray, params: None) -> None:
        """校验 ndarray 输入

        LOG_NORMAL 模式下额外检查输入全为正数。

        Args:
            x (np.ndarray): 输入 ndarray
            params (None): 无运行参数

        Raises:
            ValueError: LOG_NORMAL 模式下输入包含非正数值时
        """
        if self.config.dist == ScoreDistribution.LOG_NORMAL:
            _validate_positive_input(x)

    def _run_data(self, x: np.ndarray, params: None,
                  idx: pd.Index | None = None) -> tuple[np.ndarray, DistMergeScorerExtraOutput]:
        """推理阶段：确定参数来源并进行分布合并

        参数来源优先级:
            1. Config 中 mus 和 sigmas 均非 None → 使用预设参数
            2. 否则 → 从当前输入数据计算局部参数

        Args:
            x (np.ndarray): 输入数据（多列异常分数），形状 (n_samples, n_scores)
            params (None): 无运行参数
            idx (pd.Index | None): 输入数据的行索引

        Returns:
            tuple[np.ndarray, DistMergeScorerExtraOutput]:
                - merged_scores: 合并后异常分数，形状 (n_samples,)
                - eo: 附加输出，含 μ/σ 和后验权重矩阵
        """
        # 确定分布参数来源
        if self.config.mus is not None and self.config.sigmas is not None:
            # 使用预设参数
            mus = np.array(self.config.mus, dtype=float)
            sigmas = np.array(self.config.sigmas, dtype=float)
            # σ=0 保护
            sigmas[sigmas == 0] = 1.0
        else:
            # 从当前输入数据计算局部参数
            mus, sigmas = _compute_distribution_params(self.config.dist, x)

        # 解析权重
        weights = np.array(self.config.weights) if self.config.weights else None
        # 调用核心合并逻辑
        merged_scores, posterior_weights = _merge_scores(
            self.config.dist, x, mus, sigmas, weights
        )
        # 构造附加输出
        eo = DistMergeScorerExtraOutput(
            mus=mus,
            sigmas=sigmas,
            posterior_weights=posterior_weights,
        )
        return merged_scores, eo
