# -*- coding: utf-8 -*-

"""
分布合并评分器单元测试

对应源文件:
- dist_merge_scorer.py: DistMergeScorer、DistDirectMergeScorer、ScoreDistribution

测试范围:
- Config 参数验证（两个算子）
- 模块级共享函数（_validate_positive_input、_compute_distribution_params、_merge_scores）
- DistMergeScorer fit/run/save/load 全流程
- DistDirectMergeScorer run 流程（局部参数 + 预设参数）
- DataFrame/ndarray 双类型支持
- NORMAL / LOG_NORMAL 两种分布策略
- 等权 / 加权两种合并方式
- 后验权重矩阵正确性
- 分布保持性验证（合并后分数服从标准分布）
- 边界条件（未训练先推理、LOG_NORMAL 非正数输入等）
- 两个算子一致性对比（相同参数下输出相同）
"""

import numpy as np
import pytest
from pandas import DataFrame
from pydantic import ValidationError

from tsas.engine.operator.detection.dist_merge_scorer import (_compute_distribution_params, _merge_scores,
                                                              _validate_positive_input, DistDirectMergeScorer,
                                                              DistDirectMergeScorerConfig, DistMergeScorer,
                                                              DistMergeScorerConfig, DistMergeScorerExtraOutput,
                                                              ScoreDistribution)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_scores():
    """训练用多列异常分数（ndarray, 500x3, 正态分布随机数）"""
    np.random.seed(42)
    # 三列分数，分别来自不同均值/标准差的正态分布
    col1 = np.random.randn(500) * 1.0 + 0.0
    col2 = np.random.randn(500) * 2.0 + 1.0
    col3 = np.random.randn(500) * 0.5 + 0.5
    return np.column_stack([col1, col2, col3])


@pytest.fixture
def test_scores():
    """测试用多列异常分数（ndarray, 200x3, 含异常点）"""
    np.random.seed(123)
    normal = np.column_stack([
        np.random.randn(180) * 1.0,
        np.random.randn(180) * 2.0 + 1.0,
        np.random.randn(180) * 0.5 + 0.5,
    ])
    abnormal = np.column_stack([
        np.random.randn(20) * 3.0 + 5.0,
        np.random.randn(20) * 4.0 + 6.0,
        np.random.randn(20) * 2.0 + 3.0,
    ])
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_df(train_scores):
    """训练用多列异常分数（DataFrame）"""
    return DataFrame(train_scores, columns=["score_a", "score_b", "score_c"])


@pytest.fixture
def test_df(test_scores):
    """测试用多列异常分数（DataFrame）"""
    return DataFrame(test_scores, columns=["score_a", "score_b", "score_c"])


@pytest.fixture
def train_log_scores():
    """训练用多列异常分数（对数正态分布, 500x3, 全正数）"""
    np.random.seed(42)
    # 对数正态分布: exp(N(μ, σ²))
    col1 = np.exp(np.random.randn(500) * 0.5 + 0.0)
    col2 = np.exp(np.random.randn(500) * 0.8 + 1.0)
    col3 = np.exp(np.random.randn(500) * 0.3 + 0.5)
    return np.column_stack([col1, col2, col3])


@pytest.fixture
def test_log_scores():
    """测试用多列异常分数（对数正态分布, 200x3, 全正数, 含异常点）"""
    np.random.seed(123)
    normal = np.column_stack([
        np.exp(np.random.randn(180) * 0.5),
        np.exp(np.random.randn(180) * 0.8 + 1.0),
        np.exp(np.random.randn(180) * 0.3 + 0.5),
    ])
    abnormal = np.column_stack([
        np.exp(np.random.randn(20) * 1.5 + 3.0),
        np.exp(np.random.randn(20) * 2.0 + 4.0),
        np.exp(np.random.randn(20) * 1.0 + 2.0),
    ])
    return np.vstack([normal, abnormal])


# ============================================================================
# 模块级共享函数测试
# ============================================================================

class TestValidatePositiveInput:
    """测试 _validate_positive_input 函数"""

    def test_valid_positive_input(self):
        """
        目的：验证全正数输入通过校验
        输入：全正数 ndarray
        预期：不抛出异常
        """
        x = np.array([[1.0, 2.0], [0.1, 0.5]])
        _validate_positive_input(x)  # 不抛异常即通过

    def test_zero_value_raises(self):
        """
        目的：验证包含零值时抛出 ValueError
        输入：包含 0 的 ndarray
        预期：ValueError
        """
        x = np.array([[1.0, 0.0], [0.1, 0.5]])
        with pytest.raises(ValueError, match="正数"):
            _validate_positive_input(x)

    def test_negative_value_raises(self):
        """
        目的：验证包含负值时抛出 ValueError
        输入：包含负数的 ndarray
        预期：ValueError
        """
        x = np.array([[1.0, -0.1], [0.1, 0.5]])
        with pytest.raises(ValueError, match="正数"):
            _validate_positive_input(x)


class TestComputeDistributionParams:
    """测试 _compute_distribution_params 函数"""

    def test_normal_mode(self):
        """
        目的：验证 NORMAL 模式下直接计算 μ/σ
        输入：(100, 2) 正态分布数据
        预期：mus ≈ mean(axis=0), sigmas ≈ std(axis=0)
        """
        np.random.seed(42)
        data = np.random.randn(100, 2) * 2.0 + 1.0
        mus, sigmas = _compute_distribution_params(ScoreDistribution.NORMAL, data)
        np.testing.assert_allclose(mus, np.mean(data, axis=0), atol=1e-10)
        np.testing.assert_allclose(sigmas, np.std(data, axis=0), atol=1e-10)

    def test_log_normal_mode(self):
        """
        目的：验证 LOG_NORMAL 模式下对 log(data) 计算 μ/σ
        输入：(100, 2) 对数正态分布数据
        预期：mus ≈ mean(log(data), axis=0), sigmas ≈ std(log(data), axis=0)
        """
        np.random.seed(42)
        data = np.exp(np.random.randn(100, 2) * 0.5 + 1.0)
        mus, sigmas = _compute_distribution_params(ScoreDistribution.LOG_NORMAL, data)
        log_data = np.log(data)
        np.testing.assert_allclose(mus, np.mean(log_data, axis=0), atol=1e-10)
        np.testing.assert_allclose(sigmas, np.std(log_data, axis=0), atol=1e-10)

    def test_zero_sigma_protection(self):
        """
        目的：验证 σ=0 保护机制
        输入：第二列全为常数（σ=0）
        预期：sigmas[1] 被替换为 1.0
        """
        data = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        mus, sigmas = _compute_distribution_params(ScoreDistribution.NORMAL, data)
        assert sigmas[0] > 0  # 第一列有变化
        assert sigmas[1] == 1.0  # 第二列常数列被保护为 1.0


class TestMergeScores:
    """测试 _merge_scores 核心合并函数"""

    def test_normal_equal_weight_mean_zero(self):
        """
        目的：验证 NORMAL 等权合并后均值接近 0（标准正态 N(0,1)）
        输入：(1000, 3) 正态分布数据，使用数据自身 μ/σ
        预期：合并分数均值 ≈ 0
        """
        np.random.seed(42)
        data = np.random.randn(1000, 3)
        mus, sigmas = _compute_distribution_params(ScoreDistribution.NORMAL, data)
        merged, _ = _merge_scores(ScoreDistribution.NORMAL, data, mus, sigmas, None)
        assert abs(np.mean(merged)) < 0.1  # 均值接近 0

    def test_normal_equal_weight_std_one(self):
        """
        目的：验证 NORMAL 等权合并后标准差接近 1（标准正态 N(0,1)）
        输入：(1000, 3) 正态分布数据
        预期：合并分数标准差 ≈ 1
        """
        np.random.seed(42)
        data = np.random.randn(1000, 3)
        mus, sigmas = _compute_distribution_params(ScoreDistribution.NORMAL, data)
        merged, _ = _merge_scores(ScoreDistribution.NORMAL, data, mus, sigmas, None)
        assert abs(np.std(merged) - 1.0) < 0.1  # 标准差接近 1

    def test_normal_weighted_mean_zero(self):
        """
        目的：验证 NORMAL 加权合并后均值接近 0
        输入：(1000, 3) 正态分布数据，权重 [0.2, 0.5, 0.3]
        预期：合并分数均值 ≈ 0
        """
        np.random.seed(42)
        data = np.random.randn(1000, 3)
        mus, sigmas = _compute_distribution_params(ScoreDistribution.NORMAL, data)
        weights = np.array([0.2, 0.5, 0.3])
        merged, _ = _merge_scores(ScoreDistribution.NORMAL, data, mus, sigmas, weights)
        assert abs(np.mean(merged)) < 0.1

    def test_normal_weighted_std_one(self):
        """
        目的：验证 NORMAL 加权合并后标准差接近 1
        输入：(1000, 3) 正态分布数据，权重 [0.2, 0.5, 0.3]
        预期：合并分数标准差 ≈ 1
        """
        np.random.seed(42)
        data = np.random.randn(1000, 3)
        mus, sigmas = _compute_distribution_params(ScoreDistribution.NORMAL, data)
        weights = np.array([0.2, 0.5, 0.3])
        merged, _ = _merge_scores(ScoreDistribution.NORMAL, data, mus, sigmas, weights)
        assert abs(np.std(merged) - 1.0) < 0.1

    def test_log_normal_equal_weight_positive(self):
        """
        目的：验证 LOG_NORMAL 等权合并后全为正数（LogN(0,1) 的值域 > 0）
        输入：(1000, 3) 对数正态分布数据
        预期：合并分数全为正数
        """
        np.random.seed(42)
        data = np.exp(np.random.randn(1000, 3) * 0.5)
        mus, sigmas = _compute_distribution_params(ScoreDistribution.LOG_NORMAL, data)
        merged, _ = _merge_scores(ScoreDistribution.LOG_NORMAL, data, mus, sigmas, None)
        assert np.all(merged > 0)

    def test_log_normal_equal_weight_median_one(self):
        """
        目的：验证 LOG_NORMAL 等权合并后中位数接近 1
        输入：(1000, 3) 对数正态分布数据
        预期：合并分数中位数 ≈ 1（exp(N(0,1)) 的中位数为 exp(0)=1）
        """
        np.random.seed(42)
        data = np.exp(np.random.randn(1000, 3) * 0.5)
        mus, sigmas = _compute_distribution_params(ScoreDistribution.LOG_NORMAL, data)
        merged, _ = _merge_scores(ScoreDistribution.LOG_NORMAL, data, mus, sigmas, None)
        assert abs(np.median(merged) - 1.0) < 0.15

    def test_posterior_weights_shape(self):
        """
        目的：验证后验权重矩阵形状正确
        输入：(50, 4) 数据
        预期：后验权重形状 (50, 4)
        """
        np.random.seed(42)
        data = np.random.randn(50, 4)
        mus, sigmas = _compute_distribution_params(ScoreDistribution.NORMAL, data)
        _, posterior_weights = _merge_scores(ScoreDistribution.NORMAL, data, mus, sigmas, None)
        assert posterior_weights.shape == (50, 4)

    def test_posterior_weights_sum_approx_one(self):
        """
        目的：验证后验权重矩阵每行求和接近 1（贡献比例）
        输入：(100, 3) 正态分布数据
        预期：每行 posterior_weights 求和 ≈ 1.0
        """
        np.random.seed(42)
        data = np.random.randn(100, 3)
        mus, sigmas = _compute_distribution_params(ScoreDistribution.NORMAL, data)
        _, posterior_weights = _merge_scores(ScoreDistribution.NORMAL, data, mus, sigmas, None)
        row_sums = np.sum(posterior_weights, axis=1)
        # 排除 merged=0 的行（其权重和为 0）
        nonzero_mask = row_sums != 0
        np.testing.assert_allclose(row_sums[nonzero_mask], 1.0, atol=1e-8)

    def test_posterior_weights_with_custom_weights(self):
        """
        目的：验证加权模式下后验权重矩阵正确
        输入：(100, 3) 数据，权重 [0.5, 0.3, 0.2]
        预期：后验权重形状 (100, 3)，每行和 ≈ 1
        """
        np.random.seed(42)
        data = np.random.randn(100, 3)
        mus, sigmas = _compute_distribution_params(ScoreDistribution.NORMAL, data)
        weights = np.array([0.5, 0.3, 0.2])
        _, posterior_weights = _merge_scores(ScoreDistribution.NORMAL, data, mus, sigmas, weights)
        assert posterior_weights.shape == (100, 3)
        row_sums = np.sum(posterior_weights, axis=1)
        nonzero_mask = row_sums != 0
        np.testing.assert_allclose(row_sums[nonzero_mask], 1.0, atol=1e-8)

    def test_merged_zero_protection(self):
        """
        目的：验证合并分数为 0 时后验权重设为 0（除零保护）
        输入：构造特殊数据使某行标准化分数恰好抵消（merged=0）
        预期：该行后验权重全为 0
        """
        # 构造两列完全对称的数据，使标准化后恰好抵消
        # mus = [0, 0], sigmas = [1, 1]，data = [1, -1] → std = [1, -1], merged = 0
        data = np.array([[1.0, -1.0]])
        mus = np.array([0.0, 0.0])
        sigmas = np.array([1.0, 1.0])
        _, posterior_weights = _merge_scores(ScoreDistribution.NORMAL, data, mus, sigmas, None)
        assert np.all(posterior_weights[0] == 0.0)


# ============================================================================
# DistMergeScorer Config 测试
# ============================================================================

class TestDistMergeScorerConfig:
    """测试 DistMergeScorerConfig 参数验证"""

    def test_config_defaults(self):
        """
        目的：验证 Config 默认值
        输入：无参数构造
        预期：dist=NORMAL, weights=None
        """
        cfg = DistMergeScorerConfig()
        assert cfg.dist == ScoreDistribution.NORMAL
        assert cfg.weights is None

    def test_config_frozen(self):
        """
        目的：验证 Config 不可变
        输入：创建后尝试修改字段
        预期：抛出 ValidationError
        """
        cfg = DistMergeScorerConfig()
        with pytest.raises(ValidationError):
            cfg.dist = ScoreDistribution.LOG_NORMAL

    def test_config_dist_validation(self):
        """
        目的：验证 dist 枚举约束
        输入：dist="invalid"
        预期：ValidationError
        """
        with pytest.raises(ValidationError):
            DistMergeScorerConfig(dist="invalid")

    def test_config_custom_values(self):
        """
        目的：验证自定义参数正确传递
        输入：dist=LOG_NORMAL, weights=[0.3, 0.7]
        预期：Config 中值为自定义值
        """
        cfg = DistMergeScorerConfig(
            dist=ScoreDistribution.LOG_NORMAL,
            weights=[0.3, 0.7],
        )
        assert cfg.dist == ScoreDistribution.LOG_NORMAL
        assert cfg.weights == [0.3, 0.7]


# ============================================================================
# DistMergeScorer fit/run 测试
# ============================================================================

class TestDistMergeScorerFit:
    """测试 DistMergeScorer 训练流程"""

    def test_fit_learns_params(self, train_scores):
        """
        目的：验证 fit 后学习了分布参数
        输入：(500, 3) 训练分数
        预期：_mus 和 _sigmas 非 None，is_fitted 为 True
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)
        assert scorer._mus is not None
        assert scorer._sigmas is not None
        assert scorer.is_fitted is True

    def test_fit_normal_params_correctness(self, train_scores):
        """
        目的：验证 NORMAL 模式下学习的参数等于训练数据的 mean/std
        输入：(500, 3) 训练分数
        预期：_mus ≈ mean(train_scores, axis=0), _sigmas ≈ std(train_scores, axis=0)
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)
        np.testing.assert_allclose(scorer._mus, np.mean(train_scores, axis=0), atol=1e-10)
        np.testing.assert_allclose(scorer._sigmas, np.std(train_scores, axis=0), atol=1e-10)

    def test_fit_log_normal_params_correctness(self, train_log_scores):
        """
        目的：验证 LOG_NORMAL 模式下学习的参数等于 log(data) 的 mean/std
        输入：(500, 3) 对数正态训练分数
        预期：_mus ≈ mean(log(data), axis=0)
        """
        scorer = DistMergeScorer(dist=ScoreDistribution.LOG_NORMAL)
        scorer.fit(train_log_scores)
        log_data = np.log(train_log_scores)
        np.testing.assert_allclose(scorer._mus, np.mean(log_data, axis=0), atol=1e-10)
        np.testing.assert_allclose(scorer._sigmas, np.std(log_data, axis=0), atol=1e-10)

    def test_fit_log_normal_negative_raises(self):
        """
        目的：验证 LOG_NORMAL 模式下训练数据含非正数时报错
        输入：包含负数的数据
        预期：ValueError
        """
        scorer = DistMergeScorer(dist=ScoreDistribution.LOG_NORMAL)
        bad_data = np.array([[1.0, 2.0], [-1.0, 3.0]])
        with pytest.raises(ValueError, match="正数"):
            scorer.fit(bad_data)

    def test_fit_with_dataframe(self, train_df):
        """
        目的：验证 DataFrame 输入可以训练
        输入：DataFrame (500, 3)
        预期：训练成功，is_fitted 为 True
        """
        scorer = DistMergeScorer()
        scorer.fit(train_df)
        assert scorer.is_fitted is True


class TestDistMergeScorerRun:
    """测试 DistMergeScorer 推理流程"""

    def test_run_output_shape_and_eo_type(self, train_scores, test_scores):
        """
        目的：验证推理输出形状和 EO 类型
        输入：训练 + (200, 3) 测试数据
        预期：scores 形状 (200,)，eo 为 DistMergeScorerExtraOutput
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)
        scores, eo = scorer.run(test_scores)
        assert scores.shape == (200,)
        assert isinstance(eo, DistMergeScorerExtraOutput)

    def test_run_normal_distribution_properties(self, train_scores, test_scores):
        """
        目的：验证 NORMAL 等权合并后正常数据分数近似服从 N(0,1)
        输入：训练 + 测试数据中正常部分（前 180 行）
        预期：合并分数均值 ≈ 0，标准差 ≈ 1
        说明：使用纯正常数据验证分布保持性，异常点会偏移分布
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)
        # 仅使用正常部分验证分布属性
        normal_test = test_scores[:180]
        scores, _ = scorer.run(normal_test)
        assert abs(np.mean(scores)) < 0.2
        assert abs(np.std(scores) - 1.0) < 0.2

    def test_run_normal_weighted_distribution_properties(self, train_scores, test_scores):
        """
        目的：验证 NORMAL 加权合并后正常数据分数近似服从 N(0,1)
        输入：训练 + 测试数据中正常部分（前 180 行），权重 [0.2, 0.3, 0.5]
        预期：合并分数均值 ≈ 0，标准差 ≈ 1
        说明：使用纯正常数据验证分布保持性
        """
        scorer = DistMergeScorer(weights=[0.2, 0.3, 0.5])
        scorer.fit(train_scores)
        # 仅使用正常部分验证分布属性
        normal_test = test_scores[:180]
        scores, _ = scorer.run(normal_test)
        assert abs(np.mean(scores)) < 0.2
        assert abs(np.std(scores) - 1.0) < 0.2

    def test_run_log_normal_positive(self, train_log_scores, test_log_scores):
        """
        目的：验证 LOG_NORMAL 合并后分数全为正数
        输入：对数正态训练 + 测试数据
        预期：所有分数 > 0
        """
        scorer = DistMergeScorer(dist=ScoreDistribution.LOG_NORMAL)
        scorer.fit(train_log_scores)
        scores, _ = scorer.run(test_log_scores)
        assert np.all(scores > 0)

    def test_run_log_normal_median_near_one(self, train_log_scores, test_log_scores):
        """
        目的：验证 LOG_NORMAL 合并后中位数接近 1
        输入：对数正态训练 + 测试数据
        预期：中位数 ≈ 1（exp(N(0,1)) 的中位数为 exp(0) = 1）
        """
        scorer = DistMergeScorer(dist=ScoreDistribution.LOG_NORMAL)
        scorer.fit(train_log_scores)
        scores, _ = scorer.run(test_log_scores)
        assert abs(np.median(scores) - 1.0) < 0.3

    def test_run_before_fit_raises(self, test_scores):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        输入：未训练的 scorer
        预期：RuntimeError
        """
        scorer = DistMergeScorer()
        with pytest.raises(RuntimeError, match="训练尚未完成"):
            scorer.run(test_scores)

    def test_run_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出
        输入：DataFrame 训练 + DataFrame 测试
        预期：输出为 DataFrame，列名为 ["score"]，索引与输入一致
        """
        scorer = DistMergeScorer()
        scorer.fit(train_df)
        scores, _ = scorer.run(test_df)
        assert isinstance(scores, DataFrame)
        assert list(scores.columns) == ["score"]
        assert len(scores) == len(test_df)

    def test_run_with_ndarray(self, train_scores, test_scores):
        """
        目的：验证 ndarray 输入返回 ndarray
        输入：ndarray 训练 + ndarray 测试
        预期：scores 为 ndarray 且为 1D
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)
        scores, _ = scorer.run(test_scores)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1

    def test_run_log_normal_negative_input_raises(self, train_log_scores):
        """
        目的：验证 LOG_NORMAL 模式下推理数据含非正数时报错
        输入：包含负数的测试数据
        预期：ValueError
        """
        scorer = DistMergeScorer(dist=ScoreDistribution.LOG_NORMAL)
        scorer.fit(train_log_scores)
        bad_data = np.array([[1.0, 2.0, 3.0], [-1.0, 2.0, 3.0]])
        with pytest.raises(ValueError, match="正数"):
            scorer.run(bad_data)

    def test_run_eo_fields(self, train_scores, test_scores):
        """
        目的：验证 EO 附加输出字段正确
        输入：训练 + 测试数据
        预期：mus/sigmas 形状 (3,)，posterior_weights 形状 (200, 3)
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)
        _, eo = scorer.run(test_scores)
        assert eo.mus.shape == (3,)
        assert eo.sigmas.shape == (3,)
        assert eo.posterior_weights.shape == (200, 3)

    def test_run_anomalous_higher_scores(self, train_scores, test_scores):
        """
        目的：验证异常点合并分数高于正常点
        输入：test_scores 前 180 行为正常，后 20 行为异常
        预期：异常区平均分数高于正常区（NORMAL 模式下异常点偏离更远）
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)
        scores, _ = scorer.run(test_scores)
        # NORMAL 模式下异常点的 |z-score| 更大，合并后分数绝对值更大
        normal_avg_abs = np.mean(np.abs(scores[:180]))
        abnormal_avg_abs = np.mean(np.abs(scores[180:]))
        assert abnormal_avg_abs > normal_avg_abs


# ============================================================================
# DistMergeScorer Save/Load 测试
# ============================================================================

class TestDistMergeScorerSaveLoad:
    """测试 DistMergeScorer 持久化"""

    def test_save_load_roundtrip(self, train_scores, test_scores, tmp_path):
        """
        目的：验证 save + load 后推理结果一致
        输入：训练 → save → load → run
        预期：load 后推理输出与 save 前一致
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)
        scores_before, eo_before = scorer.run(test_scores)

        save_dir = tmp_path / "dist_merge_scorer"
        scorer.save(save_dir)

        loaded = DistMergeScorer.load(save_dir)
        scores_after, eo_after = loaded.run(test_scores)

        np.testing.assert_allclose(scores_after, scores_before)
        np.testing.assert_allclose(eo_after.mus, eo_before.mus)
        np.testing.assert_allclose(eo_after.sigmas, eo_before.sigmas)

    def test_load_restores_fitted_state(self, train_scores, tmp_path):
        """
        目的：验证 load 后 is_fitted 为 True
        输入：训练 → save → load
        预期：loaded.is_fitted == True
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)

        save_dir = tmp_path / "dist_merge_scorer"
        scorer.save(save_dir)

        loaded = DistMergeScorer.load(save_dir)
        assert loaded.is_fitted is True

    def test_load_restores_params(self, train_scores, tmp_path):
        """
        目的：验证 load 后分布参数正确恢复
        输入：训练 → save → load
        预期：loaded._mus 和 loaded._sigmas 与原始值一致
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)

        save_dir = tmp_path / "dist_merge_scorer"
        scorer.save(save_dir)

        loaded = DistMergeScorer.load(save_dir)
        np.testing.assert_allclose(loaded._mus, scorer._mus)
        np.testing.assert_allclose(loaded._sigmas, scorer._sigmas)

    def test_save_creates_expected_files(self, train_scores, tmp_path):
        """
        目的：验证 save 生成正确的文件
        输入：训练 → save
        预期：config.json、_learned_state.npz 均存在
        """
        scorer = DistMergeScorer()
        scorer.fit(train_scores)

        save_dir = tmp_path / "dist_merge_scorer"
        scorer.save(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "_learned_state.npz").exists()


# ============================================================================
# DistDirectMergeScorerConfig 测试
# ============================================================================

class TestDistDirectMergeScorerConfig:
    """测试 DistDirectMergeScorerConfig 参数验证"""

    def test_config_defaults(self):
        """
        目的：验证 Config 默认值
        输入：无参数构造
        预期：dist=NORMAL, weights=None, mus=None, sigmas=None
        """
        cfg = DistDirectMergeScorerConfig()
        assert cfg.dist == ScoreDistribution.NORMAL
        assert cfg.weights is None
        assert cfg.mus is None
        assert cfg.sigmas is None

    def test_config_frozen(self):
        """
        目的：验证 Config 不可变
        输入：创建后尝试修改字段
        预期：抛出 ValidationError
        """
        cfg = DistDirectMergeScorerConfig()
        with pytest.raises(ValidationError):
            cfg.dist = ScoreDistribution.LOG_NORMAL

    def test_config_dist_validation(self):
        """
        目的：验证 dist 枚举约束
        输入：dist="invalid"
        预期：ValidationError
        """
        with pytest.raises(ValidationError):
            DistDirectMergeScorerConfig(dist="invalid")

    def test_config_with_preset_params(self):
        """
        目的：验证预设参数正确传递
        输入：mus=[0.5, 1.0], sigmas=[0.1, 0.2]
        预期：Config 中值为自定义值
        """
        cfg = DistDirectMergeScorerConfig(
            mus=[0.5, 1.0],
            sigmas=[0.1, 0.2],
        )
        assert cfg.mus == [0.5, 1.0]
        assert cfg.sigmas == [0.1, 0.2]


# ============================================================================
# DistDirectMergeScorer Run 测试
# ============================================================================

class TestDistDirectMergeScorerRun:
    """测试 DistDirectMergeScorer 推理流程"""

    def test_run_local_params_shape(self, test_scores):
        """
        目的：验证局部参数模式下输出形状
        输入：(200, 3) 测试数据，不预设 mus/sigmas
        预期：scores 形状 (200,)，eo 为 DistMergeScorerExtraOutput
        """
        scorer = DistDirectMergeScorer()
        scores, eo = scorer.run(test_scores)
        assert scores.shape == (200,)
        assert isinstance(eo, DistMergeScorerExtraOutput)

    def test_run_local_params_distribution(self):
        """
        目的：验证局部参数模式下 NORMAL 合并后近似 N(0,1)
        输入：(500, 3) 纯正态分布数据（无异常点）
        预期：均值 ≈ 0，标准差 ≈ 1
        说明：使用纯正常数据验证分布保持性，异常点会偏移分布
        """
        np.random.seed(42)
        # 生成纯正态分布数据，无异常点
        data = np.column_stack([
            np.random.randn(500) * 1.0,
            np.random.randn(500) * 2.0 + 1.0,
            np.random.randn(500) * 0.5 + 0.5,
        ])
        scorer = DistDirectMergeScorer()
        scores, _ = scorer.run(data)
        assert abs(np.mean(scores)) < 0.2
        assert abs(np.std(scores) - 1.0) < 0.2

    def test_run_preset_params(self, test_scores):
        """
        目的：验证预设参数模式下使用预设值
        输入：(200, 3) 测试数据，预设 mus=[0, 1, 0.5], sigmas=[1, 2, 0.5]
        预期：eo.mus/sigmas 等于预设值
        """
        preset_mus = [0.0, 1.0, 0.5]
        preset_sigmas = [1.0, 2.0, 0.5]
        scorer = DistDirectMergeScorer(
            mus=preset_mus,
            sigmas=preset_sigmas,
        )
        _, eo = scorer.run(test_scores)
        np.testing.assert_allclose(eo.mus, preset_mus)
        np.testing.assert_allclose(eo.sigmas, preset_sigmas)

    def test_run_preset_vs_local_different(self, test_scores):
        """
        目的：验证预设参数与局部参数模式下输出不同
        输入：同一测试数据，分别用预设参数和局部参数
        预期：两组输出不同（因为参数来源不同）
        """
        # 局部参数
        scorer_local = DistDirectMergeScorer()
        scores_local, _ = scorer_local.run(test_scores)

        # 预设参数（故意设为不同于数据统计量的值）
        scorer_preset = DistDirectMergeScorer(
            mus=[10.0, 10.0, 10.0],
            sigmas=[1.0, 1.0, 1.0],
        )
        scores_preset, _ = scorer_preset.run(test_scores)

        # 输出应显著不同
        assert not np.allclose(scores_local, scores_preset)

    def test_run_log_normal_local_params(self, test_log_scores):
        """
        目的：验证 LOG_NORMAL 局部参数模式下输出全正数
        输入：(200, 3) 对数正态测试数据
        预期：所有分数 > 0
        """
        scorer = DistDirectMergeScorer(dist=ScoreDistribution.LOG_NORMAL)
        scores, _ = scorer.run(test_log_scores)
        assert np.all(scores > 0)

    def test_run_log_normal_negative_raises(self):
        """
        目的：验证 LOG_NORMAL 模式下输入含非正数时报错
        输入：包含负数的数据
        预期：ValueError
        """
        scorer = DistDirectMergeScorer(dist=ScoreDistribution.LOG_NORMAL)
        bad_data = np.array([[1.0, 2.0], [-1.0, 3.0]])
        with pytest.raises(ValueError, match="正数"):
            scorer.run(bad_data)

    def test_run_with_dataframe(self, test_df):
        """
        目的：验证 DataFrame 输入输出
        输入：DataFrame (200, 3)
        预期：输出为 DataFrame，列名为 ["score"]
        """
        scorer = DistDirectMergeScorer()
        scores, _ = scorer.run(test_df)
        assert isinstance(scores, DataFrame)
        assert list(scores.columns) == ["score"]
        assert len(scores) == len(test_df)

    def test_run_with_ndarray(self, test_scores):
        """
        目的：验证 ndarray 输入返回 ndarray
        输入：ndarray (200, 3)
        预期：scores 为 ndarray 且为 1D
        """
        scorer = DistDirectMergeScorer()
        scores, _ = scorer.run(test_scores)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1

    def test_run_weighted(self):
        """
        目的：验证加权合并模式
        输入：(500, 3) 纯正态分布数据，权重 [0.2, 0.5, 0.3]
        预期：输出形状正确，分布近似 N(0,1)
        说明：使用纯正常数据验证分布保持性
        """
        np.random.seed(42)
        data = np.column_stack([
            np.random.randn(500) * 1.0,
            np.random.randn(500) * 2.0 + 1.0,
            np.random.randn(500) * 0.5 + 0.5,
        ])
        scorer = DistDirectMergeScorer(weights=[0.2, 0.5, 0.3])
        scores, _ = scorer.run(data)
        assert scores.shape == (500,)
        assert abs(np.mean(scores)) < 0.2
        assert abs(np.std(scores) - 1.0) < 0.2

    def test_run_eo_fields(self, test_scores):
        """
        目的：验证 EO 附加输出字段正确
        输入：(200, 3) 测试数据
        预期：mus/sigmas 形状 (3,)，posterior_weights 形状 (200, 3)
        """
        scorer = DistDirectMergeScorer()
        _, eo = scorer.run(test_scores)
        assert eo.mus.shape == (3,)
        assert eo.sigmas.shape == (3,)
        assert eo.posterior_weights.shape == (200, 3)

    def test_run_no_fit_needed(self, test_scores):
        """
        目的：验证非训练算子不需要 fit 即可 run
        输入：未训练的 scorer
        预期：正常执行，不抛出异常
        """
        scorer = DistDirectMergeScorer()
        # 不调用 fit，直接 run
        scores, _ = scorer.run(test_scores)
        assert scores.shape == (200,)

    def test_run_preset_sigma_zero_protection(self, test_scores):
        """
        目的：验证预设 σ=0 时被保护为 1.0
        输入：预设 sigmas=[0.0, 1.0, 0.5]
        预期：eo.sigmas[0] == 1.0（被保护）
        """
        scorer = DistDirectMergeScorer(
            mus=[0.0, 1.0, 0.5],
            sigmas=[0.0, 1.0, 0.5],
        )
        _, eo = scorer.run(test_scores)
        assert eo.sigmas[0] == 1.0

    def test_run_partial_preset_uses_local(self, test_scores):
        """
        目的：验证仅预设 mus（sigmas 为 None）时使用局部计算
        输入：mus=[0, 1, 0.5], sigmas=None
        预期：eo.mus 不等于预设值（因为局部计算覆盖了）
        """
        scorer = DistDirectMergeScorer(
            mus=[0.0, 1.0, 0.5],
            sigmas=None,
        )
        _, eo = scorer.run(test_scores)
        # sigmas 为 None → 使用局部计算，mus 也用局部计算
        local_mus = np.mean(test_scores, axis=0)
        np.testing.assert_allclose(eo.mus, local_mus)


# ============================================================================
# 两个算子一致性测试
# ============================================================================

class TestCrossScorerConsistency:
    """测试可训练版与非训练版在相同参数下的一致性"""

    def test_same_params_same_output(self, train_scores, test_scores):
        """
        目的：验证 DistMergeScorer（训练参数）与 DistDirectMergeScorer（预设相同参数）
              在同一测试数据上输出完全一致
        输入：同一训练/测试数据
        预期：两个算子的输出完全相同
        """
        # 可训练版：fit 学习参数
        trainable = DistMergeScorer()
        trainable.fit(train_scores)
        scores_trainable, eo_trainable = trainable.run(test_scores)

        # 非训练版：使用可训练版学习到的参数作为预设
        direct = DistDirectMergeScorer(
            mus=eo_trainable.mus.tolist(),
            sigmas=eo_trainable.sigmas.tolist(),
        )
        scores_direct, eo_direct = direct.run(test_scores)

        # 输出应完全一致
        np.testing.assert_allclose(scores_trainable, scores_direct)
        np.testing.assert_allclose(eo_trainable.posterior_weights, eo_direct.posterior_weights)

    def test_same_params_weighted_same_output(self, train_scores, test_scores):
        """
        目的：验证加权模式下两个算子输出一致
        输入：同一数据，权重 [0.2, 0.3, 0.5]
        预期：输出完全相同
        """
        weights = [0.2, 0.3, 0.5]

        trainable = DistMergeScorer(weights=weights)
        trainable.fit(train_scores)
        scores_trainable, eo_trainable = trainable.run(test_scores)

        direct = DistDirectMergeScorer(
            weights=weights,
            mus=eo_trainable.mus.tolist(),
            sigmas=eo_trainable.sigmas.tolist(),
        )
        scores_direct, _ = direct.run(test_scores)

        np.testing.assert_allclose(scores_trainable, scores_direct)

    def test_same_params_log_normal_same_output(self, train_log_scores, test_log_scores):
        """
        目的：验证 LOG_NORMAL 模式下两个算子输出一致
        输入：对数正态数据
        预期：输出完全相同
        """
        trainable = DistMergeScorer(dist=ScoreDistribution.LOG_NORMAL)
        trainable.fit(train_log_scores)
        scores_trainable, eo_trainable = trainable.run(test_log_scores)

        direct = DistDirectMergeScorer(
            dist=ScoreDistribution.LOG_NORMAL,
            mus=eo_trainable.mus.tolist(),
            sigmas=eo_trainable.sigmas.tolist(),
        )
        scores_direct, _ = direct.run(test_log_scores)

        np.testing.assert_allclose(scores_trainable, scores_direct)


# ============================================================================
# 算子基本信息测试
# ============================================================================

class TestOperatorMetadata:
    """测试算子元信息（name、version）"""

    def test_dist_merge_scorer_name(self):
        """
        目的：验证 DistMergeScorer.name()
        预期：返回 "dist_merge_scorer"
        """
        assert DistMergeScorer.name() == "dist_merge_scorer"

    def test_dist_merge_scorer_version(self):
        """
        目的：验证 DistMergeScorer.version()
        预期：返回 (1, 0, 0)
        """
        assert DistMergeScorer.version() == (1, 0, 0)

    def test_dist_direct_merge_scorer_name(self):
        """
        目的：验证 DistDirectMergeScorer.name()
        预期：返回 "dist_direct_merge_scorer"
        """
        assert DistDirectMergeScorer.name() == "dist_direct_merge_scorer"

    def test_dist_direct_merge_scorer_version(self):
        """
        目的：验证 DistDirectMergeScorer.version()
        预期：返回 (1, 0, 0)
        """
        assert DistDirectMergeScorer.version() == (1, 0, 0)

    def test_dist_merge_scorer_has_extra_output(self):
        """
        目的：验证 DistMergeScorer 声明了附加输出
        预期：has_extra_output() 返回 True
        """
        assert DistMergeScorer.has_extra_output() is True

    def test_dist_direct_merge_scorer_has_extra_output(self):
        """
        目的：验证 DistDirectMergeScorer 声明了附加输出
        预期：has_extra_output() 返回 True
        """
        assert DistDirectMergeScorer.has_extra_output() is True
