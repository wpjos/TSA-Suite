# -*- coding: utf-8 -*-

"""
均值合并评分器单元测试

对应源文件:
- mean_merge_scorer.py: MeanMergeScorer、ScoreMergeMethod

测试范围:
- Config 参数验证
- 四种平均方法（算术、几何、调和、平方）的等权与加权计算正确性
- 后验权重矩阵正确性（形状、行和为 1、除零保护）
- DataFrame/ndarray 双类型支持
- 输入校验（GEOMETRIC/HARMONIC 正数约束）
- 算子元信息（name、version、has_extra_output）
"""

import numpy as np
import pytest
from pandas import DataFrame
from pydantic import ValidationError

from tsas.engine.operator.detection.mean_merge_scorer import (
    MeanMergeScorer,
    MeanMergeScorerConfig,
    MeanMergeScorerExtraOutput,
    ScoreMergeMethod,
)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def positive_data():
    """全正数测试数据（ndarray, 100x3），适用于所有方法"""
    np.random.seed(42)
    return np.abs(np.random.randn(100, 3)) + 0.1


@pytest.fixture
def real_data():
    """含负数测试数据（ndarray, 100x3），仅适用于 ARITHMETIC 和 QUADRATIC"""
    np.random.seed(42)
    return np.random.randn(100, 3)


@pytest.fixture
def positive_df(positive_data):
    """全正数测试数据（DataFrame）"""
    return DataFrame(positive_data, columns=["a", "b", "c"])


@pytest.fixture
def real_df(real_data):
    """含负数测试数据（DataFrame）"""
    return DataFrame(real_data, columns=["a", "b", "c"])


# ============================================================================
# Config 测试
# ============================================================================

class TestMeanMergeScorerConfig:
    """测试 MeanMergeScorerConfig 参数验证"""

    def test_config_defaults(self):
        """
        目的：验证 Config 默认值
        输入：无参数构造
        预期：method=ARITHMETIC, weights=None
        """
        cfg = MeanMergeScorerConfig()
        assert cfg.method == ScoreMergeMethod.ARITHMETIC
        assert cfg.weights is None

    def test_config_frozen(self):
        """
        目的：验证 Config 不可变
        输入：创建后尝试修改字段
        预期：抛出 ValidationError
        """
        cfg = MeanMergeScorerConfig()
        with pytest.raises(ValidationError):
            cfg.method = ScoreMergeMethod.GEOMETRIC

    def test_config_method_validation(self):
        """
        目的：验证 method 枚举约束
        输入：method="invalid"
        预期：ValidationError
        """
        with pytest.raises(ValidationError):
            MeanMergeScorerConfig(method="invalid")

    def test_config_custom_values(self):
        """
        目的：验证自定义参数正确传递
        输入：method=HARMONIC, weights=[0.3, 0.7]
        预期：Config 中值为自定义值
        """
        cfg = MeanMergeScorerConfig(
            method=ScoreMergeMethod.HARMONIC,
            weights=[0.3, 0.7],
        )
        assert cfg.method == ScoreMergeMethod.HARMONIC
        assert cfg.weights == [0.3, 0.7]

    def test_config_all_methods(self):
        """
        目的：验证所有枚举值均可构造
        输入：四种方法分别构造
        预期：均不抛出异常
        """
        for method in ScoreMergeMethod:
            cfg = MeanMergeScorerConfig(method=method)
            assert cfg.method == method


# ============================================================================
# 算术平均测试
# ============================================================================

class TestArithmeticMean:
    """测试算术平均方法"""

    def test_equal_weight_known_values(self):
        """
        目的：验证等权算术平均的计算正确性
        输入：[[1, 2, 3], [4, 5, 6]]
        预期：输出 [2.0, 5.0]（逐行均值）
        """
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        scorer = MeanMergeScorer()
        scores, _ = scorer.run(x)
        np.testing.assert_allclose(scores, [2.0, 5.0])

    def test_equal_weight_matches_np_mean(self, real_data):
        """
        目的：验证等权算术平均与 np.mean(x, axis=1) 一致
        输入：(100, 3) 数据
        预期：输出与 np.mean 一致
        """
        scorer = MeanMergeScorer()
        scores, _ = scorer.run(real_data)
        np.testing.assert_allclose(scores, np.mean(real_data, axis=1))

    def test_weighted_known_values(self):
        """
        目的：验证加权算术平均的计算正确性
        输入：[[1, 2, 3]], weights=[1, 2, 1]（归一化后 [0.25, 0.5, 0.25]）
        预期：输出 [0.25*1 + 0.5*2 + 0.25*3] = [2.0]
        """
        x = np.array([[1, 2, 3]], dtype=float)
        scorer = MeanMergeScorer(weights=[1, 2, 1])
        scores, _ = scorer.run(x)
        np.testing.assert_allclose(scores, [2.0])

    def test_weighted_matches_manual(self, real_data):
        """
        目的：验证加权算术平均与手动计算一致
        输入：(100, 3) 数据，权重 [0.2, 0.3, 0.5]
        预期：输出与手动加权平均一致
        """
        weights = [0.2, 0.3, 0.5]
        scorer = MeanMergeScorer(weights=weights)
        scores, _ = scorer.run(real_data)
        w = np.array(weights) / sum(weights)
        expected = np.sum(real_data * w, axis=1)
        np.testing.assert_allclose(scores, expected)

    def test_accepts_negative_values(self, real_data):
        """
        目的：验证算术平均接受负数输入
        输入：含负数的数据
        预期：正常执行，不抛出异常
        """
        scorer = MeanMergeScorer()
        scores, _ = scorer.run(real_data)
        assert scores.shape == (100,)


# ============================================================================
# 几何平均测试
# ============================================================================

class TestGeometricMean:
    """测试几何平均方法"""

    def test_equal_weight_known_values(self):
        """
        目的：验证等权几何平均的计算正确性
        输入：[[1, 4, 9]]
        预期：输出 [exp(mean(log([1,4,9])))] = [exp((0+ln4+ln9)/3)] ≈ [3.302]
        """
        x = np.array([[1, 4, 9]], dtype=float)
        scorer = MeanMergeScorer(method=ScoreMergeMethod.GEOMETRIC)
        scores, _ = scorer.run(x)
        expected = np.exp(np.mean(np.log(x), axis=1))
        np.testing.assert_allclose(scores, expected)

    def test_equal_weight_matches_manual(self, positive_data):
        """
        目的：验证等权几何平均与 exp(mean(log(x))) 一致
        输入：(100, 3) 正数数据
        预期：输出与手动计算一致
        """
        scorer = MeanMergeScorer(method=ScoreMergeMethod.GEOMETRIC)
        scores, _ = scorer.run(positive_data)
        expected = np.exp(np.mean(np.log(positive_data), axis=1))
        np.testing.assert_allclose(scores, expected)

    def test_weighted_matches_manual(self, positive_data):
        """
        目的：验证加权几何平均与 exp(Σ(w_i·log(x_i))) 一致
        输入：(100, 3) 正数数据，权重 [0.2, 0.3, 0.5]
        预期：输出与手动计算一致
        """
        weights = [0.2, 0.3, 0.5]
        scorer = MeanMergeScorer(method=ScoreMergeMethod.GEOMETRIC, weights=weights)
        scores, _ = scorer.run(positive_data)
        w = np.array(weights) / sum(weights)
        expected = np.exp(np.sum(np.log(positive_data) * w, axis=1))
        np.testing.assert_allclose(scores, expected)

    def test_negative_raises(self):
        """
        目的：验证几何平均拒绝负数输入
        输入：包含负数的数据
        预期：ValueError
        """
        x = np.array([[1.0, -2.0, 3.0]])
        scorer = MeanMergeScorer(method=ScoreMergeMethod.GEOMETRIC)
        with pytest.raises(ValueError, match="正数"):
            scorer.run(x)

    def test_zero_raises(self):
        """
        目的：验证几何平均拒绝零值输入
        输入：包含 0 的数据
        预期：ValueError
        """
        x = np.array([[1.0, 0.0, 3.0]])
        scorer = MeanMergeScorer(method=ScoreMergeMethod.GEOMETRIC)
        with pytest.raises(ValueError, match="正数"):
            scorer.run(x)


# ============================================================================
# 调和平均测试
# ============================================================================

class TestHarmonicMean:
    """测试调和平均方法"""

    def test_equal_weight_known_values(self):
        """
        目的：验证等权调和平均的计算正确性
        输入：[[1, 2, 4]]
        预期：输出 [3 / (1/1 + 1/2 + 1/4)] = [3 / 1.75] ≈ [1.714]
        """
        x = np.array([[1, 2, 4]], dtype=float)
        scorer = MeanMergeScorer(method=ScoreMergeMethod.HARMONIC)
        scores, _ = scorer.run(x)
        expected = 3.0 / np.sum(1.0 / x, axis=1)
        np.testing.assert_allclose(scores, expected)

    def test_equal_weight_matches_manual(self, positive_data):
        """
        目的：验证等权调和平均与 n / Σ(1/x_i) 一致
        输入：(100, 3) 正数数据
        预期：输出与手动计算一致
        """
        scorer = MeanMergeScorer(method=ScoreMergeMethod.HARMONIC)
        scores, _ = scorer.run(positive_data)
        n = positive_data.shape[1]
        expected = n / np.sum(1.0 / positive_data, axis=1)
        np.testing.assert_allclose(scores, expected)

    def test_weighted_matches_manual(self, positive_data):
        """
        目的：验证加权调和平均与 1 / Σ(w_i/x_i) 一致（归一化权重）
        输入：(100, 3) 正数数据，权重 [0.2, 0.3, 0.5]
        预期：输出与手动计算一致
        """
        weights = [0.2, 0.3, 0.5]
        scorer = MeanMergeScorer(method=ScoreMergeMethod.HARMONIC, weights=weights)
        scores, _ = scorer.run(positive_data)
        w = np.array(weights) / sum(weights)
        expected = 1.0 / np.sum(w / positive_data, axis=1)
        np.testing.assert_allclose(scores, expected)

    def test_negative_raises(self):
        """
        目的：验证调和平均拒绝负数输入
        输入：包含负数的数据
        预期：ValueError
        """
        x = np.array([[1.0, -2.0, 3.0]])
        scorer = MeanMergeScorer(method=ScoreMergeMethod.HARMONIC)
        with pytest.raises(ValueError, match="正数"):
            scorer.run(x)


# ============================================================================
# 平方平均测试
# ============================================================================

class TestQuadraticMean:
    """测试平方平均（RMS）方法"""

    def test_equal_weight_known_values(self):
        """
        目的：验证等权平方平均的计算正确性
        输入：[[1, 2, 3]]
        预期：输出 [sqrt((1+4+9)/3)] = [sqrt(14/3)] ≈ [2.160]
        """
        x = np.array([[1, 2, 3]], dtype=float)
        scorer = MeanMergeScorer(method=ScoreMergeMethod.QUADRATIC)
        scores, _ = scorer.run(x)
        expected = np.sqrt(np.mean(x ** 2, axis=1))
        np.testing.assert_allclose(scores, expected)

    def test_equal_weight_matches_manual(self, real_data):
        """
        目的：验证等权平方平均与 sqrt(mean(x²)) 一致
        输入：(100, 3) 数据
        预期：输出与手动计算一致
        """
        scorer = MeanMergeScorer(method=ScoreMergeMethod.QUADRATIC)
        scores, _ = scorer.run(real_data)
        expected = np.sqrt(np.mean(real_data ** 2, axis=1))
        np.testing.assert_allclose(scores, expected)

    def test_weighted_matches_manual(self, real_data):
        """
        目的：验证加权平方平均与 sqrt(Σ(w_i·x_i²)) 一致（归一化权重）
        输入：(100, 3) 数据，权重 [0.2, 0.3, 0.5]
        预期：输出与手动计算一致
        """
        weights = [0.2, 0.3, 0.5]
        scorer = MeanMergeScorer(method=ScoreMergeMethod.QUADRATIC, weights=weights)
        scores, _ = scorer.run(real_data)
        w = np.array(weights) / sum(weights)
        expected = np.sqrt(np.sum((real_data ** 2) * w, axis=1))
        np.testing.assert_allclose(scores, expected)

    def test_accepts_negative_values(self, real_data):
        """
        目的：验证平方平均接受负数输入
        输入：含负数的数据
        预期：所有输出 >= 0（RMS 始终非负）
        """
        scorer = MeanMergeScorer(method=ScoreMergeMethod.QUADRATIC)
        scores, _ = scorer.run(real_data)
        assert np.all(scores >= 0)


# ============================================================================
# 后验权重矩阵测试
# ============================================================================

class TestPosteriorWeights:
    """测试后验权重矩阵"""

    def test_shape(self, positive_data):
        """
        目的：验证后验权重矩阵形状正确
        输入：(100, 3) 数据
        预期：后验权重形状 (100, 3)
        """
        scorer = MeanMergeScorer()
        _, eo = scorer.run(positive_data)
        assert eo.posterior_weights.shape == (100, 3)

    def test_row_sum_one_arithmetic(self, positive_data):
        """
        目的：验证算术平均后验权重行和为 1
        输入：(100, 3) 正数数据
        预期：每行 posterior_weights 求和 ≈ 1.0
        """
        scorer = MeanMergeScorer(method=ScoreMergeMethod.ARITHMETIC)
        _, eo = scorer.run(positive_data)
        row_sums = np.sum(eo.posterior_weights, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_row_sum_one_geometric(self, positive_data):
        """
        目的：验证几何平均后验权重行和为 1
        输入：(100, 3) 正数数据
        预期：每行 posterior_weights 求和 ≈ 1.0
        """
        scorer = MeanMergeScorer(method=ScoreMergeMethod.GEOMETRIC)
        _, eo = scorer.run(positive_data)
        row_sums = np.sum(eo.posterior_weights, axis=1)
        nonzero = row_sums != 0
        np.testing.assert_allclose(row_sums[nonzero], 1.0, atol=1e-10)

    def test_row_sum_one_harmonic(self, positive_data):
        """
        目的：验证调和平均后验权重行和为 1
        输入：(100, 3) 正数数据
        预期：每行 posterior_weights 求和 ≈ 1.0
        """
        scorer = MeanMergeScorer(method=ScoreMergeMethod.HARMONIC)
        _, eo = scorer.run(positive_data)
        row_sums = np.sum(eo.posterior_weights, axis=1)
        nonzero = row_sums != 0
        np.testing.assert_allclose(row_sums[nonzero], 1.0, atol=1e-10)

    def test_row_sum_one_quadratic(self, real_data):
        """
        目的：验证平方平均后验权重行和为 1
        输入：(100, 3) 数据
        预期：每行 posterior_weights 求和 ≈ 1.0
        """
        scorer = MeanMergeScorer(method=ScoreMergeMethod.QUADRATIC)
        _, eo = scorer.run(real_data)
        row_sums = np.sum(eo.posterior_weights, axis=1)
        nonzero = row_sums != 0
        np.testing.assert_allclose(row_sums[nonzero], 1.0, atol=1e-10)

    def test_divide_zero_protection_arithmetic(self):
        """
        目的：验证算术平均除零保护
        输入：构造 [1, -1] 使算术平均为 0
        预期：该行后验权重全为 0
        """
        x = np.array([[1.0, -1.0]])
        scorer = MeanMergeScorer(method=ScoreMergeMethod.ARITHMETIC)
        _, eo = scorer.run(x)
        assert np.all(eo.posterior_weights[0] == 0.0)

    def test_weighted_posterior_weights(self, positive_data):
        """
        目的：验证加权模式下后验权重正确
        输入：(100, 3) 数据，权重 [0.5, 0.3, 0.2]
        预期：后验权重形状 (100, 3)，行和为 1
        """
        scorer = MeanMergeScorer(weights=[0.5, 0.3, 0.2])
        _, eo = scorer.run(positive_data)
        assert eo.posterior_weights.shape == (100, 3)
        row_sums = np.sum(eo.posterior_weights, axis=1)
        nonzero = row_sums != 0
        np.testing.assert_allclose(row_sums[nonzero], 1.0, atol=1e-10)


# ============================================================================
# 输入输出类型测试
# ============================================================================

class TestInputOutputTypes:
    """测试 DataFrame/ndarray 双类型支持"""

    def test_dataframe_input_arithmetic(self, positive_df):
        """
        目的：验证 DataFrame 输入时输出为 DataFrame，列名为 ["score"]
        输入：DataFrame (100, 3)
        预期：输出为 DataFrame，列名为 ["score"]
        """
        scorer = MeanMergeScorer()
        scores, _ = scorer.run(positive_df)
        assert isinstance(scores, DataFrame)
        assert list(scores.columns) == ["score"]
        assert len(scores) == 100

    def test_ndarray_input_arithmetic(self, positive_data):
        """
        目的：验证 ndarray 输入时输出为 ndarray
        输入：ndarray (100, 3)
        预期：输出为 ndarray 且为 1D
        """
        scorer = MeanMergeScorer()
        scores, _ = scorer.run(positive_data)
        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1

    def test_dataframe_input_geometric(self, positive_df):
        """
        目的：验证 DataFrame 输入在几何平均模式下正常工作
        输入：DataFrame (100, 3)
        预期：输出为 DataFrame，列名为 ["score"]
        """
        scorer = MeanMergeScorer(method=ScoreMergeMethod.GEOMETRIC)
        scores, _ = scorer.run(positive_df)
        assert isinstance(scores, DataFrame)
        assert list(scores.columns) == ["score"]

    def test_dataframe_values_match_ndarray(self, positive_data, positive_df):
        """
        目的：验证 DataFrame 和 ndarray 输入的计算结果一致
        输入：同一数据的两种格式
        预期：输出值一致
        """
        scorer1 = MeanMergeScorer()
        scorer2 = MeanMergeScorer()
        scores_ndarray, _ = scorer1.run(positive_data)
        scores_df, _ = scorer2.run(positive_df)
        np.testing.assert_allclose(scores_ndarray, scores_df.values.ravel())


# ============================================================================
# 无需训练测试
# ============================================================================

class TestNoFitRequired:
    """测试无需训练即可推理"""

    def test_run_without_fit(self, positive_data):
        """
        目的：验证无需 fit 即可直接 run
        输入：未训练的 scorer + 数据
        预期：正常执行，不抛出异常
        """
        scorer = MeanMergeScorer()
        scores, _ = scorer.run(positive_data)
        assert scores.shape == (100,)

    def test_is_fitted_false(self):
        """
        目的：验证算子默认 is_fitted 为 False（不继承 LearnableOperatorMixin）
        输入：新建实例
        预期：不抛出异常（MeanMergeScorer 不继承 LearnableOperatorMixin）
        """
        scorer = MeanMergeScorer()
        # MeanMergeScorer 不继承 LearnableOperatorMixin，没有 is_fitted 属性
        assert not hasattr(scorer, 'is_fitted')


# ============================================================================
# 算子元信息测试
# ============================================================================

class TestOperatorMetadata:
    """测试算子元信息"""

    def test_name(self):
        """
        目的：验证 name() 返回正确算子标识
        预期：返回 "mean_merge_scorer"
        """
        assert MeanMergeScorer.name() == "mean_merge_scorer"

    def test_version(self):
        """
        目的：验证 version() 返回正确版本号
        预期：返回 (1, 0, 0)
        """
        assert MeanMergeScorer.version() == (1, 0, 0)

    def test_has_extra_output_true(self):
        """
        目的：验证 has_extra_output() 返回 True
        预期：返回 True（有 EO）
        """
        assert MeanMergeScorer.has_extra_output() is True

    def test_oid(self):
        """
        目的：验证 oid 格式包含算子名称
        输入：新建 MeanMergeScorer 实例
        预期：oid 以 "mean_merge_scorer$" 开头
        """
        scorer = MeanMergeScorer()
        assert scorer.oid.startswith("mean_merge_scorer$")

    def test_eo_type(self, positive_data):
        """
        目的：验证 EO 类型为 MeanMergeScorerExtraOutput
        输入：运行后检查 EO 类型
        预期：eo 为 MeanMergeScorerExtraOutput 实例
        """
        scorer = MeanMergeScorer()
        _, eo = scorer.run(positive_data)
        assert isinstance(eo, MeanMergeScorerExtraOutput)
