# -*- coding: utf-8 -*-

"""
具体检测算子单元测试

对应源文件：
- residual_scorer.py: ResidualScorer, ResidualMapScorer

测试范围：
- 各算子的 fit/run 基本流程
- DataFrame/ndarray 双类型支持
- 边界条件（零标准差、小样本、单特征等）
- 附加输出正确性
- 参数验证
"""

import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.residual_scorer import (ResidualMapScorer, ResidualScorer,
                                                            ResidualScorerExtraOutput)


# ============================================================================
# 公共测试数据
# ============================================================================

@pytest.fixture
def train_data():
    """测试用训练数据（ndarray, 100x3）"""
    np.random.seed(42)
    return np.random.randn(100, 3)


@pytest.fixture
def test_data():
    """测试用测试数据（ndarray, 60x3，含异常点）"""
    np.random.seed(123)
    normal = np.random.randn(50, 3)
    abnormal = np.random.randn(10, 3) * 5 + 10
    return np.vstack([normal, abnormal])


@pytest.fixture
def train_df(train_data):
    """测试用训练数据（DataFrame）"""
    return DataFrame(train_data, columns=["a", "b", "c"])


@pytest.fixture
def test_df(test_data):
    """测试用测试数据（DataFrame）"""
    return DataFrame(test_data, columns=["a", "b", "c"])


# ============================================================================
# ResidualScorer 测试
# ============================================================================

class TestResidualScorer:
    """测试残差比较器的比较逻辑"""

    def test_mse_zero_residual(self):
        """
        目的：验证相同输入的 MSE 残差为零
        输入：y_pred == y_real
        预期：残差全为 0
        """
        comp = ResidualScorer(metric="mse")
        x = np.random.randn(10, 3)
        residual, eo = comp.run((x, x))
        np.testing.assert_allclose(residual, 0.0, atol=1e-10)
        assert eo is not None and isinstance(eo, ResidualScorerExtraOutput)

    def test_mse_known_value(self):
        """
        目的：验证 MSE 计算结果正确
        输入：y_pred=0, y_real=[1,2,3] 的单样本
        预期：MSE = mean(1,4,9) = 14/3
        """
        comp = ResidualScorer(metric="mse")
        y_pred = np.zeros((1, 3))
        y_real = np.array([[1.0, 2.0, 3.0]])
        residual, _ = comp.run((y_pred, y_real))
        np.testing.assert_allclose(residual[0], 14.0 / 3.0)

    def test_mae_known_value(self):
        """
        目的：验证 MAE 计算结果正确
        输入：y_pred=0, y_real=[1,2,3] 的单样本
        预期：MAE = mean(1,2,3) = 2.0
        """
        comp = ResidualScorer(metric="mae")
        y_pred = np.zeros((1, 3))
        y_real = np.array([[1.0, 2.0, 3.0]])
        residual, _ = comp.run((y_pred, y_real))
        np.testing.assert_allclose(residual[0], 2.0)

    def test_with_dataframe(self):
        """
        目的：验证 DataFrame 输入的残差计算
        输入：DataFrame 的 (y_pred, y_real)
        预期：输出为 DataFrame
        """
        comp = ResidualScorer(metric="mse")
        y_pred = DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        y_real = DataFrame({"a": [1.5, 2.5], "b": [3.5, 4.5]})
        residual, _ = comp.run((y_pred, y_real))
        assert isinstance(residual, DataFrame)

    def test_output_1d(self):
        """
        目的：验证输出为 1D
        输入：(10, 3) 的 y_pred, y_real
        预期：残差形状为 (10,)
        """
        comp = ResidualScorer()
        y_pred = np.random.randn(10, 3)
        y_real = np.random.randn(10, 3)
        residual, _ = comp.run((y_pred, y_real))
        assert residual.ndim == 1
        assert len(residual) == 10

    def test_config_default(self):
        """
        目的：验证默认配置
        输入：无参数构造
        预期：metric="mse"
        """
        comp = ResidualScorer()
        assert comp.config.metric == "mse"


# ============================================================================
# ResidualMapScorer 测试
# ============================================================================

class TestResidualMapScorer:
    """测试残差映射评分器的逐元素评分逻辑"""

    def test_mse_known_value(self):
        """
        目的：验证 MSE 模式下逐元素平方残差计算正确
        输入：y_pred=0, y_real=[[1,2],[3,4]]
        预期：输出为 [[1,4],[9,16]]（逐元素平方，不聚合）
        """
        scorer = ResidualMapScorer(metric="mse")
        y_pred = np.zeros((2, 2))
        y_real = np.array([[1.0, 2.0], [3.0, 4.0]])
        scores = scorer.run((y_pred, y_real))
        expected = np.array([[1.0, 4.0], [9.0, 16.0]])
        np.testing.assert_allclose(scores, expected)

    def test_mae_known_value(self):
        """
        目的：验证 MAE 模式下逐元素绝对残差计算正确
        输入：y_pred=0, y_real=[[1,-2],[3,-4]]
        预期：输出为 [[1,2],[3,4]]（逐元素取绝对值，不聚合）
        """
        scorer = ResidualMapScorer(metric="mae")
        y_pred = np.zeros((2, 2))
        y_real = np.array([[1.0, -2.0], [3.0, -4.0]])
        scores = scorer.run((y_pred, y_real))
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(scores, expected)

    def test_output_2d_shape(self):
        """
        目的：验证输出为 2D 且形状与输入一致（逐特征，不聚合）
        输入：(10, 3) 的 y_pred, y_real
        预期：输出形状为 (10, 3)
        """
        scorer = ResidualMapScorer()
        y_pred = np.random.randn(10, 3)
        y_real = np.random.randn(10, 3)
        scores = scorer.run((y_pred, y_real))
        assert scores.ndim == 2
        assert scores.shape == (10, 3)

    def test_zero_residual(self):
        """
        目的：验证相同输入的逐元素残差为零
        输入：y_pred == y_real
        预期：所有元素为 0
        """
        scorer = ResidualMapScorer(metric="mse")
        x = np.random.randn(10, 3)
        scores = scorer.run((x, x))
        np.testing.assert_allclose(scores, 0.0, atol=1e-10)

    def test_with_dataframe(self):
        """
        目的：验证 DataFrame 输入返回 DataFrame 且列名一致
        输入：DataFrame 的 (y_pred, y_real)，列名为 ["a", "b"]
        预期：输出为 DataFrame，列名与输入相同
        """
        scorer = ResidualMapScorer(metric="mse")
        y_pred = DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        y_real = DataFrame({"a": [1.5, 2.5], "b": [3.5, 4.5]})
        scores = scorer.run((y_pred, y_real))
        assert isinstance(scores, DataFrame)
        assert list(scores.columns) == ["a", "b"]

    def test_config_default(self):
        """
        目的：验证默认配置
        输入：无参数构造
        预期：metric="mse"
        """
        scorer = ResidualMapScorer()
        assert scorer.config.metric == "mse"

    def test_name(self):
        """
        目的：验证算子名称正确
        输入：类方法 name()
        预期：返回 "residual_map_scorer"
        """
        assert ResidualMapScorer.name() == "residual_map_scorer"

    def test_has_extra_output_false(self):
        """
        目的：验证算子无附加输出
        输入：实例方法 has_extra_output()
        预期：返回 False
        """
        scorer = ResidualMapScorer()
        assert scorer.has_extra_output() is False

    def test_no_fit_required(self):
        """
        目的：验证算子无需 fit 即可直接 run
        输入：直接调用 run，不调用 fit
        预期：正常执行并返回结果
        """
        scorer = ResidualMapScorer()
        y_pred = np.zeros((5, 2))
        y_real = np.ones((5, 2))
        scores = scorer.run((y_pred, y_real))
        assert scores.shape == (5, 2)
        np.testing.assert_allclose(scores, 1.0)
