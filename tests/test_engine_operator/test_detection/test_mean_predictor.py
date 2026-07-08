# -*- coding: utf-8 -*-

"""
均值预测器单元测试

对应源文件：
- mean_predictor.py: MeanPredictor

测试范围：
- fit/run 基本流程
- DataFrame/ndarray 双类型支持
- 边界条件（未训练先推理等）
"""

import numpy as np
import pytest
from pandas import DataFrame

from tsas.engine.operator.detection.mean_predictor import MeanPredictor


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
# MeanPredictor 测试
# ============================================================================

class TestMeanPredictor:
    """测试均值预测器的训练和推理流程"""

    def test_fit_learns_mean(self, train_data):
        """
        目的：验证 fit 正确学习列均值
        输入：(100, 3) 训练数据
        预期：_mean 与 x.mean(axis=0) 相等
        """
        predictor = MeanPredictor()
        predictor.fit(train_data)
        assert predictor._mean is not None
        np.testing.assert_allclose(predictor._mean, train_data.mean(axis=0))

    def test_run_broadcasts_mean(self, train_data, test_data):
        """
        目的：验证 run 将均值广播为预测值
        输入：训练数据 + 测试数据
        预期：每行预测值等于训练均值
        """
        predictor = MeanPredictor()
        predictor.fit(train_data)
        pred = predictor.run(test_data)
        assert pred.shape == test_data.shape
        # 每行都应等于训练均值
        for i in range(len(test_data)):
            np.testing.assert_allclose(pred[i], predictor._mean)

    def test_with_dataframe(self, train_df, test_df):
        """
        目的：验证 DataFrame 输入输出类型镜像
        输入：DataFrame 训练和测试数据
        预期：输出为 DataFrame
        """
        predictor = MeanPredictor()
        predictor.fit(train_df)
        pred = predictor.run(test_df)
        assert isinstance(pred, DataFrame)

    def test_before_fit_raises(self, test_data):
        """
        目的：验证未训练时 run 抛出 RuntimeError
        输入：未 fit 的预测器
        预期：抛出 RuntimeError
        """
        predictor = MeanPredictor()
        with pytest.raises(RuntimeError):
            predictor.run(test_data)
