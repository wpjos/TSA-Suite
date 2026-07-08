# -*- coding: utf-8 -*-

"""
无标签自评估指标算子测试

测试覆盖:
    1. 基本功能: run() 返回自评估分数
    2. scores() 方法: 按 main_scores 提取命名标量
    3. 算法验证: 变异系数和 sigmoid 映射正确性
    4. 边界条件: 空数组、全零分数、常数分数
    5. 类型兼容: ndarray、list 输入

测试约束:
    - 代码覆盖率 > 90%
    - 测试通过率 100%
    - 中文注释说明测试目的、输入、输出和预期结果
"""

import numpy as np
import pytest

from tsas.engine.operator.evaluation.self_evaluation import (SelfEvaluation, SelfEvaluationConfig)


# ============================================================================
# 基本功能测试
# ============================================================================

class TestSelfEvaluationBasic:
    """基本功能测试"""

    def test_run_returns_float(self):
        """
        测试目的: run() 返回 float 类型
        输入: 任意分数数组 [0.1, 0.2, 0.15, 5.0, 4.8, 5.2]
        输出: float
        预期: 返回值为 float，范围 [0, 1]
        """
        scores = np.array([0.1, 0.2, 0.15, 5.0, 4.8, 5.2])

        op = SelfEvaluation()
        result = op.run(scores)

        assert isinstance(result, float)
        assert 0.0 < result <= 1.0

    def test_run_with_distinct_scores(self):
        """
        测试目的: 有明显差异的分数时自评估分数较高
        输入: 正常样本分数低(~0.1)，异常样本分数高(~5.0)
        输出: float
        预期: 自评估分数 > 0.5
        """
        # 正常样本分数 ~0.1，异常样本分数 ~5.0
        scores = np.array([0.1, 0.2, 0.15, 5.0, 4.8, 5.2])

        op = SelfEvaluation()
        result = op.run(scores)

        # 明显差异 → CV 较大 → sigmoid 输出较大
        assert result > 0.5

    def test_run_with_uniform_scores(self):
        """
        测试目的: 常数分数时自评估分数接近 0.5
        输入: 所有分数相同 [1.0, 1.0, 1.0, 1.0]
        输出: float
        预期: 自评估分数 ≈ 0.5（CV=0 → sigmoid(0)=0.5）
        """
        scores = np.array([1.0, 1.0, 1.0, 1.0])

        op = SelfEvaluation()
        result = op.run(scores)

        # std=0 → CV=0 → sigmoid(0)=0.5
        assert abs(result - 0.5) < 0.01


# ============================================================================
# scores() 方法测试
# ============================================================================

class TestSelfEvaluationScores:
    """scores() 方法测试"""

    def test_scores_returns_dict(self):
        """
        测试目的: scores() 返回按 main_scores 映射的字典
        输入: 默认配置 main_scores={"self_eval": "_"}
        输出: dict[str, float]
        预期: 返回 {"self_eval": float}
        """
        scores = np.array([0.1, 0.2, 5.0])

        op = SelfEvaluation()
        result_dict = op.scores(scores)

        assert result_dict is not None
        assert "self_eval" in result_dict
        assert isinstance(result_dict["self_eval"], float)

    def test_scores_matches_run_value(self):
        """
        测试目的: scores() 提取的值与 run() 结果一致
        输入: 分数数组 [0.1, 0.2, 5.0]
        输出: dict[str, float]
        预期: scores["self_eval"] == run()
        """
        scores = np.array([0.1, 0.2, 5.0])

        op = SelfEvaluation()
        run_result = op.run(scores)
        scores_result = op.scores(scores)

        assert scores_result["self_eval"] == run_result

    def test_scores_returns_none_when_main_scores_is_none(self):
        """
        测试目的: main_scores=None 时 scores() 返回 None
        输入: main_scores=None
        输出: None
        预期: scores() 返回 None
        """
        scores = np.array([0.1, 0.2, 5.0])

        op = SelfEvaluation(main_scores=None)
        result_dict = op.scores(scores)

        assert result_dict is None


# ============================================================================
# 算法验证测试
# ============================================================================

class TestSelfEvaluationAlgorithm:
    """算法验证测试"""

    def test_cv_calculation(self):
        """
        测试目的: 验证变异系数计算正确
        输入: 已知均值和标准差的分数数组 [1.0, 2.0, 3.0, 4.0, 5.0]
        输出: float
        预期: CV = std / |mean|，sigmoid(CV) == result
        """
        # 构造已知均值和标准差的数据
        # mean = 3.0, std ≈ 2.0 (手动验证)
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected_mean = np.mean(np.abs(scores))
        expected_std = np.std(scores)
        expected_cv = expected_std / expected_mean

        op = SelfEvaluation()
        result = op.run(scores)

        # 验证 sigmoid 映射
        expected_result = 1.0 / (1.0 + np.exp(-expected_cv))
        assert abs(result - expected_result) < 1e-6

    def test_sigmoid_mapping(self):
        """
        测试目的: 验证 sigmoid 映射范围
        输入: 不同 CV 值的数据（常数 vs 大差异）
        输出: float
        预期: sigmoid 输出在 [0.5, 1) 范围（CV >= 0）
        """
        # CV=0 → sigmoid=0.5
        uniform_scores = np.array([1.0, 1.0, 1.0])
        op = SelfEvaluation()
        result_uniform = op.run(uniform_scores)
        assert abs(result_uniform - 0.5) < 0.01

        # CV 较大 → sigmoid 接近 1
        varied_scores = np.array([0.01, 0.02, 100.0, 200.0])
        result_varied = op.run(varied_scores)
        assert result_varied > 0.7

    def test_result_increases_with_cv(self):
        """
        测试目的: 自评估分数随 CV 增大而增大
        输入: 两组不同 CV 的数据（低 CV vs 高 CV）
        输出: float
        预期: CV 大的数据自评估分数更高
        """
        # 低 CV：分数接近
        low_cv_scores = np.array([1.0, 1.1, 1.2, 1.3])
        # 高 CV：分数差异大
        high_cv_scores = np.array([1.0, 1.0, 100.0, 100.0])

        op = SelfEvaluation()
        result_low = op.run(low_cv_scores)
        result_high = op.run(high_cv_scores)

        assert result_high > result_low


# ============================================================================
# 边界条件测试
# ============================================================================

class TestSelfEvaluationEdgeCases:
    """边界条件测试"""

    def test_empty_array(self):
        """
        测试目的: 空数组返回 0.0
        输入: 空数组 []
        输出: float
        预期: 返回 0.0
        """
        scores = np.array([])

        op = SelfEvaluation()
        result = op.run(scores)

        assert result == 0.0

    def test_all_zero_scores(self):
        """
        测试目的: 全零分数时自评估返回 0.0（零均值保护）
        输入: 全零数组 [0,0,...,0]（长度10）
        输出: float
        预期: 返回 0.0
        """
        scores = np.zeros(10)

        op = SelfEvaluation()
        result = op.run(scores)

        assert result == 0.0

    def test_single_element(self):
        """
        测试目的: 单元素数组场景
        输入: 长度为 1 的数组 [5.0]
        输出: float
        预期: std=0 → CV=0 → result ≈ 0.5
        """
        scores = np.array([5.0])

        op = SelfEvaluation()
        result = op.run(scores)

        # 单元素 std=0 → sigmoid(0)=0.5
        assert abs(result - 0.5) < 0.01

    def test_negative_scores(self):
        """
        测试目的: 分数含负值时正确处理（取绝对值求均值）
        输入: 含负数的分数数组 [-1.0, -2.0, 5.0, 6.0]
        输出: float
        预期: 正常计算，结果在 [0, 1] 范围
        """
        scores = np.array([-1.0, -2.0, 5.0, 6.0])

        op = SelfEvaluation()
        result = op.run(scores)

        assert 0.0 < result <= 1.0

    def test_list_input_converted_to_ndarray(self):
        """
        测试目的: list 输入自动转换为 ndarray
        输入: Python list [0.1, 0.2, 5.0, 4.8]
        输出: float
        预期: 正常计算
        """
        scores = [0.1, 0.2, 5.0, 4.8]

        op = SelfEvaluation()
        result = op.run(scores)

        assert isinstance(result, float)

    def test_large_sample_size(self):
        """
        测试目的: 大样本量场景
        输入: 10000 个样本（90%正常~0.1，10%异常~5-10）
        输出: float
        预期: 正常计算，结果 > 0.5
        """
        rng = np.random.RandomState(42)
        # 90% 正常样本分数 ~0.1，10% 异常样本分数 ~5
        scores = np.concatenate([
            rng.rand(9000) * 0.2,
            rng.rand(1000) * 5 + 5
        ])

        op = SelfEvaluation()
        result = op.run(scores)

        assert 0.0 < result <= 1.0
        assert result > 0.5  # 有明显差异

    def test_multidimensional_input_flattened(self):
        """
        测试目的: 多维输入自动展平
        输入: 二维数组 [[0.1, 0.2], [5.0, 4.8]]
        输出: float
        预期: 正常计算（展平后）
        """
        scores = np.array([[0.1, 0.2], [5.0, 4.8]])

        op = SelfEvaluation()
        result = op.run(scores)

        assert isinstance(result, float)

    def test_very_small_mean(self):
        """
        测试目的: 绝对均值极小（< 1e-10）时触发零均值保护
        输入: 极小分数 [1e-15, -1e-15, 2e-15, -2e-15]
        输出: float
        预期: 返回 0.0（零均值保护）
        """
        scores = np.array([1e-15, -1e-15, 2e-15, -2e-15])

        op = SelfEvaluation()
        result = op.run(scores)

        assert result == 0.0

    def test_two_element_different(self):
        """
        测试目的: 两个不同元素时自评估分数 > 0.5
        输入: [0.0, 1.0]
        输出: float
        预期: result > 0.5（有差异时 CV > 0）
        """
        scores = np.array([0.0, 1.0])

        op = SelfEvaluation()
        result = op.run(scores)

        assert result > 0.5

    def test_tuple_input_converted_to_ndarray(self):
        """
        测试目的: tuple 输入自动转换为 ndarray
        输入: Python tuple (0.1, 0.2, 5.0)
        输出: float
        预期: 正常计算
        """
        scores = (0.1, 0.2, 5.0)

        op = SelfEvaluation()
        result = op.run(scores)

        assert isinstance(result, float)
        assert 0.0 < result <= 1.0


# ============================================================================
# 配置测试
# ============================================================================

class TestSelfEvaluationConfig:
    """配置测试"""

    def test_custom_main_scores(self):
        """
        测试目的: 自定义 main_scores
        输入: main_scores={"custom": "_"}
        输出: dict[str, float]
        预期: scores() 返回 {"custom": float}
        """
        scores = np.array([0.1, 0.2, 5.0])

        op = SelfEvaluation(main_scores={"custom": "_"})
        result_dict = op.scores(scores)

        assert result_dict is not None
        assert "custom" in result_dict

    def test_config_frozen(self):
        """
        测试目的: SelfEvaluationConfig 为 frozen 模式，不可修改
        输入: 尝试修改 config.main_scores
        输出: ValidationError
        预期: 抛出 Exception
        """
        config = SelfEvaluationConfig()
        with pytest.raises(Exception):
            config.main_scores = {"other": "_"}

    def test_config_default_main_scores(self):
        """
        测试目的: 验证 SelfEvaluationConfig 默认配置
        输入: 不传参数创建 SelfEvaluationConfig
        输出: SelfEvaluationConfig 实例
        预期: main_scores={"self_eval": "_"}
        """
        config = SelfEvaluationConfig()
        assert config.main_scores == {"self_eval": "_"}

    def test_config_explicit_main_scores_none(self):
        """
        测试目的: 显式设置 main_scores=None
        输入: main_scores=None
        输出: SelfEvaluationConfig 实例
        预期: config.main_scores is None
        """
        config = SelfEvaluationConfig(main_scores=None)
        assert config.main_scores is None

    def test_config_used_by_operator(self):
        """
        测试目的: 验证配置通过构造函数传递到算子并正确使用
        输入: main_scores={"my_score": "_"}, 分数数组
        输出: dict[str, float]
        预期: scores() 返回 {"my_score": float}，且值等于 run() 结果
        """
        scores = np.array([0.1, 0.2, 5.0])

        op = SelfEvaluation(main_scores={"my_score": "_"})
        run_result = op.run(scores)
        scores_result = op.scores(scores)

        assert scores_result is not None
        assert "my_score" in scores_result
        assert scores_result["my_score"] == run_result


# ============================================================================
# 与 HPO metrics.py 对比测试
# ============================================================================

class TestSelfEvaluationConsistency:
    """与旧实现一致性测试"""

    def test_matches_old_compute_self_eval(self):
        """
        测试目的: 验证新实现与旧 _compute_self_eval 结果一致
        输入: 分数数组 [0.1, 0.2, 0.15, 5.0, 4.8, 5.2]
        输出: float
        预期: 新实现结果与手动计算的旧实现结果差值 < 1e-6
        """
        scores = np.array([0.1, 0.2, 0.15, 5.0, 4.8, 5.2])

        op = SelfEvaluation()
        new_result = op.run(scores)

        # 手动计算旧实现的逻辑
        mean_val = np.mean(np.abs(scores))
        cv = np.std(scores) / mean_val
        old_result = 1.0 / (1.0 + np.exp(-cv))

        assert abs(new_result - old_result) < 1e-6

    def test_name(self):
        """
        测试目的: 验证 name() 返回正确标识
        输入: 无
        预期: 返回 "self_evaluation"
        """
        assert SelfEvaluation.name() == "self_evaluation"
