# -*- coding: utf-8 -*-

"""
点调整评价指标算子测试

测试覆盖:
    1. 基本功能: run() 返回 PA 指标
    2. PA 算法正确性: 验证段级调整逻辑
    3. scores() 方法: 按 main_scores 提取命名标量
    4. 边界条件: 无异常段、全异常段、空样本
    5. 错误处理: 非一维输入、长度不一致

测试约束:
    - 代码覆盖率 > 90%
    - 测试通过率 100%
    - 中文注释说明测试目的、输入、输出和预期结果
"""

import numpy as np
import pytest

from tsas.engine.operator.evaluation.point_adjust import (PointAdjust, PointAdjustResult)


# ============================================================================
# 基本功能测试
# ============================================================================

class TestPointAdjustBasic:
    """基本功能测试"""

    def test_run_returns_complete_result(self):
        """
        测试目的: run() 返回完整 PA 指标
        输入: y_truth=[0,1,1,1,0,0,0], y_predict=[0,0,1,0,0,0,0]
        输出: PointAdjustResult
        预期: n_samples=7, n_anomaly_segments=1, pa_tp=1, pa_fn=0
        """
        # 异常段: [1,2,3]（标签为1），其他为正常（标签为0）
        y_truth = np.array([0, 1, 1, 1, 0, 0, 0])
        # 预测: 在异常段内只检出了位置2，按PA规则整段算检出
        y_predict = np.array([0, 0, 1, 0, 0, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert isinstance(result, PointAdjustResult)
        assert result.n_samples == 7
        assert result.n_anomaly_segments == 1
        assert result.pa_tp == 1  # 异常段内有检出
        assert result.pa_fn == 0

    def test_pa_f1_perfect_detection(self):
        """
        测试目的: 完美检测时 PA-F1 = 1.0
        输入: 异常段内所有点都被检出，无误报
        预期: pa_f1 = 1.0, pa_precision = 1.0, pa_recall = 1.0
        """
        y_truth = np.array([0, 1, 1, 1, 0, 0, 0])
        y_predict = np.array([0, 1, 1, 1, 0, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.pa_tp == 1
        assert result.pa_fp == 0
        assert result.pa_fn == 0
        assert result.pa_precision == 1.0
        assert result.pa_recall == 1.0
        assert result.pa_f1 == 1.0

    def test_pa_f1_partial_detection_in_segment(self):
        """
        测试目的: 异常段内部分检出仍算 TP
        输入: 异常段 [1,2,3]，只在位置1检出
        预期: pa_tp = 1（段内只要有一个检出就整段算TP）
        """
        y_truth = np.array([0, 1, 1, 1, 0])
        y_predict = np.array([0, 1, 0, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.pa_tp == 1
        assert result.pa_fn == 0

    def test_pa_fn_when_no_detection_in_segment(self):
        """
        测试目的: 异常段内完全未检出算 FN
        输入: 异常段 [1,2,3]，完全没有检出
        输出: PointAdjustResult
        预期: pa_fn = 1
        """
        y_truth = np.array([0, 1, 1, 1, 0])
        y_predict = np.array([0, 0, 0, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.pa_tp == 0
        assert result.pa_fn == 1

    def test_pa_fp_from_false_alarms(self):
        """
        测试目的: 非异常段的误报算 FP
        输入: 正常段 [0,4] 内有误报
        输出: PointAdjustResult
        预期: pa_fp 统计正常区域内的误报点数
        """
        y_truth = np.array([0, 1, 1, 1, 0, 0])
        y_predict = np.array([1, 1, 0, 0, 1, 0])  # 位置0和4误报

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        # 位置0（正常但预测异常）+ 位置4（正常但预测异常）= 2个FP
        # 异常段[1,2,3]内位置1有检出 → TP=1
        assert result.pa_fp == 2
        assert result.pa_tp == 1


# ============================================================================
# PA 算法正确性测试
# ============================================================================

class TestPointAdjustAlgorithm:
    """PA 算法正确性测试"""

    def test_multiple_anomaly_segments(self):
        """
        测试目的: 多个异常段场景
        输入: 两个异常段 [1,2] 和 [5,6]
        输出: PointAdjustResult
        预期: 分别计算各段的 TP/FN
        """
        y_truth = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        # 第一个段[1,2]有检出（位置2），第二个段[5,6]无检出
        y_predict = np.array([0, 0, 1, 0, 0, 0, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 2
        assert result.pa_tp == 1  # 第一个段检出
        assert result.pa_fn == 1  # 第二个段未检出

    def test_segment_boundary_detection(self):
        """
        测试目的: 段边界检测正确
        输入: 两个单点异常段
        输出: PointAdjustResult
        预期: 单点异常段也能正确识别
        """
        y_truth = np.array([0, 1, 0, 1, 0])  # 两个单点异常
        y_predict = np.array([0, 1, 0, 0, 0])  # 只检出第一个

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 2
        assert result.pa_tp == 1
        assert result.pa_fn == 1

    def test_adjacent_segments_merged(self):
        """
        测试目的: 相邻异常点合并为连续段
        输入: 连续异常点 y_truth=[1,1,1,0,0]
        输出: PointAdjustResult
        预期: 合并为一个段
        """
        y_truth = np.array([1, 1, 1, 0, 0])  # 一个连续段
        y_predict = np.array([1, 0, 0, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 1
        assert result.pa_tp == 1  # 位置0有检出，整段算TP

    def test_pa_f1_formula(self):
        """
        测试目的: 验证 PA-F1 公式
        输入: 已知 TP=1, FP=1, FN=1
        输出: float (pa_f1 值)
        预期: PA-F1 = 2*P*R/(P+R) = 0.5
        """
        y_truth = np.array([0, 1, 1, 0, 1, 0])
        y_predict = np.array([1, 1, 0, 0, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        # 段1[1,2]: 有检出（位置1）→ TP=1
        # 段2[4]: 无检出 → FN=1
        # FP: 位置0 → FP=1
        expected_precision = 1 / (1 + 1)  # TP/(TP+FP)
        expected_recall = 1 / (1 + 1)  # TP/(TP+FN)
        expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)

        assert abs(result.pa_precision - expected_precision) < 1e-6
        assert abs(result.pa_recall - expected_recall) < 1e-6
        assert abs(result.pa_f1 - expected_f1) < 1e-6


# ============================================================================
# scores() 方法测试
# ============================================================================

class TestPointAdjustScores:
    """scores() 方法测试"""

    def test_scores_returns_dict(self):
        """
        测试目的: scores() 返回按 main_scores 映射的字典
        输入: 默认配置 main_scores={"pa_f1": "pa_f1", "pa_recall": "pa_recall"}
        输出: dict[str, float]
        预期: 返回 {"pa_f1": float, "pa_recall": float}
        """
        y_truth = np.array([0, 1, 1, 0])
        y_predict = np.array([0, 1, 0, 0])

        op = PointAdjust()
        scores = op.scores((y_truth, y_predict))

        assert scores is not None
        assert "pa_f1" in scores
        assert "pa_recall" in scores

    def test_scores_matches_run_values(self):
        """
        测试目的: scores() 提取的值与 run() 结果一致
        输入: 同上
        输出: dict[str, float]
        预期: scores["pa_f1"] == result.pa_f1
        """
        y_truth = np.array([0, 1, 1, 0])
        y_predict = np.array([0, 1, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))
        scores = op.scores((y_truth, y_predict))

        assert scores["pa_f1"] == result.pa_f1
        assert scores["pa_recall"] == result.pa_recall

    def test_scores_with_custom_main_scores(self):
        """
        测试目的: 自定义 main_scores 提取不同指标
        输入: main_scores={"pa_precision": "pa_precision"}
        输出: dict[str, float]
        预期: 返回 {"pa_precision": float}
        """
        y_truth = np.array([0, 1, 1, 0])
        y_predict = np.array([0, 1, 0, 0])

        op = PointAdjust(main_scores={"pa_precision": "pa_precision"})
        scores = op.scores((y_truth, y_predict))

        assert scores is not None
        assert "pa_precision" in scores


# ============================================================================
# 边界条件测试
# ============================================================================

class TestPointAdjustEdgeCases:
    """边界条件测试"""

    def test_no_anomaly_segments(self):
        """
        测试目的: 无异常段场景
        输入: y_truth 全为 0
        输出: PointAdjustResult
        预期: n_anomaly_segments=0，pa_tp=pa_fn=0
        """
        y_truth = np.array([0, 0, 0, 0])
        y_predict = np.array([0, 1, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 0
        assert result.pa_tp == 0
        assert result.pa_fn == 0
        # 但有误报
        assert result.pa_fp == 1

    def test_all_anomaly_segments(self):
        """
        测试目的: 全异常段场景
        输入: y_truth 全为 1
        输出: PointAdjustResult
        预期: 一个大异常段
        """
        y_truth = np.array([1, 1, 1, 1])
        y_predict = np.array([1, 1, 1, 1])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 1
        assert result.pa_tp == 1
        assert result.pa_fp == 0

    def test_empty_input(self):
        """
        测试目的: 空输入场景
        输入: 空数组
        输出: PointAdjustResult
        预期: n_samples=0
        """
        y_truth = np.array([])
        y_predict = np.array([])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 0
        assert result.n_anomaly_segments == 0

    def test_single_sample(self):
        """
        测试目的: 单样本场景
        输入: 长度为1的数组
        输出: PointAdjustResult
        预期: n_samples=1, n_anomaly_segments=1, pa_tp=1
        """
        y_truth = np.array([1])
        y_predict = np.array([1])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 1
        assert result.n_anomaly_segments == 1
        assert result.pa_tp == 1


# ============================================================================
# 错误处理测试
# ============================================================================

class TestPointAdjustErrors:
    """错误处理测试"""

    def test_non_1d_input_raises_error(self):
        """
        测试目的: 非一维输入抛出 ValueError
        输入: 二维数组
        输出: ValueError
        预期: 抛出 ValueError，消息包含 "必须为一维数组"
        """
        y_truth = np.array([[0, 1], [1, 0]])
        y_predict = np.array([[0, 1], [1, 0]])

        op = PointAdjust()
        with pytest.raises(ValueError, match="必须为一维数组"):
            op.run((y_truth, y_predict))

    def test_length_mismatch_raises_error(self):
        """
        测试目的: y_truth 和 y_predict 长度不一致时抛出 ValueError
        输入: y_truth 长度 4, y_predict 长度 3
        输出: ValueError
        预期: 抛出 ValueError，消息包含 "长度不一致"
        """
        y_truth = np.array([0, 1, 0, 1])
        y_predict = np.array([0, 1, 0])

        op = PointAdjust()
        with pytest.raises(ValueError, match="长度不一致"):
            op.run((y_truth, y_predict))

    def test_list_input_converted_to_ndarray(self):
        """
        测试目的: list 输入自动转换为 ndarray
        输入: y_truth/y_predict 为 Python list
        输出: PointAdjustResult
        预期: 正常计算，n_samples=4
        """
        y_truth = [0, 1, 1, 0]
        y_predict = [0, 1, 0, 0]

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_samples == 4


# ============================================================================
# 配置测试
# ============================================================================

class TestPointAdjustConfig:
    """配置测试"""

    def test_positive_label_explicit(self):
        """
        测试目的: 显式指定 positive_label
        输入: positive_label=2, y_truth=[0,2,2,0], y_predict=[0,2,0,0]
        输出: PointAdjustResult
        预期: 使用 2 作为异常标签，n_anomaly_segments=1, pa_tp=1
        """
        y_truth = np.array([0, 2, 2, 0])
        y_predict = np.array([0, 2, 0, 0])

        op = PointAdjust(positive_label=2)
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 1
        assert result.pa_tp == 1

    def test_positive_label_non_standard_value(self):
        """
        测试目的: 使用非 0/1 的 positive_label（如 99）
        输入: positive_label=99, y_truth=[0,99,99,0,99]
        输出: PointAdjustResult
        预期: 使用 99 作为异常标签，正确识别异常段 [1,2] 和 [4]
        """
        y_truth = np.array([0, 99, 99, 0, 99])
        y_predict = np.array([0, 99, 0, 0, 0])

        op = PointAdjust(positive_label=99)
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 2  # [1,2] 和 [4]
        assert result.pa_tp == 1  # 第一个段检出，第二个段未检出
        assert result.pa_fn == 1

    def test_positive_label_zero(self):
        """
        测试目的: positive_label=0 时将 0 视为异常
        输入: positive_label=0, y_truth=[0,1,0,1,0]
        输出: PointAdjustResult
        预期: 0 被视为异常标签，正确识别 3 个异常段
        """
        y_truth = np.array([0, 1, 0, 1, 0])
        y_predict = np.array([0, 1, 0, 0, 1])

        op = PointAdjust(positive_label=0)
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 3  # [0], [2], [4]
        assert result.pa_tp == 2  # 位置 0 和 2 检出，位置 4 未检出
        assert result.pa_fn == 1

    def test_scores_with_none_main_scores(self):
        """
        测试目的: main_scores=None 时 scores() 返回 None
        输入: main_scores=None
        输出: None
        预期: scores() 返回 None
        """
        y_truth = np.array([0, 1, 1, 0])
        y_predict = np.array([0, 1, 0, 0])

        op = PointAdjust(main_scores=None)
        scores = op.scores((y_truth, y_predict))

        assert scores is None

    def test_main_scores_custom_keys(self):
        """
        测试目的: 自定义 main_scores 提取不同 PA 指标
        输入: main_scores={"pa_precision": "pa_precision"}
        输出: dict[str, float]
        预期: 返回 {"pa_precision": float}
        """
        y_truth = np.array([0, 1, 1, 0])
        y_predict = np.array([0, 1, 0, 0])

        op = PointAdjust(main_scores={"pa_precision": "pa_precision"})
        scores = op.scores((y_truth, y_predict))

        assert scores is not None
        assert "pa_precision" in scores


# ============================================================================
# 连续异常段边界测试
# ============================================================================

class TestPointAdjustSegmentBoundaries:
    """连续异常段边界测试"""

    def test_anomaly_at_start_of_array(self):
        """
        测试目的: 异常段位于数组起始位置的边界情况
        输入: y_truth=[1,1,0,0], y_predict=[1,0,0,0]
        输出: PointAdjustResult
        预期: 异常段 [0,1] 被正确识别，位置0检出 → pa_tp=1
        """
        y_truth = np.array([1, 1, 0, 0])
        y_predict = np.array([1, 0, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 1
        assert result.pa_tp == 1

    def test_anomaly_at_end_of_array(self):
        """
        测试目的: 异常段位于数组末尾的边界情况
        输入: y_truth=[0,0,1,1], y_predict=[0,0,0,1]
        输出: PointAdjustResult
        预期: 异常段 [2,3] 被正确识别，位置3检出 → pa_tp=1
        """
        y_truth = np.array([0, 0, 1, 1])
        y_predict = np.array([0, 0, 0, 1])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 1
        assert result.pa_tp == 1

    def test_full_array_single_segment(self):
        """
        测试目的: 整个数组为单个异常段
        输入: y_truth=[1,1,1,1,1], y_predict=[0,1,0,0,1]
        输出: PointAdjustResult
        预期: 单个异常段覆盖全部索引，2个检出点 → pa_tp=1
        """
        y_truth = np.array([1, 1, 1, 1, 1])
        y_predict = np.array([0, 1, 0, 0, 1])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 1
        assert result.pa_tp == 1  # 段内至少一个检出
        assert result.pa_fp == 0  # 无异常段外的区域

    def test_alternating_segments(self):
        """
        测试目的: 交替出现的短异常段
        输入: y_truth=[1,0,1,0,1,0], y_predict=[1,0,0,0,1,0]
        输出: PointAdjustResult
        预期: 3个单点异常段，段[0]和段[4]检出 → pa_tp=2, pa_fn=1
        """
        y_truth = np.array([1, 0, 1, 0, 1, 0])
        y_predict = np.array([1, 0, 0, 0, 1, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 3
        assert result.pa_tp == 2
        assert result.pa_fn == 1

    def test_fp_in_normal_region_between_segments(self):
        """
        测试目的: 异常段之间的正常区域有误报
        输入: y_truth=[1,1,0,1,1], y_predict=[1,1,1,1,0]
        输出: PointAdjustResult
        预期: 2个异常段 [0,1] 和 [3,4]，位置2误报 → pa_fp=1
        """
        y_truth = np.array([1, 1, 0, 1, 1])
        y_predict = np.array([1, 1, 1, 1, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 2
        assert result.pa_tp == 2  # 两个段都有检出
        assert result.pa_fp == 1  # 位置2是正常区域但预测异常

    def test_no_true_anomaly_all_false_alarms(self):
        """
        测试目的: 无真实异常但全部预测为异常
        输入: y_truth=[0,0,0,0], y_predict=[1,1,1,1]
        输出: PointAdjustResult
        预期: n_anomaly_segments=0, pa_tp=0, pa_fp=4
        """
        y_truth = np.array([0, 0, 0, 0])
        y_predict = np.array([1, 1, 1, 1])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.n_anomaly_segments == 0
        assert result.pa_tp == 0
        assert result.pa_fp == 4

    def test_safe_divide_zero_precision(self):
        """
        测试目的: PA-Precision 分母为零时的安全处理
        输入: y_truth=[1,1], y_predict=[0,0]（无TP无FP）
        输出: PointAdjustResult
        预期: pa_tp=0, pa_fn=1, pa_fp=0, pa_precision=0.0（分母为0）
        """
        y_truth = np.array([1, 1])
        y_predict = np.array([0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        assert result.pa_tp == 0
        assert result.pa_fn == 1
        assert result.pa_fp == 0
        assert result.pa_precision == 0.0  # TP+FP=0 → 默认值 0.0
        assert result.pa_recall == 0.0
        assert result.pa_f1 == 0.0


# ============================================================================
# 冻结结果测试
# ============================================================================

class TestPointAdjustFrozen:
    """冻结结果测试"""

    def test_result_is_frozen(self):
        """
        测试目的: 结果对象不可修改（frozen）
        输入: 尝试修改 result.pa_f1 = 0.99
        输出: ValidationError
        预期: 抛出 Exception
        """
        y_truth = np.array([0, 1, 1, 0])
        y_predict = np.array([0, 1, 0, 0])

        op = PointAdjust()
        result = op.run((y_truth, y_predict))

        with pytest.raises(Exception):
            result.pa_f1 = 0.99

    def test_name(self):
        """
        测试目的: 验证 name() 返回正确标识
        输入: 无
        预期: 返回 "point_adjust"
        """
        assert PointAdjust.name() == "point_adjust"
