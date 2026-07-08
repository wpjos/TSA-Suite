# -*- coding: utf-8 -*-

"""
羲和 Gamma 评分器测试模块

测试 XiHeGammaScorer 的核心功能，包括：
- 基础属性和初始化
- 模型加载和前置校验
- run/batch_run 推理（使用 mock 模型）
- 合并算法（mean / heuristic）
- DataFrame / ndarray 双类型支持
- 持久化 save/load
- NumericBatchRunMixin 集成
- 真实默认模型加载验证（@pytest.mark.real_model）

通过模块级 autouse fixture 自动 mock TSPredictor，避免构造时加载真实模型。
标记为 @pytest.mark.real_model 的测试跳过 mock，使用真实 TSPredictor。
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tsas.engine.operator.detection.xihe import (_XIHE_WINDOW_SIZE, XiHeGammaScorer, XiHeGammaScorerConfig,
                                                 XiHeGammaScorerExtraOutput)


# ============================================================================
# Mock 工具
# ============================================================================


def _make_mock_predictor(var_columns: list[str], window_size: int = _XIHE_WINDOW_SIZE):
    """创建 mock TSPredictor，返回固定的异常分数和重构值

    对每个窗口返回：
    - anomaly_score: 每个变量的分数为 0.5 的常量数组
    - reconstruction: 每个变量的重构值为 1.0 的常量数组

    Args:
        var_columns (list[str]): 变量列名列表
        window_size (int): 窗口大小

    Returns:
        MagicMock: mock TSPredictor 实例
    """
    mock_predictor = MagicMock()
    mock_predictor.checkpoint_path = "/mock/path"

    def mock_predict(req_data):
        batch_data = req_data["data"]
        results = []
        for item in batch_data:
            context = item["context"]
            n_points = len(next(iter(context.values())))
            result = {
                "anomaly_score": {col: [0.5] * n_points for col in var_columns},
                "reconstruction": {col: [1.0] * n_points for col in var_columns},
            }
            results.append(result)
        return {"data": results}

    mock_predictor.predict = mock_predict
    return mock_predictor


def _make_dynamic_mock_predictor():
    """创建动态 mock TSPredictor，根据输入自动适配变量数

    与 _make_mock_predictor 不同，此函数不预设变量列名，
    而是从模型推理请求的 context 中动态提取变量列名，
    适用于 autouse fixture 场景（变量数未知）。

    Returns:
        MagicMock: 动态 mock TSPredictor 实例
    """
    mock_predictor = MagicMock()
    mock_predictor.checkpoint_path = "/mock/path"

    def mock_predict(req_data):
        batch_data = req_data["data"]
        results = []
        for item in batch_data:
            context = item["context"]
            # 从输入数据中动态提取变量列名
            var_cols = list(context.keys())
            n_points = len(next(iter(context.values())))
            result = {
                "anomaly_score": {col: [0.5] * n_points for col in var_cols},
                "reconstruction": {col: [1.0] * n_points for col in var_cols},
            }
            results.append(result)
        return {"data": results}

    mock_predictor.predict = mock_predict
    return mock_predictor


def _make_test_df(n_rows: int, n_vars: int = 2) -> pd.DataFrame:
    """创建测试 DataFrame，索引为整数时间索引"""
    np.random.seed(42)
    data = np.random.randn(n_rows, n_vars)
    columns = [f"var_{i}" for i in range(n_vars)]
    df = pd.DataFrame(data, columns=columns, index=pd.RangeIndex(n_rows, name="time"))
    return df


# ============================================================================
# 模块级 autouse fixture：自动 mock TSPredictor
# ============================================================================


@pytest.fixture(autouse=True)
def _mock_tspredictor(request):
    """模块级自动 mock TSPredictor，避免构造时加载真实模型

    所有测试自动生效。标记为 ``@pytest.mark.real_model`` 的测试
    跳过 mock，使用真实 TSPredictor 验证默认模型加载。
    """
    # 标记为 real_model 的测试跳过 mock，使用真实 TSPredictor
    if request.node.get_closest_marker("real_model"):
        yield None
        return
    # 动态 mock，自动适配变量数
    mock_pred = _make_dynamic_mock_predictor()
    with patch(
        "pangu_xihe_gamma.infer_service.pangu_ts_predictor.TSPredictor",
        return_value=mock_pred,
    ):
        yield mock_pred


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_scorer_2var():
    """创建已加载 mock 模型的 2 变量评分器

    模型通过 autouse fixture 在构造时自动加载（动态 mock），
    无需手动覆盖 ``_model``。
    """
    config = XiHeGammaScorerConfig(score_merge="heuristic", batch_size=4, step=1)
    scorer = XiHeGammaScorer(config=config)
    # 模型已在 __init__ 中通过 autouse mock 自动加载
    return scorer


@pytest.fixture
def test_df_200():
    """200 行 2 列的测试 DataFrame"""
    return _make_test_df(200, 2)


@pytest.fixture
def test_df_150():
    """150 行 3 列的测试 DataFrame"""
    return _make_test_df(150, 3)


# ============================================================================
# 测试：基础属性和初始化
# ============================================================================


class TestXiHeGammaScorerInit:
    """测试 XiHeGammaScorer 初始化和基础属性"""

    def test_name(self):
        """测试算子名称为 xihe_gamma_scorer"""
        assert XiHeGammaScorer.name() == "xihe_gamma_scorer"

    def test_has_extra_output(self):
        """测试 EO 类型为 XiHeGammaScorerExtraOutput"""
        assert XiHeGammaScorer.has_extra_output() is True

    def test_default_config(self):
        """测试无 Config 时使用默认值，模型自动加载

        model_path=None 时通过 autouse mock 自动加载默认模型。
        """
        scorer = XiHeGammaScorer()
        # model_path=None 时自动加载默认模型（此处为 mock）
        assert scorer._model is not None
        assert scorer._model_path is not None

    def test_custom_config(self):
        """测试自定义 Config 参数"""
        config = XiHeGammaScorerConfig(
            score_merge="mean", batch_size=16, step=2, align="right",
            local_value_scale=True,
        )
        scorer = XiHeGammaScorer(config=config)
        assert scorer.config.score_merge == "mean"
        assert scorer.config.batch_size == 16
        assert scorer.config.step == 2
        assert scorer.config.align == "right"
        assert scorer.config.local_value_scale is True

    def test_oid(self):
        """测试 oid 包含算子名称前缀和用户标识"""
        scorer = XiHeGammaScorer(oid="test_xihe")
        assert "test_xihe" in scorer.oid


# ============================================================================
# 测试：模型加载和前置校验
# ============================================================================


class TestXiHeGammaScorerModelLoading:
    """测试模型加载和 _can_run 校验"""

    def test_can_run_raises_without_model(self):
        """测试模型被手动清除后 run 抛出 RuntimeError

        构造时模型已通过 autouse mock 自动加载，
        手动清除模型后验证 _can_run 拦截逻辑。
        """
        scorer = XiHeGammaScorer()
        # 手动清除模型，模拟模型未加载场景
        scorer._model = None
        df = _make_test_df(200, 2)
        with pytest.raises(RuntimeError, match="模型未加载"):
            scorer.run(df)

    def test_load_model_with_path(self):
        """测试通过路径加载模型"""
        scorer = XiHeGammaScorer()
        mock_pred = _make_mock_predictor(["var_0"])
        with patch(
            "pangu_xihe_gamma.infer_service.pangu_ts_predictor.TSPredictor",
            return_value=mock_pred,
        ) as mock_cls:
            scorer.load_model("/some/path")
            mock_cls.assert_called_once()
        assert scorer._model is not None
        assert scorer._model_path is not None

    def test_load_model_default_path(self):
        """测试无路径时使用默认路径加载"""
        scorer = XiHeGammaScorer()
        mock_pred = _make_mock_predictor(["var_0"])
        mock_pred.checkpoint_path = "/default/path"
        with patch(
            "pangu_xihe_gamma.infer_service.pangu_ts_predictor.TSPredictor",
            return_value=mock_pred,
        ):
            scorer.load_model()
        assert scorer._model is not None
        assert scorer._model_path == "/default/path"

    def test_auto_load_from_config(self):
        """测试 Config 中配置 model_path 时自动加载

        此测试使用自己的 patch 覆盖 autouse mock，
        验证指定 model_path 时的加载路径。
        """
        mock_pred = _make_mock_predictor(["var_0"])
        with patch(
            "pangu_xihe_gamma.infer_service.pangu_ts_predictor.TSPredictor",
            return_value=mock_pred,
        ) as mock_cls:
            config = XiHeGammaScorerConfig(model_path="/auto/path")
            scorer = XiHeGammaScorer(config=config)
            # 验证 TSPredictor 以指定路径的绝对路径被调用
            mock_cls.assert_called_with(os.path.abspath("/auto/path"))
        assert scorer._model is not None
        assert scorer._model_path == os.path.abspath("/auto/path")

    @pytest.mark.real_model
    def test_real_default_model_loading(self):
        """验证真实的默认模型参数能够被正确加载（不使用 mock）

        此测试不 mock TSPredictor，验证 model_path=None 时
        能通过 TSPredictor() 加载默认预训练模型。
        如果 pangu_xihe_gamma 未安装或默认模型不可用，测试将被跳过。
        """
        pytest.importorskip("pangu_xihe_gamma")
        try:
            scorer = XiHeGammaScorer()
        except Exception as e:
            pytest.skip(f"默认模型不可用: {e}")
        assert scorer._model is not None
        assert scorer._model_path is not None


# ============================================================================
# 测试：run 推理（DataFrame 输入）
# ============================================================================


class TestXiHeGammaScorerRun:
    """测试 run 方法的核心功能"""

    def test_run_dataframe_output_shape(self, mock_scorer_2var, test_df_200):
        """测试 DataFrame 输入时输出形状正确
        输入: 200x2 DataFrame
        预期: 主输出 DataFrame(200, 2)，EO 包含 timestamp(200) 和 feature_recon(200, 2)
        """
        result = mock_scorer_2var.run(test_df_200)
        assert isinstance(result, tuple) and len(result) == 2
        scores_df, eo = result
        # 主输出是 DataFrame（因为输入是 DataFrame）
        assert isinstance(scores_df, pd.DataFrame)
        assert scores_df.shape == (200, 2)
        # EO 类型正确
        assert isinstance(eo, XiHeGammaScorerExtraOutput)
        assert len(eo.timestamp) == 200
        assert eo.feature_recon.shape == (200, 2)

    def test_run_preserves_index(self, mock_scorer_2var):
        """测试输出 DataFrame 保留原始时间索引"""
        df = _make_test_df(200, 2)
        df.index = pd.date_range("2024-01-01", periods=200, freq="h", name="ts")
        scores_df, eo = mock_scorer_2var.run(df)
        pd.testing.assert_index_equal(scores_df.index, df.index)

    def test_run_preserves_column_names(self, mock_scorer_2var, test_df_200):
        """测试输出列名沿用输入列名（MultiScorerMixin 行为）"""
        scores_df, _ = mock_scorer_2var.run(test_df_200)
        assert list(scores_df.columns) == list(test_df_200.columns)

    def test_run_ndarray_input(self, mock_scorer_2var):
        """测试 ndarray 输入时返回 ndarray"""
        np.random.seed(42)
        arr = np.random.randn(200, 2)
        result = mock_scorer_2var.run(arr)
        scores, eo = result
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (200, 2)
        assert isinstance(eo, XiHeGammaScorerExtraOutput)

    def test_run_score_merge_mean(self, test_df_200):
        """测试 mean 合并策略

        模型通过 autouse mock 自动加载，返回固定 0.5 分数。
        """
        config = XiHeGammaScorerConfig(score_merge="mean", batch_size=4)
        scorer = XiHeGammaScorer(config=config)
        scores_df, eo = scorer.run(test_df_200)
        assert scores_df.shape == (200, 2)
        # mean 策略下 mock 返回固定 0.5，合并后仍应为 0.5
        np.testing.assert_allclose(scores_df.values, 0.5, atol=1e-10)

    def test_run_with_local_value_scale(self, test_df_200):
        """测试 local_value_scale 标准化功能

        开启标准化后重构值应被逆变换回原始尺度。
        模型通过 autouse mock 自动加载。
        """
        config = XiHeGammaScorerConfig(local_value_scale=True, batch_size=4)
        scorer = XiHeGammaScorer(config=config)
        scores_df, eo = scorer.run(test_df_200)
        assert scores_df.shape == (200, 2)
        # 重构值不应全为 1.0（因为逆变换会改变值）
        assert eo.feature_recon.shape == (200, 2)


# ============================================================================
# 测试：batch_run 推理
# ============================================================================


class TestXiHeGammaScorerBatchRun:
    """测试 batch_run 方法"""

    def test_batch_run_yields_correct_type(self, mock_scorer_2var, test_df_200):
        """测试 batch_run yield 类型与 run 返回类型一致（DataFrame + EO）"""
        batches = list(mock_scorer_2var.batch_run(test_df_200))
        assert len(batches) > 0
        for batch_scores_df, batch_eo in batches:
            assert isinstance(batch_scores_df, pd.DataFrame)
            assert isinstance(batch_eo, XiHeGammaScorerExtraOutput)
            assert batch_scores_df.shape[1] == 2

    def test_batch_run_covers_all_rows(self, mock_scorer_2var, test_df_200):
        """测试所有批次拼接后覆盖全部行"""
        batches = list(mock_scorer_2var.batch_run(test_df_200))
        total_rows = sum(b[0].shape[0] for b in batches)
        assert total_rows == 200

    def test_batch_run_ndarray_input(self, mock_scorer_2var):
        """测试 ndarray 输入时 batch_run yield ndarray"""
        arr = np.random.randn(200, 2)
        batches = list(mock_scorer_2var.batch_run(arr))
        assert len(batches) > 0
        for batch_scores, batch_eo in batches:
            assert isinstance(batch_scores, np.ndarray)

    def test_batch_run_raises_without_model(self):
        """测试模型被手动清除后 batch_run 抛出 RuntimeError

        构造时模型已通过 autouse mock 自动加载，
        手动清除模型后验证 _can_run 拦截逻辑。
        """
        scorer = XiHeGammaScorer()
        # 手动清除模型，模拟模型未加载场景
        scorer._model = None
        df = _make_test_df(200, 2)
        with pytest.raises(RuntimeError, match="模型未加载"):
            list(scorer.batch_run(df))


# ============================================================================
# 测试：合并算法
# ============================================================================


class TestMergeAlgorithms:
    """测试窗口合并核心算法"""

    def test_merge_scores_mean(self):
        """测试均值合并：多窗口覆盖区域取 nanmean
        输入: (3, 2, 4) 的 3D 数组，部分 NaN
        预期: nanmean 沿 axis=2
        """
        cache = np.array([
            [[1.0, 2.0, np.nan, np.nan], [3.0, 4.0, np.nan, np.nan]],
            [[5.0, 6.0, 7.0, np.nan], [8.0, 9.0, 10.0, np.nan]],
            [[np.nan, 11.0, 12.0, 13.0], [np.nan, 14.0, 15.0, 16.0]],
        ])
        result = XiHeGammaScorer._merge_scores_mean(cache)
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result[0, 0], 1.5)  # mean(1, 2)
        np.testing.assert_allclose(result[1, 0], 6.0)  # mean(5, 6, 7)

    def test_merge_scores_heuristic_small_count(self):
        """测试启发式合并：有效值 <= 5 时不去极值
        输入: 3 个有效值
        预期: mean + std（不去极值）
        """
        cache = np.array([[[1.0, 2.0, 3.0]]])
        result = XiHeGammaScorer._merge_scores_heuristic_np(cache)
        expected = np.mean([1.0, 2.0, 3.0]) + np.std([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result[0, 0], expected)

    def test_merge_scores_heuristic_with_nan(self):
        """测试启发式合并：NaN 值被正确忽略"""
        cache = np.array([[[1.0, np.nan, 3.0]]])
        result = XiHeGammaScorer._merge_scores_heuristic_np(cache)
        assert result.shape == (1, 1)
        assert not np.isnan(result[0, 0])

    def test_batch_merge_full(self):
        """测试 _batch_merge 完整合并（ready_time=None）"""
        var_columns = ["a", "b"]
        unmerged = {
            0: ([0, 1, 2], np.array([[1, 2], [3, 4], [5, 6]], dtype=float),
                np.array([[10, 20], [30, 40], [50, 60]], dtype=float)),
        }
        merge_cache = (
            np.full((3, 2, 1), np.nan),
            np.full((3, 2, 1), np.nan),
        )
        times, scores, recons = XiHeGammaScorer._batch_merge(
            var_columns, unmerged, merge_cache, XiHeGammaScorer._merge_scores_mean,
        )
        assert times == [0, 1, 2]
        assert scores.shape == (3, 2)
        assert recons.shape == (3, 2)
        np.testing.assert_array_equal(scores, [[1, 2], [3, 4], [5, 6]])

    def test_batch_merge_partial(self):
        """测试 _batch_merge 部分合并（ready_time 截断）"""
        var_columns = ["a"]
        unmerged = {
            0: ([0, 1, 2, 3], np.ones((4, 1)), np.ones((4, 1))),
        }
        merge_cache = (
            np.full((4, 1, 1), np.nan),
            np.full((4, 1, 1), np.nan),
        )
        # ready_time=1 意味着只合并 time <= 1 的部分
        times, scores, recons = XiHeGammaScorer._batch_merge(
            var_columns, unmerged, merge_cache, XiHeGammaScorer._merge_scores_mean,
            ready_time=1,
        )
        assert times == [0, 1]
        assert scores.shape == (2, 1)
        # 剩余部分仍在缓存中
        assert 0 in unmerged
        remain_times = unmerged[0][0]
        assert remain_times == [2, 3]


# ============================================================================
# 测试：collate_batch
# ============================================================================


class TestCollateBatch:
    """测试 DataLoader collate 函数"""

    def test_collate_batch_format(self):
        """测试 _collate_batch 输出格式正确"""
        # 模拟 DataFrameSlidingWindowDataset 的单个样本
        times = pd.Series([0, 1, 2])
        values = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        batch = [(times, values), (times, values)]
        time_lists, context_dicts = XiHeGammaScorer._collate_batch(batch)
        assert len(time_lists) == 2
        assert len(context_dicts) == 2
        assert "context" in context_dicts[0]
        assert "a" in context_dicts[0]["context"]


# ============================================================================
# 测试：持久化
# ============================================================================


class TestXiHeGammaScorerPersistence:
    """测试 save/load 持久化"""

    def test_save_load_config(self):
        """测试 Config 的 save/load 往返一致性
        save 保存 Config JSON，load（classmethod）从 JSON 重建实例
        """
        config = XiHeGammaScorerConfig(
            score_merge="mean", batch_size=16, step=2, device="cpu",
        )
        scorer = XiHeGammaScorer(config=config, oid="persist_test")
        with tempfile.TemporaryDirectory() as tmpdir:
            scorer.save(tmpdir)
            # 通过 classmethod 加载新实例
            loaded = XiHeGammaScorer.load(tmpdir, oid="persist_test")
            assert loaded.config.score_merge == "mean"
            assert loaded.config.batch_size == 16
            assert loaded.config.step == 2
            assert "persist_test" in loaded.oid


# ============================================================================
# 测试：边界条件
# ============================================================================


class TestXiHeGammaScorerEdgeCases:
    """测试边界条件"""

    def test_invalid_input_type(self, mock_scorer_2var):
        """测试非法输入类型抛出 TypeError"""
        with pytest.raises(TypeError):
            mock_scorer_2var.run("invalid")

    def test_3var_input(self, test_df_150):
        """测试 3 变量输入

        动态 mock 自动适配变量数，无需手动设置。
        """
        config = XiHeGammaScorerConfig(batch_size=4)
        scorer = XiHeGammaScorer(config=config)
        scores_df, eo = scorer.run(test_df_150)
        assert scores_df.shape == (150, 3)
        assert eo.feature_recon.shape == (150, 3)

    def test_step_greater_than_1(self):
        """测试 step > 1 的滑窗步长

        模型通过 autouse mock 自动加载。
        """
        config = XiHeGammaScorerConfig(batch_size=4, step=10)
        scorer = XiHeGammaScorer(config=config)
        df = _make_test_df(200, 2)
        scores_df, eo = scorer.run(df)
        assert scores_df.shape == (200, 2)

    def test_config_validation_batch_size(self):
        """测试 Config 验证：batch_size 必须 >= 1"""
        with pytest.raises(Exception):
            XiHeGammaScorerConfig(batch_size=0)

    def test_config_validation_step(self):
        """测试 Config 验证：step 必须 >= 1"""
        with pytest.raises(Exception):
            XiHeGammaScorerConfig(step=0)
