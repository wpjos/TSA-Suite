# -*- coding: utf-8 -*-

"""特征选择器 CLI 测试。

测试覆盖：
    - ``run`` 子命令：静态列选择器（无需训练）
    - ``fit`` + ``run`` 子命令：方差阈值选择器（需训练 + 加载）
    - 统一 CLI 入口模块名验证
    - 配置格式错误时的异常处理

配置文件格式采用与 detection 模块一致的嵌套 dict::

    {"operator": {"name": "column_selector", "config": {...}}}
"""

import json

import pandas as pd
import pytest

from tsas.engine.operator.cli import __main__ as cli_main
from tsas.engine.operator.cli import feature_selection


# ============================================================================
# run 子命令测试
# ============================================================================

class TestSelectionCliRun:
    """特征选择器 ``run`` 子命令测试"""

    def test_run_column_selector(self, tmp_path):
        """
        目的：验证 CLI 可运行静态列选择器并写出主输出与 EO
        输入：3 列 CSV，配置选择 c 和 a 两列
        预期：输出 CSV 仅包含 c 和 a 两列；EO 的 selected_indices 为 [2, 0]
        """
        input_path = tmp_path / 'input.csv'
        config_path = tmp_path / 'config.json'
        output_path = tmp_path / 'output.csv'
        eo_path = tmp_path / 'eo.json'

        pd.DataFrame({'a': [1], 'b': [2], 'c': [3]}).to_csv(input_path, index=False)
        config_path.write_text(
            json.dumps({
                'operator': {
                    'name': 'column_selector',
                    'config': {'input_columns': ['c', 'a']},
                }
            }),
            encoding='utf-8',
        )

        feature_selection.main([
            'run', '--input', str(input_path), '--config', str(config_path),
            '--output', str(output_path), '--eo-output', str(eo_path),
        ])

        assert pd.read_csv(output_path).to_dict(orient='list') == {'c': [3], 'a': [1]}
        assert json.loads(eo_path.read_text(encoding='utf-8')) == {'selected_indices': [2, 0]}


# ============================================================================
# fit + run 子命令测试
# ============================================================================

class TestSelectionCliFitAndRun:
    """特征选择器 ``fit`` + ``run`` 子命令测试"""

    def test_fit_and_run_loaded_variance_selector(self, tmp_path):
        """
        目的：验证 CLI 可训练方差阈值选择器并加载运行
        输入：3 列 CSV（a 列方差为 0，b 列方差 > 0），阈值 0.1
        预期：fit 保存模型后，run --load 加载模型，输出仅保留 b 列；
              EO 的 selected_indices 为 [1]
        """
        input_path = tmp_path / 'input.csv'
        config_path = tmp_path / 'config.json'
        model_dir = tmp_path / 'model'
        output_path = tmp_path / 'output.csv'
        eo_path = tmp_path / 'eo.json'

        pd.DataFrame({'a': [1, 1, 1], 'b': [1, 2, 3]}).to_csv(input_path, index=False)
        config_path.write_text(
            json.dumps({
                'operator': {
                    'name': 'variance_threshold_selector',
                    'config': {'threshold': 0.1},
                }
            }),
            encoding='utf-8',
        )

        # 训练
        feature_selection.main([
            'fit', '--input', str(input_path), '--config', str(config_path),
            '--model-dir', str(model_dir),
        ])

        # 加载并运行
        feature_selection.main([
            'run', '--input', str(input_path), '--config', str(config_path),
            '--load', str(model_dir), '--output', str(output_path),
            '--eo-output', str(eo_path),
        ])

        assert list(pd.read_csv(output_path).columns) == ['b']
        assert json.loads(eo_path.read_text(encoding='utf-8'))['selected_indices'] == [1]


# ============================================================================
# 配置格式错误处理测试
# ============================================================================

class TestSelectionCliConfigErrors:
    """特征选择器配置格式错误处理测试"""

    def test_missing_operator_field_raises(self, tmp_path):
        """
        目的：验证配置文件缺少 ``operator`` 字段时抛出 ValueError
        输入：空 JSON 对象 ``{}``
        预期：ValueError，消息包含 "缺少 'operator' 字段"
        """
        config_path = tmp_path / 'config.json'
        output_path = tmp_path / 'output.csv'
        eo_path = tmp_path / 'eo.json'
        input_path = tmp_path / 'input.csv'

        pd.DataFrame({'a': [1]}).to_csv(input_path, index=False)
        config_path.write_text(json.dumps({}), encoding='utf-8')

        with pytest.raises(ValueError, match="缺少 'operator' 字段"):
            feature_selection.main([
                'run', '--input', str(input_path), '--config', str(config_path),
                '--output', str(output_path), '--eo-output', str(eo_path),
            ])

    def test_missing_name_in_operator_raises(self, tmp_path):
        """
        目的：验证 ``operator`` 字段中缺少 ``name`` 子字段时抛出 ValueError
        输入：``{"operator": {"config": {...}}}``（无 name）
        预期：ValueError，消息包含 "缺少 'name' 子字段"
        """
        config_path = tmp_path / 'config.json'
        output_path = tmp_path / 'output.csv'
        eo_path = tmp_path / 'eo.json'
        input_path = tmp_path / 'input.csv'

        pd.DataFrame({'a': [1]}).to_csv(input_path, index=False)
        config_path.write_text(
            json.dumps({'operator': {'config': {'input_columns': ['a']}}}),
            encoding='utf-8',
        )

        with pytest.raises(ValueError, match="缺少 'name' 子字段"):
            feature_selection.main([
                'run', '--input', str(input_path), '--config', str(config_path),
                '--load', str(tmp_path / 'nonexistent'),  # 触发 load_path 分支
                '--output', str(output_path), '--eo-output', str(eo_path),
            ])


# ============================================================================
# 统一 CLI 入口测试
# ============================================================================

class TestUnifiedCliEntry:
    """统一 CLI 入口测试"""

    def test_unified_cli_exposes_feature_selection(self, capsys):
        """
        目的：验证统一 CLI 入口暴露的特征选择器模块名为 feature_selection
        输入：``python -m tsas.engine.operator.cli help``
        预期：输出中包含 "feature_selection"
        """
        cli_main.main(['help'])

        usage = capsys.readouterr().out
        assert 'feature_selection' in usage
