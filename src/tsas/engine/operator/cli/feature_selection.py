# -*- coding: utf-8 -*-

"""特征选择器 CLI。

本模块提供 ``python -m tsas.engine.operator.cli feature_selection`` 子命令，支持单个
Selector 的帮助生成、训练、运行和模型加载。
"""

import argparse
from typing import cast

import pandas as pd
from pydantic import BaseModel

from tsas.engine.operator.base import BaseOperator, LearnableOperatorMixin, NumericData
from tsas.engine.operator.cli.common import build_help_subparser, extract_encoding_arg, handle_help, \
    instantiate_operator
from tsas.engine.operator.cli.config_loader import load_config
from tsas.engine.operator.cli.io import ensure_encoding, load_data, save_data, save_json
from tsas.engine.operator.cli.registry import OperatorRegistry

__all__ = ['create_registry', 'main']


def create_registry() -> OperatorRegistry:
    """创建特征选择器注册中心。

    Returns:
        OperatorRegistry: 已完成发现流程的注册中心。
    """
    from tsas.engine.operator.feature.selection.base import BaseFeatureSelectorMixin

    registry = OperatorRegistry(
        base_class=BaseOperator,
        scan_packages=['tsas.engine.operator.feature.selection'],
        filter_fn=lambda cls: issubclass(cls, BaseFeatureSelectorMixin),
    )
    registry.discover()
    return registry


def _build_parser() -> argparse.ArgumentParser:
    """构建命令行解析器。

    Returns:
        argparse.ArgumentParser: ``feature_selection`` 子命令解析器。
    """
    parser = argparse.ArgumentParser(prog='feature_selection', description='特征选择器 CLI')
    subparsers = parser.add_subparsers(dest='command', required=True)

    build_help_subparser(subparsers)

    run_parser = subparsers.add_parser('run', help='运行特征选择器')
    run_parser.add_argument('--input', '-i', required=True, help='输入 CSV/TSV 文件')
    run_parser.add_argument('--config', '-c', required=True, help='选择器配置文件')
    run_parser.add_argument('--output', '-o', required=True, help='选择后特征输出文件')
    run_parser.add_argument('--eo-output', required=True, help='附加输出 JSON 文件')
    run_parser.add_argument('--load', default=None, help='已保存选择器模型目录')

    fit_parser = subparsers.add_parser('fit', help='训练特征选择器')
    fit_parser.add_argument('--input', '-i', required=True, help='训练输入 CSV/TSV 文件')
    fit_parser.add_argument('--config', '-c', required=True, help='选择器配置文件')
    fit_parser.add_argument('--model-dir', '-m', required=True, help='模型保存目录')
    return parser


def _load_operator(config_path: str, registry: OperatorRegistry, load_path: str | None = None) -> BaseOperator:
    """从配置文件或模型目录构造特征选择器。

    配置文件采用与 detection 模块一致的嵌套 dict 格式，``operator`` 字段
    为包含 ``name`` 和 ``config`` 的字典::

        {
            "operator": {
                "name": "column_selector",
                "config": {"input_columns": ["a", "c"]}
            }
        }

    Args:
        config_path (str): 配置文件路径。
        registry (OperatorRegistry): 特征选择器注册中心。
        load_path (str | None): 已保存模型目录。指定时从模型目录
            加载已训练的选择器，忽略配置中的实例参数。

    Returns:
        BaseOperator: 选择器实例。

    Raises:
        ValueError: 配置文件中缺少 ``operator`` 字段时。
    """
    config = load_config(config_path)
    # 提取 operator 嵌套字典（与 detection 模块格式一致）
    op_config = config.get('operator', {})
    if not op_config:
        raise ValueError("配置文件中缺少 'operator' 字段")

    if load_path:
        # 从配置中获取算子类名，通过注册中心查找类后调用 load
        op_name = op_config.get('name')
        if not op_name:
            raise ValueError("配置文件中 operator 字段缺少 'name' 子字段")
        op_cls = registry.get(op_name)
        return cast(type[BaseOperator], op_cls).load(load_path)

    # 委托公共函数完成核心实例化（查找类 → 构造 config → 创建实例）
    return instantiate_operator(op_config, registry)


def _handle_help(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """处理帮助文档生成。

    Args:
        registry (OperatorRegistry): 特征选择器注册中心。
        args (argparse.Namespace): 命令行参数。

    Returns:
        None: 本函数直接输出或写入文档。
    """
    handle_help(registry, args.operator_name)


def _handle_fit(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """处理选择器训练。

    Args:
        registry (OperatorRegistry): 特征选择器注册中心。
        args (argparse.Namespace): 命令行参数。

    Returns:
        None: 本函数将模型写入磁盘。

    Raises:
        TypeError: 非训练型选择器执行 ``fit`` 时抛出。
    """
    data = load_data(args.input)
    operator = _load_operator(args.config, registry)
    if not isinstance(operator, LearnableOperatorMixin):
        raise TypeError(f'{type(operator).__name__} 不支持 fit')
    operator.fit(data, None)
    operator.save(args.model_dir)


def _handle_run(registry: OperatorRegistry, args: argparse.Namespace) -> None:
    """处理选择器运行。

    Args:
        registry (OperatorRegistry): 特征选择器注册中心。
        args (argparse.Namespace): 命令行参数。

    Returns:
        None: 本函数写出主输出和 EO JSON。
    """
    data = load_data(args.input)
    operator = _load_operator(args.config, registry, load_path=args.load)
    output, eo = cast(tuple[NumericData, BaseModel], operator.run(data))
    save_data(cast(pd.DataFrame, output), args.output)
    save_json(eo.model_dump(mode='json'), args.eo_output)


def main(args: list[str] | None = None) -> None:
    """特征选择器 CLI 入口。

    Args:
        args (list[str] | None): 命令行参数列表。

    Returns:
        None: 本函数通过子命令完成实际操作。
    """
    encoding, remaining = extract_encoding_arg(args)
    ensure_encoding(encoding)
    parsed = _build_parser().parse_args(remaining)
    registry = create_registry()
    if parsed.command == 'help':
        _handle_help(registry, parsed)
    elif parsed.command == 'fit':
        _handle_fit(registry, parsed)
    elif parsed.command == 'run':
        _handle_run(registry, parsed)


if __name__ == '__main__':
    main()
