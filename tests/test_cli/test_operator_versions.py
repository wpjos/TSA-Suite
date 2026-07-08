# -*- coding: utf-8 -*-

"""
具体算子版本号自动化测试

通过 CLI 注册中心的自动发现机制收集全部算子类，确保新增算子无需手动修改
测试代码即可被自动覆盖。

测试范围：
    1. 所有算子的 version() 返回值为合法非空 tuple[int, ...]
    2. 所有算子的 min_compatible_version() <= version() 不变量成立

设计要点：
    - 延迟加载：通过 session-scoped fixture 在测试真正执行时才触发四个
      CLI 子模块的 create_registry() 扫描，避免收集阶段的额外开销。
    - 错误收集：循环内不直接断言，而是收集全部错误信息后统一断言，
      确保一次运行能暴露所有算子的版本问题。
    - 与 CLI 一致：直接复用各 CLI 子模块的 create_registry()，保证测试
      覆盖范围与 CLI 可见范围完全一致。

注意：
    基础类（BaseOperator）的 version 机制测试（如 _validate_version_tuple、
    __init_subclass__ 校验、save/load 版本持久化等）位于
    ``tests/test_engine_operator/test_version.py``。
"""

import pytest


# ============================================================================
# session-scoped fixture：延迟收集全部算子
# ============================================================================

@pytest.fixture(scope="session")
def all_operators() -> dict[str, type]:
    """通过 CLI 注册中心自动发现全部算子类

    在测试执行阶段（而非收集阶段）依次调用四个 CLI 子模块的
    ``create_registry()`` 函数，合并返回 ``{算子名称: 算子类}`` 映射。

    复用 CLI 子模块的注册逻辑，保证测试覆盖的算子集合与 CLI 用户可见
    的算子集合完全一致。如果某个算子因缺少可选依赖而 import 失败，
    registry 的 discover 会静默跳过它，该算子不会被测试覆盖——但它的
    功能测试用例也会同样失败，开发者修复后本测试自然恢复覆盖。

    Returns:
        dict[str, type]: 全部已注册算子的 {name(): cls} 映射，按名称排序
    """
    from tsas.engine.operator.cli.detection import create_registry as _det_factory
    from tsas.engine.operator.cli.evaluation import create_registry as _eval_factory
    from tsas.engine.operator.cli.feature_construction import create_registry as _fc_factory
    from tsas.engine.operator.cli.feature_selection import create_registry as _fs_factory

    result: dict[str, type] = {}
    for factory in (_det_factory, _eval_factory, _fc_factory, _fs_factory):
        registry = factory()
        result.update(registry.list_all())
    print(f"Found {len(result)} operators in total")
    return dict(sorted(result.items()))


# ============================================================================
# 版本号格式与不变量测试
# ============================================================================

class TestOperatorVersions:
    """所有具体算子的版本号一致性测试

    通过自动发现机制收集全部算子后，逐个校验版本号的格式合法性和
    min_compatible_version 不变量。采用"收集后统一断言"策略，
    确保一次运行能暴露所有算子的问题。
    """

    def test_all_operator_versions_are_valid(self, all_operators: dict[str, type]):
        """
        目的：验证所有算子的 version() 返回值为合法非空 tuple[int, ...]
        输入：通过 CLI 注册中心自动发现的全部算子类
        预期：每个算子的 version() 返回值满足：
            - 类型为 tuple
            - 元组非空
            - 所有元素均为 int
        策略：循环内收集错误，循环后统一断言，确保一次暴露所有问题
        """
        errors: list[str] = []

        for name, cls in all_operators.items():
            v = cls.version()

            # 校验 1：类型必须为 tuple
            if not isinstance(v, tuple):
                errors.append(
                    f"{name} ({cls.__name__}): version() 返回 "
                    f"{type(v).__name__}，期望 tuple"
                )
                continue  # 后续校验无意义，跳过

            # 校验 2：元组不能为空
            if len(v) == 0:
                errors.append(f"{name} ({cls.__name__}): version() 返回空元组")
                continue

            # 校验 3：所有元素必须为 int
            for i, element in enumerate(v):
                if not isinstance(element, int):
                    errors.append(
                        f"{name} ({cls.__name__}): version() 第 {i} 个元素 "
                        f"为 {type(element).__name__}（值: {element!r}），期望 int"
                    )

        # 统一断言，一次输出全部格式问题
        assert not errors, (
            f"发现 {len(errors)} 个算子版本格式问题:\n" + "\n".join(errors)
        )

    def test_min_compatible_version_invariant(self, all_operators: dict[str, type]):
        """
        目的：验证所有算子的 min_compatible_version() <= version() 不变量成立
        输入：通过 CLI 注册中心自动发现的全部算子类
        预期：每个算子的 min_compatible_version() 返回值 <= version() 返回值
        策略：循环内收集错误，循环后统一断言，确保一次暴露所有问题

        注意：
            本测试仅对 version() 格式合法的算子执行不变量校验。
            格式不合法的算子由 test_all_operator_versions_are_valid 负责。
        """
        errors: list[str] = []

        for name, cls in all_operators.items():
            v = cls.version()

            # 跳过格式不合法的算子（由上一个测试覆盖）
            if not isinstance(v, tuple) or len(v) == 0:
                continue
            if not all(isinstance(e, int) for e in v):
                continue

            min_v = cls.min_compatible_version()

            # 校验 min_compatible_version 自身格式
            if not isinstance(min_v, tuple):
                errors.append(
                    f"{name} ({cls.__name__}): min_compatible_version() 返回 "
                    f"{type(min_v).__name__}，期望 tuple"
                )
                continue

            if len(min_v) == 0:
                errors.append(
                    f"{name} ({cls.__name__}): min_compatible_version() 返回空元组"
                )
                continue

            if not all(isinstance(e, int) for e in min_v):
                errors.append(
                    f"{name} ({cls.__name__}): min_compatible_version() 含非 int 元素: {min_v}"
                )
                continue

            # 不变量校验：min_compatible_version() <= version()
            if min_v > v:
                errors.append(
                    f"{name} ({cls.__name__}): "
                    f"min_compatible_version()={min_v} > version()={v}"
                )

        # 统一断言
        assert not errors, (
            f"发现 {len(errors)} 个版本不变量问题:\n" + "\n".join(errors)
        )

    def test_discovered_operator_count_positive(self, all_operators: dict[str, type]):
        """
        目的：验证自动发现的算子数量大于 0，确保发现机制正常工作
        输入：通过 CLI 注册中心自动发现的全部算子类
        预期：发现的算子数量 > 0（四个模块合计至少应有数十个算子）

        注意：
            此测试作为"防遗漏"检查。如果发现数量为 0，说明注册中心扫描
            逻辑可能出现问题（如包路径变更、全部 import 失败等）。
            单个算子因依赖缺失被跳过是可接受的（该算子自身的功能测试
            也会失败），但全部算子都发现不到则说明机制本身有故障。
        """
        assert len(all_operators) > 0, (
            "自动发现的算子数量为 0，请检查 CLI 注册中心的扫描配置是否正确"
        )
