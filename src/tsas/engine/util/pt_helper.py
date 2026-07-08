# -*- coding: utf-8 -*-

"""
PyTorch辅助工具模块

该模块提供了一系列用于简化PyTorch设备管理、随机种子设置和数据类型支持检查的实用函数。
主要包括设备自动检测与分配、全局随机种子设置、特定设备数据类型支持检查等功能。

Functions:
    get_torch_device: 获取PyTorch计算设备对象
    set_global_seed: 设置全局随机种子以确保实验可重现性
    is_bf16_and_fp16_supported: 检查指定设备是否支持bfloat16和float16数据类型
"""

import logging
import os
import random
from contextlib import contextmanager

import torch

from tsas.basic.util.exception import RunError

__all__ = [
    "get_torch_device",
    "set_global_seed",
    "is_bf16_and_fp16_supported",
    "torch_device_scope"
]

logger = logging.getLogger(__name__)


def get_torch_device(
    *,
    device: str | dict | torch.device | None = None,
    device_name: str | None = None,
    gpu: str | int | None = None,
    xpu: str | int | None = None,
    npu: str | int | None = None,
    raise_error: bool = True
) -> torch.device:
    """
    获取PyTorch计算设备对象(torch.device)。
    
    支持多种方式指定设备，包括直接传入设备对象、设备名称、GPU/NPU/XPU索引等，
    并会根据设备可用性自动降级处理（默认）或抛出异常。

    Args:
        device: 设备标识，可以是torch.device对象、设备名称字符串或包含设备配置的字典
        device_name: 设备名称字符串，如"cuda:0"、"cpu"、"npu:0"等
        gpu: GPU设备索引，可以是字符串或整数
        xpu: XPU设备索引，可以是字符串或整数
        npu: NPU设备索引，可以是字符串或整数
        raise_error: 当设备不可用时是否抛出异常，若为False则降级到CPU

    Returns:
        torch.device: PyTorch设备对象

    Raises:
        RunError: 当设备不可用且raise_error为True时抛出
    """
    if device:
        if isinstance(device, torch.device):
            # 如果传递的已经是设备资源对象，直接返回
            return device
        elif isinstance(device, dict):
            # 如果传递的是字典对象，则尝试解包
            return get_torch_device(
                device_name=device.get("device_name", device_name),
                gpu=device.get("gpu", gpu),
                npu=device.get("npu", npu),
                xpu=device.get("xpu", xpu),
                raise_error=device.get("raise_error", raise_error))
        else:
            # 如果传递的是字符串，则尝试作为设备名称处理
            return get_torch_device(device_name=device if isinstance(device, str) else device_name,
                                    gpu=gpu, xpu=xpu, npu=npu, raise_error=raise_error)
    try:
        if isinstance(device_name, str) and device_name.strip():
            device_name = device_name.strip().lower()
            if device_name.startswith("cuda") and hasattr(torch, "cuda") and torch.cuda.is_available():
                pass
            elif device_name.startswith("npu") and _try_import_npu():
                pass
            elif device_name.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
                pass
            elif device_name.startswith("cpu"):
                pass
            else:
                raise RunError(f"无法获取计算资源: device_name={device_name}")
        else:
            if gpu is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                gpu = gpu.strip() if isinstance(gpu, str) else str(gpu)
                device_name = f"cuda:{gpu}"
            elif npu is not None and _try_import_npu():
                import torch_npu
                npu = npu.strip() if isinstance(npu, str) else str(npu)
                device_name = f"npu:{npu}"
            elif xpu is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
                xpu = xpu.strip() if isinstance(xpu, str) else str(xpu)
                device_name = f"xpu:{xpu}"
            elif gpu is None and npu is None and xpu is None:
                device_name = "cpu"
            else:
                raise RunError(f"无法获取计算资源: gpu={gpu},npu={npu},xpu={xpu}")
        device = torch.device(device_name)
    except Exception as ex:
        if raise_error:
            logger.error("获取计算资源失败: %s", ex)
            raise ex
        else:
            logger.warning("获取计算资源失败: '%s', 降级为CPU", ex)
            device = torch.device("cpu")  # 降级为CPU
    return device


def set_global_seed(seed: int = 47):
    """
    设定全局随机种子，增强结果可复现性。
    - 注意：完全可复现还需要固定数据加载器的采样顺序、禁用 cuDNN 的某些优化等，
      此处做较为常用的全局基础设置。
    
    Args:
        seed: 随机种子值，默认为47
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)  # 内部会将所有自动关联所有后端进行设置包括cuda/mps/xpu/npu


def _try_import_npu() -> bool:
    """
    尝试导入NPU相关模块并检查是否可用。

    Returns:
        bool: 如果NPU模块可以导入且可用返回True，否则返回False
    """
    try:
        import importlib
        import importlib.util
        if importlib.util.find_spec("torch_npu") is not None:
            import torch_npu
            return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception as _:
        pass
    return False


@contextmanager
def torch_device_scope(device: torch.device):
    """
    设备上下文管理器，在指定设备上执行代码块。

    Args:
        device: 要设置为当前设备的torch.device对象
    """
    # 检查对应后端
    if device.type == "cpu":
        yield
    if hasattr(torch, device.type):
        backend = getattr(torch, device.type)
    else:
        raise RunError(f"无法获取 device={device} 对应的后端 torch.{device.type}")
    if not hasattr(backend, "is_available") or not backend.is_available():
        raise RunError(f"后端状态不可用 torch.{device.type}.is_available()=False")
    # 设置当前设备
    previous = backend.current_device()
    try:
        torch.npu.set_device(device.index)
        yield
    finally:
        backend.set_device(previous)


def _try_float_dtype(device: torch.device, dtype: torch.dtype) -> bool:
    """
    在指定设备上用给定 dtype 做一个非常小的算子（matmul + add）测试，仅用于指定数值类型是否能跑通。

    Args:
        device: 目标设备
        dtype: 数据类型，如torch.bfloat16, torch.float16等

    Returns:
        bool: 如果指定数据类型在该设备上可用返回True，否则返回False
    """
    try:
        with torch_device_scope(device):
            # 用极小张量降低出错/溢出的风险
            a = torch.randn(32, 32, device=device, dtype=dtype)
            b = torch.randn(32, 32, device=device, dtype=dtype)
            c = a @ b
            c = c + a
            # 额外做一次归约，确保真正执行
            _ = c.sum().item()
            return True
    except Exception:
        return False


def is_bf16_and_fp16_supported(device: torch.device) -> tuple[bool, bool]:
    """
    检查指定设备是否支持bfloat16和float16数据类型。

    Args:
        device: 目标设备

    Returns:
        tuple[bool, bool]: (是否支持bfloat16, 是否支持float16)
    """
    bf16 = _try_float_dtype(device, torch.bfloat16)
    fp16 = _try_float_dtype(device, torch.float16)
    return bf16, fp16
