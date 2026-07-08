# -*- coding: utf-8 -*-

"""
异常处理模块

定义基础组件使用的两类核心异常：
1. RunError: 运行时异常基类，包含错误信息、关联对象类型和内部异常
2. ArgumentError: 参数校验异常，用于参数不符合预期时爆出
所有异常均携带错误描述、关联对象类型及嵌套异常信息
"""

__all__ = [
    "AssertError",
    "RunError",
    "ArgumentError",
    "trust_assert"
]

from logging import Logger


class AssertError(Exception):
    """条件断言异常基类"""

    def __init__(self, message: str):
        """
        Args:
            message(str): 错误描述信息
        """
        super(AssertError, self).__init__(message)
        self.message = message


class RunError(Exception):
    """算子运行时异常"""

    def __init__(self,
                 message: str,
                 obj: object = None,
                 inner_error: Exception = None,
                 ):
        """
        Args:
            message(str): 错误描述信息
            obj(object,optional): 触发异常的对象实例（可选）
            inner_error(Exception,optional): 嵌套的原始异常对象（可选）
        """
        super(RunError, self).__init__(message)
        self.message = message
        # 获取对象的类信息
        cls = obj.__class__ if obj is not None and not isinstance(obj, type) else obj
        self.class_name = cls.__name__ if cls else None
        self.inner_error = inner_error


class ArgumentError(RunError):
    """算子参数校验失败异常"""

    def __init__(self,
                 message: str,
                 args: str | list[str] | dict[str:object] | None = None,
                 obj: object = None,
                 inner_error: Exception = None,
                 ):
        """
        Args:
            message(str): 错误描述信息
            obj(object,optional): 触发异常的对象实例（可选）
            inner_error(Exception,optional): 嵌套的原始异常对象（可选）
            args(str|list[str]|dict[str:object],optional): 参数名称（可选）
        """
        super(ArgumentError, self).__init__(message, obj, inner_error)
        self.argument_details = args


def trust_assert(judgment: bool, message: str | None = None, *args, logger: Logger | None = None, **kwargs) -> None:
    """
    可信断言函数，当判断条件为False时抛出AssertError异常
    
    Args:
        judgment(bool): 判断条件
        message(str|None): 异常消息模板（可选）
        logger(Logger|None): 日志对象（可选）。若存在则只在日志中记录异常，而不抛出。
        *args: 用于格式化消息的位置参数
        **kwargs: 用于格式化消息的关键字参数
        
    Raises:
        AssertError: 当judgment为False、且logger为None时抛出
    """
    if not judgment:
        # 根据是否有参数决定是否格式化消息
        message = (message.format(*args, **kwargs) if args or kwargs else message) if message else "Assert Error"
        if logger:
            logger.error(message)
        else:
            raise AssertError(message)
