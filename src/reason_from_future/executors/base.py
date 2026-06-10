"""
ExecutionResult — 执行结果数据模型

所有 Executor 的统一输出格式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """代码执行结果。

    Attributes:
        success: 是否执行成功
        result: 执行结果值（成功时）
        result_type: 结果类型（int, float, str, sympy.Basic, ...）
        error: 错误信息（失败时）
        code: 执行的代码
        elapsed_s: 执行耗时（秒）
        variables: 执行后的变量绑定
    """

    success: bool = False
    result: Any = None
    result_type: str = ""
    error: str = ""
    code: str = ""
    elapsed_s: float = 0.0
    variables: dict[str, Any] = field(default_factory=dict)

    @property
    def numeric_value(self) -> float | None:
        """尝试提取数值结果。"""
        if self.result is None:
            return None
        try:
            return float(self.result)
        except (TypeError, ValueError):
            return None

    def __str__(self) -> str:
        if self.success:
            return f"ExecutionResult(ok, {self.result_type}={self.result})"
        return f"ExecutionResult(fail, {self.error})"
