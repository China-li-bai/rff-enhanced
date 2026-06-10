"""
Executors — 精确计算后端

GRAVEC A 步的执行层：
  LLM 心算 → 精确计算（sympy / python exec）

倪海厦类比：
  LLM 心算 = 凭经验开方（快但不精确）
  Sympy    = 药典精确配伍（慢但无误）
  Python   = 实验室化验（精确但需安全环境）

设计原则：
  - Executor 是普通 Python 类，不需要继承基类
  - 输入：代码字符串 + 变量绑定
  - 输出：ExecutionResult（成功/失败 + 结果值）
  - 沙箱化：timeout + restricted globals
"""

from .sympy_exec import SympyExecutor
from .base import ExecutionResult

__all__ = ["SympyExecutor", "ExecutionResult"]
