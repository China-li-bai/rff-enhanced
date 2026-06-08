"""
GRAVEC 工具层 — 工具注册中心

望闻问切多通道的工具注册与发现。
每个通道可以注册多个工具，Agent 按需调用。

通道映射（中医四诊 → 代码领域 → 数学领域）：
  望（看）  → py_compile, ast.parse     → AST 求值
  闻（听）  → pytest, subprocess.run    → 数量级检查
  问（问诊）→ pylint, ruff check        → 依赖追溯
  切（脉诊）→ mypy, pyright             → 边界检查
"""

from .registry import ToolRegistry

__all__ = ["ToolRegistry"]
