"""
ToolRegistry — 工具注册中心

倪师类比：工具 = 诊断器械（听诊器、血压计、舌诊仪……）
  - 注册：把器械摆上桌
  - 发现：根据症状选器械
  - 调用：用器械检查病人

设计原则：
  - 工具是普通 Python 函数，不需要继承任何基类
  - 注册时声明名称、描述、通道归属
  - 调用时自动参数传递和错误处理
  - 与 ActionAgent / EffectAgent 配合使用
"""

from __future__ import annotations

from typing import Any, Callable


class ToolRegistry:
    """工具注册中心。

    Usage:
        registry = ToolRegistry()

        # 注册工具
        registry.register(
            name="py_compile",
            fn=lambda path: __import__("py_compile").compile(path, doraise=True),
            description="编译检查 Python 文件语法",
            channel="inspect",
        )

        # 查询
        registry.has_tool("py_compile")  # True
        registry.list_tools()            # [("py_compile", "编译检查...")]
        registry.tools_by_channel("inspect")  # [("py_compile", ...)]

        # 调用
        result = registry.call("py_compile", path="main.py")
    """

    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        description: str = "",
        channel: str = "default",
    ) -> None:
        """注册一个工具。

        Args:
            name: 工具名称（唯一标识）
            fn: 工具函数
            description: 工具描述（给 Agent 看的）
            channel: 所属通道（inspect/execute/analyze/validate）
        """
        self._tools[name] = {
            "fn": fn,
            "description": description,
            "channel": channel,
        }

    def unregister(self, name: str) -> None:
        """注销一个工具。"""
        self._tools.pop(name, None)

    def has_tool(self, name: str) -> bool:
        """检查工具是否已注册。"""
        return name in self._tools

    def has_tools(self) -> bool:
        """检查是否有任何工具注册。"""
        return len(self._tools) > 0

    def call(self, name: str, **kwargs: Any) -> Any:
        """调用一个工具。

        Args:
            name: 工具名称
            **kwargs: 工具参数

        Returns:
            工具执行结果

        Raises:
            KeyError: 工具未注册
            Exception: 工具执行错误
        """
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")

        fn = self._tools[name]["fn"]
        return fn(**kwargs)

    def list_tools(self) -> list[tuple[str, str]]:
        """列出所有已注册工具。

        Returns:
            [(name, description), ...]
        """
        return [
            (name, info["description"])
            for name, info in self._tools.items()
        ]

    def tools_by_channel(self, channel: str) -> list[tuple[str, str]]:
        """列出指定通道的工具。

        Args:
            channel: 通道名称

        Returns:
            [(name, description), ...]
        """
        return [
            (name, info["description"])
            for name, info in self._tools.items()
            if info["channel"] == channel
        ]

    def get_description(self, name: str) -> str:
        """获取工具描述。"""
        if name not in self._tools:
            return ""
        return self._tools[name]["description"]
