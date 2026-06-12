"""
GRAVEC SymPy MCP Server — 基于 FastMCP 的标准工具服务

使用 MCP (Model Context Protocol) 标准暴露 sympy 计算工具，
兼容任何支持 MCP 的 LLM 客户端（Claude Desktop、Cursor、OpenAI 等）。

架构：
  ┌──────────────────────────────────────────────────────┐
  │  任何 LLM 客户端（Claude/GPT/Agnes/...）             │
  │       ↓ OpenAI Function Calling / MCP Client         │
  │  MCP Client (FastMCP.Client)                         │
  │       ↓ JSON-RPC (in-memory / stdio / HTTP)          │
  │  MCP Server (FastMCP)                                │
  │       ↓ @mcp.tool                                    │
  │  SympyExecutor (沙箱执行)                             │
  └──────────────────────────────────────────────────────┘

两种使用模式：
  1. 进程内模式（推荐）：Client(server) 直接调用，零网络开销
  2. 独立服务模式：mcp.run() 启动 HTTP/STDIO 服务器，供外部客户端连接

社区参考：
  - sdiehl/sympy-mcp: 30+ 细粒度工具的 MCP sympy 服务器
  - FastMCP 2.0: https://github.com/prefecthq/fastmcp
  - MCP 协议: https://modelcontextprotocol.io
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP

from .sympy_exec import SympyExecutor

# ============================================================================
# MCP Server 定义
# ============================================================================

mcp = FastMCP(
    "GRAVEC SymPy Server",
    instructions=(
        "You have access to the sympy_calculate tool for precise math computation. "
        "IMPORTANT RULES:\n"
        "1. Use sympy_calculate for ANY arithmetic — do NOT do mental math.\n"
        "2. Do NOT write import statements — all sympy functions are pre-imported.\n"
        "3. Use Rational(a, b) instead of a/b for exact fractions.\n"
        "4. After getting the tool result, output your answer in the requested format.\n"
    ),
)

# 共享的 SympyExecutor 实例
_executor = SympyExecutor(timeout=10)


@mcp.tool
def sympy_calculate(code: str, purpose: str = "") -> str:
    """用 SymPy 执行精确数学计算。

    适用于：方程求解、表达式化简、精确分数运算、微积分、线性代数等。
    不要用心算，用这个工具来确保计算精确。

    重要：不要写 import 语句！sympy 的函数已经预导入，直接使用即可。
    例如直接写 Rational(10, 11)，不要写 from sympy import Rational。

    可用函数：symbols, solve, Eq, Rational, simplify, expand,
    factor, sqrt, binomial, factorial, Sum, summation, Matrix, det,
    gcd, lcm, isprime, divisors, nsimplify, pi, E, oo, I 等。
    最终结果必须赋值给变量 `result`。
    用 Rational(a, b) 代替 a/b 做精确除法。

    Args:
        code: SymPy Python 代码。最终结果赋值给 `result`。
              例: x = symbols('x'); result = solve(Eq(x**2 - 4, 0), x)
        purpose: 这次计算的目的是什么（简短说明，可选）
    """
    exec_result = _executor.execute(code)

    if exec_result.success:
        parts = [f"计算结果: {exec_result.result}"]
        if exec_result.result_type:
            parts.append(f"类型: {exec_result.result_type}")
        if exec_result.variables:
            user_vars = {
                str(k): v for k, v in exec_result.variables.items()
                if k != "result"
            }
            if user_vars:
                parts.append(f"变量: {json.dumps(user_vars, default=str)}")
        return "\n".join(parts)
    else:
        # 结构化错误输出（错误码 + 修复提示）
        parts = [f"计算错误: {exec_result.error}"]
        if exec_result.error_code:
            parts.append(f"错误码: {exec_result.error_code}")
        if exec_result.error_hint:
            parts.append(f"修复提示: {exec_result.error_hint}")
        return "\n".join(parts)


# ============================================================================
# MCP Client 辅助 — 进程内调用
# ============================================================================

class MCPToolBridge:
    """MCP 工具桥接器：将 MCP Server 的工具桥接到 OpenAI Function Calling。

    用法：
        bridge = MCPToolBridge()
        # 获取 OpenAI 格式的 tools schema
        tools = await bridge.get_openai_tools()
        # 执行工具调用
        result = await bridge.call_tool("sympy_calculate", {"code": "result = 1+1"})

    也支持同步调用（自动包装 asyncio）：
        bridge = MCPToolBridge()
        tools = bridge.get_openai_tools_sync()
        result = bridge.call_tool_sync("sympy_calculate", {"code": "result = 1+1"})
    """

    def __init__(self, server: FastMCP | None = None):
        self._server = server or mcp
        self._executor = _executor  # 共享 executor，用于提取 computed_values

    async def get_openai_tools(self) -> list[dict[str, Any]]:
        """获取 OpenAI Function Calling 格式的工具列表。"""
        from fastmcp import Client

        async with Client(self._server) as client:
            mcp_tools = await client.list_tools()

        # MCP tool schema → OpenAI function calling schema
        # 两者格式直接兼容：inputSchema = parameters
        openai_tools = []
        for tool in mcp_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            })
        return openai_tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """调用 MCP 工具，返回结果字符串。"""
        from fastmcp import Client

        async with Client(self._server) as client:
            result = await client.call_tool(name, arguments)
            # CallToolResult 有 content 属性（list of content items）
            texts = []
            for content in result.content:
                if hasattr(content, "text"):
                    texts.append(content.text)
                elif hasattr(content, "data"):
                    texts.append(str(content.data))
            return "\n".join(texts) if texts else str(result)

    def get_openai_tools_sync(self) -> list[dict[str, Any]]:
        """同步版本：获取 OpenAI 格式的工具列表。"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # 在已有事件循环中（如 Jupyter），用 nest_asyncio 或新线程
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.get_openai_tools())
                return future.result()
        else:
            return asyncio.run(self.get_openai_tools())

    def call_tool_sync(self, name: str, arguments: dict[str, Any]) -> str:
        """同步版本：调用 MCP 工具。"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.call_tool(name, arguments))
                return future.result()
        else:
            return asyncio.run(self.call_tool(name, arguments))

    @property
    def executor(self) -> SympyExecutor:
        """访问底层 SympyExecutor（用于提取 computed_values）。"""
        return self._executor


# ============================================================================
# 独立运行入口
# ============================================================================

if __name__ == "__main__":
    # 启动 MCP 服务器（STDIO 模式，供 Claude Desktop 等外部客户端使用）
    mcp.run(transport="stdio")
