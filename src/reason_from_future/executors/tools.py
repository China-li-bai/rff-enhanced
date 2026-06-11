"""
LLM Tool Calling — sympy 精确计算工具

以果决其行：LLM 自己判断何时需要精确计算，调用 sympy 工具，
看到结果后再决定下一步。不再是"LLM 心算 → 外部验证"的断裂模式，
而是"LLM 推理 → 调用工具 → 看到结果 → 继续推理"的闭环。

架构：
  ┌──────────────────────────────────────────────────┐
  │  LLM 推理                                        │
  │    "我需要计算 10/11 的精确值"                     │
  │         ↓ tool_call                              │
  │  sympy_calculate(code="Rational(10, 11)")        │
  │         ↓ 执行结果                                │
  │  tool_result: 0.9090909090909091                  │
  │         ↓ 反馈给 LLM                              │
  │  LLM 继续推理："所以答案是 10/11"                  │
  └──────────────────────────────────────────────────┘

社区参考：
  - sdiehl/sympy-mcp: MCP 协议的 sympy 服务器
  - Eis4TY/sym-mcp: 带 AST 安全检查的 sympy MCP
  - rllm-project: OpenAI function calling + calculator

我们的方案不依赖 MCP，直接用 OpenAI function calling 协议，
通过 LiteLLM 兼容所有模型。
"""

from __future__ import annotations

import json
from typing import Any

import litellm

from .sympy_exec import SympyExecutor


# ============================================================================
# 工具 Schema — OpenAI Function Calling 格式
# ============================================================================

SYMPY_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "sympy_calculate",
        "description": (
            "用 SymPy 执行精确数学计算。适用于：方程求解、表达式化简、"
            "精确分数运算、微积分、线性代数等。"
            "不要用心算，用这个工具来确保计算精确。"
            "\n\n重要：不要写 import 语句！sympy 的函数已经预导入，直接使用即可。"
            "例如直接写 Rational(10, 11)，不要写 from sympy import Rational。"
            "\n\n可用函数：symbols, solve, Eq, Rational, simplify, expand, "
            "factor, sqrt, binomial, factorial, Sum, summation, Matrix, det, "
            "gcd, lcm, isprime, divisors, nsimplify, pi, E, oo, I 等。"
            "最终结果必须赋值给变量 `result`。"
            "用 Rational(a, b) 代替 a/b 做精确除法。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "SymPy Python 代码。最终结果赋值给 `result`。"
                        "例: x = symbols('x'); result = solve(Eq(x**2 - 4, 0), x)"
                    ),
                },
                "purpose": {
                    "type": "string",
                    "description": "这次计算的目的是什么（简短说明）",
                },
            },
            "required": ["code"],
        },
    },
}


# ============================================================================
# Tool Call 处理器
# ============================================================================

class SympyToolHandler:
    """处理 LLM 的 sympy_calculate 工具调用。

    Usage:
        handler = SympyToolHandler()
        result_str = handler.handle("sympy_calculate", {"code": "result = Rational(10, 11)"})
    """

    def __init__(self, timeout: int = 10):
        self._executor = SympyExecutor(timeout=timeout)

    def handle(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """执行工具调用，返回结果字符串。"""
        if tool_name != "sympy_calculate":
            return f"错误：未知工具 {tool_name}"

        code = arguments.get("code", "")
        if not code:
            return "错误：code 参数不能为空"

        exec_result = self._executor.execute(code)

        if exec_result.success:
            # 格式化结果，让 LLM 容易理解
            parts = [f"计算结果: {exec_result.result}"]
            if exec_result.result_type:
                parts.append(f"类型: {exec_result.result_type}")
            if exec_result.variables:
                # 只展示非内置变量
                user_vars = {
                    k: v for k, v in exec_result.variables.items()
                    if k != "result"
                }
                if user_vars:
                    parts.append(f"变量: {user_vars}")
            return "\n".join(parts)
        else:
            return f"计算错误: {exec_result.error}"


# ============================================================================
# LLM Tool-Call 循环
# ============================================================================

def llm_call_with_tools(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_handler: SympyToolHandler | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    max_tool_rounds: int = 5,
    timeout: int = 120,
    verbose: bool = False,
    force_tool_use: bool = False,
    final_prompt: str | None = None,
) -> dict[str, Any]:
    """带工具调用的 LLM 对话循环。

    流程：
    1. 发送 messages + tools 给 LLM
    2. 如果 LLM 返回 tool_calls → 执行工具 → 把结果追加到 messages → 回到步骤 1
    3. 如果 LLM 返回普通文本 → 返回最终消息

    当 force_tool_use=True 时，采用两阶段策略：
    - 阶段1：tool_choice="required" → 强制调用工具 → 获取计算结果
    - 阶段2：tool_choice="auto"（不带 tools）→ 让 LLM 用计算结果输出最终文本

    Args:
        messages: 对话历史
        tools: OpenAI function calling 格式的工具列表
        tool_handler: 工具调用处理器
        model: LLM 模型名
        api_key: API Key
        api_base: API Base URL
        max_tool_rounds: 最大工具调用轮数（防止无限循环）
        timeout: 单次 LLM 调用超时（秒）
        verbose: 是否打印详细日志
        force_tool_use: 是否强制模型调用工具（两阶段策略）
        final_prompt: 两阶段策略中，阶段2的追加 prompt（让 LLM 输出格式化结果）

    Returns:
        dict: {"content": str, "tool_calls_count": int, "messages": list,
               "computed_values": dict[str, Any]}
        computed_values 包含 sympy 计算的所有变量及其值。
    """
    from ..llm import DEFAULT_MODEL, DEFAULT_API_KEY, DEFAULT_API_BASE

    use_model = model or DEFAULT_MODEL
    use_api_key = api_key or DEFAULT_API_KEY
    use_api_base = api_base or DEFAULT_API_BASE

    if tool_handler is None:
        tool_handler = SympyToolHandler()

    if tools is None:
        tools = [SYMPY_TOOL_SCHEMA]

    # 在 messages 前面插入系统消息，引导 LLM 使用工具
    # 只在第一轮且 messages 中没有系统消息时插入
    tool_system_msg = {
        "role": "system",
        "content": (
            "You have access to the sympy_calculate tool for precise math computation. "
            "IMPORTANT RULES:\n"
            "1. Use sympy_calculate for ANY arithmetic — do NOT do mental math.\n"
            "2. Do NOT write import statements — all sympy functions (Rational, solve, symbols, etc.) are pre-imported.\n"
            "3. Use Rational(a, b) instead of a/b for exact fractions.\n"
            "4. After getting the tool result, output your answer in the requested format.\n"
        ),
    }
    if messages and messages[0].get("role") != "system":
        messages = [tool_system_msg] + messages
    elif messages and messages[0].get("role") == "system":
        # 追加到现有系统消息
        messages[0] = {
            "role": "system",
            "content": messages[0]["content"] + "\n\n" + tool_system_msg["content"],
        }

    tool_calls_count = 0
    computed_values: dict[str, Any] = {}  # 收集 sympy 计算的变量值

    for round_idx in range(max_tool_rounds + 1):
        kwargs: dict[str, Any] = {
            "model": use_model,
            "messages": messages,
            "timeout": timeout,
        }
        if use_api_key:
            kwargs["api_key"] = use_api_key
        if use_api_base:
            kwargs["api_base"] = use_api_base
        if tools:
            kwargs["tools"] = tools

        # 第一轮：用 auto 让 LLM 自己决定是否调用工具
        # 注意：Agnes AI 不支持 tool_choice="required"，所以不用 force 模式
        # 改为在 prompt 中明确指示使用工具

        if verbose:
            print(f"\n--- Tool-Call Round {round_idx + 1} ---")
            last_msg = messages[-1]
            if last_msg.get("role") == "tool":
                print(f"[tool result] {last_msg.get('content', '')[:200]}")
            else:
                print(f"[{last_msg.get('role')}] {str(last_msg.get('content', ''))[:200]}")

        response = litellm.completion(**kwargs)

        choice = response.choices[0]
        message = choice.message

        # 没有 tool_calls → LLM 给出了最终回复
        if not message.tool_calls:
            if verbose:
                print(f"[final] {message.content[:300] if message.content else '(empty)'}")

            return {
                "content": message.content or "",
                "tool_calls_count": tool_calls_count,
                "messages": messages,
                "computed_values": computed_values,
            }

        # 有 tool_calls → 执行工具，把结果追加到 messages
        messages.append(message.model_dump())

        for tc in message.tool_calls:
            tool_name = tc.function.name
            tool_args_str = tc.function.arguments
            tool_call_id = tc.id

            if verbose:
                print(f"[tool_call] {tool_name}({tool_args_str[:200]})")

            try:
                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
            except json.JSONDecodeError:
                tool_args = {"code": tool_args_str}

            # 执行工具
            result_str = tool_handler.handle(tool_name, tool_args)
            tool_calls_count += 1

            # 收集 sympy 计算的变量值
            if tool_name == "sympy_calculate" and tool_handler._executor:
                last_exec = tool_handler._executor._last_execution
                if last_exec and last_exec.success and last_exec.variables:
                    for k, v in last_exec.variables.items():
                        if k != "result":
                            computed_values[k] = v
                    # 也收集 result 值
                    if last_exec.result is not None:
                        computed_values["__result__"] = last_exec.result

            if verbose:
                print(f"[tool_result] {result_str[:300]}")

            # 追加 tool 结果到 messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_str,
            })

        # force_tool_use 两阶段策略：工具调用完成后，
        # 如果 LLM 没有返回文本内容，再做一次不带 tools 的调用
        # 让 LLM 用计算结果输出格式化文本
        if force_tool_use and tool_calls_count > 0:
            # 构建阶段2的 prompt
            summary_parts = []
            for msg in messages:
                if msg.get("role") == "tool":
                    summary_parts.append(f"计算结果: {msg.get('content', '')}")

            calc_summary = "\n".join(summary_parts)
            final_msg = final_prompt or (
                "Based on the calculation results above, provide your final answer. "
                "Output the result in the required format."
            )
            phase2_messages = [
                {
                    "role": "system",
                    "content": (
                        "You have just used the sympy_calculate tool. "
                        "Here are the calculation results:\n\n"
                        f"{calc_summary}\n\n"
                        f"{final_msg}"
                    ),
                },
                {"role": "user", "content": messages[0].get("content", "") if messages else ""},
            ]

            if verbose:
                print(f"\n--- Phase 2: Format output (no tools) ---")

            phase2_kwargs: dict[str, Any] = {
                "model": use_model,
                "messages": phase2_messages,
                "timeout": timeout,
            }
            if use_api_key:
                phase2_kwargs["api_key"] = use_api_key
            if use_api_base:
                phase2_kwargs["api_base"] = use_api_base

            phase2_response = litellm.completion(**phase2_kwargs)
            phase2_content = phase2_response.choices[0].message.content or ""

            if verbose:
                print(f"[phase2 final] {phase2_content[:300]}")

            return {
                "content": phase2_content,
                "tool_calls_count": tool_calls_count,
                "messages": messages,
                "computed_values": computed_values,
            }

    # 超过最大轮数，强制返回
    return {
        "content": messages[-1].get("content", "") if messages else "",
        "tool_calls_count": tool_calls_count,
        "messages": messages,
        "computed_values": computed_values,
    }
