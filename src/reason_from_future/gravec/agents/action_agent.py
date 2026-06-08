"""
A Agent — 行动执行（倪海厦的"开方+验方"）

倪师类比：
  确定性 A = 只验方（自洽性检查、AST 求值）
  Agent A  = 开新方 + 验方（可调工具、产生新信息）

当前 GSM8K 实现中 A 只能验证（自洽性 + 追溯来源），
但倪师开方是创造性的——"龙胆泻肝汤不行，改丹栀逍遥散"。

Agent A 的能力：
  1. 调工具（运行代码、执行测试、搜索文档）
  2. 产生新信息（不只是验证旧信息）
  3. 多步执行（一个 action 可以调多个工具）

设计原则：
  - 保留确定性 fallback（spec.execute_action）
  - Agent 结果与确定性接口兼容（都是 Observation）
  - 工具通过 ToolRegistry 注册和调用
"""

from __future__ import annotations

from typing import Any

from ...core import Workspace
from ..models import Observation


class ActionAgent:
    """A Agent：行动执行（可调工具）。

    与确定性 execute_action 的区别：
    - 确定性：只能验证（自洽性检查、AST 求值）
    - Agent：可以调工具、产生新信息

    Usage:
        agent = ActionAgent(llm_call=llm_call, tool_registry=registry)
        result = agent.execute(state, step, goal, fallback=deterministic_obs)
    """

    def __init__(
        self,
        llm_call: Any = None,
        model: str | None = None,
        tool_registry: Any = None,
    ):
        self._llm_call = llm_call
        self._model = model
        self._tool_registry = tool_registry

    def execute(
        self,
        state: Workspace,
        step: str,
        goal: str,
        *,
        fallback: Observation | None = None,
    ) -> Observation:
        """行动执行（可调工具）。

        Args:
            state: 当前工作台状态
            step: 待执行的步骤
            goal: 当前目标
            fallback: 确定性 fallback 结果

        Returns:
            Observation，与确定性接口完全兼容
        """
        # 如果没有工具注册中心，直接 fallback
        if self._tool_registry is None or not self._tool_registry.has_tools():
            return fallback or Observation(content="no tools available")

        # 如果没有 LLM，无法决定调哪个工具
        if self._llm_call is None:
            return fallback or Observation(content="no LLM available")

        prompt = self._build_prompt(state, step, goal)
        raw = self._llm_call(prompt, model=self._model)

        return self._execute_tools(raw, state, step, fallback)

    def _build_prompt(
        self,
        state: Workspace,
        step: str,
        goal: str,
    ) -> str:
        """构建 A Agent 的 prompt。"""
        known_vars = "\n".join(f"  {k} = {v}" for k, v in state.items()) or "  (empty)"

        available_tools = ""
        if self._tool_registry:
            tools = self._tool_registry.list_tools()
            if tools:
                available_tools = "\n可用工具:\n" + "\n".join(
                    f"  - {name}: {desc}" for name, desc in tools
                )

        return f"""你是一个执行专家，需要决定如何验证或推进当前步骤。

当前目标: {goal}
当前步骤: {step}

已知变量:
{known_vars}
{available_tools}

请决定是否需要调用工具来验证或推进。返回 JSON 格式：
{{
  "action": "use_tool" 或 "verify_only",
  "tool_name": "<工具名，如果 action=use_tool>",
  "tool_args": {{<工具参数>}},
  "reason": "<为什么选择这个行动>"
}}

如果当前步骤已经有足够信息验证，选择 verify_only。
只返回 JSON，不要其他文字。"""

    def _execute_tools(
        self,
        raw: str,
        state: Workspace,
        step: str,
        fallback: Observation | None,
    ) -> Observation:
        """解析 LLM 的工具调用决策并执行。"""
        import json

        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)

            if data.get("action") != "use_tool":
                return fallback or Observation(content="verify only, no tool needed")

            tool_name = data.get("tool_name", "")
            tool_args = data.get("tool_args", {})

            if not tool_name or not self._tool_registry:
                return fallback or Observation(content="no tool specified")

            result = self._tool_registry.call(tool_name, **tool_args)

            return Observation(
                content=f"工具 {tool_name} 执行结果: {result}",
                data={"tool": tool_name, "result": result},
                observation_type="neutral",
                confidence=0.7,
                channel="tool",
            )
        except (json.JSONDecodeError, ValueError, KeyError, Exception) as e:
            return fallback or Observation(
                content=f"A Agent 执行失败: {e}",
                observation_type="surprise",
                confidence=0.3,
            )
