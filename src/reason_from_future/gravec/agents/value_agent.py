"""
V Agent — 语义主证识别（倪海厦的"抓主证"）

倪师类比：
  确定性 V = 只看图论距离（离目标近就值钱）
  Agent V  = 语义推理"舌红是主证还是兼证"

当前 GSM8K 实现用图论距离打分，在数学领域碰巧管用，
但在代码/规划领域完全失效——import os 离目标近但价值极低，
修复核心逻辑可能离目标远但价值极高。

Agent V 用 LLM 做语义判断：
  输入：step 的语义描述 + goal + 已知变量
  输出：ValueScore（score, reason, syndrome_type, is_primary）

设计原则：
  - 保留确定性 fallback（spec.evaluate_step_value）
  - Agent 结果与确定性接口兼容（都是 ValueScore）
  - 可通过配置开关启用/禁用
"""

from __future__ import annotations

from typing import Any

from ...core import Workspace
from ..models import ValueScore


class ValueAgent:
    """V Agent：语义主证识别。

    用 LLM 推理判断一个步骤是"主证/兼证/无关/矛盾"，
    而不是只看图论距离。

    Usage:
        agent = ValueAgent(llm_call=llm_call)
        result = agent.evaluate(state, step, goal, fallback=deterministic_score)
        # result 是 ValueScore，与确定性接口完全兼容
    """

    def __init__(self, llm_call: Any = None, model: str | None = None):
        self._llm_call = llm_call
        self._model = model

    def evaluate(
        self,
        state: Workspace,
        step: str,
        goal: str,
        *,
        fallback: ValueScore | None = None,
    ) -> ValueScore:
        """语义主证识别。

        Args:
            state: 当前工作台状态
            step: 待评估的步骤
            goal: 当前目标
            fallback: 确定性 fallback 结果（spec.evaluate_step_value 的输出）

        Returns:
            ValueScore，与确定性接口完全兼容
        """
        if self._llm_call is None:
            return fallback or ValueScore(score=0.0, reason="no LLM available")

        prompt = self._build_prompt(state, step, goal, fallback)
        raw = self._llm_call(prompt, model=self._model)

        return self._parse_response(raw, fallback)

    def _build_prompt(
        self,
        state: Workspace,
        step: str,
        goal: str,
        fallback: ValueScore | None,
    ) -> str:
        """构建 V Agent 的 prompt。"""
        known_vars = "\n".join(f"  {k} = {v}" for k, v in state.items()) or "  (empty)"

        fallback_info = ""
        if fallback:
            fallback_info = (
                f"\n确定性分析（图论距离）给出的初步评估：\n"
                f"  分数: {fallback.score:.2f}\n"
                f"  原因: {fallback.reason}\n"
            )

        return f"""你是一个中医辨证专家，需要判断某个"症状"（步骤）对最终"证型"（目标）的价值。

当前目标（果）: {goal}
待评估步骤: {step}

已知变量（草稿纸）:
{known_vars}
{fallback_info}

请判断这个步骤的价值，返回 JSON 格式：
{{
  "score": <float, -1.0到1.0>,
  "reason": "<判断理由>",
  "syndrome_type": "<primary|secondary|irrelevant|contradictory>"
}}

评分标准（中医辨证类比）：
- primary (主证, 0.8~1.0): 直接决定证型的关键步骤，不攻克就无法推进
- secondary (兼证, 0.3~0.7): 辅助确认的步骤，有价值但不是最关键的
- irrelevant (无关, 0.0~0.2): 和当前目标无关的步骤
- contradictory (矛盾, -0.5~0.0): 与已有证据冲突的步骤，需要重新辨证

只返回 JSON，不要其他文字。"""

    def _parse_response(
        self, raw: str, fallback: ValueScore | None
    ) -> ValueScore:
        """解析 LLM 返回的 JSON 为 ValueScore。"""
        import json

        try:
            # 尝试提取 JSON
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)

            score = float(data.get("score", 0.0))
            reason = str(data.get("reason", ""))
            syndrome_type = str(data.get("syndrome_type", "secondary"))

            return ValueScore(
                score=score,
                reason=reason,
                syndrome_type=syndrome_type,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            # 解析失败 → fallback
            return fallback or ValueScore(score=0.0, reason="V Agent parse failed")
