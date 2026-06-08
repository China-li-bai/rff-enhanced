"""
C Agent — 综合重新辨证（倪海厦的"效不更方/不效调方"）

倪师类比：
  确定性 C = if-else 规则触发（failure_type == "wrong_direction" → refine_goal）
  Agent C  = 综合四诊、重新辨证（"等等，舌红但脉沉细，这不是单纯的肝火"）

当前代码的 C 是规则触发：
  if diagnosis.failure_type == "wrong_direction":
      revision = spec.refine_goal(...)

但倪师重新辨证时，不是看一个 failure_type 字符串，
而是重新审视全部证据——综合四诊，创造性推理。

Agent C 的能力：
  1. 拿到全部 observation_history，综合推理
  2. 判断是否需要"重新辨证"（不只是看 failure_type）
  3. 创造性地提出新方向（不只是 refine_goal 的模板化修正）
  4. 识别"真寒假热"式的深层矛盾

设计原则：
  - 保留确定性 fallback（spec.diagnose_cause + spec.refine_goal）
  - Agent 结果与确定性接口兼容（CausalDiagnosis + GoalRevision）
  - 强矛盾（score < -0.3）自动触发重新辨证
"""

from __future__ import annotations

from typing import Any, List

from ...core import Workspace
from ..models import CausalDiagnosis, GoalRevision, Observation, ValueScore


class ChiefAgent:
    """C Agent：综合重新辨证。

    与确定性 diagnose_cause + refine_goal 的区别：
    - 确定性：if-else 规则触发
    - Agent：综合全部历史，创造性推理

    Usage:
        agent = ChiefAgent(llm_call=llm_call)
        diagnosis = agent.diagnose(state, step, observation, goal, fallback=deterministic_diagnosis)
        revision = agent.revise(state, goal, observations, fallback=deterministic_revision)
    """

    def __init__(self, llm_call: Any = None, model: str | None = None):
        self._llm_call = llm_call
        self._model = model

    def diagnose(
        self,
        state: Workspace,
        step: str,
        observation: Observation,
        goal: str,
        *,
        fallback: CausalDiagnosis | None = None,
        value_score: ValueScore | None = None,
    ) -> CausalDiagnosis:
        """因果诊断（综合推理版）。

        Args:
            state: 当前工作台状态
            step: 当前步骤
            observation: 当前观察
            goal: 当前目标
            fallback: 确定性 fallback 诊断
            value_score: V 步骤的价值判断（用于识别矛盾）

        Returns:
            CausalDiagnosis，与确定性接口完全兼容
        """
        # 强矛盾 → 直接触发重新辨证，不等 E 的效果分
        if value_score and value_score.score < -0.3:
            return CausalDiagnosis(
                failure_type="wrong_direction",
                description=f"强矛盾步骤: {value_score.reason}",
                suggested_fix="重新辨证：当前方向与已有证据严重冲突",
                confidence=0.8,
            )

        if self._llm_call is None:
            return fallback or CausalDiagnosis()

        prompt = self._build_diagnose_prompt(state, step, observation, goal, fallback)
        raw = self._llm_call(prompt, model=self._model)

        return self._parse_diagnose_response(raw, fallback)

    def revise(
        self,
        state: Workspace,
        goal: str,
        observations: List[Observation],
        *,
        fallback: GoalRevision | None = None,
    ) -> GoalRevision | None:
        """果行共变（综合重新辨证版）。

        Args:
            state: 当前工作台状态
            goal: 当前目标
            observations: 全部观察历史
            fallback: 确定性 fallback 修正

        Returns:
            GoalRevision 或 None（效不更方）
        """
        if self._llm_call is None:
            return fallback

        prompt = self._build_revise_prompt(state, goal, observations)
        raw = self._llm_call(prompt, model=self._model)

        return self._parse_revise_response(raw, fallback)

    def _build_diagnose_prompt(
        self,
        state: Workspace,
        step: str,
        observation: Observation,
        goal: str,
        fallback: CausalDiagnosis | None,
    ) -> str:
        """构建因果诊断 prompt。"""
        known_vars = "\n".join(f"  {k} = {v}" for k, v in state.items()) or "  (empty)"

        fallback_info = ""
        if fallback:
            fallback_info = (
                f"\n确定性分析给出的初步诊断：\n"
                f"  类型: {fallback.failure_type}\n"
                f"  描述: {fallback.description}\n"
                f"  建议: {fallback.suggested_fix}\n"
            )

        return f"""你是一个中医主任医师，需要诊断"为什么这个治疗方案没效"。

当前目标: {goal}
当前步骤: {step}
当前观察: {observation.content} (类型={observation.observation_type}, 置信度={observation.confidence:.2f})

已知变量:
{known_vars}
{fallback_info}

请综合分析失败原因，返回 JSON 格式：
{{
  "failure_type": "<wrong_direction|insufficient_effort|confounding_factor|unexpected|unknown>",
  "description": "<详细描述>",
  "suggested_fix": "<修正建议>",
  "confidence": <0.0到1.0>
}}

失败类型说明（中医类比）：
- wrong_direction: 辨证方向就错了（热证当成寒证）
- insufficient_effort: 方向对但力度不够（药量轻了）
- confounding_factor: 兼证干扰（湿热夹着血瘀）
- unexpected: 意外情况（病人对药过敏）
- unknown: 暂时无法判断

只返回 JSON，不要其他文字。"""

    def _build_revise_prompt(
        self,
        state: Workspace,
        goal: str,
        observations: List[Observation],
    ) -> str:
        """构建果行共变 prompt。"""
        known_vars = "\n".join(f"  {k} = {v}" for k, v in state.items()) or "  (empty)"

        obs_summary = "\n".join(
            f"  [{obs.observation_type}/{obs.channel}] {obs.content} (置信度={obs.confidence:.2f})"
            for obs in observations[-5:]  # 最近5条观察
        ) or "  (无观察记录)"

        return f"""你是一个中医主任医师，需要决定是否"重新辨证"。

当前目标: {goal}

已知变量:
{known_vars}

最近观察记录:
{obs_summary}

请综合全部证据，判断是否需要调整目标。返回 JSON 格式：
{{
  "need_revision": <true|false>,
  "revised_goal": "<修正后的目标，如果 need_revision=false 则为空>",
  "revision_reason": "<为什么修正>",
  "confidence": <0.0到1.0>,
  "keep_old_as_subgoal": <true|false>
}}

原则：
- 效不更方：如果方向对只是力度不够，不要改目标
- 不效调方：如果方向根本性错误，果断调整
- 硬果不改：最终目标（如 final_answer）不允许修改，只修软果（子目标）

只返回 JSON，不要其他文字。"""

    def _parse_diagnose_response(
        self, raw: str, fallback: CausalDiagnosis | None
    ) -> CausalDiagnosis:
        """解析因果诊断 LLM 返回。"""
        import json

        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)

            return CausalDiagnosis(
                failure_type=data.get("failure_type", "unknown"),
                description=data.get("description", ""),
                suggested_fix=data.get("suggested_fix", ""),
                confidence=float(data.get("confidence", 0.5)),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return fallback or CausalDiagnosis()

    def _parse_revise_response(
        self, raw: str, fallback: GoalRevision | None
    ) -> GoalRevision | None:
        """解析果行共变 LLM 返回。"""
        import json

        try:
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)

            if not data.get("need_revision", False):
                return None

            return GoalRevision(
                revised_goal=data.get("revised_goal", ""),
                revision_reason=data.get("revision_reason", ""),
                confidence=float(data.get("confidence", 0.5)),
                keep_old_as_subgoal=data.get("keep_old_as_subgoal", True),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return fallback
