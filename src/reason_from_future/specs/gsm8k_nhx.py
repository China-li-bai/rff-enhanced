"""
================================================================================
GSM8K 倪海厦增强规约 — gsm8k_nhx.py
================================================================================

【费曼视角：一句话讲清楚这个文件是什么】
原版 GSM8KSpec 只会"算题"（推理），不会"验题"（行动），更不会"调思路"（果行共变）。
GSM8KNiHaixiaSpec 在原版基础上补上了倪海厦方法论的四个核心：

1. 价值判断 (evaluate_step_value) — 这步算出来的中间值，对最终答案有多重要？
   倪师类比：这个症状是"主证"还是"兼证"？
   
2. 行动执行 (execute_action) — 把算出的中间值代入原题验证自洽性
   倪师类比：开方后让病人服药，看反应
   
3. 验效反馈 (evaluate_observation) — 代入验证后，离正确答案更近了还是更远了？
   倪师类比：复诊——好转了还是恶化了？
   
4. 果行共变 (refine_goal) — 如果反复算不对，是不是目标理解有误？需要换角度
   倪师类比：不效调方——重新辨证

5. 因果诊断 (diagnose_cause) — 为什么这步没效果？是方向错了还是力度不够？
   倪师类比：为什么药没效？辨证错了还是药量不够？

【跨文件关系】
- 继承 core_nhx.py 的 NiHaixiaSpec（间接继承 core.py 的 ProblemSpec）
- 复用 gsm8k.py 的 GSM8KSpec 全部 8 个方法实现
- 被 core_nhx.py 的 reason_from_future_nhx() 主循环调用

【Python 入门知识重点】
- 多重继承：class Child(Parent1, Parent2) 同时继承两个父类
- super() 在多重继承中的行为：MRO（方法解析顺序）
- 组合 vs 继承：这里用组合（self._base = GSM8KSpec(...)）避免多重继承的复杂性
"""
import json
import re
import textwrap
from typing import Dict, List, Optional, Set, Tuple

from ..core import Workspace
from ..core_nhx import (
    CausalDiagnosis,
    GoalRevision,
    NiHaixiaSpec,
    Observation,
    ValueScore,
)
from ..llm import llm_call
from .gsm8k import GSM8KSpec


class GSM8KNiHaixiaSpec(NiHaixiaSpec):
    """GSM8K 倪海厦增强规约。

    【费曼解释 - 为什么用组合而不是继承】
    我们已经有了 GSM8KSpec（原版8个方法的实现），不想重写它们。
    最直觉的做法是多重继承：class GSM8KNiHaixiaSpec(GSM8KSpec, NiHaixiaSpec)
    但多重继承在 Python 中容易出问题（钻石继承、MRO 冲突等）。

    更安全的做法是"组合"：在 __init__ 中创建一个 GSM8KSpec 实例，
    然后把原版8个方法的调用"委托"给这个实例。
    这就像请了一个"原版专家"来处理基础工作，自己专注新增的4个方法。

    【倪海厦类比】
    GSM8KSpec = 只会辨证的医生（基础能力）
    NiHaixiaSpec = 完整的倪师方法论框架（增强能力）
    GSM8KNiHaixiaSpec = 一个学会了倪师方法论的数学老师
    """

    def __init__(self, problem_data: Dict[str, str]):
        self._base = GSM8KSpec(problem_data)
        self.question: str = problem_data["question"]
        self.problem_data: Dict[str, str] = problem_data

        answer_str = str(problem_data["answer"])
        match = re.search(r"(?:####\s*)?([0-9,.]+)\s*$", answer_str)
        if match:
            self.gold_numeric_answer: float = float(match.group(1).replace(",", ""))
        else:
            self.gold_numeric_answer: float = float('nan')

    # ====================================================================
    # 原版 8 个方法：委托给 GSM8KSpec 实例
    # ====================================================================

    def derive_final_target(self, problem: str) -> str:
        return self._base.derive_final_target(problem)

    def parse_workspace_update(self, raw_text: str, state: Workspace) -> Workspace:
        return self._base.parse_workspace_update(raw_text, state)

    def check_local(self, state: Workspace, target_step: str) -> bool:
        return self._base.check_local(state, target_step)

    def verify_final(self, state: Workspace) -> Tuple[bool, str, float]:
        return self._base.verify_final(state)

    def prompt_last_step(self, state: Workspace, target: str, avoid: Set[str]) -> str:
        return self._base.prompt_last_step(state, target, avoid)

    def prompt_forward_step(self, state: Workspace, target_step: str, avoid: Set[str]) -> str:
        return self._base.prompt_forward_step(state, target_step, avoid)

    def parse_target_step(self, raw_text: str) -> str:
        return self._base.parse_target_step(raw_text)

    def merge_aliases(self, state: Workspace) -> Workspace:
        return self._base.merge_aliases(state)

    # ====================================================================
    # 新增方法 1：价值判断
    # ====================================================================
    def evaluate_step_value(
        self, state: Workspace, step: str, goal: str
    ) -> ValueScore:
        """价值判断：这一步算出的中间值对最终答案有多重要？

        【费曼解释】
        不是所有中间变量都同等重要。比如一道题：
        "小明有5个苹果，小红有3个橘子，他们一共有多少水果？"
        
        - total_fruits = apples + oranges → 主证！直接决定最终答案 (score: 0.9)
        - apples = 5 → 兼证，需要但不是关键 (score: 0.5)
        - oranges = 3 → 兼证 (score: 0.5)
        - color_of_apples → 无关 (score: 0.0)

        【倪海厦类比】
        倪师看病时"抓主证"——不是所有症状都值得花精力。
        主证是"果"的直接线索，兼证是辅助，矛盾是"重新辨证"的信号。

        【实现策略】
        用 LLM 判断这个变量在解题链中的位置：
        1. 是否直接出现在原题中？（基础变量 → 兼证）
        2. 是否是最终答案的直接前驱？（关键变量 → 主证）
        3. 是否和其他变量矛盾？（矛盾 → 需要重新思考）
        """
        defined_vars = sorted(list(state.keys()))

        if step not in state:
            return ValueScore(score=0.0, reason=f"变量 {step} 不在已知状态中")

        prompt = textwrap.dedent(
            f"""
            You are evaluating the importance of a computed variable in solving a math problem.

            Problem: {self.question}

            Known variables: {json.dumps({k: v for k, v in state.items() if isinstance(v, (int, float))}, indent=2)}

            Variable to evaluate: "{step}" = {state[step]}
            Final goal: "{goal}"

            Rate how important this variable is for reaching the final goal:
            - 0.8-1.0: This variable is a DIRECT prerequisite for the final answer (主证/primary evidence)
            - 0.4-0.7: This variable is an intermediate step that contributes indirectly (兼证/secondary evidence)
            - 0.1-0.3: This variable has minor relevance (无关/minor relevance)
            - -0.5-0.0: This variable CONTRADICTS other known values (矛盾/contradiction)

            Output a single JSON object with keys:
            - "score": float between -0.5 and 1.0
            - "reason": brief explanation (one sentence)
            - "is_primary": true if score >= 0.8

            IMPORTANT: Respond with ONLY the JSON.
            """
        ).strip()

        try:
            raw = llm_call(prompt, verbose=False)
            match = re.search(r"\{[\s\S]*?\}", raw)
            if match:
                data = json.loads(match.group(0))
                return ValueScore(
                    score=float(data.get("score", 0.5)),
                    reason=str(data.get("reason", "")),
                    is_primary=bool(data.get("is_primary", False)),
                )
        except Exception:
            pass

        if step == goal:
            return ValueScore(score=1.0, reason="目标变量本身", is_primary=True)

        return ValueScore(score=0.5, reason="默认中等价值（LLM评估失败）")

    # ====================================================================
    # 新增方法 2：行动执行
    # ====================================================================
    def execute_action(
        self, state: Workspace, step: str, goal: str
    ) -> Observation:
        """行动执行：把算出的中间值代入原题验证自洽性。

        【费曼解释】
        算出一个中间值后，不是直接相信它，而是"用一下"看看：
        把这个值代入原题的语境中，检查是否自洽。

        比如原题说"小明有5个苹果"，你算出 apples = 7，
        那就矛盾了——这就是"行动"发现的"观察"。

        【倪海厦类比】
        开方后让病人服药，观察反应。
        如果病人说"吃了更难受"→ 恶化
        如果病人说"好多了"→ 好转
        如果病人说"没感觉"→ 无变化

        【实现策略】
        用 LLM 把当前所有已知变量代入原题，检查：
        1. 数值是否和题目描述矛盾？
        2. 计算链是否自洽？
        3. 是否发现了新的约束或信息？
        """
        numeric_state = {k: v for k, v in state.items() if isinstance(v, (int, float))}

        prompt = textwrap.dedent(
            f"""
            You are verifying the consistency of computed values against a math problem.

            Problem: {self.question}

            Computed variables so far:
            {json.dumps(numeric_state, indent=2)}

            Most recently computed: "{step}" = {state.get(step, "unknown")}

            Check:
            1. Does "{step}" = {state.get(step, "unknown")} contradict any information in the problem?
            2. Are all computed values mutually consistent?
            3. Does this bring us closer to the final answer?

            Output a single JSON object:
            {{
                "consistent": true/false,
                "observation_type": "improvement" | "deterioration" | "neutral" | "surprise",
                "content": "brief description of what you observed",
                "confidence": 0.0-1.0,
                "details": {{
                    "contradictions": ["list of any contradictions found"],
                    "new_insights": ["list of any new insights"],
                    "distance_to_goal": "closer" | "same" | "farther"
                }}
            }}

            IMPORTANT: Respond with ONLY the JSON.
            """
        ).strip()

        try:
            raw = llm_call(prompt, verbose=False)
            match = re.search(r"\{[\s\S]*?\}", raw)
            if match:
                data = json.loads(match.group(0))
                obs_type = data.get("observation_type", "neutral")
                if obs_type not in Observation.VALID_TYPES:
                    obs_type = "neutral"
                return Observation(
                    content=str(data.get("content", "")),
                    data=data.get("details", {}),
                    observation_type=obs_type,
                    confidence=float(data.get("confidence", 0.5)),
                )
        except Exception:
            pass

        return Observation(
            content="验证执行失败，默认中性观察",
            data={"step": step},
            observation_type="neutral",
            confidence=0.3,
        )

    # ====================================================================
    # 新增方法 3：验效反馈
    # ====================================================================
    def evaluate_observation(
        self, observation: Observation, state: Workspace, goal: str
    ) -> float:
        """验效反馈：评估行动效果。返回改善程度。

        【费曼解释】
        把 Observation 中的信息转化为一个数值：
        - 正值 = 好转（离答案更近了）
        - 负值 = 恶化（方向错了）
        - 零 = 无变化

        【倪海厦类比】
        复诊评估——好转了多少？用 0~1 的数值量化。

        【实现策略】
        基于 observation_type 和 confidence 计算：
        - improvement + high confidence → 高正值
        - deterioration + high confidence → 高负值
        - neutral → 接近零
        - surprise → 视情况而定
        """
        type_scores = {
            "improvement": 0.6,
            "deterioration": -0.4,
            "neutral": 0.0,
            "surprise": 0.2,
        }

        base_score = type_scores.get(observation.observation_type, 0.0)

        details = observation.data
        distance = details.get("distance_to_goal", "same")
        if distance == "closer":
            base_score += 0.2
        elif distance == "farther":
            base_score -= 0.2

        contradictions = details.get("contradictions", [])
        if contradictions:
            base_score -= 0.3 * len(contradictions)

        new_insights = details.get("new_insights", [])
        if new_insights:
            base_score += 0.1 * len(new_insights)

        adjusted = base_score * observation.confidence

        return max(-1.0, min(1.0, adjusted))

    # ====================================================================
    # 新增方法 4：果行共变
    # ====================================================================
    def refine_goal(
        self, state: Workspace, goal: str, observations: List[Observation]
    ) -> Optional[GoalRevision]:
        """果行共变：根据反馈修正目标。

        【费曼解释】
        如果反复算不对，可能不是计算的问题，而是"理解题意"的问题。
        比如：你以为要算"总人数"，但其实题目问的是"剩余人数"。

        这时候需要"重新理解题意"——修正目标。

        【倪海厦类比】
        不效调方——原来的辨证方向不对，需要重新辨证。
        但不是完全推翻，而是调整方向。

        【实现策略】
        分析最近的观察历史，用 LLM 判断：
        1. 当前目标是否正确理解了题意？
        2. 是否需要换一个角度来解题？
        3. 新的目标应该是什么？
        """
        if not observations:
            return None

        recent = observations[-3:]
        obs_summary = "\n".join(
            f"- [{o.observation_type}] {o.content} (置信度: {o.confidence:.2f})"
            for o in recent
        )

        deterioration_count = sum(
            1 for o in recent if o.observation_type == "deterioration"
        )
        if deterioration_count < 2:
            return None

        numeric_state = {k: v for k, v in state.items() if isinstance(v, (int, float))}

        prompt = textwrap.dedent(
            f"""
            You are re-evaluating the problem-solving goal based on feedback.

            Problem: {self.question}

            Current goal: "{goal}"
            Known variables: {json.dumps(numeric_state, indent=2)}

            Recent observations (most recent last):
            {obs_summary}

            The current approach seems ineffective (multiple deteriorations).
            Should we revise the goal or approach?

            Consider:
            1. Is the current goal correctly interpreting the problem?
            2. Should we decompose the goal differently?
            3. Is there a different variable that would be more productive to target?

            If no revision is needed, output: {{"needs_revision": false}}
            If revision is needed, output:
            {{
                "needs_revision": true,
                "revised_goal": "new target variable name",
                "reason": "why the revision is needed",
                "confidence": 0.0-1.0,
                "keep_old_as_subgoal": true/false
            }}

            IMPORTANT: Respond with ONLY the JSON.
            """
        ).strip()

        try:
            raw = llm_call(prompt, verbose=False)
            match = re.search(r"\{[\s\S]*?\}", raw)
            if match:
                data = json.loads(match.group(0))
                if not data.get("needs_revision", False):
                    return None
                return GoalRevision(
                    revised_goal=str(data.get("revised_goal", goal)),
                    revision_reason=str(data.get("reason", "")),
                    confidence=float(data.get("confidence", 0.5)),
                    keep_old_as_subgoal=bool(data.get("keep_old_as_subgoal", True)),
                )
        except Exception:
            pass

        return None

    # ====================================================================
    # 新增方法 5：因果诊断
    # ====================================================================
    def diagnose_cause(
        self, state: Workspace, step: str, observation: Observation, goal: str
    ) -> CausalDiagnosis:
        """因果诊断：行动没效果时，分析"为什么"。

        【费曼解释】
        当验效发现行动没效果时，需要诊断"为什么"。
        可能的原因：
        1. wrong_direction：方向就错了（比如把加法题当减法做）
        2. insufficient_effort：方向对但力度不够（还需要更多中间步骤）
        3. confounding_factor：有干扰因素（算出了无关的变量）
        4. unexpected：意外情况（题目理解有误）

        【倪海厦类比】
        为什么药没效？
        - 辨证错了 → wrong_direction
        - 药量不够 → insufficient_effort
        - 有兼证干扰 → confounding_factor
        - 意外反应 → unexpected

        【实现策略】
        用 LLM 分析当前状态、行动结果和目标之间的关系，
        判断失败原因并给出修正建议。
        """
        numeric_state = {k: v for k, v in state.items() if isinstance(v, (int, float))}

        prompt = textwrap.dedent(
            f"""
            You are diagnosing why a reasoning step failed to make progress.

            Problem: {self.question}
            Current goal: "{goal}"
            Known variables: {json.dumps(numeric_state, indent=2)}

            Step that failed: "{step}" = {state.get(step, "unknown")}
            Observation after action: [{observation.observation_type}] {observation.content}

            Diagnose the failure type:
            - "wrong_direction": The approach is fundamentally wrong (e.g., treating addition as subtraction)
            - "insufficient_effort": The direction is correct but more intermediate steps are needed
            - "confounding_factor": Irrelevant variables are interfering with the calculation
            - "unexpected": Something unexpected happened (e.g., problem misinterpretation)
            - "unknown": Cannot determine the cause

            Output a single JSON object:
            {{
                "failure_type": "one of the above types",
                "description": "brief explanation of why this step failed",
                "suggested_fix": "what to do differently next",
                "confidence": 0.0-1.0
            }}

            IMPORTANT: Respond with ONLY the JSON.
            """
        ).strip()

        try:
            raw = llm_call(prompt, verbose=False)
            match = re.search(r"\{[\s\S]*?\}", raw)
            if match:
                data = json.loads(match.group(0))
                ftype = data.get("failure_type", "unknown")
                if ftype not in CausalDiagnosis.VALID_TYPES:
                    ftype = "unknown"
                return CausalDiagnosis(
                    failure_type=ftype,
                    description=str(data.get("description", "")),
                    suggested_fix=str(data.get("suggested_fix", "")),
                    confidence=float(data.get("confidence", 0.5)),
                )
        except Exception:
            pass

        return CausalDiagnosis(
            failure_type="unknown",
            description="因果诊断失败，使用默认诊断",
            suggested_fix="尝试不同的中间变量",
            confidence=0.3,
        )
