"""
NiHaixiaSpec — 倪海厦「以果决其行」增强规约

从 core_nhx.py 的 NiHaixiaSpec 提取并增强，核心变化：
1. V/A/E/C 四步支持 Agent-backed 实现（智能面）
2. 保留确定性 fallback（控制面）
3. 新增工具注册（望闻问切多通道）

继承链：
  ProblemSpec (ABC, 8个@abstractmethod) — core.py
    └── NiHaixiaSpec (新增6个@abstractmethod) — 本文件
          └── GSM8KNiHaixiaSpec — specs/gsm8k_nhx.py
"""

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Set

from ..core import ProblemSpec, Workspace
from .models import (
    CausalDiagnosis,
    GoalRevision,
    Observation,
    ReasoningPolicy,
    ValueScore,
)


class NiHaixiaSpec(ProblemSpec):
    """倪海厦「以果决其行」增强规约。

    在 ProblemSpec 的 8 个方法基础上新增 6 个方法：
    1. evaluate_step_value()   — V：价值判断（主证/兼证/矛盾）
    2. execute_action()        — A：行动执行（可调工具）
    3. evaluate_observation()  — E：验效反馈
    4. refine_goal()           — C：果行共变（目标随反馈修正）
    5. diagnose_cause()        — C：因果诊断（为什么没效）
    6. select_reasoning_policy() — 策略选择（治疗流派）

    每个方法都有两种实现路径：
    - 确定性路径：spec 内部的算法（fallback，保证可测试）
    - Agent 路径：调用 LLM Agent（智能面，提供语义深度）

    子类可以选择只实现确定性路径，也可以同时实现 Agent 路径。
    """

    # ----------------------------------------------------------------
    # V (Value/价值判断)：这一步对目标的贡献度是多少？
    # ----------------------------------------------------------------
    @abstractmethod
    def evaluate_step_value(
        self, state: Workspace, step: str, goal: str
    ) -> ValueScore:
        """价值判断：这一步对目标的贡献度。

        倪师类比：这个症状（step）对判断最终证型（goal）的价值？
        - 主证 → score: 0.8~1.0
        - 兼证 → score: 0.3~0.7
        - 无关 → score: 0.0~0.2
        - 矛盾 → score: -0.5~0.0

        确定性实现：图论距离 / 结构分析
        Agent 实现：语义推理"这是主证还是兼证"
        """

    # ----------------------------------------------------------------
    # A (Action/决其行)：基于推理结果执行行动
    # ----------------------------------------------------------------
    @abstractmethod
    def execute_action(
        self, state: Workspace, step: str, goal: str
    ) -> Observation:
        """行动执行：基于推理结果执行行动并返回观察。

        倪师类比：开方/扎针（行动）→ 病人反馈（观察）

        确定性实现：AST 求值 / 自洽性检查
        Agent 实现：调工具（运行代码、执行测试、搜索文档）
        """

    # ----------------------------------------------------------------
    # E (Effect/验效)：评估行动效果
    # ----------------------------------------------------------------
    @abstractmethod
    def evaluate_observation(
        self, observation: Observation, state: Workspace, goal: str
    ) -> float:
        """验效反馈：评估行动效果。返回改善程度。

        倪师类比：复诊——病人吃了药，好转了多少？
        - 正值 = 好转
        - 负值 = 恶化
        - 零 = 无变化

        确定性实现：公式计算
        Agent 实现：多通道望闻问切
        """

    # ----------------------------------------------------------------
    # C (Check/校验)：果行共变 + 因果诊断
    # ----------------------------------------------------------------
    @abstractmethod
    def refine_goal(
        self, state: Workspace, goal: str, observations: List[Observation]
    ) -> Optional[GoalRevision]:
        """果行共变：根据反馈修正目标。

        倪师类比：复诊后重新辨证——原来的诊断方向可能不对。
        返回 None = 效不更方，返回 GoalRevision = 不效调方。

        硬果（final_answer）不允许修改，只修软果（当前子目标）。
        """

    @abstractmethod
    def diagnose_cause(
        self, state: Workspace, step: str, observation: Observation, goal: str
    ) -> CausalDiagnosis:
        """因果诊断：行动没效果时，分析"为什么"。

        倪师类比：为什么这药没效？
        - wrong_direction: 辨证方向就错了
        - insufficient_effort: 方向对但力度不够
        - confounding_factor: 兼证干扰
        - unexpected: 意外情况
        - unknown: 暂时无法判断
        """

    # ----------------------------------------------------------------
    # 策略选择：治疗流派
    # ----------------------------------------------------------------
    def select_reasoning_policy(
        self,
        state: Workspace,
        goal: str,
        iteration: int,
        observations: List[Observation],
        avoid: Set[str],
    ) -> ReasoningPolicy:
        """选择下一步的开方策略（治疗流派）。

        倪师类比：根据当前证型选择治疗方向——
        - 治本（抓主证） / 治标（兼证处理）
        - 重新辨证 / 巩固（效不更方）

        默认实现基于观察历史和迭代状态自动选择。
        子类可覆盖以实现领域特定的策略逻辑。
        """
        if self.check_local(state, goal):
            return ReasoningPolicy(
                name="format_finalize",
                treatment_approach="consolidate",
                instruction="Return only the already verified final format.",
            )

        if observations:
            recent = observations[-3:]
            bad_count = sum(1 for obs in recent if obs.observation_type == "deterioration")
            stagnant_count = sum(
                1 for obs in recent if obs.observation_type in {"neutral", "surprise"}
            )
            if bad_count >= 1 and iteration >= 1:
                latest = recent[-1]
                return ReasoningPolicy(
                    name="feedback_repair",
                    treatment_approach="re_diagnose",
                    instruction=(
                        "Use the verifier feedback to repair the next candidate while preserving the required "
                        f"output format. Latest feedback: {latest.content}"
                    ),
                )
            if stagnant_count >= 2 or avoid:
                return ReasoningPolicy(
                    name="feedback_guided_retry",
                    treatment_approach="treat_secondary",
                    instruction=(
                        "Avoid repeated low-value or failing steps and produce one parseable candidate in the "
                        f"required format. Avoid list: {sorted(avoid)}"
                    ),
                )

        if iteration == 0:
            return ReasoningPolicy(
                name="format_first_attempt",
                treatment_approach="treat_root_cause",
                instruction="Generate the first candidate in the exact domain format.",
            )

        return ReasoningPolicy(
            name="feedback_guided_retry",
            treatment_approach="treat_root_cause",
            instruction="Use accumulated feedback to produce the next parseable candidate.",
        )

    def render_prompt_with_policy(
        self,
        prompt: str,
        policy: ReasoningPolicy,
        phase: str,
    ) -> str:
        """将策略渲染到 prompt 中。默认 no-op，子类可覆盖。"""
        return prompt

    def should_attempt_direct_goal(
        self,
        state: Workspace,
        goal: str,
        iteration: int,
        avoid: Set[str],
    ) -> bool:
        """是否允许直接尝试计算最终目标。"""
        return goal not in avoid
