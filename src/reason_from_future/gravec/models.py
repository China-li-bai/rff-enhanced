"""
GRAVEC 数据结构 — 倪海厦方法论的核心概念

从 core_nhx.py 提取并增强的数据模型，保持向后兼容。

倪海厦类比：
  Observation  = 复诊记录（病人吃了药，回来报告效果）
  ValueScore   = 主证/兼证评估（舌红是主证，口微渴是兼证）
  GoalRevision = 重新辨证（原方向不对，调整目标）
  CausalDiagnosis = 因果诊断（为什么没效？辨证错？药量不够？）
  ReasoningPolicy = 开方策略（治本/治标/重新辨证/巩固）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# Observation — 行动后的观察结果（倪海厦的"复诊记录"）
# ============================================================================
@dataclass
class Observation:
    """行动后的观察结果。

    倪师看诊时，复诊记录是最重要的——
    "一诊开方，二诊看效，效不更方，不效调方"。
    Observation 就是"二诊看效"的数据载体。

    Attributes:
        content: 观察内容的自然语言描述
        data: 结构化数据（如验证结果、数值指标）
        observation_type: 观察类型
            - improvement: 好转
            - deterioration: 恶化
            - neutral: 无变化
            - surprise: 意外发现
        confidence: 观察的可靠程度 [0.0, 1.0]
        channel: 观察通道（望/闻/问/切），用于多通道验效
    """

    content: str
    data: dict[str, Any] = field(default_factory=dict)
    observation_type: str = "neutral"
    confidence: float = 0.5
    channel: str = "default"

    VALID_TYPES = frozenset({"improvement", "deterioration", "neutral", "surprise"})

    def __post_init__(self):
        if self.observation_type not in self.VALID_TYPES:
            self.observation_type = "neutral"
        self.confidence = max(0.0, min(1.0, self.confidence))


# ============================================================================
# ValueScore — 价值判断结果（倪海厦的"主证 vs 兼证"评估）
# ============================================================================
@dataclass
class ValueScore:
    """价值判断结果。

    中医辨证时，不是所有症状都同等重要：
    - 主证（如：舌红苔黄腻）→ 直接决定证型，score: 0.8~1.0
    - 兼证（如：口微渴）→ 辅助确认，score: 0.3~0.7
    - 无关（如：穿红衣服）→ 和诊断无关，score: 0.0~0.2
    - 矛盾（如：脉沉细但舌红）→ 需要重新辨证，score: -0.5~0.0

    Attributes:
        score: 价值分数 [-1.0, 1.0]
        reason: 判断理由的自然语言描述
        is_primary: 是否为主证（score >= 0.8 自动标记）
        syndrome_type: 证型分类（主证/兼证/无关/矛盾）
    """

    score: float = 0.0
    reason: str = ""
    is_primary: bool = False
    syndrome_type: str = "secondary"

    VALID_SYNDROME_TYPES = frozenset({"primary", "secondary", "irrelevant", "contradictory"})

    def __post_init__(self):
        self.score = max(-1.0, min(1.0, self.score))
        self.is_primary = self.score >= 0.8
        # 自动推断证型
        if self.syndrome_type not in self.VALID_SYNDROME_TYPES:
            if self.score >= 0.8:
                self.syndrome_type = "primary"
            elif self.score >= 0.3:
                self.syndrome_type = "secondary"
            elif self.score >= 0.0:
                self.syndrome_type = "irrelevant"
            else:
                self.syndrome_type = "contradictory"


# ============================================================================
# GoalRevision — 目标修正结果（倪海厦的"重新辨证"）
# ============================================================================
@dataclass
class GoalRevision:
    """目标修正结果。

    倪师说"效不更方，不效调方"——没效果就要重新思考。
    有时候不是药不好，而是诊断方向就错了。

    硬果 vs 软果：
    - 硬果（如 GSM8K 的 final_answer）：永不改变
    - 软果（如当前子目标）：可以随证型变化而调整

    Attributes:
        revised_goal: 修正后的目标
        revision_reason: 为什么修正
        confidence: 修正的置信度 [0.0, 1.0]
        keep_old_as_subgoal: 是否保留旧目标作为子目标
        is_hard_goal: 是否为硬果（硬果不允许修改）
    """

    revised_goal: str = ""
    revision_reason: str = ""
    confidence: float = 0.5
    keep_old_as_subgoal: bool = True
    is_hard_goal: bool = False

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


# ============================================================================
# CausalDiagnosis — 因果诊断结果（倪海厦的"为什么没效？"分析）
# ============================================================================
@dataclass
class CausalDiagnosis:
    """因果诊断结果。

    当验效发现行动没效果时，需要诊断"为什么"。
    倪师复诊时会问："是辨证错了？还是药量不够？还是兼证没处理？"

    五类失败归因：
    - wrong_direction: 辨证方向就错了（热证当成寒证）
    - insufficient_effort: 方向对，力度不够（药量轻了）
    - confounding_factor: 兼证干扰（湿热夹着血瘀）
    - unexpected: 意外情况（病人对药过敏）
    - unknown: 暂时无法判断

    Attributes:
        failure_type: 失败原因分类
        description: 自然语言描述
        suggested_fix: 修正建议
        confidence: 诊断的可靠程度 [0.0, 1.0]
    """

    failure_type: str = "unknown"
    description: str = ""
    suggested_fix: str = ""
    confidence: float = 0.5

    VALID_TYPES = frozenset({
        "wrong_direction",
        "insufficient_effort",
        "confounding_factor",
        "unexpected",
        "unknown",
    })

    def __post_init__(self):
        if self.failure_type not in self.VALID_TYPES:
            self.failure_type = "unknown"
        self.confidence = max(0.0, min(1.0, self.confidence))


# ============================================================================
# ReasoningPolicy — 开方策略（倪海厦的"治疗流派"）
# ============================================================================
@dataclass(frozen=True)
class ReasoningPolicy:
    """开方策略控制。

    倪师看诊时，不同阶段有不同的治疗策略：
    - treat_root_cause: 治本/抓主证（当 is_primary=True 时）
    - treat_secondary: 治标/兼证处理（当 score ∈ [0.3, 0.8) 时）
    - re_diagnose: 重新辨证（当矛盾出现时）
    - consolidate: 巩固/效不更方（当连续 improvement 时）

    同时保留格式控制能力，确保 LLM 输出可解析。

    Attributes:
        name: 策略名称
        treatment_approach: 治疗流派（治本/治标/重新辨证/巩固）
        borrowed_styles: 向后兼容的风格标签
        instruction: 给 LLM 的指令
    """

    name: str
    treatment_approach: str = "treat_root_cause"
    borrowed_styles: tuple[str, ...] = ()
    instruction: str = ""

    VALID_APPROACHES = frozenset({
        "treat_root_cause",
        "treat_secondary",
        "re_diagnose",
        "consolidate",
    })

    def __post_init__(self):
        if self.treatment_approach not in self.VALID_APPROACHES:
            object.__setattr__(self, "treatment_approach", "treat_root_cause")
