"""
Strategy 定义 — 推理策略枚举与决策模型

四种策略对应 PRISM 的三级路由：
  COT_DIRECT   → 简单题：1-2 次 LLM 调用直接出答案
  COT_VERIFY   → 中等题：GRAVEC G→R→A→V→E→C + sympy 验证
  TOT_VOTE     → 困难题：多路径 GRAVEC + 多数投票
  DEEP_GRAVEC  → 极难题：全量迭代 + sympy + 投票 + 目标修正

倪海厦类比：
  COT_DIRECT   = 小病开成药（感冒→银翘散，一剂见效）
  COT_VERIFY   = 常规辨证论治（望闻问切→开方→复诊）
  TOT_VOTE     = 疑难杂症会诊（多位大夫各开一方，取共识）
  DEEP_GRAVEC  = 危重证抢救（反复辨证、调方、多法并用）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Strategy(Enum):
    """推理策略枚举。"""

    COT_DIRECT = "cot_direct"
    COT_VERIFY = "cot_verify"
    TOT_VOTE = "tot_vote"
    DEEP_GRAVEC = "deep_gravec"

    @property
    def max_iters(self) -> int:
        """该策略的默认最大迭代次数。"""
        return {
            Strategy.COT_DIRECT: 2,
            Strategy.COT_VERIFY: 10,
            Strategy.TOT_VOTE: 8,
            Strategy.DEEP_GRAVEC: 16,
        }[self]

    @property
    def vote_count(self) -> int:
        """该策略的投票路径数（1 = 不投票）。"""
        return {
            Strategy.COT_DIRECT: 1,
            Strategy.COT_VERIFY: 1,
            Strategy.TOT_VOTE: 3,
            Strategy.DEEP_GRAVEC: 5,
        }[self]

    @property
    def use_sympy(self) -> bool:
        """该策略是否启用 sympy 精确计算。"""
        return self in (Strategy.COT_VERIFY, Strategy.TOT_VOTE, Strategy.DEEP_GRAVEC)

    @property
    def description(self) -> str:
        return {
            Strategy.COT_DIRECT: "简单题：1-2 次 LLM 调用直接出答案",
            Strategy.COT_VERIFY: "中等题：GRAVEC + sympy 验证",
            Strategy.TOT_VOTE: "困难题：多路径 GRAVEC + 多数投票",
            Strategy.DEEP_GRAVEC: "极难题：全量迭代 + sympy + 投票 + 目标修正",
        }[self]


@dataclass
class ProblemFeatures:
    """问题特征提取结果 — 望闻问切收集的信息。

    由 ProblemClassifier 产出，供 RoutingPolicy 决策。

    Attributes:
        subject: 学科分类 (algebra, geometry, number_theory, ...)
        difficulty_hint: 难度提示 (1-5, 来自数据集标签或 LLM 估算)
        has_variables: 是否含变量 (如 x, y, a, b)
        has_equations: 是否含方程 (等号、不等号)
        has_geometry: 是否含几何 (坐标、面积、体积)
        has_counting: 是否含计数/概率
        operation_count: 估算运算步骤数
        variable_count: 变量个数
        has_special_notation: 是否含特殊记号 (base-N, 模运算, 组合数)
        raw_features: L1/L2 层提取的原始特征字典
    """

    subject: str = "unknown"
    difficulty_hint: int = 3
    has_variables: bool = False
    has_equations: bool = False
    has_geometry: bool = False
    has_counting: bool = False
    operation_count: int = 1
    variable_count: int = 0
    has_special_notation: bool = False
    raw_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyDecision:
    """策略路由决策 — 以果决其行的"果"。

    由 RoutingPolicy 产出，决定执行哪条推理路径。

    Attributes:
        strategy: 选定的推理策略
        features: 问题特征（决策依据）
        confidence: 路由置信度 [0.0, 1.0]
        reason: 路由理由（可解释性）
        max_iters: 该策略下的最大迭代次数
        vote_count: 投票路径数
        use_sympy: 是否启用 sympy
        use_early_stop: 是否启用早停
        estimated_time_s: 预估耗时（秒）
    """

    strategy: Strategy = Strategy.COT_VERIFY
    features: ProblemFeatures = field(default_factory=ProblemFeatures)
    confidence: float = 0.5
    reason: str = ""
    max_iters: int = 10
    vote_count: int = 1
    use_sympy: bool = True
    use_early_stop: bool = True
    estimated_time_s: float = 30.0

    def __post_init__(self):
        # 从 strategy 自动填充默认值
        if self.max_iters == 10 and self.strategy != Strategy.COT_VERIFY:
            self.max_iters = self.strategy.max_iters
        if self.vote_count == 1 and self.strategy != Strategy.COT_VERIFY:
            self.vote_count = self.strategy.vote_count
        if self.use_sympy and not self.strategy.use_sympy:
            self.use_sympy = self.strategy.use_sympy
        self.confidence = max(0.0, min(1.0, self.confidence))
