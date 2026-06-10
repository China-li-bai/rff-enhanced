"""
RoutingPolicy — PRISM 式自适应路由策略

根据 ProblemFeatures 决定执行哪条推理路径。

三层路由决策：
  L1+L2 特征 → 确定性规则（快速、可预测）
  L3 LLM FC  → 语义深度判断（慢、但准）

路由逻辑（PRISM 对齐）：
  简单题 (complexity ≤ 4, difficulty ≤ 2)
    → COT_DIRECT: 1-2 次 LLM 调用直接出答案
  中等题 (complexity 5-10, difficulty 3)
    → COT_VERIFY: GRAVEC + sympy 验证
  困难题 (complexity 11-20, difficulty 4)
    → TOT_VOTE: 多路径 GRAVEC + 多数投票
  极难题 (complexity > 20, difficulty 5)
    → DEEP_GRAVEC: 全量迭代 + sympy + 投票 + 目标修正

倪海厦类比：
  简单 = 小病开成药（一剂见效）
  中等 = 常规辨证论治（望闻问切→开方→复诊）
  困难 = 疑难杂症会诊（多位大夫各开一方，取共识）
  极难 = 危重证抢救（反复辨证、调方、多法并用）
"""

from __future__ import annotations

from typing import Any, Callable

from .classifier import ProblemClassifier
from .strategy import ProblemFeatures, Strategy, StrategyDecision


class RoutingPolicy:
    """PRISM 式自适应路由策略。

    Usage:
        policy = RoutingPolicy()
        decision = policy.route("If $x = 2$, what is $x^2$?", difficulty_hint=1)
        # decision.strategy = Strategy.COT_DIRECT
        # decision.max_iters = 2

        # 带 LLM FC 的路由（L3 层）
        policy = RoutingPolicy(llm_call=my_llm_call)
        decision = policy.route(problem, features=features)
    """

    def __init__(
        self,
        llm_call: Callable[..., str] | None = None,
        model: str | None = None,
    ):
        self._classifier = ProblemClassifier()
        self._llm_call = llm_call
        self._model = model

    def route(
        self,
        problem: str,
        *,
        features: ProblemFeatures | None = None,
        difficulty_hint: int = 0,
    ) -> StrategyDecision:
        """路由决策：根据问题特征选择推理策略。

        Args:
            problem: 问题描述文本
            features: 预提取的问题特征（可选，不传则自动提取）
            difficulty_hint: 外部难度提示（如 MATH-500 的 level 标签）

        Returns:
            StrategyDecision 实例
        """
        # 提取特征（如果未提供）
        if features is None:
            features = self._classifier.classify(problem, difficulty_hint=difficulty_hint)

        # L1+L2: 确定性规则路由
        decision = self._rule_based_route(problem, features)

        # L3: LLM FC 路由（仅当置信度不够时）
        if decision.confidence < 0.7 and self._llm_call is not None:
            llm_decision = self._llm_route(problem, features)
            if llm_decision is not None:
                decision = llm_decision

        return decision

    def _rule_based_route(
        self, problem: str, features: ProblemFeatures
    ) -> StrategyDecision:
        """L1+L2: 确定性规则路由。"""
        complexity = features.raw_features.get("complexity_score", 0)
        difficulty = features.difficulty_hint

        # 规则 1: 有外部难度标签时直接用
        if difficulty >= 1:
            if difficulty <= 1:
                strategy = Strategy.COT_DIRECT
                reason = f"L{difficulty} 简单题 → 直接求解"
                confidence = 0.9
            elif difficulty <= 2:
                strategy = Strategy.COT_DIRECT
                reason = f"L{difficulty} 基础题 → 直接求解（可能需验证）"
                confidence = 0.8
            elif difficulty <= 3:
                strategy = Strategy.COT_VERIFY
                reason = f"L{difficulty} 中等题 → GRAVEC + sympy 验证"
                confidence = 0.85
            elif difficulty <= 4:
                strategy = Strategy.TOT_VOTE
                reason = f"L{difficulty} 困难题 → 多路径投票"
                confidence = 0.8
            else:
                strategy = Strategy.DEEP_GRAVEC
                reason = f"L{difficulty} 极难题 → 全量迭代 + 投票"
                confidence = 0.75

            # 复杂度修正：高复杂度升级策略
            if complexity > 15 and strategy.value <= Strategy.COT_VERIFY.value:
                strategy = Strategy.TOT_VOTE
                reason += " [复杂度修正: ↑TOT_VOTE]"
                confidence -= 0.1

            # 特殊记号修正：base-N / 模运算升级
            if features.has_special_notation and strategy == Strategy.COT_DIRECT:
                strategy = Strategy.COT_VERIFY
                reason += " [特殊记号修正: ↑COT_VERIFY]"
                confidence -= 0.05

            return StrategyDecision(
                strategy=strategy,
                features=features,
                confidence=confidence,
                reason=reason,
                max_iters=strategy.max_iters,
                vote_count=strategy.vote_count,
                use_sympy=strategy.use_sympy,
                use_early_stop=strategy != Strategy.DEEP_GRAVEC,
                estimated_time_s=self._estimate_time(strategy),
            )

        # 规则 2: 无难度标签，纯复杂度路由
        if complexity <= 4:
            strategy = Strategy.COT_DIRECT
            reason = f"低复杂度 ({complexity}) → 直接求解"
            confidence = 0.7
        elif complexity <= 10:
            strategy = Strategy.COT_VERIFY
            reason = f"中等复杂度 ({complexity}) → GRAVEC + sympy"
            confidence = 0.65
        elif complexity <= 20:
            strategy = Strategy.TOT_VOTE
            reason = f"高复杂度 ({complexity}) → 多路径投票"
            confidence = 0.6
        else:
            strategy = Strategy.DEEP_GRAVEC
            reason = f"极高复杂度 ({complexity}) → 全量迭代"
            confidence = 0.55

        return StrategyDecision(
            strategy=strategy,
            features=features,
            confidence=confidence,
            reason=reason,
            max_iters=strategy.max_iters,
            vote_count=strategy.vote_count,
            use_sympy=strategy.use_sympy,
            use_early_stop=strategy != Strategy.DEEP_GRAVEC,
            estimated_time_s=self._estimate_time(strategy),
        )

    def _llm_route(
        self, problem: str, features: ProblemFeatures
    ) -> StrategyDecision | None:
        """L3: LLM FC 路由（语义深度判断）。"""
        if self._llm_call is None:
            return None

        prompt = f"""你是一个数学问题难度评估专家。请评估以下数学题的难度和推荐解题策略。

问题: {problem[:500]}

已提取的特征:
- 学科: {features.subject}
- 变量数: {features.variable_count}
- 含方程: {features.has_equations}
- 含几何: {features.has_geometry}
- 含计数/概率: {features.has_counting}
- 含特殊记号: {features.has_special_notation}
- 估算运算步骤: {features.operation_count}

请返回 JSON:
{{
  "difficulty": <1-5>,
  "strategy": "cot_direct" | "cot_verify" | "tot_vote" | "deep_gravec",
  "reason": "<为什么选这个策略>",
  "confidence": <0.0-1.0>
}}

策略说明:
- cot_direct: 简单题，1-2步心算可解
- cot_verify: 中等题，需要分步推理+验证
- tot_vote: 困难题，需要多种解法交叉验证
- deep_gravec: 极难题，需要深度迭代+反复修正

只返回 JSON，不要其他文字。"""

        try:
            import json

            raw = self._llm_call(prompt, model=self._model)
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            strategy = Strategy(data.get("strategy", "cot_verify"))
            difficulty = int(data.get("difficulty", 3))
            reason = str(data.get("reason", "LLM 路由"))
            confidence = float(data.get("confidence", 0.7))

            features.difficulty_hint = difficulty

            return StrategyDecision(
                strategy=strategy,
                features=features,
                confidence=confidence,
                reason=f"[LLM路由] {reason}",
                max_iters=strategy.max_iters,
                vote_count=strategy.vote_count,
                use_sympy=strategy.use_sympy,
                use_early_stop=strategy != Strategy.DEEP_GRAVEC,
                estimated_time_s=self._estimate_time(strategy),
            )
        except Exception:
            return None

    @staticmethod
    def _estimate_time(strategy: Strategy) -> float:
        """预估耗时（秒）。"""
        return {
            Strategy.COT_DIRECT: 5.0,
            Strategy.COT_VERIFY: 30.0,
            Strategy.TOT_VOTE: 90.0,
            Strategy.DEEP_GRAVEC: 180.0,
        }[strategy]
