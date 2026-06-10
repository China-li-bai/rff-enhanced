"""
ProblemClassifier — 问题特征提取器

三层漏斗路由的前两层：
  L1 正则/关键词  (<1ms)   → 学科分类 + 特殊记号检测
  L2 启发式分析  (<10ms)  → 复杂度估算（变量数、运算步骤数）

L3 (LLM FC) 在 policy.py 中实现，因为需要 LLM 调用。
"""

from __future__ import annotations

import re
from typing import Any

from .strategy import ProblemFeatures


class ProblemClassifier:
    """问题特征提取器（L1 正则 + L2 启发式）。"""

    # L1 学科关键词（每个学科一组独立正则）
    _SUBJECT_RULES: list[tuple[str, str]] = [
        ("geometry", r"triangle|circle|rectangle|square|cube|sphere|cylinder|cone|area|volume|perimeter|diagonal|radius|diameter|circumference|angle|hypotenuse|isosceles|equilateral|midpoint|slope|coordinate"),
        ("number_theory", r"divisible|remainder|modulo|prime|gcd|lcm|base\s+\d|digit|binary|ternary|positive integer|consecutive|factor|multiple|\\bmod\\b|congruen"),
        ("counting_probability", r"probability|combin|permut|choose|select.*from|how many ways|arrang|order|distinct|\\binom|at least|at most|random"),
        ("precalculus", r"\\bsin\\b|\\bcos\\b|\\btan\\b|log|ln|exponent|limit|sequence|series|function|domain|range|asymptote|period|amplitude|matrix|determinant|vector|eigenvalue"),
        ("intermediate_algebra", r"quadrat|polynomial|factori|simplif|expand|absolute value|inequal|interval|system of equation|simultaneous|substitut"),
        ("algebra", r"solve for|equation|variable|express|evaluate|linear|rate|ratio|proportion|sum|product|difference|average|mean"),
        ("prealgebra", r"multiply|divide|add|subtract|fraction|decimal|percent|how many|total|remaining|left over|change|cost|price"),
    ]

    # 特殊记号模式
    _SPECIAL_NOTATION_PATTERNS: list[str] = [
        r"\d+_\d+",
        r"\\binom",
        r"\\pmod",
        r"\\lfloor|\\rfloor",
        r"\\lceil|\\rceil",
        r"\\cdot",
        r"\\times",
    ]

    def classify(self, problem: str, *, difficulty_hint: int = 0) -> ProblemFeatures:
        """提取问题特征。"""
        features = ProblemFeatures()

        # L1: 正则/关键词分类
        self._classify_subject(problem, features)
        self._detect_special_notation(problem, features)
        self._detect_variables(problem, features)
        self._detect_equations(problem, features)

        # L2: 启发式复杂度估算
        self._estimate_complexity(problem, features)

        # 外部难度提示
        if difficulty_hint > 0:
            features.difficulty_hint = difficulty_hint

        return features

    def _classify_subject(self, problem: str, features: ProblemFeatures) -> None:
        """L1: 学科分类。"""
        problem_lower = problem.lower()
        for subject, pattern in self._SUBJECT_RULES:
            try:
                if re.search(pattern, problem_lower):
                    features.subject = subject
                    features.raw_features["l1_subject"] = subject
                    return
            except re.error:
                continue
        features.subject = "general"
        features.raw_features["l1_subject"] = "general"

    def _detect_special_notation(self, problem: str, features: ProblemFeatures) -> None:
        """L1: 特殊记号检测。"""
        for pattern in self._SPECIAL_NOTATION_PATTERNS:
            try:
                if re.search(pattern, problem):
                    features.has_special_notation = True
                    features.raw_features.setdefault("special_notations", []).append(pattern)
            except re.error:
                continue
        # base-N: \d+_\d+ 特别处理
        base_n = re.findall(r"(\d+)_(\d+)", problem)
        if base_n:
            features.has_special_notation = True
            features.raw_features["base_notation"] = base_n

    def _detect_variables(self, problem: str, features: ProblemFeatures) -> None:
        """L1: 变量检测。"""
        latex_vars = set(re.findall(r"\$([a-zA-Z])\$", problem))
        inline_vars = set(re.findall(r"\b([a-zA-Z])\s*=", problem))
        all_vars = latex_vars | inline_vars
        non_vars = {"a", "i", "A", "I"}
        real_vars = all_vars - non_vars
        features.has_variables = len(real_vars) > 0
        features.variable_count = len(real_vars)
        features.raw_features["variables"] = list(real_vars)

    def _detect_equations(self, problem: str, features: ProblemFeatures) -> None:
        """L1: 方程检测。"""
        has_equals = "=" in problem or "\\equiv" in problem
        has_inequality = any(s in problem for s in ["<", ">", "\\leq", "\\geq", "\\neq"])
        features.has_equations = has_equals or has_inequality

        geometry_keywords = [
            "triangle", "circle", "rectangle", "square", "cube", "sphere",
            "cylinder", "cone", "area", "volume", "perimeter", "diagonal",
            "radius", "diameter", "angle", "side", "height", "base",
            "coordinate", "midpoint", "slope", "point",
        ]
        problem_lower = problem.lower()
        features.has_geometry = any(kw in problem_lower for kw in geometry_keywords)

        counting_keywords = [
            "probability", "how many ways", "combin", "permut", "choose",
            "arrang", "select", "at least", "at most", "random",
        ]
        features.has_counting = any(kw in problem_lower for kw in counting_keywords)

    def _estimate_complexity(self, problem: str, features: ProblemFeatures) -> None:
        """L2: 启发式复杂度估算。"""
        op_count = 0
        for op in ["+", "-", "\\times", "\\cdot", "/", "\\frac", "^", "\\sqrt"]:
            op_count += problem.count(op)
        features.operation_count = max(1, op_count)

        complexity_score = 0
        complexity_score += features.variable_count * 2
        complexity_score += features.operation_count
        complexity_score += 3 if features.has_equations else 0
        complexity_score += 4 if features.has_geometry else 0
        complexity_score += 3 if features.has_counting else 0
        complexity_score += 5 if features.has_special_notation else 0

        features.raw_features["complexity_score"] = complexity_score
        features.raw_features["operation_count"] = features.operation_count
