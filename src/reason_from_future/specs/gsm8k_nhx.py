"""
GSM8K 倪海厦增强规约：确定性验效版。

设计原则：
- 硬果固定：GSM8K 的最终目标始终是 final_answer，不因失败而改名。
- 软果可变：当前子目标、路径偏好、失败避免项可以随验效调整。
- LLM 只负责生发候选：反向提出变量、正向提出公式。
- 程序负责裁决：解析、求值、依赖抽取、价值评分、验效和因果诊断。
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core import Workspace
from ..core_nhx import (
    CausalDiagnosis,
    GoalRevision,
    NiHaixiaSpec,
    Observation,
    ValueScore,
)
from .gsm8k import GSM8KSpec


_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")
_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass
class StepRecord:
    """一次候选计算的可验证记录。"""

    var: str
    value: float
    expr: str = ""
    deps: Set[str] = field(default_factory=set)
    literal_numbers: List[float] = field(default_factory=list)
    verified: bool = False
    failure_code: str = ""
    failure_reason: str = ""
    iteration: int = 0


class _ArithmeticVerifier(ast.NodeVisitor):
    """安全求值 GSM8K 候选表达式，并抽取依赖变量。"""

    def __init__(self, variables: Dict[str, float]):
        self.variables = variables
        self.deps: Set[str] = set()
        self.literal_numbers: List[float] = []

    def visit(self, node: ast.AST) -> float:  # type: ignore[override]
        if isinstance(node, ast.Expression):
            return self.visit(node.body)

        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            value = float(node.value)
            self.literal_numbers.append(value)
            return value

        if isinstance(node, ast.Name):
            self.deps.add(node.id)
            if node.id not in self.variables:
                raise NameError(node.id)
            return float(self.variables[node.id])

        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
            return _ALLOWED_UNARY_OPS[type(node.op)](self.visit(node.operand))

        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
            left = self.visit(node.left)
            right = self.visit(node.right)
            return _ALLOWED_BIN_OPS[type(node.op)](left, right)

        raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _safe_eval_expr(expr: str, variables: Dict[str, float]) -> tuple[float, Set[str], List[float]]:
    tree = ast.parse(expr, mode="eval")
    verifier = _ArithmeticVerifier(variables)
    value = verifier.visit(tree)
    if not math.isfinite(value):
        raise ValueError("expression produced a non-finite value")
    return value, verifier.deps, verifier.literal_numbers


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric answer")
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        numeric = float(value.replace(",", "").strip())
    else:
        raise ValueError(f"unsupported numeric type: {type(value).__name__}")
    if not math.isfinite(numeric):
        raise ValueError("non-finite numeric value")
    return numeric


def _json_object(raw_text: str) -> Optional[dict[str, Any]]:
    match = _JSON_OBJECT_RE.search(raw_text.strip())
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class GSM8KNiHaixiaSpec(NiHaixiaSpec):
    """GSM8K 的 GRAVEC/NHX 规约。

    G/R 仍复用原 GSM8KSpec 的 prompt；A/V/E/C 改为确定性函数逻辑。
    """

    def __init__(self, problem_data: Dict[str, str]):
        self._base = GSM8KSpec(problem_data)
        self.question: str = problem_data["question"]
        self.problem_data: Dict[str, str] = problem_data
        self.gold_numeric_answer = self._parse_gold_answer(problem_data["answer"])

        self._mentioned_numbers = self._extract_question_numbers(self.question)
        self._step_records: dict[str, StepRecord] = {}
        self._iteration_history: list[dict[str, Any]] = []
        self._parse_failures: list[StepRecord] = []
        self._soft_goal_hint = ""
        self._soft_revision_count = 0

    # ------------------------------------------------------------------
    # 基础抽取与安全计算
    # ------------------------------------------------------------------
    def _parse_gold_answer(self, answer: Any) -> float:
        text = str(answer)
        matches = _NUMBER_RE.findall(text)
        if not matches:
            return float("nan")
        return float(matches[-1].replace(",", ""))

    def _extract_question_numbers(self, question: str) -> list[float]:
        return [float(n.replace(",", "")) for n in _NUMBER_RE.findall(question)]

    def _numeric_state(self, state: Workspace) -> dict[str, float]:
        numeric: dict[str, float] = {}
        for key, value in state.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                numeric[key] = float(value)
        return numeric

    def _record_failure(self, var: str, expr: str, code: str, reason: str) -> None:
        self._parse_failures.append(
            StepRecord(
                var=var or "__unknown__",
                expr=expr,
                value=float("nan"),
                verified=False,
                failure_code=code,
                failure_reason=reason,
                iteration=len(self._iteration_history),
            )
        )

    def _validate_var_name(self, var_name: str) -> bool:
        return bool(_VAR_RE.fullmatch(var_name))

    def _nearest_parse_failure(self, step: str) -> Optional[StepRecord]:
        for failure in reversed(self._parse_failures):
            if failure.var == step or failure.var == "__unknown__":
                return failure
        return None

    def _question_context_flags(self, var_name: str) -> dict[str, bool]:
        text = f"{self.question} {var_name}".lower()
        count_words = {
            "people", "person", "student", "students", "duck", "ducks", "egg", "eggs",
            "book", "books", "tree", "trees", "apple", "apples", "chair", "chairs",
            "copy", "copies", "item", "items", "how many", "number",
        }
        money_words = {"dollar", "dollars", "$", "cost", "price", "earn", "make", "pay", "sell", "sold"}
        non_negative_words = count_words | money_words | {
            "left", "remain", "remaining", "total", "amount", "remainder", "per day",
        }
        return {
            "count_like": any(word in text for word in count_words),
            "money_like": any(word in text for word in money_words),
            "non_negative": any(word in text for word in non_negative_words),
        }

    def _constraint_check(self, state: Workspace, step: str, value: float) -> tuple[bool, list[str], str]:
        violations: list[str] = []
        flags = self._question_context_flags(step)

        if not math.isfinite(value):
            violations.append(f"{step} is not finite")
            return False, violations, "non_finite"

        if abs(value) > 1e12:
            violations.append(f"{step}={value:g} is implausibly large")

        if flags["non_negative"] and value < -1e-9:
            violations.append(f"{step}={value:g} violates non-negative context")

        if flags["count_like"] and abs(value - round(value)) > 1e-7:
            violations.append(f"{step}={value:g} should be an integer count")

        record = self._step_records.get(step)
        if record and record.deps:
            unknown_deps = [dep for dep in record.deps if dep not in state]
            if unknown_deps:
                violations.append(f"{step} depends on unknown variables: {sorted(unknown_deps)}")

        code = "ok" if not violations else "constraint_violation"
        return not violations, violations, code

    def _relative_error_to_gold(self, value: float) -> Optional[float]:
        if math.isnan(self.gold_numeric_answer):
            return None
        if abs(self.gold_numeric_answer) < 1e-12:
            return abs(value)
        return abs(value - self.gold_numeric_answer) / abs(self.gold_numeric_answer)

    def _has_duplicate_value(self, state: Workspace, step: str, value: float) -> bool:
        for name, other in self._numeric_state(state).items():
            if name != step and abs(other - value) < 1e-9:
                return True
        return False

    # ------------------------------------------------------------------
    # 依赖图：用表达式 deps 构造 dep -> var 的 DAG
    # ------------------------------------------------------------------
    def _dependency_edges(self) -> dict[str, set[str]]:
        edges: dict[str, set[str]] = {}
        for record in self._step_records.values():
            if not record.verified:
                continue
            for dep in record.deps:
                edges.setdefault(dep, set()).add(record.var)
        return edges

    def _shortest_dependency_distance(self, source: str, target: str) -> Optional[int]:
        if source == target:
            return 0
        edges = self._dependency_edges()
        queue: deque[tuple[str, int]] = deque([(source, 0)])
        seen = {source}
        while queue:
            node, dist = queue.popleft()
            for nxt in edges.get(node, set()):
                if nxt == target:
                    return dist + 1
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        return None

    def _structural_progress(self, state: Workspace, step: str, goal: str) -> tuple[float, str]:
        record = self._step_records.get(step)
        if step == goal:
            return 1.0, "目标变量本身"
        if not record:
            return 0.25, "缺少可验证表达式记录"
        if not record.verified:
            return -0.4, record.failure_reason or "表达式未通过验证"

        dist = self._shortest_dependency_distance(step, goal)
        if dist is not None:
            score = max(0.35, 0.9 - 0.15 * dist)
            return score, f"位于通向 {goal} 的依赖路径上，距离 {dist}"

        if record.deps:
            return 0.65, "由已知变量推出的新中间量"

        if record.literal_numbers:
            mentioned = set(round(n, 10) for n in self._mentioned_numbers)
            literals = set(round(n, 10) for n in record.literal_numbers)
            if literals & mentioned:
                return 0.55, "使用题目原始数值形成候选中间量"

        if self._has_duplicate_value(state, step, record.value):
            return 0.2, "数值重复，信息增益较低"

        return 0.45, "新增有效数值，但尚未连接到目标路径"

    # ------------------------------------------------------------------
    # 原版接口：G/R prompt 复用，parse 增强为确定性记录
    # ------------------------------------------------------------------
    def derive_final_target(self, problem: str) -> str:
        return self._base.derive_final_target(problem)

    def parse_workspace_update(self, raw_text: str, state: Workspace) -> Workspace:
        data = _json_object(raw_text)
        if not data:
            parsed = self._base.parse_workspace_update(raw_text, state)
            for var, value in parsed.items():
                if isinstance(value, (int, float)):
                    self._step_records[var] = StepRecord(
                        var=var,
                        value=float(value),
                        expr=str(value),
                        verified=True,
                        iteration=len(self._iteration_history),
                    )
            return parsed

        var_name = str(data.get("var", "")).strip()
        expr = str(data.get("expr", "")).strip()

        if not var_name or not self._validate_var_name(var_name):
            self._record_failure(var_name, expr, "invalid_var", f"invalid variable name: {var_name!r}")
            return Workspace()

        try:
            provided_value = _coerce_float(data.get("value"))
        except Exception as exc:
            self._record_failure(var_name, expr, "invalid_value", str(exc))
            return Workspace()

        deps: Set[str] = set()
        literals: list[float] = []
        if expr:
            try:
                calculated, deps, literals = _safe_eval_expr(expr, self._numeric_state(state))
            except NameError as exc:
                self._record_failure(var_name, expr, "unknown_dependency", f"unknown dependency: {exc}")
                return Workspace()
            except Exception as exc:
                self._record_failure(var_name, expr, "invalid_expr", str(exc))
                return Workspace()

            tolerance = max(1e-6, abs(provided_value) * 1e-7)
            if abs(calculated - provided_value) > tolerance:
                self._record_failure(
                    var_name,
                    expr,
                    "value_mismatch",
                    f"expr evaluates to {calculated:g}, but value is {provided_value:g}",
                )
                return Workspace()

        record = StepRecord(
            var=var_name,
            value=provided_value,
            expr=expr or str(provided_value),
            deps=deps,
            literal_numbers=literals,
            verified=True,
            iteration=len(self._iteration_history),
        )
        self._step_records[var_name] = record
        return Workspace({var_name: provided_value})

    def check_local(self, state: Workspace, target_step: str) -> bool:
        return self._base.check_local(state, target_step)

    def verify_final(self, state: Workspace) -> Tuple[bool, str, float]:
        return self._base.verify_final(state)

    def prompt_last_step(self, state: Workspace, target: str, avoid: Set[str]) -> str:
        prompt = self._base.prompt_last_step(state, target, avoid)
        if self._soft_goal_hint:
            prompt += f"\n\nDeterministic feedback from the verifier:\n{self._soft_goal_hint}\n"
        return prompt

    def prompt_forward_step(self, state: Workspace, target_step: str, avoid: Set[str]) -> str:
        prompt = self._base.prompt_forward_step(state, target_step, avoid)
        prompt += (
            "\nVerifier requirement: the JSON 'expr' must be executable with only numeric "
            "literals and variables already listed in the current state. Do not reference "
            "unknown variables."
        )
        return prompt

    def parse_target_step(self, raw_text: str) -> str:
        return self._base.parse_target_step(raw_text)

    def merge_aliases(self, state: Workspace) -> Workspace:
        return self._base.merge_aliases(state)

    # ------------------------------------------------------------------
    # V：价值判断
    # ------------------------------------------------------------------
    def evaluate_step_value(self, state: Workspace, step: str, goal: str) -> ValueScore:
        if step not in state:
            failure = self._nearest_parse_failure(step)
            reason = failure.failure_reason if failure else "变量不在已知状态中"
            return ValueScore(score=0.0, reason=reason)

        value = state[step]
        if not isinstance(value, (int, float)):
            return ValueScore(score=0.1, reason="非数值变量")

        ok, violations, _ = self._constraint_check(state, step, float(value))
        if not ok:
            return ValueScore(score=-0.5, reason="; ".join(violations), is_primary=False)

        score, reason = self._structural_progress(state, step, goal)
        return ValueScore(score=score, reason=reason, is_primary=score >= 0.8)

    # ------------------------------------------------------------------
    # A：行动执行
    # ------------------------------------------------------------------
    def execute_action(self, state: Workspace, step: str, goal: str) -> Observation:
        if step not in state:
            failure = self._nearest_parse_failure(step)
            return Observation(
                content=failure.failure_reason if failure else "变量不在状态中",
                data={
                    "failure_code": failure.failure_code if failure else "missing_step",
                    "constraints_satisfied": False,
                    "distance_change": "unknown",
                    "distance_to_goal": "unknown",
                },
                observation_type="deterioration",
                confidence=0.75,
            )

        value = state[step]
        if not isinstance(value, (int, float)):
            return Observation(
                content=f"{step} 不是数值类型",
                data={
                    "failure_code": "non_numeric",
                    "constraints_satisfied": False,
                    "distance_change": "unknown",
                    "distance_to_goal": "unknown",
                },
                observation_type="deterioration",
                confidence=0.8,
            )

        numeric_value = float(value)
        ok, violations, failure_code = self._constraint_check(state, step, numeric_value)
        rel_error = self._relative_error_to_gold(numeric_value) if step == goal else None
        value_score, value_reason = self._structural_progress(state, step, goal)

        if not ok:
            obs_type = "deterioration"
            distance_change = "farther"
            content = f"约束违反: {'; '.join(violations)}"
        elif step == goal:
            if rel_error is not None and rel_error < 1e-6:
                obs_type = "improvement"
                distance_change = "closer"
                content = "最终答案通过数值验效"
            else:
                obs_type = "deterioration"
                distance_change = "farther"
                failure_code = "final_wrong"
                content = f"最终答案未通过验效，relative_error={rel_error}"
        elif value_score >= 0.45:
            obs_type = "improvement"
            distance_change = "closer"
            content = f"{step}={numeric_value:g} 是有效中间量：{value_reason}"
        else:
            obs_type = "neutral"
            distance_change = "same"
            content = f"{step}={numeric_value:g} 有效但信息增益较低：{value_reason}"

        if self._has_duplicate_value(state, step, numeric_value) and step != goal:
            obs_type = "neutral"
            distance_change = "same"
            failure_code = "duplicate_value"
            content = f"{step}={numeric_value:g} 与已有变量重复，信息增益低"

        observation_data = {
            "step": step,
            "step_value": numeric_value,
            "value_score": value_score,
            "value_reason": value_reason,
            "constraints_satisfied": ok,
            "violations": violations,
            "contradictions": violations,
            "failure_code": failure_code if not ok or obs_type != "improvement" else "",
            "distance_change": distance_change,
            "distance_to_goal": distance_change,
            "relative_error_to_gold": rel_error,
            "deps": sorted(self._step_records.get(step, StepRecord(step, numeric_value)).deps),
        }

        self._iteration_history.append(
            {
                "step": step,
                "value": numeric_value,
                "observation_type": obs_type,
                "failure_code": observation_data["failure_code"],
                "value_score": value_score,
            }
        )

        return Observation(
            content=content,
            data=observation_data,
            observation_type=obs_type,
            confidence=0.9 if ok else 0.85,
        )

    # ------------------------------------------------------------------
    # E：验效反馈
    # ------------------------------------------------------------------
    def evaluate_observation(self, observation: Observation, state: Workspace, goal: str) -> float:
        type_base = {
            "improvement": 0.55,
            "deterioration": -0.45,
            "neutral": 0.0,
            "surprise": 0.1,
        }
        score = type_base.get(observation.observation_type, 0.0)

        value_score = observation.data.get("value_score")
        if isinstance(value_score, (int, float)):
            score += 0.25 * float(value_score)

        distance_change = observation.data.get("distance_change", observation.data.get("distance_to_goal", "unknown"))
        if distance_change == "closer":
            score += 0.2
        elif distance_change == "farther":
            score -= 0.3

        rel_error = observation.data.get("relative_error_to_gold")
        if isinstance(rel_error, (int, float)):
            if rel_error < 1e-6:
                score = 1.0
            elif rel_error < 0.05:
                score += 0.2
            elif rel_error > 1.0:
                score -= 0.2

        if observation.data.get("contradictions"):
            score -= 0.35

        if not observation.data.get("constraints_satisfied", True):
            score -= 0.4

        return max(-1.0, min(1.0, score * observation.confidence))

    # ------------------------------------------------------------------
    # 果行共变：只修软果，不改 GSM8K 的硬目标 final_answer
    # ------------------------------------------------------------------
    def refine_goal(
        self,
        state: Workspace,
        goal: str,
        observations: List[Observation],
    ) -> Optional[GoalRevision]:
        if len(observations) < 3:
            return None

        recent = observations[-4:]
        bad = [o for o in recent if o.observation_type == "deterioration"]
        stagnant = [o for o in recent if o.data.get("distance_change") in {"same", "unknown"}]
        low_value = [o for o in recent if float(o.data.get("value_score", 0.0)) < 0.3]

        if len(bad) < 2 and len(stagnant) < 3 and len(low_value) < 3:
            return None

        failed_steps = [str(o.data.get("step", "")) for o in recent if o.data.get("step")]
        self._soft_revision_count += 1
        self._soft_goal_hint = (
            "Keep the hard goal as final_answer, but change the soft path: "
            f"avoid low-value or failing steps {failed_steps}; choose a prerequisite "
            "that directly combines already verified numeric variables or original question numbers."
        )

        return GoalRevision(
            revised_goal=goal,
            revision_reason="硬目标保持不变；软路径已根据验效反馈更新",
            confidence=0.75,
            keep_old_as_subgoal=True,
        )

    # ------------------------------------------------------------------
    # C：因果诊断
    # ------------------------------------------------------------------
    def diagnose_cause(
        self,
        state: Workspace,
        step: str,
        observation: Observation,
        goal: str,
    ) -> CausalDiagnosis:
        failure_code = observation.data.get("failure_code", "")

        if failure_code in {"invalid_expr", "invalid_value", "value_mismatch", "unknown_dependency"}:
            return CausalDiagnosis(
                failure_type="wrong_direction",
                description=f"候选表达式未通过确定性校验: {failure_code}",
                suggested_fix="重新生成只依赖已知变量的可执行表达式",
                confidence=0.9,
            )

        if failure_code == "constraint_violation" or not observation.data.get("constraints_satisfied", True):
            return CausalDiagnosis(
                failure_type="wrong_direction",
                description=f"数值约束违反: {'; '.join(observation.data.get('violations', []))}",
                suggested_fix="换一个满足题目数量、非负性和整数约束的中间变量",
                confidence=0.85,
            )

        if failure_code == "final_wrong":
            return CausalDiagnosis(
                failure_type="wrong_direction",
                description="最终答案与 gold answer 不一致",
                suggested_fix="回退到上一个有效中间量，重新组合公式",
                confidence=0.9,
            )

        if failure_code == "duplicate_value":
            return CausalDiagnosis(
                failure_type="confounding_factor",
                description="新变量与已有变量数值重复，信息增益不足",
                suggested_fix="选择能减少未知量的新变量",
                confidence=0.75,
            )

        recent = self._iteration_history[-4:]
        if len(recent) >= 4 and all(item.get("value_score", 0.0) < 0.35 for item in recent):
            return CausalDiagnosis(
                failure_type="wrong_direction",
                description="连续低价值步骤，当前软路径可能错误",
                suggested_fix="保持 final_answer 不变，但更换子目标分解",
                confidence=0.7,
            )

        if observation.data.get("distance_change") == "same":
            return CausalDiagnosis(
                failure_type="insufficient_effort",
                description="步骤有效但尚未形成足够的目标连接",
                suggested_fix="继续推进能组合已有变量的直接前驱",
                confidence=0.55,
            )

        return CausalDiagnosis(
            failure_type="unknown",
            description="未发现硬约束错误，继续探索",
            suggested_fix="保留当前硬目标，尝试不同候选路径",
            confidence=0.4,
        )
