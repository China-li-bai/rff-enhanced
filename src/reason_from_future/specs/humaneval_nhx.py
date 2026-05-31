"""
================================================================================
HumanEval 倪海厦增强规约 — humaneval_nhx.py
================================================================================

【费曼视角：为什么 HumanEval 比 GSM8K 更适合测试「以果决其行」】

GSM8K 是"算术题"——答案固定、一次算对就行、没有验效反馈的必要。
HumanEval 是"编程题"——有测试用例作为"果"，代码可以执行验证，
失败了需要因果诊断（逻辑错？边界条件？类型错误？），
有时候需要换思路（果行共变），完美映射倪海厦的诊疗循环：

  G (以果):     测试用例就是"果"——函数必须通过这些测试
  R (推理):     分析需求，生成代码
  A (决其行):   执行代码，运行测试
  V (价值判断): 这段代码覆盖了哪些测试？核心逻辑对不对？
  E (验效):     测试失败了？看报错信息
  C (校验):     因果诊断——是逻辑错？边界条件？类型错误？
                果行共变——也许需要换一种算法思路

【倪海厦类比】
GSM8K = 只能"望闻问切"，不能"开方试药"（无法执行验证）
HumanEval = 完整诊疗：辨证→开方→看效→不效调方

【Workspace 结构】
{
    "function_name": str,           # 函数名
    "function_signature": str,      # 函数签名
    "docstring": str,               # 文档字符串
    "current_code": str,            # 当前生成的代码
    "test_results": {               # 测试执行结果
        "passed": int,
        "failed": int,
        "errors": [str],
    },
    "debug_attempts": int,          # 调试尝试次数
    "approach": str,                # 当前算法思路描述
}
"""
from __future__ import annotations

import json
import re
import textwrap
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

from ..core import Workspace
from ..core_nhx import (
    CausalDiagnosis,
    GoalRevision,
    NiHaixiaSpec,
    Observation,
    ReasoningPolicy,
    ValueScore,
)
from ..llm import llm_call


class HumanEvalNiHaixiaSpec(NiHaixiaSpec):
    """HumanEval 倪海厦增强规约。

    HumanEval 每道题是一个 Python 函数签名 + docstring + 测试用例。
    目标是生成能通过所有测试用例的函数实现。

    与 GSM8K 的关键区别：
    1. 可以执行代码验证（A-行动执行有真实反馈）
    2. 测试失败有具体错误信息（E-验效有诊断线索）
    3. 可能需要换算法思路（C-果行共变有实际意义）
    4. 代码质量有高低之分（V-价值判断有区分度）
    """

    def __init__(self, problem_data: Dict[str, Any]):
        self.task_id: str = problem_data.get("task_id", "")
        self.prompt: str = problem_data.get("prompt", "")
        self.canonical_solution: str = problem_data.get("canonical_solution", "")
        self.test: str = problem_data.get("test", "")
        self.entry_point: str = problem_data.get("entry_point", "")

        self._full_problem = self.prompt + "\n    pass\n"

        func_match = re.search(r"def\s+(\w+)", self.prompt)
        self.function_name: str = func_match.group(1) if func_match else self.entry_point

    def _extract_code(self, raw_text: str) -> str:
        """从 LLM 输出中提取 Python 代码。"""
        code_blocks = re.findall(r"```(?:python)?\s*\n([\s\S]*?)```", raw_text)
        if code_blocks:
            return code_blocks[0].strip()

        lines = raw_text.strip().splitlines()
        code_lines = []
        in_function = False
        for line in lines:
            if re.match(r"def\s+", line):
                in_function = True
            if in_function:
                code_lines.append(line)
        if code_lines:
            return "\n".join(code_lines)

        return raw_text.strip()

    def _complete_code(self, code: str) -> str:
        """将函数实现补全为可执行代码。"""
        if code.strip().startswith("def "):
            return code
        return self.prompt + code

    def _run_tests(self, code: str) -> Dict[str, Any]:
        """执行测试用例并返回结果。"""
        full_code = self._complete_code(code)
        test_code = full_code + "\n" + self.test + f"\ncheck({self.entry_point})\n"

        try:
            exec_globals: dict[str, Any] = {}
            exec(test_code, exec_globals)
            return {"passed": True, "errors": [], "error_count": 0}
        except AssertionError as e:
            return {"passed": False, "errors": [f"AssertionError: {e}"], "error_count": 1}
        except Exception:
            tb = traceback.format_exc()
            short_tb = "\n".join(tb.splitlines()[-5:])
            return {"passed": False, "errors": [short_tb], "error_count": 1}

    def _get_current_code(self, state: Workspace) -> str:
        """从 Workspace 获取当前代码。"""
        return state.get("current_code", "")

    # ====================================================================
    # 原版 8 个方法
    # ====================================================================

    def derive_final_target(self, problem: str) -> str:
        return "pass_all_tests"

    def parse_workspace_update(self, raw_text: str, state: Workspace) -> Workspace:
        code = self._extract_code(raw_text)
        if not code:
            return Workspace()

        update = Workspace({"current_code": code})

        test_result = self._run_tests(code)
        update["test_results"] = test_result
        update["debug_attempts"] = state.get("debug_attempts", 0) + 1

        return update

    def check_local(self, state: Workspace, target_step: str) -> bool:
        if target_step == "pass_all_tests":
            results = state.get("test_results", {})
            return results.get("passed", False)
        return target_step in state

    def verify_final(self, state: Workspace) -> Tuple[bool, str, float]:
        results = state.get("test_results", {})
        passed = results.get("passed", False)
        code = state.get("current_code", "")
        return passed, code, 1.0 if passed else 0.0

    def prompt_last_step(self, state: Workspace, target: str, avoid: Set[str]) -> str:
        current_code = self._get_current_code(state)
        test_results = state.get("test_results", {})
        debug_attempts = state.get("debug_attempts", 0)

        context = f"Function signature and docstring:\n{self.prompt}"

        if current_code:
            context += f"\n\nCurrent implementation:\n```python\n{current_code}\n```"

        if test_results and not test_results.get("passed", True):
            errors = test_results.get("errors", [])
            context += "\n\nTest failures:\n" + "\n".join(errors[:3])

        avoid_str = ""
        if avoid:
            avoid_str = f"\nAvoid these approaches (already tried and failed): {', '.join(avoid)}"

        prompt = textwrap.dedent(f"""
            You are debugging a Python function. The goal is to make it pass all test cases.

            {context}

            Debug attempt #{debug_attempts + 1}

            What is the SINGLE MOST IMPORTANT next step to fix this function?
            Choose one:
            - "fix_logic_error" — the algorithm/logic is wrong
            - "fix_edge_case" — missing edge case handling
            - "fix_type_error" — type mismatch or wrong return type
            - "fix_syntax_error" — syntax or indentation error
            - "rewrite_approach" — current approach is fundamentally wrong, need new strategy
            - "pass_all_tests" — the code already passes all tests

            {avoid_str}

            Output a single JSON: {{"next_step": "step_name"}}
        """).strip()
        return prompt

    def prompt_forward_step(self, state: Workspace, target_step: str, avoid: Set[str]) -> str:
        current_code = self._get_current_code(state)
        test_results = state.get("test_results", {})
        debug_attempts = state.get("debug_attempts", 0)
        approach = state.get("approach", "")

        context = f"Function signature and docstring:\n{self.prompt}"

        if current_code:
            context += f"\n\nCurrent buggy implementation:\n```python\n{current_code}\n```"

        if test_results and not test_results.get("passed", True):
            errors = test_results.get("errors", [])
            context += "\n\nTest error output:\n" + "\n".join(errors[:3])

        if approach:
            context += f"\n\nCurrent approach: {approach}"

        avoid_str = ""
        if avoid:
            avoid_str = f"\nAvoid these approaches: {', '.join(avoid)}"

        if target_step == "pass_all_tests" and not current_code:
            instruction = "Write the complete function implementation that passes all test cases."
        elif target_step == "fix_logic_error":
            instruction = "Fix the logic/algorithm error in the implementation. Return the COMPLETE corrected function."
        elif target_step == "fix_edge_case":
            instruction = "Add edge case handling. Return the COMPLETE corrected function."
        elif target_step == "fix_type_error":
            instruction = "Fix the type error. Return the COMPLETE corrected function."
        elif target_step == "fix_syntax_error":
            instruction = "Fix the syntax/indentation error. Return the COMPLETE corrected function."
        elif target_step == "rewrite_approach":
            instruction = "The current approach is fundamentally wrong. Rewrite the function with a DIFFERENT algorithm/strategy. Return the COMPLETE new implementation."
        else:
            instruction = f"Implement step: {target_step}. Return the COMPLETE function implementation."

        prompt = textwrap.dedent(f"""
            You are implementing a Python function.

            {context}

            Task: {instruction}

            Debug attempt #{debug_attempts + 1}
            {avoid_str}

            Return the COMPLETE function implementation in a ```python code block.
            Do NOT include the test code. Only the function definition and its body.
        """).strip()
        return prompt

    def parse_target_step(self, raw_text: str) -> str:
        try:
            match = re.search(r"\{[\s\S]*?\}", raw_text)
            if match:
                data = json.loads(match.group(0))
                step = data.get("next_step", "")
                if step:
                    return step.strip()
        except (json.JSONDecodeError, KeyError):
            pass

        valid_steps = {
            "fix_logic_error", "fix_edge_case", "fix_type_error",
            "fix_syntax_error", "rewrite_approach", "pass_all_tests",
        }
        for step in valid_steps:
            if step in raw_text.lower():
                return step

        return "fix_logic_error"

    def merge_aliases(self, state: Workspace) -> Workspace:
        return state

    def render_prompt_with_policy(
        self,
        prompt: str,
        policy: ReasoningPolicy,
        phase: str,
    ) -> str:
        return textwrap.dedent(f"""
            {prompt}

            GRAVEC format/feedback control for this {phase} phase:
            - {policy.name}: {policy.instruction}
            Use this feedback internally, but still return only the complete
            function implementation in the requested Python code block.
        """).strip()

    # ====================================================================
    # 新增方法 1：价值判断
    # ====================================================================
    def evaluate_step_value(
        self, state: Workspace, step: str, goal: str
    ) -> ValueScore:
        current_code = self._get_current_code(state)
        test_results = state.get("test_results", {})
        debug_attempts = state.get("debug_attempts", 0)

        if not current_code:
            return ValueScore(score=0.9, reason="首次生成代码，价值极高", is_primary=True)

        if test_results.get("passed", False):
            return ValueScore(score=1.0, reason="代码已通过所有测试", is_primary=True)

        errors = test_results.get("errors", [])
        error_text = "\n".join(errors[:2])

        if "SyntaxError" in error_text or "IndentationError" in error_text:
            return ValueScore(score=0.8, reason="语法错误，修复成本低", is_primary=True)

        if "TypeError" in error_text:
            return ValueScore(score=0.7, reason="类型错误，方向可能正确", is_primary=False)

        if "AssertionError" in error_text:
            return ValueScore(score=0.6, reason="断言失败，逻辑基本正确但边界条件未处理", is_primary=False)

        if debug_attempts <= 2:
            return ValueScore(score=0.7, reason="早期调试，方向可能正确", is_primary=False)

        if debug_attempts <= 5:
            return ValueScore(score=0.4, reason="多次调试未果，方向可能有问题", is_primary=False)

        return ValueScore(score=0.1, reason="调试次数过多，需要换思路", is_primary=False)

    # ====================================================================
    # 新增方法 2：行动执行
    # ====================================================================
    def execute_action(
        self, state: Workspace, step: str, goal: str
    ) -> Observation:
        current_code = self._get_current_code(state)
        test_results = state.get("test_results", {})

        if not current_code:
            return Observation(
                content="没有代码可执行",
                data={},
                observation_type="neutral",
                confidence=0.3,
            )

        if test_results.get("passed", False):
            return Observation(
                content="代码通过所有测试用例",
                data={"passed": True},
                observation_type="improvement",
                confidence=1.0,
            )

        errors = test_results.get("errors", [])
        error_text = "\n".join(errors[:2])

        obs_type = "deterioration"
        if "AssertionError" in error_text:
            obs_type = "neutral"
        elif "SyntaxError" in error_text or "IndentationError" in error_text:
            obs_type = "deterioration"
        elif "TypeError" in error_text:
            obs_type = "neutral"

        return Observation(
            content=f"测试失败: {error_text[:200]}",
            data={"errors": errors, "passed": False},
            observation_type=obs_type,
            confidence=0.8,
        )

    # ====================================================================
    # 新增方法 3：验效反馈
    # ====================================================================
    def evaluate_observation(
        self, observation: Observation, state: Workspace, goal: str
    ) -> float:
        if observation.data.get("passed", False):
            return 1.0

        type_scores = {
            "improvement": 0.5,
            "deterioration": -0.3,
            "neutral": 0.0,
            "surprise": 0.2,
        }
        base = type_scores.get(observation.observation_type, 0.0)

        debug_attempts = state.get("debug_attempts", 0)
        if debug_attempts > 5:
            base -= 0.2

        return max(-1.0, min(1.0, base * observation.confidence))

    # ====================================================================
    # 新增方法 4：果行共变
    # ====================================================================
    def refine_goal(
        self, state: Workspace, goal: str, observations: List[Observation]
    ) -> Optional[GoalRevision]:
        recent = observations[-3:]
        deterioration_count = sum(
            1 for o in recent if o.observation_type == "deterioration"
        )
        if deterioration_count < 2:
            return None

        debug_attempts = state.get("debug_attempts", 0)
        if debug_attempts < 3:
            return None

        current_code = self._get_current_code(state)
        errors = []
        for o in recent:
            errors.extend(o.data.get("errors", []))

        prompt = textwrap.dedent(f"""
            You are diagnosing why a code implementation keeps failing.

            Function signature:
            {self.prompt}

            Current failing code:
            ```python
            {current_code}
            ```

            Recent errors:
            {chr(10).join(errors[:3])}

            Debug attempts so far: {debug_attempts}

            Should we completely rewrite with a different approach?
            If yes, describe the new approach briefly.

            Output JSON:
            {{
                "needs_revision": true/false,
                "revised_approach": "description of new approach if revision needed",
                "reason": "why current approach is failing"
            }}
        """).strip()

        try:
            raw = llm_call(prompt, verbose=False)
            match = re.search(r"\{[\s\S]*?\}", raw)
            if match:
                data = json.loads(match.group(0))
                if not data.get("needs_revision", False):
                    return None
                return GoalRevision(
                    revised_goal="rewrite_approach",
                    revision_reason=str(data.get("reason", "")),
                    confidence=0.7,
                    keep_old_as_subgoal=False,
                )
        except Exception:
            pass

        return GoalRevision(
            revised_goal="rewrite_approach",
            revision_reason=f"调试{debug_attempts}次仍未通过，建议换思路",
            confidence=0.5,
            keep_old_as_subgoal=False,
        )

    # ====================================================================
    # 新增方法 5：因果诊断
    # ====================================================================
    def diagnose_cause(
        self, state: Workspace, step: str, observation: Observation, goal: str
    ) -> CausalDiagnosis:
        errors = observation.data.get("errors", [])
        error_text = "\n".join(errors[:2])

        if "SyntaxError" in error_text or "IndentationError" in error_text:
            return CausalDiagnosis(
                failure_type="wrong_direction",
                description="语法错误，代码结构有问题",
                suggested_fix="fix_syntax_error",
                confidence=0.9,
            )

        if "TypeError" in error_text:
            return CausalDiagnosis(
                failure_type="wrong_direction",
                description="类型错误，返回值类型或参数类型不匹配",
                suggested_fix="fix_type_error",
                confidence=0.8,
            )

        if "AssertionError" in error_text:
            return CausalDiagnosis(
                failure_type="insufficient_effort",
                description="逻辑正确但边界条件或特殊情况未处理",
                suggested_fix="fix_edge_case",
                confidence=0.7,
            )

        debug_attempts = state.get("debug_attempts", 0)
        if debug_attempts >= 4:
            return CausalDiagnosis(
                failure_type="wrong_direction",
                description="多次调试未果，算法思路可能根本不对",
                suggested_fix="rewrite_approach",
                confidence=0.6,
            )

        return CausalDiagnosis(
            failure_type="insufficient_effort",
            description="逻辑有小错误，需要进一步调试",
            suggested_fix="fix_logic_error",
            confidence=0.5,
        )
