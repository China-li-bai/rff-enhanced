"""
GRAVEC 控制面 — 确定性 G→R→A→V→E→C 主循环

这是"以果决其行"的纪律骨架：
- 循环顺序固定：G→R→A→V→E→C
- 阈值判断确定性：value_threshold, effect_threshold
- 停滞检测确定性：stagnation_window
- 黑名单管理确定性：max_fails_per_var
- 果行共变触发确定性：failure_type + max_goal_revisions

智能面的 Agent 只在 V/A/E/C 四步被调用，
但调用时机和后续分支完全由控制面决定。

与 core_nhx.py 的关系：
  本文件是 core_nhx.py 的重构版，核心逻辑不变，
  但 V/A/E/C 四步从 spec 的确定性方法升级为 Agent-backed。
  迁移期间两者并存，迁移完成后 core_nhx.py 可废弃。
"""

from __future__ import annotations

from typing import List, Set

from ..core import Workspace
from ..llm import llm_call
from .models import (
    CausalDiagnosis,
    GoalRevision,
    Observation,
    ReasoningPolicy,
    ValueScore,
)
from .spec import NiHaixiaSpec


def reason_from_future_gravec(
    problem: str,
    spec: NiHaixiaSpec,
    *,
    max_iters: int = 16,
    min_iters: int = 1,
    require_gold: bool = True,
    model: str | None = None,
    verbose: bool = False,
    value_threshold: float = 0.2,
    effect_threshold: float = 0.1,
    max_goal_revisions: int = 3,
) -> str:
    """GRAVEC「以果决其行」主循环。

    控制面骨架 — 确定性循环纪律：
    ┌─────────────────────────────────────────────────────────────────┐
    │  G (以果): 反向推理 — "要达到目标，先得算什么？"               │
    │       ↓                                                         │
    │  R (推理): 正向计算 — 算出先决条件的值                         │
    │       ↓                                                         │
    │  A (决其行): 行动执行 — Agent 可调工具                         │
    │       ↓                                                         │
    │  V (价值判断): 语义主证识别 — Agent 推理"主证/兼证/矛盾"       │
    │       ↓                                                         │
    │  E (验效): 多通道验效 — Agent 望闻问切                         │
    │       ↓                                                         │
    │  C (校验): 综合重新辨证 — Agent 因果诊断 + 果行共变            │
    │       │                                                         │
    │       ├─ 达成 → 返回答案                                       │
    │       ├─ 未达成但效果好 → 继续推进                             │
    │       ├─ 未达成且效果差 → 因果诊断 → 果行共变                  │
    │       └─ 价值太低 → 跳过此步，尝试其他路径                     │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        problem: 问题描述文本
        spec: 倪海厦增强规约实例
        max_iters: 最大迭代次数
        min_iters: 最少迭代次数
        require_gold: 是否必须和标准答案匹配
        model: LLM 模型名
        verbose: 是否打印详细日志
        value_threshold: 价值判断阈值 — 低于此值的步骤会被降权
        effect_threshold: 验效阈值 — 低于此值触发因果诊断
        max_goal_revisions: 最大目标修正次数

    Returns:
        最终答案字符串

    Raises:
        RuntimeError: 耗尽迭代次数仍未达成目标
    """
    state: Workspace = Workspace()
    goal: str = spec.derive_final_target(problem)
    original_goal: str = goal
    avoid: Set[str] = set()

    attempt_counts: dict[str, int] = {}
    max_fails_per_var: int = 3

    stagnation_counter: int = 0
    stagnation_window: int = 4

    observation_history: List[Observation] = []
    goal_revision_count: int = 0
    low_value_steps: List[str] = []

    def register_fail(symbol: str) -> None:
        nonlocal attempt_counts, avoid
        attempt_counts[symbol] = attempt_counts.get(symbol, 0) + 1
        if attempt_counts[symbol] >= max_fails_per_var:
            avoid.add(symbol)

    # ====================================================================
    # 主迭代循环：G → R → A → V → E → C
    # ====================================================================
    for iter_idx in range(max_iters):
        made_progress: bool = False

        if verbose:
            print(f"\n{'='*60}")
            print(f"迭代 {iter_idx + 1}/{max_iters} | 目标: {goal}")
            print(f"已知变量: {list(state.keys())}")
            print(f"{'='*60}")

        policy = spec.select_reasoning_policy(
            state, goal, iter_idx, observation_history, avoid
        )

        if verbose:
            print(
                f"[策略] {policy.name} "
                f"(流派={policy.treatment_approach})"
            )

        # ================================================================
        # C 前置检查：目标是否已在草稿纸上？
        # ================================================================
        if spec.check_local(state, goal):
            if not require_gold and iter_idx >= (min_iters - 1):
                return str(state[goal])
            ok, answer_from_llm, gold_val_for_debug = spec.verify_final(state)
            if ok:
                return answer_from_llm
            register_fail(goal)

        # ================================================================
        # G (Goal/以果)：反向推理 — 从目标往回看
        # ================================================================
        if spec.should_attempt_direct_goal(state, goal, iter_idx, avoid):
            direct_prompt = spec.render_prompt_with_policy(
                spec.prompt_forward_step(state, goal, avoid),
                policy,
                "direct",
            )
            direct_raw = llm_call(direct_prompt, model=model, verbose=verbose)
            direct_state = state | spec.parse_workspace_update(direct_raw, state)

            if spec.check_local(direct_state, goal):
                if not require_gold and iter_idx >= (min_iters - 1):
                    return str(direct_state[goal])
                ok, answer_from_llm, gold_val_for_debug = spec.verify_final(direct_state)
                if ok:
                    return answer_from_llm
                elif verbose:
                    print(
                        f"[E-验效] 直接尝试失败: LLM={answer_from_llm}, "
                        f"标准={gold_val_for_debug}"
                    )

            state = direct_state
            register_fail(goal)

        # ================================================================
        # G (Goal/以果)：反向推理 — 确定先决条件
        # ================================================================
        g_prompt = spec.render_prompt_with_policy(
            spec.prompt_last_step(state, goal, avoid),
            policy,
            "backward",
        )
        raw_target_step_response = llm_call(g_prompt, model=model, verbose=verbose)
        target_step = spec.parse_target_step(raw_target_step_response)

        if not target_step or target_step in avoid:
            if verbose:
                print(f"[G-以果] 无效的先决条件: '{target_step}', 跳过本轮")
            continue

        if verbose:
            print(f"[G-以果] 先决条件: {target_step}")

        # ================================================================
        # R (Reason/推理)：正向计算
        # ================================================================
        r_prompt = spec.render_prompt_with_policy(
            spec.prompt_forward_step(state, target_step, avoid),
            policy,
            "forward",
        )
        forward_raw = llm_call(r_prompt, model=model, verbose=verbose)
        parsed_update = spec.parse_workspace_update(forward_raw, state)

        llm_provided_var = None
        if parsed_update:
            llm_provided_var_keys = list(parsed_update.keys())
            if llm_provided_var_keys:
                llm_provided_var = llm_provided_var_keys[0]

        if not parsed_update or not llm_provided_var:
            register_fail(target_step)
            if verbose:
                print("[R-推理] LLM 未返回有效结果, 跳过本轮")
            continue

        if verbose:
            print(f"[R-推理] 计算结果: {llm_provided_var} = {parsed_update.get(llm_provided_var)}")

        # ---- 情况 A：LLM 直接算了目标变量 ----
        if llm_provided_var == goal:
            temp_state = state | parsed_update
            if spec.check_local(temp_state, goal):
                if not require_gold and iter_idx >= (min_iters - 1):
                    return str(temp_state[goal])
                ok, answer_from_llm, gold_val_for_debug = spec.verify_final(temp_state)
                if ok:
                    return answer_from_llm
                elif verbose:
                    print(
                        f"[E-验效] 目标验证失败: LLM={answer_from_llm}, "
                        f"标准={gold_val_for_debug}"
                    )
                register_fail(goal)
                if target_step != goal:
                    register_fail(target_step)
                continue
            else:
                register_fail(goal)
                if target_step != goal:
                    register_fail(target_step)
                continue

        # ---- 情况 B：LLM 算了中间变量 ----
        elif llm_provided_var == target_step:
            temp_state = state | parsed_update
            if spec.check_local(temp_state, target_step):
                state = temp_state
                made_progress = True
            else:
                register_fail(target_step)
                continue

        # ---- 情况 C：LLM 给了无关内容 ----
        else:
            register_fail(target_step)
            continue

        # ================================================================
        # A (Action/决其行)：行动执行
        # ================================================================
        step_to_evaluate = target_step if llm_provided_var == target_step else llm_provided_var
        observation = spec.execute_action(state, step_to_evaluate, goal)
        observation_history.append(observation)

        if verbose:
            print(
                f"[A-决其行] step={step_to_evaluate}, "
                f"观察类型={observation.observation_type}, "
                f"通道={observation.channel}, "
                f"置信度={observation.confidence:.2f}"
            )

        # ================================================================
        # V (Value/价值判断)：评估步骤贡献度
        # ================================================================
        value_score = spec.evaluate_step_value(state, step_to_evaluate, goal)

        if verbose:
            print(
                f"[V-价值判断] 价值分={value_score.score:.2f}, "
                f"证型={value_score.syndrome_type}, "
                f"主证={value_score.is_primary}, "
                f"原因={value_score.reason}"
            )

        if value_score.score < value_threshold:
            low_value_steps.append(step_to_evaluate)
            if verbose:
                print(
                    f"[V-价值判断] 低价值步骤 (score={value_score.score:.2f} "
                    f"< threshold={value_threshold}), 降权处理"
                )
            if value_score.score < 0.0:
                register_fail(step_to_evaluate)
                if verbose:
                    print("[V-价值判断] 矛盾步骤, 加入黑名单")

        # ================================================================
        # E (Effect/验效)：检验行动效果
        # ================================================================
        effect_score = spec.evaluate_observation(observation, state, goal)

        if verbose:
            print(f"[E-验效] 效果分={effect_score:.2f}")

        if effect_score < effect_threshold:
            # ---- 效果不好 → 因果诊断 ----
            diagnosis = spec.diagnose_cause(state, step_to_evaluate, observation, goal)

            if verbose:
                print(
                    f"[E-验效] 效果差, 因果诊断: "
                    f"类型={diagnosis.failure_type}, "
                    f"描述={diagnosis.description}, "
                    f"建议={diagnosis.suggested_fix}"
                )

            # ============================================================
            # C (Check/校验) + 果行共变：根据诊断修正目标
            # ============================================================
            if (
                diagnosis.failure_type == "wrong_direction"
                and goal_revision_count < max_goal_revisions
            ):
                revision = spec.refine_goal(state, goal, observation_history)
                if revision is not None and revision.revised_goal != goal:
                    goal_revision_count += 1
                    if verbose:
                        print(
                            f"[C-果行共变] 目标修正: "
                            f"{goal} -> {revision.revised_goal} "
                            f"(原因: {revision.revision_reason}, "
                            f"置信度: {revision.confidence:.2f})"
                        )
                    if revision.keep_old_as_subgoal:
                        avoid.discard(goal)
                    goal = revision.revised_goal

        # ================================================================
        # C (Check/校验)：后处理
        # ================================================================
        state = spec.merge_aliases(state)

        if made_progress:
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= stagnation_window:
            avoid = {s for s, cnt in attempt_counts.items() if cnt >= max_fails_per_var}
            stagnation_counter = 0
            if verbose:
                print(f"[C-校验] 软重启: 放宽黑名单, 当前avoid={avoid}")

        # ---- 检查修正后的目标是否已达成 ----
        if goal != original_goal and spec.check_local(state, goal):
            if not require_gold and iter_idx >= (min_iters - 1):
                return str(state[goal])
            ok, answer_from_llm, gold_val_for_debug = spec.verify_final(state)
            if ok:
                return answer_from_llm

    raise RuntimeError(
        f"GRAVEC 耗尽迭代次数 (目标修正 {goal_revision_count} 次). "
        f"最终目标: {goal}, 原始目标: {original_goal}"
    )
