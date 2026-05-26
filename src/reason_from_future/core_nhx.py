"""
================================================================================
倪海厦「以果决其行」增强核心 — core_nhx.py
================================================================================

【费曼视角：一句话讲清楚这个文件是什么】
原来的 RFF 只会"想"（推理），不会"做"（行动），更不会"看效果调方案"（验效）。
这个文件在 RFF 的 G→R→C 三步曲基础上，补上了倪海厦方法论最核心的四个缺失：

  ┌──────────────────────────────────────────────────────────────────────┐
  │  原版 RFF：G → R → C（只会想，不会做）                              │
  │                                                                      │
  │  倪海厦版：G → R → A → V → E → C（想完就做，做完就验，验完就调）     │
  │                                                                      │
  │  G (Goal/以果):    反向推理 — 从目标往回看                           │
  │  R (Reason/推理):  正向计算 — 算出先决条件的值                       │
  │  A (Action/决其行): 行动执行 — 基于推理结果采取行动                  │
  │  V (Value/价值判断): 评估这一步对目标的贡献度                        │
  │  E (Effect/验效):  检验行动效果，获取反馈                           │
  │  C (Check/校验):   综合判断 + 果行共变（目标随反馈修正）             │
  └──────────────────────────────────────────────────────────────────────┘

【倪海厦方法论映射】
  中医实践                    →  代码对应
  ─────────────────────────────────────────
  望闻问切（收集信息）         →  R（正向推理得到中间结果）
  辨证（判断证型）             →  V（价值判断：这个结果指向什么方向）
  论治（决定治疗方案）         →  A（行动执行：基于推理采取行动）
  开方/扎针（实施治疗）        →  A（行动执行的具体实现）
  复诊（看效果）               →  E（验效：行动后观察效果）
  调方/重新辨证（果行共变）    →  C + refine_goal（目标随反馈修正）

  核心原则：「以果决其行」——不是"我觉得该这样做"，而是"要达到那个果，
  必须这样做"。每一步行动都由目标（果）驱动，每一步效果都反馈给目标修正。

【Python 入门知识】
1. @dataclass：自动生成 __init__、__repr__ 等方法的装饰器
   比喻：你定义一个"学生"类，只要写 name 和 age 两个字段，
   @dataclass 自动帮你写好构造函数和打印格式
2. dataclass.field()：给字段设默认值或默认工厂
3. Optional[X]：表示值可以是 X 类型或 None
4. List[X]：表示元素类型为 X 的列表

【跨文件关系】
- 继承 core.py 的 Workspace 和 ProblemSpec
- 使用 llm.py 的 llm_call() 与 LLM 通信
- 被 specs/gsm8k.py 等具体领域实现继承
"""
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set, Tuple

from .core import ProblemSpec, Workspace
from .llm import llm_call


# ============================================================================
# 第一部分：数据结构 — 倪海厦方法论的核心概念
# ============================================================================

@dataclass
class Observation:
    """行动后的观察结果 — 倪海厦的"复诊记录"。

    【费曼解释】
    你给病人开了药（行动），病人回来复诊（观察）。
    你需要记录：
    - 病人说了什么（content：主观描述）
    - 体征数据（data：客观指标，如体温37.2°C）
    - 整体趋势（observation_type：好转/恶化/不变/意外）
    - 你有多确信（confidence：0.0=完全不确定，1.0=非常确信）

    【倪海厦类比】
    倪师看诊时，复诊记录是最重要的——"一诊开方，二诊看效，效不更方，不效调方"。
    Observation 就是"二诊看效"的数据载体。

    【在六步曲中的位置】
    A（行动执行）产生 Observation → E（验效）消费 Observation
    """

    content: str
    data: dict[str, Any] = field(default_factory=dict)
    observation_type: str = "neutral"
    confidence: float = 0.5

    VALID_TYPES = frozenset({"improvement", "deterioration", "neutral", "surprise"})

    def __post_init__(self):
        if self.observation_type not in self.VALID_TYPES:
            self.observation_type = "neutral"
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ValueScore:
    """价值判断结果 — 倪海厦的"主证 vs 兼证"评估。

    【费曼解释】
    中医辨证时，不是所有症状都同等重要：
    - 主证（如：舌红苔黄腻）→ 直接决定证型，score: 0.8~1.0
    - 兼证（如：口微渴）→ 辅助确认，score: 0.3~0.7
    - 无关（如：穿红衣服）→ 和诊断无关，score: 0.0~0.2
    - 矛盾（如：脉沉细但舌红）→ 需要重新辨证，score: -0.5~0.0

    ValueScore 就是给每一步推理打分：这步对最终目标的贡献有多大？

    【倪海厦类比】
    倪师强调"抓主证"——不是所有信息都值得花精力。
    主证是"果"的直接线索，兼证是辅助，矛盾是"重新辨证"的信号。

    【在六步曲中的位置】
    V（价值判断）产生 ValueScore → 影响后续行动的优先级和资源分配
    """

    score: float = 0.0
    reason: str = ""
    is_primary: bool = False

    def __post_init__(self):
        self.score = max(-1.0, min(1.0, self.score))
        self.is_primary = self.score >= 0.8


@dataclass
class GoalRevision:
    """目标修正结果 — 倪海厦的"重新辨证"。

    【费曼解释】
    有时候你治了半天没效果，不是因为药不好，而是诊断方向就错了！
    倪师说"效不更方，不效调方"——没效果就要重新思考。

    GoalRevision 记录：
    - 修正后的目标（revised_goal：新的辨证方向）
    - 为什么修正（revision_reason：原方向为什么不对）
    - 修正的置信度（confidence：你有多确信新方向是对的）
    - 是否保留旧目标作为子目标（keep_old_as_subgoal：旧方向不完全放弃）

    【倪海厦类比】
    这是最深层的设计——倪师认为"果"不是固定的。
    随着治疗推进，你对"果"的理解会深化，甚至完全改变。
    比如一开始以为是"热证"，治了发现不对，重新辨证为"真寒假热"。

    【在六步曲中的位置】
    C（校验）中触发 → 如果验效反馈表明目标理解有误，调用 refine_goal()
    """

    revised_goal: str = ""
    revision_reason: str = ""
    confidence: float = 0.5
    keep_old_as_subgoal: bool = True

    def __post_init__(self):
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class CausalDiagnosis:
    """因果诊断结果 — 倪海厦的"为什么没效？"分析。

    【费曼解释】
    当验效发现行动没效果时，需要诊断"为什么"。
    就像倪师复诊时会问："为什么这药没效？是辨证错了？还是药量不够？还是兼证没处理？"

    CausalDiagnosis 记录：
    - 失败原因分类（failure_type：辨证错/药量错/兼证干扰/意外因素）
    - 具体描述（description：用自然语言解释）
    - 修正建议（suggested_fix：下一步应该怎么做）
    - 置信度（confidence：诊断的可靠程度）

    【在六步曲中的位置】
    E（验效）发现效果不好时 → 触发因果诊断 → 结果反馈给 C（校验）
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
# 第二部分：NiHaixiaSpec — 倪海厦增强规约
# ============================================================================
class NiHaixiaSpec(ProblemSpec):
    """倪海厦「以果决其行」增强规约。

    【费曼解释 - 为什么需要这个类】
    原版 ProblemSpec 定义了 8 个方法，都是关于"推理"的：
    怎么反向推理、怎么正向计算、怎么校验结果……
    但它完全没有关于"行动"和"验效"的方法！

    NiHaixiaSpec 在 ProblemSpec 的基础上新增 4 个方法：
    1. evaluate_step_value() — 价值判断（这步值不值得做？）
    2. execute_action()      — 行动执行（做了之后观察到什么？）
    3. evaluate_observation() — 验效反馈（效果好不好？）
    4. refine_goal()         — 果行共变（要不要调整目标？）

    【倪海厦类比】
    ProblemSpec = 只会"望闻问切 + 辨证"的医生（只诊断不治疗）
    NiHaixiaSpec = 完整的倪师：诊断 → 开方 → 看效 → 调方 → 再诊

    【Python知识 - 继承链】
    ProblemSpec (ABC, 8个@abstractmethod)
      └── NiHaixiaSpec (新增4个@abstractmethod)
            └── GSM8KNiHaixiaSpec (具体实现，在 gsm8k.py 中)

    子类必须实现全部 12 个方法（8个继承的 + 4个新增的）才能实例化。
    """

    # ----------------------------------------------------------------
    # 新增方法 1：价值判断
    # ----------------------------------------------------------------
    @abstractmethod
    def evaluate_step_value(
        self, state: Workspace, step: str, goal: str
    ) -> ValueScore:
        """价值判断：这一步对目标的贡献度是多少？

        倪海厦类比：这个症状（step）对判断最终证型（goal）的价值是多少？
        - 主证 → score: 0.8~1.0（如：舌红苔黄 → 直接指向"热证"）
        - 兼证 → score: 0.3~0.7（如：口微渴 → 辅助确认）
        - 无关 → score: 0.0~0.2（如：穿红衣服 → 和诊断无关）
        - 矛盾 → score: -0.5~0.0（如：脉沉细但舌红 → 需重新辨证）

        返回 ValueScore，包含分数、原因、是否为主证。

        【在六步曲中的位置】
        R（正向计算）之后 → V（价值判断）→ 决定是否值得继续推进
        """

    # ----------------------------------------------------------------
    # 新增方法 2：行动执行
    # ----------------------------------------------------------------
    @abstractmethod
    def execute_action(
        self, state: Workspace, step: str, goal: str
    ) -> Observation:
        """行动执行：基于推理结果执行一个行动并返回观察。

        倪海厦类比：开方/扎针（行动）→ 病人反馈（观察）

        例如：
        - GSM8K：用算出的中间值代入原题验证自洽性
        - CodeWriting：创建文件、运行代码、查看输出
        - Game24：验证表达式是否使用了正确的数字

        返回 Observation，包含观察内容、数据、类型和置信度。

        【在六步曲中的位置】
        V（价值判断）确认值得做 → A（行动执行）→ 产生 Observation
        """

    # ----------------------------------------------------------------
    # 新增方法 3：验效反馈
    # ----------------------------------------------------------------
    @abstractmethod
    def evaluate_observation(
        self, observation: Observation, state: Workspace, goal: str
    ) -> float:
        """验效反馈：评估行动效果。返回改善程度。

        倪海厦类比：复诊——病人吃了药，好转了多少？
        - 正值 = 好转（如：0.5 表示改善了一半）
        - 负值 = 恶化（如：-0.3 表示情况变差了）
        - 零 = 无变化

        【在六步曲中的位置】
        A（行动执行）产生 Observation → E（验效）评估效果
        """

    # ----------------------------------------------------------------
    # 新增方法 4：果行共变
    # ----------------------------------------------------------------
    @abstractmethod
    def refine_goal(
        self, state: Workspace, goal: str, observations: List[Observation]
    ) -> Optional[GoalRevision]:
        """果行共变：根据反馈修正目标。

        倪海厦类比：复诊后重新辨证——原来的诊断方向可能不对。

        返回 None 表示目标不需要修正（效不更方）。
        返回 GoalRevision 表示需要修正目标（不效调方）。

        这是倪海厦方法论最深层的设计——
        目标不是固定的，而是随着行动反馈共同演化的。

        【在六步曲中的位置】
        E（验效）发现效果不好 → C（校验）中触发 refine_goal()
        """

    # ----------------------------------------------------------------
    # 新增方法 5：因果诊断
    # ----------------------------------------------------------------
    @abstractmethod
    def diagnose_cause(
        self, state: Workspace, step: str, observation: Observation, goal: str
    ) -> CausalDiagnosis:
        """因果诊断：行动没效果时，分析"为什么"。

        倪海厦类比：为什么这药没效？是辨证错了？还是药量不够？
        - wrong_direction：辨证方向就错了（价值判断为负）
        - insufficient_effort：方向对但力度不够（需要更多步骤）
        - confounding_factor：有干扰因素（兼证没处理）
        - unexpected：意外情况（需要重新审视问题）
        - unknown：暂时无法判断

        【在六步曲中的位置】
        E（验效）发现效果差 → 触发因果诊断 → 结果反馈给 C（校验）
        """


# ============================================================================
# 第三部分：主循环 — 倪海厦版「以果决其行」六步曲
# ============================================================================
def reason_from_future_nhx(
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
    """倪海厦版「以果决其行」主循环。

    ============================================================================
    【费曼视角：完整算法解释】
    ============================================================================

    原版 RFF：G → R → C（只会想，不会做）
    倪海厦版：G → R → A → V → E → C（想完就做，做完就验，验完就调）

    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  G (以果): 反向推理 — "要达到目标，先得算什么？"                     │
    │       ↓                                                             │
    │  R (推理): 正向计算 — 算出先决条件的值                              │
    │       ↓                                                             │
    │  A (决其行): 行动执行 — 基于推理结果采取行动                        │
    │       ↓                                                             │
    │  V (价值判断): 评估这一步的贡献度 — 值不值得继续？                  │
    │       ↓                                                             │
    │  E (验效): 检验行动效果 — 效果好不好？                              │
    │       ↓                                                             │
    │  C (校验): 综合判断 — 达成目标了吗？                                │
    │       │                                                             │
    │       ├─ 达成 → 返回答案 🎉                                        │
    │       ├─ 未达成但效果好 → 继续推进                                 │
    │       ├─ 未达成且效果差 → 因果诊断 → 果行共变（修正目标）          │
    │       └─ 价值太低 → 跳过此步，尝试其他路径                         │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

    【倪海厦方法论映射】
    "以果决其行" = G（以果）→ A（决其行）
    "效不更方" = E（验效）效果好 → 继续当前方向
    "不效调方" = E（验效）效果差 → 因果诊断 + 果行共变
    "抓主证" = V（价值判断）is_primary = True → 优先处理

    ============================================================================
    【参数说明】
    ============================================================================
    problem:           问题描述文本
    spec:              倪海厦增强规约实例（必须实现12个方法）
    max_iters:         最大迭代次数
    min_iters:         最少迭代次数
    require_gold:      是否必须和标准答案匹配
    model:             LLM 模型名
    verbose:           是否打印详细日志
    value_threshold:   价值判断阈值 — 低于此值的步骤会被降权
    effect_threshold:  验效阈值 — 低于此值触发因果诊断
    max_goal_revisions:最大目标修正次数 — 防止无限修正
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
        if goal not in avoid:
            direct_prompt = spec.prompt_forward_step(state, goal, avoid)
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
                        f"[V-验效] 直接尝试失败: LLM={answer_from_llm}, "
                        f"标准={gold_val_for_debug}"
                    )

            state = direct_state
            register_fail(goal)

        # ================================================================
        # G (Goal/以果)：反向推理 — 确定先决条件
        # ================================================================
        g_prompt = spec.prompt_last_step(state, goal, avoid)
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
        r_prompt = spec.prompt_forward_step(state, target_step, avoid)
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
                print(f"[R-推理] LLM 未返回有效结果, 跳过本轮")
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
                        f"[V-验效] 目标验证失败: LLM={answer_from_llm}, "
                        f"标准={gold_val_for_debug}"
                    )
                else:
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
                register_fail(target_step)
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
                f"[A-决其行] 行动执行: step={step_to_evaluate}, "
                f"观察类型={observation.observation_type}, "
                f"置信度={observation.confidence:.2f}"
            )

        # ================================================================
        # V (Value/价值判断)：评估步骤贡献度
        # ================================================================
        value_score = spec.evaluate_step_value(state, step_to_evaluate, goal)

        if verbose:
            print(
                f"[V-价值判断] 价值分={value_score.score:.2f}, "
                f"主证={value_score.is_primary}, "
                f"原因={value_score.reason}"
            )

        if value_score.score < value_threshold:
            low_value_steps.append(step_to_evaluate)
            if verbose:
                print(
                    f"[V-价值判断] ⚠️ 低价值步骤 (score={value_score.score:.2f} "
                    f"< threshold={value_threshold}), 降权处理"
                )
            if value_score.score < 0.0:
                register_fail(step_to_evaluate)
                if verbose:
                    print(f"[V-价值判断] ❌ 矛盾步骤, 加入黑名单")

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
                    f"[E-验效] ⚠️ 效果差, 因果诊断: "
                    f"类型={diagnosis.failure_type}, "
                    f"描述={diagnosis.description}, "
                    f"建议={diagnosis.suggested_fix}"
                )

            # ============================================================
            # C (Check/校验) + 果行共变：根据诊断修正目标
            # ============================================================
            if diagnosis.failure_type == "wrong_direction" and goal_revision_count < max_goal_revisions:
                revision = spec.refine_goal(state, goal, observation_history)
                if revision is not None and revision.revised_goal != goal:
                    goal_revision_count += 1
                    if verbose:
                        print(
                            f"[C-果行共变] 🔄 目标修正: "
                            f"{goal} → {revision.revised_goal} "
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
                print(f"[C-校验] 🔄 软重启: 放宽黑名单, 当前avoid={avoid}")

        # ---- 检查修正后的目标是否已达成 ----
        if goal != original_goal and spec.check_local(state, goal):
            if not require_gold and iter_idx >= (min_iters - 1):
                return str(state[goal])
            ok, answer_from_llm, gold_val_for_debug = spec.verify_final(state)
            if ok:
                return answer_from_llm

    raise RuntimeError(
        f"倪海厦版 RFF 耗尽迭代次数 (目标修正 {goal_revision_count} 次). "
        f"最终目标: {goal}, 原始目标: {original_goal}"
    )
