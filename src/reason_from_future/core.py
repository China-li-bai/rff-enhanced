"""
================================================================================
RFF (Reason-from-Future) 核心引擎 — core.py
================================================================================

【费曼视角：一句话讲清楚 RFF 是什么】
想象你要去一个未知的城市旅游。你不知道怎么走，但你知道目的地是"故宫"。
普通的 AI 推理是："我有地图，我从当前位置一步步走到故宫"（正向推理）。
RFF 的推理是："我要到故宫。要到故宫，先得到天安门。要到天安门，先得到前门..." 
然后反过来正向算："OK，我先走到前门，再走到天安门，最后到故宫。"
——从未来（目标）往回看，找出"要达到目标，必须先完成什么"，
然后用正向计算把先决条件一个个攻克。

【Python 入门知识】
这个文件展示了 Python 中几个重要概念：
1. ABC + @abstractmethod：抽象基类 —— 定义了"子类必须实现哪些方法"的合同
   比喻：面试官的"岗位要求"，你不满足就不能上岗（不能实例化）
2. dict 子类化：Workspace 继承了 dict 的所有功能，同时加了自定义方法
3. __or__ 运算符重载：让 Workspace 支持 `a | b` 合并操作
4. nonlocal：嵌套函数中修改外层变量
5. * 在函数参数中：强制后面的参数用关键字传递

【跨文件关系】
- llm_call() 来自 llm.py：所有与 Gemini 的通信都通过它
- ProblemSpec 被 specs/ 下的 Game24Spec、GSM8KSpec 等继承
- Workspace 被 specs/ 中的所有实现用作"草稿纸"
- 入口文件是 demos/demo_xxx.py 和 interactive_cli.py
"""
from __future__ import annotations  # [Python知识] 支持 dict[str, Any] 这样的类型注解写法

from abc import ABC, abstractmethod  # [Python知识] ABC=抽象基类，abstractmethod=抽象方法的标记器
from typing import Any, Tuple, Set   # [Python知识] 类型注解工具
from .llm import llm_call            # [跨文件] 从同目录的 llm.py 导入 LLM 调用函数


# ============================================================================
# 第一部分：Workspace — 推理过程的"草稿纸"
# ============================================================================
class Workspace(dict):
    """推理过程中的草稿纸/工作台。
    
    【费曼解释】
    就像你做数学题时用的草稿纸：
    - 每算出一个中间结果，就写在草稿纸上
    - 后面的计算要用到前面的结果，就从草稿纸上找
    - 最后看草稿纸上有没有"最终答案"
    
    Workspace 本质上就是一个字典（dict），但加了几个方便的方法。
    为什么继承 dict 而不是内部包含一个 dict？
    → 因为 self["x"] = 10 比 self.data["x"] = 10 更直觉、更简洁。
    
    【跨文件】几乎每个 spec 都会读写 Workspace：
    - gsm8k.py 的 parse_workspace_update() 往里写键值对
    - core.py 的 reason_from_future() 主循环往里读/写状态
    - game24.py 的 check_local() 检查里面有没有目标值
    """

    # -------------------- 写入辅助方法 --------------------
    def add(self, key: str, val: Any) -> None:
        """以更语义化的方式存储键值对。等价于 self[key] = val。
        
        [Python知识] `val: Any` 中的 `Any` 表示可以是任意类型。
        在 Python 中类型注解只是"建议"，不强制，编译器不会阻止你传错类型。
        """
        self[key] = val

    # -------------------- 读取辅助方法（供 spec 调用） --------------------
    def get_all_data(self) -> dict[str, Any]:
        """返回整个工作台的状态（以普通 dict 形式）。
        
        [跨文件] CodeWritingSpec 调用 state.get_all_data() 判断草稿纸是否为空。
        因为 Workspace 本身就是 dict 的子类，直接返回 self 即可。
        """
        return self

    def get_internal_state_DEBUG(self) -> dict[str, Any]:
        """调试用：和 get_all_data 一样，向后兼容。"""
        return self

    # -------------------- | 运算符重载（合并两个 Workspace）--------------------
    def __or__(self, other: dict[str, Any]):  # type: ignore[override]
        """定义 Workspace | dict 的行为：返回合并后的新 Workspace。
        
        [费曼解释]
        你有两张草稿纸（Workspace），想合并成一张新的：
        new = paper_a | paper_b  
        → 先抄 paper_a 的内容，再抄 paper_b 的内容（重复的以 paper_b 为准）
        → 返回的还是一张草稿纸（Workspace），不是普通纸（dict）
        
        [Python知识]
        Python 3.9+ 的普通 dict 也支持 `|`，但返回的是普通 dict，
        会丢失 Workspace 的自定义方法。重写 __or__ 保证结果还是 Workspace。
        
        [跨文件] reason_from_future() 主循环多处使用：
        state | parsed_update  来合并 LLM 新算出的结果到当前状态
        """
        combined = Workspace()
        combined.update(self)
        if isinstance(other, dict):
            combined.update(other)
        else:
            raise TypeError("Can only merge Workspace with dict-like object using '|'")
        return combined

    def __ror__(self, other: dict[str, Any]):  # type: ignore[override]
        """定义 dict | Workspace 的行为（Workspace 在 | 右边时）。
        
        [Python知识] __or__ 处理 `self | other`（左合并），
        __ror__ 处理 `other | self`（右合并，r = reversed/reverse）。
        """
        combined = Workspace()
        if isinstance(other, dict):
            combined.update(other)
        else:
            raise TypeError("Can only merge Workspace with dict-like object using '|'")
        combined.update(self)
        return combined


# ============================================================================
# 第二部分：异常类
# ============================================================================
class LocalCheckFail(Exception):
    """正向步骤的局部校验失败时抛出的异常。
    
    [说明] 当前版本的主循环没有实际抛出这个异常（改用 register_fail 处理），
    它是预留的异常类型，供未来扩展使用。
    """


# ============================================================================
# 第三部分：ProblemSpec — 问题规约的"合同"（最核心的抽象）
# ============================================================================
class ProblemSpec(ABC):
    """所有推理领域必须遵守的"合同"。
    
    【费曼解释 - 什么是 ProblemSpec】
    想象你在一个工厂里：
    - reason_from_future() 是生产流水线的主控程序（通用的，不关心具体做什么产品）
    - ProblemSpec 是"产品规格书"（针对每种产品的特殊要求）
    - 要做新产品？只需要写一份新的"规格书"，不用改流水线！
    
    这就是"策略模式"（Strategy Pattern）：把"做什么"（领域逻辑）和"怎么做"（控制流程）分离。
    
    【Python知识 - 抽象基类（ABC）】
    ABC (Abstract Base Class) = 抽象基类，像一份"合同"：
    - 用 @abstractmethod 标记的方法，子类 MUST 实现，否则实例化时报错
    - 不能直接创建 ProblemSpec() 的实例（会报 TypeError）
    - 比喻：你不能雇一个"没有具体技能的人"，只能雇"程序员"或"设计师"
    
    【跨文件】Game24Spec、GSM8KSpec、CodeWritingSpec、GeneralProblemSolvingSpec 
    都继承 ProblemSpec，各自实现这 8 个方法。
    """

    @abstractmethod
    def derive_final_target(self, problem: str) -> str:
        """确定最终目标的变量名。在循环开始前调用一次，之后 goal 不会再变。
        
        [费曼解释] "你最终要算什么？"
        - 对于数学题：返回 "final_answer"
        - 对于 24 点游戏：返回 "24"
        - 对于代码编写：返回 "complete_code_solution"
        
        [跨文件] 由 reason_from_future() 在初始化时调用，赋值给 goal 常量。
        """

    @abstractmethod
    def parse_workspace_update(self, raw_text: str, state: Workspace) -> Workspace:
        """把 LLM 的原始输出解析成 Workspace 中的结构化数据。
        
        [费曼解释] LLM 是一只会说话但格式不稳定的"鹦鹉"。它可能返回：
        - "答案：42" 
        - {"var": "x", "value": 42}
        - "根据计算...最终答案是 42"
        这个方法就是"翻译官"，把鹦鹉的话翻译成草稿纸上规范的格式。
        
        [Python知识] 这个方法里大量使用正则表达式（re 模块）和 JSON 解析，
        因为 LLM 的输出格式不够稳定，需要多种 fallback 策略。
        
        [跨文件] reason_from_future() 每次调用 LLM 后都调用此方法解析返回结果。
        """

    @abstractmethod
    def check_local(self, state: Workspace, target_step: str) -> bool:
        """局部校验：检查某个变量是否"看起来合理"。
        
        [费曼解释] "这步算出来的结果，格式对不对、数值在不在合理范围？"
        这像是做完一道题后的"自检"——不对答案，只检查有没有明显的格式错误。
        
        和 verify_final 的区别：
        - check_local：只检查格式和合理性（便宜、快）
        - verify_final：要和标准答案比对（可能贵、慢）
        
        [跨文件] reason_from_future() 在每次 LLM 正向计算后调用。
        对于 GSM8K：检查值是否是数值类型
        对于 Game24：检查工作台中是否有表达式等于目标值
        """

    @abstractmethod
    def verify_final(self, state: Workspace) -> Tuple[bool, str, float]:
        """终极验证：检查最终答案是否正确。
        
        返回 (是否正确, LLM给出的答案字符串, 标准答案数值)。
        这是最严格的一关——和标准答案（ground truth）比对。
        
        [跨文件] reason_from_future() 在认为找到答案时调用，决定是否结束循环。
        """

    @abstractmethod
    def prompt_last_step(self, state: Workspace, target: str, avoid: Set[str]) -> str:
        """构建"反向推理"的提示词：要算出 target，先得算什么？
        
        [费曼解释] 这是 RFF 的核心！
        "我知道要去故宫（target），但我现在在前门（state）。要到故宫，先得到哪？"
        → LLM 回答："天安门"（target_step）
        → 然后我们正向去算天安门的位置
        → 直到能直接算到故宫
        
        avoid 参数是"试过但失败的变量名"集合，防止 LLM 反复提同一个方案。
        
        [跨文件] reason_from_future() 在每个迭代的第 2 步调用。
        """

    @abstractmethod
    def prompt_forward_step(self, state: Workspace, target_step: str, avoid: Set[str]) -> str:
        """构建"正向计算"的提示词：请计算 target_step 的值。
        
        [费曼解释] "OK，我们确定了要算'天安门的坐标'（target_step）。
        现在已知前门坐标是 (x1, y1)，请算出天安门坐标。"
        
        [跨文件] reason_from_future() 在反向推理确定 target_step 后调用。
        """

    @abstractmethod
    def parse_target_step(self, raw_text: str) -> str:
        """从 LLM 的反向推理输出中提取"下一步要算什么"。
        
        [费曼解释] LLM 可能说：
        "根据分析，要得到答案，需要先计算'总书籍数'..."
        → 这个方法从中提取出变量名字符串："总书籍数"
        
        [跨文件] reason_from_future() 在反向推理（prompt_last_step）的 LLM 调用后调用。
        """

    @abstractmethod
    def merge_aliases(self, state: Workspace) -> Workspace:
        """合并草稿纸中的同义词/别名。
        
        [费曼解释] LLM 可能在不同步骤中用不同名字指同一个东西：
        - "initial_science_books" 和 "science_books_before_bonus" 可能是一个意思
        - 如果不合并，后续推理会"看不到"已算出的值
        
        [跨文件] reason_from_future() 在每个迭代结束时调用。
        """


# ============================================================================
# 第四部分：主循环 — RFF 算法的核心引擎
# ============================================================================
def reason_from_future(
    problem: str,          # 问题描述文本（如数学题原文）：[跨文件] 透传到 spec 的 prompt 构建中
    spec: ProblemSpec,     # 问题规约实例：[跨文件] 所有领域逻辑都在 spec 里
    *,                     # [Python知识] * 表示后面的参数必须用关键字传递，不能按位置传
    max_iters: int = 16,   # 最大迭代次数：防止无限循环
    min_iters: int = 1,    # 最少迭代次数：防止过早接受"看起来不错"的错误答案
    require_gold: bool = True,  # 是否必须和标准答案匹配：True=严格模式，False=宽松模式
    model: str | None = None,
    verbose: bool = False,  # 是否打印详细日志
) -> str:
    """RFF (Reason-from-Future) 主循环。
    
    ============================================================================
    【费曼视角：完整算法解释】
    ============================================================================
    
    想象你要解一道复杂的数学题。正常的做法是从已知条件出发，一步步推导到答案。
    但 RFF 反其道而行之：
    
    ┌─────────────────────────────────────────────────────────────┐
    │  RFF 循环 = G → R → C 三步曲                                │
    │                                                              │
    │  G (Goal/Backward)：从目标往回看 → "要算 X，先得算什么？"     │
    │  R (Reasoning/Forward)：正向计算那个先决条件                  │
    │  C (Check)：检查结果是否合理                                 │
    │                                                              │
    │  重复以上步骤，直到最终目标被满足                              │
    └─────────────────────────────────────────────────────────────┘
    
    具体来说，每次迭代分 5 个步骤：
    
    步骤0：初始化
    → goal = "final_answer" (对数学题) 或 "24" (对24点游戏)
    → state = {} (空草稿纸)
    → avoid = {} (还没什么要避免的)
    
    步骤1：检查目标是否已在草稿纸上
    → 如果 goal 已经在 state 里且格式正确 → 验证是不是对的 → 对了就返回！
    → 如果不对 → 标记 goal 失败，下次不直接尝试它
    
    步骤1b：快速通道 — 直接用当前知识尝试计算目标
    → 跳过反向推理，直接让 LLM 用已知变量计算 goal
    → 如果对了就返回；如果错了，保留过程中发现的其他有用变量
    
    步骤2：反向推理 (G) — "要算 goal，先得算什么？"
    → 调用 spec.prompt_last_step(state, goal, avoid) 构建提示词
    → 调用 llm_call() 发送给 Gemini
    → 调用 spec.parse_target_step() 提取 LLM 返回的变量名
    → 得到 target_step（如 "total_science_books"）
    
    步骤3：正向计算 (R) — "请计算 target_step 的值"
    → 调用 spec.prompt_forward_step(state, target_step, avoid) 构建提示词
    → 调用 llm_call() 发送给 Gemini
    → 调用 spec.parse_workspace_update() 解析 LLM 的返回
    → 调用 spec.check_local() 局部校验
    
    步骤3后续：处理 LLM 的返回结果
    → 情况A：LLM 算了 goal（目标变量）→ 验证，对了返回，错了不更新 state
    → 情况B：LLM 算了 target_step（先决变量）→ 校验通过则更新 state，标记 made_progress
    → 情况C：LLM 给了无关内容 → 放弃，标记失败
    
    步骤4：后处理
    → merge_aliases(state)：合并同义词
    → 停滞检测：连续 4 轮没进展 → 软重启（放宽 avoid 限制）
    
    ============================================================================
    【跨文件追踪：一次完整的 RFF 迭代中每个方法来自哪里】
    ============================================================================
    
    调用链：
    reason_from_future()           ← core.py (本文件)
      ├── spec.derive_final_target()   ← gsm8k.py / game24.py / ...
      ├── spec.check_local()           ← gsm8k.py / game24.py / ...
      ├── spec.verify_final()          ← gsm8k.py / game24.py / ...
      ├── spec.prompt_forward_step()   ← gsm8k.py / game24.py / ...
      │     └── llm_call()             ← llm.py (Gemini API 封装)
      ├── spec.prompt_last_step()      ← gsm8k.py / game24.py / ...
      │     └── llm_call()             ← llm.py
      ├── spec.parse_target_step()     ← gsm8k.py / game24.py / ...
      ├── spec.parse_workspace_update()← gsm8k.py / game24.py / ...
      └── spec.merge_aliases()         ← gsm8k.py / game24.py / ...
    
    ============================================================================
    【Python入门知识 - 函数参数中的 *】
    ============================================================================
    `def f(a, b, *, c=1)` 中的 `*` 是一个分隔符：
    - * 前面的参数可以按位置传递：f(1, 2)
    - * 后面的参数必须用关键字传递：f(1, 2, c=3) ✅  f(1, 2, 3) ❌
    这样做的好处：防止调用者搞混参数的顺序。
    """
    
    # ---- 初始化：准备草稿纸和主循环变量 ----
    state: Workspace = Workspace()                    # 空草稿纸，啥都没写
    goal: str = spec.derive_final_target(problem)     # [跨文件] 调用 spec 的方法确定最终目标
    avoid: Set[str] = set()                           # 黑名单：已失败或已完成的变量名
    
    # [费曼解释] attempt_counts：每个变量尝试了几次、失败了几次
    # 同一个变量失败 3 次就拉黑（加入 avoid），防止 LLM 在死胡同里打转
    attempt_counts: dict[str, int] = {}
    max_fails_per_var: int = 3
    
    # [费曼解释] 停滞检测机制：如果连续几轮都没进展，说明可能卡住了
    # 此时触发"软重启"——只保留真正失败 3 次的变量在黑名单，其他放开重新试
    stagnation_counter: int = 0   # 连续没进展的轮数
    stagnation_window: int = 4    # 4 轮没进展就软重启
    
    # ---- 辅助函数：记录失败 ----
    def register_fail(symbol: str) -> None:
        """某个变量计算失败时调用。失败 3 次后加入 avoid 黑名单。
        
        [Python知识] `nonlocal` 关键字让嵌套函数能修改外层函数的变量。
        没有 nonlocal 的话，Python 会认为 attempt_counts 是局部变量（赋值即创建），
        然后因为"引用未赋值变量"而报 UnboundLocalError。
        
        类比：nonlocal 就像给内层函数一把钥匙，能打开外层的抽屉修改内容。
        """
        nonlocal attempt_counts, avoid  # [Python知识] 声明要修改外层变量
        attempt_counts[symbol] = attempt_counts.get(symbol, 0) + 1
        if attempt_counts[symbol] >= max_fails_per_var:
            avoid.add(symbol)  # 拉黑！下次 LLM 不能再提这个变量
    
    # ====================================================================
    # 主迭代循环
    # ====================================================================
    for iter_idx in range(max_iters):
        made_progress: bool = False  # 本轮是否有实质进展
        
        # ----------------------------------------------------------------
        # 步骤 1：检查目标是否已经在草稿纸上
        # ----------------------------------------------------------------
        if spec.check_local(state, goal):
            # [费曼解释] goal 已经在 state 中存在且格式正确
            # 但需要验证内容是否正确：
            #   宽松模式 (require_gold=False) + 迭代次数够 → 直接接受
            #   严格模式 (require_gold=True) → 和标准答案比对
            if not require_gold and iter_idx >= (min_iters - 1):
                return str(state[goal])  # 宽松模式：接受当前值
            ok, answer_from_llm, gold_val_for_debug = spec.verify_final(state)
            if ok:
                return answer_from_llm  # 🎉 验证通过，成功！
            # 验证失败：值存在但不对
            register_fail(goal)
        
        # ----------------------------------------------------------------
        # 步骤 1b：快速通道 — 直接尝试计算目标
        # ----------------------------------------------------------------
        # [费曼解释] 有时候不需要反向推理，LLM 直接用已知变量就能算出答案。
        # 这步先试一下，省一轮反向推理。
        if goal not in avoid:
            direct_prompt = spec.prompt_forward_step(state, goal, avoid)
            # [跨文件] llm_call() 来自 llm.py，封装了对 Gemini API 的调用
            direct_raw = llm_call(direct_prompt, model=model, verbose=verbose)
            # [Python知识] `state | parsed_update` 使用 Workspace 的 __or__ 合并
            # 结果还是 Workspace，保留了自定义方法
            direct_state = state | spec.parse_workspace_update(direct_raw, state)
            if spec.check_local(direct_state, goal):
                if not require_gold and iter_idx >= (min_iters - 1):
                    return str(direct_state[goal])
                ok, answer_from_llm, gold_val_for_debug = spec.verify_final(direct_state)
                if ok:
                    return answer_from_llm  # 🎉 直接算对了！
                elif verbose:
                    print(
                        f"INFO (direct attempt): LLM proposed final_answer="
                        f"'{answer_from_llm}', but gold_answer='{gold_val_for_debug}'. "
                        f"Adding to avoid list."
                    )
            # 关键设计：即使目标没算对，LLM 可能在过程中算出了其他有用的中间变量
            # 所以仍然更新 state（保留那些正确的新变量）
            state = direct_state
            register_fail(goal)
        
        # ----------------------------------------------------------------
        # 步骤 2：反向推理 (G) — "要算出 goal，先得算什么？"
        # ----------------------------------------------------------------
        # [费曼解释] 这就是 RFF 的核心！从未来往回看。
        #   已知：目标 = goal（如 "final_answer"）
        #   问题：在当前状态 state 下，要算出 goal，先得算出哪个变量？
        #   就像：要去故宫，先得到天安门
        g_prompt = spec.prompt_last_step(state, goal, avoid)
        raw_target_step_response = llm_call(g_prompt, model=model, verbose=verbose)
        target_step = spec.parse_target_step(raw_target_step_response)
        
        # 防御性检查：LLM 可能返回空值或已拉黑的变量
        if not target_step or target_step in avoid:
            continue  # 跳过本轮，用新的迭代
        
        # ----------------------------------------------------------------
        # 步骤 3：正向计算 (R) — "请计算 target_step 的值"
        # ----------------------------------------------------------------
        # [费曼解释] 反向推理确定了"要先算出天安门的坐标"
        # 现在正向去算：以已知状态为基础，计算 target_step
        r_prompt = spec.prompt_forward_step(state, target_step, avoid)
        forward_raw = llm_call(r_prompt, model=model, verbose=verbose)
        parsed_update = spec.parse_workspace_update(forward_raw, state)
        
        # LLM 不一定听话 —— 你让它算 x，它可能算出了 y 或 final_answer
        # 必须判断 LLM 实际返回了什么
        llm_provided_var = None
        if parsed_update:
            llm_provided_var_keys = list(parsed_update.keys())
            if llm_provided_var_keys:
                llm_provided_var = llm_provided_var_keys[0]
        
        # ---- 情况 A：LLM 直接算了目标变量 (goal) ----
        if llm_provided_var == goal:
            # LLM 声称算出了最终答案
            temp_state_for_verification = state | parsed_update
            if spec.check_local(temp_state_for_verification, goal):
                if not require_gold and iter_idx >= (min_iters - 1):
                    return str(temp_state_for_verification[goal])
                ok, answer_from_llm, gold_val_for_debug = spec.verify_final(
                    temp_state_for_verification
                )
                if ok:
                    return answer_from_llm  # 🎉
                elif verbose:
                    print(
                        f"INFO (after computing '{target_step}'): LLM proposed "
                        f"final_answer='{answer_from_llm}', but gold_answer="
                        f"'{gold_val_for_debug}'. Adding to avoid list."
                    )
                else:
                    register_fail(goal)
                    if target_step != goal:
                        register_fail(target_step)
                    # ⚠️ 关键设计：错误的答案绝不能写入 state！
                    # 否则下一轮循环会看到 goal 已存在，反复验证同一个错误值
                    continue
            else:
                # check_local 失败（格式不对等）
                register_fail(goal)
                if target_step != goal:
                    register_fail(target_step)
                continue
        
        # ---- 情况 B：LLM 算了我们要求的中间变量 (target_step) ----
        elif llm_provided_var == target_step:
            temp_state_for_target_step = state | parsed_update
            if spec.check_local(temp_state_for_target_step, target_step):
                # ✅ 校验通过！把新结果写入草稿纸
                state = temp_state_for_target_step
                register_fail(target_step)  # 防止 LLM 后面重复算这个变量
                made_progress = True
            else:
                # ❌ 校验失败
                register_fail(target_step)
                continue
        
        # ---- 情况 C：LLM 给了无关内容 ----
        else:
            register_fail(target_step)
            continue
        
        # ----------------------------------------------------------------
        # 步骤 4：后处理 — 合并同义词 + 停滞检测
        # ----------------------------------------------------------------
        # [费曼解释] LLM 可能用不同名字指同一个东西
        # 例如 "initial_science_books" 和 "science_books_before_bonus"
        # merge_aliases 尝试合并这些同义词
        state = spec.merge_aliases(state)
        
        # 停滞检测
        if made_progress:
            stagnation_counter = 0  # 有进展，重置计数器
        else:
            stagnation_counter += 1  # 没进展，+1
        
        # 软重启：连续 4 轮没进展 → 放宽约束
        if stagnation_counter >= stagnation_window:
            # 只保留真正失败 3 次的变量在黑名单中
            avoid = {s for s, cnt in attempt_counts.items() if cnt >= max_fails_per_var}
            stagnation_counter = 0
    
    # 循环耗尽：所有迭代次数用完了还没解出来
    raise RuntimeError("RFF exhausted iterations without solution.")