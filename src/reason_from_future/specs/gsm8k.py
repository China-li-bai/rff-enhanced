"""
================================================================================
GSM8K 数学题规约 — gsm8k.py
================================================================================

【费曼视角：一句话讲清楚】
这是"小学数学应用题"的问题规约。它定义了：
1. 怎么读题（derive_final_target） → 目标永远是 "final_answer"
2. 怎么理解 LLM 的回答（parse_workspace_update） → 从 JSON 或文本中提取数值
3. 怎么检查答案对不对（check_local / verify_final） → 检查数值类型 / 和标准答案比对
4. 怎么让 LLM 反向思考（prompt_last_step） → "要算答案，先得算什么？"
5. 怎么让 LLM 正向计算（prompt_forward_step） → "请计算 X 的值"

这就是实现一个 ProblemSpec 的完整模板！看懂了这个文件，你就知道怎么给其他问题类型写"规则书"。

【跨文件关系】
- 继承 core.py 的 ProblemSpec 抽象基类
- 所有方法都被 core.py 的 reason_from_future() 主循环调用
- parse_workspace_update 返回的 Workspace 数据被 core.py 存入 state

【Python 入门知识重点】
- json.loads()：把 JSON 字符串转成 Python 字典
- re.search()：正则表达式搜索，用于从 LLM 的混乱输出中提取结构化数据
- textwrap.dedent()：去掉多行字符串的公共缩进
- eval()（安全的用法）：在受限的命名空间中执行简单表达式
"""
import json
import re
import textwrap
from typing import Dict, Set, Tuple

from ..core import Workspace, ProblemSpec  # [跨文件] ..表示上级目录的 core.py


# flake8: noqa: E501  # [Python知识] 告诉代码检查工具"行太长不要报错"


class GSM8KSpec(ProblemSpec):
    """GSM8K 小学数学应用题的规约实现。
    
    GSM8K 是一个包含 8500 道小学数学题的数据集。
    每道题有一个"问题"（question）和一个"答案"（answer）。
    答案格式通常是：解释... #### 数字
    
    例如：
      question: "Natalia sold 48 clips. Half were small, half were large..."
      answer: "Natalia sold 48/2 = 24 clips. #### 24"
    """

    def __init__(self, problem_data: Dict[str, str]):
        """创建 GSM8K 规约实例。
        
        [Python知识] super().__init__() 调用父类（ProblemSpec）的构造函数。
        不调用也不会报错（因为 ABC 的 __init__ 是空的），但这是个好习惯。
        
        [参数]
        problem_data: {"question": "题目文本", "answer": "答案文本（含 #### 数字）"}
        """
        super().__init__()
        self.question: str = problem_data["question"]
        self.problem_data: Dict[str, str] = problem_data

        # [Python知识] 从答案字符串中提取数字部分
        # 格式："解释文字... #### 数字" 或 直接是数字
        answer_str = str(problem_data["answer"])
        # [Python知识] re.search() 在字符串中搜索匹配正则表达式的部分
        # r"(?:####\s*)?([0-9,.]+)\s*$" 解释：
        #   (?:####\s*)?  → 可选的 "#### " 前缀（?:表示不捕获）
        #   ([0-9,.]+)    → 捕获数字、逗号、小数点（如 "1,234.56"）
        #   \s*$          → 行末可选空格
        match = re.search(r"(?:####\s*)?([0-9,.]+)\s*$", answer_str)
        if match:
            self.gold_numeric_answer: float = float(match.group(1).replace(",", ""))
        else:
            print(f"Warning: Could not parse gold numeric answer from: '{answer_str}'")
            self.gold_numeric_answer: float = float('nan')  # [Python知识] nan = Not a Number

    # ====================================================================
    # 【8个抽象方法实现】— 以下方法都是 ProblemSpec 的"合同要求"
    # ====================================================================

    # ---- 1. derive_final_target ----
    def derive_final_target(self, problem: str) -> str:
        """确定最终目标的变量名。GSM8K 始终是 "final_answer"。
        
        [费曼解释] "你最后要算的东西叫什么名字？"
        对数学题来说，不管题怎么变，最终答案都叫 "final_answer"。
        这就是最简单的实现——直接返回常量。
        
        [跨文件] 在 reason_from_future() 初始化时调用，赋值给不变的 goal 变量。
        """
        return "final_answer"

    # ---- 2. parse_workspace_update（最复杂的方法）----
    def parse_workspace_update(self, raw_text: str, state: Workspace) -> Workspace:
        """把 LLM 的原始输出解析成 Workspace 中的键值对。
        
        [费曼解释] 这是最复杂的方法，因为 LLM 的输出太不可预测了！
        LLM 可能返回：
          - {"var": "x", "expr": "a+b", "value": 42}     ← 理想格式
          - "The final answer is $\\boxed{42}$"           ← 自然语言
          - "```json\n{\"var\": \"x\", ...}\n```"         ← Markdown 代码块
          - "根据计算...答案是 42"                         ← 纯中文
        
        这个方法就是"垃圾回收站"——用多层 fallback 策略尽量从中提取有用信息。
        
        [Python知识] 多层 fallback：try-except 处理"可能出错"的代码，出错后优雅降级。
        比喻：抢银行计划A失败 → 触发计划B → 计划B失败 → 触发计划C → 都不行就放弃。
        
        [跨文件] 被 reason_from_future() 在第1b步和第3步调用。
        """
        clean_raw_text = raw_text.strip()

        # -------- 策略 1：解析 JSON 格式 --------
        try:
            # [Python知识] re.search(r"\{[\s\S]*?\}", text) 匹配文本中第一个 JSON 对象
            # [\s\S] 匹配任意字符（包括换行），*? 是非贪婪匹配
            match = re.search(r"\{[\s\S]*?\}", clean_raw_text)
            if match:
                json_text = match.group(0)
                json_text = json_text.strip()
                # 处理 Markdown 代码块：去掉 ```json 和 ```
                if json_text.startswith("```json"):
                    json_text = json_text[7:-3].strip()
                elif json_text.startswith("```"):
                    json_text = json_text[3:-3].strip()
                clean_raw_text = json_text

            # [Python知识] json.loads() 把 JSON 字符串解析成 Python 字典
            data = json.loads(clean_raw_text)
            var_name = data.get("var")     # 变量名
            var_value = data.get("value")  # 变量值
            expr = data.get("expr")        # 计算表达式（可选）

            if var_name and var_value is not None:
                # 如果有表达式，就用安全 eval 验证
                if expr and isinstance(expr, str):
                    try:
                        # [Python知识] eval() 能执行任意 Python 代码，很危险
                        # 这里通过传入受限的局部变量空间来降低风险：
                        # - safe_locals 只包含 state 中的数值变量
                        # - 没有 __builtins__（不能调用 print、open 等）
                        safe_locals = {
                            k: v for k, v in state.items()
                            if isinstance(v, (int, float))
                        }
                        calculated = float(eval(expr, {}, safe_locals))
                        # 如果表达式结果和 value 对不上，拒绝这次更新
                        if abs(calculated - float(var_value)) > 1e-4:
                            return Workspace()
                    except NameError:
                        # [Python知识] NameError 表示表达式引用了不存在的变量
                        # 这没问题——先接受数值，后面再验证
                        pass
                    except Exception:
                        return Workspace()
                elif isinstance(var_value, str):
                    try:
                        var_value = float(var_value.replace(",", ""))
                    except ValueError:
                        pass
                elif isinstance(var_value, (int, float)):
                    var_value = float(var_value)
                # ✅ 成功解析！返回包含新变量的 Workspace
                return Workspace({var_name: var_value})

        except json.JSONDecodeError:
            # [Python知识] json.loads() 失败 → 不是合法 JSON，尝试其他策略
            pass
        except Exception:
            pass

        # -------- 策略 2：寻找 LaTeX 格式的答案 --------
        # "The final answer is $\\boxed{42}$"
        final_target_var_name = self.derive_final_target(self.question)
        boxed_match = re.search(
            r"The final answer is \$\\boxed\{([\d\.,]+)\}\$",
            clean_raw_text,
            re.IGNORECASE,
        )
        if boxed_match:
            try:
                value_str = boxed_match.group(1).replace(",", "")
                numeric_value = float(value_str)
                return Workspace({final_target_var_name: numeric_value})
            except ValueError:
                pass

        # -------- 策略 3：寻找简单的"答案是 XX"格式 --------
        simple_match = re.search(
            r"(?:final answer is|result is|is simply|is:)\s*([\d\.,]+)",
            clean_raw_text,
            re.IGNORECASE,
        )
        if simple_match:
            try:
                value_str = simple_match.group(1).replace(",", "")
                numeric_value = float(value_str)
                return Workspace({final_target_var_name: numeric_value})
            except ValueError:
                pass

        # -------- 所有策略都失败了 --------
        return Workspace()  # 返回空 Workspace → 主循环会标记失败

    # ---- 3. merge_aliases ----
    def merge_aliases(self, state: Workspace) -> Workspace:
        """合并同义词变量名。
        
        [费曼解释] LLM 可能在步骤间用不同的名字指同一个东西：
        - "initial_science_books" vs "science_books_before_bonus"
        
        忽略掉 "initial", "before", "total" 等修饰词后，
        如果两个名字的"核心部分"一样，就合并它们。
        
        [Python知识] 这里使用正则表达式 re.sub() 去除修饰词：
        re.sub(r"(?:number|num|total|...)", "", var) → 清除这些词
        然后用 setdefault 保持第一个出现的变量名。
        """
        if len(state) <= 1:
            return state

        normalized_map: dict[str, str] = {}
        for var in state:
            # 去掉常见的修饰词
            norm = re.sub(r"(?:number|num|total|initial|before|after|of|the)", "", var.lower())
            # 规范化下划线
            norm = re.sub(r"[_\s]+", "_", norm).strip("_")
            # [Python知识] setdefault(key, default)：如果 key 不在字典中则设置
            normalized_map.setdefault(norm, var)

        new_state = Workspace()
        for norm_key, representative_var in normalized_map.items():
            new_state[representative_var] = state[representative_var]
        return new_state

    # ---- 4. check_local ----
    def check_local(self, state: Workspace, target_step: str) -> bool:
        """局部校验：检查变量是否存在且类型正确。
        
        [费曼解释] 只做基础检查：
        - 变量名在不在草稿纸上？
        - 变量的值是不是数字类型？
        
        不做"对不对"的检查——那是 verify_final 的工作。
        
        [跨文件] 被 reason_from_future() 在每次正向计算后调用。
        """
        if target_step not in state:
            return False
        return isinstance(state[target_step], (int, float))

    # ---- 5. prompt_last_step（反向推理提示词）----
    def prompt_last_step(self, state: Workspace, target: str, avoid: Set[str]) -> str:
        """构建反向推理提示词："要算出 target，先得算什么？"
        
        [费曼解释] 这是 RFF 最核心的提示词构建！
        告诉 LLM：
        1. 这是题目（self.question）
        2. 目前已知这些变量和值（state）
        3. 最终目标是什么（target）
        4. 不要提这些变量（avoid）
        5. 请列出最多 3 个必须先算出的变量（next_variables）
        
        [跨文件] 被 reason_from_future() 在第2步调用。
        """
        defined_vars = sorted(list(state.keys()))
        avoid_list_str = (
            f"Do not choose any of these variables: {sorted(list(avoid))}."
            if avoid else ""
        )
        prompt = textwrap.dedent(  # [Python知识] 去掉多行字符串的公共缩进
            f"""
            You are reasoning backward through a math word problem to find the solution.
            The problem is:
            {self.question}

            Defined variables with values so far:
            {json.dumps(state, indent=2)}

            Your final goal is the variable "{target}".
            Provide up to **three** prerequisite variable names (strings) that must
            be computed *immediately before* "{target}" can be determined. These names
            must NOT already appear in the defined list above.

            {avoid_list_str}

            Output a single JSON object with one key exactly named "next_variables"
            whose value is an array of 1-3 strings. Example:
            {{"next_variables": ["bonus_science_books", "replacement_science_books"]}}

            IMPORTANT: Respond with ONLY the JSON.
            """
        ).strip()
        return prompt

    # ---- 6. prompt_forward_step（正向计算提示词）----
    def prompt_forward_step(self, state: Workspace, target_step: str, avoid: Set[str]) -> str:
        """构建正向计算提示词："请计算 target_step 的值"。
        
        [费曼解释] 告诉 LLM：
        1. 题目是什么
        2. 目前已知变量和值（让它用这些值来计算）
        3. 你需要算哪个变量（target_step）
        4. 怎么输出结果（JSON 格式，含 var、expr、value）
        
        [跨文件] 被 reason_from_future() 在第1b步和第3步调用。
        """
        final_goal_name = self.derive_final_target(self.question)
        defined_vars = sorted(list(state.keys()))

        prompt = f"""You are solving a math word problem step-by-step.
The problem is:
{self.question}

Defined variables so far (with values):
{json.dumps(state, indent=2)}

Undefined variable you must compute now: "{target_step}".
Avoid inventing synonyms for existing variable names. Stick to clear snake_case names.
"""
        if target_step == final_goal_name:
            prompt += (
                "\nWhen outputting 'final_answer', set 'expr' to the same numeric "
                "value as 'value' (do NOT reference other variables)."
            )

        prompt += textwrap.dedent(
            f"""
 
Output your answer as a single JSON object with three keys: 'var', 'expr', and 'value'.
  • 'var'  – either the target variable or "final_answer" if it's final.
  • 'expr' – a Python-style arithmetic expression that evaluates to the value
             **using only previously known variables** (listed above).
             If trivial or final, just repeat the numeric value.
  • 'value' – the numerical result (number type, **not** string).

Examples:
  Intermediate step: {{"var": "miles_per_gallon", "expr": "miles_driven / gallons", "value": 67.89}}
  Final answer:      {{"var": "final_answer", "expr": "42", "value": 42}}

IMPORTANT: Your *entire response* must be *only* the JSON object.
No additional commentary.
"""
        )
        return prompt

    # ---- 7. parse_target_step ----
    def parse_target_step(self, raw_text: str) -> str:
        """从 LLM 的反向推理输出中提取"下一步要算什么"。
        
        [费曼解释] LLM 可能返回：
        - {"next_variables": ["var1", "var2", "var3"]}  ← 理想格式
        - {"next_variable": "var1"}                      ← 单变量格式
        - $\boxed{\text{variable_name}}$                 ← LaTeX 格式
        - "variable_name"                                ← 纯文本
        
        这个方法用多层 fallback 提取第一个有效的变量名。
        
        [Python知识] 正则表达式的 r 前缀（raw string）：
        r"\n" 是两个字面字符 \ 和 n，而 "\n" 是换行符。
        在正则表达式里用 raw string 避免转义问题。
        
        [跨文件] 被 reason_from_future() 在第2步调用。
        """
        clean_raw_text = raw_text.strip()

        # 策略 1：解析 {"next_variables": [...]} 格式（最新，支持多变量）
        try:
            match = re.search(r"\{[\s\S]*?next_variables[\s\S]*?\}", clean_raw_text)
            if match:
                json_text = match.group(0).strip()
                if json_text.startswith("```json"):
                    json_text = json_text[7:-3].strip()
                data = json.loads(json_text)
                if "next_variables" in data and isinstance(data["next_variables"], list):
                    for v in data["next_variables"]:
                        if isinstance(v, str) and v:
                            return v.strip()  # 返回第一个有效的变量名
        except Exception:
            pass

        # 策略 2：解析 {"next_variable": "..."} 格式（旧版，单变量）
        try:
            match = re.search(
                r"\{[\s\S]*?\"next_variable\"\s*:\s*\"(.*?)\"[\s\S]*?\}",
                clean_raw_text,
            )
            if match:
                json_like_text = match.group(0)
                if json_like_text.startswith("```json"):
                    json_like_text = json_like_text[7:-3].strip()
                data = json.loads(json_like_text)
                return str(data.get("next_variable", "")).strip()
        except Exception:
            pass

        # 策略 3：LaTeX boxed 格式 $\boxed{\text{variable_name}}$
        boxed_text_match = re.search(
            r"\$\\boxed\{\text\{([_a-zA-Z0-9\s]+)\}\}\$", clean_raw_text
        )
        if boxed_text_match:
            return boxed_text_match.group(1).strip().replace(" ", "_")

        # 策略 4：最后一行看起来像变量名
        lines = [line.strip() for line in clean_raw_text.split("\n") if line.strip()]
        if lines:
            last_line = lines[-1]
            # 清理常见的前缀
            last_line = re.sub(
                r"^(?:The next variable to compute is|Here is the JSON output:)\s*",
                "",
                last_line,
                flags=re.IGNORECASE,
            ).strip()
            last_line = last_line.replace("`", "").replace("'", "").replace('"', "")
            # 检查是否像合法的变量名（字母开头，只含字母数字下划线）
            if re.fullmatch(r"[_a-zA-Z][_a-zA-Z0-9]*", last_line):
                return last_line

        # 最后兜底：返回最后一行
        return lines[-1] if lines else clean_raw_text

    # ---- 8. verify_final ----
    def verify_final(self, state: Workspace) -> Tuple[bool, str, float]:
        """终极验证：比对 LLM 的结果和标准答案。
        
        [费曼解释] 这是最后也是最严格的一关！
        1. 从 state 中取出 final_answer
        2. 转为数字
        3. 和标准答案（self.gold_numeric_answer）比对
        4. 允许 1e-5 的浮点误差
        
        [跨文件] 被 reason_from_future() 在第1步和第1b步调用。
        
        [返回值]
        (是否正确, LLM给出的答案字符串, 标准答案数值)
        """
        guess_val = state.get(self.derive_final_target(self.question))

        if guess_val is None:
            return False, "No final answer provided.", self.gold_numeric_answer

        try:
            # [Python知识] 处理各种可能的类型：str, int, float, dict, list...
            if isinstance(guess_val, str):
                guess_val_cleaned = guess_val.replace(",", "")
                # 正则验证是否是合法数字格式
                if not re.fullmatch(r"-?\d+(\.\d+)?", guess_val_cleaned):
                    return (
                        False,
                        f"Invalid numeric format for final answer: {guess_val}",
                        self.gold_numeric_answer,
                    )
                numeric_guess = float(guess_val_cleaned)
            elif isinstance(guess_val, (int, float)):
                numeric_guess = float(guess_val)
            else:
                return (
                    False,
                    f"Final answer '{guess_val}' has unexpected type {type(guess_val).__name__}.",
                    self.gold_numeric_answer,
                )
        except ValueError:
            return (
                False,
                f"Final answer '{guess_val}' is not a valid number.",
                self.gold_numeric_answer,
            )

        # 和标准答案比对（允许微小的浮点误差）
        is_correct = abs(numeric_guess - self.gold_numeric_answer) < 1e-5
        return is_correct, str(numeric_guess), self.gold_numeric_answer