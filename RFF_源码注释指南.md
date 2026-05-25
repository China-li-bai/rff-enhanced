# RFF (Reason-from-Future) 源码逐行注释指南

> 面向 Python 初学者。先说"这是什么"（生活比喻），再说"为什么这么写"（技术原理），最后标注跨文件逻辑关联。

---

## 目录

1. [架构总览](#1-架构总览)
2. [Python 知识预备](#2-python-知识预备)
3. [core.py — 核心抽象与主循环](#3-corepy--核心抽象与主循环)
4. [llm.py — LLM 调用封装](#4-llmpy--llm-调用封装)
5. [__init__.py — 包入口与导出](#5-__initpy--包入口与导出)
6. [specs/ — 四个领域规约](#6-specs--四个领域规约)
7. [tools/ — 文件读写工具](#7-tools--文件读写工具)
8. [demos/ — 运行入口](#8-demos--运行入口)
9. [跨文件逻辑追踪：一次完整的 G→R→C 循环](#9-跨文件逻辑追踪)
10. [四个 Spec 的对比进化线](#10-四个-spec-的对比进化线)

---

## 1. 架构总览

### 1.1 文件关系图

```
rff-enhanced/src/
│
├── demos/                          ← 🏁 运行入口（你从这里启动）
│   ├── demo_game24.py              ← 24点游戏演示
│   ├── demo_gsm8k.py               ← 数学题演示
│   └── demo_general.py             ← 通用问题演示
│
├── reason_from_future/             ← 📦 核心包
│   ├── __init__.py                 ← 包入口，统一导出
│   ├── core.py                     ← 🧠 大脑：Workspace + ProblemSpec + 主循环
│   ├── llm.py                      ← 📡 嘴巴：与 Gemini API 通话
│   │
│   ├── specs/                      ← 🎭 角色剧本：不同问题的"规则书"
│   │   ├── __init__.py             ← Spec 子包入口
│   │   ├── game24.py               ← 24点游戏规则
│   │   ├── gsm8k.py                ← 小学数学题规则
│   │   ├── general_problem_solving.py ← 通用问题求解规则
│   │   └── code_writing.py         ← 代码生成规则
│   │
│   └── tools/                      ← 🔧 工具箱：给 LLM 用的文件操作
│       ├── read_file.py            ← 读文件
│       └── write_to_file.py        ← 写文件
```

### 1.2 核心概念映射（用生活比喻理解）

| 概念 | 代码中的名字 | 生活比喻 |
|------|-------------|---------|
| 工作台/草稿纸 | `Workspace` | 做数学题时的草稿纸，上面写满了中间结果 |
| 问题规约 | `ProblemSpec` | 面试官出的"考卷"，定义了考什么、怎么判对错 |
| 最终目标 | `goal` | 考卷的最后一道大题的答案 |
| 反向推理 | `prompt_last_step` | 从答案倒推：要得出24，我先得算出什么？ |
| 正向计算 | `prompt_forward_step` | 从已知出发：我有8和3，那8×3=24！ |
| 局部校验 | `check_local` | 做完一步后自检：格式对不对？数值合不合理？ |
| 终极验证 | `verify_final` | 对答案：跟标准答案对得上吗？ |
| 避免重复 | `avoid` 集合 | "这道题我试过了，别再试了" |
| 停滞重启 | `stagnation_counter` | 连续4步没进展 → 把一些限制放开，重新来 |

### 1.3 数据流向总览

```
demo_xxx.py                创建 Spec 实例
     │                         │
     ▼                         ▼
  reason_from_future(problem, spec)  ←── core.py 中的主循环
     │
     ├── spec.derive_final_target()  → 确定 goal = "final_answer" 或 "24"
     │
     │  ┌─────── 迭代循环（最多 max_iters 次）────────┐
     │  │                                              │
     │  │  1. check_local(state, goal)  → 已有答案？   │
     │  │     ├── 是 → verify_final()  → 对了？返回！ │
     │  │     └── 否 → 继续                           │
     │  │                                              │
     │  │  2. prompt_forward_step(goal)  → 直接试试   │
     │  │     → llm_call()  → parse_workspace_update()│
     │  │     → check_local() → verify_final()        │
     │  │                                              │
     │  │  3. prompt_last_step(goal)  → 反向问：      │
     │  │     "要算 goal，先得算什么？"                 │
     │  │     → llm_call()  → parse_target_step()     │
     │  │     → 得到 target_step                       │
     │  │                                              │
     │  │  4. prompt_forward_step(target_step) → 正向算│
     │  │     → llm_call()  → parse_workspace_update()│
     │  │     → check_local()  → 通过？更新 state！    │
     │  │                                              │
     │  │  5. merge_aliases(state)  → 合并同义词       │
     │  │  6. 停滞检测 → 必要时软重启                  │
     │  └──────────────────────────────────────────────┘
     │
     ▼
  返回最终答案 或 抛出 RuntimeError
```

---

## 2. Python 知识预备

以下按代码中**首次出现**的顺序列出所有需要了解的 Python 特性，每个都用「一句话 + 比喻 + 代码中哪用到」的方式解释。

### 2.1 `from __future__ import annotations`
> **一句话**：让 Python 3.7+ 支持更先进的类型注解语法（比如 `dict[str, Any]` 而非 `Dict[str, Any]`）。
>
> **比喻**：就像提前告诉翻译官"我接下来会用一些新词汇"，这样即使字典还没收录，翻译官也能正确处理。
>
> **用到**：`core.py` 第2行、`game24.py` 第6行、`__init__.py` 第8行

### 2.2 ABC（抽象基类）+ `@abstractmethod`
> **一句话**：ABC 是"模板类"，规定了子类**必须**实现哪些方法，否则实例化时会报错。
>
> **比喻**：就像面试时发的"岗位说明书"——规定了你必须会什么技能，不会就不能上岗。
>
> **代码**：
> ```python
> from abc import ABC, abstractmethod
>
> class Animal(ABC):           # 继承 ABC → 这是一个抽象基类
>     @abstractmethod          # 装饰器：标记这个方法子类必须实现
>     def speak(self) -> str:
>         ...
>
> class Dog(Animal):
>     def speak(self) -> str:  # ✅ 子类实现了 speak
>         return "汪汪"
>
> # Dog() 正常工作
> # Animal() → ❌ TypeError: Can't instantiate abstract class
> ```
>
> **用到**：`core.py` 的 `ProblemSpec` 类，它有 8 个抽象方法

### 2.3 继承 dict（`class Workspace(dict)`）
> **一句话**：让 Workspace **就是**一个字典，同时还能加自己的方法。
>
> **比喻**：你买了一辆车，它本身就能开（dict 的所有功能），你又在上面装了导航和音响（自定义方法）。
>
> **代码**：
> ```python
> class Workspace(dict):     # 继承 dict → ws = Workspace() 就像 ws = {}
>     def add(self, key, val):  # 额外方法
>         self[key] = val
>
> ws = Workspace()
> ws["x"] = 10           # ✅ 普通 dict 用法
> ws.add("y", 20)        # ✅ 自定义方法
> ```
>
> **用到**：`core.py` 第9行

### 2.4 运算符重载（`__or__` / `__ror__`）
> **一句话**：让自定义对象支持 `|` 运算符。`__or__` 是 `self | other`，`__ror__` 是 `other | self`。
>
> **比喻**：Python 不知道你的 Workspace 怎么"合并"，你得亲自教它。就像教会一个机器人"两个盘子怎么合到一个盘子里"。
>
> **代码**：
> ```python
> class Workspace(dict):
>     def __or__(self, other):       # 定义 self | other 的行为
>         combined = Workspace()
>         combined.update(self)       # 先把自己的内容放进去
>         combined.update(other)      # 再把 other 的内容合进去
>         return combined
>
> a = Workspace({"x": 1})
> b = Workspace({"y": 2})
> c = a | b                # c = {"x": 1, "y": 2}
> ```
>
> **为什么需要？** Python 3.9 的 `dict | dict` 返回普通 `dict`，会**丢失** Workspace 的自定义方法。重写 `__or__` 确保合并结果还是 `Workspace` 类型。
>
> **用到**：`core.py` 第36-59行

### 2.5 `nonlocal` 关键字
> **一句话**：在嵌套函数中，让内层函数能**修改**外层函数的变量。
>
> **比喻**：你在家里的厨房（外层函数），你小孩在客厅（内层函数）——`nonlocal` 就是给小孩一把钥匙，能打开厨房的冰箱拿东西。
>
> **代码**：
> ```python
> def outer():
>     count = 0           # 外层变量
>     def inner():
>         nonlocal count   # 声明：我要修改外层的 count
>         count += 1       # ✅ 没有 nonlocal 会报 UnboundLocalError
>     inner()
>     print(count)         # 1
> ```
>
> **用到**：`core.py` 第160行 `register_fail` 函数修改 `attempt_counts` 和 `avoid`

### 2.6 f-string（格式化字符串）
> **一句话**：在字符串中直接嵌入变量，`f"名字是{name}"`。
>
> **比喻**：填表时用"姓名：____"的横线，f-string 就是自动帮你填上。
>
> **用到**：几乎每个 prompt 构建处都用到了

### 2.7 类型注解（Type Hints）
> **一句话**：给变量和函数标注类型，方便阅读和工具检查，**不影响运行**。
>
> **比喻**：像菜谱上写"盐 5克"——告诉你放多少，但炒菜时你放6克也不会报错。
>
> **用到**：几乎每个函数签名

### 2.8 `typing` 模块常用类型

| 类型 | 含义 | 代码示例 |
|------|------|---------|
| `Any` | 任意类型 | `val: Any` |
| `Tuple[bool, str, float]` | 三个元素的元组 | `return True, "24", 24.0` |
| `Set[str]` | 字符串集合 | `avoid: Set[str]` |
| `Dict[str, str]` | 字符串到字符串的字典 | `problem_data: Dict[str, str]` |
| `List[int]` | 整数列表 | `nums: List[int]` |
| `Union[str, Path]` | 要么 str 要么 Path | `file_path: Union[str, Path]` |

### 2.9 `*` 在函数参数中的含义
> **一句话**：`*` 后面的参数**必须用关键字**传递，不能按位置传。
>
> **比喻**：点菜时说"我要一份宫保鸡丁"（位置参数），vs 说"菜品=宫保鸡丁，辣度=微辣"（关键字参数）。`*` 就是强制你用后者。
>
> **代码**：
> ```python
> def reason_from_future(problem, spec, *, max_iters=16):
>     ...
>
> # ✅ 正确
> reason_from_future("题目", spec, max_iters=10)
> # ❌ 错误！* 后面的参数必须用关键字
> reason_from_future("题目", spec, 10)
> ```
>
> **用到**：`core.py` 第106行、`llm.py` 第11行

### 2.10 `collections.Counter`
> **一句话**：一个自动计数的字典，统计每个元素出现了几次。
>
> **比喻**：投票时用的计分板——"3号出现了2次，5号出现了1次"。
>
> **代码**：
> ```python
> from collections import Counter
> c = Counter([3, 5, 3, 3, 5])  # {3: 3, 5: 2}
> c[3]        # 3
> c1 - c2     # 支持减法！差集计数
> ```
>
> **用到**：`game24.py` 中的 `_orig_counter`、`_used_counter`、`_available_counter`

### 2.11 `ast` 模块（抽象语法树）
> **一句话**：把代码字符串解析成树状结构，用于**安全地**求值（避免 `eval()` 的安全风险）。
>
> **比喻**：`eval()` 就像让陌生人直接进你家翻箱倒柜；`ast` + `NodeVisitor` 就像让快递员只看你指定的那个箱子。
>
> **代码**：
> ```python
> # ❌ 危险！可能执行恶意代码
> eval("__import__('os').system('rm -rf /')")
> # ✅ 安全！只允许加减乘除
> tree = ast.parse("(3+5)*2", mode="eval")
> # 然后用自定义的 NodeVisitor 逐节点求值
> ```
>
> **用到**：`game24.py` 的 `_SafeEval` 类

### 2.12 `operator` 模块
> **一句话**：把运算符变成函数，方便在字典里查表调用。
>
> **比喻**：`+` 是个符号，不能放进字典当 value；`operator.add` 是个函数，可以。
>
> **代码**：
> ```python
> import operator
> ops = {ast.Add: operator.add, ast.Sub: operator.sub}
> result = ops[ast.Add](3, 5)  # 等价于 3 + 5 = 8
> ```
>
> **用到**：`game24.py` 第25-30行的 `_ALLOWED_BIN_OPS`

### 2.13 `textwrap.dedent`
> **一句话**：去掉多行字符串的公共前导空格，让代码中的长文本保持缩进美观。
>
> **用到**：所有 prompt 构建处

### 2.14 `re` 正则表达式
> **一句话**：用模式匹配来搜索/提取/替换文本。
>
> **常用模式**：
> - `r"\{[\s\S]*?\}"` — 匹配 JSON 对象
> - `r"\b\d+\b"` — 匹配独立的数字
> - `r"(?:####\s*)?([0-9,.]+)\s*$"` — 匹配行末的数字
>
> **用到**：`gsm8k.py`、`game24.py`、`general_problem_solving.py`、`code_writing.py`

### 2.15 `uuid.uuid4()`
> **一句话**：生成一个几乎不会重复的随机唯一标识符。
>
> **用到**：`game24.py` 第167行，给每条表达式生成唯一 key

### 2.16 `from pathlib import Path`
> **一句话**：比 `os.path` 更现代的文件路径操作库。
>
> **用到**：`tools/read_file.py` 和 `tools/write_to_file.py`

### 2.17 `__all__` 变量
> **一句话**：定义"当别人用 `from package import *` 时，会导入哪些名字"。
>
> **比喻**：你家有很多房间，`__all__` 就是贴在门口的"开放参观"房间清单。
>
> **用到**：`__init__.py`、`specs/__init__.py`

---

## 3. core.py — 核心抽象与主循环

### 3.0 文件头部

```python
"""Core abstractions for Reason-from-Future."""
from __future__ import annotations          # [知识2.1] 启用延迟注解

from abc import ABC, abstractmethod          # [知识2.2] 抽象基类
from typing import Any, Tuple, Set          # [知识2.8] 类型注解
from .llm import llm_call                   # 相对导入，从同级的 llm.py 导入
```

### 3.1 Workspace 类（第9-59行）

```python
class Workspace(dict):
```
> **是什么**：继承自 `dict` 的工作台，存放推理过程中所有中间变量和结果。
>
> **比喻**：做数学题的草稿纸。每算出一个中间结果就写上去，最后从上面找到"最终答案"。
>
> **为什么继承 dict**：推理中存取键值对是最核心的操作，继承 dict 让这操作零成本，同时可以加自定义方法。
>
> **跨文件关联**：几乎所有文件都用 Workspace——spec 往里写值，core 往里读值。

#### `add()` 方法

```python
def add(self, key: str, val: Any) -> None:
    self[key] = val
```
> 语义更清晰的"添加"方法，底层就是 `self[key] = val`。

#### `get_all_data()` 方法

```python
def get_all_data(self) -> dict[str, Any]:
    return self   # Workspace 本身就是 dict，直接返回自身
```
> CodeWritingSpec 用 `state.get_all_data()` 判断工作台是否为空。面向接口写法。

#### `get_internal_state_DEBUG()` 方法

```python
def get_internal_state_DEBUG(self) -> dict[str, Any]:
    return self   # 同上，向后兼容的调试方法
```

#### `__or__` 运算符重载

```python
def __or__(self, other: dict[str, Any]):
    combined = Workspace()    # 创建新的空 Workspace
    combined.update(self)     # 先放自己的内容
    if isinstance(other, dict):
        combined.update(other)  # 再放 other（同名 key 以 other 为准）
    else:
        raise TypeError("Can only merge Workspace with dict-like object using '|'")
    return combined
```
> [知识2.4] 重载 `|` 运算符。`workspace_a | workspace_b` 返回合并后的新 Workspace。
>
> **为什么需要**：Python 3.9 的 `dict | dict` 返回普通 `dict`，Workspace 的自定义方法会丢失。
>
> **跨文件关联**：core.py 第184、224、228、254行都用 `state | parsed_update` 合并新信息。

#### `__ror__` 右侧运算符重载

```python
def __ror__(self, other: dict[str, Any]):
    combined = Workspace()
    if isinstance(other, dict):
        combined.update(other)   # 先放 other
    else:
        raise TypeError(...)
    combined.update(self)         # 再放自己（自己覆盖 other 的同名 key）
    return combined
```
> 当 Workspace 在 `|` 右侧时（`dict | Workspace`），保证结果还是 Workspace。

---

### 3.2 LocalCheckFail 异常类（第62-63行）

```python
class LocalCheckFail(Exception):
    """Raised when a forward hop fails local validation."""
```
> 自定义异常，当正向步骤的局部校验失败时抛出。
> **注意**：当前代码定义了但主循环没用到（用 `register_fail()` 替代），是预留异常类。

---

### 3.3 ProblemSpec 抽象基类（第66-100行）

```python
class ProblemSpec(ABC):
    """Contract that each reasoning domain must fulfill."""
```
> [知识2.2] 抽象基类，定义了所有"问题规约"必须实现的8个方法。是"合同"——规定了每个领域必须提供的服务。
>
> **跨文件关联**：Game24Spec、GSM8KSpec、GeneralProblemSolvingSpec、CodeWritingSpec 都继承 ProblemSpec。

#### 8个抽象方法一览

| 方法 | 对应 RFF 阶段 | 作用 |
|------|--------------|------|
| `derive_final_target(problem)` | 初始化 | 确定最终目标的变量名（如 `"final_answer"` 或 `"24"`） |
| `parse_workspace_update(raw_text, state)` | 正向计算后 | 把 LLM 的原始输出解析成 Workspace 中的键值对 |
| `check_local(state, target_step)` | 局部校验 | 检查某个变量是否"看起来合理"（格式、类型等） |
| `verify_final(state)` | 终极验证 | 检查最终答案是否正确 |
| `prompt_last_step(state, target, avoid)` | 反向推理 | 构建提示词："要算出 X，先得算什么？" |
| `prompt_forward_step(state, target_step, avoid)` | 正向计算 | 构建提示词："请计算 X 的值" |
| `parse_target_step(raw_text)` | 反向推理后 | 从 LLM 的反向推理输出中提取"下一步要算什么" |
| `merge_aliases(state)` | 后处理 | 合并同义词（如 `initial_books` 和 `books_before_bonus`） |

---

### 3.4 `reason_from_future()` 主循环（第103-288行）

这是整个项目的**核心算法**，我分段逐行解释。

#### 函数签名

```python
def reason_from_future(
    problem: str,          # 问题描述（如数学题文本）
    spec: ProblemSpec,     # 问题规约（如 GSM8KSpec 实例）
    *,                     # [知识2.9] 后面参数必须用关键字传递
    max_iters: int = 16,   # 最大迭代次数
    min_iters: int = 1,    # 最少迭代次数（防止过早接受答案）
    require_gold: bool = True,  # 是否需要跟标准答案对得上
    model: str = "gemini-2.5-flash-preview-05-20",
    verbose: bool = False,
) -> str:
```

> **核心思想**：RFF = Reason From Future。像你要去一个目的地，先从目的地往回看"要到达那里得先经过哪"，然后再从当前出发往前走。

#### 初始化

```python
state: Workspace = Workspace()                    # 草稿纸，空白的
goal: str = spec.derive_final_target(problem)     # 最终目标，如 "final_answer"
avoid: Set[str] = set()                           # 已尝试/失败过的变量名集合
```
> `goal` 在整个循环中**不变**——关键设计决策。旧版让 goal 变成先决变量，导致搜索"倒退"永远不回来。

```python
attempt_counts: dict[str, int] = {}   # 每个变量失败的次数
max_fails_per_var: int = 3            # 同一个变量失败3次就加入 avoid
stagnation_counter: int = 0           # 连续没有进展的迭代次数
stagnation_window: int = 4            # 连续4次没进展就触发软重启
```
> **停滞检测比喻**：连续4次搬家都没找到工作——换个策略吧。

#### `register_fail` 嵌套函数

```python
def register_fail(symbol: str) -> None:
    """Increment failure counter and add to avoid set when threshold hit."""
    nonlocal attempt_counts, avoid      # [知识2.5] 修改外层变量
    attempt_counts[symbol] = attempt_counts.get(symbol, 0) + 1  # 计数+1
    if attempt_counts[symbol] >= max_fails_per_var:              # 达到3次？
        avoid.add(symbol)               # 加入黑名单
```
> 记录某个变量计算失败。失败3次就拉黑。

#### 主循环 — 第1步：检查目标是否已达成

```python
for iter_idx in range(max_iters):
    made_progress: bool = False

    # 1) 如果已有目标值，尝试验证并结束
    if spec.check_local(state, goal):          # goal 变量存在于 state 且格式正确？
        if not require_gold and iter_idx >= (min_iters - 1):
            return str(state[goal])             # 不需要金标 && 迭代次数够 → 直接返回
        ok, answer_from_llm, gold_val_for_debug = spec.verify_final(state)
        if ok:
            return answer_from_llm              # 验证通过！
        register_fail(goal)                     # 验证失败，记录
```

#### 主循环 — 第1b步：直接尝试计算目标

```python
    # 1b) 尝试用当前知识直接计算目标（快速通道）
    if goal not in avoid:
        direct_prompt = spec.prompt_forward_step(state, goal, avoid)
        direct_raw = llm_call(direct_prompt, model=model, verbose=verbose)
        direct_state = state | spec.parse_workspace_update(direct_raw, state)
        if spec.check_local(direct_state, goal):
            # ... 验证逻辑 ...
        state = direct_state      # 即使目标算错了，也保留其他有用的新变量
        register_fail(goal)
```
> **为什么有这步**：有时 LLM 直接就能算出最终答案，不需要走反向推理。是"快速通道"。
>
> **关键设计**：`state = direct_state`——即使目标没算对，LLM 可能算出了其他有用的中间变量。

#### 主循环 — 第2步：反向推理

```python
    # 2) 问 LLM："要算出 goal，先得算什么？"
    g_prompt = spec.prompt_last_step(state, goal, avoid)
    raw_target_step_response = llm_call(g_prompt, model=model, verbose=verbose)
    target_step = spec.parse_target_step(raw_target_step_response)

    # 2a) 防御：LLM 返回空值或已尝试过的变量
    if not target_step or target_step in avoid:
        continue
```
> 这是 RFF 的"从未来推理"——从目标往回推，找到先决条件。

#### 主循环 — 第3步：正向计算先决变量

```python
    # 3) 让 LLM 计算那个先决变量
    r_prompt = spec.prompt_forward_step(state, target_step, avoid)
    forward_raw = llm_call(r_prompt, model=model, verbose=verbose)
    parsed_update = spec.parse_workspace_update(forward_raw, state)
```

#### 主循环 — 第3步后续：判断 LLM 到底算出了什么

```python
    llm_provided_var = None
    if parsed_update:
        llm_provided_var_keys = list(parsed_update.keys())
        if llm_provided_var_keys:
            llm_provided_var = llm_provided_var_keys[0]
```
> LLM 可能"不听话"——你让它算 x，它可能算了 y 或 final_answer。必须分情况处理。

**情况A：LLM 算了目标变量**

```python
    if llm_provided_var == goal:
        temp_state_for_verification = state | parsed_update
        if spec.check_local(temp_state_for_verification, goal):
            # ... 验证逻辑，对了返回，错了 register_fail ...
        # ⚠️ 关键：不更新 state！错误的答案不能写入草稿纸
        continue
```
> 如果 LLM 给了错误的目标值，**不写入 state**。否则下一轮会发现"目标已存在"然后反复验证同一个错误值。

**情况B：LLM 算了先决中间变量**

```python
    elif llm_provided_var == target_step:
        temp_state_for_target_step = state | parsed_update
        if spec.check_local(temp_state_for_target_step, target_step):
            state = temp_state_for_target_step  # ✅ 提交新状态
            register_fail(target_step)
            made_progress = True                 # 标记本轮有进展
        else:
            register_fail(target_step)
            continue                             # 不更新 state
```
> 最顺利的情况：LLM 算出了我们想要的中间变量，且通过了局部校验。

**情况C：LLM 给了无关内容**

```python
    else:
        register_fail(target_step)
        continue   # 直接放弃，不更新 state
```

#### 后处理

```python
    state = spec.merge_aliases(state)    # 合并同义词

    if made_progress:
        stagnation_counter = 0           # 有进展，重置
    else:
        stagnation_counter += 1          # 没进展，+1

    # 软重启
    if stagnation_counter >= stagnation_window:
        avoid = {s for s, cnt in attempt_counts.items() if cnt >= max_fails_per_var}
        stagnation_counter = 0
```
> **软重启逻辑**：停滞时把 avoid 缩小到"真正失败3次"的变量，其他放开重新尝试。

#### 循环结束

```python
    raise RuntimeError("RFF exhausted iterations without solution.")
```

---

## 4. llm.py — LLM 调用封装

```python
"""LLM glue using Google GenAI SDK"""
import os
from google import genai
from google.genai import types

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # 从环境变量读取 API Key
client = genai.Client(api_key=GEMINI_API_KEY)  # 创建全局客户端（模块级初始化）
```
> ⚠️ 没设 `GEMINI_API_KEY` 环境变量，`client` 会用 None 创建，后续调用会报错。

```python
def llm_call(
    prompt: str,
    *,                                # [知识2.9] 后面必须用关键字
    model: str = "gemini-2.5-flash-preview-05-20",
    verbose: bool = False,
    tools: list | None = None,
) -> str:
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")

    if verbose:
        print(f"--- LLM PROMPT ({model}) ---")
        print(prompt)

    # 构建 SDK 要求的 content 对象
    contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
    # 比喻：prompt 是货物，types.Content 是快递盒，types.Part 是填充物

    cfg = None
    if tools:
        cfg = types.GenerateContentConfig(tools=tools)  # 工具配置

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=cfg,
    )
    result_text = response.text  # LLM 返回的纯文本

    if verbose:
        print(f"--- LLM RESPONSE ({model}) ---")
        print(result_text)
    return result_text
```
> **跨文件关联**：core.py 中的 `reason_from_future()` 在每次需要 LLM 时调用 `llm_call()`。

---

## 5. __init__.py — 包入口与导出

### 5.1 主包入口 `reason_from_future/__init__.py`

```python
from __future__ import annotations

# 从子模块导入核心类和函数
from .core import ProblemSpec, Workspace, LocalCheckFail, reason_from_future
from .llm import llm_call
from .specs import Game24Spec, GSM8KSpec

__all__ = [  # [知识2.17] 定义对外暴露的名称
    "ProblemSpec", "Workspace", "LocalCheckFail", "reason_from_future",
    "llm_call", "Game24Spec", "GSM8KSpec",
]
```
> **是什么**：Python 包的入口文件。`import reason_from_future` 时 Python 会执行这个文件。
>
> **`.core` 和 `..core` 的区别**：`.core` = 同级目录的 `core.py`；`..core` = 上级目录的 `core.py`

### 5.2 Spec 子包入口 `reason_from_future/specs/__init__.py`

```python
from .game24 import Game24Spec
from .gsm8k import GSM8KSpec
from .code_writing import CodeWritingSpec
from .general_problem_solving import GeneralProblemSolvingSpec

__all__ = ["Game24Spec", "GSM8KSpec", "CodeWritingSpec", "GeneralProblemSolvingSpec"]
```
> 集中导入所有 Spec 类，方便 `from reason_from_future.specs import Game24Spec`。

---

## 6. specs/ — 四个领域规约

### 6.1 Game24Spec（game24.py）— 最简单的 Spec

#### 辅助工具：安全表达式求值

```python
_ALLOWED_BIN_OPS = {  # [知识2.11/2.12] AST 节点类型 → 运算函数的映射
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
```

```python
class _SafeEval(ast.NodeVisitor):
    """安全求值器：只允许 + - * / 和括号，其他一律拒绝。"""
    def visit(self, node: ast.AST):
        if isinstance(node, ast.Expression):       # 根节点 → 递归访问子节点
            return self.visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)               # 数字 → 返回浮点数
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.visit(node.operand)        # 负号 → 取反
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
            left = self.visit(node.left)            # 递归求左操作数
            right = self.visit(node.right)          # 递归求右操作数
            return _ALLOWED_BIN_OPS[type(node.op)](left, right)
        raise ValueError("Unsafe or unsupported expression component encountered.")

def safe_eval(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    return _SafeEval().visit(tree)
```
> **为什么不用 `eval()`**：`eval()` 可执行任意 Python 代码（包括删文件）。`_SafeEval` 通过 AST 解析只处理已知节点类型，杜绝代码注入。

#### Game24Spec 类

```python
class Game24Spec(ProblemSpec):
    TARGET_VALUE = 24.0

    def __init__(self, nums: List[int]):
        super().__init__()
        self.original_nums: List[int] = list(nums)       # 原始4个数字
        self._orig_counter: Counter[int] = Counter(nums)  # [知识2.10] 每个数字出现次数
```
> **为什么用 Counter**：24点游戏要求每个数字恰好用一次。Counter 可追踪"哪些数字用了、还剩哪些"。

**`derive_final_target`** → 返回 `"24"`（常量）

**`parse_target_step`** → 从 LLM 输出提取算术表达式
```python
    def parse_target_step(self, raw_text: str) -> str:
        raw = raw_text.strip()
        if "=" in raw:
            left, _ = raw.split("=", 1)  # "3+5=8" → 取 "3+5"
            return left.strip()
        return raw
```

**`check_local`** → 检查目标值是否在工作台中
```python
    def check_local(self, state: Workspace, target_step: str) -> bool:
        try:
            target_val = safe_eval(target_step)   # 算出目标值
        except Exception:
            return False
        for item in state.values():
            if isinstance(item, dict) and abs(item.get("value", 1e9) - target_val) < 1e-9:
                return True   # 工作台中有表达式的值等于目标值
        return False
```

**`verify_final`** → 验证是否得到24且使用了所有数字
```python
    def verify_final(self, state: Workspace) -> Tuple[bool, str, float]:
        for item in state.values():
            if not isinstance(item, dict): continue
            if abs(item.get("value", 0.0) - self.TARGET_VALUE) > 1e-9: continue
            if Counter(item.get("nums", [])) == self._orig_counter:
                return True, item["expr"], self.TARGET_VALUE  # ✅ 24 且用了全部数字
        return False, "", self.TARGET_VALUE
```
> 两个条件：值等于24 **且** 使用的数字集合等于原始数字集合。

**`parse_workspace_update`** → 解析 LLM 输出的新表达式（带三重校验）
```python
    def parse_workspace_update(self, raw_text: str, state: Workspace) -> Workspace:
        expr_line = raw_text.strip()
        if expr_line == "CANNOT_ACHIEVE_WITH_AVAILABLE_NUMBERS":
            return Workspace()  # LLM 说做不到

        # ... 预处理（去等号右边、去分号句号）...

        try:
            value = safe_eval(expr_line)  # 校验1：表达式能安全求值
        except Exception:
            return Workspace()

        nums_in_expr = self._extract_numbers(expr_line)
        avail_counter = self._available_counter(state)

        # 校验2+3：使用的数字是可用数字的子集，且没有超量
        if not all(avail_counter[n] >= c for n, c in Counter(nums_in_expr).items()):
            return Workspace()

        key = str(uuid.uuid4())  # [知识2.15] 随机唯一 key
        return Workspace({key: {"expr": expr_line, "value": value, "nums": nums_in_expr}})
```
> **三重校验**：1) 表达式可安全求值 2) 使用的数字是可用数字子集 3) 没超量使用任何数字

**`merge_aliases`** → 直接返回原状态（24点不需要合并同义词）

---

### 6.2 GSM8KSpec（gsm8k.py）— 数学题 Spec

#### 初始化

```python
class GSM8KSpec(ProblemSpec):
    def __init__(self, problem_data: Dict[str, str]):
        super().__init__()
        self.question = problem_data["question"]   # 题目文本
        self.problem_data = problem_data

        # 从答案字符串中提取数值
        # GSM8K 答案格式："解释过程... #### 24" 或直接 "24"
        answer_str = str(problem_data["answer"])
        match = re.search(r"(?:####\s*)?([0-9,.]+)\s*$", answer_str)
        if match:
            self.gold_numeric_answer = float(match.group(1).replace(",", ""))
        else:
            self.gold_numeric_answer = float('nan')  # 解析失败用 NaN
```

#### `derive_final_target` → `"final_answer"`

#### `parse_workspace_update` — 三层降级解析

```python
    def parse_workspace_update(self, raw_text: str, state: Workspace) -> Workspace:
        # 尝试1：JSON 格式 {"var": "x", "expr": "a+b", "value": 123}
        # 尝试2：LaTeX 格式 "The final answer is $\boxed{24}$"
        # 尝试3：简单格式 "final answer is 24"
        return Workspace()  # 全都解析失败
```
> **为什么三层**：LLM 输出格式不可预测。可能返回 JSON、LaTeX、自然语言。三层降级确保总能提取答案。

**JSON 解析中的表达式验证**：
```python
        if expr and isinstance(expr, str):
            try:
                safe_locals = {k: v for k, v in state.items() if isinstance(v, (int, float))}
                calculated = float(eval(expr, {}, safe_locals))  # 用当前 state 计算表达式
                if abs(calculated - float(var_value)) > 1e-4:
                    return Workspace()  # 表达式计算值 ≠ 声称值 → 拒绝
            except NameError:
                pass  # 表达式引用了未知变量 → 暂时接受，后续验证
```
> **是什么**：如果 LLM 既给了表达式又给了值，验证表达式计算结果是否跟声称的值一致。
>
> **安全措施**：`eval(expr, {}, safe_locals)` 中 `{}` 不提供全局变量，`safe_locals` 只包含数值。即使表达式有恶意代码也找不到可调用函数。
>
> **注意**：这里用 `eval` 而非 `safe_eval`，因为需要支持变量引用（如 `a + b`）。

#### `merge_aliases` — 同义词合并

```python
    def merge_aliases(self, state: Workspace) -> Workspace:
        if len(state) <= 1:
            return state

        normalized_map: dict[str, str] = {}
        for var in state:
            # 去掉常见修饰词（number/num/total/initial/before/after/of/the）
            norm = re.sub(r"(?:number|num|total|initial|before|after|of|the)", "", var.lower())
            norm = re.sub(r"[_\s]+", "_", norm).strip("_")
            normalized_map.setdefault(norm, var)  # 同一个 norm 只保留第一个

        new_state = Workspace()
        for norm_key, representative_var in normalized_map.items():
            new_state[representative_var] = state[representative_var]
        return new_state
```
> **是什么**：合并变量名中的同义词。
>
> **比喻**：草稿纸上写了 `initial_science_books = 150` 和 `science_books_before_bonus = 150`——这两个其实是同一个东西！合并后只保留一个。
>
> **归一化逻辑**：去掉 `number/num/total/initial/before/after/of/the` 这些修饰词后，如果两个变量名一样，就是同义词。

#### `check_local` — 检查变量是否为数值

```python
    def check_local(self, state: Workspace, target_step: str) -> bool:
        if target_step not in state:
            return False
        return isinstance(state[target_step], (int, float))  # 只检查是否为数值
```
> 比 Game24Spec 的 `check_local` 简单——GSM8K 只需检查变量存在且是数值。

#### `verify_final` — 跟标准答案比对

```python
    def verify_final(self, state: Workspace) -> Tuple[bool, str, float]:
        guess_val = state.get(self.derive_final_target(self.question))
        if guess_val is None:
            return False, "No final answer provided."

        # 把 guess 转成 float
        if isinstance(guess_val, str):
            guess_val_cleaned = guess_val.replace(",", "")
            numeric_guess = float(guess_val_cleaned)
        elif isinstance(guess_val, (int, float)):
            numeric_guess = float(guess_val)
        else:
            return False, f"Unexpected type {type(guess_val).__name__}."

        is_correct = abs(numeric_guess - self.gold_numeric_answer) < 1e-5
        return is_correct, str(numeric_guess), self.gold_numeric_answer
```

#### prompt 方法

- **`prompt_last_step`**：反向推理——"要算 final_answer，先得算什么？"，要求 LLM 返回 `{"next_variables": ["var1", "var2"]}`
- **`prompt_forward_step`**：正向计算——"请计算变量 X 的值"，要求 LLM 返回 `{"var": "X", "expr": "a+b", "value": 123}`
- **`parse_target_step`**：从 LLM 反向推理输出提取变量名，支持 `{"next_variables": [...]}` 和 `{"next_variable": "x"}` 两种格式

---

### 6.3 GeneralProblemSolvingSpec（general_problem_solving.py）— 通用问题 Spec

#### 核心区别

| 对比项 | Game24Spec / GSM8KSpec | GeneralProblemSolvingSpec |
|--------|----------------------|--------------------------|
| 答案类型 | 数值（24 / 数学答案） | 结构化方案（系统设计/策略/决策） |
| workspace 结构 | 扁平的 `{var: value}` | 嵌套的 `{components: {...}, decisions: {...}, ...}` |
| 验证方式 | 有金标准 | 无金标准，用质量评分 |
| 目标名 | `"24"` / `"final_answer"` | `"complete_solution"` |

#### workspace 结构

```python
self.workspace_schema = {
    "components": {},      # 设计组件
    "decisions": {},       # 关键决策
    "constraints": {},     # 需求和限制
    "dependencies": {},    # 元素间依赖关系
    "rationales": {},       # 选择理由
    "open_questions": [],   # 未解决的问题
    "assumptions": []       # 工作假设
}
```
> **比喻**：Game24Spec 的 workspace 是一张白纸，上面只有"表达式=值"；GeneralProblemSolvingSpec 的 workspace 是一个文件柜，有很多抽屉（components、decisions 等），每个抽屉里还有文件夹。

#### `parse_workspace_update` — 按 `update_type` 分发

```python
    def parse_workspace_update(self, raw_text: str, state: Workspace) -> Workspace:
        # 解析 JSON → 根据 update_type 字段分发：
        if update_type == "component":    # 添加设计组件
        elif update_type == "decision":   # 记录决策
        elif update_type == "dependency": # 添加依赖关系
        elif update_type == "solution_summary":  # 最终方案
        elif update_type == "batch":      # 批量更新（递归调用自身）
```
> **`batch` 模式**：LLM 一次返回多个更新，通过递归调用 `parse_workspace_update` 逐个处理。
>
> **比喻**：点外卖时一次点了5个菜（batch），厨房还是一个一个做（递归调用）。

#### `check_local` — 按 target_step 前缀判断

```python
    def check_local(self, state: Workspace, target_step: str) -> bool:
        if target_step == "complete_solution":
            return "complete_solution" in state and bool(state["complete_solution"].get("summary"))

        if target_step.startswith("define_"):    # 检查组件是否已定义
        elif target_step.startswith("decide_"):  # 检查决策是否已做出
        elif target_step.startswith("analyze_"): # 检查分析是否存在

        return self._find_in_nested_dict(state, target_step)  # 通用：在嵌套字典中搜索
```
> **`_find_in_nested_dict`**：递归搜索嵌套字典。像在文件柜的每个抽屉、每个文件夹中找文件。

#### `verify_final` — 质量评分（非精确匹配）

```python
    def verify_final(self, state: Workspace) -> Tuple[bool, str, float]:
        # 计算质量分数：
        quality_score = 0.0
        if has_summary:        quality_score += 0.4   # 有方案摘要
        if has_components:     quality_score += 0.3 * min(len(components) / 3, 1.0)
        if has_decisions:      quality_score += 0.3 * min(len(decisions) / 2, 1.0)

        is_acceptable = quality_score > 0.7  # 超过 0.7 分算通过
        return is_acceptable, solution_text, quality_score
```
> **没有金标准**：通用问题不像数学题有唯一正确答案，所以用质量评分替代精确匹配。

#### prompt 方法

- **`prompt_last_step`**：反向规划——"要达成 complete_solution，最关键的下一步是什么？"，要求返回 `{"next_task": "define_core_services"}`
- **`prompt_forward_step`**：正向执行——根据 `target_step` 的前缀（`define_`/`decide_`/`analyze_`）给出不同指令

---

### 6.4 CodeWritingSpec（code_writing.py）— 代码生成 Spec

#### 核心区别

| 对比项 | GeneralProblemSolvingSpec | CodeWritingSpec |
|--------|--------------------------|-----------------|
| 产出 | 系统设计方案 | 可执行代码 |
| workspace 结构 | components/decisions/dependencies | modules/functions/classes/test_cases |
| 目标名 | `"complete_solution"` | `"complete_code_solution"` 或 `"implement_function_<name>"` |
| 验证方式 | 质量评分 > 0.7 | 质量评分 > 0.5 |

#### workspace 结构

```python
self.workspace_schema = {
    "modules": {},      # module_name: { "content": "...", "description": "..." }
    "functions": {},    # func_name: { "signature": "...", "body": "...", "module": "..." }
    "classes": {},      # class_name: { "attributes": {}, "methods": {}, "module": "..." }
    "test_cases": {},   # test_name: { "input": "...", "expected_output": "..." }
    "decisions": {},    # decision_name: { "choice": "...", "rationale": "..." }
    "dependencies": {}, # element_name: { "depends_on": [], "type": "..." }
    "solution_code": None   # 最终代码字符串
}
```

#### `derive_final_target` — 智能推断目标

```python
    def derive_final_target(self, problem: str) -> str:
        # 尝试从问题描述中提取函数名
        # 如 "function should be named 'factorial'" → "implement_function_factorial"
        patterns = [
            r"function\s+should\s+be\s+named\s+['\"]([A-Za-z_]\w*)['\"]",
            r"function\s+must\s+be\s+named\s+['\"]([A-Za-z_]\w*)['\"]",
            r"['\"]([A-Za-z_]\w*)['\"]\s*(?:function|func)",
        ]
        for pat in patterns:
            m = re.search(pat, problem, re.IGNORECASE)
            if m:
                return f"implement_function_{m.group(1)}"

        # 默认返回通用目标
        return "complete_code_solution"
```
> **是什么**：如果题目明确要求实现某个函数，直接把目标定位到该函数，避免漫无目的地规划。

#### `parse_workspace_update` — 按 `update_type` 分发

支持的更新类型：`module` / `function` / `class` / `test_case` / `decision` / `dependency` / `solution_code` / `batch`

**特殊逻辑**：如果 LLM 返回的不是 JSON 而是一大段代码（超过3行且包含 `def`/`class`/`import`），直接存入 `solution_code`。

#### `check_local` — 按 target_step 前缀判断

```python
    def check_local(self, state: Workspace, target_step: str) -> bool:
        if target_step == "complete_code_solution":
            # 检查有 solution_code 或有 modules
        elif target_step.startswith("define_module_"):
            # 检查 modules 中有该模块且有 content
        elif target_step.startswith("implement_function_"):
            # 检查 functions 中有该函数且有 body
            # ⚠️ 通过后还会在 state 中写入完成标记
            if done:
                state[target_step] = f"function '{function_name}' implemented"
        elif target_step.startswith("define_class_"):
            # 检查 classes 中有该类且有 methods 或 attributes
        elif target_step.startswith("write_tests_for_"):
            # 检查 test_cases 中有针对该元素的测试
        elif target_step.startswith("decide_on_"):
            # 检查 decisions 中有该决策且有 choice 和 rationale
```

#### `verify_final` — 代码质量评分

```python
    def verify_final(self, state: Workspace) -> Tuple[bool, str, float]:
        # 质量分数分布：
        # 0.4 分：有 solution_code 或 modules
        # 0.2 分：有 functions（每个0.05，上限0.2）
        # 0.2 分：有 classes（每个0.1，上限0.2）
        # 0.2 分：有 test_cases（每个0.05，上限0.2）

        is_acceptable = normalized_score >= 0.5  # 0.5 分算通过
```

---

## 7. tools/ — 文件读写工具

### 7.1 read_file.py

```python
from pathlib import Path
from typing import Union

def read_file(file_path: Union[str, Path]) -> str:
    path = Path(file_path).expanduser().resolve()  # 展开 ~ 转绝对路径
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file but found directory: {path}")
    return path.read_text(encoding="utf-8")  # [知识2.16] Path 的便捷读文件方法
```
> **是什么**：安全的文件读取工具。做了两重防御检查：文件是否存在、是否是目录。

### 7.2 write_to_file.py

```python
def write_to_file(file_path: Union[str, Path], content: str, *, overwrite: bool = True) -> None:
    path = Path(file_path).expanduser().resolve()
    if path.exists() and path.is_dir():
        raise IsADirectoryError(f"Cannot write to a directory: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"File already exists and overwrite=False: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建父目录
    path.write_text(content, encoding="utf-8")
```
> **是什么**：安全的文件写入工具。
> - `overwrite=False` 可防止意外覆盖
> - `path.parent.mkdir(parents=True, exist_ok=True)` 自动创建所有中间目录

---

## 8. demos/ — 运行入口

### 8.1 demo_game24.py

```python
from reason_from_future.core import reason_from_future
from reason_from_future.specs import Game24Spec

easy = [1, 3, 6, 11]     # (11-3) * (6-3) == 24
easy_2 = [1, 2, 5, 9]    # (9-1) * (5-2) == 24
hard = [2, 3, 5, 12]     # (12/(3-(5/2))) == 24

current = easy_2
spec = Game24Spec(current)   # 创建 24点 Spec
answer = reason_from_future(  # 调用主循环
    "Reach 24 with all the numbers {}".format(current),
    spec,
    verbose=True
)
print("Solution:", answer)
```
> **运行流程**：
> 1. 创建 Game24Spec（传入4个数字）
> 2. 调用 `reason_from_future()`，自动执行 G→R→C 循环
> 3. 打印最终答案

### 8.2 demo_gsm8k.py

```python
from reason_from_future.core import reason_from_future
from reason_from_future.specs.gsm8k import GSM8KSpec

medium = {
    "question": "A school library ordered 600 new books...",
    "answer": "198",
}

def main(verbose=True):
    spec = GSM8KSpec(medium)
    answer = reason_from_future(
        problem=medium["question"],
        spec=spec,
        max_iters=7,
        verbose=verbose,
        require_gold=False,   # 不要求跟金标准完全匹配
        min_iters=2,         # 至少迭代2次
    )
    print("FINAL ANSWER:", answer)

if __name__ == "__main__":
    main()
```
> **`require_gold=False`**：GSM8K 标准答案的解析可能出错，设为 False 让循环更宽松。
> **`if __name__ == "__main__"`**：Python 的惯用写法——只有直接运行此文件时才执行 `main()`，被 import 时不执行。

### 8.3 demo_general.py

```python
system_design_problem = {
    "problem_statement": "Design a scalable URL shortening service...",
    "problem_type": "system_design",
    "requirements": [...],
    "evaluation_criteria": [...]
}

def run_example(problem_config, max_iters=10, verbose=True):
    spec = GeneralProblemSolvingSpec(problem_config)
    solution = reason_from_future(
        problem=problem_config["problem_statement"],
        spec=spec,
        max_iters=max_iters,
        min_iters=3,           # 至少迭代3次
        require_gold=False,    # 通用问题没有金标准
        verbose=verbose
    )
    return solution
```

---

## 9. 跨文件逻辑追踪

### 一次完整的 G→R→C 循环（以 demo_gsm8k.py 为例）

**第0步：准备**

```
demo_gsm8k.py                          reason_from_future/core.py
    │                                       │
    ├─ GSM8KSpec(medium)  ──────────────→  spec.__init__()
    │  创建 Spec 实例                        提取 question + gold_numeric_answer
    │
    ├─ reason_from_future(                   state = Workspace()
    │      problem=...,                      goal = spec.derive_final_target(problem)
    │      spec=spec,                        → goal = "final_answer"
    │      max_iters=7,                      avoid = set()
    │  )                                     attempt_counts = {}
    │                                       ...
```

**第1轮迭代：直接尝试**

```
core.py 主循环
    │
    ├─ spec.check_local(state, "final_answer")  → False（state 是空的）
    │
    ├─ spec.prompt_forward_step(state, "final_answer", avoid)
    │  └─ gsm8k.py 构建 prompt：
    │     "你正在解决一道数学题...请计算 final_answer"
    │
    ├─ llm_call(prompt)  ───────────────────→ llm.py
    │                                       ├─ 构建 types.Content
    │                                       ├─ client.models.generate_content()
    │                                       └─ 返回 LLM 文本响应
    │
    ├─ spec.parse_workspace_update(raw_text, state)
    │  └─ gsm8k.py 解析 JSON：
    │     {"var": "final_answer", "expr": "300-102", "value": 198}
    │     → Workspace({"final_answer": 198})
    │
    ├─ state | parsed_update → state = {"final_answer": 198}
    │
    ├─ spec.check_local(state, "final_answer")  → True（值存在且是数值）
    │
    └─ spec.verify_final(state)
       └─ gsm8k.py 比对：198 == gold_numeric_answer?
          → True! 返回 "198"
```

**如果直接尝试失败，则进入反向推理**：

```
core.py 主循环（续）
    │
    ├─ spec.prompt_last_step(state, "final_answer", avoid)
    │  └─ gsm8k.py 构建反向推理 prompt：
    │     "要算出 final_answer，先得算什么变量？"
    │
    ├─ llm_call(prompt)  ─────────────────→ 返回 LLM 文本
    │
    ├─ spec.parse_target_step(raw_text)
    │  └─ gsm8k.py 解析 JSON：
    │     {"next_variables": ["science_books", "bonus_books"]}
    │     → 返回 "science_books"
    │
    ├─ spec.prompt_forward_step(state, "science_books", avoid)
    │  └─ gsm8k.py 构建正向计算 prompt
    │
    ├─ llm_call(prompt)  ─────────────────→ 返回 LLM 文本
    │
    ├─ spec.parse_workspace_update(raw_text, state)
    │  └─ 解析 → Workspace({"science_books": 150})
    │
    ├─ spec.check_local(合并后state, "science_books")  → True
    │
    ├─ state = state | parsed_update  → 提交新状态
    │
    └─ spec.merge_aliases(state)  → 合并同义词
       回到循环顶部，检查 "final_answer" 是否现在可算...
```

### 跨文件调用链总结

```
demo → 创建 Spec ──→ spec.__init__()
     → reason_from_future()
          ├── spec.derive_final_target()  → 确定目标
          ├── spec.check_local()          → 局部检查
          ├── spec.prompt_forward_step()  → 构建正向 prompt
          │       └── llm_call()          → 调 Gemini API
          │               └── google.genai SDK
          ├── spec.parse_workspace_update() → 解析 LLM 输出
          ├── spec.verify_final()          → 终极验证
          ├── spec.prompt_last_step()      → 构建反向 prompt
          │       └── llm_call()
          ├── spec.parse_target_step()     → 提取下一步
          └── spec.merge_aliases()         → 合并同义词
```

---

## 10. 四个 Spec 的对比进化线

### 从简单到复杂的进化

```
Game24Spec ──→ GSM8KSpec ──→ GeneralProblemSolvingSpec ──→ CodeWritingSpec
  (纯算术)      (数学推理)        (系统设计/策略)              (代码生成)
```

| 特性 | Game24Spec | GSM8KSpec | GeneralProblemSolving | CodeWritingSpec |
|------|-----------|-----------|----------------------|-----------------|
| **目标** | 常量 `"24"` | `"final_answer"` | `"complete_solution"` | `"complete_code_solution"` 或 `"implement_function_*"` |
| **Workspace 结构** | 扁平：`{uuid: {expr, value, nums}}` | 扁平：`{var: number}` | 嵌套：`{components, decisions, ...}` | 嵌套：`{modules, functions, classes, ...}` |
| **check_local** | safe_eval 比较 | isinstance 数值检查 | 前缀判断 + 嵌套搜索 | 前缀判断 + 嵌套搜索 + 完成标记 |
| **verify_final** | 值=24 && 数字全用 | 数值精确匹配 | 质量评分 > 0.7 | 质量评分 > 0.5 |
| **merge_aliases** | 无操作 | 去修饰词归一化 | 无操作 | 无操作 |
| **parse_update** | 单一格式（表达式） | 三层降级（JSON→LaTeX→自然语言） | 按 update_type 分发 | 按 update_type 分发 + 代码块兜底 |
| **安全求值** | ast._SafeEval | eval + 沙箱 | 无 | 无 |
| **数字追踪** | Counter 追踪使用情况 | 无 | 无 | 无 |

### 进化规律

1. **Game24Spec → GSM8KSpec**：从"纯算术验证"到"语义推理"，增加了同义词合并、多层解析、表达式交叉验证
2. **GSM8KSpec → GeneralProblemSolvingSpec**：从"数值精确匹配"到"质量评分"，因为系统设计没有唯一正确答案
3. **GeneralProblemSolvingSpec → CodeWritingSpec**：从"方案描述"到"可执行产物"，workspace 结构从"文档型"变为"代码型"，增加了智能目标推断

### 共同骨架

尽管差异很大，四个 Spec 都遵循 `ProblemSpec` 定义的8个方法。这就是**抽象基类**的威力——主循环 `reason_from_future()` 完全不需要知道自己在处理什么类型的问题，它只调用 spec 的方法，具体行为由子类决定。

这就像一个万能遥控器——按钮（8个方法）是一样的，但控制电视、空调、音响时（不同的 Spec），效果完全不同。

---

> 📝 本文档由 AI 辅助生成，如有疑问请参考源码原文。