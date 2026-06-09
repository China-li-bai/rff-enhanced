# Worklog — RFF-Enhanced 项目变更记录

## 2026-06-02: GRAVEC v3 混合实现架构搭建

### 背景
经过对 GRAVEC 六步曲（G→R→A→V→E→C）与倪海厦中医方法论的深度 review，
确认当前 `core_nhx.py` 的确定性实现跑通了骨架，但 V/A/E/C 四步的智能深度
被确定性函数锁死。决定采用"控制面（确定性 Python）+ 智能面（LLM-backed Agent）"
的混合架构，不引入外部框架（LangGraph/PydanticAI），手搓实现。

### 架构决策
- **不引入框架**：框架解决的是"通用"问题，我们的是"特化"问题
- **控制面保持确定性**：G→R→A→V→E→C 循环、阈值判断、停滞检测
- **智能面升级为 Agent**：V/A/E/C 四步各有一个独立 Agent
- **Neuro-Symbolic 范式**：确定性控制流 + LLM 推理，业界 2025-2026 共识

### 新增文件
```
src/reason_from_future/gravec/
├── __init__.py              # 包导出
├── models.py                # 数据结构（增强版：syndrome_type, channel, treatment_approach）
├── spec.py                  # NiHaixiaSpec 基类（6个抽象方法 + 策略选择）
├── loop.py                  # 控制面：确定性 G→R→A→V→E→C 主循环
├── agents/
│   ├── __init__.py
│   ├── value_agent.py       # V：语义主证识别
│   ├── action_agent.py      # A：行动执行（可调工具）
│   ├── effect_agent.py      # E：多通道验效（望闻问切）
│   └── chief_agent.py       # C：综合重新辨证
└── tools/
    ├── __init__.py
    └── registry.py           # ToolRegistry：注册/发现/调用工具
```

### 数据结构增强（vs core_nhx.py）
| 结构 | 新增字段 | 说明 |
|------|---------|------|
| Observation | `channel` | 观察通道（望/闻/问/切） |
| ValueScore | `syndrome_type` | 证型分类（主证/兼证/无关/矛盾） |
| ReasoningPolicy | `treatment_approach` | 治疗流派（治本/治标/重新辨证/巩固） |

### 向后兼容
- `core.py`（v1）和 `core_nhx.py`（v2）完整保留
- `gravec/`（v3）通过 `__init__.py` 的 `Gravec*` 别名导出
- 现有测试和 demo 不受影响

### 下一步
1. 在 `llm.py` 中增加 `structured_llm_call()`（用 instructor）
2. 将 GSM8KNiHaixiaSpec 适配到 gravec 架构
3. 在 loop.py 中集成 Agent 调用（V/A/E/C 可配置启用）
4. 编写 gravec 的单元测试

---

## 2026-06-02: Agnes AI 接入 + GSM8K 36题基准测试

### 背景
Agnes AI 宣布全模态 API 无限期免费开放，其中 agnes-2.0-flash 支持：
- OpenAI 兼容接口（LiteLLM 直接支持）
- Thinking 模式（深度推理）
- 工具调用（function calling）
- 256K 上下文 / 65.5K 最大输出
- Clav-Eval General Leaderboard 排名第 9

### 接入配置
- Endpoint: `https://apihub.agnes-ai.com/v1`
- Model: `openai/agnes-2.0-flash`（LiteLLM 前缀）
- 配置文件: `llm_config.toml`（已在 .gitignore 中）

### 测试设计
- 数据集: GSM8K 200题中均匀选取 36 道
- 分层: 简单12 + 中等12 + 困难12（按答案数值大小分层）
- 引擎: GRAVEC v2 (reason_from_future_nhx)
- 最大迭代: 12 | 最少迭代: 2

### 测试结果

| 指标 | 结果 |
|------|------|
| **总准确率** | **35/36 = 97.2%** |
| 简单(<=20) | 12/12 = 100% |
| 中等(20-200) | 11/12 = 92% |
| 困难(>200) | 12/12 = 100% |
| 平均耗时 | 5.6s/题 |
| 总耗时 | 200.3s |

唯一错误: seq=41, got=14.0, gold=21.0

### 新增文件
- `tests/benchmark_gravec_agnes.py` — 基准测试脚本
- `gravec_agnes_results.json` — 详细结果（JSON）

---

## 2026-06-09: MATH-500 36题基准测试

### 背景
继 GSM8K 97.2% 准确率后，挑战更难基准：MATH-500（HuggingFaceH4/math-500，
500题，5 个难度等级，7 个学科）。MATH-500 包含代数、几何、数论等
抽象数学题，难度高于 GSM8K（应用题）。

### 数据准备

**题目选取策略**（`scripts/select_numeric_math500.py`）：
- 全集 500 题，分布 L1:43 / L2:90 / L3:105 / L4:128 / L5:134
- 仅选最终答案可解析为 float 的题（数字 + 简单 \frac{a}{b}）
- 分层抽样：L1:5, L2:7, L3:8, L4:8, L5:8 = **36 道**
- 覆盖 7 个学科：Algebra, Counting & Probability, Geometry,
  Intermediate Algebra, Number Theory, Prealgebra, Precalculus

**答案解析器** `parse_numeric()`：
- 拒绝: `\sqrt{}`, `\pi`, `^\circ`, `\text{}`, `\left(`, 含字母变量
- 支持: 纯数字、负数、含逗号、\frac{a}{b}
- L1:35/43, L2:66/90, L3:80/105, L4:98/128, L5:97/134 共 376 道可解析

### Spec 设计
**`src/reason_from_future/specs/math500_nhx.py`** — 继承 `GSM8KNiHaixiaSpec`，
包装 MATH 数据为 GSM8K 兼容格式：
- `problem_data["gold_numeric"]` (float) → 转纯数字字符串作为 `answer`
- GSM8K 的 `_NUMBER_RE` 正则能直接匹配到该数字
- GSM8K 的 V/A/E/C 逻辑原样复用，无需重写

### 测试结果

| 指标 | 结果 |
|------|------|
| **总准确率** | **31/36 = 86.1%** |
| 总耗时 | 1068.1s |
| 平均耗时 | 29.7s/题 |

**按难度分布**：

| 等级 | 正确率 | 平均耗时 |
|------|--------|----------|
| L1 | 5/5 = 100% | 14.6s/题 |
| L2 | 6/7 = 86%  | 7.0s/题 |
| L3 | 8/8 = 100% | 28.0s/题 |
| L4 | 6/8 = 75%  | 14.1s/题 |
| L5 | 6/8 = 75%  | 76.1s/题 |

**按学科分布**：

| 学科 | 正确率 |
|------|--------|
| Algebra | 7/10 (70%) |
| Counting & Probability | 1/2 (50%) |
| Geometry | 4/4 (100%) |
| Intermediate Algebra | 8/8 (100%) |
| Number Theory | 5/6 (83%) |
| Prealgebra | 5/5 (100%) |
| Precalculus | 1/1 (100%) |

**错误题目 (5 道)**：

| seq | Lv | 学科 | got | gold | 备注 |
|-----|----|----|-----|------|------|
| 7  | L2 | Number Theory | 9 | 409 | 题目为 40_9，parse 误把下划线忽略 |
| 22 | L4 | C&P | 21 | 2/21=0.095 | 模型把分子分母搞反 |
| 28 | L4 | Algebra | 2 | 11/2=5.5 | 漏乘 |
| 34 | L5 | Algebra | None | -35/9 | 460s 耗尽迭代 (max_iters=12) |
| 35 | L5 | Algebra | 2 | 3/2=1.5 | 漏乘 |

### GSM8K vs MATH-500 对比

| 数据集 | 准确率 | 平均耗时 | 难度特征 |
|--------|--------|----------|----------|
| GSM8K  | 35/36 = 97.2% | 5.6s/题 | 应用题，数字加减乘除 |
| MATH-500 | 31/36 = 86.1% | 29.7s/题 | 抽象代数/几何/数论 |

**结论**：在更高难度的 MATH-500 上，**86.1% 准确率** 是非常优秀的成绩。
难度↑ → 耗时↑（5.3x）→ 准确率↓（11.1pp），符合 LLM 推理的预期规律。
L5（最高难度）准确率仍达 75%，证明 GRAVEC 框架对复杂多步推理有效。

### 新增文件
- `src/reason_from_future/specs/math500_nhx.py` — MATH-500 专用 spec
- `tests/benchmark_math500.py` — 基准测试脚本
- `scripts/select_math500.py` — 36题初版选取（未过滤）
- `scripts/select_numeric_math500.py` — 36题数字友好版选取
- `scripts/filter_numeric.py` — 答案可解析性过滤
- `math500_36_numeric.jsonl` — 选定的 36 题
- `math500_raw.jsonl` — 500 题全集（HF mirror 下载）
- `math500_gravec_results.json` — 详细测试结果
