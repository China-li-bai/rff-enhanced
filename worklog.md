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
