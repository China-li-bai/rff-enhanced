"""
GRAVEC — 「以果决其行」混合实现

架构：控制面（确定性 Python）+ 智能面（LLM-backed Agent）

  控制面（Control Plane）— loop.py
    G→R→A→V→E→C 循环骨架，阈值判断，停滞检测，黑名单管理
    这是"以果决其行"的纪律——确定性代码，不交给 Agent 随意改

  智能面（Intelligence Plane）— agents/
    V: value_agent.py   — 语义主证识别（"舌红是主证还是兼证？"）
    A: action_agent.py  — 行动执行（可调工具，"开方+验方"）
    E: effect_agent.py  — 多通道验效（"望闻问切"四诊）
    C: chief_agent.py   — 综合重新辨证（"效不更方/不效调方"）

  数据层 — models.py
    Observation, ValueScore, GoalRevision, CausalDiagnosis, ReasoningPolicy

  工具层 — tools/
    ToolRegistry: 注册/发现/调用工具

倪海厦方法论映射：
  G (以果)    → loop.py 中的反向推理
  R (推理)    → loop.py 中的正向计算
  A (决其行)  → agents/action_agent.py
  V (价值判断) → agents/value_agent.py
  E (验效)    → agents/effect_agent.py
  C (校验)    → agents/chief_agent.py
"""

from .models import CausalDiagnosis, GoalRevision, Observation, ReasoningPolicy, ValueScore
from .spec import NiHaixiaSpec

__all__ = [
    "NiHaixiaSpec",
    "Observation",
    "ValueScore",
    "GoalRevision",
    "CausalDiagnosis",
    "ReasoningPolicy",
]
