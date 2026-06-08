"""
GRAVEC 智能面 — LLM-backed Agents

四个 Agent 对应倪海厦方法论的四步核心判断：
  V: value_agent.py   — 语义主证识别
  A: action_agent.py  — 行动执行（可调工具）
  E: effect_agent.py  — 多通道验效
  C: chief_agent.py   — 综合重新辨证

每个 Agent 的设计原则：
  1. 接受 spec 的确定性 fallback 结果作为输入
  2. 用 LLM 推理增强判断的语义深度
  3. 返回与确定性接口兼容的数据结构
  4. 可独立测试（mock LLM）和独立禁用（fallback to 确定性）
"""

from .action_agent import ActionAgent
from .chief_agent import ChiefAgent
from .effect_agent import EffectAgent
from .value_agent import ValueAgent

__all__ = [
    "ValueAgent",
    "ActionAgent",
    "EffectAgent",
    "ChiefAgent",
]
