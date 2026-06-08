"""Reason-from-Future – Near-Executable Reference Implementation.

This package contains a domain-agnostic controller implementing the
Reason-from-Future algorithm with example domain specs.  See
`PLANS/inital_setup.md` for the original design rationale.

倪海厦「以果决其行」增强版：在原版 RFF 的 G→R→C 三步曲基础上，
新增 A（行动执行）、V（价值判断）、E（验效反馈）三步，
以及果行共变（目标随反馈修正）机制，形成完整的 G→R→A→V→E→C 六步曲。

架构演进：
  v1 (core.py):        G→R→C（只会想，不会做）
  v2 (core_nhx.py):    G→R→A→V→E→C（确定性六步曲）
  v3 (gravec/):        控制面（确定性循环）+ 智能面（LLM-backed Agent）
"""

from __future__ import annotations

# Core abstractions and controller (原版)
from .core import ProblemSpec, Workspace, LocalCheckFail, reason_from_future

# 倪海厦增强核心 v2（确定性版本，保留向后兼容）
from .core_nhx import (
    CausalDiagnosis,
    GoalRevision,
    NiHaixiaSpec,
    Observation,
    ReasoningPolicy,
    ValueScore,
    reason_from_future_nhx,
)

# 倪海厦增强核心 v3（混合实现：控制面+智能面）
from .gravec import (
    NiHaixiaSpec as GravecSpec,
    Observation as GravecObservation,
    ValueScore as GravecValueScore,
    GoalRevision as GravecGoalRevision,
    CausalDiagnosis as GravecCausalDiagnosis,
    ReasoningPolicy as GravecReasoningPolicy,
)
from .gravec.agents import ActionAgent, ChiefAgent, EffectAgent, ValueAgent
from .gravec.loop import reason_from_future_gravec
from .gravec.tools import ToolRegistry

# LLM interface（底层用 LiteLLM 抹平 100+ LLM 差异）
from .llm import llm_call

# Example problem specifications
from .specs import Game24Spec, GSM8KSpec, GSM8KNiHaixiaSpec, HumanEvalNiHaixiaSpec

__all__ = [
    # from .core (原版 v1)
    "ProblemSpec",
    "Workspace",
    "LocalCheckFail",
    "reason_from_future",
    # from .core_nhx (v2 确定性)
    "NiHaixiaSpec",
    "Observation",
    "ReasoningPolicy",
    "ValueScore",
    "GoalRevision",
    "CausalDiagnosis",
    "reason_from_future_nhx",
    # from .gravec (v3 混合实现)
    "GravecSpec",
    "GravecObservation",
    "GravecValueScore",
    "GravecGoalRevision",
    "GravecCausalDiagnosis",
    "GravecReasoningPolicy",
    "reason_from_future_gravec",
    "ValueAgent",
    "ActionAgent",
    "EffectAgent",
    "ChiefAgent",
    "ToolRegistry",
    # from .llm
    "llm_call",
    # from .specs
    "Game24Spec",
    "GSM8KSpec",
    "GSM8KNiHaixiaSpec",
    "HumanEvalNiHaixiaSpec",
]
