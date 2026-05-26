"""Reason-from-Future – Near-Executable Reference Implementation.

This package contains a domain-agnostic controller implementing the
Reason-from-Future algorithm with example domain specs.  See
`PLANS/inital_setup.md` for the original design rationale.

倪海厦「以果决其行」增强版：在原版 RFF 的 G→R→C 三步曲基础上，
新增 A（行动执行）、V（价值判断）、E（验效反馈）三步，
以及果行共变（目标随反馈修正）机制，形成完整的 G→R→A→V→E→C 六步曲。
"""

from __future__ import annotations

# Core abstractions and controller (原版)
from .core import ProblemSpec, Workspace, LocalCheckFail, reason_from_future

# 倪海厦增强核心
from .core_nhx import (
    NiHaixiaSpec,
    Observation,
    ValueScore,
    GoalRevision,
    CausalDiagnosis,
    reason_from_future_nhx,
)

# LLM interface（底层用 LiteLLM 抹平 100+ LLM 差异）
from .llm import llm_call

# Example problem specifications
from .specs import Game24Spec, GSM8KSpec, GSM8KNiHaixiaSpec

__all__ = [
    # from .core (原版)
    "ProblemSpec",
    "Workspace",
    "LocalCheckFail",
    "reason_from_future",
    # from .core_nhx (倪海厦增强)
    "NiHaixiaSpec",
    "Observation",
    "ValueScore",
    "GoalRevision",
    "CausalDiagnosis",
    "reason_from_future_nhx",
    # from .llm
    "llm_call",
    # from .specs
    "Game24Spec",
    "GSM8KSpec",
    "GSM8KNiHaixiaSpec",
]
