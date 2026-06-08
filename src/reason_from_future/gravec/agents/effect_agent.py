"""
E Agent — 多通道验效（倪海厦的"望闻问切"）

倪师有四种诊断方法，每种独立提供信息，互相印证：
  望（看）  → 代码领域：语法检查 / 数学领域：算术求值
  闻（听）  → 代码领域：运行测试 / 数学领域：数量级合理性
  问（问诊）→ 代码领域：静态分析 / 数学领域：依赖追溯
  切（脉诊）→ 代码领域：类型检查 / 数学领域：边界检查

当前代码只有一个 E 通道（LLM 自报），所有信息来自单一来源。
Agent E 引入多通道验证，每个通道独立给出观察结果，
最终综合所有通道的判断。

设计原则：
  - 每个通道独立返回 Observation
  - 综合时考虑通道的可靠度和一致性
  - 保留确定性 fallback（spec.evaluate_observation）
"""

from __future__ import annotations

from typing import Any

from ...core import Workspace
from ..models import Observation


# 四诊通道定义
CHANNEL_INSPECT = "inspect"    # 望：静态检查
CHANNEL_EXECUTE = "execute"    # 闻：动态执行
CHANNEL_ANALYZE = "analyze"    # 问：语义分析
CHANNEL_VALIDATE = "validate"  # 切：约束验证

ALL_CHANNELS = [CHANNEL_INSPECT, CHANNEL_EXECUTE, CHANNEL_ANALYZE, CHANNEL_VALIDATE]


class EffectAgent:
    """E Agent：多通道验效。

    与确定性 evaluate_observation 的区别：
    - 确定性：单通道公式计算
    - Agent：多通道望闻问切，综合判断

    Usage:
        agent = EffectAgent(llm_call=llm_call, tool_registry=registry)
        result = agent.evaluate(observation, state, goal, fallback=deterministic_score)
    """

    def __init__(
        self,
        llm_call: Any = None,
        model: str | None = None,
        tool_registry: Any = None,
    ):
        self._llm_call = llm_call
        self._model = model
        self._tool_registry = tool_registry

    def evaluate(
        self,
        observation: Observation,
        state: Workspace,
        goal: str,
        *,
        fallback: float = 0.0,
    ) -> float:
        """多通道验效。

        Args:
            observation: A 步骤产生的观察结果
            state: 当前工作台状态
            goal: 当前目标
            fallback: 确定性 fallback 分数

        Returns:
            效果分数（float），与确定性接口完全兼容
        """
        # 如果没有 LLM，直接 fallback
        if self._llm_call is None:
            return fallback

        # 收集多通道观察
        channel_observations = self._gather_channels(observation, state, goal)

        if not channel_observations:
            return fallback

        # 综合多通道判断
        return self._synthesize(channel_observations, state, goal, fallback)

    def _gather_channels(
        self,
        primary_observation: Observation,
        state: Workspace,
        goal: str,
    ) -> list[Observation]:
        """收集多通道观察结果。"""
        observations = [primary_observation]

        # 如果有工具注册中心，尝试各通道
        if self._tool_registry:
            for channel in ALL_CHANNELS:
                obs = self._run_channel(channel, state, goal)
                if obs is not None:
                    observations.append(obs)

        return observations

    def _run_channel(
        self,
        channel: str,
        state: Workspace,
        goal: str,
    ) -> Observation | None:
        """运行单个验证通道。"""
        # 通道到工具的映射
        channel_tools = {
            CHANNEL_INSPECT: "py_compile",
            CHANNEL_EXECUTE: "pytest",
            CHANNEL_ANALYZE: "pylint",
            CHANNEL_VALIDATE: "mypy",
        }

        tool_name = channel_tools.get(channel)
        if tool_name and self._tool_registry and self._tool_registry.has_tool(tool_name):
            try:
                result = self._tool_registry.call(tool_name)
                return Observation(
                    content=f"[{channel}] {result}",
                    data={"channel": channel, "result": result},
                    observation_type="neutral",
                    confidence=0.6,
                    channel=channel,
                )
            except Exception:
                return None

        return None

    def _synthesize(
        self,
        observations: list[Observation],
        state: Workspace,
        goal: str,
        fallback: float,
    ) -> float:
        """综合多通道观察，返回效果分数。"""
        if not observations:
            return fallback

        # 简单综合：加权平均
        # 主观察权重最高，通道观察辅助
        total_weight = 0.0
        weighted_score = 0.0

        for obs in observations:
            weight = obs.confidence
            # 根据观察类型映射分数
            if obs.observation_type == "improvement":
                score = 0.7
            elif obs.observation_type == "deterioration":
                score = -0.3
            elif obs.observation_type == "surprise":
                score = 0.1
            else:
                score = 0.3

            weighted_score += score * weight
            total_weight += weight

        if total_weight == 0:
            return fallback

        return weighted_score / total_weight
