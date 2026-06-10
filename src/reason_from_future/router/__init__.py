"""
Strategy Router — 以果决其行的"果"层

PRISM (Plan before Solving) 式策略路由：
  在执行 GRAVEC 之前，先判断"该用什么策略到达目标"。

这就是"以果决其行"的工程化实现——
  果 = 策略目标（用哪种推理路径）
  行 = 执行路径（CoT / GRAVEC / ToT+Vote / Deep GRAVEC）

三层漏斗路由（2026 业界共识）：
  L1 正则/关键词  (<1ms,   60-80% 请求) → 快速分类
  L2 启发式分析  (<10ms,  15-25% 请求) → 复杂度估算
  L3 LLM FC      (1-2s,   5-15%  请求) → 语义深度判断

与 GRAVEC 六步曲的映射：
  Router = G(以果)的前置增强
  在"要达到目标"之前，先确定"该走哪条路到达目标"
"""

from .classifier import ProblemClassifier
from .strategy import Strategy, StrategyDecision
from .policy import RoutingPolicy

__all__ = ["ProblemClassifier", "Strategy", "StrategyDecision", "RoutingPolicy"]
