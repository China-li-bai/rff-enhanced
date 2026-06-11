"""调试预计算模式：看 sympy 工具调用的结果。"""
import sys
sys.path.insert(0, "src")
from reason_from_future.executors.tools import llm_call_with_tools, SympyToolHandler

handler = SympyToolHandler()

# 测试：计算概率 11/36
result = llm_call_with_tools(
    messages=[{"role": "user", "content": "Two fair 6-sided dice are thrown. What is the probability that the product is a multiple of 5? Use sympy_calculate to compute."}],
    tool_handler=handler,
    max_tool_rounds=3,
    verbose=True,
)
print(f"\n=== Final content: {repr(result['content'])} ===")
print(f"Tool calls: {result['tool_calls_count']}")
