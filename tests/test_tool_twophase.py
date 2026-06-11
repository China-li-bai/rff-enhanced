"""测试完整的 tool-calling 两阶段流程。"""
import sys
sys.path.insert(0, "src")
from reason_from_future.executors.tools import llm_call_with_tools, SympyToolHandler

handler = SympyToolHandler()

# 测试1: force_tool_use=True, 两阶段
print("=== Test: force_tool_use=True, 两阶段 ===")
result = llm_call_with_tools(
    messages=[{"role": "user", "content": "Calculate 11/36 as a fraction using sympy_calculate. Then output JSON: {\"var\": \"prob\", \"expr\": \"11/36\", \"value\": 0.3056}"}],
    tool_handler=handler,
    force_tool_use=True,
    max_tool_rounds=3,
    verbose=True,
    final_prompt='Output a JSON with keys var, expr, value using the calculation result.',
)
print(f"\nResult content: {repr(result['content'])}")
print(f"Tool calls: {result['tool_calls_count']}")

# 测试2: force_tool_use=False, 单阶段
print("\n=== Test: force_tool_use=False, 单阶段 ===")
result2 = llm_call_with_tools(
    messages=[{"role": "user", "content": "Calculate 11/36 as a fraction using sympy_calculate. Then output JSON: {\"var\": \"prob\", \"expr\": \"11/36\", \"value\": 0.3056}"}],
    tool_handler=handler,
    force_tool_use=False,
    max_tool_rounds=3,
    verbose=True,
)
print(f"\nResult content: {repr(result2['content'])}")
print(f"Tool calls: {result2['tool_calls_count']}")
