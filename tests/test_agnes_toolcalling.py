"""测试 Agnes AI 的 tool-calling 支持。"""
import sys
sys.path.insert(0, "src")
import litellm
from reason_from_future.llm import DEFAULT_MODEL, DEFAULT_API_KEY, DEFAULT_API_BASE
from reason_from_future.executors.tools import SYMPY_TOOL_SCHEMA

# 测试1: tool_choice=auto
print("=== Test 1: tool_choice=auto ===")
resp = litellm.completion(
    model=DEFAULT_MODEL,
    api_key=DEFAULT_API_KEY,
    api_base=DEFAULT_API_BASE,
    messages=[{"role": "user", "content": "Please use the sympy_calculate tool to compute 11/36 as a fraction."}],
    tools=[SYMPY_TOOL_SCHEMA],
    tool_choice="auto",
    timeout=30,
)
msg = resp.choices[0].message
print(f"Content: {repr(msg.content)}")
print(f"Tool calls: {msg.tool_calls}")
if msg.tool_calls:
    for tc in msg.tool_calls:
        print(f"  Tool: {tc.function.name}({tc.function.arguments})")

# 测试2: 不指定 tool_choice
print("\n=== Test 2: no tool_choice ===")
resp2 = litellm.completion(
    model=DEFAULT_MODEL,
    api_key=DEFAULT_API_KEY,
    api_base=DEFAULT_API_BASE,
    messages=[{"role": "user", "content": "Use sympy_calculate to compute Rational(10, 11)."}],
    tools=[SYMPY_TOOL_SCHEMA],
    timeout=30,
)
msg2 = resp2.choices[0].message
print(f"Content: {repr(msg2.content)}")
print(f"Tool calls: {msg2.tool_calls}")
if msg2.tool_calls:
    for tc in msg2.tool_calls:
        print(f"  Tool: {tc.function.name}({tc.function.arguments})")

# 测试3: tool_choice=required
print("\n=== Test 3: tool_choice=required ===")
try:
    resp3 = litellm.completion(
        model=DEFAULT_MODEL,
        api_key=DEFAULT_API_KEY,
        api_base=DEFAULT_API_BASE,
        messages=[{"role": "user", "content": "Compute 11/36"}],
        tools=[SYMPY_TOOL_SCHEMA],
        tool_choice="required",
        timeout=30,
    )
    msg3 = resp3.choices[0].message
    print(f"Content: {repr(msg3.content)}")
    print(f"Tool calls: {msg3.tool_calls}")
except Exception as e:
    print(f"Error: {e}")
