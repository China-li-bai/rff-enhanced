#!/usr/bin/env python3
"""测试 GSM8KNiHaixiaSpec 确定性方法"""
from reason_from_future.specs.gsm8k_nhx import GSM8KNiHaixiaSpec
from reason_from_future.core import Workspace

spec = GSM8KNiHaixiaSpec({
    "question": "Janet has 5 apples. She buys 3 more. How many does she have?",
    "answer": "8"
})
print("GSM8K gold:", spec.gold_numeric_answer)
print("Constraints:", spec._extracted_constraints)

state = Workspace({"apples": 5, "bought": 3, "total": 8})

# 测试价值判断
vs = spec.evaluate_step_value(state, "total", "total")
print(f"Value(total): score={vs.score}, is_primary={vs.is_primary}, reason={vs.reason}")

vs2 = spec.evaluate_step_value(state, "apples", "total")
print(f"Value(apples): score={vs2.score}, is_primary={vs2.is_primary}, reason={vs2.reason}")

# 测试行动执行
obs = spec.execute_action(state, "total", "total")
print(f"Action: type={obs.observation_type}, confidence={obs.confidence}")
print(f"Action data: {obs.data}")

# 测试验效反馈
eff = spec.evaluate_observation(obs, state, "total")
print(f"Effect score: {eff}")

# 测试因果诊断
diag = spec.diagnose_cause(state, "total", obs, "total")
print(f"Diagnosis: type={diag.failure_type}, confidence={diag.confidence}")

# 测试约束违反场景
state_bad = Workspace({"apples": 5, "bought": 3, "total": -8})
obs_bad = spec.execute_action(state_bad, "total", "total")
print(f"\nBad value action: type={obs_bad.observation_type}, content={obs_bad.content}")

vs_bad = spec.evaluate_step_value(state_bad, "total", "total")
print(f"Bad value score: {vs_bad.score}, reason={vs_bad.reason}")

print("\nAll deterministic methods work without LLM!")
