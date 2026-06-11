"""多题测试：验证 tool-calling 预计算模式。"""
import json
import time
import sys

sys.path.insert(0, "src")
from reason_from_future.core_nhx import reason_from_future_nhx
from reason_from_future.specs.math500_nhx import MATHNiHaixiaSpec

test_seqs = [55, 62, 79, 46, 3, 15, 27, 88]  # 8道题

with open("math500_100_numeric.jsonl") as f:
    lines = f.readlines()

results = []
for seq in test_seqs:
    for line in lines:
        d = json.loads(line)
        if d.get("seq") == seq:
            break
    else:
        print(f"seq={seq} not found")
        continue

    gold = float(d["gold_numeric"])
    spec = MATHNiHaixiaSpec(d)

    start = time.time()
    try:
        answer = reason_from_future_nhx(
            problem=d["problem"],
            spec=spec,
            max_iters=8,
            verbose=False,
            require_gold=False,
            min_iters=2,
            use_tools=True,
        )
        elapsed = time.time() - start
        try:
            numeric = float(str(answer).replace(",", "").strip())
            correct = abs(numeric - gold) / abs(gold) < 1e-3 if abs(gold) > 1e-9 else abs(numeric) < 1e-6
        except (ValueError, TypeError):
            correct = False
            numeric = None

        status = "OK" if correct else "FAIL"
        results.append(correct)
        print(f"seq={seq} {status} got={numeric} gold={gold} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start
        results.append(False)
        print(f"seq={seq} ERROR: {e} ({elapsed:.1f}s)")

print(f"\n=== 准确率: {sum(results)}/{len(results)} = {sum(results)/len(results)*100:.1f}% ===")
