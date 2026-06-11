"""单题测试：验证 tool-calling 模式修复分数反转问题。"""
import json
import time
import sys

sys.path.insert(0, "src")
from reason_from_future.core_nhx import reason_from_future_nhx
from reason_from_future.specs.math500_nhx import MATHNiHaixiaSpec

# 测试之前出错的分数反转题
test_seqs = [46, 55, 62, 79]  # gold=10/11, 11/36, 1/3, 1/4

with open("math500_100_numeric.jsonl") as f:
    lines = f.readlines()

for seq in test_seqs:
    for line in lines:
        d = json.loads(line)
        if d.get("seq") == seq:
            break
    else:
        print(f"seq={seq} not found")
        continue

    gold = float(d["gold_numeric"])
    print(f"\n{'='*60}")
    print(f"seq={seq} L{d['level']} {d['subject']} gold={gold}")
    print(f"问题: {d['problem'][:150]}...")

    spec = MATHNiHaixiaSpec(d)

    # Tool-calling 模式
    start = time.time()
    try:
        answer = reason_from_future_nhx(
            problem=d["problem"],
            spec=spec,
            max_iters=10,
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
        print(f"[tools=True]  {status} got={numeric} gold={gold} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start
        print(f"[tools=True]  ERROR: {e} ({elapsed:.1f}s)")
