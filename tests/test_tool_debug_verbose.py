"""调试：单题 verbose 测试，看 computed_values 是否正确。"""
import json
import time
import sys

sys.path.insert(0, "src")
from reason_from_future.core_nhx import reason_from_future_nhx
from reason_from_future.specs.math500_nhx import MATHNiHaixiaSpec

with open("math500_100_numeric.jsonl") as f:
    lines = f.readlines()

d = json.loads(lines[55])  # seq=55
gold = float(d["gold_numeric"])
print(f"seq=55 gold={gold}")

spec = MATHNiHaixiaSpec(d)
start = time.time()
try:
    answer = reason_from_future_nhx(
        problem=d["problem"],
        spec=spec,
        max_iters=5,
        verbose=True,
        require_gold=False,
        min_iters=1,
        use_tools=True,
    )
    elapsed = time.time() - start
    try:
        numeric = float(str(answer).replace(",", "").strip())
        correct = abs(numeric - gold) / abs(gold) < 1e-3
    except:
        correct = False
        numeric = None
    print(f"\n=== {'OK' if correct else 'FAIL'} got={numeric} gold={gold} ({elapsed:.1f}s) ===")
except Exception as e:
    elapsed = time.time() - start
    print(f"\n=== ERROR: {e} ({elapsed:.1f}s) ===")
