"""调试 seq=55 的分数题。"""
import json
import time
import sys

sys.path.insert(0, "src")
from reason_from_future.core_nhx import reason_from_future_nhx
from reason_from_future.specs.math500_nhx import MATHNiHaixiaSpec

with open("math500_100_numeric.jsonl") as f:
    lines = f.readlines()

for line in lines:
    d = json.loads(line)
    if d.get("seq") == 55:
        break

gold = float(d["gold_numeric"])
print(f"seq=55 L{d['level']} {d['subject']} gold={gold}")
print(f"问题: {d['problem'][:200]}")

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
