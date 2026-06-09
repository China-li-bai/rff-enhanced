"""从 MATH-500 选取 36 道题（按难度分层）。"""
import json
import random
from collections import Counter, defaultdict

random.seed(42)

data = [json.loads(line) for line in open("math500_raw.jsonl")]
by_level = defaultdict(list)
for d in data:
    by_level[d["level"]].append(d)

# Level 分布: L1:43, L2:90, L3:105, L4:128, L5:134
# 36题: L1:5, L2:7, L3:8, L4:8, L5:8
def pick(pool, n):
    return random.sample(pool, min(n, len(pool)))


selected = (
    pick(by_level[1], 5)
    + pick(by_level[2], 7)
    + pick(by_level[3], 8)
    + pick(by_level[4], 8)
    + pick(by_level[5], 8)
)
print(f"Selected: {len(selected)}")

out_path = "math500_36.jsonl"
with open(out_path, "w") as f:
    for i, item in enumerate(selected, 1):
        f.write(
            json.dumps(
                {
                    "seq": i,
                    "level": item["level"],
                    "subject": item["subject"],
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "answer": item["answer"],
                    "unique_id": item["unique_id"],
                }
            )
            + "\n"
        )
print(f"Saved to {out_path}")
print("Level dist:", dict(Counter([s["level"] for s in selected])))
print("Subject dist:", dict(Counter([s["subject"] for s in selected])))
