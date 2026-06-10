"""从 500 题全集选 100 道数字题（分层抽样）。"""
import json
import re
import random
from collections import Counter, defaultdict

random.seed(42)

data = [json.loads(line) for line in open("math500_raw.jsonl")]
by_level = defaultdict(list)
for d in data:
    by_level[d["level"]].append(d)

print("Level distribution:")
for lv in sorted(by_level):
    print(f"  L{lv}: {len(by_level[lv])}")


def parse_numeric(ans: str) -> float | None:
    """尝试把 MATH 答案字符串解析为数字。"""
    a = ans.strip()
    if "\\sqrt" in a or "\\pi" in a or "^\\circ" in a or "\\text" in a or "\\left" in a:
        return None
    if re.search(r"\d+_\d+", a):
        return None
    m = re.match(r"^(-)?\\+frac\{(-?\d+)\}\{(-?\d+)\}$", a.replace(" ", ""))
    if m:
        sign, num, den = m.groups()
        try:
            v = float(num) / float(den)
            return -v if sign else v
        except Exception:
            return None
    if re.search(r"[a-zA-Z]", a):
        return None
    a = re.sub(r"\\!| ", "", a)
    try:
        return float(a.replace(",", ""))
    except Exception:
        return None


# 统计每级可解析数
print("\nPer level parseable count:")
for lv in sorted(by_level):
    parseable = [d for d in by_level[lv] if parse_numeric(d["answer"]) is not None]
    print(f"  L{lv}: {len(parseable)}/{len(by_level[lv])}")

# 目标: 100 题，按比例分配
total_parseable = sum(
    len([d for d in by_level[lv] if parse_numeric(d["answer"]) is not None])
    for lv in sorted(by_level)
)
print(f"\nTotal parseable: {total_parseable}")

# 按比例分配，最少每级 5 道
targets = {}
remaining = 100
for lv in sorted(by_level):
    parseable = [d for d in by_level[lv] if parse_numeric(d["answer"]) is not None]
    alloc = max(5, round(len(parseable) / total_parseable * 100))
    targets[lv] = min(alloc, len(parseable))
    remaining -= targets[lv]

# 调整到恰好 100
if remaining > 0:
    # 给最多的级补
    max_lv = max(targets, key=lambda lv: len([d for d in by_level[lv] if parse_numeric(d["answer"]) is not None]) - targets[lv])
    targets[max_lv] += remaining
elif remaining < 0:
    # 从最多的级减
    max_lv = max(targets, key=targets.get)
    targets[max_lv] += remaining

print(f"\nTargets: {targets} = {sum(targets.values())}")


def pick_parseable(pool, n):
    avail = [d for d in pool if parse_numeric(d["answer"]) is not None]
    if len(avail) <= n:
        return avail
    return random.sample(avail, n)


selected = []
for lv, n in sorted(targets.items()):
    picked = pick_parseable(by_level[lv], n)
    selected.extend(picked)
    print(f"L{lv}: picked {len(picked)} (target {n})")

# 补足到 100
if len(selected) < 100:
    used_ids = {d["unique_id"] for d in selected}
    extras = [
        d for d in data
        if parse_numeric(d["answer"]) is not None and d["unique_id"] not in used_ids
    ]
    random.shuffle(extras)
    need = 100 - len(selected)
    selected.extend(extras[:need])
    print(f"\n补足 {need} 道，共 {len(selected)}")

# 重新编号并保存
print(f"\nFinal: {len(selected)} problems")
out_path = "math500_100_numeric.jsonl"
with open(out_path, "w") as f:
    for i, item in enumerate(selected, 1):
        gold = parse_numeric(item["answer"])
        f.write(
            json.dumps(
                {
                    "seq": i,
                    "level": item["level"],
                    "subject": item["subject"],
                    "problem": item["problem"],
                    "solution": item["solution"],
                    "answer": item["answer"],
                    "gold_numeric": gold,
                    "unique_id": item["unique_id"],
                }
            )
            + "\n"
        )
print(f"Saved to {out_path}")
print("Level dist:", dict(Counter([s["level"] for s in selected])))
print("Subject dist:", dict(Counter([s["subject"] for s in selected])))
