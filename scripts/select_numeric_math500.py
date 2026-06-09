"""从 500 题全集选 36 道数字题（用更宽松的过滤）。"""
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
    """尝试把 MATH 答案字符串解析为数字。返回 None 表示不可解析。"""
    a = ans.strip()
    # 拒绝: 根号、π、角度、文本、坐标、变量表达式
    if "\\sqrt" in a or "\\pi" in a or "^\\circ" in a or "\\text" in a or "\\left" in a:
        return None
    # 处理 \frac{a}{b} → a/b（必须在判断字母前处理，否则 frac 会被字母拒绝）
    m = re.match(r"^(-)?\\+frac\{(-?\d+)\}\{(-?\d+)\}$", a.replace(" ", ""))
    if m:
        sign, num, den = m.groups()
        try:
            v = float(num) / float(den)
            return -v if sign else v
        except Exception:
            return None
    # 拒绝含字母的（变量名如 p-q, Evelyn, 1\\frac 等等）
    if re.search(r"[a-zA-Z]", a):
        return None
    # 处理逗号分隔: 11,\! 111,\! 111,\! 100 → 11111111100
    a = re.sub(r"\\!| ", "", a)
    # 处理普通数字（含逗号）
    try:
        return float(a.replace(",", ""))
    except Exception:
        return None


# 验证过滤逻辑
for d in data[:5]:
    print(f"  L{d['level']} {d['subject']:25s} ans={d['answer']:30s} → {parse_numeric(d['answer'])}")

# 先看每级可解析的数量
print("\nPer level parseable count:")
for lv in sorted(by_level):
    parseable = [d for d in by_level[lv] if parse_numeric(d["answer"]) is not None]
    print(f"  L{lv}: {len(parseable)}/{len(by_level[lv])}")

# 目标: L1:5, L2:7, L3:8, L4:8, L5:8 = 36
# L1 总共只有 43 题，能解析的更少


def pick_parseable(pool, n):
    avail = [d for d in pool if parse_numeric(d["answer"]) is not None]
    if len(avail) <= n:
        return avail
    return random.sample(avail, n)


# 缺额时跨级补
selected = []
targets = {1: 5, 2: 7, 3: 8, 4: 8, 5: 8}
for lv, n in targets.items():
    picked = pick_parseable(by_level[lv], n)
    selected.extend(picked)
    print(f"\nL{lv}: picked {len(picked)} (target {n})")
    if len(picked) < n:
        print(f"  ⚠️  缺 {n - len(picked)} 道")

# 总数不够时从剩余 parseable 池里补
total = len(selected)
if total < 36:
    used_ids = {d["unique_id"] for d in selected}
    extras = [
        d for d in data
        if parse_numeric(d["answer"]) is not None and d["unique_id"] not in used_ids
    ]
    random.shuffle(extras)
    need = 36 - total
    selected.extend(extras[:need])
    print(f"\n补足 {need} 道，共 {len(selected)}")

# 重新编号并保存
print(f"\nFinal: {len(selected)} problems")
out_path = "math500_36_numeric.jsonl"
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
