"""筛选 MATH-500 36 题中数字友好的子集。"""
import json
import re

data = [json.loads(l) for l in open("math500_36.jsonl")]
print(f"Total: {len(data)}")


def is_numeric_friendly(ans: str) -> bool:
    a = ans.strip()
    # Reject: 根号、π、角度、文本、坐标、复合表达式
    if "\\sqrt" in a or "\\pi" in a or "^\\circ" in a or "\\text" in a or "\\left" in a:
        return False
    # 拒绝任何字母（除分数外）
    if re.search(r"[a-zA-Z]", a):
        return False
    return True


filtered = [d for d in data if is_numeric_friendly(d["answer"])]
print(f"Numeric-friendly: {len(filtered)}\n")

print("=== Skipped (non-numeric) ===")
for d in data:
    if not is_numeric_friendly(d["answer"]):
        print(f"  L{d['level']} {d['subject']:25s} ans={d['answer']}")

print("\n=== Kept (numeric) ===")
for d in filtered:
    print(f"  L{d['level']} {d['subject']:25s} ans={d['answer']}")

# Save filtered
with open("math500_36_numeric.jsonl", "w") as f:
    for d in filtered:
        f.write(json.dumps(d) + "\n")
print(f"\nSaved numeric subset: {len(filtered)} problems")
