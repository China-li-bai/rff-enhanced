"""调试 \frac 解析。"""
import re
import json

# 实际数据
data = [json.loads(l) for l in open("math500_raw.jsonl")]
for d in data:
    if "frac" in d["answer"] and "Evelyn" not in d["answer"] and "pi" not in d["answer"]:
        print(f"Raw answer: {d['answer']!r}")
        a = re.sub(r"\\!| ", "", d["answer"])
        print(f"After cleanup: {a!r}")
        m = re.match(r"^(-)?\\+frac\{(-?\d+)\}\{(-?\d+)\}$", a)
        print(f"Match: {m}")
        if m:
            print(f"  Groups: {m.groups()}")
        print()
        if data.index(d) > 50:
            break
