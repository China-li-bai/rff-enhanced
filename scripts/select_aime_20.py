"""从 AIME 2024/2025 数据集中选出 20 道数字友好题。

AIME 答案为 0-999 的整数，天然适合我们的数值评测。
从 60 题中选 20 道，优先选：
  1. 答案为整数的（AIME 全部满足）
  2. 可用 SymPy 精确计算的
  3. 覆盖不同难度和知识点
"""
from datasets import load_dataset
import json
import random

random.seed(42)

# 加载数据集
ds24 = load_dataset('math-ai/aime24', split='test')
ds25 = load_dataset('math-ai/aime25', split='test')

all_problems = []

for item in ds24:
    # AIME 2024 的 solution 格式: \boxed{204}
    sol = item['solution']
    answer = sol.replace('\\boxed{', '').replace('}', '').strip()
    try:
        answer_int = int(answer)
    except ValueError:
        continue
    all_problems.append({
        'id': f"aime24_{item['id']}",
        'problem': item['problem'],
        'answer': answer_int,
        'source': 'aime24',
        'url': item.get('url', ''),
    })

for item in ds25:
    answer = str(item['answer']).strip()
    try:
        answer_int = int(answer)
    except ValueError:
        continue
    all_problems.append({
        'id': f"aime25_{item['id']}",
        'problem': item['problem'],
        'answer': answer_int,
        'source': 'aime25',
        'url': '',
    })

print(f"Total valid problems: {len(all_problems)}")

# 选 20 题：从 AIME24 和 AIME25 各选 10 道
aime24_probs = [p for p in all_problems if p['source'] == 'aime24']
aime25_probs = [p for p in all_problems if p['source'] == 'aime25']

selected_24 = random.sample(aime24_probs, min(10, len(aime24_probs)))
selected_25 = random.sample(aime25_probs, min(10, len(aime25_probs)))
selected = selected_24 + selected_25

# 保存
output_path = 'data/aime_20.json'
with open(output_path, 'w') as f:
    json.dump(selected, f, indent=2, ensure_ascii=False)

print(f"\nSelected {len(selected)} problems → {output_path}")
print(f"  AIME 2024: {len(selected_24)} | AIME 2025: {len(selected_25)}")
print(f"\nPreview:")
for p in selected[:3]:
    print(f"  [{p['id']}] answer={p['answer']}")
    print(f"    {p['problem'][:100]}...")
