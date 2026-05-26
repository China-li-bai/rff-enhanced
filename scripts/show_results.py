"""列出 GSM8K 200 题完整列表 + 基准测试结果。"""
import json

with open("gsm8k_200_questions.json", "r") as f:
    questions = json.load(f)

with open("benchmark_results.json", "r") as f:
    results = json.load(f)

nhx_results = results["nhx"]
correct_count = sum(1 for r in nhx_results if r["correct"])

print("=" * 100)
print("GSM8K 200 题完整列表 + 基准测试结果")
print(f"模型: {results['meta']['model']}")
print(f"准确率: {correct_count}/{len(nhx_results)} = {correct_count/len(nhx_results)*100:.1f}%")
print("=" * 100)

for i, q in enumerate(questions):
    r = nhx_results[i]
    status = "✅" if r["correct"] else "❌"
    ans = str(r.get("numeric_answer", r.get("answer", "N/A")))
    short_q = q["question"][:55] + ("..." if len(q["question"]) > 55 else "")
    print(f"  {status} {i+1:3d}. [GSM8K#{q['dataset_index']:4d}] 答案={ans:>12s} 标准={q['answer']:>10s} {r['elapsed']:5.1f}s | {short_q}")

wrong = [i for i, r in enumerate(nhx_results) if not r["correct"]]
print(f"\n{'='*100}")
print(f"错误题目详情 ({len(wrong)} 题):")
print(f"{'='*100}")
for i in wrong:
    q = questions[i]
    r = nhx_results[i]
    print(f"\n  ❌ 第 {i+1} 题 [GSM8K#{q['dataset_index']}]")
    print(f"  问题: {q['question'][:120]}...")
    print(f"  模型答案: {r.get('numeric_answer', r.get('answer', 'N/A'))}")
    print(f"  标准答案: {q['answer']}")
    print(f"  耗时: {r['elapsed']}s")
