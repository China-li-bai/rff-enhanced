"""
GRAVEC + Agnes AI — GSM8K 36题基准测试

使用 Agnes 2.0 Flash 模型，从 GSM8K 200 题中均匀选取 36 道题，
覆盖不同难度级别，运行 GRAVEC v2 (reason_from_future_nhx) 进行测试。

运行方式：
    cd /Users/mac/WorkBuddy/Claw/rff-enhanced
    PYTHONPATH=src uv run python tests/benchmark_gravec_agnes.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reason_from_future.core_nhx import reason_from_future_nhx
from reason_from_future.specs.gsm8k_nhx import GSM8KNiHaixiaSpec


def select_36_questions(data: list[dict]) -> list[dict]:
    """从 200 题中均匀选取 36 道，覆盖不同难度。

    策略：按答案数值大小分层（简单/中等/困难），每层均匀采样。
    - 简单 (answer <= 20): 12 题
    - 中等 (20 < answer <= 200): 12 题
    - 困难 (answer > 200): 12 题
    """
    def parse_answer(a: str) -> float:
        try:
            return float(a.replace(",", ""))
        except ValueError:
            return 0.0

    easy = [d for d in data if parse_answer(d["answer"]) <= 20]
    medium = [d for d in data if 20 < parse_answer(d["answer"]) <= 200]
    hard = [d for d in data if parse_answer(d["answer"]) > 200]

    # 均匀采样，步长 = len(pool) // 12
    def sample(pool: list[dict], n: int) -> list[dict]:
        if len(pool) <= n:
            return pool
        step = len(pool) / n
        return [pool[int(i * step)] for i in range(n)]

    selected = sample(easy, 12) + sample(medium, 12) + sample(hard, 12)
    return selected


def run_single(problem_data: dict, max_iters: int = 12, verbose: bool = False) -> dict:
    """运行单题 GRAVEC v2。"""
    spec = GSM8KNiHaixiaSpec(problem_data)
    start = time.time()
    try:
        answer = reason_from_future_nhx(
            problem=problem_data["question"],
            spec=spec,
            max_iters=max_iters,
            verbose=verbose,
            require_gold=False,
            min_iters=2,
        )
        elapsed = time.time() - start
        gold = float(problem_data["answer"].replace(",", ""))
        try:
            numeric_answer = float(answer.replace(",", ""))
            correct = abs(numeric_answer - gold) < 1e-5
        except (ValueError, TypeError):
            correct = False
            numeric_answer = None
        return {
            "seq": problem_data["seq"],
            "question": problem_data["question"][:60] + "...",
            "answer": answer,
            "numeric_answer": numeric_answer,
            "gold": gold,
            "correct": correct,
            "elapsed": round(elapsed, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "seq": problem_data["seq"],
            "question": problem_data["question"][:60] + "...",
            "answer": None,
            "numeric_answer": None,
            "gold": float(problem_data["answer"].replace(",", "")),
            "correct": False,
            "elapsed": round(time.time() - start, 2),
            "error": str(e)[:100],
        }


def main():
    # 加载 200 题数据集
    data_path = os.path.join(os.path.dirname(__file__), "..", "gsm8k_200_questions.json")
    with open(data_path) as f:
        all_data = json.load(f)

    # 选取 36 题
    selected = select_36_questions(all_data)

    print(f"\n{'='*70}")
    print(f"GRAVEC v2 + Agnes 2.0 Flash — GSM8K 36题基准测试")
    print(f"模型: openai/agnes-2.0-flash")
    print(f"题目: {len(selected)} 道 (简单12 + 中等12 + 困难12)")
    print(f"最大迭代: 12 | 最少迭代: 2")
    print(f"{'='*70}\n")

    results = []
    correct_count = 0
    total_time = 0

    for i, problem_data in enumerate(selected):
        print(f"[{i+1:2d}/36] seq={problem_data['seq']}, answer={problem_data['answer']}...", end=" ", flush=True)
        result = run_single(problem_data, max_iters=12, verbose=False)
        results.append(result)

        status = "OK" if result["correct"] else "FAIL"
        if result["correct"]:
            correct_count += 1
        total_time += result["elapsed"]

        answer_str = str(result.get("numeric_answer", result.get("answer", "ERR")))
        err_str = f" | err: {result['error'][:40]}" if result.get("error") else ""
        print(f"{status} got={answer_str} gold={result['gold']} ({result['elapsed']:.1f}s){err_str}")

        # 避免速率限制
        time.sleep(0.5)

    # 汇总
    print(f"\n{'='*70}")
    print(f"测试结果汇总")
    print(f"{'='*70}")
    print(f"  总题数:   {len(results)}")
    print(f"  正确数:   {correct_count}")
    print(f"  准确率:   {correct_count/len(results)*100:.1f}%")
    print(f"  总耗时:   {total_time:.1f}s")
    print(f"  平均耗时: {total_time/len(results):.1f}s/题")

    # 按难度分类统计
    easy_results = [r for r in results if r["gold"] <= 20]
    medium_results = [r for r in results if 20 < r["gold"] <= 200]
    hard_results = [r for r in results if r["gold"] > 200]

    for label, subset in [("简单(<=20)", easy_results), ("中等(20-200)", medium_results), ("困难(>200)", hard_results)]:
        if subset:
            c = sum(1 for r in subset if r["correct"])
            print(f"  {label}: {c}/{len(subset)} ({c/len(subset)*100:.0f}%)")

    # 错误详情
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n错误题目详情:")
        for r in errors:
            answer_str = str(r.get("numeric_answer", r.get("answer", "ERR")))
            err_str = f" | {r['error'][:50]}" if r.get("error") else ""
            print(f"  seq={r['seq']} got={answer_str} gold={r['gold']}{err_str}")

    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "..", "gravec_agnes_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "openai/agnes-2.0-flash",
            "engine": "GRAVEC v2 (reason_from_future_nhx)",
            "total": len(results),
            "correct": correct_count,
            "accuracy": correct_count / len(results) * 100,
            "total_time": round(total_time, 1),
            "avg_time": round(total_time / len(results), 1),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
