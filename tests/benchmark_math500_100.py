"""
GRAVEC v2 — MATH-500 100题基准测试

使用 Agnes 2.0 Flash 模型，对 MATH-500 分层选取的 100 道数字友好题，
运行 GRAVEC v2 (reason_from_future_nhx) 进行测试。

运行方式：
    cd /Users/mac/WorkBuddy/Claw/rff-enhanced
    PYTHONPATH=src uv run python tests/benchmark_math500_100.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reason_from_future.core_nhx import reason_from_future_nhx
from reason_from_future.specs.math500_nhx import MATHNiHaixiaSpec


def run_single(problem_data: dict, verbose: bool = False, timeout: int = 180, use_tools: bool = False) -> dict:
    """运行单题 GRAVEC v2。"""
    import threading

    spec = MATHNiHaixiaSpec(problem_data)
    start = time.time()
    result_holder: dict = {"done": False}

    def _worker():
        try:
            answer = reason_from_future_nhx(
                problem=problem_data["problem"],
                spec=spec,
                max_iters=10,
                verbose=verbose,
                require_gold=False,
                min_iters=2,
                use_tools=use_tools,
            )
            result_holder["answer"] = answer
        except Exception as e:
            result_holder["error"] = str(e)[:120]
        finally:
            result_holder["done"] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    elapsed = time.time() - start
    gold = float(problem_data["gold_numeric"])

    if not result_holder["done"]:
        return {
            "seq": problem_data["seq"],
            "level": problem_data["level"],
            "subject": problem_data["subject"],
            "question": problem_data["problem"][:80] + "...",
            "answer": None,
            "numeric_answer": None,
            "gold": gold,
            "gold_latex": problem_data["answer"],
            "correct": False,
            "elapsed": round(elapsed, 2),
            "error": f"超时 ({timeout}s)",
        }

    if "error" in result_holder:
        return {
            "seq": problem_data["seq"],
            "level": problem_data["level"],
            "subject": problem_data["subject"],
            "question": problem_data["problem"][:80] + "...",
            "answer": None,
            "numeric_answer": None,
            "gold": gold,
            "gold_latex": problem_data["answer"],
            "correct": False,
            "elapsed": round(elapsed, 2),
            "error": result_holder["error"],
        }

    answer = result_holder["answer"]
    try:
        numeric_answer = float(str(answer).replace(",", "").strip())
        if abs(gold) < 1e-9:
            correct = abs(numeric_answer) < 1e-6
        else:
            correct = abs(numeric_answer - gold) / abs(gold) < 1e-3
    except (ValueError, TypeError):
        correct = False
        numeric_answer = None
    return {
        "seq": problem_data["seq"],
        "level": problem_data["level"],
        "subject": problem_data["subject"],
        "question": problem_data["problem"][:80] + "...",
        "answer": answer,
        "numeric_answer": numeric_answer,
        "gold": gold,
        "gold_latex": problem_data["answer"],
        "correct": correct,
        "elapsed": round(elapsed, 2),
        "error": None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-tools", action="store_true", help="启用 SymPy tool-calling 预计算")
    parser.add_argument("--limit", type=int, default=0, help="限制测试题数（0=全部）")
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), "..", "math500_100_numeric.jsonl")
    with open(data_path) as f:
        selected = [json.loads(line) for line in f]

    if args.limit > 0:
        selected = selected[:args.limit]

    tools_label = " + SymPy Tool-Calling" if args.use_tools else ""
    print(f"\n{'='*70}")
    print(f"GRAVEC v2 + Agnes 2.0 Flash{tools_label} — MATH-500 基准测试")
    print(f"模型: openai/agnes-2.0-flash")
    print(f"题目: {len(selected)} 道")
    print(f"引擎: reason_from_future_nhx (GRAVEC 六步曲)")
    print(f"Tool-Calling: {'ON' if args.use_tools else 'OFF'}")
    print(f"最大迭代: 10 | 最少迭代: 2 | 超时: 180s/题")
    print(f"容差: 1e-3 相对误差")
    print(f"{'='*70}\n")

    results = []
    correct_count = 0
    total_time = 0.0

    for i, problem_data in enumerate(selected):
        print(
            f"[{i+1:3d}/100] L{problem_data['level']} {problem_data['subject']:25s} "
            f"gold={problem_data['answer'][:15]:15s}",
            end=" ",
            flush=True,
        )
        result = run_single(problem_data, verbose=False, use_tools=args.use_tools)
        results.append(result)

        status = "OK" if result["correct"] else "FAIL"
        if result["correct"]:
            correct_count += 1
        total_time += result["elapsed"]

        answer_str = str(result.get("numeric_answer", result.get("answer", "ERR")))
        err_str = f" | err: {result['error'][:50]}" if result.get("error") else ""
        print(f"{status} got={answer_str:>12s} gold={result['gold']:>12g} ({result['elapsed']:.1f}s){err_str}")

        # 实时准确率
        if (i + 1) % 10 == 0:
            print(f"  >>> 进度 {i+1}/100 | 当前准确率: {correct_count/(i+1)*100:.1f}%")

        time.sleep(0.3)

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
    by_level = {}
    for r in results:
        by_level.setdefault(r["level"], []).append(r)

    print(f"\n  按难度 (L1-L5):")
    for lv in sorted(by_level):
        subset = by_level[lv]
        if subset:
            c = sum(1 for r in subset if r["correct"])
            avg_t = sum(r["elapsed"] for r in subset) / len(subset)
            print(f"    L{lv}: {c}/{len(subset)} ({c/len(subset)*100:.0f}%)  avg={avg_t:.1f}s/题")

    # 按学科统计
    print(f"\n  按学科:")
    by_subject = {}
    for r in results:
        by_subject.setdefault(r["subject"], []).append(r)
    for subj, subset in sorted(by_subject.items()):
        c = sum(1 for r in subset if r["correct"])
        print(f"    {subj:30s}: {c}/{len(subset)} ({c/len(subset)*100:.0f}%)")

    # 错误详情
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n错误题目详情 ({len(errors)} 道):")
        for r in errors:
            answer_str = str(r.get("numeric_answer", r.get("answer", "ERR")))
            err_str = f" | {r['error'][:60]}" if r.get("error") else ""
            print(f"  L{r['level']} seq={r['seq']:3d} {r['subject']:25s} "
                  f"got={answer_str:>12s} gold={r['gold']:>12g}{err_str}")

    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "..", "math500_100_gravec_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "openai/agnes-2.0-flash",
            "engine": "GRAVEC v2 (reason_from_future_nhx)",
            "dataset": "MATH-500 (100 numeric subset)",
            "total": len(results),
            "correct": correct_count,
            "accuracy": correct_count / len(results) * 100,
            "total_time": round(total_time, 1),
            "avg_time": round(total_time / len(results), 1),
            "by_level": {
                str(lv): {
                    "total": len(by_level[lv]),
                    "correct": sum(1 for r in by_level[lv] if r["correct"]),
                    "accuracy": sum(1 for r in by_level[lv] if r["correct"]) / max(len(by_level[lv]), 1) * 100,
                }
                for lv in sorted(by_level)
            },
            "by_subject": {
                s: {
                    "total": len(sub),
                    "correct": sum(1 for r in sub if r["correct"]),
                    "accuracy": sum(1 for r in sub if r["correct"]) / max(len(sub), 1) * 100,
                }
                for s, sub in sorted(by_subject.items())
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
