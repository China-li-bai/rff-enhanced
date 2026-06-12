"""
GRAVEC v2 — AIME 20题基准测试

AIME (American Invitational Mathematics Examination) 是美国数学邀请赛，
答案为 0-999 的整数，是当前前沿推理模型的标准分水岭。

运行方式：
    cd /Users/mac/WorkBuddy/Claw/rff-enhanced
    PYTHONPATH=src uv run python tests/benchmark_aime_20.py
    PYTHONPATH=src uv run python tests/benchmark_aime_20.py --use-tools
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reason_from_future.core_nhx import reason_from_future_nhx
from reason_from_future.specs.aime_nhx import AIMENiHaixiaSpec


def run_single(
    problem_data: dict,
    verbose: bool = False,
    timeout: int = 300,
    use_tools: bool = False,
) -> dict:
    """运行单题 GRAVEC v2。AIME 题更难，给 300s 超时。"""
    import threading

    spec = AIMENiHaixiaSpec(problem_data)
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
            result_holder["error"] = str(e)[:200]
        finally:
            result_holder["done"] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    elapsed = time.time() - start
    gold = int(problem_data["answer"])

    if not result_holder["done"]:
        return {
            "id": problem_data["id"],
            "source": problem_data["source"],
            "question": problem_data["problem"][:100] + "...",
            "answer": None,
            "numeric_answer": None,
            "gold": gold,
            "correct": False,
            "elapsed": round(elapsed, 2),
            "error": f"超时 ({timeout}s)",
        }

    if "error" in result_holder:
        return {
            "id": problem_data["id"],
            "source": problem_data["source"],
            "question": problem_data["problem"][:100] + "...",
            "answer": None,
            "numeric_answer": None,
            "gold": gold,
            "correct": False,
            "elapsed": round(elapsed, 2),
            "error": result_holder["error"],
        }

    answer = result_holder["answer"]
    try:
        # AIME 答案是整数，尝试解析
        ans_str = str(answer).replace(",", "").strip()
        # 去掉可能的 LaTeX 包装
        ans_str = ans_str.replace("\\boxed{", "").replace("}", "")
        numeric_answer = int(float(ans_str))
        correct = numeric_answer == gold
    except (ValueError, TypeError):
        correct = False
        numeric_answer = None
    return {
        "id": problem_data["id"],
        "source": problem_data["source"],
        "question": problem_data["problem"][:100] + "...",
        "answer": answer,
        "numeric_answer": numeric_answer,
        "gold": gold,
        "correct": correct,
        "elapsed": round(elapsed, 2),
        "error": None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-tools", action="store_true", help="启用 SymPy tool-calling 预计算")
    parser.add_argument("--limit", type=int, default=0, help="限制测试题数（0=全部）")
    parser.add_argument("--timeout", type=int, default=300, help="单题超时秒数")
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "aime_20.json")
    with open(data_path, encoding="utf-8") as f:
        selected = json.load(f)

    if args.limit > 0:
        selected = selected[:args.limit]

    tools_label = " + SymPy Tool-Calling" if args.use_tools else ""
    print(f"\n{'='*70}")
    print(f"GRAVEC v2 + Agnes 2.0 Flash{tools_label} — AIME 基准测试")
    print(f"模型: openai/agnes-2.0-flash")
    print(f"题目: {len(selected)} 道 (AIME 2024/2025)")
    print(f"引擎: reason_from_future_nhx (GRAVEC 六步曲)")
    print(f"Tool-Calling: {'ON' if args.use_tools else 'OFF'}")
    print(f"最大迭代: 10 | 最少迭代: 2 | 超时: {args.timeout}s/题")
    print(f"判定: 精确整数匹配 (AIME 答案 0-999)")
    print(f"{'='*70}\n")

    results = []
    correct_count = 0
    total_time = 0.0

    for i, problem_data in enumerate(selected):
        print(
            f"[{i+1:2d}/{len(selected)}] {problem_data['id']:15s} "
            f"gold={problem_data['answer']:>4d}",
            end=" ",
            flush=True,
        )
        result = run_single(
            problem_data,
            verbose=False,
            timeout=args.timeout,
            use_tools=args.use_tools,
        )
        results.append(result)

        status = "OK" if result["correct"] else "FAIL"
        if result["correct"]:
            correct_count += 1
        total_time += result["elapsed"]

        answer_str = str(result.get("numeric_answer", result.get("answer", "ERR")))
        err_str = f" | err: {result['error'][:60]}" if result.get("error") else ""
        print(f"{status} got={answer_str:>6s} gold={result['gold']:>4d} ({result['elapsed']:.1f}s){err_str}")

        # 实时准确率
        if (i + 1) % 5 == 0:
            print(f"  >>> 进度 {i+1}/{len(selected)} | 当前准确率: {correct_count/(i+1)*100:.1f}%")

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

    # 按来源统计
    by_source = {}
    for r in results:
        by_source.setdefault(r["source"], []).append(r)
    print(f"\n  按来源:")
    for src, subset in sorted(by_source.items()):
        c = sum(1 for r in subset if r["correct"])
        avg_t = sum(r["elapsed"] for r in subset) / len(subset)
        print(f"    {src}: {c}/{len(subset)} ({c/len(subset)*100:.0f}%)  avg={avg_t:.1f}s/题")

    # 错误详情
    errors = [r for r in results if not r["correct"]]
    if errors:
        print(f"\n错误题目详情 ({len(errors)} 道):")
        for r in errors:
            answer_str = str(r.get("numeric_answer", r.get("answer", "ERR")))
            err_str = f" | {r['error'][:80]}" if r.get("error") else ""
            print(f"  {r['id']:15s} got={answer_str:>6s} gold={r['gold']:>4d}{err_str}")

    # 正确题目
    corrects = [r for r in results if r["correct"]]
    if corrects:
        print(f"\n正确题目 ({len(corrects)} 道):")
        for r in corrects:
            print(f"  {r['id']:15s} got={r['numeric_answer']:>4d} gold={r['gold']:>4d} ({r['elapsed']:.1f}s)")

    # 保存结果
    output_path = os.path.join(os.path.dirname(__file__), "..", "aime_20_gravec_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "openai/agnes-2.0-flash",
            "engine": "GRAVEC v2 (reason_from_future_nhx)",
            "dataset": "AIME 2024/2025 (20 selected)",
            "tool_calling": args.use_tools,
            "total": len(results),
            "correct": correct_count,
            "accuracy": correct_count / len(results) * 100,
            "total_time": round(total_time, 1),
            "avg_time": round(total_time / len(results), 1),
            "by_source": {
                s: {
                    "total": len(sub),
                    "correct": sum(1 for r in sub if r["correct"]),
                    "accuracy": sum(1 for r in sub if r["correct"]) / max(len(sub), 1) * 100,
                }
                for s, sub in sorted(by_source.items())
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
