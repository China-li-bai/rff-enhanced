"""
================================================================================
GSM8K 公开数据集验证 — validate_gsm8k.py
================================================================================

【费曼视角：为什么需要这个文件】
纯逻辑测试验证了"代码没bug"，但没验证"方法好不好用"。
就像你造了一辆汽车，单元测试验证了"刹车能刹住"，但没验证"能开多快"。

这个文件用 GSM8K 公开数据集的 10 道标准题来验证：
1. 原版 RFF 能解对几道？
2. 倪海厦版 RFF 能解对几道？
3. 倪海厦版的价值判断和验效是否真的有帮助？

【GSM8K 数据集】
GSM8K 是 OpenAI 发布的小学数学应用题数据集，包含 8,500+ 道题，
每道题都有标准答案（ground truth），是学术界公认的推理能力基准。

数据集地址：https://huggingface.co/datasets/gsm8k
论文：Cobbe et al., "Training Verifiers to Solve Math Word Problems" (2021)

【运行方式】
需要先设置 Gemini API Key：
    export GEMINI_API_KEY="your_key_here"

然后运行：
    cd /Users/mac/WorkBuddy/Claw/rff-enhanced
    PYTHONPATH=src python3 tests/validate_gsm8k.py

    只跑倪海厦版：
    PYTHONPATH=src python3 tests/validate_gsm8k.py --mode nhx

    只跑原版：
    PYTHONPATH=src python3 tests/validate_gsm8k.py --mode original

    对比模式：
    PYTHONPATH=src python3 tests/validate_gsm8k.py --mode compare
"""
import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reason_from_future.core import reason_from_future
from reason_from_future.specs.gsm8k import GSM8KSpec
from reason_from_future.core_nhx import reason_from_future_nhx
from reason_from_future.specs.gsm8k_nhx import GSM8KNiHaixiaSpec


# ============================================================================
# GSM8K 标准测试题（来自公开数据集）
# ============================================================================
GSM8K_SAMPLES = [
    {
        "id": 1,
        "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 each. How much does she make every day at the farmers' market?",
        "answer": "18",
        "category": "算术-基础",
    },
    {
        "id": 2,
        "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
        "answer": "3",
        "category": "算术-分数",
    },
    {
        "id": 3,
        "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
        "answer": "70000",
        "category": "算术-百分比",
    },
    {
        "id": 4,
        "question": "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many meters does he run a week?",
        "answer": "540",
        "category": "算术-乘法",
    },
    {
        "id": 5,
        "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of mixed chicken feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
        "answer": "20",
        "category": "算术-多步",
    },
    {
        "id": 6,
        "question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. How much did he pay for 6 glasses?",
        "answer": "24",
        "category": "算术-规律",
    },
    {
        "id": 7,
        "question": "Marissa is hiking a 12-mile trail. She took 1 hour to walk the first 4 miles, then another hour to walk the next 2 miles. If she wants her average speed for the whole hike to be 4 miles per hour, how fast should she walk the remaining distance?",
        "answer": "6",
        "category": "算术-速率",
    },
    {
        "id": 8,
        "question": "Carlos and Ben are working on a puzzle. Carlos can place 8 pieces per minute. Ben can place 6 pieces per minute. If they work together for 10 minutes, then Ben works alone for another 5 minutes, how many pieces do they place in total?",
        "answer": "170",
        "category": "算术-协作",
    },
    {
        "id": 9,
        "question": "A piece of square paper has a side length of 6 inches. Amy cuts the paper into two equal rectangles. What is the perimeter of one of the rectangles?",
        "answer": "18",
        "category": "几何-周长",
    },
    {
        "id": 10,
        "question": "There were 15 trees in the grove. 3 were cut down. Then, after some time, 2 more were cut down. But 1 grew back. How many are left?",
        "answer": "11",
        "category": "算术-增减",
    },
]


def run_original(problem_data: dict, max_iters: int = 10, verbose: bool = False) -> dict:
    """运行原版 RFF。"""
    spec = GSM8KSpec(problem_data)
    start = time.time()
    try:
        answer = reason_from_future(
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
            "answer": answer,
            "numeric_answer": numeric_answer,
            "gold": gold,
            "correct": correct,
            "elapsed": round(elapsed, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "answer": None,
            "numeric_answer": None,
            "gold": float(problem_data["answer"].replace(",", "")),
            "correct": False,
            "elapsed": round(time.time() - start, 2),
            "error": str(e),
        }


def run_nhx(problem_data: dict, max_iters: int = 10, verbose: bool = False) -> dict:
    """运行倪海厦版 RFF。"""
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
            "answer": answer,
            "numeric_answer": numeric_answer,
            "gold": gold,
            "correct": correct,
            "elapsed": round(elapsed, 2),
            "error": None,
        }
    except Exception as e:
        return {
            "answer": None,
            "numeric_answer": None,
            "gold": float(problem_data["answer"].replace(",", "")),
            "correct": False,
            "elapsed": round(time.time() - start, 2),
            "error": str(e),
        }


def print_result_row(idx: int, sample: dict, result: dict, mode: str):
    """打印单题结果。"""
    status = "✅" if result["correct"] else "❌"
    answer_str = str(result.get("numeric_answer", result.get("answer", "N/A")))
    gold_str = str(result["gold"])
    error_str = f" | 错误: {result['error'][:50]}" if result.get("error") else ""
    print(
        f"  {status} #{sample['id']:2d} [{sample['category']:8s}] "
        f"答案={answer_str:>8s} 标准={gold_str:>8s} "
        f"耗时={result['elapsed']:5.1f}s{error_str}"
    )


def run_validation(mode: str = "compare", max_iters: int = 10, verbose: bool = False):
    """运行验证。"""
    print(f"\n{'='*70}")
    print(f"GSM8K 公开数据集验证 — 模式: {mode}")
    print(f"题目数量: {len(GSM8K_SAMPLES)} | 最大迭代: {max_iters}")
    print(f"{'='*70}\n")

    original_results = []
    nhx_results = []

    if mode in ("original", "compare"):
        print("📊 原版 RFF 结果:")
        print("-" * 70)
        for sample in GSM8K_SAMPLES:
            result = run_original(sample, max_iters=max_iters, verbose=verbose)
            original_results.append(result)
            print_result_row(len(original_results), sample, result, "original")
            time.sleep(1)

        orig_correct = sum(1 for r in original_results if r["correct"])
        orig_avg_time = sum(r["elapsed"] for r in original_results) / len(original_results)
        print(f"\n  📈 原版 RFF: {orig_correct}/{len(GSM8K_SAMPLES)} 正确 "
              f"({orig_correct/len(GSM8K_SAMPLES)*100:.0f}%) | "
              f"平均耗时: {orig_avg_time:.1f}s\n")

    if mode in ("nhx", "compare"):
        print("📊 倪海厦版 RFF 结果:")
        print("-" * 70)
        for sample in GSM8K_SAMPLES:
            result = run_nhx(sample, max_iters=max_iters, verbose=verbose)
            nhx_results.append(result)
            print_result_row(len(nhx_results), sample, result, "nhx")
            time.sleep(1)

        nhx_correct = sum(1 for r in nhx_results if r["correct"])
        nhx_avg_time = sum(r["elapsed"] for r in nhx_results) / len(nhx_results)
        print(f"\n  📈 倪海厦版 RFF: {nhx_correct}/{len(GSM8K_SAMPLES)} 正确 "
              f"({nhx_correct/len(GSM8K_SAMPLES)*100:.0f}%) | "
              f"平均耗时: {nhx_avg_time:.1f}s\n")

    if mode == "compare" and original_results and nhx_results:
        print("=" * 70)
        print("📊 对比总结:")
        print("-" * 70)
        print(f"  {'题目':>6s} | {'原版':>8s} | {'倪海厦版':>8s} | {'差异':>8s}")
        print(f"  {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8}")
        for i, sample in enumerate(GSM8K_SAMPLES):
            orig_status = "✅" if original_results[i]["correct"] else "❌"
            nhx_status = "✅" if nhx_results[i]["correct"] else "❌"
            if original_results[i]["correct"] and not nhx_results[i]["correct"]:
                diff = "原版胜"
            elif not original_results[i]["correct"] and nhx_results[i]["correct"]:
                diff = "倪师胜"
            elif original_results[i]["correct"] and nhx_results[i]["correct"]:
                diff = "都对了"
            else:
                diff = "都错了"
            print(f"  #{sample['id']:4d} | {orig_status:>8s} | {nhx_status:>8s} | {diff:>8s}")

        orig_total = sum(1 for r in original_results if r["correct"])
        nhx_total = sum(1 for r in nhx_results if r["correct"])
        print(f"\n  总计: 原版 {orig_total}/{len(GSM8K_SAMPLES)} | "
              f"倪海厦版 {nhx_total}/{len(GSM8K_SAMPLES)}")

        nhx_only = sum(
            1 for i in range(len(GSM8K_SAMPLES))
            if nhx_results[i]["correct"] and not original_results[i]["correct"]
        )
        orig_only = sum(
            1 for i in range(len(GSM8K_SAMPLES))
            if original_results[i]["correct"] and not nhx_results[i]["correct"]
        )
        print(f"  倪师独有正确: {nhx_only} | 原版独有正确: {orig_only}")

    print(f"\n{'='*70}")
    print("验证完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K 公开数据集验证")
    parser.add_argument(
        "--mode",
        choices=["original", "nhx", "compare"],
        default="compare",
        help="验证模式: original=原版, nhx=倪海厦版, compare=对比",
    )
    parser.add_argument("--max-iters", type=int, default=10, help="最大迭代次数")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️ 需要设置 GEMINI_API_KEY 环境变量！")
        print("   运行: export GEMINI_API_KEY='your_key_here'")
        print()
        print("如果你只想运行纯逻辑测试（不需要 API Key），请运行：")
        print("   PYTHONPATH=src python3 tests/test_nhx_logic.py")
        sys.exit(1)

    run_validation(mode=args.mode, max_iters=args.max_iters, verbose=args.verbose)
