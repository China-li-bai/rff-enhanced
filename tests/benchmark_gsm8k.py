"""
================================================================================
GSM8K 大规模验证 — benchmark_gsm8k.py
================================================================================

【费曼视角】
10 道题只能"尝味道"，200 道题才能"吃饱"。
这个脚本从 HuggingFace 下载 GSM8K 原始测试集，随机抽取 200 道题，
用倪海厦版 RFF 逐一验证，输出准确率和详细报告。

【运行方式】
    cd /Users/mac/WorkBuddy/Claw/rff-enhanced
    uv run python3 tests/benchmark_gsm8k.py

    只跑 50 题（快速验证）：
    uv run python3 tests/benchmark_gsm8k.py --num 50

    指定模式：
    uv run python3 tests/benchmark_gsm8k.py --mode nhx
    uv run python3 tests/benchmark_gsm8k.py --mode original
    uv run python3 tests/benchmark_gsm8k.py --mode compare

    保存结果到文件：
    uv run python3 tests/benchmark_gsm8k.py --num 200 --save results.json

【配置】
LLM 配置从项目根目录 llm_config.toml 读取，无需硬编码。
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reason_from_future.llm import DEFAULT_MODEL


def load_gsm8k_test_set(num_samples: int = 200, seed: int = 42):
    """从 HuggingFace 加载 GSM8K 测试集并随机抽取指定数量。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 需要安装 datasets 库: uv pip install datasets")
        sys.exit(1)

    print("📥 正在下载/加载 GSM8K 数据集...")
    dataset = load_dataset("gsm8k", "main", split="test")
    total = len(dataset)
    print(f"   GSM8K 测试集共 {total} 道题")

    if num_samples >= total:
        num_samples = total
        print(f"   使用全部 {num_samples} 道题")
        samples = list(dataset)
    else:
        random.seed(seed)
        indices = random.sample(range(total), num_samples)
        indices.sort()
        samples = [dataset[i] for i in indices]
        print(f"   随机抽取 {num_samples} 道题 (seed={seed})")

    results = []
    for i, item in enumerate(samples):
        question = item["question"]
        answer_str = item["answer"]
        if "####" in answer_str:
            gold = answer_str.split("####")[-1].strip().replace(",", "")
        else:
            gold = answer_str.strip().replace(",", "")
        results.append({
            "id": i + 1,
            "question": question,
            "answer": gold,
        })
    return results


def run_nhx(problem_data: dict, max_iters: int = 10, verbose: bool = False) -> dict:
    """运行倪海厦版 RFF。"""
    from reason_from_future.core_nhx import reason_from_future_nhx
    from reason_from_future.specs.gsm8k_nhx import GSM8KNiHaixiaSpec

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


def run_original(problem_data: dict, max_iters: int = 10, verbose: bool = False) -> dict:
    """运行原版 RFF。"""
    from reason_from_future.core import reason_from_future
    from reason_from_future.specs.gsm8k import GSM8KSpec

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


def run_benchmark(
    mode: str = "nhx",
    num_samples: int = 200,
    max_iters: int = 10,
    verbose: bool = False,
    save_path: str | None = None,
):
    """运行大规模基准测试。"""
    samples = load_gsm8k_test_set(num_samples)

    print(f"\n{'='*70}")
    print(f"GSM8K 大规模验证")
    print(f"模型: {DEFAULT_MODEL}")
    print(f"模式: {mode} | 题数: {len(samples)} | 最大迭代: {max_iters}")
    print(f"{'='*70}\n")

    all_results = {"meta": {"model": DEFAULT_MODEL, "mode": mode, "num_samples": len(samples), "max_iters": max_iters}, "nhx": [], "original": []}

    if mode in ("nhx", "compare"):
        print("📊 倪海厦版 RFF:")
        print("-" * 70)
        nhx_correct = 0
        nhx_errors = 0
        nhx_total_time = 0
        for i, sample in enumerate(samples):
            result = run_nhx(sample, max_iters=max_iters, verbose=verbose)
            all_results["nhx"].append({"id": sample["id"], **result})
            if result["correct"]:
                nhx_correct += 1
            if result.get("error"):
                nhx_errors += 1
            nhx_total_time += result["elapsed"]

            status = "✅" if result["correct"] else "❌"
            ans = str(result.get("numeric_answer", result.get("answer", "N/A")))
            err = f" | ERR: {result['error'][:40]}" if result.get("error") else ""
            print(f"  {status} #{i+1:3d}/{len(samples)} 答案={ans:>10s} 标准={result['gold']:>10.0f} {result['elapsed']:5.1f}s{err}")

            if (i + 1) % 20 == 0:
                pct = nhx_correct / (i + 1) * 100
                print(f"  --- 进度: {i+1}/{len(samples)} | 当前准确率: {pct:.1f}% ---")

        nhx_pct = nhx_correct / len(samples) * 100
        nhx_avg = nhx_total_time / len(samples)
        print(f"\n  📈 倪海厦版 RFF: {nhx_correct}/{len(samples)} 正确 ({nhx_pct:.1f}%) | "
              f"平均耗时: {nhx_avg:.1f}s | 错误: {nhx_errors}\n")

    if mode in ("original", "compare"):
        print("📊 原版 RFF:")
        print("-" * 70)
        orig_correct = 0
        orig_errors = 0
        orig_total_time = 0
        for i, sample in enumerate(samples):
            result = run_original(sample, max_iters=max_iters, verbose=verbose)
            all_results["original"].append({"id": sample["id"], **result})
            if result["correct"]:
                orig_correct += 1
            if result.get("error"):
                orig_errors += 1
            orig_total_time += result["elapsed"]

            status = "✅" if result["correct"] else "❌"
            ans = str(result.get("numeric_answer", result.get("answer", "N/A")))
            err = f" | ERR: {result['error'][:40]}" if result.get("error") else ""
            print(f"  {status} #{i+1:3d}/{len(samples)} 答案={ans:>10s} 标准={result['gold']:>10.0f} {result['elapsed']:5.1f}s{err}")

            if (i + 1) % 20 == 0:
                pct = orig_correct / (i + 1) * 100
                print(f"  --- 进度: {i+1}/{len(samples)} | 当前准确率: {pct:.1f}% ---")

        orig_pct = orig_correct / len(samples) * 100
        orig_avg = orig_total_time / len(samples)
        print(f"\n  📈 原版 RFF: {orig_correct}/{len(samples)} 正确 ({orig_pct:.1f}%) | "
              f"平均耗时: {orig_avg:.1f}s | 错误: {orig_errors}\n")

    if mode == "compare":
        print("=" * 70)
        print("📊 对比总结:")
        print("-" * 70)
        nhx_only = sum(
            1 for i in range(len(samples))
            if all_results["nhx"][i]["correct"] and not all_results["original"][i]["correct"]
        )
        orig_only = sum(
            1 for i in range(len(samples))
            if all_results["original"][i]["correct"] and not all_results["nhx"][i]["correct"]
        )
        both = sum(
            1 for i in range(len(samples))
            if all_results["nhx"][i]["correct"] and all_results["original"][i]["correct"]
        )
        neither = sum(
            1 for i in range(len(samples))
            if not all_results["nhx"][i]["correct"] and not all_results["original"][i]["correct"]
        )
        print(f"  两者都对: {both} | 两者都错: {neither}")
        print(f"  倪师独有正确: {nhx_only} | 原版独有正确: {orig_only}")
        print(f"  倪海厦版: {nhx_correct}/{len(samples)} ({nhx_pct:.1f}%)")
        print(f"  原版:     {orig_correct}/{len(samples)} ({orig_pct:.1f}%)")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存到: {save_path}")

    print(f"\n{'='*70}")
    print("验证完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K 大规模验证")
    parser.add_argument("--mode", choices=["original", "nhx", "compare"], default="nhx", help="验证模式")
    parser.add_argument("--num", type=int, default=200, help="测试题数（默认200）")
    parser.add_argument("--max-iters", type=int, default=10, help="最大迭代次数")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    parser.add_argument("--save", type=str, default=None, help="保存结果到 JSON 文件")
    args = parser.parse_args()

    run_benchmark(
        mode=args.mode,
        num_samples=args.num,
        max_iters=args.max_iters,
        verbose=args.verbose,
        save_path=args.save,
    )
