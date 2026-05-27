"""
================================================================================
HumanEval 大规模验证 — benchmark_humaneval.py
================================================================================

【费曼视角】
GSM8K 只能验证"算得对不对"，HumanEval 能验证"做出来的东西能不能用"。
这是「以果决其行」方法论的最佳测试场——

  果 = 测试用例（明确的、可执行的验证标准）
  行 = 代码实现（可执行、可验证的行动）
  验效 = 运行测试（真实的、客观的反馈）
  因果诊断 = 分析报错（为什么失败？逻辑错？边界？类型？）
  果行共变 = 换思路重写（当前方向走不通，换算法）

【对比实验设计】
  模式1: baseline  — 直接让 LLM 生成代码（一次生成，无反馈）
  模式2: nhx       — 倪海厦版 RFF（G→R→A→V→E→C，含验效反馈）
  模式3: compare   — 两者对比，证明方法论价值

【运行方式】
    cd /Users/mac/WorkBuddy/Claw/rff-enhanced
    uv run python3 tests/benchmark_humaneval.py

    只跑 20 题（快速验证）：
    uv run python3 tests/benchmark_humaneval.py --num 20

    对比模式：
    uv run python3 tests/benchmark_humaneval.py --mode compare --num 50

    保存结果：
    uv run python3 tests/benchmark_humaneval.py --num 164 --save humaneval_results.json
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reason_from_future.llm import DEFAULT_MODEL


def load_humaneval(num_samples: int | None = None, seed: int = 42):
    """从 HuggingFace 加载 HumanEval 数据集。"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ 需要安装 datasets 库: uv pip install datasets")
        sys.exit(1)

    print("📥 正在下载/加载 HumanEval 数据集...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    total = len(dataset)
    print(f"   HumanEval 共 {total} 道题")

    if num_samples is None or num_samples >= total:
        num_samples = total
        print(f"   使用全部 {num_samples} 道题")
        return list(dataset)

    import random
    random.seed(seed)
    indices = sorted(random.sample(range(total), num_samples))
    samples = [dataset[i] for i in indices]
    print(f"   随机抽取 {num_samples} 道题 (seed={seed})")
    return samples


def run_test(code: str, test: str, entry_point: str) -> dict:
    """安全执行测试用例。"""
    full_code = code + "\n" + test + f"\ncheck({entry_point})\n"
    try:
        exec_globals: dict = {}
        exec(full_code, exec_globals)
        return {"passed": True, "error": None}
    except AssertionError as e:
        return {"passed": False, "error": f"AssertionError: {e}"}
    except Exception as e:
        tb = traceback.format_exc()
        short = "\n".join(tb.splitlines()[-3:])
        return {"passed": False, "error": short}


def extract_code(raw_text: str) -> str:
    """从 LLM 输出中提取 Python 代码。"""
    import re
    code_blocks = re.findall(r"```(?:python)?\s*\n([\s\S]*?)```", raw_text)
    if code_blocks:
        return code_blocks[0].strip()

    lines = raw_text.strip().splitlines()
    code_lines = []
    in_function = False
    for line in lines:
        if line.strip().startswith("def "):
            in_function = True
        if in_function:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines)

    return raw_text.strip()


def run_baseline(problem_data: dict, verbose: bool = False) -> dict:
    """Baseline：直接让 LLM 生成代码，一次生成无反馈。"""
    from reason_from_future.llm import llm_call

    prompt = problem_data["prompt"]
    start = time.time()

    try:
        raw = llm_call(
            f"Complete the following Python function. Return ONLY the function implementation in a ```python code block.\n\n{prompt}",
            verbose=verbose,
        )
        code = extract_code(raw)
        if not code.strip().startswith("def "):
            code = prompt + code

        test_result = run_test(code, problem_data["test"], problem_data["entry_point"])
        elapsed = time.time() - start

        return {
            "passed": test_result["passed"],
            "error": test_result["error"],
            "code": code,
            "elapsed": round(elapsed, 2),
            "attempts": 1,
        }
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "code": "",
            "elapsed": round(time.time() - start, 2),
            "attempts": 1,
        }


def run_nhx(problem_data: dict, max_iters: int = 8, verbose: bool = False) -> dict:
    """倪海厦版 RFF：G→R→A→V→E→C，含验效反馈和因果诊断。"""
    from reason_from_future.core_nhx import reason_from_future_nhx
    from reason_from_future.specs.humaneval_nhx import HumanEvalNiHaixiaSpec

    spec = HumanEvalNiHaixiaSpec(problem_data)
    start = time.time()

    try:
        answer = reason_from_future_nhx(
            problem=problem_data["prompt"],
            spec=spec,
            max_iters=max_iters,
            verbose=verbose,
            require_gold=False,
            min_iters=1,
        )

        code = spec._get_current_code(spec.merge_aliases(Workspace()))
        if not code:
            code = answer if isinstance(answer, str) else ""

        test_result = run_test(code, problem_data["test"], problem_data["entry_point"])
        elapsed = time.time() - start

        return {
            "passed": test_result["passed"],
            "error": test_result["error"],
            "code": code,
            "elapsed": round(elapsed, 2),
            "attempts": 1,
        }
    except Exception as e:
        return {
            "passed": False,
            "error": str(e),
            "code": "",
            "elapsed": round(time.time() - start, 2),
            "attempts": 1,
        }


from reason_from_future.core import Workspace


def run_nhx_direct(problem_data: dict, max_iters: int = 8, verbose: bool = False) -> dict:
    """倪海厦版 RFF（直接执行版）：自己管理循环，确保代码执行验证。"""
    from reason_from_future.specs.humaneval_nhx import HumanEvalNiHaixiaSpec
    from reason_from_future.llm import llm_call

    spec = HumanEvalNiHaixiaSpec(problem_data)
    start = time.time()

    state = Workspace()
    goal = "pass_all_tests"
    avoid = set()
    debug_attempts = 0
    best_code = ""
    best_passed = False

    for iter_idx in range(max_iters):
        current_code = spec._get_current_code(state)
        test_results = state.get("test_results", {})

        if current_code and test_results.get("passed", False):
            best_code = current_code
            best_passed = True
            break

        if iter_idx == 0:
            target_step = "pass_all_tests"
        else:
            g_prompt = spec.prompt_last_step(state, goal, avoid)
            raw = llm_call(g_prompt, verbose=verbose)
            target_step = spec.parse_target_step(raw)

            if not target_step or target_step in avoid:
                target_step = "fix_logic_error"

        r_prompt = spec.prompt_forward_step(state, target_step, avoid)
        raw = llm_call(r_prompt, verbose=verbose)
        update = spec.parse_workspace_update(raw, state)

        if update and update.get("current_code"):
            state = state | update
            new_code = update["current_code"]
            new_results = update.get("test_results", {})

            if new_results.get("passed", False):
                best_code = new_code
                best_passed = True
                break

            observation = spec.execute_action(state, target_step, goal)
            effect = spec.evaluate_observation(observation, state, goal)

            if effect < 0:
                avoid.add(target_step)
                debug_attempts += 1

            if debug_attempts >= 4 and not best_passed:
                target_step = "rewrite_approach"
                avoid.clear()
                debug_attempts = 0

            state["debug_attempts"] = debug_attempts
        else:
            avoid.add(target_step)
            debug_attempts += 1

    if not best_code:
        best_code = spec._get_current_code(state)

    if best_code and not best_passed:
        test_result = run_test(best_code, problem_data["test"], problem_data["entry_point"])
        best_passed = test_result["passed"]

    elapsed = time.time() - start

    return {
        "passed": best_passed,
        "error": None if best_passed else "未通过测试",
        "code": best_code,
        "elapsed": round(elapsed, 2),
        "attempts": debug_attempts + 1,
    }


def run_benchmark(
    mode: str = "nhx",
    num_samples: int | None = None,
    max_iters: int = 8,
    verbose: bool = False,
    save_path: str | None = None,
):
    """运行 HumanEval 基准测试。"""
    samples = load_humaneval(num_samples)

    print(f"\n{'='*70}")
    print(f"HumanEval 基准测试 — 以果决其行验证")
    print(f"模型: {DEFAULT_MODEL}")
    print(f"模式: {mode} | 题数: {len(samples)} | 最大迭代: {max_iters}")
    print(f"{'='*70}\n")

    all_results = {
        "meta": {
            "model": DEFAULT_MODEL,
            "mode": mode,
            "num_samples": len(samples),
            "max_iters": max_iters,
        },
        "baseline": [],
        "nhx": [],
    }

    if mode in ("baseline", "compare"):
        print("📊 Baseline（直接 LLM 生成，无反馈）:")
        print("-" * 70)
        bl_correct = 0
        bl_total_time = 0
        for i, sample in enumerate(samples):
            result = run_baseline(sample, verbose=verbose)
            all_results["baseline"].append({
                "task_id": sample["task_id"],
                **result,
            })
            if result["passed"]:
                bl_correct += 1
            bl_total_time += result["elapsed"]

            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {i+1:3d}/{len(samples)} {sample['task_id']:12s} {result['elapsed']:5.1f}s")

            if (i + 1) % 20 == 0:
                pct = bl_correct / (i + 1) * 100
                print(f"  --- 进度: {i+1}/{len(samples)} | pass@1: {pct:.1f}% ---")

        bl_pct = bl_correct / len(samples) * 100
        bl_avg = bl_total_time / len(samples)
        print(f"\n  📈 Baseline: {bl_correct}/{len(samples)} pass@1 ({bl_pct:.1f}%) | "
              f"平均耗时: {bl_avg:.1f}s\n")

    if mode in ("nhx", "compare"):
        print("📊 倪海厦版 RFF（G→R→A→V→E→C，含验效反馈）:")
        print("-" * 70)
        nhx_correct = 0
        nhx_total_time = 0
        nhx_fixed_count = 0
        for i, sample in enumerate(samples):
            result = run_nhx_direct(sample, max_iters=max_iters, verbose=verbose)
            all_results["nhx"].append({
                "task_id": sample["task_id"],
                **result,
            })
            if result["passed"]:
                nhx_correct += 1
            nhx_total_time += result["elapsed"]

            status = "✅" if result["passed"] else "❌"
            attempts = result.get("attempts", 1)
            print(f"  {status} {i+1:3d}/{len(samples)} {sample['task_id']:12s} {result['elapsed']:5.1f}s iter={attempts}")

            if (i + 1) % 20 == 0:
                pct = nhx_correct / (i + 1) * 100
                print(f"  --- 进度: {i+1}/{len(samples)} | pass@1: {pct:.1f}% ---")

        nhx_pct = nhx_correct / len(samples) * 100
        nhx_avg = nhx_total_time / len(samples)
        print(f"\n  📈 倪海厦版 RFF: {nhx_correct}/{len(samples)} pass@1 ({nhx_pct:.1f}%) | "
              f"平均耗时: {nhx_avg:.1f}s\n")

    if mode == "compare":
        print("=" * 70)
        print("📊 对比总结 — 以果决其行 vs 直接生成:")
        print("-" * 70)

        bl_results = {r["task_id"]: r for r in all_results["baseline"]}
        nhx_results = {r["task_id"]: r for r in all_results["nhx"]}

        both_pass = 0
        bl_only = 0
        nhx_only = 0
        neither = 0
        nhx_fixed_tasks = []

        for task_id in bl_results:
            bl_pass = bl_results[task_id]["passed"]
            nhx_pass = nhx_results[task_id]["passed"]

            if bl_pass and nhx_pass:
                both_pass += 1
            elif bl_pass and not nhx_pass:
                bl_only += 1
            elif not bl_pass and nhx_pass:
                nhx_only += 1
                nhx_fixed_tasks.append(task_id)
            else:
                neither += 1

        print(f"  两者都通过: {both_pass}")
        print(f"  两者都失败: {neither}")
        print(f"  Baseline 独有通过: {bl_only}")
        print(f"  倪海厦版独有通过: {nhx_only} ← 方法论修复的题目")
        print(f"")
        print(f"  Baseline:     {bl_correct}/{len(samples)} ({bl_pct:.1f}%)")
        print(f"  倪海厦版 RFF: {nhx_correct}/{len(samples)} ({nhx_pct:.1f}%)")
        print(f"  方法论提升:   +{nhx_pct - bl_pct:.1f}%")

        if nhx_fixed_tasks:
            print(f"\n  🔧 倪海厦版修复的题目:")
            for tid in nhx_fixed_tasks:
                print(f"     - {tid}")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存到: {save_path}")

    print(f"\n{'='*70}")
    print("验证完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HumanEval 基准测试 — 以果决其行验证")
    parser.add_argument("--mode", choices=["baseline", "nhx", "compare"], default="nhx",
                        help="测试模式: baseline=直接生成, nhx=倪海厦版, compare=对比")
    parser.add_argument("--num", type=int, default=None,
                        help="测试题数（默认全部164题）")
    parser.add_argument("--max-iters", type=int, default=8,
                        help="NHX最大迭代次数")
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
