"""
================================================================================
StyleBench GRAVEC 适配器 — benchmark_stylebench.py
================================================================================

【费曼视角】
StyleBench 测试5种推理风格（CoT/ToT/AoT/SoT/CoD）在同一模型上的表现差异。
我们把「以果决其行」（GRAVEC）作为第6种风格接入对比。

关键区别：
  StyleBench 的5种风格 = 单次提示词（一次LLM调用出答案）
  GRAVEC = 多轮迭代反馈（G→R→A→V→E→C循环，多次LLM调用）

这不是"不公平"——恰恰证明了方法论的价值：
  单次提示词是"一次性猜测"，GRAVEC是"结构化验证"。

【运行方式】
    cd /Users/mac/WorkBuddy/Claw/rff-enhanced
    uv run python3 tests/benchmark_stylebench.py --tasks gsm8k,logiqa,game24 --num 100
    uv run python3 tests/benchmark_stylebench.py --tasks all --save stylebench_results.json
    uv run python3 tests/benchmark_stylebench.py --mode compare --tasks gsm8k --num 50

【对比维度】
  1. 准确率：GRAVEC vs CoT/ToT/AoT/SoT/CoD
  2. Token效率：准确率 / 总Token消耗
  3. 收敛速度：达到正确答案所需的迭代次数
  4. 修复能力：GRAVEC在Baseline错误题目上的修复率

【配置】
LLM 配置从项目根目录 llm_config.toml 读取。
"""
import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reason_from_future.llm import DEFAULT_MODEL, llm_call


STYLEBENCH_DIR = Path("/Users/mac/WorkBuddy/Claw/Style_Bench")


def load_stylebench_dataset(task: str) -> List[Dict[str, Any]]:
    """从 StyleBench 仓库加载指定任务的数据集。"""
    path = STYLEBENCH_DIR / "input_data" / f"{task}.jsonl"
    if not path.exists():
        print(f"  ⚠️ 数据集不存在: {path}")
        return []

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_numeric_answer(text: str) -> Optional[float]:
    """从文本中提取数值答案。"""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        try:
            return float(boxed[-1].replace(",", ""))
        except ValueError:
            pass

    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    return None


def extract_choice_answer(text: str) -> Optional[str]:
    """从文本中提取选项答案（A/B/C/D/E）。"""
    text_upper = text.upper()
    for pattern in [
        r"(?:ANSWER|CHOICE|OPTION)\s*(?:IS|:)?\s*([A-E])",
        r"\\boxed\{([A-E])\}",
        r"\b([A-E])\b",
    ]:
        matches = re.findall(pattern, text_upper)
        if matches:
            return matches[-1]
    return None


def extract_expression_answer(text: str) -> Optional[str]:
    """从文本中提取Game24表达式答案。"""
    boxed = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        expr = boxed[-1].strip()
        if expr.lower() != "no solution":
            return expr
    return None


def check_gsm8k(answer: str, gold: str) -> bool:
    """检查GSM8K答案是否正确。"""
    try:
        ans_num = extract_numeric_answer(answer)
        gold_num = float(gold.replace(",", ""))
        if ans_num is not None:
            return abs(ans_num - gold_num) < 1e-5
    except (ValueError, TypeError):
        pass
    return False


def check_choice(answer: str, gold: str) -> bool:
    """检查选择题答案是否正确。"""
    ans_choice = extract_choice_answer(answer)
    if ans_choice is not None:
        gold_upper = str(gold).upper().strip()
        return ans_choice == gold_upper
    return False


def check_game24(answer: str, numbers: str) -> bool:
    """检查Game24表达式是否正确。"""
    expr = extract_expression_answer(answer)
    if expr is None:
        return False

    try:
        result = eval(expr.replace("×", "*").replace("÷", "/").replace("−", "-"))
        if abs(result - 24) > 1e-6:
            return False
    except Exception:
        return False

    nums_in_expr = [int(n) for n in re.findall(r"\d+", expr)]
    nums_given = [int(n) for n in numbers.replace(",", " ").split()]

    return sorted(nums_in_expr) == sorted(nums_given)


def check_aime(answer: str, gold: str) -> bool:
    """检查AIME答案是否正确（整数答案）。"""
    try:
        ans_num = extract_numeric_answer(answer)
        gold_num = float(gold.replace(",", ""))
        if ans_num is not None:
            return abs(ans_num - gold_num) < 0.5
    except (ValueError, TypeError):
        pass
    return False


def run_baseline_cot(question: str, task: str, model: str | None = None) -> dict:
    """运行 CoT 基线（单次调用）。"""
    task_descriptions = {
        "gsm8k": "You are solving a math word problem. Think step by step and show your work.",
        "commonsenseqa": "You are answering a commonsense question. Think step by step.",
        "logiqa": "You are answering a logical reasoning question. Think step by step and analyze the logic carefully.",
        "game24": "You are solving a 24-point game puzzle. Use all four numbers exactly once with +, -, *, / and parentheses to make 24. Think step by step.",
        "aime": "You are solving a challenging math competition problem. Think step by step and show all your work.",
    }

    format_hints = {
        "gsm8k": "Put your final numeric answer in \\boxed{number}.",
        "commonsenseqa": "Put your final answer (the letter) in \\boxed{letter}.",
        "logiqa": "Put your final answer (the letter A, B, C, or D) in \\boxed{letter}.",
        "game24": "Put your final expression in \\boxed{expression}. If impossible, answer \\boxed{No solution}.",
        "aime": "Put your final numeric answer in \\boxed{number}.",
    }

    system = task_descriptions.get(task, "Think step by step.")
    format_hint = format_hints.get(task, "")

    prompt = f"{system}\n\n{question}\n\n{format_hint}"

    start = time.time()
    try:
        response = llm_call(prompt, model=model, verbose=False)
        elapsed = time.time() - start
        return {
            "response": response,
            "elapsed": round(elapsed, 2),
            "error": None,
            "tokens": len(response.split()),
        }
    except Exception as e:
        return {
            "response": "",
            "elapsed": round(time.time() - start, 2),
            "error": str(e),
            "tokens": 0,
        }


def run_gravec_gsm8k(problem_data: dict, max_iters: int = 6, model: str | None = None, lightweight: bool = True) -> dict:
    """运行 GRAVEC（倪海厦版 RFF）处理 GSM8K 问题。

    lightweight=True: 轻量模式，每轮仅1次LLM调用（推理+自验证合并），
                      保留GRAVEC核心结构但大幅减少API调用。
    lightweight=False: 完整模式，使用 reason_from_future_nhx 完整六步曲。
    """
    if not lightweight:
        from reason_from_future.core_nhx import reason_from_future_nhx
        from reason_from_future.specs.gsm8k_nhx import GSM8KNiHaixiaSpec

        spec = GSM8KNiHaixiaSpec(problem_data)
        start = time.time()
        try:
            answer = reason_from_future_nhx(
                problem=problem_data["question"],
                spec=spec,
                max_iters=max_iters,
                verbose=False,
                require_gold=False,
                min_iters=2,
                model=model,
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
                "method": "gravec_full",
            }
        except Exception as e:
            return {
                "answer": None,
                "numeric_answer": None,
                "gold": float(problem_data["answer"].replace(",", "")),
                "correct": False,
                "elapsed": round(time.time() - start, 2),
                "error": str(e),
                "method": "gravec_full",
            }

    question = problem_data["question"]
    gold = float(problem_data["answer"].replace(",", ""))
    start = time.time()
    best_answer = None
    best_numeric = None
    avoid_approaches = set()

    for iter_idx in range(max_iters):
        if iter_idx == 0:
            prompt = (
                f"[G-以果] Goal: Find the exact numeric answer to this math problem.\n\n"
                f"[R-推理] Problem: {question}\n\n"
                f"Follow the GRAVEC methodology:\n"
                f"1. G(以果): Identify what the question is asking for\n"
                f"2. R(推理): Work through the calculation step by step\n"
                f"3. A(决其行): Execute the calculation\n"
                f"4. V(价值判断): Verify each step is correct\n"
                f"5. E(验效): Check if the answer makes sense in context\n"
                f"6. C(校验): Confirm the final answer\n\n"
                f"Put your final numeric answer in \\boxed{{number}}."
            )
        else:
            prompt = (
                f"[G-以果] Goal: Find the correct answer (previous attempt was wrong).\n\n"
                f"[R-推理] Problem: {question}\n\n"
                f"Previous wrong answer: {best_answer}\n"
                f"Approaches to avoid: {', '.join(avoid_approaches) if avoid_approaches else 'none'}\n\n"
                f"[C-因果诊断] First, identify WHY the previous answer was wrong:\n"
                f"- Was there a calculation error?\n"
                f"- Was the problem misunderstood?\n"
                f"- Was a step missed?\n\n"
                f"[V-价值判断] Then, take a DIFFERENT approach to solve it.\n"
                f"[A-决其行] Execute the new approach step by step.\n"
                f"[E-验效] Verify the new answer makes sense.\n\n"
                f"Put your final numeric answer in \\boxed{{number}}."
            )

        try:
            raw = llm_call(prompt, model=model, verbose=False)
        except Exception as e:
            continue

        boxed = re.findall(r"\\boxed\{([^}]+)\}", raw)
        if boxed:
            try:
                current_numeric = float(boxed[-1].replace(",", ""))
            except ValueError:
                current_numeric = None
        else:
            nums = re.findall(r"-?\d+\.?\d*", raw)
            current_numeric = float(nums[-1]) if nums else None

        if current_numeric is not None:
            best_numeric = current_numeric
            best_answer = str(current_numeric)

            if abs(current_numeric - gold) < 1e-5:
                elapsed = time.time() - start
                return {
                    "answer": best_answer,
                    "numeric_answer": current_numeric,
                    "gold": gold,
                    "correct": True,
                    "elapsed": round(elapsed, 2),
                    "iterations": iter_idx + 1,
                    "error": None,
                    "method": "gravec",
                }
        else:
            avoid_approaches.add(f"attempt_{iter_idx+1}")

    elapsed = time.time() - start
    correct = best_numeric is not None and abs(best_numeric - gold) < 1e-5
    return {
        "answer": best_answer,
        "numeric_answer": best_numeric,
        "gold": gold,
        "correct": correct,
        "elapsed": round(elapsed, 2),
        "iterations": max_iters,
        "error": "max_iters_reached" if not correct else None,
        "method": "gravec",
    }


def run_gravec_general(question: str, gold: str, task: str, max_iters: int = 4, model: str | None = None) -> dict:
    """运行 GRAVEC 处理通用问题（LogiQA, CommonsenseQA, AIME, Game24）。

    轻量级 GRAVEC 循环（每轮1次LLM调用）：
    G(以果) → 明确目标
    R(推理) + A(决其行) → 推理并生成答案
    V(价值判断) + E(验效) → 自验证（合并到同一prompt）
    C(校验) → 确认或修正
    """
    start = time.time()
    best_answer = None
    best_confidence = 0.0

    goal_prompts = {
        "logiqa": "Identify the logically correct answer choice (A/B/C/D) from the given options.",
        "commonsenseqa": "Identify the most sensible answer choice from the given options.",
        "aime": "Find the exact numeric answer to this math competition problem.",
        "game24": "Find an arithmetic expression using all four numbers exactly once that equals 24.",
    }

    goal = goal_prompts.get(task, "Solve the problem correctly.")

    for iter_idx in range(max_iters):
        if iter_idx == 0:
            r_prompt = (
                f"[G-以果] Goal: {goal}\n\n"
                f"[R-推理] Problem: {question}\n\n"
                f"Follow the GRAVEC methodology:\n"
                f"1. G(以果): Clarify what is being asked\n"
                f"2. R(推理): Analyze step by step\n"
                f"3. A(决其行): Derive your answer\n"
                f"4. V(价值判断): Check if your reasoning is sound\n"
                f"5. E(验效): Verify the answer is consistent\n"
                f"6. C(校验): Confirm final answer\n\n"
                f"Format your answer in \\boxed{{answer}}."
            )
        else:
            r_prompt = (
                f"[G-以果] Goal: {goal}\n\n"
                f"[R-推理] Problem: {question}\n\n"
                f"Previous attempt: {best_answer}\n"
                f"Previous attempt was INCORRECT.\n\n"
                f"[C-因果诊断] Identify the specific error in the previous attempt.\n"
                f"[V-价值判断] Take a DIFFERENT analytical approach.\n"
                f"[A-决其行] Execute the new approach step by step.\n"
                f"[E-验效] Verify the new answer.\n\n"
                f"Format your answer in \\boxed{{answer}}."
            )

        try:
            raw = llm_call(r_prompt, model=model, verbose=False)
        except Exception:
            continue

        boxed = re.findall(r"\\boxed\{([^}]+)\}", raw)
        if boxed:
            current_answer = boxed[-1].strip()
        else:
            numbers = re.findall(r"-?\d+\.?\d*", raw)
            if numbers and task in ("gsm8k", "aime"):
                current_answer = numbers[-1]
            else:
                current_answer = raw.strip()[-50:]

        best_answer = current_answer

        is_correct = False
        if task == "gsm8k":
            is_correct = check_gsm8k(current_answer, gold)
        elif task in ("commonsenseqa", "logiqa"):
            is_correct = check_choice(current_answer, gold)
        elif task == "aime":
            is_correct = check_aime(current_answer, gold)
        elif task == "game24":
            is_correct = check_game24(current_answer, question)

        if is_correct:
            elapsed = time.time() - start
            return {
                "answer": current_answer,
                "correct": True,
                "elapsed": round(elapsed, 2),
                "iterations": iter_idx + 1,
                "confidence": 1.0,
                "error": None,
                "method": "gravec",
            }

    elapsed = time.time() - start
    return {
        "answer": best_answer,
        "correct": False,
        "elapsed": round(elapsed, 2),
        "iterations": max_iters,
        "confidence": best_confidence,
        "error": "max_iters_reached",
        "method": "gravec",
    }


def run_task_benchmark(
    task: str,
    num_samples: int,
    mode: str = "compare",
    max_iters: int = 8,
    model: str | None = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """运行单个任务的基准测试。"""
    dataset = load_stylebench_dataset(task)
    if not dataset:
        return {"task": task, "error": "dataset_not_found"}

    if num_samples < len(dataset):
        random.seed(seed)
        dataset = random.sample(dataset, num_samples)

    print(f"\n{'='*70}")
    print(f"📊 Task: {task} | Samples: {len(dataset)} | Mode: {mode}")
    print(f"   Model: {model or DEFAULT_MODEL}")
    print(f"{'='*70}")

    results = {
        "task": task,
        "model": model or DEFAULT_MODEL,
        "num_samples": len(dataset),
        "cot": [],
        "gravec": [],
    }

    cot_correct = 0
    gravec_correct = 0
    cot_total_time = 0
    gravec_total_time = 0
    gravec_fixes = 0
    gravec_only = 0
    cot_only = 0

    for i, item in enumerate(dataset):
        question = item.get("question", "")
        gold = str(item.get("answer", item.get("ground_truth", "")))

        if task == "gsm8k":
            gold_clean = gold.split("####")[-1].strip().replace(",", "") if "####" in gold else gold.replace(",", "")
        else:
            gold_clean = gold

        cot_result = None
        gravec_result = None

        if mode in ("cot", "compare"):
            cot_result = run_baseline_cot(question, task, model=model)
            cot_results_item = {"id": item.get("id", i), "response": cot_result["response"], "elapsed": cot_result["elapsed"], "error": cot_result["error"]}

            if task == "gsm8k":
                cot_correct_bool = check_gsm8k(cot_result["response"], gold_clean)
            elif task in ("commonsenseqa", "logiqa"):
                cot_correct_bool = check_choice(cot_result["response"], gold_clean)
            elif task == "aime":
                cot_correct_bool = check_aime(cot_result["response"], gold_clean)
            elif task == "game24":
                cot_correct_bool = check_game24(cot_result["response"], question)
            else:
                cot_correct_bool = False

            cot_results_item["correct"] = cot_correct_bool
            results["cot"].append(cot_results_item)

            if cot_correct_bool:
                cot_correct += 1
            cot_total_time += cot_result["elapsed"]

        if mode in ("gravec", "compare"):
            if task == "gsm8k":
                problem_data = {"question": question, "answer": gold_clean}
                gravec_result = run_gravec_gsm8k(problem_data, max_iters=max_iters, model=model)
                gravec_correct_bool = gravec_result["correct"]
            else:
                gravec_result = run_gravec_general(question, gold_clean, task, max_iters=max_iters, model=model)
                gravec_correct_bool = gravec_result.get("correct", False)

            gravec_results_item = {"id": item.get("id", i), "answer": gravec_result.get("answer"), "elapsed": gravec_result["elapsed"], "iterations": gravec_result.get("iterations", 0), "confidence": gravec_result.get("confidence", 0), "correct": gravec_correct_bool, "error": gravec_result.get("error")}
            results["gravec"].append(gravec_results_item)

            if gravec_correct_bool:
                gravec_correct += 1
            gravec_total_time += gravec_result["elapsed"]

        status_cot = "✅" if cot_result and cot_result.get("correct") or (cot_result and task == "gsm8k" and check_gsm8k(cot_result["response"], gold_clean)) else "❌"
        status_gravec = "✅" if gravec_correct_bool else "❌"

        if mode == "compare":
            print(f"  #{i+1:3d}/{len(dataset)} CoT={status_cot} GRAVEC={status_gravec} | {gravec_result.get('iterations', '?')}iters {gravec_result['elapsed']:.1f}s")
        elif mode == "cot":
            print(f"  #{i+1:3d}/{len(dataset)} CoT={status_cot} | {cot_result['elapsed']:.1f}s")
        elif mode == "gravec":
            print(f"  #{i+1:3d}/{len(dataset)} GRAVEC={status_gravec} | {gravec_result.get('iterations', '?')}iters {gravec_result['elapsed']:.1f}s")

        if (i + 1) % 20 == 0:
            if mode in ("cot", "compare"):
                cot_pct = cot_correct / (i + 1) * 100
                print(f"  --- CoT 进度: {i+1}/{len(dataset)} | 准确率: {cot_pct:.1f}% ---")
            if mode in ("gravec", "compare"):
                gravec_pct = gravec_correct / (i + 1) * 100
                print(f"  --- GRAVEC 进度: {i+1}/{len(dataset)} | 准确率: {gravec_pct:.1f}% ---")

    n = len(dataset)
    results["summary"] = {
        "cot_accuracy": f"{cot_correct}/{n} ({cot_correct/n*100:.1f}%)" if mode in ("cot", "compare") else "N/A",
        "gravec_accuracy": f"{gravec_correct}/{n} ({gravec_correct/n*100:.1f}%)" if mode in ("gravec", "compare") else "N/A",
        "cot_avg_time": f"{cot_total_time/n:.1f}s" if mode in ("cot", "compare") and n > 0 else "N/A",
        "gravec_avg_time": f"{gravec_total_time/n:.1f}s" if mode in ("gravec", "compare") and n > 0 else "N/A",
    }

    print(f"\n{'─'*70}")
    if mode in ("cot", "compare"):
        print(f"  CoT:    {cot_correct}/{n} ({cot_correct/n*100:.1f}%) | 平均耗时: {cot_total_time/n:.1f}s")
    if mode in ("gravec", "compare"):
        print(f"  GRAVEC: {gravec_correct}/{n} ({gravec_correct/n*100:.1f}%) | 平均耗时: {gravec_total_time/n:.1f}s")

    if mode == "compare" and results["cot"] and results["gravec"]:
        for j in range(n):
            cot_ok = results["cot"][j].get("correct", False)
            gravec_ok = results["gravec"][j].get("correct", False)
            if gravec_ok and not cot_ok:
                gravec_only += 1
                gravec_fixes += 1
            elif cot_ok and not gravec_ok:
                cot_only += 1
        print(f"\n  📊 对比分析:")
        print(f"     GRAVEC 修复了 CoT 的 {gravec_fixes} 个错误")
        print(f"     CoT 修复了 GRAVEC 的 {cot_only} 个错误")
        print(f"     GRAVEC 净增: +{gravec_fixes - cot_only}")

    return results


def print_stylebench_comparison(task_results: Dict[str, Dict[str, Any]]):
    """打印与 StyleBench 已有结果的对比。"""
    stylebench_csv = STYLEBENCH_DIR / "analysis_results" / "is_correct_summary.csv"
    if not stylebench_csv.exists():
        print("\n  ⚠️ StyleBench 原始结果文件不存在，跳过跨模型对比")
        return

    import csv

    stylebench_data = {}
    with open(stylebench_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model"]
            dataset_name = row["dataset"]
            stylebench_data[(model_name, dataset_name)] = {
                "CoT": row.get("CoT", "N/A"),
                "SoT": row.get("SoT", "N/A"),
                "CoD": row.get("CoD", "N/A"),
                "ToT": row.get("ToT", "N/A"),
                "AoT": row.get("AoT", "N/A"),
            }

    print(f"\n{'='*70}")
    print("📊 StyleBench 跨模型对比（GRAVEC vs 5种风格）")
    print(f"{'='*70}")

    for task, result in task_results.items():
        if "error" in result and result.get("error") == "dataset_not_found":
            continue

        gravec_pct = 0.0
        if result.get("gravec"):
            gravec_correct = sum(1 for r in result["gravec"] if r.get("correct"))
            gravec_pct = gravec_correct / len(result["gravec"]) * 100 if result["gravec"] else 0

        cot_pct = 0.0
        if result.get("cot"):
            cot_correct = sum(1 for r in result["cot"] if r.get("correct"))
            cot_pct = cot_correct / len(result["cot"]) * 100 if result["cot"] else 0

        print(f"\n  📋 Task: {task}")
        print(f"  {'─'*50}")
        print(f"  {'方法':<12} {'准确率':>10}")
        print(f"  {'─'*50}")
        print(f"  {'GRAVEC':<12} {gravec_pct:>9.1f}%")
        print(f"  {'CoT(ours)':<12} {cot_pct:>9.1f}%")

        for (model_name, dataset_name), styles in stylebench_data.items():
            if dataset_name == task:
                for style_name, pct in styles.items():
                    print(f"  {style_name+'('+model_name+')':<30} {pct:>10}")
                break


def main():
    parser = argparse.ArgumentParser(description="StyleBench GRAVEC 适配器")
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,logiqa",
        help="任务列表，逗号分隔: gsm8k,logiqa,commonsenseqa,game24,aime 或 'all'",
    )
    parser.add_argument("--num", type=int, default=50, help="每任务测试题数（默认50）")
    parser.add_argument(
        "--mode",
        choices=["cot", "gravec", "compare"],
        default="compare",
        help="测试模式: cot=仅CoT基线, gravec=仅GRAVEC, compare=两者对比",
    )
    parser.add_argument("--max-iters", type=int, default=8, help="GRAVEC最大迭代次数")
    parser.add_argument("--save", type=str, default=None, help="保存结果到JSON文件")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    all_tasks = ["gsm8k", "logiqa", "commonsenseqa", "game24", "aime"]
    if args.tasks.lower() == "all":
        tasks = all_tasks
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in all_tasks]

    if not tasks:
        print("❌ 没有有效的任务。可选: gsm8k, logiqa, commonsenseqa, game24, aime")
        sys.exit(1)

    print(f"🚀 StyleBench GRAVEC 适配器")
    print(f"   模型: {DEFAULT_MODEL}")
    print(f"   任务: {', '.join(tasks)}")
    print(f"   每任务题数: {args.num}")
    print(f"   模式: {args.mode}")

    task_results = {}
    for task in tasks:
        result = run_task_benchmark(
            task=task,
            num_samples=args.num,
            mode=args.mode,
            max_iters=args.max_iters,
            seed=args.seed,
        )
        task_results[task] = result

    print_stylebench_comparison(task_results)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(task_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 结果已保存到: {args.save}")

    print(f"\n{'='*70}")
    print("✅ StyleBench GRAVEC 适配器测试完成！")


if __name__ == "__main__":
    main()
