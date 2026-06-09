"""
MATH-500 专用 NiHaixiaSpec（继承 GSM8KNiHaixiaSpec，包装 MATH 数据为 GSM8K 格式）。

MATH-500 数据格式：
    problem: str
    solution: str
    answer: str         (LaTeX, e.g. "2", "\\frac{14}{3}")
    level: int
    subject: str
    unique_id: str

GSM8K 期望的格式：
    question: str
    answer: str         (e.g. "#### 18", "42", "2.5")

包装策略：
    1. 在调用方（benchmark 脚本）预先把 MATH 答案解析为 float → gold_numeric
    2. 此 spec 把 gold_numeric 转成纯数字字符串作为 GSM8K 的 answer
    3. GSM8K 的 _NUMBER_RE 正则能直接匹配到该数字
    4. GSM8K 的所有 V/A/E/C 逻辑原样复用
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reason_from_future.specs.gsm8k_nhx import GSM8KNiHaixiaSpec


class MATHNiHaixiaSpec(GSM8KNiHaixiaSpec):
    """MATH-500 题目专用的 GRAVEC NiHaixiaSpec。"""

    def __init__(self, problem_data: Dict[str, Any]):
        gold_num = problem_data.get("gold_numeric")
        if gold_num is None:
            raise ValueError(
                "problem_data['gold_numeric'] 必填（由 benchmark 脚本预解析）。"
            )
        # GSM8K 的 _NUMBER_RE 会从 answer 字符串里找数字
        # 直接传 str(float) 确保只匹配到一个数字
        gsm8k_format: Dict[str, str] = {
            "question": problem_data["problem"],
            "answer": f"{gold_num:g}",
        }
        super().__init__(gsm8k_format)
        # 保留 MATH 元数据
        self.level = problem_data.get("level", 0)
        self.subject = problem_data.get("subject", "")
        self.unique_id = problem_data.get("unique_id", "")
        self.solution = problem_data.get("solution", "")
        self.gold_numeric: float = float(gold_num)


if __name__ == "__main__":
    # 简单自检
    test_data = {
        "problem": "If $x = 2$, what is $x^2$?",
        "answer": "4",
        "gold_numeric": 4.0,
        "level": 1,
        "subject": "Algebra",
        "unique_id": "test-1",
    }
    spec = MATHNiHaixiaSpec(test_data)
    print(f"question: {spec.question}")
    print(f"gold_numeric_answer (parsed): {spec.gold_numeric_answer}")
    print(f"gold_numeric: {spec.gold_numeric}")
    print(f"level: {spec.level}, subject: {spec.subject}")
