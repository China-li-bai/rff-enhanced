"""AIME 专用 NiHaixiaSpec（继承 GSM8KNiHaixiaSpec，适配 AIME 数据格式）。

AIME 数据格式：
    id: str          (e.g. "aime24_60")
    problem: str     (LaTeX 格式题目)
    answer: int      (0-999 整数)
    source: str      ("aime24" / "aime25")

AIME 答案全部是 0-999 的整数，天然适合数值评测。
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from reason_from_future.specs.gsm8k_nhx import GSM8KNiHaixiaSpec


class AIMENiHaixiaSpec(GSM8KNiHaixiaSpec):
    """AIME 题目专用的 GRAVEC NiHaixiaSpec。"""

    def __init__(self, problem_data: Dict[str, Any]):
        gold_num = problem_data.get("answer")
        if gold_num is None:
            raise ValueError("problem_data['answer'] 必填。")
        # AIME 答案是 0-999 的整数
        gold_int = int(gold_num)
        gsm8k_format: Dict[str, str] = {
            "question": problem_data["problem"],
            "answer": str(gold_int),
        }
        super().__init__(gsm8k_format)
        self.aime_id = problem_data.get("id", "")
        self.source = problem_data.get("source", "")
        self.gold_numeric: float = float(gold_int)
