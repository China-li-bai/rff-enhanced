"""
================================================================================
倪海厦版 RFF Demo — demo_gsm8k_nhx.py
================================================================================

【费曼视角：一句话讲清楚】
这是倪海厦版 RFF 的快速体验入口。选一道数学题，看六步曲怎么跑。

【运行方式】
    export GEMINI_API_KEY="your_key_here"
    cd /Users/mac/WorkBuddy/Claw/rff-enhanced
    PYTHONPATH=src python3 src/demos/demo_gsm8k_nhx.py
"""
from reason_from_future.core_nhx import reason_from_future_nhx
from reason_from_future.specs.gsm8k_nhx import GSM8KNiHaixiaSpec


simple = {
    "question": "There were 15 trees in the grove. 3 were cut down. Then, after some time, 2 more were cut down. But 1 grew back. How many are left?",
    "answer": "11",
}

medium = {
    "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 each. How much does she make every day at the farmers' market?",
    "answer": "18",
}

hard = {
    "question": """A school library ordered 600 new books for three new sections: Fiction, Science, and History. Exactly half of the order was Fiction. The remaining books were split equally between Science and History.
    During shipping the library discovered that
    10 percent of the Fiction books and 10 percent of the Science books were damaged, and
    15 History books were lost.

    The library reordered one replacement copy for every damaged or lost book.
    As a thank-you, the supplier added bonus copies equal to 40 percent of the total number of replacements, and half of these bonus copies were Science books.

    Then, the library received a donation of science books equal to 50 percent of the number of Science books it already had, but 5/9 of those were stolen.

    After the replacements and bonus copies arrived, how many Science books did the library have in total?""",
    "answer": "198",
}

CURRENT_SAMPLE = medium


def main(verbose=True):
    print("=" * 60)
    print("倪海厦「以果决其行」RFF Demo")
    print("=" * 60)
    print(f"\n题目: {CURRENT_SAMPLE['question'][:80]}...")
    print(f"标准答案: {CURRENT_SAMPLE['answer']}")
    print()

    spec = GSM8KNiHaixiaSpec(CURRENT_SAMPLE)

    answer = reason_from_future_nhx(
        problem=CURRENT_SAMPLE["question"],
        spec=spec,
        max_iters=10,
        verbose=verbose,
        require_gold=False,
        min_iters=2,
    )

    print(f"\n{'='*60}")
    print(f"最终答案: {answer}")
    print(f"标准答案: {CURRENT_SAMPLE['answer']}")
    try:
        numeric = float(answer.replace(",", ""))
        gold = float(CURRENT_SAMPLE["answer"].replace(",", ""))
        if abs(numeric - gold) < 1e-5:
            print("🎉 正确！")
        else:
            print(f"❌ 错误 (差值: {abs(numeric - gold)})")
    except (ValueError, TypeError):
        print("⚠️ 无法解析为数字")


if __name__ == "__main__":
    main()
