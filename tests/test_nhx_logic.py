"""
================================================================================
倪海厦「以果决其行」纯逻辑单元测试
================================================================================

【费曼视角：为什么需要这个文件】
验证代码不需要真的去调 Gemini API！就像你验证一个计算器，
不需要真的去买菜——只需要输入 2+2，看输出是不是 4。

这个文件测试的是"逻辑"而非"智能"：
- 数据结构的约束是否正确（ValueScore 不能超过 1.0）
- 继承关系是否正确（GSM8KNiHaixiaSpec 是 NiHaixiaSpec 的子类）
- 验效计算是否合理（improvement → 正值，deterioration → 负值）
- 果行共变的触发条件是否正确

【运行方式】
    cd /Users/mac/WorkBuddy/Claw/rff-enhanced
    PYTHONPATH=src python3 -m pytest tests/test_nhx_logic.py -v

    或者不用 pytest：
    PYTHONPATH=src python3 tests/test_nhx_logic.py
"""
import sys
import os

os.environ.setdefault("GEMINI_API_KEY", "test_key_for_unit_tests_only")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reason_from_future.core import ProblemSpec, Workspace
from reason_from_future.core_nhx import (
    CausalDiagnosis,
    GoalRevision,
    NiHaixiaSpec,
    Observation,
    ValueScore,
    reason_from_future_nhx,
)
from reason_from_future.specs.gsm8k_nhx import GSM8KNiHaixiaSpec


# ============================================================================
# 第一层：数据结构约束测试
# ============================================================================
class TestObservation:
    def test_valid_types(self):
        for t in Observation.VALID_TYPES:
            obs = Observation(content="test", observation_type=t)
            assert obs.observation_type == t, f"类型 {t} 应该被接受"

    def test_invalid_type_defaults_to_neutral(self):
        obs = Observation(content="test", observation_type="invalid_type")
        assert obs.observation_type == "neutral"

    def test_confidence_clamped(self):
        obs_high = Observation(content="test", confidence=2.0)
        assert obs_high.confidence == 1.0, "置信度应被限制在 1.0"

        obs_low = Observation(content="test", confidence=-1.0)
        assert obs_low.confidence == 0.0, "置信度应被限制在 0.0"

        obs_ok = Observation(content="test", confidence=0.7)
        assert obs_ok.confidence == 0.7, "正常置信度不变"


class TestValueScore:
    def test_score_clamped(self):
        vs_high = ValueScore(score=5.0)
        assert vs_high.score == 1.0, "分数应被限制在 1.0"

        vs_low = ValueScore(score=-3.0)
        assert vs_low.score == -1.0, "分数应被限制在 -1.0"

    def test_primary_flag_auto_set(self):
        vs_primary = ValueScore(score=0.9)
        assert vs_primary.is_primary is True, "score >= 0.8 应自动标记为主证"

        vs_secondary = ValueScore(score=0.5)
        assert vs_secondary.is_primary is False, "score < 0.8 不应标记为主证"

    def test_negative_score_not_primary(self):
        vs_neg = ValueScore(score=-0.5, is_primary=True)
        assert vs_neg.is_primary is False, "负分不应是主证（__post_init__ 修正）"


class TestGoalRevision:
    def test_confidence_clamped(self):
        gr = GoalRevision(confidence=3.0)
        assert gr.confidence == 1.0

    def test_default_keep_old(self):
        gr = GoalRevision(revised_goal="new_goal")
        assert gr.keep_old_as_subgoal is True, "默认保留旧目标"


class TestCausalDiagnosis:
    def test_valid_types(self):
        for t in CausalDiagnosis.VALID_TYPES:
            cd = CausalDiagnosis(failure_type=t)
            assert cd.failure_type == t

    def test_invalid_type_defaults(self):
        cd = CausalDiagnosis(failure_type="invalid")
        assert cd.failure_type == "unknown"

    def test_confidence_clamped(self):
        cd = CausalDiagnosis(confidence=-5.0)
        assert cd.confidence == 0.0


# ============================================================================
# 第二层：继承关系测试
# ============================================================================
class TestInheritance:
    def test_nihaixia_spec_is_problemspec(self):
        assert issubclass(NiHaixiaSpec, ProblemSpec), \
            "NiHaixiaSpec 必须是 ProblemSpec 的子类"

    def test_gsm8k_nihaixia_spec_is_nihaixia_spec(self):
        assert issubclass(GSM8KNiHaixiaSpec, NiHaixiaSpec), \
            "GSM8KNiHaixiaSpec 必须是 NiHaixiaSpec 的子类"

    def test_gsm8k_nihaixia_spec_is_problemspec(self):
        assert issubclass(GSM8KNiHaixiaSpec, ProblemSpec), \
            "GSM8KNiHaixiaSpec 必须是 ProblemSpec 的子类"

    def test_cannot_instantiate_nihaixia_spec_directly(self):
        try:
            NiHaixiaSpec()
            assert False, "抽象类不应该能实例化"
        except TypeError:
            pass

    def test_gsm8k_nihaixia_spec_can_instantiate(self):
        spec = GSM8KNiHaixiaSpec({
            "question": "2 + 3 = ?",
            "answer": "5"
        })
        assert spec is not None
        assert spec.gold_numeric_answer == 5.0


# ============================================================================
# 第三层：GSM8KNiHaixiaSpec 原版方法测试（无需 LLM）
# ============================================================================
class TestGSM8KNiHaixiaSpecBasic:
    def setup_method(self):
        self.spec = GSM8KNiHaixiaSpec({
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 each. How much does she make every day at the farmers' market?",
            "answer": "18"
        })

    def test_derive_final_target(self):
        assert self.spec.derive_final_target("any problem") == "final_answer"

    def test_check_local_with_numeric(self):
        state = Workspace({"final_answer": 18.0})
        assert self.spec.check_local(state, "final_answer") is True

    def test_check_local_missing(self):
        state = Workspace({"other_var": 5.0})
        assert self.spec.check_local(state, "final_answer") is False

    def test_check_local_non_numeric(self):
        state = Workspace({"final_answer": "not a number"})
        assert self.spec.check_local(state, "final_answer") is False

    def test_verify_final_correct(self):
        state = Workspace({"final_answer": 18.0})
        ok, answer, gold = self.spec.verify_final(state)
        assert ok is True
        assert answer == "18.0"
        assert gold == 18.0

    def test_verify_final_wrong(self):
        state = Workspace({"final_answer": 20.0})
        ok, answer, gold = self.spec.verify_final(state)
        assert ok is False
        assert gold == 18.0

    def test_parse_workspace_update_json(self):
        raw = '{"var": "eggs_sold", "expr": "16-3-4", "value": 9}'
        result = self.spec.parse_workspace_update(raw, Workspace())
        assert "eggs_sold" in result
        assert result["eggs_sold"] == 9.0

    def test_parse_workspace_update_empty(self):
        result = self.spec.parse_workspace_update("gibberish text", Workspace())
        assert len(result) == 0

    def test_merge_aliases_basic(self):
        state = Workspace({"total_books": 100, "books_total": 100})
        merged = self.spec.merge_aliases(state)
        assert len(merged) <= 2

    def test_gold_numeric_answer_parsed(self):
        assert self.spec.gold_numeric_answer == 18.0


# ============================================================================
# 第四层：验效逻辑测试（无需 LLM）
# ============================================================================
class TestEvaluateObservationLogic:
    def setup_method(self):
        self.spec = GSM8KNiHaixiaSpec({
            "question": "2 + 3 = ?",
            "answer": "5"
        })

    def test_improvement_positive(self):
        obs = Observation(
            content="好转",
            observation_type="improvement",
            confidence=0.8,
        )
        score = self.spec.evaluate_observation(obs, Workspace(), "final_answer")
        assert score > 0, "improvement 应该产生正分"

    def test_deterioration_negative(self):
        obs = Observation(
            content="恶化",
            observation_type="deterioration",
            confidence=0.8,
        )
        score = self.spec.evaluate_observation(obs, Workspace(), "final_answer")
        assert score < 0, "deterioration 应该产生负分"

    def test_neutral_zero(self):
        obs = Observation(
            content="无变化",
            observation_type="neutral",
            confidence=0.8,
        )
        score = self.spec.evaluate_observation(obs, Workspace(), "final_answer")
        base = 0.0
        assert abs(score - base) < 0.3, "neutral 应该接近零"

    def test_closer_distance_boosts(self):
        obs = Observation(
            content="更近了",
            observation_type="improvement",
            confidence=1.0,
            data={"distance_to_goal": "closer"},
        )
        obs_far = Observation(
            content="更远了",
            observation_type="improvement",
            confidence=1.0,
            data={"distance_to_goal": "farther"},
        )
        score_close = self.spec.evaluate_observation(obs, Workspace(), "final_answer")
        score_far = self.spec.evaluate_observation(obs_far, Workspace(), "final_answer")
        assert score_close > score_far, "closer 应该比 farther 分数高"

    def test_contradictions_penalty(self):
        obs_clean = Observation(
            content="无矛盾",
            observation_type="improvement",
            confidence=1.0,
            data={"contradictions": []},
        )
        obs_contra = Observation(
            content="有矛盾",
            observation_type="improvement",
            confidence=1.0,
            data={"contradictions": ["value mismatch"]},
        )
        score_clean = self.spec.evaluate_observation(obs_clean, Workspace(), "final_answer")
        score_contra = self.spec.evaluate_observation(obs_contra, Workspace(), "final_answer")
        assert score_clean > score_contra, "矛盾应该扣分"

    def test_low_confidence_dampens(self):
        obs_high = Observation(
            content="test",
            observation_type="improvement",
            confidence=1.0,
        )
        obs_low = Observation(
            content="test",
            observation_type="improvement",
            confidence=0.1,
        )
        score_high = self.spec.evaluate_observation(obs_high, Workspace(), "final_answer")
        score_low = self.spec.evaluate_observation(obs_low, Workspace(), "final_answer")
        assert abs(score_high) > abs(score_low), "低置信度应该衰减效果分"


# ============================================================================
# 第五层：Workspace 合并逻辑测试
# ============================================================================
class TestWorkspaceMerge:
    def test_or_merge(self):
        a = Workspace({"x": 1, "y": 2})
        b = {"y": 3, "z": 4}
        c = a | b
        assert c["x"] == 1
        assert c["y"] == 3
        assert c["z"] == 4
        assert isinstance(c, Workspace)

    def test_ror_merge(self):
        a = {"x": 1}
        b = Workspace({"y": 2})
        c = a | b
        assert isinstance(c, Workspace)
        assert c["x"] == 1
        assert c["y"] == 2

    def test_add_method(self):
        ws = Workspace()
        ws.add("key", "value")
        assert ws["key"] == "value"


# ============================================================================
# 运行入口（不用 pytest 也能跑）
# ============================================================================
if __name__ == "__main__":
    import traceback

    test_classes = [
        TestObservation,
        TestValueScore,
        TestGoalRevision,
        TestCausalDiagnosis,
        TestInheritance,
        TestGSM8KNiHaixiaSpecBasic,
        TestEvaluateObservationLogic,
        TestWorkspaceMerge,
    ]

    total = 0
    passed = 0
    failed = 0

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in methods:
            total += 1
            try:
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                getattr(instance, method_name)()
                passed += 1
                print(f"  ✅ {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                print(f"  ❌ {cls.__name__}.{method_name}: {e}")
                traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"测试结果: {passed}/{total} 通过, {failed} 失败")
    if failed == 0:
        print("🎉 所有纯逻辑测试通过！无需 LLM，无需 API Key！")
    else:
        print(f"⚠️ {failed} 个测试失败，请检查")
