"""Week 2 tests for the scoring layer."""
import json

from agentbridge.agent import Agent, AgentResult, TraceStep
from agentbridge.llm import MockLLM
from agentbridge.scoring import (
    CaseOutcome,
    HeuristicJudge,
    hard_check,
    quality_gate,
    summarize,
)


def _result(answer, tools=()):
    return AgentResult(answer=answer,
                       trace=[TraceStep(kind="tool", name=t) for t in tools])


def test_hard_check_pass():
    case = {"expect_tool": "calculator", "expect_contains": ["1000"]}
    assert hard_check(case, _result("The result is 1000.", ["calculator"])) == []


def test_hard_check_wrong_tool():
    case = {"expect_tool": "calculator", "expect_contains": ["1000"]}
    reasons = hard_check(case, _result("The result is 1000."))
    assert reasons and "expected tool" in reasons[0]


def test_hard_check_missing_needle():
    case = {"expect_tool": None, "expect_contains": ["approved"]}
    reasons = hard_check(case, _result("no idea"))
    assert reasons and "missing" in reasons[0]


def test_judge_penalises_raw_json_leak():
    judge = HeuristicJudge()
    case = {"expect_tool": "calculator", "expect_contains": ["1000"]}
    scores = judge.score(case, _result('{"expression": "125 * 8", "result": 1000}', ["calculator"]))
    assert scores["groundedness"] == 0.0


def test_judge_rewards_clean_sentence():
    judge = HeuristicJudge()
    case = {"expect_tool": "calculator", "expect_contains": ["1000"]}
    scores = judge.score(case, _result("The result is 1000.", ["calculator"]))
    assert all(v == 1.0 for v in scores.values())


def test_quality_gate_logic():
    assert quality_gate(0.95, 0.90) is True
    assert quality_gate(0.90, 0.90) is True
    assert quality_gate(0.85, 0.90) is False


def test_summarize_counts_and_quality():
    outcomes = [
        CaseOutcome("a", [], {"correctness": 1.0, "conciseness": 1.0,
                              "groundedness": 1.0, "helpfulness": 1.0}),
        CaseOutcome("b", ["boom"], {"correctness": 0.0, "conciseness": 1.0,
                                    "groundedness": 1.0, "helpfulness": 1.0}),
    ]
    stats = summarize(outcomes)
    assert stats["cases"] == 2 and stats["hard_passed"] == 1
    assert 0.8 < stats["quality_score"] < 0.9


def test_mock_end_to_end_answers_are_speakable():
    """The full pipeline (agent -> mock formatting) must never emit raw JSON."""
    agent = Agent(provider=MockLLM())
    for task in ("What is 125 * 8?", "Check claim #CLM-1001"):
        answer = agent.run(task).answer
        assert '"result"' not in answer and '":' not in answer
        assert answer.endswith((".", "!", "?"))


def test_golden_set_is_well_formed():
    cases = json.loads(open("evals/golden_set.json", encoding="utf-8").read())
    assert len(cases) >= 20
    for c in cases:
        assert set(c) >= {"id", "input", "expect_tool", "expect_contains"}
