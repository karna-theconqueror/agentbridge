"""Week 2: scoring layer for the eval gate.

Two independent gates:

1. Functional gate — hard checks (tool used, answer contains expected text).
   Must be 100%. A miss = the feature is broken.
2. Quality gate — judge score (correctness, conciseness, groundedness,
   helpfulness) compared against evals/baseline.json. A drop below baseline
   = the feature still works but got worse (wordy, raw JSON leaks, etc.).

Judges:
- HeuristicJudge (default): deterministic, offline, free.
- LLMJudge: enabled automatically when OPENAI_API_KEY is set.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .agent import AgentResult


# ------------------------------------------------------------- hard checks

def expected_needles(case: dict) -> list[str]:
    needle = case.get("expect_contains")
    if needle is None:
        return []
    if isinstance(needle, str):
        return [needle]
    return list(needle)


def hard_check(case: dict, result: AgentResult) -> list[str]:
    """Returns a list of failure reasons (empty = pass)."""
    reasons: list[str] = []

    expected_tool = case.get("expect_tool")
    if expected_tool and expected_tool not in result.tool_names:
        reasons.append(f"expected tool '{expected_tool}', used {result.tool_names or 'none'}")

    for needle in expected_needles(case):
        if needle.lower() not in result.answer.lower():
            reasons.append(f"answer missing {needle!r}")

    return reasons


# ------------------------------------------------------------- judges

CRITERIA = ("correctness", "conciseness", "groundedness", "helpfulness")


class HeuristicJudge:
    """Deterministic rubric scorer — the offline default."""

    name = "heuristic"

    def score(self, case: dict, result: AgentResult) -> dict[str, float]:
        answer = result.answer or ""
        lower = answer.lower()

        # 1. correctness: every expected needle is present
        needles = expected_needles(case)
        scores = {
            "correctness": 1.0 if (not needles or all(n.lower() in lower for n in needles))
            else 0.0
        }

        # 2. conciseness: non-empty and not a wall of text
        scores["conciseness"] = 1.0 if 0 < len(answer) <= 400 else 0.0

        # 3. groundedness: no raw JSON blob leaking into the final answer,
        #    no internal error markers
        raw_json = "{" in answer and "}" in answer and '":' in answer
        internal_error = "ERROR" in answer or "unknown tool" in lower
        scores["groundedness"] = 0.0 if (raw_json or internal_error) else 1.0

        # 4. helpfulness: tool results summarized as complete sentences;
        #    free-form answers say something substantive
        if case.get("expect_tool"):
            scores["helpfulness"] = 1.0 if answer.rstrip().endswith((".", "!", "?")) else 0.0
        else:
            scores["helpfulness"] = 1.0 if len(answer.split()) >= 3 else 0.0

        return scores


class LLMJudge:
    """LLM-as-judge: used automatically when OPENAI_API_KEY is set."""

    name = "llm"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # optional dependency
        self.client = OpenAI()
        self.model = model

    def score(self, case: dict, result: AgentResult) -> dict[str, float]:
        prompt = (
            "Score this agent answer 1.0 or 0.0 for each criterion. "
            "Reply as JSON only: {\"correctness\":..,\"conciseness\":..,"
            "\"groundedness\":..,\"helpfulness\":..}\n\n"
            f"Task: {case['input']}\n"
            f"Expected to contain: {expected_needles(case)}\n"
            f"Answer: {result.answer}\n"
            "correctness=expected content present; conciseness=short and direct; "
            "groundedness=no invented facts or raw JSON; helpfulness=complete, "
            "useful sentence."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = resp.choices[0].message.content or "{}"
        raw = raw[raw.find("{"): raw.rfind("}") + 1] or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        return {c: 1.0 if float(data.get(c, 0)) >= 0.5 else 0.0 for c in CRITERIA}


def get_judge():
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return LLMJudge()
        except Exception:
            pass
    return HeuristicJudge()


# ------------------------------------------------------------- aggregation

@dataclass
class CaseOutcome:
    case_id: str
    hard_failures: list[str]
    scores: dict[str, float]

    @property
    def passed(self) -> bool:
        return not self.hard_failures

    @property
    def quality(self) -> float:
        return sum(self.scores.values()) / len(self.scores) if self.scores else 0.0


def quality_gate(score: float, baseline: float) -> bool:
    """Quality must not regress below the recorded baseline."""
    return score >= baseline - 1e-9


def summarize(outcomes: list[CaseOutcome]) -> dict:
    hard_passed = sum(1 for o in outcomes if o.passed)
    quality = sum(o.quality for o in outcomes) / len(outcomes) if outcomes else 0.0
    return {
        "cases": len(outcomes),
        "hard_passed": hard_passed,
        "hard_score": hard_passed / len(outcomes) if outcomes else 0.0,
        "quality_score": quality,
        "per_criterion": {
            c: sum(o.scores.get(c, 0.0) for o in outcomes) / len(outcomes)
            for c in CRITERIA
        },
    }
