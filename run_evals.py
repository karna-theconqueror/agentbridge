"""Golden eval harness — the CI gate.

Runs every case in evals/golden_set.json and exits non-zero on any failure,
so a regression in prompts/tools/agent logic blocks the deploy.

    python run_evals.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from agentbridge.agent import Agent

GOLDEN_PATH = Path(__file__).parent / "evals" / "golden_set.json"


def load_cases() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def run_case(agent: Agent, case: dict) -> tuple[bool, str]:
    result = agent.run(case["input"])

    expected_tool = case.get("expect_tool")
    if expected_tool:
        if expected_tool not in result.tool_names:
            return False, f"expected tool '{expected_tool}', used {result.tool_names}"

    needle = case.get("expect_contains")
    if needle and needle.lower() not in result.answer.lower():
        return False, f"answer missing '{needle}': {result.answer!r}"

    return True, "ok"


def main() -> int:
    cases = load_cases()
    agent = Agent()
    passed = 0

    print(f"Running {len(cases)} golden cases (provider: {agent.provider.name})")
    print("-" * 60)
    for case in cases:
        ok, detail = run_case(agent, case)
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {case['id']:16} {detail}")

    score = passed / len(cases)
    print("-" * 60)
    print(f"Score: {passed}/{len(cases)} ({score:.0%})")

    if score < 1.0:
        print("GATE FAILED — fix before merging.")
        return 1
    print("GATE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
