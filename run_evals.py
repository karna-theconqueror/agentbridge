"""Golden eval harness — the CI gate (Week 2: two gates).

Gate 1 (functional): every case must pass its hard checks (tool + answer
content). Any miss exits non-zero.

Gate 2 (quality): the judge's mean quality score must not fall below the
baseline recorded in evals/baseline.json. This catches "still works but got
worse" regressions — wordy answers, raw JSON leaking, lost punctuation.

    python run_evals.py                   # run both gates
    python run_evals.py --update-baseline # accept current quality as the new floor

Judge: heuristic by default (offline, deterministic); switches to an LLM
judge automatically when OPENAI_API_KEY is set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from agentbridge.agent import Agent
from agentbridge.llm import MockLLM
from agentbridge.scoring import (
    CaseOutcome,
    get_judge,
    hard_check,
    quality_gate,
    summarize,
)

GOLDEN_PATH = Path(__file__).parent / "evals" / "golden_set.json"
BASELINE_PATH = Path(__file__).parent / "evals" / "baseline.json"


def load_cases() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def load_baseline() -> float | None:
    if not BASELINE_PATH.exists():
        return None
    return float(json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["min_quality"])


def save_baseline(score: float) -> None:
    BASELINE_PATH.write_text(
        json.dumps({"min_quality": round(score, 4)}, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str]) -> int:
    update_baseline = "--update-baseline" in argv
    # Evals are pinned to the offline mock so the CI gate is deterministic and
    # never hits the network — even when GEMINI_API_KEY is set for the API.
    # Opt into a real model with: python run_evals.py --live
    use_live = "--live" in argv

    cases = load_cases()
    agent = Agent() if use_live else Agent(provider=MockLLM())
    judge = get_judge()

    print(f"Running {len(cases)} golden cases")
    print(f"  provider : {agent.provider.name}{' (live)' if use_live else ''}")
    print(f"  judge    : {judge.name}")
    print("-" * 72)

    outcomes: list[CaseOutcome] = []
    for case in cases:
        result = agent.run(case["input"])
        failures = hard_check(case, result)
        scores = judge.score(case, result)
        outcomes.append(CaseOutcome(case_id=case["id"],
                                    hard_failures=failures, scores=scores))
        mark = "PASS" if not failures else "FAIL"
        detail = "; ".join(failures) if failures else \
            f"tools={result.tool_names or 'none'} quality={outcomes[-1].quality:.2f}"
        print(f"[{mark}] {case['id']:24} {detail}")

    stats = summarize(outcomes)
    print("-" * 72)
    print(f"Gate 1 (functional) : {stats['hard_passed']}/{stats['cases']} "
          f"({stats['hard_score']:.0%})")
    crit = "  ".join(f"{k}={v:.2f}" for k, v in stats["per_criterion"].items())
    print(f"Gate 2 (quality)    : {stats['quality_score']:.3f}   [{crit}]")

    exit_code = 0

    if stats["hard_score"] < 1.0:
        print("GATE 1 FAILED — functional regression, fix before merging.")
        exit_code = 1

    baseline = load_baseline()
    if update_baseline:
        save_baseline(stats["quality_score"])
        print(f"Baseline updated to {stats['quality_score']:.3f}")
    elif baseline is None:
        print("No quality baseline yet — run with --update-baseline to record one.")
    elif quality_gate(stats["quality_score"], baseline):
        print(f"Gate 2 passed — quality {stats['quality_score']:.3f} >= baseline {baseline:.3f}.")
    else:
        print(f"GATE 2 FAILED — quality {stats['quality_score']:.3f} dropped below "
              f"baseline {baseline:.3f}.")
        exit_code = 1

    if exit_code == 0:
        print("ALL GATES PASSED.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
