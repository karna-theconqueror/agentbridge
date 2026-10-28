"""Week 5 tests: observability (TraceLogger) + /metrics endpoint wiring."""
import json

from agentbridge.agent import AgentResult, TraceStep
from agentbridge.observability import TraceLogger


def make_result(blocked=False, tools=(), guardrails=(), latency=10.0):
    trace = [TraceStep(kind="llm", detail="thinking", latency_ms=latency)]
    for t in tools:
        trace.append(TraceStep(kind="tool", name=t, detail="{}", latency_ms=latency))
    for g in guardrails:
        trace.append(TraceStep(kind="guardrail", detail=g, latency_ms=0.0))
    trace.append(TraceStep(kind="final", detail="answer", latency_ms=latency))
    return AgentResult(answer="answer", trace=trace, iterations=1, blocked=blocked)


def test_record_and_metrics(tmp_path):
    logger = TraceLogger(log_path=str(tmp_path / "traces.jsonl"))
    logger.record("prompt 1", make_result(tools=["calculator"], latency=20.0))
    logger.record("prompt 2", make_result(blocked=True, guardrails=["BLOCKED input: injection"]))

    m = logger.metrics()
    assert m["total_runs"] == 2
    assert m["blocked_runs"] == 1
    assert m["tool_usage"] == {"calculator": 1}
    assert "BLOCKED" in m["guardrail_events"]
    assert m["latency_ms"]["max"] >= m["latency_ms"]["p50"] >= 0


def test_persists_jsonl(tmp_path):
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(log_path=str(path))
    logger.record("p", make_result())
    logger.record("p2", make_result())

    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    assert lines[0]["prompt"] == "p"
    assert "run_id" in lines[0]


def test_empty_metrics(tmp_path):
    logger = TraceLogger(log_path=str(tmp_path / "t.jsonl"))
    m = logger.metrics()
    assert m["total_runs"] == 0
    assert m["latency_ms"]["p50"] == 0.0
