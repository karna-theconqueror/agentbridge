"""Week 5: Observability.

A lightweight, dependency-free observability layer that:

1. Persists every agent run (trace, latency, tools used, guardrail events,
   blocked status) as a JSON line in `logs/traces.jsonl`.
2. Provides fast aggregate stats for the `/metrics` endpoint: call volume,
   tool usage, guardrail hits, blocked count, latency percentiles.

This is the same data you would ship to Langfuse/Phoenix/Datadog in a real
deployment — the interface is intentionally provider-agnostic.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .agent import AgentResult, TraceStep


@dataclass
class RunRecord:
    """One agent invocation, persisted to the trace log."""
    run_id: str
    ts: float
    prompt: str
    answer: str
    blocked: bool
    iterations: int
    total_latency_ms: float
    tools_used: list[str] = field(default_factory=list)
    guardrail_events: list[str] = field(default_factory=list)


class TraceLogger:
    """Append-only JSONL trace log + in-memory aggregation.

    Thread-safe: FastAPI runs sync endpoints in a threadpool.
    """

    def __init__(self, log_path: str = "logs/traces.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # Aggregated counters, kept in memory for cheap /metrics reads.
        self._runs = 0
        self._blocked = 0
        self._tool_counts: dict[str, int] = {}
        self._guardrail_counts: dict[str, int] = {}
        self._latencies: list[float] = []

    def record(self, prompt: str, result: AgentResult) -> RunRecord:
        total_latency = sum(s.latency_ms for s in result.trace)
        tools = [s.name for s in result.trace if s.kind == "tool"]
        guardrails = [s.detail for s in result.trace if s.kind == "guardrail"]
        rec = RunRecord(
            run_id=uuid.uuid4().hex[:12],
            ts=time.time(),
            prompt=prompt,
            answer=result.answer,
            blocked=result.blocked,
            iterations=result.iterations,
            total_latency_ms=round(total_latency, 2),
            tools_used=tools,
            guardrail_events=guardrails,
        )
        with self._lock:
            self._runs += 1
            if result.blocked:
                self._blocked += 1
            for t in tools:
                self._tool_counts[t] = self._tool_counts.get(t, 0) + 1
            # Bucket guardrail events by kind (first word of the detail).
            for g in guardrails:
                kind = g.split(" ", 1)[0].rstrip(":")
                self._guardrail_counts[kind] = self._guardrail_counts.get(kind, 0) + 1
            self._latencies.append(total_latency)
            # Persist (append one JSON line per run).
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__) + "\n")
        return rec

    def metrics(self) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            n = len(lat)
            def pct(p: float) -> float:
                if n == 0:
                    return 0.0
                idx = min(n - 1, int(p / 100 * n))
                return round(lat[idx], 2)
            return {
                "total_runs": self._runs,
                "blocked_runs": self._blocked,
                "tool_usage": dict(sorted(self._tool_counts.items())),
                "guardrail_events": dict(sorted(self._guardrail_counts.items())),
                "latency_ms": {
                    "p50": pct(50),
                    "p95": pct(95),
                    "max": round(lat[-1], 2) if n else 0.0,
                },
            }
