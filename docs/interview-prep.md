# AgentBridge — Interview Prep (Deep-Dive Q&A)

Hard questions an interviewer might ask about this project, with strong answers.
Use these to sound like someone who *built and reasoned about* the system, not
someone who generated it.

---

## Architecture & design

**Q: Walk me through what happens when a request hits `/invoke`.**
A: The prompt goes in. First the input guardrail checks for prompt injection
(block) and PII (redact). Then the LLM decides whether to call a tool or answer.
If it wants a tool, the HITL gate checks the approval policy for that tool —
deny by default. If approved, the tool runs and its output is validated against
its schema before it goes back into the model's context. The loop repeats until
the model produces a final answer, which passes through the output guardrail.
Every step is recorded as a trace step and persisted by the TraceLogger.

**Q: Why a two-gate eval instead of one accuracy number?**
A: Because they catch different failures. The functional gate is binary and hard —
every case must pass its checks, so a tool that stops working fails immediately.
The quality gate is graded and comparative — it scores correctness, conciseness,
groundedness, helpfulness against a recorded baseline, so a change that technically
"works" but produces worse, rambling, or ungrounded answers also fails. One catches
breakage; the other catches degradation.

**Q: How do you keep the quality gate from being gamed or drifting?**
A: The baseline is a recorded floor, updated only deliberately with
`--update-baseline` after a human reviews the change — it's not auto-tuned. The
judge is deterministic offline (heuristic) so results are reproducible; set an API
key and it swaps to an LLM-as-judge. Determinism matters: a flaky gate trains people
to ignore it.

**Q: Why did you build your own eval harness instead of using an off-the-shelf one?**
A: For a portfolio piece, building it forces me to understand what an eval gate
actually is — a hard functional check plus a graded quality check against a baseline.
Off-the-shelf tools (Ragas, Braintrust, LangSmith) are the right call in production;
I'd reach for them next. But the core idea — gate on regression, don't trust vibes —
is the same.

## Safety & guardrails

**Q: How does the prompt-injection guardrail actually work? What's its weakness?**
A: It's pattern-based — regexes for known injection phrases ("ignore previous
instructions", "reveal your system prompt", etc.), plus structural heuristics. Its
honest weakness: pattern-matching can't catch novel or obfuscated injections. In
production I'd add a classifier-based detector and, more importantly, treat the
guardrail as defense-in-depth — not the only line. The real protection is that tools
are schema-validated and high-risk ones need HITL, so even a successful injection has
a limited blast radius.

**Q: Why is HITL deny-by-default?**
A: For high-risk tools, the failure mode of a wrong approval is worse than the failure
mode of a blocked legitimate call. Deny-by-default means a misconfiguration fails safe.
You can widen the policy deliberately once you trust the tool, but you can't un-send an
email or un-refund money.

**Q: Tell me about a bug you found in this project.**
A: The one I'm proudest of: the HITL policy was stored on a shared agent object and only
reset when the request explicitly passed a `hitl` flag. So after one request approved the
calculator tool, that approval leaked into every subsequent request — a safety hole. I
caught it in live testing (a request with no flag came back unblocked), fixed it by
resetting the policy on every request, and added a regression test so it can't come back.

## Observability & ops

**Q: What does your observability layer give you that print statements don't?**
A: Aggregates and persistence. Every run becomes a structured record (trace, tools used,
guardrail events, latency, blocked status) appended to JSONL, plus in-memory aggregates:
total/blocked runs, tool-usage counts, guardrail-event buckets, and latency p50/p95/max.
That's what lets you answer "is the agent getting slower?" or "how often are guardrails
firing?" — questions print statements can't.

**Q: How would you take this to real production?**
A: Swap the JSONL logger for a real backend (OpenTelemetry → Datadog/Grafana, or
Langfuse/Phoenix for LLM-specific traces). Add token/cost tracking per run. Put the
service behind auth and rate limiting. Run evals on a schedule and on every model-prompt
change, not just in CI. Add streaming responses and a real queue for tool calls. And
replace the heuristic judge with an LLM-as-judge calibrated against human labels.

**Q: The latency numbers are sub-millisecond. Is that real?**
A: Those are the MockLLM provider — deterministic and offline, so tool execution and
loop overhead are microseconds. The latency plumbing (per-step timing, p50/p95/max
aggregation) is real and provider-agnostic; with a real model the p95 would be dominated
by LLM call time, which is exactly what the instrumentation is there to measure.

## Trade-offs & self-critique

**Q: What's the weakest part of this project?**
A: The guardrails are pattern-based, so they're demonstrable but not robust against a
determined adversary — I'd say that plainly rather than oversell it. The eval set is 22
hand-written cases, which proves the harness works but isn't statistically strong. And
the observability store is in-memory + JSONL, fine for a demo, not for many concurrent
users. All three are known, documented trade-offs, not accidents.

**Q: If you had two more weeks, what would you build?**
A: A second domain demo (telco/insurance) to prove the framework generalizes beyond
calculator + claims. Streaming responses. Token/cost accounting in the metrics. An
LLM-judge calibration against a small human-labeled set. And a real trace backend so
it survives a restart under load.

**Q: Why stdlib-only core?**
A: To keep the reasoning visible and the barrier to entry zero — a reviewer can read
the whole agent loop and eval gate without installing a framework. It's a teaching
choice as much as an engineering one. In production I'd happily use established libs;
the point here was to demonstrate I understand what's inside them.
