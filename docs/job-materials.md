# AgentBridge — Job Materials

Tailored resume bullets, a LinkedIn summary, and interview talking points for
the AgentBridge portfolio project. Map every claim to something you can
demonstrate live or point to in the repo.

---

## 1. Resume bullets

Pick 2–3 depending on the role. Each is written in the X → Y → Z style
(what you built, what it does, what result it produces).

### For AI/ML Engineer, GenAI, or LLM-application roles

- Designed and built **AgentBridge**, an agentic-AI pilot-to-production accelerator
  (Python/FastAPI) that takes a prototype LLM agent to a production-ready service —
  with a **two-gate evaluation harness** (functional + quality-vs-baseline), runtime
  **guardrails**, **human-in-the-loop** tool approval, and an **observability** layer
  (JSONL traces + aggregate `/metrics`). **41 tests, 22/22 golden evals at quality 1.000.**

- Implemented a **golden-set eval harness** with two CI gates — a hard functional gate
  (100% required) and a **quality gate** that scores correctness / conciseness /
  groundedness / helpfulness against a recorded baseline, blocking any regression —
  wired into **GitHub Actions** so a drop in quality fails the build.

- Built layered **guardrails** for LLM agents — prompt-injection detection, PII
  redaction (email/SSN), and tool-output **schema validation** — applied at input,
  tool-output, and answer checkpoints, cutting unsafe/leaked responses before they
  reach the user.

- Added a **human-in-the-loop** approval gate for high-risk tool calls with a
  deny-by-default policy, plus an **observability** layer (per-run traces, tool-usage
  and guardrail-event counts, latency p50/p95/max) surfaced through a live FastAPI
  dashboard.

### For Platform / Backend / MLOps-leaning roles

- Containerized and deployed an **LLM agent service** (FastAPI + Docker + uvicorn)
  exposing `/invoke`, `/health`, and `/metrics`, with a live **observability
  dashboard** — from zero to a running, tested service.

- Ship-it discipline: **pytest (41 tests)** + **eval gates** in CI, JSONL run-trace
  logging, and a metrics endpoint (latency percentiles, tool usage, guardrail hits)
  — the operational backbone that turns an AI demo into something you can run.

---

## 2. LinkedIn "About" / project post (short)

> I built **AgentBridge** — an agentic-AI pilot-to-production accelerator.
>
> Most LLM agents die in the demo phase. AgentBridge is the layer that gets them
> to production: a **two-gate eval harness** (functional + quality-vs-baseline) that
> blocks regressions in CI, layered **guardrails** (prompt-injection + PII redaction +
> tool-output schema validation), **human-in-the-loop** approval for high-risk tool
> calls, and an **observability** layer with a live dashboard.
>
> Stack: Python · FastAPI · Docker · GitHub Actions. 41 tests · 22/22 golden evals ·
> quality 1.000. Built offline-friendly (deterministic MockLLM) with a pluggable
> provider for real models.
>
> Repo: https://github.com/karna-theconqueror/agentbridge

---

## 3. Interview talking points (the "tell me about this project" answer)

**30-second version:**
"AgentBridge is a framework that makes prototype LLM agents production-ready.
It adds four things a demo doesn't have: an evaluation gate so you don't ship a
quality regression, runtime guardrails so the agent can't be injected on or leak
PII, human approval on high-risk tool calls, and observability so you can see
what every run actually did."

**Go deeper on whichever they bite on:**

- **Evals** — "I use two gates. A functional gate where every golden case must pass
  its hard checks (100%). And a quality gate that scores each answer on correctness,
  conciseness, groundedness, helpfulness against a recorded baseline — so if a change
  makes answers worse, CI fails instead of shipping. That's the difference between
  'it works on my example' and 'it's measurably not regressing.'"

- **Guardrails** — "Three layers, checked at three points: input, tool output, and
  final answer. Prompt-injection patterns block the request, PII like emails and SSNs
  gets redacted, and tool output is validated against its schema so a malformed tool
  result can't poison the model's context."

- **HITL** — "High-risk tools (say, anything that spends money or sends a message)
  require human approval before execution, deny-by-default. I made sure the policy
  resets per request so one approval can't leak into the next call — I actually caught
  that state-leak bug in live testing."

- **Observability** — "Every step of every run is a trace step, persisted to JSONL and
  aggregated into a metrics endpoint: runs, blocked runs, tool usage, guardrail events,
  and latency percentiles. A dashboard reads it live."

---

## 4. How to demo it live (for a video call)

1. `python run_demo.py` — show the trace: guardrail redaction, tool call, final answer.
2. `python run_evals.py` — show the two gates passing (22/22, quality 1.000).
3. `uvicorn app:app --port 8000` → open the dashboard, run a prompt, watch metrics update.
4. Show a **blocked** injection attempt and a **blocked** HITL call — safety is memorable.
