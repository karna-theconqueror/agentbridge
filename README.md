# AgentBridge — Agentic-AI Pilot-to-Production Accelerator

A toolkit that takes a prototype LLM agent and makes it production-ready:
**evaluation before deployment, guardrails during runtime, observability after.**

Built as the portfolio project aligned to Tech Mahindra's "AI Delivered Right"
strategy (see `TechMahindra_Project_Blueprint.pdf`).

## Quick start (no API key needed)

```bash
python run_demo.py            # run the agent on sample tasks
python run_evals.py           # run the golden eval suite (this is the CI gate)
python -m pytest              # unit tests
```

The default provider is a deterministic **MockLLM** so everything runs offline.
Drop in a real model by setting an env var:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-..."
python run_demo.py            # auto-detects the key and switches provider
```

## What's inside (mapped to the 8-week plan)

| Module | Status | Week |
|---|---|---|
| `agentbridge/agent.py` — tool-calling agent loop with trace recording | ✅ done | 1 |
| `agentbridge/llm.py` — pluggable providers (Mock / OpenAI / Anthropic) | ✅ done | 1 |
| `agentbridge/tools.py` — tool registry + sample domain tools | ✅ done | 1 |
| `evals/golden_set.json` + `run_evals.py` — golden eval harness (20 cases) | ✅ v2 | 2 |
| Two gates: functional (100%) + quality vs `evals/baseline.json` | ✅ | 2 |
| `agentbridge/scoring.py` — heuristic judge + optional LLM-as-judge | ✅ | 2 |
| `tests/` — pytest suite (32 tests, CI-ready) | ✅ v3 | 2–3 |
| `agentbridge/guardrails.py` — injection blocking, PII redaction, tool-output schema validation | ✅ | 3 |
| Guardrails wired into agent loop (input / tool-output / answer checkpoints) | ✅ | 3 |
| Guardrails: PII redaction, injection detection, schema validation | ✅ | 3 |
| Human-in-the-loop approval + CI gate on quality drop | ⬜ | 4 |
| Observability dashboard (Langfuse): traces, cost, latency | ⬜ | 5 |
| FastAPI + Docker packaging | ⬜ | 6 |
| Streamlit/React dashboard + domain demo pack | ⬜ | 7 |
| README polish, architecture diagram, case study | ⬜ | 8 |

## Architecture (v1)

```
user input ──▶ Agent.run() ──▶ LLM provider (Mock | OpenAI | Anthropic)
                    │                 │
                    │◀── tool_call ───┘
                    ▼
              tool registry (calculator | get_claim_status)
                    │
                    ▼
              trace: [step, step, ...]   (feeds observability in Week 5)
                    │
                    ▼
              final answer + trace
```

## Repository conventions

- Plain Python 3.10+, stdlib only for the core (real providers are optional extras).
- `run_evals.py` runs **two gates** and exits non-zero if either fails:
  1. *Functional* — every case must pass its hard checks (100% required).
  2. *Quality* — judge score (correctness, conciseness, groundedness, helpfulness)
     must not drop below `evals/baseline.json`. Accept a new floor with
     `python run_evals.py --update-baseline`.
- Judge is the deterministic heuristic offline; set `OPENAI_API_KEY` to
  switch to an LLM-as-judge automatically.
- Every change: `python run_evals.py && python -m pytest` must pass.

## Deploy locally (Docker + FastAPI)

```bash
docker build -t agentbridge .
docker run -p 8000:8000 agentbridge
```

FastAPI exposes:

- `GET /health` → `{"status":"ok"}`
- `POST /invoke` with JSON `{ "prompt": "...", "hitl": true/false }`

Example:

```bash
curl -X POST http://localhost:8000/invoke \
     -H "Content-Type: application/json" \
     -d '{"prompt":"What is 125 * 8?"}'
```

---

## Roadmap

See Section 6 and Section 9 of `TechMahindra_Project_Blueprint.pdf`.
