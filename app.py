from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from agentbridge.hitl import Hitl, AgentWithHitl
from agentbridge.observability import TraceLogger

app = FastAPI(title="AgentBridge API")

agent = AgentWithHitl()          # default HITL policy set per-request below
trace_logger = TraceLogger()     # Week 5: observability layer


class PromptRequest(BaseModel):
    prompt: str
    hitl: bool | None = None  # True = human approval gate enabled for the call


@app.post("/invoke")
def invoke(req: PromptRequest):
    # HITL policy is reset on EVERY request so state can't leak between calls.
    # - hitl omitted / False → calculator is blocked (default, deny-by-default).
    # - hitl=True           → human gate approves calculator (demo approval).
    approved = bool(req.hitl)
    agent.hitl = Hitl({"calculator": lambda: approved})

    try:
        result = agent.run(req.prompt)
    except Exception as exc:  # provider down / rate-limited / unreachable
        # Fail gracefully with a clear message instead of a bare 500.
        return JSONResponse(
            status_code=503,
            content={
                "error": str(exc),
                "answer": "The model provider is temporarily unavailable. "
                          "Please try again in a moment.",
                "iterations": 0,
                "blocked": False,
                "run_id": None,
                "latency_ms": 0.0,
                "tools_used": [],
                "guardrail_events": [],
            },
        )

    rec = trace_logger.record(req.prompt, result)  # persist + aggregate for /metrics
    return {
        "answer": result.answer,
        "iterations": result.iterations,
        "blocked": result.blocked,
        # extras for the dashboard (real per-run data):
        "run_id": rec.run_id,
        "latency_ms": rec.total_latency_ms,
        "tools_used": rec.tools_used,
        "guardrail_events": rec.guardrail_events,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return trace_logger.metrics()


@app.get("/")
def dashboard():
    """Serve the observability dashboard at the root path."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")
