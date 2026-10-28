from fastapi import FastAPI
from pydantic import BaseModel

from agentbridge.hitl import Hitl, AgentWithHitl

app = FastAPI(title="AgentBridge API")

# Default HITL policy – here we block the calculator tool for demo purposes
hitl_policy = {"calculator": False}
hitl = Hitl(hitl_policy)
agent = AgentWithHitl(hitl=hitl)


class PromptRequest(BaseModel):
    prompt: str
    hitl: bool | None = None  # optional override per request


@app.post("/invoke")
def invoke(req: PromptRequest):
    # Optional per‑request HITL override
    if req.hitl is not None:
        # For demo: toggle calculator between blocked/allowed
        agent.hitl = Hitl({"calculator": not req.hitl})

    result = agent.run(req.prompt)
    return {"answer": result.answer, "iterations": result.iterations}


@app.get("/health")
def health():
    return {"status": "ok"}
