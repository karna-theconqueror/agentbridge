from fastapi import FastAPI
from pydantic import BaseModel

from agentbridge.hitl import Hitl, AgentWithHitl

app = FastAPI(title="AgentBridge API")

hitl = Hitl({"calculator": False})  # default: block calculator
agent = AgentWithHitl(hitl=hitl)


class PromptRequest(BaseModel):
    prompt: str
    hitl: bool | None = None  # True = human approval gate enabled for the call


@app.post("/invoke")
def invoke(req: PromptRequest):
    if req.hitl is not None:
        # hitl == True → enable the human gate (demo policy approves calculator)
        agent.hitl = Hitl({"calculator": lambda: True}) if req.hitl else Hitl({"calculator": False})

    result = agent.run(req.prompt)
    return {"answer": result.answer, "iterations": result.iterations, "blocked": result.blocked}


@app.get("/health")
def health():
    return {"status": "ok"}
