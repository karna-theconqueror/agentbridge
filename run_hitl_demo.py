import json

import time
from dataclasses import dataclass

from .llm import LLMResponse, Message, get_provider
from .tools import TOOL_SCHEMAS, run_tool
from .guardrails import REFUSAL, Guardrails

from .hitl import Hitl, AgentWithHitl


# ----------------------------------------------------------------- Demo agent

def demo_agent_with_hitl() -> None:
    policy = {"calculator": lambda: False, "get_claim_status": lambda: True}
    agent = AgentWithHitl(provider=get_provider(), hitl=Hitl(policy))
    # This call should hit the HITL block and return refusal
    result = agent.run("What is 125 * 8?")
    print("Result when block calculator:", result.answer)
    # This call should pass as the policy allows claim status
    result2 = agent.run("Check claim #CLM-1001")
    print("Result when allowed claim:", result2.answer)


if __name__ == "__main__":
    demo_agent_with_hitl()
