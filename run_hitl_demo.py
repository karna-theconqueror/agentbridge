"""HITL (human-in-the-loop) demo.

Run directly: python run_hitl_demo.py
"""
from agentbridge.hitl import Hitl, AgentWithHitl
from agentbridge.llm import get_provider


def demo_agent_with_hitl() -> None:
    # Policy: calculator requires human approval (blocked here), claim lookup allowed.
    policy = {"calculator": lambda: False, "get_claim_status": lambda: True}
    agent = AgentWithHitl(provider=get_provider(), hitl=Hitl(policy))

    # This call hits the HITL block and returns a refusal.
    result = agent.run("What is 125 * 8?")
    print("Result when calculator is blocked:", result.answer)

    # This call passes because the policy allows claim status lookup.
    result2 = agent.run("Check claim #CLM-1001")
    print("Result when claim lookup is allowed:", result2.answer)


if __name__ == "__main__":
    demo_agent_with_hitl()
