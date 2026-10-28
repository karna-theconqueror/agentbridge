"""Regression: both agent loops must forward Gemini's thought_signature.

`Agent.run` (agent.py) and `AgentWithHitl.run` (hitl.py) each build the
message history themselves. Gemini 3.x rejects a replayed functionCall that
is missing the thoughtSignature, so a dropped signature shows up as a 400
only when talking to the live API — never against MockLLM.
"""
from agentbridge.hitl import AgentWithHitl, Hitl
from agentbridge.llm import LLMResponse, Message, MockLLM, ToolCall


class SignatureCapturingProvider(MockLLM):
    """Records the message history the agent hands back on each call."""

    def __init__(self):
        super().__init__()
        self.seen: list[list[Message]] = []

    def complete(self, messages, tools):
        self.seen.append(list(messages))
        # First call asks for the tool (carrying a signature), second finishes.
        if len(self.seen) == 1:
            return LLMResponse(tool_call=ToolCall(
                name="calculator",
                arguments={"expression": "125 * 8"},
                thought_signature="sig-xyz",
            ))
        return LLMResponse(content="The result is 1000.")


def _history_of_signature():
    provider = SignatureCapturingProvider()
    agent = AgentWithHitl(provider=provider, hitl=Hitl({"calculator": True}))
    agent.run("What is 125 * 8?")
    return provider.seen[-1]


def test_hitl_loop_forwards_thought_signature():
    """The HITL loop (hitl.py) must echo the signature back to the provider."""
    history = _history_of_signature()
    assistant = next(m for m in history if m.role == "assistant")
    assert assistant.thought_signature == "sig-xyz"


def test_base_loop_forwards_thought_signature():
    """Same guarantee for the plain Agent loop (agent.py)."""
    from agentbridge.agent import Agent

    provider = SignatureCapturingProvider()
    Agent(provider=provider).run("What is 125 * 8?")
    assistant = next(m for m in provider.seen[-1] if m.role == "assistant")
    assert assistant.thought_signature == "sig-xyz"


def test_signature_survives_into_gemini_contents():
    """End-to-end: the signature must land on the replayed functionCall."""
    from agentbridge.llm import GeminiProvider

    p = GeminiProvider(api_key="k", model="m")
    history = _history_of_signature()
    contents = p._to_contents(history)
    model_turn = next(c for c in contents if c["role"] == "model")
    assert model_turn["parts"][0]["thoughtSignature"] == "sig-xyz"
