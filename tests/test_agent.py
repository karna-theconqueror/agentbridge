"""Week 1 unit tests — run with `python -m pytest`."""
import json

from agentbridge.agent import Agent
from agentbridge.llm import MockLLM
from agentbridge.tools import calculator, get_claim_status, run_tool


def test_calculator_math():
    assert json.loads(calculator("125 * 8"))["result"] == 1000


def test_calculator_rejects_junk():
    assert "error" in json.loads(calculator("__import__('os')"))


def test_claim_lookup():
    assert json.loads(get_claim_status("CLM-1001"))["status"] == "approved"


def test_unknown_claim():
    assert "error" in json.loads(get_claim_status("CLM-9999"))


def test_unknown_tool_fails_safely():
    assert "error" in run_tool("no_such_tool", {})


def test_agent_uses_tool_and_answers():
    agent = Agent(provider=MockLLM())
    result = agent.run("What is 125 * 8?")
    assert "calculator" in result.tool_names
    assert "1000" in result.answer


def test_agent_direct_answer_has_no_tools():
    agent = Agent(provider=MockLLM())
    result = agent.run("Hello, what can you do?")
    assert result.tool_names == []
    assert result.answer


def test_agent_records_trace():
    agent = Agent(provider=MockLLM())
    result = agent.run("Check claim #CLM-1001")
    kinds = [s.kind for s in result.trace]
    assert kinds[0] == "llm" and kinds[-1] == "final"
    assert "tool" in kinds
