"""Week 4: Human‑in‑The‑Loop approval and quality gate integration.

The gate already exists as a two-part test in ``run_evals.py`` (`Gate 2` quality versus
`baseline.json`).  The *human‑in‑the‑loop* (HITL) layer is new – it allows manual
review of certain tool calls before they propagate to the model.  It is
parameterised so you can toggle it on/off (default off to avoid blocking the
demo).  The policy demonstrates a *pull request style* gate that a human may
act on.

Implementation:

1. ``agentbridge/hitl.py`` – a very light fake reviewer that can be
   switched on to either approve or block selected tools (via a
   configuration map).  The default map blocks no tools.
2. ``agentbridge/agent.py`` – taps into the loop right after the tool call
   and before the returned JSON is sent back to the model.  If HITL is
   enabled and the tool name is in the ``review_map``, the call waits for a
   decision; the test suite simulates a “manual” approval.
3. ``tests/test_hitl.py`` – exercises the approval gate – blocks a
   tool call, then approves it, and checks that the agent behaves as
   expected.

The repository has **all** necessary hooks; the CI flow now includes both
*automatic* quality gate and an *optionally* interactive human gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import time
from .llm import LLMResponse, Message, get_provider
from .tools import TOOL_SCHEMAS, run_tool
from .guardrails import REFUSAL, Guardrails


# ----------------------------------------------------------- HITL policy

class Hitl:
    """Very small manual review simulator.

    The policy is a mapping ``tool_name -> decision_func`` where the decision
    function returns ``True`` to *approve* the tool call or ``False`` to
    *block* it.  In a real system you might hook into a comments API, slack
    bot, or a simple web UI.
    """

    def __init__(self, policy: dict[str, Callable[[], bool]] | None = None):
        self.policy = policy or {}

    def review(self, tool_name: str) -> bool:
        if tool_name not in self.policy:
            return True  # default allow
        return self.policy[tool_name]()


# ----------------------------------------------------------- agent wrapper

class AgentWithHitl:
    def __init__(self, *, provider=None, tools_schemas=TOOL_SCHEMAS,
                 max_iterations=5, guardrails: Guardrails | None = None,
                 hitl: Hitl | None = None):
        self.provider = provider or get_provider()
        self.tools_schemas = tools_schemas
        self.max_iterations = max_iterations
        self.guardrails = guardrails if guardrails is not None else Guardrails()
        self.hitl = hitl if hitl is not None else Hitl({})

    def run(self, user_input: str) -> AgentResult:
        # Input guardrail – same as before
        checked = self.guardrails.check_input(user_input)
        if not checked.allowed:
            return AgentResult(answer=REFUSAL, trace=[
                TraceStep(kind="guardrail", detail=f"BLOCKED input (injection: {checked.findings})")
            ], iterations=0, blocked=True)

        if checked.findings:
            trace = [TraceStep(kind="guardrail", detail=f"redacted from input: {', '.join(checked.findings)}")]
        else:
            trace = []
        messages: list[Message] = [Message(role="user", content=checked.text)]

        for iteration in range(1, self.max_iterations + 1):
            t0 = time.perf_counter()
            resp: LLMResponse = self.provider.complete(messages, self.tools_schemas)
            llm_ms = (time.perf_counter() - t0) * 1000

            if resp.tool_call:
                tc = resp.tool_call
                trace.append(TraceStep(kind="llm", name=tc.name,
                                       detail="requested tool", latency_ms=llm_ms))

                # ----- HITL check before executing
                approved = self.hitl.review(tc.name)
                if not approved:
                    trace.append(TraceStep(kind="hitl", name=tc.name,
                                           detail="blocked by human reviewer", latency_ms=0))
                    return AgentResult(answer=REFUSAL,
                                       trace=trace,
                                       iterations=iteration,
                                       blocked=True)

                t0 = time.perf_counter()
                result = run_tool(tc.name, tc.arguments)
                tool_ms = (time.perf_counter() - t0) * 1000

                # Tool guardrail
                checked_tool = self.guardrails.check_tool_output(tc.name, result)
                if not checked_tool.allowed:
                    trace.append(TraceStep(kind="guardrail", name=tc.name,
                                           detail=f"quarantined tool output: {checked_tool.findings}",
                                           latency_ms=tool_ms))
                    result = json.dumps({"error": f"{tc.name} output failed validation"})
                else:
                    if checked_tool.findings:
                        trace.append(TraceStep(kind="guardrail", name=tc.name,
                                               detail=f"scrubbed tool output: {', '.join(checked_tool.findings)}",
                                               latency_ms=tool_ms))
                    result = checked_tool.text
                trace.append(TraceStep(kind="tool", name=tc.name,
                                       detail=result, latency_ms=tool_ms))

                messages.append(Message(role="assistant",
                                        content=f"[tool_call:{tc.name}] {json.dumps(tc.arguments)}"))
                messages.append(Message(role="tool", content=result))
                continue

            # Final answer guardrail
            answer = resp.content or "(empty response)"
            checked_out = self.guardrails.check_output(answer)
            if checked_out.findings:
                trace.append(TraceStep(kind="guardrail",
                                       detail=f"scrubbed from answer: {', '.join(checked_out.findings)}"))
            answer = checked_out.text
            trace.append(TraceStep(kind="final", detail=answer, latency_ms=llm_ms))
            return AgentResult(answer=answer, trace=trace, iterations=iteration)


# ----------------------------------------------------------------------------- test helper
@dataclass
class HITLAgentResult:
    """Simple relaxed result type used only by the hitl tests – extends the core AgentResult without recursion."""
    answer: str
    trace: list


# The tests import ``HitlAgentResult`` as a stand‑in for the original
@dataclass
class DummyAgentResult:
    answer: str
    trace: list
# We'll use this in the test helper; the core AgentWithHitl still returns its own AgentResult.

