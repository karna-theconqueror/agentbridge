"""Week 4: Human-in-the-Loop approval and quality gate integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import time
import json

from .agent import TraceStep, AgentResult
from .llm import LLMResponse, Message, ToolCall, get_provider
from .tools import TOOL_SCHEMAS, run_tool
from .guardrails import REFUSAL, Guardrails


@dataclass
class HitlDecision:
    allowed: bool
    reason: str = "policy"


class Hitl:
    """Manual reviewer policy.

    Values in the policy map may be:
      - a bool (True = allow, False = block)  [also what the API sets],
      - a zero-arg callable returning bool.
    """

    def __init__(self, policy: dict[str, Callable[[], bool] | bool] | None = None):
        self.policy = policy or {}

    def review(self, tool_name: str) -> bool:
        if tool_name not in self.policy:
            return True  # default allow
        decision = self.policy[tool_name]
        if callable(decision):
            return bool(decision())
        return bool(decision)


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
        checked = self.guardrails.check_input(user_input)
        trace: list[TraceStep] = []
        if not checked.allowed:
            trace.append(TraceStep(kind="guardrail", detail=f"BLOCKED input (injection: {checked.findings})"))
            return AgentResult(answer=REFUSAL, trace=trace, iterations=0, blocked=True)

        if checked.findings:
            trace.append(TraceStep(kind="guardrail", detail=f"redacted from input: {', '.join(checked.findings)}"))
        messages: list[Message] = [Message(role="user", content=checked.text)]

        for iteration in range(1, self.max_iterations + 1):
            t0 = time.perf_counter()
            resp: LLMResponse = self.provider.complete(messages, self.tools_schemas)
            llm_ms = (time.perf_counter() - t0) * 1000

            if resp.tool_call is not None:
                tc: ToolCall = resp.tool_call
                trace.append(TraceStep(kind="llm", name=tc.name, detail="requested tool", latency_ms=llm_ms))

                approved = self.hitl.review(tc.name)
                if not approved:
                    trace.append(TraceStep(kind="hitl", name=tc.name, detail="blocked by human reviewer", latency_ms=0))
                    return AgentResult(answer=REFUSAL, trace=trace, iterations=iteration, blocked=True)

                t0 = time.perf_counter()
                result = run_tool(tc.name, tc.arguments)
                tool_ms = (time.perf_counter() - t0) * 1000

                checked_tool = self.guardrails.check_tool_output(tc.name, result)
                if not checked_tool.allowed:
                    trace.append(TraceStep(kind="guardrail", name=tc.name, detail=f"quarantined tool output: {checked_tool.findings}", latency_ms=tool_ms))
                    result = json.dumps({"error": f"{tc.name} output failed validation"})
                else:
                    if checked_tool.findings:
                        trace.append(TraceStep(kind="guardrail", name=tc.name, detail=f"scrubbed tool output: {', '.join(checked_tool.findings)}", latency_ms=tool_ms))
                    result = checked_tool.text
                trace.append(TraceStep(kind="tool", name=tc.name, detail=result, latency_ms=tool_ms))

                messages.append(Message(role="assistant",
                                        content=f"[tool_call:{tc.name}] {json.dumps(tc.arguments)}",
                                        thought_signature=tc.thought_signature))
                messages.append(Message(role="tool", content=result))
                continue

            answer = resp.content or "(empty response)"
            checked_out = self.guardrails.check_output(answer)
            if checked_out.findings:
                trace.append(TraceStep(kind="guardrail", detail=f"scrubbed from answer: {', '.join(checked_out.findings)}"))
            answer = checked_out.text
            trace.append(TraceStep(kind="final", detail=answer, latency_ms=llm_ms))
            return AgentResult(answer=answer, trace=trace, iterations=iteration)

        return AgentResult(answer="I could not complete that within the iteration budget.", trace=trace, iterations=self.max_iterations)
