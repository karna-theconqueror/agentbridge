"""Agent runtime: the tool-calling loop with trace recording.

The trace is the foundation for Week 5 observability (cost, latency,
failure taxonomy) — every step is already captured here.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .guardrails import REFUSAL, Guardrails
from .llm import LLMResponse, Message, get_provider
from .tools import TOOL_SCHEMAS, run_tool

MAX_ITERATIONS = 5  # safety cap so a confused model can't spin forever


@dataclass
class TraceStep:
    kind: str            # "llm" | "tool" | "guardrail" | "final"
    name: str = ""
    detail: str = ""
    latency_ms: float = 0.0


@dataclass
class AgentResult:
    answer: str
    trace: list[TraceStep] = field(default_factory=list)
    iterations: int = 0
    blocked: bool = False

    @property
    def tool_names(self) -> list[str]:
        return [s.name for s in self.trace if s.kind == "tool"]


class Agent:
    def __init__(self, provider=None, tools_schemas=TOOL_SCHEMAS,
                 max_iterations=MAX_ITERATIONS, guardrails=None):
        self.provider = provider or get_provider()
        self.tools_schemas = tools_schemas
        self.max_iterations = max_iterations
        self.guardrails = guardrails if guardrails is not None else Guardrails()

    def run(self, user_input: str) -> AgentResult:
        trace: list[TraceStep] = []

        # ---- INPUT guardrail: injection blocks, PII gets masked
        checked = self.guardrails.check_input(user_input)
        if not checked.allowed:
            trace.append(TraceStep(kind="guardrail",
                                   detail=f"BLOCKED input (injection patterns: {checked.findings})"))
            return AgentResult(answer=REFUSAL, trace=trace, iterations=0, blocked=True)
        if checked.findings:
            trace.append(TraceStep(kind="guardrail",
                                   detail=f"redacted from input: {', '.join(checked.findings)}"))
        messages: list[Message] = [Message(role="user", content=checked.text)]

        for iteration in range(1, self.max_iterations + 1):
            t0 = time.perf_counter()
            response: LLMResponse = self.provider.complete(messages, self.tools_schemas)
            llm_ms = (time.perf_counter() - t0) * 1000

            if response.tool_call is not None:
                tc = response.tool_call
                trace.append(TraceStep(kind="llm", name=tc.name,
                                       detail="requested tool", latency_ms=llm_ms))
                t0 = time.perf_counter()
                result = run_tool(tc.name, tc.arguments)
                tool_ms = (time.perf_counter() - t0) * 1000

                # ---- TOOL guardrail: schema-validate + scrub PII in tool output
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

            # No tool call -> final answer.
            # ---- OUTPUT guardrail: scrub PII before it reaches the user
            answer = response.content or "(empty response)"
            checked_out = self.guardrails.check_output(answer)
            if checked_out.findings:
                trace.append(TraceStep(kind="guardrail",
                                       detail=f"scrubbed from answer: {', '.join(checked_out.findings)}"))
            answer = checked_out.text
            trace.append(TraceStep(kind="final", detail=answer, latency_ms=llm_ms))
            return AgentResult(answer=answer, trace=trace, iterations=iteration)

        # Hit the cap: fail closed, with the trace intact for debugging.
        trace.append(TraceStep(kind="final",
                               detail="ERROR: max iterations exceeded",
                               latency_ms=0))
        return AgentResult(
            answer="I could not complete that within the iteration budget.",
            trace=trace,
            iterations=self.max_iterations,
        )
