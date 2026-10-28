"""Pluggable LLM providers.

The MockLLM is deterministic and offline so the demo and tests always run.
OpenAI/Anthropic providers activate automatically when their API key is set.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """Either a tool call or a final answer (never both)."""
    content: str | None = None
    tool_call: ToolCall | None = None


@dataclass
class Message:
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None


class MockLLM:
    """Rule-based stand-in: answers math and claim questions via tools,
    everything else directly. Deterministic = reproducible evals."""

    name = "mock"

    # Word-based operators normalised into symbols before pattern matching,
    # so "What is 15 plus 7?" behaves like "15 + 7".
    WORD_OPERATORS = [
        (re.compile(r"(-?\d+)\s*plus\s*(-?\d+)", re.I), r"\1 + \2"),
        (re.compile(r"(-?\d+)\s*minus\s*(-?\d+)", re.I), r"\1 - \2"),
        (re.compile(r"(-?\d+)\s*times\s*(-?\d+)", re.I), r"\1 * \2"),
        (re.compile(r"(-?\d+)\s*divided\s+by\s*(-?\d+)", re.I), r"\1 / \2"),
        (re.compile(r"divide\s+(-?\d+)\s+by\s+(-?\d+)", re.I), r"\1 / \2"),
    ]

    @staticmethod
    def _final_from_tool_result(raw: str) -> str:
        """Turn a raw tool result into a human sentence (no JSON leakage)."""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return f"The tool returned: {raw}"
        if isinstance(data, dict):
            if "error" in data:
                return f"Sorry, {data['error']}."
            if "result" in data:
                return f"The result is {data['result']}."
            return "Here's what I found: " + ", ".join(
                f"{k}: {v}" for k, v in data.items()
            ) + "."
        return str(data)

    def complete(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        last = messages[-1].content if messages else ""

        # A tool result coming back means we should produce the final answer.
        if messages and messages[-1].role == "tool":
            return LLMResponse(content=self._final_from_tool_result(last))

        text = last.strip()

        normalized = text
        for pattern, repl in self.WORD_OPERATORS:
            normalized = pattern.sub(repl, normalized)

        math = re.search(r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)", normalized)
        if math and any(w in text.lower() for w in (
                "calculate", "compute", "what is", "plus", "minus", "times",
                "divide", "/", "*", "+", "-")):
            return LLMResponse(tool_call=ToolCall(
                name="calculator",
                arguments={"expression": f"{math.group(1)} {math.group(2)} {math.group(3)}"},
            ))

        claim = re.search(r"claim\s*#?([\w-]+)", text, re.I)
        if claim:
            return LLMResponse(tool_call=ToolCall(
                name="get_claim_status",
                arguments={"claim_id": claim.group(1)},
            ))

        return LLMResponse(
            content="I can calculate math expressions or look up claim status. "
                    "Try: 'What is 125 * 8?' or 'Check claim #CLM-1001'."
        )


class OpenAIProvider:
    """Thin wrapper around the chat-completions tool-calling API."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI  # optional dependency
        self.client = OpenAI()
        self.model = model

    def complete(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            tools=tools,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            return LLMResponse(tool_call=ToolCall(
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            ))
        return LLMResponse(content=msg.content or "")


def get_provider() -> MockLLM | OpenAIProvider:
    """Factory: real provider if a key exists, otherwise the offline mock."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIProvider()
        except Exception:
            pass  # fall back to mock if the SDK/key is broken
    return MockLLM()
