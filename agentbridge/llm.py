"""Pluggable LLM providers.

The MockLLM is deterministic and offline so the demo, evals, and tests always
run. Real providers activate automatically when their API key is set:

    GEMINI_API_KEY   → GeminiProvider   (REST, stdlib-only — no SDK needed)
    OPENAI_API_KEY   → OpenAIProvider

Eval harness deliberately pins itself to MockLLM so the CI gate never depends
on the network (see run_evals.py --live to opt in).
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    arguments: dict
    # Gemini 3.x requires the model's thoughtSignature to be echoed back when
    # a functionCall is replayed in multi-turn history. None for other providers.
    thought_signature: str | None = None


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
    thought_signature: str | None = None


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


class GeminiProvider:
    """Google Gemini via the REST API — stdlib only, no SDK dependency.

    Uses `urllib.request` against:
        POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    with the key sent in the `x-goog-api-key` header.

    Converts our OpenAI-style tool schemas into Gemini `functionDeclarations`
    and our Message history into Gemini `contents` (roles: user / model).
    """

    name = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 timeout: float = 30.0, retries: int = 3, backoff: float = 0.8,
                 max_delay: float = 65.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        # A pinned model (explicit arg or GEMINI_MODEL) is the only one tried;
        # otherwise fall back through DEFAULT_MODELS when quota runs out.
        pinned = model or os.environ.get("GEMINI_MODEL")
        self.models = [pinned] if pinned else list(self.DEFAULT_MODELS)
        self.model = self.models[0]
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.max_delay = max_delay

    # ---------------------------------------------------------- converters
    @staticmethod
    def _to_gemini_tools(tools: list[dict]) -> list[dict]:
        """OpenAI [{type,function:{...}}] → Gemini [{functionDeclarations:[...]}]."""
        decls = []
        for t in tools:
            fn = t.get("function", t)
            decls.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        return [{"functionDeclarations": decls}]

    @staticmethod
    def _parse_tool_call(content: str):
        """Recover (name, args) from the '[tool_call:name] {json}' marker."""
        m = re.match(r"^\[tool_call:([^\]]+)\]\s*(\{.*\})$", content or "", re.S)
        if not m:
            return None
        try:
            return m.group(1), json.loads(m.group(2))
        except json.JSONDecodeError:
            return None

    def _to_contents(self, messages: list[Message]) -> list[dict]:
        """Our Message list → Gemini contents list.

        role mapping: user→user, assistant→model, tool→model(functionCall)
        followed by a user turn carrying functionResponse.
        """
        contents: list[dict] = []
        pending_tool: str | None = None

        for msg in messages:
            if msg.role == "user":
                contents.append({"role": "user", "parts": [{"text": msg.content}]})
            elif msg.role == "assistant":
                parsed = self._parse_tool_call(msg.content)
                if parsed:
                    name, args = parsed
                    pending_tool = name
                    part: dict = {"functionCall": {"name": name, "args": args}}
                    # Gemini 3.x rejects a replayed functionCall without the
                    # thoughtSignature the model attached to it.
                    if msg.thought_signature:
                        part["thoughtSignature"] = msg.thought_signature
                    contents.append({"role": "model", "parts": [part]})
                else:
                    contents.append({"role": "model", "parts": [{"text": msg.content}]})
            elif msg.role == "tool":
                # tool result goes back as a user turn with functionResponse
                try:
                    payload = json.loads(msg.content)
                except (json.JSONDecodeError, TypeError):
                    payload = {"result": msg.content}
                if not isinstance(payload, dict):
                    payload = {"result": payload}
                name = pending_tool or "tool"
                pending_tool = None
                contents.append({"role": "user",
                                 "parts": [{"functionResponse": {"name": name,
                                                                 "response": payload}}]})
        return contents

    # ---------------------------------------------------------- call
    RETRYABLE = {429, 500, 502, 503, 504}

    @staticmethod
    def _retry_delay(detail: str) -> float | None:
        """Pull the server's RetryInfo delay out of an error body, so we wait
        exactly as long as we're told to. Matches both shapes:
            "retryDelay": "57s"          (details[].retryDelay)
            Please retry in 57.37s.      (error.message)
        """
        m = (re.search(r'"retryDelay"\s*:\s*"([\d.]+)', detail, re.I)
             or re.search(r"retry(?:Delay)?[^0-9]{0,20}([\d.]+)\s*s\b", detail, re.I))
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    # Models tried in order when GEMINI_MODEL is not pinned. Free-tier quota
    # is tracked per model, so one model being exhausted doesn't take the app
    # down — we simply move to the next that still has room.
    DEFAULT_MODELS = (
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash",
    )

    class RateLimitedError(RuntimeError):
        """Quota is exhausted for long enough that retrying now cannot help."""

    def _request(self, model: str, payload: bytes) -> dict:
        """POST to one model, retrying transient errors with the server's own
        retryDelay. Raises RateLimitedError when waiting is pointless."""
        url = f"{self.BASE_URL}/{model}:generateContent"
        last_err: Exception | None = None
        delay = self.backoff

        for attempt in range(self.retries + 1):
            if attempt:
                time.sleep(delay)
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
                last_err = RuntimeError(f"Gemini API error {exc.code}: {detail}")
                if exc.code not in self.RETRYABLE:
                    raise last_err from exc
                server_delay = self._retry_delay(detail)
                if server_delay and server_delay > self.max_delay:
                    # e.g. an exhausted daily quota asking us to wait hours —
                    # retrying sooner cannot help, so fail fast (and let the
                    # caller fall back to another model).
                    raise self.RateLimitedError(
                        f"{model}: quota exhausted, API asked us to wait "
                        f"~{server_delay / 60:.0f} min"
                    ) from exc
                delay = min(server_delay or self.backoff * (2 ** attempt),
                            self.max_delay)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # network blip or read timeout — retryable
                reason = getattr(exc, "reason", exc)
                last_err = RuntimeError(f"Gemini API unreachable: {reason}")
                delay = min(self.backoff * (2 ** attempt), self.max_delay)

        raise last_err if last_err else RuntimeError(f"Gemini request to {model} failed")

    def complete(self, messages: list[Message], tools: list[dict]) -> LLMResponse:
        body: dict = {"contents": self._to_contents(messages)}
        if tools:
            body["tools"] = self._to_gemini_tools(tools)
        payload = json.dumps(body).encode("utf-8")

        # Try each model in turn; a rate-limited one is skipped (not fatal).
        # The first model that answers is promoted to the front so we don't
        # pay the failover cost on every request.
        rate_limited: list[str] = []
        data: dict | None = None
        for idx, model in enumerate(self.models):
            try:
                data = self._request(model, payload)
            except self.RateLimitedError as exc:
                rate_limited.append(str(exc))
                continue
            if idx:
                self.models.insert(0, self.models.pop(idx))
                self.model = self.models[0]
            break

        if data is None:
            if len(self.models) == 1:
                raise RuntimeError(
                    f"Gemini rate limit exceeded — {rate_limited[0]}. "
                    f"Unset GEMINI_MODEL to allow automatic failover."
                )
            raise RuntimeError(
                "No Gemini model has quota right now. " + "; ".join(rate_limited)
            )

        # pick the first part: either a functionCall or text
        candidates = data.get("candidates") or []
        if not candidates:
            # surface safety/other blocks instead of silently returning empty
            feedback = data.get("promptFeedback") or {}
            reason = feedback.get("blockReason") or data.get("error", {}).get("message") or "no candidates"
            return LLMResponse(content=f"[gemini blocked: {reason}]")

        parts = (candidates[0].get("content") or {}).get("parts") or []
        for part in parts:
            if "functionCall" in part:
                fc = part["functionCall"]
                return LLMResponse(tool_call=ToolCall(
                    name=fc.get("name", ""),
                    arguments=fc.get("args") or {},
                    thought_signature=part.get("thoughtSignature"),
                ))
            if "text" in part:
                return LLMResponse(content=part["text"] or "")
        return LLMResponse(content="")


def get_provider():
    """Factory: real provider if a key exists, otherwise the offline mock."""
    if os.environ.get("GEMINI_API_KEY"):
        try:
            return GeminiProvider()
        except Exception:
            pass  # fall back to mock if the key/endpoint is broken
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIProvider()
        except Exception:
            pass  # fall back to mock if the SDK/key is broken
    return MockLLM()
