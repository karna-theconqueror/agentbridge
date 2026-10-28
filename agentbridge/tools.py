"""Tool registry — the tools an agent is allowed to call.

Week 3 note: every tool's *output* will pass through the guardrails layer
(PII redaction, schema validation) before it re-enters the model context.
"""
from __future__ import annotations

import json
from typing import Callable

# ---------------------------------------------------------------- tools

def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if not expression or not set(expression) <= allowed:
        return json.dumps({"error": "unsupported characters in expression"})
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - sandboxed above
    except Exception as exc:
        return json.dumps({"error": f"could not evaluate: {exc}"})
    return json.dumps({"expression": expression, "result": result})


def get_claim_status(claim_id: str) -> str:
    """Sample domain tool: insurance claim lookup (synthetic data)."""
    claims = {
        "CLM-1001": {"status": "approved", "amount_usd": 4200, "adjuster": "R. Patel"},
        "CLM-1002": {"status": "under_review", "amount_usd": 1150, "note": "awaiting documents"},
        "CLM-1003": {"status": "denied", "amount_usd": 800, "reason": "policy lapse"},
    }
    record = claims.get(claim_id.upper())
    if not record:
        return json.dumps({"error": f"claim {claim_id} not found"})
    return json.dumps({"claim_id": claim_id.upper(), **record})


# ---------------------------------------------------------------- registry

REGISTRY: dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "get_claim_status": get_claim_status,
}

# Tool schemas in OpenAI format (also used as the eval contract).
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression like '125 * 8'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_claim_status",
            "description": "Look up the status of an insurance claim by id.",
            "parameters": {
                "type": "object",
                "properties": {"claim_id": {"type": "string"}},
                "required": ["claim_id"],
            },
        },
    },
]


def run_tool(name: str, arguments: dict) -> str:
    """Execute a registered tool; unknown tools fail safely (not loudly)."""
    fn = REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        return fn(**arguments)
    except TypeError as exc:
        return json.dumps({"error": f"bad arguments for {name}: {exc}"})


# Declared output contracts — enforced by the guardrails layer (Week 3).
# {"error": ...} responses are always permitted (legitimate tool failures).
TOOL_OUTPUT_SCHEMAS: dict[str, dict[str, str]] = {
    "calculator": {"expression": "str", "result": "num"},
    "get_claim_status": {"claim_id": "str", "status": "str"},
}
