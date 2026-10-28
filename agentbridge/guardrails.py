"""Week 3: guardrails layer.

Three chokepoints around the agent loop:

1. INPUT  — prompt-injection detection (blocks) + PII redaction (masks, then
   lets the request through with the PII stripped before it reaches the model).
2. TOOL   — every tool result must be valid JSON matching the tool's declared
   output schema; anything else is quarantined before it re-enters the model
   context ("garbage in the context window" is how agents derail).
3. OUTPUT — the final answer is scrubbed for PII before it reaches the user.

Fail-closed on injection, fail-safe (quarantine) on malformed tool data.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .tools import TOOL_OUTPUT_SCHEMAS

REFUSAL = "I can't help with that request. I can calculate math expressions or look up claim status."


@dataclass
class GuardrailResult:
    allowed: bool
    text: str
    findings: list[str] = field(default_factory=list)


# ------------------------------------------------------------- PII

PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Order matters: specific formats first. CARD must precede AADHAAR so a
    # 16-digit card isn't partially eaten by the 12-digit Aadhaar pattern.
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}[\s-]\d{4}[\s-]\d{4}\b")),
    ("PHONE", re.compile(r"\+?\d[\d\s\-]{8,}\d")),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII with [TAG] markers. Returns (redacted_text, findings)."""
    findings: list[str] = []
    for tag, pattern in PII_PATTERNS:
        def _mask(match: re.Match) -> str:
            if tag not in findings:
                findings.append(tag)
            return f"[{tag}]"
        text = pattern.sub(_mask, text)
    return text, findings


# ------------------------------------------------------------- injection

INJECTION_PATTERNS = [
    re.compile(p, re.I) for p in (
        r"\bignore\s+(?:all\s+|previous|prior|above)\b",
        r"\bdisregard\s+(?:all\s+|previous|prior)\b",
        r"\b(?:reveal|show|print|repeat)\s+(?:your|the)\s+(?:system|hidden|initial)\b",
        r"\bsystem\s+prompt\b",
        r"\bjailbreak\b",
        r"\byou\s+are\s+now\b",
        r"\bdeveloper\s+mode\b",
        r"\bact\s+as\s+if\b",
        r"\byour\s+secrets?\b",
        r"\bpretend\s+(?:to\s+be|you\s+are)\b",
    )
]


def detect_injection(text: str) -> list[str]:
    """Returns the patterns that hit (empty list = looks clean)."""
    return [p.pattern for p in INJECTION_PATTERNS if p.search(text)]


# ------------------------------------------------------------- tool schemas

def _type_ok(value, expected: str) -> bool:
    if expected == "str":
        return isinstance(value, str)
    if expected == "num":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def validate_tool_result(tool_name: str, raw: str) -> GuardrailResult:
    """Tool output must be JSON matching the declared schema.

    {"error": ...} responses are allowed through (they are legitimate
    tool-level failures the model should see); malformed data is quarantined.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return GuardrailResult(allowed=False, text=raw,
                               findings=[f"{tool_name}: non-JSON output quarantined"])

    if not isinstance(data, dict):
        return GuardrailResult(allowed=False, text=raw,
                               findings=[f"{tool_name}: non-object JSON quarantined"])

    if "error" in data:
        return GuardrailResult(allowed=True, text=raw)

    schema = TOOL_OUTPUT_SCHEMAS.get(tool_name)
    if not schema:
        return GuardrailResult(allowed=True, text=raw)

    problems = [f"missing '{k}'" for k, t in schema.items()
                if k not in data and not _type_ok(data.get(k), t)]
    problems += [f"wrong type for '{k}' (expected {t})"
                 for k, t in schema.items() if k in data and not _type_ok(data[k], t)]
    if problems:
        return GuardrailResult(allowed=False, text=raw,
                               findings=[f"{tool_name}: " + "; ".join(problems)])
    return GuardrailResult(allowed=True, text=raw)


# ------------------------------------------------------------- facade

class Guardrails:
    """The three checkpoints. Stateless; safe to reuse across agents."""

    def check_input(self, text: str) -> GuardrailResult:
        hits = detect_injection(text)
        if hits:
            return GuardrailResult(allowed=False, text="", findings=hits)
        redacted, findings = redact_pii(text)
        return GuardrailResult(allowed=True, text=redacted, findings=findings)

    def check_tool_output(self, tool_name: str, raw: str) -> GuardrailResult:
        validated = validate_tool_result(tool_name, raw)
        if not validated.allowed:
            return validated
        redacted, findings = redact_pii(validated.text)
        return GuardrailResult(allowed=True, text=redacted,
                               findings=[f"PII:{f}" for f in findings])

    def check_output(self, text: str) -> GuardrailResult:
        redacted, findings = redact_pii(text)
        return GuardrailResult(allowed=True, text=redacted,
                               findings=[f"PII:{f}" for f in findings])
