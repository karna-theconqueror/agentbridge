"""Gemini provider tests — converter logic only (no network calls)."""
import json

from agentbridge.llm import GeminiProvider, Message


def make_provider():
    # api_key passed explicitly so tests never depend on the environment
    return GeminiProvider(api_key="test-key", model="gemini-test")


def test_converts_openai_tool_schemas_to_gemini():
    tools = [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic.",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                           "required": ["expression"]},
        },
    }]
    out = GeminiProvider._to_gemini_tools(tools)
    assert out == [{
        "functionDeclarations": [{
            "name": "calculator",
            "description": "Evaluate arithmetic.",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                           "required": ["expression"]},
        }]
    }]


def test_parse_tool_call_marker():
    name, args = GeminiProvider._parse_tool_call('[tool_call:calculator] {"expression": "125 * 8"}')
    assert name == "calculator"
    assert args == {"expression": "125 * 8"}


def test_parse_tool_call_rejects_plain_text():
    assert GeminiProvider._parse_tool_call("hello there") is None


def test_message_roles_convert_to_gemini_contents():
    p = make_provider()
    msgs = [
        Message(role="user", content="What is 125 * 8?"),
        Message(role="assistant", content='[tool_call:calculator] {"expression": "125 * 8"}'),
        Message(role="tool", content='{"expression": "125 * 8", "result": 1000}'),
    ]
    contents = p._to_contents(msgs)
    assert contents[0] == {"role": "user", "parts": [{"text": "What is 125 * 8?"}]}
    # assistant tool request becomes a model functionCall
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["functionCall"]["name"] == "calculator"
    # tool result becomes a user functionResponse bound to that tool name
    fr = contents[2]["parts"][0]["functionResponse"]
    assert contents[2]["role"] == "user"
    assert fr["name"] == "calculator"
    assert fr["response"]["result"] == 1000


def test_plain_final_answer_converts_to_model_text():
    p = make_provider()
    contents = p._to_contents([
        Message(role="user", content="hi"),
        Message(role="assistant", content="Hello! I can help."),
    ])
    assert contents[1] == {"role": "model", "parts": [{"text": "Hello! I can help."}]}


def test_requires_api_key():
    import os
    from agentbridge.llm import GeminiProvider as G
    old = os.environ.pop("GEMINI_API_KEY", None)
    try:
        import pytest
        with pytest.raises(ValueError):
            G()
    finally:
        if old is not None:
            os.environ["GEMINI_API_KEY"] = old


def test_tool_response_payload_is_dict():
    """Non-dict JSON (e.g. a bare list/number) must still be a dict payload."""
    p = make_provider()
    contents = p._to_contents([
        Message(role="user", content="x"),
        Message(role="assistant", content='[tool_call:calculator] {"expression": "1+1"}'),
        Message(role="tool", content='[1, 2]'),
    ])
    resp = contents[2]["parts"][0]["functionResponse"]["response"]
    assert isinstance(resp, dict)


def test_thought_signature_is_echoed_back():
    """Gemini 3.x rejects a replayed functionCall missing thoughtSignature."""
    p = make_provider()
    contents = p._to_contents([
        Message(role="user", content="What is 125 * 8?"),
        Message(role="assistant",
                content='[tool_call:calculator] {"expression": "125 * 8"}',
                thought_signature="sig-abc123"),
    ])
    part = contents[1]["parts"][0]
    assert part["functionCall"]["name"] == "calculator"
    assert part["thoughtSignature"] == "sig-abc123"


def test_thought_signature_omitted_when_none():
    """No signature attached → the key must be absent, not null."""
    p = make_provider()
    contents = p._to_contents([
        Message(role="user", content="hi"),
        Message(role="assistant", content='[tool_call:calculator] {"expression": "1+1"}'),
    ])
    assert "thoughtSignature" not in contents[1]["parts"][0]


def test_retry_delay_parsing():
    """Both server shapes of the retry delay must be recognised."""
    body = json.dumps({"error": {
        "code": 429,
        "message": "Quota exceeded. Please retry in 57.373731235s.",
        "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                     "retryDelay": "57s"}],
    }})
    assert GeminiProvider._retry_delay(body) == 57.0
    assert GeminiProvider._retry_delay("no delay here") is None
