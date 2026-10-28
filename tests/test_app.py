"""Regression tests for the FastAPI app endpoints (Week 4/5 wiring)."""
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_invoke_hll_allows_calculator():
    r = client.post("/invoke", json={"prompt": "What is 125 * 8?", "hitl": True})
    body = r.json()
    assert r.status_code == 200
    assert body["blocked"] is False
    assert "1000" in body["answer"]


def test_invoke_default_blocks_calculator():
    r = client.post("/invoke", json={"prompt": "What is 125 * 8?"})
    body = r.json()
    assert body["blocked"] is True


def test_hitl_state_does_not_leak_across_requests():
    """hitl:true in one request must NOT approve calculator in the next."""
    client.post("/invoke", json={"prompt": "What is 2 * 2?", "hitl": True})
    r = client.post("/invoke", json={"prompt": "What is 2 * 2?"})  # no hitl flag
    assert r.json()["blocked"] is True


def test_metrics_endpoint_reflects_runs():
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    assert m["total_runs"] > 0
    assert "latency_ms" in m


def test_dashboard_served_at_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "AgentBridge" in r.text
