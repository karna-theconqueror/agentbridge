"""Rate limiting tests (protects the shared model quota on a public deploy)."""
import pytest
from fastapi.testclient import TestClient

import app as app_module
from agentbridge.ratelimit import RateLimiter


def test_allows_up_to_limit_then_blocks():
    rl = RateLimiter(limit=3, window=60)
    assert [rl.check("a")[0] for _ in range(3)] == [True, True, True]

    allowed, retry_after = rl.check("a")
    assert allowed is False
    assert retry_after > 0          # caller needs a Retry-After


def test_clients_are_independent():
    rl = RateLimiter(limit=1, window=60)
    assert rl.check("ip-1")[0] is True
    assert rl.check("ip-1")[0] is False
    assert rl.check("ip-2")[0] is True     # different client has its own budget


def test_window_resets_after_it_expires():
    rl = RateLimiter(limit=1, window=0.05)
    assert rl.check("a")[0] is True
    assert rl.check("a")[0] is False

    import time
    time.sleep(0.06)
    assert rl.check("a")[0] is True         # new window, budget restored


def test_zero_limit_disables_the_limiter():
    rl = RateLimiter(limit=0)
    assert rl.enabled is False
    assert all(rl.check(f"ip-{i}")[0] for i in range(100))


def test_default_limit_comes_from_env(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "7")
    assert RateLimiter().limit == 7
    # explicit arg still wins over env
    assert RateLimiter(limit=2).limit == 2


def _swap_limiter(limit: int):
    """Temporarily replace the app's limiter; returns a restore callable."""
    original = app_module.rate_limiter
    app_module.rate_limiter = RateLimiter(limit=limit, window=60)
    return lambda: setattr(app_module, "rate_limiter", original)


def test_invoke_returns_429_with_retry_after_once_exhausted():
    restore = _swap_limiter(limit=2)
    client = TestClient(app_module.app)
    try:
        assert client.post("/invoke", json={"prompt": "hi"}).status_code == 200
        assert client.post("/invoke", json={"prompt": "hi"}).status_code == 200

        r = client.post("/invoke", json={"prompt": "hi"})
        assert r.status_code == 429
        assert "Retry-After" in r.headers
        assert int(r.headers["Retry-After"]) >= 1
        # body stays dashboard-friendly: a readable answer + error marker
        body = r.json()
        assert "Slow down" in body["answer"]
        assert body["error"]
        assert body["run_id"] is None
    finally:
        restore()


def test_health_and_dashboard_are_not_rate_limited():
    restore = _swap_limiter(limit=1)
    client = TestClient(app_module.app)
    try:
        client.post("/invoke", json={"prompt": "hi"})          # uses the only slot
        assert client.post("/invoke", json={"prompt": "hi"}).status_code == 429
        assert client.get("/health").status_code == 200        # still open
        assert client.get("/").status_code == 200              # still open
    finally:
        restore()


def test_disabled_limiter_never_returns_429():
    restore = _swap_limiter(limit=0)
    client = TestClient(app_module.app)
    try:
        codes = [client.post("/invoke", json={"prompt": "hi"}).status_code
                 for _ in range(15)]
        assert all(c == 200 for c in codes)
    finally:
        restore()
