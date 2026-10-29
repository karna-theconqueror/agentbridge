"""Fixed-window per-client rate limiting for public deployments.

Stdlib-only and in-memory — deliberately. A shared demo endpoint backed by a
free-tier model key gets found by scanners within hours; this stops one client
from draining the quota without pulling in Redis or any dependency.

This is a *demo-grade* control, not a security boundary: state is per-process
and resets on restart. For real production put it in a shared store (Redis /
API gateway) — see docs/interview-prep.md.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

DEFAULT_LIMIT = 10          # requests per client per window
DEFAULT_WINDOW = 60.0       # seconds


@dataclass
class _Window:
    count: int = 0
    reset_at: float = 0.0


class RateLimiter:
    """Fixed-window counter keyed by client.

    Thread-safe because FastAPI runs sync endpoints in a thread pool.
    """

    def __init__(self, limit: int | None = None, window: float = DEFAULT_WINDOW):
        if limit is None:
            limit = int(os.environ.get("RATE_LIMIT_PER_MIN", DEFAULT_LIMIT))
        self.limit = max(0, int(limit))   # 0 disables the limiter
        self.window = window
        self._lock = threading.Lock()
        self._hits: dict[str, _Window] = {}

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def check(self, client: str) -> tuple[bool, float]:
        """Consume one slot.

        Returns (allowed, seconds_until_next_slot). When not allowed the
        caller should send Retry-After = ceil(seconds_until_next_slot).
        """
        if not self.enabled:
            return True, 0.0

        now = time.monotonic()
        with self._lock:
            w = self._hits.get(client)
            if w is None or now >= w.reset_at:
                w = _Window(count=0, reset_at=now + self.window)
                self._hits[client] = w

            if w.count >= self.limit:
                return False, max(0.0, w.reset_at - now)

            w.count += 1
            self._evict(now)
            return True, 0.0

    def _evict(self, now: float) -> None:
        """Keep the dict bounded if a client rotates IPs/hosts."""
        if len(self._hits) <= 10_000:
            return
        for key, win in list(self._hits.items()):
            if now >= win.reset_at:
                del self._hits[key]
