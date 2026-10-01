"""Per-provider pacing shared by every caller in the process.

Semantic Scholar allows a keyed client one request per second *across all
endpoints*, and arXiv one request every three seconds on a single connection.
Search, the citation graph, triage and the metadata tools all call these
providers, so a limiter per call site does not bound the total — two of them
running concurrently is already over the limit. These are process-wide.

The limiters hold a `threading.Lock`, not an `asyncio.Lock`: the search
adapters run their HTTP in worker threads and the app runs more than one event
loop over its lifetime (warm-up, tests), and an asyncio primitive binds to the
first loop that touches it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Mapping


class QuotaExhaustedError(RuntimeError):
    """The provider's daily budget is spent; retrying before the reset is waste."""

    def __init__(self, provider: str, reset_in_s: float | None = None) -> None:
        detail = f"; resets in {int(reset_in_s)}s" if reset_in_s else ""
        super().__init__(f"{provider} daily quota exhausted{detail}")
        self.provider = provider
        self.reset_in_s = reset_in_s


class MissingCredentialError(RuntimeError):
    """The provider cannot be used without a key that is not configured."""


class IntervalLimiter:
    """Spaces calls at least `min_interval` seconds apart, whoever makes them.

    Each caller reserves the next free slot under the lock and then sleeps
    outside it, so waiting never blocks the event loop and concurrent callers
    queue in arrival order instead of all firing when the interval elapses.
    """

    def __init__(self, min_interval: float) -> None:
        self.min_interval = float(min_interval)
        self._lock = threading.Lock()
        self._next_free = 0.0

    def _reserve(self) -> float:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_free)
            self._next_free = slot + self.min_interval
            return slot - now

    async def wait(self) -> None:
        delay = self._reserve()
        if delay > 0:
            await asyncio.sleep(delay)

    def reset(self) -> None:
        with self._lock:
            self._next_free = 0.0


@dataclass
class DailyBudget:
    """What the provider last told us about its daily budget.

    OpenAlex reports the remaining budget on every response. Once it answers 429
    with nothing left, every further request until the reset is refused anyway,
    so callers check `exhausted()` first and skip the provider — the search
    fan-out then reports it as `quota_exhausted` instead of a generic failure,
    and falls back to the other sources.
    """

    provider: str
    remaining_usd: float | None = None
    _exhausted_until: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def observe(self, headers: Mapping[str, str] | None) -> None:
        if not headers:
            return
        raw = _header(headers, "x-ratelimit-remaining-usd")
        if raw is None:
            return
        try:
            remaining = float(raw)
        except ValueError:
            return
        with self._lock:
            self.remaining_usd = remaining
            if remaining > 0:
                self._exhausted_until = 0.0

    def mark_exhausted(self, headers: Mapping[str, str] | None = None) -> float:
        reset = _header(headers or {}, "x-ratelimit-reset")
        try:
            reset_s = float(reset) if reset is not None else 3600.0
        except ValueError:
            reset_s = 3600.0
        # Capped: a clock skew or a bogus header must not disable the provider
        # for longer than a day.
        reset_s = min(max(reset_s, 60.0), 86400.0)
        with self._lock:
            self.remaining_usd = 0.0
            self._exhausted_until = time.monotonic() + reset_s
        return reset_s

    def exhausted(self) -> bool:
        with self._lock:
            return self._exhausted_until > time.monotonic()

    def reset_in(self) -> float | None:
        with self._lock:
            left = self._exhausted_until - time.monotonic()
        return left if left > 0 else None

    def clear(self) -> None:
        with self._lock:
            self.remaining_usd = None
            self._exhausted_until = 0.0


def _header(headers: Mapping[str, str], name: str) -> str | None:
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def is_quota_response(status_code: int, headers: Mapping[str, str] | None) -> bool:
    """A 429 that means "budget spent", as opposed to "too many per second"."""
    if status_code != 429:
        return False
    raw = _header(headers or {}, "x-ratelimit-remaining-usd")
    if raw is None:
        return False
    try:
        return float(raw) <= 0
    except ValueError:
        return False


# One keyed S2 client gets 1 request/s across endpoints; the margin absorbs
# clock jitter between our slot and their window.
S2_LIMITER = IntervalLimiter(1.05)
# arXiv: "no more than one request every three seconds".
ARXIV_LIMITER = IntervalLimiter(3.1)
# dblp's SPARQL endpoint publishes no limit; one query at a time is polite.
DBLP_LIMITER = IntervalLimiter(1.0)
# OpenReview allows 180 requests per minute per client.
OPENREVIEW_LIMITER = IntervalLimiter(0.35)

OPENALEX_BUDGET = DailyBudget("OpenAlex")


def reset_all() -> None:
    """For tests: forget every pacing slot and budget observation."""
    for limiter in (S2_LIMITER, ARXIV_LIMITER, DBLP_LIMITER, OPENREVIEW_LIMITER):
        limiter.reset()
    OPENALEX_BUDGET.clear()
