"""Rotating key pools, one per provider, shared by every caller in the process.

The `.env` carries several keys per provider — mostly free-tier keys, each with
its own monthly credit and per-minute limit. A key that stops working must not
take the provider down with it, and a provider whose every key has stopped
working must say *why*, because "all keys refused" (fix the config), "credit
spent" (wait for the month) and "rate limited" (wait a minute) call for
different responses from both the operator and the agent.

What a response does to a key:

- refused (401): disabled for the life of the process — it will not start
  working by itself;
- out of credit (Tavily 432/433, Exa and Firecrawl 402): cooled down for a
  day, since the free tiers reset monthly and a paid key may be topped up;
- rate limited (429): cooled down for `Retry-After`, and the next key is
  tried at once.

Each key is also paced on its own, since the providers meter per key.

State is guarded by a `threading.Lock` for the same reason as
`paper_search.ratelimit`: the app runs more than one event loop over its
lifetime, and an asyncio primitive binds to the first loop that touches it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Mapping

from app.services.paper_search.ratelimit import IntervalLimiter
from app.services.web_search.credentials import (
    EXA,
    FIRECRAWL,
    TAVILY,
    mask_key,
    provider_keys,
)

QUOTA_COOLDOWN_SECONDS = 24 * 3600.0
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 30.0
MAX_RATE_LIMIT_COOLDOWN_SECONDS = 600.0

# Minimum spacing between two requests on the *same key*. Tavily dev keys
# allow 100 requests/minute, Exa 10/s on search, Firecrawl's free plan 10/min
# on scrape and search.
KEY_INTERVALS: dict[str, float] = {
    TAVILY: 0.65,
    EXA: 0.12,
    FIRECRAWL: 6.2,
}


@dataclass
class _KeyState:
    key: str
    limiter: IntervalLimiter
    disabled: bool = False
    quota_until: float = 0.0
    rate_until: float = 0.0
    last_error: str = ""

    def usable(self, now: float) -> bool:
        return not self.disabled and self.quota_until <= now and self.rate_until <= now


@dataclass
class KeyPool:
    """Round-robin over the keys of one provider, skipping the unusable ones."""

    provider: str
    keys: tuple[str, ...]
    min_interval: float = 0.0
    _states: list[_KeyState] = field(default_factory=list, repr=False)
    _cursor: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self._states = [_KeyState(k, IntervalLimiter(self.min_interval)) for k in self.keys]

    @property
    def configured(self) -> bool:
        return bool(self._states)

    @property
    def size(self) -> int:
        return len(self._states)

    def acquire(self) -> str | None:
        """The next usable key, or `None` when none is usable right now."""
        with self._lock:
            now = time.monotonic()
            for step in range(len(self._states)):
                idx = (self._cursor + step) % len(self._states)
                if self._states[idx].usable(now):
                    self._cursor = idx + 1
                    return self._states[idx].key
        return None

    async def pace(self, key: str) -> None:
        state = self._state(key)
        if state is not None:
            await state.limiter.wait()

    def mark_ok(self, key: str) -> None:
        with self._lock:
            state = self._state_locked(key)
            if state is not None:
                state.last_error = ""

    def mark_refused(self, key: str, detail: str = "") -> None:
        with self._lock:
            state = self._state_locked(key)
            if state is not None:
                state.disabled = True
                state.last_error = detail or "refused"

    def mark_out_of_credit(self, key: str, detail: str = "", seconds: float = QUOTA_COOLDOWN_SECONDS) -> None:
        with self._lock:
            state = self._state_locked(key)
            if state is not None:
                state.quota_until = time.monotonic() + seconds
                state.last_error = detail or "out of credit"

    def mark_rate_limited(self, key: str, retry_after: float | None = None, detail: str = "") -> None:
        wait = DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS if retry_after is None else retry_after
        wait = min(max(wait, 1.0), MAX_RATE_LIMIT_COOLDOWN_SECONDS)
        with self._lock:
            state = self._state_locked(key)
            if state is not None:
                state.rate_until = time.monotonic() + wait
                state.last_error = detail or "rate limited"

    def unavailable_status(self) -> str:
        """Why no key is usable, in `PlatformStatus` vocabulary.

        The most recoverable cause wins: one key merely rate-limited means the
        provider will be back within a minute, which matters more to the caller
        than the other keys being dead.
        """
        with self._lock:
            if not self._states:
                return "skipped_no_key"
            now = time.monotonic()
            if any(not s.disabled and s.rate_until > now for s in self._states):
                return "rate_limited"
            if any(not s.disabled and s.quota_until > now for s in self._states):
                return "quota_exhausted"
            if all(s.disabled for s in self._states):
                return "auth_failed"
            return "failed"

    def describe(self) -> list[dict[str, object]]:
        """Masked per-key state, for logs and diagnostics — never the keys."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "key": mask_key(s.key),
                    "usable": s.usable(now),
                    "disabled": s.disabled,
                    "cooling_s": round(max(s.quota_until, s.rate_until) - now, 1)
                    if not s.usable(now) and not s.disabled
                    else 0.0,
                    "last_error": s.last_error,
                }
                for s in self._states
            ]

    def _state(self, key: str) -> _KeyState | None:
        with self._lock:
            return self._state_locked(key)

    def _state_locked(self, key: str) -> _KeyState | None:
        return next((s for s in self._states if s.key == key), None)


_POOLS: dict[str, KeyPool] = {}
_POOLS_LOCK = threading.Lock()


def pool_for(provider: str) -> KeyPool:
    """The process-wide pool of `provider`, built from settings on first use."""
    with _POOLS_LOCK:
        pool = _POOLS.get(provider)
        if pool is None:
            pool = KeyPool(provider, provider_keys(provider), KEY_INTERVALS.get(provider, 0.0))
            _POOLS[provider] = pool
        return pool


def single_key_pool(provider: str, key: str) -> KeyPool:
    """A private pool around one explicitly supplied key."""
    keys = (key.strip(),) if key and key.strip() else ()
    return KeyPool(provider, keys, KEY_INTERVALS.get(provider, 0.0))


def resolve_pool(provider: str, pools: Mapping[str, KeyPool] | None) -> KeyPool:
    """The pool a caller supplied for `provider`, else the process-wide one.

    Membership, not truthiness: an empty pool supplied on purpose must stay
    empty rather than fall through to the configured keys.
    """
    if pools is not None and provider in pools:
        return pools[provider]
    return pool_for(provider)


def reset_pools() -> None:
    """For tests and settings reloads: rebuild every pool on next use."""
    with _POOLS_LOCK:
        _POOLS.clear()
