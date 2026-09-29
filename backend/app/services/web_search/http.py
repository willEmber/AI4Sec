"""One request through a key pool, with every failure mapped to a status.

The providers disagree on how they say "this key is out of credit" (Tavily
432/433, Exa and Firecrawl 402), but agree on the rest, so the loop lives here
and each adapter only names its quota codes. A problem with the request itself
(400, 422) is not blamed on the key: every key would get the same answer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import httpx

from app.services.web_search.credentials import mask_key
from app.services.web_search.keypool import KeyPool
from app.services.web_search.models import ProviderError

logger = logging.getLogger("scholar.web_search")

USER_AGENT = "scholar-agent/1.0"
DEFAULT_TIMEOUT = 30.0
# Pause before retrying a 5xx or a timeout. Tests set it to zero.
SERVER_RETRY_DELAY = 1.0
_SERVER_ERRORS = {408, 500, 502, 503, 504}


def new_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=httpx.Timeout(timeout, connect=15.0),
        follow_redirects=True,
    )


def _retry_after(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except ValueError:
        return None


def _excerpt(response: httpx.Response, key: str) -> str:
    text = (response.text or "")[:300].replace("\n", " ")
    return text.replace(key, "***") if key else text


async def pooled_post(
    client: httpx.AsyncClient,
    pool: KeyPool,
    url: str,
    *,
    body: dict[str, Any],
    auth: Callable[[str], dict[str, str]],
    quota_statuses: frozenset[int],
    timeout: float | None = None,
    server_retries: int = 1,
) -> dict[str, Any]:
    """POST `body` with the next usable key until one answers.

    Raises `ProviderError` with the pool's reason once no key is usable, or
    with `failed` when the provider rejects the request itself or keeps
    failing on its side.
    """
    provider = pool.provider
    if not pool.configured:
        raise ProviderError(provider, "skipped_no_key", "no API key configured")

    server_failures = 0
    while True:
        key = pool.acquire()
        if key is None:
            status = pool.unavailable_status()
            raise ProviderError(provider, status, f"no usable key ({status})")
        await pool.pace(key)

        try:
            response = await client.post(
                url,
                json=body,
                headers=auth(key),
                **({"timeout": timeout} if timeout is not None else {}),
            )
        except httpx.TimeoutException:
            server_failures += 1
            if server_failures > server_retries:
                raise ProviderError(provider, "failed", "timed out") from None
            await asyncio.sleep(SERVER_RETRY_DELAY * server_failures)
            continue
        except httpx.HTTPError as exc:
            server_failures += 1
            if server_failures > server_retries:
                raise ProviderError(provider, "failed", f"network error: {type(exc).__name__}") from None
            await asyncio.sleep(SERVER_RETRY_DELAY * server_failures)
            continue

        code = response.status_code
        if 200 <= code < 300:
            pool.mark_ok(key)
            try:
                data = response.json()
            except ValueError:
                raise ProviderError(provider, "failed", "response was not JSON") from None
            if not isinstance(data, dict):
                raise ProviderError(provider, "failed", "unexpected response shape")
            return data
        if code == 401:
            logger.warning("%s refused key %s (HTTP 401)", provider, mask_key(key))
            pool.mark_refused(key, "HTTP 401")
            continue
        if code in quota_statuses:
            logger.warning("%s key %s is out of credit (HTTP %d)", provider, mask_key(key), code)
            pool.mark_out_of_credit(key, f"HTTP {code}")
            continue
        if code == 429:
            pool.mark_rate_limited(key, _retry_after(response), "HTTP 429")
            continue
        if code in _SERVER_ERRORS:
            server_failures += 1
            if server_failures > server_retries:
                raise ProviderError(provider, "failed", f"HTTP {code}: {_excerpt(response, key)}")
            await asyncio.sleep(SERVER_RETRY_DELAY * server_failures)
            continue
        raise ProviderError(provider, "failed", f"HTTP {code}: {_excerpt(response, key)}")
