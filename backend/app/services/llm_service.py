from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, AsyncIterator

import httpx

from app.config import get_settings

logger = logging.getLogger("scholar.llm")

# Maximum timeout cap for any single LLM request (seconds)
_TIMEOUT_CAP = 900.0


class LLMService:
    """Async OpenAI-compatible LLM client with retry and backoff."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ):
        settings = get_settings()
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = api_key or settings.llm_api_key
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _is_retryable(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    def _compute_delay(self, attempt: int) -> float:
        delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
        jitter = random.uniform(0.8, 1.2)
        return delay * jitter

    @staticmethod
    def _compute_read_timeout(prompt_chars: int, max_tokens: int) -> float:
        """Compute read timeout that accounts for prompt size and expected output."""
        # prompt_chars/4 ≈ rough token estimate; each prompt token adds ~0.02s processing
        # each output token adds ~0.04s generation time
        timeout = 90.0 + (prompt_chars / 4) * 0.02 + max_tokens * 0.04
        return min(max(180.0, timeout), _TIMEOUT_CAP)

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request, return assistant content."""
        settings = get_settings()
        model = model or settings.thinking_model

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Scale timeout with expected output size + prompt size
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        base_read_timeout = self._compute_read_timeout(prompt_chars, max_tokens)

        logger.info(
            f"LLM chat: model={model} prompt={prompt_chars} chars "
            f"max_tokens={max_tokens} timeout={base_read_timeout:.0f}s"
        )
        t0 = time.perf_counter()

        attempt = 0
        timeout_escalations = 0
        connect_failures = 0
        while True:
            attempt += 1
            # Progressive timeout escalation: multiply base by 1.5^n on ReadTimeout retries
            read_timeout = min(
                base_read_timeout * (1.5 ** timeout_escalations),
                _TIMEOUT_CAP,
            )
            try:
                t_req = time.perf_counter()
                timeout = httpx.Timeout(
                    connect=30.0, read=read_timeout, write=30.0, pool=30.0,
                )
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )
                req_elapsed = time.perf_counter() - t_req

                # ── HTTP 429 special handling ──
                if resp.status_code == 429 and attempt <= self.max_retries:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = min(float(retry_after), 120.0)
                        except ValueError:
                            delay = min(self._compute_delay(attempt) * 2, 120.0)
                    else:
                        delay = min(self._compute_delay(attempt) * 2, 120.0)
                    logger.warning(
                        f"LLM chat: HTTP 429 rate-limited (attempt {attempt}/{self.max_retries}), "
                        f"retry in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue

                # ── Non-retryable client errors: fail immediately ──
                if resp.status_code in (401, 403):
                    logger.error(
                        f"LLM chat: HTTP {resp.status_code} — auth/permission error, not retrying"
                    )
                    resp.raise_for_status()

                # ── Other retryable status codes ──
                if self._is_retryable(resp.status_code) and attempt <= self.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.warning(
                        f"LLM chat: HTTP {resp.status_code} (attempt {attempt}/{self.max_retries}), "
                        f"retry in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                total_elapsed = time.perf_counter() - t0

                # Log usage if available
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", "?")
                completion_tokens = usage.get("completion_tokens", "?")
                logger.info(
                    f"LLM chat: DONE in {total_elapsed:.1f}s (http={req_elapsed:.1f}s) — "
                    f"tokens={prompt_tokens}+{completion_tokens} response={len(content)} chars"
                )
                return content

            except httpx.ReadTimeout as e:
                req_elapsed = time.perf_counter() - t_req if 't_req' in dir() else 0
                if attempt > self.max_retries:
                    logger.error(
                        f"LLM chat: ReadTimeout FAILED after {attempt} attempts "
                        f"in {time.perf_counter()-t0:.1f}s — {e}"
                    )
                    raise
                timeout_escalations += 1
                new_timeout = min(
                    base_read_timeout * (1.5 ** timeout_escalations),
                    _TIMEOUT_CAP,
                )
                delay = self._compute_delay(attempt)
                logger.warning(
                    f"LLM chat: ReadTimeout after {req_elapsed:.0f}s "
                    f"(attempt {attempt}/{self.max_retries}), "
                    f"escalating timeout to {new_timeout:.0f}s, retry in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

            except httpx.ConnectError as e:
                connect_failures += 1
                if connect_failures >= 2 or attempt > self.max_retries:
                    logger.error(
                        f"LLM chat: ConnectError giving up after {connect_failures} connect failures "
                        f"in {time.perf_counter()-t0:.1f}s — {e}"
                    )
                    raise
                delay = self._compute_delay(attempt)
                logger.warning(
                    f"LLM chat: ConnectError (attempt {attempt}/{self.max_retries}), "
                    f"retry in {delay:.1f}s — {e}"
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                req_elapsed = time.perf_counter() - t_req if 't_req' in dir() else 0
                if attempt > self.max_retries:
                    logger.error(
                        f"LLM chat: HTTPStatusError FAILED after {attempt} attempts "
                        f"in {time.perf_counter()-t0:.1f}s — {e}"
                    )
                    raise
                delay = self._compute_delay(attempt)
                logger.warning(
                    f"LLM chat: {type(e).__name__} (attempt {attempt}/{self.max_retries}), "
                    f"retry in {delay:.1f}s — {e}"
                )
                await asyncio.sleep(delay)

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens."""
        settings = get_settings()
        model = model or settings.thinking_model

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        read_timeout = self._compute_read_timeout(prompt_chars, max_tokens)
        logger.info(
            f"LLM stream: model={model} prompt={prompt_chars} chars timeout={read_timeout:.0f}s"
        )
        t0 = time.perf_counter()
        token_count = 0

        timeout = httpx.Timeout(connect=30.0, read=read_timeout, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            token_count += 1
                            yield content
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue

        logger.info(f"LLM stream: DONE in {time.perf_counter()-t0:.1f}s — {token_count} chunks")


def get_llm_service() -> LLMService:
    return LLMService()
