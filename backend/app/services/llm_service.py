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

# Built-in Responses API tools documented for Qwen3.5+ / Qwen3.6+ / Qwen3.7+ / Qwen3.8
WEB_SEARCH_TOOL: dict[str, str] = {"type": "web_search"}
WEB_EXTRACTOR_TOOL: dict[str, str] = {"type": "web_extractor"}
CODE_INTERPRETER_TOOL: dict[str, str] = {"type": "code_interpreter"}


class LLMEmptyResponseError(RuntimeError):
    """The gateway accepted the request but returned no assistant text.

    ``reason`` carries the truncation cause when there is one. The common case
    is ``max_output_tokens``: on the Responses API that budget covers reasoning
    *and* visible output, so a thinking model can spend the whole allowance
    before emitting a single token. Callers must not retry at the same size —
    the failure is deterministic. Raising beats returning ``""`` because an
    empty string reaches JSON parsers as "unparseable", which hides the cause.
    """

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason


class LLMService:
    """Async Qwen / DashScope MaaS Responses API client with retry and backoff.

    Uses the OpenAI-compatible Responses endpoint::

        POST {base_url}/responses
        {
          "model": "...",
          "input": [{"role": "...", "content": "..."}],
          "enable_thinking": true,
          "tools": [{"type": "web_search"}, ...],   # optional
          ...
        }
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ):
        settings = get_settings()
        self.base_url = self._clean_str(base_url or settings.llm_base_url).rstrip("/")
        self.api_key = self._clean_str(api_key or settings.llm_api_key)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    @staticmethod
    def _clean_str(value: str | None) -> str:
        """Strip whitespace / CRLF that often leaks from Windows .env files."""
        return (value or "").strip().strip("\r").strip()

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _resolve_model(self, model: str) -> str:
        settings = get_settings()
        # THINKING_MODELNAME may be a comma-separated list; the first entry is
        # the default when the caller does not pick a specific model.
        return self._clean_str(model) or settings.default_thinking_model

    def _build_payload(
        self,
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        enable_thinking: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build a Responses API body (MaaS / DashScope compatible-mode)."""
        payload: dict[str, Any] = {
            "model": model,
            "input": messages,
            "temperature": temperature,
            # Required by Qwen thinking models on the MaaS Responses endpoint
            # (qwen3.8-max, qwen3.7-plus/max, etc.).
            "enable_thinking": bool(enable_thinking),
        }
        if tools:
            payload["tools"] = tools
        if max_tokens and max_tokens > 0:
            # OpenAI Responses field name; DashScope compatible-mode accepts it.
            payload["max_output_tokens"] = max_tokens
        if stream:
            payload["stream"] = True
        return payload

    @staticmethod
    def _extract_output_text(data: dict[str, Any]) -> str:
        """Pull final assistant text out of a Responses API payload.

        Thinking models return a ``reasoning`` item before the ``message`` item;
        tool calls (web_search etc.) may also appear. Only message text is returned.
        """
        content = ""
        for item in data.get("output", []) or []:
            if item.get("type") != "message":
                continue
            parts = item.get("content", []) or []
            if isinstance(parts, str):
                content += parts
                break
            for part in parts:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype in {"output_text", "text"}:
                    content += part.get("text", "") or ""
                elif "text" in part and isinstance(part.get("text"), str):
                    content += part["text"]
            break

        if content:
            return content

        # Fallbacks seen on some gateways
        for key in ("output_text", "text"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    @staticmethod
    def _truncation_reason(data: dict[str, Any], *, max_tokens: int) -> str:
        """Return why generation stopped early, or ``""`` if it ended normally.

        The Responses API marks a budget-exhausted answer with
        ``status=incomplete`` plus ``incomplete_details.reason``; some gateways
        only set a per-item ``finish_reason``. When neither is present, output
        tokens landing exactly on the ceiling means the same thing, so that is
        the last check rather than the first.
        """
        details = data.get("incomplete_details")
        if isinstance(details, dict):
            reason = str(details.get("reason") or "").strip()
            if reason:
                return reason
        if str(data.get("status") or "").strip() == "incomplete":
            return "incomplete"
        for item in data.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            finish = str(item.get("finish_reason") or "").strip()
            if finish and finish != "stop":
                return finish
        if max_tokens > 0:
            usage = data.get("usage", {}) or {}
            produced = usage.get("output_tokens", usage.get("completion_tokens"))
            if isinstance(produced, int) and produced >= max_tokens:
                return "max_output_tokens"
        return ""

    @staticmethod
    def _is_retryable(status_code: int) -> bool:
        return status_code in {408, 429, 500, 502, 503, 504}

    def _compute_delay(self, attempt: int) -> float:
        delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
        jitter = random.uniform(0.8, 1.2)
        return delay * jitter

    @staticmethod
    def _compute_read_timeout(prompt_chars: int, max_tokens: int, *, tools: bool) -> float:
        """Compute read timeout that accounts for prompt size and expected output."""
        # Thinking models spend extra time on hidden reasoning tokens; tool-using
        # calls (web_search) add another network round-trip on the provider side.
        # The per-token term is 0.03s (~33 tok/s) against ~43 tok/s measured on
        # qwen3.8-max: enough headroom to absorb a slow hour, while keeping the
        # large budgets these calls now pass off the _TIMEOUT_CAP ceiling — a
        # request that caps out waits 15 minutes per attempt before failing.
        base = 150.0 if tools else 120.0
        timeout = base + (prompt_chars / 4) * 0.02 + max_tokens * 0.03
        return min(max(180.0, timeout), _TIMEOUT_CAP)

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        *,
        enable_thinking: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Send a Responses API request, return final assistant text."""
        model = self._resolve_model(model)
        payload = self._build_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            enable_thinking=enable_thinking,
            tools=tools,
        )

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        base_read_timeout = self._compute_read_timeout(
            prompt_chars, max_tokens, tools=bool(tools)
        )
        tool_names = ",".join(
            str(t.get("type", "?")) for t in (tools or []) if isinstance(t, dict)
        ) or "-"

        logger.info(
            f"LLM chat: model={model} prompt={prompt_chars} chars "
            f"max_tokens={max_tokens} timeout={base_read_timeout:.0f}s "
            f"url={self.base_url}/responses thinking={enable_thinking} tools={tool_names}"
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
                        f"{self.base_url}/responses",
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
                if resp.status_code in (400, 401, 403, 404):
                    body = (resp.text or "")[:800]
                    logger.error(
                        f"LLM chat: HTTP {resp.status_code} — not retrying; body={body!r}"
                    )
                    resp.raise_for_status()

                # ── Other retryable status codes ──
                if self._is_retryable(resp.status_code) and attempt <= self.max_retries:
                    delay = self._compute_delay(attempt)
                    logger.warning(
                        f"LLM chat: HTTP {resp.status_code} (attempt {attempt}/{self.max_retries}), "
                        f"retry in {delay:.1f}s body={(resp.text or '')[:300]!r}"
                    )
                    await asyncio.sleep(delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                content = self._extract_output_text(data)
                truncated = self._truncation_reason(data, max_tokens=max_tokens)
                total_elapsed = time.perf_counter() - t0

                usage = data.get("usage", {}) or {}
                prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens", "?"))
                completion_tokens = usage.get(
                    "output_tokens", usage.get("completion_tokens", "?")
                )
                logger.info(
                    f"LLM chat: DONE in {total_elapsed:.1f}s (http={req_elapsed:.1f}s) — "
                    f"tokens={prompt_tokens}+{completion_tokens} response={len(content)} chars"
                    + (f" truncated={truncated}" if truncated else "")
                )

                if not content:
                    # Retrying at the same budget would fail identically, so this
                    # leaves the retry loop instead of burning another attempt.
                    raise LLMEmptyResponseError(
                        f"LLM returned no text (reason={truncated or 'unknown'}); "
                        f"model={model} max_tokens={max_tokens} "
                        f"output_tokens={completion_tokens} thinking={enable_thinking}. "
                        "On the Responses API max_output_tokens covers reasoning as "
                        "well, so raise the budget for this call.",
                        reason=truncated,
                    )
                if truncated:
                    logger.warning(
                        f"LLM chat: output truncated ({truncated}) at "
                        f"max_tokens={max_tokens} — answer may be cut off"
                    )
                return content

            except httpx.ReadTimeout as e:
                req_elapsed = time.perf_counter() - t_req if "t_req" in dir() else 0
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
                req_elapsed = time.perf_counter() - t_req if "t_req" in dir() else 0
                # 4xx already logged and raised above; do not retry them here.
                status = e.response.status_code if e.response is not None else 0
                if status and status < 500:
                    raise
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
        *,
        enable_thinking: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream final answer tokens from the Responses API (skips reasoning deltas)."""
        model = self._resolve_model(model)
        payload = self._build_payload(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            enable_thinking=enable_thinking,
            tools=tools,
        )

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        read_timeout = self._compute_read_timeout(
            prompt_chars, max_tokens, tools=bool(tools)
        )
        tool_names = ",".join(
            str(t.get("type", "?")) for t in (tools or []) if isinstance(t, dict)
        ) or "-"
        logger.info(
            f"LLM stream: model={model} prompt={prompt_chars} chars "
            f"timeout={read_timeout:.0f}s url={self.base_url}/responses "
            f"thinking={enable_thinking} tools={tool_names}"
        )
        t0 = time.perf_counter()
        token_count = 0

        timeout = httpx.Timeout(connect=30.0, read=read_timeout, write=30.0, pool=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/responses",
                headers=self._headers(),
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode("utf-8", errors="replace")[:800]
                    logger.error(
                        f"LLM stream: HTTP {resp.status_code} body={body!r}"
                    )
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str or data_str == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    event_type = chunk.get("type", "")
                    # Final answer tokens only — ignore reasoning/thinking deltas.
                    if event_type in {
                        "response.output_text.delta",
                        "response.content_part.delta",
                    }:
                        content = chunk.get("delta", "")
                        if isinstance(content, dict):
                            content = content.get("text", "") or content.get("delta", "")
                        if content:
                            token_count += 1
                            yield content
                    elif event_type == "response.completed":
                        break

        logger.info(f"LLM stream: DONE in {time.perf_counter()-t0:.1f}s — {token_count} chunks")
        if token_count == 0:
            # Same failure `chat` raises on: reasoning can spend the whole
            # max_output_tokens budget and the text deltas never arrive.
            logger.error(
                f"LLM stream: no text deltas — model={model} max_tokens={max_tokens} "
                f"thinking={enable_thinking}; the budget covers reasoning too"
            )


def get_llm_service() -> LLMService:
    return LLMService()
