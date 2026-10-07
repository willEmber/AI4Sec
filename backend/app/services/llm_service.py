from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any

import httpx

from app.services.llm_gateway import RESPONSES, ModelProfile, Provider, get_registry
from app.services.llm_gateway.adapters import ResponsesAdapter, adapter_for, usage_tokens

logger = logging.getLogger("scholar.llm")

# Maximum timeout cap for any single LLM request (seconds). The answer arrives
# in one piece, so this has to outlast a call that uses its whole budget: the
# largest one (40960 tokens, the Lens report) takes ~1040s at the slowest rate
# measured on qwen3.8-max (39.5 tok/s).
_TIMEOUT_CAP = 1500.0

class LLMEmptyResponseError(RuntimeError):
    """The gateway accepted the request but returned no assistant text.

    ``reason`` carries the truncation cause when there is one. The common case
    is the output ceiling: on both protocols that budget covers reasoning
    *and* visible output, so a thinking model can spend the whole allowance
    before emitting a single token. Callers must not retry at the same size —
    the failure is deterministic. Raising beats returning ``""`` because an
    empty string reaches JSON parsers as "unparseable", which hides the cause.
    """

    def __init__(self, message: str, *, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason


class LLMTruncatedResponseError(RuntimeError):
    """The answer was cut off, and the caller asked for a complete one.

    Raised only under ``chat(..., require_complete=True)``. The usual cause is
    the same as for an empty answer — reasoning took most of the output
    ceiling — but some text got out, and that is the dangerous case: a report
    that stops after its first paragraph is not empty, so it is persisted as a
    finished one. ``partial`` keeps what did arrive.
    """

    def __init__(self, message: str, *, reason: str, partial: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.partial = partial


class LLMService:
    """Async chat client over the model registry, with retry and backoff.

    The model decides the gateway and the wire protocol (`llm_gateway`): the
    base gateway speaks the Responses API, an extension gateway speaks
    chat/completions. Retries, timeouts and the empty-answer rule are shared.

    Constructed with an explicit ``base_url``, the service talks to that one
    Responses endpoint for every model, as it did before there was a registry.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        max_retries: int = 5,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 30.0,
    ):
        base = get_registry().base
        self._pinned = base_url is not None
        self.base_url = self._clean_str(base_url if base_url is not None else base.base_url).rstrip("/")
        self.api_key = self._clean_str(api_key if api_key is not None else base.api_key)
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay

    @staticmethod
    def _clean_str(value: str | None) -> str:
        """Strip whitespace / CRLF that often leaks from Windows .env files."""
        return (value or "").strip().strip("\r").strip()

    @staticmethod
    def _headers(provider: Provider) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider.api_key}",
        }

    def _profile(self, model: str) -> ModelProfile:
        """The model's profile; empty picks the first configured model."""
        if self._pinned:
            pinned = Provider("pinned", self.base_url, self.api_key, RESPONSES)
            name = self._clean_str(model) or get_registry().default_model
            return ModelProfile(name=name, provider=pinned, sends_enable_thinking=True)
        return get_registry().profile(self._clean_str(model))

    @staticmethod
    def _extract_output_text(data: dict[str, Any]) -> str:
        return ResponsesAdapter.extract_text(data)

    @staticmethod
    def _truncation_reason(data: dict[str, Any], *, max_tokens: int) -> str:
        return ResponsesAdapter.truncation_reason(data, max_tokens=max_tokens)

    @staticmethod
    def _error_code(resp: httpx.Response) -> str:
        try:
            error = (resp.json() or {}).get("error")
        except Exception:  # noqa: BLE001 — a non-JSON error body has no code
            return ""
        return str(error.get("code") or "") if isinstance(error, dict) else ""

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
        # qwen3.8-max: enough headroom to absorb a slow hour. Only the largest
        # budget (40960) reaches _TIMEOUT_CAP — a request that caps out waits
        # that long per attempt before failing.
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
        require_complete: bool = False,
    ) -> str:
        """Send one chat request, return final assistant text.

        A truncated answer is returned as it is, with a warning, unless
        ``require_complete`` is set — then it raises `LLMTruncatedResponseError`.
        Set it where the text is kept as a finished document: a caller that
        parses the answer finds out from the parser, a report writer never does.
        """
        profile = self._profile(model)
        model = profile.name
        provider = profile.provider
        adapter = adapter_for(profile)
        url = f"{provider.base_url}{adapter.path}"
        payload = adapter.build_payload(
            profile,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
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
            f"url={url} thinking={enable_thinking} tools={tool_names}"
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
                        url,
                        headers=self._headers(provider),
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
                # New-API reports an unknown model as 503 `model_not_found`;
                # the status alone would have it retried five times.
                if resp.status_code in (400, 401, 403, 404) or (
                    resp.status_code >= 400 and self._error_code(resp) == "model_not_found"
                ):
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
                content = adapter.extract_text(data)
                truncated = adapter.truncation_reason(data, max_tokens=max_tokens)
                total_elapsed = time.perf_counter() - t0

                prompt_tokens, completion_tokens = (
                    "?" if count is None else count for count in usage_tokens(data)
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
                        "The output ceiling covers reasoning as well, so raise the "
                        "budget for this call.",
                        reason=truncated,
                    )
                if truncated and require_complete:
                    raise LLMTruncatedResponseError(
                        f"LLM answer cut off after {len(content)} chars "
                        f"(reason={truncated}); model={model} max_tokens={max_tokens} "
                        f"output_tokens={completion_tokens} thinking={enable_thinking}. "
                        "The output ceiling covers reasoning as well, so a retry "
                        "needs a larger budget or thinking switched off.",
                        reason=truncated,
                        partial=content,
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
                if status and (
                    status < 500 or self._error_code(e.response) == "model_not_found"
                ):
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


async def chat_complete(
    llm: Any,
    messages: list[dict[str, str]],
    *,
    model: str = "",
    temperature: float = 0.3,
    max_tokens: int,
    log_label: str,
) -> str:
    """A whole answer, or an exception — never one that was cut off.

    For calls on the reader's model with thinking on, where nothing downstream
    would notice a truncated answer: a report or translation is kept as it
    came, and a cut-off JSON object is reported as a parse error. When the
    ceiling is hit the call is repeated once with thinking off, which gives the
    whole budget to the answer. A larger budget is not the retry: the answer
    arrives in one piece, and the call that just ran out already took most of
    its timeout. Where the gateway takes no thinking switch the second call
    would be the first one again, so the failure is raised as it is.
    """
    try:
        return await llm.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            require_complete=True,
        )
    except (LLMEmptyResponseError, LLMTruncatedResponseError) as exc:
        if not exc.reason or not get_registry().profile(model).sends_enable_thinking:
            raise
        logger.warning(
            f"{log_label}: answer hit the output ceiling ({exc.reason}) at "
            f"max_tokens={max_tokens} — retrying with thinking off"
        )
    return await llm.chat(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_thinking=False,
        require_complete=True,
    )


def get_llm_service() -> LLMService:
    return LLMService()
