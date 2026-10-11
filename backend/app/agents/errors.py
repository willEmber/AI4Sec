"""What went wrong with a turn, as a code a reader can be shown (P13 §4.3).

A failed turn used to put `str(exc)` on the event stream. That is the
gateway's own wording — its URL, sometimes the request it rejected — and it
tells a reader nothing they can act on. A turn now ends with one of a few
codes; the text behind it stays in the log, and what is stored on the run is
the exception's type and a redacted summary.
"""

from __future__ import annotations

import re

from app.models.agent_models import ErrorCode


class EmptyModelResponse(RuntimeError):
    """The model returned neither text nor a tool call, and did so again."""


# Substrings a gateway uses for "this request did not fit". Kept beside the
# ones deepagents recognises (it retries those after compacting), plus the
# wording the base gateway uses.
_CONTEXT_MARKERS = (
    "context_length_exceeded",
    "contextwindowexceedederror",
    "maximum context length",
    "exceeds the context window",
    "exceeds the available context size",
    "context window exceeded",
    "context limit exceeded",
    "input tokens exceed the configured limit",
    "prompt is too long",
    "range of input length",
)
_FILTER_MARKERS = (
    "content_filter",
    "content management policy",
    "data_inspection_failed",
    "datainspectionfailed",
    "inappropriate content",
    "safety system",
)

_URL = re.compile(r"https?://\S+", re.IGNORECASE)
_BEARER = re.compile(r"(?i)\b(bearer|api[-_ ]?key|authorization)\b[\s:=]+\S+")
# Long unbroken tokens: API keys, signed ids. A sentence has spaces.
_SECRET = re.compile(r"\b[A-Za-z0-9_\-]{24,}\b")


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def classify_error(exc: BaseException) -> ErrorCode:
    """The code a failed turn is closed with.

    Decided on exception types and HTTP statuses where there are any, and on
    the gateway's wording only for the two cases it does not give a status of
    their own (context length, content filtering).
    """
    chain = _chain(exc)
    if any(isinstance(e, EmptyModelResponse) for e in chain):
        return ErrorCode.EMPTY_RESPONSE
    names = {type(e).__name__ for e in chain}
    text = " ".join(str(e) for e in chain).lower()
    statuses = {getattr(e, "status_code", None) for e in chain}

    if "ContextOverflowError" in names or (
        statuses & {400, 413, 422} and any(m in text for m in _CONTEXT_MARKERS)
    ):
        return ErrorCode.CONTEXT_TOO_LONG
    if any(m in text for m in _FILTER_MARKERS):
        return ErrorCode.CONTENT_FILTERED
    if 429 in statuses or "RateLimitError" in names:
        return ErrorCode.RATE_LIMITED
    if any(isinstance(e, TimeoutError) for e in chain) or names & {
        "APITimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "PoolTimeout",
        "WriteTimeout",
    }:
        return ErrorCode.TIMEOUT
    if any(isinstance(s, int) for s in statuses) or any(
        type(e).__module__.split(".")[0] in {"openai", "httpx", "httpcore"} for e in chain
    ):
        return ErrorCode.UPSTREAM_ERROR
    return ErrorCode.INTERNAL_ERROR


def redact(text: str, *, limit: int = 300) -> str:
    """`text` without the parts that identify a gateway or a credential."""
    text = _URL.sub("[url]", text or "")
    text = _BEARER.sub(r"\1 [redacted]", text)
    text = _SECRET.sub("[redacted]", text)
    return " ".join(text.split())[:limit]


def error_summary(exc: BaseException) -> str:
    """What is stored on the run: the exception's type and a redacted message."""
    return redact(f"{type(exc).__name__}: {exc}")
