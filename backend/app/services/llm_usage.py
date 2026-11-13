"""Who pays for a model call made through `llm_service`.

A conversation's own model calls are counted by the turn from the stream. The
calls a tool makes on its way — the three report modes above all, but also a
rank lookup's extraction pass — go through `llm_service` instead and used to be
counted by nobody: a Research Sphere report could spend more than the turn
that asked for it, outside both the turn's token ceiling and the reader's
daily quota.

Whoever wants those tokens opens a meter around the work. The meter rides on a
context variable, so it reaches every call made inside the block — including
the ones issued from tasks the block starts, since a task copies the context it
was created in — without a parameter threaded through every subgraph. Meters
nest: a mode report made inside a turn is counted by the report's run row and
by the turn at once, and the daily quota reads only one of the two.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_METERS: contextvars.ContextVar[tuple[UsageMeter, ...]] = contextvars.ContextVar(
    "scholar_llm_usage_meters", default=()
)


@dataclass
class UsageMeter:
    """Tokens reported by the calls made while this meter was open.

    `sink`, when given, receives each call's total as it is reported, so a
    turn's ceiling sees a report's spending while the report is still running.
    """

    sink: Callable[[int], None] | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    # Calls whose response carried no usage at all: their cost is unknown,
    # which is not the same as zero.
    unreported_calls: int = 0
    _tokens: list[contextvars.Token[Any]] = field(default_factory=list, repr=False)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.calls += 1
        if input_tokens is None and output_tokens is None:
            self.unreported_calls += 1
            return
        spent = int(input_tokens or 0) + int(output_tokens or 0)
        self.input_tokens += int(input_tokens or 0)
        self.output_tokens += int(output_tokens or 0)
        if self.sink is not None and spent:
            self.sink(spent)

    # Usable as `with` and as `async with`, so it can share a line with
    # `asyncio.timeout(...)`.
    def __enter__(self) -> UsageMeter:
        self._tokens.append(_METERS.set((*_METERS.get(), self)))
        return self

    def __exit__(self, *exc: object) -> None:
        _METERS.reset(self._tokens.pop())

    async def __aenter__(self) -> UsageMeter:
        return self.__enter__()

    async def __aexit__(self, *exc: object) -> None:
        self.__exit__(*exc)


def metering(sink: Callable[[int], None] | None = None) -> UsageMeter:
    """A meter to open around work whose model calls someone must pay for."""
    return UsageMeter(sink=sink)


def record(input_tokens: Any, output_tokens: Any) -> None:
    """Credit one call's usage to every open meter. Called by `llm_service`."""
    meters = _METERS.get()
    if not meters:
        return
    parsed = tuple(_as_int(v) for v in (input_tokens, output_tokens))
    for meter in meters:
        meter.add(*parsed)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
