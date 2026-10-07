from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from app.services import llm_service
from app.services.llm_service import LLMEmptyResponseError
from app.workflows.snap_subgraph import _generate_report


class _FakeLLM:
    """Answers each call with the next scripted response and records its kwargs."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _registry():
    profile = SimpleNamespace(sends_enable_thinking=True)
    return patch.object(
        llm_service, "get_registry", lambda: SimpleNamespace(profile=lambda model="": profile)
    )


def _spent() -> LLMEmptyResponseError:
    return LLMEmptyResponseError("nothing", reason="max_output_tokens")


async def _report(llm: _FakeLLM):
    with _registry():
        return await _generate_report(
            llm, context="paper", language="en", model="m", log_label="t"
        )


class TestSnapReportBudget(unittest.IsolatedAsyncioTestCase):
    async def test_a_spent_ceiling_fails_the_run_without_a_repair_attempt(self) -> None:
        """Thinking on, then off — and no repair prompt for a budget failure."""
        llm = _FakeLLM([_spent(), _spent(), "never asked"])
        with self.assertRaises(RuntimeError) as ctx:
            await _report(llm)
        self.assertEqual(len(llm.calls), 2)
        self.assertFalse(llm.calls[1]["enable_thinking"])
        self.assertIn("no text", str(ctx.exception))

    async def test_two_failed_calls_do_not_become_an_empty_report(self) -> None:
        llm = _FakeLLM([RuntimeError("gateway down"), RuntimeError("gateway down")])
        with self.assertRaises(RuntimeError):
            await _report(llm)

    async def test_prose_that_never_parses_is_still_kept(self) -> None:
        llm = _FakeLLM(["not json", "still not json"])
        report = await _report(llm)
        self.assertTrue(report.degraded)
        self.assertEqual(report.raw_markdown, "still not json")


if __name__ == "__main__":
    unittest.main()
