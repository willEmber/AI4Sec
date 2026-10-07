from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from app.services import llm_service
from app.services.llm_service import (
    LLMEmptyResponseError,
    LLMTruncatedResponseError,
    chat_complete,
)

_MESSAGES = [{"role": "user", "content": "paper"}]


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


def _registry(*, sends_enable_thinking: bool):
    profile = SimpleNamespace(sends_enable_thinking=sends_enable_thinking)
    return patch.object(
        llm_service, "get_registry", lambda: SimpleNamespace(profile=lambda model="": profile)
    )


def _truncated() -> LLMTruncatedResponseError:
    return LLMTruncatedResponseError("cut off", reason="max_output_tokens", partial="# 1.")


class TestChatComplete(unittest.IsolatedAsyncioTestCase):
    """The production failure: a 238-character report saved as a finished run."""

    async def test_a_complete_report_makes_one_call(self) -> None:
        llm = _FakeLLM(["# report"])
        with _registry(sends_enable_thinking=True):
            got = await chat_complete(
                llm, _MESSAGES, model="m", max_tokens=32768, log_label="t"
            )
        self.assertEqual(got, "# report")
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0]["max_tokens"], 32768)
        self.assertTrue(llm.calls[0]["require_complete"])

    async def test_a_truncated_report_is_rewritten_with_thinking_off(self) -> None:
        llm = _FakeLLM([_truncated(), "# report"])
        with _registry(sends_enable_thinking=True):
            got = await chat_complete(
                llm, _MESSAGES, model="m", max_tokens=32768, log_label="t"
            )
        self.assertEqual(got, "# report")
        self.assertNotIn("enable_thinking", llm.calls[0])
        self.assertFalse(llm.calls[1]["enable_thinking"])
        self.assertTrue(llm.calls[1]["require_complete"])

    async def test_an_empty_report_is_rewritten_too(self) -> None:
        llm = _FakeLLM(
            [LLMEmptyResponseError("nothing", reason="max_output_tokens"), "# report"]
        )
        with _registry(sends_enable_thinking=True):
            got = await chat_complete(
                llm, _MESSAGES, model="m", max_tokens=32768, log_label="t"
            )
        self.assertEqual(got, "# report")

    async def test_truncated_twice_fails_the_run(self) -> None:
        llm = _FakeLLM([_truncated(), _truncated()])
        with _registry(sends_enable_thinking=True):
            with self.assertRaises(LLMTruncatedResponseError):
                await chat_complete(
                    llm, _MESSAGES, model="m", max_tokens=32768, log_label="t"
                )
        self.assertEqual(len(llm.calls), 2)

    async def test_no_thinking_switch_means_no_second_call(self) -> None:
        """Without the switch the retry would be the first request again."""
        llm = _FakeLLM([_truncated(), "# never asked"])
        with _registry(sends_enable_thinking=False):
            with self.assertRaises(LLMTruncatedResponseError):
                await chat_complete(
                    llm, _MESSAGES, model="m", max_tokens=32768, log_label="t"
                )
        self.assertEqual(len(llm.calls), 1)

    async def test_an_empty_answer_with_no_cause_is_not_retried(self) -> None:
        llm = _FakeLLM([LLMEmptyResponseError("nothing"), "# never asked"])
        with _registry(sends_enable_thinking=True):
            with self.assertRaises(LLMEmptyResponseError):
                await chat_complete(
                    llm, _MESSAGES, model="m", max_tokens=32768, log_label="t"
                )
        self.assertEqual(len(llm.calls), 1)


if __name__ == "__main__":
    unittest.main()
