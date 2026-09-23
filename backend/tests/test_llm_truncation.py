from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

import httpx

from app.services.llm_service import LLMEmptyResponseError, LLMService


def _response(
    *,
    output: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    status: str = "completed",
    incomplete_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "status": status,
        "output": output if output is not None else [],
        "usage": usage or {"input_tokens": 7288, "output_tokens": 4096},
    }
    if incomplete_details is not None:
        body["incomplete_details"] = incomplete_details
    return body


def _message(text: str) -> dict[str, Any]:
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


_REASONING_ONLY = [{"type": "reasoning", "summary": []}]


class TestTruncationReason(unittest.TestCase):
    """`_truncation_reason` has to read three gateway dialects, in priority order."""

    def test_incomplete_details_is_preferred(self) -> None:
        reason = LLMService._truncation_reason(
            _response(
                status="incomplete",
                incomplete_details={"reason": "max_output_tokens"},
            ),
            max_tokens=4096,
        )
        self.assertEqual(reason, "max_output_tokens")

    def test_bare_incomplete_status_is_reported(self) -> None:
        reason = LLMService._truncation_reason(
            _response(status="incomplete"), max_tokens=0
        )
        self.assertEqual(reason, "incomplete")

    def test_per_item_finish_reason_is_read(self) -> None:
        data = _response(
            output=[{"type": "message", "finish_reason": "length", "content": []}],
            usage={"output_tokens": 10},
        )
        self.assertEqual(LLMService._truncation_reason(data, max_tokens=4096), "length")

    def test_output_tokens_on_the_ceiling_is_the_fallback(self) -> None:
        """The failure seen in production: no status field, output pinned at the cap."""
        data = _response(output=_REASONING_ONLY, usage={"output_tokens": 4096})
        self.assertEqual(
            LLMService._truncation_reason(data, max_tokens=4096), "max_output_tokens"
        )

    def test_normal_completion_reports_nothing(self) -> None:
        data = _response(output=[_message("hello")], usage={"output_tokens": 120})
        self.assertEqual(LLMService._truncation_reason(data, max_tokens=4096), "")

    def test_finish_reason_stop_is_not_truncation(self) -> None:
        data = _response(
            output=[{"type": "message", "finish_reason": "stop", "content": []}],
            usage={"output_tokens": 120},
        )
        self.assertEqual(LLMService._truncation_reason(data, max_tokens=4096), "")


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, payload: dict[str, Any], **_: Any) -> None:
        self._payload = payload

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, *_: Any) -> bool:
        return False

    async def post(self, *_: Any, **__: Any) -> _FakeResponse:
        return _FakeResponse(self._payload)


def _patched_client(payload: dict[str, Any]):
    return patch.object(
        httpx, "AsyncClient", lambda **kw: _FakeClient(payload, **kw)
    )


class TestChatEmptyResponse(unittest.IsolatedAsyncioTestCase):
    """An empty answer must surface as an error, not as an empty string.

    Returning "" made the reasoning-ate-the-budget failure reach the JSON
    parsers, which reported it as "unparseable" — the wrong cause, and the
    reason the production run retried at a budget that could not work.
    """

    async def test_reasoning_only_response_raises_with_reason(self) -> None:
        payload = _response(
            output=_REASONING_ONLY,
            usage={"input_tokens": 7288, "output_tokens": 4096},
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
        service = LLMService(base_url="https://gw.test/v1", api_key="k")
        with _patched_client(payload):
            with self.assertRaises(LLMEmptyResponseError) as ctx:
                await service.chat([{"role": "user", "content": "hi"}], model="m", max_tokens=4096)
        self.assertEqual(ctx.exception.reason, "max_output_tokens")
        self.assertIn("max_tokens=4096", str(ctx.exception))

    async def test_empty_response_without_any_marker_still_raises(self) -> None:
        payload = _response(output=[], usage={"output_tokens": 0})
        service = LLMService(base_url="https://gw.test/v1", api_key="k")
        with _patched_client(payload):
            with self.assertRaises(LLMEmptyResponseError) as ctx:
                await service.chat([{"role": "user", "content": "hi"}], model="m", max_tokens=4096)
        self.assertEqual(ctx.exception.reason, "")

    async def test_truncated_but_non_empty_answer_is_returned(self) -> None:
        """Partial text is still worth having — only a total absence is fatal."""
        payload = _response(
            output=[_message("partial answer")],
            usage={"output_tokens": 4096},
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        )
        service = LLMService(base_url="https://gw.test/v1", api_key="k")
        with _patched_client(payload):
            got = await service.chat(
                [{"role": "user", "content": "hi"}], model="m", max_tokens=4096
            )
        self.assertEqual(got, "partial answer")

    async def test_normal_answer_is_unaffected(self) -> None:
        payload = _response(output=[_message("all good")], usage={"output_tokens": 12})
        service = LLMService(base_url="https://gw.test/v1", api_key="k")
        with _patched_client(payload):
            got = await service.chat([{"role": "user", "content": "hi"}], model="m")
        self.assertEqual(got, "all good")


class TestReadTimeoutBudget(unittest.TestCase):
    def test_large_budget_stays_off_the_cap(self) -> None:
        """The Snap report call must not sit at the 900s ceiling per attempt."""
        timeout = LLMService._compute_read_timeout(25654, 16384, tools=False)
        self.assertLess(timeout, 900.0)
        self.assertGreater(timeout, 600.0)

    def test_timeout_still_grows_with_the_budget(self) -> None:
        small = LLMService._compute_read_timeout(25654, 4096, tools=False)
        large = LLMService._compute_read_timeout(25654, 16384, tools=False)
        self.assertGreater(large, small)


if __name__ == "__main__":
    unittest.main()
