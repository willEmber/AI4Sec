"""What a tool spends through `llm_service` reaches whoever pays for it, and a
Logic Lens report reads the reader's focus instead of dropping it."""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any
from unittest.mock import patch

from app.agents.context import AgentContext, BudgetUsage, RunBudget
from app.services import llm_service, llm_usage
from app.services.llm_service import LLMEmptyResponseError, LLMService


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def post(self, *_a: Any, **_k: Any) -> _FakeResponse:
        await asyncio.sleep(0)
        return _FakeResponse(self._payload)


def _answer(text: str, *, input_tokens: int | None = 100, output_tokens: int | None = 20) -> dict[str, Any]:
    usage = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    output = [{"type": "message", "content": [{"type": "output_text", "text": text}]}] if text else []
    return {"status": "completed", "output": output, "usage": usage}


def _gateway(payload: dict[str, Any]):
    return patch.object(llm_service, "_shared_client", lambda: _FakeClient(payload))


class MeterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.llm = LLMService(base_url="http://gateway.test", api_key="k", max_retries=0)

    async def test_a_call_inside_the_meter_is_counted_and_one_outside_is_not(self) -> None:
        with _gateway(_answer("hi")):
            await self.llm.chat([{"role": "user", "content": "x"}])
            with llm_usage.metering() as meter:
                await self.llm.chat([{"role": "user", "content": "x"}])
        self.assertEqual((meter.input_tokens, meter.output_tokens, meter.calls), (100, 20, 1))
        self.assertEqual(meter.total, 120)

    async def test_meters_nest_and_tasks_started_inside_report_to_both(self) -> None:
        # A mode report inside a turn: the report's run row and the turn both pay,
        # and the report's passes run concurrently in tasks of their own.
        seen: list[int] = []
        with _gateway(_answer("hi")):
            async with llm_usage.metering(sink=seen.append) as turn:
                with llm_usage.metering() as report:
                    await asyncio.gather(*(
                        self.llm.chat([{"role": "user", "content": "x"}]) for _ in range(3)
                    ))
                await self.llm.chat([{"role": "user", "content": "x"}])
        self.assertEqual(report.total, 360)
        self.assertEqual(turn.total, 480)
        self.assertEqual(seen, [120] * 4, "the sink is fed per call, as it is spent")

    async def test_an_empty_answer_was_billed_all_the_same(self) -> None:
        with _gateway(_answer("", output_tokens=4096)):
            with llm_usage.metering() as meter:
                with self.assertRaises(LLMEmptyResponseError):
                    await self.llm.chat([{"role": "user", "content": "x"}])
        self.assertEqual(meter.total, 4196)

    async def test_a_call_without_usage_is_unknown_not_free(self) -> None:
        with _gateway(_answer("hi", input_tokens=None, output_tokens=None)):
            with llm_usage.metering() as meter:
                await self.llm.chat([{"role": "user", "content": "x"}])
        self.assertEqual((meter.total, meter.calls, meter.unreported_calls), (0, 1, 1))


class HardCeilingTests(unittest.TestCase):
    """The closing answer's allowance is measured from where the gathering stopped."""

    def _ctx(self, max_tokens: int = 1000) -> AgentContext:
        return AgentContext(owner_id="o", session_id="s", budget=RunBudget(max_tokens=max_tokens))

    def test_no_hard_stop_before_the_wrap_up(self) -> None:
        # A report pushed the turn far past its ceiling in one tool call; the
        # next model call starts the wrap-up rather than the stream ending here.
        ctx = self._ctx()
        ctx.usage.tokens = 5000
        self.assertEqual(ctx.usage.hard_exceeded(ctx.budget), "")
        self.assertEqual(ctx.usage.exceeded(ctx.budget), "max_tokens")

    def test_the_allowance_starts_where_the_wrap_up_began(self) -> None:
        ctx = self._ctx()
        ctx.usage.tokens = 5000
        ctx.begin_wrapup("max_tokens")
        self.assertEqual(ctx.usage.tokens_at_wrapup, 5000)
        ctx.usage.tokens = 5090
        self.assertEqual(ctx.usage.hard_exceeded(ctx.budget), "", "the closing answer may be written")
        ctx.usage.tokens = 5100
        self.assertEqual(ctx.usage.hard_exceeded(ctx.budget), "max_tokens")

    def test_a_wrap_up_below_the_ceiling_keeps_the_old_limit(self) -> None:
        ctx = self._ctx()
        ctx.usage.tokens = 300
        ctx.begin_wrapup("max_tool_calls")
        ctx.usage.tokens = 1099
        self.assertEqual(ctx.usage.hard_exceeded(ctx.budget), "")
        ctx.usage.tokens = 1100
        self.assertEqual(ctx.usage.hard_exceeded(ctx.budget), "max_tokens")

    def test_usage_without_any_report_has_no_hard_stop(self) -> None:
        usage = BudgetUsage()
        self.assertEqual(usage.hard_exceeded(RunBudget(max_tokens=10)), "")


if __name__ == "__main__":
    unittest.main()
