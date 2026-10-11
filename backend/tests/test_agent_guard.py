"""P13: what a turn does when it runs out of something, or something breaks.

The model is scripted and the awkward tools are made up here, so every case is
deterministic: a tool that raises, one that never returns, a model that
answers nothing, a gateway that dies before or after it has written text.
Acceptance cases A32–A56 of `DEV_DOC/P13_对话运行兜底与安全加固.md`.
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool

from app.agents.context import AgentContext, BudgetUsage, RunBudget
from tests.test_agent_p2 import AgentP2TestCase


class GatewayDown(Exception):
    """Shaped like an SDK error: a status code and the gateway's own wording."""

    def __init__(self, status_code: int = 503, text: str = "") -> None:
        super().__init__(
            text or "Service unavailable at https://gw.internal.example/v1/responses "
            "key sk-abcdefghijklmnopqrstuvwxyz123456"
        )
        self.status_code = status_code


class StepModel(BaseChatModel):
    """Plays a list of steps, one per model call.

    A step is `("tools", [(name, args), ...], prose)`, `("text", str)`,
    `("empty",)`, `("cut", str, metadata)`, `("raise", exc)` or
    `("raise_after_text", exc)`. A call made without tools — the closing call —
    answers `closing` instead of taking a step, unless `ignores_closing`.
    """

    steps: list[tuple] = []
    closing: str = "Closing answer from what was gathered."
    ignores_closing: bool = False
    tools_bound: bool = False
    # Shared between the copies `bind_tools` makes.
    state: dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return "step-model"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "scholar_maas"
        return params

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self.model_copy(update={"tools_bound": bool(tools)})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        calls = self.state.setdefault("calls", [])
        calls.append({"tools_bound": self.tools_bound, "messages": list(messages)})
        if not self.tools_bound and not self.ignores_closing:
            return self._result(AIMessage(content=self.closing))
        idx = self.state.get("step", 0)
        self.state["step"] = idx + 1
        step = self.steps[idx] if idx < len(self.steps) else ("text", "done")
        kind = step[0]
        if kind == "tools":
            prose = step[2] if len(step) > 2 else ""
            return self._result(
                AIMessage(
                    content=prose,
                    tool_calls=[
                        {"name": name, "args": args, "id": f"c{idx}_{n}", "type": "tool_call"}
                        for n, (name, args) in enumerate(step[1])
                    ],
                )
            )
        if kind == "text":
            return self._result(AIMessage(content=step[1]))
        if kind == "empty":
            return self._result(AIMessage(content=""))
        if kind == "cut":
            return self._result(AIMessage(content=step[1], response_metadata=step[2]))
        if kind == "raise_after_text":
            from app.agents.model_factory import STREAM_PROBE

            probe = STREAM_PROBE.get()
            if probe is not None:
                probe.text = True
            raise step[1]
        raise step[1]

    @staticmethod
    def _result(message: AIMessage) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=message)])


class StreamingStepModel(StepModel):
    """`StepModel` that streams, so a call can die with half its text sent.

    Adds the step `("stream_then_raise", text, exc)`.
    """

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        idx = self.state.get("step", 0)
        step = self.steps[idx] if idx < len(self.steps) else ("text", "done")
        if self.tools_bound and step[0] == "stream_then_raise":
            self.state["step"] = idx + 1
            self.state.setdefault("calls", []).append(
                {"tools_bound": True, "messages": list(messages)}
            )
            for word in step[1].split(" "):
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=word + " ", id="m-cut"))
                if run_manager is not None:
                    await run_manager.on_llm_new_token(word + " ", chunk=chunk)
                yield chunk
            raise step[2]
        message = self._generate(messages).generations[0].message
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=message.content,
                tool_call_chunks=[
                    {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": n}
                    for n, c in enumerate(message.tool_calls)
                ],
                response_metadata=message.response_metadata,
            )
        )


def _envelope(**data: Any) -> str:
    from app.models.agent_models import ToolResult

    return ToolResult.ok(data).to_json()


@tool
async def note_tool(n: int) -> str:
    """Returns at once."""
    return _envelope(n=n)


@tool
async def slow_tool(seconds: float) -> str:
    """Takes its time."""
    await asyncio.sleep(seconds)
    return _envelope(slept=seconds)


@tool
async def boom_tool(n: int) -> str:
    """Raises something no tool maps."""
    raise RuntimeError(f"relation agent_things does not exist at http://db.internal:5432/x ({n})")


TEST_TOOLS = [note_tool, slow_tool, boom_tool]


class GuardTestCase(AgentP2TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from app.agents import guard
        from app.services import agent_runner

        self._runner = agent_runner
        self._saved = (
            agent_runner.build_chat_model,
            agent_runner.ALL_AGENT_TOOLS,
            agent_runner._fallback_model,
            dict(guard.TOOL_SECONDS),
        )
        agent_runner.ALL_AGENT_TOOLS = [*agent_runner.ALL_AGENT_TOOLS, *TEST_TOOLS]
        agent_runner._fallback_model = lambda _name: (None, "")
        guard.TOOL_SECONDS.update({"note_tool": 5.0, "boom_tool": 5.0, "slow_tool": None})

    async def asyncTearDown(self) -> None:
        from app.agents import guard

        (
            self._runner.build_chat_model,
            self._runner.ALL_AGENT_TOOLS,
            self._runner._fallback_model,
            seconds,
        ) = self._saved
        guard.TOOL_SECONDS.clear()
        guard.TOOL_SECONDS.update(seconds)
        for name in ("AGENT_WRAPUP_SECONDS", "AGENT_RUN_CONCURRENCY", "AGENT_FALLBACK_MODELNAME"):
            os.environ.pop(name, None)
        await super().asyncTearDown()

    def _model(self, steps: list[tuple], **fields: Any) -> StepModel:
        model = StepModel(steps=steps, state={}, **fields)
        self._runner.build_chat_model = lambda *_a, **_k: model
        return model

    async def _new_run(self, budget: dict[str, Any] | None = None, session=None):
        from app.db import agent_repository as repo

        session = session or self.session
        run, _ = await repo.create_run(
            session_id=session.session_id, owner_id=self.owner, budget=budget or {}
        )
        return run

    async def _turn(self, steps, *, budget=None, **fields):
        model = self._model(steps, **fields)
        run = await self._new_run(budget)
        await self._runner._execute_turn(session=self.session, run=run, question="What?")
        return run, model

    async def _finished(self, run):
        from app.db import agent_repository as repo

        return await repo.get_run(run.run_id, owner_id=self.owner)

    async def _events(self, run, type: str = ""):
        from app.db import agent_repository as repo

        events = [e for e in await repo.list_events(self.session.session_id) if e.run_id == run.run_id]
        return [e for e in events if not type or e.type.value == type]

    async def _answer(self, run):
        from app.db import agent_repository as repo

        answers = [
            m
            for m in await repo.list_messages(self.session.session_id)
            if m.role.value == "assistant" and m.run_id == run.run_id
        ]
        return answers[-1] if answers else None

    async def _tool_codes(self, run) -> list[str]:
        out = []
        for e in await self._events(run):
            if e.type.value in {"tool.completed", "tool.failed"}:
                out.append((e.payload.get("error") or {}).get("code") or e.payload.get("status"))
        return out


def _notes(count: int, start: int = 0) -> list[tuple]:
    return [("tools", [("note_tool", {"n": start + i})]) for i in range(count)]


class BudgetWrapUpTests(GuardTestCase):
    async def test_a32_a_spent_tool_budget_ends_in_a_closing_answer(self) -> None:
        run, model = await self._turn(_notes(6), budget={"max_tool_calls": 3})

        finished = await self._finished(run)
        self.assertEqual(finished.status.value, "done")
        self.assertEqual(finished.usage["stopped_by"], "max_tool_calls")
        self.assertEqual(finished.usage["tool_calls"], 3)

        calls = model.state["calls"]
        self.assertEqual([c["tools_bound"] for c in calls], [True, True, True, False])
        self.assertIn("ceiling", calls[-1]["messages"][-1].content)

        answer = await self._answer(run)
        self.assertEqual(answer.content, model.closing)
        self.assertNotIn("[stopped", answer.content)
        self.assertEqual(answer.completion, "complete")
        self.assertEqual(answer.stop_reason, "max_tool_calls")
        completed = (await self._events(run, "run.completed"))[0]
        self.assertEqual(completed.payload["stop_reason"], "max_tool_calls")

    async def test_a33_calls_issued_together_cannot_overrun_the_budget(self) -> None:
        together = [("note_tool", {"n": i}) for i in range(5)]
        run, _ = await self._turn([("tools", together)], budget={"max_tool_calls": 2})

        finished = await self._finished(run)
        self.assertEqual(finished.usage["tool_calls"], 2)
        self.assertEqual(finished.usage["refused_tool_calls"], 3)
        codes = await self._tool_codes(run)
        self.assertEqual(sorted(codes), ["budget_exceeded"] * 3 + ["ok"] * 2)

    def test_a34_a_spent_parse_or_download_budget_does_not_end_the_turn(self) -> None:
        budget = RunBudget(max_parses=1, max_downloads=1)
        self.assertEqual(BudgetUsage(parses=1, downloads=1).exceeded(budget), "")

    async def test_a39_the_last_model_call_allowed_is_the_closing_one(self) -> None:
        run, model = await self._turn(_notes(4), budget={"max_llm_calls": 2})
        self.assertEqual([c["tools_bound"] for c in model.state["calls"]], [True, False])
        finished = await self._finished(run)
        self.assertEqual(finished.usage["stopped_by"], "max_llm_calls")
        self.assertEqual(finished.usage["llm_calls"], 2)

    async def test_a40_a_tool_call_in_the_closing_answer_is_dropped(self) -> None:
        steps = [("tools", [("note_tool", {"n": 0})], "First I looked at the outline. ")]
        steps += _notes(5, start=1)
        run, model = await self._turn(steps, budget={"max_tool_calls": 1}, ignores_closing=True)

        finished = await self._finished(run)
        self.assertEqual(finished.status.value, "done")
        self.assertEqual(finished.usage["tool_calls"], 1)
        # One closing call, asked once more because it wrote nothing; no third.
        self.assertEqual([c["tools_bound"] for c in model.state["calls"]], [True, False, False])
        self.assertEqual(len(await self._events(run, "tool.started")), 1)
        self.assertIn("First I looked", (await self._answer(run)).content)


class ToolFaultTests(GuardTestCase):
    async def test_a41_a_tool_that_raises_becomes_a_result_the_model_reads(self) -> None:
        run, model = await self._turn(
            [("tools", [("boom_tool", {"n": 1})]), ("text", "I could not look that up.")]
        )
        finished = await self._finished(run)
        self.assertEqual(finished.status.value, "done")
        self.assertEqual(finished.usage["errors"], 1)

        failed = (await self._events(run, "tool.failed"))[0]
        self.assertEqual(failed.payload["error"]["code"], "internal_error")
        seen = json.dumps([e.payload for e in await self._events(run)])
        seen += str(model.state["calls"][-1]["messages"][-1].content)
        self.assertNotIn("db.internal", seen)
        self.assertNotIn("agent_things", seen)

    async def test_a42_cancellation_passes_through_the_guard(self) -> None:
        from app.agents.guard import TurnGuardMiddleware

        async def cancelled(_request):
            raise asyncio.CancelledError

        request = SimpleNamespace(
            tool_call={"name": "note_tool", "args": {}, "id": "c1"},
            runtime=SimpleNamespace(context=self.ctx),
        )
        with self.assertRaises(asyncio.CancelledError):
            await TurnGuardMiddleware().awrap_tool_call(request, cancelled)

    async def test_a35_a_tool_past_its_time_limit_is_answered_for(self) -> None:
        from app.agents import guard

        guard.TOOL_SECONDS["slow_tool"] = 0.1
        run, _ = await self._turn(
            [("tools", [("slow_tool", {"seconds": 5})]), ("text", "Answered without it.")]
        )
        finished = await self._finished(run)
        self.assertEqual(finished.status.value, "done")
        self.assertEqual(finished.usage["tool_timeouts"], 1)
        self.assertLess(finished.usage["wall_seconds"], 4)
        self.assertEqual(await self._tool_codes(run), ["timeout"])

    async def test_a36_the_wall_clock_is_read_after_a_tool_that_yielded_nothing(self) -> None:
        run, model = await self._turn(
            [("tools", [("slow_tool", {"seconds": 1.2})]), ("tools", [("note_tool", {"n": 1})])],
            budget={"max_wall_seconds": 1},
        )
        finished = await self._finished(run)
        self.assertEqual(finished.status.value, "done")
        self.assertEqual(finished.usage["stopped_by"], "max_wall_seconds")
        self.assertEqual([c["tools_bound"] for c in model.state["calls"]], [True, False])

    async def test_a36_past_the_hard_limit_the_turn_ends_and_keeps_its_text(self) -> None:
        from app.config import get_settings

        os.environ["AGENT_WRAPUP_SECONDS"] = "0"
        get_settings.cache_clear()
        run, _ = await self._turn(
            [("tools", [("slow_tool", {"seconds": 30})], "Reading the method section. ")],
            budget={"max_wall_seconds": 1},
        )
        finished = await self._finished(run)
        self.assertEqual(finished.status.value, "failed")
        self.assertEqual(finished.error_code, "timeout")
        self.assertLess(finished.usage["wall_seconds"], 10)
        answer = await self._answer(run)
        self.assertIn("Reading the method", answer.content)
        self.assertEqual((answer.completion, answer.stop_reason), ("failed", "timeout"))

    async def test_a37_the_third_identical_call_is_not_run(self) -> None:
        same = ("tools", [("note_tool", {"n": 7})])
        run, _ = await self._turn([same, same, same, ("text", "ok")])
        self.assertEqual(await self._tool_codes(run), ["ok", "ok", "repeated_call"])

    def test_a37_repeats_reset_when_something_readable_changed(self) -> None:
        from app.agents.guard import TurnGuardMiddleware
        from app.models.agent_models import ToolResult
        from langchain_core.messages import ToolMessage

        guard = TurnGuardMiddleware()
        ctx = AgentContext(owner_id="o", session_id="s")

        def ask(name: str, **args: Any):
            request = SimpleNamespace(
                tool_call={"name": name, "args": args, "id": "c"},
                runtime=SimpleNamespace(context=ctx),
            )
            return guard._admit(ctx, request)

        self.assertIsNone(ask("search_papers", query="moe"))
        self.assertIsNone(ask("search_papers", query="moe"))
        guard._observe(
            ctx,
            "download_paper",
            ToolMessage(content=ToolResult.ok({}).to_json(), tool_call_id="c"),
        )
        self.assertIsNone(ask("search_papers", query="moe"))
        # A tool that polls outside state is never a repeat.
        for _ in range(4):
            self.assertIsNone(ask("ensure_paper_parsed", paper_id="p"))

    async def test_a38_repeated_failures_stop_the_gathering(self) -> None:
        steps = [("tools", [("boom_tool", {"n": i})]) for i in range(8)]
        run, model = await self._turn(steps)
        finished = await self._finished(run)
        self.assertEqual(finished.status.value, "done")
        self.assertEqual(finished.usage["stopped_by"], "tool_failures")
        self.assertEqual(finished.usage["tool_calls"], 5)
        self.assertFalse(model.state["calls"][-1]["tools_bound"])


class ModelFaultTests(GuardTestCase):
    async def test_a43_a_failure_keeps_what_was_written_and_hides_the_gateway(self) -> None:
        run, _ = await self._turn(
            [
                ("tools", [("note_tool", {"n": 1})], "The paper proposes a top-2 router. "),
                ("raise", GatewayDown()),
            ]
        )
        finished = await self._finished(run)
        self.assertEqual(finished.status.value, "failed")
        self.assertEqual(finished.error_code, "upstream_error")
        self.assertIn("GatewayDown", finished.error_msg)
        self.assertNotIn("gw.internal", finished.error_msg)
        self.assertNotIn("sk-abcdef", finished.error_msg)

        failed = (await self._events(run, "run.failed"))[0]
        self.assertEqual(failed.payload["code"], "upstream_error")
        self.assertTrue(failed.payload["partial"])
        self.assertNotIn("error", failed.payload)
        self.assertNotIn("gw.internal", json.dumps(failed.payload))

        answer = await self._answer(run)
        self.assertIn("top-2 router", answer.content)
        self.assertEqual((answer.completion, answer.stop_reason), ("failed", "upstream_error"))

    async def test_a44_stopping_a_turn_keeps_what_it_had_written(self) -> None:
        self._model([("tools", [("slow_tool", {"seconds": 30})], "So far: the method is sparse. ")])
        run = await self._new_run()
        task = asyncio.create_task(
            self._runner._execute_turn(session=self.session, run=run, question="What?")
        )
        for _ in range(100):
            await asyncio.sleep(0.05)
            if await self._events(run, "tool.started"):
                break
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual((await self._finished(run)).status.value, "cancelled")
        answer = await self._answer(run)
        self.assertIn("the method is sparse", answer.content)
        self.assertEqual((answer.completion, answer.stop_reason), ("stopped", "cancelled"))

    async def test_a45_an_empty_reply_is_asked_for_once_more(self) -> None:
        run, model = await self._turn([("empty",), ("text", "Here it is.")])
        self.assertEqual((await self._finished(run)).status.value, "done")
        self.assertEqual((await self._answer(run)).content, "Here it is.")
        self.assertEqual(len(model.state["calls"]), 2)

    async def test_a45_two_empty_replies_fail_the_turn_rather_than_finish_it(self) -> None:
        run, _ = await self._turn([("empty",), ("empty",), ("text", "too late")])
        finished = await self._finished(run)
        self.assertEqual(finished.status.value, "failed")
        self.assertEqual(finished.error_code, "empty_response")
        self.assertIsNone(await self._answer(run))

    async def test_a46_an_answer_cut_at_the_output_ceiling_is_marked(self) -> None:
        run, _ = await self._turn([("cut", "The three contributions are", {"finish_reason": "length"})])
        answer = await self._answer(run)
        self.assertEqual((answer.completion, answer.stop_reason), ("truncated", "max_output_tokens"))
        self.assertEqual((await self._finished(run)).status.value, "done")

    def test_a46_both_protocols_report_a_cut_answer(self) -> None:
        cut = self._runner._cut_short
        self.assertEqual(
            cut(AIMessage(content="x", response_metadata={"finish_reason": "length"})),
            "max_output_tokens",
        )
        self.assertEqual(
            cut(
                AIMessage(
                    content="x",
                    response_metadata={
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                    },
                )
            ),
            "max_output_tokens",
        )
        self.assertEqual(cut(AIMessage(content="x", response_metadata={"finish_reason": "stop"})), "")

    async def test_a47_a_model_that_fails_unwritten_hands_over_to_the_fallback(self) -> None:
        fallback = StepModel(steps=[("text", "Answer from the fallback.")], state={})
        self._runner._fallback_model = lambda _name: (fallback, "backup-model")
        run, _ = await self._turn([("raise", GatewayDown())])

        finished = await self._finished(run)
        self.assertEqual(finished.status.value, "done")
        self.assertEqual(finished.usage["model_used"], "backup-model")
        self.assertEqual((await self._answer(run)).content, "Answer from the fallback.")
        event = (await self._events(run, "model.fallback"))[0]
        self.assertEqual(event.payload["to"], "backup-model")
        self.assertEqual(event.payload["code"], "upstream_error")

    async def test_a47_the_turn_stays_on_the_fallback_once_it_moved(self) -> None:
        fallback = StepModel(
            steps=[("tools", [("note_tool", {"n": 1})]), ("text", "Still the fallback.")], state={}
        )
        self._runner._fallback_model = lambda _name: (fallback, "backup-model")
        run, primary = await self._turn([("raise", GatewayDown()), ("text", "primary again")])
        self.assertEqual((await self._answer(run)).content, "Still the fallback.")
        self.assertEqual(len(primary.state["calls"]), 1)

    async def test_a47_a_model_that_fails_after_writing_is_not_replaced(self) -> None:
        fallback = StepModel(steps=[("text", "Answer from the fallback.")], state={})
        self._runner._fallback_model = lambda _name: (fallback, "backup-model")
        run, _ = await self._turn([("raise_after_text", GatewayDown())])

        self.assertEqual((await self._finished(run)).status.value, "failed")
        self.assertFalse(await self._events(run, "model.fallback"))
        self.assertFalse(fallback.state.get("calls"))

    async def test_a47_a_request_that_cannot_succeed_elsewhere_is_not_retried(self) -> None:
        fallback = StepModel(steps=[("text", "Answer from the fallback.")], state={})
        self._runner._fallback_model = lambda _name: (fallback, "backup-model")
        run, _ = await self._turn(
            [("raise", GatewayDown(400, "data_inspection_failed: inappropriate content"))]
        )
        finished = await self._finished(run)
        self.assertEqual(finished.error_code, "content_filtered")
        self.assertFalse(fallback.state.get("calls"))

    def test_a48_a_fallback_that_is_not_a_conversation_model_is_refused(self) -> None:
        from app.agents.model_factory import fallback_model_name
        from app.config import get_settings

        os.environ["AGENT_FALLBACK_MODELNAME"] = "no-such-model"
        get_settings.cache_clear()
        with self.assertRaises(RuntimeError):
            fallback_model_name()

    def test_a49_errors_are_named_by_what_a_reader_can_do_about_them(self) -> None:
        from app.agents.errors import EmptyModelResponse, classify_error

        cases = [
            (GatewayDown(400, "This model's maximum context length is 131072 tokens"), "context_too_long"),
            (GatewayDown(429, "slow down"), "rate_limited"),
            (GatewayDown(502), "upstream_error"),
            (TimeoutError(), "timeout"),
            (EmptyModelResponse("nothing"), "empty_response"),
            (KeyError("missing"), "internal_error"),
        ]
        for exc, code in cases:
            self.assertEqual(classify_error(exc).value, code, repr(exc))


class ConversationStateTests(GuardTestCase):
    """A50: what the model is shown of a turn that did not end on its own."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        await self._runner.open_agent_checkpointer()

    async def asyncTearDown(self) -> None:
        await self._runner.close_agent_checkpointer()
        await super().asyncTearDown()

    async def _next_turn_sees(self) -> list[str]:
        model = self._model([("text", "second answer")])
        run = await self._new_run()
        await self._runner._execute_turn(session=self.session, run=run, question="And then?")
        self.assertEqual((await self._finished(run)).status.value, "done")
        return [str(m.content) for m in model.state["calls"][0]["messages"]]

    async def test_the_closing_answer_stays_and_its_instruction_does_not(self) -> None:
        await self._turn(_notes(3), budget={"max_tool_calls": 1})
        seen = await self._next_turn_sees()
        self.assertFalse([t for t in seen if "[System notice]" in t])
        self.assertTrue([t for t in seen if "Closing answer" in t])

    async def test_a50_text_streamed_before_a_failure_is_in_the_next_turn(self) -> None:
        model = StreamingStepModel(
            steps=[("stream_then_raise", "Half an answer about routing", GatewayDown())], state={}
        )
        self._runner.build_chat_model = lambda *_a, **_k: model
        run = await self._new_run()
        await self._runner._execute_turn(session=self.session, run=run, question="What?")
        self.assertEqual((await self._finished(run)).status.value, "failed")
        self.assertIn("Half an answer", (await self._answer(run)).content)

        seen = await self._next_turn_sees()
        self.assertTrue([t for t in seen if "Half an answer about routing" in t], seen[1:])

    async def test_a50_a_turn_stopped_inside_a_tool_leaves_a_readable_history(self) -> None:
        self._model([("tools", [("slow_tool", {"seconds": 30})], "So far: the method is sparse. ")])
        run = await self._new_run()
        task = asyncio.create_task(
            self._runner._execute_turn(session=self.session, run=run, question="What?")
        )
        for _ in range(100):
            await asyncio.sleep(0.05)
            if await self._events(run, "tool.started"):
                break
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        seen = await self._next_turn_sees()
        self.assertTrue([t for t in seen if "the method is sparse" in t], seen[1:])


class LimitsTests(GuardTestCase):
    def test_a54_a_turn_cannot_be_allowed_more_than_is_left_of_today(self) -> None:
        from app.api.agent import _run_budget
        from app.services.accounts import QuotaUsage

        ceiling = RunBudget.from_settings().max_tokens
        nearly = QuotaUsage(runs=3, tokens=950_000, runs_limit=10, tokens_limit=1_000_000)
        self.assertEqual(_run_budget(nearly)["max_tokens"], 50_000)
        unlimited = QuotaUsage(runs=3, tokens=950_000, runs_limit=0, tokens_limit=0)
        self.assertEqual(_run_budget(unlimited)["max_tokens"], ceiling)
        roomy = QuotaUsage(runs=0, tokens=0, runs_limit=10, tokens_limit=ceiling * 10)
        self.assertEqual(_run_budget(roomy)["max_tokens"], ceiling)

    async def _session(self):
        from app.db import agent_repository as repo

        return await repo.create_session(owner_id=self.owner, language="en")

    async def test_a55_a_principal_has_a_ceiling_on_turns_in_flight(self) -> None:
        from app.db import agent_repository as repo

        first, second, third = self.session, await self._session(), await self._session()
        for session in (first, second):
            await repo.create_run(
                session_id=session.session_id, owner_id=self.owner, max_active_for_owner=2
            )
        with self.assertRaises(repo.TooManyActiveRuns):
            await repo.create_run(
                session_id=third.session_id, owner_id=self.owner, max_active_for_owner=2
            )
        # No ceiling asked for, none applied.
        await repo.create_run(session_id=third.session_id, owner_id=self.owner)

    async def test_a55_two_requests_cannot_both_take_the_last_slot(self) -> None:
        from app.db import agent_repository as repo

        sessions = [self.session, await self._session(), await self._session()]
        results = await asyncio.gather(
            *(
                repo.create_run(
                    session_id=s.session_id, owner_id=self.owner, max_active_for_owner=1
                )
                for s in sessions
            ),
            return_exceptions=True,
        )
        refused = [r for r in results if isinstance(r, repo.TooManyActiveRuns)]
        self.assertEqual(len(refused), 2, results)

    async def test_a56_turns_past_the_process_ceiling_wait_and_then_run(self) -> None:
        from app.config import get_settings
        from app.db import agent_repository as repo

        os.environ["AGENT_RUN_CONCURRENCY"] = "1"
        get_settings.cache_clear()
        other = await self._session()

        self._model([("tools", [("slow_tool", {"seconds": 0.6})]), ("text", "first")])
        first_run = await self._new_run()
        first = asyncio.create_task(
            self._runner._execute_turn(session=self.session, run=first_run, question="one")
        )
        await asyncio.sleep(0.2)
        self._model([("text", "second")])
        second_run = await self._new_run(session=other)
        second = asyncio.create_task(
            self._runner._execute_turn(session=other, run=second_run, question="two")
        )
        await asyncio.sleep(0.2)
        waiting = await repo.get_run(second_run.run_id, owner_id=self.owner)
        # Claimed and heartbeating, so recovery leaves it alone; not answering yet.
        self.assertEqual(waiting.status.value, "running")
        self.assertEqual(waiting.worker_id, self._runner.WORKER_ID)
        self.assertFalse(await repo.list_messages(other.session_id))

        await asyncio.gather(first, second)
        done = await repo.get_run(second_run.run_id, owner_id=self.owner)
        self.assertEqual(done.status.value, "done")
        # Its clock started when it got the slot, not when it began to wait.
        self.assertLess(done.usage["wall_seconds"], 0.5)


class InputLimitApiTests(unittest.TestCase):
    """A53: what `POST /messages` refuses before anything is created."""

    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        os.environ["DATA_DIR"] = self._tmp.name
        from tests.pg_support import use_database_env

        use_database_env(self)
        from app.config import get_settings

        get_settings.cache_clear()
        from app.services import identity

        identity.reset_secret_cache()
        from fastapi.testclient import TestClient

        from app.main import app

        self._client_cm = TestClient(app)
        self.client = self._client_cm.__enter__()
        response = self.client.post("/api/agent/sessions", json={"language": "en"})
        self.session_id = response.json()["session_id"]
        self.headers = {"X-Agent-Token": response.headers["X-Agent-Token"]}

    def tearDown(self) -> None:
        self._client_cm.__exit__(None, None, None)
        os.environ.pop("DATA_DIR", None)
        from app.config import get_settings
        from app.services import identity

        identity.reset_secret_cache()
        get_settings.cache_clear()
        self._tmp.cleanup()

    def _post(self, **body: Any):
        return self.client.post(
            f"/api/agent/sessions/{self.session_id}/messages", json=body, headers=self.headers
        )

    def test_a53_an_overlong_message_creates_nothing(self) -> None:
        response = self._post(content="x" * 20_001)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"]["code"], "message_too_long")
        detail = self.client.get(f"/api/agent/sessions/{self.session_id}", headers=self.headers)
        self.assertEqual(detail.json()["runs"], [])
        self.assertEqual(detail.json()["messages"], [])

    def test_a53_blank_messages_and_long_paper_lists_are_refused(self) -> None:
        self.assertEqual(self._post(content="   \n ").status_code, 422)
        self.assertEqual(
            self._post(content="hi", paper_ids=[f"p{i}" for i in range(21)]).status_code, 422
        )


if __name__ == "__main__":
    unittest.main()
