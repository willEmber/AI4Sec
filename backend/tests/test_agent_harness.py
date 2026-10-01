from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from app.agents.harness import (
    BUILTIN_HOST_TOOLS,
    DELEGATION_TOOL,
    create_paper_agent,
    model_visible_tool_names,
)
from app.agents.model_factory import SCHOLAR_PROVIDER, ScholarChatModel


@tool
def get_paper_outline(paper_id: str) -> str:
    """Return the section outline of a parsed paper."""
    return '{"status": "ok", "data": {"sections": []}, "evidence_ids": [], "error": null}'


@tool
def search_paper_content(question: str) -> str:
    """Search evidence inside parsed papers."""
    return '{"status": "ok", "data": {"hits": []}, "evidence_ids": [], "error": null}'


class ScriptedModel(BaseChatModel):
    """Emits canned tool calls turn by turn, then a plain answer.

    Lets the tests drive calls a correctly-configured model would never make,
    which is the only way to check what the harness does if one ever arrives.
    """

    script: list[tuple[str, dict[str, Any]]] = []
    turn: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = SCHOLAR_PROVIDER
        return params

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        idx = self.turn
        self.turn += 1
        if idx < len(self.script):
            name, args = self.script[idx]
            msg = AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": f"call_{idx}", "type": "tool_call"}],
            )
        else:
            msg = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _run(script: list[tuple[str, dict[str, Any]]], *, thread: str = "t") -> dict[str, Any]:
    model = ScriptedModel()
    model.script = script
    agent = create_paper_agent(
        tools=[get_paper_outline],
        system_prompt="test",
        model=model,
    )
    return agent.invoke(
        {"messages": [{"role": "user", "content": "go"}]},
        config={"configurable": {"thread_id": thread}},
    )


def _tool_replies(state: dict[str, Any]) -> list[str]:
    return [
        str(m.content)
        for m in state["messages"]
        if type(m).__name__ == "ToolMessage"
    ]


class ToolSurfaceTest(unittest.TestCase):
    """The model must be offered domain tools and nothing else."""

    def test_only_domain_tools_are_exposed(self) -> None:
        names = model_visible_tool_names([get_paper_outline, search_paper_content])
        self.assertEqual(sorted(names), ["get_paper_outline", "search_paper_content"])

    def test_no_builtin_or_delegation_tool_leaks(self) -> None:
        names = set(model_visible_tool_names([get_paper_outline]))
        self.assertEqual(names & BUILTIN_HOST_TOOLS, set())
        self.assertNotIn(DELEGATION_TOOL, names)


class HostBoundaryTest(unittest.TestCase):
    """Even a hallucinated built-in call must not reach the host.

    The harness profile does more than hide the built-ins from the model:
    `_ToolExclusionMiddleware` refuses them at execution too, so a call that
    arrives anyway is answered with "not available" instead of running.
    """

    def test_write_file_neither_touches_disk_nor_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "escaped.txt"
            state = _run(
                [("write_file", {"file_path": str(target), "content": "pwned"})],
                thread="write",
            )
            self.assertFalse(target.exists(), "built-in write_file reached the host filesystem")
            self.assertNotIn(str(target), state.get("files") or {})
            self.assertIn("not available", "\n".join(_tool_replies(state)).lower())

    def test_read_file_cannot_see_host_paths(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
            fh.write("host-secret")
            host_path = fh.name
        try:
            state = _run([("read_file", {"file_path": host_path})], thread="read")
            joined = "\n".join(_tool_replies(state))
            self.assertNotIn("host-secret", joined)
            self.assertIn("not available", joined.lower())
        finally:
            Path(host_path).unlink(missing_ok=True)

    def test_execute_has_no_shell(self) -> None:
        state = _run([("execute", {"command": "id"})], thread="exec")
        self.assertIn("not available", "\n".join(_tool_replies(state)).lower())


class CheckpointerTest(unittest.IsolatedAsyncioTestCase):
    """A thread's history must survive the saver being closed and reopened."""

    async def asyncSetUp(self) -> None:
        from tests.pg_support import use_database_env

        self.db_url = use_database_env(self)

    async def test_history_reloads_from_the_database(self) -> None:
        from app.agents.checkpointer import open_checkpointer

        config = {"configurable": {"thread_id": "resume-me"}}

        async with open_checkpointer(self.db_url) as saver:
            model = ScriptedModel()
            agent = create_paper_agent(
                tools=[get_paper_outline],
                system_prompt="test",
                model=model,
                checkpointer=saver,
            )
            await agent.ainvoke(
                {"messages": [{"role": "user", "content": "first turn"}]}, config=config
            )

        # Fresh saver, fresh agent, fresh pool — only the database carries state.
        async with open_checkpointer(self.db_url) as saver:
            model = ScriptedModel()
            agent = create_paper_agent(
                tools=[get_paper_outline],
                system_prompt="test",
                model=model,
                checkpointer=saver,
            )
            state = await agent.aget_state(config)
            restored = state.values.get("messages", [])
            self.assertTrue(restored, "no messages restored from the checkpoint tables")
            self.assertIn("first turn", str(restored[0].content))

            out = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "second turn"}]}, config=config
            )
            texts = [str(m.content) for m in out["messages"]]
            self.assertTrue(any("first turn" in t for t in texts))
            self.assertTrue(any("second turn" in t for t in texts))

    async def test_threads_are_isolated(self) -> None:
        from app.agents.checkpointer import open_checkpointer

        async with open_checkpointer(self.db_url) as saver:
            agent = create_paper_agent(
                tools=[get_paper_outline],
                system_prompt="test",
                model=ScriptedModel(),
                checkpointer=saver,
            )
            await agent.ainvoke(
                {"messages": [{"role": "user", "content": "session A secret"}]},
                config={"configurable": {"thread_id": "A"}},
            )
            state_b = await agent.aget_state({"configurable": {"thread_id": "B"}})
            self.assertEqual(state_b.values.get("messages", []), [])


class ModelFactoryTest(unittest.TestCase):
    """The gateway only serves /responses, so that path must be pinned on."""

    def test_model_pins_responses_api_and_provider(self) -> None:
        model = ScholarChatModel(
            model="qwen3.8-max",
            base_url="https://example.invalid/v1",
            api_key="test-key",
            use_responses_api=True,
        )
        self.assertTrue(model.use_responses_api)
        self.assertEqual(model._get_ls_params()["ls_provider"], SCHOLAR_PROVIDER)

    def test_build_chat_model_uses_settings(self) -> None:
        from app.agents import model_factory

        model = model_factory.build_chat_model("qwen3.7-plus")
        self.assertEqual(model.model_name, "qwen3.7-plus")
        self.assertTrue(model.use_responses_api)
        self.assertEqual(model._get_ls_params()["ls_provider"], SCHOLAR_PROVIDER)


if __name__ == "__main__":
    unittest.main()
