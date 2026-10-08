"""P5: the reading agent as the single entry point.

Modes as tools, long-term memory, and context management. Nothing here calls a
model or a network: the mode subgraph is stubbed, the memory and eviction
logic are exercised directly, and the API is driven through TestClient.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.test_agent_p2 import AgentP2TestCase


# ── Memory ──────────────────────────────────────────────────────────────────


class MemoryRepositoryTests(AgentP2TestCase):
    async def test_memories_are_listed_newest_first_and_deduplicated(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import MemoryKind

        first = await repo.add_memory(owner_id=self.owner, content="Prefers tables", kind=MemoryKind.PREFERENCE)
        again = await repo.add_memory(owner_id=self.owner, content="Prefers tables", kind=MemoryKind.INSTRUCTION)
        self.assertEqual(first.memory_id, again.memory_id)
        self.assertEqual(again.kind, MemoryKind.INSTRUCTION)

        await repo.add_memory(owner_id=self.owner, content="Works on MoE routing")
        listed = await repo.list_memories(self.owner)
        self.assertEqual([m.content for m in listed][:1], ["Works on MoE routing"])
        self.assertEqual(len(listed), 2)

    async def test_memories_are_scoped_to_their_owner(self) -> None:
        from app.db import agent_repository as repo
        from app.services import identity

        other, _ = await identity.create_principal()
        mine = await repo.add_memory(owner_id=self.owner, content="Answer in Chinese")
        self.assertEqual(await repo.list_memories(other), [])
        self.assertFalse(await repo.deactivate_memory(mine.memory_id, owner_id=other))
        self.assertTrue(await repo.deactivate_memory(mine.memory_id, owner_id=self.owner))
        self.assertEqual(await repo.list_memories(self.owner), [])


class MemoryToolTests(AgentP2TestCase):
    async def test_save_list_forget_round_trip(self) -> None:
        from app.agents.tools import forget_memory, list_memories, save_memory
        from app.db import agent_repository as repo

        saved = await self._call(save_memory, content="Always compare methods in a table", kind="instruction")
        self.assertEqual(saved["status"], "ok")
        memory_id = saved["data"]["memory_id"]

        listed = await self._call(list_memories)
        self.assertEqual([m["memory_id"] for m in listed["data"]["memories"]], [memory_id])

        # The turn hears about it, so the sidebar can refresh while it runs.
        events = await repo.list_events(self.session.session_id)
        self.assertIn("memory.saved", [e.type.value for e in events])

        forgotten = await self._call(forget_memory, memory_id=memory_id)
        self.assertTrue(forgotten["data"]["forgotten"])
        self.assertEqual((await self._call(list_memories))["data"]["memories"], [])

    async def test_credentials_are_refused(self) -> None:
        from app.agents.tools import save_memory

        result = await self._call(save_memory, content="my api key is sk-abcdefghijklmnopqrstuvwxyz123456")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "forbidden")

    async def test_a_paper_handle_is_not_mistaken_for_a_credential(self) -> None:
        from app.agents.tools import save_memory
        from app.agents.tools.memory import looks_like_secret

        # A real `paper_id` is the SHA-1 of the file; the fixture's is not.
        paper_id = "a94a8fe5ccb19ba61c4c0873d391e987982fbbd3"
        self.assertTrue(looks_like_secret(f"baseline is {paper_id}"))
        with mock.patch(
            "app.agents.tools.memory._paper_handles", mock.AsyncMock(return_value={paper_id})
        ):
            saved = await self._call(
                save_memory, content=f"Thesis baseline is paper_id={paper_id}", kind="project"
            )
        self.assertEqual(saved["status"], "ok")

        # Only a handle this conversation holds is exempt: the same shape
        # from anywhere else is still treated as a token.
        stranger = "0123456789abcdef0123456789abcdef01234567"
        self.assertTrue(looks_like_secret(f"see {stranger}", {paper_id}))
        refused = await self._call(save_memory, content=f"Baseline is {stranger}")
        self.assertEqual(refused["error"]["code"], "forbidden")

    async def test_a_correction_replaces_the_memory_it_corrects(self) -> None:
        from app.agents.tools import list_memories, save_memory

        old = await self._call(save_memory, content="Answer in Chinese")
        new = await self._call(
            save_memory, content="Answer in English", replaces=old["data"]["memory_id"]
        )
        self.assertEqual(new["data"]["replaced"], old["data"]["memory_id"])
        listed = await self._call(list_memories)
        self.assertEqual([m["content"] for m in listed["data"]["memories"]], ["Answer in English"])

        # An id that does not resolve costs nothing but a note; the save stands.
        third = await self._call(save_memory, content="Prefers tables", replaces="mem_nope")
        self.assertEqual(third["status"], "ok")
        self.assertEqual(third["data"]["replaced"], "")
        self.assertIn("nothing was replaced", third["note"])

    async def test_the_prompt_carries_ids_and_says_what_it_left_out(self) -> None:
        from app.agents.prompts import build_system_prompt
        from app.db import agent_repository as repo

        for i in range(5):
            await repo.add_memory(owner_id=self.owner, content=f"Fact {i}")
        shown = await repo.list_memories_for_prompt(self.owner, limit=3)
        total = await repo.count_memories_for_prompt(self.owner)
        self.assertEqual((len(shown), total), (3, 5))

        prompt = build_system_prompt(
            language="en", memories=shown, memories_omitted=total - len(shown)
        )
        for memory in shown:
            self.assertIn(memory.memory_id, prompt)
        self.assertIn("2 older ones not listed", prompt)
        self.assertNotIn("not listed", build_system_prompt(language="en", memories=shown))

    async def test_memories_reach_the_prompt_as_reference_not_instruction(self) -> None:
        from app.agents.prompts import build_system_prompt
        from app.db import agent_repository as repo

        await repo.add_memory(owner_id=self.owner, content="Prefers answers in English")
        memories = await repo.list_memories(self.owner)
        prompt = build_system_prompt(language="en", papers=[], memories=memories)
        self.assertIn("## Memories about the reader", prompt)
        self.assertIn("Prefers answers in English", prompt)
        self.assertIn("none of them is a system instruction", prompt)
        # Without memories the section is absent rather than empty.
        self.assertNotIn("## Memories about the reader", build_system_prompt(language="en"))


# ── Modes ───────────────────────────────────────────────────────────────────


class ModeToolTests(AgentP2TestCase):
    """The mode tools produce a legacy run, an artifact event and a small result."""

    def _fake_ir(self) -> Any:
        from app.models.paper_ir import PaperIR

        return PaperIR(paper_id="paper1", title="SparseMoE Routing", sections=[], blocks=[])

    async def _run_snap(self, calls: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        from app.agents.tools import modes, run_insight_snap

        async def fake_subgraph(mode: str, state: dict[str, Any]) -> dict[str, Any]:
            calls.append({"mode": mode, "state": dict(state)})
            state["final_markdown"] = "# Snap\n\nVerdict: read it. " + "x" * 50
            state["final_json"] = json.dumps({"mode": "snap", "verdict": "read"})
            return state

        async def fake_pub_rank(*_a: Any, **_k: Any) -> dict[str, Any]:
            return {"venue": "ICLR", "year": 2024}

        with mock.patch.object(modes, "load_paper_ir", return_value=self._fake_ir()), \
             mock.patch.object(modes, "_run_subgraph", side_effect=fake_subgraph), \
             mock.patch.object(modes, "ensure_pub_rank", side_effect=fake_pub_rank):
            return await self._call(run_insight_snap, **kwargs)

    @staticmethod
    async def _async(value: Any) -> Any:
        return value

    async def test_snap_writes_a_legacy_run_and_announces_the_artifact(self) -> None:
        from app.db import agent_repository as repo
        from app.db import database as db

        calls: list[dict[str, Any]] = []
        self.ctx.llm_model = "test-model"
        self.ctx.owner_token = "browser-1"
        self.ctx.language = "en"
        result = await self._run_snap(calls, focus="Is the routing loss novel?")

        self.assertEqual(result["status"], "ok", result)
        data = result["data"]
        self.assertEqual(data["mode"], "snap")
        self.assertFalse(data["reused"])
        self.assertIn("Verdict: read it", data["report_excerpt"])
        self.assertEqual(data["report_url"], f"/paper/paper1/run/{data['run_id']}")
        # The report is on screen; the model is told to summarise, not cite.
        self.assertIn("do not reproduce", result["note"])

        run = await db.fetch_one("SELECT * FROM runs WHERE run_id = ?", (data["run_id"],))
        self.assertEqual(run["status"], "done")
        self.assertEqual(run["mode"], "snap")
        self.assertEqual(run["agent_session_id"], self.session.session_id)
        self.assertEqual(run["agent_run_id"], self.ctx.run_id)
        self.assertEqual(run["owner_token"], "browser-1")
        self.assertEqual(run["llm_model"], "test-model")
        self.assertEqual(run["user_question"], "Is the routing loss novel?")
        output = await db.fetch_one("SELECT * FROM run_outputs WHERE run_id = ?", (data["run_id"],))
        self.assertIn("Verdict", output["markdown"])

        # The subgraph got the pipeline's inputs.
        self.assertEqual(len(calls), 1)
        state = calls[0]["state"]
        self.assertEqual(state["paper_id"], "paper1")
        self.assertEqual(state["language"], "en")
        self.assertIn("paper_ir_json", state)
        self.assertEqual(json.loads(state["pub_rank_json"])["venue"], "ICLR")

        events = await repo.list_events(self.session.session_id)
        artifact = [e for e in events if e.type.value == "artifact.created"]
        self.assertEqual(len(artifact), 1)
        self.assertEqual(artifact[0].payload["run_id"], data["run_id"])
        self.assertEqual(artifact[0].payload["mode"], "snap")

        listed = await repo.list_session_artifacts(self.session.session_id)
        self.assertEqual([a.run_id for a in listed], [data["run_id"]])
        self.assertEqual(listed[0].paper_title, "SparseMoE Routing")

    async def test_asking_again_reuses_the_report(self) -> None:
        calls: list[dict[str, Any]] = []
        first = await self._run_snap(calls)
        second = await self._run_snap(calls)
        self.assertEqual(len(calls), 1, "the analysis ran twice for the same paper and mode")
        self.assertTrue(second["data"]["reused"])
        self.assertEqual(second["data"]["run_id"], first["data"]["run_id"])
        self.assertIn("already existed", second["note"])

    async def test_a_reused_report_is_listed_in_the_session_that_asked_again(self) -> None:
        from app.db import agent_repository as repo

        calls: list[dict[str, Any]] = []
        first = await self._run_snap(calls)
        other = await repo.create_session(owner_id=self.owner, language="en")
        await self._attach("paper1", other.session_id)
        self.ctx.session_id = other.session_id
        self.ctx.run_id = "run-other"
        second = await self._run_snap(calls)

        self.assertTrue(second["data"]["reused"])
        listed_other = await repo.list_session_artifacts(other.session_id)
        self.assertEqual([a.run_id for a in listed_other], [first["data"]["run_id"]])
        self.assertEqual(listed_other[0].agent_run_id, "run-other")
        listed_orig = await repo.list_session_artifacts(self.session.session_id)
        self.assertEqual([a.run_id for a in listed_orig], [first["data"]["run_id"]])
        self.assertEqual(listed_orig[0].agent_run_id, "run1")

    async def test_a_failing_analysis_closes_the_run_as_failed(self) -> None:
        from app.agents.tools import modes, run_logic_lens
        from app.db import database as db

        async def boom(mode: str, state: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("lens exploded")

        with mock.patch.object(modes, "load_paper_ir", return_value=self._fake_ir()), \
             mock.patch.object(modes, "_run_subgraph", side_effect=boom), \
             mock.patch.object(modes, "ensure_pub_rank", return_value={}):
            result = await self._call(run_logic_lens)

        self.assertEqual(result["status"], "error")
        self.assertIn("lens exploded", result["error"]["message"])
        run = await db.fetch_one(
            "SELECT status, error_msg FROM runs WHERE agent_session_id = ?",
            (self.session.session_id,),
        )
        self.assertEqual(run["status"], "failed")
        self.assertIn("lens exploded", run["error_msg"])

    async def test_a_paper_outside_the_session_cannot_be_analysed(self) -> None:
        from app.agents.tools import run_research_sphere

        result = await self._call(run_research_sphere, paper_id="someone-elses")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["error"]["code"], "paper_not_found")

    async def test_progress_from_the_pipeline_becomes_tool_progress_events(self) -> None:
        from app.agents.tools import modes, run_insight_snap
        from app.db import agent_repository as repo
        from app.workflows.progress import emit_progress

        async def fake_subgraph(mode: str, state: dict[str, Any]) -> dict[str, Any]:
            await emit_progress(state["run_id"], "sphere_init_from_pdf", "running")
            await emit_progress(state["run_id"], "sphere_init_from_pdf", "done")
            state["final_markdown"] = "# Report\n\n" + "y" * 40
            state["final_json"] = "{}"
            return state

        with mock.patch.object(modes, "load_paper_ir", return_value=self._fake_ir()), \
             mock.patch.object(modes, "_run_subgraph", side_effect=fake_subgraph), \
             mock.patch.object(modes, "ensure_pub_rank", return_value={}):
            result = await self._call(run_insight_snap)

        self.assertEqual(result["status"], "ok")
        events = await repo.list_events(self.session.session_id)
        progress = [e.payload for e in events if e.type.value == "tool.progress"]
        self.assertEqual([p["step"] for p in progress], ["sphere_init_from_pdf", "sphere_init_from_pdf"])
        self.assertEqual(progress[0]["tool"], "run_insight_snap")
        self.assertEqual(progress[1]["status"], "done")


# ── Context management ──────────────────────────────────────────────────────


class EvictionTests(unittest.TestCase):
    def _tool_message(self, name: str, size: int, evidence: list[str]) -> ToolMessage:
        payload = {
            "status": "ok",
            "data": {"paper_id": "p1", "section": "2 Method", "blocks": [{"text": "z" * size}]},
            "evidence_ids": evidence,
            "note": "",
        }
        return ToolMessage(content=json.dumps(payload), name=name, tool_call_id=f"c-{name}-{size}")

    def test_old_large_results_are_stubbed_and_keep_their_evidence_ids(self) -> None:
        from app.agents.middleware import evict_tool_results

        old = self._tool_message("read_paper_section", 5000, ["ev_a", "ev_b"])
        recent = self._tool_message("read_paper_section", 5000, ["ev_c"])
        messages = [HumanMessage("q1"), AIMessage(""), old, AIMessage("a1"), HumanMessage("q2"), AIMessage(""), recent]

        out, evicted = evict_tool_results(messages, keep_recent=1, max_chars=1000)
        self.assertEqual(evicted, 1)
        stub = json.loads(out[2].content)
        self.assertTrue(stub["evicted"])
        self.assertEqual(stub["evidence_ids"], ["ev_a", "ev_b"])
        self.assertEqual(stub["data"]["section"], "2 Method")
        self.assertLess(len(out[2].content), 600)
        # The recent one is untouched, and so is the state we were given.
        self.assertIs(out[6], recent)
        self.assertIn("zzzz", messages[2].content)

    def test_nothing_is_cut_while_the_conversation_is_small(self) -> None:
        from types import SimpleNamespace

        from app.agents.middleware import ToolResultEvictionMiddleware

        old = self._tool_message("read_paper_section", 5000, ["ev_a"])
        recent = self._tool_message("read_paper_section", 5000, ["ev_b"])
        messages = [HumanMessage("q1"), AIMessage(""), old, AIMessage("a1"), HumanMessage("q2"), AIMessage(""), recent]

        def request():
            return SimpleNamespace(
                messages=messages, override=lambda **kw: SimpleNamespace(**kw)
            )

        roomy = ToolResultEvictionMiddleware(keep_recent=1, max_chars=1000, trigger_tokens=50_000)
        self.assertIs(roomy._apply(request()).messages[2], old)

        full = ToolResultEvictionMiddleware(keep_recent=1, max_chars=1000, trigger_tokens=1_000)
        self.assertTrue(json.loads(full._apply(request()).messages[2].content)["evicted"])

        always = ToolResultEvictionMiddleware(keep_recent=1, max_chars=1000)
        self.assertTrue(json.loads(always._apply(request()).messages[2].content)["evicted"])

    def test_small_results_and_non_evictable_tools_are_left_alone(self) -> None:
        from app.agents.middleware import evict_tool_results

        small = self._tool_message("read_paper_section", 100, ["ev_a"])
        rank = self._tool_message("query_publication_rank", 5000, [])
        out, evicted = evict_tool_results([small, rank, self._tool_message("read_paper_section", 10, [])], keep_recent=1, max_chars=500)
        self.assertEqual(evicted, 0)
        self.assertIs(out[0], small)
        self.assertIs(out[1], rank)

    def test_results_the_model_has_not_seen_are_never_evicted(self) -> None:
        from app.agents.middleware import evict_tool_results

        earlier = self._tool_message("read_paper_section", 5000, ["ev_old"])
        batch = [self._tool_message("read_paper_section", 5000 + i, [f"ev_{i}"]) for i in range(8)]
        messages = [HumanMessage("q1"), AIMessage(""), earlier, AIMessage("a1"), HumanMessage("q2"), AIMessage(""), *batch]

        # One step of eight parallel reads against a window of six: all eight
        # reach the model, and only the result it has already answered goes.
        out, evicted = evict_tool_results(messages, keep_recent=6, max_chars=1000)
        self.assertEqual(evicted, 1)
        self.assertTrue(json.loads(out[2].content)["evicted"])
        for original, kept in zip(batch, out[6:]):
            self.assertIs(kept, original)

        # Once the model has answered them they age like any other result.
        out, evicted = evict_tool_results([*messages, AIMessage("a2")], keep_recent=6, max_chars=1000)
        self.assertEqual(evicted, 3)
        self.assertTrue(json.loads(out[7].content)["evicted"])
        self.assertIs(out[8], batch[2])


class TokenCountTests(unittest.TestCase):
    def test_chinese_is_counted_at_about_a_token_per_character(self) -> None:
        from langchain_core.messages.utils import count_tokens_approximately

        from app.agents.middleware import count_tokens_cjk_aware

        english = [HumanMessage("word " * 800)]
        self.assertEqual(count_tokens_cjk_aware(english), count_tokens_approximately(english))

        chinese = [HumanMessage("论文的方法部分提出了什么" * 100), AIMessage("", tool_calls=[
            {"name": "search_paper_content", "args": {"question": "实验设置" * 50}, "id": "c1"}
        ])]
        counted = count_tokens_cjk_aware(chinese)
        self.assertGreaterEqual(counted, 1400)
        self.assertLess(count_tokens_approximately(chinese), 500)

    def test_the_summariser_uses_it_and_passes_tool_schemas(self) -> None:
        from app.agents.middleware import build_agent_middleware, count_tokens_cjk_aware
        from tests.test_agent_p2 import ScriptedModel

        middleware = build_agent_middleware(ScriptedModel())[0]
        self.assertIs(middleware.token_counter, count_tokens_cjk_aware)
        self.assertTrue(middleware._counter_accepts_tools)
        # Nothing is cut from what the summariser reads: a ceiling keeps the
        # last tokens, and the previous compaction's summary is the first.
        earlier_summary = HumanMessage("上一次压缩的摘要：读者要求用中文回答。")
        to_summarize = [earlier_summary, *[HumanMessage("问题" * 20_000) for _ in range(8)]]
        self.assertGreater(count_tokens_cjk_aware(to_summarize), 260_000)
        kept = middleware._lc_helper._trim_messages_for_summary(to_summarize)
        self.assertEqual(len(kept), len(to_summarize))
        self.assertIs(kept[0], earlier_summary)

        tool = {"name": "t", "description": "x" * 4000, "parameters": {}}
        self.assertGreater(
            count_tokens_cjk_aware([HumanMessage("q")], tools=[tool]),
            count_tokens_cjk_aware([HumanMessage("q")]) + 900,
        )


class CompactionTests(unittest.TestCase):
    def test_a_compaction_leaves_no_file_and_no_pointer_to_one(self) -> None:
        """The model has no file tools, so the summary must not send it to a file."""
        from langgraph.checkpoint.memory import InMemorySaver

        from app.agents.harness import create_paper_agent
        from app.agents.middleware import ScholarSummarizationMiddleware
        from tests.test_agent_p2 import ScriptedModel

        agent = create_paper_agent(
            tools=[],
            system_prompt="read papers",
            model=ScriptedModel(final="an answer"),
            checkpointer=InMemorySaver(),
            middleware=[
                ScholarSummarizationMiddleware(
                    ScriptedModel(final="THE SUMMARY"), trigger_tokens=2_000, keep_messages=2
                )
            ],
        )
        config = {"configurable": {"thread_id": "compaction"}}
        for i in range(4):
            agent.invoke({"messages": [{"role": "user", "content": f"q{i} " + "word " * 1500}]}, config=config)

        state = agent.get_state(config).values
        event = state.get("_summarization_event")
        self.assertIsNotNone(event, "the conversation should have been compacted")
        summary = event["summary_message"].content
        self.assertIn("THE SUMMARY", summary)
        self.assertNotIn("conversation_history", summary)
        self.assertFalse(event.get("file_path"))
        self.assertFalse(state.get("files"))
        # Nothing was removed from the checkpoint: compaction is a cutoff.
        self.assertEqual(len(state["messages"]), 8)


class MiddlewareAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        from app.config import get_settings

        get_settings.cache_clear()

    def test_summarization_replaces_the_default_by_name(self) -> None:
        from app.agents.middleware import ScholarSummarizationMiddleware, build_agent_middleware
        from tests.test_agent_p2 import ScriptedModel

        stack = build_agent_middleware(ScriptedModel())
        self.assertEqual([m.name for m in stack], ["SummarizationMiddleware"])
        self.assertIsInstance(stack[0], ScholarSummarizationMiddleware)
        self.assertEqual(stack[0]._trigger_tokens, 260_000)
        self.assertEqual(stack[0]._keep_messages, 12)

    def test_without_a_model_only_eviction_is_installed(self) -> None:
        from app.agents.middleware import build_agent_middleware

        self.assertEqual([m.name for m in build_agent_middleware(None)], ["ToolResultEvictionMiddleware"])

    def test_the_agent_offers_the_mode_and_memory_tools(self) -> None:
        from app.agents.harness import model_visible_tool_names
        from app.agents.tools import ALL_AGENT_TOOLS

        names = set(model_visible_tool_names(ALL_AGENT_TOOLS))
        for expected in (
            "run_insight_snap", "run_logic_lens", "run_research_sphere",
            "save_memory", "list_memories", "forget_memory",
            "search_paper_content", "download_paper",
        ):
            self.assertIn(expected, names)
        self.assertNotIn("write_file", names)
        self.assertNotIn("task", names)


class ModeInstructionTests(unittest.TestCase):
    def test_mode_note_travels_with_the_turn_only(self) -> None:
        from app.agents.prompts import mode_instruction
        from app.services.agent_runner import _model_input

        self.assertEqual(mode_instruction("auto", "zh"), "")
        self.assertIn("run_insight_snap", mode_instruction("snap", "zh"))
        self.assertIn("run_research_sphere", mode_instruction("sphere", "en"))
        self.assertEqual(_model_input("hi", "auto", "zh"), "hi")
        self.assertTrue(_model_input("hi", "lens", "en").startswith("hi\n\n"))


# ── API ─────────────────────────────────────────────────────────────────────


class UnifiedApiTests(unittest.TestCase):
    def setUp(self) -> None:
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

    def tearDown(self) -> None:
        self._client_cm.__exit__(None, None, None)
        os.environ.pop("DATA_DIR", None)
        from app.config import get_settings
        from app.services import identity

        identity.reset_secret_cache()
        get_settings.cache_clear()
        self._tmp.cleanup()

    def _session(self) -> tuple[str, dict[str, str]]:
        response = self.client.post(
            "/api/agent/sessions", json={"language": "en", "owner_token": "browser-xyz"}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["session_id"], {"X-Agent-Token": response.headers["X-Agent-Token"]}

    def test_memories_can_be_added_listed_and_deleted(self) -> None:
        _sid, headers = self._session()
        created = self.client.post(
            "/api/agent/memories", json={"content": "Prefers short answers", "kind": "preference"}, headers=headers
        )
        self.assertEqual(created.status_code, 200, created.text)
        memory_id = created.json()["memory_id"]

        listed = self.client.get("/api/agent/memories", headers=headers).json()["memories"]
        self.assertEqual([m["memory_id"] for m in listed], [memory_id])

        refused = self.client.post(
            "/api/agent/memories", json={"content": "token=abcdefgh12345678"}, headers=headers
        )
        self.assertEqual(refused.status_code, 422)

        deleted = self.client.delete(f"/api/agent/memories/{memory_id}", headers=headers)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/api/agent/memories", headers=headers).json()["memories"], [])
        self.assertEqual(self.client.delete(f"/api/agent/memories/{memory_id}", headers=headers).status_code, 404)

    def test_session_detail_carries_artifacts_and_context_stats(self) -> None:
        sid, headers = self._session()
        detail = self.client.get(f"/api/agent/sessions/{sid}", headers=headers).json()
        self.assertEqual(detail["artifacts"], [])
        self.assertEqual(detail["context"], {"compactions": 0, "last_turn_tokens": None})
        self.assertEqual(detail["session"]["config"]["owner_token"], "browser-xyz")

    def test_attaching_papers_ignores_unknown_ids(self) -> None:
        sid, headers = self._session()
        response = self.client.post(
            f"/api/agent/sessions/{sid}/papers", json={"paper_ids": ["nope"]}, headers=headers
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["attached"], [])
        self.assertEqual(response.json()["papers"], [])

    def test_a_stranger_cannot_attach_to_the_session(self) -> None:
        sid, _headers = self._session()
        _other, other_headers = self._session()
        response = self.client.post(
            f"/api/agent/sessions/{sid}/papers", json={"paper_ids": []}, headers=other_headers
        )
        self.assertEqual(response.status_code, 404)

    def test_an_unknown_mode_is_rejected_by_validation(self) -> None:
        sid, headers = self._session()
        response = self.client.post(
            f"/api/agent/sessions/{sid}/messages",
            json={"content": "hi", "mode": "turbo"},
            headers=headers,
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
