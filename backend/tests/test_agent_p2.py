"""P2: the reading tools, the API surface and one full turn end to end.

No network: the model is scripted, so what is exercised is the harness around
it — authorization, evidence registration, event translation, idempotency and
the stream. Whether a *real* model reads well is P2's other half and lives in
`scripts/verify_reading_agent.py`.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agents.context import AgentContext


class ScriptedModel(BaseChatModel):
    """Emits canned tool calls turn by turn, then a final answer."""

    script: list[tuple[str, dict[str, Any]]] = []
    final: str = "done"
    turn: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "scholar_maas"
        return params

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        idx = self.turn
        self.turn += 1
        if idx < len(self.script):
            name, args = self.script[idx]
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {"name": name, "args": args, "id": f"c{idx}", "type": "tool_call"}
                            ],
                        )
                    )
                ]
            )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.final))])


class AgentP2TestCase(unittest.IsolatedAsyncioTestCase):
    """Temp database seeded with one parsed paper attached to one session."""

    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["DATA_DIR"] = self._tmp.name

        from app.config import get_settings

        get_settings.cache_clear()

        from app.services import identity
        from tests.pg_support import open_fresh_database

        identity.reset_secret_cache()
        await open_fresh_database(self)

        from app.db import agent_repository as repo

        self.owner, self.credential = await identity.create_principal()
        self.session = await repo.create_session(owner_id=self.owner, language="en")
        await self._seed_paper("paper1")
        await self._attach("paper1", self.session.session_id)

        self.ctx = AgentContext(
            owner_id=self.owner,
            session_id=self.session.session_id,
            run_id="run1",
            thread_id=self.session.thread_id,
        )

    async def asyncTearDown(self) -> None:
        os.environ.pop("DATA_DIR", None)
        from app.config import get_settings
        from app.services import identity

        identity.reset_secret_cache()
        get_settings.cache_clear()
        self._tmp.cleanup()

    async def _seed_paper(self, paper_id: str, title: str = "SparseMoE Routing") -> None:
        """Insert a paper plus the node hierarchy the reading tools query."""
        from app.db import database as db

        await db.execute(
            "INSERT INTO papers (paper_id, file_path, title) VALUES (?, ?, ?)",
            (paper_id, f"papers/{paper_id}/original.pdf", title),
        )
        nodes = [
            (f"{paper_id}:paper", "", 0, "paper", "", "SparseMoE Routing",
             "SparseMoE Routing", 0, 7, 0, 99, "SparseMoE Routing", 0),
            (f"{paper_id}:s1", f"{paper_id}:paper", 1, "section", "", "1 Introduction",
             "1 Introduction", 0, 1, 0, 9, "", 1),
            (f"{paper_id}:s2", f"{paper_id}:paper", 1, "section", "", "2 Method",
             "2 Method", 2, 4, 10, 29, "", 2),
            (f"{paper_id}:s3", f"{paper_id}:paper", 1, "section", "", "3 Experiments",
             "3 Experiments", 5, 7, 30, 49, "", 3),
        ]
        for node_id, parent, depth, ntype, btype, title, path, p0, p1, b0, b1, text, order in nodes:
            await db.execute(
                """INSERT INTO paper_nodes
                       (node_id, paper_id, parent_id, depth, node_type, block_type, sub_type,
                        title, title_path, page_start, page_end, block_start, block_end,
                        text, text_for_search, order_idx)
                   VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (node_id, paper_id, parent, depth, ntype, btype, title, path,
                 p0, p1, b0, b1, text, f"{title} {text}", order),
            )
        chunks = [
            (f"{paper_id}:c10", f"{paper_id}:s2", "text", 2, 10,
             "We propose a top-2 gated mixture-of-experts router trained with an "
             "auxiliary load-balancing loss."),
            (f"{paper_id}:c11", f"{paper_id}:s2", "equation", 3, 11,
             r"$$g(x) = \mathrm{softmax}(W_g x)$$"),
            (f"{paper_id}:c30", f"{paper_id}:s3", "text", 5, 30,
             "On WMT14 English-German the router reaches 29.7 BLEU."),
            (f"{paper_id}:c31", f"{paper_id}:s3", "table", 6, 31,
             "| model | BLEU |\n| --- | --- |\n| dense | 28.4 |\n| ours | 29.7 |"),
        ]
        for node_id, parent, btype, page, order, text in chunks:
            section_title = "2 Method" if parent.endswith("s2") else "3 Experiments"
            await db.execute(
                """INSERT INTO paper_nodes
                       (node_id, paper_id, parent_id, depth, node_type, block_type, sub_type,
                        title, title_path, page_start, page_end, block_start, block_end,
                        text, text_for_search, order_idx)
                   VALUES (?, ?, ?, 2, 'chunk', ?, '', '', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (node_id, paper_id, parent, btype, section_title, page, page,
                 order, order, text, text, order),
            )

    async def _attach(self, paper_id: str, session_id: str) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import Availability
        from app.services import paper_catalog

        literature_id = await paper_catalog.ensure_literature_for_local_paper(paper_id)
        await repo.attach_session_paper(
            session_id=session_id,
            literature_id=literature_id,
            paper_id=paper_id,
            availability=Availability.PARSED,
        )

    async def _call(self, tool, **kwargs) -> dict[str, Any]:
        """Invoke a tool with this test's runtime context injected.

        Builds the `ToolRuntime` the agent would normally construct, so the
        tool runs exactly as it does inside a turn.
        """
        from langchain.tools import ToolRuntime

        runtime = ToolRuntime(
            state={},
            context=self.ctx,
            config={"configurable": {"thread_id": self.ctx.thread_id}},
            stream_writer=lambda _chunk: None,
            tool_call_id="test-call",
            store=None,
            tools=[],
        )
        raw = await tool.ainvoke({**kwargs, "runtime": runtime})
        return json.loads(raw)


class ReadingToolTests(AgentP2TestCase):
    async def test_outline_lists_sections_with_reader_page_numbers(self) -> None:
        from app.agents.tools import get_paper_outline

        result = await self._call(get_paper_outline)
        self.assertEqual(result["status"], "ok")
        titles = [s["title"] for s in result["data"]["sections"]]
        self.assertEqual(titles, ["1 Introduction", "2 Method", "3 Experiments"])
        # Stored 0-based, shown 1-based.
        self.assertEqual(result["data"]["sections"][1]["page"], 3)

    async def test_search_returns_citable_evidence(self) -> None:
        from app.agents.tools import search_paper_content
        from app.services import evidence_service

        result = await self._call(search_paper_content, question="What BLEU score?")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["evidence_ids"])
        hit = result["data"]["hits"][0]
        self.assertIn("evidence_id", hit)

        resolved = await evidence_service.resolve(
            result["evidence_ids"][0], owner_id=self.owner
        )
        self.assertEqual(resolved.paper_id, "paper1")
        self.assertTrue(resolved.quote)

    async def test_search_with_no_match_is_ok_not_an_error(self) -> None:
        from app.agents.tools import search_paper_content

        result = await self._call(
            search_paper_content, question="quantum chromodynamics lattice"
        )
        # "The paper does not discuss this" is a finding, not a failure; calling
        # it an error would invite a pointless retry.
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["hits"], [])
        self.assertIn("outline", result["note"])

    async def test_section_read_includes_equations_and_tables(self) -> None:
        from app.agents.tools import read_paper_section

        method = await self._call(read_paper_section, section_title="2 Method")
        self.assertEqual(method["status"], "ok")
        kinds = {b["type"] for b in method["data"]["blocks"]}
        self.assertIn("equation", kinds)

        experiments = await self._call(read_paper_section, section_title="Experiments")
        kinds = {b["type"] for b in experiments["data"]["blocks"]}
        self.assertIn("table", kinds)
        self.assertTrue(experiments["evidence_ids"])

    async def test_section_can_be_found_by_id_and_by_loose_title(self) -> None:
        from app.agents.tools import read_paper_section

        by_id = await self._call(read_paper_section, section_id="paper1:s2")
        by_title = await self._call(read_paper_section, section_title="Method")
        self.assertEqual(by_id["data"]["section"], by_title["data"]["section"])

    async def test_unknown_section_lists_what_exists(self) -> None:
        from app.agents.tools import read_paper_section

        result = await self._call(read_paper_section, section_title="Related Work")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "section_not_found")
        self.assertFalse(result["error"]["retryable"])
        self.assertIn("2 Method", result["error"]["message"])

    async def test_a_paper_outside_the_session_is_not_readable(self) -> None:
        """Guessing a paper_id must not read someone else's upload."""
        from app.agents.tools import get_paper_outline

        await self._seed_paper("paper2", title="Someone Else's Upload")   # attached to no session
        result = await self._call(get_paper_outline, paper_id="paper2")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["error"]["code"], "paper_not_found")

    async def test_ambiguous_paper_is_reported_rather_than_guessed(self) -> None:
        from app.agents.tools import get_paper_outline

        # A different work, not another file of the same one — two files of one
        # work share a literature_id and collapse into a single session entry.
        await self._seed_paper("paper3", title="A Second, Unrelated Paper")
        await self._attach("paper3", self.session.session_id)
        result = await self._call(get_paper_outline)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_argument")


class TurnExecutionTests(AgentP2TestCase):
    async def _run_turn(self, script, final: str):
        from app.db import agent_repository as repo
        from app.services import agent_runner

        run, _ = await repo.create_run(
            session_id=self.session.session_id, owner_id=self.owner
        )
        model = ScriptedModel()
        model.script = script
        model.final = final

        original = agent_runner.build_chat_model
        agent_runner.build_chat_model = lambda *_a, **_k: model
        try:
            await agent_runner._execute_turn(
                session=self.session, run=run, question="What did they measure?"
            )
        finally:
            agent_runner.build_chat_model = original
        return run

    async def test_turn_emits_the_contract_events_and_stores_citations(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import RunStatus

        # Discover the evidence id the search tool will mint, so the scripted
        # answer can cite it the way a real model would.
        from app.agents.tools import search_paper_content

        search = await self._call(search_paper_content, question="BLEU")
        evidence_id = search["evidence_ids"][0]

        run = await self._run_turn(
            [("search_paper_content", {"question": "BLEU"})],
            final=f"They report 29.7 BLEU [{evidence_id}].",
        )

        finished = await repo.get_run(run.run_id, owner_id=self.owner)
        self.assertEqual(finished.status, RunStatus.DONE)

        events = await repo.list_events(self.session.session_id)
        types = [e.type.value for e in events if e.run_id == run.run_id]
        self.assertEqual(types[0], "run.started")
        self.assertIn("tool.started", types)
        self.assertIn("tool.completed", types)
        self.assertIn("message.delta", types)
        self.assertEqual(types[-1], "run.completed")

        messages = await repo.list_messages(self.session.session_id)
        answer = [m for m in messages if m.role.value == "assistant"][-1]
        self.assertIn("29.7", answer.content)
        self.assertEqual(answer.citations, [evidence_id])

    async def test_an_invented_citation_is_not_recorded(self) -> None:
        from app.db import agent_repository as repo

        run = await self._run_turn(
            [("search_paper_content", {"question": "BLEU"})],
            final="They report 29.7 BLEU [ev_totally_made_up].",
        )
        messages = await repo.list_messages(self.session.session_id)
        answer = [m for m in messages if m.role.value == "assistant"][-1]
        # The text is preserved as the model wrote it, but a citation that was
        # never handed to it does not become a resolvable reference.
        self.assertEqual(answer.citations, [])
        self.assertEqual(
            (await repo.get_run(run.run_id, owner_id=self.owner)).status.value, "done"
        )

    async def test_tool_events_carry_the_status_not_the_payload(self) -> None:
        from app.db import agent_repository as repo

        run = await self._run_turn(
            [("read_paper_section", {"section_title": "Nowhere"})],
            final="That section does not exist in this paper.",
        )
        events = await repo.list_events(self.session.session_id)
        failed = [e for e in events if e.type.value == "tool.failed" and e.run_id == run.run_id]
        self.assertTrue(failed)
        self.assertEqual(failed[0].payload["error"]["code"], "section_not_found")
        # The full section text must not be duplicated into the event log.
        self.assertNotIn("blocks", failed[0].payload)

    async def test_a_failing_model_still_closes_its_run(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import RunStatus
        from app.services import agent_runner

        run, _ = await repo.create_run(
            session_id=self.session.session_id, owner_id=self.owner
        )

        def explode(*_a, **_k):
            raise RuntimeError("gateway exploded")

        original = agent_runner.build_chat_model
        agent_runner.build_chat_model = explode
        try:
            await agent_runner._execute_turn(
                session=self.session, run=run, question="anything"
            )
        finally:
            agent_runner.build_chat_model = original

        finished = await repo.get_run(run.run_id, owner_id=self.owner)
        self.assertEqual(finished.status, RunStatus.FAILED)
        self.assertIn("gateway exploded", finished.error_msg)
        events = await repo.list_events(self.session.session_id)
        self.assertIn("run.failed", [e.type.value for e in events])


class AgentApiTests(unittest.TestCase):
    """The HTTP surface, driven through TestClient with a scripted model."""

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

    def _new_session(self) -> tuple[str, str]:
        response = self.client.post("/api/agent/sessions", json={"language": "en"})
        self.assertEqual(response.status_code, 200, response.text)
        credential = response.headers.get("X-Agent-Token", "")
        self.assertTrue(credential, "a first-time caller was not issued a credential")
        return response.json()["session_id"], credential

    def test_first_request_mints_a_credential(self) -> None:
        session_id, credential = self._new_session()
        detail = self.client.get(
            f"/api/agent/sessions/{session_id}", headers={"X-Agent-Token": credential}
        )
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["session"]["session_id"], session_id)

    def test_another_principal_cannot_read_the_session(self) -> None:
        session_id, _ = self._new_session()
        _, intruder = self._new_session()
        response = self.client.get(
            f"/api/agent/sessions/{session_id}", headers={"X-Agent-Token": intruder}
        )
        self.assertEqual(response.status_code, 404)

    def test_missing_credential_is_rejected(self) -> None:
        session_id, _ = self._new_session()
        self.assertEqual(
            self.client.get(f"/api/agent/sessions/{session_id}").status_code, 401
        )

    def test_repeat_submission_returns_the_same_run(self) -> None:
        session_id, credential = self._new_session()
        headers = {"X-Agent-Token": credential}
        body = {"content": "hello", "client_request_id": "req-1"}

        first = self.client.post(
            f"/api/agent/sessions/{session_id}/messages", json=body, headers=headers
        )
        second = self.client.post(
            f"/api/agent/sessions/{session_id}/messages", json=body, headers=headers
        )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(second.status_code, 200, second.text)
        self.assertEqual(first.json()["run_id"], second.json()["run_id"])
        self.assertTrue(second.json()["deduplicated"])

    def test_empty_message_is_rejected(self) -> None:
        session_id, credential = self._new_session()
        response = self.client.post(
            f"/api/agent/sessions/{session_id}/messages",
            json={"content": "   "},
            headers={"X-Agent-Token": credential},
        )
        self.assertEqual(response.status_code, 422)

    def test_cancel_is_owner_scoped(self) -> None:
        session_id, credential = self._new_session()
        run_id = self.client.post(
            f"/api/agent/sessions/{session_id}/messages",
            json={"content": "hello"},
            headers={"X-Agent-Token": credential},
        ).json()["run_id"]

        _, intruder = self._new_session()
        self.assertEqual(
            self.client.post(
                f"/api/agent/runs/{run_id}/cancel", headers={"X-Agent-Token": intruder}
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.post(
                f"/api/agent/runs/{run_id}/cancel", headers={"X-Agent-Token": credential}
            ).status_code,
            200,
        )

    def test_unknown_evidence_is_404(self) -> None:
        _, credential = self._new_session()
        response = self.client.get(
            "/api/agent/evidence/ev_nope", headers={"X-Agent-Token": credential}
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
