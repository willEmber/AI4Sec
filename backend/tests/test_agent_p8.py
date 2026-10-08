"""P8: research projects and cross-session recall.

Repository, tools, prompt and API, all against a throwaway PostgreSQL schema
and none of them calling a model: the message search, the turn folding and the
evidence hydration are exercised on seeded conversations.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from tests.test_agent_p2 import AgentP2TestCase


class P8TestCase(AgentP2TestCase):
    """The P2 fixture (one parsed paper on one session) plus helpers for turns and evidence."""

    async def _turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        citations: list[str] | None = None,
        run_id: str = "",
    ) -> str:
        """Record one finished question/answer pair the way a turn would."""
        import uuid

        from app.db import agent_repository as repo
        from app.models.agent_models import MessageRole

        run_id = run_id or f"ar_{uuid.uuid4().hex[:12]}"
        await repo.append_message(
            session_id=session_id, role=MessageRole.USER, content=question, run_id=run_id
        )
        await repo.append_message(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=answer,
            run_id=run_id,
            citations=citations or [],
        )
        return run_id

    async def _evidence(self, owner_id: str, session_id: str, quote: str, page: int = 5) -> str:
        from app.db import agent_repository as repo
        from app.models.evidence_models import Evidence, Locator, SourceLevel, compute_evidence_id
        from app.services import paper_catalog

        literature_id = await paper_catalog.ensure_literature_for_local_paper("paper1")
        locator = Locator(section_path="3 Experiments", page_index=page)
        evidence_id = compute_evidence_id(
            source_level=SourceLevel.FULLTEXT,
            quote=quote,
            literature_id=literature_id,
            paper_id="paper1",
            locator=locator,
            scope=owner_id,
        )
        await repo.insert_evidence(
            Evidence(
                evidence_id=evidence_id,
                owner_id=owner_id,
                session_id=session_id,
                literature_id=literature_id,
                paper_id="paper1",
                source_level=SourceLevel.FULLTEXT,
                locator=locator,
                quote=quote,
            )
        )
        return evidence_id


# ── Repository ──────────────────────────────────────────────────────────────


class ProjectRepositoryTests(P8TestCase):
    async def test_projects_are_owned_listed_and_archived(self) -> None:
        from app.db import agent_repository as repo
        from app.services import identity

        project = await repo.create_project(owner_id=self.owner, title=" MoE routing ", description="d")
        self.assertEqual(project.title, "MoE routing")
        self.assertEqual([p.project_id for p in await repo.list_projects(self.owner)], [project.project_id])

        other, _ = await identity.create_principal()
        with self.assertRaises(repo.ProjectNotFound):
            await repo.get_project(project.project_id, owner_id=other)
        with self.assertRaises(repo.ProjectNotFound):
            await repo.update_project(project.project_id, owner_id=other, title="mine now")
        self.assertEqual(await repo.list_projects(other), [])

        archived = await repo.update_project(project.project_id, owner_id=self.owner, status="archived")
        self.assertEqual(archived.status, "archived")
        self.assertEqual(archived.title, "MoE routing")   # untouched fields stay
        self.assertEqual(await repo.list_projects(self.owner), [])
        self.assertEqual(len(await repo.list_projects(self.owner, include_archived=True)), 1)

    async def test_a_session_cannot_be_filed_under_someone_elses_project(self) -> None:
        from app.db import agent_repository as repo
        from app.services import identity

        other, _ = await identity.create_principal()
        theirs = await repo.create_project(owner_id=other, title="theirs")
        with self.assertRaises(repo.ProjectNotFound):
            await repo.create_session(owner_id=self.owner, project_id=theirs.project_id)
        with self.assertRaises(repo.ProjectNotFound):
            await repo.set_session_project(
                self.session.session_id, owner_id=self.owner, project_id=theirs.project_id
            )
        mine = await repo.create_project(owner_id=self.owner, title="mine")
        with self.assertRaises(repo.SessionNotFound):
            await repo.set_session_project(
                self.session.session_id, owner_id=other, project_id=mine.project_id
            )

    async def test_project_papers_follow_their_sessions(self) -> None:
        """A30: papers are derived from the project's sessions, so moving a session out takes them along."""
        from app.db import agent_repository as repo

        project = await repo.create_project(owner_id=self.owner, title="P")
        await repo.set_session_project(self.session.session_id, owner_id=self.owner, project_id=project.project_id)
        second = await repo.create_session(owner_id=self.owner, project_id=project.project_id)
        self.assertEqual(second.project_id, project.project_id)
        self.assertEqual((await repo.get_project(project.project_id, owner_id=self.owner)).session_count, 2)

        others = await repo.list_project_papers(project.project_id, exclude_session_id=second.session_id)
        self.assertEqual([p.paper_id for p in others], ["paper1"])
        # The session that already has it does not see it as "other".
        self.assertEqual(
            await repo.list_project_papers(project.project_id, exclude_session_id=self.session.session_id),
            [],
        )
        self.assertIsNotNone(await repo.find_project_paper(project.project_id, "paper1"))

        await repo.set_session_project(self.session.session_id, owner_id=self.owner, project_id="")
        self.assertEqual(await repo.list_project_papers(project.project_id), [])
        self.assertIsNone(await repo.find_project_paper(project.project_id, "paper1"))

    async def test_project_papers_report_the_best_availability_once(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import Availability
        from app.services import paper_catalog

        project = await repo.create_project(owner_id=self.owner, title="P")
        await repo.set_session_project(self.session.session_id, owner_id=self.owner, project_id=project.project_id)
        weaker = await repo.create_session(owner_id=self.owner, project_id=project.project_id)
        literature_id = await paper_catalog.ensure_literature_for_local_paper("paper1")
        await repo.attach_session_paper(
            session_id=weaker.session_id,
            literature_id=literature_id,
            paper_id="paper1",
            availability=Availability.PDF_READY,
        )
        papers = await repo.list_project_papers(project.project_id)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].availability, Availability.PARSED)


class MessageSearchTests(P8TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from app.db import agent_repository as repo

        self.project = await repo.create_project(owner_id=self.owner, title="P")
        await repo.set_session_project(
            self.session.session_id, owner_id=self.owner, project_id=self.project.project_id
        )
        await self._turn(
            self.session.session_id,
            "稀疏专家路由在 WMT14 上的结果是多少？",
            "论文报告路由器在 WMT14 英德翻译上达到 29.7 BLEU。",
        )
        await self._turn(
            self.session.session_id,
            "Which gating does the router use?",
            "It uses top-2 gating with an auxiliary load-balancing loss.",
        )

    async def test_terms_segment_chinese_and_keep_compounds_whole_and_split(self) -> None:
        from app.db import agent_repository as repo

        terms = await repo.search_terms("专家路由 top-2 Gating of 29.7")
        for expected in ("专家", "家路", "路由", "top-2", "top", "gating", "29.7"):
            self.assertIn(expected, terms)
        self.assertNotIn("of", terms)   # stopword
        self.assertEqual(await repo.search_terms(""), [])

    async def test_chinese_query_finds_a_chinese_conversation(self) -> None:
        """A27: the 'simple' parser would have kept the whole Han run as one token."""
        from app.db import agent_repository as repo

        hits = await repo.search_messages(self.owner, "专家路由的结果")
        self.assertTrue(hits)
        self.assertIn("专家路由", hits[0][0].content)
        self.assertEqual(hits[0][1]["session_title"], self.session.title)

    async def test_a_compound_and_its_part_both_match(self) -> None:
        from app.db import agent_repository as repo

        for query in ("top-2", "top gating"):
            hits = await repo.search_messages(self.owner, query)
            self.assertTrue(any("top-2 gating" in m.content for m, _ in hits), query)

    async def test_search_is_scoped_to_owner_project_and_excludes_the_asking_session(self) -> None:
        """A26."""
        from app.db import agent_repository as repo
        from app.services import identity

        other, _ = await identity.create_principal()
        self.assertEqual(await repo.search_messages(other, "BLEU"), [])
        self.assertTrue(await repo.search_messages(self.owner, "BLEU", project_id=self.project.project_id))
        self.assertEqual(await repo.search_messages(self.owner, "BLEU", project_id=""), [])
        self.assertEqual(
            await repo.search_messages(self.owner, "BLEU", exclude_session_id=self.session.session_id),
            [],
        )


# ── Recall ──────────────────────────────────────────────────────────────────


class RecallTests(P8TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from app.agents.context import AgentContext
        from app.db import agent_repository as repo

        self.project = await repo.create_project(owner_id=self.owner, title="MoE")
        await repo.set_session_project(
            self.session.session_id, owner_id=self.owner, project_id=self.project.project_id
        )
        self.evidence_id = await self._evidence(
            self.owner, self.session.session_id, "On WMT14 English-German the router reaches 29.7 BLEU."
        )
        await self._turn(
            self.session.session_id,
            "What BLEU does the router reach on WMT14?",
            f"It reaches 29.7 BLEU on WMT14 En-De [{self.evidence_id}].",
            citations=[self.evidence_id],
        )
        # A new conversation in the same project, asking later.
        self.later = await repo.create_session(
            owner_id=self.owner, language="en", project_id=self.project.project_id
        )
        self.ctx = AgentContext(
            owner_id=self.owner,
            session_id=self.later.session_id,
            run_id="run2",
            thread_id=self.later.thread_id,
            project_id=self.project.project_id,
        )

    async def test_recall_returns_the_earlier_turn_with_citable_evidence(self) -> None:
        """A25."""
        from app.agents.tools import recall_conversations

        result = await self._call(recall_conversations, query="BLEU on WMT14")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["scope"], "project")
        turn = result["data"]["turns"][0]
        self.assertEqual(turn["session_id"], self.session.session_id)
        self.assertIn("What BLEU", turn["question"])
        self.assertIn("29.7", turn["answer_excerpt"])
        evidence = turn["evidence"][0]
        self.assertEqual(evidence["evidence_id"], self.evidence_id)
        self.assertEqual(evidence["page"], 6)                     # 0-based 5, shown 1-based
        self.assertEqual(evidence["source_level"], "fulltext")
        self.assertIn("29.7 BLEU", evidence["quote"])
        # Listed as evidence, so the runner lets an answer cite it.
        self.assertEqual(result["evidence_ids"], [self.evidence_id])
        self.assertIn("leads, not evidence", result["note"])

    async def test_the_asking_session_and_other_projects_are_not_searched(self) -> None:
        from app.agents.tools import recall_conversations
        from app.db import agent_repository as repo

        # Asked from the session that holds the turn: nothing, it is already in context.
        self.ctx.session_id = self.session.session_id
        result = await self._call(recall_conversations, query="BLEU")
        self.assertEqual(result["data"]["turns"], [])

        # Moved to another project, the turn is outside "project" scope but inside "all".
        self.ctx.session_id = self.later.session_id
        other_project = await repo.create_project(owner_id=self.owner, title="Q")
        await repo.set_session_project(
            self.session.session_id, owner_id=self.owner, project_id=other_project.project_id
        )
        self.assertEqual((await self._call(recall_conversations, query="BLEU"))["data"]["turns"], [])
        everywhere = await self._call(recall_conversations, query="BLEU", scope="all")
        self.assertEqual(len(everywhere["data"]["turns"]), 1)

    async def test_outside_a_project_recall_searches_everything_and_says_so(self) -> None:
        from app.agents.tools import recall_conversations

        self.ctx.project_id = ""
        result = await self._call(recall_conversations, query="BLEU")
        self.assertEqual(result["data"]["scope"], "all")
        self.assertIn("not in a project", result["note"])
        self.assertEqual(len(result["data"]["turns"]), 1)

    async def test_evidence_of_another_reader_is_never_returned(self) -> None:
        """A26: an id in a message is not a licence to read the evidence behind it."""
        from app.db import agent_repository as repo
        from app.services import conversation_recall, identity

        other, _ = await identity.create_principal()
        other_session = await repo.create_session(owner_id=other)
        foreign = await self._evidence(other, other_session.session_id, "Their private passage.", page=1)
        await self._turn(
            self.session.session_id,
            "Where is the private passage?",
            f"Here [{foreign}].",
            citations=[foreign],
        )
        turns = await conversation_recall.recall(self.owner, "private passage", project_id=None)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].evidence, [])

    async def test_a_turn_matching_on_question_and_answer_is_returned_once(self) -> None:
        from app.services import conversation_recall

        turns = await conversation_recall.recall(self.owner, "WMT14 BLEU router", project_id=None)
        self.assertEqual(len(turns), 1)

    async def test_excerpt_centres_on_the_matching_terms(self) -> None:
        from app.services.conversation_recall import excerpt

        text = "filler " * 300 + "the decisive 29.7 BLEU result " + "tail " * 300
        clipped = excerpt(text, ["29.7", "bleu"], limit=200)
        self.assertIn("29.7 BLEU", clipped)
        self.assertTrue(clipped.startswith("…") and clipped.endswith("…"))
        self.assertLessEqual(len(clipped), 202)
        self.assertEqual(excerpt("short", ["x"]), "short")


# ── Project papers ──────────────────────────────────────────────────────────


class OpenProjectPaperTests(P8TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from app.agents.context import AgentContext
        from app.db import agent_repository as repo

        self.project = await repo.create_project(owner_id=self.owner, title="P")
        await repo.set_session_project(
            self.session.session_id, owner_id=self.owner, project_id=self.project.project_id
        )
        self.fresh = await repo.create_session(owner_id=self.owner, project_id=self.project.project_id)
        self.ctx = AgentContext(
            owner_id=self.owner,
            session_id=self.fresh.session_id,
            run_id="run3",
            thread_id=self.fresh.thread_id,
            project_id=self.project.project_id,
        )

    async def test_a_project_paper_is_readable_only_after_it_is_opened(self) -> None:
        """A28."""
        from langchain_core.messages import ToolMessage

        from app.agents.tools import get_paper_outline, open_project_paper
        from app.services.agent_runner import _tool_completion_payload

        refused = await self._call(get_paper_outline, paper_id="paper1")
        self.assertEqual(refused["status"], "unavailable")

        opened = await self._call(open_project_paper, paper="paper1")
        self.assertEqual(opened["status"], "ok")
        self.assertEqual(opened["data"]["availability"], "parsed")
        # The runner turns this result into `paper.added` for the sidebar.
        payload = _tool_completion_payload(
            ToolMessage(content=__import__("json").dumps(opened), tool_call_id="c", name="open_project_paper")
        )
        self.assertEqual(payload["_attached_paper"]["paper_id"], "paper1")

        outline = await self._call(get_paper_outline, paper_id="paper1")
        self.assertEqual(outline["status"], "ok")

    async def test_papers_outside_the_project_cannot_be_opened(self) -> None:
        from app.agents.tools import open_project_paper
        from app.db import agent_repository as repo

        # paper2 lives in a session outside the project.
        await self._seed_paper("paper2", title="Elsewhere")
        stray = await repo.create_session(owner_id=self.owner)
        await self._attach("paper2", stray.session_id)
        result = await self._call(open_project_paper, paper="paper2")
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["error"]["code"], "paper_not_found")

        self.ctx.project_id = ""
        result = await self._call(open_project_paper, paper="paper1")
        self.assertEqual(result["status"], "unavailable")


# ── Memory scope and prompt ─────────────────────────────────────────────────


class ProjectMemoryAndPromptTests(P8TestCase):
    async def test_project_memories_stay_in_their_project(self) -> None:
        """A29."""
        from app.agents.tools import list_memories, save_memory
        from app.db import agent_repository as repo

        project = await repo.create_project(owner_id=self.owner, title="P")
        other_project = await repo.create_project(owner_id=self.owner, title="Q")
        self.ctx.project_id = project.project_id

        saved = await self._call(save_memory, content="Works on MoE routing", kind="project")
        self.assertEqual(saved["data"]["scope"], "project")
        glob = await self._call(save_memory, content="Prefers tables", kind="preference")
        self.assertEqual(glob["data"]["scope"], "global")

        in_p = await repo.list_memories_for_prompt(self.owner, project_id=project.project_id)
        self.assertEqual([m.content for m in in_p], ["Works on MoE routing", "Prefers tables"])
        in_q = await repo.list_memories_for_prompt(self.owner, project_id=other_project.project_id)
        self.assertEqual([m.content for m in in_q], ["Prefers tables"])
        outside = await repo.list_memories_for_prompt(self.owner)
        self.assertEqual([m.content for m in outside], ["Prefers tables"])

        self.ctx.project_id = other_project.project_id
        listed = await self._call(list_memories)
        self.assertEqual([m["content"] for m in listed["data"]["memories"]], ["Prefers tables"])

    async def test_another_projects_memory_cannot_be_forgotten_or_replaced(self) -> None:
        from app.agents.tools import forget_memory, save_memory
        from app.db import agent_repository as repo

        here = await repo.create_project(owner_id=self.owner, title="P")
        elsewhere = await repo.create_project(owner_id=self.owner, title="Q")
        theirs = await repo.add_memory(
            owner_id=self.owner, content="Q uses JAX", project_id=elsewhere.project_id
        )
        self.ctx.project_id = here.project_id

        refused = await self._call(forget_memory, memory_id=theirs.memory_id)
        self.assertEqual(refused["status"], "unavailable")
        saved = await self._call(save_memory, content="Uses PyTorch", replaces=theirs.memory_id)
        self.assertEqual(saved["data"]["replaced"], "")
        kept = await repo.get_memory(theirs.memory_id, owner_id=self.owner)
        self.assertTrue(kept.active)

        # A global memory is everyone's to drop, and the project's own is too.
        mine = await self._call(save_memory, content="Works on routing", kind="project")
        self.assertTrue((await self._call(forget_memory, memory_id=mine["data"]["memory_id"]))["data"]["forgotten"])

    async def test_project_scope_without_a_project_falls_back_to_global(self) -> None:
        from app.agents.tools import save_memory

        saved = await self._call(save_memory, content="Studies routing", kind="project", scope="project")
        self.assertEqual(saved["data"]["scope"], "global")
        self.assertIn("not in a project", saved["note"])

    async def test_the_same_text_is_kept_once_per_scope(self) -> None:
        from app.db import agent_repository as repo

        project = await repo.create_project(owner_id=self.owner, title="P")
        a = await repo.add_memory(owner_id=self.owner, content="Uses PyTorch")
        b = await repo.add_memory(owner_id=self.owner, content="Uses PyTorch", project_id=project.project_id)
        c = await repo.add_memory(owner_id=self.owner, content="Uses PyTorch", project_id=project.project_id)
        self.assertNotEqual(a.memory_id, b.memory_id)
        self.assertEqual(b.memory_id, c.memory_id)

    async def test_prompt_lists_the_project_and_its_other_papers(self) -> None:
        from app.agents.prompts import PROMPT_VERSION, build_system_prompt
        from app.db import agent_repository as repo

        project = await repo.create_project(
            owner_id=self.owner, title="MoE routing", description="Compare routers on WMT14."
        )
        await repo.set_session_project(self.session.session_id, owner_id=self.owner, project_id=project.project_id)
        fresh = await repo.create_session(owner_id=self.owner, project_id=project.project_id)
        others = await repo.list_project_papers(project.project_id, exclude_session_id=fresh.session_id)
        await repo.add_memory(owner_id=self.owner, content="Works on MoE", project_id=project.project_id)
        memories = await repo.list_memories_for_prompt(self.owner, project_id=project.project_id)

        prompt = build_system_prompt(
            language="en", papers=[], memories=memories, project=project, project_papers=others
        )
        self.assertEqual(PROMPT_VERSION, "p9-memory-1")
        self.assertIn('## Research project', prompt)
        self.assertIn('"MoE routing"', prompt)
        self.assertIn("Compare routers on WMT14.", prompt)
        self.assertIn("paper_id=paper1, parsed", prompt)
        self.assertIn("open_project_paper", prompt)
        self.assertRegex(prompt, r"\[preference · this project · mem_\w+\] Works on MoE")

        zh = build_system_prompt(language="zh", project=project, project_papers=others)
        self.assertIn("## 研究项目", zh)
        self.assertIn("「MoE routing」", zh)
        # Outside a project there is no project section at all.
        self.assertNotIn("## Research project", build_system_prompt(language="en"))


class AccountMergeTests(P8TestCase):
    async def test_projects_move_with_an_anonymous_visitor_into_their_account(self) -> None:
        """A31."""
        from app.db import agent_repository as repo
        from app.db import database as db
        from app.services import accounts, identity

        project = await repo.create_project(owner_id=self.owner, title="P")
        await repo.set_session_project(self.session.session_id, owner_id=self.owner, project_id=project.project_id)
        account, _ = await identity.create_principal()
        await db.execute("UPDATE agent_principals SET kind = 'user' WHERE principal_id = ?", (account,))

        self.assertTrue(await accounts.merge_anonymous(self.owner, account))
        moved = await repo.get_project(project.project_id, owner_id=account)
        self.assertEqual(moved.session_count, 1)
        session = await repo.get_session(self.session.session_id, owner_id=account)
        self.assertEqual(session.project_id, project.project_id)


# ── API ─────────────────────────────────────────────────────────────────────


class ProjectApiTests(unittest.TestCase):
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

    def _as(self, headers: dict[str, str], method: str, url: str, **kwargs):
        # The client keeps the last principal's cookie, and the cookie wins over
        # the header; clear it so each call acts as the principal it names.
        self.client.cookies.clear()
        return self.client.request(method, url, headers=headers, **kwargs)

    def _visitor(self) -> dict[str, str]:
        self.client.cookies.clear()
        response = self.client.post("/api/agent/projects", json={"title": "First"})
        self.assertEqual(response.status_code, 200, response.text)
        return {"X-Agent-Token": response.headers["X-Agent-Token"]}

    def test_project_lifecycle_and_session_membership(self) -> None:
        me = self._visitor()
        project = self._as(me, "POST", "/api/agent/projects", json={"title": "MoE", "description": "routers"}).json()["project"]
        pid = project["project_id"]

        created = self._as(me, "POST", "/api/agent/sessions", json={"language": "en", "project_id": pid})
        self.assertEqual(created.status_code, 200, created.text)
        sid = created.json()["session_id"]
        loose = self._as(me, "POST", "/api/agent/sessions", json={"language": "en"}).json()["session_id"]

        detail = self._as(me, "GET", f"/api/agent/projects/{pid}").json()
        self.assertEqual([s["session_id"] for s in detail["sessions"]], [sid])
        self.assertEqual(detail["project"]["session_count"], 1)

        in_project = self._as(me, "GET", f"/api/agent/sessions?project_id={pid}").json()["sessions"]
        self.assertEqual([s["session_id"] for s in in_project], [sid])
        unfiled = self._as(me, "GET", "/api/agent/sessions?project_id=").json()["sessions"]
        self.assertEqual([s["session_id"] for s in unfiled], [loose])

        moved = self._as(me, "PATCH", f"/api/agent/sessions/{loose}", json={"project_id": pid, "title": "Renamed"})
        self.assertEqual(moved.status_code, 200, moved.text)
        self.assertEqual(moved.json()["session"]["project_id"], pid)
        self.assertEqual(moved.json()["session"]["title"], "Renamed")

        session_detail = self._as(me, "GET", f"/api/agent/sessions/{loose}").json()
        self.assertEqual(session_detail["project"]["project_id"], pid)
        self.assertEqual(session_detail["project_papers"], [])

        archived = self._as(me, "PATCH", f"/api/agent/projects/{pid}", json={"status": "archived"})
        self.assertEqual(archived.json()["project"]["status"], "archived")
        listed = self._as(me, "GET", "/api/agent/projects").json()["projects"]
        self.assertNotIn(pid, [p["project_id"] for p in listed])
        listed_all = self._as(me, "GET", "/api/agent/projects?include_archived=true").json()["projects"]
        self.assertIn(pid, [p["project_id"] for p in listed_all])

    def test_other_principals_cannot_see_or_use_a_project(self) -> None:
        me = self._visitor()
        pid = self._as(me, "GET", "/api/agent/projects").json()["projects"][0]["project_id"]
        stranger = self._visitor()

        self.assertEqual(self._as(stranger, "GET", f"/api/agent/projects/{pid}").status_code, 404)
        self.assertEqual(
            self._as(stranger, "PATCH", f"/api/agent/projects/{pid}", json={"title": "x"}).status_code, 404
        )
        self.assertEqual(
            self._as(stranger, "POST", "/api/agent/sessions", json={"project_id": pid}).status_code, 404
        )
        theirs = self._as(stranger, "POST", "/api/agent/sessions", json={}).json()["session_id"]
        self.assertEqual(
            self._as(stranger, "PATCH", f"/api/agent/sessions/{theirs}", json={"project_id": pid}).status_code,
            404,
        )
        self.assertEqual(
            self._as(stranger, "GET", f"/api/agent/search?q=x&project_id={pid}").status_code, 404
        )
        self.assertEqual(
            self._as(stranger, "POST", "/api/agent/memories", json={"content": "c", "project_id": pid}).status_code,
            404,
        )

    def test_search_returns_turns_and_project_memories_are_listed(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import MessageRole

        me = self._visitor()
        pid = self._as(me, "GET", "/api/agent/projects").json()["projects"][0]["project_id"]
        sid = self._as(me, "POST", "/api/agent/sessions", json={"project_id": pid}).json()["session_id"]

        async def seed() -> None:
            await repo.append_message(session_id=sid, role=MessageRole.USER, content="路由器的 BLEU 是多少", run_id="r1")
            await repo.append_message(session_id=sid, role=MessageRole.ASSISTANT, content="29.7 BLEU", run_id="r1")

        # On the client's own loop: the pool belongs to it.
        self.client.portal.call(seed)

        found = self._as(me, "GET", f"/api/agent/search?q=路由器&project_id={pid}").json()["turns"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["answer_excerpt"], "29.7 BLEU")

        saved = self._as(me, "POST", "/api/agent/memories", json={"content": "Works on MoE", "project_id": pid})
        self.assertEqual(saved.status_code, 200, saved.text)
        self.assertEqual(saved.json()["project_id"], pid)
        detail = self._as(me, "GET", f"/api/agent/projects/{pid}").json()
        self.assertEqual([m["content"] for m in detail["memories"]], ["Works on MoE"])


if __name__ == "__main__":
    unittest.main()
