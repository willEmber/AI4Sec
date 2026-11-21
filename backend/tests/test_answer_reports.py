"""Reports written as the turn's answer (P16 phase 2).

Pure pieces first — citation conversion, the method files, the tool list and
the prompt — then whole turns on a scripted model: a report opened from the
composer, one opened by the model, and the ways a report does not get saved.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from tests.test_agent_aliases import passages_tool
from tests.test_agent_guard import GuardTestCase, StepModel

A = "ev_" + "a" * 20
B = "ev_" + "b" * 20
C = "ev_" + "c" * 20
D = "ev_" + "d" * 20

METHOD = """---
version: 3
required: Core idea | Method
---
Read the method section whole, then write:

# Core idea
# Method
"""


def _table() -> dict[str, dict[str, Any]]:
    return {
        A: {"n": 1, "source_level": "fulltext", "paper_id": "p1", "title": "This paper",
            "page_label": 5, "quote": "Routing   is\nsparse.", "source_url": "", "provider": "mineru"},
        B: {"n": 2, "source_level": "fulltext", "paper_id": "p2", "title": "Other paper",
            "page_label": 3, "quote": "We disagree.", "source_url": "", "provider": "mineru"},
        C: {"n": 3, "source_level": "metadata", "paper_id": "", "title": "This paper",
            "page_label": None, "quote": "Cited 123 times.", "source_url": "https://openalex.org/W1",
            "provider": "OpenAlex"},
        D: {"n": 4, "source_level": "external_web", "paper_id": "", "title": "blog.example",
            "page_label": None, "quote": "A summary.", "source_url": "https://blog.example/x",
            "provider": "tavily"},
    }


class ReportCitationTests(unittest.TestCase):
    def test_groups_become_one_bracket_per_id(self) -> None:
        from app.services.report_citations import cited_ids, normalise

        text = f"x [{A}, {B}] y 【{C}、{A}】 z [{D}]"
        self.assertEqual(normalise(text), f"x [{A}][{B}] y [{C}][{A}] z [{D}]")
        self.assertEqual(cited_ids(text), [A, B, C, D])

    def test_work_form_numbers_citations_by_the_paper_they_are_from(self) -> None:
        from app.services.report_citations import to_works

        table = _table()
        table[B] = {**table[B], "literature_id": "lit_b"}
        table[D] = {**table[D], "literature_id": "lit_d"}
        text = f"This [{A}], rival [{B}][{B}], rank [{C}], blog [{D}]."
        self.assertEqual(
            to_works(text, table, paper_id="p1", numbers={"lit_b": 2}),
            "This [#0], rival [#2], rank , blog .",
        )

    def test_page_form_keeps_this_papers_passages_only(self) -> None:
        from app.services.report_citations import to_pages

        table = _table()
        table[D] = {**table[A], "n": 4, "page_label": 5}
        text = f"Sparse [{A}][{D}], disputed [{B}], cited [{C}, {A}]."
        self.assertEqual(to_pages(text, table, paper_id="p1"), "Sparse [p.5], disputed , cited [p.5].")

    def test_export_numbers_citations_and_lists_what_each_quotes(self) -> None:
        from app.services.report_citations import to_numbered

        text = f"Sparse [{A}][{B}], cited widely [{C}], blogged [{D}]. Again [{A}]. [p.4]"
        out = to_numbered(text, _table(), paper_id="p1", language="en")
        body, refs = out.split("## References")
        self.assertIn("Sparse [1][2], cited widely [3], blogged [4]. Again [1]. [p.4]", body)
        self.assertNotIn("ev_", out)
        lines = refs.strip().splitlines()
        self.assertEqual(lines[0], "1. p.5 — “Routing is sparse.”")
        self.assertEqual(lines[1], "2. Other paper, p.3 — “We disagree.”")
        self.assertEqual(lines[2], "3. OpenAlex · This paper — “Cited 123 times.” — https://openalex.org/W1")
        self.assertEqual(lines[3], "4. blog.example — “A summary.” — https://blog.example/x")

    def test_chinese_export_and_an_id_missing_from_the_table(self) -> None:
        from app.services.report_citations import to_numbered

        out = to_numbered(f"稀疏 [{A}]，未知 [ev_{'e' * 20}]。", {A: _table()[A]}, paper_id="p1", language="zh")
        self.assertIn("稀疏 [1]，未知 。", out)
        self.assertIn("## 引用\n\n1. p.5 —「Routing is sparse.」", out)

    def test_a_report_without_a_table_goes_out_as_stored(self) -> None:
        from app.services.report_citations import to_numbered

        self.assertEqual(to_numbered("Old [p.3] report", {}, paper_id="p1", language="en"), "Old [p.3] report")

    def test_the_export_endpoints_convert_only_answer_reports(self) -> None:
        from app.api.runs import _exported_markdown

        run = {"paper_id": "p1", "language": "en"}
        new = {"markdown": f"x [{A}]", "json_data": json.dumps({"citations": {A: _table()[A]}})}
        old = {"markdown": "x [p.2]", "json_data": json.dumps({"digest": {}})}
        self.assertTrue(_exported_markdown(new, run).startswith("x [1]\n\n## References"))
        self.assertEqual(_exported_markdown(old, run), "x [p.2]")
        self.assertEqual(_exported_markdown({"markdown": "y", "json_data": "not json"}, run), "y")
        self.assertEqual(_exported_markdown(None, run), "")


class ReportBodyTests(unittest.TestCase):
    REPORT = "# Core idea\nSparse [E1].\n\n# Method\nTwo steps [E2]."
    REQUIRED = ("Core idea", "Method")

    def _body(self, *stretches: str, required: tuple[str, ...] = REQUIRED) -> str:
        from app.services.answer_reports import report_body

        shown, marks = "", []
        for i, stretch in enumerate(stretches):
            shown += stretch
            if i < len(stretches) - 1:
                marks.append(len(shown))
        return report_body(shown, marks, required)

    def test_the_report_is_written_after_the_last_read(self) -> None:
        self.assertEqual(self._body("Reading the method. ", "", self.REPORT), self.REPORT)
        self.assertEqual(self._body(self.REPORT), self.REPORT)

    def test_a_tool_called_after_the_report_does_not_replace_it(self) -> None:
        self.assertEqual(self._body("Reading. ", self.REPORT, "Saved to memory."), self.REPORT)

    def test_a_draft_outline_loses_to_the_report_written_after_the_reads(self) -> None:
        draft = "Plan:\n# Core idea\n# Method\n"
        self.assertEqual(self._body(draft, "", self.REPORT), self.REPORT)

    def test_without_required_headings_the_last_stretch_stands(self) -> None:
        self.assertEqual(self._body("Reading. ", "The answer.", required=()), "The answer.")
        # Renamed headings everywhere: nothing to prefer, the last stretch as before.
        self.assertEqual(self._body("# Idea\nX", "Done."), "Done.")


class SkillsDirMixin:
    """Points the method loader at a temporary directory holding a Lens method."""

    def use_skills(self, files: dict[str, str]) -> None:
        """Methods from `files` only: the modes offered are those they cover."""
        from app.agents import skills

        self._skills_dir = tempfile.TemporaryDirectory()
        for name, text in files.items():
            Path(self._skills_dir.name, name).write_text(text, encoding="utf-8")
        self._saved_dir = skills.SKILLS_DIR
        skills.SKILLS_DIR = Path(self._skills_dir.name)
        skills._load.cache_clear()

    def drop_skills(self) -> None:
        from app.agents import skills

        if not hasattr(self, "_saved_dir"):
            return
        skills.SKILLS_DIR = self._saved_dir
        skills._load.cache_clear()
        self._skills_dir.cleanup()


class SkillTests(SkillsDirMixin, unittest.TestCase):
    def tearDown(self) -> None:
        self.drop_skills()

    def test_a_method_is_read_with_its_header(self) -> None:
        from app.agents import skills

        self.use_skills({"logic-lens.en.md": METHOD})
        found = skills.method("lens", "zh")  # no Chinese file: English
        self.assertEqual((found.language, found.version, found.required), ("en", "3", ("Core idea", "Method")))
        self.assertTrue(found.text.startswith("Read the method section whole"))
        self.assertEqual(skills.version_tag(frozenset({"lens"})), "lens3")

    def test_a_mode_is_offered_only_with_a_method(self) -> None:
        from app.agents import skills

        self.assertEqual(skills.enabled_modes(), frozenset({"lens", "snap", "sphere"}))
        self.use_skills({"logic-lens.en.md": METHOD})
        self.assertEqual(skills.enabled_modes(), frozenset({"lens"}))

    def test_the_shipped_methods_parse(self) -> None:
        from app.agents import skills

        for path in skills.SKILLS_DIR.glob("*.md"):
            header, body = skills._parse(path.read_text(encoding="utf-8"))
            self.assertTrue(body, path.name)
            self.assertTrue(header.get("required"), path.name)


class ToolAndPromptTests(unittest.TestCase):
    def test_each_mode_adds_begin_report_and_its_reading_tools(self) -> None:
        from app.agents.tools import ALL_AGENT_TOOLS, agent_tools

        self.assertIs(agent_tools(frozenset()), ALL_AGENT_TOOLS)
        self.assertNotIn("begin_report", [t.name for t in ALL_AGENT_TOOLS])
        names = [t.name for t in agent_tools(frozenset({"lens"}))]
        self.assertIn("begin_report", names)
        # A Lens written as the answer reads elements with their own tool.
        self.assertIn("get_paper_elements", names)
        self.assertNotIn("get_paper_elements", [t.name for t in ALL_AGENT_TOOLS])
        self.assertEqual(len(names), len(ALL_AGENT_TOOLS) + 2)
        snap_only = [t.name for t in agent_tools(frozenset({"snap"}))]
        self.assertNotIn("get_paper_elements", snap_only)
        self.assertLessEqual({"get_triage_signals", "review_paper_blind"}, set(snap_only))
        self.assertNotIn("get_triage_signals", [t.name for t in ALL_AGENT_TOOLS])
        every = [t.name for t in agent_tools(frozenset({"lens", "snap", "sphere"}))]
        self.assertFalse([n for n in every if n.startswith("run_")])  # the old report tools
        self.assertIn("build_literature_map", every)
        self.assertNotIn("build_literature_map", snap_only)

    def test_the_prompt_lists_the_report_modes_offered(self) -> None:
        from app.agents.prompts import build_system_prompt, current_prompt_version

        none = build_system_prompt(language="zh")
        self.assertNotIn("## 三种整篇报告模式", none)
        self.assertIn("相反结论。\n\n## 长期记忆", none)
        self.assertEqual(current_prompt_version(frozenset()), "p15-aliases-1")

        lens = build_system_prompt(language="en", answer_modes=frozenset({"lens"}))
        self.assertIn('`begin_report(mode="lens")`', lens)
        self.assertNotIn('begin_report(mode="snap")', lens)
        self.assertIn("\n\n## Long-term memory", lens)

        every = build_system_prompt(language="zh", answer_modes=frozenset({"lens", "snap", "sphere"}))
        section = every.split("## 长期记忆")[0].split("## 三种整篇报告模式")[1]
        self.assertNotIn("run_", section)
        self.assertEqual(section.count("begin_report(mode="), 3)


class RecordingStepModel(StepModel):
    """`StepModel` that remembers the settings each call was bound with."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self.state.setdefault("bind_kwargs", []).append(kwargs)
        return super().bind_tools(tools, **kwargs)


@tool
async def dismiss_report_tool(runtime: ToolRuntime[AgentContext]) -> str:
    """Closes the turn's report row, as the report page's dismiss would."""
    from app.models.agent_models import ToolResult
    from app.services import mode_runs

    await mode_runs.cancel(runtime.context.report.run_id)
    return ToolResult.ok({"dismissed": True}).to_json()


class AnswerReportTurnTests(SkillsDirMixin, GuardTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from app.agents import guard
        from app.agents.tools import REPORT_TOOLS

        self.use_skills({"logic-lens.en.md": METHOD})
        # The structured view is tested on its own; here it only records
        # what the snapshot hands it.
        from app.services import report_views

        self.view_calls: list[tuple] = []

        async def fake_view(report, text, citations, language):
            self.view_calls.append((report.mode, text, citations, language))
            return {"num_equations": 7}

        views = mock.patch.dict(report_views.BUILDERS, {"lens": fake_view}, clear=True)
        views.start()
        self.addCleanup(views.stop)
        extra = [passages_tool, dismiss_report_tool]
        self._saved_tools = self._runner.agent_tools
        # Every turn's tools and the guard tests' (switched off), with these.
        self._runner.agent_tools = lambda _modes: [
            *self._saved_tools(frozenset()), *extra, *REPORT_TOOLS
        ]
        guard.TOOL_SECONDS.update({"passages_tool": 5.0, "dismiss_report_tool": 5.0})

    async def asyncTearDown(self) -> None:
        self._runner.agent_tools = self._saved_tools
        self.drop_skills()
        await super().asyncTearDown()

    async def _run(self, steps: list[tuple], *, mode: str = "auto"):
        model = RecordingStepModel(steps=steps, state={})
        self._runner.build_chat_model = lambda *_a, **_k: model
        run = await self._new_run()
        await self._runner._execute_turn(session=self.session, run=run, question="Explain it.", mode=mode)
        return run, model

    async def _report_rows(self):
        from app.db import database as db

        return await db.fetch_all(
            """SELECT r.run_id, r.status, r.mode, r.agent_run_id, r.error_msg, r.tokens,
                      o.markdown, o.json_data
                 FROM runs r LEFT JOIN run_outputs o ON o.run_id = r.run_id
                WHERE r.agent_session_id = ? ORDER BY r.started_at""",
            (self.session.session_id,),
        )

    async def test_a_report_picked_in_the_composer_is_the_answer(self) -> None:
        run, model = await self._run(
            [
                ("tools", [("passages_tool", {"n": 2})], "Reading the method first."),
                ("text", "# Core idea\nRouting is sparse [E1].\n\n# Method\nTwo steps [E2]."),
            ],
            mode="lens",
        )
        self.assertEqual((await self._finished(run)).status.value, "done")

        # The method followed the reader's message; the reader's own message is stored as written.
        first = model.state["calls"][0]["messages"]
        self.assertIn("Read the method section whole", first[-1].content)
        self.assertIn("the report has begun", first[-1].content)

        [row] = await self._report_rows()
        self.assertEqual((row["status"], row["mode"], row["agent_run_id"]), ("done", "lens", run.run_id))
        data = json.loads(row["json_data"])
        citations = data["citations"]
        self.assertEqual(sorted(e["n"] for e in citations.values()), [1, 2])
        first_id = next(i for i, e in citations.items() if e["n"] == 1)
        self.assertEqual(citations[first_id]["quote"], "Passage 0 on sparse routing.")
        self.assertEqual(citations[first_id]["page_label"], 1)
        # The report is the closing answer, not the narration before the reads.
        self.assertTrue(row["markdown"].startswith("# Core idea"))
        self.assertIn(f"[{first_id}]", row["markdown"])
        self.assertTrue(data["structure_ok"])
        self.assertEqual((data["written_as"], data["stop_reason"]), ("answer", ""))
        # The mode's structured view saw the frozen text and its table.
        self.assertEqual(data["num_equations"], 7)
        [(view_mode, view_text, view_citations, _lang)] = self.view_calls
        self.assertEqual((view_mode, view_text, view_citations), ("lens", row["markdown"], citations))

        answer = await self._answer(run)
        self.assertTrue(answer.content.startswith("Reading the method first."))
        # Every model call of a report turn carries the report output ceiling.
        self.assertTrue(model.state["bind_kwargs"])
        self.assertTrue(all(k.get("max_tokens") == 40960 for k in model.state["bind_kwargs"]))

        types = [e.type.value for e in await self._events(run)]
        self.assertEqual(types.count("artifact.created"), 1)
        self.assertLess(types.index("artifact.created"), types.index("run.completed"))
        self.assertGreater(types.index("artifact.created"), max(i for i, t in enumerate(types) if t == "message.delta"))
        [card] = await self._events(run, "artifact.created")
        self.assertTrue(card.payload["answer"])
        self.assertEqual(card.payload["run_id"], row["run_id"])
        started = [e.payload for e in await self._events(run, "tool.started")]
        self.assertEqual(started[0]["tool"], "begin_report")

        from app.db import agent_repository as repo

        [artifact] = await repo.list_session_artifacts(self.session.session_id)
        self.assertEqual((artifact.run_id, artifact.answer, artifact.stop_reason), (row["run_id"], True, ""))

    async def test_a_tool_called_with_the_report_leaves_the_report_whole(self) -> None:
        report = "# Core idea\nRouting is sparse [E1].\n\n# Method\nTwo steps [E2]."
        run, _ = await self._run(
            [
                ("tools", [("passages_tool", {"n": 2})], "Reading the method first."),
                # The report, and a tool called with it; then a closing line.
                ("tools", [("passages_tool", {"n": 1})], report),
                ("text", "I also checked the first passage."),
            ],
            mode="lens",
        )
        self.assertEqual((await self._finished(run)).status.value, "done")
        [row] = await self._report_rows()
        self.assertTrue(row["markdown"].startswith("# Core idea"))
        self.assertNotIn("I also checked", row["markdown"])
        self.assertTrue(json.loads(row["json_data"])["structure_ok"])
        # The answer the reader saw keeps every word.
        self.assertTrue((await self._answer(run)).content.endswith("I also checked the first passage."))

    async def test_the_model_may_open_a_report_itself_once(self) -> None:
        run, _ = await self._run(
            [
                ("tools", [("begin_report", {"mode": "lens"})]),
                ("tools", [("begin_report", {"mode": "lens"}), ("passages_tool", {"n": 1})]),
                ("text", "# Core idea\nOnly this [E1]."),
            ]
        )
        [row] = await self._report_rows()
        data = json.loads(row["json_data"])
        self.assertEqual(row["status"], "done")
        self.assertEqual((data["structure_ok"], data["missing_sections"]), (False, ["Method"]))
        codes = await self._tool_codes(run)
        self.assertIn("invalid_argument", codes)

    async def test_a_mode_not_switched_on_is_refused(self) -> None:
        await self._run(
            [("tools", [("begin_report", {"mode": "snap"})]), ("text", "Could not.")]
        )
        self.assertEqual(await self._report_rows(), [])

    async def test_a_failed_turn_closes_its_report_and_keeps_no_output(self) -> None:
        run, _ = await self._run(
            [("tools", [("begin_report", {"mode": "lens"})]), ("raise", RuntimeError("gateway down"))]
        )
        self.assertEqual((await self._finished(run)).status.value, "failed")
        [row] = await self._report_rows()
        self.assertEqual(row["status"], "failed")
        self.assertIsNone(row["markdown"])
        self.assertEqual(await self._events(run, "artifact.created"), [])

    async def test_a_report_dismissed_while_written_stays_an_answer(self) -> None:
        run, _ = await self._run(
            [
                ("tools", [("begin_report", {"mode": "lens"})]),
                ("tools", [("dismiss_report_tool", {})]),
                ("text", "# Core idea\nDone.\n# Method\nDone."),
            ]
        )
        self.assertEqual((await self._finished(run)).status.value, "done")
        [row] = await self._report_rows()
        self.assertEqual(row["status"], "cancelled")
        self.assertIsNone(row["markdown"])
        self.assertEqual(await self._events(run, "artifact.created"), [])
        self.assertIn("# Core idea", (await self._answer(run)).content)

    async def test_several_papers_leave_the_choice_to_the_model(self) -> None:
        await self._seed_paper("paper2", title="Dense Routing")
        await self._attach("paper2", self.session.session_id)
        run, model = await self._run([("text", "Which paper do you mean?")], mode="lens")
        first = model.state["calls"][0]["messages"][-1].content
        self.assertIn("could not begin", first)
        self.assertIn('begin_report(mode="lens"', first)
        self.assertEqual(await self._report_rows(), [])

    async def test_recovery_and_the_heartbeat_reach_the_report_row(self) -> None:
        from app.db import database as db
        from app.services import answer_reports

        run_id = await answer_reports.open_run(
            mode="lens", paper_id="paper1", focus="", llm_model="m", language="en",
            owner_id=self.owner, owner_token="", session_id=self.session.session_id,
            agent_run_id="turn-x",
        )
        await db.execute("UPDATE runs SET heartbeat_at = now() - interval '1 hour' WHERE run_id = ?", (run_id,))
        await answer_reports.beat(run_id)
        row = await db.fetch_one(
            "SELECT heartbeat_at > now() - interval '1 minute' AS fresh FROM runs WHERE run_id = ?", (run_id,)
        )
        self.assertTrue(row["fresh"])
        self.assertEqual(await answer_reports.close_for_turns(["turn-x"], reason="gone"), [run_id])
        self.assertEqual(await answer_reports.close_for_turns(["turn-x"], reason="again"), [])
        row = await db.fetch_one("SELECT status, error_msg FROM runs WHERE run_id = ?", (run_id,))
        self.assertEqual((row["status"], row["error_msg"]), ("failed", "gone"))

    async def test_a_citation_table_is_frozen_as_the_owner_sees_it(self) -> None:
        from app.services import report_citations

        [eid] = (await self._call(passages_tool, n=1))["evidence_ids"]
        text, table = await report_citations.freeze(f"Mine [{eid}], invented [ev_{'f' * 20}].", owner_id=self.owner)
        self.assertEqual(text, f"Mine [{eid}], invented .")
        self.assertEqual(list(table), [eid])
        stranger, _ = await self._stranger()
        text, table = await report_citations.freeze(f"Mine [{eid}].", owner_id=stranger)
        self.assertEqual((text, table), ("Mine .", {}))

    async def _stranger(self):
        from app.services import identity

        return await identity.create_principal()


class LensViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_lens_cards_come_from_the_parse_and_the_report(self) -> None:
        from app.models.lens_models import LensDigest
        from app.models.paper_ir import Block, PaperIR
        from app.services import report_views
        from app.services.answer_reports import ReportTurn

        paper_ir = PaperIR(
            paper_id="p1", title="T", sections=[],
            blocks=[
                Block(type="equation", text="$$a=b\\tag{1}$$", page_idx=1, order_idx=1),
                Block(type="table", text="<table><tr><td>x</td></tr></table>", page_idx=2, order_idx=2),
                Block(type="image", text="Figure 1. Overall framework.", img_path="images/f.jpg",
                      page_idx=1, order_idx=3),
                Block(type="image", text="[image]", img_path="images/g.jpg", page_idx=1, order_idx=4),
            ],
        )
        seen: dict[str, Any] = {}

        async def fake_digest(_llm, *, markdown, model, log_label):
            seen["markdown"] = markdown
            return LensDigest(core_idea="Sparse routing.", available=True)

        report = ReportTurn(run_id="r1", mode="lens", paper_id="p1")
        with mock.patch.object(report_views, "load_paper_ir", return_value=paper_ir), \
             mock.patch.object(report_views, "generate_lens_digest", side_effect=fake_digest), \
             mock.patch.object(report_views, "get_llm_service", return_value=object()):
            data = await report_views.lens_view(
                report, f"# T\n## 1. Overview\nSparse [{A}], other [{B}].", _table(), "en"
            )

        self.assertEqual(
            (data["num_equations"], data["num_algorithms"], data["num_tables"], data["num_figures"]),
            (1, 0, 1, 1),
        )
        self.assertEqual(
            data["key_figures"],
            [{"page": 2, "caption": "Figure 1. Overall framework.", "url": "/api/papers/p1/images/f.jpg"}],
        )
        self.assertTrue(data["digest"]["available"])
        # The digest copies page citations into the cards: it read this
        # paper's passages by page and nothing of the other paper.
        self.assertIn("Sparse [p.5], other .", seen["markdown"])



class SnapViewTests(unittest.IsolatedAsyncioTestCase):
    async def _view(self, parts: dict[str, Any]):
        from app.models.snap_models import SnapClaim, SnapReport
        from app.services import report_views
        from app.services.answer_reports import ReportTurn

        seen: dict[str, Any] = {}

        async def fake_digest(_llm, *, markdown, model, log_label):
            seen["markdown"] = markdown
            return SnapReport(one_liner="Single-step SR.", contributions=[SnapClaim(text="Distils.", page=5)])

        report = ReportTurn(run_id="r1", mode="snap", paper_id="p1", parts=parts)
        with mock.patch.object(report_views, "generate_snap_digest", side_effect=fake_digest), \
             mock.patch.object(report_views, "get_llm_service", return_value=object()):
            data = await report_views.snap_view(report, f"# T\n## One-line verdict\nFast [{A}].", _table(), "en")
        return data, seen

    async def test_the_scorecard_takes_the_tools_results_and_the_texts_fields(self) -> None:
        signals = {"venue": "CVPR", "ccf_rank": "A", "external_score": 0.8, "resolved": True}
        review = {"novelty": 3, "available": True}
        verdict = {"tier": "must_read", "combined_score": 0.7}
        data, seen = await self._view({"signals": signals, "content_review": review, "verdict": verdict})
        self.assertEqual((data["signals"], data["content_review"], data["verdict"]), (signals, review, verdict))
        self.assertEqual(data["report"]["one_liner"], "Single-step SR.")
        self.assertFalse(data["report"]["degraded"])
        self.assertIn("Fast [p.5].", seen["markdown"])

    async def test_without_a_review_the_verdict_rests_on_the_signals(self) -> None:
        data, _ = await self._view({"signals": {"external_score": 0.9, "resolved": True}})
        self.assertFalse(data["verdict"]["content_available"])
        self.assertFalse(data["content_review"]["available"])

    async def test_without_signals_there_is_no_scorecard_verdict(self) -> None:
        data, _ = await self._view({})
        self.assertNotIn("verdict", data)
        self.assertIn("report", data)



def _sphere_map() -> dict[str, Any]:
    from app.models.sphere_models import (
        CorePartition,
        KeyHub,
        SphereNode,
        SphereOutput,
        SphereState,
        TimelineEntry,
    )

    center = SphereNode(node_id="c", title="SinSR", year=2024)
    nodes = {
        "c": center,
        "n1": SphereNode(node_id="n1", title="ResShift", year=2023, layer=1, cluster_id=0),
        "n2": SphereNode(node_id="n2", title="DDPM", year=2020, layer=1, cluster_id=0),
        "h": SphereNode(node_id="h", title="Latent Diffusion", year=2022),
    }
    state = SphereState(
        center_node=center,
        nodes=nodes,
        layer1_node_ids=["n1", "n2"],
        output=SphereOutput(
            partitions=[CorePartition(key="foundation", node_ids=["n2"]),
                        CorePartition(key="method", node_ids=["n1"])],
            timeline=[TimelineEntry(year=2020, node_ids=["n2"]), TimelineEntry(year=2023, node_ids=["n1"])],
            key_hubs=[KeyHub(node_id="h", title="Latent Diffusion", pagerank=0.3)],
        ),
        funnel=[{"stage": "select", "core": 2}],
    )
    return {
        "state": state.model_dump(mode="json"),
        "num_nodes": 180,
        "num_edges": 900,
        "literature_ids": {"n1": "lit_b", "n2": "lit_x", "h": "lit_h"},
    }


class SphereViewTests(unittest.IsolatedAsyncioTestCase):
    async def _view(self, parts: dict[str, Any], digest: dict[str, Any] | None):
        from app.services import report_views
        from app.services.answer_reports import ReportTurn

        seen: dict[str, Any] = {}

        async def fake_digest(_llm, *, markdown, papers, language, model, log_label):
            seen["markdown"] = markdown
            seen["papers"] = [p.title for p in papers]
            seen["language"] = language
            return digest

        table = _table()
        table[B] = {**table[B], "literature_id": "lit_b"}
        report = ReportTurn(run_id="r1", mode="sphere", paper_id="p1", parts=parts)
        with mock.patch.object(report_views, "generate_sphere_digest", side_effect=fake_digest), \
             mock.patch.object(report_views, "get_llm_service", return_value=object()):
            data = await report_views.sphere_view(
                report, f"# T\n## Themes\nResShift [{B}] builds on this [{A}].", table, "en"
            )
        return data, seen

    async def test_the_map_view_is_the_tools_map_and_the_texts_reading(self) -> None:
        digest = {
            "overview": "Diffusion SR moved to one step [#1].",
            "themes": [{"name": "Few-step diffusion", "definition": "Fewer steps.", "papers": [1, 0, 9]}],
            "timeline": [{"year": 2023, "summary": "ResShift shortens the chain."},
                         {"year": 2024, "summary": "One step."}],
            "comparison": [{"paper": 0, "method": "distillation"}, {"paper": 1, "method": "residual shift"},
                           {"paper": 1, "method": "again"}],
            "gaps": [{"title": "Fidelity", "description": "PSNR drops.", "papers": [0, 1]}],
            "reading_paths": [{"name": "Fast", "papers": [2, 1]}, {"name": "odd", "papers": [3]},
                              {"name": "frontier", "papers": [42]}],
        }
        data, seen = await self._view({"map": _sphere_map()}, digest)
        # The map's papers, numbered as the tool numbered them, the center first;
        # citations become those numbers.
        self.assertEqual(seen["papers"], ["SinSR", "DDPM", "ResShift", "Latent Diffusion"])
        self.assertIn("ResShift [#2] builds on this [#0].", seen["markdown"])
        self.assertEqual(seen["language"], "en")
        out = data["sphere_output"]
        self.assertEqual(out["sphere_overview"], "Diffusion SR moved to one step.")
        # The center is not a theme member, an unknown number is dropped.
        self.assertEqual(out["themes"][0]["representative_node_ids"], ["n2"])
        self.assertEqual([r["node_id"] for r in out["comparison_table"]], ["c", "n2"])
        self.assertEqual(out["comparison_table"][1]["title"], "DDPM")
        self.assertEqual([(p["name"], p["node_ids"]) for p in out["reading_paths"]],
                         [("fast", ["n1", "n2"]), ("deep", ["h"])])
        self.assertEqual(
            [(e["year"], e["node_ids"], e["summary"]) for e in out["timeline"]],
            [(2020, ["n2"], ""), (2023, ["n1"], "ResShift shortens the chain."), (2024, [], "One step.")],
        )
        # Partitions, hubs, nodes and counts are the tool's; identity is the snapshot's.
        self.assertEqual([p["key"] for p in data["partitions"]], ["foundation", "method"])
        self.assertEqual(out["key_hubs"][0]["node_id"], "h")
        self.assertEqual(set(data["nodes"]), {"c", "n1", "n2", "h"})
        self.assertEqual((data["num_nodes"], data["num_edges"], data["num_core"]), (180, 900, 2))
        self.assertEqual(data["center_node_id"], "c")
        self.assertEqual(data["debug"]["funnel"], [{"stage": "select", "core": 2}])
        self.assertFalse({"mode", "paper_id", "title", "language"} & set(data))

    async def test_an_unreadable_digest_leaves_the_map(self) -> None:
        data, _ = await self._view({"map": _sphere_map()}, None)
        self.assertEqual(data["sphere_output"]["themes"], [])
        self.assertEqual(len(data["partitions"]), 2)

    async def test_without_a_map_there_is_no_view(self) -> None:
        data, seen = await self._view({}, {"overview": "x"})
        self.assertEqual(data, {})
        self.assertEqual(seen, {})


if __name__ == "__main__":
    unittest.main()
