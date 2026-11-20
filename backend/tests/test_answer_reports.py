"""Reports written as the turn's answer (P16 phase 2).

Pure pieces first — citation conversion, the method files, the tool list and
the prompt — then whole turns on a scripted model: a report opened from the
composer, one opened by the model, and the ways a report does not get saved.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

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


class SkillsDirMixin:
    """Points the method loader at a temporary directory holding a Lens method."""

    def use_skills(self, files: dict[str, str], modes: str) -> None:
        from app.agents import skills
        from app.config import get_settings

        self._skills_dir = tempfile.TemporaryDirectory()
        for name, text in files.items():
            Path(self._skills_dir.name, name).write_text(text, encoding="utf-8")
        self._saved_dir = skills.SKILLS_DIR
        skills.SKILLS_DIR = Path(self._skills_dir.name)
        skills._load.cache_clear()
        os.environ["AGENT_ANSWER_REPORT_MODES"] = modes
        get_settings.cache_clear()

    def drop_skills(self) -> None:
        from app.agents import skills
        from app.config import get_settings

        if not hasattr(self, "_saved_dir"):
            return
        skills.SKILLS_DIR = self._saved_dir
        skills._load.cache_clear()
        self._skills_dir.cleanup()
        os.environ.pop("AGENT_ANSWER_REPORT_MODES", None)
        get_settings.cache_clear()


class SkillTests(SkillsDirMixin, unittest.TestCase):
    def tearDown(self) -> None:
        self.drop_skills()

    def test_a_method_is_read_with_its_header(self) -> None:
        from app.agents import skills

        self.use_skills({"logic-lens.en.md": METHOD}, "lens")
        found = skills.method("lens", "zh")  # no Chinese file: English
        self.assertEqual((found.language, found.version, found.required), ("en", "3", ("Core idea", "Method")))
        self.assertTrue(found.text.startswith("Read the method section whole"))
        self.assertEqual(skills.version_tag(frozenset({"lens"})), "lens3")

    def test_a_mode_needs_both_the_switch_and_a_method(self) -> None:
        from app.agents import skills

        self.use_skills({"logic-lens.en.md": METHOD}, "lens, snap,bogus")
        self.assertEqual(skills.enabled_modes(), frozenset({"lens"}))
        os.environ["AGENT_ANSWER_REPORT_MODES"] = ""
        from app.config import get_settings

        get_settings.cache_clear()
        self.assertEqual(skills.enabled_modes(), frozenset())

    def test_the_shipped_methods_parse(self) -> None:
        from app.agents import skills

        for path in skills.SKILLS_DIR.glob("*.md"):
            header, body = skills._parse(path.read_text(encoding="utf-8"))
            self.assertTrue(body, path.name)
            self.assertTrue(header.get("required"), path.name)


class ToolAndPromptTests(unittest.TestCase):
    def test_a_mode_written_as_an_answer_trades_its_tool_for_begin_report(self) -> None:
        from app.agents.tools import ALL_AGENT_TOOLS, agent_tools

        self.assertIs(agent_tools(frozenset()), ALL_AGENT_TOOLS)
        names = [t.name for t in agent_tools(frozenset({"lens"}))]
        self.assertIn("begin_report", names)
        self.assertNotIn("run_logic_lens", names)
        self.assertIn("run_insight_snap", names)
        self.assertEqual(len(names), len(ALL_AGENT_TOOLS))
        every = [t.name for t in agent_tools(frozenset({"lens", "snap", "sphere"}))]
        self.assertFalse({"run_insight_snap", "run_logic_lens", "run_research_sphere"} & set(every))

    def test_the_prompt_is_unchanged_until_a_mode_is_switched_on(self) -> None:
        from app.agents.prompts import _EN, _ZH, build_system_prompt, current_prompt_version

        self.assertTrue(build_system_prompt(language="zh").startswith(_ZH))
        self.assertTrue(build_system_prompt(language="en").startswith(_EN))
        self.assertEqual(current_prompt_version(frozenset()), "p15-aliases-1")

        lens = build_system_prompt(language="en", answer_modes=frozenset({"lens"}))
        self.assertIn('`begin_report(mode="lens")`', lens)
        self.assertNotIn("run_logic_lens", lens)
        self.assertIn("`run_insight_snap`, `run_research_sphere`: never run the same mode twice", lens)
        self.assertIn("\n\n## Long-term memory", lens)

        every = build_system_prompt(language="zh", answer_modes=frozenset({"lens", "snap", "sphere"}))
        self.assertNotIn("run_", every.split("## 长期记忆")[0].split("## 三种整篇报告模式")[1])


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

        self.use_skills({"logic-lens.en.md": METHOD}, "lens")
        extra = [passages_tool, dismiss_report_tool]
        self._saved_tools = self._runner.agent_tools
        self._runner.agent_tools = lambda _modes: [
            *self._runner.ALL_AGENT_TOOLS, *extra, *REPORT_TOOLS
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



if __name__ == "__main__":
    unittest.main()
