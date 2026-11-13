"""The eval runner against a real database, with the turn itself faked.

What is checked is the runner's own part: setup turns run first in the same
conversation, the mode picked for each turn reaches it, only the scored turn's
tools and reports are recorded, and a report case starts from no cached report.
"""

from __future__ import annotations

import itertools
import json
from typing import Any
from unittest import mock

from tests.test_agent_p2 import AgentP2TestCase


class EvalRunnerTests(AgentP2TestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        from evals.dataset import Dataset

        self.dataset = Dataset.model_validate({
            "name": "d", "version": "1",
            "papers": {"p": {"paper_id": "a" * 40, "title": "Paper"}},
            "cases": [],
        })
        await self._seed_paper("a" * 40)
        self.turns: list[tuple[str, str, str]] = []
        self.report_ids = itertools.count(1)

    async def _fake_turn(self, *, session: Any, run: Any, question: str, mode: str = "auto") -> None:
        from app.db import agent_repository as repo
        from app.db import database as db
        from app.models.agent_models import EventType, MessageRole, RunStatus
        from app.services import agent_runner

        self.turns.append((session.session_id, question, mode))
        if mode == "lens":
            report_id = f"rep{next(self.report_ids)}"
            await db.execute(
                "INSERT INTO runs (run_id, paper_id, mode, status, owner_id, agent_run_id, tokens) "
                "VALUES (?, ?, 'lens', 'done', ?, ?, 900)",
                (report_id, "a" * 40, self.owner, run.run_id),
            )
            await db.execute(
                "INSERT INTO run_outputs (run_id, markdown, json_data) VALUES (?, ?, '{}')",
                (report_id, f"# Lens {report_id}"),
            )
            await agent_runner.emit(session_id=session.session_id, run_id=run.run_id,
                                    type=EventType.TOOL_STARTED, payload={"tool": "run_logic_lens"})
            await agent_runner.emit(session_id=session.session_id, run_id=run.run_id,
                                    type=EventType.ARTIFACT_CREATED,
                                    payload={"run_id": report_id, "mode": "lens", "reused": False})
        else:
            await agent_runner.emit(session_id=session.session_id, run_id=run.run_id,
                                    type=EventType.TOOL_STARTED, payload={"tool": "read_paper_section"})
        await repo.append_message(session_id=session.session_id, role=MessageRole.ASSISTANT,
                                  content=f"answer: {question}", run_id=run.run_id)
        await repo.finish_run(run.run_id, status=RunStatus.DONE, usage={"tokens": 1000})

    async def _run(self, case: Any, attempt: int = 0) -> dict[str, Any]:
        from evals import runner

        with mock.patch.object(runner.agent_runner, "_execute_turn", self._fake_turn):
            return await runner.run_case(
                self.dataset, case, owner_id=self.owner, model="m", attempt=attempt
            )

    async def test_setup_turns_run_first_and_only_the_scored_turn_is_measured(self) -> None:
        from evals.dataset import EvalCase

        case = EvalCase.model_validate({
            "id": "f", "category": "follow_up", "papers": ["p"],
            "setup": [{"question": "explain the method", "mode": "lens"}],
            "question": "what is the PSNR?", "key_points": ["24.56"],
            "forbid_tools": ["run_logic_lens"],
        })
        record = await self._run(case, attempt=1)

        self.assertEqual([(q, m) for _, q, m in self.turns],
                         [("explain the method", "lens"), ("what is the PSNR?", "auto")])
        self.assertEqual(len({s for s, _, _ in self.turns}), 1, "one conversation")
        self.assertEqual(record["attempt"], 1)
        self.assertEqual(record["answer"], "answer: what is the PSNR?")
        self.assertTrue(record["completed"])
        self.assertEqual([c["tool"] for c in record["tool_calls"]], ["read_paper_section"])
        self.assertEqual(record["reports"], [])
        (setup,) = record["setup"]
        self.assertEqual(setup["tools"], ["run_logic_lens"])
        self.assertEqual(setup["reports"][0]["tokens"], 900)
        self.assertNotIn("markdown", setup["reports"][0], "setup reports are not judged or kept")

    async def test_a_report_case_records_the_report_and_starts_without_a_cached_one(self) -> None:
        from app.db import database as db
        from evals.dataset import EvalCase

        for key, status in (("mode:" + "a" * 40 + ":lens:zh:m", "done"),
                            ("mode:" + "a" * 40 + ":snap:zh:m", "running"),
                            ("mode:" + "b" * 40 + ":lens:zh:m", "done")):
            await db.execute(
                "INSERT INTO agent_jobs (job_id, kind, idempotency_key, status, request_json, result_json) "
                "VALUES (?, 'mode', ?, ?, '{}', ?)",
                (key[-12:] + status, key, status, json.dumps({"run_id": "old"})),
            )
        case = EvalCase.model_validate({
            "id": "r", "category": "report", "papers": ["p"], "mode": "lens", "cites": False,
            "question": "explain", "report_key_points": ["x"], "expect_tools": ["run_logic_lens"],
        })
        record = await self._run(case)

        self.assertEqual(record["reports_forgotten"], 1)
        left = {r["idempotency_key"] for r in await db.fetch_all("SELECT idempotency_key FROM agent_jobs")}
        self.assertEqual(left, {"mode:" + "a" * 40 + ":snap:zh:m", "mode:" + "b" * 40 + ":lens:zh:m"},
                         "a running job and another paper's report are left alone")
        (report,) = record["reports"]
        self.assertEqual((report["mode"], report["tokens"], report["markdown"]), ("lens", 900, "# Lens rep1"))
        self.assertEqual(record["mode"], "lens")

    async def test_a_reading_case_keeps_cached_reports(self) -> None:
        from app.db import database as db
        from evals.dataset import EvalCase

        await db.execute(
            "INSERT INTO agent_jobs (job_id, kind, idempotency_key, status, request_json, result_json) "
            "VALUES ('j1', 'mode', ?, 'done', '{}', '{}')",
            ("mode:" + "a" * 40 + ":lens:zh:m",),
        )
        case = EvalCase.model_validate({
            "id": "n", "category": "numeric", "papers": ["p"], "question": "psnr?", "key_points": ["1"],
        })
        record = await self._run(case)
        self.assertEqual(record["reports_forgotten"], 0)
        self.assertEqual(len(await db.fetch_all("SELECT 1 FROM agent_jobs")), 1)
