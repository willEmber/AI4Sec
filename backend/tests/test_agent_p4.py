"""P4: what happens to work whose worker stops existing, and to the record of it.

Nothing here kills a process. Recovery is driven by observable state — a
heartbeat that stopped, a lease that lapsed, a batch id that was recorded — so
every fault this stage claims to handle can be set up directly and asserted on,
rather than inferred from a restart that happened to look right once.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tests.test_agent_p2 import AgentP2TestCase


class RunLeaseTests(AgentP2TestCase):
    """A run says who is executing it, and keeps saying so."""

    async def _new_run(self, client_request_id: str = "") -> Any:
        from app.db import agent_repository as repo

        run, _ = await repo.create_run(
            session_id=self.session.session_id,
            owner_id=self.owner,
            client_request_id=client_request_id,
        )
        return run

    async def test_claiming_a_run_records_the_worker_and_starts_the_heartbeat(self) -> None:
        from app.db import agent_repository as repo

        run = await self._new_run()
        self.assertTrue(await repo.claim_run(run.run_id, worker_id="w1"))

        claimed = await repo.get_run(run.run_id, owner_id=self.owner)
        self.assertEqual(claimed.status.value, "running")
        self.assertEqual(claimed.worker_id, "w1")
        self.assertIsNotNone(claimed.heartbeat_at)

    async def test_a_run_cannot_be_claimed_twice(self) -> None:
        from app.db import agent_repository as repo

        run = await self._new_run()
        self.assertTrue(await repo.claim_run(run.run_id, worker_id="w1"))
        # Two workers racing for the same pending run: exactly one wins, and the
        # loser must not overwrite the winner's identity.
        self.assertFalse(await repo.claim_run(run.run_id, worker_id="w2"))
        self.assertEqual(
            (await repo.get_run(run.run_id, owner_id=self.owner)).worker_id, "w1"
        )

    async def test_a_heartbeat_from_another_worker_does_nothing(self) -> None:
        from app.db import agent_repository as repo
        from app.db import database as db

        run = await self._new_run()
        await repo.claim_run(run.run_id, worker_id="w1")
        await db.execute(
            "UPDATE agent_runs SET heartbeat_at = datetime('now', '-600 seconds') "
            "WHERE run_id = ?",
            (run.run_id,),
        )
        # A process whose run was already recovered must not be able to make it
        # look alive again.
        await repo.heartbeat_run(run.run_id, worker_id="w2")
        stale = await repo.list_stale_runs(stale_after_seconds=60)
        self.assertIn(run.run_id, [r.run_id for r in stale])

        await repo.heartbeat_run(run.run_id, worker_id="w1")
        stale = await repo.list_stale_runs(stale_after_seconds=60)
        self.assertNotIn(run.run_id, [r.run_id for r in stale])

    async def test_a_pending_run_nobody_claimed_eventually_counts_as_stale(self) -> None:
        from app.db import agent_repository as repo
        from app.db import database as db

        run = await self._new_run()
        await db.execute(
            "UPDATE agent_runs SET started_at = datetime('now', '-600 seconds') "
            "WHERE run_id = ?",
            (run.run_id,),
        )
        # A crash between persisting a run and scheduling it leaves exactly this.
        # Nothing else would ever pick it up.
        stale = await repo.list_stale_runs(stale_after_seconds=60)
        self.assertEqual([r.run_id for r in stale], [run.run_id])

    async def test_a_fresh_run_is_never_stale(self) -> None:
        from app.db import agent_repository as repo

        run = await self._new_run()
        await repo.claim_run(run.run_id, worker_id="w1")
        self.assertEqual(await repo.list_stale_runs(stale_after_seconds=60), [])


class RunRecoveryTests(AgentP2TestCase):
    """An abandoned turn reaches a terminal state instead of hanging forever."""

    async def _abandoned_run(self) -> Any:
        from app.db import agent_repository as repo
        from app.db import database as db

        run, _ = await repo.create_run(
            session_id=self.session.session_id, owner_id=self.owner
        )
        await repo.claim_run(run.run_id, worker_id="dead-worker")
        await db.execute(
            "UPDATE agent_runs SET heartbeat_at = datetime('now', '-600 seconds') "
            "WHERE run_id = ?",
            (run.run_id,),
        )
        return run

    async def test_an_orphaned_run_is_closed_as_interrupted(self) -> None:
        from app.db import agent_repository as repo
        from app.services import agent_worker

        run = await self._abandoned_run()
        recovered = await agent_worker.recover_stale_runs(stale_after_seconds=60)

        self.assertEqual(recovered, [run.run_id])
        closed = await repo.get_run(run.run_id, owner_id=self.owner)
        self.assertEqual(closed.status.value, "failed")
        # Not `upstream_error`: nothing failed, and the distinction is what tells
        # the reader that asking again is cheap.
        self.assertEqual(closed.error_code, "interrupted")
        self.assertIsNotNone(closed.finished_at)

    async def test_recovery_writes_the_terminal_event_the_client_is_waiting_for(self) -> None:
        from app.db import agent_repository as repo
        from app.services import agent_worker

        run = await self._abandoned_run()
        await agent_worker.recover_stale_runs(stale_after_seconds=60)

        events = await repo.list_run_events(run.run_id)
        self.assertEqual([e.type.value for e in events], ["run.failed"])
        self.assertEqual(events[0].payload["code"], "interrupted")

    async def test_a_live_subscriber_hears_about_it_immediately(self) -> None:
        from app.services import agent_runner, agent_worker

        run = await self._abandoned_run()
        queue = agent_runner.subscribe(self.session.session_id)
        try:
            await agent_worker.recover_stale_runs(stale_after_seconds=60)
            event = await asyncio.wait_for(queue.get(), timeout=2)
        finally:
            agent_runner.unsubscribe(self.session.session_id, queue)
        self.assertEqual(event.type.value, "run.failed")
        self.assertEqual(event.run_id, run.run_id)

    async def test_a_run_this_process_is_executing_is_left_alone(self) -> None:
        from app.db import agent_repository as repo
        from app.services import agent_runner, agent_worker

        run = await self._abandoned_run()
        # A loaded event loop can be late with a heartbeat without being dead.
        # Ending that turn would kill work the reader is watching.
        task = asyncio.create_task(asyncio.sleep(5))
        agent_runner._tasks[run.run_id] = task
        try:
            recovered = await agent_worker.recover_stale_runs(stale_after_seconds=60)
        finally:
            agent_runner._tasks.pop(run.run_id, None)
            task.cancel()

        self.assertEqual(recovered, [])
        self.assertEqual(
            (await repo.get_run(run.run_id, owner_id=self.owner)).status.value, "running"
        )

    async def test_recovery_frees_the_session_for_a_new_turn(self) -> None:
        from app.db import agent_repository as repo
        from app.services import agent_worker

        await self._abandoned_run()
        # The one-active-run index is a partial unique index, so an abandoned
        # run holds the session's only slot until something closes it.
        with self.assertRaises(repo.ActiveRunConflict):
            await repo.create_run(session_id=self.session.session_id, owner_id=self.owner)

        await agent_worker.recover_stale_runs(stale_after_seconds=60)
        run, deduplicated = await repo.create_run(
            session_id=self.session.session_id, owner_id=self.owner
        )
        self.assertFalse(deduplicated)
        self.assertEqual(run.status.value, "pending")


class JobRecoveryTests(AgentP2TestCase):
    """A job left `running` by a dead worker is found and dealt with."""

    async def _running_job(self, *, kind: str, key: str, lease_offset: str) -> str:
        from app.db import database as db

        job_id = f"job_{kind}_{key}"
        await db.execute(
            """INSERT INTO agent_jobs
                   (job_id, kind, idempotency_key, session_id, status, attempts,
                    lease_owner, lease_expires_at, request_json)
               VALUES (?, ?, ?, ?, 'running', 1, 'dead', datetime('now', ?), ?)""",
            (job_id, kind, key, self.session.session_id, lease_offset,
             json.dumps({"paper_id": "paper1"})),
        )
        # Stored as naive UTC by SQLite; the scan compares ISO strings, so make
        # the lease explicit rather than relying on the two formats matching.
        row = await db.fetch_one(
            "SELECT datetime('now', ?) AS t", (lease_offset,)
        )
        await db.execute(
            "UPDATE agent_jobs SET lease_expires_at = ? WHERE job_id = ?",
            (row["t"].replace(" ", "T"), job_id),
        )
        return job_id

    async def test_a_lapsed_lease_makes_a_job_visible_to_recovery(self) -> None:
        from app.services import agent_jobs

        lapsed = await self._running_job(kind="download", key="k-old", lease_offset="-60 seconds")
        await self._running_job(kind="download", key="k-live", lease_offset="+600 seconds")

        stale = await agent_jobs.list_stale_jobs()
        self.assertEqual([j["job_id"] for j in stale], [lapsed])

    async def test_releasing_a_lease_hands_the_job_back(self) -> None:
        from app.services import agent_jobs

        job_id = await self._running_job(
            kind="download", key="k-release", lease_offset="-60 seconds"
        )
        await agent_jobs.release_lease(job_id)

        row = await agent_jobs.get_job_by_id(job_id)
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["lease_owner"], "")
        self.assertIsNone(row["lease_expires_at"])

    async def test_a_download_left_behind_is_released_not_resumed(self) -> None:
        from app.services import agent_jobs, agent_worker

        job_id = await self._running_job(
            kind="download", key="download:lit-1", lease_offset="-60 seconds"
        )
        outcome = await agent_worker.recover_stale_jobs()

        self.assertEqual(outcome["released"], [job_id])
        self.assertEqual(outcome["resumed"], [])
        self.assertEqual((await agent_jobs.get_job_by_id(job_id))["status"], "pending")

    async def test_a_parse_with_no_batch_yet_is_released(self) -> None:
        from app.services import agent_worker

        job_id = await self._running_job(
            kind="parse", key="parse:paper1:vlm", lease_offset="-60 seconds"
        )
        # It crashed before MinerU was asked for anything. There is nothing to
        # rejoin, so the only correct move is to let the next caller submit.
        outcome = await agent_worker.recover_stale_jobs()
        self.assertEqual(outcome["released"], [job_id])


class ParseResumeTests(AgentP2TestCase):
    """The expensive half of a parse is collected, not paid for twice."""

    async def _job_with_parse(self, *, parse_status: str, batch_id: str) -> tuple[str, str]:
        from app.db import database as db
        from app.services import agent_jobs

        key = "parse:paper1:vlm"
        job_id = "job_parse_1"
        await db.execute(
            """INSERT INTO agent_jobs (job_id, kind, idempotency_key, status, request_json)
               VALUES (?, 'parse', ?, 'running', ?)""",
            (job_id, key, json.dumps({"paper_id": "paper1"})),
        )
        parse_id = "parse-abc"
        await db.execute(
            "INSERT INTO mineru_parses (parse_id, paper_id, status, remote_batch_id) "
            "VALUES (?, 'paper1', ?, ?)",
            (parse_id, parse_status, batch_id),
        )
        await agent_jobs.set_remote_id(job_id, parse_id)
        return job_id, parse_id

    async def _stored_pdf(self) -> None:
        from app.config import get_settings

        pdf = get_settings().data_dir / "papers" / "paper1" / "original.pdf"
        pdf.parent.mkdir(parents=True, exist_ok=True)
        pdf.write_bytes(b"%PDF-1.4\n")

    async def test_a_submitted_batch_is_rejoined_rather_than_resubmitted(self) -> None:
        from app.services import parse_service

        job_id, parse_id = await self._job_with_parse(parse_status="running", batch_id="b-42")
        await self._stored_pdf()

        with (
            mock.patch(
                "app.services.mineru_adapter.resume_parse",
                new=mock.AsyncMock(return_value=Path("/tmp/out")),
            ) as resume,
            mock.patch("app.services.mineru_adapter.parse_pdf", new=mock.AsyncMock()) as submit,
            mock.patch(
                "app.services.parse_service.build_and_store_paper_ir",
                new=mock.AsyncMock(return_value=mock.Mock(sections=[1, 2])),
            ),
        ):
            result = await parse_service.run_parse_job(job_id=job_id, paper_id="paper1")

        resume.assert_awaited_once_with("paper1", parse_id, "b-42")
        submit.assert_not_awaited()
        self.assertTrue(result["resumed"])
        self.assertEqual(result["parse_id"], parse_id)

    async def test_a_parse_that_already_failed_is_submitted_afresh(self) -> None:
        from app.services import parse_service

        job_id, _ = await self._job_with_parse(parse_status="failed", batch_id="b-42")
        await self._stored_pdf()

        with (
            mock.patch("app.services.mineru_adapter.resume_parse", new=mock.AsyncMock()) as resume,
            mock.patch(
                "app.services.mineru_adapter.parse_pdf",
                new=mock.AsyncMock(return_value=Path("/tmp/out")),
            ) as submit,
            mock.patch(
                "app.services.parse_service.build_and_store_paper_ir",
                new=mock.AsyncMock(return_value=mock.Mock(sections=[])),
            ),
        ):
            result = await parse_service.run_parse_job(job_id=job_id, paper_id="paper1")

        # Waiting on a batch that already failed would wait forever for a result
        # that is not coming.
        resume.assert_not_awaited()
        submit.assert_awaited_once()
        self.assertFalse(result["resumed"])

    async def test_the_batch_id_is_recorded_before_the_first_poll(self) -> None:
        from app.db import database as db
        from app.services import agent_jobs, parse_service

        await self._stored_pdf()
        job_id = "job_fresh"
        await db.execute(
            """INSERT INTO agent_jobs (job_id, kind, idempotency_key, status, request_json)
               VALUES (?, 'parse', 'parse:paper1:vlm', 'running', ?)""",
            (job_id, json.dumps({"paper_id": "paper1"})),
        )

        seen: dict[str, str] = {}

        async def _capture(paper_id: str, parse_id: str) -> Path:
            row = await agent_jobs.get_job_by_id(job_id)
            seen["remote_id"] = row["remote_id"]
            return Path("/tmp/out")

        with (
            mock.patch("app.services.mineru_adapter.parse_pdf", new=_capture),
            mock.patch(
                "app.services.parse_service.build_and_store_paper_ir",
                new=mock.AsyncMock(return_value=mock.Mock(sections=[])),
            ),
        ):
            await parse_service.run_parse_job(job_id=job_id, paper_id="paper1")

        # A crash between submission and the first poll must still leave
        # something to rejoin.
        self.assertTrue(seen["remote_id"])

    async def test_the_sweep_resumes_an_abandoned_parse(self) -> None:
        from app.db import database as db
        from app.services import agent_jobs, agent_worker

        job_id, parse_id = await self._job_with_parse(parse_status="running", batch_id="b-42")
        await db.execute(
            "UPDATE agent_jobs SET lease_expires_at = ?, session_id = ? WHERE job_id = ?",
            ("2000-01-01T00:00:00+00:00", self.session.session_id, job_id),
        )
        await self._stored_pdf()

        with (
            mock.patch(
                "app.services.mineru_adapter.resume_parse",
                new=mock.AsyncMock(return_value=Path("/tmp/out")),
            ) as resume,
            mock.patch(
                "app.services.parse_service.build_and_store_paper_ir",
                new=mock.AsyncMock(return_value=mock.Mock(sections=[])),
            ),
        ):
            outcome = await agent_worker.recover_stale_jobs()
            self.assertEqual(outcome["resumed"], [job_id])
            # The sweep hands the work to a background task so one slow parse
            # cannot stall recovery of everything behind it.
            await asyncio.gather(*list(agent_worker._resumes))

        resume.assert_awaited_once_with("paper1", parse_id, "b-42")
        self.assertEqual((await agent_jobs.get_job_by_id(job_id))["status"], "done")

    async def test_a_resumed_parse_leaves_the_paper_readable_in_its_session(self) -> None:
        from app.db import agent_repository as repo
        from app.db import database as db
        from app.models.agent_models import Availability
        from app.services import agent_worker, paper_catalog

        job_id, _ = await self._job_with_parse(parse_status="running", batch_id="b-42")
        literature_id = await paper_catalog.ensure_literature_for_local_paper("paper1")
        await repo.attach_session_paper(
            session_id=self.session.session_id,
            literature_id=literature_id,
            paper_id="paper1",
            availability=Availability.PDF_READY,
        )
        await db.execute(
            "UPDATE agent_jobs SET lease_expires_at = ?, session_id = ? WHERE job_id = ?",
            ("2000-01-01T00:00:00+00:00", self.session.session_id, job_id),
        )
        await self._stored_pdf()

        with (
            mock.patch(
                "app.services.mineru_adapter.resume_parse",
                new=mock.AsyncMock(return_value=Path("/tmp/out")),
            ),
            mock.patch(
                "app.services.parse_service.build_and_store_paper_ir",
                new=mock.AsyncMock(return_value=mock.Mock(sections=[])),
            ),
        ):
            await agent_worker.recover_stale_jobs()
            await asyncio.gather(*list(agent_worker._resumes))

        # The turn that asked is long gone, so the only place this result can
        # surface is the session's paper list when the reader comes back.
        papers = await repo.list_session_papers(self.session.session_id)
        by_id = {p.literature_id: p for p in papers}
        self.assertEqual(by_id[literature_id].availability.value, "parsed")


class CancellationTests(AgentP2TestCase):
    """Stop means stop, including while a tool is waiting on something slow."""

    async def test_stopping_a_turn_interrupts_the_task_not_just_the_flag(self) -> None:
        from app.services import agent_runner

        started = asyncio.Event()

        async def _blocked() -> None:
            started.set()
            await asyncio.sleep(300)

        task = asyncio.create_task(_blocked())
        agent_runner._tasks["run-blocked"] = task
        await started.wait()
        try:
            # The cancel flag is only read between streaming steps, so a turn
            # waiting on a parse would ignore stop for minutes without this.
            self.assertTrue(await agent_runner.request_stop("run-blocked"))
            with self.assertRaises(asyncio.CancelledError):
                await task
        finally:
            agent_runner._tasks.pop("run-blocked", None)

    async def test_stopping_a_run_this_process_does_not_hold_is_not_an_error(self) -> None:
        from app.services import agent_runner

        self.assertFalse(await agent_runner.request_stop("run-elsewhere"))

    async def test_a_cancelled_job_keeps_its_remote_id_and_becomes_claimable(self) -> None:
        from app.services import agent_jobs

        started = asyncio.Event()

        async def _work(job_id: str) -> dict[str, Any]:
            await agent_jobs.set_remote_id(job_id, "batch-in-flight")
            started.set()
            await asyncio.sleep(300)
            return {}

        task = asyncio.create_task(
            agent_jobs.run_once(kind="parse", idempotency_key="parse:cancelled", work=_work)
        )
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        row = await agent_jobs.get_job("parse:cancelled")
        # Not `failed`: nothing went wrong, and MinerU may well finish. Spending
        # an attempt on that would be spending it on a non-event.
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["remote_id"], "batch-in-flight")
        self.assertEqual(row["attempts"], 1)


class ActivityReplayTests(AgentP2TestCase):
    """A finished turn can still say what it read."""

    async def _run_with_events(self) -> str:
        from app.db import agent_repository as repo
        from app.models.agent_models import EventType

        run, _ = await repo.create_run(
            session_id=self.session.session_id, owner_id=self.owner
        )
        for type_, payload in [
            (EventType.RUN_STARTED, {"question": "q"}),
            (EventType.TOOL_STARTED, {"tool": "read_paper_section", "call_id": "c1"}),
            (EventType.MESSAGE_DELTA, {"text": "partial "}),
            (EventType.TOOL_COMPLETED, {"tool": "read_paper_section", "status": "ok"}),
            (EventType.MESSAGE_DELTA, {"text": "answer"}),
            (EventType.RUN_COMPLETED, {"citations": []}),
        ]:
            await repo.append_event(
                session_id=self.session.session_id,
                run_id=run.run_id,
                type=type_,
                payload=payload,
            )
        return run.run_id

    async def test_replay_drops_the_message_fragments(self) -> None:
        from app.db import agent_repository as repo

        run_id = await self._run_with_events()
        events = await repo.list_run_events(run_id, exclude_types=("message.delta",))

        # The answer is already a stored message; replaying its fragments would
        # rebuild text that is on screen.
        self.assertEqual(
            [e.type.value for e in events],
            ["run.started", "tool.started", "tool.completed", "run.completed"],
        )

    async def test_replay_is_scoped_to_one_run(self) -> None:
        from app.db import agent_repository as repo
        from app.models.agent_models import EventType

        run_id = await self._run_with_events()
        await repo.append_event(
            session_id=self.session.session_id,
            run_id="some-other-run",
            type=EventType.TOOL_STARTED,
            payload={"tool": "search_papers"},
        )
        events = await repo.list_run_events(run_id)
        self.assertTrue(all(e.run_id == run_id for e in events))

    async def test_events_come_back_in_order(self) -> None:
        from app.db import agent_repository as repo

        run_id = await self._run_with_events()
        events = await repo.list_run_events(run_id)
        self.assertEqual([e.seq for e in events], sorted(e.seq for e in events))


class ActivityApiTests(unittest.TestCase):
    """The replay endpoint, including who is allowed to read it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["DATA_DIR"] = self._tmp.name

        from app.config import get_settings
        from app.services import identity

        get_settings.cache_clear()
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

    def _session(self) -> tuple[str, str]:
        response = self.client.post("/api/agent/sessions", json={"language": "en"})
        return response.json()["session_id"], response.headers["X-Agent-Token"]

    def _run(self, session_id: str, credential: str) -> str:
        return self.client.post(
            f"/api/agent/sessions/{session_id}/messages",
            json={"content": "hello"},
            headers={"X-Agent-Token": credential},
        ).json()["run_id"]

    def test_activity_is_owner_scoped(self) -> None:
        session_id, credential = self._session()
        run_id = self._run(session_id, credential)
        _, intruder = self._session()

        self.assertEqual(
            self.client.get(
                f"/api/agent/runs/{run_id}/activity", headers={"X-Agent-Token": intruder}
            ).status_code,
            404,
        )
        response = self.client.get(
            f"/api/agent/runs/{run_id}/activity", headers={"X-Agent-Token": credential}
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["run_id"], run_id)

    def test_activity_of_an_unknown_run_is_404(self) -> None:
        _, credential = self._session()
        self.assertEqual(
            self.client.get(
                "/api/agent/runs/ar_nope/activity", headers={"X-Agent-Token": credential}
            ).status_code,
            404,
        )

    def test_cancel_reports_whether_it_interrupted_anything(self) -> None:
        session_id, credential = self._session()
        run_id = self._run(session_id, credential)
        response = self.client.post(
            f"/api/agent/runs/{run_id}/cancel", headers={"X-Agent-Token": credential}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["cancel_requested"])
        self.assertIn("interrupted", response.json())


if __name__ == "__main__":
    unittest.main()
