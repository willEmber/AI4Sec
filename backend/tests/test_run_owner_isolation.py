"""Integration tests for per-browser owner_token isolation on /runs endpoints.

Runs are inserted directly into a temporary database so no LLM/MinerU
background work or network access is triggered; only the read/dismiss
endpoints (the owner-scoping logic) are exercised.
"""
from __future__ import annotations

import os
import tempfile
import unittest

import psycopg


class RunOwnerIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["DATA_DIR"] = self._tmp.name

        from tests.pg_support import use_database_env

        self.db_url = use_database_env(self)

        from app.config import get_settings

        get_settings.cache_clear()

        from fastapi.testclient import TestClient

        from app.main import app

        self._client_cm = TestClient(app)
        self.client = self._client_cm.__enter__()  # runs lifespan -> init_db()

        self._seed()

    def tearDown(self) -> None:
        self._client_cm.__exit__(None, None, None)
        os.environ.pop("DATA_DIR", None)
        from app.config import get_settings

        get_settings.cache_clear()
        self._tmp.cleanup()

    def _seed(self) -> None:
        with psycopg.connect(self.db_url) as con:
            con.execute(
                "INSERT INTO papers (paper_id, file_path, title) VALUES (%s, %s, %s)",
                ("paperX", "x.pdf", "Paper X"),
            )

            def add_run(run_id: str, owner: str, started: str) -> None:
                con.execute(
                    "INSERT INTO runs (run_id, paper_id, mode, language, status, started_at, owner_token) "
                    f"VALUES (%s, %s, 'sphere', 'en', 'pending', {started}, %s)",
                    (run_id, "paperX", owner),
                )

            add_run("r_a", "A", "now()")
            add_run("r_b", "B", "now()")
            add_run("r_legacy", "", "now()")
            add_run("r_old", "A", "now() - interval '100 days'")  # stale + outside 7-day window

    def _recent_ids(self, owner: str) -> set[str]:
        resp = self.client.get("/api/runs/recent", params={"owner_token": owner})
        self.assertEqual(resp.status_code, 200, resp.text)
        return {r["run_id"] for r in resp.json()}

    def _status(self, run_id: str) -> str:
        with psycopg.connect(self.db_url) as con:
            row = con.execute("SELECT status FROM runs WHERE run_id = %s", (run_id,)).fetchone()
        return row[0] if row else ""

    def test_migration_added_owner_index(self) -> None:
        with psycopg.connect(self.db_url) as con:
            rows = con.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = current_schema() AND indexname = 'idx_runs_owner_started'"
            ).fetchall()
        self.assertEqual(len(rows), 1)

    def test_recent_runs_are_owner_scoped(self) -> None:
        self.assertEqual(self._recent_ids("A"), {"r_a"})  # r_old excluded by 7-day window
        self.assertEqual(self._recent_ids("B"), {"r_b"})
        self.assertEqual(self._recent_ids(""), {"r_legacy"})

    def test_a_run_whose_worker_is_gone_is_run_again_up_to_a_limit(self) -> None:
        from unittest import mock

        from app.services import mode_runs

        with psycopg.connect(self.db_url) as con:
            con.execute(
                "UPDATE runs SET status = 'running', attempts = 3, "
                "heartbeat_at = now() - interval '1 hour' WHERE run_id = 'r_b'"
            )
            con.execute(
                "UPDATE runs SET status = 'running', agent_run_id = 'ar_1', "
                "heartbeat_at = now() - interval '1 hour' WHERE run_id = 'r_legacy'"
            )
        with mock.patch.object(mode_runs, "start") as start:
            result = self.client.portal.call(mode_runs.recover_stale)
        # r_old never reported at all; r_a was created a moment ago and is left alone.
        self.assertEqual(result["mode_runs_resumed"], ["r_old"])
        start.assert_called_once_with("r_old")
        self.assertEqual(self._status("r_old"), "pending")
        self.assertEqual(self._status("r_a"), "pending")
        # Out of attempts, and a run whose conversation turn died with it: closed.
        self.assertEqual(sorted(result["mode_runs_closed"]), ["r_b", "r_legacy"])
        self.assertEqual(self._status("r_b"), "failed")
        self.assertEqual(self._status("r_legacy"), "failed")

        with mock.patch.object(mode_runs, "start") as start:
            again = self.client.portal.call(mode_runs.recover_stale)
        self.assertEqual(again, {"mode_runs_resumed": [], "mode_runs_closed": []})
        start.assert_not_called()

    def test_dismiss_requires_ownership(self) -> None:
        # Wrong owner cannot dismiss (and cannot learn the run exists).
        resp = self.client.post("/api/runs/r_a/dismiss", params={"owner_token": "B"})
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(self._status("r_a"), "pending")

        # Owner can dismiss.
        resp = self.client.post("/api/runs/r_a/dismiss", params={"owner_token": "A"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "cancelled")
        self.assertEqual(body["error_msg"], "Cancelled by user")

    def test_delete_requires_ownership_and_removes_the_report(self) -> None:
        with psycopg.connect(self.db_url) as con:
            con.execute("INSERT INTO run_outputs (run_id, markdown) VALUES ('r_a', '# report')")
        resp = self.client.delete("/api/runs/r_a", params={"owner_token": "B"})
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(self.client.get("/api/runs/r_a/output").status_code, 200)

        resp = self.client.delete("/api/runs/r_a", params={"owner_token": "A"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self.client.get("/api/runs/r_a").status_code, 404)
        self.assertEqual(self.client.get("/api/runs/r_a/output").status_code, 404)
        self.assertEqual(self._status("r_b"), "pending")

    def test_legacy_run_dismissible_by_anyone(self) -> None:
        resp = self.client.post("/api/runs/r_legacy/dismiss", params={"owner_token": "whoever"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self._status("r_legacy"), "cancelled")


if __name__ == "__main__":
    unittest.main()


class RunProgressAndDismissTests(RunOwnerIsolationTests):
    """What a run records while it executes, and what closing it stops."""

    def test_progress_is_persisted(self) -> None:
        from app.workflows.progress import emit_progress

        self.client.portal.call(emit_progress, "r_a", "mineru_parse", "running")
        self.client.portal.call(lambda: emit_progress("r_a", "build_paper_ir", "done", blocks=3))

        import json

        run = self.client.get("/api/runs/r_a").json()
        self.assertEqual(run["current_step"], "build_paper_ir")
        self.assertEqual(
            json.loads(run["progress_json"]),
            [
                {"step": "mineru_parse", "status": "running"},
                {"step": "build_paper_ir", "status": "done", "blocks": 3},
            ],
        )

    def test_dismiss_stops_the_task_and_the_run_stays_closed(self) -> None:
        import asyncio
        from unittest import mock

        from app.workflows.main_graph import persist_output

        seen: dict[str, bool] = {}

        async def _never_finishes(run_id: str, *args: object, **kwargs: object) -> None:
            seen["started"] = True
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                seen["cancelled"] = True
                raise

        with mock.patch("app.services.mode_runs._execute", new=_never_finishes):
            created = self.client.post("/api/runs", json={"paper_id": "paperX", "mode": "snap"})
            self.assertEqual(created.status_code, 200, created.text)
            run_id = created.json()["run_id"]
            # The cookie is Secure and this client speaks http, so the credential
            # goes back the way pre-cookie clients sent it.
            credential = created.headers["X-Agent-Token"]
            dismissed = self.client.post(
                f"/api/runs/{run_id}/dismiss", headers={"X-Agent-Token": credential}
            )
            self.assertEqual(dismissed.status_code, 200, dismissed.text)
            self.assertEqual(dismissed.json()["status"], "cancelled")
            self.client.portal.call(asyncio.sleep, 0.05)
        self.assertEqual(seen, {"started": True, "cancelled": True})

        # A graph that reaches its last node anyway must not reopen the run.
        self.client.portal.call(
            persist_output, {"run_id": run_id, "paper_id": "paperX", "final_markdown": "# late"}
        )
        after = self.client.get(f"/api/runs/{run_id}").json()
        self.assertEqual((after["status"], after["error_msg"]), ("cancelled", "Cancelled by user"))

    def test_the_stream_replays_recorded_steps_and_ends_with_the_run(self) -> None:
        import json

        from app.workflows.progress import emit_progress

        self.client.portal.call(emit_progress, "r_a", "mineru_parse", "done")
        self.client.post("/api/runs/r_a/cancel", params={"owner_token": "A"})

        body = self.client.get("/api/runs/r_a/stream").text
        events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
        self.assertEqual(
            events,
            [
                {"event": "progress", "data": {"step": "mineru_parse", "status": "done"}},
                {"event": "cancelled", "data": {"run_id": "r_a", "status": "cancelled"}},
                {"event": "end"},
            ],
        )
        self.assertEqual(self.client.get("/api/runs/nope/stream").status_code, 404)

    def test_the_executor_takes_a_pending_run_to_its_end(self) -> None:
        import json
        from unittest import mock

        from app.services import mode_runs
        from app.workflows.main_graph import persist_output

        class _Graph:
            async def astream(self, state):
                yield {"ingest_pdf": {"progress": [{"step": "ingest_pdf", "status": "done"}]}}
                await persist_output({**state, "final_markdown": "# report"})
                yield {"persist_output": {"progress": [{"step": "persist_output", "status": "done"}]}}

        class _Broken:
            async def astream(self, state):
                raise RuntimeError("model unavailable")
                yield {}

        with mock.patch.object(mode_runs, "_get_graph", return_value=_Graph()):
            self.client.portal.call(mode_runs._execute, "r_a")
        run = self.client.get("/api/runs/r_a").json()
        self.assertEqual(run["status"], "done")
        self.assertEqual(
            [s["step"] for s in json.loads(run["progress_json"])], ["ingest_pdf", "persist_output"]
        )
        self.assertEqual(self.client.get("/api/runs/r_a/output").json()["markdown"], "# report")
        with psycopg.connect(self.db_url) as con:
            attempts, worker = con.execute(
                "SELECT attempts, worker_id FROM runs WHERE run_id = 'r_a'"
            ).fetchone()
        self.assertEqual(attempts, 1)
        self.assertEqual(worker, mode_runs.WORKER_ID)

        with mock.patch.object(mode_runs, "_get_graph", return_value=_Broken()):
            self.client.portal.call(mode_runs._execute, "r_b")
        failed = self.client.get("/api/runs/r_b").json()
        self.assertEqual((failed["status"], failed["error_msg"]), ("failed", "model unavailable"))

        # A run that is no longer pending is not taken at all.
        with mock.patch.object(mode_runs, "_get_graph", return_value=_Graph()):
            self.client.portal.call(mode_runs._execute, "r_b")
        self.assertEqual(self._status("r_b"), "failed")
