"""Integration tests for per-browser owner_token isolation on /runs endpoints.

Runs are inserted directly into a temporary database so no LLM/MinerU
background work or network access is triggered; only the read/delete
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

    def test_a_run_whose_worker_is_gone_is_closed(self) -> None:
        from app.services import mode_runs

        with psycopg.connect(self.db_url) as con:
            con.execute(
                "UPDATE runs SET status = 'running', agent_run_id = 'ar_1', "
                "heartbeat_at = now() - interval '1 hour' WHERE run_id = 'r_legacy'"
            )
            # A report whose tool is still beating is left alone.
            con.execute(
                "UPDATE runs SET status = 'running', agent_run_id = 'ar_2', "
                "heartbeat_at = now() WHERE run_id = 'r_b'"
            )
        result = self.client.portal.call(mode_runs.recover_stale)
        # r_old never reported at all — a run left waiting by the upload page —
        # and r_legacy's turn died with its worker. r_a was created a moment ago.
        self.assertEqual(result, {"mode_runs_closed": ["r_legacy", "r_old"]})
        self.assertEqual(self._status("r_old"), "failed")
        self.assertEqual(self._status("r_legacy"), "failed")
        self.assertEqual(self._status("r_a"), "pending")
        self.assertEqual(self._status("r_b"), "running")

        again = self.client.portal.call(mode_runs.recover_stale)
        self.assertEqual(again, {"mode_runs_closed": []})

    def test_runs_are_no_longer_started_over_http(self) -> None:
        resp = self.client.post("/api/runs", json={"paper_id": "paperX", "mode": "snap"})
        self.assertIn(resp.status_code, (404, 405), resp.text)

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

    def test_a_legacy_run_can_be_deleted_by_anyone(self) -> None:
        resp = self.client.delete("/api/runs/r_legacy", params={"owner_token": "whoever"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self.client.get("/api/runs/r_legacy").status_code, 404)


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

    def test_deleting_a_report_being_written_stops_it(self) -> None:
        import asyncio
        import time
        from unittest import mock

        from app.services import mode_runs

        with psycopg.connect(self.db_url) as con:
            con.execute("UPDATE runs SET status = 'running' WHERE run_id = 'r_a'")
        seen: dict[str, bool] = {}

        async def make_report() -> None:
            # What a turn does while it writes a report: beat, and be
            # interrupted once the row is closed — on whichever replica.
            try:
                async with mode_runs.heartbeating("r_a"):
                    seen["started"] = True
                    await asyncio.Event().wait()
            except asyncio.CancelledError:
                seen["cancelled"] = True

        with mock.patch.object(mode_runs, "HEARTBEAT_SECONDS", 0.05):
            made = self.client.portal.start_task_soon(make_report)
            deadline = time.monotonic() + 2
            while not seen.get("started") and time.monotonic() < deadline:
                time.sleep(0.01)
            deleted = self.client.delete("/api/runs/r_a", params={"owner_token": "A"})
            self.assertEqual(deleted.status_code, 200, deleted.text)
            made.result(timeout=2)
        self.assertEqual(seen, {"started": True, "cancelled": True})
        self.assertEqual(self.client.get("/api/runs/r_a").status_code, 404)
