"""Integration tests for per-browser owner_token isolation on /runs endpoints.

Runs are inserted directly into a temporary database so no LLM/MinerU
background work or network access is triggered; only the read/dismiss
endpoints (the owner-scoping logic) are exercised.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path


class RunOwnerIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["DATA_DIR"] = self._tmp.name

        from app.config import get_settings

        get_settings.cache_clear()

        from fastapi.testclient import TestClient

        from app.main import app

        self._client_cm = TestClient(app)
        self.client = self._client_cm.__enter__()  # runs lifespan -> init_db()

        self.db_file = Path(self._tmp.name) / "app.db"
        self._seed()

    def tearDown(self) -> None:
        self._client_cm.__exit__(None, None, None)
        os.environ.pop("DATA_DIR", None)
        from app.config import get_settings

        get_settings.cache_clear()
        self._tmp.cleanup()

    def _seed(self) -> None:
        con = sqlite3.connect(self.db_file)
        con.execute(
            "INSERT INTO papers (paper_id, file_path, title) VALUES (?, ?, ?)",
            ("paperX", "x.pdf", "Paper X"),
        )

        def add_run(run_id: str, owner: str, started: str) -> None:
            con.execute(
                "INSERT INTO runs (run_id, paper_id, mode, language, status, started_at, owner_token) "
                f"VALUES (?, ?, 'sphere', 'en', 'pending', {started}, ?)",
                (run_id, "paperX", owner),
            )

        add_run("r_a", "A", "datetime('now')")
        add_run("r_b", "B", "datetime('now')")
        add_run("r_legacy", "", "datetime('now')")
        add_run("r_old", "A", "datetime('now', '-100 days')")  # stale + outside 7-day window
        con.commit()
        con.close()

    def _recent_ids(self, owner: str) -> set[str]:
        resp = self.client.get("/api/runs/recent", params={"owner_token": owner})
        self.assertEqual(resp.status_code, 200, resp.text)
        return {r["run_id"] for r in resp.json()}

    def _status(self, run_id: str) -> str:
        con = sqlite3.connect(self.db_file)
        try:
            row = con.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        finally:
            con.close()
        return row[0] if row else ""

    def test_migration_added_owner_index(self) -> None:
        con = sqlite3.connect(self.db_file)
        try:
            rows = con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_runs_owner_started'"
            ).fetchall()
        finally:
            con.close()
        self.assertEqual(len(rows), 1)

    def test_recent_runs_are_owner_scoped(self) -> None:
        self.assertEqual(self._recent_ids("A"), {"r_a"})  # r_old excluded by 7-day window
        self.assertEqual(self._recent_ids("B"), {"r_b"})
        self.assertEqual(self._recent_ids(""), {"r_legacy"})

    def test_stale_run_reconciled_on_read(self) -> None:
        self.assertEqual(self._status("r_old"), "pending")
        self._recent_ids("A")  # triggers _reconcile_stale_runs
        self.assertEqual(self._status("r_old"), "failed")  # 100 days old -> abandoned
        self.assertEqual(self._status("r_a"), "pending")    # fresh run untouched

    def test_dismiss_requires_ownership(self) -> None:
        # Wrong owner cannot dismiss (and cannot learn the run exists).
        resp = self.client.post("/api/runs/r_a/dismiss", params={"owner_token": "B"})
        self.assertEqual(resp.status_code, 404, resp.text)
        self.assertEqual(self._status("r_a"), "pending")

        # Owner can dismiss.
        resp = self.client.post("/api/runs/r_a/dismiss", params={"owner_token": "A"})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["error_msg"], "Dismissed by user")

    def test_legacy_run_dismissible_by_anyone(self) -> None:
        resp = self.client.post("/api/runs/r_legacy/dismiss", params={"owner_token": "whoever"})
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(self._status("r_legacy"), "failed")


if __name__ == "__main__":
    unittest.main()
