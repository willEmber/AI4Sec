from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.db import database as db
from app.services.mineru_adapter import (
    MinerUPollTimeoutError,
    _poll_until_done_sync,
    _update_parse_poll_sync,
)
from app.workflows.main_graph import mineru_parse


class _AlwaysRunningClient:
    def __init__(self) -> None:
        self.calls = 0

    def get_batch_results(self, batch_id: str) -> dict:
        self.calls += 1
        return {"extract_result": [{"state": "running"}]}


class MinerUPollTimeoutTests(unittest.TestCase):
    def test_poll_timeout_reports_batch_poll_count_and_last_state_counts(self) -> None:
        client = _AlwaysRunningClient()
        now = iter([0.0, 11.0])
        poll_events: list[dict] = []

        with self.assertRaises(MinerUPollTimeoutError) as caught:
            _poll_until_done_sync(
                client,
                "batch-1",
                sleep_s=0,
                timeout_s=10,
                time_fn=lambda: next(now),
                sleep_fn=lambda seconds: None,
                on_poll=lambda event: poll_events.append(event),
            )

        err = caught.exception
        self.assertEqual(err.batch_id, "batch-1")
        self.assertEqual(err.poll_count, 1)
        self.assertEqual(err.last_state_counts, {"running": 1})
        self.assertIn("batch-1", str(err))
        self.assertIn("running", str(err))
        self.assertEqual(poll_events[0]["poll_count"], 1)
        self.assertEqual(poll_events[0]["state_counts"], {"running": 1})


class MinerUProgressEventTests(unittest.IsolatedAsyncioTestCase):
    async def test_mineru_parse_emits_running_progress_before_remote_parse(self) -> None:
        asyncio.get_running_loop().slow_callback_duration = 10.0
        with tempfile.TemporaryDirectory() as tmp:
            db.set_db_path(Path(tmp) / "app.db")
            await db.init_db()

            queue = asyncio.Queue()
            state = {
                "paper_id": "paper",
                "run_id": "run-1",
                "progress": [],
            }

            async def fake_parse_pdf(paper_id: str, parse_id: str) -> Path:
                event = await queue.get()
                self.assertEqual(event["event"], "progress")
                self.assertEqual(event["data"]["step"], "mineru_parse")
                self.assertEqual(event["data"]["status"], "running")
                raise RuntimeError("stop after progress")

            with patch("app.api.runs._run_queues", {"run-1": queue}), patch(
                "app.workflows.main_graph.mineru_adapter.parse_pdf",
                new=AsyncMock(side_effect=fake_parse_pdf),
            ):
                result = await mineru_parse(state)

            self.assertIn("MinerU parse failed", result["error"])

    async def test_poll_metadata_updates_parse_row_for_diagnostics(self) -> None:
        asyncio.get_running_loop().slow_callback_duration = 10.0
        with tempfile.TemporaryDirectory() as tmp:
            db.set_db_path(Path(tmp) / "app.db")
            await db.init_db()
            await db.execute(
                "INSERT INTO papers (paper_id, file_path) VALUES (?, ?)",
                ("paper", "papers/paper/original.pdf"),
            )
            await db.execute(
                "INSERT INTO mineru_parses (parse_id, paper_id, status) VALUES (?, ?, ?)",
                ("parse-1", "paper", "running"),
            )

            _update_parse_poll_sync(
                "parse-1",
                remote_batch_id="batch-1",
                poll_count=7,
                state_counts={"running": 1},
            )

            row = await db.fetch_one(
                "SELECT remote_batch_id, poll_count, last_state_counts, last_poll_at "
                "FROM mineru_parses WHERE parse_id = ?",
                ("parse-1",),
            )
            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row["remote_batch_id"], "batch-1")
            self.assertEqual(row["poll_count"], 7)
            self.assertEqual(row["last_state_counts"], '{"running": 1}')
            self.assertTrue(row["last_poll_at"])


if __name__ == "__main__":
    unittest.main()
