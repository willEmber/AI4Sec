"""Integration tests for anonymous traffic persistence and logging."""
from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

import httpx


class TrafficTrackingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db_file = Path(self._tmp.name) / "app.db"
        from app.db import database as db
        from app.main import app

        db.set_db_path(self.db_file)
        await db.init_db()
        transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(transport=transport, base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self._tmp.cleanup()

    async def _visit(self, owner_token: str, path: str = "/") -> httpx.Response:
        return await self.client.post(
            "/api/traffic/visit",
            json={"owner_token": owner_token, "path": path},
        )

    async def test_visits_are_aggregated_by_anonymous_browser(self) -> None:
        self.assertEqual((await self._visit("browser-a")).status_code, 200)
        self.assertEqual((await self._visit("browser-a", "/upload")).status_code, 200)
        self.assertEqual((await self._visit("browser-b", "/library")).status_code, 200)

        con = sqlite3.connect(self.db_file)
        try:
            rows = con.execute(
                "SELECT visitor_hash, visit_count, last_path "
                "FROM traffic_visitors ORDER BY visitor_hash"
            ).fetchall()
        finally:
            con.close()

        self.assertEqual(len(rows), 2)
        by_hash = {row[0]: (row[1], row[2]) for row in rows}
        browser_a_hash = hashlib.sha256(b"browser-a").hexdigest()
        self.assertEqual(by_hash[browser_a_hash], (2, "/upload"))
        self.assertNotIn("browser-a", by_hash)

    async def test_log_reports_uv_and_pv_totals(self) -> None:
        await self._visit("browser-a")
        with self.assertLogs("scholar.traffic", level="INFO") as captured:
            response = await self._visit("browser-b", "/library")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"recorded": True})
        message = captured.output[-1]
        self.assertIn("new_user=True", message)
        self.assertIn("unique_users=2", message)
        self.assertIn("total_visits=2", message)
        self.assertIn("path='/library'", message)

    async def test_invalid_visit_is_rejected(self) -> None:
        response = await self._visit("", "/")
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
