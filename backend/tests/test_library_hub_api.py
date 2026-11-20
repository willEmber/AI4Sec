"""Unit tests for library hub capability discovery and owner-papers endpoint."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from app.main import app
from app.models.agent_models import Availability, SessionPaper
from app.services.accounts import Caller


class LibraryHubApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        transport = httpx.ASGITransport(app=app)
        self.client = httpx.AsyncClient(transport=transport, base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_models_exposes_library_enabled_flag(self) -> None:
        fake_registry = SimpleNamespace(
            selectable_models=lambda: ["qwen-max"],
            default_model="qwen-max",
            agent_models=lambda: ["qwen-max"],
            agent_default_model="qwen-max",
        )
        with (
            patch("app.api.system.get_registry", return_value=fake_registry),
            patch("app.api.system.get_settings", return_value=SimpleNamespace(dify_enabled=True)),
        ):
            resp = await self.client.get("/api/models")
            self.assertEqual(resp.status_code, 200)
            self.assertIs(resp.json()["library_enabled"], True)

        with (
            patch("app.api.system.get_registry", return_value=fake_registry),
            patch("app.api.system.get_settings", return_value=SimpleNamespace(dify_enabled=False)),
        ):
            resp = await self.client.get("/api/models")
            self.assertEqual(resp.status_code, 200)
            self.assertIs(resp.json()["library_enabled"], False)

    async def test_agent_papers_returns_empty_without_caller_and_lists_for_caller(self) -> None:
        with patch("app.api.deps.resolve_caller", new=AsyncMock(return_value=None)):
            resp = await self.client.get("/api/agent/papers")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json(), {"papers": []})

        sample = SessionPaper(
            session_id="sess_1",
            literature_id="lit_1",
            paper_id="sha1_abc",
            availability=Availability.PARSED,
            added_by="user",
            note="",
            title="Test Paper",
            year=2025,
            year_known=True,
            venue="NeurIPS",
            doi="",
            arxiv_id="",
            updated_at="2026-10-01 00:00:00",
        )
        caller = Caller("pr_user1", "user", "cookie")
        with (
            patch("app.api.deps.resolve_caller", new=AsyncMock(return_value=caller)),
            patch("app.api.agent.repo.list_owner_papers", new=AsyncMock(return_value=[sample])) as mocked,
        ):
            resp = await self.client.get("/api/agent/papers?limit=50")
            self.assertEqual(resp.status_code, 200)
            mocked.assert_awaited_once_with("pr_user1", limit=50)
            data = resp.json()
            self.assertEqual(len(data["papers"]), 1)
            self.assertEqual(data["papers"][0]["paper_id"], "sha1_abc")


if __name__ == "__main__":
    unittest.main()
