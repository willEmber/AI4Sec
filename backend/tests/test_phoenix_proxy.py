from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from app.api import phoenix
from app.api.admin import require_admin_token


class _Body(httpx.AsyncByteStream):
    """A body still to be read, as a real upstream's is when the proxy streams it."""

    async def __aiter__(self):
        yield b"<html>phoenix</html>"


def _settings(**overrides):
    values = {
        "tracing_ui_url": "http://phoenix:6006",
        "admin_api_token": "",
        "auth_mode": "multi_user",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PhoenixProxyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.seen: list[httpx.Request] = []

        def upstream(request: httpx.Request) -> httpx.Response:
            self.seen.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "text/html", "set-cookie": "phoenix=1"},
                stream=_Body(),
            )

        transport = httpx.MockTransport(upstream)
        self.patches = [
            patch.object(phoenix, "_client", lambda: httpx.AsyncClient(transport=transport)),
            patch.object(phoenix, "get_settings", _settings),
        ]
        for p in self.patches:
            p.start()
        self.app = FastAPI()
        self.app.include_router(phoenix.router, prefix="/api")

    def tearDown(self) -> None:
        for p in self.patches:
            p.stop()

    def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://scholar")

    async def test_strips_prefix_and_keeps_credentials_here(self) -> None:
        self.app.dependency_overrides[require_admin_token] = lambda: None
        async with self._http() as http:
            res = await http.get(
                "/api/admin/phoenix/projects/abc?x=1&token=st_secret",
                headers={"cookie": "scholar_auth=st_secret", "x-admin-token": "t", "accept": "text/html"},
            )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.text, "<html>phoenix</html>")
        self.assertNotIn("set-cookie", res.headers)
        sent = self.seen[0]
        self.assertEqual(str(sent.url), "http://phoenix:6006/projects/abc?x=1")
        self.assertNotIn("cookie", sent.headers)
        self.assertNotIn("x-admin-token", sent.headers)
        self.assertEqual(sent.headers["accept"], "text/html")

    async def test_bare_prefix_and_post_body(self) -> None:
        self.app.dependency_overrides[require_admin_token] = lambda: None
        async with self._http() as http:
            await http.get("/api/admin/phoenix")
            await http.post("/api/admin/phoenix/graphql", json={"query": "{__typename}"})
        self.assertEqual(str(self.seen[0].url), "http://phoenix:6006/")
        self.assertEqual(self.seen[1].method, "POST")
        self.assertEqual(self.seen[1].content, b'{"query":"{__typename}"}')

    async def test_nobody_without_admin_login(self) -> None:
        with patch("app.api.admin.get_settings", _settings):
            async with self._http() as http:
                res = await http.get("/api/admin/phoenix/")
        self.assertEqual(res.status_code, 401)
        self.assertEqual(self.seen, [])

    async def test_unconfigured_is_404(self) -> None:
        self.app.dependency_overrides[require_admin_token] = lambda: None
        with patch.object(phoenix, "get_settings", lambda: _settings(tracing_ui_url="")):
            async with self._http() as http:
                res = await http.get("/api/admin/phoenix/")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
