from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from app.services import dify_client
from app.services.dify_client import DifyError


def _settings(base="http://kb.test", dataset_id="", method="full_text_search", timeout=90):
    return SimpleNamespace(
        dify_api_base=base,
        dify_default_dataset_id=dataset_id,
        dify_search_method=method,
        dify_timeout_seconds=timeout,
    )


class DifyClientTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, factory, *, settings, response):
        """Invoke a dify_client coroutine, capturing the outgoing HTTP request."""
        calls: dict = {}

        async def fake_request(self, method, url, params=None, json=None):  # noqa: ANN001
            calls.update(method=method, url=url, params=params, json=json)
            return response

        with patch.object(dify_client, "get_settings", return_value=settings), patch(
            "httpx.AsyncClient.request", new=fake_request
        ):
            result = await factory()
        return calls, result

    async def test_documents_uses_short_path_without_dataset(self) -> None:
        calls, _ = await self._run(
            lambda: dify_client.list_documents(),
            settings=_settings(dataset_id=""),
            response=httpx.Response(200, json={"data": []}),
        )
        self.assertEqual(calls["method"], "GET")
        self.assertTrue(calls["url"].endswith("/api/documents"), calls["url"])

    async def test_documents_uses_full_path_with_default_dataset(self) -> None:
        calls, _ = await self._run(
            lambda: dify_client.list_documents(),
            settings=_settings(dataset_id="DS123"),
            response=httpx.Response(200, json={"data": []}),
        )
        self.assertIn("/api/datasets/DS123/documents", calls["url"])

    async def test_explicit_dataset_overrides_default(self) -> None:
        calls, _ = await self._run(
            lambda: dify_client.get_document("doc1", dataset_id="EXPL"),
            settings=_settings(dataset_id="DS123"),
            response=httpx.Response(200, json={"id": "doc1"}),
        )
        self.assertIn("/api/datasets/EXPL/documents/doc1", calls["url"])

    async def test_search_payload_and_short_path(self) -> None:
        calls, _ = await self._run(
            lambda: dify_client.search("badclip", top_k=5, search_method="semantic_search"),
            settings=_settings(dataset_id=""),
            response=httpx.Response(200, json={"query": "badclip", "records": []}),
        )
        self.assertEqual(calls["method"], "POST")
        self.assertTrue(calls["url"].endswith("/api/search"), calls["url"])
        self.assertEqual(
            calls["json"],
            {"query": "badclip", "top_k": 5, "search_method": "semantic_search"},
        )

    async def test_search_defaults_method_from_settings(self) -> None:
        calls, _ = await self._run(
            lambda: dify_client.search("q"),
            settings=_settings(method="hybrid_search"),
            response=httpx.Response(200, json={"records": []}),
        )
        self.assertEqual(calls["json"]["search_method"], "hybrid_search")

    async def test_search_invalid_method_falls_back(self) -> None:
        calls, _ = await self._run(
            lambda: dify_client.search("q", search_method="nonsense"),
            settings=_settings(),
            response=httpx.Response(200, json={"records": []}),
        )
        self.assertEqual(calls["json"]["search_method"], "full_text_search")

    async def test_search_clamps_top_k(self) -> None:
        calls, _ = await self._run(
            lambda: dify_client.search("q", top_k=10_000),
            settings=_settings(),
            response=httpx.Response(200, json={"records": []}),
        )
        self.assertEqual(calls["json"]["top_k"], 100)

    async def test_search_records_returns_list(self) -> None:
        records = [{"document_id": "d1", "content": "x"}]
        _, result = await self._run(
            lambda: dify_client.search_records("q"),
            settings=_settings(),
            response=httpx.Response(200, json={"records": records}),
        )
        self.assertEqual(result, records)

    async def test_upstream_error_maps_to_dify_error(self) -> None:
        with self.assertRaises(DifyError) as ctx:
            await self._run(
                lambda: dify_client.list_datasets(),
                settings=_settings(),
                response=httpx.Response(401, json={"detail": {"message": "unauthorized"}}),
            )
        self.assertEqual(ctx.exception.upstream_status, 401)
        self.assertEqual(ctx.exception.detail, {"message": "unauthorized"})

    async def test_not_configured_raises_without_request(self) -> None:
        # base URL empty → DifyError before any HTTP call is attempted.
        with patch.object(dify_client, "get_settings", return_value=_settings(base="")):
            with self.assertRaises(DifyError):
                await dify_client.list_datasets()


if __name__ == "__main__":
    unittest.main()
