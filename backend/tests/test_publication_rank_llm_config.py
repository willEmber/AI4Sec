from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.publication_rank.llm_rank import LLMRankClient, UnifiedRankClient
from app.services.publication_rank.publication_rank import PublicationRankResult


class _CacheWithStaleFailure:
    def __init__(self) -> None:
        self.put_results: list[PublicationRankResult] = []

    async def init(self) -> None:
        pass

    async def get(self, name: str) -> PublicationRankResult:
        return PublicationRankResult(
            name=name,
            success=False,
            error=(
                "LLM 查询失败 (3 次重试后): UnsupportedProtocol: "
                "Request URL is missing an 'http://' or 'https://' protocol."
            ),
        )

    async def put(self, result: PublicationRankResult, source: str) -> None:
        self.put_results.append(result)

    async def close(self) -> None:
        pass


class _SuccessfulLLMRank:
    async def query(self, name: str) -> PublicationRankResult:
        return PublicationRankResult(name=name, sci=None, ccf="A", success=True)


class LLMRankClientConfigTests(unittest.TestCase):
    def test_uses_app_settings_when_process_env_does_not_export_llm_config(self) -> None:
        settings = SimpleNamespace(
            llm_base_url="https://example.test/compatible-mode/v1",
            llm_api_key="settings-key",
            thinking_model="settings-model",
        )

        with patch.dict(os.environ, {}, clear=True), patch(
            "app.services.publication_rank.llm_rank.get_settings",
            return_value=settings,
            create=True,
        ):
            client = LLMRankClient()

        self.assertEqual(client.base_url, "https://example.test/compatible-mode/v1")
        self.assertEqual(client.api_key, "settings-key")
        self.assertEqual(client.model, "settings-model")

    def test_invalid_base_url_returns_configuration_error_without_http_request(self) -> None:
        client = LLMRankClient(
            base_url="",
            api_key="settings-key",
            model="settings-model",
        )

        with patch("app.services.publication_rank.llm_rank.httpx.AsyncClient") as client_cls:
            result = asyncio.run(client.query("Neural Information Processing Systems"))

        self.assertFalse(result.success)
        self.assertIn("LLM_BASEURL", result.error or "")
        self.assertIn("http://", result.error or "")
        client_cls.assert_not_called()


class UnifiedRankClientCacheTests(unittest.TestCase):
    def test_stale_llm_configuration_failure_cache_does_not_block_retry(self) -> None:
        cache = _CacheWithStaleFailure()
        client = UnifiedRankClient(
            cache=cache,
            llm_client=_SuccessfulLLMRank(),
            use_easyscholar=False,
        )

        result = asyncio.run(client.query("Neural Information Processing Systems"))

        self.assertTrue(result.success)
        self.assertEqual(result.ccf, "A")
        self.assertEqual(len(cache.put_results), 1)
        self.assertTrue(cache.put_results[0].success)


if __name__ == "__main__":
    unittest.main()
