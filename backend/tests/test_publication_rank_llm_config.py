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
            llm_rank_api_style="responses",
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
        self.assertEqual(client.api_style, "responses")
        self.assertFalse(client._use_chat_completions)

    def test_invalid_base_url_returns_configuration_error_without_http_request(self) -> None:
        client = LLMRankClient(
            base_url="",
            api_key="settings-key",
            model="settings-model",
            api_style="responses",
        )

        with patch("app.services.publication_rank.llm_rank.httpx.AsyncClient") as client_cls:
            result = asyncio.run(client.query("Neural Information Processing Systems"))

        self.assertFalse(result.success)
        self.assertIn("LLM_BASEURL", result.error or "")
        self.assertIn("http://", result.error or "")
        client_cls.assert_not_called()

    def test_api_style_explicit_chat_completions_targets_chat_endpoint(self) -> None:
        client = LLMRankClient(
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="k",
            model="m",
            api_style="chat_completions",
        )
        url, payload = client._build_payload("CVPR")
        self.assertEqual(
            url, "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        )
        self.assertIn("messages", payload)
        self.assertTrue(payload.get("enable_search"))

    def test_api_style_default_responses_targets_responses_endpoint(self) -> None:
        # Even when URL contains "dashscope", default api_style=responses must NOT
        # silently switch to /chat/completions (regression guard for the Bailian
        # apps protocol URL that only exposes /responses).
        client = LLMRankClient(
            base_url="https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1",
            api_key="k",
            model="m",
            api_style="responses",
        )
        url, payload = client._build_payload("CVPR")
        self.assertEqual(
            url,
            "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1/responses",
        )
        self.assertIn("input", payload)
        self.assertEqual(payload.get("tools"), [{"type": "web_search"}])

    def test_unknown_api_style_falls_back_to_responses(self) -> None:
        client = LLMRankClient(
            base_url="https://example.test/v1",
            api_key="k",
            model="m",
            api_style="gibberish",
        )
        self.assertEqual(client.api_style, "responses")


class EasyScholarSettingsTests(unittest.TestCase):
    def test_secret_key_read_from_app_settings_not_only_os_environ(self) -> None:
        from app.services.publication_rank import publication_rank as pr_mod

        settings = SimpleNamespace(
            easyscholar_secret_key="settings-secret",
            easyscholar_api_url="https://example.test/getPublicationRank",
        )
        with patch.dict(os.environ, {}, clear=True), patch(
            "app.config.get_settings", return_value=settings, create=True,
        ):
            client = pr_mod.EasyScholarClient()
        self.assertEqual(client.secret_key, "settings-secret")
        self.assertEqual(client.api_url, "https://example.test/getPublicationRank")

    def test_os_environ_still_works_as_fallback(self) -> None:
        from app.services.publication_rank import publication_rank as pr_mod

        # Force AppSettings import to raise so we fall back to os.getenv.
        with patch.dict(os.environ, {"EASYSCHOLAR_SECRET_KEY": "env-secret"}, clear=True), patch(
            "app.config.get_settings", side_effect=RuntimeError("settings unavailable"),
            create=True,
        ):
            client = pr_mod.EasyScholarClient()
        self.assertEqual(client.secret_key, "env-secret")


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
