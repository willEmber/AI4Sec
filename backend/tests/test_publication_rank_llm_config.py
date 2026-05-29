from __future__ import annotations

import asyncio
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.services.publication_rank.llm_rank import (
    LLMRankClient,
    UnifiedRankClient,
    _is_transient_failure,
)
from app.services.publication_rank.publication_rank import PublicationRankResult


# ---------------------------------------------------------------------------
# Test doubles for the Tavily search + LLM extraction pipeline
# ---------------------------------------------------------------------------

class _FakeTavily:
    def __init__(self, context: str = "result", configured: bool = True, raise_exc: Exception | None = None) -> None:
        self._context = context
        self._configured = configured
        self._raise = raise_exc
        self.called = False

    @property
    def configured(self) -> bool:
        return self._configured

    async def search_context(self, query: str, **kwargs: object) -> str:
        self.called = True
        if self._raise is not None:
            raise self._raise
        return self._context


class _FakeLLM:
    def __init__(self, response: str = "{}") -> None:
        self._response = response
        self.called = False
        self.last_messages: list[dict[str, str]] | None = None

    async def chat(self, messages, model="", temperature=0.3, max_tokens=4096):
        self.called = True
        self.last_messages = messages
        return self._response


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


# ---------------------------------------------------------------------------
# LLMRankClient (Tavily + LLM) tests
# ---------------------------------------------------------------------------

class LLMRankClientConfigTests(unittest.TestCase):
    def test_uses_app_settings_when_process_env_does_not_export_llm_config(self) -> None:
        settings = SimpleNamespace(
            llm_base_url="https://example.test/compatible-mode/v1",
            llm_api_key="settings-key",
            thinking_model="settings-model",
            tavily_api_key="tavily-key",
        )

        with patch.dict(os.environ, {}, clear=True), patch(
            "app.services.publication_rank.llm_rank.get_settings",
            return_value=settings,
            create=True,
        ):
            client = LLMRankClient(
                tavily_client=_FakeTavily(), llm_service=_FakeLLM(),
            )

        self.assertEqual(client.base_url, "https://example.test/compatible-mode/v1")
        self.assertEqual(client.api_key, "settings-key")
        self.assertEqual(client.model, "settings-model")

    def test_model_uses_first_of_comma_separated_list(self) -> None:
        client = LLMRankClient(
            base_url="https://example.test/v1",
            api_key="k",
            model="qwen3.6-plus, qwen3.7-max",
            tavily_client=_FakeTavily(),
            llm_service=_FakeLLM(),
        )
        self.assertEqual(client.model, "qwen3.6-plus")

    def test_invalid_base_url_returns_configuration_error_without_network(self) -> None:
        tavily = _FakeTavily()
        llm = _FakeLLM()
        client = LLMRankClient(
            base_url="",
            api_key="settings-key",
            model="settings-model",
            tavily_client=tavily,
            llm_service=llm,
        )

        result = asyncio.run(client.query("Neural Information Processing Systems"))

        self.assertFalse(result.success)
        self.assertIn("LLM_BASEURL", result.error or "")
        self.assertIn("http://", result.error or "")
        self.assertFalse(tavily.called)
        self.assertFalse(llm.called)

    def test_missing_tavily_key_returns_error_without_llm(self) -> None:
        tavily = _FakeTavily(configured=False)
        llm = _FakeLLM()
        client = LLMRankClient(
            base_url="https://example.test/v1",
            api_key="k",
            model="m",
            tavily_client=tavily,
            llm_service=llm,
        )

        result = asyncio.run(client.query("Nature"))

        self.assertFalse(result.success)
        self.assertIn("TAVILY_KEY", result.error or "")
        self.assertFalse(tavily.called)
        self.assertFalse(llm.called)

    def test_tavily_plus_llm_extracts_rank(self) -> None:
        tavily = _FakeTavily(context="CVPR 是 CCF A 类会议，无 SCI 分区")
        llm = _FakeLLM(response='{"sci": null, "ccf": "A"}')
        client = LLMRankClient(
            base_url="https://example.test/v1",
            api_key="k",
            model="m",
            tavily_client=tavily,
            llm_service=llm,
        )

        result = asyncio.run(client.query("CVPR"))

        self.assertTrue(result.success)
        self.assertIsNone(result.sci)
        self.assertEqual(result.ccf, "A")
        self.assertTrue(tavily.called)
        self.assertTrue(llm.called)
        # The Tavily search context must be passed into the LLM prompt.
        joined = "".join(m["content"] for m in (llm.last_messages or []))
        self.assertIn("CCF A 类会议", joined)

    def test_tavily_failure_is_transient_and_skips_llm(self) -> None:
        tavily = _FakeTavily(raise_exc=RuntimeError("boom"))
        llm = _FakeLLM()
        client = LLMRankClient(
            base_url="https://example.test/v1",
            api_key="k",
            model="m",
            tavily_client=tavily,
            llm_service=llm,
        )

        result = asyncio.run(client.query("Some Venue"))

        self.assertFalse(result.success)
        self.assertIn("Tavily 搜索失败", result.error or "")
        self.assertTrue(_is_transient_failure(result))
        self.assertFalse(llm.called)


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
