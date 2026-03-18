"""
PublicationRank LLM web_search fallback — 单元测试 + 集成测试。

运行单元测试（无需网络）：
    python -m pytest PublicationRank/test_llm_rank.py -v
    python PublicationRank/test_llm_rank.py

运行集成测试（需要 .env 中配置 LLM_BASEURL 和 LLM_APIKEY）：
    python PublicationRank/test_llm_rank.py --integration
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PublicationRank.publication_rank import EasyScholarClient, PublicationRankResult
from PublicationRank.rank_cache import RankCache
from PublicationRank.llm_rank import (
    LLMRankClient,
    UnifiedRankClient,
    _parse_llm_response,
)


# ===================================================================
# TestParseLLMResponse
# ===================================================================

class TestParseLLMResponse(unittest.TestCase):
    """测试 _parse_llm_response 的各种输入场景。"""

    def test_standard_json(self):
        text = '{"sci": "Q1", "ccf": "A"}'
        r = _parse_llm_response(text, "Test Journal")
        self.assertTrue(r.success)
        self.assertEqual(r.sci, "Q1")
        self.assertEqual(r.ccf, "A")

    def test_null_values(self):
        text = '{"sci": null, "ccf": "B"}'
        r = _parse_llm_response(text, "Test Conf")
        self.assertTrue(r.success)
        self.assertIsNone(r.sci)
        self.assertEqual(r.ccf, "B")

    def test_both_null(self):
        text = '{"sci": null, "ccf": null}'
        r = _parse_llm_response(text, "Unknown")
        self.assertTrue(r.success)
        self.assertIsNone(r.sci)
        self.assertIsNone(r.ccf)

    def test_markdown_fence(self):
        text = '这是查询结果：\n```json\n{"sci": "Q2", "ccf": null}\n```\n以上是结果。'
        r = _parse_llm_response(text, "Journal X")
        self.assertTrue(r.success)
        self.assertEqual(r.sci, "Q2")

    def test_embedded_json(self):
        text = '根据搜索结果，该期刊的等级信息如下：{"sci": "Q3", "ccf": "C"}。希望对你有帮助。'
        r = _parse_llm_response(text, "Journal Y")
        self.assertTrue(r.success)
        self.assertEqual(r.sci, "Q3")
        self.assertEqual(r.ccf, "C")

    def test_invalid_sci_filtered(self):
        text = '{"sci": "Q5", "ccf": "A"}'
        r = _parse_llm_response(text, "Bad SCI")
        self.assertTrue(r.success)
        self.assertIsNone(r.sci)
        self.assertEqual(r.ccf, "A")

    def test_invalid_ccf_filtered(self):
        text = '{"sci": "Q1", "ccf": "D"}'
        r = _parse_llm_response(text, "Bad CCF")
        self.assertTrue(r.success)
        self.assertEqual(r.sci, "Q1")
        self.assertIsNone(r.ccf)

    def test_case_normalization(self):
        text = '{"sci": "q1", "ccf": "a"}'
        r = _parse_llm_response(text, "Case Test")
        self.assertTrue(r.success)
        self.assertEqual(r.sci, "Q1")
        self.assertEqual(r.ccf, "A")

    def test_garbage_input(self):
        text = "I don't know anything about that."
        r = _parse_llm_response(text, "Garbage")
        self.assertFalse(r.success)
        self.assertIn("非法 JSON", r.error)

    def test_empty_input(self):
        r = _parse_llm_response("", "Empty")
        self.assertFalse(r.success)

    def test_none_like_empty(self):
        r = _parse_llm_response("   ", "Whitespace")
        self.assertFalse(r.success)


# ===================================================================
# TestRankCache
# ===================================================================

class TestRankCache(unittest.IsolatedAsyncioTestCase):
    """测试 RankCache 的增删查逻辑。"""

    async def asyncSetUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test_cache.sqlite3"
        self.cache = RankCache(db_path=self._db_path, ttl_days=180, negative_ttl_days=7)
        await self.cache.init()

    async def asyncTearDown(self):
        await self.cache.close()
        self._tmpdir.cleanup()

    async def test_put_and_get(self):
        result = PublicationRankResult(name="Nature", sci="Q1", ccf=None, success=True)
        await self.cache.put(result, source="easyscholar")
        cached = await self.cache.get("Nature")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.sci, "Q1")
        self.assertEqual(cached.name, "Nature")

    async def test_cache_miss(self):
        cached = await self.cache.get("Nonexistent Journal")
        self.assertIsNone(cached)

    async def test_expired_entry_returns_none(self):
        # 用极短 TTL 的缓存
        short_cache = RankCache(db_path=self._db_path, ttl_days=0, negative_ttl_days=0)
        await short_cache.init()

        result = PublicationRankResult(name="Old", sci="Q4", success=True)
        await short_cache.put(result, source="test")

        # ttl_days=0 → expires_at ≈ now，立刻过期
        cached = await short_cache.get("Old")
        self.assertIsNone(cached)
        await short_cache.close()

    async def test_name_normalization_dedup(self):
        r1 = PublicationRankResult(name="  IEEE TMI  ", sci="Q1", success=True)
        await self.cache.put(r1, source="test")

        # 用不同空格查询应该命中
        cached = await self.cache.get("IEEE TMI")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.sci, "Q1")

    async def test_batch_query(self):
        r1 = PublicationRankResult(name="A", sci="Q1", success=True)
        r2 = PublicationRankResult(name="B", ccf="A", success=True)
        await self.cache.put(r1, source="test")
        await self.cache.put(r2, source="test")

        batch = await self.cache.get_batch(["A", "B", "C"])
        self.assertIsNotNone(batch["A"])
        self.assertIsNotNone(batch["B"])
        self.assertIsNone(batch["C"])


# ===================================================================
# TestLLMRankClient
# ===================================================================

class TestLLMRankClient(unittest.IsolatedAsyncioTestCase):
    """用 mock httpx 响应测试 LLMRankClient。"""

    def _make_api_response(self, text: str, status_code: int = 200) -> MagicMock:
        """构造模拟的 httpx.Response。"""
        resp = MagicMock()
        resp.status_code = status_code
        resp.headers = {}
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                message=f"HTTP {status_code}",
                request=MagicMock(),
                response=resp,
            )
        resp.json.return_value = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                }
            ]
        }
        return resp

    @patch("PublicationRank.llm_rank.httpx.AsyncClient")
    async def test_successful_query(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            return_value=self._make_api_response('{"sci": "Q1", "ccf": "A"}')
        )
        mock_client_cls.return_value = mock_client

        client = LLMRankClient(base_url="http://test", api_key="key", model="m")
        result = await client.query("CVPR")
        self.assertTrue(result.success)
        self.assertEqual(result.ccf, "A")

    @patch("PublicationRank.llm_rank.httpx.AsyncClient")
    async def test_garbage_response(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            return_value=self._make_api_response("Sorry, I cannot help with that.")
        )
        mock_client_cls.return_value = mock_client

        client = LLMRankClient(base_url="http://test", api_key="key", model="m")
        result = await client.query("Unknown Conf")
        self.assertFalse(result.success)

    @patch("PublicationRank.llm_rank.httpx.AsyncClient")
    async def test_batch_concurrency(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            return_value=self._make_api_response('{"sci": null, "ccf": "B"}')
        )
        mock_client_cls.return_value = mock_client

        client = LLMRankClient(base_url="http://test", api_key="key", model="m")
        results = await client.query_batch(["A", "B", "C"], concurrency=2)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertTrue(r.success)
            self.assertEqual(r.ccf, "B")


# 需要在模块级导入 httpx 用于测试
import httpx


# ===================================================================
# TestUnifiedRankClient
# ===================================================================

class TestUnifiedRankClient(unittest.IsolatedAsyncioTestCase):
    """测试 UnifiedRankClient 的分层查询逻辑。"""

    async def asyncSetUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test_unified.sqlite3"
        self.cache = RankCache(db_path=self._db_path)
        await self.cache.init()

        self.es_client = MagicMock(spec=EasyScholarClient)
        self.llm_client = MagicMock(spec=LLMRankClient)

    async def asyncTearDown(self):
        await self.cache.close()
        self._tmpdir.cleanup()

    async def test_cache_hit_skips_api(self):
        # 预填缓存
        cached = PublicationRankResult(name="CVPR", ccf="A", success=True)
        await self.cache.put(cached, source="test")

        client = UnifiedRankClient(
            cache=self.cache, easyscholar=self.es_client, llm_client=self.llm_client
        )
        result = await client.query("CVPR")
        self.assertEqual(result.ccf, "A")
        # EasyScholar 和 LLM 不应被调用
        self.es_client.query.assert_not_called()
        self.llm_client.query.assert_not_called()

    async def test_easyscholar_hit_cached(self):
        es_result = PublicationRankResult(name="Nature", sci="Q1", success=True)
        self.es_client.query.return_value = es_result

        client = UnifiedRankClient(
            cache=self.cache, easyscholar=self.es_client, llm_client=self.llm_client
        )
        result = await client.query("Nature")
        self.assertEqual(result.sci, "Q1")
        self.llm_client.query.assert_not_called()

        # 验证已缓存
        cached = await self.cache.get("Nature")
        self.assertIsNotNone(cached)

    async def test_easyscholar_empty_fallback_to_llm(self):
        # EasyScholar 返回空结果
        es_result = PublicationRankResult(name="CVPR", sci=None, ccf=None, success=True)
        self.es_client.query.return_value = es_result

        llm_result = PublicationRankResult(name="CVPR", ccf="A", success=True)
        self.llm_client.query = AsyncMock(return_value=llm_result)

        client = UnifiedRankClient(
            cache=self.cache, easyscholar=self.es_client, llm_client=self.llm_client
        )
        result = await client.query("CVPR")
        self.assertEqual(result.ccf, "A")
        self.llm_client.query.assert_called_once()

    async def test_all_fail(self):
        es_result = PublicationRankResult(name="XXX", success=True)
        self.es_client.query.return_value = es_result

        llm_result = PublicationRankResult(name="XXX", success=False, error="timeout")
        self.llm_client.query = AsyncMock(return_value=llm_result)

        client = UnifiedRankClient(
            cache=self.cache, easyscholar=self.es_client, llm_client=self.llm_client
        )
        result = await client.query("XXX")
        self.assertFalse(result.success)


# ===================================================================
# Integration test
# ===================================================================

async def _run_integration_test():
    """集成测试：查询 6 个知名出版物并打印表格。"""
    from PublicationRank.llm_rank import UnifiedRankClient

    publications = [
        "IEEE Transactions on Medical Imaging",
        "CVPR",
        "NeurIPS",
        "AAAI",
        "Nature",
        "ACL",
    ]

    print("\n" + "=" * 70)
    print("PublicationRank LLM Fallback — 集成测试")
    print("=" * 70)

    async with UnifiedRankClient() as client:
        results = await client.query_batch(publications)

    print(f"\n{'出版物':<45} {'SCI':<8} {'CCF':<8} {'状态'}")
    print("-" * 70)
    for r in results:
        status = "OK" if r.success else f"FAIL: {r.error}"
        sci = r.sci or "-"
        ccf = r.ccf or "-"
        print(f"{r.name:<45} {sci:<8} {ccf:<8} {status}")
    print()


if __name__ == "__main__":
    if "--integration" in sys.argv:
        asyncio.run(_run_integration_test())
    else:
        # 移除 --integration 避免 unittest 报错
        sys.argv = [a for a in sys.argv if a != "--integration"]
        unittest.main()
