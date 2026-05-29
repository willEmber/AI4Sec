from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from app.services.publication_rank import PublicationRankResult, RankCache


class RankCacheCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        db_path = Path(self._tmp.name) / "rank_cache.sqlite3"
        self.cache = RankCache(db_path=db_path)
        await self.cache.init()

    async def asyncTearDown(self) -> None:
        await self.cache.close()
        self._tmp.cleanup()

    async def test_clear_only_failures_keeps_successes(self) -> None:
        await self.cache.put(
            PublicationRankResult(name="CVPR", sci=None, ccf="A", success=True),
            source="easyscholar",
        )
        await self.cache.put(
            PublicationRankResult(name="NeurIPS", success=False, error="boom"),
            source="llm",
        )
        await self.cache.put(
            PublicationRankResult(name="ICML", success=False, error="timeout"),
            source="llm",
        )

        self.assertEqual(await self.cache.count(), 3)
        self.assertEqual(await self.cache.count(only_failures=True), 2)

        deleted = await self.cache.clear(only_failures=True)
        self.assertEqual(deleted, 2)
        self.assertEqual(await self.cache.count(), 1)
        self.assertEqual(await self.cache.count(only_failures=True), 0)

        survivor = await self.cache.get("CVPR")
        self.assertIsNotNone(survivor)
        assert survivor is not None
        self.assertTrue(survivor.success)
        self.assertEqual(survivor.ccf, "A")

    async def test_clear_all_removes_everything(self) -> None:
        await self.cache.put(
            PublicationRankResult(name="CVPR", sci=None, ccf="A", success=True),
            source="easyscholar",
        )
        await self.cache.put(
            PublicationRankResult(name="NeurIPS", success=False, error="boom"),
            source="llm",
        )

        deleted = await self.cache.clear(only_failures=False)
        self.assertEqual(deleted, 2)
        self.assertEqual(await self.cache.count(), 0)

    async def test_delete_single_entry_by_name(self) -> None:
        await self.cache.put(
            PublicationRankResult(name="CVPR", sci=None, ccf="A", success=True),
            source="easyscholar",
        )
        await self.cache.put(
            PublicationRankResult(name="NeurIPS", sci=None, ccf="A", success=True),
            source="easyscholar",
        )

        deleted = await self.cache.delete("neurips")  # name normalization is case-insensitive… or is it?
        # Spec: _normalize_publication_name does NOT lower-case (it only trims/collapses spaces).
        # So a different case will *not* match. We assert exact-match behaviour.
        self.assertEqual(deleted, 0)
        self.assertEqual(await self.cache.count(), 2)

        deleted = await self.cache.delete("NeurIPS")
        self.assertEqual(deleted, 1)
        self.assertEqual(await self.cache.count(), 1)
        self.assertIsNone(await self.cache.get("NeurIPS"))
        self.assertIsNotNone(await self.cache.get("CVPR"))


if __name__ == "__main__":
    asyncio.run(unittest.main())
