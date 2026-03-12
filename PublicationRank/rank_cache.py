"""
SQLite 持久缓存 — 存储 PublicationRankResult，支持 TTL 自动过期。
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

from PublicationRank.publication_rank import (
    PublicationRankResult,
    _normalize_publication_name,
)

logger = logging.getLogger("scholar.rank_cache")

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS publication_rank_cache (
    name_normalized  TEXT PRIMARY KEY,
    name_display     TEXT NOT NULL,
    sci              TEXT,
    ccf              TEXT,
    source           TEXT NOT NULL,
    success          INTEGER NOT NULL DEFAULT 1,
    error            TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at       TEXT NOT NULL
);
"""

_DEFAULT_DB = Path(__file__).parent / "rank_cache.sqlite3"


class RankCache:
    """异步 SQLite 缓存，按归一化名称去重，支持正/负缓存 TTL。"""

    def __init__(
        self,
        db_path: Path | str = _DEFAULT_DB,
        ttl_days: int = 180,
        negative_ttl_days: int = 7,
    ):
        self._db_path = str(db_path)
        self._ttl_days = ttl_days
        self._negative_ttl_days = negative_ttl_days
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """建表并启用 WAL 模式。"""
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def get(self, name: str) -> PublicationRankResult | None:
        """命中且未过期 → 返回结果；过期 → 删除并返回 None。"""
        assert self._db is not None, "call init() first"
        key = _normalize_publication_name(name)
        cursor = await self._db.execute(
            "SELECT name_display, sci, ccf, success, error, expires_at "
            "FROM publication_rank_cache WHERE name_normalized = ?",
            (key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        name_display, sci, ccf, success, error, expires_at = row
        exp = datetime.fromisoformat(expires_at).replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) >= exp:
            await self._db.execute(
                "DELETE FROM publication_rank_cache WHERE name_normalized = ?",
                (key,),
            )
            await self._db.commit()
            logger.debug("cache expired for %s", key)
            return None

        return PublicationRankResult(
            name=name_display,
            sci=sci,
            ccf=ccf,
            success=bool(success),
            error=error,
        )

    async def put(self, result: PublicationRankResult, source: str) -> None:
        """UPSERT 一条结果到缓存。"""
        assert self._db is not None, "call init() first"
        key = _normalize_publication_name(result.name)
        ttl = self._ttl_days if result.success else self._negative_ttl_days
        expires = datetime.now(timezone.utc) + timedelta(days=ttl)

        await self._db.execute(
            "INSERT INTO publication_rank_cache "
            "(name_normalized, name_display, sci, ccf, source, success, error, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(name_normalized) DO UPDATE SET "
            "name_display=excluded.name_display, sci=excluded.sci, ccf=excluded.ccf, "
            "source=excluded.source, success=excluded.success, error=excluded.error, "
            "created_at=datetime('now'), expires_at=excluded.expires_at",
            (
                key,
                result.name,
                result.sci,
                result.ccf,
                source,
                int(result.success),
                result.error,
                expires.isoformat(),
            ),
        )
        await self._db.commit()

    async def get_batch(
        self, names: list[str]
    ) -> dict[str, PublicationRankResult | None]:
        """批量查缓存，返回 {原始名称: 结果或None}。"""
        results: dict[str, PublicationRankResult | None] = {}
        for name in names:
            results[name] = await self.get(name)
        return results

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
