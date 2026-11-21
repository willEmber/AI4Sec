"""
出版物等级动态缓存（L2）— 默认使用 PostgreSQL `publication_rank_cache` 表，支持多副本共享与源优先级单调保护；
当显式传入 ``db_path``（或 PostgreSQL 连接池未初始化的独立离线脚本环境）时自动回退至 SQLite，完全兼容已有工具与单测。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiosqlite

from .models import source_priority
from .normalizer import normalize_issn, normalize_name_key, strip_venue_decorations
from .publication_rank import (
    PublicationRankResult,
    _normalize_publication_name,
)

logger = logging.getLogger("scholar.rank_cache")

_SQLITE_SCHEMA = """\
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
_SQLITE_ADD_EXTRA = "ALTER TABLE publication_rank_cache ADD COLUMN extra_json TEXT NOT NULL DEFAULT ''"
_SQLITE_ADD_PRIORITY = "ALTER TABLE publication_rank_cache ADD COLUMN source_priority INTEGER NOT NULL DEFAULT 0"


def _default_db_path() -> Path:
    from app.config import get_settings

    return get_settings().data_dir / "rank_cache.sqlite3"


def _pg_pool_ready() -> bool:
    try:
        from app.db import database as db

        return db._pool is not None
    except Exception:
        return False


def canonical_cache_key(name: str, *, issn: str = "") -> str:
    """Canonical case-insensitive lookup key for PostgreSQL L2 cache."""
    clean_issn = normalize_issn(issn)
    if clean_issn:
        return f"issn:{clean_issn}"
    stripped, _ = strip_venue_decorations(name)
    nk = normalize_name_key(stripped or name)
    return f"name:{nk}" if nk else f"raw:{_normalize_publication_name(name)}"


def _ttl_days_for_source(source: str, *, success: bool, default_ttl: int, negative_ttl: int) -> int:
    if not success:
        return negative_ttl
    src = (source or "").strip().lower()
    if src == "llm_websearch":
        return min(default_ttl, 14)
    if src == "openalex":
        return min(default_ttl, 90)
    return default_ttl


class RankCache:
    """持久缓存：PostgreSQL 主模式（多副本共享 + 源优先级保护）+ SQLite 兼容回退模式。"""

    def __init__(
        self,
        db_path: Path | str | None = None,
        ttl_days: int = 180,
        negative_ttl_days: int = 7,
    ):
        self._explicit_db_path = db_path is not None
        self._db_path = str(db_path) if db_path is not None else ""
        self._ttl_days = ttl_days
        self._negative_ttl_days = negative_ttl_days
        self._use_pg = False
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        if not self._explicit_db_path and _pg_pool_ready():
            self._use_pg = True
            return

        self._use_pg = False
        if not self._db_path:
            self._db_path = str(_default_db_path())
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.executescript(_SQLITE_SCHEMA)
        for stmt in (_SQLITE_ADD_EXTRA, _SQLITE_ADD_PRIORITY):
            try:
                await self._db.execute(stmt)
            except Exception:
                pass
        await self._db.commit()

    def _row_to_result(
        self,
        *,
        name_display: str,
        sci: str | None,
        ccf: str | None,
        success: int | bool,
        error: str | None,
        source: str | None,
        extra_json: str | None,
    ) -> PublicationRankResult | None:
        if success and extra_json == "":
            return None
        try:
            extra = json.loads(extra_json or "{}")
        except (TypeError, ValueError):
            extra = {}
        return PublicationRankResult(
            name=name_display,
            sci=sci,
            ccf=ccf,
            success=bool(success),
            error=error,
            extra=extra if isinstance(extra, dict) else {},
            source=source or "",
        )

    async def get(self, name: str, *, issn: str = "") -> PublicationRankResult | None:
        if self._use_pg:
            from app.db import database as db

            keys: list[str] = []
            if issn and normalize_issn(issn):
                keys.append(canonical_cache_key(name, issn=issn))
            if name:
                nk = canonical_cache_key(name)
                if nk not in keys:
                    keys.append(nk)
            for key in keys:
                row = await db.fetch_one(
                    "SELECT name_display, sci, ccf, success, error, expires_at, source, extra_json, "
                    "       (expires_at <= now()) AS expired "
                    "FROM publication_rank_cache WHERE lookup_key = ?",
                    (key,),
                )
                if row is None:
                    continue
                if row.get("expired"):
                    await db.execute("DELETE FROM publication_rank_cache WHERE lookup_key = ?", (key,))
                    continue
                parsed = self._row_to_result(
                    name_display=row["name_display"],
                    sci=row.get("sci"),
                    ccf=row.get("ccf"),
                    success=row.get("success", 1),
                    error=row.get("error"),
                    source=row.get("source"),
                    extra_json=row.get("extra_json"),
                )
                if parsed is not None:
                    return parsed
            return None

        assert self._db is not None, "call init() first"
        key = _normalize_publication_name(name)
        cursor = await self._db.execute(
            "SELECT name_display, sci, ccf, success, error, expires_at, source, extra_json "
            "FROM publication_rank_cache WHERE name_normalized = ?",
            (key,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        name_display, sci, ccf, success, error, expires_at, source, extra_json = row
        exp = datetime.fromisoformat(expires_at).replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) >= exp:
            await self._db.execute(
                "DELETE FROM publication_rank_cache WHERE name_normalized = ?",
                (key,),
            )
            await self._db.commit()
            return None

        return self._row_to_result(
            name_display=name_display,
            sci=sci,
            ccf=ccf,
            success=success,
            error=error,
            source=source,
            extra_json=extra_json,
        )

    async def put(self, result: PublicationRankResult, source: str, *, issn: str = "") -> None:
        prio = source_priority(source, success=result.success)
        ttl = _ttl_days_for_source(
            source,
            success=result.success,
            default_ttl=self._ttl_days,
            negative_ttl=self._negative_ttl_days,
        )
        expires = datetime.now(timezone.utc) + timedelta(days=ttl)
        extra_str = json.dumps(result.extra or {}, ensure_ascii=False)
        clean_issn = normalize_issn(issn)

        if self._use_pg:
            from app.db import database as db

            keys = [canonical_cache_key(result.name)]
            if clean_issn:
                ik = canonical_cache_key(result.name, issn=clean_issn)
                if ik not in keys:
                    keys.insert(0, ik)

            for key in keys:
                await db.execute(
                    """INSERT INTO publication_rank_cache
                           (lookup_key, name_display, issn, sci, ccf, extra_json,
                            source, source_priority, success, error, created_at, expires_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now(), ?)
                       ON CONFLICT (lookup_key) DO UPDATE SET
                           name_display = excluded.name_display,
                           issn = CASE WHEN excluded.issn <> '' THEN excluded.issn ELSE publication_rank_cache.issn END,
                           sci = excluded.sci,
                           ccf = excluded.ccf,
                           extra_json = excluded.extra_json,
                           source = excluded.source,
                           source_priority = excluded.source_priority,
                           success = excluded.success,
                           error = excluded.error,
                           created_at = now(),
                           expires_at = excluded.expires_at
                       WHERE publication_rank_cache.expires_at <= now()
                          OR excluded.source_priority >= publication_rank_cache.source_priority""",
                    (
                        key,
                        result.name,
                        clean_issn,
                        result.sci,
                        result.ccf,
                        extra_str,
                        source,
                        prio,
                        int(result.success),
                        result.error,
                        expires.isoformat(),
                    ),
                )
            return

        assert self._db is not None, "call init() first"
        key = _normalize_publication_name(result.name)
        cursor = await self._db.execute(
            "SELECT expires_at, source_priority FROM publication_rank_cache WHERE name_normalized = ?",
            (key,),
        )
        existing = await cursor.fetchone()
        if existing is not None:
            old_exp_s, old_prio = existing
            old_exp = datetime.fromisoformat(old_exp_s).replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) < old_exp and int(old_prio or 0) > prio:
                return

        await self._db.execute(
            "INSERT INTO publication_rank_cache "
            "(name_normalized, name_display, sci, ccf, source, source_priority, success, error, expires_at, extra_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(name_normalized) DO UPDATE SET "
            "name_display=excluded.name_display, sci=excluded.sci, ccf=excluded.ccf, "
            "source=excluded.source, source_priority=excluded.source_priority, "
            "success=excluded.success, error=excluded.error, "
            "created_at=datetime('now'), expires_at=excluded.expires_at, "
            "extra_json=excluded.extra_json",
            (
                key,
                result.name,
                result.sci,
                result.ccf,
                source,
                prio,
                int(result.success),
                result.error,
                expires.isoformat(),
                extra_str,
            ),
        )
        await self._db.commit()

    async def get_batch(self, names: list[str]) -> dict[str, PublicationRankResult | None]:
        if self._use_pg and names:
            from app.db import database as db

            key_by_name = {n: canonical_cache_key(n) for n in names if n}
            unique_keys = list(dict.fromkeys(key_by_name.values()))
            if not unique_keys:
                return {n: None for n in names}
            placeholders = ",".join("?" for _ in unique_keys)
            rows = await db.fetch_all(
                f"SELECT lookup_key, name_display, sci, ccf, success, error, source, extra_json "
                f"FROM publication_rank_cache "
                f"WHERE lookup_key IN ({placeholders}) AND expires_at > now()",
                tuple(unique_keys),
            )
            by_key: dict[str, PublicationRankResult | None] = {}
            for r in rows:
                by_key[r["lookup_key"]] = self._row_to_result(
                    name_display=r["name_display"],
                    sci=r.get("sci"),
                    ccf=r.get("ccf"),
                    success=r.get("success", 1),
                    error=r.get("error"),
                    source=r.get("source"),
                    extra_json=r.get("extra_json"),
                )
            return {n: by_key.get(key_by_name.get(n, "")) for n in names}

        results: dict[str, PublicationRankResult | None] = {}
        for name in names:
            results[name] = await self.get(name)
        return results

    async def delete(self, name: str) -> int:
        if self._use_pg:
            from app.db import database as db

            key = canonical_cache_key(name)
            return await db.execute(
                "DELETE FROM publication_rank_cache WHERE lookup_key = ? OR name_display = ?",
                (key, _normalize_publication_name(name)),
            )

        assert self._db is not None, "call init() first"
        key = _normalize_publication_name(name)
        cursor = await self._db.execute(
            "DELETE FROM publication_rank_cache WHERE name_normalized = ?",
            (key,),
        )
        await self._db.commit()
        return cursor.rowcount or 0

    async def clear(self, *, only_failures: bool = False) -> int:
        if self._use_pg:
            from app.db import database as db

            if only_failures:
                deleted = await db.execute("DELETE FROM publication_rank_cache WHERE success = 0")
            else:
                deleted = await db.execute("DELETE FROM publication_rank_cache")
            if deleted:
                logger.info("rank_cache (pg) cleared: %d row(s) deleted (only_failures=%s)", deleted, only_failures)
            return deleted

        assert self._db is not None, "call init() first"
        if only_failures:
            cursor = await self._db.execute("DELETE FROM publication_rank_cache WHERE success = 0")
        else:
            cursor = await self._db.execute("DELETE FROM publication_rank_cache")
        await self._db.commit()
        deleted = cursor.rowcount or 0
        if deleted:
            logger.info("rank_cache cleared: %d row(s) deleted (only_failures=%s)", deleted, only_failures)
        return deleted

    async def count(self, *, only_failures: bool = False) -> int:
        if self._use_pg:
            from app.db import database as db

            sql = (
                "SELECT COUNT(*) AS n FROM publication_rank_cache WHERE success = 0"
                if only_failures
                else "SELECT COUNT(*) AS n FROM publication_rank_cache"
            )
            row = await db.fetch_one(sql)
            return int(row["n"]) if row else 0

        assert self._db is not None, "call init() first"
        if only_failures:
            cursor = await self._db.execute("SELECT COUNT(*) FROM publication_rank_cache WHERE success = 0")
        else:
            cursor = await self._db.execute("SELECT COUNT(*) FROM publication_rank_cache")
        row = await cursor.fetchone()
        return int(row[0]) if row else 0

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None
