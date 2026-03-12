from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_db_path: Path | None = None


def set_db_path(path: Path) -> None:
    global _db_path
    _db_path = path


def _get_db_path() -> Path:
    if _db_path is None:
        raise RuntimeError("Database path not initialized. Call set_db_path() first.")
    return _db_path


async def init_db() -> None:
    path = _get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as db:
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        await db.executescript(schema)
        await db.commit()
        # Migrate existing papers table: add new columns if missing
        for col, col_def in [
            ("venue", "TEXT DEFAULT ''"),
            ("year", "INTEGER DEFAULT 0"),
            ("sci_rank", "TEXT DEFAULT ''"),
            ("ccf_rank", "TEXT DEFAULT ''"),
        ]:
            try:
                await db.execute(f"ALTER TABLE papers ADD COLUMN {col} {col_def}")
                await db.commit()
            except Exception:
                pass  # column already exists


async def execute(sql: str, params: tuple[Any, ...] = ()) -> None:
    async with aiosqlite.connect(_get_db_path()) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute(sql, params)
        await db.commit()


async def execute_many(sql: str, params_seq: list[tuple[Any, ...]]) -> None:
    async with aiosqlite.connect(_get_db_path()) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.executemany(sql, params_seq)
        await db.commit()


async def fetch_one(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    async with aiosqlite.connect(_get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        cursor = await db.execute(sql, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)


async def fetch_all(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    async with aiosqlite.connect(_get_db_path()) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
