from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("scholar.db")

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_db_path: Path | None = None


def set_db_path(path: Path) -> None:
    global _db_path
    _db_path = path


def _get_db_path() -> Path:
    if _db_path is None:
        raise RuntimeError("Database path not initialized. Call set_db_path() first.")
    return _db_path


def get_db_path() -> Path:
    return _get_db_path()


async def init_db() -> None:
    path = _get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(path) as db:
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        await db.executescript(schema)
        await db.commit()
        # Migrate existing tables: add new columns if missing
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
        # Migrate runs table: add language column if missing
        try:
            await db.execute("ALTER TABLE runs ADD COLUMN language TEXT NOT NULL DEFAULT 'en'")
            await db.commit()
        except Exception:
            pass  # column already exists
        # Migrate runs table: add user_question + detected_intent for Smart Q&A,
        # current_step + progress_json for resumable progress display.
        for col, col_def in [
            ("user_question", "TEXT DEFAULT ''"),
            ("detected_intent", "TEXT DEFAULT ''"),
            ("current_step", "TEXT DEFAULT ''"),
            ("progress_json", "TEXT DEFAULT '[]'"),
        ]:
            try:
                await db.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_def}")
                await db.commit()
            except Exception:
                pass  # column already exists
        for index_sql in (
            "CREATE INDEX IF NOT EXISTS idx_runs_status_started ON runs(status, started_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC)",
        ):
            try:
                await db.execute(index_sql)
                await db.commit()
            except Exception:
                logger.exception("Failed to create runs index")
        # Reconcile abandoned runs: any run still pending/running at startup lost
        # its owning background task when the previous process exited, so it can
        # never finish. Mark them failed to avoid zombie "running" entries that
        # would otherwise linger in the recent-runs banner forever.
        try:
            cursor = await db.execute(
                "UPDATE runs SET status = 'failed', "
                "error_msg = 'Interrupted (server restarted)', "
                "finished_at = datetime('now') "
                "WHERE status IN ('pending', 'running')"
            )
            await db.commit()
            if cursor.rowcount:
                logger.info("Reconciled %d interrupted run(s) on startup", cursor.rowcount)
        except Exception:
            logger.exception("Failed to reconcile interrupted runs on startup")
        # Migrate mineru_parses table: add remote poll diagnostics.
        for col, col_def in [
            ("remote_batch_id", "TEXT DEFAULT ''"),
            ("poll_count", "INTEGER DEFAULT 0"),
            ("last_state_counts", "TEXT DEFAULT ''"),
            ("last_poll_at", "TEXT DEFAULT ''"),
        ]:
            try:
                await db.execute(f"ALTER TABLE mineru_parses ADD COLUMN {col} {col_def}")
                await db.commit()
            except Exception:
                pass  # column already exists
        # Optional FTS index for Smart Q&A hierarchy nodes. The regular
        # paper_nodes table remains the source of truth if FTS5 is unavailable.
        try:
            await db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS paper_node_fts "
                "USING fts5(node_id UNINDEXED, paper_id UNINDEXED, title_path, text_for_search)"
            )
            await db.commit()
        except Exception:
            pass


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
