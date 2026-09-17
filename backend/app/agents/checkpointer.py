"""Durable LangGraph checkpointer for agent sessions.

A session's `thread_id` is the LangGraph thread; the checkpointer is what lets
a run resume after the process restarts. The first version keeps this in SQLite
alongside the application database — one writer, `WAL` journalling — which the
development plan flags for re-evaluation (PostgreSQL) before multi-instance
deployment.

The checkpoint database is deliberately a *separate* file from `app.db`:
checkpoint rows churn far faster than business rows, and keeping them apart
means a checkpoint prune or reset never risks the papers/runs tables.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import get_settings

logger = logging.getLogger("scholar.agents.checkpointer")


def agent_checkpoint_db_path() -> Path:
    """Resolve the checkpoint database path from settings."""
    settings = get_settings()
    configured = (settings.agent_checkpoint_db or "").strip()
    if configured:
        return Path(configured)
    return settings.data_dir / "agent_checkpoints.db"


@asynccontextmanager
async def open_checkpointer(path: Path | None = None) -> AsyncIterator[AsyncSqliteSaver]:
    """Open a durable checkpointer, creating its tables on first use.

    Yields a saver bound to an open connection; the connection closes on exit,
    so long-lived callers (the FastAPI lifespan, the agent worker) should hold
    this open for their whole lifetime rather than per request.
    """
    db_path = path or agent_checkpoint_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(db_path))
    try:
        # WAL lets readers run while the single writer commits; without it a
        # concurrent read during a checkpoint write raises "database is locked".
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA busy_timeout=5000")
        await conn.commit()
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        logger.info("Agent checkpointer ready at %s", db_path)
        yield saver
    finally:
        await conn.close()
