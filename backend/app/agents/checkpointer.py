"""Durable LangGraph checkpointer for agent sessions.

A session's `thread_id` is the LangGraph thread; the checkpointer is what lets
a run resume after the process restarts, on this instance or another one.

Checkpoints live in the application database (`checkpoints`,
`checkpoint_blobs`, `checkpoint_writes` — table names keep them apart) but on a
pool of their own. The saver requires autocommit connections with dict rows
and no server-side prepared statements, none of which the application pool
uses, and checkpoint writes are bursty enough that they should not queue behind
ordinary queries.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.db import database

logger = logging.getLogger("scholar.agents.checkpointer")

CHECKPOINT_POOL_MAX = 4


@asynccontextmanager
async def open_checkpointer(url: str | None = None) -> AsyncIterator[AsyncPostgresSaver]:
    """Open a durable checkpointer, creating its tables on first use.

    Long-lived callers (the FastAPI lifespan, the agent worker) should hold
    this open for their whole lifetime rather than per request.
    """
    pool = AsyncConnectionPool(
        url or database.get_database_url(),
        min_size=1,
        max_size=CHECKPOINT_POOL_MAX,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
        name="scholar-checkpoints",
    )
    await pool.open(wait=True, timeout=30)
    try:
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await saver.setup()
        logger.info("Agent checkpointer ready")
        yield saver
    finally:
        await pool.close()
