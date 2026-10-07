from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from app.db import database as db

logger = logging.getLogger("scholar.progress")


# Progress as it happens, for a caller in the same process that started the run
# and wants each step (a conversation showing a mode tool's steps). Everything
# else reads the steps back from the run's row.
_listeners: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}


def subscribe(run_id: str) -> asyncio.Queue[dict[str, Any]]:
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _listeners.setdefault(run_id, set()).add(queue)
    return queue


def unsubscribe(run_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
    listeners = _listeners.get(run_id)
    if listeners is None:
        return
    listeners.discard(queue)
    if not listeners:
        _listeners.pop(run_id, None)


async def emit_progress(run_id: str, step: str, status: str, **extra: Any) -> None:
    """Single source of truth for per-step progress.

    1. Hand it to in-process listeners (see `subscribe`).
    2. Persist to `runs.current_step` + append to `runs.progress_json`, which
       is what the event stream and a reloaded page read.

    Both legs are best-effort: failures are logged but never raised.
    """
    if not run_id:
        return

    payload: dict[str, Any] = {"step": step, "status": status}
    if extra:
        payload.update(extra)

    for queue in list(_listeners.get(run_id, ())):
        queue.put_nowait(dict(payload))

    try:
        await db.execute(
            """
            UPDATE runs
               SET current_step = ?,
                   progress_json = (
                       COALESCE(NULLIF(progress_json, ''), '[]')::jsonb
                       || jsonb_build_array(?::jsonb)
                   )::text
             WHERE run_id = ?
            """,
            (step, json.dumps(payload, ensure_ascii=False), run_id),
        )
    except Exception as e:
        logger.warning("emit_progress: DB persist failed run=%s step=%s: %s", run_id, step, e)
