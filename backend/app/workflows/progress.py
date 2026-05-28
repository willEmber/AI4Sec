from __future__ import annotations

import json
import logging
from typing import Any

from app.db import database as db

logger = logging.getLogger("scholar.progress")


async def emit_progress(run_id: str, step: str, status: str, **extra: Any) -> None:
    """Single source of truth for per-step progress.

    1. Push to the in-memory SSE queue (live clients receive it instantly).
    2. Persist to `runs.current_step` + append to `runs.progress_json`.
       The DB write lets a client that reconnected after navigating away
       (queue already popped, status still `running`) recover the step list.

    Both legs are best-effort: failures are logged but never raised.
    """
    if not run_id:
        return

    payload: dict[str, Any] = {"step": step, "status": status}
    if extra:
        payload.update(extra)

    # Leg 1: in-memory queue for live SSE consumers.
    try:
        from app.api.runs import _run_queues

        queue = _run_queues.get(run_id)
        if queue is not None:
            await queue.put({"event": "progress", "data": payload})
    except Exception as e:
        logger.debug("emit_progress: queue push failed run=%s step=%s: %s", run_id, step, e)

    # Leg 2: persist so reloads / late polls can see what happened.
    try:
        await db.execute(
            """
            UPDATE runs
               SET current_step = ?,
                   progress_json = json_insert(
                       COALESCE(NULLIF(progress_json, ''), '[]'),
                       '$[#]',
                       json(?)
                   )
             WHERE run_id = ?
            """,
            (step, json.dumps(payload, ensure_ascii=False), run_id),
        )
    except Exception as e:
        logger.debug("emit_progress: DB persist failed run=%s step=%s: %s", run_id, step, e)
