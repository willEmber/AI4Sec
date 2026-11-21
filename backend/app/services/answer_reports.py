"""A report written as the turn's own answer: its `runs` row, from start to snapshot (P16).

`begin_report` (`agents/tools/report.py`) opens the row and hangs a
:class:`ReportTurn` on the turn's context; from there the conversation reads
with its ordinary tools and writes the report as its answer. When the turn
ends, the runner calls :func:`snapshot` (finished) or :func:`close` (stopped,
cut off, failed), and nothing else ever writes the row.

The row is the same `runs` / `run_outputs` pair a report tool writes, so the
report page, the library's compare matrix and both exports read it unchanged.
What differs is `json_data`: it holds the citation table the report page
renders from (`report_citations`), the structured parts the report's tools
handed over, and what each mode extracts from the finished text — never
anything the model was asked to produce in a second format.

The turn keeps the row alive: the runner's heartbeat beats it with the turn
(:func:`beat`), and recovery closes it with the turn (:func:`close_for_turns`).
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.db import database as db
from app.services import report_citations
from app.services.agent_worker import WORKER_ID

logger = logging.getLogger("scholar.answer_reports")

ACTIVE = "status IN ('pending', 'running')"


@dataclass
class ReportTurn:
    """The report a turn is writing. Hung on `AgentContext.report` by `begin_report`."""

    run_id: str
    mode: str
    paper_id: str
    focus: str = ""
    title: str = ""
    # The headings the method requires, checked when the report is saved.
    required: tuple[str, ...] = ()
    # `usage.tokens` when the report began, so the row records what it cost.
    tokens_at_start: int = 0
    # Structured results the report's tools hand over (signals, the blind
    # review, the literature map), keyed by what `json_data` calls them.
    parts: dict[str, Any] = field(default_factory=dict)



async def open_run(
    *,
    run_id: str = "",
    mode: str,
    paper_id: str,
    focus: str,
    llm_model: str,
    language: str,
    owner_id: str,
    owner_token: str,
    session_id: str,
    agent_run_id: str,
) -> str:
    """Insert the report's row, already `running` and stamped with this worker."""
    run_id = run_id or uuid.uuid4().hex[:16]
    await db.execute(
        """INSERT INTO runs
               (run_id, paper_id, mode, llm_model, language, status, user_question,
                owner_token, owner_id, agent_session_id, agent_run_id, started_at,
                worker_id, heartbeat_at)
           VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, now(), ?, now())""",
        (
            run_id, paper_id, mode, llm_model, language, focus[:2000],
            owner_token, owner_id, session_id, agent_run_id, WORKER_ID,
        ),
    )
    return run_id


async def beat(run_id: str) -> None:
    """Keep the row looking alive while its turn is."""
    await db.execute(
        f"UPDATE runs SET heartbeat_at = now(), worker_id = ? WHERE run_id = ? AND {ACTIVE}",
        (WORKER_ID, run_id),
    )


async def close(run_id: str, *, status: str, reason: str, tokens: int = 0) -> bool:
    """Close an unfinished report. Conditional: a closed row keeps its first reason."""
    closed = await db.execute(
        f"UPDATE runs SET status = ?, error_msg = ?, finished_at = now(), "
        f"tokens = tokens + ? WHERE run_id = ? AND {ACTIVE}",
        (status, reason[:500], max(0, tokens), run_id),
    )
    return bool(closed)


async def close_for_turns(agent_run_ids: list[str], *, reason: str) -> list[str]:
    """Close the reports of turns that recovery closed. Returns their run ids."""
    if not agent_run_ids:
        return []
    placeholders = ",".join("?" for _ in agent_run_ids)
    rows = await db.fetch_all(
        f"UPDATE runs SET status = 'failed', error_msg = ?, finished_at = now() "
        f"WHERE agent_run_id IN ({placeholders}) AND {ACTIVE} RETURNING run_id",
        (reason[:500], *agent_run_ids),
    )
    return sorted(row["run_id"] for row in rows)


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)


def missing_sections(body: str, required: tuple[str, ...]) -> list[str]:
    """Required headings the report does not have. Matched loosely: a heading
    that *contains* the required words counts ("2. 方法流程" for "方法流程")."""
    headings = [h.casefold() for h in _HEADING_RE.findall(body)]
    return [r for r in required if not any(r.casefold() in h for h in headings)]


def report_body(shown: str, marks: list[int], required: tuple[str, ...]) -> str:
    """The report among what the turn wrote between its tool results.

    `marks` are where the answer stood after each tool result. The report is
    normally the last stretch — narration between reads is not the report —
    but a model can write the report and call a tool with it (save a memory
    the reader asked for, check one number) and close with a line, which
    would then be saved as the report. The stretch missing the fewest
    required headings is taken, the latest among equals: a draft outline
    written before the reads loses to the report written after them.
    """
    bounds = [0, *marks, len(shown)]
    stretches = [shown[a:b].strip() for a, b in itertools.pairwise(bounds)]
    best = min(
        range(len(stretches)),
        key=lambda i: (len(missing_sections(stretches[i], required)), -i),
    )
    return stretches[best]


async def _paper_title(paper_id: str) -> str:
    row = await db.fetch_one("SELECT title FROM papers WHERE paper_id = ?", (paper_id,))
    return (row or {}).get("title") or ""


async def snapshot(
    report: ReportTurn,
    *,
    body: str,
    owner_id: str,
    language: str,
    stop_reason: str,
    spent: Callable[[], int],
) -> dict[str, Any] | None:
    """Save the finished report. Returns the `artifact.created` payload, or `None`.

    `None` when there is nothing to save (the answer after `begin_report` is
    empty) or the row was closed meanwhile (dismissed from the report page);
    in both cases the text stays an ordinary answer. `spent` reads the turn's
    tokens now, so the row records the report's cost including the
    extraction this function runs.
    """
    if not body.strip():
        await close(report.run_id, status="failed", reason="The report was empty.")
        return None

    text, citations = await report_citations.freeze(body, owner_id=owner_id)
    missing = missing_sections(text, report.required)
    title = report.title or await _paper_title(report.paper_id)
    data: dict[str, Any] = {
        "mode": report.mode,
        "paper_id": report.paper_id,
        "title": title,
        "language": language,
        "written_as": "answer",
        "focus": report.focus,
        "citations": citations,
        "stop_reason": stop_reason,
        "structure_ok": not missing,
        "missing_sections": missing,
    }
    # What a mode adds from the finished text (Lens its digest, Snap its
    # scorecard, Sphere its map). Imported here: the views pull in the report
    # modes' extraction, which the turn path needs only when a report is saved.
    from app.services.report_views import BUILDERS

    builder = BUILDERS.get(report.mode)
    if builder is not None:
        try:
            data.update(await builder(report, text, citations, language))
        except Exception:  # noqa: BLE001 — the report stands without its structured view
            logger.exception("[run:%s] %s structured view failed", report.run_id, report.mode)

    async with db.transaction() as tx:
        await tx.execute(
            "INSERT INTO run_outputs (run_id, markdown, json_data) VALUES (?, ?, ?) "
            "ON CONFLICT (run_id) DO UPDATE SET markdown = excluded.markdown, "
            "json_data = excluded.json_data",
            (report.run_id, text, json.dumps(data, ensure_ascii=False)),
        )
        closed = await tx.execute(
            f"UPDATE runs SET status = 'done', finished_at = now(), tokens = tokens + ? "
            f"WHERE run_id = ? AND {ACTIVE}",
            (max(0, spent() - report.tokens_at_start), report.run_id),
        )
        if not closed:
            # Dismissed while it was being written: the row keeps its reason,
            # and an output under a closed run would only confuse the page.
            await tx.execute("DELETE FROM run_outputs WHERE run_id = ?", (report.run_id,))
            return None
    logger.info(
        "[run:%s] %s report saved as an answer (%d chars, %d citations%s)",
        report.run_id, report.mode, len(text), len(citations),
        f", missing {missing}" if missing else "",
    )
    return {
        "run_id": report.run_id,
        "paper_id": report.paper_id,
        "mode": report.mode,
        "title": title,
        "language": language,
        "reused": False,
        "answer": True,
        "stop_reason": stop_reason,
    }
