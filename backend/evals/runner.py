"""Run a case through the agent and record what it did.

A case takes the path a reader's message takes — session, user message, run
with the configured budget, `_execute_turn` — so what is measured is the
product, not a harness beside it. Each attempt gets its own session: attempts
are independent, and a shared conversation would let one answer feed the next.
A case's setup turns run first in that session, unscored, the way a reader's
earlier messages would have.

A case that can make a report starts with its papers' cached reports
forgotten (`forget_mode_reports`), so every attempt pays for the report it
measures; the caller keeps two such cases on the same paper from overlapping.
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.context import RunBudget
from app.agents.prompts import current_prompt_version
from app.db import agent_repository as repo
from app.db import database as db
from app.models.agent_models import Availability, EventType, MessageRole, RunStatus
from app.services import agent_runner, evidence_service, paper_catalog

from evals.dataset import Dataset, EvalCase
from evals.statements import CITATION_RE, GROUPED_RE, normalise_citations


async def parsed_paper_ids(paper_ids: list[str]) -> set[str]:
    """Which of these files this database holds a parse of."""
    if not paper_ids:
        return set()
    placeholders = ",".join("?" for _ in paper_ids)
    rows = await db.fetch_all(
        f"SELECT DISTINCT paper_id FROM paper_nodes WHERE paper_id IN ({placeholders})",
        tuple(paper_ids),
    )
    return {r["paper_id"] for r in rows}


# The cached work a report reuses, by key prefix: a report tool's whole report,
# and the parts a report written as the answer caches — the literature map and
# the blind review (P16 §3.4). Each attempt has to pay for them, or its cost
# is not comparable with the baseline's.
REPORT_JOB_PREFIXES = ("mode", "sphere_map", "blind_review")


async def forget_mode_reports(paper_ids: list[str]) -> int:
    """Drop the finished report jobs of these papers, so the next one is made anew.

    The reports themselves (`runs`) stay; only the "already made" shortcut is
    removed — for every reader of this database, since a report job is keyed by
    paper, not by owner. A job still running belongs to whoever is running it
    and is left alone.
    """
    removed = 0
    for paper_id in paper_ids:
        for prefix in REPORT_JOB_PREFIXES:
            removed += await db.execute(
                "DELETE FROM agent_jobs WHERE idempotency_key LIKE ? AND status <> 'running'",
                (f"{prefix}:{paper_id}:%",),
            )
    return removed


def _evidence_record(evidence: Any) -> dict[str, Any]:
    return {
        "quote": evidence.quote,
        "source_level": evidence.source_level.value,
        "paper_id": evidence.paper_id,
        "page": evidence.locator.page_label,
        "section": evidence.locator.section_path,
    }


async def refresh_evidence(record: dict[str, Any]) -> None:
    """Bring a stored record up to what the current reader of markers sees.

    A run recorded before a marker form was recognised lacks the passages for
    those ids; left alone they would re-score as invented. Only evidence of the
    record's own session is added.
    """
    text = record.get("answer") or ""
    record["grouped_citation_markers"] = len(GROUPED_RE.findall(text))
    evidence = record.setdefault("evidence", {})
    missing = [
        evidence_id
        for evidence_id in dict.fromkeys(CITATION_RE.findall(normalise_citations(text)))
        if evidence_id not in evidence
    ]
    for found in await evidence_service.resolve_many(missing):
        if found.session_id == record.get("session_id"):
            evidence[found.evidence_id] = _evidence_record(found)


async def _reports(events: list[Any]) -> list[dict[str, Any]]:
    """The reports a turn put on screen, with their text and what they cost."""
    out = []
    for event in events:
        if event.type != EventType.ARTIFACT_CREATED:
            continue
        run_id = str(event.payload.get("run_id") or "")
        row = await db.fetch_one(
            """SELECT r.mode, r.status, r.tokens, o.markdown
                 FROM runs r LEFT JOIN run_outputs o ON o.run_id = r.run_id
                WHERE r.run_id = ?""",
            (run_id,),
        )
        row = row or {}
        out.append({
            "run_id": run_id,
            "mode": row.get("mode") or event.payload.get("mode", ""),
            "status": row.get("status", ""),
            "reused": bool(event.payload.get("reused")),
            "tokens": int(row.get("tokens") or 0),
            "markdown": row.get("markdown") or "",
        })
    return out


async def _turn(
    session: Any, *, owner_id: str, model: str, question: str, mode: str
) -> dict[str, Any]:
    """One message and its answer, as `post_message` would run it."""
    run, _ = await repo.create_run(
        session_id=session.session_id,
        owner_id=owner_id,
        llm_model=model,
        prompt_version=current_prompt_version(),
        budget=RunBudget.from_settings().as_dict(),
    )
    await repo.append_message(
        session_id=session.session_id,
        role=MessageRole.USER,
        content=question,
        run_id=run.run_id,
    )
    before = await repo.last_event_seq(session.session_id)
    started = time.perf_counter()
    await agent_runner._execute_turn(session=session, run=run, question=question, mode=mode)
    elapsed = round(time.perf_counter() - started, 1)

    events = await repo.list_events(session.session_id, after_seq=before, limit=5000)
    finished = await repo.get_run(run.run_id, owner_id=owner_id)
    answer = next(
        (
            m for m in reversed(await repo.list_messages(session.session_id))
            if m.run_id == run.run_id and m.role == MessageRole.ASSISTANT
        ),
        None,
    )
    return {
        "run_id": run.run_id,
        "status": finished.status.value,
        "completion": answer.completion if answer else "",
        "stop_reason": answer.stop_reason if answer else "",
        # Finished on its own with a whole answer; a wrap-up forced by a budget
        # is `done` + `complete` with a stop_reason, and is not counted as that.
        "completed": bool(
            answer
            and finished.status == RunStatus.DONE
            and answer.completion == "complete"
            and not answer.stop_reason
        ),
        "error_code": finished.error_code,
        "elapsed_seconds": elapsed,
        "usage": finished.usage,
        "tool_calls": [
            {"tool": e.payload.get("tool", ""), "input": e.payload.get("input", {})}
            for e in events
            if e.type == EventType.TOOL_STARTED
        ],
        "tool_failures": sum(1 for e in events if e.type == EventType.TOOL_FAILED),
        "reports": await _reports(events),
        "answer": answer.content if answer else "",
        "citations": answer.citations if answer else [],
    }


async def run_case(
    dataset: Dataset, case: EvalCase, *, owner_id: str, model: str, attempt: int = 0
) -> dict[str, Any]:
    """Execute one attempt of a case and return its trajectory record."""
    paper_ids = [dataset.papers[key].paper_id for key in case.papers]
    forgotten = await forget_mode_reports(paper_ids) if case.runs_modes else 0

    session = await repo.create_session(
        owner_id=owner_id,
        title=f"eval:{dataset.name}:{case.id}#{attempt}",
        language=case.language,
        llm_model=model,
    )
    for paper_id in paper_ids:
        await repo.attach_session_paper(
            session_id=session.session_id,
            literature_id=await paper_catalog.ensure_literature_for_local_paper(paper_id),
            paper_id=paper_id,
            availability=Availability.PARSED,
            added_by="user",
        )

    setup = []
    for turn in case.setup:
        done = await _turn(
            session, owner_id=owner_id, model=model, question=turn.question, mode=turn.mode
        )
        setup.append({
            "question": turn.question,
            "mode": turn.mode,
            **{k: done[k] for k in ("run_id", "status", "completed", "elapsed_seconds", "usage")},
            "tools": [call["tool"] for call in done["tool_calls"]],
            "reports": [{k: v for k, v in r.items() if k != "markdown"} for r in done["reports"]],
        })

    scored = await _turn(
        session, owner_id=owner_id, model=model, question=case.question, mode=case.mode
    )
    text = scored["answer"]
    # Every id written in the text, kept or not: one the runner dropped as
    # never returned by a tool resolves to nothing below and scores as invented.
    written = list(dict.fromkeys(CITATION_RE.findall(normalise_citations(text))))
    evidence = await evidence_service.resolve_many(written, owner_id=owner_id)

    return {
        "session_id": session.session_id,
        "attempt": attempt,
        "mode": case.mode,
        "reports_forgotten": forgotten,
        "setup": setup,
        # The scored turn only: a follow-up case measures the follow-up.
        **scored,
        # `[ev_a, ev_b]`: counted as citations by the runner, shown raw by the page.
        "grouped_citation_markers": len(GROUPED_RE.findall(text)),
        "evidence": {e.evidence_id: _evidence_record(e) for e in evidence},
    }
