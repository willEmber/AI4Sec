"""Run a case through the agent and record what it did.

A case takes the path a reader's message takes — session, user message, run
with the configured budget, `_execute_turn` — so what is measured is the
product, not a harness beside it. Each case gets its own session: cases are
independent, and a shared conversation would let one answer feed the next.
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.context import RunBudget
from app.agents.prompts import PROMPT_VERSION
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


async def run_case(
    dataset: Dataset, case: EvalCase, *, owner_id: str, model: str
) -> dict[str, Any]:
    """Execute one case and return its trajectory record."""
    session = await repo.create_session(
        owner_id=owner_id,
        title=f"eval:{dataset.name}:{case.id}",
        language=case.language,
        llm_model=model,
    )
    for key in case.papers:
        paper_id = dataset.papers[key].paper_id
        await repo.attach_session_paper(
            session_id=session.session_id,
            literature_id=await paper_catalog.ensure_literature_for_local_paper(paper_id),
            paper_id=paper_id,
            availability=Availability.PARSED,
            added_by="user",
        )

    run, _ = await repo.create_run(
        session_id=session.session_id,
        owner_id=owner_id,
        llm_model=model,
        prompt_version=PROMPT_VERSION,
        budget=RunBudget.from_settings().as_dict(),
    )
    await repo.append_message(
        session_id=session.session_id,
        role=MessageRole.USER,
        content=case.question,
        run_id=run.run_id,
    )
    before = await repo.last_event_seq(session.session_id)
    started = time.perf_counter()
    await agent_runner._execute_turn(session=session, run=run, question=case.question)
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
    text = answer.content if answer else ""
    # Every id written in the text, kept or not: one the runner dropped as
    # never returned by a tool resolves to nothing below and scores as invented.
    written = list(dict.fromkeys(CITATION_RE.findall(normalise_citations(text))))
    evidence = await evidence_service.resolve_many(written, owner_id=owner_id)

    return {
        "session_id": session.session_id,
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
        "answer": text,
        # `[ev_a, ev_b]`: counted as citations by the runner, shown raw by the page.
        "grouped_citation_markers": len(GROUPED_RE.findall(text)),
        "citations": answer.citations if answer else [],
        "evidence": {e.evidence_id: _evidence_record(e) for e in evidence},
    }
