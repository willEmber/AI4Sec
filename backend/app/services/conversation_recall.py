"""Recall earlier conversations: what a reader asked, what was answered, and on what evidence.

Used by the agent's `recall_conversations` tool and by the conversation
search in the UI, so the two can never disagree about what matches (P8).

The unit returned is a *turn* — one question and its answer — rather than a
message, because a matching question is useless without its answer and a
matching answer is ambiguous without its question. Each turn carries the
evidence its answer cited, hydrated from the evidence table: those snapshots
are what the earlier answer actually rested on, pinned to the parse version
read at the time, and they are what a new answer may cite. The earlier answer's
own wording is a lead, not evidence.

Ownership is the caller's `owner_id` throughout: the message search is scoped
to that principal's sessions, and evidence is re-checked against it, so an id
that somehow appears in a message does not hand over someone else's quote.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.db import agent_repository as repo
from app.models.agent_models import AgentMessage, MessageRole

QUESTION_CHARS = 300
ANSWER_CHARS = 800
QUOTE_CHARS = 300
EVIDENCE_PER_TURN = 6
# Messages fetched per turn requested: a turn can match on both its question
# and its answer, and several turns of one session often match together.
_OVERFETCH = 8


@dataclass
class RecalledEvidence:
    evidence_id: str
    source_level: str
    paper_title: str
    page: int | None
    section: str
    quote: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_level": self.source_level,
            "paper_title": self.paper_title,
            "page": self.page,
            "section": self.section,
            "quote": self.quote,
        }


@dataclass
class RecalledTurn:
    session_id: str
    session_title: str
    project_id: str
    run_id: str
    asked_at: str
    question: str
    answer_excerpt: str
    rank: float
    evidence: list[RecalledEvidence] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_title": self.session_title,
            "project_id": self.project_id,
            "run_id": self.run_id,
            "asked_at": self.asked_at,
            "question": self.question,
            "answer_excerpt": self.answer_excerpt,
            "evidence": [e.as_dict() for e in self.evidence],
        }


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def excerpt(text: str, terms: list[str], limit: int = ANSWER_CHARS) -> str:
    """The `limit`-character window of `text` that holds the most distinct query terms.

    Terms are compared against the lower-cased text, which is how the index
    produced them (Han bigrams are unaffected by lower-casing).
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    lowered = text.lower()
    lead = limit // 5
    best_start, best_score = 0, -1
    starts = sorted(
        {max(0, i - lead) for t in set(terms) if t for i in _occurrences(lowered, t)}
    )
    for start in starts or [0]:
        window = lowered[start : start + limit]
        score = sum(1 for t in set(terms) if t and t in window)
        if score > best_score:
            best_start, best_score = start, score
    end = min(len(text), best_start + limit)
    start = max(0, end - limit)
    body = text[start:end].strip()
    return ("…" if start > 0 else "") + body + ("…" if end < len(text) else "")


def _occurrences(text: str, term: str, cap: int = 20) -> list[int]:
    found: list[int] = []
    i = text.find(term)
    while i >= 0 and len(found) < cap:
        found.append(i)
        i = text.find(term, i + 1)
    return found


async def recall(
    owner_id: str,
    query: str,
    *,
    project_id: str | None = None,
    exclude_session_id: str = "",
    limit: int = 5,
) -> list[RecalledTurn]:
    """Earlier turns of `owner_id` matching `query`, best first.

    `project_id` narrows to one project (`None` = every session the owner
    has); `exclude_session_id` leaves out the session that is asking.
    """
    limit = max(1, min(int(limit), 10))
    terms = await repo.search_terms(query)
    if not terms:
        return []
    matches = await repo.search_messages(
        owner_id,
        query,
        project_id=project_id,
        exclude_session_id=exclude_session_id,
        limit=limit * _OVERFETCH,
    )

    # Collapse to turns, keeping the order of each turn's best-ranked message.
    order: list[tuple[str, str]] = []
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    lone: dict[tuple[str, str], AgentMessage] = {}
    for message, info in matches:
        key = (message.session_id, message.run_id or message.message_id)
        if key in meta:
            continue
        order.append(key)
        meta[key] = info
        if not message.run_id:
            lone[key] = message
        if len(order) >= limit:
            break

    by_turn: dict[tuple[str, str], list[AgentMessage]] = {}
    for message in await repo.list_turn_messages([k for k in order if k not in lone]):
        by_turn.setdefault((message.session_id, message.run_id), []).append(message)
    for key, message in lone.items():
        by_turn[key] = [message]

    turns: list[RecalledTurn] = []
    for key in order:
        messages = by_turn.get(key) or []
        question = next((m for m in messages if m.role == MessageRole.USER), None)
        answer = next((m for m in reversed(messages) if m.role == MessageRole.ASSISTANT), None)
        if question is None and answer is None:
            continue
        info = meta[key]
        turns.append(
            RecalledTurn(
                session_id=key[0],
                session_title=info["session_title"],
                project_id=info["project_id"],
                run_id=(answer or question).run_id,  # type: ignore[union-attr]
                asked_at=(question or answer).created_at,  # type: ignore[union-attr]
                question=_clip(question.content if question else "", QUESTION_CHARS),
                answer_excerpt=excerpt(answer.content if answer else "", terms),
                rank=info["rank"],
                evidence=[],
            )
        )
        if answer is not None:
            turns[-1].evidence = [
                RecalledEvidence(evidence_id=eid, source_level="", paper_title="",
                                 page=None, section="", quote="")
                for eid in list(dict.fromkeys(answer.citations))[:EVIDENCE_PER_TURN]
            ]

    await _hydrate_evidence(turns, owner_id)
    return turns


async def _hydrate_evidence(turns: list[RecalledTurn], owner_id: str) -> None:
    """Replace each turn's cited ids with the owner's evidence rows; drop the rest."""
    wanted = [e.evidence_id for t in turns for e in t.evidence]
    if not wanted:
        return
    found = {
        ev.evidence_id: ev
        for ev in await repo.get_evidence_many(list(dict.fromkeys(wanted)), owner_id=owner_id)
    }
    titles = await repo.literature_titles([ev.literature_id for ev in found.values()])
    for turn in turns:
        hydrated: list[RecalledEvidence] = []
        for stub in turn.evidence:
            ev = found.get(stub.evidence_id)
            if ev is None:
                continue
            hydrated.append(
                RecalledEvidence(
                    evidence_id=ev.evidence_id,
                    source_level=ev.source_level.value,
                    paper_title=titles.get(ev.literature_id, ""),
                    page=ev.locator.page_label,
                    section=ev.locator.section_path,
                    quote=_clip(ev.quote, QUOTE_CHARS),
                )
            )
        turn.evidence = hydrated
