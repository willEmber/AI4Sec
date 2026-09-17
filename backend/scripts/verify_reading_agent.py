"""P2 exit criteria, checked against a real model and a real parsed paper.

The plan's bar for P2 is not "the tools work" — a direct tool call proves only
that an interface exists. It is that the *model*, given a goal and no
prescribed order, goes and reads what it needs:

* three consecutive follow-ups, where later turns read material the earlier
  ones did not, rather than recycling the first retrieval;
* equations and tables located in the paper, not paraphrased;
* citations that resolve back to a page.

Talks to the live gateway and spends tokens. Run it explicitly::

    cd backend
    uv run python -m scripts.verify_reading_agent
    uv run python -m scripts.verify_reading_agent --paper-id <sha1> --model qwen3.7-plus

Records land in `data/reading_agent_report.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db import agent_repository as repo  # noqa: E402
from app.db import database as db  # noqa: E402
from app.db.database import init_db, set_db_path  # noqa: E402
from app.models.agent_models import Availability, EventType, RunStatus  # noqa: E402
from app.services import agent_runner, evidence_service, identity, paper_catalog  # noqa: E402

# Three turns that deliberately need different parts of a paper. The first is
# broad, the second forces a numeric/tabular lookup, the third an equation —
# so a run that answers all three from one retrieval is visibly wrong.
TURNS = [
    "这篇论文要解决什么问题，提出的方法核心是什么？",
    "它在实验里用了哪些数据集和指标，主要结果的具体数值是多少？",
    "论文里的关键公式是什么？请写出公式并解释每个符号的含义。",
]


async def _pick_paper(explicit: str) -> tuple[str, str]:
    """Choose a parsed paper: the one asked for, else the richest available."""
    if explicit:
        row = await db.fetch_one(
            "SELECT paper_id, title FROM papers WHERE paper_id = ?", (explicit,)
        )
        if row is None:
            raise SystemExit(f"No such paper: {explicit}")
        return row["paper_id"], row["title"]

    rows = await db.fetch_all(
        """SELECT p.paper_id, p.title, COUNT(n.node_id) AS nodes,
                  SUM(CASE WHEN n.block_type = 'equation' THEN 1 ELSE 0 END) AS eqs,
                  SUM(CASE WHEN n.block_type = 'table' THEN 1 ELSE 0 END) AS tbls
             FROM papers p JOIN paper_nodes n ON n.paper_id = p.paper_id
            GROUP BY p.paper_id
           HAVING nodes > 20 AND eqs > 0 AND tbls > 0
            ORDER BY eqs DESC LIMIT 1""",
        (),
    )
    if not rows:
        raise SystemExit(
            "No parsed paper with equations and tables in this database. "
            "Upload and analyse one first, or pass --paper-id."
        )
    return rows[0]["paper_id"], rows[0]["title"]


async def _run_turn(session, question: str, model: str) -> dict[str, Any]:
    """Execute one turn and summarise what the model actually did."""
    run, _ = await repo.create_run(
        session_id=session.session_id, owner_id=session.owner_id, llm_model=model
    )
    before = await repo.last_event_seq(session.session_id)
    t0 = time.perf_counter()
    await agent_runner._execute_turn(session=session, run=run, question=question)
    elapsed = round(time.perf_counter() - t0, 1)

    events = await repo.list_events(session.session_id, after_seq=before)
    tool_calls = [
        e.payload.get("tool", "") for e in events if e.type == EventType.TOOL_STARTED
    ]
    sections_read = [
        e.payload.get("summary", {}).get("section", "")
        for e in events
        if e.type == EventType.TOOL_COMPLETED and e.payload.get("summary", {}).get("section")
    ]
    finished = await repo.get_run(run.run_id, owner_id=session.owner_id)
    messages = await repo.list_messages(session.session_id)
    answer = next(
        (m for m in reversed(messages) if m.run_id == run.run_id and m.role.value == "assistant"),
        None,
    )

    return {
        "question": question,
        "run_id": run.run_id,
        "status": finished.status.value,
        "elapsed_seconds": elapsed,
        "tool_calls": tool_calls,
        "sections_read": [s for s in sections_read if s],
        "answer": (answer.content if answer else "")[:1200],
        "citations": answer.citations if answer else [],
        "usage": finished.usage,
        "error": finished.error_msg,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-id", default="", help="Parsed paper to read")
    parser.add_argument("--model", default="", help="Model id; defaults to the configured one")
    parser.add_argument("--report", default="", help="Where to write the JSON record")
    args = parser.parse_args()

    settings = get_settings()
    set_db_path(settings.data_dir / "app.db")
    await init_db()
    await agent_runner.open_agent_checkpointer()

    paper_id, title = await _pick_paper(args.paper_id)
    print(f"paper:  {title[:70]} ({paper_id})")
    print(f"model:  {args.model or settings.default_thinking_model}\n")

    principal_id, _ = await identity.create_principal(label="reading-agent-verification")
    session = await repo.create_session(
        owner_id=principal_id, language="zh", llm_model=args.model
    )
    literature_id = await paper_catalog.ensure_literature_for_local_paper(paper_id)
    await repo.attach_session_paper(
        session_id=session.session_id,
        literature_id=literature_id,
        paper_id=paper_id,
        availability=Availability.PARSED,
        added_by="user",
    )

    turns: list[dict[str, Any]] = []
    for index, question in enumerate(TURNS, start=1):
        print(f"→ turn {index}: {question}")
        result = await _run_turn(session, question, args.model)
        turns.append(result)
        print(f"  {result['status']} in {result['elapsed_seconds']}s — "
              f"tools={result['tool_calls']} citations={len(result['citations'])}")
        print(f"  {result['answer'][:200]}...\n")

    # Did later turns read material the earlier ones had not? Recycling the
    # first retrieval across three questions is the failure this looks for.
    seen: set[str] = set()
    fresh_reads = []
    for turn in turns:
        new = [s for s in turn["sections_read"] if s and s not in seen]
        seen.update(turn["sections_read"])
        fresh_reads.append(new)

    all_citations = [c for turn in turns for c in turn["citations"]]
    checks = await evidence_service.validate_citations(
        all_citations, owner_id=principal_id
    )
    resolvable = sum(1 for c in checks if c.ok)

    # The equation turn must have produced evidence from an equation block,
    # rather than a paraphrase of one.
    equation_evidence = 0
    for evidence_id in turns[-1]["citations"]:
        try:
            evidence = await evidence_service.resolve(evidence_id, owner_id=principal_id)
        except Exception:  # noqa: BLE001
            continue
        if any(marker in evidence.quote for marker in ("$", "\\", "=")):
            equation_evidence += 1

    criteria = {
        "all_turns_completed": all(t["status"] == RunStatus.DONE.value for t in turns),
        "every_turn_used_tools": all(t["tool_calls"] for t in turns),
        "later_turns_read_new_material": any(bool(f) for f in fresh_reads[1:]),
        "answers_are_cited": all(t["citations"] for t in turns),
        "citations_resolve": bool(all_citations) and resolvable == len(all_citations),
        "equation_located_in_source": equation_evidence > 0,
    }

    record = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model or settings.default_thinking_model,
        "paper_id": paper_id,
        "paper_title": title,
        "session_id": session.session_id,
        "turns": turns,
        "new_sections_per_turn": fresh_reads,
        "citations_total": len(all_citations),
        "citations_resolvable": resolvable,
        "criteria": criteria,
        "all_passed": all(criteria.values()),
    }
    path = Path(args.report) if args.report else settings.data_dir / "reading_agent_report.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("─" * 60)
    for name, passed in criteria.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nreport: {path}")
    await agent_runner.close_agent_checkpointer()
    return 0 if record["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
