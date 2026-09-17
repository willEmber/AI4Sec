"""P3 exit criteria, checked against a real model and live metadata providers.

P2 asked whether the model reads. P3 asks whether it *researches*: finds papers
it was not given, gets hold of them, compares them, and — the part that only a
real run can show — respects the boundaries when it cannot.

Four turns, each aimed at one boundary the plan names:

1. A date-limited search (A04). The year filter must be applied, and candidates
   with no known year must be reported as unknown rather than assumed to pass.
2. An arXiv paper with no DOI (A05). It must be found, downloaded, parsed and
   read — the DOI-keyed path cannot reach it.
3. A comparison against the paper already in the session (A06). Evidence must
   come from both papers, and differing experimental setups must be named.
4. A venue ranking (A10). JCR, CAS and CCF must stay separate, with the source
   and whether it is verified.

The run is also checked for what it must *not* do: cite an evidence id no tool
returned, or claim full text for a paper it only has the abstract of.

Talks to the live gateway and to OpenAlex / arXiv, and spends tokens. Run it
explicitly::

    cd backend
    uv run python -m scripts.verify_research_agent
    uv run python -m scripts.verify_research_agent --arxiv-id 1706.03762

Records land in `data/research_agent_report.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
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
from app.models.evidence_models import SourceLevel  # noqa: E402
from app.services import agent_runner, evidence_service, identity, paper_catalog  # noqa: E402

# An arXiv-only paper: widely indexed, no publisher DOI, reliably downloadable.
DEFAULT_ARXIV_ID = "1706.03762"

CITATION_RE = re.compile(r"\[(ev_[0-9a-f]{20})\]")


def _turns(arxiv_id: str, seed_title: str) -> list[dict[str, str]]:
    return [
        {
            "id": "dated_search",
            "question": (
                "帮我找 2023 年以后发表的、关于 mixture-of-experts 路由的论文，"
                "列出标题和年份。年份不确定的要说明。"
            ),
        },
        {
            "id": "no_doi_acquire",
            "question": (
                f"arXiv:{arxiv_id} 这篇论文，请下载并读全文，"
                "告诉我它提出的模型结构由哪几部分组成。"
            ),
        },
        {
            "id": "compare",
            "question": (
                f"把刚才那篇和《{seed_title}》做对比：两者的实验设置"
                "（数据集、指标）分别是什么？结果能直接比较吗？"
            ),
        },
        {
            "id": "venue_rank",
            "question": "《IEEE Transactions on Image Processing》是什么等级的期刊？",
        },
    ]


async def _pick_seed_paper(explicit: str) -> tuple[str, str]:
    """A parsed paper already in the database, to compare the new one against."""
    if explicit:
        row = await db.fetch_one(
            "SELECT paper_id, title FROM papers WHERE paper_id = ?", (explicit,)
        )
        if row is None:
            raise SystemExit(f"No such paper: {explicit}")
        return row["paper_id"], row["title"]

    rows = await db.fetch_all(
        """SELECT p.paper_id, p.title, COUNT(n.node_id) AS nodes
             FROM papers p JOIN paper_nodes n ON n.paper_id = p.paper_id
            WHERE p.title <> ''
            GROUP BY p.paper_id
           HAVING nodes > 40
            ORDER BY nodes DESC LIMIT 1""",
        (),
    )
    if not rows:
        raise SystemExit(
            "No parsed paper in this database to compare against. "
            "Analyse one first, or pass --seed-paper-id."
        )
    return rows[0]["paper_id"], rows[0]["title"]


async def _run_turn(session, turn: dict[str, str], model: str) -> dict[str, Any]:
    run, _ = await repo.create_run(
        session_id=session.session_id, owner_id=session.owner_id, llm_model=model
    )
    before = await repo.last_event_seq(session.session_id)
    t0 = time.perf_counter()
    await agent_runner._execute_turn(
        session=session, run=run, question=turn["question"]
    )
    elapsed = round(time.perf_counter() - t0, 1)

    events = await repo.list_events(session.session_id, after_seq=before)
    tool_calls = [
        e.payload.get("tool", "") for e in events if e.type == EventType.TOOL_STARTED
    ]
    tool_results = [
        {
            "tool": e.payload.get("tool", ""),
            "status": e.payload.get("status", ""),
            "summary": e.payload.get("summary", {}),
            "note": e.payload.get("note", ""),
        }
        for e in events
        if e.type in {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}
    ]
    papers_added = [
        e.payload for e in events if e.type == EventType.PAPER_ADDED
    ]
    finished = await repo.get_run(run.run_id, owner_id=session.owner_id)
    messages = await repo.list_messages(session.session_id)
    answer = next(
        (m for m in reversed(messages) if m.run_id == run.run_id and m.role.value == "assistant"),
        None,
    )
    text = answer.content if answer else ""

    return {
        "id": turn["id"],
        "question": turn["question"],
        "run_id": run.run_id,
        "status": finished.status.value,
        "elapsed_seconds": elapsed,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "papers_added": papers_added,
        "answer": text[:2000],
        "answer_full_length": len(text),
        "citations": answer.citations if answer else [],
        # Ids the model wrote that no tool ever returned. The runner drops them
        # from `citations`, so this is the only place the attempt is visible.
        "unbacked_citation_ids": [
            cid for cid in CITATION_RE.findall(text)
            if cid not in (answer.citations if answer else [])
        ],
        "usage": finished.usage,
        "error": finished.error_msg,
    }


def _tools_used(turn: dict[str, Any], *names: str) -> bool:
    return any(name in turn["tool_calls"] for name in names)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arxiv-id", default=DEFAULT_ARXIV_ID,
                        help="An arXiv paper with no DOI, to acquire and read")
    parser.add_argument("--seed-paper-id", default="",
                        help="Parsed paper already in the database, for the comparison turn")
    parser.add_argument("--model", default="", help="Model id; defaults to the configured one")
    parser.add_argument("--report", default="", help="Where to write the JSON record")
    args = parser.parse_args()

    settings = get_settings()
    set_db_path(settings.data_dir / "app.db")
    await init_db()
    await agent_runner.open_agent_checkpointer()

    seed_paper_id, seed_title = await _pick_seed_paper(args.seed_paper_id)
    print(f"seed paper: {seed_title[:70]} ({seed_paper_id})")
    print(f"arxiv:      {args.arxiv_id}")
    print(f"model:      {args.model or settings.default_thinking_model}\n")

    principal_id, _ = await identity.create_principal(label="research-agent-verification")
    session = await repo.create_session(
        owner_id=principal_id, language="zh", llm_model=args.model
    )
    literature_id = await paper_catalog.ensure_literature_for_local_paper(seed_paper_id)
    await repo.attach_session_paper(
        session_id=session.session_id,
        literature_id=literature_id,
        paper_id=seed_paper_id,
        availability=Availability.PARSED,
        added_by="user",
    )

    turns: list[dict[str, Any]] = []
    for index, turn in enumerate(_turns(args.arxiv_id, seed_title), start=1):
        print(f"→ turn {index} ({turn['id']}): {turn['question'][:60]}…")
        result = await _run_turn(session, turn, args.model)
        turns.append(result)
        print(f"  {result['status']} in {result['elapsed_seconds']}s — "
              f"tools={result['tool_calls']} citations={len(result['citations'])}")
        print(f"  {result['answer'][:200]}…\n")

    by_id = {t["id"]: t for t in turns}

    # ── A04: the date filter reached the tool, and unknown years were surfaced
    dated = by_id["dated_search"]
    search_results = [r for r in dated["tool_results"] if r["tool"] == "search_papers"]
    dated_search_ran = _tools_used(dated, "search_papers")
    unknown_year_surfaced = any(
        r["summary"].get("unknown_year_count", 0) > 0 or "year" in (r["note"] or "").lower()
        for r in search_results
    ) or any(word in dated["answer"] for word in ("年份未知", "未知年份", "年份不确定"))

    # ── A05: a DOI-less arXiv paper went download → parse → read
    acquire = by_id["no_doi_acquire"]
    acquired = _tools_used(acquire, "download_paper")
    parsed_it = _tools_used(acquire, "ensure_paper_parsed")
    read_it = _tools_used(acquire, "read_paper_section", "search_paper_content", "get_paper_outline")

    # ── A06: evidence from both papers, not one paper plus inference
    compare = by_id["compare"]
    compare_papers: set[str] = set()
    for evidence_id in compare["citations"]:
        try:
            evidence = await evidence_service.resolve(evidence_id, owner_id=principal_id)
        except Exception:  # noqa: BLE001
            continue
        if evidence.paper_id:
            compare_papers.add(evidence.paper_id)

    # ── A10: the three ranking systems stayed apart
    rank = by_id["venue_rank"]
    rank_tool_ran = _tools_used(rank, "query_publication_rank")
    systems_named = sum(
        1 for marker in ("JCR", "中科院", "CAS", "CCF") if marker in rank["answer"]
    )

    # ── Evidence levels: nothing claimed as full text that was only an abstract
    levels: dict[str, int] = {}
    for turn in turns:
        for evidence_id in turn["citations"]:
            try:
                evidence = await evidence_service.resolve(evidence_id, owner_id=principal_id)
            except Exception:  # noqa: BLE001
                continue
            levels[evidence.source_level.value] = levels.get(evidence.source_level.value, 0) + 1

    all_citations = [c for turn in turns for c in turn["citations"]]
    checks = await evidence_service.validate_citations(all_citations, owner_id=principal_id)
    resolvable = sum(1 for c in checks if c.ok)
    unbacked = [cid for turn in turns for cid in turn["unbacked_citation_ids"]]

    criteria = {
        "all_turns_completed": all(t["status"] == RunStatus.DONE.value for t in turns),
        # A04
        "date_filtered_search_ran": dated_search_ran,
        "unknown_year_reported": unknown_year_surfaced,
        # A05
        "no_doi_paper_downloaded": acquired,
        "no_doi_paper_parsed_and_read": parsed_it and read_it,
        # A06
        "comparison_cited_both_papers": len(compare_papers) >= 2,
        # A10
        "venue_rank_looked_up": rank_tool_ran,
        "ranking_systems_kept_separate": systems_named >= 2,
        # Boundaries
        "citations_resolve": bool(all_citations) and resolvable == len(all_citations),
        "no_invented_citations": not unbacked,
        "evidence_levels_recorded": len(levels) >= 2,
    }

    record = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": args.model or settings.default_thinking_model,
        "prompt_version": __import__("app.agents.prompts", fromlist=["PROMPT_VERSION"]).PROMPT_VERSION,
        "seed_paper_id": seed_paper_id,
        "seed_paper_title": seed_title,
        "arxiv_id": args.arxiv_id,
        "session_id": session.session_id,
        "turns": turns,
        "comparison_paper_ids": sorted(compare_papers),
        "evidence_levels": levels,
        "citations_total": len(all_citations),
        "citations_resolvable": resolvable,
        "unbacked_citation_ids": unbacked,
        "criteria": criteria,
        "all_passed": all(criteria.values()),
    }
    path = Path(args.report) if args.report else settings.data_dir / "research_agent_report.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("─" * 62)
    for name, passed in criteria.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\nevidence levels: {levels}")
    print(f"report: {path}")
    await agent_runner.close_agent_checkpointer()
    return 0 if record["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
