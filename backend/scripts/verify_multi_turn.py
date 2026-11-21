"""Multi-turn conversations against the live gateway, checked turn by turn.

`verify_reading_agent` asks three questions of one paper; this runs several
conversations that each lean on a different part of the tool surface — follow
ups that should be answered from what was already read, a two-paper
comparison, memory and a preference change, search → rank → download → read,
and a mode report with a follow-up on it. Each turn carries checks on what the
agent *did* (which tools, how many calls, whether it cited, whether it reused
what it had), not on its wording, which is the evals' job.

With `TRACING_OTLP_ENDPOINT` set (Phoenix running), every turn is also a trace,
and the record gains what only the trace shows: model calls per turn, prompt
tokens per call, how many tool calls each step issued together, and how large
each tool result was (`scripts.phoenix_traces`).

Talks to the live gateway, spends tokens; the acquisition scenario also
downloads and parses a paper. Run it explicitly::

    cd backend
    uv run python -m scripts.verify_multi_turn --model qwen3.8-flash
    uv run python -m scripts.verify_multi_turn --only followups,compare
    uv run python -m scripts.verify_multi_turn --list

Records land in `data/multi_turn_report.json`. Exit status 1 when a turn did
not finish or a hard check failed; soft checks (agent behaviour worth a look)
are reported but do not fail the run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.context import RunBudget  # noqa: E402
from app.agents.prompts import PROMPT_VERSION  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import agent_repository as repo  # noqa: E402
from app.db import database as db  # noqa: E402
from app.models.agent_models import Availability, EventType, MessageRole, RunStatus  # noqa: E402
from app.services import agent_runner, evidence_service, identity, paper_catalog  # noqa: E402

from scripts import phoenix_traces  # noqa: E402

# Papers the scenarios are written against, by title so a re-upload (new
# SHA-1, same paper) still matches. The richest parse of each title wins.
PAPERS = {
    "robin": "ROBIN: Robust and Invisible Watermarks",
    "gaussian_shading": "Gaussian Shading: Provable Performance-Lossless",
    "transformer": "Attention Is All You Need",
}


@dataclass
class Turn:
    question: str
    mode: str = "auto"
    # Hard: a turn that breaks one of these is a defect, not a style choice.
    must_cite: bool = False
    tools_any: tuple[str, ...] = ()       # at least one of these is called
    tools_none: tuple[str, ...] = ()      # none of these is called
    # Soft: behaviour worth flagging, which a stronger model may simply do better.
    max_tool_calls: int | None = None
    answer_matches: str = ""              # regex the answer should match
    english_answer: bool = False
    cites_papers: tuple[str, ...] = ()    # paper keys the citations must cover


@dataclass
class Scenario:
    name: str
    description: str
    papers: tuple[str, ...]
    turns: list[Turn]
    costly: bool = False                  # downloads / parses / mode reports
    meta: dict[str, Any] = field(default_factory=dict)


SCENARIOS = [
    Scenario(
        name="followups",
        description="One paper, five turns: broad → numbers → follow-up on the "
        "previous answer → a point the paper does not cover → an equation.",
        papers=("robin",),
        turns=[
            Turn("这篇论文要解决什么问题？方法的核心思路是什么？",
                 must_cite=True, tools_any=("search_paper_content", "read_paper_section", "get_paper_outline")),
            Turn("它在哪些攻击下测试了鲁棒性？给出主要结果表里的具体数值。",
                 must_cite=True, tools_any=("read_paper_section", "search_paper_content")),
            # Answerable from the table the previous turn read.
            Turn("你刚才给的结果里，哪种攻击下表现最差？",
                 max_tool_calls=1),
            Turn("论文有没有做视频扩散模型上的实验？",
                 answer_matches=r"没有|未|不涉及|没提|并未|未见|找不到|没有找到"),
            Turn("把这篇论文最核心的公式写出来，并解释每个符号。",
                 must_cite=True, tools_any=("read_paper_section", "search_paper_content")),
        ],
    ),
    Scenario(
        name="compare",
        description="Two papers: embedding mechanism, comparability of the "
        "robustness setups, image-quality numbers.",
        papers=("gaussian_shading", "robin"),
        turns=[
            Turn("比较这两篇论文嵌入水印的方式有什么本质不同。",
                 must_cite=True, cites_papers=("gaussian_shading", "robin")),
            Turn("它们的鲁棒性实验设置可以直接比较吗？分别说出数据集、攻击类型和指标。",
                 must_cite=True, cites_papers=("gaussian_shading", "robin")),
            Turn("哪一篇对生成图像质量的影响更小？有数字支撑吗？",
                 must_cite=True),
        ],
    ),
    Scenario(
        name="memory",
        description="Save a research topic and a language preference, answer "
        "under it, then replace the preference.",
        papers=("robin",),
        turns=[
            Turn("我在做扩散模型水印鲁棒性方面的研究，请记住这一点。",
                 tools_any=("save_memory",), max_tool_calls=1),
            Turn("以后都用英文回答我。",
                 tools_any=("save_memory",), max_tool_calls=1),
            Turn("这篇论文的主要局限是什么？",
                 must_cite=True, english_answer=True),
            Turn("我改主意了，还是用中文回答吧。",
                 tools_any=("save_memory", "forget_memory"), max_tool_calls=2),
        ],
    ),
    Scenario(
        name="acquisition",
        description="Search with a date range, ask about a venue, then fetch "
        "and read one of the results.",
        papers=(),
        costly=True,
        turns=[
            Turn("帮我找 2024 年以后发表的、关于扩散模型图像水印鲁棒性的论文，列 5 篇引用较多的。",
                 tools_any=("search_papers",)),
            Turn("第一篇发表在哪个会议或期刊？它的 CCF 等级是多少？",
                 tools_any=("query_publication_rank", "get_paper_metadata", "resolve_paper")),
            Turn("下载列表里第一篇能拿到全文的论文，读它的方法部分，概括核心思路。",
                 must_cite=True, tools_any=("download_paper",)),
        ],
    ),
    Scenario(
        name="snap",
        description="Insight Snap on request, then a follow-up the report answers.",
        papers=("transformer",),
        costly=True,
        turns=[
            Turn("快速评估一下这篇论文值不值得读。", tools_any=("begin_report",)),
            Turn("报告里提到的外部信号具体有哪些？引用数是多少？",
                 tools_none=("begin_report", "get_triage_signals"), max_tool_calls=2),
        ],
    ),
]


# ── Checks ──────────────────────────────────────────────────────────────────


def _english(text: str) -> bool:
    """Mostly Latin letters, ignoring citation markers, LaTeX and numbers."""
    body = re.sub(r"\[ev_[^\]]*\]|\$[^$]*\$|[\d\W_]+", "", text)
    if not body:
        return False
    han = len(re.findall(r"[一-鿿]", body))
    return han / len(body) < 0.1


def check_turn(turn: Turn, record: dict[str, Any], paper_ids: dict[str, str]) -> list[dict[str, Any]]:
    """The checks a turn's record passes or fails, each marked hard or soft."""
    tools = [call["tool"] for call in record["tool_calls"]]
    results: list[dict[str, Any]] = []

    def add(name: str, ok: bool, hard: bool, detail: str = "") -> None:
        results.append({"check": name, "ok": ok, "hard": hard, "detail": detail})

    add("finished", record["status"] == RunStatus.DONE.value and bool(record["answer"]), True,
        record.get("error_code") or "")
    if turn.must_cite:
        add("cited", bool(record["citations"]), True, f"{len(record['citations'])} citations")
    if turn.tools_any:
        add("tools_any", any(t in tools for t in turn.tools_any), True,
            f"wanted one of {list(turn.tools_any)}, called {tools}")
    if turn.tools_none:
        hit = [t for t in tools if t in turn.tools_none]
        add("tools_none", not hit, True, f"called {hit}" if hit else "")
    if turn.max_tool_calls is not None:
        add("max_tool_calls", len(tools) <= turn.max_tool_calls, False,
            f"{len(tools)} calls (≤{turn.max_tool_calls} expected): {tools}")
    if turn.answer_matches:
        add("answer_matches", bool(re.search(turn.answer_matches, record["answer"])), False,
            turn.answer_matches)
    if turn.english_answer:
        add("english_answer", _english(record["answer"]), False)
    if turn.cites_papers:
        cited = {e["paper_id"] for e in record["evidence"].values()}
        missing = [k for k in turn.cites_papers if paper_ids.get(k) not in cited]
        add("cites_papers", not missing, False, f"no citation from {missing}" if missing else "")
    # The model cites by alias (`[E12]`); what is stored has been rewritten to
    # stored ids, so an alias left in it, or a grouped marker, is a rewrite miss.
    residue = record.get("alias_residue", 0)
    if record["grouped_citation_markers"] or residue:
        add("citation_format", False, False,
            f"{record['grouped_citation_markers']} grouped markers, "
            f"{residue} alias markers left in the stored answer")
    dropped = record.get("citations_dropped", 0)
    if record["invented_citations"] or dropped:
        add("no_invented_citations", False, False,
            f"{len(record['invented_citations'])} stored ids no tool returned, "
            f"{dropped} aliases no tool handed out")
    return results


# ── Running ─────────────────────────────────────────────────────────────────

CITATION_RE = re.compile(r"ev_[A-Za-z0-9_-]{6,}")
GROUPED_RE = re.compile(r"\[ev_[^\]]*,\s*ev_[^\]]*\]")
ALIAS_RESIDUE_RE = re.compile(r"[\[【]\s*[Ee]\d{1,5}\s*[,，;；、\]】]")


async def _pick_papers() -> dict[str, str]:
    """`paper key -> paper_id`, the parse with the most nodes per title."""
    found: dict[str, str] = {}
    for key, title in PAPERS.items():
        row = await db.fetch_one(
            """SELECT p.paper_id FROM papers p JOIN paper_nodes n ON n.paper_id = p.paper_id
                WHERE p.title LIKE ? GROUP BY p.paper_id ORDER BY COUNT(*) DESC LIMIT 1""",
            (title + "%",),
        )
        if row:
            found[key] = row["paper_id"]
    return found


async def _run_turn(session, turn: Turn, model: str) -> dict[str, Any]:
    run, _ = await repo.create_run(
        session_id=session.session_id,
        owner_id=session.owner_id,
        llm_model=model,
        prompt_version=PROMPT_VERSION,
        budget=RunBudget.from_settings().as_dict(),
    )
    await repo.append_message(
        session_id=session.session_id, role=MessageRole.USER,
        content=turn.question, run_id=run.run_id,
    )
    before = await repo.last_event_seq(session.session_id)
    started_at = time.time()
    t0 = time.perf_counter()
    await agent_runner._execute_turn(session=session, run=run, question=turn.question, mode=turn.mode)
    elapsed = round(time.perf_counter() - t0, 1)

    events = await repo.list_events(session.session_id, after_seq=before, limit=5000)
    finished = await repo.get_run(run.run_id, owner_id=session.owner_id)
    answer = next(
        (m for m in reversed(await repo.list_messages(session.session_id))
         if m.run_id == run.run_id and m.role == MessageRole.ASSISTANT),
        None,
    )
    text = answer.content if answer else ""
    written = list(dict.fromkeys(CITATION_RE.findall(text)))
    resolved = await evidence_service.resolve_many(written, owner_id=session.owner_id)
    evidence = {
        e.evidence_id: {"paper_id": e.paper_id, "level": e.source_level.value,
                        "page": e.locator.page_label, "section": e.locator.section_path}
        for e in resolved
    }
    completions = [e for e in events if e.type in (EventType.TOOL_COMPLETED, EventType.TOOL_FAILED)]
    return {
        "question": turn.question,
        "run_id": run.run_id,
        "status": finished.status.value,
        "error_code": finished.error_code,
        "started_at": started_at,
        "elapsed_seconds": elapsed,
        "usage": finished.usage,
        "tool_calls": [
            {"tool": e.payload.get("tool", ""), "input": e.payload.get("input", {})}
            for e in events if e.type == EventType.TOOL_STARTED
        ],
        "tool_results": [
            {k: e.payload.get(k) for k in ("tool", "status", "evidence_count", "note", "error", "summary")
             if e.payload.get(k) not in (None, "", {})}
            for e in completions
        ],
        "answer": text,
        "citations": answer.citations if answer else [],
        "grouped_citation_markers": len(GROUPED_RE.findall(text)),
        "invented_citations": [i for i in written if i not in evidence],
        "alias_residue": len(ALIAS_RESIDUE_RE.findall(text)),
        "citations_dropped": int((finished.usage or {}).get("citations_dropped") or 0),
        "citation_groups_expanded": int((finished.usage or {}).get("citation_groups_expanded") or 0),
        "evidence": evidence,
    }


async def run_scenario(
    scenario: Scenario, *, owner_id: str, model: str, paper_ids: dict[str, str]
) -> dict[str, Any]:
    session = await repo.create_session(
        owner_id=owner_id, title=f"multi-turn:{scenario.name}", language="zh", llm_model=model
    )
    for key in scenario.papers:
        await repo.attach_session_paper(
            session_id=session.session_id,
            literature_id=await paper_catalog.ensure_literature_for_local_paper(paper_ids[key]),
            paper_id=paper_ids[key],
            availability=Availability.PARSED,
            added_by="user",
        )

    turns: list[dict[str, Any]] = []
    for index, turn in enumerate(scenario.turns, start=1):
        print(f"  → {index}. {turn.question}")
        record = await _run_turn(session, turn, model)
        record["checks"] = check_turn(turn, record, paper_ids)
        turns.append(record)
        failed = [c["check"] for c in record["checks"] if not c["ok"]]
        print(f"     {record['status']} {record['elapsed_seconds']}s  "
              f"llm={record['usage'].get('llm_calls')} tools={[c['tool'] for c in record['tool_calls']]} "
              f"cites={len(record['citations'])}" + (f"  ✗ {failed}" if failed else ""))

    # Repeated identical calls across the conversation: what a follow-up
    # should have taken from context instead of fetching again.
    seen: dict[str, int] = {}
    repeats = []
    for index, record in enumerate(turns, start=1):
        for call in record["tool_calls"]:
            key = json.dumps(call, ensure_ascii=False, sort_keys=True)
            if key in seen:
                repeats.append({"turn": index, "first_turn": seen[key], **call})
            else:
                seen[key] = index

    memories = await repo.list_memories(owner_id) if scenario.name == "memory" else []
    return {
        "name": scenario.name,
        "description": scenario.description,
        "session_id": session.session_id,
        "thread_id": session.thread_id or session.session_id,
        "turns": turns,
        "repeated_calls": repeats,
        "memories_after": [
            {"kind": m.kind.value, "content": m.content, "pending": m.pending} for m in memories
        ],
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="", help="Model id; defaults to the configured one")
    parser.add_argument("--only", default="", help="Comma-separated scenario names")
    parser.add_argument("--skip-costly", action="store_true", help="Skip downloads, parses and reports")
    parser.add_argument("--list", action="store_true", help="List scenarios and exit")
    parser.add_argument("--report", default="", help="Where to write the JSON record")
    args = parser.parse_args()

    if args.list:
        for s in SCENARIOS:
            print(f"{s.name:12s} {len(s.turns)} turns{' (costly)' if s.costly else ''}  {s.description}")
        return 0

    chosen = [s for s in SCENARIOS if not args.only or s.name in args.only.split(",")]
    if args.skip_costly:
        chosen = [s for s in chosen if not s.costly]

    settings = get_settings()
    model = args.model or settings.default_thinking_model
    tracing = bool(settings.tracing_otlp_endpoint.strip())
    db.configure(settings.database_url)
    await db.init_db()
    await agent_runner.open_agent_checkpointer()
    if tracing:
        from app.services import tracing as tracing_service

        tracing_service.start_tracing()

    paper_ids = await _pick_papers()
    runnable = [s for s in chosen if all(k in paper_ids for k in s.papers)]
    for s in chosen:
        if s not in runnable:
            print(f"skip {s.name}: papers not parsed locally ({[k for k in s.papers if k not in paper_ids]})")

    print(f"model: {model}   tracing: {'on' if tracing else 'off'}\n")
    principal_id, _ = await identity.create_principal(label="multi-turn-verification")
    scenarios = []
    for scenario in runnable:
        print(f"[{scenario.name}] {scenario.description}")
        scenarios.append(await run_scenario(scenario, owner_id=principal_id, model=model, paper_ids=paper_ids))
        print()

    if tracing:
        from app.services import tracing as tracing_service

        tracing_service.stop_tracing()   # flushes the batch processor
        base = settings.tracing_ui_url.strip() or settings.tracing_otlp_endpoint.split("/v1/")[0]
        for scenario in scenarios:
            try:
                spans = await phoenix_traces.fetch_session_spans(
                    base, settings.tracing_project or "scholar", scenario["thread_id"]
                )
            except Exception as exc:  # noqa: BLE001
                scenario["trace"] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            scenario["trace"] = phoenix_traces.analyse(spans)

    hard_failures = [
        (s["name"], i, c["check"])
        for s in scenarios for i, t in enumerate(s["turns"], start=1)
        for c in t["checks"] if c["hard"] and not c["ok"]
    ]
    soft_failures = [
        (s["name"], i, c["check"], c["detail"])
        for s in scenarios for i, t in enumerate(s["turns"], start=1)
        for c in t["checks"] if not c["hard"] and not c["ok"]
    ]
    record = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "paper_ids": paper_ids,
        "scenarios": scenarios,
        "hard_failures": hard_failures,
        "soft_failures": soft_failures,
    }
    out = Path(args.report) if args.report else Path(settings.data_dir) / "multi_turn_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print("── summary ──")
    for s in scenarios:
        print(phoenix_traces.format_summary(s))
    print(f"\nhard failures: {len(hard_failures)}  soft: {len(soft_failures)}")
    for failure in hard_failures:
        print("  ✗", *failure)
    for failure in soft_failures:
        print("  ·", *failure)
    print(f"\nrecord: {out}")
    await agent_runner.close_agent_checkpointer()
    return 1 if hard_failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
