"""The two halves of an Insight Snap the conversation cannot do by reading (P16 §4.1).

A Snap written as the answer reads the paper with the reading tools. What it
cannot do that way is look the paper up in the world and judge it blind:

* `get_triage_signals` — venue, CCF / JCR, citation counts per index, citation
  intents, open access, retraction, code links. Each is one sourced fact,
  recorded as `metadata` evidence, so the report cites it. Cached per paper in
  `paper_signals` for `SNAP_SIGNALS_TTL_HOURS`.
* `review_paper_blind` — a reviewer pass over the paper's triage context with
  nothing external in its prompt, then the three-tier verdict code combines
  from that review and the signals (`triage_verdict.synthesize_verdict`). The
  review is a judgment, not paper text, so it is not evidence. Cached as a
  `blind_review:` job, keyed with the focus like a mode report was.

Both hand their structured results to the open report (`ReportTurn.parts`), so
the scorecard renders what the tools found, not what the report retold.
Concurrent calls for one paper share one signal lookup.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

from langchain.tools import ToolRuntime, tool

from app.agents.context import AgentContext
from app.agents.tools.reading import _authorize
from app.config import get_settings
from app.models.agent_models import ErrorCode, ToolResult
from app.models.paper_ir import PaperIR
from app.models.snap_models import CitationIntents, TriageSignals
from app.services import agent_jobs, evidence_service, paper_catalog
from app.services.paper_ir_loader import PaperNotParsed, ensure_pub_rank, load_paper_ir
from app.services.snap_context import build_triage_context
from app.services.triage_verdict import ContentReview, synthesize_verdict

logger = logging.getLogger("scholar.agents.tools.triage")

# One signal lookup per paper at a time in this process: the two tools are
# called in one step, and each would otherwise query every index.
_signal_lookups: dict[str, asyncio.Task[TriageSignals]] = {}

TIER_LABELS = {
    "must_read": {"zh": "必读", "en": "Must read"},
    "selective": {"zh": "选读", "en": "Selective read"},
    "skip": {"zh": "可跳过", "en": "Can skip"},
}


def blind_review_key(*, paper_id: str, language: str, model: str, focus: str = "") -> str:
    key = f"blind_review:{paper_id}:{language}:{model or 'default'}"
    focus = " ".join(focus.split()).casefold()
    if focus:
        key += ":focus-" + hashlib.sha1(focus.encode("utf-8")).hexdigest()[:12]
    return key


async def _paper(ctx: AgentContext, paper_id: str) -> tuple[str, PaperIR | None, ToolResult | None]:
    paper_id, denial = await _authorize(ctx, paper_id)
    if denial is not None:
        return paper_id, None, denial
    try:
        return paper_id, await load_paper_ir(paper_id), None
    except PaperNotParsed:
        return paper_id, None, ToolResult.unavailable(
            ErrorCode.NOT_PARSED, f"Paper {paper_id} is not parsed; call ensure_paper_parsed."
        )


async def signals_for(paper_id: str, paper_ir: PaperIR) -> TriageSignals:
    """The paper's external signals, from the cache or one shared lookup."""
    from app.workflows.snap_subgraph import collect_signals_safe

    task = _signal_lookups.get(paper_id)
    if task is None or task.done():
        # Registered before anything is awaited, so a call arriving meanwhile joins it.

        async def _lookup() -> TriageSignals:
            try:
                pdf_path = get_settings().data_dir / "papers" / paper_id / "original.pdf"
                pub_rank = await ensure_pub_rank(
                    paper_id, paper_ir, pdf_path if pdf_path.exists() else None
                )
                return await collect_signals_safe(
                    paper_ir, paper_id=paper_id, pub_rank=pub_rank, log_label=f"[{paper_id}] triage"
                )
            finally:
                _signal_lookups.pop(paper_id, None)

        task = asyncio.ensure_future(_lookup())
        _signal_lookups[paper_id] = task
    return await asyncio.shield(task)


def snap_signal_facts(signals: dict[str, Any], title: str) -> list[tuple[str, str]]:
    """The external signals of a Snap report as `(fact, source)` sentences.

    One fact per signal, each with the index that counted it, so a follow-up
    can cite exactly the one it states — "cited 196,706 times" as Semantic
    Scholar's count, the CCF tier as EasyScholar's — and none is merged with
    another system's. Signals the report could not obtain are left out.
    """
    provenance = signals.get("provenance") or {}
    missing = set(signals.get("unavailable") or [])
    name = title or "This paper"
    facts: list[tuple[str, str]] = []

    venue = str(signals.get("venue") or "").strip()
    short = str(signals.get("venue_normalized") or "").strip()
    if venue and "venue" not in missing:
        where = venue + (f" ({short})" if short and short.lower() != venue.lower() else "")
        year = signals.get("year")
        facts.append((
            f"{name} — published at {where}" + (f", {year}" if year else ""),
            str(provenance.get("venue") or ""),
        ))
    label = short or venue or "Its venue"
    if signals.get("ccf_rank") and "ccf_rank" not in missing:
        facts.append((f"{label} — CCF {signals['ccf_rank']}", str(provenance.get("ccf_rank") or "")))
    if signals.get("sci_rank") and "sci_rank" not in missing:
        facts.append((f"{label} — JCR {signals['sci_rank']}", str(provenance.get("sci_rank") or "")))
    extra_ranks = signals.get("venue_ranks") or {}
    if isinstance(extra_ranks, dict):
        rank_prov = str(provenance.get("venue_ranks") or provenance.get("ccf_rank") or provenance.get("sci_rank") or "")
        for sys_key in ("CORE", "CAS", "TH_CPL", "CAAI", "CCF_CN"):
            if extra_ranks.get(sys_key):
                facts.append((f"{label} — {sys_key} {extra_ranks[sys_key]}", rank_prov))
    if signals.get("cas_warning"):
        facts.append((
            f"{label} — CAS International Journal Early Warning: {signals['cas_warning']}",
            str(provenance.get("cas_warning") or "CAS Early Warning List"),
        ))
    counts = signals.get("citation_counts") or {}
    if "citations" not in missing and (counts or signals.get("cited_by_count") is not None):
        # One fact per index: they count differently. Signals cached before
        # the counts were kept apart hold only the larger, under its source.
        source = str(provenance.get("citations") or "")
        for index, cited in (counts or {source: signals["cited_by_count"]}).items():
            parts = [f"cited by {int(cited):,} ({index})" if counts else f"cited by {int(cited):,}"]
            # The per-year rate is computed from the larger count.
            if index == source and signals.get("citations_per_year"):
                parts.append(f"{float(signals['citations_per_year']):,.1f} per year")
            facts.append((f"{name} — " + ", ".join(parts), index))
    # Semantic Scholar's count even when OpenAlex gave the citations: in the
    # same sentence it was sourced to OpenAlex. Without a provenance S2 did
    # not count it, and the 0 it defaults to is not a count.
    if provenance.get("influential_citations") and signals.get("influential_citation_count") is not None:
        facts.append((
            f"{name} — {int(signals['influential_citation_count']):,} influential citations",
            str(provenance["influential_citations"]),
        ))
    intents = CitationIntents.model_validate(signals.get("intents") or {})
    # A sample without a single label is no measurement (`available`).
    if intents.available and "intents" not in missing:
        sample = f"a sample of {intents.sampled:,} citing papers"
        if intents.labelled:
            sample += f", {intents.labelled:,} of them labelled"
        facts.append((
            (f"{name} — citation intents in {sample}: background {intents.background:,}, "
             f"methodology {intents.methodology:,}, result {intents.result:,}"),
            str(provenance.get("intents") or ""),
        ))
    if signals.get("is_open_access") is not None and "open_access" not in missing:
        if signals["is_open_access"]:
            url = signals.get("oa_url") or ""
            fact = f"{name} — open access" + (f" at {url}" if url else "")
        else:
            fact = f"{name} — no open-access copy found"
        facts.append((fact, str(provenance.get("open_access") or "")))
    if signals.get("is_retracted") is not None and "retraction" not in missing:
        fact = f"{name} — " + ("marked as retracted" if signals["is_retracted"] else "not marked as retracted")
        facts.append((fact, str(provenance.get("retraction") or "OpenAlex")))
    if "repos" in signals and "repos" not in missing:
        repos = [str(r.get("url") if isinstance(r, dict) else r) for r in signals.get("repos") or []]
        fact = (
            f"{name} — code repositories linked in the paper: {', '.join(repos[:5])}"
            if repos else f"{name} — no code repository link found in the paper"
        )
        facts.append((fact, str(provenance.get("repos") or "PDF")))
    return facts


async def register_signal_facts(
    ctx: AgentContext, *, paper_id: str, title: str, signals: dict[str, Any], as_of: str
) -> list[dict[str, Any]]:
    """Each signal as one `metadata` evidence row: `[{fact, evidence_id}]`."""
    facts = snap_signal_facts(signals, title)
    if not facts:
        return []
    literature_id = await paper_catalog.ensure_literature_for_local_paper(paper_id)
    out = []
    for fact, source in facts:
        quote = fact + (f" (source: {source}" if source else " (") + (f"; as of {as_of})" if as_of else ")")
        quote = quote.replace(" (; ", " (").replace(" ()", "")
        evidence = await evidence_service.record_metadata_evidence(
            literature_id=literature_id,
            quote=quote,
            owner_id=ctx.owner_id,
            session_id=ctx.session_id,
            provider=source or "insight_snap",
        )
        out.append({"fact": quote, "evidence_id": evidence.evidence_id})
    return out


def _hand_to_report(ctx: AgentContext, paper_id: str, **parts: Any) -> None:
    report = ctx.report
    if report is not None and report.mode == "snap" and report.paper_id == paper_id:
        report.parts.update(parts)


@tool(parse_docstring=False)
async def get_triage_signals(runtime: ToolRuntime[AgentContext], paper_id: str = "") -> str:
    """Look a parsed paper up in the bibliographic indexes, each fact citable.

    Returns where it was published (with its CCF / JCR tier), its citation
    counts per index and how citing papers use it, whether an open-access copy
    exists, whether it was retracted, and code repositories linked in it. Each
    fact carries an evidence number; a fact an index could not supply is listed
    under `unavailable` rather than guessed. Omit paper_id when the session
    holds one paper.
    """
    ctx = runtime.context
    paper_id, paper_ir, denial = await _paper(ctx, paper_id)
    if denial is not None:
        return denial.to_json()
    signals = await signals_for(paper_id, paper_ir)
    data = signals.model_dump(mode="json")
    facts = await register_signal_facts(
        ctx,
        paper_id=paper_id,
        title=paper_ir.title,
        signals=data,
        as_of=datetime.now(UTC).date().isoformat(),
    )
    _hand_to_report(ctx, paper_id, signals=data)
    note = (
        "Not matched in OpenAlex or Semantic Scholar: citations, open access and "
        "retraction are unknown, not zero."
        if not signals.resolved
        else ""
    )
    return ToolResult.ok(
        {
            "paper_id": paper_id,
            "title": paper_ir.title,
            "facts": facts,
            "unavailable": signals.unavailable,
            "external_score": round(signals.external_score, 3),
        },
        evidence_ids=[f["evidence_id"] for f in facts],
        note=note,
    ).to_json()


@tool(parse_docstring=False)
async def review_paper_blind(
    runtime: ToolRuntime[AgentContext], paper_id: str = "", focus: str = ""
) -> str:
    """A blind reviewer's scores for a parsed paper, and the reading verdict they give.

    A separate reviewer reads the paper's abstract, introduction, results and
    conclusion with no venue, citation count or code status in view, and scores
    contribution, evidence, novelty, reproducibility and how severe the
    limitations are, with reasons and red flags. The verdict tier (must read /
    selective / skip) is then combined by fixed rules from those scores and the
    external signals. The review is an opinion: attribute it ("the blind
    review…"), do not cite it as the paper. Pass `focus` when the reader has a
    concern the reviewer should weigh. Takes a minute; asking again returns the
    same review.
    """
    ctx = runtime.context
    paper_id, paper_ir, denial = await _paper(ctx, paper_id)
    if denial is not None:
        return denial.to_json()
    focus = (focus or "").strip()[:500]
    settings = get_settings()

    async def _work(_job_id: str) -> dict[str, Any]:
        from app.services.llm_service import get_llm_service
        from app.workflows.snap_subgraph import run_content_review

        triage = build_triage_context(paper_ir, total_budget=settings.snap_context_budget_chars)
        if not triage.text.strip():
            raise RuntimeError("No content could be extracted from the paper.")
        review = await run_content_review(
            get_llm_service(),
            context=triage.text,
            question=focus,
            language=ctx.language,
            model=ctx.llm_model,
            log_label=f"[{paper_id}] blind review",
        )
        if not review.available:
            # Not cached: the next call should try again rather than reuse a miss.
            raise RuntimeError("The blind review did not return a usable verdict.")
        return {
            "review": review.model_dump(mode="json"),
            "context_stats": {
                "chars": len(triage.text),
                "budget": triage.total_budget,
                "slots": triage.used,
                "tables": triage.tables_used,
                "figures": triage.figures_used,
                "equations": triage.equations_used,
                "truncated": triage.truncated,
            },
        }

    try:
        handle, signals = await asyncio.gather(
            agent_jobs.run_once(
                kind="blind_review",
                idempotency_key=blind_review_key(
                    paper_id=paper_id, language=ctx.language, model=ctx.llm_model, focus=focus
                ),
                work=_work,
                session_id=ctx.session_id,
                run_id=ctx.run_id,
                request={"paper_id": paper_id, "focus": focus},
            ),
            signals_for(paper_id, paper_ir),
        )
    except agent_jobs.JobFailed as exc:
        return ToolResult.failed(
            ErrorCode.UPSTREAM_ERROR, f"The blind review failed: {exc.message}"[:400], retryable=False
        ).to_json()
    if handle.status == "running":
        return ToolResult.partial(
            {"paper_id": paper_id},
            note="Another turn is running this review. Do not start it again; write the "
            "verdict from the signals and your own reading, and say the review is pending.",
        ).to_json()

    review = ContentReview.model_validate(handle.result.get("review") or {})
    verdict = synthesize_verdict(review, signals)
    _hand_to_report(
        ctx,
        paper_id,
        signals=signals.model_dump(mode="json"),
        content_review=review.model_dump(mode="json"),
        verdict=verdict.model_dump(mode="json"),
        context_stats=handle.result.get("context_stats") or None,
    )
    language = "zh" if ctx.language == "zh" else "en"
    data: dict[str, Any] = {
        "paper_id": paper_id,
        "scores": {
            "contribution_strength": review.contribution_strength,
            "evidence_strength": review.evidence_strength,
            "novelty": review.novelty,
            "reproducibility": review.reproducibility,
            "limitation_severity": review.limitation_severity,
        },
        "must_read_sections": review.must_read_sections,
        "target_readers": review.target_readers,
        "reasons": review.reasons,
        "red_flags": review.red_flags,
        "verdict": {
            "tier": verdict.tier,
            "label": TIER_LABELS.get(verdict.tier, {}).get(language, verdict.tier),
            "content_score": verdict.content_score,
            "external_score": verdict.external_score,
            "combined_score": verdict.combined_score,
            "drivers": verdict.drivers,
            "overrides": verdict.overrides,
        },
        "reused": handle.reused,
    }
    if review.question_relevance >= 0:
        data["question_relevance"] = review.question_relevance
        data["question_note"] = review.question_note
    return ToolResult.ok(
        data,
        note="Scores are 0-3 (limitation_severity: higher is worse). The verdict tier is "
        "fixed by rules; state it as given. Page marks inside the reasons are the "
        "reviewer's, not evidence numbers.",
    ).to_json()


TRIAGE_TOOLS = [get_triage_signals, review_paper_blind]
