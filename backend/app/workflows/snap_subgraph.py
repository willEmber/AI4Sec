from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from app.config import get_settings
from app.models.paper_ir import PaperIR
from app.models.snap_models import SnapReport, TriageSignals
from app.services.citation_validator import (
    format_coverage_summary,
    validate_citation_coverage,
)
from app.services.llm_service import get_llm_service
from app.services.snap_context import build_triage_context
from app.services.snap_report import (
    citation_audit as report_citation_audit,
    parse_snap_report,
    render_markdown as render_report_markdown,
    system_prompt as report_system_prompt,
)
from app.services.triage_signals import get_or_collect_triage_signals
from app.services.triage_verdict import (
    ContentReview,
    content_review_prompt,
    parse_content_review,
    render_verdict_markdown,
    synthesize_verdict,
)
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.graph")



async def _generate_report(
    llm: Any,
    *,
    context: str,
    language: str,
    model: str,
    log_label: str,
) -> SnapReport:
    """Fill the ``SnapReport`` schema, retrying once if the JSON does not parse.

    A model that ignores the schema on the first attempt usually complies when
    told exactly what went wrong, so one repair round-trip is worth the latency.
    If it fails twice the raw text is kept as Markdown (``degraded=True``) — a
    prose report is worse than a structured one but far better than no run.
    """
    user_message = (
        "请分析这篇论文:\n\n" if language == "zh" else "Analyze this paper:\n\n"
    ) + context

    raw = ""
    for attempt in (1, 2):
        try:
            raw = await llm.chat(
                [
                    {"role": "system", "content": report_system_prompt(language, repair=attempt == 2)},
                    {"role": "user", "content": user_message},
                ],
                model=model,
                temperature=0.2 if attempt == 1 else 0.0,
                max_tokens=4096,
            )
        except Exception as exc:
            logger.warning("%s: report call attempt %d failed: %s", log_label, attempt, exc)
            continue

        report = parse_snap_report(raw)
        if not report.degraded:
            return report
        if attempt == 1:
            logger.info("%s: report JSON unparseable — retrying with repair prompt", log_label)

    logger.warning("%s: report JSON failed twice — falling back to raw text", log_label)
    return SnapReport(degraded=True, raw_markdown=raw)


async def _run_content_review(
    llm: Any,
    *,
    context: str,
    question: str,
    language: str,
    model: str,
    log_label: str,
) -> ContentReview:
    """Second, independent LLM pass that scores the paper on its contents alone.

    Deliberately given the same paper context as the report but none of the
    external signals — see the module docstring of ``triage_verdict`` for why.
    Failures degrade to an unavailable review, never to an exception.
    """
    try:
        raw = await llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": content_review_prompt(language, question=question),
                },
                {"role": "user", "content": context},
            ],
            model=model,
            temperature=0.0,
            max_tokens=2000,
        )
    except Exception as exc:
        logger.warning("%s: content review call failed: %s", log_label, exc)
        return ContentReview()

    review = parse_content_review(raw)
    if review.available:
        logger.info(
            "%s: content review — contribution=%d evidence=%d novelty=%d "
            "reproducibility=%d limitations=%d score=%.2f",
            log_label, review.contribution_strength, review.evidence_strength,
            review.novelty, review.reproducibility, review.limitation_severity,
            review.content_score,
        )
    return review


async def _collect_signals_safe(
    paper_ir: PaperIR,
    *,
    paper_id: str,
    pub_rank: dict[str, Any],
    log_label: str,
) -> TriageSignals:
    """Cache-backed signal collection with a hard timeout and a failure fallback.

    Signals are corroboration, not a prerequisite: if every index is unreachable
    the run still produces a report, with the affected rows marked unknown.
    """
    settings = get_settings()
    try:
        return await asyncio.wait_for(
            get_or_collect_triage_signals(
                paper_ir,
                paper_id=paper_id,
                ttl_hours=settings.snap_signals_ttl_hours,
                pub_rank=pub_rank,
                doi=pub_rank.get("doi", ""),
                arxiv_id=pub_rank.get("arxiv_id", ""),
                enable_network=settings.snap_signals_enabled,
                probe_repos=settings.snap_probe_repos,
                github_token=settings.github_token,
                fetch_intents=settings.snap_citation_intents,
                log_label=log_label,
            ),
            timeout=60.0,
        )
    except Exception as exc:
        logger.warning("%s: signal collection failed (%s) — verdict falls back to content only", log_label, exc)
        from app.services.triage_signals import compute_external_score

        fallback = TriageSignals(
            venue=pub_rank.get("venue", ""),
            year=int(pub_rank.get("year") or 0),
            sci_rank=pub_rank.get("sci", ""),
            ccf_rank=pub_rank.get("ccf", ""),
            unavailable=["citations", "open_access", "retraction"],
        )
        return compute_external_score(fallback)


async def run_insight_snap(state: MainGraphState) -> dict[str, Any]:
    """Run Insight Snap analysis on the paper."""
    paper_id = state["paper_id"]
    t0 = time.perf_counter()
    settings = get_settings()

    paper_ir = PaperIR.model_validate_json(state["paper_ir_json"])
    logger.info(f"[{paper_id}] snap: Parsed PaperIR — {len(paper_ir.sections)} sections, {len(paper_ir.blocks)} blocks")

    t_extract = time.perf_counter()
    ctx = build_triage_context(paper_ir, total_budget=settings.snap_context_budget_chars)
    logger.info(
        f"[{paper_id}] snap: built triage context in {time.perf_counter()-t_extract:.3f}s — {ctx.summary()}"
    )

    if not ctx.text.strip():
        return {
            "final_markdown": "# Error\n\nNo content could be extracted from the paper.",
            "final_json": json.dumps({"error": "no_content"}),
            "progress": state.get("progress", []) + [{"step": "run_snap", "status": "failed"}],
        }

    key_content = ctx.text

    try:
        pub_rank = json.loads(state.get("pub_rank_json", "") or "{}")
        if not isinstance(pub_rank, dict):
            pub_rank = {}
    except (json.JSONDecodeError, TypeError):
        pub_rank = {}

    # The report prompt may see the venue (it is asked to mention it in passing);
    # the *content review* must not, so it gets the untouched context.
    report_content = key_content
    meta_parts: list[str] = []
    if pub_rank.get("venue"):
        meta_parts.append(f"Published in: {pub_rank['venue']}")
    if pub_rank.get("sci"):
        meta_parts.append(f"SCI Tier: {pub_rank['sci']}")
    if pub_rank.get("ccf"):
        meta_parts.append(f"CCF Rating: {pub_rank['ccf']}")
    if meta_parts:
        report_content = f"[Publication Info: {' | '.join(meta_parts)}]\n\n" + key_content

    language = state.get("language", "en")
    llm = get_llm_service()
    model = state.get("llm_model", "")
    question = state.get("user_question", "") or ""

    # Three independent passes, run concurrently: the structured report, the
    # blind content review, and the external-signal lookups. Total latency is the
    # slowest one rather than their sum — the old sequential report +
    # evidence-pool pair was strictly slower than this while producing less.
    logger.info(
        f"[{paper_id}] snap: Calling LLM (model={model or '(default)'}) "
        f"+ content review + signal lookup concurrently..."
    )
    t_llm = time.perf_counter()
    report, review, signals = await asyncio.gather(
        _generate_report(
            llm,
            context=report_content,
            language=language,
            model=model,
            log_label=f"[{paper_id}] snap",
        ),
        _run_content_review(
            llm,
            context=key_content,
            question=question,
            language=language,
            model=model,
            log_label=f"[{paper_id}] snap",
        ),
        _collect_signals_safe(
            paper_ir, paper_id=paper_id, pub_rank=pub_rank, log_label=f"[{paper_id}] snap"
        ),
    )
    logger.info(
        f"[{paper_id}] snap: all three passes done in {time.perf_counter()-t_llm:.1f}s — "
        f"contributions={len(report.contributions)} findings={len(report.findings)} "
        f"limitations={len(report.limitations)} degraded={report.degraded}"
    )

    markdown = render_report_markdown(report, language=language)
    verdict = synthesize_verdict(review, signals)
    logger.info(
        f"[{paper_id}] snap: verdict={verdict.tier} combined={verdict.combined_score:.2f} "
        f"(content={verdict.content_score:.2f} external={verdict.external_score:.2f}) "
        f"drivers={','.join(verdict.drivers) or '-'} overrides={','.join(verdict.overrides) or '-'}"
    )

    # Coverage over the model's citable claims. A structured report can be audited
    # exactly (a claim has a page or it does not); only the degraded prose path
    # needs the regex heuristic. Either way the verdict section is excluded — it is
    # assembled from typed signals and names a source per row, so auditing it for
    # [p.X] would report our own table cells as uncited claims.
    audit = (
        validate_citation_coverage(markdown)
        if report.degraded
        else report_citation_audit(report)
    )
    logger.info(f"[{paper_id}] snap: {format_coverage_summary(audit)}")
    if audit["claims_uncited"] > 0 and audit["uncited_samples"]:
        logger.warning(
            f"[{paper_id}] snap: {audit['claims_uncited']} uncited claims — samples: {audit['uncited_samples'][:3]}"
        )

    verdict_markdown = render_verdict_markdown(verdict, review, signals, language=language)
    markdown = f"{markdown.rstrip()}\n\n{verdict_markdown}"

    logger.info(f"[{paper_id}] snap: TOTAL {time.perf_counter()-t0:.1f}s")
    return {
        "final_markdown": markdown,
        "analysis_language": language,
        "final_json": json.dumps({
            "mode": "snap",
            "paper_id": state["paper_id"],
            "title": paper_ir.title,
            "sections_used": [s.title for s in paper_ir.sections if s.title],
            "citation_audit": audit,
            "report": report.model_dump(mode="json"),
            "signals": signals.model_dump(mode="json"),
            "content_review": review.model_dump(mode="json"),
            "verdict": verdict.model_dump(mode="json"),
            "context_stats": {
                "chars": len(ctx.text),
                "budget": ctx.total_budget,
                "slots": ctx.used,
                "tables": ctx.tables_used,
                "figures": ctx.figures_used,
                "equations": ctx.equations_used,
                "truncated": ctx.truncated,
            },
        }),
        "progress": state.get("progress", []) + [{"step": "run_snap", "status": "done"}],
    }
