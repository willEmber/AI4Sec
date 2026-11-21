"""What Insight Snap reads with besides the paper: the blind content review and
the external signals.

The Snap report is written by the conversation (`agents/skills`, P16); its
tools (`agents/tools/triage.py`) call these two. The subgraph that wrote the
report from them is gone.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import get_settings
from app.models.paper_ir import PaperIR
from app.models.snap_models import TriageSignals
from app.services.llm_service import chat_complete
from app.services.triage_signals import get_or_collect_triage_signals
from app.services.triage_verdict import (
    ContentReview,
    content_review_prompt,
    parse_content_review,
)

logger = logging.getLogger("scholar.graph")

# The output ceiling covers reasoning as well as visible text, on both gateway
# protocols, and these models think by default — so it has to leave room for a
# full reasoning trace *plus* the answer; one sized for the answer alone lets
# reasoning consume all of it and the call returns nothing at all.
# `chat_complete` answers a spent ceiling by asking again with thinking off.
_CONTENT_REVIEW_MAX_TOKENS = 16384


async def run_content_review(
    llm: Any,
    *,
    context: str,
    question: str,
    language: str,
    model: str,
    log_label: str,
) -> ContentReview:
    """An independent LLM pass that scores the paper on its contents alone.

    Given the budgeted triage context of the paper and none of the external
    signals — see the module docstring of ``triage_verdict`` for why.
    Failures degrade to an unavailable review, never to an exception.
    """
    try:
        raw = await chat_complete(
            llm,
            [
                {
                    "role": "system",
                    "content": content_review_prompt(language, question=question),
                },
                {"role": "user", "content": context},
            ],
            model=model,
            temperature=0.0,
            max_tokens=_CONTENT_REVIEW_MAX_TOKENS,
            log_label=f"{log_label} content review",
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


async def collect_signals_safe(
    paper_ir: PaperIR,
    *,
    paper_id: str,
    pub_rank: dict[str, Any],
    log_label: str,
) -> TriageSignals:
    """Cache-backed signal collection with a hard timeout and a failure fallback.

    Signals are corroboration, not a prerequisite: if every index is unreachable
    the report is still written, with the affected signals marked unknown.
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
