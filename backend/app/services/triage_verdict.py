"""Turn evidence into a triage verdict — the part the report used to guess at.

The old Snap produced "Worth Reading? Yes" as the last section of the *same*
generation that had just written "Core Contributions". A decoder that has spent
400 tokens describing a contribution favourably cannot then argue against it:
the verdict was a rationalization of the summary, not a judgement of the paper.

So the verdict is assembled from two sources that never see each other:

* **A content review** (``CONTENT_REVIEW_SYSTEM``) — its own LLM call, scoring
  contribution / evidence / novelty / reproducibility / limitation severity. It
  is shown the paper only. It does *not* get the venue, the citation count, or
  the code links, because a "CCF A" in the prompt anchors every content score
  upward, and the whole point is to have two independent readings.
* **The external signals** (``TriageSignals``) — venue rank, citation impact,
  artifact availability, retraction status. Deterministic, no model involved.

``synthesize_verdict`` combines them into three tiers rather than a yes/no,
because the honest answer for most papers is "read one section of it", and it
records *which* signals drove the call so the report can show its work.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from app.models.snap_models import TriageSignals

logger = logging.getLogger("scholar.triage")


# ──────────────────────────────────────────────────────────────────────
# Content review (independent LLM pass, blind to external signals)
# ──────────────────────────────────────────────────────────────────────

CONTENT_REVIEW_SYSTEM = """You are a demanding peer reviewer performing triage. You will be given excerpts from ONE paper. Judge the paper on its own contents alone.

You are deliberately NOT told where it was published, how often it has been cited, or whether code was released. Do not speculate about any of those — another process supplies them. Judge only what the text supports.

Return ONLY a JSON object (no markdown fences, no commentary):

{
  "contribution_strength": 0-3,   // 0 none/unclear, 1 marginal, 2 solid, 3 major
  "evidence_strength": 0-3,       // are the central claims actually demonstrated? 0 asserted only, 3 thoroughly shown
  "novelty": 0-3,                 // 0 rehash, 1 incremental, 2 clearly new combination, 3 genuinely new idea
  "reproducibility": 0-3,         // enough detail (data, setup, hyperparameters) to reproduce the core result?
  "limitation_severity": 0-3,     // 0 none material, 3 severe enough to undermine the conclusions
  "must_read_sections": [         // 0-3 items; the parts that are worth a reader's time
    {"where": "§4.3 Ablation [p.7]", "why": "first to isolate the effect of A vs B"}
  ],
  "target_readers": "one sentence: who specifically should read it",
  "reasons": ["2-4 short justifications, each carrying a [p.X] citation"],
  "red_flags": ["0-3 concrete methodological problems, each with [p.X]; omit if none"]
}

Rules:
- Score strictly. A 3 must be earned; most competent papers are 2s.
- Every entry in "reasons" and "red_flags" MUST cite a page as [p.X].
- Base "reproducibility" on what the excerpt reports, not on whether code exists.
- If the excerpt is too thin to judge a dimension, score it 1 and say so in "reasons".
- Never mention venues, citation counts, or code availability.
"""

QUESTION_DIRECTIVE = """
The reader asked: "{question}"

Add two fields to the JSON:
  "question_relevance": 0-3,      // how well THIS paper answers that question
  "question_note": "one sentence on what it does or does not answer, with [p.X]"
"""

# The review's prose lands directly in the rendered report, so it has to be in
# the report's language — without this the Chinese output showed English reasons
# under Chinese headings.
_REVIEW_ZH_DIRECTIVE = """
Output language: write every string value in Simplified Chinese — "must_read_sections",
"target_readers", "reasons", "red_flags" and "question_note". Keep these
in their original English form: page citations [p.X], LaTeX, section numbers, metric
and dataset names, paper titles and author names. JSON keys and the numeric scores
stay exactly as specified.
"""


def content_review_prompt(language: str = "en", *, question: str = "") -> str:
    """The review prompt, localized, with the reader's question when there is one."""
    prompt = CONTENT_REVIEW_SYSTEM
    if question.strip():
        prompt += QUESTION_DIRECTIVE.format(question=question.strip()[:500])
    if language == "zh":
        prompt += _REVIEW_ZH_DIRECTIVE
    return prompt


class ContentReview(BaseModel):
    """Parsed content-review scores. Defaults are the 'unjudged' middle."""

    contribution_strength: int = 1
    evidence_strength: int = 1
    novelty: int = 1
    reproducibility: int = 1
    limitation_severity: int = 1
    must_read_sections: list[dict[str, str]] = Field(default_factory=list)
    target_readers: str = ""
    reasons: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    question_relevance: int = -1
    question_note: str = ""
    available: bool = False  # False when the review call failed

    @property
    def content_score(self) -> float:
        """Content quality in [0,1], penalized by limitation severity."""
        positives = (
            0.35 * self.contribution_strength
            + 0.30 * self.evidence_strength
            + 0.20 * self.novelty
            + 0.15 * self.reproducibility
        ) / 3.0
        penalty = 0.12 * (self.limitation_severity / 3.0)
        return max(0.0, min(1.0, positives - penalty))


_FENCE_RE = re.compile(r"^```\w*\n?|\n?```$", re.MULTILINE)


def _clamp(value: Any, low: int = 0, high: int = 3, default: int = 1) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError):
        return default


def parse_content_review(raw: str) -> ContentReview:
    """Parse the review call's JSON. Returns an unavailable review on any failure."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    # Models occasionally prepend prose; salvage the outermost object.
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            logger.warning("content review: no JSON object found; sample=%r", (raw or "")[:200])
            return ContentReview()
        text = text[start : end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("content review: invalid JSON (%s); sample=%r", exc, (raw or "")[:200])
        return ContentReview()
    if not isinstance(data, dict):
        return ContentReview()

    def _str_list(key: str, limit: int) -> list[str]:
        items = data.get(key) or []
        if not isinstance(items, list):
            return []
        return [str(i).strip() for i in items if str(i).strip()][:limit]

    sections: list[dict[str, str]] = []
    for item in (data.get("must_read_sections") or [])[:3]:
        if isinstance(item, dict) and (item.get("where") or item.get("why")):
            sections.append({
                "where": str(item.get("where", "")).strip()[:120],
                "why": str(item.get("why", "")).strip()[:240],
            })
        elif isinstance(item, str) and item.strip():
            sections.append({"where": item.strip()[:120], "why": ""})

    return ContentReview(
        contribution_strength=_clamp(data.get("contribution_strength")),
        evidence_strength=_clamp(data.get("evidence_strength")),
        novelty=_clamp(data.get("novelty")),
        reproducibility=_clamp(data.get("reproducibility")),
        limitation_severity=_clamp(data.get("limitation_severity")),
        must_read_sections=sections,
        target_readers=str(data.get("target_readers", "")).strip()[:400],
        reasons=_str_list("reasons", 4),
        red_flags=_str_list("red_flags", 3),
        question_relevance=_clamp(data.get("question_relevance"), default=-1)
        if data.get("question_relevance") is not None
        else -1,
        question_note=str(data.get("question_note", "")).strip()[:400],
        available=True,
    )


# ──────────────────────────────────────────────────────────────────────
# Synthesis
# ──────────────────────────────────────────────────────────────────────

# Content leads: "worth reading" is mostly a property of the argument, and
# external signals are corroboration. They still carry real weight — a Q1 venue
# and 300 citations are evidence the field found it useful — but they must not
# be able to sink a strong new preprint on their own.
_W_CONTENT = 0.60
_W_EXTERNAL = 0.40

_MUST_READ_THRESHOLD = 0.62
_SELECTIVE_THRESHOLD = 0.38

# A strong content reading floors the verdict at "selective" no matter how thin
# the external record is — that is the case of a good paper published last month.
_STRONG_CONTENT = 0.62

TIER_MUST_READ = "must_read"
TIER_SELECTIVE = "selective"
TIER_SKIP = "skip"


class Verdict(BaseModel):
    tier: str = TIER_SELECTIVE
    combined_score: float = 0.0
    content_score: float = 0.0
    external_score: float = 0.0
    # Short machine-readable notes on what drove the tier, for the report.
    drivers: list[str] = Field(default_factory=list)
    overrides: list[str] = Field(default_factory=list)
    content_available: bool = False


def synthesize_verdict(review: ContentReview, signals: TriageSignals) -> Verdict:
    """Combine the two independent readings into a three-tier verdict."""
    content = review.content_score if review.available else 0.0
    external = signals.external_score

    if review.available:
        combined = _W_CONTENT * content + _W_EXTERNAL * external
    else:
        # No content reading — the external record is all we have. Say so rather
        # than pretending the content scored zero.
        combined = external

    verdict = Verdict(
        combined_score=round(combined, 3),
        content_score=round(content, 3),
        external_score=round(external, 3),
        content_available=review.available,
    )

    if combined >= _MUST_READ_THRESHOLD:
        verdict.tier = TIER_MUST_READ
    elif combined >= _SELECTIVE_THRESHOLD:
        verdict.tier = TIER_SELECTIVE
    else:
        verdict.tier = TIER_SKIP

    # --- hard overrides, highest priority first -------------------------
    if signals.is_retracted:
        verdict.tier = TIER_SKIP
        verdict.overrides.append("retracted")
    elif not review.available:
        # Nothing read the paper. A strong venue and a big citation count are
        # not grounds to tell someone this is a must-read — that would be the
        # same unfounded verdict this pipeline exists to remove. Cap at
        # "selective" and let the rendered notice explain why.
        if verdict.tier == TIER_MUST_READ:
            verdict.tier = TIER_SELECTIVE
        verdict.overrides.append("no_content_review")
    elif content >= _STRONG_CONTENT and verdict.tier == TIER_SKIP:
        # A good paper published last month has no citation record yet; the
        # content reading alone floors it at "selective".
        verdict.tier = TIER_SELECTIVE
        verdict.overrides.append("strong_content_thin_record")

    # --- drivers, strongest first --------------------------------------
    if signals.sci_rank or signals.ccf_rank:
        verdict.drivers.append("venue_rank")
    if signals.citations_known and signals.citation_score >= 0.5:
        verdict.drivers.append("citation_impact")
    if signals.has_official_code:
        verdict.drivers.append("official_code")
    if review.available:
        if review.contribution_strength >= 3 or review.novelty >= 3:
            verdict.drivers.append("strong_contribution")
        if review.evidence_strength <= 1:
            verdict.drivers.append("weak_evidence")
        if review.limitation_severity >= 3:
            verdict.drivers.append("severe_limitations")
    if signals.is_preprint:
        verdict.drivers.append("preprint")
    if signals.is_survey:
        verdict.drivers.append("survey")
    return verdict
