"""Budget-aware triage context for Insight Snap.

The original extractor concatenated a few keyword-matched sections in document
order and then hard-cut the result at 12k characters. Two things went wrong:

1. Only ``text``/``title``/``list`` blocks were kept, so every results **table**
   was dropped — while the prompt asked for "main metrics, improvements,
   comparisons". The model had no numbers to report and fell back to restating
   the abstract's adjectives.
2. Because the cut was a prefix of a document-ordered string, the *conclusion*
   (last in the document, and the densest section for triage) was the first
   thing discarded on any long paper.

This module allocates a character budget across named slots in **priority**
order — abstract, conclusion, results tables, then introduction and experiment
prose — and only afterwards assembles them in reading order. A slot that does
not use its allowance releases it to the next one, and long prose slots are
fitted head+tail so an introduction keeps both its problem statement and its
trailing contribution list.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.models.paper_ir import PaperIR
from app.services.ir_extract import (
    ABSTRACT_KEYWORDS,
    CONCLUSION_KEYWORDS,
    EXPERIMENT_KEYWORDS,
    INTRO_KEYWORDS,
    compact_table_text,
    extract_equations,
    extract_figures,
    extract_tables,
    section_match_keys,
    section_text,
)

logger = logging.getLogger("scholar.snap")

_ELISION = "\n\n[… omitted for length …]\n\n"


@dataclass(frozen=True)
class _Slot:
    """One budgeted piece of the triage context.

    ``head_frac`` < 1 fits over-long text as head+tail instead of a prefix, so a
    section whose payload sits at both ends (an introduction: problem first,
    contribution list last) survives truncation.
    """

    key: str
    heading: str
    budget: int
    head_frac: float = 1.0


# Priority order. Earlier slots are funded first and inherit unspent budget.
_SLOTS: tuple[_Slot, ...] = (
    _Slot("abstract", "## Abstract", 4_000),
    _Slot("conclusion", "## Conclusion & Discussion", 4_500, head_frac=0.7),
    _Slot("tables", "## Results Tables", 9_000),
    _Slot("intro", "## Introduction & Contributions", 6_500, head_frac=0.55),
    _Slot("experiments", "## Experimental Setup & Results (prose)", 7_000, head_frac=0.6),
    _Slot("figures", "## Figure Captions", 2_000),
    _Slot("equations", "## Key Equations", 1_200),
)

# Assembly order handed to the model (differs from funding order: the reader
# wants abstract → intro → tables → experiments → conclusion).
_RENDER_ORDER: tuple[str, ...] = (
    "abstract", "intro", "tables", "figures", "equations", "experiments", "conclusion",
)

_MAX_TABLES = 6
_MAX_FIGURES = 12
_MAX_EQUATIONS = 3

# Captions that mark a table as reporting results rather than configuration.
_RESULT_CAPTION_CUES = (
    "result", "comparison", "compare", "ablation", "performance", "accuracy",
    "benchmark", "state-of-the-art", "sota", "baseline", "evaluation",
    "score", "error rate", "f1", "bleu", "auc", "map", "precision", "recall",
    "对比", "结果", "消融", "性能", "准确",
)
# Captions that mark a table as reference material — useful, but not the
# evidence a triage verdict needs.
_CONFIG_CAPTION_CUES = (
    "notation", "symbol", "hyperparameter", "hyper-parameter", "parameter setting",
    "statistics of", "dataset statistics", "glossary", "abbreviation",
)

_DIGIT_RE = re.compile(r"\d")


@dataclass
class TriageContext:
    """Assembled context plus enough instrumentation to debug a thin report."""

    text: str = ""
    used: dict[str, int] = field(default_factory=dict)
    truncated: list[str] = field(default_factory=list)
    tables_used: int = 0
    figures_used: int = 0
    equations_used: int = 0
    total_budget: int = 0

    @property
    def slots_present(self) -> list[str]:
        return [k for k, v in self.used.items() if v > 0]

    def summary(self) -> str:
        parts = [f"{k}={v}" for k, v in self.used.items() if v > 0]
        trunc = f" truncated={','.join(self.truncated)}" if self.truncated else ""
        return (
            f"{len(self.text)}/{self.total_budget} chars [{' '.join(parts)}] "
            f"tables={self.tables_used} figs={self.figures_used} eqs={self.equations_used}{trunc}"
        )


def _fit(text: str, budget: int, head_frac: float) -> tuple[str, bool]:
    """Trim ``text`` to ``budget`` chars, snapping to line boundaries.

    With ``head_frac`` < 1 the result keeps both ends of the text separated by an
    elision marker. Returns ``(text, was_truncated)``.
    """
    if budget <= 0:
        return "", bool(text)
    if len(text) <= budget:
        return text, False

    def _snap_end(s: str) -> str:
        cut = s.rfind("\n")
        return s[:cut] if cut > len(s) * 0.6 else s

    def _snap_start(s: str) -> str:
        cut = s.find("\n")
        return s[cut + 1:] if -1 < cut < len(s) * 0.4 else s

    if head_frac >= 1.0:
        return _snap_end(text[:budget]), True

    head_chars = int(budget * head_frac)
    tail_chars = budget - head_chars - len(_ELISION)
    if tail_chars <= 200:
        return _snap_end(text[:budget]), True
    return _snap_end(text[:head_chars]) + _ELISION + _snap_start(text[-tail_chars:]), True


def _score_table(record: dict) -> float:
    """Rank tables by how likely they carry the paper's headline numbers."""
    text = record.get("text", "") or ""
    caption = text[: text.find("<table")] if "<table" in text else text[:300]
    caption_low = caption.lower()
    section_low = (record.get("section", "") or "").lower()

    score = 0.0
    if any(kw in section_low for kw in EXPERIMENT_KEYWORDS):
        score += 3.0
    if any(cue in caption_low for cue in _RESULT_CAPTION_CUES):
        score += 2.5
    if any(cue in caption_low for cue in _CONFIG_CAPTION_CUES):
        score -= 2.0

    # Numeric density: a results table is mostly digits, a notation table is not.
    body = text[len(caption):] or text
    if body:
        digits = len(_DIGIT_RE.findall(body))
        if digits / max(1, len(body)) > 0.05:
            score += 2.0
    return score


def _collect_tables(paper_ir: PaperIR) -> list[str]:
    """Compact the most result-bearing tables, best first."""
    records = extract_tables(paper_ir)
    if not records:
        return []
    ranked = sorted(
        records,
        key=lambda r: (-_score_table(r), r.get("page", 0)),
    )
    out: list[str] = []
    for record in ranked[:_MAX_TABLES]:
        compact = compact_table_text(record["text"])
        if compact:
            out.append(f"[p.{record['page']}] {compact}")
    return out


def _collect_figures(paper_ir: PaperIR) -> list[str]:
    return [
        f"[p.{f['page']}] {f['text'].strip()}"
        for f in extract_figures(paper_ir)[:_MAX_FIGURES]
        if f.get("text", "").strip()
    ]


def _collect_equations(paper_ir: PaperIR) -> list[str]:
    return [
        f"[p.{e['page']}] {e['text'].strip()}"
        for e in extract_equations(paper_ir)[:_MAX_EQUATIONS]
        if e.get("text", "").strip()
    ]


def _fallback_body_text(paper_ir: PaperIR) -> str:
    """Every text block, page-tagged — used when section matching finds nothing.

    Papers whose headings MinerU failed to detect (scans, unusual layouts) end up
    with a single untitled section, so keyword matching yields empty slots. Rather
    than report "not reported in extracted text" for the whole paper, fall back to
    the raw block stream.
    """
    parts = [
        f"{b.text.strip()} [p.{b.page_idx + 1}]"
        for b in paper_ir.blocks
        if b.type in ("text", "title", "list") and b.text.strip()
    ]
    return "\n".join(parts)


def build_triage_context(paper_ir: PaperIR, *, total_budget: int = 40_000) -> TriageContext:
    """Assemble the Insight Snap context under a character budget.

    Slots are funded in the priority order of ``_SLOTS`` (unspent allowance
    cascades forward, then a second pass hands leftovers back to slots that were
    truncated), and rendered in ``_RENDER_ORDER``.
    """
    match_keys = section_match_keys(paper_ir)

    def _sec(keywords) -> str:
        return section_text(paper_ir, keywords, match_keys=match_keys)

    candidates: dict[str, str] = {
        "abstract": _sec(ABSTRACT_KEYWORDS),
        "conclusion": _sec(CONCLUSION_KEYWORDS),
        "intro": _sec(INTRO_KEYWORDS),
        "experiments": section_text(
            paper_ir,
            EXPERIMENT_KEYWORDS,
            block_types=("text", "title", "list"),
            match_keys=match_keys,
        ),
        "tables": "\n\n".join(_collect_tables(paper_ir)),
        "figures": "\n".join(_collect_figures(paper_ir)),
        "equations": "\n\n".join(_collect_equations(paper_ir)),
    }

    # Heading detection failed → no prose slot matched. Feed the raw block stream
    # into the intro slot so the run still sees the paper.
    if not any(candidates[k] for k in ("abstract", "conclusion", "intro", "experiments")):
        candidates["intro"] = _fallback_body_text(paper_ir)
        logger.info("snap: section matching found nothing — using raw block fallback")

    ctx = TriageContext(total_budget=total_budget)
    fitted: dict[str, str] = {}
    # Reserve what assembly itself will add (title line + slot headings) so the
    # budget bounds the string actually sent to the model, not just its content.
    overhead = len(paper_ir.title) + 4 + sum(len(s.heading) + 4 for s in _SLOTS)
    remaining = max(1_000, total_budget - overhead)

    # Pass 1 — fund in priority order.
    for slot in _SLOTS:
        raw = candidates.get(slot.key, "")
        if not raw:
            ctx.used[slot.key] = 0
            continue
        allowance = min(slot.budget, remaining)
        text, was_truncated = _fit(raw, allowance, slot.head_frac)
        fitted[slot.key] = text
        ctx.used[slot.key] = len(text)
        remaining -= len(text)
        if was_truncated:
            ctx.truncated.append(slot.key)

    # Pass 2 — redistribute what pass 1 left over to the slots it had to cut.
    for slot in _SLOTS:
        if remaining <= 0 or slot.key not in ctx.truncated:
            continue
        raw = candidates.get(slot.key, "")
        allowance = min(len(raw), ctx.used[slot.key] + remaining)
        text, was_truncated = _fit(raw, allowance, slot.head_frac)
        remaining -= len(text) - ctx.used[slot.key]
        fitted[slot.key] = text
        ctx.used[slot.key] = len(text)
        if not was_truncated:
            ctx.truncated.remove(slot.key)

    ctx.tables_used = fitted.get("tables", "").count("[p.")
    ctx.figures_used = len([ln for ln in fitted.get("figures", "").splitlines() if ln.strip()])
    ctx.equations_used = fitted.get("equations", "").count("[p.")

    parts: list[str] = []
    if paper_ir.title:
        parts.append(f"# {paper_ir.title}")
    slot_by_key = {s.key: s for s in _SLOTS}
    for key in _RENDER_ORDER:
        body = fitted.get(key, "").strip()
        if body:
            parts.append(f"{slot_by_key[key].heading}\n{body}")
    ctx.text = "\n\n".join(parts)
    return ctx
