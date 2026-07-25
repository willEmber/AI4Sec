"""Extract a structured digest from a finished Logic Lens report.

Insight Snap and Research Sphere both ship a structured twin of their report, so
the reader can flip between cards and Markdown. Logic Lens could not: its output
is free prose by design, and there was nothing to render as cards.

This module supplies the missing twin without touching what makes the mode
useful. The deep report is generated first, exactly as before; then one cheap
pass reads *the report* — not the paper — and fills ``LensDigest``. Extracting
from the finished text is what keeps the two views honest: every card quotes a
claim the Markdown makes, carries the page the Markdown cited, and can be checked
by toggling back. A digest generated independently from the paper would be a
second opinion wearing the first one's clothes.

The pass is best-effort throughout. A failed call, unparseable JSON, or an empty
result all return ``available=False``; the run keeps its Markdown and the UI
renders it the way it did before the digest existed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.models.lens_models import (
    LensAlgorithm,
    LensClaim,
    LensDataset,
    LensDigest,
    LensFinding,
    LensFormula,
    LensReproducibility,
    LensStage,
    LensStep,
    LensSymbol,
)

logger = logging.getLogger("scholar.lens")


LENS_DIGEST_SYSTEM = """You convert a finished "Logic Lens" analysis of ONE paper into a structured digest that a UI renders as cards.

The report you are given is the source of truth. Extract only what it states, reuse its wording, and copy its `[p.X]` page numbers. Add no analysis of your own, and do not restate the report in full — the digest is a navigable index of it, not a second copy.

Return ONLY a JSON object — no markdown fences, no commentary:

{
  "core_idea": "2-3 sentences: the central insight and why it works.",
  "problem": "The concrete problem the paper attacks and why it matters.",
  "gap": "The specific limitation in prior work this paper targets.",
  "contributions": [{"text": "what the paper contributes", "page": 3}],
  "pipeline": [
    {"name": "Encoder stack", "role": "what this stage is responsible for", "page": 4}
  ],
  "formulas": [
    {"name": "Scaled dot-product attention",
     "latex": "\\\\mathrm{Attention}(Q,K,V)=\\\\mathrm{softmax}\\\\left(\\\\frac{QK^{\\\\top}}{\\\\sqrt{d_k}}\\\\right)V",
     "page": 4,
     "role": "what it computes and why it is formulated this way",
     "symbols": [{"symbol": "d_k", "meaning": "key dimension"}]}
  ],
  "algorithm": {"name": "Training procedure", "page": 5, "complexity": "O(n^2 d)",
                "steps": [{"step": "what happens in this step", "note": "why / cost"}]},
  "datasets": [{"name": "WMT14 EN-DE", "metrics": "BLEU",
                "measures": "n-gram overlap with reference translations", "page": 8}],
  "setup": [{"text": "Adam, 100k steps, 8x P100", "page": 7}],
  "findings": [{"metric": "BLEU", "dataset": "WMT14 EN-DE", "value": "28.4",
                "baseline": "26.3", "delta": "+2.1", "page": 8, "note": "single run"}],
  "takeaways": [{"text": "what the numbers demonstrate, not the numbers again", "page": 8}],
  "why_it_works": [{"text": "a likely source of the method's effectiveness", "page": 6}],
  "limitations": [{"text": "an assumption, confound or generalization risk", "page": 9}],
  "reproducibility": {"score": 2, "available": ["code released", "hyperparameters"],
                      "missing": ["random seeds"]},
  "open_questions": [{"text": "a direction the report calls out", "page": 0}]
}

Rules:
1. `latex` is the formula body only — no `$` or `$$` delimiters and no `\\\\begin{equation}` wrapper. Keep inline math inside other strings as `$...$`.
2. Every `findings` row needs a real `metric` and `value` copied from the report. Drop rows that would be prose; if the report gives no numbers, return an empty list.
3. `page` is the integer from the report's nearest `[p.X]`. Use 0 when the report gives none — never invent a page.
4. Omit what the report does not support: an empty list, an empty string, or `"algorithm": null` is correct. Padding is not.
5. `reproducibility.score`: 0 = nothing usable, 1 = partial, 2 = most of it, 3 = enough to reproduce the core result. Judge from what the report says, not from what a good paper would say.
6. Keep every string terse — each one is a card in a UI, not a paragraph.
7. Limits: 6 contributions, 10 pipeline stages, 6 formulas (8 symbols each), 12 algorithm steps, 8 datasets, 8 setup items, 10 findings, 6 takeaways, 5 why_it_works, 6 limitations, 5 open questions.
8. Write every string in the SAME language the report is written in. JSON keys stay in English exactly as specified.
"""

_REPAIR_DIRECTIVE = """
Your previous response could not be parsed as JSON. Return the same digest as a
single valid JSON object and nothing else: no prose before or after it, no
markdown fences, no trailing commas, all strings double-quoted, and every
backslash in LaTeX escaped (`\\\\frac`, not `\\frac`).
"""

# The report can run long; the digest only needs the report, so cap the input
# rather than let a 60k-char analysis inflate every run's prompt.
_MAX_REPORT_CHARS = 60_000

_FENCE_RE = re.compile(r"^```\w*\n?|\n?```$", re.MULTILINE)
# Strip delimiters a model leaves on `latex` despite rule 1.
_MATH_DELIM_RE = re.compile(r"^\s*\$\$?|\$\$?\s*$")
_EQ_ENV_RE = re.compile(
    r"^\s*\\begin\{(equation|align|gather|displaymath)\*?\}|\\end\{(equation|align|gather|displaymath)\*?\}\s*$"
)


def system_prompt(*, repair: bool = False) -> str:
    """The digest prompt, optionally with the JSON-repair nudge.

    No language variant: the report is already written in the requested
    language and rule 8 tells the model to follow it, so the digest inherits the
    language instead of re-deciding it.
    """
    return LENS_DIGEST_SYSTEM + (_REPAIR_DIRECTIVE if repair else "")


def _as_int(value: Any) -> int:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return 0
    return page if page > 0 else 0


def _text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _claims(items: Any, limit: int) -> list[LensClaim]:
    out: list[LensClaim] = []
    for item in (items or [])[:limit]:
        if isinstance(item, dict):
            text, page = _text(item.get("text"), 600), _as_int(item.get("page"))
        elif isinstance(item, str):
            text, page = _text(item, 600), 0
        else:
            continue
        if text:
            out.append(LensClaim(text=text, page=page))
    return out


def _strings(items: Any, limit: int) -> list[str]:
    out: list[str] = []
    for item in (items or [])[:limit]:
        text = _text(item.get("text") if isinstance(item, dict) else item, 200)
        if text:
            out.append(text)
    return out


def clean_latex(raw: Any) -> str:
    """Strip delimiters and equation environments off a formula body.

    Models add `$$` or `\\begin{equation}` back roughly as often as they follow
    rule 1, and the card typesets the body in display mode itself — leaving the
    wrapper in place makes KaTeX render the delimiters as text.
    """
    latex = str(raw or "").strip()
    for _ in range(2):
        latex = _EQ_ENV_RE.sub("", latex).strip()
        latex = _MATH_DELIM_RE.sub("", latex).strip()
    return latex[:1000]


def _formulas(items: Any, limit: int) -> list[LensFormula]:
    out: list[LensFormula] = []
    for item in (items or [])[:limit]:
        if not isinstance(item, dict):
            continue
        latex = clean_latex(item.get("latex"))
        if not latex:
            continue
        symbols = [
            LensSymbol(symbol=clean_latex(s.get("symbol")), meaning=_text(s.get("meaning"), 200))
            for s in (item.get("symbols") or [])[:8]
            if isinstance(s, dict) and str(s.get("symbol", "")).strip()
        ]
        out.append(LensFormula(
            name=_text(item.get("name"), 120),
            latex=latex,
            page=_as_int(item.get("page")),
            role=_text(item.get("role"), 600),
            symbols=symbols,
        ))
    return out


def _stages(items: Any, limit: int) -> list[LensStage]:
    out: list[LensStage] = []
    for item in (items or [])[:limit]:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"), 120)
        role = _text(item.get("role"), 500)
        if name or role:
            out.append(LensStage(name=name, role=role, page=_as_int(item.get("page"))))
    return out


def _algorithm(item: Any) -> LensAlgorithm | None:
    if not isinstance(item, dict):
        return None
    steps: list[LensStep] = []
    for raw in (item.get("steps") or [])[:12]:
        if isinstance(raw, dict):
            step, note = _text(raw.get("step"), 400), _text(raw.get("note"), 300)
        elif isinstance(raw, str):
            step, note = _text(raw, 400), ""
        else:
            continue
        if step:
            steps.append(LensStep(step=step, note=note))
    if not steps:
        return None
    return LensAlgorithm(
        name=_text(item.get("name"), 120),
        page=_as_int(item.get("page")),
        complexity=_text(item.get("complexity"), 120),
        steps=steps,
    )


def _datasets(items: Any, limit: int) -> list[LensDataset]:
    out: list[LensDataset] = []
    for item in (items or [])[:limit]:
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"), 120)
        if not name:
            continue
        out.append(LensDataset(
            name=name,
            metrics=_text(item.get("metrics"), 160),
            measures=_text(item.get("measures"), 400),
            page=_as_int(item.get("page")),
        ))
    return out


def _findings(items: Any, limit: int) -> list[LensFinding]:
    """Same rule as Insight Snap: a row without a number is not a finding."""
    out: list[LensFinding] = []
    for item in (items or [])[:limit]:
        if not isinstance(item, dict):
            continue
        metric, value = _text(item.get("metric"), 80), _text(item.get("value"), 80)
        if not metric or not value:
            continue
        out.append(LensFinding(
            metric=metric,
            dataset=_text(item.get("dataset"), 120),
            value=value,
            baseline=_text(item.get("baseline"), 120),
            delta=_text(item.get("delta"), 40),
            page=_as_int(item.get("page")),
            note=_text(item.get("note"), 200),
        ))
    return out


def _reproducibility(item: Any) -> LensReproducibility:
    if not isinstance(item, dict):
        return LensReproducibility()
    try:
        score = int(item.get("score", 0) or 0)
    except (TypeError, ValueError):
        score = 0
    return LensReproducibility(
        score=max(0, min(3, score)),
        available=_strings(item.get("available"), 6),
        missing=_strings(item.get("missing"), 6),
    )


def parse_lens_digest(raw: str) -> LensDigest:
    """Parse the digest JSON. Anything unusable returns ``available=False``."""
    text = (raw or "").strip()
    if not text:
        return LensDigest()

    candidate = _FENCE_RE.sub("", text).strip() if text.startswith("```") else text
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        logger.warning("lens digest: invalid JSON (%s); sample=%r", exc, text[:200])
        return LensDigest()
    if not isinstance(data, dict):
        return LensDigest()

    digest = LensDigest(
        core_idea=_text(data.get("core_idea"), 1200),
        problem=_text(data.get("problem"), 1000),
        gap=_text(data.get("gap"), 1000),
        contributions=_claims(data.get("contributions"), 6),
        pipeline=_stages(data.get("pipeline"), 10),
        formulas=_formulas(data.get("formulas"), 6),
        algorithm=_algorithm(data.get("algorithm")),
        datasets=_datasets(data.get("datasets"), 8),
        setup=_claims(data.get("setup"), 8),
        findings=_findings(data.get("findings"), 10),
        takeaways=_claims(data.get("takeaways"), 6),
        why_it_works=_claims(data.get("why_it_works"), 5),
        limitations=_claims(data.get("limitations"), 6),
        reproducibility=_reproducibility(data.get("reproducibility")),
        open_questions=_claims(data.get("open_questions"), 5),
    )

    # A digest with nothing but a core idea is a card view with one card — worse
    # than the Markdown it would replace, so treat it as unavailable.
    substance = (
        len(digest.contributions) + len(digest.pipeline) + len(digest.formulas)
        + len(digest.findings) + len(digest.datasets) + len(digest.limitations)
    )
    if substance < 2:
        logger.warning("lens digest: parsed but too thin to render; sample=%r", text[:200])
        return LensDigest()

    digest.available = True
    return digest


async def generate_lens_digest(
    llm: Any,
    *,
    markdown: str,
    model: str = "",
    max_tokens: int = 6000,
    log_label: str = "lens",
) -> LensDigest:
    """Best-effort digest of a finished report; never raises.

    Retries once with the repair directive, because a model that ignores the
    JSON contract on the first attempt usually complies when told exactly what
    broke — and unescaped LaTeX backslashes are the failure it makes most.

    Args:
        llm: the shared LLM service (must expose ``async chat``).
        markdown: the Lens report as generated, in its final language.
        model: optional model override.
        max_tokens: response budget; formulas and symbol tables are the bulk.
        log_label: prefix for log lines.
    """
    report = (markdown or "").strip()
    if not report:
        return LensDigest()
    if len(report) > _MAX_REPORT_CHARS:
        report = report[:_MAX_REPORT_CHARS] + "\n[...truncated...]"

    for attempt in (1, 2):
        try:
            raw = await llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt(repair=attempt == 2)},
                    {"role": "user", "content": f"Logic Lens report:\n\n{report}"},
                ],
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.warning("%s: digest call attempt %d failed: %s", log_label, attempt, exc)
            continue

        digest = parse_lens_digest(raw)
        if digest.available:
            return digest
        if attempt == 1:
            logger.info("%s: digest JSON unusable — retrying with repair prompt", log_label)

    logger.warning("%s: digest failed twice — the run keeps its markdown only", log_label)
    return LensDigest()
