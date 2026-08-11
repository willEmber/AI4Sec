"""Generate the Insight Snap report as fields, then render the Markdown from them.

The report used to be free Markdown, which made it impossible to do anything
with afterwards: no cards, no sorting, no comparing two papers' results, no
export of anything but the prose. Worse, free prose is where "significantly
outperforms prior work" hides — a sentence that survives review precisely
because it commits to no number.

So the model fills a schema (``SnapReport``) instead, and ``render_markdown``
produces the document. Two things fall out of that:

* Every experimental finding has to name a metric, a dataset and a value, or it
  cannot be represented at all.
* The Markdown structure lives in code, so the English and Chinese outputs are
  guaranteed to have the same sections — the two hand-maintained prompts had
  already drifted apart.

If the JSON pass fails twice, ``parse_snap_report`` degrades to keeping the raw
text as Markdown rather than losing the run.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.models.snap_models import SnapClaim, SnapFinding, SnapReport

logger = logging.getLogger("scholar.snap")


SNAP_REPORT_SYSTEM = """You are a research-paper triage assistant. Read the provided excerpts of ONE paper and fill in a JSON report that lets a researcher decide, in about 30 seconds, whether to read it in depth.

Return ONLY a JSON object — no markdown fences, no commentary:

{
  "one_liner": "One sentence: the problem, the method, and the headline result.",
  "contributions": [
    {"text": "what the paper actually contributes", "page": 3}
  ],
  "findings": [
    {"metric": "BLEU", "dataset": "WMT14 EN-DE", "value": "28.4",
     "baseline": "26.3 (best prior)", "delta": "+2.1", "page": 8,
     "note": "single run"}
  ],
  "suitable_for": "One or two sentences: which tasks, data regimes or settings this is actually useful for.",
  "limitations": [
    {"text": "a concrete limitation, assumption or failure mode", "page": 9}
  ]
}

Rules:
1. 3-5 contributions, 2-6 findings, 1-4 limitations. Fewer is fine; padding is not.
2. `findings` MUST come from the `## Results Tables` block or explicit numbers in
   the prose. Every finding needs a real `metric`, `dataset` and `value`. If the
   paper reports no numbers at all, return an empty `findings` list — do NOT
   invent one, and do NOT fill it with unquantified claims.
3. `baseline` and `delta` are optional per finding; include them whenever the
   comparison is in the source. Keep `delta` signed ("+2.1", "-0.4 ppl").
4. `page` is the integer from the nearest `[p.X]` marker. Never guess a page.
5. Keep every string terse and factual. No praise, no hedging, no "significantly".
6. Use LaTeX for mathematical notation inside strings: $inline$.
7. Do not judge whether the paper is worth reading — a separate reviewer decides
   that, and it is not part of this schema.
"""

_ZH_DIRECTIVE = """
Output language: write every string value in Simplified Chinese. Keep these in
their original English form: LaTeX, metric names, dataset names, paper titles,
author names, and venue names. On first use of a technical term, keep the English
and add the Chinese in parentheses — e.g. "Transformer(变换器)".
JSON keys stay in English exactly as specified.
"""

_REPAIR_DIRECTIVE = """
Your previous response could not be parsed as JSON. Return the same report as a
single valid JSON object and nothing else: no prose before or after it, no
markdown fences, no trailing commas, all strings double-quoted.
"""

_FENCE_RE = re.compile(r"^```\w*\n?|\n?```$", re.MULTILINE)


def system_prompt(language: str = "en", *, repair: bool = False) -> str:
    """The report prompt, localized, optionally with the JSON-repair nudge."""
    prompt = SNAP_REPORT_SYSTEM
    if language == "zh":
        prompt += _ZH_DIRECTIVE
    if repair:
        prompt += _REPAIR_DIRECTIVE
    return prompt


def _as_int(value: Any) -> int:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return 0
    return page if page > 0 else 0


def _claims(items: Any, limit: int) -> list[SnapClaim]:
    out: list[SnapClaim] = []
    for item in (items or [])[:limit]:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            page = _as_int(item.get("page"))
        elif isinstance(item, str):
            text, page = item.strip(), 0
        else:
            continue
        if text:
            out.append(SnapClaim(text=text[:600], page=page))
    return out


def _findings(items: Any, limit: int) -> list[SnapFinding]:
    """Keep only findings that actually carry a number.

    A row with no ``value`` is the unquantified claim the schema exists to
    exclude; dropping it here means a weak model cannot smuggle prose into the
    findings table by leaving the numeric columns blank.
    """
    out: list[SnapFinding] = []
    for item in (items or [])[:limit]:
        if not isinstance(item, dict):
            continue
        value = str(item.get("value", "")).strip()
        metric = str(item.get("metric", "")).strip()
        if not value or not metric:
            continue
        out.append(SnapFinding(
            metric=metric[:80],
            dataset=str(item.get("dataset", "")).strip()[:120],
            value=value[:80],
            baseline=str(item.get("baseline", "")).strip()[:120],
            delta=str(item.get("delta", "")).strip()[:40],
            page=_as_int(item.get("page")),
            note=str(item.get("note", "")).strip()[:200],
        ))
    return out


def parse_snap_report(raw: str) -> SnapReport:
    """Parse the report JSON. On failure returns ``degraded=True`` with the text."""
    text = (raw or "").strip()
    if not text:
        return SnapReport(degraded=True, raw_markdown="")

    candidate = _FENCE_RE.sub("", text).strip() if text.startswith("```") else text
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        logger.warning("snap report: invalid JSON (%s); sample=%r", exc, text[:200])
        return SnapReport(degraded=True, raw_markdown=text)
    if not isinstance(data, dict):
        return SnapReport(degraded=True, raw_markdown=text)

    report = SnapReport(
        one_liner=str(data.get("one_liner", "")).strip()[:800],
        contributions=_claims(data.get("contributions"), 6),
        findings=_findings(data.get("findings"), 8),
        suitable_for=str(data.get("suitable_for", "")).strip()[:800],
        limitations=_claims(data.get("limitations"), 5),
    )
    if not (report.one_liner or report.contributions or report.findings):
        logger.warning("snap report: JSON parsed but empty; sample=%r", text[:200])
        return SnapReport(degraded=True, raw_markdown=text)
    return report


# ──────────────────────────────────────────────────────────────────────
# Markdown rendering
# ──────────────────────────────────────────────────────────────────────

_L = {
    "en": {
        "summary": "## One-Sentence Summary",
        "contributions": "## Core Contributions",
        "findings": "## Key Experimental Findings",
        "applicability": "## Applicability & Limitations",
        "suitable": "**Suitable for**",
        "limitations": "**Limitations**",
        "metric": "Metric",
        "dataset": "Dataset",
        "result": "This paper",
        "baseline": "Baseline",
        "delta": "Δ",
        "page": "Page",
        "no_findings": "_The paper reports no quantified results in the extracted text._",
        "not_reported": "_Not reported in extracted text._",
    },
    "zh": {
        "summary": "## 一句话总结",
        "contributions": "## 核心贡献",
        "findings": "## 关键实验发现",
        "applicability": "## 适用性与局限",
        "suitable": "**适用于**",
        "limitations": "**局限**",
        "metric": "指标",
        "dataset": "数据集",
        "result": "本文",
        "baseline": "基线",
        "delta": "变化",
        "page": "页码",
        "no_findings": "_提取文本中未给出可量化的实验结果。_",
        "not_reported": "_提取文本中未提及。_",
    },
}


def _cite(page: int) -> str:
    return f" [p.{page}]" if page > 0 else ""


def citation_audit(report: SnapReport) -> dict[str, int | float | list[str]]:
    """Exact page-citation coverage over the claims that should carry one.

    The regex auditor in ``citation_validator`` has to guess which sentences are
    factual claims, and over this report it guesses wrong in a specific way: the
    one-liner and the applicability blurb are whole-paper syntheses with no single
    page to point at, so counting them as uncited reported ~67% coverage for a
    report where every citable claim was in fact cited.

    With the structured report the question is no longer a guess — a claim either
    has a page or it does not. Contributions, findings and limitations are the
    citable units; the syntheses are excluded by construction.
    """
    units: list[tuple[str, int]] = []
    units += [(c.text, c.page) for c in report.contributions]
    units += [(limitation.text, limitation.page) for limitation in report.limitations]
    units += [(f"{f.metric} {f.value}", f.page) for f in report.findings]

    uncited = [text for text, page in units if page <= 0]
    total = len(units)
    return {
        "claims_total": total,
        "claims_uncited": len(uncited),
        "coverage": round((total - len(uncited)) / total, 3) if total else 1.0,
        "uncited_samples": uncited[:5],
    }


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def render_markdown(report: SnapReport, *, language: str = "en") -> str:
    """Render the structured report as the Markdown document users read/export."""
    if report.degraded:
        return report.raw_markdown

    lang = _L.get(language, _L["en"])
    parts: list[str] = []

    if report.one_liner:
        parts += [lang["summary"], "", report.one_liner, ""]

    parts += [lang["contributions"], ""]
    if report.contributions:
        parts += [f"- {c.text}{_cite(c.page)}" for c in report.contributions]
    else:
        parts.append(f"- {lang['not_reported']}")
    parts.append("")

    parts += [lang["findings"], ""]
    if report.findings:
        header = [lang["metric"], lang["dataset"], lang["result"], lang["baseline"], lang["delta"], lang["page"]]
        parts.append("| " + " | ".join(header) + " |")
        parts.append("|" + "|".join(["---"] * len(header)) + "|")
        for f in report.findings:
            page = f"[p.{f.page}]" if f.page > 0 else "—"
            cells = [
                _escape_cell(f.metric),
                _escape_cell(f.dataset) or "—",
                _escape_cell(f.value),
                _escape_cell(f.baseline) or "—",
                _escape_cell(f.delta) or "—",
                page,
            ]
            parts.append("| " + " | ".join(cells) + " |")
        parts.append("")
        notes = [f for f in report.findings if f.note]
        if notes:
            parts += [f"- {f.metric}: {f.note}{_cite(f.page)}" for f in notes]
            parts.append("")
    else:
        parts += [lang["no_findings"], ""]

    parts += [lang["applicability"], ""]
    parts.append(f"{lang['suitable']}: {report.suitable_for or lang['not_reported']}")
    parts.append("")
    parts.append(f"{lang['limitations']}:")
    if report.limitations:
        parts += [f"- {limitation.text}{_cite(limitation.page)}" for limitation in report.limitations]
    else:
        parts.append(f"- {lang['not_reported']}")

    return "\n".join(parts).rstrip() + "\n"
