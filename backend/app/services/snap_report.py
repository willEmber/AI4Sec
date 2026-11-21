"""The Insight Snap report as fields: `parse_snap_report`.

A Snap report used to be a JSON schema the model filled, rendered to Markdown
here. The report is now written by the conversation (P16); `snap_digest`
extracts the same fields from its text and parses them with this, so every
finding names a metric, a dataset and a value or is dropped, and a payload
that does not parse keeps the raw text (``degraded=True``) rather than losing
the report.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.models.snap_models import SnapClaim, SnapFinding, SnapReport

logger = logging.getLogger("scholar.snap")


_FENCE_RE = re.compile(r"^```\w*\n?|\n?```$", re.MULTILINE)


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
