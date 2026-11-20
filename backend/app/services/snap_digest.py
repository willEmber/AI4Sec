"""The Snap scorecard's report fields, extracted from a Snap written as an answer (P16 §5.2).

The scorecard renders `json_data.report` — one-liner, contributions,
findings, who it suits, limitations — in the shape `snap_report.SnapReport`
has always had. A Snap made by its tool filled that schema from the paper; a
Snap written as the answer has the prose first, so the fields are extracted
from it, the way `lens_digest` indexes a Lens report: the cards can then never
say something the report does not.

Best effort: a digest that will not parse leaves `degraded=True`, and the page
shows the report's text instead of the scorecard.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.snap_models import SnapReport
from app.services.snap_report import parse_snap_report

logger = logging.getLogger(__name__)

SNAP_DIGEST_SYSTEM = """You convert a finished "Insight Snap" triage report of ONE paper into the JSON a scorecard renders.

The report you are given is the source of truth. Extract only what it states, reuse its wording, and copy its `[p.X]` page numbers. Add nothing of your own.

Return ONLY a JSON object — no markdown fences, no commentary:

{
  "one_liner": "One sentence: the problem, the method, and the headline result, from the report's verdict line.",
  "contributions": [{"text": "a contribution the report lists", "page": 3}],
  "findings": [
    {"metric": "BLEU", "dataset": "WMT14 EN-DE", "value": "28.4",
     "baseline": "26.3 (best prior)", "delta": "+2.1", "page": 8, "note": "single run"}
  ],
  "suitable_for": "The report's 'who should read it', in one or two sentences.",
  "limitations": [{"text": "a limitation the report names", "page": 9}]
}

Rules:
1. At most 6 contributions, 8 findings, 5 limitations — what the report gives, no padding.
2. Every finding needs a real `metric`, `dataset` and `value` copied from the report's results. If the report gives no numbers, return an empty `findings` list.
3. `baseline` and `delta` only when the report states the comparison. Keep `delta` signed ("+2.1").
4. `page` is the integer from the report's nearest `[p.X]`; 0 when it gives none. Never invent one.
5. Leave out the external signals, the blind review and the reading advice: the scorecard shows those from their own sources.
6. Write every string in the SAME language as the report. JSON keys stay in English exactly as specified.
"""

_REPAIR_DIRECTIVE = """
Your previous response could not be parsed as JSON. Return the same digest as a
single valid JSON object and nothing else: no prose before or after it, no
markdown fences, no trailing commas, all strings double-quoted.
"""

_MAX_REPORT_CHARS = 30_000


async def generate_snap_digest(
    llm: Any,
    *,
    markdown: str,
    model: str = "",
    max_tokens: int = 8192,
    log_label: str = "snap",
) -> SnapReport:
    """Best-effort scorecard fields of a finished report; never raises."""
    report = (markdown or "").strip()
    if not report:
        return SnapReport(degraded=True)
    if len(report) > _MAX_REPORT_CHARS:
        report = report[:_MAX_REPORT_CHARS] + "\n[...truncated...]"

    for attempt in (1, 2):
        try:
            raw = await llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": SNAP_DIGEST_SYSTEM + (_REPAIR_DIRECTIVE if attempt == 2 else ""),
                    },
                    {"role": "user", "content": f"Insight Snap report:\n\n{report}"},
                ],
                model=model,
                temperature=0.0,
                max_tokens=max_tokens if attempt == 1 else int(max_tokens * 1.5),
                # Utility call: thinking off (see `llm_gateway.registry`).
                enable_thinking=False,
            )
        except Exception as exc:  # noqa: BLE001 — the report stands without its cards
            logger.warning("%s: snap digest attempt %d failed: %s", log_label, attempt, exc)
            continue
        digest = parse_snap_report(raw)
        if not digest.degraded:
            return digest
        if attempt == 1:
            logger.info("%s: snap digest JSON unusable — retrying with repair prompt", log_label)

    logger.warning("%s: snap digest failed twice — the page shows the report's text", log_label)
    return SnapReport(degraded=True)
