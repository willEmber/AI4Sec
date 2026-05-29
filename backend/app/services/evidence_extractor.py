"""P0 stub for evidence-card extraction.

Runs as a best-effort second LLM call after the main report is generated. We
hand the model the same context the main prompt saw plus the list of slots it
just had to fill, and ask it to return a JSON array of supporting evidence
cards. The result lands in ``final_json.evidence_pool`` for the frontend to
surface (P1 will wire this through the renderer); failures are swallowed so
the primary markdown is never blocked.

This module deliberately keeps no Pydantic schema and no DB writes — both are
P1 work (see ``backend/app/models/evidence.py`` in the roadmap).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("scholar.evidence")


_EVIDENCE_SYSTEM = """You extract supporting evidence from an academic paper excerpt.
Given the paper context (each block prefixed with `[p.X]`) and a list of report
slot names, produce a JSON array of evidence cards. Each card MUST be backed by
an exact short quote that appears verbatim in the provided context.

Return ONLY the JSON array, no markdown fences, no commentary.

Schema for each item:
{
  "id": "E01",                     // unique id: E01, E02, ...
  "page": 3,                       // integer from the nearest [p.X] marker
  "slot": "method",                // best-matching slot name from the slot list
  "quote": "exact substring...",   // <=400 chars, copied verbatim from context
  "paraphrase": "<=25 word gist"   // optional plain-English summary
}

Rules:
- Return at most 16 cards.
- Cover slots that have clear evidence; skip slots without any.
- ``quote`` must be a substring of the context (whitespace differences are OK).
- Prefer quotes that contain numbers, named methods, or definitive claims.
- Never invent page numbers — only use page numbers that appear in the context.
"""


_FENCE_RE = re.compile(r"^```\w*\n?|\n?```$", re.MULTILINE)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()
    return text


def _coerce_card(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    quote = str(item.get("quote", "")).strip()
    if not quote:
        return None
    try:
        page = int(item.get("page", 0) or 0)
    except (TypeError, ValueError):
        page = 0
    return {
        "id": str(item.get("id", "")).strip()[:16],
        "page": page,
        "slot": str(item.get("slot", "")).strip()[:64],
        "quote": quote[:400],
        "paraphrase": str(item.get("paraphrase", "")).strip()[:200],
    }


async def extract_evidence_pool(
    llm: Any,
    *,
    context: str,
    slots: list[str],
    model: str = "",
    max_cards: int = 16,
    max_tokens: int = 3000,
    log_label: str = "evidence",
) -> list[dict[str, Any]]:
    """Best-effort: returns ``[]`` if anything goes wrong.

    Args:
        llm: the shared LLM service (must expose ``async chat``).
        context: the exact context the main prompt saw (with ``[p.X]`` markers).
        slots: ordered list of slot names the report is supposed to cover.
        model: optional model override.
        max_cards: hard cap on returned items (post-LLM, in case the model
            ignores the prompt).
        max_tokens: response budget for the extraction call.
        log_label: prefix for log lines so callers can tell snap from lens.
    """
    if not context.strip() or not slots:
        return []

    slot_list = ", ".join(slots)
    user_msg = f"Slot list: {slot_list}\n\nPaper context:\n{context}"

    try:
        resp = await llm.chat(
            messages=[
                {"role": "system", "content": _EVIDENCE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("%s: evidence extraction LLM call failed: %s", log_label, exc)
        return []

    try:
        data = json.loads(_strip_fences(resp))
    except Exception as exc:
        logger.warning(
            "%s: evidence extraction returned non-JSON (%s); sample=%r",
            log_label,
            exc,
            (resp or "")[:200],
        )
        return []

    if not isinstance(data, list):
        logger.warning("%s: evidence extraction returned non-array: %r", log_label, type(data))
        return []

    cards: list[dict[str, Any]] = []
    for item in data:
        card = _coerce_card(item)
        if card is not None:
            cards.append(card)
        if len(cards) >= max_cards:
            break

    logger.info("%s: extracted %d evidence cards (raw=%d)", log_label, len(cards), len(data))
    return cards
