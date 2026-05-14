from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.models.paper_ir import PaperIR
from app.services.llm_service import get_llm_service
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.graph")


_VALID_INTENTS = ("snap", "lens", "sphere", "qa")


_SYSTEM_PROMPT = """You are a question-routing classifier for an academic-paper analysis system.
Given the user's question about ONE paper, pick the single most appropriate intent:

- snap   : quick triage; main contributions; "is this paper worth reading"; high-level summary
- lens   : deep technical details — formulas, algorithms, hyperparameters, methodology, reproduction
- sphere : relationship to OTHER works; related papers; research landscape; cross-paper comparison
- qa     : a specific factual question answerable from this single paper (e.g., "what dataset is used?", "what's the learning rate?")

The question may be written in any language; classify by semantics, not language.

Return STRICT JSON only, no prose, no code fences:
{"intent": "snap|lens|sphere|qa", "reason": "<= 10 words"}"""


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from JSON response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _build_context(paper_ir: PaperIR) -> str:
    """Tiny grounding snippet (title + first-page text) so the LLM has something to anchor on."""
    title = (paper_ir.title or "").strip()
    early_text_parts: list[str] = []
    for block in sorted(paper_ir.blocks, key=lambda b: (b.page_idx, b.order_idx)):
        if block.page_idx > 1:
            break
        text = (block.text or "").strip()
        if not text:
            continue
        if block.type in ("text", "title"):
            early_text_parts.append(text)
        if sum(len(p) for p in early_text_parts) > 1500:
            break
    context_body = " ".join(early_text_parts)[:1500]
    return f"Title: {title}\nContext: {context_body}"


async def classify_intent(state: MainGraphState) -> dict[str, Any]:
    """Classify the user's question into snap|lens|sphere|qa when mode == 'auto'.

    No-op (with a 'skipped' progress event) for explicit modes.
    """
    if state.get("error"):
        return {}

    paper_id = state.get("paper_id", "")
    mode = state.get("mode", "snap")
    progress = state.get("progress", [])

    if mode != "auto":
        return {
            "progress": progress + [{"step": "classify_intent", "status": "skipped"}],
        }

    question = (state.get("user_question") or "").strip()
    if not question:
        logger.warning(f"[{paper_id}] classify_intent: empty question under auto mode — falling back to snap")
        return {
            "mode": "snap",
            "detected_intent": "snap",
            "progress": progress + [{"step": "classify_intent", "status": "fallback", "reason": "empty"}],
        }

    t0 = time.perf_counter()
    paper_ir_json = state.get("paper_ir_json", "")
    try:
        paper_ir = PaperIR.model_validate_json(paper_ir_json) if paper_ir_json else None
    except Exception:
        paper_ir = None
    context = _build_context(paper_ir) if paper_ir else ""

    user_msg = (
        f"{context}\n\nUser question: {question}" if context else f"User question: {question}"
    )

    llm = get_llm_service()
    model = state.get("llm_model", "")
    logger.info(f"[{paper_id}] classify_intent: calling LLM (model={model or '(default)'}, q={question[:80]!r})")

    intent = "qa"
    reason = ""
    raw = ""
    try:
        raw = await llm.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            temperature=0.0,
            max_tokens=120,
        )
        cleaned = _strip_json_fences(raw)
        parsed = json.loads(cleaned)
        candidate = str(parsed.get("intent", "")).strip().lower()
        if candidate in _VALID_INTENTS:
            intent = candidate
        reason = str(parsed.get("reason", ""))[:200]
    except json.JSONDecodeError as e:
        logger.warning(f"[{paper_id}] classify_intent: JSON parse failed ({e}); raw={raw[:200]!r}; defaulting to qa")
    except Exception as e:
        logger.warning(f"[{paper_id}] classify_intent: LLM call failed ({e}); defaulting to qa")

    elapsed = time.perf_counter() - t0
    logger.info(f"[{paper_id}] classify_intent: -> {intent} in {elapsed:.2f}s reason={reason!r}")

    return {
        "mode": intent,
        "detected_intent": intent,
        "progress": progress + [{"step": "classify_intent", "status": "done", "intent": intent, "reason": reason}],
    }
