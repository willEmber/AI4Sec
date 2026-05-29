from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.services.llm_service import get_llm_service
from app.workflows.state import MainGraphState

logger = logging.getLogger("scholar.translate")

_TRANSLATE_SYSTEM = """You are a professional academic translator. Translate the following research analysis from English to Chinese (Simplified).

RULES:
1. Preserve ALL markdown formatting exactly (headings, tables, lists, bold, italic, code blocks, etc.)
2. Preserve ALL LaTeX expressions ($...$, $$...$$) without ANY modification.
3. Preserve ALL page citations [p.X] without modification.
4. Preserve ALL Markdown image embeds ![alt](url) — keep the URL byte-for-byte and do not drop, duplicate, or relocate the image. You may translate the alt text inside the brackets.
5. Keep paper titles, author names, and venue names in their original English form.
6. Translate section headings and prose into natural, academic Chinese.
7. Keep technical terms in English with Chinese explanation in parentheses on first use (e.g. "Transformer (变换器模型)").
8. Do NOT add or remove any content — this is a strict translation.
9. Output ONLY the translated markdown, no explanation or preamble."""

# Section-boundary pattern for chunk splitting
_SECTION_RE = re.compile(r"(?=^## )", re.MULTILINE)

# Maximum characters per translation chunk (~2000 tokens)
_MAX_CHUNK_CHARS = 8000


def _split_at_sections(markdown: str) -> list[str]:
    """Split markdown at ## headings, merging small chunks."""
    parts = _SECTION_RE.split(markdown)
    parts = [p for p in parts if p.strip()]

    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(current) + len(part) > _MAX_CHUNK_CHARS and current:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)
    return chunks


async def _translate_chunk(text: str, model: str) -> str:
    """Translate a single chunk of markdown via LLM."""
    llm = get_llm_service()
    messages = [
        {"role": "system", "content": _TRANSLATE_SYSTEM},
        {"role": "user", "content": text},
    ]
    return await llm.chat(messages, model=model, temperature=0.2, max_tokens=8192)


async def translate_output(state: MainGraphState) -> dict[str, Any]:
    """Translate final_markdown to the requested language. Passes through if language is 'en'."""
    language = state.get("language", "en")
    markdown = state.get("final_markdown", "")

    if language == "en" or not markdown:
        logger.info("[%s] translate_output: SKIPPED (lang=%s, md=%d chars)",
                     state.get("paper_id", "?"), language, len(markdown))
        return {
            "progress": state.get("progress", []) + [{"step": "translate_output", "status": "skipped"}],
        }

    t0 = time.perf_counter()
    model = state.get("llm_model", "")
    logger.info("[%s] translate_output: Translating %d chars to %s...",
                 state.get("paper_id", "?"), len(markdown), language)

    try:
        if len(markdown) <= _MAX_CHUNK_CHARS:
            translated = await _translate_chunk(markdown, model)
        else:
            chunks = _split_at_sections(markdown)
            logger.info("[%s] translate_output: Split into %d chunks", state.get("paper_id", "?"), len(chunks))
            translated_parts: list[str] = []
            for i, chunk in enumerate(chunks):
                logger.info("[%s] translate_output: Translating chunk %d/%d (%d chars)...",
                             state.get("paper_id", "?"), i + 1, len(chunks), len(chunk))
                translated_parts.append(await _translate_chunk(chunk, model))
            translated = "\n\n".join(translated_parts)

        elapsed = time.perf_counter() - t0
        logger.info("[%s] translate_output: DONE in %.1fs — %d -> %d chars",
                     state.get("paper_id", "?"), elapsed, len(markdown), len(translated))
        return {
            "final_markdown": translated,
            "progress": state.get("progress", []) + [{"step": "translate_output", "status": "done"}],
        }

    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error("[%s] translate_output: FAILED in %.1fs — %s (keeping original)",
                      state.get("paper_id", "?"), elapsed, e)
        # On failure, keep original English markdown rather than losing all output
        return {
            "progress": state.get("progress", []) + [
                {"step": "translate_output", "status": "failed", "error": str(e)},
            ],
        }
