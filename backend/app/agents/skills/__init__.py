"""Report methods: how a report written as the turn's answer is read and written (P16).

`begin_report` hands the model the method of the mode it starts — the sections
the report has, which tools to read with and in what order, how to cite — so
the system prompt carries one line per mode rather than three methods nobody
uses in most turns.

Each method is a Markdown file beside this module, `<name>.<zh|en>.md`, with a
short header:

    ---
    version: 1
    required: Core idea | Method | Equations and algorithms | Experiments
    ---

`required` lists the headings a finished report must have; the snapshot checks
them (`structure_ok`). `version` is part of the prompt version a run records,
so an edited method is told apart from the one a baseline measured.

A mode is written as an answer only when it is switched on
(`AGENT_ANSWER_REPORT_MODES`) *and* its method exists; a switch naming a mode
without one leaves that mode on its report tool.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger("scholar.agents.skills")

SKILLS_DIR = Path(__file__).resolve().parent

SKILL_NAMES = {"lens": "logic-lens", "snap": "insight-snap", "sphere": "research-sphere"}

# Modes already warned about, so a misconfiguration is logged once, not per turn.
_warned: set[str] = set()


@dataclass(frozen=True)
class Method:
    mode: str
    language: str
    version: str
    text: str
    required: tuple[str, ...]


def _parse(raw: str) -> tuple[dict[str, str], str]:
    """Split the `---` header from the body. A file without one is all body."""
    if not raw.startswith("---\n"):
        return {}, raw.strip()
    end = raw.find("\n---", 4)
    if end < 0:
        return {}, raw.strip()
    header: dict[str, str] = {}
    for line in raw[4:end].splitlines():
        key, sep, value = line.partition(":")
        if sep:
            header[key.strip().casefold()] = value.strip()
    return header, raw[end + 4 :].strip()


@functools.lru_cache(maxsize=16)
def _load(directory: str, mode: str, language: str) -> Method | None:
    name = SKILL_NAMES.get(mode)
    if name is None:
        return None
    for lang in (language, "en", "zh"):
        path = Path(directory) / f"{name}.{lang}.md"
        if path.is_file():
            header, body = _parse(path.read_text(encoding="utf-8"))
            required = tuple(
                part.strip() for part in header.get("required", "").split("|") if part.strip()
            )
            return Method(
                mode=mode,
                language=lang,
                version=header.get("version", "0") or "0",
                text=body,
                required=required,
            )
    return None


def method(mode: str, language: str) -> Method | None:
    """The method of `mode` in the conversation's language (English, failing that)."""
    return _load(str(SKILLS_DIR), mode, "zh" if language == "zh" else "en")


def enabled_modes() -> frozenset[str]:
    """The modes this deployment writes as answers: switched on and with a method."""
    switched = get_settings().answer_report_modes
    ready = frozenset(m for m in switched if method(m, "en") is not None)
    for missing in sorted(switched - ready - _warned):
        _warned.add(missing)
        logger.warning("AGENT_ANSWER_REPORT_MODES names %s, which has no method; ignored", missing)
    return ready


def version_tag(modes: frozenset[str]) -> str:
    """`lens1,snap2`: the methods a turn was offered, for its prompt version."""
    parts = []
    for mode in sorted(modes):
        found = method(mode, "en")
        parts.append(f"{mode}{found.version if found else '?'}")
    return ",".join(parts)
