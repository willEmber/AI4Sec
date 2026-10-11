"""Split an answer into the statements its citations attach to.

Citation precision and recall are sentence-level measures (as in ALCE): a
citation is judged against the sentence that carries it, not the whole answer.
The agent writes Markdown, so the split has to step over headings, formulas,
code and table rules, and must not break a sentence at "0.611" or "Eq. 5".

Markdown also widens what one citation covers, and a split that ignores it
reports well-cited answers as uncited. Three scopes are read the way a person
reads them, each becoming *one* statement so nothing is judged twice:

* a table is one statement, citing whatever any of its rows cites;
* a cited line ending in a colon ("采样配置 [ev]：") takes in the uncited list
  items and the table directly under it;
* an uncited table followed by a short cited line ("数据来源：[ev]") takes it in;
* a block quote is one statement, and a display formula under a cited lead-in
  belongs to it.

A merged statement is harder to support in full, not easier: every claim in it
has to be backed by the passages it cites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# What the prompt asks for and the reader's page renders as a badge.
CITATION_RE = re.compile(r"\[(ev_[0-9a-f]{20})\]")
# What the model sometimes writes instead: several ids in one bracket. The
# runner still counts these as citations (it scans for known ids), so they are
# scored as citations here too — and counted, because the page shows them raw.
GROUPED_RE = re.compile(r"\[\s*ev_[0-9a-f]{20}(?:\s*[,，、;；]\s*ev_[0-9a-f]{20})+\s*\]")
_ID_RE = re.compile(r"ev_[0-9a-f]{20}")


def normalise_citations(text: str) -> str:
    """Rewrite `[ev_a, ev_b]` as `[ev_a][ev_b]`, the form everything else reads."""
    return GROUPED_RE.sub(
        lambda m: "".join(f"[{evidence_id}]" for evidence_id in _ID_RE.findall(m.group(0))), text
    )

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_LIST_MARK_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)、])\s+")
_DISPLAY_MATH_RE = re.compile(r"\$\$.+?\$\$", re.S)
# A sentence ends at CJK punctuation, or at Latin punctuation followed by
# whitespace — "0.611" and "Eq.5" have none after the dot. Citation markers
# that trail the punctuation stay with the sentence they follow. A semicolon
# does not end one: "A；B [ev]。" cites both clauses.
_SENTENCE_END_RE = re.compile(
    r"(?:[。！？]|[.!?](?=\s|\[ev_|$))(?:\s*\[ev_[0-9a-f]{20}\])*"
)
# Abbreviations whose dot is followed by a space but ends nothing.
_ABBREVIATION_RE = re.compile(r"(?:\b(?:Eq|Eqs|Fig|Figs|Sec|Tab|et al|i\.e|e\.g|vs|cf|No|resp)\.)$", re.I)
_MATH_RE = re.compile(r"\$\$.+?\$\$|\$[^$]+?\$")
_STOPS = ".!?。！？"
_HIDE = {ord(c): 0xE000 + i for i, c in enumerate(_STOPS)}
_SHOW = {v: k for k, v in _HIDE.items()}
_CONTENT_RE = re.compile(r"[0-9A-Za-z一-鿿]")
MIN_STATEMENT_CHARS = 8
# A line this short after a table that carries citations is its source note.
SOURCE_NOTE_CHARS = 60
_LEAD_IN_RE = re.compile(r"[:：]\s*[*_]*\s*$")
TEXT, ITEM, ROW, MATH, QUOTE = "text", "item", "row", "math", "quote"


@dataclass
class Statement:
    index: int
    text: str                                  # the sentence, markers removed
    citations: list[str] = field(default_factory=list)


def strip_citations(text: str) -> str:
    return re.sub(r"\s{2,}", " ", CITATION_RE.sub("", text)).strip()


def _blocks(answer: str) -> list[tuple[str, str]]:
    """`(kind, line)` for every line that can hold a claim; display math is
    folded onto one line and consecutive table rows into one block."""
    answer = _DISPLAY_MATH_RE.sub(lambda m: " " + m.group(0).replace("\n", " ") + " ", answer)
    blocks: list[tuple[str, str]] = []
    fenced = False
    for raw in answer.splitlines():
        if _FENCE_RE.match(raw):
            fenced = not fenced
            continue
        if fenced or not raw.strip() or _HEADING_RE.match(raw):
            continue
        line = raw.strip()
        if line.startswith(">"):
            quoted = line.lstrip("> ").strip()
            if not quoted:
                continue
            if blocks and blocks[-1][0] == QUOTE:
                blocks[-1] = (QUOTE, blocks[-1][1] + " " + quoted)
            else:
                blocks.append((QUOTE, quoted))
            continue
        if line.startswith("$$") and line.endswith("$$") and len(line) > 4:
            blocks.append((MATH, line))
            continue
        if line.startswith("|"):
            if _TABLE_RULE_RE.match(line):
                continue
            row = " | ".join(cell.strip() for cell in line.strip("|").split("|"))
            if blocks and blocks[-1][0] == ROW:
                blocks[-1] = (ROW, blocks[-1][1] + "\n" + row)
            else:
                blocks.append((ROW, row))
        elif _LIST_MARK_RE.match(line):
            blocks.append((ITEM, _LIST_MARK_RE.sub("", line)))
        else:
            blocks.append((TEXT, line))
    return blocks


def _hide_math_stops(line: str) -> str:
    """Swap sentence punctuation inside `$…$` for placeholders: "$x_0. y$" ends nothing."""
    return _MATH_RE.sub(lambda m: m.group(0).translate(_HIDE), line)


def _sentences(line: str) -> list[str]:
    return [part.translate(_SHOW) for part in _split(_hide_math_stops(line))]


def _split(line: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(line):
        if _ABBREVIATION_RE.search(line[start:match.start() + 1]):
            continue
        parts.append(line[start:match.end()])
        start = match.end()
    if line[start:].strip():
        parts.append(line[start:])
    return parts


def _cites(text: str) -> bool:
    return bool(CITATION_RE.search(text))


def _scoped(blocks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge the blocks one citation covers (see the module docstring)."""
    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(blocks):
        kind, line = blocks[i]
        i += 1
        if kind in (TEXT, ITEM) and _cites(line) and _LEAD_IN_RE.search(strip_citations(line)):
            # A cited lead-in: take the uncited items and table under it. An
            # item with its own citation stands alone and does not end the list.
            kept: list[tuple[str, str]] = []
            alone = line
            while i < len(blocks) and blocks[i][0] in (ITEM, ROW, MATH):
                if _cites(blocks[i][1]):
                    kept.append(blocks[i])
                else:
                    line += "\n" + blocks[i][1]
                i += 1
            merged.append((kind, line) if line == alone else (ROW, line))
            merged.extend(kept)
            continue
        if kind == ROW and not _cites(line) and i < len(blocks):
            after_kind, after = blocks[i]
            if (
                after_kind != ROW
                and _cites(after)
                and len(strip_citations(after)) <= SOURCE_NOTE_CHARS
            ):
                line += "\n" + after
                i += 1
        merged.append((kind, line))
    return merged


def split_statements(answer: str) -> list[Statement]:
    """The answer's statements in order, each with the evidence ids it cites.

    A fragment holding only citation markers belongs to the sentence before it
    ("……结果更好。 [ev_a]" written with the marker on its own line).
    """
    statements: list[Statement] = []
    for kind, line in _scoped(_blocks(normalise_citations(answer or ""))):
        # A table, a quote, a formula, or a lead-in with what it covers, is read whole.
        for sentence in [line] if kind in (ROW, MATH, QUOTE) else _sentences(line):
            cited = list(dict.fromkeys(CITATION_RE.findall(sentence)))
            text = strip_citations(sentence).strip(" *_")
            if len(text) < MIN_STATEMENT_CHARS or not _CONTENT_RE.search(text):
                if cited and statements:
                    previous = statements[-1]
                    previous.citations = list(dict.fromkeys(previous.citations + cited))
                continue
            statements.append(Statement(index=len(statements), text=text, citations=cited))
    return statements
