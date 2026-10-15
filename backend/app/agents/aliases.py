"""Short per-session evidence numbers: the model cites `[E12]`, not `[ev_…]` (P15).

An evidence id is a 20-hex hash: right as a key, wrong as something a model
copies. Tokenised digit by digit it is twenty near-identical tokens, and a
conversation holds a hundred or more of them, so every two-character run of
an id also occurs in a dozen others. Writing an id means picking the evidence
(the first token or two) and then copying the rest, and the copy follows
whichever earlier occurrence matches the last few tokens — often another id
sharing a prefix. A controlled test on the gateway's small model cited a
wrong passage five times as often with hash ids as with `E12`, and the wrong
id usually existed, which the citation filter cannot catch.

So the model never sees a hash. Two translation points, one book:

* tool results going into the conversation state have every `ev_…` replaced
  by the session's alias for it (`EvidenceAliasMiddleware`, allocating new
  aliases as tools return evidence);
* text coming out of the model has every citation group rewritten back to
  `[ev_…]` before it is streamed or stored (`CitationRewriter`).

Everything outside the model's own context — events, stored messages,
citations, the frontend, recall — keeps the stored ids, and an alias never
leaves its session. A session started before aliases has hashes in its
history; a hash the model copies from there is passed through for the usual
citation check.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from app.agents.context import AgentContext, BudgetUsage

logger = logging.getLogger(__name__)

ALIAS_PREFIX = "E"
EVIDENCE_ID_RE = re.compile(r"ev_[0-9a-f]{20}(?![0-9a-f])")
_TOKEN = r"(?:[Ee]\d{1,5}|ev_[0-9a-f]{20})"
# What the model puts between two citations in one bracket.
_SEP = r"[,，;；、\s]+"
_OPEN = "[【"
GROUP_RE = re.compile(rf"[\[【]\s*({_TOKEN}(?:{_SEP}{_TOKEN})*)\s*[\]】]")
_SPLIT_RE = re.compile(_SEP)
# A bracket that could still become a group once more text arrives.
_PARTIAL_RE = re.compile(rf"[\[【]\s*(?:{_TOKEN}{_SEP})*(?:[Ee]\d{{0,5}}|ev_?[0-9a-f]{{0,20}})?")
# Longer than eight stored ids: not a citation group. The same bound applies to
# a whole text and to a stream, so both rewrite one text identically.
MAX_GROUP_CHARS = 200
# A bracketed stored id in a tool result, or a bare one.
_TOOL_ID_RE = re.compile(r"\[(ev_[0-9a-f]{20})\]|ev_[0-9a-f]{20}(?![0-9a-f])")


def format_alias(number: int) -> str:
    return f"{ALIAS_PREFIX}{number}"


@dataclass
class AliasBook:
    """One session's aliases, mirrored in memory for the turn.

    Loaded once when the turn starts; `assign` writes new aliases to the
    database before they are used, so a restart or another replica sees the
    numbers the conversation state already holds. With no `session_id` (unit
    tests) numbers are kept in memory only.
    """

    session_id: str = ""
    by_id: dict[str, int] = field(default_factory=dict)
    by_no: dict[int, str] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @classmethod
    async def load(cls, session_id: str) -> AliasBook:
        from app.db import agent_repository as repo

        book = cls(session_id=session_id)
        book._take(await repo.load_evidence_aliases(session_id))
        return book

    def _take(self, numbers: dict[str, int]) -> None:
        for evidence_id, number in numbers.items():
            self.by_id[evidence_id] = number
            self.by_no[number] = evidence_id

    async def assign(self, evidence_ids: list[str]) -> None:
        """Give these ids aliases, keeping the ones they have. Order sets numbers."""
        missing = [i for i in dict.fromkeys(evidence_ids) if i and i not in self.by_id]
        if not missing:
            return
        async with self._lock:
            missing = [i for i in missing if i not in self.by_id]
            if not missing:
                return
            if self.session_id:
                from app.db import agent_repository as repo

                self._take(await repo.allocate_evidence_aliases(self.session_id, missing))
            else:
                top = max(self.by_no, default=0)
                self._take({eid: top + n for n, eid in enumerate(missing, 1)})

    def alias(self, evidence_id: str) -> str | None:
        number = self.by_id.get(evidence_id)
        return format_alias(number) if number is not None else None

    def resolve(self, token: str) -> str | None:
        """The stored id an alias (`E12`, `e12`) or a known stored id names."""
        token = token.strip()
        if token[:1] in ("E", "e") and token[1:].isdigit():
            return self.by_no.get(int(token[1:]))
        if EVIDENCE_ID_RE.fullmatch(token):
            return token if token in self.by_id else None
        return None

    def resolve_all(self, tokens: list[str]) -> list[str]:
        """Stored ids for a list of aliases or stored ids; unknown aliases dropped."""
        out: list[str] = []
        for token in tokens:
            resolved = self.resolve(token)
            if resolved is None and EVIDENCE_ID_RE.fullmatch(token.strip()):
                resolved = token.strip()   # an id from before aliases
            if resolved:
                out.append(resolved)
        return out


# ── Into the model ──────────────────────────────────────────────────────────


def envelope_ids(content: str) -> list[str]:
    """The ids a tool result offers for citation, in its order, then any others."""
    ids: list[str] = []
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            ids = [i for i in parsed.get("evidence_ids") or [] if isinstance(i, str)]
    except (TypeError, ValueError):
        pass
    return ids


def rewrite_tool_content(content: str, book: AliasBook) -> str:
    """Every stored id in a tool result replaced by its alias.

    A stored id with no alias is not something this result offered for
    citation — an earlier answer's marker quoted by recall whose evidence is
    gone. It is removed rather than shown, so the model never sees a hash.
    """

    def swap(match: re.Match[str]) -> str:
        evidence_id = match.group(1) or match.group(0)
        alias = book.alias(evidence_id)
        if alias is None:
            return ""
        return f"[{alias}]" if match.group(1) else alias

    return _TOOL_ID_RE.sub(swap, content)


class EvidenceAliasMiddleware(AgentMiddleware):
    """Puts aliases into every tool result before it enters the conversation.

    Outside the turn guard, so its refusals pass through here too (they carry
    no evidence). The checkpoint stores the rewritten result: the model's
    history is consistent with what it was shown, and eviction stubs and
    summaries built from it carry aliases as well.
    """

    @property
    def name(self) -> str:
        return "EvidenceAliasMiddleware"

    async def awrap_tool_call(  # type: ignore[override]
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        message = await handler(request)
        context = getattr(request.runtime, "context", None)
        book = context.aliases if isinstance(context, AgentContext) else None
        if book is None or not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            return message
        content = message.content
        if "ev_" not in content:
            return message
        await book.assign(envelope_ids(content) + EVIDENCE_ID_RE.findall(content))
        message.content = rewrite_tool_content(content, book)
        return message

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:  # type: ignore[override]
        # The synchronous path is the tool-surface probe; a turn is async.
        return handler(request)


# ── Out of the model ────────────────────────────────────────────────────────


class CitationRewriter:
    """Rewrites the model's citation groups to stored ids, as text streams by.

    A group is a bracket (`[]` or `【】`) holding one or more aliases or stored
    ids, separated by commas, semicolons, `、` or spaces; it becomes one
    `[ev_…]` per citation. An alias the book does not know is dropped (and
    counted); a stored id it does not know is kept as written, for the citation
    check that handles sessions older than aliases. Any other bracket passes
    through untouched.

    Streaming holds back only an unclosed bracket that could still become a
    group, at most `MAX_GROUP_CHARS`. Fed in any number of pieces, the output
    equals rewriting the whole text at once (with the same book).
    """

    def __init__(self, book: AliasBook, *, usage: BudgetUsage | None = None) -> None:
        self.book = book
        self._pending = ""
        self.dropped = 0
        self.expanded = 0
        # The run's usage, where the counts are reported; optional for tests.
        self._usage = usage

    def feed(self, text: str) -> str:
        self._pending += text
        return self._drain(final=False)

    def finish(self) -> str:
        """Release what is held back: an unclosed bracket is plain text."""
        return self._drain(final=True)

    def rewrite(self, text: str) -> str:
        """A whole text at once; leaves nothing pending."""
        return self.feed(text) + self.finish()

    def _drain(self, *, final: bool) -> str:
        out: list[str] = []
        buf = self._pending
        while buf:
            positions = [p for p in (buf.find(c) for c in _OPEN) if p >= 0]
            if not positions:
                out.append(buf)
                buf = ""
                break
            start = min(positions)
            out.append(buf[:start])
            buf = buf[start:]
            match = GROUP_RE.match(buf)
            if match and match.end() <= MAX_GROUP_CHARS:
                out.append(self._expand(match.group(1)))
                buf = buf[match.end():]
                continue
            if (
                not final
                and len(buf) < MAX_GROUP_CHARS
                and _PARTIAL_RE.fullmatch(buf)
            ):
                break   # might still close as a group
            out.append(buf[0])
            buf = buf[1:]
        self._pending = buf
        return "".join(out)

    def _expand(self, inner: str) -> str:
        tokens = [t for t in _SPLIT_RE.split(inner.strip()) if t]
        if len(tokens) > 1:
            self.expanded += 1
            if self._usage is not None:
                self._usage.citation_groups_expanded += 1
        parts = []
        for token in tokens:
            resolved = self.book.resolve(token)
            if resolved is not None:
                parts.append(f"[{resolved}]")
            elif EVIDENCE_ID_RE.fullmatch(token):
                parts.append(f"[{token}]")
            else:
                self.dropped += 1
                if self._usage is not None:
                    self._usage.citations_dropped += 1
                logger.info("Dropped unknown evidence alias %s", token)
        return "".join(parts)


def strip_alias_markers(text: str) -> str:
    """Text with citation groups removed — for previews, which do not cite."""
    return GROUP_RE.sub("", text)
