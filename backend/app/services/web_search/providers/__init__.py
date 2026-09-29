"""Provider adapters. Each turns one provider's API into `WebResult` /
`WebPage` and names its own quota status codes; key rotation and failure
mapping are shared in `web_search.http`."""

from __future__ import annotations

import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime

# What a single page may contribute before chunking. A page this long is a
# book or a data dump; the reader gets its relevant chunks either way.
MAX_PAGE_CHARS = 300_000


def iso_date(value: object) -> str | None:
    """A provider's date as YYYY-MM-DD, or None when it is absent or vague.

    Providers send ISO timestamps (Exa), RFC 2822 (Tavily news) or relative
    phrases ("3 days ago", Firecrawl news). A relative phrase is dropped rather
    than resolved: it is relative to when the provider indexed the page, which
    we do not know.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
        except ValueError:
            return None
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError, IndexError):
        pass
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def clean_snippet(text: object, limit: int = 1500) -> str:
    out = re.sub(r"\s+", " ", str(text or "")).strip()
    return out if len(out) <= limit else out[: limit - 1].rstrip() + "…"
