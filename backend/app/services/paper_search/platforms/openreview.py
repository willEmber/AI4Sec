"""OpenReview — submissions, acceptance decisions and reviews (API v2).

The only source that says whether a paper was accepted, rejected or withdrawn
at ICLR/NeurIPS/ICML, and the only one with the reviews themselves.

Access has two tiers. Full-text search (`/notes/search`) works anonymously
and returns each submission with its venue line ("ICLR 2025 Poster",
"Submitted to ICLR 2025", "… Withdrawn Submission"). Reading a forum's
replies (`/notes?forum=…`) is behind a challenge for anonymous clients and
needs an account: set OPENREVIEW_USERNAME / OPENREVIEW_PASSWORD. Without one
the review tool still reports the decision, and says the reviews need a login.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

from ..config import Settings
from ..credentials import openreview_credentials
from ..http_client import HTTPClient, HTTPStatusError
from ..models import FILTER_OPEN_ACCESS, FILTER_VENUES, FILTER_YEAR, Paper, SearchFilters
from ..ratelimit import OPENREVIEW_LIMITER
from ..utils import normalize_whitespace

OPENREVIEW_API = "https://api2.openreview.net"
OPENREVIEW_WEB = "https://openreview.net"

SUPPORTED_FILTERS = frozenset({FILTER_YEAR, FILTER_VENUES, FILTER_OPEN_ACCESS})
SUPPORTED_SORTS = frozenset({"relevance"})

# Venue groups for the conferences that run their reviewing on OpenReview.
_VENUE_GROUPS: dict[str, str] = {
    "iclr": "ICLR.cc/{year}/Conference",
    "neurips": "NeurIPS.cc/{year}/Conference",
    "nips": "NeurIPS.cc/{year}/Conference",
    "icml": "ICML.cc/{year}/Conference",
    "colm": "colmweb.org/COLM/{year}/Conference",
    "tmlr": "TMLR",
}
_MAX_VENUE_YEARS = 4

# A line that still ends in "Submission" names a submission, not a publication:
# ARR's rolling cycles ("ACL ARR 2025 February Submission") never say more.
_NOT_ACCEPTED_RE = re.compile(
    r"submitted to|withdrawn|desk[\s_-]*reject|rejected|under review|submission\s*$", re.IGNORECASE
)
_TIER_RE = re.compile(r"\b(oral|spotlight|poster|notable[- ]top[- ]\d+%?)\b", re.IGNORECASE)


class OpenReviewLoginRequired(RuntimeError):
    """The request needs a logged-in client (anonymous access hit the challenge)."""


def venue_group(venue: str, year: int) -> str:
    template = _VENUE_GROUPS.get(re.sub(r"[^a-z]", "", (venue or "").lower()))
    if not template:
        return ""
    return template.format(year=year) if "{year}" in template else template


def known_venue(venue: str) -> bool:
    return re.sub(r"[^a-z]", "", (venue or "").lower()) in _VENUE_GROUPS


def classify_acceptance(venue_line: str) -> tuple[str, str]:
    """(`accepted`|`not_accepted`|`withdrawn`|`desk_rejected`|`unknown`, tier)."""
    line = normalize_whitespace(venue_line)
    if not line:
        return "unknown", ""
    low = line.lower()
    if "withdrawn" in low:
        return "withdrawn", ""
    if "desk" in low and "reject" in low:
        return "desk_rejected", ""
    if _NOT_ACCEPTED_RE.search(line):
        return "not_accepted", ""
    tier = _TIER_RE.search(line)
    return "accepted", tier.group(1).lower() if tier else ""


def _content_value(content: dict[str, Any], key: str) -> Any:
    raw = content.get(key)
    if isinstance(raw, dict) and "value" in raw:
        return raw["value"]
    return raw


def _year_of(note: dict[str, Any]) -> int:
    for key in ("pdate", "odate", "cdate", "tcdate"):
        ms = note.get(key)
        if isinstance(ms, (int, float)) and ms > 0:
            return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year
    return 0


def is_submission(note: dict[str, Any]) -> bool:
    return any(str(i).endswith("/-/Submission") for i in note.get("invitations") or [])


def note_to_paper(note: dict[str, Any]) -> Paper:
    content = note.get("content") or {}
    forum = note.get("forum") or note.get("id") or ""
    venue_line = normalize_whitespace(str(_content_value(content, "venue") or ""))
    state, _tier = classify_acceptance(venue_line)
    # A rejected or withdrawn submission was not published at the venue; the
    # venue field stays empty so a venue filter cannot count it.
    venue = ""
    if state == "accepted":
        venue = _TIER_RE.sub("", venue_line).strip()
    authors = _content_value(content, "authors") or []
    year = _year_of(note)
    m = re.search(r"\b(19|20)\d{2}\b", venue_line)
    if m:
        year = int(m.group(0))
    has_pdf = bool(_content_value(content, "pdf"))
    return Paper(
        title=normalize_whitespace(str(_content_value(content, "title") or "")),
        abstract=normalize_whitespace(str(_content_value(content, "abstract") or "")),
        url=f"{OPENREVIEW_WEB}/forum?id={forum}" if forum else "",
        doi="",
        authors="; ".join(normalize_whitespace(str(a)) for a in authors if a)
        if isinstance(authors, list)
        else "",
        source_platform="OpenReview",
        year=year,
        venue=venue,
        openreview_id=forum,
        oa_pdf_url=f"{OPENREVIEW_WEB}/pdf?id={forum}" if (forum and has_pdf) else "",
        venue_type=("journal" if venue.upper().startswith("TMLR") else "conference") if venue else "",
        is_open_access=True if has_pdf else None,
        acceptance=venue_line,
        venue_aliases=[venue.split()[0]] if venue else [],
    )


# ── auth ────────────────────────────────────────────────────────────────────

_TOKEN_LOCK = threading.Lock()
_TOKEN: tuple[str, float] = ("", 0.0)
_TOKEN_TTL_S = 45 * 60


async def auth_headers(client: HTTPClient) -> dict[str, str]:
    """Bearer token when an account is configured, else no auth at all."""
    global _TOKEN
    user, password = openreview_credentials()
    if not (user and password):
        return {}
    with _TOKEN_LOCK:
        token, expires = _TOKEN
    if token and expires > time.monotonic():
        return {"Authorization": f"Bearer {token}"}
    await OPENREVIEW_LIMITER.wait()
    data = await client.post_json(
        f"{OPENREVIEW_API}/login", json_body={"id": user, "password": password}
    )
    token = (data or {}).get("token") or ""
    if not token:
        return {}
    with _TOKEN_LOCK:
        _TOKEN = (token, time.monotonic() + _TOKEN_TTL_S)
    return {"Authorization": f"Bearer {token}"}


def has_account() -> bool:
    user, password = openreview_credentials()
    return bool(user and password)


async def _get(client: HTTPClient, path: str, params: dict[str, str]) -> Any:
    await OPENREVIEW_LIMITER.wait()
    headers = await auth_headers(client)
    try:
        return await client.get_json(f"{OPENREVIEW_API}{path}", params=params, headers=headers)
    except HTTPStatusError as exc:
        if exc.status_code == 403 and "Challenge" in (exc.body or ""):
            raise OpenReviewLoginRequired(
                "OpenReview requires a logged-in client for this request"
            ) from exc
        raise


# ── search ──────────────────────────────────────────────────────────────────


# `/notes/search` ANDs its terms, and stopwords are not indexed: one "of" in the
# query matches nothing ("Mixture of Experts" → 0 hits, "Mixture Experts" → 10).
# A quoted phrase is matched as a phrase and survives its stopwords.
_TERM_RE = re.compile(r"[\w\-]+", re.UNICODE)
_STOPWORDS = {
    "a", "about", "all", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "its", "not", "of", "on", "or", "our",
    "the", "their", "this", "to", "via", "we", "what", "when", "which", "why", "with", "you",
}
_MAX_TERMS = 8


def search_term(query: str) -> str:
    """Free-text query as OpenReview can match it: stopwords dropped."""
    terms = [t for t in _TERM_RE.findall(query or "") if t.lower() not in _STOPWORDS]
    return " ".join(terms[:_MAX_TERMS])


def title_phrase(title: str) -> str:
    """A title as one quoted phrase, for looking a known paper up."""
    return '"' + normalize_whitespace((title or "").replace('"', " ")) + '"'


async def search_submissions(
    client: HTTPClient, term: str, *, limit: int, offset: int = 0, group: str = ""
) -> list[dict[str, Any]]:
    params = {"term": term, "source": "forum", "limit": str(limit), "offset": str(offset)}
    if group:
        params["group"] = group
    data = await _get(client, "/notes/search", params)
    return [n for n in (data or {}).get("notes") or [] if isinstance(n, dict) and is_submission(n)]


def _groups_for(filters: SearchFilters | None) -> list[str]:
    if filters is None or not filters.venues:
        return []
    this_year = datetime.now(timezone.utc).year
    lo = filters.year_from or (filters.year_to or this_year) - (_MAX_VENUE_YEARS - 1)
    hi = filters.year_to or this_year
    years = list(range(hi, lo - 1, -1))[:_MAX_VENUE_YEARS]
    groups: list[str] = []
    for venue in filters.venues:
        for year in years:
            g = venue_group(venue, year)
            if g and g not in groups:
                groups.append(g)
    return groups


async def search_openreview(
    client: HTTPClient,
    *,
    query: str,
    limit: int,
    settings: Settings,
    filters: SearchFilters | None = None,
    offset: int = 0,
) -> list[Paper]:
    _ = settings
    limit = max(1, min(int(limit), 50))
    query = search_term(query)
    if not query:
        return []
    if filters is not None and filters.venues:
        groups = _groups_for(filters)
        if not groups:
            # None of the requested venues reviews on OpenReview.
            return []
        per_group = max(3, limit // len(groups) + 1)
        notes: list[dict[str, Any]] = []
        for group in groups:
            notes.extend(await search_submissions(client, query, limit=per_group, offset=offset, group=group))
    else:
        # Over-fetch: search also returns reviews and dblp mirror records,
        # which are dropped here.
        notes = await search_submissions(client, query, limit=limit * 2, offset=offset)

    papers = [note_to_paper(n) for n in notes]
    return [p for p in papers if p.title][:limit]


# ── forum and reviews ───────────────────────────────────────────────────────

_REVIEW_TEXT_FIELDS = (
    "summary", "strengths", "weaknesses", "strengths_and_weaknesses", "questions", "limitations",
)
_REVIEW_SCORE_FIELDS = ("rating", "confidence", "soundness", "presentation", "contribution")


def _score(value: Any) -> str:
    return normalize_whitespace(str(value)) if value not in (None, "") else ""


def parse_forum_notes(notes: list[dict[str, Any]]) -> dict[str, Any]:
    """Split a forum's replies into decision, meta-review and reviews."""
    out: dict[str, Any] = {"decision": "", "meta_review": "", "reviews": []}
    for note in notes:
        invitations = [str(i) for i in note.get("invitations") or []]
        kind = invitations[0].rsplit("/-/", 1)[-1] if invitations else ""
        content = note.get("content") or {}
        if kind == "Decision":
            out["decision"] = _score(_content_value(content, "decision"))
        elif kind == "Meta_Review":
            text = _content_value(content, "metareview") or _content_value(content, "summary") or ""
            rec = _content_value(content, "recommendation") or ""
            out["meta_review"] = normalize_whitespace(f"{rec} {text}".strip())
        elif kind == "Official_Review":
            review: dict[str, Any] = {"review_id": note.get("id") or ""}
            for key in _REVIEW_SCORE_FIELDS:
                value = _score(_content_value(content, key))
                if value:
                    review[key] = value
            for key in _REVIEW_TEXT_FIELDS:
                value = normalize_whitespace(str(_content_value(content, key) or ""))
                if value:
                    review[key] = value
            out["reviews"].append(review)
    return out


async def get_forum_notes(client: HTTPClient, forum_id: str) -> list[dict[str, Any]]:
    data = await _get(client, "/notes", {"forum": forum_id, "limit": "200"})
    return [n for n in (data or {}).get("notes") or [] if isinstance(n, dict)]
