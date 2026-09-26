from __future__ import annotations

import re
from urllib.parse import urlencode

from ..config import Settings
from ..http_client import HTTPClient, HTTPStatusError
from ..models import FILTER_OPEN_ACCESS, FILTER_YEAR, Paper, SearchFilters
from ..ratelimit import ARXIV_LIMITER
from ..utils import normalize_doi, normalize_whitespace, safe_xml_fromstring


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

ARXIV_QUERY_ENDPOINTS = (
    "https://export.arxiv.org/api/query?",
    "http://export.arxiv.org/api/query?",
)

# Everything on arXiv is open access, so that filter is satisfied by construction.
SUPPORTED_FILTERS = frozenset({FILTER_YEAR, FILTER_OPEN_ACCESS})
SUPPORTED_SORTS = frozenset({"relevance", "recent"})

_ABS_ID_RE = re.compile(r"arxiv\.org/abs/(.+?)(?:v\d+)?$", re.IGNORECASE)
# arXiv's query language treats these as operators or syntax.
_QUERY_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)
_BOOLEAN_WORDS = {"and", "or", "andnot"}
# Stopwords as `all:` terms make arXiv's AND match nothing.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is",
    "of", "on", "or", "the", "to", "via", "with", "using", "based", "towards", "toward",
}
# arXiv's edge rejects scripted requests that do not say what they accept.
_HEADERS = {"Accept": "application/atom+xml"}
_MAX_TERMS = 8


def build_search_query(query: str, filters: SearchFilters | None) -> str:
    """`all:` terms joined by AND, plus a submittedDate range when asked.

    A bare `all:word1 word2` only prefixes the first word, and the rest match
    any field in any combination; ANDing the terms keeps results on topic.
    """
    terms = [
        t
        for t in _QUERY_TOKEN_RE.findall(query or "")
        if t.lower() not in _BOOLEAN_WORDS and t.lower() not in _STOPWORDS
    ][:_MAX_TERMS]
    parts = [f"all:{t}" for t in terms] or ["all:*"]
    q = " AND ".join(parts)
    if filters and (filters.year_from or filters.year_to):
        lo = f"{filters.year_from:04d}01010000" if filters.year_from else "199101010000"
        hi = f"{filters.year_to:04d}12312359" if filters.year_to else "209912312359"
        q = f"({q}) AND submittedDate:[{lo} TO {hi}]"
    return q


def arxiv_id_from_abs_url(url: str) -> str:
    m = _ABS_ID_RE.search((url or "").strip())
    return m.group(1) if m else ""


async def search_arxiv(
    client: HTTPClient,
    *,
    query: str,
    limit: int,
    settings: Settings,
    filters: SearchFilters | None = None,
    offset: int = 0,
) -> list[Paper]:
    _ = settings  # unused but kept for signature consistency
    sort = (filters.sort if filters else "relevance") or "relevance"
    qs = urlencode(
        {
            "search_query": build_search_query(query, filters),
            "start": str(max(int(offset), 0)),
            "max_results": str(int(limit)),
            "sortBy": "submittedDate" if sort == "recent" else "relevance",
            "sortOrder": "descending",
        }
    )
    text = ""
    last_exc: Exception | None = None
    for base in ARXIV_QUERY_ENDPOINTS:
        await ARXIV_LIMITER.wait()
        try:
            text = await client.get_text(f"{base}{qs}", headers=_HEADERS)
            break
        except HTTPStatusError as exc:
            # arXiv answers a throttled host with 406, not 429 — and sometimes
            # with the feed in the body anyway. The plain-HTTP mirror is the
            # same service, so an HTTP-level refusal is not retried there.
            if exc.status_code == 406 and "<feed" in (exc.body or ""):
                text = exc.body
            else:
                last_exc = exc
            break
        except Exception as exc:  # noqa: BLE001 — connection trouble: try the next endpoint
            last_exc = exc
            continue
    if not text:
        # Raised, not swallowed: the fan-out reports a failed platform
        # differently from one that found nothing.
        if last_exc is not None:
            raise last_exc
        return []

    root = safe_xml_fromstring(text)

    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ATOM_NS)[:limit]:
        title = normalize_whitespace((entry.findtext("atom:title", default="", namespaces=ATOM_NS) or ""))
        abstract = normalize_whitespace(
            (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "")
        )
        abs_url = normalize_whitespace((entry.findtext("atom:id", default="", namespaces=ATOM_NS) or ""))
        doi = normalize_doi((entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS) or ""))
        arxiv_id = arxiv_id_from_abs_url(abs_url)

        author_names: list[str] = []
        for author in entry.findall("atom:author", ATOM_NS):
            name = normalize_whitespace((author.findtext("atom:name", default="", namespaces=ATOM_NS) or ""))
            if name:
                author_names.append(name)

        # year: extract from <published> tag (format: 2023-01-15T...)
        published = normalize_whitespace(
            entry.findtext("atom:published", default="", namespaces=ATOM_NS) or ""
        )
        year = 0
        if published and len(published) >= 4:
            try:
                year = int(published[:4])
            except ValueError:
                pass

        # venue: arXiv has primary_category as the closest equivalent
        category_el = entry.find("arxiv:primary_category", ATOM_NS)
        venue = ""
        if category_el is not None:
            venue = f"arXiv [{category_el.get('term', '')}]"
        journal_ref = normalize_whitespace(
            entry.findtext("arxiv:journal_ref", default="", namespaces=ATOM_NS) or ""
        )

        papers.append(
            Paper(
                title=title,
                abstract=abstract,
                url=abs_url,
                doi=doi,
                authors="; ".join(author_names),
                source_platform="arXiv",
                year=year,
                venue=venue,
                arxiv_id=arxiv_id,
                oa_pdf_url=f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
                publication_date=published[:10],
                venue_type="preprint",
                is_open_access=True,
                extra={"journal_ref": journal_ref} if journal_ref else {},
            )
        )
    return papers
