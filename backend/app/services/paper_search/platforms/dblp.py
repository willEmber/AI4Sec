"""dblp — the computer-science bibliography.

What dblp adds that the general indexes do not: the venue a CS paper was
actually published at, under the name the community uses ("ICLR", "NeurIPS",
"ACL (1)"), which is what a CCF lookup and a "papers from ICLR 2025" question
both need. It has no abstracts and no citation counts; those come from the
other platforms when the same paper turns up there.

dblp's JSON search API now answers scripts with a proof-of-work bot challenge,
so this goes through the public SPARQL endpoint instead, which is ungated.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from ..config import Settings
from ..http_client import HTTPClient
from ..models import FILTER_PUBLICATION_TYPES, FILTER_VENUES, FILTER_YEAR, Paper, SearchFilters
from ..ratelimit import DBLP_LIMITER
from ..utils import normalize_doi, normalize_whitespace

DBLP_SPARQL = "https://sparql.dblp.org/sparql"

SUPPORTED_FILTERS = frozenset({FILTER_YEAR, FILTER_VENUES, FILTER_PUBLICATION_TYPES})
SUPPORTED_SORTS = frozenset({"relevance", "recent"})

# Words that match nearly every title and only slow the substring scan down.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is",
    "of", "on", "or", "the", "to", "via", "with", "using", "based", "towards", "toward",
}
_MAX_TERMS = 6
DEFAULT_YEARS = 5
_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)
_ARXIV_EE_RE = re.compile(r"arxiv\.org/abs/([^\s?#v]+(?:v\d+)?)", re.IGNORECASE)
_OPENREVIEW_EE_RE = re.compile(r"openreview\.net/(?:forum|pdf)\?id=([\w\-]+)", re.IGNORECASE)
_VERSION_RE = re.compile(r"v\d+$")

_TYPE_TO_DBLP = {"conference": "Inproceedings", "journal": "Article", "preprint": "Informal"}
_DBLP_TO_TYPE = {"Inproceedings": "conference", "Article": "journal", "Informal": "preprint"}


def _sparql_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def query_terms(query: str) -> list[str]:
    terms = [t.lower() for t in _TOKEN_RE.findall(query or "")]
    kept = [t for t in terms if len(t) >= 3 and t not in _STOPWORDS]
    # Longest first: the rarest words narrow the scan fastest.
    kept = sorted(dict.fromkeys(kept), key=len, reverse=True)
    return kept[:_MAX_TERMS]


def _year_floor(filters: SearchFilters | None) -> int:
    """The year the scan starts from.

    Without any year or venue bound, a title scan covers all of dblp and the
    endpoint times out; the most recent five years is the default bound, and
    the tool asks for an explicit year when it needs older work.
    """
    if filters is not None and (filters.year_from or filters.year_to or filters.venues):
        return filters.year_from
    return datetime.now(timezone.utc).year - (DEFAULT_YEARS - 1)


def build_sparql(query: str, *, limit: int, offset: int, filters: SearchFilters | None) -> str:
    terms = query_terms(query)
    if not terms:
        raise ValueError("query has no searchable words")
    # QLever's text index: each word must occur in the title. Far cheaper than
    # a CONTAINS scan over every title literal.
    words = "\n  ".join(f"?text ql:contains-word {_sparql_string(t)} ." for t in terms)

    clauses: list[str] = []
    year_from = _year_floor(filters)
    if year_from:
        clauses.append(f'FILTER(?year >= "{year_from}"^^xsd:gYear)')
    if filters is not None:
        if filters.year_to:
            clauses.append(f'FILTER(?year <= "{filters.year_to}"^^xsd:gYear)')
        if filters.venues:
            # dblp venue labels carry volume suffixes: "ACL (1)", "EMNLP (Findings)".
            alternatives = " || ".join(
                f"?venue = {_sparql_string(v.strip())} || STRSTARTS(?venue, {_sparql_string(v.strip() + ' (')})"
                for v in filters.venues
                if v.strip()
            )
            clauses.append(f"FILTER({alternatives})")
        types = [_TYPE_TO_DBLP[t] for t in filters.publication_types if t in _TYPE_TO_DBLP]
        if types:
            clauses.append(
                "FILTER(?type IN (" + ", ".join(f"dblp:{t}" for t in types) + "))"
            )
    body = "\n  ".join(clauses)
    return f"""PREFIX dblp: <https://dblp.org/rdf/schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX ql: <http://qlever.cs.uni-freiburg.de/builtin-functions/>
SELECT DISTINCT ?pub ?title ?year ?venue ?type WHERE {{
  ?pub dblp:title ?title ; dblp:yearOfPublication ?year ; rdf:type ?type .
  FILTER(?type != dblp:Publication)
  OPTIONAL {{ ?pub dblp:publishedIn ?venue }}
  ?text ql:contains-entity ?title .
  {words}
  {body}
}}
ORDER BY DESC(?year)
LIMIT {int(limit)} OFFSET {int(offset)}"""


def build_details_sparql(pub_uris: list[str]) -> str:
    """DOI, links and ordered authors for publications already found.

    A second query because joining these into the title scan multiplies the
    rows it has to sort, which is what made a search take ten seconds or more.
    """
    values = " ".join(f"<{u}>" for u in pub_uris if u.startswith("https://dblp.org/rec/"))
    return f"""PREFIX dblp: <https://dblp.org/rdf/schema#>
SELECT ?pub (SAMPLE(?doiv) AS ?doi)
  (GROUP_CONCAT(DISTINCT STR(?ee); separator=" ") AS ?ees)
  (GROUP_CONCAT(DISTINCT CONCAT(STR(?ord), "|", ?aname); separator=";;") AS ?authors)
WHERE {{
  VALUES ?pub {{ {values} }}
  OPTIONAL {{ ?pub dblp:doi ?doiv }}
  OPTIONAL {{ ?pub dblp:primaryDocumentPage ?ee }}
  OPTIONAL {{ ?pub dblp:hasSignature ?sig . ?sig dblp:signatureOrdinal ?ord ; dblp:signatureDblpName ?aname }}
}} GROUP BY ?pub"""


def _value(binding: dict[str, Any], key: str) -> str:
    return normalize_whitespace(((binding.get(key) or {}).get("value")) or "")


def _authors(raw: str) -> str:
    pairs: list[tuple[int, str]] = []
    for chunk in (raw or "").split(";;"):
        if "|" not in chunk:
            continue
        ord_s, name = chunk.split("|", 1)
        try:
            pairs.append((int(ord_s), normalize_whitespace(name)))
        except ValueError:
            continue
    # dblp disambiguates homonyms with a numeric suffix ("Wei Wang 0001").
    names = [re.sub(r"\s+\d{4}$", "", n) for _, n in sorted(pairs)]
    return "; ".join(n for n in names if n)


def binding_to_paper(binding: dict[str, Any]) -> Paper:
    pub = _value(binding, "pub")
    type_name = _value(binding, "type").rsplit("#", 1)[-1]
    venue = _value(binding, "venue")
    venue_type = _DBLP_TO_TYPE.get(type_name, "")
    if venue.lower() == "corr":
        venue_type = "preprint"
    year_s = _value(binding, "year")
    return Paper(
        title=_value(binding, "title").rstrip("."),
        abstract="",
        url=pub,
        doi="",
        authors="",
        source_platform="DBLP",
        year=int(year_s) if year_s.isdigit() else 0,
        venue=venue,
        dblp_key=pub.replace("https://dblp.org/rec/", ""),
        venue_type=venue_type,
        venue_aliases=[venue] if venue else [],
    )


def apply_details(paper: Paper, row: dict[str, Any]) -> None:
    ees = _value(row, "ees").split()
    for ee in ees:
        m = _ARXIV_EE_RE.search(ee)
        if m and not paper.arxiv_id:
            paper.arxiv_id = _VERSION_RE.sub("", m.group(1))
        m = _OPENREVIEW_EE_RE.search(ee)
        if m and not paper.openreview_id:
            paper.openreview_id = m.group(1)
    if ees:
        paper.url = ees[0]
    if paper.arxiv_id:
        paper.oa_pdf_url = f"https://arxiv.org/pdf/{paper.arxiv_id}"
    paper.doi = normalize_doi(_value(row, "doi"))
    paper.authors = _authors(((row.get("authors") or {}).get("value")) or "")


async def search_dblp(
    client: HTTPClient,
    *,
    query: str,
    limit: int,
    settings: Settings,
    filters: SearchFilters | None = None,
    offset: int = 0,
) -> list[Paper]:
    _ = settings
    bindings = await _sparql(client, build_sparql(query, limit=limit, offset=offset, filters=filters))
    papers = [binding_to_paper(b) for b in bindings]
    if not papers:
        return papers

    # DOI, links and authors are enrichment; failing here must not lose the hits.
    try:
        rows = await _sparql(client, build_details_sparql([p.url for p in papers]))
    except Exception:  # noqa: BLE001
        return papers
    by_pub = {_value(r, "pub"): r for r in rows}
    for paper in papers:
        row = by_pub.get(paper.url)
        if row:
            apply_details(paper, row)
    return papers


async def _sparql(client: HTTPClient, sparql: str) -> list[dict[str, Any]]:
    await DBLP_LIMITER.wait()
    data = await client.get_json(
        DBLP_SPARQL,
        params={"query": sparql},
        headers={"Accept": "application/sparql-results+json"},
    )
    if isinstance(data, dict) and data.get("exception"):
        raise RuntimeError(f"dblp SPARQL error: {str(data['exception'])[:200]}")
    return [b for b in ((data or {}).get("results") or {}).get("bindings") or [] if isinstance(b, dict)]
