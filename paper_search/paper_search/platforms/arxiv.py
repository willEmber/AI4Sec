from __future__ import annotations

from urllib.parse import quote_plus

from ..config import Settings
from ..http_client import HTTPClient
from ..models import Paper
from ..utils import normalize_doi, normalize_whitespace, safe_xml_fromstring


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

ARXIV_QUERY_ENDPOINTS = (
    "https://export.arxiv.org/api/query?",
    "http://export.arxiv.org/api/query?",
)


def _arxiv_pdf_url(arxiv_abs_url: str) -> str | None:
    url = (arxiv_abs_url or "").strip()
    if "/abs/" not in url:
        return None
    arxiv_id = url.split("/abs/", 1)[1].strip()
    if not arxiv_id:
        return None
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


async def search_arxiv(
    client: HTTPClient, *, query: str, limit: int, settings: Settings
) -> list[Paper]:
    _ = settings  # unused but kept for signature consistency
    qs = f"search_query=all:{quote_plus(query)}&start=0&max_results={int(limit)}"
    text = ""
    for base in ARXIV_QUERY_ENDPOINTS:
        try:
            text = await client.get_text(f"{base}{qs}")
            break
        except Exception:
            continue
    if not text:
        return []

    try:
        root = safe_xml_fromstring(text)
    except Exception:
        return []

    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ATOM_NS)[:limit]:
        title = normalize_whitespace((entry.findtext("atom:title", default="", namespaces=ATOM_NS) or ""))
        abstract = normalize_whitespace(
            (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "")
        )
        abs_url = normalize_whitespace((entry.findtext("atom:id", default="", namespaces=ATOM_NS) or ""))
        doi = normalize_doi((entry.findtext("arxiv:doi", default="", namespaces=ATOM_NS) or ""))

        author_names: list[str] = []
        for author in entry.findall("atom:author", ATOM_NS):
            name = normalize_whitespace((author.findtext("atom:name", default="", namespaces=ATOM_NS) or ""))
            if name:
                author_names.append(name)
        authors = "; ".join(author_names)

        p = Paper(
            title=title,
            abstract=abstract,
            url=abs_url,
            doi=doi,
            authors=authors,
            source_platform="arXiv",
        )
        p.oa_paper_url = _arxiv_pdf_url(abs_url) or ""
        papers.append(p)
    return papers
