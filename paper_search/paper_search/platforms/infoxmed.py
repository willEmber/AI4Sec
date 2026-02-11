from __future__ import annotations

from ..config import Settings
from ..http_client import HTTPClient
from ..models import Paper
from ..utils import normalize_doi, normalize_whitespace


INFOXMED_API_URL = "https://api.infox-med.com/search/home/keywords"
INFOXMED_WEB_ORIGIN = "https://www.infox-med.com"
INFOXMED_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def _coalesce(*values: str) -> str:
    for v in values:
        if v:
            return v
    return ""


async def search_infoxmed(
    client: HTTPClient, *, query: str, limit: int, settings: Settings
) -> list[Paper]:
    """
    InfoX-Med keyword search.
    API: POST https://api.infox-med.com/search/home/keywords
    """
    payload = {
        "category": int(settings.infoxmed_category),
        "filter": settings.infoxmed_filter,
        "field": settings.infoxmed_field,
        "keywords": query,
        "pageSize": str(int(limit)),
        "pageNum": "1",
        "sort": settings.infoxmed_sort,
    }
    headers = {
        "Content-Type": "application/json",
        "Origin": INFOXMED_WEB_ORIGIN,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "User-Agent": INFOXMED_USER_AGENT,
    }

    data = await client.post_json(INFOXMED_API_URL, json_body=payload, headers=headers)
    if not isinstance(data, dict):
        return []
    if str(data.get("code")) != "0":
        return []

    records = ((data.get("data") or {}).get("records")) or []
    papers: list[Paper] = []
    for record in records[:limit]:
        title_en = normalize_whitespace((record.get("docTitle") or "").strip())
        title_zh = normalize_whitespace((record.get("docTitleZh") or "").strip())
        title = _coalesce(title_zh, title_en)

        abstract_en = normalize_whitespace((record.get("docAbstract") or "").strip())
        abstract_zh = normalize_whitespace((record.get("docAbstractZh") or "").strip())
        abstract = _coalesce(abstract_zh, abstract_en)

        doi = normalize_doi((record.get("docDoi") or "").strip())
        pmid = normalize_whitespace(str(record.get("pmid") or "").strip())
        url = ""
        if doi:
            url = f"https://doi.org/{doi}"
        elif pmid:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        authors = normalize_whitespace((record.get("docAuthor") or "").strip())

        p = Paper(
            title=title,
            abstract=abstract,
            url=url,
            doi=doi,
            authors=authors,
            source_platform="InfoXMed",
        )
        papers.append(p)
    return papers

