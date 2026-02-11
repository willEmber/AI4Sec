from __future__ import annotations

import xml.etree.ElementTree as ET

from ..config import Settings
from ..http_client import HTTPClient
from ..models import Paper
from ..utils import normalize_doi, normalize_whitespace, safe_xml_fromstring


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _param_identity(settings: Settings) -> dict[str, str]:
    params: dict[str, str] = {}
    if settings.tool_name:
        params["tool"] = settings.tool_name
    email = settings.pick_contact_email()
    if email:
        params["email"] = email
    api_key = (settings.ncbi_api_key or "").strip()
    if api_key:
        params["api_key"] = api_key
    return params


def _extract_doi(article: ET.Element) -> str:
    # Prefer PubmedData/ArticleIdList/ArticleId[@IdType="doi"]
    for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        if (aid.get("IdType") or "").lower() == "doi":
            return normalize_doi(aid.text or "")
    # Fallback: MedlineCitation/Article/ELocationID[@EIdType="doi"]
    for eloc in article.findall(".//MedlineCitation/Article/ELocationID"):
        if (eloc.get("EIdType") or "").lower() == "doi":
            return normalize_doi(eloc.text or "")
    return ""


def _extract_abstract(article: ET.Element) -> str:
    parts: list[str] = []
    for node in article.findall(".//MedlineCitation/Article/Abstract/AbstractText"):
        text = "".join(node.itertext()).strip()
        if text:
            label = (node.get("Label") or "").strip()
            if label:
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)
    return normalize_whitespace(" ".join(parts))


def _extract_title(article: ET.Element) -> str:
    title_node = article.find(".//MedlineCitation/Article/ArticleTitle")
    if title_node is None:
        return ""
    return normalize_whitespace("".join(title_node.itertext()))


def _extract_authors(article: ET.Element) -> str:
    names: list[str] = []
    for author in article.findall(".//MedlineCitation/Article/AuthorList/Author"):
        last = normalize_whitespace((author.findtext("LastName", default="") or ""))
        fore = normalize_whitespace((author.findtext("ForeName", default="") or ""))
        collective = normalize_whitespace((author.findtext("CollectiveName", default="") or ""))
        if collective:
            names.append(collective)
            continue
        name = normalize_whitespace(f"{fore} {last}".strip())
        if name:
            names.append(name)
    return "; ".join(names)


async def search_pubmed(
    client: HTTPClient, *, query: str, limit: int, settings: Settings
) -> list[Paper]:
    base_params = _param_identity(settings)
    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(int(limit)),
        "retmode": "json",
    }
    esearch_params.update(base_params)

    esearch_data = await client.get_json(f"{EUTILS_BASE}/esearch.fcgi", params=esearch_params)
    ids = ((esearch_data.get("esearchresult") or {}).get("idlist")) or []
    ids = [i for i in ids if i]
    if not ids:
        return []

    efetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
    efetch_params.update(base_params)

    efetch_xml = await client.get_text(f"{EUTILS_BASE}/efetch.fcgi", params=efetch_params)

    try:
        root = safe_xml_fromstring(efetch_xml)
    except Exception:
        return []

    papers: list[Paper] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = normalize_whitespace((article.findtext(".//MedlineCitation/PMID", default="") or ""))
        title = _extract_title(article)
        abstract = _extract_abstract(article)
        doi = _extract_doi(article)
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        authors = _extract_authors(article)

        papers.append(
            Paper(
                title=title,
                abstract=abstract,
                url=url,
                doi=doi,
                authors=authors,
                source_platform="PubMed",
            )
        )
    return papers[:limit]
