from __future__ import annotations

"""
Resolve an open-access (OA) fulltext URL from a DOI.

Provider order:
  1) Unpaywall (requires a contact email)
  2) CORE API v3 (requires an API key)

This module purposefully does *not* include a CLI entrypoint; the project exposes
one public CLI via `python -m papersdownload`.
"""

import re
from dataclasses import dataclass
from typing import Any, Mapping

import requests

from .credentials import env_first

_UNPAYWALL_EMAIL_SPLIT_RE = re.compile(r"[\\s,;]+")

_DOI_PREFIX_RE = re.compile(r"^https?://(dx\\.)?doi\\.org/", re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\\s+", " ", (text or "").strip())


def normalize_doi(doi: str) -> str:
    doi = (doi or "").strip()
    doi = re.sub(r"^doi:\\s*", "", doi, flags=re.IGNORECASE)
    doi = _DOI_PREFIX_RE.sub("", doi).strip()
    return doi.lower().strip().strip("/")


class HTTPStatusError(RuntimeError):
    def __init__(self, method: str, url: str, status_code: int, body: str) -> None:
        super().__init__(f"{method} {url} -> HTTP {status_code}: {body[:500]}")
        self.method = method
        self.url = url
        self.status_code = status_code
        self.body = body


def _merged_headers(
    base: Mapping[str, str] | None, extra: Mapping[str, str] | None
) -> dict[str, str]:
    merged: dict[str, str] = {}
    if base:
        merged.update(dict(base))
    if extra:
        merged.update(dict(extra))
    return merged


def _get_json(
    url: str,
    *,
    params: Mapping[str, str] | None = None,
    timeout: float = 10.0,
    headers: Mapping[str, str] | None = None,
) -> Any:
    resp = requests.get(
        url,
        params=params,
        headers=dict(headers or {}),
        timeout=timeout,
        allow_redirects=True,
    )
    if not resp.ok:
        raise HTTPStatusError("GET", url, resp.status_code, resp.text)
    return resp.json()


def _get_status_and_json(
    url: str,
    *,
    params: Mapping[str, str] | None = None,
    timeout: float = 10.0,
    headers: Mapping[str, str] | None = None,
) -> tuple[int, Any]:
    resp = requests.get(
        url,
        params=params,
        headers=dict(headers or {}),
        timeout=timeout,
        allow_redirects=True,
    )
    status = resp.status_code
    try:
        data = resp.json()
    except Exception:
        data = None
    return status, data


def unpaywall_find_pdf_url(
    *,
    doi: str,
    email: str,
    timeout: float = 10.0,
    headers: Mapping[str, str] | None = None,
) -> str | None:
    """
    Return the best OA PDF URL for a DOI via Unpaywall, or None.

    Notes:
    - Unpaywall requires a (contact) email address passed as ?email=...
    - Prefer best_oa_location.url_for_pdf, then scan oa_locations[].url_for_pdf.
    """
    doi_norm = normalize_doi(doi)
    email = (email or "").strip()
    if not doi_norm or not email:
        return None

    url = f"https://api.unpaywall.org/v2/{doi_norm}"
    status, data = _get_status_and_json(
        url,
        params={"email": email},
        timeout=timeout,
        headers=headers,
    )
    if status == 404 or not isinstance(data, dict):
        return None

    best = data.get("best_oa_location") or {}
    pdf = normalize_whitespace(best.get("url_for_pdf") or "")
    if pdf:
        return pdf

    for loc in data.get("oa_locations") or []:
        pdf = normalize_whitespace((loc or {}).get("url_for_pdf") or "")
        if pdf:
            return pdf

    return None


def core_find_oa_url(
    *,
    doi: str,
    title: str = "",
    api_key: str,
    timeout: float = 10.0,
    headers: Mapping[str, str] | None = None,
) -> str | None:
    """
    Best-effort OA fulltext URL discovery via CORE API v3.

    Preference order within the first result:
      - downloadUrl
      - sourceFulltextUrls (first item)
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return None

    doi_norm = normalize_doi(doi)
    query = doi_norm or normalize_whitespace(title)
    if not query:
        return None

    url = "https://api.core.ac.uk/v3/search/works/"
    params = {"q": query, "limit": "1", "offset": "0", "sort": "relevance"}
    req_headers = _merged_headers(headers, {"Authorization": f"Bearer {api_key}"})

    try:
        data = _get_json(url, params=params, timeout=timeout, headers=req_headers)
    except Exception:
        return None

    results = (data or {}).get("results") or []
    if not results:
        return None

    first = results[0] or {}
    download_url = normalize_whitespace(first.get("downloadUrl") or "")
    if download_url:
        return download_url

    fulltext_urls = first.get("sourceFulltextUrls")
    if isinstance(fulltext_urls, str):
        candidate = normalize_whitespace(fulltext_urls)
        return candidate or None
    if isinstance(fulltext_urls, list):
        for u in fulltext_urls:
            candidate = normalize_whitespace(u or "")
            if candidate:
                return candidate

    return None


@dataclass
class OAResolveResult:
    doi: str
    oa_url: str | None
    source: str | None


def resolve_open_access_url(
    doi: str,
    *,
    title: str = "",
    unpaywall_email: str = "",
    core_api_key: str = "",
    timeout: float = 10.0,
    user_agent: str = "papersdownload/0.1",
) -> OAResolveResult:
    """
    Resolve an OA URL for the DOI using Unpaywall -> CORE order.
    """
    doi_norm = normalize_doi(doi)
    headers = {"User-Agent": user_agent}

    if unpaywall_email:
        try:
            oa = unpaywall_find_pdf_url(
                doi=doi_norm,
                email=unpaywall_email,
                timeout=timeout,
                headers=headers,
            )
        except Exception:
            oa = None
        if oa:
            return OAResolveResult(doi=doi_norm, oa_url=oa, source="unpaywall")

    if core_api_key:
        try:
            oa = core_find_oa_url(
                doi=doi_norm,
                title=title,
                api_key=core_api_key,
                timeout=timeout,
                headers=headers,
            )
        except Exception:
            oa = None
        if oa:
            return OAResolveResult(doi=doi_norm, oa_url=oa, source="core")

    return OAResolveResult(doi=doi_norm, oa_url=None, source=None)


def env_default_unpaywall_email() -> str:
    """
    获取 Unpaywall 邮箱（contact email）。

    出于安全考虑，本项目不再内置任何邮箱；请通过环境变量配置：
      - UNPAYWALL_EMAIL
      - PAPERSEARCH_UNPAYWALL_EMAIL（兼容旧变量名）
    """
    emails = env_default_unpaywall_emails()
    return emails[0] if emails else ""


def parse_unpaywall_emails(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = [p.strip() for p in _UNPAYWALL_EMAIL_SPLIT_RE.split(raw) if p.strip()]
    # de-dup, keep order
    return list(dict.fromkeys(parts))


def env_default_unpaywall_emails() -> list[str]:
    """
    Get Unpaywall contact emails as a list.

    Supports multi-email rotation via env like:
      UNPAYWALL_EMAIL=a@x.com,b@y.com,c@z.com
    """
    raw = env_first("UNPAYWALL_EMAIL", "PAPERSEARCH_UNPAYWALL_EMAIL")
    return parse_unpaywall_emails(raw)


def env_default_core_api_key() -> str:
    """
    获取 CORE API Key。

    出于安全考虑，本项目不再内置任何 API Key；请通过环境变量配置：
      - CORE_API_KEY
      - PAPERSEARCH_CORE_API_KEY（兼容旧变量名）
    """
    return env_first("CORE_API_KEY", "PAPERSEARCH_CORE_API_KEY")
