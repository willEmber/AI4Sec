from __future__ import annotations

"""Paper PDF downloader integrated directly into the backend.

Supports three OA / TDM strategies (Europe PMC and Sci-Hub intentionally
excluded), tried in order until one succeeds:

  1. OA Resolver — Unpaywall (preferred) then CORE
  2. Elsevier TDM — ScienceDirect Article Retrieval API
  3. Wiley TDM   — Wiley Online Library TDM API (Wiley DOIs only)

All HTTP I/O is async via httpx. PDFs are written atomically via ``*.part``
rename. Per-attempt timeout and bounded retries with backoff are applied
uniformly; 429 responses trigger a longer cool-down.
"""

import asyncio
import ipaddress
import logging
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import quote, urlsplit

import httpx

logger = logging.getLogger("scholar.downloader")

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"
CORE_SEARCH_URL = "https://api.core.ac.uk/v3/search/works/"
ELSEVIER_ARTICLE_URL = "https://api.elsevier.com/content/article/doi/{doi}"
WILEY_TDM_URL = "https://api.wiley.com/onlinelibrary/tdm/v1/articles/{doi}"

_DOI_PREFIX_RE = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)
_DOI_SCHEME_RE = re.compile(r"^doi:\s*", re.IGNORECASE)
_FILENAME_FORBIDDEN_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
_WHITESPACE_RE = re.compile(r"\s+")


# ─────────────────────────────────────────────────────────────────────
# Common helpers
# ─────────────────────────────────────────────────────────────────────

def normalize_doi(raw: str) -> str:
    """Strip URL/scheme prefixes and lower-case the DOI."""
    doi = (raw or "").strip()
    doi = _DOI_SCHEME_RE.sub("", doi)
    doi = _DOI_PREFIX_RE.sub("", doi)
    return doi.strip().strip("/").lower()


def safe_filename(stem: str, *, max_len: int = 180) -> str:
    """Sanitize a string for use as a filename stem (no extension)."""
    stem = (stem or "").strip()
    stem = _FILENAME_FORBIDDEN_RE.sub("_", stem)
    stem = stem.replace(" ", "_")
    stem = re.sub(r"_+", "_", stem).strip("_")
    if not stem:
        stem = "paper"
    return stem[:max_len]


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", (text or "").strip())


def _looks_like_pdf_bytes(data: bytes) -> bool:
    return data.lstrip().startswith(b"%PDF")


def looks_like_wiley_doi(doi: str) -> bool:
    """Heuristic to avoid calling Wiley TDM for obviously non-Wiley DOIs."""
    d = (doi or "").strip().lower()
    return d.startswith("10.1002/") or d.startswith("10.1111/")


@dataclass(frozen=True)
class DownloadResult:
    doi: str
    ok: bool
    source: str
    pdf_path: str | None
    detail: str | None


# ─────────────────────────────────────────────────────────────────────
# SSRF guard — externally-supplied URLs must resolve to a public host
# ─────────────────────────────────────────────────────────────────────

class _RateLimited(Exception):
    """Internal signal that a hop returned HTTP 429 (retry with backoff)."""


def _ip_is_blocked(ip_text: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_text)
    except ValueError:
        return True  # unparseable → treat as unsafe
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def _url_is_public(url: str) -> bool:
    """True only for http(s) URLs whose host resolves entirely to public IPs.

    Downloaded PDF URLs come from third-party OA databases (Unpaywall / CORE /
    publisher TDM APIs), so a poisoned record could otherwise point the backend
    at an internal address (cloud metadata endpoints, localhost services).
    Resolve the host and reject if *any* resolved address is
    private/loopback/link-local/reserved/multicast/unspecified.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    host = parts.hostname
    if not host:
        return False
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return False
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    if not infos:
        return False
    return not any(_ip_is_blocked(info[4][0]) for info in infos)


# ─────────────────────────────────────────────────────────────────────
# Streaming write — shared by all strategies
# ─────────────────────────────────────────────────────────────────────

async def _stream_pdf_to_path(
    client: httpx.AsyncClient,
    url: str,
    dest_path: Path,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    retries: int = 3,
    backoff_seconds: float = 1.5,
    max_redirects: int = 5,
) -> tuple[bool, str | None]:
    """Stream a URL into ``dest_path``. Returns (ok, error_detail).

    Redirects are followed manually so the SSRF guard (:func:`_url_is_public`)
    re-checks every hop — httpx's automatic redirects would otherwise let a
    public URL bounce to an internal one unchecked.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    req_headers = dict(headers or {})
    last_err: str | None = None
    retries = max(1, int(retries))

    for attempt in range(1, retries + 1):
        try:
            current_url = url
            redirects = 0
            while True:
                if not await asyncio.to_thread(_url_is_public, current_url):
                    raise ValueError(f"Refusing to fetch non-public URL: {current_url!r}")
                async with client.stream(
                    "GET",
                    current_url,
                    headers=req_headers,
                    timeout=timeout,
                    follow_redirects=False,
                ) as resp:
                    if resp.status_code in (301, 302, 303, 307, 308):
                        location = resp.headers.get("Location")
                        if not location:
                            raise ValueError("Redirect without Location header")
                        redirects += 1
                        if redirects > max_redirects:
                            raise ValueError("Too many redirects")
                        current_url = str(resp.url.join(location))
                        continue
                    if resp.status_code == 429:
                        raise _RateLimited()
                    resp.raise_for_status()

                    aiter = resp.aiter_bytes(chunk_size=1024 * 1024)
                    first = b""
                    async for chunk in aiter:
                        first = chunk
                        break
                    if not first:
                        raise ValueError("Empty response body")

                    content_type = (resp.headers.get("Content-Type") or "").lower()
                    if "application/pdf" not in content_type and not _looks_like_pdf_bytes(first[:2048]):
                        raise ValueError(f"Not a PDF (Content-Type={content_type!r})")

                    with open(tmp_path, "wb") as f:
                        f.write(first)
                        async for chunk in aiter:
                            if chunk:
                                f.write(chunk)
                    break  # body read — leave the redirect loop

            os.replace(tmp_path, dest_path)
            return True, None

        except _RateLimited:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            last_err = "HTTP 429 (rate limited)"
            if attempt < retries:
                await asyncio.sleep(min(60.0, backoff_seconds * attempt * 2))
        except Exception as e:
            last_err = str(e)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            if attempt < retries:
                await asyncio.sleep(min(60.0, backoff_seconds * attempt))

    return False, last_err


# ─────────────────────────────────────────────────────────────────────
# Strategy 1: OA Resolver (Unpaywall → CORE)
# ─────────────────────────────────────────────────────────────────────

async def _unpaywall_find_pdf_url(
    client: httpx.AsyncClient,
    doi: str,
    *,
    email: str,
    timeout: float = 10.0,
) -> str | None:
    email = (email or "").strip()
    if not doi or not email:
        return None

    try:
        resp = await client.get(
            UNPAYWALL_URL.format(doi=doi),
            params={"email": email},
            timeout=timeout,
            follow_redirects=True,
        )
    except httpx.HTTPError as e:
        logger.debug("unpaywall error for %s: %s", doi, e)
        return None
    if resp.status_code == 404:
        return None
    if not resp.is_success:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    best = data.get("best_oa_location") or {}
    pdf = _normalize_whitespace(best.get("url_for_pdf") or "")
    if pdf:
        return pdf

    for loc in data.get("oa_locations") or []:
        pdf = _normalize_whitespace((loc or {}).get("url_for_pdf") or "")
        if pdf:
            return pdf
    return None


async def _core_find_oa_url(
    client: httpx.AsyncClient,
    doi: str,
    *,
    api_key: str,
    title: str = "",
    timeout: float = 10.0,
) -> str | None:
    api_key = (api_key or "").strip()
    if not api_key:
        return None
    query = doi or _normalize_whitespace(title)
    if not query:
        return None

    try:
        resp = await client.get(
            CORE_SEARCH_URL,
            params={"q": query, "limit": "1", "offset": "0", "sort": "relevance"},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            follow_redirects=True,
        )
    except httpx.HTTPError as e:
        logger.debug("core error for %s: %s", doi, e)
        return None
    if not resp.is_success:
        return None
    try:
        data = resp.json()
    except ValueError:
        return None

    results = (data or {}).get("results") or []
    if not results:
        return None
    first = results[0] or {}

    download_url = _normalize_whitespace(first.get("downloadUrl") or "")
    if download_url:
        return download_url

    fulltext = first.get("sourceFulltextUrls")
    if isinstance(fulltext, str):
        url = _normalize_whitespace(fulltext)
        return url or None
    if isinstance(fulltext, list):
        for u in fulltext:
            url = _normalize_whitespace(u or "")
            if url:
                return url
    return None


async def download_via_oa_resolver(
    client: httpx.AsyncClient,
    doi: str,
    dest_path: Path,
    *,
    unpaywall_email: str,
    core_api_key: str,
    title: str = "",
    timeout: float = 60.0,
    retries: int = 3,
) -> DownloadResult | None:
    """Resolve an OA PDF URL (Unpaywall → CORE) and download it.

    Returns ``None`` when no resolver is configured (no email and no key).
    """
    if not unpaywall_email and not core_api_key:
        return None

    source: str | None = None
    oa_url: str | None = None

    if unpaywall_email:
        oa_url = await _unpaywall_find_pdf_url(
            client, doi, email=unpaywall_email, timeout=min(timeout, 15.0)
        )
        if oa_url:
            source = "unpaywall"

    if not oa_url and core_api_key:
        oa_url = await _core_find_oa_url(
            client, doi, api_key=core_api_key, title=title, timeout=min(timeout, 15.0)
        )
        if oa_url:
            source = "core"

    if not oa_url:
        return None

    ok, err = await _stream_pdf_to_path(
        client, oa_url, dest_path, timeout=timeout, retries=retries
    )
    return DownloadResult(
        doi=doi,
        ok=ok,
        source=source or "oa_resolver",
        pdf_path=str(dest_path) if ok else None,
        detail=oa_url if ok else err,
    )


# ─────────────────────────────────────────────────────────────────────
# Strategy 2: Elsevier TDM
# ─────────────────────────────────────────────────────────────────────

async def download_via_elsevier_tdm(
    client: httpx.AsyncClient,
    doi: str,
    dest_path: Path,
    *,
    api_key: str,
    inst_token: str = "",
    timeout: float = 60.0,
    retries: int = 3,
) -> DownloadResult | None:
    api_key = (api_key or "").strip()
    if not api_key:
        return None

    headers: dict[str, str] = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/pdf",
    }
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token

    url = ELSEVIER_ARTICLE_URL.format(doi=doi)
    ok, err = await _stream_pdf_to_path(
        client, url, dest_path, headers=headers, timeout=timeout, retries=retries
    )
    return DownloadResult(
        doi=doi,
        ok=ok,
        source="elsevier_tdm",
        pdf_path=str(dest_path) if ok else None,
        detail=None if ok else err,
    )


# ─────────────────────────────────────────────────────────────────────
# Strategy 3: Wiley TDM
# ─────────────────────────────────────────────────────────────────────

async def download_via_wiley_tdm(
    client: httpx.AsyncClient,
    doi: str,
    dest_path: Path,
    *,
    token: str,
    timeout: float = 60.0,
    retries: int = 3,
    require_wiley_doi: bool = True,
) -> DownloadResult | None:
    token = (token or "").strip()
    if not token:
        return None
    if require_wiley_doi and not looks_like_wiley_doi(doi):
        return None

    url = WILEY_TDM_URL.format(doi=quote(doi, safe=""))
    headers = {
        "Wiley-TDM-Client-Token": token,
        "Accept": "application/pdf",
    }
    ok, err = await _stream_pdf_to_path(
        client, url, dest_path, headers=headers, timeout=timeout, retries=retries
    )
    return DownloadResult(
        doi=doi,
        ok=ok,
        source="wiley_tdm",
        pdf_path=str(dest_path) if ok else None,
        detail=None if ok else err,
    )


# ─────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DownloaderCredentials:
    unpaywall_email: str = ""
    core_api_key: str = ""
    elsevier_api_key: str = ""
    elsevier_inst_token: str = ""
    wiley_tdm_token: str = ""


async def download_paper(
    doi: str,
    dest_path: Path,
    *,
    credentials: DownloaderCredentials,
    title: str = "",
    client: httpx.AsyncClient | None = None,
    timeout: float = 60.0,
    retries: int = 3,
) -> DownloadResult:
    """Try OA Resolver → Elsevier TDM → Wiley TDM in order.

    Re-uses an existing ``httpx.AsyncClient`` when provided; otherwise opens a
    short-lived one. If ``dest_path`` already exists and is non-empty, returns
    immediately without hitting the network.
    """
    doi_norm = normalize_doi(doi)
    if not doi_norm:
        return DownloadResult(
            doi=doi, ok=False, source="input", pdf_path=None, detail="Empty DOI"
        )

    dest_path = Path(dest_path)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return DownloadResult(
            doi=doi_norm,
            ok=True,
            source="local",
            pdf_path=str(dest_path),
            detail="Already exists",
        )

    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            headers={"User-Agent": DEFAULT_USER_AGENT},
            timeout=timeout,
        )

    attempts: list[DownloadResult] = []
    try:
        oa = await download_via_oa_resolver(
            client,
            doi_norm,
            dest_path,
            unpaywall_email=credentials.unpaywall_email,
            core_api_key=credentials.core_api_key,
            title=title,
            timeout=timeout,
            retries=retries,
        )
        if oa and oa.ok:
            return oa
        if oa:
            attempts.append(oa)

        els = await download_via_elsevier_tdm(
            client,
            doi_norm,
            dest_path,
            api_key=credentials.elsevier_api_key,
            inst_token=credentials.elsevier_inst_token,
            timeout=timeout,
            retries=retries,
        )
        if els and els.ok:
            return els
        if els:
            attempts.append(els)

        wiley = await download_via_wiley_tdm(
            client,
            doi_norm,
            dest_path,
            token=credentials.wiley_tdm_token,
            timeout=timeout,
            retries=retries,
        )
        if wiley and wiley.ok:
            return wiley
        if wiley:
            attempts.append(wiley)
    finally:
        if owns_client:
            await client.aclose()

    if attempts:
        return attempts[-1]
    return DownloadResult(
        doi=doi_norm,
        ok=False,
        source="lookup",
        pdf_path=None,
        detail="No download path configured (set UNPAYWALL_EMAIL / CORE_API_KEY / ELSEVIER_API_KEY / WILEY_TDM_TOKEN)",
    )
