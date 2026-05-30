"""Async client for the self-hosted Dify knowledge base proxy.

The proxy (``DIFY_API_BASE``, e.g. ``http://8.217.68.153:3002``) sits in front of
a Dify dataset and holds the real Dify API key server-side, so this client never
sends an Authorization header. Endpoints mirror the proxy's documented surface
(see ``/api-docs``): datasets, documents, segments, markdown, download URL, and
``POST /api/search``.

Path selection: when an effective dataset id is known (explicit ``dataset_id``
argument, else ``DIFY_DEFAULT_DATASET_ID``) the full ``/api/datasets/{id}/...``
paths are used; otherwise the short ``/api/...`` paths let the proxy apply its
own default dataset.

Retrieval note: ``full_text_search`` is fast (~1.5s) but unranked; the semantic /
hybrid methods are high quality but slow (8B reranker, tens of seconds), so the
configured ``dify_timeout_seconds`` defaults generously.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

from app.config import get_settings

logger = logging.getLogger("scholar.dify")

_VALID_SEARCH_METHODS = frozenset({"semantic_search", "full_text_search", "hybrid_search"})


class DifyError(RuntimeError):
    """Raised when the Dify proxy is unreachable or returns a non-2xx status.

    ``upstream_status`` is the proxy's HTTP status when the request reached it
    (``None`` for transport-level failures); ``detail`` is the parsed error body.
    """

    def __init__(self, message: str, *, upstream_status: int | None = None, detail: Any = None):
        super().__init__(message)
        self.message = message
        self.upstream_status = upstream_status
        self.detail = detail


def _base() -> str:
    settings = get_settings()
    base = settings.dify_api_base.strip().rstrip("/")
    if not base:
        raise DifyError("Dify knowledge base is not configured (DIFY_API_BASE is empty)")
    return base


def _resolve_dataset(dataset_id: str | None) -> str:
    """Effective dataset id: explicit arg wins, else the project default."""
    if dataset_id and dataset_id.strip():
        return dataset_id.strip()
    return get_settings().dify_default_dataset_id.strip()


def _timeout() -> httpx.Timeout:
    read = float(get_settings().dify_timeout_seconds)
    return httpx.Timeout(connect=15.0, read=read, write=30.0, pool=read)


async def _request(method: str, path: str, *, params: dict | None = None, json: dict | None = None) -> Any:
    url = f"{_base()}{path}"
    try:
        async with httpx.AsyncClient(timeout=_timeout()) as client:
            resp = await client.request(method, url, params=params, json=json)
    except httpx.HTTPError as exc:
        logger.warning("dify: transport error on %s %s — %s", method, path, exc)
        raise DifyError(f"Dify request failed: {exc}") from exc

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text[:500]
        logger.warning("dify: upstream %s on %s %s — %s", resp.status_code, method, path, detail)
        raise DifyError(
            "Dify request failed",
            upstream_status=resp.status_code,
            detail=detail,
        )

    try:
        return resp.json()
    except Exception as exc:
        raise DifyError(f"Dify returned a non-JSON response: {exc}") from exc


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(int(value), hi))


# ── Datasets / documents ──────────────────────────────────────────────────

async def list_datasets(page: int = 1, limit: int = 20) -> dict[str, Any]:
    return await _request(
        "GET", "/api/datasets",
        params={"page": _clamp(page, 1, 10_000), "limit": _clamp(limit, 1, 100)},
    )


async def list_documents(dataset_id: str | None = None, page: int = 1, limit: int = 20) -> dict[str, Any]:
    ds = _resolve_dataset(dataset_id)
    params = {"page": _clamp(page, 1, 100_000), "limit": _clamp(limit, 1, 100)}
    if ds:
        return await _request("GET", f"/api/datasets/{quote(ds, safe='')}/documents", params=params)
    return await _request("GET", "/api/documents", params=params)


async def get_document(document_id: str, dataset_id: str | None = None) -> dict[str, Any]:
    ds = _resolve_dataset(dataset_id)
    doc = quote(document_id, safe="")
    if ds:
        return await _request("GET", f"/api/datasets/{quote(ds, safe='')}/documents/{doc}")
    return await _request("GET", f"/api/documents/{doc}")


async def list_segments(
    document_id: str, dataset_id: str | None = None, page: int = 1, limit: int = 100
) -> dict[str, Any]:
    ds = _resolve_dataset(dataset_id)
    doc = quote(document_id, safe="")
    params = {"page": _clamp(page, 1, 100_000), "limit": _clamp(limit, 1, 100)}
    if ds:
        return await _request(
            "GET", f"/api/datasets/{quote(ds, safe='')}/documents/{doc}/segments", params=params
        )
    return await _request("GET", f"/api/documents/{doc}/segments", params=params)


async def get_markdown(document_id: str, dataset_id: str | None = None) -> dict[str, Any]:
    ds = _resolve_dataset(dataset_id)
    doc = quote(document_id, safe="")
    if ds:
        return await _request("GET", f"/api/datasets/{quote(ds, safe='')}/documents/{doc}/markdown")
    return await _request("GET", f"/api/documents/{doc}/markdown")


async def get_download_url(document_id: str, dataset_id: str | None = None) -> dict[str, Any]:
    ds = _resolve_dataset(dataset_id)
    doc = quote(document_id, safe="")
    if ds:
        return await _request("GET", f"/api/datasets/{quote(ds, safe='')}/documents/{doc}/download")
    return await _request("GET", f"/api/documents/{doc}/download")


# ── Search ────────────────────────────────────────────────────────────────

async def search(
    query: str,
    *,
    top_k: int = 10,
    score_threshold: float | None = None,
    search_method: str | None = None,
    dataset_id: str | None = None,
) -> dict[str, Any]:
    """Run a retrieval query. Returns the raw ``{"query", "records": [...]}`` dict.

    Each record is ``{document_id, document_name, segment_id, content, score, metadata}``.
    Raises :class:`DifyError` on transport / upstream failure (callers on the
    analysis pipeline should wrap this in try/except to stay fault-tolerant).
    """
    method = (search_method or get_settings().dify_search_method or "full_text_search").strip()
    if method not in _VALID_SEARCH_METHODS:
        method = "full_text_search"

    body: dict[str, Any] = {
        "query": query,
        "top_k": _clamp(top_k, 1, 100),
        "search_method": method,
    }
    if score_threshold is not None:
        body["score_threshold"] = float(score_threshold)

    ds = _resolve_dataset(dataset_id)
    if ds:
        return await _request("POST", f"/api/datasets/{quote(ds, safe='')}/search", json=body)
    return await _request("POST", "/api/search", json=body)


async def search_records(query: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Convenience wrapper returning just the ``records`` list (still raises)."""
    data = await search(query, **kwargs)
    records = data.get("records")
    return records if isinstance(records, list) else []
