"""The Phoenix tracing UI, served to admins on the frontend's own origin.

The browser opens ``/api/admin/phoenix`` on the frontend; the ``/api`` rewrite
brings it here, where the admin check runs, and this forwards it to Phoenix
with the prefix stripped. Phoenix runs with
``PHOENIX_HOST_ROOT_PATH=/api/admin/phoenix``: it still serves from ``/`` but
writes every asset, API and route URL with that prefix, so the page keeps
coming back through this check. It sits under ``/api`` rather than a nicer
``/admin/phoenix`` because the login cookie is scoped to ``Path=/api``; a page
outside it would reach the backend without the admin's credential.

Phoenix runs without its own authentication — this check is the only lock, so
it must never be reachable from outside the backend's network. Traces hold the
full prompts, paper text and readers' messages included.

Forwarding is a whitelist: the reader's cookies, the admin token, a ``?token=``
credential and anything else Phoenix has no use for stay here, and Phoenix
cannot set cookies on our origin. WebSocket upgrades are not forwarded; viewing traces does not use them.
"""
from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from app.api.admin import require_admin_token
from app.config import get_settings

_REQUEST_HEADERS = {
    "accept",
    "accept-encoding",
    "accept-language",
    "cache-control",
    "content-type",
    "if-modified-since",
    "if-none-match",
    "range",
}
# Hop-by-hop headers, the ones uvicorn writes itself, and the one Phoenix could
# use to write on our origin.
_DROPPED_RESPONSE_HEADERS = {
    "date",
    "server",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "set-cookie",
}
_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

router = APIRouter(
    prefix="/admin/phoenix",
    tags=["admin"],
    dependencies=[Depends(require_admin_token)],
    include_in_schema=False,
)


def _client() -> httpx.AsyncClient:
    # Some GraphQL queries over a large trace table take a while.
    return httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))


@router.api_route("", methods=_METHODS)
@router.api_route("/{path:path}", methods=_METHODS)
async def phoenix_proxy(request: Request, path: str = "") -> StreamingResponse:
    base = get_settings().tracing_ui_url.strip().rstrip("/")
    if not base:
        raise HTTPException(status_code=404, detail="Tracing UI is not configured")

    headers = {k: v for k, v in request.headers.items() if k.lower() in _REQUEST_HEADERS}
    client = _client()
    upstream_request = client.build_request(
        request.method,
        f"{base}/{path}",
        # `?token=` is one of the ways a credential reaches us; it stays here.
        params=[(k, v) for k, v in request.query_params.multi_items() if k != "token"],
        headers=headers,
        content=await request.body(),
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Phoenix unreachable: {type(exc).__name__}") from exc

    async def _close() -> None:
        await upstream.aclose()
        await client.aclose()

    # Raw bytes: a gzip body is passed on as gzip, with its own encoding header.
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers={
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in _DROPPED_RESPONSE_HEADERS
        },
        background=BackgroundTask(_close),
    )
