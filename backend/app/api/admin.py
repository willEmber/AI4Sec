"""Administrative endpoints (rank cache cleanup, etc.).

These routes perform destructive cache operations. They are gated behind an
optional ``ADMIN_API_TOKEN``: when that env var is set, every request must carry
a matching ``X-Admin-Token`` header. When it is unset (the default) the routes
stay open so existing trusted/local deployments keep working — set the token
before exposing the API to the public internet.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel

from app.config import get_settings
from app.rate_limit import limiter
from app.services.publication_rank import RankCache

logger = logging.getLogger("scholar.admin")


async def require_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    """Require a matching ``X-Admin-Token`` header when ADMIN_API_TOKEN is set."""
    expected = get_settings().admin_api_token
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Admin token required")


router = APIRouter(
    tags=["admin"], prefix="/admin", dependencies=[Depends(require_admin_token)]
)


class RankCacheClearResponse(BaseModel):
    deleted: int
    only_failures: bool
    remaining: int


class RankCacheDeleteResponse(BaseModel):
    deleted: int
    name: str


class RankCacheStatsResponse(BaseModel):
    total: int
    failures: int


@router.get("/rank-cache/stats", response_model=RankCacheStatsResponse)
@limiter.limit("30/minute")
async def rank_cache_stats(request: Request) -> RankCacheStatsResponse:
    """统计当前 publication_rank 缓存条目总数与失败条目数。"""
    cache = RankCache()
    await cache.init()
    try:
        total = await cache.count()
        failures = await cache.count(only_failures=True)
    finally:
        await cache.close()
    return RankCacheStatsResponse(total=total, failures=failures)


@router.delete("/rank-cache", response_model=RankCacheClearResponse)
@limiter.limit("6/minute")
async def clear_rank_cache(
    request: Request,
    only_failures: bool = Query(
        False,
        description="True 时仅清除失败的负缓存；False 时清空全部缓存",
    ),
) -> RankCacheClearResponse:
    """清空（或部分清空）publication_rank 缓存。"""
    cache = RankCache()
    await cache.init()
    try:
        deleted = await cache.clear(only_failures=only_failures)
        remaining = await cache.count()
    finally:
        await cache.close()
    logger.info(
        "admin: rank-cache cleared deleted=%d only_failures=%s remaining=%d",
        deleted, only_failures, remaining,
    )
    return RankCacheClearResponse(
        deleted=deleted, only_failures=only_failures, remaining=remaining,
    )


@router.delete("/rank-cache/{name}", response_model=RankCacheDeleteResponse)
@limiter.limit("30/minute")
async def delete_rank_cache_entry(request: Request, name: str) -> RankCacheDeleteResponse:
    """删除单个出版物名称对应的缓存（按归一化名称匹配）。"""
    cache = RankCache()
    await cache.init()
    try:
        deleted = await cache.delete(name)
    finally:
        await cache.close()
    logger.info("admin: rank-cache delete name=%r deleted=%d", name, deleted)
    return RankCacheDeleteResponse(deleted=deleted, name=name)
