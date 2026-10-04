from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, Request

from app.services.llm_gateway import get_registry
from app.db import database as db
from app.models.schemas import (
    ModelListResponse,
    TrafficVisitCreate,
    TrafficVisitResponse,
)
from app.rate_limit import limiter

router = APIRouter(tags=["system"])
logger = logging.getLogger("scholar.traffic")


@router.get("/models", response_model=ModelListResponse)
@limiter.limit("60/minute")
async def list_models(request: Request) -> ModelListResponse:
    """Return the selectable LLM models and the defaults, by name only.

    ``models`` is what the report modes offer; ``agent_models`` is the subset
    verified for conversations. Which gateway serves a model is not exposed.
    """
    registry = get_registry()
    return ModelListResponse(
        models=registry.selectable_models(),
        default=registry.default_model,
        agent_models=registry.agent_models(),
        agent_default=registry.agent_default_model,
    )


@router.post("/traffic/visit", response_model=TrafficVisitResponse)
@limiter.limit("120/minute")
async def record_traffic_visit(
    request: Request,
    visit: TrafficVisitCreate,
) -> TrafficVisitResponse:
    """Record an anonymous browser visit and log cumulative UV/PV totals."""
    visitor_hash = hashlib.sha256(visit.owner_token.encode("utf-8")).hexdigest()
    totals = await db.record_traffic_visit(visitor_hash, visit.path)
    logger.info(
        "traffic: visit path=%r new_user=%s unique_users=%d total_visits=%d",
        visit.path,
        totals["new_user"],
        totals["unique_users"],
        totals["total_visits"],
    )
    return TrafficVisitResponse()
