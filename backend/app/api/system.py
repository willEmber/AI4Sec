from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import get_settings
from app.models.schemas import ModelListResponse
from app.rate_limit import limiter

router = APIRouter(tags=["system"])


@router.get("/models", response_model=ModelListResponse)
@limiter.limit("60/minute")
async def list_models(request: Request) -> ModelListResponse:
    """Return the selectable LLM models (from THINKING_MODELNAME) and the default.

    THINKING_MODELNAME may be a comma-separated list, e.g.
    ``qwen3.6-plus,qwen3.7-max``. The frontend renders these as a dropdown so the
    user picks instead of typing a model name.
    """
    settings = get_settings()
    return ModelListResponse(
        models=settings.thinking_models,
        default=settings.default_thinking_model,
    )
