"""`paper_search` settings as the backend runs it.

The search package reads its own `PAPERSEARCH_*` variables, but the model it
reranks with is the backend's: same gateway, same key, `RERANK_MODELNAME`.
Sphere and the agent's search tool both build their settings here, so the two
cannot drift onto different rerank configurations.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.config import get_settings
from app.services.paper_search import Settings as SearchSettings
from app.services.paper_search import load_env_file

_ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"


def backend_search_settings() -> SearchSettings:
    load_env_file(str(_ROOT_ENV))
    base = SearchSettings.from_env()
    cfg = get_settings()
    return replace(
        base,
        llm_base_url=base.llm_base_url or cfg.llm_base_url,
        llm_api_key=base.llm_api_key or cfg.llm_api_key,
        rerank_model=base.rerank_model or cfg.rerank_model,
    )
