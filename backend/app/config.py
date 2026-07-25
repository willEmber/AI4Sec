from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    # --- paths ---
    data_dir: Path = Path("data")

    # --- LLM (Qwen Responses API) ---
    llm_base_url: str = Field(default="", alias="LLM_BASEURL")
    llm_api_key: str = Field(default="", alias="LLM_APIKEY")
    # May hold a comma-separated list of selectable models, e.g.
    # "qwen3.6-plus,qwen3.7-max". The first entry is the default; the full list
    # is offered to the frontend as a dropdown (see `thinking_models`).
    thinking_model: str = Field(default="", alias="THINKING_MODELNAME")
    embed_model: str = Field(default="", alias="EMBED_MODELNAME")
    rerank_model: str = Field(default="", alias="RERANK_MODELNAME")

    # --- Tavily web search (used by publication-rank fallback) ---
    tavily_api_key: str = Field(default="", alias="TAVILY_KEY")

    # --- Publication rank (EasyScholar) ---
    easyscholar_secret_key: str = Field(default="", alias="EASYSCHOLAR_SECRET_KEY")
    easyscholar_api_url: str = Field(
        default="https://www.easyscholar.cc/open/getPublicationRank",
        alias="EASYSCHOLAR_API_URL",
    )

    # --- MinerU ---
    mineru_token: str = Field(default="", alias="MINERU_TOKEN")
    mineru_model_version: str = Field(default="vlm", alias="MINERU_MODEL_VERSION")
    mineru_poll_interval_seconds: int = Field(default=6, alias="MINERU_POLL_INTERVAL_SECONDS")
    mineru_parse_timeout_seconds: int = Field(default=1800, alias="MINERU_PARSE_TIMEOUT_SECONDS")
    mineru_batch_timeout_seconds: int = Field(default=7200, alias="MINERU_BATCH_TIMEOUT_SECONDS")

    # --- Paper downloader (OA Resolver / Elsevier TDM / Wiley TDM) ---
    unpaywall_email: str = Field(default="", alias="UNPAYWALL_EMAIL")
    core_api_key: str = Field(default="", alias="CORE_API_KEY")
    elsevier_api_key: str = Field(default="", alias="ELSEVIER_API_KEY")
    elsevier_inst_token: str = Field(default="", alias="ELSEVIER_INSTTOKEN")
    wiley_tdm_token: str = Field(default="", alias="WILEY_TDM_TOKEN")

    # --- Insight Snap ---
    # Character budget for the triage context handed to the LLM. The old 12k
    # prefix-cut dropped results tables and conclusions on any long paper; 40k
    # (~10k tokens) fits comfortably in every model offered here.
    snap_context_budget_chars: int = Field(default=40_000, alias="SNAP_CONTEXT_BUDGET_CHARS")
    # Enrich the triage verdict with external evidence (venue rank, citation
    # impact, code availability, retraction status). Adds a few seconds of
    # network I/O per new paper; results are cached on the papers row.
    snap_signals_enabled: bool = Field(default=True, alias="SNAP_SIGNALS_ENABLED")
    # Probe resolved code repositories for stars / last-push date. Off by
    # default: unauthenticated GitHub API allows only 60 requests/hour per IP.
    snap_probe_repos: bool = Field(default=False, alias="SNAP_PROBE_REPOS")
    # Optional token for the repo probe above; lifts the rate limit to 5000/h.
    github_token: str = Field(default="", alias="GITHUB_TOKEN")
    # How long cached per-paper signals stay fresh. Citation counts drift slowly,
    # so a week avoids re-querying every index on each re-run. 0 disables caching.
    snap_signals_ttl_hours: int = Field(default=168, alias="SNAP_SIGNALS_TTL_HOURS")
    # Tally Semantic Scholar citation *intents* (background / methodology /
    # result) — one extra request that runs concurrently with the LLM calls.
    snap_citation_intents: bool = Field(default=True, alias="SNAP_CITATION_INTENTS")

    # --- GROBID (structured header extraction fallback) ---
    # Base URL of a GROBID server, e.g. http://localhost:8070. Empty disables it;
    # the regex DOI/arXiv scan over the first pages remains the primary path.
    grobid_url: str = Field(default="", alias="GROBID_URL")
    grobid_timeout_seconds: int = Field(default=60, alias="GROBID_TIMEOUT_SECONDS")

    # --- Research Sphere ---
    sphere_radius: int = Field(default=1, alias="SPHERE_RADIUS")
    sphere_candidate_cap: int = Field(default=200, alias="SPHERE_CANDIDATE_CAP")
    # Max candidates entering the LLM relevance gate (after quality gates)
    sphere_gate_cap: int = Field(default=120, alias="SPHERE_GATE_CAP")
    # Final core-set size (papers used for all synthesis/reporting)
    sphere_core_cap: int = Field(default=40, alias="SPHERE_CORE_CAP")
    sphere_pdf_parse_cap: int = Field(default=0, alias="SPHERE_PDF_PARSE_CAP")
    # Keyword-search-only candidates need at least this many citations or a
    # ranked venue to survive (they lack citation-graph evidence)
    sphere_t3_min_citations: int = Field(default=5, alias="SPHERE_T3_MIN_CITATIONS")
    # EasyScholar venue-rank lookup during scoring (cache-backed, no LLM fallback)
    sphere_venue_rank_enabled: bool = Field(default=True, alias="SPHERE_VENUE_RANK_ENABLED")
    # A merely-newest post-center-year paper needs at least this many citations
    # (or a ranked venue) to claim a "frontier" seat — otherwise low-impact
    # recent citers crowd out real cutting-edge follow-ups
    sphere_frontier_min_citations: int = Field(default=10, alias="SPHERE_FRONTIER_MIN_CITATIONS")

    # --- Dify knowledge base (self-hosted proxy) ---
    # Base URL of the Dify knowledge API proxy (e.g. http://8.217.68.153:3002).
    # Empty disables every library feature (see `dify_enabled`); the proxy holds
    # the real Dify API key server-side, so no key is configured here.
    dify_api_base: str = Field(default="", alias="DIFY_API_BASE")
    # Optional explicit dataset id. When empty, the proxy's own default dataset
    # (DIFY_DEFAULT_DATASET_ID on the proxy) is used via the short `/api/...` paths.
    dify_default_dataset_id: str = Field(default="", alias="DIFY_DEFAULT_DATASET_ID")
    # Default retrieval mode. `full_text_search` is fast (~1.5s); `semantic_search`
    # / `hybrid_search` are higher quality but slow (8B reranker, tens of seconds).
    dify_search_method: str = Field(default="full_text_search", alias="DIFY_SEARCH_METHOD")
    dify_timeout_seconds: int = Field(default=90, alias="DIFY_TIMEOUT_SECONDS")
    # How many library candidates Research Sphere pulls per run (0 disables the
    # library channel in Sphere while leaving the standalone library API on).
    dify_sphere_top_k: int = Field(default=10, alias="DIFY_SPHERE_TOP_K")

    # --- R2 / S3-compatible object storage (figure hosting for exports) ---
    # When fully configured, figures embedded in reports are uploaded here and
    # referenced by public URL in the Zotero note (external images render in
    # Zotero notes; base64 data URIs do not sync and hit note-size limits).
    r2_account_id: str = Field(default="", alias="R2_ACCOUNT_ID")
    r2_access_key_id: str = Field(default="", alias="R2_ACCESS_KEY_ID")
    r2_secret_access_key: str = Field(default="", alias="R2_SECRET_ACCESS_KEY")
    r2_bucket: str = Field(default="", alias="R2_BUCKET")
    r2_endpoint: str = Field(default="", alias="R2_ENDPOINT")
    r2_public_base_url: str = Field(default="", alias="R2_PUBLIC_BASE_URL")

    # --- server ---
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]

    # --- security / ops ---
    # When set, /api/admin/* requires a matching `X-Admin-Token` header.
    # Empty (default) keeps admin routes open for backward compatibility on
    # trusted/local deployments — set it before exposing the API publicly.
    admin_api_token: str = Field(default="", alias="ADMIN_API_TOKEN")
    # Swagger UI / ReDoc / openapi.json. Safe to leave on for local dev; set
    # ENABLE_DOCS=false to stop leaking the full API surface in production.
    enable_docs: bool = Field(default=True, alias="ENABLE_DOCS")

    model_config = {
        "env_file": str(Path(__file__).resolve().parents[2] / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @property
    def thinking_models(self) -> list[str]:
        """Selectable thinking models, parsed from the comma-separated env var."""
        return [m.strip() for m in self.thinking_model.split(",") if m.strip()]

    @property
    def default_thinking_model(self) -> str:
        """First configured model — used when the caller does not pick one."""
        models = self.thinking_models
        return models[0] if models else ""

    @property
    def dify_enabled(self) -> bool:
        """Whether the Dify knowledge base integration is configured."""
        return bool(self.dify_api_base.strip())

    @property
    def r2_enabled(self) -> bool:
        """Whether object storage is fully configured for figure hosting."""
        return all(
            v.strip()
            for v in (
                self.r2_access_key_id,
                self.r2_secret_access_key,
                self.r2_bucket,
                self.r2_endpoint,
                self.r2_public_base_url,
            )
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
