from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    # --- paths ---
    data_dir: Path = Path("data")

    # --- Database (PostgreSQL) ---
    # Every business table and the agent checkpoints live here. The pool is
    # per process; with several uvicorn workers the server sees
    # workers x DATABASE_POOL_MAX connections, plus a small checkpoint pool each.
    database_url: str = Field(
        default="postgresql://scholar:scholar@localhost:5432/scholar", alias="DATABASE_URL"
    )
    database_pool_min: int = Field(default=1, alias="DATABASE_POOL_MIN")
    database_pool_max: int = Field(default=10, alias="DATABASE_POOL_MAX")

    # --- LLM (Qwen / DashScope MaaS Responses API) ---
    # Compatible-mode base URL ending in /v1; client POSTs to {base}/responses
    # with enable_thinking=true for every thinking model.
    llm_base_url: str = Field(default="", alias="LLM_BASEURL")
    llm_api_key: str = Field(default="", alias="LLM_APIKEY")
    # May hold a comma-separated list of selectable models, e.g.
    # "qwen3.8-max,qwen3.7-plus,qwen3.7-max". The first entry is the default;
    # the full list is offered to the frontend as a dropdown (see `thinking_models`).
    thinking_model: str = Field(default="", alias="THINKING_MODELNAME")
    embed_model: str = Field(default="", alias="EMBED_MODELNAME")
    rerank_model: str = Field(default="", alias="RERANK_MODELNAME")

    # --- Web search providers (app/services/web_search) ---
    # Each may hold one key or a comma-separated pool; the pool rotates past a
    # key that is refused, rate-limited or out of credit. The plural name comes
    # first; the singular spellings are accepted because deployed `.env` files
    # use them, and a key that is set but never read fails silently — the
    # publication-rank Tavily fallback read only `TAVILY_KEY` for that reason
    # and never ran.
    tavily_api_keys: str = Field(
        default="",
        validation_alias=AliasChoices("TAVILY_API_KEYS", "TAVILY_API_KEY", "TAVILY_KEY"),
    )
    exa_api_keys: str = Field(
        default="", validation_alias=AliasChoices("EXA_API_KEYS", "EXA_API_KEY")
    )
    firecrawl_api_keys: str = Field(
        default="",
        validation_alias=AliasChoices("FIRECRAWL_API_KEYS", "FIRECRAWL_API_KEY"),
    )
    # Off stops every web search and page read: queries leave the machine for
    # third-party services, which some deployments must not allow.
    web_search_enabled: bool = Field(default=True, alias="WEB_SEARCH_ENABLED")

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
    # Stop blocking a conversation on a remote queue; the submitted batch is
    # retained so a later turn can collect it without another paid submission.
    mineru_queue_timeout_seconds: int = Field(default=300, ge=0, alias="MINERU_QUEUE_TIMEOUT_SECONDS")
    mineru_parse_timeout_seconds: int = Field(default=1800, alias="MINERU_PARSE_TIMEOUT_SECONDS")
    mineru_batch_timeout_seconds: int = Field(default=7200, alias="MINERU_BATCH_TIMEOUT_SECONDS")
    # Start parsing a PDF when it is attached to a conversation instead of when
    # the agent first asks for it, so the reader's typing time is parse time.
    # Off means a paper nobody asks about is never billed.
    parse_on_attach: bool = Field(default=True, alias="PARSE_ON_ATTACH")

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

    # --- Logic Lens ---
    # After the deep report is written, extract a structured digest from it so
    # the run view can offer the same structured/Markdown toggle as Snap and
    # Sphere. One extra LLM call per run; off means markdown-only, as before.
    lens_digest_enabled: bool = Field(default=True, alias="LENS_DIGEST_ENABLED")

    # --- Paper agent (Deep Agents Smart Q&A) ---
    # Model for the conversational agent. Empty falls back to the first entry of
    # THINKING_MODELNAME. The gateway serves /responses only (no
    # /chat/completions), which app/agents/model_factory.py accounts for.
    agent_model: str = Field(default="", alias="AGENT_MODELNAME")
    # Per-request timeout handed to the model client. Agent turns carry long
    # tool transcripts, so this sits well above a plain chat call.
    agent_request_timeout_seconds: int = Field(default=300, alias="AGENT_REQUEST_TIMEOUT_SECONDS")
    # Transport-level retries inside the OpenAI SDK (connection resets, 5xx).
    agent_max_retries: int = Field(default=3, alias="AGENT_MAX_RETRIES")
    # HMAC secret signing anonymous agent credentials, event-stream tickets and
    # the OAuth login state. Empty generates one and
    # persists it under the data dir, so local dev needs no configuration;
    # set it explicitly in production, and across every instance, or a restart
    # invalidates every issued credential.
    agent_identity_secret: str = Field(default="", alias="AGENT_IDENTITY_SECRET")
    # Per-run ceilings, enforced by the executor rather than asked of the model.
    # Reaching one ends the turn with whatever was found, reported as such.
    agent_max_tool_calls: int = Field(default=40, alias="AGENT_MAX_TOOL_CALLS")
    # Each download is a PDF fetch; each parse is a MinerU job costing minutes
    # and quota. These are the two that cost real money, so they are the two
    # most worth tuning per deployment.
    agent_max_downloads: int = Field(default=5, alias="AGENT_MAX_DOWNLOADS")
    agent_max_parses: int = Field(default=3, alias="AGENT_MAX_PARSES")
    # Web searches and page reads per run. Each spends third-party credit and
    # sends the query off the machine; neither ends the turn when reached —
    # the tool refuses and the agent answers from what it has.
    agent_max_web_searches: int = Field(default=10, alias="AGENT_MAX_WEB_SEARCHES")
    agent_max_web_fetches: int = Field(default=8, alias="AGENT_MAX_WEB_FETCHES")
    agent_max_tokens: int = Field(default=400_000, alias="AGENT_MAX_TOKENS")
    # Generous because one turn may now download and parse a paper it just
    # found, which a pure reading turn never did.
    agent_max_wall_seconds: int = Field(default=1800, alias="AGENT_MAX_WALL_SECONDS")
    # Streamed answer text is buffered and written as one `message.delta` event
    # per window instead of one per model chunk. Each event is a committed row,
    # and per-chunk writes made the event log ~98% deltas and a turn ~450
    # commits — the dominant write load under concurrent readers.
    agent_delta_flush_ms: int = Field(default=150, alias="AGENT_DELTA_FLUSH_MS")
    agent_delta_flush_chars: int = Field(default=1500, alias="AGENT_DELTA_FLUSH_CHARS")
    # --- Context management ---
    # Conversation size (approximate tokens, system prompt and tool schemas
    # included) at which older turns are summarised. The gateway models have
    # no published profile, so this is an absolute number rather than a fraction
    # of the context window; 80k leaves room for a long section read on top.
    agent_context_trigger_tokens: int = Field(default=80_000, alias="AGENT_CONTEXT_TRIGGER_TOKENS")
    # How many recent messages survive a compaction untouched. Counted in
    # messages, not tokens, because a tool-call/tool-result pair must never be
    # split — the summariser keeps whole exchanges.
    agent_context_keep_messages: int = Field(default=12, alias="AGENT_CONTEXT_KEEP_MESSAGES")
    # Tool results older than the most recent N are cut down to a stub before
    # the model sees them again; a section read is tens of KB and is rarely
    # needed verbatim two questions later. Evidence ids survive the cut.
    agent_tool_result_keep: int = Field(default=6, alias="AGENT_TOOL_RESULT_KEEP")
    agent_tool_result_max_chars: int = Field(default=1_500, alias="AGENT_TOOL_RESULT_MAX_CHARS")
    # --- Long-term memory ---
    # Ceiling on memories injected into the prompt per turn (newest first).
    agent_memory_max_items: int = Field(default=30, alias="AGENT_MEMORY_MAX_ITEMS")

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

    # --- accounts (P7.5 I2) ---
    # `multi_user`: OAuth accounts, optionally beside anonymous trial use.
    # `single_user`: a self-hosted instance for one person — every request is
    # the same local principal and nobody logs in.
    auth_mode: str = Field(default="multi_user", alias="AUTH_MODE")
    # Whether a visitor may use the app before logging in (multi_user only).
    auth_allow_anonymous: bool = Field(default=True, alias="AUTH_ALLOW_ANONYMOUS")
    # The origin the browser sees (the Next.js frontend). OAuth callbacks are
    # built from it, so it must match what is registered with each provider.
    public_base_url: str = Field(default="http://localhost:3001", alias="PUBLIC_BASE_URL")
    # Cookies are `Secure` unless this is turned off for plain-http local dev.
    auth_cookie_secure: bool = Field(default=True, alias="AUTH_COOKIE_SECURE")
    # Login sessions last this long since they were last used.
    auth_session_days: int = Field(default=30, alias="AUTH_SESSION_DAYS")
    # A provider is offered once both its id and secret are set.
    oauth_github_client_id: str = Field(default="", alias="OAUTH_GITHUB_CLIENT_ID")
    oauth_github_client_secret: str = Field(default="", alias="OAUTH_GITHUB_CLIENT_SECRET")
    oauth_google_client_id: str = Field(default="", alias="OAUTH_GOOGLE_CLIENT_ID")
    oauth_google_client_secret: str = Field(default="", alias="OAUTH_GOOGLE_CLIENT_SECRET")
    oauth_linuxdo_client_id: str = Field(default="", alias="OAUTH_LINUXDO_CLIENT_ID")
    oauth_linuxdo_client_secret: str = Field(default="", alias="OAUTH_LINUXDO_CLIENT_SECRET")
    # A username/password login for one administrator account (multi_user
    # only), offered once both are set — for local testing and personal
    # deployments without an OAuth app. The account is unlimited by quotas;
    # changing either value logs out every session it has.
    admin_username: str = Field(default="", alias="ADMIN_USERNAME")
    admin_password: str = Field(default="", alias="ADMIN_PASSWORD")
    # Per-principal daily agent quotas (UTC day), counted from agent_runs.
    # 0 means unlimited; single_user mode is never limited.
    quota_anon_daily_runs: int = Field(default=20, alias="QUOTA_ANON_DAILY_RUNS")
    quota_anon_daily_tokens: int = Field(default=1_000_000, alias="QUOTA_ANON_DAILY_TOKENS")
    quota_user_daily_runs: int = Field(default=200, alias="QUOTA_USER_DAILY_RUNS")
    quota_user_daily_tokens: int = Field(default=10_000_000, alias="QUOTA_USER_DAILY_TOKENS")

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
