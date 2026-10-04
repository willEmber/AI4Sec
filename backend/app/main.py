from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.services.llm_gateway import get_registry
from app.db import database
from app.rate_limit import limiter
from app.services.identity import AGENT_TOKEN_HEADER


def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    return Response(
        content=f'{{"detail":"Rate limit exceeded: {exc.detail}"}}',
        status_code=429,
        media_type="application/json",
    )


def _setup_logging() -> None:
    """Configure logging so all scholar.* loggers output at INFO level."""
    fmt = "%(asctime)s [%(name)s] %(levelname)s %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, stream=sys.stdout, force=True)
    # Ensure our loggers are at INFO even if root is higher
    for name in (
        "scholar",
        "scholar.runs",
        "scholar.graph",
        "scholar.llm",
        "scholar.mineru",
        "scholar.papers",
        "scholar.traffic",
    ):
        logging.getLogger(name).setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    logger = logging.getLogger("scholar")
    logger.info("Scholar Platform starting up...")

    settings = get_settings()
    data_dir = settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "papers").mkdir(exist_ok=True)

    database.configure(
        settings.database_url,
        min_size=settings.database_pool_min,
        max_size=settings.database_pool_max,
    )
    await database.init_db()
    logger.info("Database ready")
    if settings.auth_mode == "single_user":
        from app.services.accounts import ensure_local_principal

        await ensure_local_principal()
        logger.info("Auth: single_user — every request is the local principal")
    else:
        from app.services.accounts import admin_login_enabled, ensure_admin_account
        from app.services.oauth_providers import enabled_providers

        await ensure_admin_account()
        providers = [p.id for p in enabled_providers()]
        if admin_login_enabled():
            providers.append("admin")
        logger.info(
            f"Auth: multi_user providers={providers or '(none)'} "
            f"anonymous={'on' if settings.auth_allow_anonymous else 'off'}"
        )
        if not providers and not settings.auth_allow_anonymous:
            logger.warning("No login provider is configured and anonymous use is off: nobody can sign in")
    registry = get_registry()
    logger.info(
        f"LLM base_url={settings.llm_base_url} "
        f"default_model={registry.default_model or '(none)'} "
        f"utility_model={registry.utility_model or '(none)'} "
        f"models={registry.selectable_models()} agent_models={registry.agent_models()}"
    )

    # The agent checkpointer holds its own small pool for the process. Failing
    # to open it must not stop the rest of the app, which does not depend on
    # it — the agent routes report the failure instead.
    try:
        from app.services.agent_runner import open_agent_checkpointer

        await open_agent_checkpointer()
    except Exception:
        logger.exception("Agent checkpointer unavailable; agent sessions will not persist")

    # Importing the agent stack takes tens of seconds on a cold process. In a
    # background thread it costs nothing anyone waits for; on the request path
    # it costs every concurrent request.
    async def _warm_agent() -> None:
        try:
            from app.services.agent_runner import warm_up_agent

            await warm_up_agent()
            logger.info("Agent stack warmed up")
        except Exception:
            logger.exception("Agent warm-up failed; the first turn will be slow")

    warm_task = asyncio.create_task(_warm_agent())

    # Whatever the previous process was doing when it stopped is still recorded
    # as in progress. The sweep closes turns nobody is running any more and
    # rejoins parses that outlived their worker, so a restart costs a repeated
    # question rather than a repeated parse.
    try:
        from app.services import agent_worker

        await agent_worker.start()
    except Exception:
        logger.exception("Agent recovery worker did not start; interrupted runs will linger")

    yield

    warm_task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await warm_task

    try:
        from app.services import agent_worker

        await agent_worker.stop()
    except Exception:
        logger.exception("Failed to stop the agent recovery worker")

    try:
        from app.services import parse_service

        await parse_service.stop_background_parses()
    except Exception:
        logger.exception("Failed to stop background parses")

    try:
        from app.services.agent_runner import close_agent_checkpointer

        await close_agent_checkpointer()
    except Exception:
        logger.exception("Failed to close the agent checkpointer")

    await database.close_pool()
    logger.info("Scholar Platform shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    # Interactive docs / openapi schema leak the full API surface; allow turning
    # them off in production via ENABLE_DOCS=false (kept on by default for dev).
    if settings.enable_docs:
        app = FastAPI(title="Scholar Platform", version="0.1.0", lifespan=lifespan)
    else:
        app = FastAPI(
            title="Scholar Platform",
            version="0.1.0",
            lifespan=lifespan,
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
        )

    # Rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS: the login cookie only ever travels same-origin, through the Next.js
    # proxy, so cross-origin requests need no credentials. The one cross-origin
    # call — the SSE stream, a plain GET straight to the backend — carries a
    # short-lived stream ticket in its URL instead. Keeping
    # allow_credentials=False also neutralises the unsafe "*" + credentials
    # combination should cors_origins ever be set to ["*"].
    # `expose_headers` is what lets a cross-origin client read the agent
    # credential the server mints on a first request; without it the browser
    # hides the header and the client would ask for a new principal every call.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=[AGENT_TOKEN_HEADER],
    )

    from app.api.admin import router as admin_router
    from app.api.agent import router as agent_router
    from app.api.auth import router as auth_router
    from app.api.library import router as library_router
    from app.api.papers import router as papers_router
    from app.api.runs import router as runs_router
    from app.api.system import router as system_router

    app.include_router(papers_router, prefix="/api")
    app.include_router(runs_router, prefix="/api")
    app.include_router(admin_router, prefix="/api")
    app.include_router(system_router, prefix="/api")
    app.include_router(library_router, prefix="/api")
    app.include_router(agent_router, prefix="/api")
    app.include_router(auth_router, prefix="/api")

    return app


app = create_app()
