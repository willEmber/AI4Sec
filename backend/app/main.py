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
from app.db.database import init_db, set_db_path
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

    set_db_path(data_dir / "app.db")
    await init_db()
    logger.info(f"Database initialized at {data_dir / 'app.db'}")
    logger.info(
        f"LLM base_url={settings.llm_base_url} "
        f"default_model={settings.default_thinking_model or '(none)'} "
        f"models={settings.thinking_models}"
    )

    # The agent checkpointer holds one SQLite connection for the process: a
    # per-turn connection would serialise behind the previous turn's WAL
    # checkpoint. Failing to open it must not stop the rest of the app, which
    # does not depend on it — the agent routes report the failure instead.
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
        from app.services.agent_runner import close_agent_checkpointer

        await close_agent_checkpointer()
    except Exception:
        logger.exception("Failed to close the agent checkpointer")
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

    # CORS: the app authenticates with an owner_token carried in the query/body,
    # never via cookies, so credentials are not needed. Keeping
    # allow_credentials=False also neutralises the unsafe "*" + credentials
    # combination should cors_origins ever be set to ["*"]. In practice only the
    # cross-origin SSE stream (a plain GET straight to the backend) relies on
    # CORS — every other /api call is same-origin through the Next.js proxy.
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

    return app


app = create_app()
