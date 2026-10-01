"""PostgreSQL for tests: one throwaway schema per test.

Set `TEST_DATABASE_URL` to a database the tests may create schemas in, e.g.
`postgresql://postgres@localhost:5432/scholar_test`. Each test gets a fresh
schema, reached through the connection's `search_path`, so migrations, the
application pool and the agent checkpointer all land in it without knowing
they are under test; the schema is dropped afterwards.

Without `TEST_DATABASE_URL`, or when it cannot be reached, tests that need the
database are skipped with the reason — the summary line says how many.
"""

from __future__ import annotations

import os
import unittest
import uuid
from functools import lru_cache

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo

ENV_VAR = "TEST_DATABASE_URL"


@lru_cache(maxsize=1)
def _unavailable_reason() -> str:
    url = os.environ.get(ENV_VAR, "").strip()
    if not url:
        return f"{ENV_VAR} is not set; database tests need a PostgreSQL they may create schemas in"
    try:
        with psycopg.connect(url, connect_timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:  # noqa: BLE001 — any failure means "cannot run here"
        return f"{ENV_VAR} is not reachable: {exc}"
    return ""


def require_database() -> str:
    """The base URL, or raise `SkipTest` explaining why there is none."""
    reason = _unavailable_reason()
    if reason:
        raise unittest.SkipTest(reason)
    return os.environ[ENV_VAR].strip()


def url_for_schema(base_url: str, schema: str) -> str:
    """`base_url` with `search_path` pointed at `schema` (then `public`)."""
    params = conninfo_to_dict(base_url)
    existing = params.pop("options", "") or ""
    options = f"{existing} -c search_path={schema},public".strip()
    return make_conninfo(**params, options=options)


def create_schema() -> tuple[str, str]:
    """Create a fresh schema. Returns `(schema, url_scoped_to_it)`."""
    base = require_database()
    schema = f"t_{uuid.uuid4().hex[:16]}"
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    return schema, url_for_schema(base, schema)


def drop_schema(schema: str) -> None:
    base = os.environ.get(ENV_VAR, "").strip()
    if not base:
        return
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def use_database_env(test: unittest.TestCase) -> str:
    """Give `test` a fresh schema and point `DATABASE_URL` at it.

    For tests that boot the app (TestClient): the lifespan reads the URL from
    settings. Cleanup drops the schema after the test's own cleanups have run,
    so the app has shut down and released its connections first.
    """
    from app.config import get_settings

    schema, url = create_schema()
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    get_settings.cache_clear()

    def _restore() -> None:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        get_settings.cache_clear()
        drop_schema(schema)

    test.addCleanup(_restore)
    return url


async def open_fresh_database(test: unittest.IsolatedAsyncioTestCase) -> str:
    """Fresh schema, migrated, with the application pool open on it.

    For tests that call repositories directly rather than through the app.
    """
    from app.db import database

    url = use_database_env(test)
    await database.close_pool()
    database.configure(url, min_size=1, max_size=4)
    await database.init_db()
    test.addAsyncCleanup(database.close_pool)
    return url
