"""P4 fault drills against real server processes.

The unit tests set up each failure directly, which is what makes them fast and
exact. What they cannot do is prove that a *process* dying leaves recoverable
state, or that a second process actually picks it up — the recovery could be
correct and still never run, because nothing wired it into the app's lifespan.

So this script boots uvicorn for real, kills it with SIGKILL (no shutdown hook,
no flush, no chance to tidy up), boots a second one on the same data directory
and asks the HTTP API what became of the work. Two things are simulated and
said so plainly:

- the model is a socket that accepts the connection and never answers, so a turn
  is genuinely in flight when the process dies;
- MinerU is not called at all. The parse drill seeds a job whose batch id was
  recorded before the crash and checks that recovery *rejoins* it — that no
  second submission is created — which is the part that costs money.

Run: uv run python -m scripts.verify_agent_recovery
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1]

# The heartbeat window the server actually uses. Waiting it out is the point:
# shortening it for the drill would verify a configuration nobody runs.
from app.services.agent_worker import RUN_STALE_SECONDS  # noqa: E402

RECOVERY_DEADLINE = RUN_STALE_SECONDS + 90


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class StallingModel:
    """A socket that accepts an HTTP request and never replies.

    Enough to hold a turn open: the agent's provider call blocks, the run stays
    `running` and keeps its heartbeat, which is the state a crash has to be
    tested against.
    """

    def __init__(self) -> None:
        self.port = _free_port()
        self._server: asyncio.AbstractServer | None = None
        self._held: list[asyncio.StreamWriter] = []

    async def start(self) -> None:
        async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            self._held.append(writer)
            with contextlib.suppress(Exception):
                await reader.read(65536)
                await asyncio.sleep(3600)

        self._server = await asyncio.start_server(_handle, "127.0.0.1", self.port)

    async def stop(self) -> None:
        for writer in self._held:
            with contextlib.suppress(Exception):
                writer.close()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()


class Server:
    """One uvicorn process, killable the way a crash kills one."""

    def __init__(
        self, *, data_dir: Path, db_url: str, model_port: int, secret: str, role: str
    ) -> None:
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}/api"
        self._data_dir = data_dir
        self._db_url = db_url
        self._model_port = model_port
        self._secret = secret
        self._proc: subprocess.Popen[bytes] | None = None
        # Named by role, not by port, so a rerun overwrites rather than piles up.
        self._log = open(data_dir / f"recovery-drill-{role}.log", "wb")

    def start(self) -> None:
        env = {
            **os.environ,
            "DATA_DIR": str(self._data_dir),
            # Both processes share one throwaway schema, as two instances of a
            # deployment share one database.
            "DATABASE_URL": self._db_url,
            "LLM_BASEURL": f"http://127.0.0.1:{self._model_port}/v1",
            "LLM_APIKEY": "drill",
            "THINKING_MODELNAME": "stall-model",
            "AGENT_MODELNAME": "stall-model",
            # Fixed across both processes: a generated secret would change on
            # restart and invalidate the credential this drill holds, which
            # would look like a recovery failure and is not one.
            "AGENT_IDENTITY_SECRET": self._secret,
            "MINERU_TOKEN": "drill-not-a-real-token",
            "ENABLE_DOCS": "true",
        }
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(self.port), "--log-level", "warning"],
            cwd=str(BACKEND_DIR), env=env, stdout=self._log, stderr=subprocess.STDOUT,
        )

    async def wait_ready(self, timeout: float = 180.0) -> None:
        # Generous: a cold import of the agent stack on a WSL /mnt checkout
        # alone takes over a minute.
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(timeout=5) as client:
            while time.monotonic() < deadline:
                if self._proc is not None and self._proc.poll() is not None:
                    raise RuntimeError(
                        f"server exited with {self._proc.returncode}; see {self._log.name}"
                    )
                with contextlib.suppress(Exception):
                    if (await client.get(f"{self.base}/models")).status_code < 500:
                        return
                await asyncio.sleep(0.4)
        raise RuntimeError(f"server on {self.port} never became ready")

    def kill(self) -> None:
        """SIGKILL: no lifespan shutdown, no flush, nothing tidied up."""
        if self._proc is None:
            return
        self._proc.send_signal(signal.SIGKILL)
        self._proc.wait(timeout=30)
        self._proc = None

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            with contextlib.suppress(Exception):
                self._proc.wait(timeout=15)
            self._proc = None
        self._log.close()


class Drill:
    def __init__(self) -> None:
        self.results: dict[str, tuple[bool, str]] = {}

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.results[name] = (bool(passed), detail)
        mark = "PASS" if passed else "FAIL"
        print(f"  {mark}  {name}" + (f"   {detail}" if detail else ""), flush=True)

    @property
    def ok(self) -> bool:
        return all(passed for passed, _ in self.results.values())


async def _sql(db_url: str, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Read the database the servers share, from outside either of them."""
    import psycopg
    from psycopg.rows import dict_row

    from app.db.database import to_pg_sql

    async with await psycopg.AsyncConnection.connect(db_url, row_factory=dict_row) as conn:
        cursor = await conn.execute(to_pg_sql(sql), params)
        return list(await cursor.fetchall())


async def _exec(db_url: str, sql: str, params: tuple[Any, ...] = ()) -> None:
    import psycopg

    from app.db.database import to_pg_sql

    async with await psycopg.AsyncConnection.connect(db_url) as conn:
        await conn.execute(to_pg_sql(sql), params)


def _create_drill_schema() -> tuple[str, str]:
    """A fresh schema in DATABASE_URL's database, and a URL scoped to it."""
    import uuid

    import psycopg
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    from app.config import get_settings

    base = get_settings().database_url
    schema = f"drill_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
    params = conninfo_to_dict(base)
    options = f"{params.pop('options', '') or ''} -c search_path={schema},public".strip()
    return schema, make_conninfo(**params, options=options)


def _drop_drill_schema(schema: str) -> None:
    import psycopg

    from app.config import get_settings

    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


async def main() -> int:
    drill = Drill()
    tmp = tempfile.TemporaryDirectory()
    data_dir = Path(tmp.name)
    schema, db_url = _create_drill_schema()
    model = StallingModel()
    await model.start()
    secret = "recovery-drill-secret"

    first = Server(
        data_dir=data_dir, db_url=db_url, model_port=model.port, secret=secret, role="first"
    )
    second: Server | None = None
    try:
        print("── booting the first server ──", flush=True)
        first.start()
        await first.wait_ready()

        async with httpx.AsyncClient(timeout=20) as client:
            created = await client.post(f"{first.base}/agent/sessions", json={"language": "en"})
            session_id = created.json()["session_id"]
            token = created.headers["X-Agent-Token"]
            headers = {"X-Agent-Token": token}

            # ── A12: the same request twice is one turn ──────────────────────
            body = {"content": "what does this paper claim?", "client_request_id": "drill-1"}
            one = (await client.post(
                f"{first.base}/agent/sessions/{session_id}/messages", json=body, headers=headers
            )).json()
            two = (await client.post(
                f"{first.base}/agent/sessions/{session_id}/messages", json=body, headers=headers
            )).json()
            run_id = one["run_id"]
            drill.check(
                "duplicate_request_is_one_run",
                one["run_id"] == two["run_id"] and two["deduplicated"],
                f"run={run_id}",
            )
            messages = await _sql(
                db_url, "SELECT * FROM agent_messages WHERE session_id = ? AND role = 'user'",
                (session_id,),
            )
            drill.check("duplicate_request_stored_one_message", len(messages) == 1)

            # The turn is now blocked on a provider that will never answer.
            await asyncio.sleep(3)
            rows = await _sql(db_url, "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,))
            drill.check(
                "turn_is_in_flight_when_killed",
                bool(rows) and rows[0]["status"] == "running" and bool(rows[0]["worker_id"]),
                f"worker={rows[0]['worker_id'] if rows else '?'}",
            )

            # ── A13 groundwork: a parse whose batch was already submitted ────
            await _exec(db_url,
                "INSERT INTO papers (paper_id, file_path, title) VALUES ('drillpaper', 'x', 'Drill')")
            pdf = data_dir / "papers" / "drillpaper" / "original.pdf"
            pdf.parent.mkdir(parents=True, exist_ok=True)
            pdf.write_bytes(b"%PDF-1.4\n")
            await _exec(db_url,
                "INSERT INTO mineru_parses (parse_id, paper_id, status, remote_batch_id) "
                "VALUES ('drill-parse', 'drillpaper', 'running', 'drill-batch-42')")
            await _exec(db_url,
                """INSERT INTO agent_jobs
                       (job_id, kind, idempotency_key, session_id, status, attempts,
                        lease_owner, lease_expires_at, remote_id, request_json)
                   VALUES ('drill-job', 'parse', 'parse:drillpaper:vlm', ?, 'running', 1,
                           'dead-worker', '2000-01-01T00:00:00+00:00', 'drill-parse', ?)""",
                (session_id, json.dumps({"paper_id": "drillpaper"})))

        print("── SIGKILL ──", flush=True)
        first.kill()

        print("── booting the second server on the same data ──", flush=True)
        second = Server(
            data_dir=data_dir, db_url=db_url, model_port=model.port, secret=secret, role="second"
        )
        second.start()
        await second.wait_ready()

        # ── A13: the abandoned run reaches a terminal state ──────────────────
        print(f"── waiting up to {RECOVERY_DEADLINE}s for the heartbeat to go stale ──", flush=True)
        deadline = time.monotonic() + RECOVERY_DEADLINE
        status = ""
        error_code = ""
        while time.monotonic() < deadline:
            rows = await _sql(db_url, "SELECT * FROM agent_runs WHERE run_id = ?", (run_id,))
            status, error_code = rows[0]["status"], rows[0]["error_code"]
            if status not in ("pending", "running"):
                break
            await asyncio.sleep(3)
        drill.check("interrupted_run_is_closed", status == "failed", f"status={status}")
        drill.check("interrupted_run_says_why", error_code == "interrupted", f"code={error_code}")

        async with httpx.AsyncClient(timeout=20) as client:
            headers = {"X-Agent-Token": token}

            # ── A14: a browser still watching gets closure, by replay ────────
            activity = await client.get(f"{second.base}/agent/runs/{run_id}/activity", headers=headers)
            types = [e["type"] for e in activity.json()["events"]]
            drill.check(
                "terminal_event_is_replayable",
                activity.status_code == 200 and types and types[-1] == "run.failed",
                f"events={types}",
            )

            resumed_stream = await _read_sse(
                f"{second.base}/agent/runs/{run_id}/events?after=0&token={token}"
            )
            drill.check(
                "sse_replays_to_a_terminal_event",
                bool(resumed_stream) and resumed_stream[-1]["event"] == "run.failed",
                f"frames={len(resumed_stream)}",
            )
            if resumed_stream:
                after = resumed_stream[0]["id"]
                gap = await _read_sse(
                    f"{second.base}/agent/runs/{run_id}/events?after={after}&token={token}"
                )
                replayed_ids = {f["id"] for f in gap}
                drill.check(
                    "sse_resume_skips_what_was_seen",
                    after not in replayed_ids and len(gap) == len(resumed_stream) - 1,
                    f"after={after} then {len(gap)} frames",
                )

            # ── The session is usable again ──────────────────────────────────
            again = await client.post(
                f"{second.base}/agent/sessions/{session_id}/messages",
                json={"content": "asking again", "client_request_id": "drill-2"},
                headers=headers,
            )
            drill.check(
                "session_accepts_a_new_turn_after_recovery",
                again.status_code == 200,
                f"http={again.status_code}",
            )
            if again.status_code == 200:
                new_run = again.json()["run_id"]
                # ── A15: stop reaches a turn blocked on the provider ─────────
                #
                # Measured as "when did the run actually stop", by a watcher
                # started *before* the request goes out. Timing it by when the
                # POST returns would measure this script instead: the response
                # is written in milliseconds (verified against the server's own
                # clock) but this process does not always get round to reading
                # it for several seconds, and that delay is the harness's, not
                # the product's.
                async def _watch_until_terminal() -> tuple[str, float]:
                    at = time.monotonic()
                    while time.monotonic() - at < 40:
                        rows = await _sql(
                            db_url, "SELECT status FROM agent_runs WHERE run_id = ?",
                            (new_run,),
                        )
                        if rows and rows[0]["status"] not in ("pending", "running"):
                            return rows[0]["status"], time.monotonic() - at
                        await asyncio.sleep(0.2)
                    return "timeout", time.monotonic() - at

                # A trivial request alongside the cancel: if that one is slow
                # too, the event loop is starved and every client is waiting,
                # which needs a different fix from a slow handler.
                async def _probe() -> float:
                    at = time.monotonic()
                    await client.get(f"{second.base}/models", timeout=60)
                    return time.monotonic() - at

                watcher = asyncio.create_task(_watch_until_terminal())
                await asyncio.sleep(0)  # let the watcher take its first sample
                cancelled, idle_latency = await asyncio.gather(
                    client.post(
                        f"{second.base}/agent/runs/{new_run}/cancel",
                        headers=headers,
                        timeout=60,
                    ),
                    _probe(),
                )
                terminal, elapsed = await watcher
                print(
                    f"  ·  run reached {terminal} in {elapsed:.1f}s; an unrelated "
                    f"request alongside the cancel took {idle_latency:.1f}s",
                    flush=True,
                )
                drill.check(
                    "server_stays_responsive_while_a_turn_starts",
                    idle_latency < 2.0,
                    f"unrelated request {idle_latency:.1f}s",
                )
                drill.check(
                    "cancel_stops_a_turn_blocked_on_the_model",
                    terminal == "cancelled" and elapsed < 10,
                    f"status={terminal} after {elapsed:.1f}s, interrupted="
                    f"{cancelled.json().get('interrupted')}",
                )

        # ── A13: the parse was rejoined, not resubmitted ─────────────────────
        parses = await _sql(db_url,
            "SELECT * FROM mineru_parses WHERE paper_id = 'drillpaper'")
        job = (await _sql(db_url, "SELECT * FROM agent_jobs WHERE job_id = 'drill-job'"))[0]
        drill.check(
            "recovery_did_not_resubmit_the_parse",
            len(parses) == 1 and parses[0]["parse_id"] == "drill-parse",
            f"{len(parses)} parse row(s)",
        )
        drill.check(
            "recovery_kept_the_remote_batch_id",
            job["remote_id"] == "drill-parse"
            and parses[0]["remote_batch_id"] == "drill-batch-42",
            f"remote_id={job['remote_id']} batch={parses[0]['remote_batch_id']}",
        )
        drill.check(
            "abandoned_job_is_no_longer_claimed_by_a_dead_worker",
            job["lease_owner"] != "dead-worker",
            f"status={job['status']} owner={job['lease_owner'] or '(none)'}",
        )

    finally:
        first.stop()
        if second is not None:
            second.stop()
        await model.stop()
        report = {
            "criteria": {k: {"passed": v[0], "detail": v[1]} for k, v in drill.results.items()},
            "run_stale_seconds": RUN_STALE_SECONDS,
        }
        out = BACKEND_DIR / "data" / "recovery_drill_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        for log in sorted(data_dir.glob("recovery-drill-*.log")):
            (out.parent / log.name).write_bytes(log.read_bytes())
        print(f"\nreport: {out.relative_to(BACKEND_DIR)}", flush=True)
        tmp.cleanup()
        _drop_drill_schema(schema)

    print("\n" + ("ALL DRILLS PASSED" if drill.ok else "SOME DRILLS FAILED"), flush=True)
    return 0 if drill.ok else 1


async def _read_sse(url: str, *, limit: int = 100) -> list[dict[str, str]]:
    """Read a stream to its terminal event. Returns `{id, event}` per frame."""
    frames: list[dict[str, str]] = []
    current: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                return frames
            async for line in response.aiter_lines():
                if line.startswith("id: "):
                    current["id"] = line[4:].strip()
                elif line.startswith("event: "):
                    current["event"] = line[7:].strip()
                elif line == "" and current.get("event"):
                    frames.append(current)
                    if current["event"].startswith("run.") and current["event"] != "run.started":
                        return frames
                    if len(frames) >= limit:
                        return frames
                    current = {}
    return frames


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
