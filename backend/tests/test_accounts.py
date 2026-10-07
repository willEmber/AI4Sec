"""Accounts (P7.5 I2): login, merging anonymous data, cookies, quotas, modes.

The provider round trip is mocked at `oauth_providers.exchange_code` /
`fetch_profile`; everything on our side of it — the sealed state cookie, the
callback, the merge, the session cookie — runs for real against PostgreSQL.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse

from tests.pg_support import open_fresh_database, use_database_env


def _set_env(test: unittest.TestCase, **values: str) -> None:
    from app.config import get_settings

    previous = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    get_settings.cache_clear()

    def _restore() -> None:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        get_settings.cache_clear()

    test.addCleanup(_restore)


class SealTests(unittest.TestCase):
    def setUp(self) -> None:
        from app.services import identity

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        _set_env(self, DATA_DIR=self._tmp.name, AGENT_IDENTITY_SECRET="test-secret")
        identity.reset_secret_cache()
        self.addCleanup(identity.reset_secret_cache)

    def test_round_trip_and_tamper(self) -> None:
        from app.services import identity

        sealed = identity.seal("stream", {"p": "pr_a", "r": "ar_1"}, ttl_seconds=60)
        self.assertEqual(identity.unseal("stream", sealed)["r"], "ar_1")
        body, _, sig = sealed.rpartition(".")
        self.assertIsNone(identity.unseal("stream", body[:-2] + "AA." + sig))

    def test_purposes_do_not_cross(self) -> None:
        from app.services import identity

        sealed = identity.seal("oauth", {"p": "pr_a"}, ttl_seconds=60)
        self.assertIsNone(identity.unseal("stream", sealed))
        # Nor can a sealed value pass for an anonymous credential.
        self.assertIsNone(identity.verify_credential(sealed))

    def test_expiry(self) -> None:
        from app.services import identity

        sealed = identity.seal("stream", {"p": "pr_a"}, ttl_seconds=60)
        with mock.patch("app.services.identity.time.time", return_value=time.time() + 120):
            self.assertIsNone(identity.unseal("stream", sealed))

    def test_safe_next_refuses_other_sites(self) -> None:
        from app.api.auth import safe_next

        self.assertEqual(safe_next("/chat/s1?x=1"), "/chat/s1?x=1")
        for hostile in ("https://evil.example", "//evil.example", "/\\evil.example", "chat"):
            self.assertEqual(safe_next(hostile), "/", hostile)


class AccountServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        await open_fresh_database(self)

    async def _profile(self, subject: str = "42", **kw):
        from app.services.accounts import ExternalProfile

        return ExternalProfile(provider="github", subject=subject, display_name="Ada", **kw)

    async def test_same_subject_is_the_same_account(self) -> None:
        from app.services import accounts

        first = await accounts.upsert_user(await self._profile())
        again = await accounts.upsert_user(await self._profile(avatar_url="https://a/1.png"))
        other = await accounts.upsert_user(await self._profile(subject="43"))
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual((await accounts.get_user(first))["avatar_url"], "https://a/1.png")

    async def test_merge_moves_everything_once(self) -> None:
        from app.db import agent_repository as repo
        from app.db import database as db
        from app.services import accounts, identity

        anon, credential = await identity.create_principal()
        session = await repo.create_session(owner_id=anon)
        await repo.add_memory(owner_id=anon, content="Prefers tables")
        await db.execute(
            "INSERT INTO papers (paper_id, file_path) VALUES ('p1', 'x')"
        )
        await db.execute(
            "INSERT INTO runs (run_id, paper_id, mode, owner_id) VALUES ('r1', 'p1', 'snap', ?)", (anon,)
        )
        user = await accounts.upsert_user(await self._profile())

        self.assertTrue(await accounts.merge_anonymous(anon, user))
        self.assertEqual((await repo.get_session(session.session_id, owner_id=user)).session_id, session.session_id)
        self.assertEqual(len(await repo.list_memories(user)), 1)
        self.assertEqual((await db.fetch_one("SELECT owner_id FROM runs WHERE run_id = 'r1'"))["owner_id"], user)
        # The old credential is dead, and a second merge is a no-op.
        self.assertIsNone(await identity.resolve_principal(credential))
        self.assertFalse(await accounts.merge_anonymous(anon, user))

    async def test_an_account_is_never_merged(self) -> None:
        from app.services import accounts

        a = await accounts.upsert_user(await self._profile("1"))
        b = await accounts.upsert_user(await self._profile("2"))
        self.assertFalse(await accounts.merge_anonymous(a, b))

    async def test_session_expiry_revocation_and_disabled_accounts(self) -> None:
        from app.db import database as db
        from app.services import accounts

        user = await accounts.upsert_user(await self._profile())
        token = await accounts.create_login_session(user)
        self.assertTrue(token.startswith("st_"))
        self.assertEqual(await accounts.resolve_login_session(token), user)
        # Only the hash is stored.
        self.assertIsNone(await db.fetch_one("SELECT 1 AS hit FROM auth_sessions WHERE token_hash = ?", (token,)))

        await db.execute("UPDATE users SET status = 'disabled' WHERE principal_id = ?", (user,))
        self.assertIsNone(await accounts.resolve_login_session(token))
        with self.assertRaises(accounts.AccountDisabled):
            await accounts.upsert_user(await self._profile())
        await db.execute("UPDATE users SET status = 'active' WHERE principal_id = ?", (user,))

        await db.execute("UPDATE auth_sessions SET expires_at = now() - interval '1 second'")
        self.assertIsNone(await accounts.resolve_login_session(token))
        fresh = await accounts.create_login_session(user)
        await accounts.revoke_login_session(fresh)
        self.assertIsNone(await accounts.resolve_login_session(fresh))

    async def test_an_account_credential_is_not_an_hmac_credential(self) -> None:
        from app.services import accounts, identity

        user = await accounts.upsert_user(await self._profile())
        self.assertIsNone(await identity.resolve_principal(identity.issue_credential(user)))

    async def test_daily_usage_counts_runs_and_tokens(self) -> None:
        from app.db import agent_repository as repo
        from app.db import database as db
        from app.services import accounts, identity

        anon, _ = await identity.create_principal()
        session = await repo.create_session(owner_id=anon)
        run, _ = await repo.create_run(session_id=session.session_id, owner_id=anon)
        await db.execute(
            "UPDATE agent_runs SET status = 'done', usage_json = ? WHERE run_id = ?",
            ('{"tokens": 1234}', run.run_id),
        )
        usage = await accounts.daily_usage(accounts.Caller(anon, "anonymous", "cookie"))
        self.assertEqual((usage.runs, usage.tokens), (1, 1234))

    async def test_admin_account_follows_its_configuration(self) -> None:
        from app.db import database as db
        from app.services import accounts

        self.assertIsNone(await accounts.ensure_admin_account())   # never configured
        _set_env(self, ADMIN_USERNAME="root", ADMIN_PASSWORD="pw-1")
        self.assertTrue(accounts.check_admin_credentials(" root ", "pw-1"))
        self.assertFalse(accounts.check_admin_credentials("root", "pw-2"))
        self.assertFalse(accounts.check_admin_credentials("other", "pw-1"))

        admin = await accounts.ensure_admin_account()
        self.assertEqual((await accounts.get_user(admin))["role"], "admin")
        stored = await db.fetch_one("SELECT profile_json FROM user_identities WHERE principal_id = ?", (admin,))
        self.assertNotIn("pw-1", stored["profile_json"])
        token = await accounts.create_login_session(admin)
        usage = await accounts.daily_usage(accounts.Caller(admin, "user", "cookie"))
        self.assertEqual((usage.runs_limit, usage.tokens_limit), (0, 0))

        # A restart with the same configuration keeps the session...
        self.assertEqual(await accounts.ensure_admin_account(), admin)
        self.assertEqual(await accounts.resolve_login_session(token), admin)
        # ...a renamed admin is still the same account, but its logins end...
        _set_env(self, ADMIN_USERNAME="boss")
        self.assertEqual(await accounts.ensure_admin_account(), admin)
        self.assertIsNone(await accounts.resolve_login_session(token))
        self.assertEqual((await accounts.get_user(admin))["display_name"], "boss")
        # ...and so do they when the admin login is switched off.
        token = await accounts.create_login_session(admin)
        _set_env(self, ADMIN_PASSWORD="")
        self.assertFalse(accounts.admin_login_enabled())
        await accounts.ensure_admin_account()
        self.assertIsNone(await accounts.resolve_login_session(token))


class _AppTestCase(unittest.TestCase):
    """Boots the app on a fresh schema with plain-http cookies."""

    env: dict[str, str] = {}

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        _set_env(
            self,
            DATA_DIR=self._tmp.name,
            AUTH_COOKIE_SECURE="false",
            OAUTH_GITHUB_CLIENT_ID="gh-id",
            OAUTH_GITHUB_CLIENT_SECRET="gh-secret",
            PUBLIC_BASE_URL="http://testserver",
            **self.env,
        )
        use_database_env(self)

        from app.services import identity

        identity.reset_secret_cache()
        self.addCleanup(identity.reset_secret_cache)

        from fastapi.testclient import TestClient

        from app.main import app

        self._client_cm = TestClient(app)
        self.client = self._client_cm.__enter__()
        self.addCleanup(self._client_cm.__exit__, None, None, None)
        # Turns are not executed here; only whether one may start.
        patcher = mock.patch("app.services.agent_runner.start_turn", new=mock.AsyncMock())
        patcher.start()
        self.addCleanup(patcher.stop)

    @contextlib.contextmanager
    def _new_client(self):
        """The same app as another browser: no cookies, restored afterwards.

        A second TestClient would run the lifespan again and swap the global
        connection pool out from under the first.
        """
        saved = list(self.client.cookies.jar)
        self.client.cookies.clear()
        try:
            yield self.client
        finally:
            self.client.cookies.clear()
            for cookie in saved:
                self.client.cookies.jar.set_cookie(cookie)

    def _login(self, *, subject: str = "42", next_path: str = "/chat"):
        from app.services.accounts import ExternalProfile

        start = self.client.post("/api/auth/login/github", json={"next": next_path})
        self.assertEqual(start.status_code, 200, start.text)
        state = parse_qs(urlparse(start.json()["authorize_url"]).query)["state"][0]
        profile = ExternalProfile(provider="github", subject=subject, display_name="Ada")
        with mock.patch(
            "app.services.oauth_providers.exchange_code", new=mock.AsyncMock(return_value="tok")
        ), mock.patch(
            "app.services.oauth_providers.fetch_profile", new=mock.AsyncMock(return_value=profile)
        ):
            return self.client.get(
                "/api/auth/callback/github",
                params={"code": "c", "state": state},
                follow_redirects=False,
            )


class LoginFlowTests(_AppTestCase):
    def test_conversation_before_login_belongs_to_the_account_after(self) -> None:
        created = self.client.post("/api/agent/sessions", json={"language": "en"})
        session_id = created.json()["session_id"]
        old_credential = created.headers["X-Agent-Token"]
        self.assertEqual(self.client.cookies.get("scholar_auth"), old_credential)

        done = self._login()
        self.assertEqual(done.status_code, 302)
        self.assertEqual(done.headers["location"], "/chat")
        self.assertTrue(self.client.cookies.get("scholar_auth", "").startswith("st_"))

        me = self.client.get("/api/auth/me").json()
        self.assertTrue(me["authenticated"])
        self.assertEqual(me["user"]["display_name"], "Ada")
        listed = self.client.get("/api/agent/sessions").json()["sessions"]
        self.assertEqual([s["session_id"] for s in listed], [session_id])

        # The anonymous credential no longer opens anything.
        with self._new_client() as stranger:
            response = stranger.get(
                f"/api/agent/sessions/{session_id}", headers={"X-Agent-Token": old_credential}
            )
            self.assertEqual(response.status_code, 401)

    def test_logout_ends_the_session(self) -> None:
        self._login()
        token = self.client.cookies.get("scholar_auth")
        self.assertEqual(self.client.post("/api/auth/logout").status_code, 200)
        self.assertIsNone(self.client.cookies.get("scholar_auth"))
        with self._new_client() as other:
            me = other.get("/api/auth/me", headers={"Cookie": f"scholar_auth={token}"}).json()
            self.assertFalse(me["authenticated"])

    def test_state_must_match(self) -> None:
        self.client.post("/api/auth/login/github", json={"next": "/chat"})
        response = self.client.get(
            "/api/auth/callback/github",
            params={"code": "c", "state": "forged"},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("auth_error=state_mismatch", response.headers["location"])
        self.assertIsNone(self.client.cookies.get("scholar_auth"))

    def test_callback_without_a_login_in_progress(self) -> None:
        response = self.client.get(
            "/api/auth/callback/github", params={"code": "c", "state": "s"}, follow_redirects=False
        )
        self.assertIn("auth_error=login_expired", response.headers["location"])

    def test_unconfigured_provider_is_not_offered(self) -> None:
        providers = self.client.get("/api/auth/providers").json()["providers"]
        self.assertEqual([p["id"] for p in providers], ["github"])
        self.assertEqual(self.client.post("/api/auth/login/google", json={}).status_code, 404)

    def test_header_credential_moves_into_the_cookie(self) -> None:
        with self._new_client() as first:
            credential = first.post("/api/agent/sessions", json={}).headers["X-Agent-Token"]
            first.cookies.clear()
        with self._new_client() as legacy:
            response = legacy.get("/api/agent/sessions", headers={"X-Agent-Token": credential})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(legacy.cookies.get("scholar_auth"), credential)

    def test_me_moves_a_header_credential_into_the_cookie(self) -> None:
        with self._new_client() as first:
            credential = first.post("/api/agent/sessions", json={}).headers["X-Agent-Token"]
            first.cookies.clear()
            me = first.get("/api/auth/me", headers={"X-Agent-Token": credential}).json()
            self.assertEqual(me["kind"], "anonymous")
            self.assertFalse(me["authenticated"])
            self.assertEqual(first.cookies.get("scholar_auth"), credential)

    def test_classic_runs_follow_the_principal(self) -> None:
        from app.db import database as db

        self.client.post("/api/agent/sessions", json={})   # become a principal
        me = self.client.get("/api/auth/me").json()["principal_id"]
        db.execute_sync("INSERT INTO papers (paper_id, file_path) VALUES ('p1', 'x')")
        db.execute_sync(
            "INSERT INTO runs (run_id, paper_id, mode, owner_id) VALUES ('mine', 'p1', 'snap', ?)", (me,)
        )
        db.execute_sync(
            "INSERT INTO runs (run_id, paper_id, mode, owner_token) VALUES ('legacy', 'p1', 'snap', 'tok-1')"
        )

        ids = {r["run_id"] for r in self.client.get("/api/runs/recent").json()}
        self.assertEqual(ids, {"mine"})
        # The browser's old token adopts the legacy row for this principal...
        ids = {r["run_id"] for r in self.client.get("/api/runs/recent", params={"owner_token": "tok-1"}).json()}
        self.assertEqual(ids, {"mine", "legacy"})
        ids = {r["run_id"] for r in self.client.get("/api/runs/recent").json()}
        self.assertEqual(ids, {"mine", "legacy"})
        # ...and nobody else sees either, token or not.
        with self._new_client() as stranger:
            ids = {r["run_id"] for r in stranger.get("/api/runs/recent", params={"owner_token": "tok-1"}).json()}
            self.assertEqual(ids, set())
            self.assertEqual(stranger.post("/api/runs/mine/dismiss").status_code, 404)

    def test_stream_ticket_opens_only_its_run(self) -> None:
        session_id = self.client.post("/api/agent/sessions", json={}).json()["session_id"]
        run_id = self.client.post(
            f"/api/agent/sessions/{session_id}/messages", json={"content": "hi"}
        ).json()["run_id"]
        ticket = self.client.post(f"/api/agent/runs/{run_id}/stream-ticket").json()["ticket"]

        # End the turn, so the stream replays to a terminal event and closes.
        from app.db import agent_repository as repo
        from app.models.agent_models import EventType

        async def _finish() -> None:
            await repo.append_event(
                session_id=session_id, type=EventType.RUN_COMPLETED, payload={}, run_id=run_id
            )

        self.client.portal.call(_finish)

        with self._new_client() as browser:   # no cookie: the ticket alone
            ok = browser.get(f"/api/agent/runs/{run_id}/events", params={"ticket": ticket})
            self.assertEqual(ok.status_code, 200)
            self.assertIn("run.completed", ok.text)
            other = browser.get("/api/agent/runs/ar_other/events", params={"ticket": ticket})
            self.assertEqual(other.status_code, 401)
            forged = browser.get(f"/api/agent/runs/{run_id}/events", params={"ticket": ticket + "x"})
            self.assertEqual(forged.status_code, 401)


class AdminLoginTests(_AppTestCase):
    env = {
        "ADMIN_USERNAME": "root",
        "ADMIN_PASSWORD": "pw-1",
        "ADMIN_API_TOKEN": "ops-token",
        "QUOTA_USER_DAILY_RUNS": "1",
    }

    def test_password_login_takes_the_visitor_along(self) -> None:
        self.assertTrue(self.client.get("/api/auth/providers").json()["admin_login"])
        session_id = self.client.post("/api/agent/sessions", json={}).json()["session_id"]

        wrong = self.client.post("/api/auth/admin/login", json={"username": "root", "password": "nope"})
        self.assertEqual(wrong.status_code, 401)
        self.assertEqual(wrong.json()["detail"]["code"], "invalid_credentials")

        done = self.client.post("/api/auth/admin/login", json={"username": "root", "password": "pw-1"})
        self.assertEqual(done.status_code, 200, done.text)
        self.assertTrue(self.client.cookies.get("scholar_auth", "").startswith("st_"))
        me = self.client.get("/api/auth/me").json()
        self.assertTrue(me["authenticated"])
        self.assertEqual((me["user"]["display_name"], me["user"]["role"]), ("root", "admin"))
        self.assertEqual(me["quota"]["runs_limit"], 0)   # the user quota does not apply
        listed = self.client.get("/api/agent/sessions").json()["sessions"]
        self.assertEqual([s["session_id"] for s in listed], [session_id])

    def test_admin_routes_open_to_the_admin_login(self) -> None:
        with self._new_client() as visitor:
            visitor.post("/api/agent/sessions", json={})   # an anonymous principal
            self.assertEqual(visitor.get("/api/admin/rank-cache/stats").status_code, 401)
            ok = visitor.get("/api/admin/rank-cache/stats", headers={"X-Admin-Token": "ops-token"})
            self.assertEqual(ok.status_code, 200)
        self.client.post("/api/auth/admin/login", json={"username": "root", "password": "pw-1"})
        self.assertEqual(self.client.get("/api/admin/rank-cache/stats").status_code, 200)


class AdminLoginOffTests(_AppTestCase):
    def test_not_offered_without_configuration(self) -> None:
        self.assertFalse(self.client.get("/api/auth/me").json()["admin_login"])
        response = self.client.post("/api/auth/admin/login", json={"username": "", "password": ""})
        self.assertEqual(response.status_code, 404)

    def test_admin_routes_are_closed_without_a_token_or_an_admin(self) -> None:
        self.assertEqual(self.client.get("/api/admin/rank-cache/stats").status_code, 401)
        self.client.post("/api/agent/sessions", json={})   # an anonymous principal
        self.assertEqual(self.client.get("/api/admin/rank-cache/stats").status_code, 401)
        self.assertEqual(self.client.delete("/api/admin/rank-cache").status_code, 401)


class QuotaTests(_AppTestCase):
    env = {"QUOTA_ANON_DAILY_RUNS": "1"}

    def test_second_turn_of_the_day_is_refused_but_a_retry_is_not(self) -> None:
        from app.db import database as db

        session_id = self.client.post("/api/agent/sessions", json={}).json()["session_id"]
        first = self.client.post(
            f"/api/agent/sessions/{session_id}/messages",
            json={"content": "one", "client_request_id": "req-1"},
        )
        self.assertEqual(first.status_code, 200, first.text)
        # Let the session accept another turn.
        db.execute_sync("UPDATE agent_runs SET status = 'done'")

        second = self.client.post(
            f"/api/agent/sessions/{session_id}/messages", json={"content": "two"}
        )
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.json()["detail"]["code"], "quota_exceeded")

        retry = self.client.post(
            f"/api/agent/sessions/{session_id}/messages",
            json={"content": "one", "client_request_id": "req-1"},
        )
        self.assertEqual(retry.status_code, 200)
        self.assertTrue(retry.json()["deduplicated"])

    def test_a_mode_run_from_the_upload_page_counts_as_a_run(self) -> None:
        from app.db import database as db

        db.execute_sync("INSERT INTO papers (paper_id, file_path) VALUES ('p1', 'papers/p1/original.pdf')")
        with mock.patch("app.services.mode_runs._execute", new=mock.AsyncMock()):
            first = self.client.post("/api/runs", json={"paper_id": "p1", "mode": "snap"})
            self.assertEqual(first.status_code, 200, first.text)
            second = self.client.post("/api/runs", json={"paper_id": "p1", "mode": "snap"})
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.json()["detail"]["code"], "quota_exceeded")
        self.assertEqual(self.client.get("/api/auth/me").json()["quota"]["runs"], 1)
        # The same day's agent turn is refused too: one allowance, not one per door.
        session_id = self.client.post("/api/agent/sessions", json={}).json()["session_id"]
        turn = self.client.post(f"/api/agent/sessions/{session_id}/messages", json={"content": "one"})
        self.assertEqual(turn.status_code, 429)

    def test_a_mode_run_made_inside_a_turn_is_not_counted_twice(self) -> None:
        from app.db import database as db

        session_id = self.client.post("/api/agent/sessions", json={}).json()["session_id"]
        self.client.post(f"/api/agent/sessions/{session_id}/messages", json={"content": "one"})
        owner = self.client.portal.call(db.fetch_one, "SELECT owner_id, run_id FROM agent_runs")
        db.execute_sync("INSERT INTO papers (paper_id, file_path) VALUES ('p1', 'papers/p1/original.pdf')")
        db.execute_sync(
            "INSERT INTO runs (run_id, paper_id, owner_id, agent_run_id) VALUES ('r1', 'p1', ?, ?)",
            (owner["owner_id"], owner["run_id"]),
        )
        self.assertEqual(self.client.get("/api/auth/me").json()["quota"]["runs"], 1)


class LibraryQuotaTests(_AppTestCase):
    env = {"QUOTA_ANON_DAILY_RUNS": "1", "DIFY_API_BASE": "http://dify.invalid"}

    def test_a_library_question_counts_once_answered(self) -> None:
        from app.services.dify_client import DifyError

        ask = {"question": "what is attention?"}
        down = mock.AsyncMock(side_effect=DifyError("down", upstream_status=503))
        with mock.patch("app.services.corpus_qa.answer_corpus_question", new=down):
            self.assertEqual(self.client.post("/api/library/ask", json=ask).status_code, 502)
        answered = mock.AsyncMock(return_value={"answer": "a"})
        with mock.patch("app.services.corpus_qa.answer_corpus_question", new=answered):
            first = self.client.post("/api/library/ask", json=ask)
            self.assertEqual(first.status_code, 200, first.text)
            second = self.client.post("/api/library/ask", json=ask)
        self.assertEqual(second.status_code, 429)
        self.assertEqual(second.json()["detail"]["code"], "quota_exceeded")
        self.assertEqual(answered.await_count, 1)


class DeleteDataTests(_AppTestCase):
    env = {"ADMIN_USERNAME": "root", "ADMIN_PASSWORD": "pw-1"}

    def test_a_reader_deletes_a_session_a_project_and_then_everything(self) -> None:
        session_id = self.client.post("/api/agent/sessions", json={}).json()["session_id"]
        project_id = self.client.post("/api/agent/projects", json={"title": "MoE"}).json()["project"]["project_id"]
        with self._new_client() as visitor:
            visitor.post("/api/agent/sessions", json={})
            self.assertEqual(visitor.delete(f"/api/agent/sessions/{session_id}").status_code, 404)
            self.assertEqual(visitor.delete(f"/api/agent/projects/{project_id}").status_code, 404)

        self.assertEqual(self.client.get("/api/auth/me/data").json(), {"sessions": 1, "runs": 0, "projects": 1})
        self.assertEqual(self.client.delete(f"/api/agent/projects/{project_id}").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/agent/sessions/{session_id}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/agent/sessions/{session_id}").status_code, 404)

        self.client.post("/api/agent/sessions", json={})
        gone = self.client.delete("/api/auth/me")
        self.assertEqual(gone.status_code, 200, gone.text)
        self.assertEqual(gone.json()["sessions"], 1)
        self.assertEqual(self.client.get("/api/agent/sessions").status_code, 401)

    def test_the_configured_admin_is_not_deletable(self) -> None:
        self.client.post("/api/auth/admin/login", json={"username": "root", "password": "pw-1"})
        refused = self.client.delete("/api/auth/me")
        self.assertEqual(refused.status_code, 400)
        self.assertEqual(refused.json()["detail"]["code"], "not_deletable")


class NoAnonymousTests(_AppTestCase):
    env = {"AUTH_ALLOW_ANONYMOUS": "false"}

    def test_costly_routes_need_a_login(self) -> None:
        self.assertEqual(self.client.post("/api/agent/sessions", json={}).status_code, 401)
        upload = self.client.post(
            "/api/papers/upload", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}
        )
        self.assertEqual(upload.status_code, 401)

        self._login()
        self.assertEqual(self.client.post("/api/agent/sessions", json={}).status_code, 200)


class SingleUserTests(_AppTestCase):
    env = {"AUTH_MODE": "single_user"}

    def test_everyone_is_the_local_user_and_old_data_follows(self) -> None:
        from app.services import identity
        from app.services.accounts import LOCAL_PRINCIPAL_ID

        me = self.client.get("/api/auth/me").json()
        self.assertEqual((me["mode"], me["principal_id"]), ("single_user", LOCAL_PRINCIPAL_ID))
        self.assertEqual(me["quota"]["runs_limit"], 0)   # never limited
        # A private instance's only visitor is its owner.
        self.assertEqual(self.client.get("/api/admin/rank-cache/stats").status_code, 200)

        # Data an anonymous browser made before the switch.
        from app.db import agent_repository as repo

        async def _legacy() -> tuple[str, str]:
            anon, credential = await identity.create_principal()
            session = await repo.create_session(owner_id=anon)
            return credential, session.session_id

        credential, session_id = self.client.portal.call(_legacy)
        listed = self.client.get(
            "/api/agent/sessions", headers={"X-Agent-Token": credential}
        ).json()["sessions"]
        self.assertIn(session_id, [s["session_id"] for s in listed])
        # And without the old credential, it is still there.
        with self._new_client() as fresh:
            listed = fresh.get("/api/agent/sessions").json()["sessions"]
            self.assertIn(session_id, [s["session_id"] for s in listed])
        self.assertEqual(self.client.post("/api/auth/login/github", json={}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
