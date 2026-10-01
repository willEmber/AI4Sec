-- Accounts (P7.5 I2).
--
-- A user is a principal of kind 'user': every existing owner_id keeps pointing
-- at agent_principals, so nothing that owns data needs to learn about users.

CREATE TABLE users (
    principal_id  TEXT PRIMARY KEY REFERENCES agent_principals(principal_id),
    email         TEXT NOT NULL DEFAULT '',
    display_name  TEXT NOT NULL DEFAULT '',
    avatar_url    TEXT NOT NULL DEFAULT '',
    role          TEXT NOT NULL DEFAULT 'user',     -- user | admin
    status        TEXT NOT NULL DEFAULT 'active',   -- active | disabled
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per external login. Keyed by the provider's own subject, never by
-- email: an address can change hands, and two providers can report the same
-- one without either proving it belongs to the same person.
CREATE TABLE user_identities (
    provider      TEXT NOT NULL,
    subject       TEXT NOT NULL,
    principal_id  TEXT NOT NULL REFERENCES users(principal_id),
    email         TEXT NOT NULL DEFAULT '',
    profile_json  TEXT NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (provider, subject)
);
CREATE INDEX idx_user_identities_principal ON user_identities(principal_id);

-- Login sessions. Only the SHA-256 of the cookie value is stored, so a read
-- of this table does not hand out working credentials.
CREATE TABLE auth_sessions (
    token_hash   TEXT PRIMARY KEY,
    principal_id TEXT NOT NULL REFERENCES users(principal_id),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at   TIMESTAMPTZ DEFAULT NULL,
    user_agent   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX idx_auth_sessions_principal ON auth_sessions(principal_id);

-- An anonymous principal whose data moved into an account. Its credential
-- stops resolving the moment this is set.
ALTER TABLE agent_principals ADD COLUMN merged_into TEXT DEFAULT NULL
    REFERENCES agent_principals(principal_id);

-- Classic mode runs gain a server-verified owner beside the browser's
-- self-declared owner_token, which keeps scoping rows made before this.
ALTER TABLE runs ADD COLUMN owner_id TEXT DEFAULT NULL;
CREATE INDEX idx_runs_owner_id_started ON runs(owner_id, started_at DESC);
