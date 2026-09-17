-- P1: Agent sessions, literature identity, versioned parses and immutable evidence.
--
-- Deliberately parallel to the existing `runs` tables rather than folded into
-- them: `runs.paper_id` is NOT NULL, which cannot express a research task that
-- starts with no attachment or one that spans several papers.

-- ── Identity ────────────────────────────────────────────────────────────────
-- Server-issued anonymous principals. The browser's `owner_token` is client
-- generated and therefore unforgeable only for listing; agent resources are
-- scoped to a principal whose credential the server signs and verifies.
CREATE TABLE IF NOT EXISTS agent_principals (
    principal_id TEXT PRIMARY KEY,
    kind         TEXT NOT NULL DEFAULT 'anonymous',  -- anonymous | user
    label        TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Sessions and runs ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id   TEXT PRIMARY KEY,
    owner_id     TEXT NOT NULL REFERENCES agent_principals(principal_id),
    -- LangGraph thread. Held as its own column so a session could be re-threaded
    -- (e.g. after a compaction) without changing its public id.
    thread_id    TEXT NOT NULL,
    title        TEXT NOT NULL DEFAULT '',
    language     TEXT NOT NULL DEFAULT 'zh',
    llm_model    TEXT NOT NULL DEFAULT '',
    config_json  TEXT NOT NULL DEFAULT '{}',
    status       TEXT NOT NULL DEFAULT 'active',   -- active | archived
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_owner
    ON agent_sessions(owner_id, updated_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_sessions_thread
    ON agent_sessions(thread_id);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id            TEXT PRIMARY KEY,
    session_id        TEXT NOT NULL REFERENCES agent_sessions(session_id),
    owner_id          TEXT NOT NULL,
    -- Client-supplied idempotency key: resubmitting the same key returns the
    -- original run instead of starting a second one (acceptance case A12).
    client_request_id TEXT NOT NULL DEFAULT '',
    status            TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed|cancelled
    cancel_requested  INTEGER NOT NULL DEFAULT 0,
    error_code        TEXT NOT NULL DEFAULT '',
    error_msg         TEXT NOT NULL DEFAULT '',
    -- Budget ceilings for this run and what it actually consumed, both JSON so
    -- new dimensions do not need a migration.
    budget_json       TEXT NOT NULL DEFAULT '{}',
    usage_json        TEXT NOT NULL DEFAULT '{}',
    llm_model         TEXT NOT NULL DEFAULT '',
    prompt_version    TEXT NOT NULL DEFAULT '',
    started_at        TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at       TEXT DEFAULT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_session ON agent_runs(session_id, started_at DESC);
-- One in-flight run per session; a second concurrent submit must get a 409.
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runs_active
    ON agent_runs(session_id) WHERE status IN ('pending', 'running');
-- Idempotency is per session, and only for callers that supplied a key.
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runs_idempotent
    ON agent_runs(session_id, client_request_id) WHERE client_request_id <> '';

CREATE TABLE IF NOT EXISTS agent_messages (
    message_id     TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES agent_sessions(session_id),
    run_id         TEXT NOT NULL DEFAULT '',
    role           TEXT NOT NULL,                  -- user | assistant | system
    content        TEXT NOT NULL DEFAULT '',
    citations_json TEXT NOT NULL DEFAULT '[]',     -- evidence_id list backing the text
    seq            INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_agent_messages_session ON agent_messages(session_id, seq);

-- Durable event log. Events are persisted before publishing so a reconnecting
-- client can replay from `seq` (`Last-Event-ID`) and de-duplicate.
CREATE TABLE IF NOT EXISTS agent_events (
    session_id     TEXT NOT NULL,
    seq            INTEGER NOT NULL,
    run_id         TEXT NOT NULL DEFAULT '',
    schema_version INTEGER NOT NULL DEFAULT 1,
    type           TEXT NOT NULL,
    payload_json   TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_agent_events_run ON agent_events(run_id, seq);

-- ── Literature identity ─────────────────────────────────────────────────────
-- A candidate paper, which may never acquire a local PDF. Kept distinct from
-- `papers` (whose id is sha1 of the file) because one work can have several
-- files and a file can exist with no external identifier.
CREATE TABLE IF NOT EXISTS literature_items (
    literature_id     TEXT PRIMARY KEY,
    doi               TEXT NOT NULL DEFAULT '',
    arxiv_id          TEXT NOT NULL DEFAULT '',
    openalex_id       TEXT NOT NULL DEFAULT '',
    s2_paper_id       TEXT NOT NULL DEFAULT '',
    pmid              TEXT NOT NULL DEFAULT '',
    title             TEXT NOT NULL DEFAULT '',
    title_fingerprint TEXT NOT NULL DEFAULT '',
    authors_json      TEXT NOT NULL DEFAULT '[]',
    year              INTEGER NOT NULL DEFAULT 0,
    -- Explicit, because "no year" and "year 0" must not be confused when a
    -- date filter is applied (acceptance case A04).
    year_known        INTEGER NOT NULL DEFAULT 0,
    venue             TEXT NOT NULL DEFAULT '',
    abstract          TEXT NOT NULL DEFAULT '',
    url               TEXT NOT NULL DEFAULT '',
    oa_pdf_url        TEXT NOT NULL DEFAULT '',
    source            TEXT NOT NULL DEFAULT '',    -- which platform produced it
    retrieved_at      TEXT NOT NULL DEFAULT (datetime('now')),
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_literature_doi ON literature_items(doi) WHERE doi <> '';
CREATE INDEX IF NOT EXISTS idx_literature_arxiv ON literature_items(arxiv_id) WHERE arxiv_id <> '';
CREATE INDEX IF NOT EXISTS idx_literature_fingerprint
    ON literature_items(title_fingerprint) WHERE title_fingerprint <> '';

-- Candidate ↔ local file. Many-to-many on purpose: one work may have several
-- downloaded files (preprint and version of record).
CREATE TABLE IF NOT EXISTS literature_files (
    literature_id TEXT NOT NULL REFERENCES literature_items(literature_id),
    paper_id      TEXT NOT NULL REFERENCES papers(paper_id),
    origin        TEXT NOT NULL DEFAULT '',        -- upload | arxiv | oa_resolver | ...
    is_primary    INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (literature_id, paper_id)
);
CREATE INDEX IF NOT EXISTS idx_literature_files_paper ON literature_files(paper_id);

CREATE TABLE IF NOT EXISTS session_papers (
    session_id    TEXT NOT NULL REFERENCES agent_sessions(session_id),
    literature_id TEXT NOT NULL REFERENCES literature_items(literature_id),
    paper_id      TEXT NOT NULL DEFAULT '',
    -- candidate: metadata only. pdf_ready: file stored. parsed: PaperIR built.
    -- unavailable: full text could not be obtained (acceptance case A08).
    availability  TEXT NOT NULL DEFAULT 'candidate',
    added_by      TEXT NOT NULL DEFAULT 'agent',   -- agent | user
    note          TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, literature_id)
);
CREATE INDEX IF NOT EXISTS idx_session_papers_session ON session_papers(session_id, updated_at DESC);

-- ── Versioned parses ────────────────────────────────────────────────────────
-- A parse of one PDF with one parser configuration. Evidence points at a
-- version, so re-parsing a paper cannot invalidate answers already given.
CREATE TABLE IF NOT EXISTS paper_versions (
    version_id    TEXT PRIMARY KEY,
    paper_id      TEXT NOT NULL REFERENCES papers(paper_id),
    parse_id      TEXT NOT NULL DEFAULT '',        -- mineru_parses.parse_id, when applicable
    parser        TEXT NOT NULL DEFAULT 'mineru',
    parser_config TEXT NOT NULL DEFAULT '{}',
    output_dir    TEXT NOT NULL DEFAULT '',
    -- Hash of the parsed content, so an identical re-parse reuses the version
    -- instead of creating a duplicate.
    content_hash  TEXT NOT NULL DEFAULT '',
    block_count   INTEGER NOT NULL DEFAULT 0,
    page_count    INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'ready',   -- ready | superseded | failed
    is_current    INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_paper_versions_paper ON paper_versions(paper_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_paper_versions_current
    ON paper_versions(paper_id) WHERE is_current = 1;

-- ── Evidence ────────────────────────────────────────────────────────────────
-- Immutable. Rows are never updated: a changed quote is a different evidence
-- id, so a citation stored in an old answer always resolves to what was
-- actually read at the time.
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id   TEXT PRIMARY KEY,
    owner_id      TEXT NOT NULL DEFAULT '',
    session_id    TEXT NOT NULL DEFAULT '',
    literature_id TEXT NOT NULL DEFAULT '',
    paper_id      TEXT NOT NULL DEFAULT '',        -- empty for metadata/web evidence
    parse_version TEXT NOT NULL DEFAULT '',        -- empty for non-PDF sources
    source_level  TEXT NOT NULL,                   -- fulltext|abstract|metadata|external_web
    -- Structured position: section path, 0-based page index, block range, bbox.
    -- Shape varies by source level, so it is JSON rather than columns.
    locator_json  TEXT NOT NULL DEFAULT '{}',
    quote         TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL DEFAULT '',
    source_url    TEXT NOT NULL DEFAULT '',
    provider      TEXT NOT NULL DEFAULT '',
    retrieved_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_evidence_session ON evidence(session_id, retrieved_at DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_paper ON evidence(paper_id, parse_version);
CREATE INDEX IF NOT EXISTS idx_evidence_literature ON evidence(literature_id);

-- ── Long-running jobs ───────────────────────────────────────────────────────
-- Downloads and parses. The idempotency key is derived from the work identity
-- plus configuration, so a restart re-attaches to the job instead of
-- resubmitting it to the external service (acceptance case A13).
CREATE TABLE IF NOT EXISTS agent_jobs (
    job_id          TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,                 -- download | parse
    idempotency_key TEXT NOT NULL,
    session_id      TEXT NOT NULL DEFAULT '',
    run_id          TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending', -- pending|running|done|failed|cancelled
    attempts        INTEGER NOT NULL DEFAULT 0,
    -- Worker lease: a crashed worker's job is reclaimable once this passes.
    lease_owner     TEXT NOT NULL DEFAULT '',
    lease_expires_at TEXT DEFAULT NULL,
    remote_id       TEXT NOT NULL DEFAULT '',      -- e.g. MinerU batch id, for recovery
    request_json    TEXT NOT NULL DEFAULT '{}',
    result_json     TEXT NOT NULL DEFAULT '{}',
    error_code      TEXT NOT NULL DEFAULT '',
    error_msg       TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_jobs_idempotent ON agent_jobs(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_agent_jobs_claimable ON agent_jobs(status, lease_expires_at);
