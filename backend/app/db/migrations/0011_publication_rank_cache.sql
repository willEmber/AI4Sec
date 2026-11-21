-- Dynamic L2 cache for external publication-rank lookups (P17).
-- Static authoritative directories (CCF, CORE, TH-CPL, CAAI, CCF-CN, JCR/CAS
-- seed and early-warning list) live in the read-only in-process catalog so
-- throwaway test schemas do not insert tens of thousands of rows per test.
-- This table replaces the standalone `rank_cache.sqlite3` file so replicas
-- share cached EasyScholar / OpenAlex / web-fallback lookups.
CREATE TABLE publication_rank_cache (
    lookup_key      TEXT        PRIMARY KEY,
    name_display    TEXT        NOT NULL,
    issn            TEXT        NOT NULL DEFAULT '',
    sci             TEXT,
    ccf             TEXT,
    extra_json      TEXT        NOT NULL DEFAULT '{}',
    source          TEXT        NOT NULL DEFAULT '',
    source_priority INTEGER     NOT NULL DEFAULT 0,
    success         INTEGER     NOT NULL DEFAULT 1,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ NOT NULL
);
CREATE INDEX idx_publication_rank_cache_expires ON publication_rank_cache(expires_at);
CREATE INDEX idx_publication_rank_cache_failures ON publication_rank_cache(success) WHERE success = 0;

-- Rich multi-system venue profile snapshot (CCF, CORE, TH-CPL, CAAI, JCR, CAS,
-- CAS early-warning flag, subtrack) alongside legacy `sci_rank` / `ccf_rank`.
ALTER TABLE papers ADD COLUMN venue_ranks_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE literature_items ADD COLUMN venue_ranks_json TEXT NOT NULL DEFAULT '{}';
