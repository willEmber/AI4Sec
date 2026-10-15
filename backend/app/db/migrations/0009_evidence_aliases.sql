-- The model cites evidence by a short per-session number (`E12`), not by its
-- 20-hex id (P15). Copying a hash out of a context holding a hundred others
-- went wrong often enough to matter, and mostly by landing on another id that
-- shares its first characters, which the citation filter cannot catch. The
-- stored id stays the key everywhere else; this table is the translation.
--   alias_no: the number after `E`, from 1, stable for the session's life
-- No foreign key to `evidence`: recall hands a session evidence recorded under
-- another one, which may be deleted before this session is.
CREATE TABLE agent_evidence_aliases (
    session_id  TEXT        NOT NULL,
    alias_no    INTEGER     NOT NULL,
    evidence_id TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, alias_no),
    UNIQUE (session_id, evidence_id)
);
