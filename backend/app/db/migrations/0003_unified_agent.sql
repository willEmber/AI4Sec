-- P5: the reading agent becomes the single entry point.
--
-- Two additions. Long-term memory is per principal, not per session: a
-- preference stated in one conversation should shape the next one, and the
-- reader must be able to see and delete what was kept. Mode reports produced
-- inside a conversation are ordinary `runs` rows — that is what keeps the
-- report page, the compare matrix and the exports working unchanged — so a
-- run only needs to say which conversation, and which turn, produced it.

CREATE TABLE IF NOT EXISTS agent_memories (
    memory_id         TEXT PRIMARY KEY,
    owner_id          TEXT NOT NULL,
    kind              TEXT NOT NULL DEFAULT 'preference',  -- preference | fact | project | instruction
    content           TEXT NOT NULL,
    source_session_id TEXT NOT NULL DEFAULT '',
    source_run_id     TEXT NOT NULL DEFAULT '',
    active            INTEGER NOT NULL DEFAULT 1,
    created_at        TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_agent_memories_owner
    ON agent_memories(owner_id, active, updated_at DESC);

ALTER TABLE runs ADD COLUMN agent_session_id TEXT NOT NULL DEFAULT '';
ALTER TABLE runs ADD COLUMN agent_run_id     TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_runs_agent_session
    ON runs(agent_session_id, started_at DESC) WHERE agent_session_id <> '';
