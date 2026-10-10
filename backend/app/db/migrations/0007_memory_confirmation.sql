-- Forgetting a memory now deletes it. Rows forgotten before this were only
-- flagged, so what a reader asked to have forgotten was still stored; they go
-- here. `active` stays as a column (an imported SQLite deployment still
-- carries it) but nothing sets it to 0 any more.
DELETE FROM agent_memories WHERE active = 0;

-- A memory the agent saved without the reader having said anything like it
-- waits here until the reader confirms it: a turn never sees a pending one.
ALTER TABLE agent_memories ADD COLUMN pending INTEGER NOT NULL DEFAULT 0;
-- The memory a pending correction would replace, dropped when it is confirmed.
ALTER TABLE agent_memories ADD COLUMN replaces_id TEXT NOT NULL DEFAULT '';

-- "The same text once per scope" was a read followed by an insert, so two
-- saves arriving together both inserted. Keep the most recently touched copy
-- and let the index hold the rule from here on.
DELETE FROM agent_memories a
 USING agent_memories b
 WHERE a.owner_id = b.owner_id
   AND a.project_id = b.project_id
   AND a.content = b.content
   AND (a.updated_at, a.memory_id) < (b.updated_at, b.memory_id);

CREATE UNIQUE INDEX idx_agent_memories_unique
    ON agent_memories(owner_id, project_id, content) WHERE active = 1;
