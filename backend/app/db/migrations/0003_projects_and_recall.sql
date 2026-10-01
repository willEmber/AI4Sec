-- Research projects and cross-session recall (P8).
--
-- A project groups sessions. Its papers are not stored: they are the union of
-- its sessions' papers, so moving a session in or out cannot leave the two
-- disagreeing. `project_id = ''` means "no project", following the schema's
-- convention for optional references (`paper_id = ''`), and is checked by the
-- repository on write rather than by a foreign key.

CREATE TABLE agent_projects (
    project_id  TEXT PRIMARY KEY,
    owner_id    TEXT NOT NULL REFERENCES agent_principals(principal_id),
    title       TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active',   -- active | archived
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_agent_projects_owner ON agent_projects(owner_id, status, updated_at DESC);

ALTER TABLE agent_sessions ADD COLUMN project_id TEXT NOT NULL DEFAULT '';
CREATE INDEX idx_agent_sessions_project
    ON agent_sessions(project_id, updated_at DESC) WHERE project_id <> '';

-- '' is a global memory; otherwise it is injected only into that project's sessions.
ALTER TABLE agent_memories ADD COLUMN project_id TEXT NOT NULL DEFAULT '';

-- Search terms for conversation recall.
--
-- The 'simple' text-search parser does not segment Chinese: a run of Han
-- characters is one token, so a Chinese question matches almost nothing.
-- This function produces the terms itself — lower-cased Latin words and
-- numbers (compounds such as `top-2` or `29.7` kept whole *and* split), plus
-- character bigrams for Han runs — and both the index and the query use it, so
-- the two can never segment differently. Going through `array_to_tsvector`
-- also bypasses the parser, whose treatment of non-ASCII depends on the
-- database locale.
CREATE FUNCTION scholar_search_terms(src text) RETURNS text[]
LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $$
DECLARE
    terms     text[] := '{}';
    lowered   text;
    run       text;
    i         int;
    stopwords text[] := ARRAY[
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'do', 'does', 'for',
        'from', 'how', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the',
        'this', 'to', 'was', 'were', 'what', 'which', 'with'
    ];
BEGIN
    IF src IS NULL OR src = '' THEN
        RETURN terms;
    END IF;
    lowered := lower(src);
    -- Compounds joined by . - _ (top-2, gpt-4o, 29.7), kept whole.
    FOR run IN
        SELECT m[1] FROM regexp_matches(lowered, '([a-z0-9]+(?:[._-][a-z0-9]+)+)', 'g') AS m
    LOOP
        terms := terms || run;
    END LOOP;
    -- Their parts, and every plain word, so `top` finds `top-2`.
    FOR run IN
        SELECT m[1] FROM regexp_matches(lowered, '([a-z0-9]+)', 'g') AS m
    LOOP
        IF char_length(run) >= 2 AND NOT run = ANY (stopwords) THEN
            terms := terms || run;
        END IF;
    END LOOP;
    -- Han runs as character bigrams; a lone character stands for itself.
    FOR run IN
        SELECT m[1] FROM regexp_matches(src, '([㐀-鿿]+)', 'g') AS m
    LOOP
        IF char_length(run) = 1 THEN
            terms := terms || run;
        ELSE
            FOR i IN 1 .. char_length(run) - 1 LOOP
                terms := terms || substr(run, i, 2);
            END LOOP;
        END IF;
    END LOOP;
    RETURN terms;
END
$$;

-- Generated, so existing messages are indexed by this migration and new ones
-- as they are written; nothing has to remember to keep it in step.
ALTER TABLE agent_messages ADD COLUMN search_tsv TSVECTOR
    GENERATED ALWAYS AS (array_to_tsvector(scholar_search_terms(content))) STORED;
CREATE INDEX idx_agent_messages_search ON agent_messages USING GIN (search_tsv);
