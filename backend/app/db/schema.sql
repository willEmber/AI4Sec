-- Scholar Platform Database Schema

CREATE TABLE IF NOT EXISTS papers (
    paper_id   TEXT PRIMARY KEY,             -- sha1(pdf_bytes)
    file_path  TEXT NOT NULL,                -- relative to data_dir
    title      TEXT DEFAULT '',
    doi        TEXT DEFAULT '',
    venue      TEXT DEFAULT '',              -- journal/conference name from Crossref
    year       INTEGER DEFAULT 0,            -- publication year
    sci_rank   TEXT DEFAULT '',              -- SCI tier: Q1/Q2/Q3/Q4
    ccf_rank   TEXT DEFAULT '',              -- CCF rating: A/B/C
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS mineru_parses (
    parse_id   TEXT PRIMARY KEY,
    paper_id   TEXT NOT NULL REFERENCES papers(paper_id),
    backend    TEXT NOT NULL DEFAULT 'vlm',  -- vlm | pipeline
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | failed
    output_dir TEXT DEFAULT '',
    error_msg  TEXT DEFAULT '',
    remote_batch_id TEXT DEFAULT '',
    poll_count INTEGER DEFAULT 0,
    last_state_counts TEXT DEFAULT '',
    last_poll_at TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_mineru_parses_paper ON mineru_parses(paper_id);

CREATE TABLE IF NOT EXISTS blocks (
    block_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id     TEXT NOT NULL REFERENCES papers(paper_id),
    type         TEXT NOT NULL DEFAULT '',    -- text | title | table | image | equation | code | list | ref_text ...
    sub_type     TEXT DEFAULT '',
    page_idx     INTEGER DEFAULT 0,
    bbox_json    TEXT DEFAULT '[]',           -- [x0, y0, x1, y1]
    text         TEXT DEFAULT '',
    section_path TEXT DEFAULT '',             -- e.g. "2.Method/2.1 Model"
    order_idx    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_blocks_paper ON blocks(paper_id);
CREATE INDEX IF NOT EXISTS idx_blocks_type  ON blocks(paper_id, type);

CREATE TABLE IF NOT EXISTS paper_nodes (
    node_id         TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL REFERENCES papers(paper_id),
    parent_id       TEXT DEFAULT '',
    depth           INTEGER DEFAULT 0,
    node_type       TEXT NOT NULL DEFAULT '',    -- paper | section | chunk
    block_type      TEXT DEFAULT '',             -- original block type for chunk nodes
    sub_type        TEXT DEFAULT '',
    title           TEXT DEFAULT '',
    title_path      TEXT DEFAULT '',
    page_start      INTEGER DEFAULT 0,
    page_end        INTEGER DEFAULT 0,
    block_start     INTEGER DEFAULT 0,
    block_end       INTEGER DEFAULT 0,
    text            TEXT DEFAULT '',
    text_for_search TEXT DEFAULT '',
    order_idx       INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_paper_nodes_paper ON paper_nodes(paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_nodes_parent ON paper_nodes(paper_id, parent_id);
CREATE INDEX IF NOT EXISTS idx_paper_nodes_type ON paper_nodes(paper_id, node_type);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL REFERENCES papers(paper_id),
    mode            TEXT NOT NULL DEFAULT 'snap', -- snap | lens | sphere | auto | qa
    llm_model       TEXT DEFAULT '',
    language        TEXT NOT NULL DEFAULT 'en',   -- en | zh
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | running | done | failed
    error_msg       TEXT DEFAULT '',
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    finished_at     TEXT DEFAULT NULL,
    user_question   TEXT DEFAULT '',
    detected_intent TEXT DEFAULT '',
    current_step    TEXT DEFAULT '',              -- last step name pushed via progress (for resume UI)
    progress_json   TEXT DEFAULT '[]',            -- JSON array of {step,status,...} events emitted so far
    owner_token     TEXT DEFAULT ''               -- per-browser owner token, scopes recent-runs listing
);
CREATE INDEX IF NOT EXISTS idx_runs_status_started ON runs(status, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_paper ON runs(paper_id);
-- NOTE: idx_runs_owner_started is created in database.py AFTER the owner_token
-- column migration. It must not live here: on legacy DBs the runs table already
-- exists without owner_token, so this script runs before the column is added and
-- an inline index on owner_token would raise "no such column" and abort init.

CREATE TABLE IF NOT EXISTS run_outputs (
    run_id   TEXT PRIMARY KEY REFERENCES runs(run_id),
    markdown TEXT DEFAULT '',
    json_data TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sphere_nodes (
    node_id TEXT NOT NULL,
    run_id  TEXT NOT NULL REFERENCES runs(run_id),
    doi     TEXT DEFAULT '',
    arxiv_id TEXT DEFAULT '',
    openalex_id TEXT DEFAULT '',
    s2_paper_id TEXT DEFAULT '',
    title   TEXT DEFAULT '',
    year    INTEGER DEFAULT 0,
    venue   TEXT DEFAULT '',
    authors TEXT DEFAULT '',
    abstract_text TEXT DEFAULT '',
    cited_by_count INTEGER DEFAULT 0,
    pdf_path TEXT DEFAULT '',
    mineru_parsed INTEGER DEFAULT 0,
    source  TEXT DEFAULT 'seed_ref',
    score_total REAL DEFAULT 0.0,
    layer   INTEGER DEFAULT 0,
    cluster_id INTEGER DEFAULT -1,
    PRIMARY KEY (node_id, run_id)
);
CREATE INDEX IF NOT EXISTS idx_sphere_nodes_run ON sphere_nodes(run_id);

CREATE TABLE IF NOT EXISTS sphere_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    edge_type TEXT NOT NULL DEFAULT 'cites',
    weight REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_sphere_edges_run ON sphere_edges(run_id);
