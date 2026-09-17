-- ── Run leases and heartbeats ───────────────────────────────────────────────
-- A turn is executed by a worker. Without a record of *which* worker and when
-- it was last alive, a run left in `running` by a crash is indistinguishable
-- from one that is merely slow, and the only available recovery is "assume
-- everything running at startup is dead" — which is wrong the moment a second
-- process exists, and untestable without actually killing one.
--
-- With a heartbeat, an orphan is a fact rather than an assumption: a run whose
-- worker has not checked in for longer than the interval is not being executed
-- by anyone, whoever is asking and whenever they ask (acceptance case A13).

ALTER TABLE agent_runs ADD COLUMN worker_id TEXT NOT NULL DEFAULT '';
ALTER TABLE agent_runs ADD COLUMN heartbeat_at TEXT DEFAULT NULL;

-- The orphan scan reads exactly this: active runs ordered by how stale they are.
CREATE INDEX IF NOT EXISTS idx_agent_runs_heartbeat
    ON agent_runs(status, heartbeat_at) WHERE status IN ('pending', 'running');

-- Jobs already carry a lease. What they lacked is a way to find the ones a
-- crash left behind without scanning every job ever run.
CREATE INDEX IF NOT EXISTS idx_agent_jobs_running
    ON agent_jobs(status, lease_expires_at) WHERE status = 'running';
