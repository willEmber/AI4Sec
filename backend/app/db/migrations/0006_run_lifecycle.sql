-- Mode runs get the same liveness signal agent turns have: the worker
-- executing one stamps its identity and reports every few seconds, so a run
-- whose worker is gone can be told from one that is merely slow.
ALTER TABLE runs ADD COLUMN worker_id    TEXT NOT NULL DEFAULT '';
ALTER TABLE runs ADD COLUMN heartbeat_at TIMESTAMPTZ DEFAULT NULL;
ALTER TABLE runs ADD COLUMN attempts     INTEGER NOT NULL DEFAULT 0;
CREATE INDEX idx_runs_active ON runs(heartbeat_at) WHERE status IN ('pending', 'running');

-- Whatever is still marked active belongs to a process from before heartbeats.
-- Closing it here keeps the first recovery sweep from re-running, and paying
-- for, a run that was abandoned long ago.
UPDATE runs
   SET status = 'failed',
       error_msg = 'Interrupted (task no longer running)',
       finished_at = now()
 WHERE status IN ('pending', 'running');
