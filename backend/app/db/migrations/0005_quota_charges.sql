-- Costly actions that leave no row of their own to count against the daily
-- quota (a library question). Agent turns and mode runs are counted from
-- agent_runs and runs.
CREATE TABLE quota_charges (
    charge_id  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    owner_id   TEXT NOT NULL,
    kind       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_quota_charges_owner_created ON quota_charges(owner_id, created_at DESC);
