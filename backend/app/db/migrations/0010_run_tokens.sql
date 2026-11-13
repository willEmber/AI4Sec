-- What a mode run and a library question spent on the model, so the daily
-- token quota counts them. Until now it summed agent turns only: a report from
-- the upload page and a library answer used tokens that no quota saw, and a
-- report made inside a conversation was missing from that turn's total too.
--
-- A report made inside a turn also records its tokens here, but the quota
-- reads them from the turn (`agent_runs.usage_json`), which includes them;
-- here they only say what that one report cost.
ALTER TABLE runs ADD COLUMN tokens BIGINT NOT NULL DEFAULT 0;
ALTER TABLE quota_charges ADD COLUMN tokens BIGINT NOT NULL DEFAULT 0;
