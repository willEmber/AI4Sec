-- An answer is no longer either whole or absent (P13). A turn that fails, is
-- stopped by the reader or is cut off by the model's output ceiling keeps the
-- text it had produced, and the row says which of those it is.
--   completion:  complete | stopped | truncated | failed
--   stop_reason: why — a budget ceiling, `cancelled`, an error code. Also set
--                on a complete answer that was written because a budget ran out.
ALTER TABLE agent_messages ADD COLUMN completion TEXT NOT NULL DEFAULT 'complete';
ALTER TABLE agent_messages ADD COLUMN stop_reason TEXT NOT NULL DEFAULT '';
