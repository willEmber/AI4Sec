-- A resumable, reader-visible snapshot, including queue and page progress.
ALTER TABLE mineru_parses ADD COLUMN progress_json TEXT NOT NULL DEFAULT '{}';
