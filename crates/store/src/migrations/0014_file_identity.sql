-- File identity is deliberately separate from the historical file_state
-- layout so existing databases migrate without rewriting that hot table.
CREATE TABLE IF NOT EXISTS file_identity (
    project  TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    ctime_ns INTEGER,
    file_id  INTEGER,
    PRIMARY KEY (project, rel_path),
    FOREIGN KEY (project, rel_path)
        REFERENCES file_state(project, rel_path) ON DELETE CASCADE
);

-- The experimental deterministic/Qwen summary cache was never used by the
-- production query path. V2 stores deliberately do not persist summaries.
DROP TABLE IF EXISTS project_summaries;
