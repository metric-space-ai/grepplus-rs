-- 0012: first-class multi-chunk vector embeddings.
--
-- SQLite cannot add a column to the table-level UNIQUE constraint created in
-- 0010, so rebuild the table without that constraint and replace it with an
-- explicit unique index that includes chunk_idx. Existing single-span rows
-- become chunk 0.

DROP INDEX IF EXISTS idx_vector_embeddings_scope;
DROP INDEX IF EXISTS idx_vector_embeddings_file;
DROP INDEX IF EXISTS idx_vector_embeddings_node;
DROP INDEX IF EXISTS idx_vector_embeddings_uniq;

ALTER TABLE vector_embeddings RENAME TO vector_embeddings_old_0012;

CREATE TABLE vector_embeddings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    project          TEXT    NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    model_id         TEXT    NOT NULL,
    prompt_version   TEXT    NOT NULL,
    task             TEXT    NOT NULL,
    node_id          INTEGER REFERENCES nodes(id) ON DELETE CASCADE,
    chunk_idx        INTEGER NOT NULL DEFAULT 0,
    qualified_name   TEXT    NOT NULL,
    file_path        TEXT    NOT NULL,
    start_line       INTEGER NOT NULL DEFAULT 0,
    end_line         INTEGER NOT NULL DEFAULT 0,
    content_sha256   TEXT    NOT NULL,
    graph_generation INTEGER NOT NULL DEFAULT 0,
    dim              INTEGER NOT NULL,
    vector_norm      REAL    NOT NULL,
    vector           BLOB    NOT NULL,
    created_at       TEXT    NOT NULL,
    vector_i8        BLOB,
    i8_scale         REAL
);

INSERT INTO vector_embeddings (
    id, project, model_id, prompt_version, task, node_id, chunk_idx,
    qualified_name, file_path, start_line, end_line, content_sha256,
    graph_generation, dim, vector_norm, vector, created_at, vector_i8, i8_scale
)
SELECT
    id, project, model_id, prompt_version, task, node_id, 0,
    qualified_name, file_path, start_line, end_line, content_sha256,
    graph_generation, dim, vector_norm, vector, created_at, vector_i8, i8_scale
FROM vector_embeddings_old_0012;

DROP TABLE vector_embeddings_old_0012;

CREATE UNIQUE INDEX IF NOT EXISTS idx_vector_embeddings_uniq
    ON vector_embeddings(
        project, model_id, prompt_version, task, qualified_name, chunk_idx, content_sha256
    );

CREATE INDEX IF NOT EXISTS idx_vector_embeddings_scope
    ON vector_embeddings(project, model_id, prompt_version, task, graph_generation);

CREATE INDEX IF NOT EXISTS idx_vector_embeddings_file
    ON vector_embeddings(project, file_path);

CREATE INDEX IF NOT EXISTS idx_vector_embeddings_node
    ON vector_embeddings(node_id);
