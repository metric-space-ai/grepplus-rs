-- 0011: int8-quantized copy of each code-span vector.
--
-- WHY: the exact scan is memory-bound. 200k f32 vectors are ~615 MB of
-- blobs, which thrashes SQLite's page cache and turned the measured scan
-- superlinear (50k = 71 ms but 200k = 2.3 s per query). The i8 copy is 4x
-- smaller (768 B/row), restoring cache-resident scans well past the old
-- 50k-span limit. Scoring uses the i8 dot for CANDIDATE SELECTION only;
-- the winners are re-scored exactly from the f32 blob, so ranking quality
-- is unchanged. (A kd-tree was evaluated for this and rejected with data:
-- at 768 dims the measured pruning rate is 0%.)
--
-- Both columns are nullable: rows written before this migration simply
-- have no i8 copy and scan via the f32 path until re-embedded.
ALTER TABLE vector_embeddings ADD COLUMN vector_i8 BLOB;
ALTER TABLE vector_embeddings ADD COLUMN i8_scale REAL;
