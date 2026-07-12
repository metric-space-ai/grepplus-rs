#!/usr/bin/env bash
# End-to-end acceptance for an unpacked Unix release artifact.

set -euo pipefail

BIN="${1:?usage: release_package_smoke.sh /path/to/greppy [work-dir]}"
WORK="${2:-$(mktemp -d "${TMPDIR:-/tmp}/greppy-release-smoke-XXXXXX")}"
mkdir -p "$WORK/repo/src" "$WORK/repo/.git" "$WORK/store"

cat >"$WORK/repo/src/lib.rs" <<'RS'
pub fn apply_limit(value: i32) -> i32 { value.clamp(0, 100) }
pub fn process_value(value: i32) -> i32 { apply_limit(value) }
pub fn normalize_score(value: i32) -> i32 { value.max(0) }
pub fn validate_score(value: i32) -> bool { value <= 100 }
pub fn default_score() -> i32 { 50 }
pub fn minimum_score() -> i32 { 0 }
pub fn maximum_score() -> i32 { 100 }
RS

export GREPPY_STORE_DIR="$WORK/store"
export GREPPY_EMBED_DAEMON_MODEL_TTL_S=5
export GREPPY_EMBED_DAEMON_EXIT_TTL_S=15
export GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S=5
export GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S=15

"$BIN" --help >/dev/null
"$BIN" --device cpu --root "$WORK/repo" doctor --json >"$WORK/doctor.json" || test $? -eq 1
jq -e '.command == "doctor" and .inference.registry.selected_backend == "cpu"' "$WORK/doctor.json" >/dev/null

"$BIN" --device cpu --root "$WORK/repo" index "$WORK/repo" >"$WORK/index.txt"
"$BIN" --device cpu --root "$WORK/repo" brief apply_limit --json >"$WORK/brief.json"
jq -e '
  .schema_version == "greppy.brief.v1" and
  .status == "ok" and
  (.definitions | length) >= 1 and
  (.definitions[0].end_line >= .definitions[0].start_line) and
  (.definitions[0].signature | type == "string" and length > 0) and
  (.definitions[0].summary | length) >= 1 and
  (.expand_id | type == "string" and length > 0)
' "$WORK/brief.json" >/dev/null
brief_expand="$(jq -r '.expand_id' "$WORK/brief.json")"
"$BIN" --root "$WORK/repo" expand "$brief_expand" --json >"$WORK/brief-expand.json"
jq -e --arg id "$brief_expand" '.id == $id and (.payload_text | contains("apply_limit"))' "$WORK/brief-expand.json" >/dev/null

"$BIN" --device cpu --root "$WORK/repo" semantic-search \
  "restrict a numeric value to an allowed range" --json >"$WORK/semantic.json"
jq -e '
  .schema_version == "greppy.semantic-search.v1" and
  .status == "ok" and
  (.hits | length) >= 1 and
  (all(.hits[]; (.end_line >= .start_line) and (.signature | type == "string" and length > 0))) and
  (any(.hits[]; (.summary | length) >= 1)) and
  (.expand_id | type == "string" and length > 0)
' "$WORK/semantic.json" >/dev/null
semantic_expand="$(jq -r '.expand_id' "$WORK/semantic.json")"
"$BIN" --root "$WORK/repo" expand "$semantic_expand" --json >"$WORK/semantic-expand.json"
jq -e --arg id "$semantic_expand" '.id == $id and (.payload_text | length > 0)' "$WORK/semantic-expand.json" >/dev/null

printf 'release package inference smoke passed: %s\n' "$BIN"
