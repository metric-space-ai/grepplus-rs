#!/usr/bin/env bash
# End-to-end acceptance for an unpacked Unix release artifact.
#
# The script is copied verbatim into the release tarball (see
# .github/workflows/release.yml) and must stay self-contained: every fixture
# is generated inline, and only POSIX-ish tooling that exists on the ubuntu
# and macos runners is used (bash 3.2+, jq, cmp, find, pgrep, shasum or
# sha256sum).

set -euo pipefail

BIN="${1:?usage: release_package_smoke.sh /path/to/greppy [work-dir]}"
WORK="${2:-$(mktemp -d "${TMPDIR:-/tmp}/greppy-release-smoke-XXXXXX")}"
mkdir -p "$WORK/repo/src" "$WORK/repo/.git" "$WORK/store"

section() { printf '\n=== %s ===\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }


cat >"$WORK/repo/src/lib.rs" <<'RS'
pub fn apply_limit(value: i32) -> i32 { value.clamp(0, 100) }
pub fn process_value(value: i32) -> i32 { apply_limit(value) }
pub fn normalize_score(value: i32) -> i32 { value.max(0) }
pub fn validate_score(value: i32) -> bool { value <= 100 }
pub fn default_score() -> i32 { 50 }
pub fn minimum_score() -> i32 { 0 }
pub fn maximum_score() -> i32 { 100 }
RS

# serde-shaped fixture: mirrors serde_derive/src/internals/{case,attr}.rs so
# the exact-hit assertions exercise the symbols the public benchmarks use
# (bench/agent_efficiency, bench/runtime_footprint.py: apply_to_field).
cat >"$WORK/repo/src/case.rs" <<'RS'
#[derive(Copy, Clone)]
pub enum RenameRule {
    LowerCase,
    UpperCase,
    SnakeCase,
}

impl RenameRule {
    /// Apply a rename case rule to a struct field name.
    pub fn apply_to_field(self, field: &str) -> String {
        match self {
            RenameRule::LowerCase => field.to_lowercase(),
            RenameRule::UpperCase => field.to_uppercase(),
            RenameRule::SnakeCase => field.to_string(),
        }
    }
}

pub struct RenameAllRules {
    pub serialize: RenameRule,
    pub deserialize: RenameRule,
}

pub struct Name {
    pub serialize: String,
    pub deserialize: String,
}

impl Name {
    /// Rename the serialize and deserialize names by the container rules.
    pub fn rename_by_rules(&mut self, rules: &RenameAllRules) {
        self.serialize = rules.serialize.apply_to_field(&self.serialize);
        self.deserialize = rules.deserialize.apply_to_field(&self.deserialize);
    }

    /// Return the field name used when serializing.
    pub fn serialize_name(&self) -> &str {
        &self.serialize
    }
}
RS

export GREPPY_STORE_DIR="$WORK/store"
export GREPPY_EMBED_DAEMON_MODEL_TTL_S=5
export GREPPY_EMBED_DAEMON_EXIT_TTL_S=15
export GREPPY_SUMMARIZE_DAEMON_MODEL_TTL_S=5
export GREPPY_SUMMARIZE_DAEMON_EXIT_TTL_S=15

# --- baseline: doctor, index, JSON brief/semantic-search, expand ------------
section "baseline: doctor, index, JSON brief + semantic-search + expand"

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
semantic_omitted="$(jq -r '.omitted' "$WORK/semantic.json")"
jq -e --arg id "$semantic_expand" --argjson omitted "$semantic_omitted" '
  .id == $id and
  (.payload_text | length > 0) and
  .payload_json.further_hits == $omitted and
  (.payload_json.hits | length) == $omitted
' "$WORK/semantic-expand.json" >/dev/null

# --- text output mode: prescribed shape and deterministic ordering ----------
# Contracts under test:
# * brief (text): dispatch_brief in crates/cli/src/lib.rs prints, in this
#   fixed order: the definition header `== NAME (file:start-end) ==`, then
#   `-- CALLERS (n) --`, then (non-callable targets only) `-- REFERENCES
#   (n) --`, then `-- CALLS (n) --`, then the trailing
#   `Expand: greppy expand <id>` line (ExpandHandle::text_line).
# * semantic-search (text): print_semantic_vector_hit in crates/cli/src/lib.rs
#   prints one block per hit — a bare `file:start-end` locator line, an
#   indented signature, indented purpose bullets — followed by the trailing
#   `greppy expand <id>  → source evidence …` line
#   (ExpandHandle::semantic_text_line).
# * Hit ordering: crates/store/src/vector_embedding.rs vector_search_exact:
#   "Ranking is total and deterministic: score descending, then
#   `qualified_name`, then row id." The JSON hits array is rendered from the
#   same ranked slice, so text order must equal JSON order, and JSON scores
#   must be non-increasing.
section "text output mode: prescribed shape and deterministic ordering"

"$BIN" --device cpu --root "$WORK/repo" brief apply_limit >"$WORK/brief.txt"
first_match_line() { grep -n "$1" "$2" | head -1 | cut -d: -f1 || true; }
def_line="$(first_match_line '^== .*apply_limit (src/lib.rs:1-1) ==$' "$WORK/brief.txt")"
callers_line="$(first_match_line '^-- CALLERS ([0-9]*) --$' "$WORK/brief.txt")"
calls_line="$(first_match_line '^-- CALLS ([0-9]*) --$' "$WORK/brief.txt")"
expand_line="$(first_match_line '^Expand: greppy expand ' "$WORK/brief.txt")"
[ -n "$def_line" ] || fail "brief text: missing '== …apply_limit (src/lib.rs:1-1) ==' definition header"
[ -n "$callers_line" ] || fail "brief text: missing '-- CALLERS (n) --' section"
[ -n "$calls_line" ] || fail "brief text: missing '-- CALLS (n) --' section"
[ -n "$expand_line" ] || fail "brief text: missing trailing 'Expand: greppy expand' line"
[ "$def_line" -lt "$callers_line" ] || fail "brief text: definition must precede CALLERS"
[ "$callers_line" -lt "$calls_line" ] || fail "brief text: CALLERS must precede CALLS"
[ "$calls_line" -lt "$expand_line" ] || fail "brief text: CALLS must precede the Expand line"
[ "$expand_line" -eq "$(grep -c '' "$WORK/brief.txt")" ] || fail "brief text: Expand line must be the last line"
grep -q 'process_value src/lib.rs:2-2$' "$WORK/brief.txt" \
  || fail "brief text: expected caller row for process_value at src/lib.rs:2-2"

# JSON scores must be non-increasing (the ranked half of the contract).
jq -e '[.hits[].score] | . == (sort | reverse)' "$WORK/semantic.json" >/dev/null \
  || fail "semantic-search JSON: hit scores are not in descending order"

semantic_locs_from_text() {
  # Locator lines are the only non-indented lines apart from the trailing
  # expand handle; blocks are blank-line separated.
  awk '/^[^ ]/ && $0 !~ /^greppy expand / && NF > 0' "$1"
}

"$BIN" --device cpu --root "$WORK/repo" semantic-search \
  "restrict a numeric value to an allowed range" >"$WORK/semantic.txt"
grep -Eq '^greppy expand [^ ]+  → source evidence for ' "$WORK/semantic.txt" \
  || fail "semantic-search text: missing trailing 'greppy expand <id>' evidence line"
semantic_locs_from_text "$WORK/semantic.txt" >"$WORK/semantic-locs-text.txt"
[ -s "$WORK/semantic-locs-text.txt" ] || fail "semantic-search text: no hit locator lines found"
grep -Eq '^src/[a-z_]+\.rs:[0-9]+(-[0-9]+)?$' "$WORK/semantic-locs-text.txt" \
  || fail "semantic-search text: locator lines do not look like file:start-end"

# Text order must equal the ranked JSON order for the same query.
jq -r '.hits[] | .summary_loc // "\(.file_path):\(.start_line)-\(.end_line)"' \
  "$WORK/semantic.json" >"$WORK/semantic-locs-json.txt"
cmp -s "$WORK/semantic-locs-text.txt" "$WORK/semantic-locs-json.txt" \
  || { diff -u "$WORK/semantic-locs-json.txt" "$WORK/semantic-locs-text.txt" >&2 || true; \
       fail "semantic-search: text hit order diverges from ranked JSON order"; }

# Repeating the query must reproduce the same ordering (determinism).
"$BIN" --device cpu --root "$WORK/repo" semantic-search \
  "restrict a numeric value to an allowed range" >"$WORK/semantic-rerun.txt"
semantic_locs_from_text "$WORK/semantic-rerun.txt" >"$WORK/semantic-locs-rerun.txt"
cmp -s "$WORK/semantic-locs-text.txt" "$WORK/semantic-locs-rerun.txt" \
  || fail "semantic-search text: hit ordering is not deterministic across reruns"

# --- exact serde-repo hits ---------------------------------------------------
# The serde-shaped fixture (src/case.rs above) must be resolvable exactly:
# `brief SYMBOL` resolves symbol names via the graph, so each of the three
# serde symbols must come back as a definition, and a targeted semantic query
# must surface each symbol among the retrieved hits (shown hits + the
# expand-pack remainder = the full ranked retrieval set).
section "exact serde-repo hits: apply_to_field, rename_by_rules, serialize_name"

assert_brief_exact() {
  local symbol="$1"
  "$BIN" --device cpu --root "$WORK/repo" brief "$symbol" --json >"$WORK/brief-$symbol.json"
  jq -e --arg sym "$symbol" '
    .status == "ok" and
    ([.definitions[].qualified_name] | any(contains($sym))) and
    ([.definitions[].file_path] | any(. == "src/case.rs"))
  ' "$WORK/brief-$symbol.json" >/dev/null \
    || fail "brief $symbol: expected an exact definition hit in src/case.rs"
}
assert_brief_exact apply_to_field
assert_brief_exact rename_by_rules
assert_brief_exact serialize_name

assert_semantic_retrieves() {
  local symbol="$1"
  local query="$2"
  local out="$WORK/semantic-$symbol.json"
  "$BIN" --device cpu --root "$WORK/repo" semantic-search "$query" --json >"$out"
  jq -e '.status == "ok" and (.hits | length) >= 1' "$out" >/dev/null \
    || fail "semantic-search '$query': expected status ok with hits"
  jq -r '.hits[].qualified_name' "$out" >"$WORK/semantic-$symbol-names.txt"
  local expand_id
  expand_id="$(jq -r '.expand_id // empty' "$out")"
  if [ -n "$expand_id" ]; then
    "$BIN" --root "$WORK/repo" expand "$expand_id" --json \
      | jq -r '.payload_json.hits[].qualified_name' >>"$WORK/semantic-$symbol-names.txt"
  fi
  grep -q "$symbol" "$WORK/semantic-$symbol-names.txt" \
    || fail "semantic-search '$query': $symbol not in retrieved hit set: $(tr '\n' ' ' <"$WORK/semantic-$symbol-names.txt")"
}
assert_semantic_retrieves apply_to_field "apply a rename case rule to a struct field"
assert_semantic_retrieves rename_by_rules "rename the serialize and deserialize names using the container rules"
assert_semantic_retrieves serialize_name "return the field name used when serializing"

# --- text/JSON parity --------------------------------------------------------
# The same query in text and JSON mode must surface the same hit set: both
# renderers consume the identical ranked slice (dispatch_semantic in
# crates/cli/src/lib.rs), so the normalized `file:start-end` sets must match.
section "text/JSON parity: identical hit set in both modes"

parity_query="apply a rename case rule to a struct field"
"$BIN" --device cpu --root "$WORK/repo" semantic-search "$parity_query" >"$WORK/parity.txt"
"$BIN" --device cpu --root "$WORK/repo" semantic-search "$parity_query" --json >"$WORK/parity.json"
jq -e '.status == "ok"' "$WORK/parity.json" >/dev/null
semantic_locs_from_text "$WORK/parity.txt" | LC_ALL=C sort >"$WORK/parity-locs-text.txt"
jq -r '.hits[] | .summary_loc // "\(.file_path):\(.start_line)-\(.end_line)"' \
  "$WORK/parity.json" | LC_ALL=C sort >"$WORK/parity-locs-json.txt"
[ -s "$WORK/parity-locs-text.txt" ] || fail "parity: text mode returned no hits"
cmp -s "$WORK/parity-locs-text.txt" "$WORK/parity-locs-json.txt" \
  || { diff -u "$WORK/parity-locs-json.txt" "$WORK/parity-locs-text.txt" >&2 || true; \
       fail "parity: text and JSON modes returned different hit sets"; }


printf '\nrelease package inference smoke passed: %s\n' "$BIN"
