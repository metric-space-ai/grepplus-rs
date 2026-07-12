#!/usr/bin/env bash
# Agent-style grep passthrough corpus. Semantic value is exposed only through
# explicit greppy commands; ordinary grep invocations remain byte-exact.

set -uo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GREPPY_BIN="${GREPPY_BIN:-$WORKSPACE_ROOT/target/debug/greppy}"
REAL_GREP="${REAL_GREP:-$(command -v grep)}"
CORPUS_SRC="${CORPUS_SRC:-$WORKSPACE_ROOT/bench/fixtures/sample}"

CORPUS_ROOT="$(mktemp -d -t greppy-agent-utility.XXXXXX)"
STORE_ROOT="$(mktemp -d -t greppy-agent-store.XXXXXX)"
trap 'rm -rf "$CORPUS_ROOT" "$STORE_ROOT"' EXIT
cp -R "$CORPUS_SRC/." "$CORPUS_ROOT/"
rm -rf "$CORPUS_ROOT/.greppy" "$CORPUS_ROOT/.git"
export GREPPY_STORE_DIR="$STORE_ROOT"

# Keep a fresh graph in scope to prove it cannot alter ordinary grep output.
GREPPY_TEST_SKIP_INFERENCE=1 "$GREPPY_BIN" index "$CORPUS_ROOT" >/dev/null 2>&1

CORPUS=(
  "-R|hello|."
  "-R|ProcessOrder|."
  "-R|UserService|."
  "-R|total|."
  "-R|build_default_order|."
  "-R|Greeter|."
  "-R|fmt|."
  "-R|payment_retry|."
  "-R|process_payment|."
  "-R|nonexistent_symbol_xyz|."
  "-Rn|hello|."
  "-Rc|hello|."
)

printf "%-50s %-6s %-12s %-12s %s\n" "command" "rc" "stdout_b" "stderr_b" "contract"
echo "-----------------------------------------------------------------------------------------------"

pass=0
fail=0
declare -a failures

for entry in "${CORPUS[@]}"; do
  IFS='|' read -r -a argv <<< "$entry"
  pretty="${argv[*]}"
  sub_out="$(mktemp)"; sub_err="$(mktemp)"
  ref_out="$(mktemp)"; ref_err="$(mktemp)"

  ( cd "$CORPUS_ROOT" && "$GREPPY_BIN" "${argv[@]}" ) >"$sub_out" 2>"$sub_err"
  sub_rc=$?
  ( cd "$CORPUS_ROOT" && "$REAL_GREP" "${argv[@]}" ) >"$ref_out" 2>"$ref_err"
  ref_rc=$?

  stdout_b=$(wc -c <"$sub_out" | tr -d ' ')
  stderr_b=$(wc -c <"$sub_err" | tr -d ' ')
  if [[ "$sub_rc" -eq "$ref_rc" ]] && cmp -s "$sub_out" "$ref_out" \
      && cmp -s "$sub_err" "$ref_err"; then
    printf "%-50s %-6s %-12s %-12s %s\n" "$pretty" "$sub_rc" "$stdout_b" "$stderr_b" byte-exact
    pass=$((pass + 1))
  else
    printf "%-50s %-6s %-12s %-12s %s\n" "$pretty" "$sub_rc" "$stdout_b" "$stderr_b" FAIL
    failures+=("$pretty: rc $sub_rc/$ref_rc or byte mismatch")
    diff -u "$ref_out" "$sub_out" | head -12
    diff -u "$ref_err" "$sub_err" | head -12
    fail=$((fail + 1))
  fi
  rm -f "$sub_out" "$sub_err" "$ref_out" "$ref_err"
done

echo ""
echo "=== agent_utility.sh summary ==="
echo "pass: $pass"
echo "fail: $fail"
if [[ "$fail" -gt 0 ]]; then
  printf '  - %s\n' "${failures[@]}"
fi
[[ "$fail" -eq 0 ]]
