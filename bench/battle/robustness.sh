#!/usr/bin/env bash
# ROBUSTNESS battle — lock in the field bugs found in the 2026-07 hardening
# batch so they can never silently regress. Every assertion is on COMMAND
# OUTPUT of the built binary (black-box), never on raw SQLite.
#
# Covered:
#   O9  parent .gitignore must NOT gut a nested repo's index
#   P10 find-usages must not answer a false "(no usages)" for a call-only
#       symbol that who-calls demonstrably links
#   P4  who-calls / find-usages carry the grep-shaped call-site line
#
# Black-box: drives the built binary only; touches no crate source.

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

NAME="robustness"
require_bins "$GREPPY_BIN" || { emit_summary "$NAME"; exit 1; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/battle-robust-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
export GREPPY_STORE_DIR="$WORK/store"
export GREPPY_AUTO_REINDEX=1

# ---------------------------------------------------------------------------
# O9 — a nested git repo under a parent whose .gitignore has an unanchored
# `*/` must still index all its own subdirectory files.
# ---------------------------------------------------------------------------
PARENT="$WORK/vendored"
REPO="$PARENT/dep"
mkdir -p "$REPO/src"
printf '*/\n' > "$PARENT/.gitignore"          # would hide every subdir
mkdir -p "$REPO/.git"                           # mark REPO as its own repo
cat > "$REPO/src/lib.rs" <<'RS'
pub fn wrap_helper() { let _ = crate::inner::do_inner(); }
RS
mkdir -p "$REPO/src/inner"
cat > "$REPO/src/inner/mod.rs" <<'RS'
pub fn do_inner() -> i32 { 42 }
RS

"$GREPPY_BIN" index "$REPO" --root "$REPO" >/dev/null 2>&1
who_out="$("$GREPPY_BIN" who-calls do_inner --root "$REPO" 2>&1)"
if grep -q "wrap_helper" <<<"$who_out"; then
    pass "O9: nested-repo src/ files indexed (who-calls do_inner finds wrap_helper)"
else
    fail "O9: parent '*/' gutted the nested index — who-calls do_inner missed wrap_helper: $who_out"
fi

# P4 — the call-site line must appear grep-shaped under the caller.
if grep -Eq "src/lib.rs:[0-9]+: .*do_inner" <<<"$who_out"; then
    pass "P4: who-calls prints the grep-shaped call-site line"
else
    fail "P4: who-calls missing the call-site evidence line: $who_out"
fi

# ---------------------------------------------------------------------------
# P10 — find-usages of a call-only symbol must not answer "(no usages)"
# when who-calls links it. Same fixture: do_inner is only ever called.
# ---------------------------------------------------------------------------
fu_out="$("$GREPPY_BIN" find-usages do_inner --root "$REPO" 2>&1)"
if grep -q "no usages" <<<"$fu_out"; then
    fail "P10: find-usages answered a false '(no usages)' for a linked call target: $fu_out"
else
    if grep -q "wrap_helper" <<<"$fu_out"; then
        pass "P10: find-usages reports the call reference (no false zero)"
    else
        fail "P10: find-usages did not report the known reference: $fu_out"
    fi
fi

emit_summary "$NAME"
