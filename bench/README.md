# greppy benchmark suite

The `bench/` directory contains the empirical-heuristic benchmark
corpus required before the default heuristic is
declared stable. There are three scripts, each covering one of the
three sub-areas:

| Script | Corpus | What it measures |
|--------|--------|------------------|
| `grep_compat.sh`     | grep-compatibility corpus | Byte-exact stdout/stderr/exit-code preservation against real `/usr/bin/grep` for 35 representative invocations, split across the three heuristic classes (Strict / Sidecar / VisibleAugment). |
| `agent_utility.sh`   | agent-utility corpus        | Real agent-style invocations: sidecar presence, sentinel, exit-code preservation, and the "no synthetic line on miss" rule. |
| `freshness_bench.sh` | freshness benchmark          | 9 mutation scenarios (cold start, fresh after index, edit, delete, add, rename, commit, branch, agent-temp-file) asserted against `greppy_freshness::check_files` via the `freshness-probe` example binary. |

A combined runner is provided as `run_all.sh`.

## Running

```bash
cargo build --workspace --examples
bash bench/run_all.sh
```

You can also run an individual script directly:

```bash
bash bench/grep_compat.sh
bash bench/agent_utility.sh
bash bench/freshness_bench.sh
```

All scripts are non-destructive: they reindex the fixture at
`bench/fixtures/sample/` from scratch and clean up after themselves.
The fixture's git state is committed at the start of the run; each
mutation scenario resets the working tree via `git clean -fdx` +
`git checkout -- .` so successive runs are reproducible.

## How to read the output

Each script prints a `summary` block at the end:

```text
=== grep_compat.sh summary ===
pass: 35
fail: 0
```

A `pass` is an invocation that satisfied the class's contract:

- **Strict / Sidecar** — full byte-exact stdout + stderr + exit code
  against real grep.
- **VisibleAugment** — real-grep output is a byte-exact prefix of
  subject output; the suffix contains at least one labelled synthetic
  line (`GREPPY_NON_CANONICAL_HIT`); exit code matches.

A `fail` is an invocation that violated the contract. The script
prints the expected vs. actual output so a regression can be
diagnosed quickly.

`agent_utility.sh` adds a per-invocation table with `rc / real_b /
sub_b / delta_b / sidecar_b / synth_n / side / sentinel?` columns.
These help you see how much extra context greppy surfaces on top
of raw grep.

`freshness_bench.sh` prints `expect / actual / elapsed_ms` per
scenario. Elapsed is the per-check wall time; the production
`greppy-grep` gate is budgeted at 200 ms (per-invocation,
search-path-scoped), while the bench probe uses 30 s because it
walks the whole repo.

## Fixture

`bench/fixtures/sample/` is a hand-crafted Rust project with the
symbols the corpora query. It is committed to a local git repo so
the freshness bench can also exercise the git-fingerprint path.

- `src/lib.rs` — `Greeter`, `ProcessOrder`, `UserService`,
  `InMemoryUserService`, `hello`
- `src/greeter.rs` — secondary module
- `src/orders.rs` — order-handling helpers
- `src/script.py` — Python file (exercises `Language::Unsupported`)

## Adding a new bench entry

1. Decide which script the entry belongs to:
   - Pipeline-sensitive grep invocation → `grep_compat.sh` with
     class `STRICT` or `SIDECAR`.
   - Agent-style exploration with augmentation → `grep_compat.sh`
     with class `VISIBLE_AUGMENT`, or `agent_utility.sh`.
   - Workspace mutation → `freshness_bench.sh`.
2. Append the entry to the `CORPUS` (or `probe` call) in the
   appropriate script.
3. Re-run `bench/run_all.sh` and verify the entry passes.
4. Commit the new entry + any fixture changes together.

## What this suite does NOT measure

- Statistical agent-utility numbers (real LLM traces, real Bash
  chains, real tool-call counts) — those belong to the
  agent-utility corpus but require running an actual coding agent.
  This corpus is a smoke-test, not a benchmark.
- Performance / wall-time regression detection — the freshness
  bench reports `elapsed_ms` but does not assert on it. A real
  performance suite would gate on a budget.
- Cross-platform behavior — the scripts assume `/usr/bin/grep` and
  macOS-style temp dirs (`$TMPDIR`). A Windows run would need the
  paths and the `find` calls adjusted; that is a future task.
