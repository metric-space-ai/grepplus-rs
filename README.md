# grepplus

**Standard `grep`, plus a few commands your coding agent can use to navigate code — `who-calls`, `impact`, `context`, `brief`. Agents finish code-navigation tasks ~2× faster and ~3–4× cheaper. One native Rust binary.**

You install it as `grep`, and everything works exactly as before — same flags, same output, same exit code. The same binary just *also* answers the questions an agent normally burns rounds on: *who calls this function, what breaks if I change it, where is the code that does X.* One line in your agent's config (below) tells it the extra commands exist, and it stops looping.

```bash
# Standard grep — every command works, unchanged:
grep -rn "TODO" src/
grep -i "connection refused" server.log

# A few extra commands, on the same `grep`:
grep who-calls parse_config                 # who calls this function
grep impact User --direction incoming       # what breaks if I change User
grep context "restrict a value to a range"  # find code by meaning, not keyword
grep brief _split_blueprint_path            # definition + callers + callees, one call
```

<video src="docs/assets/grepplus-pi-code-benchmark-demo.mp4" controls muted width="100%"></video>

---

## Setup — two steps

**1. Install it as `grep`** and index the repo (one time):

```bash
cargo build --release --bin grepplus                              # build the one binary
sudo install -m 0755 target/release/grepplus /usr/local/bin/grep  # install it AS grep (system /bin/grep is untouched)
grepplus index /path/to/repo                                      # index once (setup cost, not per query)
#   for semantic ("find code by meaning") search, add:
#     --embeddings --embedding-gguf <embeddinggemma-300M-Q4_K.gguf> --embedding-tokenizer <tokenizer.json>
```

Prefer a prebuilt binary? Download one for macOS / Linux / Windows from the [Releases](../../releases) page and put it on your `PATH` as `grep`. (Uninstall: remove the binary and `rm -rf "${GREPPLUS_STORE_DIR:-$HOME/Library/Caches/grepplus}"`.)

**2. Tell your agent** — paste the text below into the file your agent reads for project instructions. This is the *only* integration:

| Agent | Paste it into |
|---|---|
| **Claude Code** | `CLAUDE.md` in the repo root |
| **OpenAI Codex** | `AGENTS.md` in the repo root |
| **Cursor** | `.cursor/rules` |
| **Windsurf** | `.windsurfrules` |
| **Anything else / raw API** | the model's **system prompt** |

The exact text to paste:

```text
This project's `grep` is standard grep plus a few extra commands. Every normal
grep command works as usual. It also answers code-navigation questions — prefer
these over grep+read loops:
- grep who-calls SYM / grep callees SYM / grep find-usages SYM
- grep impact SYM --direction incoming      # what breaks if SYM changes
- grep path --from A --to B                  # the call chain from A to B
- grep context "plain-English description"   # find code by meaning, not keyword
- grep brief SYM                             # definition + callers + callees, one call
Be efficient: one impact/context/who-calls call beats many greps. Stop as soon
as you can answer. If a command replies "no index for this repo", run
`grep index .` once (it takes a few seconds), then retry the command.
```

> **On indexing:** the code commands read a small on-disk index built by `grep index .` (step 1 above builds it once; plain grep never needs it). If the repo drifts, grepplus re-indexes the changed files automatically on the next command. A repo that was never indexed returns the `no index` message above rather than blocking — the prompt line tells the agent to run `grep index .` and continue.

That prompt is the whole integration. **Bonus:** even without it, an exploratory `grep` over an indexed repo appends one self-describing context file (definition, callers, suggested next reads, and the command list), so a capable agent can discover the commands from grep's own output.

---

## What it saves

What an agent actually pays for is **billed tokens** and **wall-clock time.**

The benchmark: a real coding agent (MiniMax-M3, driven by [Pi Code](https://pi.dev)) answers **94 code questions** — *who-calls*, impact/blast-radius, call-chain traces, and vocabulary-gap "find the code that does X" — across **four real pinned repositories** (Rust `serde`, Python `flask`, Java `gson`, TypeScript `zod`). The same agent runs each task twice: once with plain `grep`, once with grepplus. The fixed system prompt is warmup and is excluded from every ratio. The harness is in [`bench/agent_efficiency/`](bench/agent_efficiency/) and is reproducible. Medians:

| What you actually pay | Median | |
|---|---:|---|
| **Billed input tokens** | **~3.7×** | cheaper |
| **Output tokens** | **~2.9×** | fewer |
| **Wall-clock time** | **~2.0×** | faster |
| **Tool-call rounds** | **~4.0×** | fewer |

It all comes from one thing: **fewer model round-trips.** The win is largest on structural questions (`who-calls`/`impact`) and vocabulary-gap searches (`context`), and ~1× on a plain literal search, where `grep` is already the right tool.

---

## How it works

- **Standard grep.** Any invocation that isn't one of the extra commands runs the real `grep` and returns its output and exit code unchanged — even for non-UTF-8 patterns. Scripts don't notice.
- **A precomputed code graph.** An indexed, typed symbol graph (`CALLS`/`USES`/`TYPE_REF`/`IMPORTS`) answers `who-calls`/`callees`/`find-usages`/`impact`/`path` directly — resolved relationships with `file:line`, not textual name matches — collapsing several grep+read rounds into one call.
- **Native semantic search.** For a natural-language query that shares no words with the code, it embeds the query with Google's **EmbeddingGemma** (pure-Rust [candle](https://github.com/huggingface/candle) — no llama.cpp, no Python, no HTTP) and does exact cosine nearest-neighbour search over code-span embeddings.
- **One-shot briefings.** `brief SYM` returns definition + callers + callees in one call; `impact SYM` returns the whole transitive blast-radius in one call.
- **Freshness-gated incremental index** so a stale graph never returns a wrong answer; re-indexing reparses only changed files.
- **One native Rust binary.** At runtime it links only system libraries; tree-sitter parsers and SQLite are compiled in statically.

---

## Status

Early and evolving. Language coverage is growing — today ~18 languages are wired for cross-file `CALLS`/`IMPORTS` (Rust also resolves `TYPE_REF`/`USES`). The passthrough intercepts `grep`; `ripgrep` (`rg`) interception is next. Not yet production-ready — use it as a fast code-navigation aid, not a system of record.

The exact benchmark harness lives in [`bench/agent_efficiency/`](bench/agent_efficiency/).

## License

MIT — see [LICENSE](LICENSE). Third-party notices: [THIRD_PARTY.md](THIRD_PARTY.md).
