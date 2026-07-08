#!/usr/bin/env python3
"""Auto-generate config_env spectrum tasks (BENCH_SPECTRUM.md).

"Where does this app read the ENV_VAR environment variable?" is a real,
common developer question, and one plain grep answers only if you already
know the exact spelling and access idiom. The FLOOR is fully mechanical:
rg finds every file that reads a given env var, so the gold answer is the
set of those files — no authoring, no LLM judge.

For each repo we find env-var reads (Python os.environ / getenv, Rust
std::env::var, JS process.env, Go os.Getenv), pick vars read in exactly
one or two files (a crisp, checkable answer), and emit one natural
question per var with a floor_terms check on the reading file(s).

Usage:
    python3 spectrum_env.py --repos-root realcorpus \
        --repos serde flask gson zod tokio django \
        --out tasks_config_env.json [--per-repo 4]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

# (language label, regex capturing the env-var NAME) — fixed idioms only,
# so a match is unambiguously an env-var read.
PATTERNS = [
    r"os\.environ(?:\.get)?\(\s*['\"]([A-Z_][A-Z0-9_]+)['\"]",   # py subscript/get
    r"getenv\(\s*['\"]([A-Z_][A-Z0-9_]+)['\"]",                    # py/c getenv
    r"std::env::var(?:_os)?\(\s*['\"]([A-Z_][A-Z0-9_]+)['\"]",     # rust
    r"process\.env\.([A-Z_][A-Z0-9_]+)\b",                          # js
    r"os\.Getenv\(\s*['\"]([A-Z_][A-Z0-9_]+)['\"]",               # go
]


def env_reads(root: pathlib.Path) -> dict[str, set[str]]:
    """Map env-var name -> set of files (repo-relative) that read it."""
    out: dict[str, set[str]] = {}
    for pat in PATTERNS:
        p = subprocess.run(
            ["rg", "--no-heading", "--with-filename", "-o", "-r", "$1", pat, str(root)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        # rg -o -r prints "path:VARNAME"; recover both.
        p2 = subprocess.run(
            ["rg", "--no-heading", "--with-filename", "-o", pat, str(root)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        for line in p2.stdout.splitlines():
            if ":" not in line:
                continue
            path, _, hit = line.partition(":")
            m = re.search(pat, hit)
            if not m:
                continue
            var = m.group(1)
            rel = str(pathlib.Path(path).relative_to(root))
            out.setdefault(var, set()).add(rel)
    return out


def build(repos_root: pathlib.Path, repos: list[str], per_repo: int) -> list[dict]:
    tasks: list[dict] = []
    n = 0
    for repo in repos:
        root = repos_root / repo
        if not root.exists():
            print(f"[env] skip missing {root}", file=sys.stderr)
            continue
        reads = env_reads(root)
        # crisp answers: vars read in 1-2 files, deterministic order.
        crisp = sorted(
            (v for v, files in reads.items() if 1 <= len(files) <= 2 and len(v) >= 5),
            key=lambda v: (len(reads[v]), v),
        )
        picked = 0
        for var in crisp:
            if picked >= per_repo:
                break
            files = sorted(reads[var])
            n += 1
            picked += 1
            tasks.append({
                "id": f"ce{n:03d}",
                "repo": repo,
                "lang": "mixed",
                "type": "where",
                "class": "config_env",
                "q": f"Where does this codebase read the {var} environment "
                     f"variable, and what does it do with it?",
                "ground_truth": f"{var} is read in: {', '.join(files)}.",
                "check": {
                    "kind": "floor_terms",
                    # the var name plus a reading file; naming either the
                    # file OR the var + purpose counts as finding it.
                    "terms": [var] + [f.rsplit("/", 1)[-1] for f in files],
                    "min_hits": 2,
                    "semantics": "floor",
                },
            })
        print(f"[env] {repo}: {picked} tasks from {len(crisp)} crisp vars "
              f"({len(reads)} total env reads)")
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-root", type=pathlib.Path, required=True)
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--per-repo", type=int, default=4)
    a = ap.parse_args()
    tasks = build(a.repos_root, a.repos, a.per_repo)
    a.out.write_text(json.dumps(tasks, indent=1))
    print(f"[env] wrote {len(tasks)} config_env tasks -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
