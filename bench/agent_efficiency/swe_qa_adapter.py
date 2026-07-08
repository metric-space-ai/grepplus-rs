#!/usr/bin/env python3
"""Adapt SWE-QA-Pro (TIGER-Lab, ACL 2026, MIT) into our tasks_v2 schema.

SWE-QA-Pro is 260 repository questions seeded from real GitHub issues,
human-verified, over 26 pinned repos, with a difficulty filter that removes
anything answerable without exploring the code — exactly the realistic,
hard, non-command-shaped questions the owner asked for.

We keep our mechanical-grading discipline: extract code-shaped identifiers
and file paths from each GOLD answer as FLOOR TERMS, verify each against the
pinned commit with rg (a term rg cannot find is dropped, never kept), and
emit a `floor_terms` check (answer must name >= min_hits of them). No
LLM judge; the question text is the natural gold question, unchanged.

Usage:
    # 1. clone the pinned repos (idempotent):
    python3 swe_qa_adapter.py clone   --src SWEQA.jsonl --repos DIR
    # 2. build the task set (rg-verifies floors against DIR):
    python3 swe_qa_adapter.py build   --src SWEQA.jsonl --repos DIR \
            --out tasks_sweqa.json [--min-hits 2] [--cap-per-repo 10]

Output tasks carry class `swe_qa_real`; run them through the same
run_bench.py + grade_answers.py (floor_terms kind) as every other class.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

# Code-shaped tokens in a gold answer: snake_case, camelCase, PascalCase with
# an inner cap, dotted paths, or a source file name. Plain English words are
# excluded (no separator / no inner capital), so "the class inherits" yields
# nothing but `FALQON`, `QAOA`, `models/variational.py` do.
IDENT = re.compile(
    r"`([^`]+)`"  # anything explicitly backticked (authors mark code)
    r"|\b([A-Za-z_][A-Za-z0-9_]*\.(?:py|rs|ts|tsx|js|go|java|rb|c|cc|cpp|h|hpp))\b"
    r"|\b([a-z][a-z0-9]*_[a-z0-9_]+)\b"  # snake_case
    r"|\b([a-z]+[A-Z][A-Za-z0-9]*)\b"  # camelCase
    r"|\b([A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b"  # PascalCase w/ inner cap
)
STOP = {"README", "TODO", "HTTP", "JSON", "API", "URL", "ID"}


def floor_terms_from_answer(answer: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in IDENT.finditer(answer):
        tok = next((g for g in m.groups() if g), None)
        if not tok:
            continue
        tok = tok.strip().strip("`")
        # A backticked span may hold a call like `foo(x)` — keep the head.
        tok = re.split(r"[(\[<{,\s]", tok, maxsplit=1)[0]
        if len(tok) < 4 or tok in STOP or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def repo_dir(repos: pathlib.Path, repo: str) -> pathlib.Path:
    return repos / repo.replace("/", "__")


def clone(src: pathlib.Path, repos: pathlib.Path) -> int:
    rows = [json.loads(l) for l in src.open()]
    pins = {}
    for r in rows:
        pins.setdefault(r["repo"], r["commit_id"])
    repos.mkdir(parents=True, exist_ok=True)
    fail = 0
    for repo, commit in sorted(pins.items()):
        dst = repo_dir(repos, repo)
        marker = dst / ".sweqa_commit"
        if marker.exists() and marker.read_text().strip() == commit:
            print(f"[clone] {repo} already at {commit[:10]}")
            continue
        if not dst.exists():
            url = f"https://github.com/{repo}.git"
            if subprocess.run(
                ["git", "clone", "--quiet", url, str(dst)]
            ).returncode:
                print(f"[clone] FAILED clone {repo}", file=sys.stderr)
                fail += 1
                continue
        if subprocess.run(
            ["git", "-C", str(dst), "checkout", "--quiet", commit]
        ).returncode:
            print(f"[clone] FAILED checkout {repo}@{commit[:10]}", file=sys.stderr)
            fail += 1
            continue
        marker.write_text(commit)
        print(f"[clone] {repo} -> {commit[:10]}")
    return fail


def rg_present(term: str, root: pathlib.Path) -> bool:
    # Fixed-string, whole-repo; a term the pinned tree does not contain is
    # not a checkable floor and is dropped.
    p = subprocess.run(
        ["rg", "-l", "--fixed-strings", "--max-count", "1", term, str(root)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    return bool(p.stdout.strip())


def build(
    src: pathlib.Path, repos: pathlib.Path, out: pathlib.Path,
    min_hits: int, cap_per_repo: int,
) -> int:
    rows = [json.loads(l) for l in src.open()]
    tasks = []
    per_repo: dict[str, int] = {}
    dropped_no_floor = 0
    n = 0
    for r in sorted(rows, key=lambda r: (r["repo"], r["question"])):
        repo = r["repo"]
        root = repo_dir(repos, repo)
        if not root.exists():
            continue
        if per_repo.get(repo, 0) >= cap_per_repo:
            continue
        raw = floor_terms_from_answer(r["answer"])
        verified = [t for t in raw if rg_present(t, root)]
        if len(verified) < min_hits:
            dropped_no_floor += 1
            continue
        n += 1
        per_repo[repo] = per_repo.get(repo, 0) + 1
        tasks.append({
            "id": f"sq{n:03d}",
            "repo": repo,
            "commit": r["commit_id"],
            "lang": "mixed",
            "type": r["qa_type"]["class_name"].split()[0].lower(),  # what/where/how/why
            "class": "swe_qa_real",
            "q": r["question"],
            "ground_truth": r["answer"][:600],
            "check": {
                "kind": "floor_terms",
                "terms": verified[:12],
                "min_hits": min_hits,
                "semantics": "floor",
            },
            "source": {"benchmark": "SWE-QA-Pro", "cluster": r["cluster"]["name"],
                       "sub_class": r["qa_type"]["sub_class_name"]},
        })
    out.write_text(json.dumps(tasks, indent=1))
    print(f"[build] wrote {len(tasks)} tasks -> {out}")
    print(f"[build] per-repo: {dict(sorted(per_repo.items()))}")
    print(f"[build] dropped (floor < {min_hits} verified terms): {dropped_no_floor}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("clone")
    c.add_argument("--src", type=pathlib.Path, required=True)
    c.add_argument("--repos", type=pathlib.Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", type=pathlib.Path, required=True)
    b.add_argument("--repos", type=pathlib.Path, required=True)
    b.add_argument("--out", type=pathlib.Path, required=True)
    b.add_argument("--min-hits", type=int, default=2)
    b.add_argument("--cap-per-repo", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "clone":
        return clone(a.src, a.repos)
    return build(a.src, a.repos, a.out, a.min_hits, a.cap_per_repo)


if __name__ == "__main__":
    sys.exit(main())
