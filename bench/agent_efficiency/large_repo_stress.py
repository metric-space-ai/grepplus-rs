#!/usr/bin/env python3
"""Reproducible large-repo stress gate for greppy indexing.

This is a black-box harness for the R3/R8 production-hardening contract. It
generates a deterministic git Rust repo, indexes it with an isolated
GREPPY_STORE_DIR, mutates exactly one file, re-indexes, and records:

* initial index wall time and peak whole-process-tree RSS,
* incremental re-index wall time, peak RSS and indexed-file count,
* graph.db size, node/edge counts and integrity_check,
* CLI symbol lookup proof for an unchanged and a newly-added symbol.

The one-time index build cost is measured here because this script validates
index hardening, not the agent-efficiency query benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_BIN = REPO / "target" / "release" / "greppy"


@dataclass
class CommandResult:
    argv: list[str]
    exit_code: int
    elapsed_s: float
    peak_rss_kb: int
    output: str
    timed_out: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "argv": redact_argv(self.argv),
            "exit_code": self.exit_code,
            "elapsed_s": round(self.elapsed_s, 3),
            "peak_rss_kb": self.peak_rss_kb,
            "peak_rss_mib": round(self.peak_rss_kb / 1024, 3),
            "timed_out": self.timed_out,
            "indexed_files": indexed_file_count(self.output),
            "panic_detected": has_panic(self.output),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--greppy-bin", type=pathlib.Path, default=DEFAULT_BIN)
    ap.add_argument("--files", type=int, default=300)
    ap.add_argument("--functions-per-file", type=int, default=5)
    ap.add_argument("--fanout", type=int, default=2)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--incremental-timeout-s", type=float, default=120.0)
    ap.add_argument("--sample-interval-s", type=float, default=0.05)
    ap.add_argument("--work-dir", type=pathlib.Path)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--max-initial-seconds", type=float)
    ap.add_argument("--max-incremental-seconds", type=float)
    ap.add_argument("--max-peak-rss-mib", type=float)
    ap.add_argument("--max-db-mib", type=float)
    ap.add_argument("--max-incremental-indexed", type=int, default=1)
    args = ap.parse_args()

    if args.files < 2:
        raise SystemExit("--files must be >= 2")
    if args.functions_per_file < 1:
        raise SystemExit("--functions-per-file must be >= 1")
    if args.fanout < 0:
        raise SystemExit("--fanout must be >= 0")
    if not args.greppy_bin.exists():
        raise SystemExit(f"greppy binary missing: {args.greppy_bin}")

    base_temp = None
    if args.work_dir:
        work = args.work_dir.resolve()
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
    else:
        base_temp = tempfile.TemporaryDirectory(prefix="greppy-large-stress-")
        work = pathlib.Path(base_temp.name)

    repo = work / "repo"
    store = work / "store"
    report: dict[str, Any] = {
        "status": "running",
        "parameters": {
            "files": args.files,
            "functions_per_file": args.functions_per_file,
            "fanout": args.fanout,
            "max_incremental_indexed": args.max_incremental_indexed,
            "max_initial_seconds": args.max_initial_seconds,
            "max_incremental_seconds": args.max_incremental_seconds,
            "max_peak_rss_mib": args.max_peak_rss_mib,
            "max_db_mib": args.max_db_mib,
        },
        "work_dir": str(work) if args.keep or args.work_dir else None,
        "checks": [],
    }

    failures: list[str] = []

    try:
        generate_rust_repo(repo, args.files, args.functions_per_file, args.fanout)
        git_init(repo)
        generated_rs = len(list((repo / "src").glob("*.rs")))
        check(report, failures, generated_rs >= args.files + 1, f"generated Rust files >= requested files ({generated_rs})")
        check(report, failures, (repo / ".git").is_dir(), "synthetic corpus is a git repository")

        env = os.environ.copy()
        env["GREPPY_STORE_DIR"] = str(store)

        initial = run_with_rss(
            [str(args.greppy_bin), "index", str(repo)],
            cwd=repo,
            env=env,
            timeout_s=args.timeout_s,
            sample_interval_s=args.sample_interval_s,
            log_path=work / "initial-index.log",
        )
        report["initial_index"] = initial.as_json()
        check(report, failures, initial.exit_code == 0, f"initial index exits 0 ({initial.exit_code})")
        check(report, failures, not initial.timed_out, "initial index does not time out")
        check(report, failures, not has_panic(initial.output), "initial index output has no panic")

        db = find_graph_db(store)
        check(report, failures, db is not None, "graph.db exists after initial index")
        if db:
            report["db_after_initial"] = db_report(db)
            check(report, failures, report["db_after_initial"]["integrity_check"] == "ok", "initial graph.db integrity_check is ok")
            check(report, failures, report["db_after_initial"]["nodes"] > 0, "initial graph has nodes")
            check(report, failures, report["db_after_initial"]["edges"] > 0, "initial graph has edges")

        unchanged_symbol = f"large_repo_anchor_{args.files - 1:04d}_0"
        initial_lookup = run_plain(
            [str(args.greppy_bin), "--root", str(repo), "search-symbols", unchanged_symbol],
            cwd=repo,
            env=env,
        )
        report["initial_lookup"] = command_json(initial_lookup)
        check(report, failures, unchanged_symbol in initial_lookup.stdout, f"initial lookup finds unchanged symbol {unchanged_symbol}")

        changed_file_index = args.files // 2
        changed_file = repo / "src" / f"mod{changed_file_index:04d}.rs"
        new_symbol = f"large_repo_incremental_marker_{changed_file_index:04d}"
        with changed_file.open("a", encoding="utf-8") as f:
            f.write(f"\npub fn {new_symbol}() -> u64 {{ {changed_file_index} }}\n")

        incremental = run_with_rss(
            [str(args.greppy_bin), "index", str(repo)],
            cwd=repo,
            env=env,
            timeout_s=args.incremental_timeout_s,
            sample_interval_s=args.sample_interval_s,
            log_path=work / "incremental-index.log",
        )
        report["incremental_index"] = incremental.as_json()
        check(report, failures, incremental.exit_code == 0, f"incremental index exits 0 ({incremental.exit_code})")
        check(report, failures, not incremental.timed_out, "incremental index does not time out")
        check(report, failures, not has_panic(incremental.output), "incremental index output has no panic")

        indexed = indexed_file_count(incremental.output)
        check(report, failures, indexed is not None, "incremental index reports indexed-file count")
        if indexed is not None:
            check(
                report,
                failures,
                indexed <= args.max_incremental_indexed,
                f"incremental index touches <= {args.max_incremental_indexed} file(s) ({indexed})",
            )

        db = find_graph_db(store)
        if db:
            report["db_after_incremental"] = db_report(db)
            check(report, failures, report["db_after_incremental"]["integrity_check"] == "ok", "incremental graph.db integrity_check is ok")
            check(report, failures, report["db_after_incremental"]["nodes"] > 0, "incremental graph has nodes")
            check(report, failures, report["db_after_incremental"]["edges"] > 0, "incremental graph has edges")

        unchanged_lookup = run_plain(
            [str(args.greppy_bin), "--root", str(repo), "search-symbols", unchanged_symbol],
            cwd=repo,
            env=env,
        )
        new_lookup = run_plain(
            [str(args.greppy_bin), "--root", str(repo), "search-symbols", new_symbol],
            cwd=repo,
            env=env,
        )
        report["post_incremental_unchanged_lookup"] = command_json(unchanged_lookup)
        report["post_incremental_new_lookup"] = command_json(new_lookup)
        check(report, failures, unchanged_symbol in unchanged_lookup.stdout, "unchanged symbol survives incremental publish")
        check(report, failures, new_symbol in new_lookup.stdout, "new symbol appears after one-file incremental publish")

        max_peak_mib = max(
            initial.peak_rss_kb / 1024,
            incremental.peak_rss_kb / 1024,
        )
        report["max_peak_rss_mib"] = round(max_peak_mib, 3)
        if args.max_peak_rss_mib is not None:
            check(
                report,
                failures,
                max_peak_mib <= args.max_peak_rss_mib,
                f"peak RSS <= {args.max_peak_rss_mib} MiB ({max_peak_mib:.3f})",
            )
        if args.max_initial_seconds is not None:
            check(
                report,
                failures,
                initial.elapsed_s <= args.max_initial_seconds,
                f"initial index <= {args.max_initial_seconds}s ({initial.elapsed_s:.3f})",
            )
        if args.max_incremental_seconds is not None:
            check(
                report,
                failures,
                incremental.elapsed_s <= args.max_incremental_seconds,
                f"incremental index <= {args.max_incremental_seconds}s ({incremental.elapsed_s:.3f})",
            )
        if args.max_db_mib is not None and db:
            db_mib = db.stat().st_size / (1024 * 1024)
            check(
                report,
                failures,
                db_mib <= args.max_db_mib,
                f"graph.db <= {args.max_db_mib} MiB ({db_mib:.3f})",
            )

        report["status"] = "fail" if failures else "pass"
        report["failures"] = failures
        emit(report, args.json)
        return 1 if failures else 0
    finally:
        if base_temp is not None and not args.keep:
            base_temp.cleanup()


def generate_rust_repo(root: pathlib.Path, files: int, functions_per_file: int, fanout: int) -> None:
    src = root / "src"
    src.mkdir(parents=True)
    (root / "Cargo.toml").write_text(
        "[package]\nname = \"greppy_large_stress\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[lib]\npath = \"src/lib.rs\"\n",
        encoding="utf-8",
    )
    lib_lines = [f"pub mod mod{i:04d};" for i in range(files)]
    lib_lines.append("")
    lib_lines.append("pub fn run_large_repo(seed: u64) -> u64 {")
    terms = [f"mod{i:04d}::large_repo_anchor_{i:04d}_0(seed)" for i in range(max(0, files - fanout), files)]
    lib_lines.append("    " + " + ".join(terms))
    lib_lines.append("}")
    (src / "lib.rs").write_text("\n".join(lib_lines) + "\n", encoding="utf-8")

    for i in range(files):
        lines = [
            f"pub struct LargeRepoWidget{i:04d} {{",
            "    pub value: u64,",
            "}",
            "",
            f"impl LargeRepoWidget{i:04d} {{",
            "    pub fn new(value: u64) -> Self {",
            "        Self { value }",
            "    }",
            "    pub fn score(&self) -> u64 {",
            f"        self.value + {i}",
            "    }",
            "}",
            "",
        ]
        for j in range(functions_per_file):
            symbol = f"large_repo_anchor_{i:04d}_{j}"
            terms = [f"LargeRepoWidget{i:04d}::new(seed + {i + j}).score()"]
            for back in range(1, fanout + 1):
                prev = i - back
                if prev >= 0:
                    terms.append(f"crate::mod{prev:04d}::large_repo_anchor_{prev:04d}_0(seed)")
            lines.extend(
                [
                    f"pub fn {symbol}(seed: u64) -> u64 {{",
                    "    " + " + ".join(terms),
                    "}",
                    "",
                ]
            )
        (src / f"mod{i:04d}.rs").write_text("\n".join(lines), encoding="utf-8")


def git_init(root: pathlib.Path) -> None:
    run_plain(["git", "init", "-q", "."], cwd=root)
    run_plain(["git", "add", "-A"], cwd=root)
    run_plain(
        [
            "git",
            "-c",
            "user.email=large-stress@greppy.test",
            "-c",
            "user.name=large-stress",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "large repo stress corpus",
        ],
        cwd=root,
    )


def run_with_rss(
    argv: list[str],
    cwd: pathlib.Path,
    env: dict[str, str],
    timeout_s: float,
    sample_interval_s: float,
    log_path: pathlib.Path,
) -> CommandResult:
    start = time.monotonic()
    peak = 0
    timed_out = False
    with log_path.open("w+b") as log:
        proc = subprocess.Popen(argv, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            peak = max(peak, rss_kb_tree(proc.pid))
            if time.monotonic() - start > timeout_s:
                timed_out = True
                proc.kill()
                break
            time.sleep(sample_interval_s)
        proc.wait()
        elapsed = time.monotonic() - start
        peak = max(peak, rss_kb_tree(proc.pid))
        log.flush()
        log.seek(0)
        output = log.read().decode("utf-8", "replace")
    return CommandResult(
        argv=argv,
        exit_code=proc.returncode if proc.returncode is not None else -1,
        elapsed_s=elapsed,
        peak_rss_kb=peak,
        output=output,
        timed_out=timed_out,
    )


def rss_kb_tree(root_pid: int) -> int:
    try:
        ps = subprocess.run(
            ["ps", "-A", "-o", "pid=", "-o", "ppid=", "-o", "rss="],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return 0
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in ps.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, rss_kb = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        rss[pid] = rss_kb
    stack = [root_pid]
    total = 0
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        total += rss.get(pid, 0)
        stack.extend(children.get(pid, []))
    return total


def run_plain(argv: list[str], cwd: pathlib.Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def find_graph_db(store: pathlib.Path) -> pathlib.Path | None:
    matches = sorted(store.rglob("graph.db"))
    return matches[0] if matches else None


def db_report(db: pathlib.Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db))
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        nodes = conn.execute("SELECT count(*) FROM nodes").fetchone()[0]
        edges = conn.execute("SELECT count(*) FROM edges").fetchone()[0]
        files = conn.execute("SELECT count(*) FROM file_state").fetchone()[0]
    finally:
        conn.close()
    return {
        "path": str(db),
        "size_bytes": db.stat().st_size,
        "size_mib": round(db.stat().st_size / (1024 * 1024), 3),
        "integrity_check": integrity,
        "nodes": nodes,
        "edges": edges,
        "file_state_rows": files,
    }


def check(report: dict[str, Any], failures: list[str], ok: bool, message: str) -> None:
    report["checks"].append({"ok": bool(ok), "message": message})
    if not ok:
        failures.append(message)


def indexed_file_count(output: str) -> int | None:
    match = re.search(r"\bindexed\s+(\d+)\s+files?\b", output)
    return int(match.group(1)) if match else None


def has_panic(output: str) -> bool:
    return bool(re.search(r"panic|thread .* panicked|RUST_BACKTRACE", output, re.IGNORECASE))


def command_json(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "argv": redact_argv([str(p) for p in result.args]),
        "exit_code": result.returncode,
        "stdout_preview": result.stdout[:4000],
    }


def redact_argv(argv: list[str]) -> list[str]:
    redacted = []
    for part in argv:
        if part.startswith("sk-"):
            redacted.append("<redacted>")
        else:
            redacted.append(part)
    return redacted


def emit(report: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"large_repo_stress: {report['status']}")
    for item in report["checks"]:
        print(("PASS " if item["ok"] else "FAIL ") + item["message"])
    initial = report.get("initial_index", {})
    incremental = report.get("incremental_index", {})
    print(
        "initial: "
        f"{initial.get('elapsed_s')}s, "
        f"{initial.get('peak_rss_mib')} MiB RSS, "
        f"indexed={initial.get('indexed_files')}"
    )
    print(
        "incremental: "
        f"{incremental.get('elapsed_s')}s, "
        f"{incremental.get('peak_rss_mib')} MiB RSS, "
        f"indexed={incremental.get('indexed_files')}"
    )
    db = report.get("db_after_incremental") or report.get("db_after_initial") or {}
    if db:
        print(
            "db: "
            f"{db.get('size_mib')} MiB, "
            f"nodes={db.get('nodes')}, "
            f"edges={db.get('edges')}, "
            f"integrity={db.get('integrity_check')}"
        )


if __name__ == "__main__":
    sys.exit(main())
