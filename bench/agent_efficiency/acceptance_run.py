#!/usr/bin/env python3
"""Run the full greppy agent-efficiency acceptance pipeline.

This script exists to prevent "benchmark by anecdote":

1. build greppy,
2. verify the synthetic task regression/router classes,
3. verify/index the synthetic corpus,
4. run pi/MiniMax-M3 with raw trajectory capture for the product comparison,
5. attach accepted mechanical quality grades for the synthetic tasks,
6. run mandatory forensics for every candidate.

The MiniMax API key is read only from MINIMAX_API_KEY or the user's launchd
environment and is never written to commands, logs, reports, or result files.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import subprocess
import sys
import time


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
# Product default: trio with explorer as the FIXED product-gate baseline
# (BENCHMARK_CONTRACT §Baselines). Any other --baseline is research-only and
# is stamped DIAGNOSTIC on every artifact — it can never produce product-gate
# status (Codex-Review P0-2). Prefer parallel_acceptance_run.py for product
# runs (R8: parallel, never serial).
DEFAULT_AGENTS = "grep,greppy,explorer"
GATE_BASELINE = "explorer"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", default=DEFAULT_AGENTS)
    ap.add_argument("--baseline", default=GATE_BASELINE)
    ap.add_argument(
        "--candidates",
        help="comma-separated candidates for forensics; default = agents minus baseline",
    )
    ap.add_argument("--repo", help="optional corpus repo filter passed to run_bench.py")
    ap.add_argument("task_ids", nargs="*", help="optional task IDs passed to run_bench.py")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--output-dir", type=pathlib.Path)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--allow-unaccepted",
        action="store_true",
        help="exit 0 even when one or more forensics gates return not accepted",
    )
    args = ap.parse_args()

    agents = [a.strip() for a in args.agents.split(",") if a.strip()]
    if not agents:
        raise SystemExit("--agents must name at least one agent")
    if args.baseline not in agents:
        raise SystemExit(f"--baseline {args.baseline!r} must be included in --agents")
    diagnostic_run = args.baseline != GATE_BASELINE
    if diagnostic_run:
        print(
            f"== DIAGNOSTIC RUN: baseline {args.baseline!r} != gate baseline "
            f"{GATE_BASELINE!r} — artifacts are stamped DIAGNOSTIC and this run "
            "produces NO product-gate status (Codex-Review P0-2).",
            file=sys.stderr,
        )
    candidates = (
        [c.strip() for c in args.candidates.split(",") if c.strip()]
        if args.candidates
        else [a for a in agents if a != args.baseline]
    )
    unknown_candidates = sorted(set(candidates) - set(agents))
    if unknown_candidates:
        raise SystemExit(
            "all --candidates must be included in --agents: "
            + ", ".join(unknown_candidates)
        )

    ensure_minimax_api_key()
    if not args.dry_run and not args.skip_bench and not os.environ.get("MINIMAX_API_KEY"):
        raise SystemExit(
            "MINIMAX_API_KEY is missing. Export it or set it with launchctl; do not pass it on argv."
        )

    run_dir = args.output_dir or (HERE / "acceptance_runs" / args.run_id)
    raw_dir = run_dir / "raw"
    results = run_dir / "results.json"
    graded_results = run_dir / "results.mechanical.json"
    aggregate = run_dir / "aggregate.txt"
    summary = run_dir / "SUMMARY.md"
    logs_dir = run_dir / "logs"
    if not args.dry_run:
        logs_dir.mkdir(parents=True, exist_ok=True)

    steps: list[tuple[str, int]] = []
    forensics_status: dict[str, int] = {}

    def step(name: str, cmd: list[str], allowed: set[int] | None = None, tee: pathlib.Path | None = None) -> int:
        allowed = allowed or {0}
        rc = run_command(
            name=name,
            cmd=cmd,
            cwd=REPO,
            log_path=logs_dir / f"{safe_name(name)}.log",
            dry_run=args.dry_run,
            allowed=allowed,
            tee_path=tee,
        )
        steps.append((name, rc))
        return rc

    if not args.skip_build:
        step("build-greppy", ["cargo", "build", "--release", "--bin", "greppy"])

    if not args.skip_verify:
        step(
            "verify-task-classes",
            [sys.executable, str(HERE / "verify_task_classes.py")],
        )
        step(
            "verify-tasks-index",
            [sys.executable, str(HERE / "verify_tasks.py"), "--index"],
        )

    bench_args = [
        sys.executable,
        str(HERE / "run_bench.py"),
        "--results",
        str(results),
        "--agents",
        ",".join(agents),
        "--save-raw",
        "--raw-dir",
        str(raw_dir),
    ]
    if args.repo:
        bench_args.extend(["--repo", args.repo])
    bench_args.extend(args.task_ids)
    if not args.skip_bench:
        step("run-bench", bench_args)

    if not args.skip_bench or results.exists() or args.dry_run:
        if "greppy" in agents:
            step(
                "aggregate-report",
                [sys.executable, str(HERE / "run_bench.py"), "--results", str(results), "--report"],
                tee=aggregate,
            )
        step(
            "mechanical-grade",
            [
                sys.executable,
                str(HERE / "grade_answers.py"),
                "--mode",
                "mechanical",
                "--accept-mechanical",
                "--results",
                str(results),
                "--output",
                str(graded_results),
                "--agents",
                ",".join(agents),
            ],
        )
        for candidate in candidates:
            prefix = "DIAGNOSTIC_" if diagnostic_run else ""
            report_path = run_dir / f"{prefix}FORENSICS_{args.baseline}_VS_{candidate}.md"
            rc = step(
                f"forensics-{args.baseline}-vs-{candidate}",
                [
                    sys.executable,
                    str(HERE / "forensics.py"),
                    "--results",
                    str(graded_results),
                    "--baseline",
                    args.baseline,
                    "--candidate",
                    candidate,
                    "--output",
                    str(report_path),
                    "--enforce",
                ],
                allowed={0, 2},
            )
            forensics_status[candidate] = rc

    if not args.dry_run:
        write_summary(
            summary=summary,
            run_dir=run_dir,
            agents=agents,
            baseline=args.baseline,
            candidates=candidates,
            key_set=bool(os.environ.get("MINIMAX_API_KEY")),
            steps=steps,
            forensics_status=forensics_status,
            results=results,
            graded_results=graded_results,
            raw_dir=raw_dir,
            aggregate=aggregate,
        )

    unaccepted = [c for c, rc in forensics_status.items() if rc != 0]
    if unaccepted and not args.allow_unaccepted:
        print(
            "not accepted: " + ", ".join(unaccepted) + f" (see {summary})",
            file=sys.stderr,
        )
        return 2
    return 0


def run_command(
    name: str,
    cmd: list[str],
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    dry_run: bool,
    allowed: set[int],
    tee_path: pathlib.Path | None = None,
) -> int:
    rendered = shlex.join(cmd)
    print(f"== {name}: {rendered}", file=sys.stderr)
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if tee_path:
        tee_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n\n")
        tee_file = tee_path.open("w", encoding="utf-8") if tee_path else None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
                if tee_file:
                    tee_file.write(line)
            rc = proc.wait()
        finally:
            if tee_file:
                tee_file.close()
        log.write(f"\nexit={rc}\n")

    if rc not in allowed:
        raise SystemExit(f"{name} exited {rc}, expected one of {sorted(allowed)}")
    return rc


def ensure_minimax_api_key() -> None:
    """Populate this process env from launchd when shell env is not inherited."""
    if os.environ.get("MINIMAX_API_KEY"):
        return
    try:
        proc = subprocess.run(
            ["launchctl", "getenv", "MINIMAX_API_KEY"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except (OSError, ValueError):
        return
    value = proc.stdout.strip()
    if value:
        os.environ["MINIMAX_API_KEY"] = value


def write_summary(
    summary: pathlib.Path,
    run_dir: pathlib.Path,
    agents: list[str],
    baseline: str,
    candidates: list[str],
    key_set: bool,
    steps: list[tuple[str, int]],
    forensics_status: dict[str, int],
    results: pathlib.Path,
    graded_results: pathlib.Path,
    raw_dir: pathlib.Path,
    aggregate: pathlib.Path,
) -> None:
    lines = [
        f"# greppy Acceptance Run - {run_dir.name}",
        "",
        "## Configuration",
        "",
        f"- Agents: `{','.join(agents)}`",
        f"- Baseline: `{baseline}`",
        f"- Candidates: `{','.join(candidates)}`",
        f"- MINIMAX_API_KEY set: `{str(key_set).lower()}`",
        "",
        "## Artifacts",
        "",
        f"- Results: `{results}`",
        f"- Mechanical results: `{graded_results}`",
        f"- Raw trajectories: `{raw_dir}`",
        f"- Aggregate report: `{aggregate}`",
        "",
        "## Steps",
        "",
        "| Step | Exit |",
        "|---|---:|",
    ]
    for name, rc in steps:
        lines.append(f"| `{name}` | {rc} |")
    lines.extend(["", "## Forensics Gates", "", "| Candidate | Status | Exit |", "|---|---|---:|"])
    for candidate in candidates:
        rc = forensics_status.get(candidate)
        if rc is None:
            status = "not run"
            rc_text = "n/a"
        elif rc == 0:
            status = "accepted"
            rc_text = "0"
        else:
            status = "not accepted"
            rc_text = str(rc)
        lines.append(f"| `{candidate}` | {status} | {rc_text} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A candidate is accepted only when its forensics gate exits 0. Exit 2 means the run produced useful optimization evidence, but it does not prove the greppy product claim.",
            "",
        ]
    )
    summary.write_text("\n".join(lines), encoding="utf-8")


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


if __name__ == "__main__":
    raise SystemExit(main())
