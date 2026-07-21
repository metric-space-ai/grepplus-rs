#!/usr/bin/env python3
"""Forensics bench: run agents on REAL PR-derived edit tasks, keep every trace.

Separate from run_benchmark.py (which is the frozen v0.2 registered-gate harness
on synthetic mutation tasks) so that harness stays untouched. This one drives
the validated real tasks from swe_bench_adapter.py through the pi agent, grades
with the PR's own FAIL_TO_PASS / PASS_TO_PASS tests, and — the whole point —
saves the reasoning trace (agent.jsonl) for friction_catalog.py.

Lifecycle per (task, arm):
  clone@base_commit -> toolchain setup -> apply test_patch (adds the failing
  test) -> warm greppy index -> run pi agent (issue text as the task, arm's
  system prompt) -> grade: every FAIL_TO_PASS must pass and every PASS_TO_PASS
  must stay green. The gold source patch is NEVER shown to the agent.

Model is chosen via GREPPY_BENCH_PROVIDER/MODEL/THINKING/EXTENSION (Kimi K3 for
reasoning-visible forensics). Reuses run_benchmark's arm policies + pi argv.

Usage:
  forensics_bench.py --tasks tasks_real_python.json --arms greppy-edit \
      --greppy-bin .../greppy --out-dir runs/forensics-YYYYmmdd [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import time

import run_benchmark as RB  # frozen harness: reuse policies, pi argv, parsing
import swe_bench_adapter as A  # toolchain setup + clone + patch helpers


def clone_worktree(task: dict, workdir: str) -> str:
    cfg = A.REPOS[task["repo"]]
    url = f"https://github.com/{cfg['owner']}/{cfg['name']}"
    path = os.path.join(workdir, cfg["name"])
    A.git(["clone", "--quiet", url, path], cwd=workdir)
    A.git(["fetch", "--quiet", "origin", task["base_commit"]], cwd=path, check=False)
    A.git(["checkout", "--quiet", "--force", task["base_commit"]], cwd=path)
    return path


def grade(repo: str, task: dict) -> dict:
    """Run FAIL_TO_PASS + PASS_TO_PASS after the agent; correctness iff all
    FAIL_TO_PASS pass and all PASS_TO_PASS stay green."""
    results = A.run_tests(repo, task["toolchain"], task["changed_tests"])
    if results is None:
        return {"correctness": None, "reason": "grading harness error"}
    ftp = {t: results.get(t, False) for t in task["fail_to_pass"]}
    ptp = {t: results.get(t, False) for t in task["pass_to_pass"]}
    ok = all(ftp.values()) and all(ptp.values())
    return {
        "correctness": ok,
        "fail_to_pass_passed": sum(ftp.values()),
        "fail_to_pass_total": len(ftp),
        "pass_to_pass_passed": sum(ptp.values()),
        "pass_to_pass_total": len(ptp),
    }


def issue_prompt(task: dict) -> str:
    body = task["issue_body"].strip()
    body = body[:4000] + ("\n…" if len(body) > 4000 else "")
    return (
        f"Resolve this issue in the current repository.\n\n"
        f"Title: {task['issue_title']}\n\n{body}\n\n"
        f"Make the source changes that fix it. Do not edit test files; the fix "
        f"is verified by the project's own test suite."
    )


def run_one(task: dict, arm: str, greppy_bin: str, pi_bin: str, out_dir: pathlib.Path) -> dict:
    tid = f"{task['repo']}-{task['pr_number']}"
    raw_dir = out_dir / tid / arm
    raw_dir.mkdir(parents=True, exist_ok=True)
    # ignore_cleanup_errors: an agent subprocess can leave a file mid-write when
    # the workspace teardown races it ("Directory not empty"); a best-effort
    # cleanup must never kill the whole run on its last task.
    with tempfile.TemporaryDirectory(
        prefix=f"fbench-{tid}-", ignore_cleanup_errors=True
    ) as wd:
        try:
            repo = clone_worktree(task, wd)
            if not A.run_setup(repo, task["toolchain"]):
                return {"task": tid, "arm": arm, "valid": False, "reason": "setup failed"}
            if not A.apply_patch(repo, task["test_patch"]):
                return {"task": tid, "arm": arm, "valid": False, "reason": "test_patch apply failed"}
            store = os.path.join(wd, "greppy-store")
            picfg = os.path.join(wd, "pi-cfg")
            env = os.environ.copy()
            env["GREPPY_STORE_DIR"] = store
            env["PI_CODING_AGENT_DIR"] = picfg
            # Pin bare `greppy` on PATH to the binary under test. The system
            # prompt hands the agent an absolute path, but agents drift to bare
            # `greppy`; without this they silently hit a stale system/ctox shim
            # whose passthrough routes unknown subcommands (e.g. `edit text-cas`)
            # to grep -> contaminated friction measurement.
            binshim = os.path.join(wd, "binshim")
            os.makedirs(binshim, exist_ok=True)
            shim = os.path.join(binshim, "greppy")
            if os.path.lexists(shim):
                os.unlink(shim)
            os.symlink(os.path.abspath(greppy_bin), shim)
            env["PATH"] = binshim + os.pathsep + env.get("PATH", "")
            if arm in ("greppy", "greppy-edit"):
                subprocess.run([greppy_bin, "--root", ".", "index", "."], cwd=repo,
                               env=env, capture_output=True, timeout=1200)
            argv = [
                pi_bin, "-p", "--extension", str(RB.PROVIDER_EXTENSION),
                "--provider", RB.DEFAULT_PROVIDER, "--model", RB.DEFAULT_MODEL,
                "--mode", "json", "--no-session", "--thinking", RB.DEFAULT_THINKING,
                "--tools", RB.ARM_TOOLS[arm], "--no-context-files", "--no-skills",
                "--no-prompt-templates", "--no-extensions", "--approve",
                "--append-system-prompt",
                RB.system_prompt(arm, pathlib.Path(greppy_bin)),
                issue_prompt(task),
            ]
            t0 = time.monotonic()
            # Retry transient provider failures (rate-limit contention with a
            # running panel, upstream 5xx) so infrastructure noise does not
            # masquerade as a benchmark result.
            proc = None
            for attempt in range(3):
                try:
                    proc = subprocess.run(argv, cwd=repo, env=env, capture_output=True,
                                          timeout=task.get("timeout_seconds", 2400))
                except subprocess.TimeoutExpired:
                    return {"task": tid, "arm": arm, "valid": False, "reason": "agent timed out"}
                tail = proc.stderr[-400:].decode("utf-8", "replace")
                if proc.returncode == 0 or b'"type"' in proc.stdout[:200]:
                    break
                (raw_dir / f"pi.stderr.{attempt}").write_bytes(proc.stderr)
                if attempt < 2:
                    time.sleep(20 * (attempt + 1))
            wall = round(time.monotonic() - t0, 1)
            if proc is None or not proc.stdout.strip():
                return {"task": tid, "arm": arm, "valid": False,
                        "reason": f"pi produced no trace: {tail[:200]}"}
            (raw_dir / "agent.jsonl").write_bytes(proc.stdout)
            metrics = RB.parse_pi_jsonl(proc.stdout)
            g = grade(repo, task)
            return {"task": tid, "arm": arm, "valid": True, "wall_seconds": wall,
                    "turns": metrics.get("turns"), "tool_calls": metrics.get("tool_calls"),
                    "reported_error": metrics.get("reported_error"), **g}
        except Exception as e:
            return {"task": tid, "arm": arm, "valid": False, "reason": str(e)[:120]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--arms", default="greppy-edit", help="comma list")
    ap.add_argument("--greppy-bin", required=True)
    ap.add_argument("--pi-bin", default="pi")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]
    arms = args.arms.split(",")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = set()
    if args.resume and results_path.exists():
        done = {(json.loads(l)["task"], json.loads(l)["arm"]) for l in open(results_path) if l.strip()}

    with open(results_path, "a") as rf:
        for task in tasks:
            for arm in arms:
                tid = f"{task['repo']}-{task['pr_number']}"
                if (tid, arm) in done:
                    continue
                row = run_one(task, arm, args.greppy_bin, args.pi_bin, out_dir)
                rf.write(json.dumps(row) + "\n")
                rf.flush()
                c = row.get("correctness")
                mark = "OK" if c else ("x" if c is False else "-")
                print(f"  {tid:22s} {arm:12s} {mark} turns={row.get('turns')} valid={row.get('valid')}",
                      file=sys.stderr, flush=True)
    print(f"-> {results_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
