#!/usr/bin/env python3
"""Aggregate N graded benchmark runs into the pre-registered report.

Per BENCHMARK_CONTRACT.md pre-registration (2026-07-06): published numbers
are the MEDIAN over N complete runs of the identical task set, reported with
the min-max span per headline metric. Single-run numbers are never published.

Usage:
    python3 aggregate_runs.py RUN_DIR [RUN_DIR ...] [--output REPORT.md]

Each RUN_DIR must contain results_graded.json (grade_answers.py output).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

METRIC_FIELDS = ["total", "variable_input", "output", "tool_calls", "wall_s", "turns"]


def ok(res: dict | None) -> bool:
    return (
        isinstance(res, dict)
        and "wall_s" in res
        and not res.get("error")
        and (res.get("answer") or "").strip() != ""
    )


def run_metrics(rows: list[dict]) -> dict:
    """Per-run medians: per-task grep/greppy factor per field + quality tally."""
    out: dict = {"n_rows": len(rows)}
    complete = [r for r in rows if ok(r.get("grep")) and ok(r.get("greppy"))]
    out["n_complete"] = len(complete)
    for field in METRIC_FIELDS:
        factors = []
        for r in complete:
            g, p = r["grep"].get(field), r["greppy"].get(field)
            if isinstance(g, (int, float)) and isinstance(p, (int, float)) and p > 0:
                factors.append(g / p)
        out[f"factor_{field}"] = statistics.median(factors) if factors else None
    for arm in ("grep", "greppy"):
        tally: dict[str, int] = {}
        for r in complete:
            q = ((r[arm].get("quality") or {}).get("verdict")) or "ungraded"
            tally[q] = tally.get(q, 0) + 1
        out[f"quality_{arm}"] = tally

    # Win-rate (the headline the owner wants: is grep+ a no-brainer?).
    # Per task, greppy is "not worse" when it is at least as cheap in
    # total tokens (10% tolerance for LLM sampling noise) AND not lower
    # quality. Reported as the share of complete tasks, plus the two
    # loss buckets so regressions are never hidden behind an average.
    rank = {"pass": 2, "partial": 1, "fail": 0, "ungraded": 0, None: 0}
    not_worse = q_loss = cost_loss = 0
    for r in complete:
        g, p = r["grep"], r["greppy"]
        gq = rank.get((g.get("quality") or {}).get("verdict"), 0)
        pq = rank.get((p.get("quality") or {}).get("verdict"), 0)
        gt, pt = g.get("total"), p.get("total")
        cheaper = (
            isinstance(gt, (int, float))
            and isinstance(pt, (int, float))
            and pt <= gt * 1.10
        )
        if pq < gq:
            q_loss += 1
        elif pq > gq or cheaper:
            not_worse += 1
        else:
            cost_loss += 1
    n = len(complete) or 1
    out["win_rate"] = round(100 * not_worse / n, 1)
    out["win_not_worse"] = not_worse
    out["loss_quality"] = q_loss
    out["loss_cost"] = cost_loss
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", type=pathlib.Path)
    ap.add_argument("--output", type=pathlib.Path, default=None)
    args = ap.parse_args()

    per_run = []
    for run in args.runs:
        f = run / "results_graded.json"
        if not f.exists():
            sys.exit(f"[aggregate] missing {f}")
        rows = json.loads(f.read_text())
        m = run_metrics(rows)
        m["run"] = run.name
        per_run.append(m)

    # cross-run: median + min-max span per headline metric
    lines = [
        "# Benchmark aggregate — median over runs (pre-registered protocol)",
        "",
        f"Runs: {', '.join(m['run'] for m in per_run)}",
        f"Rows per run (complete/total): "
        + ", ".join(f"{m['n_complete']}/{m['n_rows']}" for m in per_run),
        "",
        "| metric | " + " | ".join(m["run"] for m in per_run)
        + " | MEDIAN | span |",
        "|---|" + "---|" * (len(per_run) + 2),
    ]
    for field in METRIC_FIELDS:
        vals = [m[f"factor_{field}"] for m in per_run]
        nums = [v for v in vals if v is not None]
        if not nums:
            continue
        med, lo, hi = statistics.median(nums), min(nums), max(nums)
        cells = " | ".join(f"{v:.2f}x" if v is not None else "—" for v in vals)
        lines.append(
            f"| grep/greppy {field} | {cells} | **{med:.2f}x** | {lo:.2f}–{hi:.2f}x |"
        )
    lines.append("")
    lines.append("## Win-rate — is grep+ a no-brainer? (per-task, not worse)")
    wins = [m["win_rate"] for m in per_run]
    lines.append(
        f"greppy NOT WORSE on **{statistics.median(wins):.1f}%** of tasks "
        f"(median over runs; span {min(wins):.1f}–{max(wins):.1f}%). "
        "Not-worse = at least as cheap in total tokens (10% tolerance) AND "
        "not lower quality."
    )
    for m in per_run:
        lines.append(
            f"- {m['run']}: win {m['win_rate']}% "
            f"(not-worse {m['win_not_worse']}, "
            f"quality-loss {m['loss_quality']}, cost-loss {m['loss_cost']} "
            f"of {m['n_complete']})"
        )
    lines.append("")
    lines.append("## Answer quality (mechanical grade tallies per arm)")
    for m in per_run:
        lines.append(
            f"- {m['run']}: grep={json.dumps(m['quality_grep'], sort_keys=True)} "
            f"greppy={json.dumps(m['quality_greppy'], sort_keys=True)}"
        )
    report = "\n".join(lines) + "\n"
    print(report)
    if args.output:
        args.output.write_text(report)
        print(f"[aggregate] wrote {args.output}", file=sys.stderr)
    # machine-readable sidecar next to --output (or cwd)
    sidecar = (args.output.with_suffix(".json") if args.output
               else pathlib.Path("aggregate_runs.json"))
    sidecar.write_text(json.dumps(per_run, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
