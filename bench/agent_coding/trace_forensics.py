#!/usr/bin/env python3
"""Forensics over pi agent traces: what agents actually do with greppy.

Reads the ``agent.jsonl`` a benchmark arm records, reconstructs the
turn-by-turn timeline (assistant text, every tool call verbatim, what the
tool returned, per-message token usage), classifies every greppy
invocation, and flags flaw patterns.

MiniMax does not expose its reasoning content through the API; reasoning
is billed inside ``usage.output``. The per-turn output-token count against
the visible text length is therefore reported as the hidden-deliberation
proxy: a turn with 2 visible characters and 300 output tokens burned its
budget thinking, and whatever it was thinking about is the previous tool
result — which is why the timeline keeps result excerpts.

Usage:
  python3 trace_forensics.py TRACE.jsonl               # one arm, full timeline
  python3 trace_forensics.py --summary RUN_DIR         # raw_traces run dir,
                                                       # per-arm aggregates
Output is plain text, sized for reading, not for grading. Grading stays in
run_benchmark.py; this tool exists to explain gradings.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Heuristics for classifying bash commands the agent runs.
SOURCE_OPEN_RE = re.compile(
    r"\b(cat|head|tail|less|more|sed\s+-n|awk\s+'?NR)\b.*?"
    r"([\w./-]+\.(?:rs|py|ts|js|tsx|jsx|go|java|c|h|cpp|hpp|rb|php|swift|kt|scala|zig|vue|svelte|toml|json|yaml|yml))",
    re.S,
)
GREPPY_RE = re.compile(r"(?:^|[;&|]\s*|\$\()\s*(?:[\w./]*/)?greppy\b")
EDIT_VERB_RE = re.compile(r"\bgreppy\s+edit\s+([a-z-]+)")
READ_RE = re.compile(r"\bgreppy\s+read\b")
RG_STYLE_RE = re.compile(r"--smart-case|\s-S\s|\s-t[a-z]+\b|--glob|\s-g\s|\brg\s")
# Exit-code contract of the edit surface (docs/contracts/EDIT_CONTRACT.md).
EDIT_EXITS = {
    0: "applied/already-satisfied",
    10: "not-found",
    11: "ambiguous",
    12: "stale-hash",
    13: "syntax/postcondition",
    14: "validator-failed",
    15: "concurrent-modification",
    16: "publish-failed",
    17: "unsafe-path",
    20: "invalid-spec",
}
FALLBACK_RE = re.compile(r"\b(sed\s+-i|patch\b|ed\s|tee\s|>\s*[\w./-]+\.(rs|py|ts|js)|applypatch|git\s+apply)")


def load_events(path: Path) -> list[dict[str, Any]]:
    events = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def result_text(result: Any) -> str:
    if isinstance(result, dict):
        parts = result.get("content") or []
        texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(texts)
    return str(result or "")


def first_command_word(cmd: str) -> str:
    try:
        words = shlex.split(cmd)
    except ValueError:
        words = cmd.split()
    return words[0] if words else ""


class ToolCallRecord:
    def __init__(self, call_id: str, name: str, arguments: dict[str, Any]):
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
        self.result = ""
        self.is_error = False

    @property
    def command(self) -> str:
        return str(self.arguments.get("command", "")) if isinstance(self.arguments, dict) else ""


def build_timeline(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[ToolCallRecord]]:
    """Return (turns, calls). Each turn: text, calls, usage."""
    turns: list[dict[str, Any]] = []
    calls_by_id: dict[str, ToolCallRecord] = {}
    order: list[ToolCallRecord] = []
    for e in events:
        etype = e.get("type")
        if etype == "message_end" and e.get("message", {}).get("role") == "assistant":
            m = e["message"]
            text_parts, call_records = [], []
            for block in m.get("content", []):
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype in ("thinking", "reasoning"):
                    # If a provider ever starts returning reasoning, keep it.
                    text_parts.append(f"[reasoning] {block.get('text') or block.get('thinking', '')}")
                elif btype == "toolCall":
                    rec = ToolCallRecord(block.get("id", ""), block.get("name", ""), block.get("arguments", {}))
                    calls_by_id[rec.call_id] = rec
                    call_records.append(rec)
                    order.append(rec)
            usage = m.get("usage", {})
            turns.append(
                {
                    "text": "\n".join(text_parts),
                    "calls": call_records,
                    "output_tokens": usage.get("output", 0),
                    "input_tokens": usage.get("input", 0),
                    "cache_read": usage.get("cacheRead", 0),
                }
            )
        elif etype == "tool_execution_end":
            rec = calls_by_id.get(e.get("toolCallId", ""))
            if rec is not None:
                rec.result = result_text(e.get("result"))
                rec.is_error = bool(e.get("isError"))
    return turns, order


def classify_calls(order: list[ToolCallRecord]) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "tool_calls": len(order),
        "greppy_calls": 0,
        "greppy_subcommands": Counter(),
        "edit_verbs": Counter(),
        "edit_failures": Counter(),  # exit label -> count
        "reads": 0,
        "read_handles": 0,
        "rg_style_calls": 0,
        "source_opens": 0,
        "post_edit_reopens": 0,
        "fallback_edits": 0,
        "repeated_commands": 0,
        "errored_calls": 0,
        "usage_errors": 0,
    }
    seen_commands: Counter = Counter()
    edited_files: set[str] = set()
    flaws: list[str] = []

    for idx, rec in enumerate(order):
        cmd = rec.command
        if rec.is_error:
            stats["errored_calls"] += 1
        if not cmd:
            continue
        seen_commands[cmd] += 1
        if seen_commands[cmd] == 2:
            stats["repeated_commands"] += 1
            flaws.append(f"call#{idx}: identical command repeated: {cmd[:120]}")

        if GREPPY_RE.search(cmd):
            stats["greppy_calls"] += 1
            sub = re.search(r"\bgreppy\s+(?:--root\s+\S+\s+)?([a-z-]+)", cmd)
            if sub:
                stats["greppy_subcommands"][sub.group(1)] += 1
            if READ_RE.search(cmd):
                stats["reads"] += 1
                if "--handle" in cmd:
                    stats["read_handles"] += 1
            verb = EDIT_VERB_RE.search(cmd)
            if verb:
                stats["edit_verbs"][verb.group(1)] += 1
                low = rec.result.lower()
                for code, label in EDIT_EXITS.items():
                    if code and (f"exit {code}" in low or f'"exit_code": {code}' in low or f"status\": \"{label.split('/')[0]}" in low):
                        stats["edit_failures"][label] += 1
                        flaws.append(f"call#{idx}: edit {verb.group(1)} -> {label}: {cmd[:120]}")
                        break
                for m in re.finditer(r'"path"\s*:\s*"([^"]+)"', rec.result):
                    edited_files.add(m.group(1))
            if "usage:" in rec.result[:200] or "error:" in rec.result[:200].lower():
                stats["usage_errors"] += 1
                flaws.append(f"call#{idx}: greppy call rejected: {cmd[:120]} -> {rec.result[:160]!r}")
            if RG_STYLE_RE.search(cmd):
                stats["rg_style_calls"] += 1
        else:
            mopen = SOURCE_OPEN_RE.search(cmd)
            if mopen:
                stats["source_opens"] += 1
                opened = mopen.group(2)
                if any(opened.endswith(Path(f).name) for f in edited_files):
                    stats["post_edit_reopens"] += 1
                    flaws.append(f"call#{idx}: re-read of already-edited file {opened}: {cmd[:120]}")
            if FALLBACK_RE.search(cmd):
                stats["fallback_edits"] += 1
                flaws.append(f"call#{idx}: non-greppy edit fallback: {cmd[:120]}")
    stats["flaws"] = flaws
    return stats


def print_timeline(turns: list[dict[str, Any]], excerpt: int) -> None:
    for i, t in enumerate(turns, 1):
        visible = len(t["text"])
        print(f"\n=== turn {i}  out={t['output_tokens']}tok visible={visible}ch "
              f"cache={t['cache_read']}")
        if t["text"]:
            print("  say:", t["text"][:excerpt].replace("\n", "\n       "))
        for rec in t["calls"]:
            arg = rec.command or json.dumps(rec.arguments)[:excerpt]
            print(f"  call[{rec.name}]: {arg[:excerpt]}")
            res = rec.result.strip()
            marker = " (ERROR)" if rec.is_error else ""
            print(f"  saw{marker} ({len(res)}ch): {res[:excerpt]!r}")


def print_stats(stats: dict[str, Any]) -> None:
    ordered = [
        "tool_calls", "greppy_calls", "reads", "read_handles", "rg_style_calls",
        "source_opens", "post_edit_reopens", "fallback_edits",
        "repeated_commands", "errored_calls", "usage_errors",
    ]
    print("\n--- stats ---")
    for k in ordered:
        print(f"  {k}: {stats[k]}")
    for counter_key in ("greppy_subcommands", "edit_verbs", "edit_failures"):
        if stats[counter_key]:
            print(f"  {counter_key}: {dict(stats[counter_key])}")
    if stats["flaws"]:
        print("\n--- flaw candidates ---")
        for f in stats["flaws"]:
            print("  *", f)
    else:
        print("\n--- flaw candidates: none ---")


def analyze_one(path: Path, excerpt: int, timeline: bool) -> dict[str, Any]:
    events = load_events(path)
    turns, order = build_timeline(events)
    stats = classify_calls(order)
    stats["turns"] = len(turns)
    stats["output_tokens"] = sum(t["output_tokens"] for t in turns)
    stats["visible_chars"] = sum(len(t["text"]) for t in turns)
    if timeline:
        print(f"### {path}")
        print_timeline(turns, excerpt)
        print_stats(stats)
        hidden = stats["output_tokens"] - stats["visible_chars"] // 4
        print(f"\n  hidden-deliberation proxy: {stats['output_tokens']} output tokens "
              f"vs ~{stats['visible_chars'] // 4} visible-text tokens "
              f"(~{max(hidden, 0)} spent on unseen reasoning)")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="agent.jsonl file, or raw_traces run dir with --summary")
    ap.add_argument("--summary", action="store_true", help="aggregate per arm over a run dir")
    ap.add_argument("--excerpt", type=int, default=400, help="chars of text/result excerpts")
    args = ap.parse_args()

    target = Path(args.target)
    if not args.summary:
        analyze_one(target, args.excerpt, timeline=True)
        return 0

    per_arm: dict[str, list[dict[str, Any]]] = {}
    for trace in sorted(target.glob("*/*/agent.jsonl")):
        arm = trace.parent.name
        per_arm.setdefault(arm, []).append(
            {"task": trace.parent.parent.name, **analyze_one(trace, args.excerpt, timeline=False)}
        )
    for arm, rows in sorted(per_arm.items()):
        print(f"\n===== arm: {arm} ({len(rows)} tasks) =====")
        for k in ("tool_calls", "greppy_calls", "source_opens", "post_edit_reopens",
                  "fallback_edits", "usage_errors", "output_tokens"):
            print(f"  {k}: total {sum(r[k] for r in rows)}")
        verbs: Counter = Counter()
        fails: Counter = Counter()
        for r in rows:
            verbs.update(r["edit_verbs"])
            fails.update(r["edit_failures"])
        if verbs:
            print(f"  edit_verbs: {dict(verbs)}")
        if fails:
            print(f"  edit_failures: {dict(fails)}")
        for r in rows:
            if r["flaws"]:
                print(f"  -- {r['task']}:")
                for f in r["flaws"]:
                    print("     *", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
