#!/usr/bin/env python3
"""Select the highest-scoring diverse hard agent-coding candidates."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "validated_v2.jsonl"
DEFAULT_OUTPUT = HERE / "hard_selection.json"
PREFERRED_TYPES = {
    "cross-cutting-change",
    "refactor-mit-verhalten",
    "feature-implementation",
    "reported-bugfix",
}
LANGUAGE_NAMES = {
    "go": "Go",
    "java": "Java",
    "python": "Python",
    "rust": "Rust",
    "typescript": "TypeScript",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--max-per-language", type=int, default=6)
    parser.add_argument("--min-languages", type=int, default=3)
    parser.add_argument("--min-files", type=int, default=3)
    parser.add_argument("--min-lines", type=int, default=30)
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument(
        "--eligible-ids",
        type=Path,
        help="Restrict eligibility to ids from a JSON array or one-per-line text file",
    )
    parser.add_argument(
        "--verdict",
        action="append",
        help="Restrict eligibility to a verdict (repeatable); defaults to every verdict",
    )
    args = parser.parse_args()
    for name in ("count", "max_per_language", "min_languages", "min_files", "min_lines"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    return args


def slug_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._").lower()


def task_type(row: dict[str, Any]) -> str:
    value = row.get("type") or row.get("category")
    if not isinstance(value, str) or not value:
        raise ValueError("candidate has no type/category")
    return value


def task_id(row: dict[str, Any]) -> str:
    repo = str(row["repo"]).removesuffix(".git").rstrip("/").rsplit("/", 1)[-1]
    return f"{slug_component(repo)}-{slug_component(task_type(row))}-{row['commit'][:12].lower()}"


def normalized_language(row: dict[str, Any]) -> str:
    value = row.get("language")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("candidate has no language")
    return value.strip().lower()


def read_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        if not isinstance(document, list):
            raise SystemExit(f"{path}: expected a JSON array of task ids")
        values = document
    if not values or not all(isinstance(value, str) and value for value in values):
        raise SystemExit(f"{path}: task ids must be non-empty strings")
    return set(values)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SystemExit(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def decorate(row: dict[str, Any]) -> dict[str, Any]:
    files = row.get("files_touched")
    lines = row.get("lines_changed")
    if not isinstance(files, list) or isinstance(lines, bool) or not isinstance(lines, int):
        raise ValueError("candidate has malformed files_touched/lines_changed")
    language_key = normalized_language(row)
    kind = task_type(row)
    return {
        "id": task_id(row),
        "language": LANGUAGE_NAMES.get(language_key, language_key),
        "language_key": language_key,
        "type": kind,
        "files_touched": len(files),
        "lines_changed": lines,
        "score": len(files) * math.log(lines),
        "repo": row["repo"],
        "commit": row["commit"],
        "parent": row["parent"],
        "verdict": row.get("verdict"),
        "preferred_type": kind in PREFERRED_TYPES,
    }


def main() -> int:
    args = parse_args()
    excluded = set(args.exclude_id)
    eligible_ids = read_ids(args.eligible_ids) if args.eligible_ids else None
    verdicts = set(args.verdict or [])
    candidates = []
    for raw in read_rows(args.validated):
        row = decorate(raw)
        if row["files_touched"] < args.min_files or row["lines_changed"] < args.min_lines:
            continue
        if row["id"] in excluded or (eligible_ids is not None and row["id"] not in eligible_ids):
            continue
        if verdicts and row["verdict"] not in verdicts:
            continue
        candidates.append(row)

    # Score is primary. Preferred task types and stable ids only break exact score ties.
    candidates.sort(key=lambda row: (-row["score"], not row["preferred_type"], row["id"]))
    selected: list[dict[str, Any]] = []
    language_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    commit_keys: set[tuple[str, str]] = set()
    max_per_type = args.count // 2
    for row in candidates:
        commit_key = (row["repo"], row["commit"])
        if commit_key in commit_keys:
            continue
        if language_counts[row["language_key"]] >= args.max_per_language:
            continue
        if type_counts[row["type"]] >= max_per_type:
            continue
        selected.append(row)
        language_counts[row["language_key"]] += 1
        type_counts[row["type"]] += 1
        commit_keys.add(commit_key)
        if len(selected) == args.count:
            break

    selected_ids = {row["id"] for row in selected}
    reserve = [row for row in candidates if row["id"] not in selected_ids]
    if len({row["language_key"] for row in selected}) < args.min_languages:
        raise SystemExit(f"selection has fewer than {args.min_languages} languages")
    document = {
        "criteria": {
            "count": args.count,
            "min_files_touched": args.min_files,
            "min_lines_changed": args.min_lines,
            "score": "files_touched * log(lines_changed)",
            "max_per_language": args.max_per_language,
            "min_languages": args.min_languages,
            "max_type_share": 0.5,
            "preferred_types": sorted(PREFERRED_TYPES),
        },
        "selected": selected,
        "reserve": reserve,
        "distribution": {
            "languages": dict(sorted(Counter(row["language"] for row in selected).items())),
            "types": dict(sorted(Counter(row["type"] for row in selected).items())),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"selected {len(selected)} of {len(candidates)} eligible hard candidates")
    print(f"languages: {document['distribution']['languages']}")
    print(f"types: {document['distribution']['types']}")
    print(f"output: {args.output.resolve()}")
    return 0 if len(selected) == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
