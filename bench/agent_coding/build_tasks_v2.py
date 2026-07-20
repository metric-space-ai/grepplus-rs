#!/usr/bin/env python3
"""Build validated v2 agent-coding tasks from harvested commits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_VALIDATED = HERE / "validated_v2.jsonl"
DEFAULT_SERIOUS = HERE / "harvest_candidates_v2_serious.jsonl"
DEFAULT_MEDIUM = HERE / "harvest_candidates_v2.jsonl"
DEFAULT_V1 = HERE / "tasks_v1.json"
DEFAULT_SCHEMA = HERE / "task_v2.schema.json"
DEFAULT_OUTPUT = HERE / "tasks_v2.json"
SCHEMA_VERSION = "greppy.agent-coding-tasks.v2"
TIMEOUTS = {"S": 1800, "M": 1200}

# Function declarations on changed lines. The check intentionally does not treat
# every call-site identifier as a declaration; doing so would flag ordinary prose
# words such as "object", "format", and "drop" as implementation leaks.
FUNCTION_DECLARATION_PATTERNS = (
    re.compile(r"\b(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\("),
    re.compile(r"\bfunc\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\("),
    re.compile(r"\bfn\s+([A-Za-z_]\w*)\s*(?:<[^>{}]*>)?\s*\("),
    re.compile(r"\bfunction\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("),
    re.compile(
        r"\b(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*="
        r"\s*(?:async\s*)?(?:<[^>]+>\s*)?\([^)]*\)\s*=>"
    ),
)


class BuildSkip(Exception):
    """A candidate cannot safely become a task."""


class SchemaValidationError(ValueError):
    """An instance does not satisfy the loaded JSON Schema."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated", type=Path, default=DEFAULT_VALIDATED)
    parser.add_argument("--serious", type=Path, default=DEFAULT_SERIOUS)
    parser.add_argument("--medium", type=Path, default=DEFAULT_MEDIUM)
    parser.add_argument("--tasks-v1", type=Path, default=DEFAULT_V1)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--harvest-root",
        type=Path,
        help="Directory containing one local clone per repository",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Write at most this many passing tasks (used for staged validation)",
    )
    parser.add_argument(
        "--min-tasks",
        type=int,
        default=40,
        help="Fail without replacing the output if fewer tasks pass (default: 40)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.min_tasks < 1:
        parser.error("--min-tasks must be at least 1")
    return args


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON from {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(row, dict):
                    raise SystemExit(f"{path}:{line_number}: expected a JSON object")
                rows.append(row)
    except OSError as exc:
        raise SystemExit(f"cannot read {path}: {exc}") from exc
    return rows


def json_type_matches(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "null":
        return instance is None
    raise SchemaValidationError(f"unsupported schema type {expected!r}")


def resolve_local_ref(root_schema: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise SchemaValidationError(f"unsupported non-local $ref {reference!r}")
    node: Any = root_schema
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SchemaValidationError(f"unresolvable $ref {reference!r}")
        node = node[part]
    if not isinstance(node, dict):
        raise SchemaValidationError(f"$ref {reference!r} does not point to a schema")
    return node


def validate_against_schema(
    instance: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: str = "$",
) -> None:
    """Validate the subset of Draft 2020-12 used by task_v2.schema.json."""
    if "$ref" in schema:
        validate_against_schema(instance, resolve_local_ref(root_schema, schema["$ref"]), root_schema, path)
        return

    if "const" in schema and instance != schema["const"]:
        raise SchemaValidationError(f"{path}: expected constant {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaValidationError(f"{path}: {instance!r} is not in {schema['enum']!r}")

    expected_type = schema.get("type")
    if expected_type is not None and not json_type_matches(instance, expected_type):
        raise SchemaValidationError(f"{path}: expected {expected_type}, got {type(instance).__name__}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in instance]
        if missing:
            raise SchemaValidationError(f"{path}: missing required properties {missing!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(instance) - set(properties))
            if extras:
                raise SchemaValidationError(f"{path}: unexpected properties {extras!r}")
        for key, value in instance.items():
            if key in properties:
                validate_against_schema(value, properties[key], root_schema, f"{path}.{key}")

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise SchemaValidationError(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise SchemaValidationError(f"{path}: more than {schema['maxItems']} items")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in instance]
            if len(encoded) != len(set(encoded)):
                raise SchemaValidationError(f"{path}: duplicate array items")
        if "items" in schema:
            for index, value in enumerate(instance):
                validate_against_schema(value, schema["items"], root_schema, f"{path}[{index}]")

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise SchemaValidationError(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise SchemaValidationError(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise SchemaValidationError(f"{path}: does not match {schema['pattern']!r}")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaValidationError(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise SchemaValidationError(f"{path}: above maximum {schema['maximum']}")


def run_git(repo: Path, args: list[str], *, text: bool = False) -> bytes | str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        errors="replace" if text else None,
    )
    if proc.returncode:
        stderr = proc.stderr if text else proc.stderr.decode("utf-8", "replace")
        raise BuildSkip(f"git {' '.join(args)} failed: {stderr.strip()[-500:]}")
    return proc.stdout


def object_exists(repo: Path, object_name: str) -> bool:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{object_name}^{{commit}}"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def normalize_repo_url(url: str) -> str:
    return url.removesuffix(".git").rstrip("/")


def repo_name(url: str) -> str:
    name = normalize_repo_url(url).rsplit("/", 1)[-1]
    if not name:
        raise BuildSkip(f"cannot derive repository name from {url!r}")
    return name


def infer_harvest_root(valid_rows: Iterable[dict[str, Any]]) -> Path:
    configured = os.environ.get("GREPPY_HARVEST_REPOS")
    if configured:
        return Path(configured).expanduser().resolve()

    marker = "/validation-v2/"
    for row in valid_rows:
        for field in ("proof_a", "proof_b"):
            proof = row.get(field)
            if not isinstance(proof, str) or marker not in proof:
                continue
            match = re.search(r"(/[^\s;\"']*/scratchpad)/validation-v2/", proof)
            if match is None:
                continue
            candidate = Path(match.group(1)) / "harvest-repos"
            if candidate.is_dir():
                return candidate.resolve()

    raise SystemExit(
        "cannot infer harvest clone root; pass --harvest-root or set GREPPY_HARVEST_REPOS"
    )


def load_setup_commands(path: Path) -> dict[str, list[list[str]]]:
    document = read_json(path)
    if not isinstance(document, dict) or not isinstance(document.get("tasks"), list):
        raise SystemExit(f"{path}: expected a tasks_v1 object with a tasks array")
    variants: dict[str, set[str]] = {}
    decoded: dict[str, list[list[str]]] = {}
    for task in document["tasks"]:
        try:
            url = normalize_repo_url(task["repository"]["url"])
            commands = task["setup_commands"]
        except (KeyError, TypeError) as exc:
            raise SystemExit(f"{path}: malformed v1 task setup data") from exc
        encoded = json.dumps(commands, sort_keys=True)
        variants.setdefault(url, set()).add(encoded)
        decoded[url] = commands
    conflicts = sorted(url for url, values in variants.items() if len(values) != 1)
    if conflicts:
        raise SystemExit(f"{path}: conflicting setup_commands for {conflicts!r}")
    return decoded


def index_candidates(rows: Iterable[dict[str, Any]], candidate_class: str) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        try:
            key = (normalize_repo_url(row["repo"]), row["commit"])
        except KeyError as exc:
            raise SystemExit(f"{candidate_class} candidate is missing {exc.args[0]!r}") from exc
        if key in result:
            raise SystemExit(f"duplicate {candidate_class} candidate for {key[0]} {key[1]}")
        result[key] = row
    return result


def task_type(row: dict[str, Any]) -> str:
    value = row.get("type") or row.get("category")
    if not isinstance(value, str) or not value:
        raise BuildSkip("candidate has no type/category")
    return value


def slug_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not slug:
        raise BuildSkip(f"cannot form id component from {value!r}")
    return slug.lower()


def expected_task_id(row: dict[str, Any]) -> str:
    return f"{slug_component(repo_name(row['repo']))}-{slug_component(task_type(row))}-{row['commit'][:12].lower()}"


def make_user_task(candidate_class: str, source: dict[str, Any]) -> str:
    intent = source.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise BuildSkip("source candidate has no intent")
    intent = intent.strip()
    if candidate_class == "S":
        title = source.get("issue_title")
        if isinstance(title, str) and title.strip():
            return f"{title.strip()}\n\n{intent}"
    return intent


def changed_function_identifiers(code_patch: str) -> set[str]:
    identifiers: set[str] = set()
    for line in code_patch.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        changed_line = line[1:]
        for pattern in FUNCTION_DECLARATION_PATTERNS:
            identifiers.update(pattern.findall(changed_line))
    return identifiers


def leaked_identifiers(user_task: str, identifiers: Iterable[str]) -> list[str]:
    leaks: list[str] = []
    for identifier in sorted(set(identifiers)):
        pattern = rf"(?<![A-Za-z0-9_$]){re.escape(identifier)}(?![A-Za-z0-9_$])"
        if re.search(pattern, user_task):
            leaks.append(identifier)
    return leaks


def verify_clone(repo: Path, expected_url: str, parent: str, commit: str) -> None:
    if not (repo / ".git").exists():
        raise BuildSkip(f"local clone missing: {repo}")
    origin = run_git(repo, ["remote", "get-url", "origin"], text=True).strip()
    if normalize_repo_url(origin) != normalize_repo_url(expected_url):
        raise BuildSkip(f"clone origin mismatch: expected {expected_url}, found {origin}")
    for object_name in (parent, commit):
        if not object_exists(repo, object_name):
            raise BuildSkip(f"commit object missing from clone: {object_name}")


def extract_task_patches(repo: Path, row: dict[str, Any]) -> tuple[str, str]:
    parent = row["parent"]
    commit = row["commit"]
    tests = list(dict.fromkeys(row.get("tests_touched", [])))
    if not tests or not all(isinstance(path, str) and path for path in tests):
        raise BuildSkip("tests_touched is empty or malformed")

    changed_output = run_git(repo, ["diff", "--name-only", "--no-renames", parent, commit], text=True)
    changed_paths = [path for path in changed_output.splitlines() if path]
    missing_tests = sorted(set(tests) - set(changed_paths))
    if missing_tests:
        raise BuildSkip(f"tests_touched paths absent from commit diff: {missing_tests!r}")
    test_set = set(tests)
    code_paths = [path for path in changed_paths if path not in test_set]
    common = ["diff", "--binary", "--full-index", "--no-renames", parent, commit, "--"]
    test_patch_bytes = run_git(repo, [*common, *tests])
    code_patch_bytes = run_git(repo, [*common, *code_paths]) if code_paths else b""
    if not test_patch_bytes.strip():
        raise BuildSkip("test patch is empty")

    expected_hash = row.get("test_patch_sha256")
    actual_hash = hashlib.sha256(test_patch_bytes).hexdigest()
    if expected_hash and actual_hash != expected_hash:
        raise BuildSkip(f"test patch hash mismatch: expected {expected_hash}, got {actual_hash}")
    try:
        test_patch = test_patch_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BuildSkip(f"test patch is not UTF-8: {exc}") from exc
    code_patch = code_patch_bytes.decode("utf-8", "replace")
    return test_patch, code_patch


def build_one(
    row: dict[str, Any],
    source_indexes: dict[str, dict[tuple[str, str], dict[str, Any]]],
    setup_commands: dict[str, list[list[str]]],
    harvest_root: Path,
    task_schema: dict[str, Any],
    root_schema: dict[str, Any],
) -> dict[str, Any]:
    candidate_class = row.get("candidate_class")
    if candidate_class not in ("S", "M"):
        raise BuildSkip(f"invalid candidate_class {candidate_class!r}")
    url = normalize_repo_url(row["repo"])
    source = source_indexes[candidate_class].get((url, row["commit"]))
    if source is None:
        raise BuildSkip(f"candidate missing from class {candidate_class} source JSONL")
    if url not in setup_commands:
        raise BuildSkip("repository has no tasks_v1 setup_commands pattern")

    local_repo = harvest_root / repo_name(url)
    verify_clone(local_repo, url, row["parent"], row["commit"])
    test_patch, code_patch = extract_task_patches(local_repo, row)
    user_task = make_user_task(candidate_class, source)
    leaks = leaked_identifiers(user_task, changed_function_identifiers(code_patch))
    if leaks:
        raise BuildSkip(f"user_task contains changed function identifier(s): {', '.join(leaks)}")

    command = row.get("scoped_test_command")
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) and arg for arg in command):
        raise BuildSkip("scoped_test_command is empty or malformed")

    task = {
        "id": expected_task_id(row),
        "class": candidate_class,
        "type": task_type(row),
        "repository": {"url": url, "commit": row["parent"]},
        "test_patch": test_patch,
        "user_task": user_task,
        "setup_commands": copy.deepcopy(setup_commands[url]),
        "test_command": list(command),
        "timeout_seconds": TIMEOUTS[candidate_class],
    }
    try:
        validate_against_schema(task, task_schema, root_schema)
    except SchemaValidationError as exc:
        raise BuildSkip(f"task schema validation failed: {exc}") from exc
    return task


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    root_schema = read_json(args.schema)
    if not isinstance(root_schema, dict):
        raise SystemExit(f"{args.schema}: schema root must be an object")
    task_schema = resolve_local_ref(root_schema, "#/$defs/task")

    validated_rows = [row for row in read_jsonl(args.validated) if row.get("verdict") == "valid"]
    serious_index = index_candidates(read_jsonl(args.serious), "S")
    medium_index = index_candidates(read_jsonl(args.medium), "M")
    source_indexes = {"S": serious_index, "M": medium_index}
    setup_commands = load_setup_commands(args.tasks_v1)
    harvest_root = args.harvest_root.resolve() if args.harvest_root else infer_harvest_root(validated_rows)

    ordered_rows = sorted(validated_rows, key=lambda row: (expected_task_id(row), row.get("candidate_class", "")))
    tasks: list[dict[str, Any]] = []
    skips: list[tuple[str, str]] = []
    for row in ordered_rows:
        candidate_id = expected_task_id(row)
        try:
            task = build_one(row, source_indexes, setup_commands, harvest_root, task_schema, root_schema)
        except (BuildSkip, KeyError, TypeError) as exc:
            skips.append((candidate_id, str(exc)))
            continue
        tasks.append(task)
        if args.limit is not None and len(tasks) >= args.limit:
            break

    tasks.sort(key=lambda task: task["id"])
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        duplicates = sorted(task_id for task_id, count in Counter(ids).items() if count > 1)
        raise SystemExit(f"duplicate generated task ids: {duplicates!r}")

    required_count = min(args.min_tasks, args.limit) if args.limit is not None else args.min_tasks
    if len(tasks) < required_count:
        print_report(validated_rows, tasks, skips, harvest_root, args.output)
        raise SystemExit(f"only {len(tasks)} tasks passed; required at least {required_count}")

    document = {"schema_version": SCHEMA_VERSION, "tasks": tasks}
    try:
        validate_against_schema(document, root_schema, root_schema)
    except SchemaValidationError as exc:
        raise SystemExit(f"generated collection fails schema validation: {exc}") from exc
    atomic_write_json(args.output, document)
    print_report(validated_rows, tasks, skips, harvest_root, args.output)
    return 0


def print_report(
    validated_rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    skips: list[tuple[str, str]],
    harvest_root: Path,
    output: Path,
) -> None:
    counts = Counter((task["class"], task["type"]) for task in tasks)
    print(f"Validated candidates: {len(validated_rows)}")
    print(f"Built tasks: {len(tasks)}")
    print("Built by class/type:")
    for (candidate_class, kind), count in sorted(counts.items()):
        print(f"  {candidate_class}/{kind}: {count}")
    print(f"Skipped: {len(skips)}")
    for candidate_id, reason in skips:
        print(f"  {candidate_id}: {reason}")
    print(f"Harvest root: {harvest_root}")
    print(f"Output: {output.resolve()}")


if __name__ == "__main__":
    sys.exit(main())
