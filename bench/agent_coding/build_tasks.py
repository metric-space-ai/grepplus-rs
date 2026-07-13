#!/usr/bin/env python3
"""Build and prove the public v1 coding-outcome task bank."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass


HERE = pathlib.Path(__file__).resolve().parent
MANIFEST_PATH = HERE.parent / "agent_efficiency" / "realcorpus" / "MANIFEST.json"
OUTPUT_PATH = HERE / "tasks_v1.json"
SCHEMA_VERSION = "greppy.agent-coding-tasks.v1"


@dataclass(frozen=True)
class Mutation:
    task_id: str
    repo: str
    path: str
    old: str
    new: str
    user_task: str
    test_command: tuple[str, ...]
    setup_commands: tuple[tuple[str, ...], ...] = ()
    timeout_seconds: int = 600


FLASK_VENV = ".tox/greppy-bench-venv"
PYTEST = (f"{FLASK_VENV}/bin/python3", "-m", "pytest", "-q")
FLASK_SETUP = (
    ("python3", "-m", "venv", "--clear", FLASK_VENV),
    (
        f"{FLASK_VENV}/bin/python3",
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "pytest==8.4.2",
        "asgiref==3.11.1",
        "python-dotenv==1.2.2",
    ),
)
GSON_TEST = ("mvn", "-q", "-pl", "gson")
ZOD_TEST = ("pnpm", "exec", "vitest", "run")
ZOD_SETUP = (("pnpm", "install", "--frozen-lockfile", "--prefer-offline"),)
SERDE_TEST = ("cargo", "test", "-p", "serde_test_suite")
TOKIO_STREAM_TEST = ("cargo", "test", "-p", "tokio-stream")
GO_TEST = ("go", "test")


MUTATIONS = (
    Mutation(
        "flask-debug-env-values",
        "flask",
        "src/flask/helpers.py",
        'return bool(val and val.lower() not in {"0", "false", "no"})',
        "return bool(val)",
        "Restore the documented handling of false-like values in the debug environment setting.",
        PYTEST + ("tests/test_helpers.py::TestHelpers::test_get_debug_flag",),
        setup_commands=FLASK_SETUP,
    ),
    Mutation(
        "flask-context-global-default",
        "flask",
        "src/flask/ctx.py",
        "return self.__dict__.get(name, default)",
        "return self.__dict__.get(name)",
        "Restore the application-context namespace lookup behavior when a caller supplies a fallback.",
        PYTEST + ("tests/test_appctx.py::test_app_ctx_globals_methods",),
        setup_commands=FLASK_SETUP,
    ),
    Mutation(
        "flask-method-view-dispatch",
        "flask",
        "src/flask/views.py",
        "meth = getattr(self, request.method.lower(), None)",
        "meth = getattr(self, request.method.upper(), None)",
        "Restore class-based request dispatch for the normal HTTP method handlers.",
        PYTEST + ("tests/test_views.py::test_method_based_view",),
        setup_commands=FLASK_SETUP,
    ),
    Mutation(
        "flask-session-application-root",
        "flask",
        "src/flask/sessions.py",
        'return app.config["SESSION_COOKIE_PATH"] or app.config["APPLICATION_ROOT"]  # type: ignore[no-any-return]',
        'return app.config["SESSION_COOKIE_PATH"] or "/"  # type: ignore[no-any-return]',
        "Restore the configured application-root fallback used for session cookie paths.",
        PYTEST + ("tests/test_basic.py::test_session_using_application_root",),
        setup_commands=FLASK_SETUP,
    ),
    Mutation(
        "flask-json-date-format",
        "flask",
        "src/flask/json/provider.py",
        "return http_date(o)",
        "return str(o)",
        "Restore the public JSON representation used for date and datetime values.",
        PYTEST + ("tests/test_json.py::test_jsonify_datetime",),
        setup_commands=FLASK_SETUP,
    ),
    Mutation(
        "hugo-string-equal-fold",
        "hugo",
        "common/hstrings/strings.go",
        "return strings.EqualFold(string(s), s2)",
        "return string(s) == s2",
        "Restore case-insensitive comparison for strings used by template equality operations.",
        GO_TEST + ("./common/hstrings", "-run", "^TestStringEqualFold$", "-count=1"),
    ),
    Mutation(
        "hugo-uppercase-detection",
        "hugo",
        "common/hstrings/strings.go",
        "if 'A' <= r && r <= 'Z' {",
        "if 'B' <= r && r <= 'Z' {",
        "Restore uppercase detection across the complete ASCII uppercase range.",
        GO_TEST + ("./common/hstrings", "-run", "^TestHasUppercase$", "-count=1"),
    ),
    Mutation(
        "hugo-unique-strings",
        "hugo",
        "common/hstrings/strings.go",
        "if !seen {\n\t\t\tunique = append(unique, val)\n\t\t}",
        "if seen {\n\t\t\tunique = append(unique, val)\n\t\t}",
        "Restore duplicate removal while preserving the first occurrence of each string.",
        GO_TEST + ("./common/hstrings", "-run", "^TestUniqueStrings$", "-count=1"),
    ),
    Mutation(
        "hugo-unique-strings-sorted",
        "hugo",
        "common/hstrings/strings.go",
        "return s[:i+1]",
        "return s[:i]",
        "Restore the complete sorted unique result, including its final distinct value.",
        GO_TEST + ("./common/hstrings", "-run", "^TestUniqueStringsSorted$", "-count=1"),
    ),
    Mutation(
        "hugo-base-url-trailing-slash",
        "hugo",
        "common/urls/baseURL.go",
        "if !strings.HasSuffix(u.Path, \"/\") {",
        "if strings.HasSuffix(u.Path, \"/\") {",
        "Restore canonical trailing-slash normalization for base URLs.",
        GO_TEST + ("./common/urls", "-run", "^TestBaseURL$", "-count=1"),
    ),
    Mutation(
        "gson-camel-case-separation",
        "gson",
        "gson/src/main/java/com/google/gson/FieldNamingPolicy.java",
        "if (Character.isUpperCase(character) && translation.length() != 0) {",
        "if (Character.isUpperCase(character) && translation.length() == 0) {",
        "Restore word separation when translating camel-cased field names.",
        GSON_TEST + ("-Dtest=FieldNamingPolicyTest#testSeparateCamelCase", "test"),
    ),
    Mutation(
        "gson-leading-letter-uppercase",
        "gson",
        "gson/src/main/java/com/google/gson/FieldNamingPolicy.java",
        "if (Character.isLetter(c)) {",
        "if (Character.isUpperCase(c)) {",
        "Restore capitalization of the first eligible letter in translated field names.",
        GSON_TEST + ("-Dtest=FieldNamingPolicyTest#testUpperCaseFirstLetter", "test"),
    ),
    Mutation(
        "gson-default-long-number",
        "gson",
        "gson/src/main/java/com/google/gson/LongSerializationPolicy.java",
        "return new JsonPrimitive(value);",
        "return new JsonPrimitive(value.toString());",
        "Restore the default numeric JSON representation for long values.",
        GSON_TEST + ("-Dtest=LongSerializationPolicyTest#testDefaultLongSerialization", "test"),
    ),
    Mutation(
        "gson-string-long-policy",
        "gson",
        "gson/src/main/java/com/google/gson/LongSerializationPolicy.java",
        "return new JsonPrimitive(value.toString());",
        "return new JsonPrimitive(value);",
        "Restore quoted JSON output when the string-based long serialization policy is selected.",
        GSON_TEST + ("-Dtest=LongSerializationPolicyTest#testStringLongSerialization", "test"),
    ),
    Mutation(
        "gson-boolean-string-case",
        "gson",
        "gson/src/main/java/com/google/gson/JsonPrimitive.java",
        "return Boolean.parseBoolean(getAsString());",
        'return "true".equals(getAsString());',
        "Restore case-insensitive boolean conversion for string-backed JSON primitives.",
        GSON_TEST + ("-Dtest=JsonPrimitiveTest#testBoolean", "test"),
    ),
    Mutation(
        "zod-inclusive-number-maximum",
        "zod",
        "packages/zod/src/v4/core/checks.ts",
        "if (def.inclusive ? payload.value <= def.value : payload.value < def.value) {",
        "if (payload.value < def.value) {",
        "Restore inclusive numeric maximum validation at the configured boundary.",
        ZOD_TEST + ("packages/zod/src/v4/classic/tests/number.test.ts", "-t", "lte"),
        setup_commands=ZOD_SETUP,
    ),
    Mutation(
        "zod-inclusive-number-minimum",
        "zod",
        "packages/zod/src/v4/core/checks.ts",
        "if (def.inclusive ? payload.value >= def.value : payload.value > def.value) {",
        "if (payload.value > def.value) {",
        "Restore inclusive numeric minimum validation at the configured boundary.",
        ZOD_TEST + ("packages/zod/src/v4/classic/tests/number.test.ts", "-t", "gte"),
        setup_commands=ZOD_SETUP,
    ),
    Mutation(
        "zod-set-maximum-boundary",
        "zod",
        "packages/zod/src/v4/core/checks.ts",
        "if (size <= def.maximum) return;",
        "if (size < def.maximum) return;",
        "Restore acceptance of sets whose size is exactly the configured maximum.",
        ZOD_TEST + ("packages/zod/src/v4/classic/tests/set.test.ts", "-t", "valid parse: size-related methods"),
        setup_commands=ZOD_SETUP,
    ),
    Mutation(
        "zod-string-maximum-boundary",
        "zod",
        "packages/zod/src/v4/core/checks.ts",
        "if (length <= def.maximum) return;",
        "if (length < def.maximum) return;",
        "Restore acceptance of strings whose length is exactly the configured maximum.",
        ZOD_TEST + ("packages/zod/src/v4/classic/tests/string.test.ts", "-t", "length checks"),
        setup_commands=ZOD_SETUP,
    ),
    Mutation(
        "zod-string-prefix-check",
        "zod",
        "packages/zod/src/v4/core/checks.ts",
        "if (payload.value.startsWith(def.prefix)) return;",
        "if (payload.value.endsWith(def.prefix)) return;",
        "Restore prefix validation for strings without changing suffix validation.",
        ZOD_TEST + ("packages/zod/src/v4/classic/tests/string.test.ts", "-t", "startswith/endswith"),
        setup_commands=ZOD_SETUP,
    ),
    Mutation(
        "serde-option-some-token",
        "serde",
        "serde_core/src/ser/impls.rs",
        "Some(ref value) => serializer.serialize_some(value),",
        "Some(_) => serializer.serialize_none(),",
        "Restore serialization of present optional values as present values rather than null-like values.",
        SERDE_TEST + ("--test", "test_ser", "test_option", "--", "--exact"),
    ),
    Mutation(
        "serde-duration-seconds-field",
        "serde",
        "serde_core/src/ser/impls.rs",
        'tri!(state.serialize_field("secs", &self.as_secs()));',
        'tri!(state.serialize_field("seconds", &self.as_secs()));',
        "Restore the stable field representation used when serializing standard durations.",
        SERDE_TEST + ("--test", "test_ser", "test_duration", "--", "--exact"),
    ),
    Mutation(
        "serde-system-time-nanos-field",
        "serde",
        "serde_core/src/ser/impls.rs",
        'tri!(state.serialize_field("nanos_since_epoch", &duration_since_epoch.subsec_nanos()));',
        'tri!(state.serialize_field("nanos", &duration_since_epoch.subsec_nanos()));',
        "Restore the stable subsecond field representation used for system timestamps.",
        SERDE_TEST + ("--test", "test_ser", "test_system_time", "--", "--exact"),
    ),
    Mutation(
        "serde-bound-included-variant",
        "serde",
        "serde_core/src/ser/impls.rs",
        'serializer.serialize_newtype_variant("Bound", 1, "Included", value)',
        'serializer.serialize_newtype_variant("Bound", 1, "Inclusive", value)',
        "Restore the serialized variant identity for inclusive range bounds.",
        SERDE_TEST + ("--test", "test_ser", "test_bound", "--", "--exact"),
    ),
    Mutation(
        "serde-range-start-field",
        "serde",
        "serde_core/src/ser/impls.rs",
        'let mut state = tri!(serializer.serialize_struct("Range", 2));\n        tri!(state.serialize_field("start", &self.start));',
        'let mut state = tri!(serializer.serialize_struct("Range", 2));\n        tri!(state.serialize_field("begin", &self.start));',
        "Restore the stable starting-bound field representation for standard ranges.",
        SERDE_TEST + ("--test", "test_ser", "test_range", "--", "--exact"),
    ),
    Mutation(
        "tokio-stream-map-termination",
        "tokio",
        "tokio-stream/src/stream_ext/map.rs",
        "self.stream.is_terminated()",
        "!self.stream.is_terminated()",
        "Restore fused-stream termination reporting for mapped streams.",
        TOKIO_STREAM_TEST + ("--test", "stream_fused", "map_not_terminated_before_done", "--", "--exact"),
    ),
    Mutation(
        "tokio-stream-filter-termination",
        "tokio",
        "tokio-stream/src/stream_ext/filter.rs",
        "self.stream.is_terminated()",
        "!self.stream.is_terminated()",
        "Restore fused-stream termination reporting for filtered streams.",
        TOKIO_STREAM_TEST + ("--test", "stream_fused", "filter_not_terminated_before_done", "--", "--exact"),
    ),
    Mutation(
        "tokio-stream-skip-termination",
        "tokio",
        "tokio-stream/src/stream_ext/skip.rs",
        "self.stream.is_terminated()",
        "!self.stream.is_terminated()",
        "Restore fused-stream termination reporting for streams that skip initial items.",
        TOKIO_STREAM_TEST + ("--test", "stream_fused", "skip_not_terminated_before_done", "--", "--exact"),
    ),
    Mutation(
        "tokio-stream-take-termination",
        "tokio",
        "tokio-stream/src/stream_ext/take.rs",
        "fn is_terminated(&self) -> bool {\n        self.remaining == 0\n    }",
        "fn is_terminated(&self) -> bool {\n        self.remaining != 0\n    }",
        "Restore termination reporting when a stream's item limit has not yet been reached.",
        TOKIO_STREAM_TEST + ("--test", "stream_fused", "take_not_terminated_before_limit", "--", "--exact"),
    ),
    Mutation(
        "tokio-stream-take-while-termination",
        "tokio",
        "tokio-stream/src/stream_ext/take_while.rs",
        "fn is_terminated(&self) -> bool {\n        self.done\n    }",
        "fn is_terminated(&self) -> bool {\n        !self.done\n    }",
        "Restore termination reporting after a take-while predicate stops the stream.",
        TOKIO_STREAM_TEST + ("--test", "stream_fused", "take_while_terminated_after_predicate_fails", "--", "--exact"),
    ),
)


def run(argv: list[str] | tuple[str, ...], cwd: pathlib.Path, timeout: int, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def source_at(repo: pathlib.Path, commit: str, path: str) -> str:
    result = run(("git", "show", f"{commit}:{path}"), repo, 60)
    if result.returncode:
        raise RuntimeError(f"cannot read {repo.name}:{path} at {commit}")
    return result.stdout


def mutation_patch(path: str, source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected one occurrence in {path}, found {source.count(old)}: {old!r}")
    changed = source.replace(old, new, 1)
    body = "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            changed.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            n=3,
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}"


def build(repos_root: pathlib.Path) -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["repos"]
    tasks = []
    for mutation in MUTATIONS:
        repo_data = manifest[mutation.repo]
        repo = repos_root / mutation.repo
        source = source_at(repo, repo_data["commit"], mutation.path)
        tasks.append(
            {
                "id": mutation.task_id,
                "repository": {"url": repo_data["url"], "commit": repo_data["commit"]},
                "mutation_patch": mutation_patch(mutation.path, source, mutation.old, mutation.new),
                "user_task": mutation.user_task,
                "setup_commands": [list(command) for command in mutation.setup_commands],
                "test_command": list(mutation.test_command),
                "timeout_seconds": mutation.timeout_seconds,
            }
        )
    return {"schema_version": SCHEMA_VERSION, "tasks": tasks}


def prove(document: dict[str, object], repos_root: pathlib.Path, only_repo: str | None = None) -> None:
    by_repo: dict[str, list[dict[str, object]]] = {}
    task_to_repo = {mutation.task_id: mutation.repo for mutation in MUTATIONS}
    for task in document["tasks"]:  # type: ignore[index]
        by_repo.setdefault(task_to_repo[task["id"]], []).append(task)  # type: ignore[index]

    if only_repo is not None:
        by_repo = {only_repo: by_repo[only_repo]}

    with tempfile.TemporaryDirectory(prefix="agent-coding-task-proof-") as temp_name:
        temp_root = pathlib.Path(temp_name)
        for repo_name, tasks in by_repo.items():
            source = repos_root / repo_name
            repo = temp_root / repo_name
            subprocess.run(("git", "clone", "--quiet", "--shared", str(source), str(repo)), check=True)
            commit = tasks[0]["repository"]["commit"]  # type: ignore[index]
            subprocess.run(("git", "checkout", "--quiet", "--detach", commit), cwd=repo, check=True)
            env = os.environ.copy()
            if repo_name == "gson":
                java_home = pathlib.Path("/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home")
                if java_home.is_dir():
                    env["JAVA_HOME"] = str(java_home)
                    env["PATH"] = f"{java_home / 'bin'}{os.pathsep}{env['PATH']}"

            for task in tasks:
                subprocess.run(("git", "reset", "--hard", "--quiet", commit), cwd=repo, check=True)
                for index, setup_command in enumerate(task["setup_commands"]):
                    setup = run(setup_command, repo, task["timeout_seconds"], env)  # type: ignore[arg-type]
                    if setup.returncode:
                        raise RuntimeError(
                            f"{task['id']}: setup command {index} failed:\n{setup.stdout[-4000:]}"
                        )
                ignored_path = ".tox" if repo_name == "flask" else "node_modules" if repo_name == "zod" else None
                if ignored_path is not None:
                    ignored = run(("git", "check-ignore", "-q", ignored_path), repo, 60, env)
                    if ignored.returncode:
                        raise RuntimeError(f"{task['id']}: setup path is not ignored: {ignored_path}")
                setup_diff = run(("git", "diff", "--quiet", commit, "--"), repo, 60, env)
                if setup_diff.returncode:
                    raise RuntimeError(f"{task['id']}: setup modified tracked files")
                setup_status = run(("git", "status", "--porcelain", "--untracked-files=all"), repo, 60, env)
                if setup_status.returncode or setup_status.stdout:
                    raise RuntimeError(f"{task['id']}: setup left non-ignored files:\n{setup_status.stdout}")
                command = task["test_command"]
                timeout = task["timeout_seconds"]
                clean = run(command, repo, timeout, env)  # type: ignore[arg-type]
                if clean.returncode:
                    raise RuntimeError(f"{task['id']}: clean test failed:\n{clean.stdout[-4000:]}")
                applied = subprocess.run(
                    ("git", "apply", "--whitespace=nowarn", "-"),
                    cwd=repo,
                    input=task["mutation_patch"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if applied.returncode:
                    raise RuntimeError(f"{task['id']}: mutation did not apply:\n{applied.stdout}")
                mutated = run(command, repo, timeout, env)  # type: ignore[arg-type]
                if mutated.returncode == 0:
                    raise RuntimeError(f"{task['id']}: mutated test unexpectedly passed")
                print(f"PASS {task['id']}: clean=0 mutated={mutated.returncode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT_PATH)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-repo", choices=sorted({mutation.repo for mutation in MUTATIONS}))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = build(args.repos_root.resolve())
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.verify:
        prove(document, args.repos_root.resolve(), args.verify_repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
