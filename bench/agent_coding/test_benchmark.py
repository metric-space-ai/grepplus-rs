#!/usr/bin/env python3
"""Network-free standard-library tests for the agent coding harness."""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
import unittest

import run_benchmark as bench


def git(cwd: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class GitFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="agent-coding-test-")
        self.root = pathlib.Path(self.tempdir.name)
        self.source = self.root / "source"
        self.source.mkdir()
        git(self.source, "init", "-q")
        git(self.source, "config", "user.name", "Benchmark Test")
        git(self.source, "config", "user.email", "benchmark@example.invalid")
        (self.source / "value.txt").write_text("old\n", encoding="utf-8")
        git(self.source, "add", "value.txt")
        git(self.source, "commit", "-qm", "fixture")
        self.commit = git(self.source, "rev-parse", "HEAD")
        self.backing = self.root / "repo.git"
        git(self.root, "clone", "--mirror", "--no-local", str(self.source), str(self.backing))

    def tearDown(self) -> None:
        self.tempdir.cleanup()


class PatchTests(GitFixture):
    PATCH = """diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-old
+new
"""

    def test_patch_applies_and_binary_diff_is_hashed(self) -> None:
        worktree_path = self.root / "patch-worktree"
        with bench.temporary_worktree(self.backing, self.commit, worktree_path, 10) as worktree:
            bench.apply_mutation(worktree, self.PATCH, 10)
            self.assertEqual((worktree / "value.txt").read_text(encoding="utf-8"), "new\n")
            (worktree / "asset.bin").write_bytes(b"\x00\x01\xff\x00")
            diff = bench.capture_binary_diff(worktree, self.commit, 10)
            self.assertIn(b"+new", diff)
            self.assertIn(b"GIT binary patch", diff)
            self.assertRegex(bench.sha256_bytes(diff), r"^[0-9a-f]{64}$")

    def test_invalid_patch_does_not_modify_worktree(self) -> None:
        worktree_path = self.root / "bad-patch-worktree"
        with bench.temporary_worktree(self.backing, self.commit, worktree_path, 10) as worktree:
            with self.assertRaises(bench.HarnessError):
                bench.apply_mutation(worktree, self.PATCH.replace("-old", "-missing"), 10)
            self.assertEqual((worktree / "value.txt").read_text(encoding="utf-8"), "old\n")


class WorktreeTests(GitFixture):
    def test_repository_clone_resolves_exact_pinned_commit(self) -> None:
        clone_parent = self.root / "clone-parent"
        clone_parent.mkdir()
        task = {
            "repository": {"url": str(self.source), "commit": self.commit},
            "timeout_seconds": 10,
        }
        backing = bench.clone_pinned_repository(task, clone_parent)
        resolved = git(clone_parent, "--git-dir", str(backing), "rev-parse", "HEAD")
        self.assertEqual(resolved, self.commit)

    def test_worktree_is_removed_after_exception(self) -> None:
        worktree_path = self.root / "cleanup-worktree"
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with bench.temporary_worktree(self.backing, self.commit, worktree_path, 10):
                self.assertTrue(worktree_path.is_dir())
                raise RuntimeError("intentional")
        self.assertFalse(worktree_path.exists())
        listing = git(self.root, "--git-dir", str(self.backing), "worktree", "list", "--porcelain")
        self.assertNotIn(str(worktree_path), listing)


def result_row(
    task_id: str,
    arm: str,
    *,
    passed: bool,
    tools: int,
    inputs: int,
    wall: float,
    source_opens: int = 1,
    valid: bool = True,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "arm": arm,
        "valid": valid,
        "correctness": passed,
        "agent": {
            "tool_calls": tools,
            "source_opens": source_opens,
            "input_tokens": inputs,
            "wall_seconds": wall,
        },
    }


class GradingTests(unittest.TestCase):
    def test_gate_requires_twenty_percent_reduction_in_all_three_metrics(self) -> None:
        task_ids = [f"t{i}" for i in range(30)]
        rows: list[dict[str, object]] = []
        for task_id in task_ids:
            rows.extend(
                [
                    result_row(task_id, "explorer", passed=True, tools=10, source_opens=5, inputs=1000, wall=10),
                    result_row(task_id, "greppy", passed=True, tools=8, source_opens=4, inputs=800, wall=8),
                ]
            )
        grade = bench.grade_results(rows, task_ids)
        self.assertTrue(grade["passed"])
        self.assertEqual(
            grade["efficiency_on_solved_pairs"]["greppy_to_explorer_tool_calls"],
            0.8,
        )
        self.assertEqual(grade["efficiency_on_solved_pairs"]["greppy_to_explorer_source_opens"], 0.8)
        self.assertEqual(grade["efficiency_on_solved_pairs"]["greppy_to_explorer_input_tokens"], 0.8)

        rows[-1]["agent"]["source_opens"] = 5
        grade = bench.grade_results(rows, task_ids)
        self.assertFalse(grade["efficiency_on_solved_pairs"]["all_metrics_pass"])
        self.assertFalse(grade["passed"])

    def test_one_task_cannot_pass_the_benchmark(self) -> None:
        rows = [
            result_row("t1", "explorer", passed=True, tools=10, source_opens=5, inputs=1000, wall=10),
            result_row("t1", "greppy", passed=True, tools=1, source_opens=1, inputs=100, wall=1),
        ]
        grade = bench.grade_results(rows, ["t1"])
        self.assertFalse(grade["sample_size"]["passes"])
        self.assertFalse(grade["passed"])

    def test_gate_requires_at_least_twenty_solved_pairs(self) -> None:
        task_ids = [f"t{i}" for i in range(30)]
        rows: list[dict[str, object]] = []
        for index, task_id in enumerate(task_ids):
            passed = index < 19
            rows.extend(
                [
                    result_row(task_id, "explorer", passed=passed, tools=10, source_opens=5, inputs=1000, wall=10),
                    result_row(task_id, "greppy", passed=passed, tools=8, source_opens=4, inputs=800, wall=8),
                ]
            )
        grade = bench.grade_results(rows, task_ids)
        self.assertEqual(grade["complete_pair_count"], 30)
        self.assertEqual(grade["solved_pair_count"], 19)
        self.assertFalse(grade["sample_size"]["passes"])
        self.assertFalse(grade["passed"])

    def test_exact_paired_test_detects_significant_regression(self) -> None:
        rows: list[dict[str, object]] = []
        task_ids = [f"t{i}" for i in range(5)]
        for task_id in task_ids:
            rows.extend(
                [
                    result_row(task_id, "explorer", passed=True, tools=2, inputs=100, wall=1),
                    result_row(task_id, "greppy", passed=False, tools=1, inputs=50, wall=0.1),
                ]
            )
        grade = bench.grade_results(rows, task_ids)
        self.assertFalse(grade["correctness"]["no_significant_regression"])
        self.assertEqual(grade["correctness"]["one_sided_exact_mcnemar_p"], 0.03125)
        self.assertFalse(grade["passed"])

    def test_failed_pair_never_receives_wall_time_credit(self) -> None:
        rows = [
            result_row("solved", "explorer", passed=True, tools=10, inputs=100, wall=1),
            result_row("solved", "greppy", passed=True, tools=8, inputs=100, wall=2),
            result_row("failed", "explorer", passed=True, tools=100, inputs=1000, wall=100),
            result_row("failed", "greppy", passed=False, tools=1, inputs=10, wall=0.01),
        ]
        grade = bench.grade_results(rows, ["solved", "failed"])
        self.assertEqual(grade["solved_pair_count"], 1)
        self.assertEqual(grade["wall_time_on_solved_pairs_only"]["credited_greppy_wins"], 0)
        self.assertFalse(grade["failed_tests_receive_speed_credit"])


class ContractTests(unittest.TestCase):
    def test_arm_validity_requires_success_even_when_turns_exist(self) -> None:
        self.assertFalse(bench.agent_result_is_valid({"success": False, "turns": 3, "timed_out": True}))
        self.assertFalse(bench.agent_result_is_valid({"success": False, "turns": 3, "return_code": 1}))
        self.assertTrue(bench.agent_result_is_valid({"success": True, "turns": 1, "return_code": 0}))

    def test_publishable_manifest_includes_platform_and_versions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-coding-manifest-") as tmp_name:
            root = pathlib.Path(tmp_name)
            executable = root / "fake-tool"
            executable.write_text("#!/bin/sh\nprintf 'fake-tool 1.2.3\\n'\n", encoding="utf-8")
            executable.chmod(0o755)
            task = {
                "id": "sample",
                "repository": {"url": "https://example.invalid/repo.git", "commit": "a" * 40},
                "mutation_patch": "diff --git a/a b/a\n",
                "user_task": "Fix it.",
                "test_command": ["true"],
                "timeout_seconds": 60,
            }
            document = {"schema_version": bench.TASK_SCHEMA_VERSION, "tasks": [task]}
            task_path = root / "tasks.json"
            task_path.write_text(json.dumps(document), encoding="utf-8")
            manifest = bench.build_base_manifest(
                run_id="test-run",
                task_path=task_path,
                task_document=document,
                tasks=[task],
                pi_bin=executable,
                greppy_bin=executable,
                warm_greppy=False,
            )
            self.assertEqual(manifest["executables"]["pi"]["version"], "fake-tool 1.2.3")
            self.assertEqual(manifest["executables"]["greppy"]["version"], "fake-tool 1.2.3")

    def test_resume_rejects_changed_identity_and_duplicate_rows(self) -> None:
        current = {field: {"value": field} for field in bench.RESUME_IDENTITY_FIELDS}
        previous = json.loads(json.dumps(current))
        bench.validate_resume_identity(previous, current)
        previous["prompt_contract"] = {"value": "changed"}
        with self.assertRaisesRegex(bench.HarnessError, "prompt_contract"):
            bench.validate_resume_identity(previous, current)

        row = {
            "schema_version": bench.RESULT_SCHEMA_VERSION,
            "task_id": "sample",
            "arm": "explorer",
        }
        self.assertEqual(bench.validate_resume_rows([row], ["sample"]), [row])
        with self.assertRaisesRegex(bench.HarnessError, "duplicate"):
            bench.validate_resume_rows([row, dict(row)], ["sample"])
        with self.assertRaisesRegex(bench.HarnessError, "selected task set"):
            bench.validate_resume_rows([{**row, "task_id": "other"}], ["sample"])
            self.assertTrue(manifest["platform"]["operating_system"])
            self.assertTrue(manifest["platform"]["architecture"])

    def test_schema_and_runtime_validator_agree_on_minimal_task(self) -> None:
        schema = json.loads((bench.HERE / "task.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], bench.TASK_SCHEMA_VERSION)
        document = {
            "schema_version": bench.TASK_SCHEMA_VERSION,
            "tasks": [
                {
                    "id": "sample",
                    "repository": {"url": "/tmp/repo", "commit": "a" * 40},
                    "mutation_patch": "diff --git a/a b/a\n",
                    "user_task": "Fix the regression.",
                    "test_command": ["python3", "-m", "unittest"],
                    "timeout_seconds": 60,
                }
            ],
        }
        self.assertEqual(bench.validate_task_document(document)[0]["id"], "sample")

    def test_secret_redaction_and_metric_parsing(self) -> None:
        secret = "sk-never-log-this"
        event = {
            "type": "turn_end",
            "toolResults": [{"content": [{"type": "text", "text": secret}]}],
            "message": {
                "usage": {"input": 100, "output": 20, "cacheRead": 10},
                "content": [
                    {"type": "toolCall", "name": "read", "arguments": {"path": "src/lib.rs"}}
                ],
            },
        }
        raw = (json.dumps(event) + "\n").encode()
        redacted = bench.redact(raw, [secret])
        self.assertNotIn(secret.encode(), redacted)
        metrics = bench.parse_pi_jsonl(redacted)
        self.assertEqual(metrics["input_tokens"], 110)
        self.assertEqual(metrics["uncached_input_tokens"], 100)
        self.assertEqual(metrics["output_tokens"], 20)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["source_opens"], 1)


if __name__ == "__main__":
    unittest.main()
