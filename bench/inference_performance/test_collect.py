#!/usr/bin/env python3
"""Unit tests for fail-closed benchmark collection."""

from __future__ import annotations

import argparse
import json
import stat
import tempfile
import unittest
from pathlib import Path

from bench.inference_performance import collect, contract


class InferencePerformanceCollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"model")
        self.tokenizer = self.root / "tokenizer.json"
        self.tokenizer.write_text("{}", encoding="ascii")
        self.hardware = self.root / "hardware.json"
        self.hardware.write_text('{"cpu":"fixture"}', encoding="ascii")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_collects_and_hashes_raw_measurement(self) -> None:
        raw = {
            "schema_version": contract.RAW_SCHEMA_VERSION,
            "model_family": "embeddinggemma",
            "workload": contract.EMBEDDING_ENCODER,
            "semantics": contract.SEMANTICS[contract.EMBEDDING_ENCODER],
            "generation_path": "encoder",
            "case_id": "fixture",
            "sample_index": 0,
            "elapsed_ns": 2_000_000,
            "input_token_ids": [1, 2, 3, 4],
            "output_token_ids": [],
            "output_limit": 0,
        }
        binary = self._producer(f"print({json.dumps(json.dumps(raw))})\n")
        records = collect.collect(self._args(binary))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["input_token_ids_sha256"], contract.token_ids_sha256([1, 2, 3, 4]))
        self.assertEqual(records[0]["tokens_per_second"], 2000.0)
        self.assertRegex(records[0]["binary_sha256"], r"^[0-9a-f]{64}$")

    def test_failed_producer_is_a_hard_failure(self) -> None:
        binary = self._producer("raise SystemExit(7)\n")
        with self.assertRaisesRegex(contract.ContractError, "failed with exit code 7"):
            collect.collect(self._args(binary, engine="llama.cpp"))

    def test_missing_llama_binary_is_a_hard_failure(self) -> None:
        with self.assertRaisesRegex(contract.ContractError, "binary file is missing"):
            collect.collect(self._args(self.root / "missing", engine="llama.cpp"))

    def test_declared_p_core_count_must_match_threads(self) -> None:
        binary = self._producer("print('{}')\n")
        args = self._args(binary)
        args.p_core_set = ["p0"]
        with self.assertRaisesRegex(contract.ContractError, "one entry per benchmark thread"):
            collect.collect(args)

    def _producer(self, body: str) -> Path:
        path = self.root / "producer"
        path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def _args(self, binary: Path, *, engine: str = "native") -> argparse.Namespace:
        return argparse.Namespace(
            run_id="fixture-run",
            platform="apple_cpu",
            engine=engine,
            model_family="embeddinggemma",
            binary=binary,
            model=self.model,
            tokenizer=self.tokenizer,
            source_root=self.root,
            hardware=self.hardware,
            threads=2,
            p_core_set=["p0", "p1"],
            device_kind="cpu",
            device_id="cpu",
            gpu_count=0,
            visible_gpu_ids=[],
            producer_args=[],
        )


if __name__ == "__main__":
    unittest.main()
