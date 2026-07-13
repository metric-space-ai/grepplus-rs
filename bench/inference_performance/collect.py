#!/usr/bin/env python3
"""Run one benchmark producer and enrich its raw JSONL with provenance hashes."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .contract import (
        ENGINES,
        MODEL_FAMILIES,
        PLATFORMS,
        RAW_SCHEMA_VERSION,
        SCHEMA_VERSION,
        WORKLOADS,
        ContractError,
        canonical_json,
        hardware_sha256,
        load_jsonl,
        sha256_file,
        source_tree_sha256,
        token_ids_sha256,
    )
except ImportError:  # pragma: no cover - direct script execution
    from contract import (  # type: ignore
        ENGINES,
        MODEL_FAMILIES,
        PLATFORMS,
        RAW_SCHEMA_VERSION,
        SCHEMA_VERSION,
        WORKLOADS,
        ContractError,
        canonical_json,
        hardware_sha256,
        load_jsonl,
        sha256_file,
        source_tree_sha256,
        token_ids_sha256,
    )


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.threads < 1:
        raise ContractError("threads must be positive")
    if len(args.p_core_set) != args.threads:
        raise ContractError("p_core_set must contain exactly one entry per benchmark thread")
    binary = args.binary.resolve()
    model = args.model.resolve()
    tokenizer = args.tokenizer.resolve()
    source_root = args.source_root.resolve()
    hardware_path = args.hardware.resolve()
    for label, path in (("binary", binary), ("model", model), ("tokenizer", tokenizer)):
        if not path.is_file():
            raise ContractError(f"{label} file is missing: {path}")
    if not source_root.is_dir():
        raise ContractError(f"source root is missing: {source_root}")
    if not hardware_path.is_file():
        raise ContractError(f"hardware JSON is missing: {hardware_path}")

    try:
        hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read hardware JSON: {error}") from error
    if not isinstance(hardware, dict) or not hardware:
        raise ContractError("hardware JSON must contain one non-empty object")

    device = _device_from_args(args)
    command = [os.fspath(binary), *args.producer_args]
    env = os.environ.copy()
    if args.device_kind == "cuda":
        visible = str(args.visible_gpu_ids[0])
        existing = env.get("CUDA_VISIBLE_DEVICES")
        if existing is not None and existing.strip() != visible:
            raise ContractError(
                "CUDA_VISIBLE_DEVICES conflicts with --visible-gpu-id; refusing an ambiguous GPU run"
            )
        env["CUDA_VISIBLE_DEVICES"] = visible

    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
    except OSError as error:
        raise ContractError(f"cannot execute benchmark producer {binary}: {error}") from error
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise ContractError(
            f"benchmark producer failed with exit code {completed.returncode}; "
            "a failed llama.cpp baseline is a hard failure"
        )
    raw_records = load_jsonl(completed.stdout.splitlines(), os.fspath(binary))

    provenance = {
        "binary_sha256": sha256_file(binary),
        "model_sha256": sha256_file(model),
        "tokenizer_sha256": sha256_file(tokenizer),
        "source_sha256": source_tree_sha256(source_root),
        "hardware_sha256": hardware_sha256(hardware),
    }
    records = []
    for index, raw in enumerate(raw_records):
        records.append(
            _enrich_raw_record(
                raw,
                raw_index=index,
                args=args,
                device=device,
                hardware=hardware,
                provenance=provenance,
            )
        )
    return records


def _device_from_args(args: argparse.Namespace) -> dict[str, Any]:
    visible = args.visible_gpu_ids
    if args.device_kind == "cpu":
        if args.gpu_count != 0 or visible:
            raise ContractError("CPU collection requires --gpu-count 0 and no --visible-gpu-id")
    elif args.gpu_count != 1 or len(visible) != 1:
        raise ContractError("Metal/CUDA collection requires exactly one visible GPU")
    if args.device_kind != "cpu" and str(visible[0]) != args.device_id:
        raise ContractError("--device-id must equal the sole --visible-gpu-id")
    return {
        "kind": args.device_kind,
        "id": args.device_id,
        "gpu_count": args.gpu_count,
        "visible_gpu_ids": visible,
    }


def _enrich_raw_record(
    raw: Mapping[str, Any],
    *,
    raw_index: int,
    args: argparse.Namespace,
    device: Mapping[str, Any],
    hardware: Mapping[str, Any],
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    label = f"raw record[{raw_index}]"
    if raw.get("schema_version") != RAW_SCHEMA_VERSION:
        raise ContractError(f"{label}: schema_version must be {RAW_SCHEMA_VERSION!r}")
    if raw.get("model_family") != args.model_family:
        raise ContractError(f"{label}: producer model_family does not match collection arguments")
    if "threads" in raw and raw.get("threads") != args.threads:
        raise ContractError(f"{label}: producer threads do not match collection arguments")
    if "device" in raw and raw.get("device") != args.device_kind:
        raise ContractError(f"{label}: producer device does not match collection arguments")
    if raw.get("workload") not in WORKLOADS:
        raise ContractError(f"{label}: unknown workload {raw.get('workload')!r}")
    if not isinstance(raw.get("case_id"), str) or not raw.get("case_id"):
        raise ContractError(f"{label}: case_id must be non-empty")
    sample_index = raw.get("sample_index")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int) or sample_index < 0:
        raise ContractError(f"{label}: sample_index must be a non-negative integer")
    elapsed_ns = raw.get("elapsed_ns")
    if isinstance(elapsed_ns, bool) or not isinstance(elapsed_ns, int) or elapsed_ns <= 0:
        raise ContractError(f"{label}: elapsed_ns must be a positive integer")

    input_ids = _token_vector(raw.get("input_token_ids"), f"{label}.input_token_ids")
    output_ids = _token_vector(raw.get("output_token_ids", []), f"{label}.output_token_ids")
    attention_mask = _token_vector(raw.get("attention_mask", []), f"{label}.attention_mask")
    input_tokens = len(input_ids)
    output_tokens = len(output_ids)
    workload = str(raw["workload"])
    tokens_for_rate = output_tokens if workload == "qwen_tg128" else input_tokens
    if tokens_for_rate <= 0:
        raise ContractError(f"{label}: cannot derive throughput from zero tokens")

    value = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "platform": args.platform,
        "engine": args.engine,
        "model_family": args.model_family,
        "workload": workload,
        "semantics": raw.get("semantics"),
        "generation_path": raw.get("generation_path"),
        "case_id": raw["case_id"],
        "sample_index": sample_index,
        "elapsed_ns": elapsed_ns,
        "latency_ms": elapsed_ns / 1_000_000.0,
        "tokens_per_second": tokens_for_rate * 1_000_000_000.0 / elapsed_ns,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "output_limit": raw.get("output_limit", 0),
        "input_token_ids": input_ids,
        "output_token_ids": output_ids,
        "attention_mask": attention_mask,
        "input_token_ids_sha256": token_ids_sha256(input_ids),
        "output_token_ids_sha256": token_ids_sha256(output_ids) if output_ids else None,
        "attention_mask_sha256": token_ids_sha256(attention_mask) if attention_mask else None,
        "threads": args.threads,
        "p_core_set": args.p_core_set,
        "device": dict(device),
        "hardware": dict(hardware),
        **provenance,
    }
    try:
        canonical_json(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{label}: enriched record is not canonical JSON: {error}") from error
    if not math.isfinite(value["tokens_per_second"]):
        raise ContractError(f"{label}: non-finite throughput")
    return value


def _token_vector(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ContractError(f"{label} must be an array")
    result = []
    for index, token_id in enumerate(value):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id <= 0xFFFF_FFFF
        ):
            raise ContractError(f"{label}[{index}] must be an unsigned u32")
        result.append(token_id)
    return result


def _csv_items(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    if len(items) != len(set(items)):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--platform", required=True, choices=PLATFORMS)
    parser.add_argument("--engine", required=True, choices=ENGINES)
    parser.add_argument("--model-family", required=True, choices=MODEL_FAMILIES)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--hardware", required=True, type=Path)
    parser.add_argument("--threads", required=True, type=int)
    parser.add_argument("--p-core-set", required=True, type=_csv_items)
    parser.add_argument("--device-kind", required=True, choices=("cpu", "metal", "cuda"))
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--gpu-count", required=True, type=int)
    parser.add_argument(
        "--visible-gpu-id",
        action="append",
        default=[],
        dest="visible_gpu_ids",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "producer_args",
        nargs=argparse.REMAINDER,
        help="arguments passed to the producer binary after --",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.producer_args[:1] == ["--"]:
        args.producer_args = args.producer_args[1:]
    if args.threads < 1:
        print("collection failed: --threads must be positive", file=sys.stderr)
        return 2
    try:
        records = collect(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="ascii", newline="\n") as handle:
            for record in records:
                handle.write(canonical_json(record).decode("ascii") + "\n")
    except (OSError, ContractError) as error:
        print(f"collection failed: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(records)} raw samples to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
