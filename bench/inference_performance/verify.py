#!/usr/bin/env python3
"""Verify native-vs-llama.cpp inference performance JSONL."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .contract import (
        EMBEDDING_ENCODER,
        ENGINES,
        GATED_WORKLOADS,
        GREPPY_BRIEF,
        HASH_FIELDS,
        MODEL_FAMILIES,
        PLATFORMS,
        QWEN_PP512,
        QWEN_TG128,
        SCHEMA_VERSION,
        SEMANTICS,
        WORKLOADS,
        ContractError,
        hardware_sha256,
        load_jsonl,
        token_ids_sha256,
    )
except ImportError:  # pragma: no cover - direct script execution
    from contract import (  # type: ignore
        EMBEDDING_ENCODER,
        ENGINES,
        GATED_WORKLOADS,
        GREPPY_BRIEF,
        HASH_FIELDS,
        MODEL_FAMILIES,
        PLATFORMS,
        QWEN_PP512,
        QWEN_TG128,
        SCHEMA_VERSION,
        SEMANTICS,
        WORKLOADS,
        ContractError,
        hardware_sha256,
        load_jsonl,
        token_ids_sha256,
    )


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MIN_MEDIAN_SPEEDUP = 1.05
MIN_SAMPLE_SPEEDUP = 1.00
DEFAULT_MIN_SAMPLES = 5


class VerificationError(ContractError):
    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("\n".join(self.errors))


def verify_records(
    records: Sequence[Mapping[str, Any]],
    *,
    required_platforms: Sequence[str] = PLATFORMS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    require_greppy: bool = True,
) -> dict[str, Any]:
    """Validate records and return gate ratios; raise on any violation."""

    errors: list[str] = []
    normalized = [dict(record) for record in records]
    if not normalized:
        raise VerificationError(["result set is empty; llama baseline is mandatory"])
    if min_samples < 1:
        raise ValueError("min_samples must be positive")

    for index, record in enumerate(normalized):
        _validate_record(record, index, errors)

    run_ids = {record.get("run_id") for record in normalized}
    if len(run_ids) != 1 or None in run_ids or "" in run_ids:
        errors.append(f"records must contain exactly one non-empty run_id, found {sorted(map(str, run_ids))}")

    required = set(required_platforms)
    unknown_required = required.difference(PLATFORMS)
    if unknown_required:
        errors.append(f"unknown required platforms: {sorted(unknown_required)}")
    present_platforms = {record.get("platform") for record in normalized}
    missing_platforms = required.difference(present_platforms)
    if missing_platforms:
        errors.append(f"missing required platforms: {sorted(missing_platforms)}")

    _validate_global_hash_stability(normalized, errors)
    _validate_platform_hardware(normalized, errors)

    groups: dict[tuple[str, str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in normalized:
        platform = record.get("platform")
        workload = record.get("workload")
        model_family = record.get("model_family")
        case_id = record.get("case_id")
        engine = record.get("engine")
        if all(isinstance(value, str) for value in (platform, workload, model_family, case_id, engine)):
            groups[(platform, workload, model_family, case_id)][engine].append(record)

    report_groups = []
    for platform in sorted(required):
        for workload in GATED_WORKLOADS:
            expected_model = "embeddinggemma" if workload == EMBEDDING_ENCODER else "qwen35_mtp"
            case_keys = sorted(
                key
                for key in groups
                if key[0] == platform and key[1] == workload and key[2] == expected_model
            )
            if not case_keys:
                errors.append(f"{platform}/{workload}: missing all samples")
                continue
            for key in case_keys:
                engines = groups[key]
                missing_engines = set(ENGINES).difference(engines)
                if missing_engines:
                    errors.append(
                        f"{_group_name(key)}: missing mandatory engines {sorted(missing_engines)}"
                    )
                    continue
                _validate_engine_group(key, engines["native"], min_samples, errors)
                _validate_engine_group(key, engines["llama.cpp"], min_samples, errors)
                _validate_pair(key, engines["native"], engines["llama.cpp"], errors)
                ratio = _gate_pair(key, engines["native"], engines["llama.cpp"], errors)
                if ratio is not None:
                    report_groups.append(ratio)

        if require_greppy:
            greppy_keys = sorted(
                key
                for key in groups
                if key[0] == platform
                and key[1] == GREPPY_BRIEF
                and key[2] == "qwen35_mtp"
            )
            if not greppy_keys:
                errors.append(f"{platform}/{GREPPY_BRIEF}: missing production Greppy samples")
            for key in greppy_keys:
                engines = groups[key]
                if "native" not in engines:
                    errors.append(f"{_group_name(key)}: missing native production samples")
                    continue
                if "llama.cpp" in engines:
                    errors.append(
                        f"{_group_name(key)}: Greppy production records are native-only; "
                        "do not label a target-only llama path as production MTP"
                    )
                _validate_engine_group(key, engines["native"], min_samples, errors)

    if errors:
        raise VerificationError(errors)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "required_platforms": sorted(required),
        "minimum_median_speedup": MIN_MEDIAN_SPEEDUP,
        "minimum_sample_speedup": MIN_SAMPLE_SPEEDUP,
        "gates": report_groups,
    }


def _validate_record(record: dict[str, Any], index: int, errors: list[str]) -> None:
    label = f"record[{index}]"
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{label}: schema_version must be {SCHEMA_VERSION!r}")
    _enum(record, "platform", PLATFORMS, label, errors)
    _enum(record, "engine", ENGINES, label, errors)
    _enum(record, "model_family", MODEL_FAMILIES, label, errors)
    _enum(record, "workload", WORKLOADS, label, errors)
    _string(record, "run_id", label, errors)
    _string(record, "case_id", label, errors)
    _integer(record, "sample_index", label, errors, minimum=0)
    _integer(record, "elapsed_ns", label, errors, minimum=1)
    _integer(record, "input_tokens", label, errors, minimum=1)
    _integer(record, "threads", label, errors, minimum=1)

    workload = record.get("workload")
    engine = record.get("engine")
    model_family = record.get("model_family")
    expected_model = "embeddinggemma" if workload == EMBEDDING_ENCODER else "qwen35_mtp"
    if workload in WORKLOADS and model_family != expected_model:
        errors.append(f"{label}: {workload} requires model_family={expected_model}")
    expected_semantics = SEMANTICS.get(workload)
    if expected_semantics and record.get("semantics") != expected_semantics:
        errors.append(f"{label}: {workload} semantics must be {expected_semantics!r}")

    input_ids = record.get("input_token_ids")
    if not _token_vector(input_ids, f"{label}.input_token_ids", errors):
        input_ids = None
    elif len(input_ids) != record.get("input_tokens"):
        errors.append(
            f"{label}: input_tokens={record.get('input_tokens')} does not match "
            f"input_token_ids length {len(input_ids)}"
        )
    if input_ids is not None:
        try:
            expected_hash = token_ids_sha256(input_ids)
            if record.get("input_token_ids_sha256") != expected_hash:
                errors.append(f"{label}: input_token_ids_sha256 does not match raw token IDs")
        except ContractError as error:
            errors.append(f"{label}: {error}")

    for field in HASH_FIELDS:
        if not SHA256_RE.fullmatch(str(record.get(field, ""))):
            errors.append(f"{label}: {field} must be a lowercase SHA-256")

    hardware = record.get("hardware")
    if not isinstance(hardware, dict) or not hardware:
        errors.append(f"{label}: hardware must be a non-empty object")
    else:
        try:
            if record.get("hardware_sha256") != hardware_sha256(hardware):
                errors.append(f"{label}: hardware_sha256 does not match hardware object")
        except (TypeError, ValueError) as error:
            errors.append(f"{label}: hardware is not canonically serializable: {error}")

    p_core_set = record.get("p_core_set")
    if not isinstance(p_core_set, list) or not p_core_set:
        errors.append(f"{label}: p_core_set must be a non-empty list")
    elif len(p_core_set) != len({str(item) for item in p_core_set}):
        errors.append(f"{label}: p_core_set contains duplicates")

    device = record.get("device")
    _validate_device(device, record.get("platform"), label, errors)

    if workload == QWEN_PP512:
        if record.get("input_tokens") != 512:
            errors.append(f"{label}: PP512 must process exactly 512 token IDs")
        if record.get("output_tokens") != 0:
            errors.append(f"{label}: PP512 output_tokens must be 0")
        if record.get("generation_path") != "target_prefill":
            errors.append(f"{label}: PP512 generation_path must be target_prefill")
    elif workload == QWEN_TG128:
        if record.get("output_tokens") != 128 or record.get("output_limit") != 128:
            errors.append(f"{label}: TG128 must commit exactly 128 tokens with output_limit=128")
        expected_path = "production_mtp" if engine == "native" else "target_greedy_reference"
        if record.get("generation_path") != expected_path:
            errors.append(f"{label}: {engine} TG128 generation_path must be {expected_path}")
        output_ids = record.get("output_token_ids")
        if _token_vector(output_ids, f"{label}.output_token_ids", errors):
            if len(output_ids) != 128:
                errors.append(f"{label}: TG128 output_token_ids must contain exactly 128 IDs")
            try:
                if record.get("output_token_ids_sha256") != token_ids_sha256(output_ids):
                    errors.append(f"{label}: output_token_ids_sha256 does not match raw token IDs")
            except ContractError as error:
                errors.append(f"{label}: {error}")
        if not SHA256_RE.fullmatch(str(record.get("output_token_ids_sha256", ""))):
            errors.append(f"{label}: TG128 requires output_token_ids_sha256")
    elif workload == EMBEDDING_ENCODER:
        if record.get("output_tokens") != 0:
            errors.append(f"{label}: embedding encoder output_tokens must be 0; TG is not applicable")
        if record.get("output_limit") not in (0, None):
            errors.append(f"{label}: embedding encoder output_limit must be 0")
        if record.get("generation_path") != "encoder":
            errors.append(f"{label}: embedding generation_path must be encoder")
        attention_mask = record.get("attention_mask")
        if _token_vector(attention_mask, f"{label}.attention_mask", errors):
            if len(attention_mask) != record.get("input_tokens"):
                errors.append(f"{label}: attention_mask length must equal input_tokens")
            if any(value not in (0, 1) for value in attention_mask):
                errors.append(f"{label}: attention_mask values must be 0 or 1")
            if record.get("attention_mask_sha256") != token_ids_sha256(attention_mask):
                errors.append(f"{label}: attention_mask_sha256 does not match raw mask")
        if not SHA256_RE.fullmatch(str(record.get("attention_mask_sha256", ""))):
            errors.append(f"{label}: embedding encoder requires attention_mask_sha256")
    elif workload == GREPPY_BRIEF:
        input_tokens = record.get("input_tokens")
        if (
            isinstance(input_tokens, bool)
            or not isinstance(input_tokens, int)
            or not 100 <= input_tokens <= 500
        ):
            errors.append(f"{label}: Greppy prompt must contain 100-500 model input tokens")
        output_limit = record.get("output_limit")
        if not isinstance(output_limit, int) or isinstance(output_limit, bool) or not 1 <= output_limit <= 64:
            errors.append(f"{label}: Greppy output_limit must be <=64")
        if engine != "native" or record.get("generation_path") != "production_mtp":
            errors.append(f"{label}: Greppy workload must use the native production MTP path")

    elapsed_ns = record.get("elapsed_ns")
    if isinstance(elapsed_ns, int) and not isinstance(elapsed_ns, bool) and elapsed_ns > 0:
        rate_tokens = _tokens_for_rate(record)
        expected_rate = (
            rate_tokens * 1_000_000_000.0 / elapsed_ns
            if rate_tokens is not None
            else None
        )
        rate = record.get("tokens_per_second")
        if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not math.isfinite(rate):
            errors.append(f"{label}: tokens_per_second must be finite")
        elif expected_rate is not None and not math.isclose(
            float(rate), expected_rate, rel_tol=1e-9, abs_tol=1e-9
        ):
            errors.append(f"{label}: tokens_per_second must be derived from raw counts and elapsed_ns")
        latency = record.get("latency_ms")
        expected_latency = elapsed_ns / 1_000_000.0
        if not isinstance(latency, (int, float)) or isinstance(latency, bool) or not math.isfinite(latency):
            errors.append(f"{label}: latency_ms must be finite")
        elif not math.isclose(float(latency), expected_latency, rel_tol=1e-9, abs_tol=1e-9):
            errors.append(f"{label}: latency_ms must be derived from elapsed_ns")


def _validate_device(device: Any, platform: Any, label: str, errors: list[str]) -> None:
    if not isinstance(device, dict):
        errors.append(f"{label}: device must be an object")
        return
    expected_kind = {"apple_cpu": "cpu", "x86_cpu": "cpu", "metal": "metal", "cuda": "cuda"}.get(platform)
    if expected_kind and device.get("kind") != expected_kind:
        errors.append(f"{label}: {platform} requires device.kind={expected_kind}")
    if not isinstance(device.get("id"), str) or not device.get("id"):
        errors.append(f"{label}: device.id must be non-empty")
    gpu_count = device.get("gpu_count")
    visible = device.get("visible_gpu_ids")
    if expected_kind in {"metal", "cuda"}:
        if gpu_count != 1:
            errors.append(f"{label}: GPU benchmarks must enumerate exactly one GPU")
        if not isinstance(visible, list) or len(visible) != 1:
            errors.append(f"{label}: GPU benchmarks must expose exactly one GPU ID")
        elif str(visible[0]) != str(device.get("id")):
            errors.append(f"{label}: selected device.id must equal the sole visible GPU ID")
    else:
        if gpu_count != 0 or visible != []:
            errors.append(f"{label}: CPU benchmarks must record zero visible GPUs")


def _validate_global_hash_stability(records: Sequence[dict[str, Any]], errors: list[str]) -> None:
    for model_family in MODEL_FAMILIES:
        family = [record for record in records if record.get("model_family") == model_family]
        for field in ("model_sha256", "tokenizer_sha256"):
            values = {record.get(field) for record in family}
            if len(values) > 1:
                errors.append(f"{model_family}: all platforms and engines must use one {field}")
    for engine in ENGINES:
        for model_family in MODEL_FAMILIES:
            values = {
                record.get("source_sha256")
                for record in records
                if record.get("engine") == engine and record.get("model_family") == model_family
            }
            if len(values) > 1:
                errors.append(f"{engine}/{model_family}: source_sha256 changed across the result set")


def _validate_platform_hardware(records: Sequence[dict[str, Any]], errors: list[str]) -> None:
    for platform in PLATFORMS:
        platform_records = [record for record in records if record.get("platform") == platform]
        hashes = {record.get("hardware_sha256") for record in platform_records}
        if len(hashes) > 1:
            errors.append(f"{platform}: native and llama.cpp must run on identical hardware")


def _validate_engine_group(
    key: tuple[str, str, str, str],
    records: Sequence[dict[str, Any]],
    min_samples: int,
    errors: list[str],
) -> None:
    name = _group_name(key) + f"/{records[0].get('engine', '?')}"
    if len(records) < min_samples:
        errors.append(f"{name}: requires at least {min_samples} raw samples, found {len(records)}")
    indices = [record.get("sample_index") for record in records]
    comparable_indices = [_comparable(index) for index in indices]
    if len(comparable_indices) != len(set(comparable_indices)):
        errors.append(f"{name}: duplicate sample_index values")
    if all(isinstance(index, int) and not isinstance(index, bool) for index in indices):
        expected = list(range(len(indices)))
        if sorted(indices) != expected:
            errors.append(f"{name}: sample_index values must be contiguous from zero")
    stable_fields = (
        "binary_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "source_sha256",
        "hardware_sha256",
        "threads",
        "p_core_set",
        "device",
        "input_tokens",
        "input_token_ids_sha256",
        "attention_mask_sha256",
        "semantics",
        "generation_path",
    )
    for field in stable_fields:
        values = {_comparable(record.get(field)) for record in records}
        if len(values) > 1:
            errors.append(f"{name}: {field} changed between samples")


def _validate_pair(
    key: tuple[str, str, str, str],
    native: Sequence[dict[str, Any]],
    llama: Sequence[dict[str, Any]],
    errors: list[str],
) -> None:
    name = _group_name(key)
    parity_fields = (
        "model_sha256",
        "tokenizer_sha256",
        "hardware_sha256",
        "threads",
        "p_core_set",
        "device",
        "input_tokens",
        "input_token_ids_sha256",
        "attention_mask_sha256",
        "semantics",
        "output_limit",
    )
    for field in parity_fields:
        if _comparable(native[0].get(field)) != _comparable(llama[0].get(field)):
            errors.append(f"{name}: native and llama.cpp differ in {field}")
    native_indices = {
        record.get("sample_index")
        for record in native
        if isinstance(record.get("sample_index"), int)
        and not isinstance(record.get("sample_index"), bool)
    }
    llama_indices = {
        record.get("sample_index")
        for record in llama
        if isinstance(record.get("sample_index"), int)
        and not isinstance(record.get("sample_index"), bool)
    }
    if native_indices != llama_indices:
        errors.append(f"{name}: native and llama.cpp sample indices differ")
    if key[1] == QWEN_TG128:
        native_by_index = {record.get("sample_index"): record for record in native}
        llama_by_index = {record.get("sample_index"): record for record in llama}
        for sample_index in sorted(native_indices.intersection(llama_indices)):
            native_hash = native_by_index[sample_index].get("output_token_ids_sha256")
            llama_hash = llama_by_index[sample_index].get("output_token_ids_sha256")
            if native_hash != llama_hash:
                errors.append(
                    f"{name}/sample[{sample_index}]: committed TG128 token IDs differ; "
                    "the paths are not comparable"
                )


def _gate_pair(
    key: tuple[str, str, str, str],
    native: Sequence[dict[str, Any]],
    llama: Sequence[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any] | None:
    name = _group_name(key)
    try:
        native_rates = [float(record["tokens_per_second"]) for record in native]
        llama_rates = [float(record["tokens_per_second"]) for record in llama]
    except (KeyError, TypeError, ValueError):
        errors.append(f"{name}: cannot compute gate from tokens_per_second")
        return None
    if not native_rates or not llama_rates or min(native_rates + llama_rates) <= 0:
        errors.append(f"{name}: gate rates must be positive")
        return None
    native_median = statistics.median(native_rates)
    llama_median = statistics.median(llama_rates)
    median_ratio = native_median / llama_median
    sample_floor_ratio = min(native_rates) / llama_median
    if median_ratio < MIN_MEDIAN_SPEEDUP:
        errors.append(
            f"{name}: median speedup {median_ratio:.6f}x is below {MIN_MEDIAN_SPEEDUP:.2f}x"
        )
    if sample_floor_ratio < MIN_SAMPLE_SPEEDUP:
        errors.append(
            f"{name}: slowest native sample is {sample_floor_ratio:.6f}x the llama.cpp "
            f"median, below {MIN_SAMPLE_SPEEDUP:.2f}x"
        )
    return {
        "platform": key[0],
        "workload": key[1],
        "model_family": key[2],
        "case_id": key[3],
        "native_median_tokens_per_second": native_median,
        "llama_median_tokens_per_second": llama_median,
        "median_speedup": median_ratio,
        "slowest_native_vs_llama_median": sample_floor_ratio,
        "native_median_latency_ms": statistics.median(
            float(record["latency_ms"]) for record in native
        ),
        "llama_median_latency_ms": statistics.median(
            float(record["latency_ms"]) for record in llama
        ),
    }


def _tokens_for_rate(record: Mapping[str, Any]) -> int | None:
    field = "output_tokens" if record.get("workload") == QWEN_TG128 else "input_tokens"
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _group_name(key: tuple[str, str, str, str]) -> str:
    return "/".join(key)


def _comparable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _enum(
    record: Mapping[str, Any], field: str, allowed: Iterable[str], label: str, errors: list[str]
) -> None:
    if record.get(field) not in allowed:
        errors.append(f"{label}: {field} must be one of {sorted(allowed)}")


def _string(record: Mapping[str, Any], field: str, label: str, errors: list[str]) -> None:
    if not isinstance(record.get(field), str) or not record.get(field):
        errors.append(f"{label}: {field} must be a non-empty string")


def _integer(
    record: Mapping[str, Any],
    field: str,
    label: str,
    errors: list[str],
    *,
    minimum: int,
) -> None:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        errors.append(f"{label}: {field} must be an integer >= {minimum}")


def _token_vector(value: Any, label: str, errors: list[str]) -> bool:
    if not isinstance(value, list):
        errors.append(f"{label} must be an array")
        return False
    valid = True
    for index, token_id in enumerate(value):
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id <= 0xFFFF_FFFF
        ):
            errors.append(f"{label}[{index}] must be an unsigned u32")
            valid = False
    return valid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="combined raw JSONL sample file")
    parser.add_argument(
        "--platform",
        action="append",
        choices=PLATFORMS,
        dest="platforms",
        help="required platform; defaults to all four release platforms",
    )
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument(
        "--allow-missing-greppy",
        action="store_true",
        help="development-only: do not require production Greppy prompt samples",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with args.results.open(encoding="utf-8") as handle:
            records = load_jsonl(handle, str(args.results))
        report = verify_records(
            records,
            required_platforms=args.platforms or PLATFORMS,
            min_samples=args.min_samples,
            require_greppy=not args.allow_missing_greppy,
        )
    except (OSError, ContractError) as error:
        print(f"inference performance verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
