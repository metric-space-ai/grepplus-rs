#!/usr/bin/env python3
"""Shared schema and hashing helpers for inference performance records."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "greppy.inference-performance.sample.v1"
RAW_SCHEMA_VERSION = "greppy.inference-performance.raw.v1"

PLATFORMS = ("apple_cpu", "x86_cpu", "metal", "cuda")
ENGINES = ("native", "llama.cpp")
MODEL_FAMILIES = ("qwen35_mtp", "embeddinggemma")

QWEN_PP512 = "qwen_pp512"
QWEN_TG128 = "qwen_tg128"
EMBEDDING_ENCODER = "embedding_encoder"
GREPPY_BRIEF = "greppy_brief"

GATED_WORKLOADS = (QWEN_PP512, QWEN_TG128, EMBEDDING_ENCODER)
WORKLOADS = GATED_WORKLOADS + (GREPPY_BRIEF,)

SEMANTICS = {
    QWEN_PP512: "qwen_target_prefill_exact_512_v1",
    QWEN_TG128: "qwen_greedy_generation_exact_128_v1",
    EMBEDDING_ENCODER: "embeddinggemma_encoder_forward_v1",
    GREPPY_BRIEF: "greppy_brief_production_mtp_v1",
}

HASH_FIELDS = (
    "binary_sha256",
    "model_sha256",
    "tokenizer_sha256",
    "source_sha256",
    "hardware_sha256",
    "input_token_ids_sha256",
)


class ContractError(ValueError):
    """Raised when a producer or record violates the benchmark contract."""


def canonical_json(value: Any) -> bytes:
    """Encode a value using the canonical form used by contract hashes."""

    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash a token vector as length-prefixed little-endian unsigned u32s."""

    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(token_ids)))
    for index, token_id in enumerate(token_ids):
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ContractError(f"token ID {index} is not an integer")
        if not 0 <= token_id <= 0xFFFF_FFFF:
            raise ContractError(f"token ID {index} is outside u32: {token_id}")
        digest.update(struct.pack("<I", token_id))
    return digest.hexdigest()


def hardware_sha256(hardware: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(hardware))


def source_tree_sha256(root: Path) -> str:
    """Hash tracked and non-ignored source files, including local modifications."""

    root = root.resolve()
    if not root.is_dir():
        raise ContractError(f"source root is not a directory: {root}")
    paths = _git_source_paths(root)
    if paths is None:
        paths = _fallback_source_paths(root)
    if not paths:
        raise ContractError(f"source root contains no hashable files: {root}")

    digest = hashlib.sha256()
    for relative in paths:
        encoded = relative.as_posix().encode("utf-8")
        path = root / relative
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _git_source_paths(root: Path) -> list[Path] | None:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                os.fspath(root),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    paths = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        if _source_path_allowed(relative) and (root / relative).is_file():
            paths.append(relative)
    return sorted(set(paths), key=lambda value: value.as_posix())


def _fallback_source_paths(root: Path) -> list[Path]:
    paths = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if _source_path_allowed(relative):
            paths.append(relative)
    return sorted(paths, key=lambda value: value.as_posix())


def _source_path_allowed(path: Path) -> bool:
    excluded_parts = {
        ".git",
        ".cache",
        ".idea",
        ".venv",
        "__pycache__",
        "build",
        "dev",
        "target",
    }
    if any(part in excluded_parts for part in path.parts):
        return False
    if path.suffix.lower() in {
        ".a",
        ".bin",
        ".dylib",
        ".dll",
        ".exe",
        ".gguf",
        ".gz",
        ".o",
        ".obj",
        ".onnx",
        ".pyc",
        ".safetensors",
        ".so",
        ".tar",
        ".zip",
    }:
        return False
    return True


def load_jsonl(lines: Iterable[str], source: str = "<stream>") -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ContractError(f"{source}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise ContractError(f"{source}:{line_number}: record must be a JSON object")
        records.append(value)
    if not records:
        raise ContractError(f"{source}: no JSONL records")
    return records


def dump_jsonl(records: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(record).decode("ascii") + "\n" for record in records)
