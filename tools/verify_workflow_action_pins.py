#!/usr/bin/env python3
"""Reject mutable third-party GitHub Action references."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


USES = re.compile(r"^\s*(?:-\s*)?uses:\s*([^\s#]+)")
PINNED_ACTION = re.compile(r"^[^/@\s]+/[^@\s]+@[0-9a-f]{40}$")
RUST_TOOLCHAIN_ACTION = "dtolnay/rust-toolchain@"
RUST_TOOLCHAIN_CHANNEL = re.compile(r'^channel\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


def _rust_toolchain_channel() -> str:
    document = pathlib.Path("rust-toolchain.toml").read_text(encoding="utf-8")
    match = RUST_TOOLCHAIN_CHANNEL.search(document)
    if match is None:
        raise ValueError("rust-toolchain.toml has no quoted channel")
    return match.group(1)


def _step_lines(lines: list[str], uses_index: int) -> list[str]:
    uses_indent = len(lines[uses_index]) - len(lines[uses_index].lstrip())
    end = len(lines)
    for index in range(uses_index + 1, len(lines)):
        stripped = lines[index].lstrip()
        indent = len(lines[index]) - len(stripped)
        if stripped.startswith("- ") and indent < uses_indent:
            end = index
            break
    return lines[uses_index:end]


def verify(paths: list[pathlib.Path]) -> list[str]:
    errors: list[str] = []
    try:
        toolchain = _rust_toolchain_channel()
    except (OSError, UnicodeError, ValueError) as exc:
        return [f"rust-toolchain.toml: cannot determine pinned channel: {exc}"]
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path}: cannot read workflow: {exc}")
            continue
        for line_number, line in enumerate(lines, 1):
            match = USES.match(line)
            if match is None:
                continue
            reference = match.group(1)
            if reference.startswith("./"):
                continue
            if not PINNED_ACTION.fullmatch(reference):
                errors.append(
                    f"{path}:{line_number}: external action is not pinned to a "
                    f"40-character commit SHA: {reference}"
                )
            if reference.startswith(RUST_TOOLCHAIN_ACTION):
                expected = re.compile(
                    rf"^\s*toolchain:\s*['\"]?{re.escape(toolchain)}['\"]?\s*$"
                )
                if not any(expected.match(step_line) for step_line in _step_lines(lines, line_number - 1)):
                    errors.append(
                        f"{path}:{line_number}: immutable dtolnay/rust-toolchain pin must "
                        f"declare toolchain {toolchain!r} from rust-toolchain.toml"
                    )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        default=sorted(pathlib.Path(".github/workflows").glob("*.yml")),
    )
    args = parser.parse_args(argv)
    errors = verify(args.paths)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        return 1
    print(f"verified immutable Action pins in {len(args.paths)} workflows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
