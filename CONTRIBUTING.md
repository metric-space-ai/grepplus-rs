# Contributing to Greppy

Greppy accepts focused bug fixes, tests, documentation, parser improvements,
and performance work that preserve its evidence and release contracts.

## Before opening a change

- Use a GitHub issue for behavioral changes that affect CLI output, storage,
  inference, model assets, or supported platforms.
- Report security issues through GitHub private vulnerability reporting, not a
  public issue. See [SECURITY.md](SECURITY.md).
- Keep unrelated refactors out of a functional change.
- Do not add model weights, generated kernels, or third-party source without a
  documented license, source URL, immutable revision, and digest.

## Development setup

The repository pins its Rust toolchain in `rust-toolchain.toml`. Git LFS is
required for real embedded-model builds.

```bash
git lfs install
git lfs pull
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all --check
```

Fast CI uses fixture assets for most tests. Changes to inference, packaging, or
the product binary must also pass the real-asset and target-hardware jobs. Keep
temporary models, benchmark output, copied binaries, and external repositories
under the ignored `dev/` directory.

## Contract-sensitive changes

- Passthrough changes must retain byte-exact stdout, stderr, and exit status
  parity with the real system `grep` and must not create Greppy state.
- Structured-output changes require text and JSON contract tests, exact source
  spans, source-faithful signatures, freshness failure, and working expansion
  handles.
- Prompt changes require a prompt-version bump and quality-gate updates.
- Inference changes require token/logit parity and matched native-versus-
  llama.cpp measurements on affected CPU/GPU targets.
- Cache, daemon, or backend-loader changes require fault, ownership,
  permissions, and concurrency tests.

## Pull requests

Open pull requests against `main`. Branch protection requires the platform CI,
real embedded-asset tests, Windows daemon tests, CodeQL, an approving review,
and resolved conversations. Describe the behavioral contract, verification,
platforms tested, and any remaining risk.

By submitting a contribution, you agree that it is licensed under the
repository's MIT license and that you have the right to contribute it.
