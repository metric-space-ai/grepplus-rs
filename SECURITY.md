# Security Policy

## Supported versions

Security fixes are provided for the newest published Greppy release. The
unreleased `main` branch is tested continuously but is not a supported release
channel. Pin production installations to an immutable tag and verify the
published checksum and build provenance.

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's **Security > Report a
vulnerability** flow for this repository. Do not open a public issue for a
suspected vulnerability that exposes repository contents, cache paths, local
privilege boundaries, model/backend loading, or daemon transport.

Include the affected version and platform, reproduction steps, expected impact,
and whether the issue requires local access. Maintainers will acknowledge a
complete report within five business days and coordinate disclosure after a fix
is available.

## Security boundaries

- Greppy processes local source code and stores indexed source spans in a local
  SQLite cache. It does not send code or model prompts to a network service.
- Ordinary grep passthrough invokes the real system `grep` and must not open an
  index, load a model, or mutate a Greppy cache.
- Structured commands treat source and graph evidence as authoritative. Qwen
  summaries are untrusted navigation hints and may be omitted on any inference
  or validation failure.
- Model and embedded CUDA artifacts are extracted only into private,
  content-addressed cache paths and are verified before loading. External
  backend-library overrides are not supported by release builds.
- Greppy does not install drivers, toolkits, updates, or other software. Release
  upgrades are explicit and use signed/checksummed artifacts.

## Sensitive repositories

Set `GREPPY_STORE_DIR` to an encrypted or ephemeral user-private location when
repository contents require additional at-rest protection. Use `greppy cache
status --json` to audit stored paths and `greppy cache clear --root DIR --yes`
or `greppy cache clear --all --yes` to remove managed data.
