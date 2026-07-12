# Support Policy

## Release targets

The `v0.2.x` release gate covers:

| Platform | Inference backend | Daemon transport |
|---|---|---|
| macOS Apple Silicon | CPU and Metal | Unix-domain socket |
| Linux x86_64 | CPU and NVIDIA CUDA | Unix-domain socket |
| Windows x86_64 | CPU | named pipe with user ACL |

Windows CUDA, macOS Intel, and Linux ARM64 are not release-gated for `v0.2.x`.
An unsupported accelerator must not prevent CPU operation in automatic mode.
An explicitly selected unavailable backend fails with a diagnostic instead of
silently switching devices.

## Language support

Rust, Python, Java, JavaScript, TypeScript, and Go are production-certified by
language fixtures and real-repository acceptance tests. Other bundled parsers
can provide indexing and text/definition search, but graph completeness is not
claimed until that language receives the same test coverage.

Static analysis can miss reflection, runtime dependency injection, generated
code, macro expansion, monkeypatching, and dynamic dispatch. Greppy fails closed
when indexed source evidence is stale; verify proposed changes with the
language toolchain and test suite.

## Getting help

Open a GitHub issue with:

- `greppy --version` and the exact release checksum;
- operating system, CPU, GPU, and driver version;
- `greppy doctor --json` with private paths redacted;
- the command, exit code, and minimal reproducible repository when possible.

Use GitHub's private vulnerability-reporting flow for security issues. See
[`SECURITY.md`](SECURITY.md).

## Operational defaults

Embedding and summary models remain resident for 300 idle seconds. Their daemon
processes exit after 1800 idle seconds. Workspace cache entries use a 14-day
default TTL plus an independent size quota. These values can be inspected with
`greppy cache status --json`.
