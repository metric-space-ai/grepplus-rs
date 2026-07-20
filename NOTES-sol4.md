# NOTES — M4 CLI wiring outside crate-only scope

The graph engines are complete in `crates/edit`, but the registered user-facing
`change-signature --spec sig.json [--backend graph|lsp]` path requires changes
in the forbidden CLI crate.

- `crates/cli/src/lib.rs:862-878` still declares the old `--parameters` surface
  and has no `--spec` or `--backend` argument. It must read and deserialize
  `greppy_edit::verbs::ChangeSignatureSpec`; selecting `lsp` should map
  `require_semantic_backend("lsp")` to invalid-spec exit 20 with its clear
  unavailable message.
- `crates/cli/src/lib.rs:6660-6714` still calls the compatibility
  parameters-only `change_signature` entry point and supplies candidate
  path/line checklists. It must instead build a `SignatureDefinition`, convert
  every incoming graph node to a `RenameFileScope` byte range (including its
  end line), and call `change_signature_files`; the compatibility entry point
  now fails closed rather than publishing a definition-only partial edit.
- `crates/cli/src/lib.rs:6864-6868` handles rename-symbol LSP selection through
  the older `NotImplemented` error. For the exact M4 wording and exit-20
  behavior, that arm should call `require_semantic_backend(&backend)` before
  graph resolution.

No CLI files were changed, per the `crates/edit/**` hard scope.
