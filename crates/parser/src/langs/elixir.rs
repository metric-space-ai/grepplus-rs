//! Elixir — the first language onboarded via the parallel-safe registry
//! (`crate::registry`). This whole file is the entire surface: it declares the
//! spec + queries + grammar and self-registers with `inventory::submit!`. No
//! shared file is edited (build.rs discovers this module automatically).
//!
//! Status: **experimental**. Elixir's tree-sitter grammar models `def` /
//! `defp` / `defmodule` as generic `call` nodes (there is no
//! `function_definition` kind), so definition/call extraction is
//! predicate-based and less precise than a language with distinct def nodes.
//! It is intentionally NOT claimed as `supported` (no verification corpus yet).

use crate::registry::LangDef;
use crate::spec::{CallSpec, DefRule, DocStyle, ImportStrategy, LangSpec, NameStrategy};

/// `def`/`defp`/`defmacro` become Function definitions. The captured `@def`
/// node kind is `call` (Elixir has no distinct def kind), so the DefRule keys
/// on `"call"`; only the def-keyword calls reach it because the DEFINITIONS
/// query filters by keyword.
static ELIXIR_SPEC: LangSpec = LangSpec {
    name: NameStrategy::Capture,
    defs: &[DefRule::func("call")],
    owner_kinds: &[],
    calls: CallSpec { skip_callees: &[] },
    // The bespoke Elixir extractor consumes IMPORTS below directly. The
    // strategy remains inert on this path, but keeping the query registered
    // makes the provider's declared capabilities match its extraction output.
    imports: ImportStrategy::Bash,
    docs: DocStyle::LineHashComment,
};

/// `def add(...)` parses as `(call (identifier "def") (arguments (call
/// (identifier "add") …)))`; capture the inner identifier as the name.
const DEFINITIONS: &str = r#"
    (call
      (identifier) @_kw
      (arguments (call (identifier) @name))
      (#any-of? @_kw "def" "defp" "defmacro" "defmacrop")) @def
"#;

/// Bare `foo(...)` calls, excluding the def/module/import keyword-calls.
const CALLS: &str = r#"
    (call
      (identifier) @callee
      (arguments)
      (#not-any-of? @callee
        "def" "defp" "defmacro" "defmacrop" "defmodule"
        "import" "alias" "require" "use"))
"#;

/// Module dependencies declared through Elixir's alias-bearing macros. The
/// bespoke extractor consumes the `@imported` alias and emits one IMPORTS edge
/// keyed by its final dotted segment.
pub(crate) const IMPORTS: &str = r#"
    (call
      (identifier) @_kw
      (arguments (alias) @imported)
      (#any-of? @_kw "alias" "import" "require")) @import
"#;

inventory::submit! {
    LangDef {
        name: "elixir",
        extensions: &["ex", "exs"],
        filenames: &[],
        grammar: || tree_sitter_elixir::LANGUAGE.into(),
        spec: &ELIXIR_SPEC,
        def_query: DEFINITIONS,
        call_query: CALLS,
        import_query: IMPORTS,
    }
}
