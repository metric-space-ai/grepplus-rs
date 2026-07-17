//! Idempotent `ensure-*` operations.
//!
//! `ensure-import`: make sure a module/name import exists. Present →
//! `already-satisfied` (no write, no error); absent → inserted at the
//! canonical position; conflicting (same name bound from a different
//! module) → `invalid-result`, nothing written.

use std::path::Path;

use crate::certificate::{Certificate, SelectorClass, SelectorEngine, Status};
use crate::txn::{PlannedOp, Snapshot};
use crate::verbs::{run_pipeline_public, single_refusal_certificate, VerbOptions};
use greppy_core::Result;
use greppy_parser::Language;

/// The import line we would write, per language.
fn import_line(language: Language, module: &str, name: Option<&str>) -> Option<String> {
    Some(match (language, name) {
        (Language::Python, Some(n)) => format!("from {module} import {n}"),
        (Language::Python, None) => format!("import {module}"),
        (Language::Rust, Some(n)) => format!("use {module}::{n};"),
        (Language::Rust, None) => format!("use {module};"),
        (Language::Go, _) => format!("import \"{module}\""),
        (Language::TypeScript { .. } | Language::JavaScript, Some(n)) => {
            format!("import {{ {n} }} from \"{module}\";")
        }
        (Language::TypeScript { .. } | Language::JavaScript, None) => {
            format!("import \"{module}\";")
        }
        _ => return None,
    })
}

/// Node kinds that represent import statements per grammar.
fn import_kinds(language: Language) -> &'static [&'static str] {
    match language {
        Language::Python => &["import_statement", "import_from_statement"],
        Language::Rust => &["use_declaration"],
        Language::Go => &["import_declaration"],
        Language::TypeScript { .. } | Language::JavaScript => &["import_statement"],
        _ => &[],
    }
}

struct ImportScan {
    /// byte offset AFTER the last top-level import (insertion point), or the
    /// canonical start-of-file position when no import exists
    insert_at: usize,
    /// an import binding `name` (or bare `module`) already exists
    satisfied: bool,
    /// `name` is already bound from a DIFFERENT module
    conflict: Option<String>,
}

fn scan_imports(
    language: Language,
    content: &[u8],
    module: &str,
    name: Option<&str>,
) -> Option<ImportScan> {
    let tree = greppy_parser::parse(language, content).ok()?;
    let kinds = import_kinds(language);
    let mut insert_at = 0usize;
    let mut satisfied = false;
    let mut conflict = None;
    let root = tree.root_node();
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if !kinds.contains(&node.kind()) {
            continue;
        }
        let text = String::from_utf8_lossy(&content[node.start_byte()..node.end_byte()]);
        let mut end = node.end_byte();
        if content.get(end) == Some(&b'\n') {
            end += 1;
        }
        insert_at = end;
        let mentions_module = text.contains(module);
        match name {
            Some(n) => {
                let mentions_name = text
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .any(|tok| tok == n);
                if mentions_module && mentions_name {
                    satisfied = true;
                } else if mentions_name && !mentions_module {
                    conflict = Some(text.trim().to_string());
                }
            }
            None => {
                if mentions_module {
                    satisfied = true;
                }
            }
        }
    }
    Some(ImportScan {
        insert_at,
        satisfied,
        conflict,
    })
}

/// `greppy edit ensure-import --file F --module M [--name N]`.
pub fn ensure_import(
    workspace_root: &Path,
    file: &Path,
    module: &str,
    name: Option<&str>,
    options: &VerbOptions,
) -> Result<Certificate> {
    let snapshot = Snapshot::read(file)?;
    let language = greppy_parser::language_for_path(file);
    let Some(line) = import_line(language, module, name) else {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::NotFound,
            options,
        ));
    };
    let Some(scan) = scan_imports(language, &snapshot.content, module, name) else {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::NotFound,
            options,
        ));
    };
    if scan.satisfied {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::AlreadySatisfied,
            options,
        ));
    }
    if scan.conflict.is_some() {
        return Ok(single_refusal_certificate(
            workspace_root,
            &snapshot,
            SelectorEngine::TreeSitter,
            SelectorClass::Structural,
            Status::InvalidResult,
            options,
        ));
    }
    let mut block = line.into_bytes();
    block.push(b'\n');
    let ops = vec![PlannedOp {
        id: "ensure-import".into(),
        range: (scan.insert_at, scan.insert_at),
        replacement: block,
    }];
    run_pipeline_public(
        workspace_root,
        snapshot,
        ops,
        SelectorEngine::TreeSitter,
        SelectorClass::Structural,
        Some(language),
        options,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn python_insert_after_last_import() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"import os\n\ndef run():\n    pass\n").unwrap();
        let cert = ensure_import(
            dir.path(),
            &f,
            "auth.validators",
            Some("validate"),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(
            out.starts_with("import os\nfrom auth.validators import validate\n"),
            "{out}"
        );
    }

    #[test]
    fn second_run_already_satisfied() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"from auth.validators import validate\n").unwrap();
        let cert = ensure_import(
            dir.path(),
            &f,
            "auth.validators",
            Some("validate"),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::AlreadySatisfied);
        assert_eq!(cert.exit_code(), 0);
    }

    #[test]
    fn conflicting_binding_refuses() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"from other.module import validate\n").unwrap();
        let cert = ensure_import(
            dir.path(),
            &f,
            "auth.validators",
            Some("validate"),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::InvalidResult);
        assert!(std::fs::read_to_string(&f)
            .unwrap()
            .starts_with("from other.module"));
    }

    #[test]
    fn rust_use_insertion() {
        let dir = ws();
        let f = dir.path().join("m.rs");
        std::fs::write(&f, b"use std::io::Write;\n\nfn main() {}\n").unwrap();
        let cert = ensure_import(
            dir.path(),
            &f,
            "crate::auth",
            Some("validate"),
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(cert.status, Status::Applied);
        let out = std::fs::read_to_string(&f).unwrap();
        assert!(out.contains("use crate::auth::validate;\n"), "{out}");
        assert_eq!(cert.operations[0].syntax.new_errors, 0);
    }

    #[test]
    fn file_without_imports_inserts_at_top() {
        let dir = ws();
        let f = dir.path().join("m.py");
        std::fs::write(&f, b"def run():\n    pass\n").unwrap();
        let cert = ensure_import(dir.path(), &f, "os", None, &VerbOptions::default()).unwrap();
        assert_eq!(cert.status, Status::Applied);
        assert!(std::fs::read_to_string(&f)
            .unwrap()
            .starts_with("import os\n"));
    }
}
