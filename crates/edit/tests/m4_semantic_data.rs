//! M4 semantic and structured-data contract tests.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use greppy_edit::certificate::{Guarantee, SelectorClass};
use greppy_edit::data::data_set;
use greppy_edit::verbs::{
    change_signature_files, rename_symbol_files, require_semantic_backend, ChangeSignatureSpec,
    RenameFileScope, SignatureDefinition, VerbOptions,
};
use greppy_edit::{EditHandle, Language, Status};

fn workspace() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

fn write(root: &Path, name: &str, content: &[u8]) -> PathBuf {
    let path = root.join(name);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&path, content).unwrap();
    path
}

fn whole_file_scope(name: &str) -> RenameFileScope {
    RenameFileScope {
        rel_path: name.into(),
        spans: vec![(0, usize::MAX)],
    }
}

#[cfg(unix)]
#[test]
fn rename_symbol_rolls_back_all_three_files_when_file_two_publish_fails() {
    let ws = workspace();
    let originals: [&[u8]; 3] = [
        b"pub fn old_name() {}\n",
        b"pub fn second() { old_name(); }\n",
        b"pub fn third() { old_name(); }\n",
    ];
    let paths = [
        write(ws.path(), "1.rs", originals[0]),
        write(ws.path(), "2.rs", originals[1]),
        write(ws.path(), "3.rs", originals[2]),
    ];
    std::fs::hard_link(&paths[1], ws.path().join("2-alias.rs")).unwrap();

    let certificate = rename_symbol_files(
        ws.path(),
        &[
            whole_file_scope("1.rs"),
            whole_file_scope("2.rs"),
            whole_file_scope("3.rs"),
        ],
        "old_name",
        "new_name",
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(certificate.status, Status::PublishFailed, "{certificate:?}");
    assert!(!certificate.published);
    assert_eq!(certificate.operations.len(), 3);
    assert!(certificate
        .operations
        .iter()
        .all(|operation| operation.guarantees.no_clobber == Guarantee::Failed));
    for (path, original) in paths.iter().zip(originals) {
        assert_eq!(std::fs::read(path).unwrap(), original);
    }
}

#[test]
fn rename_symbol_default_zero_residual_fires_after_publish() {
    let ws = workspace();
    write(ws.path(), "definition.rs", b"pub fn old_name() {}\n");
    write(ws.path(), "missed.rs", b"pub fn caller() { old_name(); }\n");

    let certificate = rename_symbol_files(
        ws.path(),
        &[whole_file_scope("definition.rs")],
        "old_name",
        "new_name",
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(certificate.status, Status::InvalidResult);
    assert_eq!(certificate.exit_code(), 13);
    assert!(certificate.published);
    assert_eq!(certificate.operations[0].residual_occurrences, Some(1));
    assert!(!certificate.operations[0].postconditions_passed);
    assert!(std::fs::read_to_string(ws.path().join("definition.rs"))
        .unwrap()
        .contains("new_name"));
}

#[test]
fn change_signature_updates_definition_and_every_call_site_transactionally() {
    let ws = workspace();
    let definition = b"def compute(a, b):\n    return a + b\n";
    write(ws.path(), "definition.py", definition);
    write(
        ws.path(),
        "one.py",
        b"from definition import compute\nvalue = compute(x, y)\n",
    );
    write(
        ws.path(),
        "two.py",
        b"from definition import compute\nother = compute(1, 2)\n",
    );
    let spec: ChangeSignatureSpec = serde_json::from_str(
        r#"{
            "old_parameters": "(a, b)",
            "new_parameters": "(b, a, timeout=30)",
            "added_arguments": {"timeout": "30"},
            "expect_call_sites": 2
        }"#,
    )
    .unwrap();

    let certificate = change_signature_files(
        ws.path(),
        &SignatureDefinition {
            rel_path: "definition.py".into(),
            range: (0, definition.len()),
        },
        &[whole_file_scope("one.py"), whole_file_scope("two.py")],
        "compute",
        &spec,
        Language::Python,
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(certificate.status, Status::Applied, "{certificate:?}");
    assert!(certificate.published);
    assert_eq!(certificate.operations.len(), 3);
    assert!(certificate
        .operations
        .iter()
        .all(|operation| operation.selector_class == SelectorClass::Semantic));
    assert_eq!(certificate.operations[0].residual_occurrences, Some(0));
    let definition = std::fs::read_to_string(ws.path().join("definition.py")).unwrap();
    let one = std::fs::read_to_string(ws.path().join("one.py")).unwrap();
    let two = std::fs::read_to_string(ws.path().join("two.py")).unwrap();
    assert!(
        definition.contains("def compute(b, a, timeout=30):"),
        "{definition}"
    );
    assert!(one.contains("compute(y, x, 30)"), "{one}");
    assert!(two.contains("compute(2, 1, 30)"), "{two}");
}

#[test]
fn change_signature_cardinality_mismatch_changes_nothing() {
    let ws = workspace();
    let definition = b"def compute(a, b):\n    return a + b\n";
    let definition_path = write(ws.path(), "definition.py", definition);
    let call = b"value = compute(x, y)\n";
    let call_path = write(ws.path(), "one.py", call);
    let spec = ChangeSignatureSpec {
        old_parameters: "(a, b)".into(),
        new_parameters: "(b, a)".into(),
        added_arguments: BTreeMap::new(),
        expect_call_sites: 2,
    };

    let certificate = change_signature_files(
        ws.path(),
        &SignatureDefinition {
            rel_path: "definition.py".into(),
            range: (0, definition.len()),
        },
        &[whole_file_scope("one.py")],
        "compute",
        &spec,
        Language::Python,
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(certificate.status, Status::Ambiguous);
    assert_eq!(certificate.exit_code(), 11);
    assert!(!certificate.published);
    assert_eq!(certificate.operations[0].target_matches, 1);
    assert_eq!(std::fs::read(definition_path).unwrap(), definition);
    assert_eq!(std::fs::read(call_path).unwrap(), call);
}

#[test]
fn change_signature_residual_fails_loudly_after_publish() {
    let ws = workspace();
    let definition = b"def compute(a, b):\n    return a + b\n";
    write(ws.path(), "definition.py", definition);
    write(ws.path(), "one.py", b"value = compute(x, y)\n");
    write(ws.path(), "missed.py", b"other = compute(1, 2)\n");
    let spec = ChangeSignatureSpec {
        old_parameters: "(a, b)".into(),
        new_parameters: "(b, a)".into(),
        added_arguments: BTreeMap::new(),
        expect_call_sites: 1,
    };

    let certificate = change_signature_files(
        ws.path(),
        &SignatureDefinition {
            rel_path: "definition.py".into(),
            range: (0, definition.len()),
        },
        &[whole_file_scope("one.py")],
        "compute",
        &spec,
        Language::Python,
        &VerbOptions::default(),
    )
    .unwrap();

    assert_eq!(certificate.status, Status::InvalidResult);
    assert_eq!(certificate.operations[0].residual_occurrences, Some(1));
    assert!(certificate.published);
    assert!(std::fs::read_to_string(ws.path().join("definition.py"))
        .unwrap()
        .contains("compute(b, a)"));
    assert!(std::fs::read_to_string(ws.path().join("one.py"))
        .unwrap()
        .contains("compute(y, x)"));
    assert!(std::fs::read_to_string(ws.path().join("missed.py"))
        .unwrap()
        .contains("compute(1, 2)"));
}

fn assert_only_target_changed(before: &[u8], after: &[u8], old: &[u8], new: &[u8]) {
    let start = before
        .windows(old.len())
        .position(|window| window == old)
        .unwrap();
    assert_eq!(&before[..start], &after[..start]);
    assert_eq!(
        &before[start + old.len()..],
        &after[start + new.len()..],
        "bytes outside the selected value changed"
    );
}

#[test]
fn data_edits_preserve_json_yaml_and_toml_bytes_outside_the_target() {
    let ws = workspace();
    let fixtures: [(&str, &[u8], &str, &[u8], &[u8]); 3] = [
        (
            "config.json",
            b"{\n  \"server\": {\n    \"port\": 9000,\n    \"host\": \"x\"\n  }\n}\n",
            "$.server.port",
            b"9000",
            b"8080",
        ),
        (
            "config.yaml",
            b"# heading\nserver:\n    port: 9000  # keep this comment\n    host: x\n",
            "$.server.port",
            b"9000",
            b"8080",
        ),
        (
            "config.toml",
            b"# heading\n[server]\nhost = \"x\"\nport = 9000 # keep this comment\n",
            "$.server.port",
            b"9000",
            b"8080",
        ),
    ];

    for (name, before, path, old, new) in fixtures {
        let file = write(ws.path(), name, before);
        let certificate = data_set(
            ws.path(),
            &file,
            path,
            "8080",
            false,
            &VerbOptions::default(),
        )
        .unwrap();
        assert_eq!(
            certificate.status,
            Status::Applied,
            "{name}: {certificate:?}"
        );
        assert!(certificate.operations[0].outside_declared_ranges_unchanged);
        let after = std::fs::read(file).unwrap();
        assert_only_target_changed(before, &after, old, new);
    }
}

#[test]
fn data_ensure_is_idempotent_and_structured_errors_are_cardinal() {
    let ws = workspace();
    let file = write(ws.path(), "config.json", b"{\n  \"port\": 8080\n}\n");
    let before = std::fs::read(&file).unwrap();
    let satisfied = data_set(
        ws.path(),
        &file,
        "$.port",
        "8080",
        true,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(satisfied.status, Status::AlreadySatisfied);
    assert_eq!(satisfied.operations[0].target_matches, 1);
    assert_eq!(std::fs::read(&file).unwrap(), before);

    let missing = data_set(
        ws.path(),
        &file,
        "$.missing",
        "1",
        false,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(missing.status, Status::NotFound);
    assert_eq!(missing.exit_code(), 10);

    let duplicate = write(ws.path(), "duplicate.json", b"{\"port\": 1, \"port\": 2}\n");
    let ambiguous = data_set(
        ws.path(),
        &duplicate,
        "$.port",
        "3",
        false,
        &VerbOptions::default(),
    )
    .unwrap();
    assert_eq!(ambiguous.status, Status::Ambiguous);
    assert_eq!(ambiguous.exit_code(), 11);
    assert_eq!(ambiguous.operations[0].target_matches, 2);
}

#[test]
fn data_edit_honors_planned_file_and_target_hashes() {
    let ws = workspace();
    let original = b"{\"port\": 9000}\n";
    let file = write(ws.path(), "config.json", original);
    let handle = EditHandle::for_range(
        ws.path(),
        Path::new("config.json"),
        original,
        0,
        original.len(),
    )
    .unwrap();
    let options = VerbOptions {
        planned_file_sha256: Some(handle.file_sha256),
        planned_target_sha256: Some(handle.target_sha256),
        planned_target_range: Some((0, original.len())),
        ..Default::default()
    };
    let concurrent = b"{\"port\": 9001}\n";
    std::fs::write(&file, concurrent).unwrap();

    let certificate = data_set(ws.path(), &file, "$.port", "8080", false, &options).unwrap();
    assert_eq!(certificate.status, Status::Stale);
    assert_eq!(certificate.exit_code(), 12);
    assert_eq!(std::fs::read(&file).unwrap(), concurrent);
}

#[test]
fn unavailable_lsp_backend_is_a_clear_invalid_specification() {
    let error = require_semantic_backend("lsp").unwrap_err();
    assert!(error.to_string().contains("unavailable"), "{error}");
    assert!(error.to_string().contains("--backend graph"), "{error}");
}
