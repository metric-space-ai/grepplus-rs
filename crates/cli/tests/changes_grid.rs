//! End-to-end grid for `greppy changes` symbol and graph impact reporting.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

static COUNTER: AtomicU32 = AtomicU32::new(0);
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_repo(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-changes-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join("tests")).unwrap();
    std::fs::write(
        repo.join("Cargo.toml"),
        "[package]\nname = \"changes-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        r#"pub fn target(value: u32) -> u32 {
    value + 1
}

pub fn removed() -> u32 {
    9
}

pub fn wrapper() -> u32 {
    target(1)
}

pub fn base_only() -> u32 {
    1
}

#[test]
fn target_test() {
    assert_eq!(wrapper(), 2);
}
"#,
    )
    .unwrap();
    std::fs::write(
        repo.join("tests/integration.rs"),
        r#"use changes_fixture::wrapper;

#[test]
fn target_test() {
    assert_eq!(wrapper(), 2);
}
"#,
    )
    .unwrap();
    std::fs::write(
        repo.join("tests/impact.rs"),
        r#"pub fn test_target(value: u32) -> u32 {
    value + 1
}

pub fn impact_test() -> u32 {
    test_target(1)
}
"#,
    )
    .unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.email", "changes@example.invalid"]);
    git(&repo, &["config", "user.name", "Changes Fixture"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-qm", "initial"]);
    (repo, base.join("store"))
}

fn git(repo: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .expect("run git");
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn index(repo: &Path, store: &Path) {
    let (code, stdout, stderr) = run(repo, store, &["index", "."]);
    assert_eq!(code, 0, "index failed\nstdout={stdout}\nstderr={stderr}");
}

fn apply_worktree_change(repo: &Path) {
    std::fs::write(
        repo.join("src/lib.rs"),
        r#"pub fn target(value: u64, extra: u64) -> u64 {
    value + extra
}

pub fn added() -> u32 {
    7
}

pub fn wrapper() -> u32 {
    target(1, 1) as u32
}

pub fn base_only() -> u32 {
    1
}

#[test]
fn target_test() {
    assert_eq!(wrapper(), 2);
}
"#,
    )
    .unwrap();
    std::fs::write(
        repo.join("tests/impact.rs"),
        r#"pub fn test_target(value: u64) -> u64 {
    value + 2
}

pub fn impact_test() -> u64 {
    test_target(1)
}
"#,
    )
    .unwrap();
    std::fs::write(repo.join("tests/manual.case"), "manual verification\n").unwrap();
}

fn find_definition<'a>(file: &'a Value, group: &str, name: &str) -> &'a Value {
    file["definitions"][group]
        .as_array()
        .unwrap()
        .iter()
        .find(|definition| definition["name"] == name)
        .unwrap_or_else(|| panic!("missing {group} definition {name}: {file}"))
}

#[test]
fn groups_symbols_detects_signatures_and_reports_callers_and_tests() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let (repo, store) = fresh_repo("impact");
    index(&repo, &store);
    apply_worktree_change(&repo);
    git(&repo, &["add", "src/lib.rs"]);
    let status_before = git(&repo, &["status", "--porcelain=v1"]);

    let (code, stdout, stderr) = run(&repo, &store, &["changes", "--json"]);
    assert_eq!(code, 0, "changes failed\nstdout={stdout}\nstderr={stderr}");
    assert_eq!(
        git(&repo, &["status", "--porcelain=v1"]),
        status_before,
        "changes must not mutate VCS state"
    );
    let value: Value = serde_json::from_str(&stdout).expect("valid changes JSON");
    let lib = value["files"]
        .as_array()
        .unwrap()
        .iter()
        .find(|file| file["path"] == "src/lib.rs")
        .expect("src/lib.rs summary");

    let target = find_definition(lib, "modified", "target");
    assert_eq!(target["signature_changed"], true);
    assert!(target["before_signature"]
        .as_str()
        .unwrap()
        .contains("value: u32"));
    assert!(target["after_signature"]
        .as_str()
        .unwrap()
        .contains("extra: u64"));
    find_definition(lib, "added", "added");
    find_definition(lib, "deleted", "removed");

    let callsites = value["callsite_impact"].as_array().unwrap();
    assert!(callsites.iter().any(|impact| {
        impact["changed_symbol"]
            .as_str()
            .unwrap()
            .contains("target")
            && impact["caller"].as_str().unwrap().contains("wrapper")
    }));
    let known = value["tests"]["known_impacted"].as_array().unwrap();
    assert!(known.iter().any(|impact| {
        impact["test_symbol"]
            .as_str()
            .unwrap()
            .contains("impact_test")
            && impact["hops"].as_u64().unwrap() <= 2
    }));
    let unknown = value["tests"]["unknown_or_unindexed"].as_array().unwrap();
    assert!(unknown
        .iter()
        .any(|impact| impact["path"] == "tests/manual.case"));
}

#[test]
fn base_revision_includes_committed_and_worktree_deltas() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let (repo, store) = fresh_repo("base");
    let initial = git(&repo, &["rev-parse", "HEAD"]);
    let original = std::fs::read_to_string(repo.join("src/lib.rs")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        original.replace(
            "pub fn base_only() -> u32 {\n    1",
            "pub fn base_only() -> u32 {\n    2",
        ),
    )
    .unwrap();
    git(&repo, &["add", "src/lib.rs"]);
    git(&repo, &["commit", "-qm", "committed delta"]);
    index(&repo, &store);
    let committed = std::fs::read_to_string(repo.join("src/lib.rs")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        committed.replace("value + 1", "value + 2"),
    )
    .unwrap();

    let (code, default_stdout, stderr) = run(&repo, &store, &["changes", "--json"]);
    assert_eq!(code, 0, "default changes failed: {stderr}");
    let default_value: Value = serde_json::from_str(&default_stdout).unwrap();
    let default_lib = &default_value["files"][0];
    find_definition(default_lib, "modified", "target");
    assert!(default_lib["definitions"]["modified"]
        .as_array()
        .unwrap()
        .iter()
        .all(|definition| definition["name"] != "base_only"));

    let (code, base_stdout, stderr) =
        run(&repo, &store, &["changes", "--base", &initial, "--json"]);
    assert_eq!(code, 0, "base changes failed: {stderr}");
    let base_value: Value = serde_json::from_str(&base_stdout).unwrap();
    let base_lib = &base_value["files"][0];
    find_definition(base_lib, "modified", "target");
    find_definition(base_lib, "modified", "base_only");
    assert_eq!(base_value["base"]["requested"], initial);
}

#[test]
fn json_fields_are_stable_and_certification_is_omitted_without_history() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let (repo, store) = fresh_repo("shape");
    apply_worktree_change(&repo);
    let (code, stdout, stderr) = run(&repo, &store, &["changes", "--json"]);
    assert_eq!(code, 0, "changes failed\nstdout={stdout}\nstderr={stderr}");
    let value: Value = serde_json::from_str(&stdout).unwrap();
    let keys = value
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    assert_eq!(
        keys,
        vec![
            "base",
            "callsite_impact",
            "files",
            "schema_version",
            "tests"
        ]
    );
    assert_eq!(value["schema_version"], "greppy.changes.v1");
    assert!(value.get("certification").is_none());
    assert!(value["tests"]["known_impacted"].is_array());
    assert!(value["tests"]["unknown_or_unindexed"].is_array());
    assert_eq!(value["tests"]["graph_depth"], 2);
}

#[test]
fn text_output_keeps_known_and_unknown_tests_strictly_separate() {
    let _guard = TEST_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let (repo, store) = fresh_repo("text");
    index(&repo, &store);
    apply_worktree_change(&repo);
    let (code, stdout, stderr) = run(&repo, &store, &["changes"]);
    assert_eq!(code, 0, "changes failed\nstdout={stdout}\nstderr={stderr}");
    let known = stdout.find("  known_impacted:").expect("known heading");
    let unknown = stdout
        .find("  unknown_or_unindexed:")
        .expect("unknown heading");
    assert!(
        known < unknown,
        "test classifications must be separate: {stdout}"
    );
    assert!(stdout[known..unknown].contains("impact_test"));
    assert!(stdout[unknown..].contains("tests/manual.case"));
}
