//! End-to-end smoke coverage for the M4 semantic and structured-data CLI verbs.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_workspace(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-edit-m4-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    (repo, base.join("store"))
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

#[test]
fn change_signature_spec_rewrites_definition_and_graph_call_sites() {
    let (repo, store) = fresh_workspace("signature");
    std::fs::write(
        repo.join("lib.rs"),
        "pub fn compute(a: i32, b: i32) -> i32 {\n    a + b\n}\n\npub fn caller() -> i32 {\n    compute(1, 2)\n}\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("sig.json"),
        r#"{
  "old_parameters": "(a: i32, b: i32)",
  "new_parameters": "(b: i32, a: i32)",
  "expect_call_sites": 1
}
"#,
    )
    .unwrap();

    let repo_arg = repo.to_string_lossy().into_owned();
    let (index_code, _, index_stderr) =
        run(&repo, &store, &["--root", &repo_arg, "index", &repo_arg]);
    assert_eq!(index_code, 0, "index failed: {index_stderr}");

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &[
            "--root",
            &repo_arg,
            "edit",
            "change-signature",
            "--symbol",
            "compute",
            "--spec",
            "sig.json",
        ],
    );
    assert_eq!(code, 0, "change-signature failed: {stderr}\n{stdout}");
    assert!(stdout.contains("\"status\": \"applied\""), "{stdout}");
    let changed = std::fs::read_to_string(repo.join("lib.rs")).unwrap();
    assert!(changed.contains("compute(b: i32, a: i32)"), "{changed}");
    assert!(changed.contains("compute(2, 1)"), "{changed}");

    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn data_set_and_ensure_run_through_the_cli() {
    let (repo, store) = fresh_workspace("data");
    std::fs::write(
        repo.join("config.json"),
        "{\n  \"server\": {\n    \"port\": 9000,\n    \"host\": \"localhost\"\n  }\n}\n",
    )
    .unwrap();
    let repo_arg = repo.to_string_lossy().into_owned();

    let (set_code, set_stdout, set_stderr) = run(
        &repo,
        &store,
        &[
            "--root",
            &repo_arg,
            "edit",
            "data",
            "set",
            "--file",
            "config.json",
            "--path",
            "$.server.port",
            "--value-json",
            "8080",
        ],
    );
    assert_eq!(set_code, 0, "data set failed: {set_stderr}\n{set_stdout}");
    assert!(set_stdout.contains("\"status\": \"applied\""));
    let changed = std::fs::read_to_string(repo.join("config.json")).unwrap();
    assert!(changed.contains("\"port\": 8080"), "{changed}");
    assert!(changed.contains("\"host\": \"localhost\""), "{changed}");

    let (ensure_code, ensure_stdout, ensure_stderr) = run(
        &repo,
        &store,
        &[
            "--root",
            &repo_arg,
            "edit",
            "data",
            "ensure",
            "--file",
            "config.json",
            "--path",
            "$.server.port",
            "--value-json",
            "8080",
        ],
    );
    assert_eq!(
        ensure_code, 0,
        "data ensure failed: {ensure_stderr}\n{ensure_stdout}"
    );
    assert!(
        ensure_stdout.contains("\"status\": \"already-satisfied\""),
        "{ensure_stdout}"
    );

    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn semantic_lsp_backend_is_invalid_spec_before_graph_resolution() {
    let (repo, store) = fresh_workspace("lsp");
    std::fs::write(
        repo.join("sig.json"),
        r#"{
  "old_parameters": "(a)",
  "new_parameters": "(a)",
  "expect_call_sites": 0
}
"#,
    )
    .unwrap();
    let repo_arg = repo.to_string_lossy().into_owned();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &[
            "--root",
            &repo_arg,
            "edit",
            "change-signature",
            "--symbol",
            "missing",
            "--spec",
            "sig.json",
            "--backend",
            "lsp",
        ],
    );
    assert_eq!(code, 20, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.is_empty(), "{stdout}");
    assert!(
        stderr.contains("--backend lsp is unavailable in this build; use --backend graph"),
        "{stderr}"
    );

    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}
