//! File-oriented `greppy read` hardening coverage.

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
        "greppy-cli-read-file-{tag}-{}-{n}",
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
fn read_path_prints_numbered_file_lines_without_an_index() {
    let (repo, store) = fresh_workspace("numbered");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "alpha\nbeta\ngamma\ndelta\n").unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["read", "src/lib.rs"]);

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.starts_with("src/lib.rs:1-4\n"), "{stdout}");
    assert!(stdout.contains("1 | alpha"), "{stdout}");
    assert!(stdout.contains("4 | delta"), "{stdout}");
}

#[test]
fn read_path_lines_selects_an_inclusive_range() {
    let (repo, store) = fresh_workspace("range");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "alpha\nbeta\ngamma\ndelta\n").unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read", "src/lib.rs", "--lines", "2:3"],
    );

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.starts_with("src/lib.rs:2-3\n"), "{stdout}");
    assert!(stdout.contains("2 | beta"), "{stdout}");
    assert!(stdout.contains("3 | gamma"), "{stdout}");
    assert!(!stdout.contains("alpha"), "{stdout}");
    assert!(!stdout.contains("delta"), "{stdout}");
}

#[test]
fn misspelled_read_path_suggests_paths_not_symbols() {
    let (repo, store) = fresh_workspace("suggestion");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn target() {}\n").unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["read", "src/lbi.rs"]);

    assert_eq!(code, 10, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("closest paths"), "{stdout}");
    assert!(stdout.contains("src/lib.rs"), "{stdout}");
    assert!(stdout.contains("try: greppy read src/lib.rs"), "{stdout}");
    assert!(!stdout.contains("closest definitions"), "{stdout}");
}
