//! End-to-end grid for the deterministic `greppy map` orientation verb.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_repo(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base =
        std::env::temp_dir().join(format!("greppy-cli-map-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join("tests")).unwrap();
    std::fs::create_dir_all(repo.join("python_pkg")).unwrap();
    std::fs::create_dir_all(repo.join("vendor/biglib")).unwrap();

    std::fs::write(
        repo.join("Cargo.toml"),
        "[package]\nname = \"map-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn answer() -> u32 { 42 }\n").unwrap();
    std::fs::write(repo.join("tests/smoke.rs"), "#[test] fn smoke() {}\n").unwrap();
    std::fs::write(
        repo.join("python_pkg/app.py"),
        "def run():\n    return 42\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("tests/test_app.py"),
        "def test_app():\n    assert True\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("vendor/biglib/generated.py"),
        "def vendored():\n    pass\n",
    )
    .unwrap();
    std::fs::write(repo.join("pytest.ini"), "[pytest]\ntestpaths = tests\n").unwrap();
    std::fs::write(repo.join("tox.ini"), "[tox]\nenvlist = py\n").unwrap();
    std::fs::write(
        repo.join("package.json"),
        r#"{"scripts":{"test":"node test.js","test:unit":"node unit.js","build":"node build.js"}}"#,
    )
    .unwrap();
    std::fs::write(repo.join("Makefile"), "test:\n\t@true\ncheck:\n\t@true\n").unwrap();

    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.email", "map@example.invalid"]);
    git(&repo, &["config", "user.name", "Map Fixture"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-qm", "fixture"]);
    (repo, base.join("store"))
}

fn git(repo: &Path, args: &[&str]) {
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
fn mixed_repo_reports_languages_tests_commands_and_vendor_marker() {
    let (repo, store) = fresh_repo("mixed");
    let (code, stdout, stderr) = run(&repo, &store, &["index", "."]);
    assert_eq!(code, 0, "index failed\nstdout={stdout}\nstderr={stderr}");

    let (code, stdout, stderr) = run(&repo, &store, &["map"]);
    assert_eq!(code, 0, "map failed\nstdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("rust"), "missing Rust language: {stdout}");
    assert!(
        stdout.contains("python"),
        "missing Python language: {stdout}"
    );
    assert!(
        stdout.contains("indexed: yes"),
        "missing index coverage: {stdout}"
    );
    assert!(stdout.contains("tests"), "missing test root: {stdout}");
    for command in [
        "cargo test",
        "pytest",
        "tox",
        "npm test",
        "npm run test:unit",
        "make test",
        "make check",
    ] {
        assert!(
            stdout.contains(command),
            "missing command {command:?}: {stdout}"
        );
    }
    let vendor_line = stdout
        .lines()
        .find(|line| line.contains("vendor"))
        .expect("vendor module line");
    assert!(
        vendor_line.contains("[collapsed]"),
        "vendor was expanded: {vendor_line}"
    );
}

#[test]
fn json_shape_is_stable_and_complete() {
    let (repo, store) = fresh_repo("json");
    let (code, stdout, stderr) = run(&repo, &store, &["map", "--json"]);
    assert_eq!(code, 0, "map failed\nstdout={stdout}\nstderr={stderr}");
    let value: Value = serde_json::from_str(&stdout).expect("valid map JSON");
    let object = value.as_object().expect("map JSON object");
    let keys = object.keys().map(String::as_str).collect::<Vec<_>>();
    assert_eq!(
        keys,
        vec![
            "commands",
            "index",
            "languages",
            "large_subtrees",
            "modules",
            "path",
            "root",
            "schema_version",
            "suggestions",
            "test_roots",
        ]
    );
    assert_eq!(value["schema_version"], "greppy.map.v1");
    assert_eq!(value["path"], ".");
    assert!(value["languages"].is_array());
    assert!(value["modules"].is_array());
    assert!(value["test_roots"].is_array());
    assert!(value["commands"].is_array());
    assert!(value["large_subtrees"].is_array());
    assert!(value["suggestions"].is_array());
    assert!(value["index"]["available"].is_boolean());
}

#[test]
fn text_output_stays_within_sixty_lines_and_ends_with_drilldowns() {
    let (repo, store) = fresh_repo("budget");
    for index in 0..24 {
        let dir = repo.join(format!("module_{index:02}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("code.rs"), format!("pub fn f_{index}() {{}}\n")).unwrap();
    }
    let (code, stdout, stderr) = run(&repo, &store, &["map"]);
    assert_eq!(code, 0, "map failed\nstdout={stdout}\nstderr={stderr}");
    assert!(
        stdout.lines().count() <= 60,
        "{} lines\n{stdout}",
        stdout.lines().count()
    );
    let tries = stdout
        .lines()
        .filter(|line| line.starts_with("try: greppy map "))
        .count();
    assert!(
        (2..=3).contains(&tries),
        "expected 2-3 drilldowns: {stdout}"
    );
    assert!(
        stdout
            .lines()
            .rev()
            .take(3)
            .all(|line| line.starts_with("try: ")),
        "drilldowns must end the output: {stdout}"
    );
}

#[test]
fn path_argument_scopes_modules_and_files() {
    let (repo, store) = fresh_repo("path");
    std::fs::create_dir_all(repo.join("python_pkg/tests")).unwrap();
    std::fs::write(
        repo.join("python_pkg/tests/test_nested.py"),
        "def test_nested(): pass\n",
    )
    .unwrap();
    let (code, stdout, stderr) = run(&repo, &store, &["map", "python_pkg", "--json"]);
    assert_eq!(code, 0, "map failed\nstdout={stdout}\nstderr={stderr}");
    let value: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["path"], "python_pkg");
    let languages = value["languages"].as_array().unwrap();
    assert_eq!(languages.len(), 1);
    assert_eq!(languages[0]["language"], "python");
    assert_eq!(languages[0]["files"], 2);
}
