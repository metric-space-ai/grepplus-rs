use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct FixtureRepo {
    root: PathBuf,
    store: PathBuf,
}

impl FixtureRepo {
    fn new(label: &str) -> Self {
        let nonce = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "greppy-verify-grid-{label}-{}-{epoch}-{nonce}",
            std::process::id()
        ));
        let root = base.join("repo");
        let store = base.join("store");
        fs::create_dir_all(&root).unwrap();
        git(&root, &["init", "-q"]);
        git(&root, &["config", "user.name", "Greppy Verify Test"]);
        git(&root, &["config", "user.email", "verify@example.invalid"]);
        Self { root, store }
    }

    fn commit_all(&self, message: &str) {
        git(&self.root, &["add", "-A"]);
        git(&self.root, &["commit", "-q", "-m", message]);
    }

    fn verify(&self, command: &[&str]) -> (Output, Value) {
        let mut process = Command::new(env!("CARGO_BIN_EXE_greppy"));
        process
            .current_dir(&self.root)
            .env("GREPPY_STORE_DIR", &self.store)
            .args(["verify", "--json", "--no-cache", "--timeout", "120", "--"])
            .args(command);
        let output = process.output().unwrap();
        let report: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
            panic!(
                "verify stdout was not JSON: {error}\nstdout={}\nstderr={}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        });
        (output, report)
    }
}

impl Drop for FixtureRepo {
    fn drop(&mut self) {
        if let Some(base) = self.root.parent() {
            let _ = fs::remove_dir_all(base);
        }
    }
}

fn git(root: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn write_pytest_fixture(repo: &FixtureRepo, state: &str) {
    fs::create_dir_all(repo.root.join("tests")).unwrap();
    fs::write(repo.root.join("state.txt"), state).unwrap();
    fs::write(
        repo.root.join("pytest_fixture.py"),
        r#"import pathlib
import sys
state = pathlib.Path("state.txt").read_text().strip()
test_id = "tests/test_demo.py::test_value"
if state == "green":
    print(f"{test_id} PASSED [100%]")
    print("============================== 1 passed in 0.01s ==============================")
    raise SystemExit(0)
print(f"{test_id} FAILED [100%]")
print(f"FAILED {test_id} - assert 2 == 1")
print("============================== 1 failed in 0.01s ==============================")
raise SystemExit(1)
"#,
    )
    .unwrap();
    fs::write(
        repo.root.join("tests/test_demo.py"),
        "def test_value():\n    assert True\n",
    )
    .unwrap();
}

fn first_id<'a>(report: &'a Value, class: &str) -> Option<&'a str> {
    report[class]
        .as_array()
        .and_then(|cases| cases.first())
        .and_then(|case| case["test_id"].as_str())
}

#[test]
fn pytest_newly_failed_preexisting_and_fixed_are_distinguished() {
    let newly = FixtureRepo::new("pytest-new");
    write_pytest_fixture(&newly, "green");
    newly.commit_all("green baseline");
    fs::write(newly.root.join("state.txt"), "red").unwrap();
    let (output, report) = newly.verify(&["python3", "pytest_fixture.py"]);
    assert_eq!(output.status.code(), Some(21));
    assert_eq!(report["framework"], "pytest");
    assert_eq!(
        first_id(&report, "newly_failed"),
        Some("tests/test_demo.py::test_value")
    );

    let preexisting = FixtureRepo::new("pytest-preexisting");
    write_pytest_fixture(&preexisting, "red");
    preexisting.commit_all("red baseline");
    let (output, report) = preexisting.verify(&["python3", "pytest_fixture.py"]);
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        first_id(&report, "preexisting_failed"),
        Some("tests/test_demo.py::test_value")
    );
    assert!(report["newly_failed"].as_array().unwrap().is_empty());

    let fixed = FixtureRepo::new("pytest-fixed");
    write_pytest_fixture(&fixed, "red");
    fixed.commit_all("red baseline");
    fs::write(fixed.root.join("state.txt"), "green").unwrap();
    let (output, report) = fixed.verify(&["python3", "pytest_fixture.py"]);
    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        first_id(&report, "fixed"),
        Some("tests/test_demo.py::test_value")
    );
}

#[test]
fn cargo_fixture_detects_new_failure_and_mirrors_target() {
    let repo = FixtureRepo::new("cargo-new");
    fs::write(
        repo.root.join("Cargo.toml"),
        "[package]\nname = \"verify-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(repo.root.join(".gitignore"), "/target\n").unwrap();
    fs::create_dir_all(repo.root.join("src")).unwrap();
    fs::write(
        repo.root.join("src/lib.rs"),
        "#[cfg(test)]\nmod tests {\n    #[test]\n    fn value_is_one() { assert_eq!(1, 1); }\n}\n",
    )
    .unwrap();
    repo.commit_all("passing cargo baseline");
    fs::write(
        repo.root.join("src/lib.rs"),
        "#[cfg(test)]\nmod tests {\n    #[test]\n    fn value_is_one() { assert_eq!(2, 1); }\n}\n",
    )
    .unwrap();

    let (output, report) = repo.verify(&["cargo", "test"]);
    assert_eq!(output.status.code(), Some(21));
    assert_eq!(report["framework"], "cargo-test");
    assert_eq!(
        first_id(&report, "newly_failed"),
        Some("tests::value_is_one")
    );
    assert!(report["mirrored_paths"]
        .as_array()
        .unwrap()
        .iter()
        .any(|path| path == "target"));
}

#[test]
fn clean_workspace_digest_status_and_worktree_list_are_unchanged() {
    let repo = FixtureRepo::new("workspace");
    write_pytest_fixture(&repo, "green");
    repo.commit_all("clean passing fixture");
    let worktrees_before = git(&repo.root, &["worktree", "list", "--porcelain"]);
    assert!(git(&repo.root, &["status", "--porcelain"]).is_empty());

    let (output, report) = repo.verify(&["python3", "pytest_fixture.py"]);

    assert_eq!(output.status.code(), Some(0));
    assert_eq!(
        report["workspace_digest_before"],
        report["workspace_digest_after"]
    );
    assert_eq!(report["workspace_unchanged"], true);
    assert!(git(&repo.root, &["status", "--porcelain"]).is_empty());
    assert_eq!(
        git(&repo.root, &["worktree", "list", "--porcelain"]),
        worktrees_before
    );
}

#[test]
fn nonexistent_command_is_infrastructure_error() {
    let repo = FixtureRepo::new("missing-command");
    write_pytest_fixture(&repo, "green");
    repo.commit_all("fixture");
    let command = "greppy-command-that-does-not-exist-verify-grid";

    let (output, report) = repo.verify(&[command]);

    assert_eq!(output.status.code(), Some(22));
    assert_eq!(report["exit_code"], 22);
    let infrastructure = report["infrastructure_error"].as_array().unwrap();
    assert!(infrastructure.iter().any(|case| {
        case["test_id"]
            .as_str()
            .is_some_and(|id| id.starts_with("after:test_process_spawn_error"))
    }));
}
