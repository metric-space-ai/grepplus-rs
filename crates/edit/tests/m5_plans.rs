use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use greppy_edit::journal::{
    publish_journal, recover_with_report, FilePublication, RecoveryAction, WorkspaceLock,
};
use greppy_edit::plan::{
    apply_plan, Plan, PlanAction, PlanOperation, PlanPreconditions, PlanPublish, PlanPublishMode,
    PlanSelector, PlanWorkspace, PLAN_SCHEMA,
};
use greppy_edit::Status;
use sha2::{Digest, Sha256};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    TEST_MUTEX.lock().unwrap()
}

fn sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn file_hashes(root: &std::path::Path, files: &[&str]) -> BTreeMap<String, String> {
    files
        .iter()
        .map(|file| {
            (
                (*file).to_string(),
                sha256(&std::fs::read(root.join(file)).unwrap()),
            )
        })
        .collect()
}

fn text_operation(
    id: &str,
    file: &str,
    old_text: &str,
    replacement: &str,
    file_sha256: String,
) -> PlanOperation {
    PlanOperation {
        id: id.into(),
        file: file.into(),
        selector: PlanSelector::Text {
            old_text: old_text.into(),
            expect: 1,
        },
        action: PlanAction::Replace {
            content: replacement.into(),
        },
        preconditions: PlanPreconditions {
            file_sha256: Some(file_sha256),
            target_sha256: None,
        },
    }
}

fn journal_plan(root: &std::path::Path, operations: Vec<PlanOperation>) -> Plan {
    Plan {
        schema_version: PLAN_SCHEMA.into(),
        workspace: PlanWorkspace {
            root: root.to_string_lossy().into_owned(),
            expect_git_head: None,
            require_unchanged_files: true,
        },
        operations,
        validators: vec![],
        publish: PlanPublish {
            mode: PlanPublishMode::Journal,
        },
    }
}

#[test]
fn stale_third_operation_keeps_all_three_files_byte_identical() {
    let _guard = serial();
    let workspace = tempfile::tempdir().unwrap();
    for (file, content) in [
        ("one.txt", b"one = old\n".as_slice()),
        ("two.txt", b"two = old\n".as_slice()),
        ("three.txt", b"three = old\n".as_slice()),
    ] {
        std::fs::write(workspace.path().join(file), content).unwrap();
    }
    let files = ["one.txt", "two.txt", "three.txt"];
    let before = file_hashes(workspace.path(), &files);
    let plan = journal_plan(
        workspace.path(),
        vec![
            text_operation(
                "operation-1",
                "one.txt",
                "old",
                "new",
                before["one.txt"].clone(),
            ),
            text_operation(
                "operation-2",
                "two.txt",
                "old",
                "new",
                before["two.txt"].clone(),
            ),
            text_operation(
                "operation-3",
                "three.txt",
                "old",
                "new",
                "stale-file-hash".into(),
            ),
        ],
    );

    let certificate = apply_plan(&plan, false).unwrap();

    assert_eq!(certificate.status, Status::Stale);
    assert_eq!(certificate.exit_code(), 12);
    assert!(!certificate.published);
    assert_eq!(certificate.operations.len(), 3);
    assert_eq!(
        certificate
            .operations
            .iter()
            .map(|report| report.id.as_str())
            .collect::<Vec<_>>(),
        ["operation-1", "operation-2", "operation-3"]
    );
    assert_eq!(file_hashes(workspace.path(), &files), before);
}

#[test]
fn overlapping_operations_name_both_reports_and_mutate_nothing() {
    let _guard = serial();
    let workspace = tempfile::tempdir().unwrap();
    let file = workspace.path().join("overlap.txt");
    let original = b"0123456789\n";
    std::fs::write(&file, original).unwrap();
    let hash = sha256(original);
    let plan = journal_plan(
        workspace.path(),
        vec![
            PlanOperation {
                id: "wide-range".into(),
                file: "overlap.txt".into(),
                selector: PlanSelector::Resolved {
                    byte_start: 1,
                    byte_end: 7,
                },
                action: PlanAction::Replace {
                    content: "WIDE".into(),
                },
                preconditions: PlanPreconditions {
                    file_sha256: Some(hash.clone()),
                    target_sha256: None,
                },
            },
            PlanOperation {
                id: "inner-range".into(),
                file: "overlap.txt".into(),
                selector: PlanSelector::Resolved {
                    byte_start: 4,
                    byte_end: 9,
                },
                action: PlanAction::Delete,
                preconditions: PlanPreconditions {
                    file_sha256: Some(hash),
                    target_sha256: None,
                },
            },
        ],
    );

    let certificate = apply_plan(&plan, false).unwrap();

    assert_eq!(certificate.status, Status::InvalidResult);
    assert_eq!(certificate.exit_code(), 13);
    assert!(!certificate.published);
    assert_eq!(certificate.operations.len(), 2);
    assert!(certificate.operations[0]
        .postconditions
        .iter()
        .any(|result| result.detail.as_deref().is_some_and(|detail| {
            detail.contains("wide-range") && detail.contains("inner-range")
        })));
    assert!(certificate.operations[1]
        .postconditions
        .iter()
        .any(|result| result.detail.as_deref().is_some_and(|detail| {
            detail.contains("wide-range") && detail.contains("inner-range")
        })));
    assert_eq!(std::fs::read(file).unwrap(), original);
}

#[test]
fn non_overlapping_same_file_operations_use_original_offsets() {
    let _guard = serial();
    let workspace = tempfile::tempdir().unwrap();
    let file = workspace.path().join("offsets.txt");
    let original = b"alpha beta gamma\n";
    std::fs::write(&file, original).unwrap();
    let hash = sha256(original);
    let plan = journal_plan(
        workspace.path(),
        vec![
            PlanOperation {
                id: "grow-first".into(),
                file: "offsets.txt".into(),
                selector: PlanSelector::Resolved {
                    byte_start: 0,
                    byte_end: 5,
                },
                action: PlanAction::Replace {
                    content: "ALPHA-LONG".into(),
                },
                preconditions: PlanPreconditions {
                    file_sha256: Some(hash.clone()),
                    target_sha256: Some(sha256(b"alpha")),
                },
            },
            PlanOperation {
                id: "shrink-last".into(),
                file: "offsets.txt".into(),
                selector: PlanSelector::Resolved {
                    byte_start: 11,
                    byte_end: 16,
                },
                action: PlanAction::Replace {
                    content: "G".into(),
                },
                preconditions: PlanPreconditions {
                    file_sha256: Some(hash),
                    target_sha256: Some(sha256(b"gamma")),
                },
            },
        ],
    );

    let certificate = apply_plan(&plan, false).unwrap();

    assert_eq!(certificate.status, Status::Applied, "{certificate:#?}");
    assert!(certificate.published);
    assert_eq!(certificate.operations.len(), 2);
    assert_eq!(std::fs::read(file).unwrap(), b"ALPHA-LONG beta G\n");
    assert_eq!(certificate.operations[0].changed_byte_ranges, [(0, 5)]);
    assert_eq!(certificate.operations[1].changed_byte_ranges, [(11, 16)]);
}

#[test]
fn active_lock_fails_immediately_and_dead_pid_lock_is_taken_over() {
    let _guard = serial();
    let workspace = tempfile::tempdir().unwrap();
    let file = workspace.path().join("locked.py");
    let original = b"state = old\n";
    std::fs::write(&file, original).unwrap();
    let make_plan = || {
        journal_plan(
            workspace.path(),
            vec![text_operation(
                "locked-operation",
                "locked.py",
                "old",
                "new",
                sha256(original),
            )],
        )
    };

    let held = WorkspaceLock::acquire(workspace.path()).unwrap();
    let started = Instant::now();
    let refused = apply_plan(&make_plan(), false).unwrap();
    assert!(started.elapsed() < Duration::from_secs(1));
    assert_eq!(refused.status, Status::PublishFailed);
    assert_eq!(refused.exit_code(), 16);
    assert!(!refused.published);
    assert_eq!(std::fs::read(&file).unwrap(), original);
    drop(held);

    std::fs::write(workspace.path().join(".greppy-edit.lock"), b"999999999\n").unwrap();
    let applied = apply_plan(&make_plan(), false).unwrap();
    assert_eq!(applied.status, Status::Applied);
    assert!(applied.published);
    assert!(applied.operations[0]
        .postconditions
        .iter()
        .any(|result| result.name == "workspace-lock-takeover" && result.passed));
    assert_eq!(std::fs::read(file).unwrap(), b"state = new\n");
}

#[test]
fn explicit_recover_reports_and_restores_interrupted_publish() {
    let _guard = serial();
    const CRASH_AFTER_ENV: &str = "GREPPY_EDIT_TEST_JOURNAL_CRASH_AFTER";
    let workspace = tempfile::tempdir().unwrap();
    let first = workspace.path().join("first.txt");
    let second = workspace.path().join("second.txt");
    std::fs::write(&first, b"first-before").unwrap();
    std::fs::write(&second, b"second-before").unwrap();
    let publications = [
        FilePublication {
            rel_path: "first.txt".into(),
            expected_live_sha256: sha256(b"first-before"),
            content: b"first-after".to_vec(),
        },
        FilePublication {
            rel_path: "second.txt".into(),
            expected_live_sha256: sha256(b"second-before"),
            content: b"second-after".to_vec(),
        },
    ];

    std::env::set_var(CRASH_AFTER_ENV, "published-0");
    let interrupted = publish_journal(workspace.path(), "tx-m5-recover", &publications);
    std::env::remove_var(CRASH_AFTER_ENV);
    assert!(interrupted.is_err());

    let report = recover_with_report(workspace.path()).unwrap();

    assert!(report.found_journal);
    assert!(report.committed);
    assert_eq!(report.action, RecoveryAction::RolledBack);
    assert_eq!(report.transaction_id.as_deref(), Some("tx-m5-recover"));
    assert_eq!(report.files_considered, 2);
    assert_eq!(report.files_restored, 1);
    assert_eq!(std::fs::read(first).unwrap(), b"first-before");
    assert_eq!(std::fs::read(second).unwrap(), b"second-before");
    assert!(!workspace.path().join(".greppy-edit-journal").exists());
    assert!(!workspace.path().join(".greppy-edit.lock").exists());
}
