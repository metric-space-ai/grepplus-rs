use greppy_edit::journal::{publish_journal, recover, FilePublication};
use sha2::{Digest, Sha256};

const CRASH_AFTER_ENV: &str = "GREPPY_EDIT_TEST_JOURNAL_CRASH_AFTER";
const BOUNDARIES: &[&str] = &[
    "lock-acquired",
    "cas-verified",
    "journal-dir-created",
    "pre-image-0",
    "pre-image-1",
    "journal-uncommitted",
    "journal-committed",
    "published-0",
    "published-1",
    "journal-removed",
];

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn publication(path: &str, before: &[u8], after: &[u8]) -> FilePublication {
    FilePublication {
        rel_path: path.into(),
        expected_live_sha256: sha256_hex(before),
        content: after.to_vec(),
    }
}

#[test]
fn recover_restores_consistency_after_every_journal_boundary() {
    for boundary in BOUNDARIES {
        let workspace = tempfile::tempdir().unwrap();
        let a = workspace.path().join("a.txt");
        let b = workspace.path().join("b.txt");
        std::fs::write(&a, b"a-before").unwrap();
        std::fs::write(&b, b"b-before").unwrap();
        let files = [
            publication("a.txt", b"a-before", b"a-after"),
            publication("b.txt", b"b-before", b"b-after"),
        ];

        std::env::set_var(CRASH_AFTER_ENV, boundary);
        let result = publish_journal(workspace.path(), "tx-crash-boundary", &files);
        std::env::remove_var(CRASH_AFTER_ENV);
        assert!(
            result.is_err(),
            "boundary {boundary} did not inject a crash"
        );

        recover(workspace.path()).unwrap();
        let live_a = std::fs::read(&a).unwrap();
        let live_b = std::fs::read(&b).unwrap();
        if *boundary == "journal-removed" {
            assert_eq!(live_a, b"a-after", "boundary {boundary}");
            assert_eq!(live_b, b"b-after", "boundary {boundary}");
        } else {
            assert_eq!(live_a, b"a-before", "boundary {boundary}");
            assert_eq!(live_b, b"b-before", "boundary {boundary}");
        }
        assert!(
            !workspace.path().join(".greppy-edit-journal").exists(),
            "recovery left journal state after {boundary}"
        );
        assert!(
            !workspace.path().join(".greppy-edit.lock").exists(),
            "recovery left workspace lock after {boundary}"
        );
    }
}
