//! Logical all-or-nothing multi-file publish with a pre-image journal.
//!
//! Protocol:
//! 1. take the workspace lock (`.greppy-edit.lock` in the workspace root)
//! 2. re-verify every input hash (CAS) under the lock
//! 3. write the journal: per file the pre-image bytes + both hashes,
//!    fsynced, then mark it `committed`
//! 4. publish every file atomically (tmp+fsync+rename)
//! 5. remove the journal (success) — or roll back every already-published
//!    file from its pre-image and remove the journal (failure)
//!
//! A crash between 3 and 5 leaves a committed journal on disk;
//! `greppy edit recover` restores every pre-image and removes it. A crash
//! before the `committed` marker means nothing was published; the journal
//! is discarded.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::hash::sha256_hex;
use crate::publish::{publish_atomic, require_inside_workspace};
use greppy_core::{Error, Result};

const JOURNAL_DIR: &str = ".greppy-edit-journal";
const LOCK_NAME: &str = ".greppy-edit.lock";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub rel_path: String,
    pub pre_image_file: String,
    pub pre_sha256: String,
    pub post_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Journal {
    pub schema_version: String,
    pub transaction_id: String,
    pub committed: bool,
    pub entries: Vec<JournalEntry>,
}

/// One planned file publication.
pub struct FilePublication {
    pub rel_path: String,
    pub expected_live_sha256: String,
    pub content: Vec<u8>,
}

/// Simple advisory lock: exclusive-create of a lock file. Stale locks (from
/// a crashed process) are taken over if older than 10 minutes.
struct WorkspaceLock {
    path: PathBuf,
}

impl WorkspaceLock {
    fn acquire(workspace_root: &Path) -> Result<Self> {
        let path = workspace_root.join(LOCK_NAME);
        for _ in 0..2 {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut f) => {
                    let _ = writeln!(f, "{}", std::process::id());
                    return Ok(Self { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    let stale = std::fs::metadata(&path)
                        .and_then(|m| m.modified())
                        .ok()
                        .and_then(|t| t.elapsed().ok())
                        .map(|age| age.as_secs() > 600)
                        .unwrap_or(true);
                    if stale {
                        let _ = std::fs::remove_file(&path);
                        continue;
                    }
                    return Err(Error::Workspace(
                        "another greppy edit transaction holds the workspace lock".into(),
                    ));
                }
                Err(source) => {
                    return Err(Error::Io {
                        context: format!("create lock {}", path.display()),
                        source,
                    })
                }
            }
        }
        Err(Error::Workspace("could not acquire workspace lock".into()))
    }
}

impl Drop for WorkspaceLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn journal_dir(workspace_root: &Path) -> PathBuf {
    workspace_root.join(JOURNAL_DIR)
}

fn journal_path(workspace_root: &Path) -> PathBuf {
    journal_dir(workspace_root).join("journal.json")
}

/// Publish `files` as one logical transaction. Returns the transaction id.
pub fn publish_journal(
    workspace_root: &Path,
    transaction_id: &str,
    files: &[FilePublication],
) -> Result<()> {
    let _lock = WorkspaceLock::acquire(workspace_root)?;

    // CAS for every file under the lock, before anything is written
    for f in files {
        let abs = require_inside_workspace(workspace_root, &workspace_root.join(&f.rel_path))?;
        let live = std::fs::read(&abs).map_err(|source| Error::Io {
            context: format!("read {}", abs.display()),
            source,
        })?;
        if sha256_hex(&live) != f.expected_live_sha256 {
            return Err(Error::Workspace(format!(
                "stale plan: {} changed since planning; nothing was written",
                f.rel_path
            )));
        }
    }

    // write pre-images + journal, fsync, then mark committed
    let dir = journal_dir(workspace_root);
    std::fs::create_dir_all(&dir).map_err(|source| Error::Io {
        context: format!("create {}", dir.display()),
        source,
    })?;
    let mut entries = Vec::new();
    for (i, f) in files.iter().enumerate() {
        let abs = workspace_root.join(&f.rel_path);
        let pre = std::fs::read(&abs).map_err(|source| Error::Io {
            context: format!("read {}", abs.display()),
            source,
        })?;
        let pre_name = format!("pre-{i:04}.bin");
        let pre_path = dir.join(&pre_name);
        std::fs::write(&pre_path, &pre).map_err(|source| Error::Io {
            context: format!("write {}", pre_path.display()),
            source,
        })?;
        if let Ok(h) = std::fs::File::open(&pre_path) {
            let _ = h.sync_all();
        }
        entries.push(JournalEntry {
            rel_path: f.rel_path.clone(),
            pre_image_file: pre_name,
            pre_sha256: sha256_hex(&pre),
            post_sha256: sha256_hex(&f.content),
        });
    }
    let mut journal = Journal {
        schema_version: "greppy.edit-journal.v1".into(),
        transaction_id: transaction_id.to_string(),
        committed: false,
        entries,
    };
    write_journal(workspace_root, &journal)?;
    journal.committed = true;
    write_journal(workspace_root, &journal)?;

    // publish; roll back from pre-images on any failure
    let mut published = 0usize;
    let mut failure: Option<Error> = None;
    for f in files {
        match publish_atomic(
            workspace_root,
            &workspace_root.join(&f.rel_path),
            &f.content,
            &f.expected_live_sha256,
        ) {
            Ok(_) => published += 1,
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }
    if let Some(e) = failure {
        for (f, entry) in files.iter().zip(&journal.entries).take(published) {
            let pre =
                std::fs::read(dir.join(&entry.pre_image_file)).map_err(|source| Error::Io {
                    context: format!("read pre-image for {}", f.rel_path),
                    source,
                })?;
            // rollback ignores CAS: restoring the pre-image is the contract
            let abs = workspace_root.join(&f.rel_path);
            std::fs::write(&abs, &pre).map_err(|source| Error::Io {
                context: format!("rollback {}", abs.display()),
                source,
            })?;
        }
        let _ = std::fs::remove_dir_all(&dir);
        return Err(e);
    }
    std::fs::remove_dir_all(&dir).map_err(|source| Error::Io {
        context: format!("remove journal {}", dir.display()),
        source,
    })?;
    Ok(())
}

fn write_journal(workspace_root: &Path, journal: &Journal) -> Result<()> {
    let path = journal_path(workspace_root);
    let bytes = serde_json::to_vec_pretty(journal)
        .map_err(|e| Error::Invalid(format!("serialize journal: {e}")))?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, &bytes).map_err(|source| Error::Io {
        context: format!("write {}", tmp.display()),
        source,
    })?;
    if let Ok(h) = std::fs::File::open(&tmp) {
        let _ = h.sync_all();
    }
    std::fs::rename(&tmp, &path).map_err(|source| Error::Io {
        context: format!("rename {}", path.display()),
        source,
    })?;
    Ok(())
}

/// Recovery outcome for `greppy edit recover`.
#[derive(Debug, PartialEq, Eq)]
pub enum Recovery {
    NothingToRecover,
    RolledBack {
        transaction_id: String,
        files: usize,
    },
    DiscardedUncommitted,
}

/// Restore pre-images from a committed journal left by a crash.
pub fn recover(workspace_root: &Path) -> Result<Recovery> {
    let path = journal_path(workspace_root);
    if !path.exists() {
        return Ok(Recovery::NothingToRecover);
    }
    let journal: Journal =
        serde_json::from_slice(&std::fs::read(&path).map_err(|source| Error::Io {
            context: format!("read {}", path.display()),
            source,
        })?)
        .map_err(|e| Error::Invalid(format!("journal unreadable: {e}")))?;
    let dir = journal_dir(workspace_root);
    if !journal.committed {
        std::fs::remove_dir_all(&dir).map_err(|source| Error::Io {
            context: format!("remove journal {}", dir.display()),
            source,
        })?;
        return Ok(Recovery::DiscardedUncommitted);
    }
    let mut restored = 0usize;
    for entry in &journal.entries {
        let abs = workspace_root.join(&entry.rel_path);
        let live_sha = std::fs::read(&abs)
            .map(|b| sha256_hex(&b))
            .unwrap_or_default();
        if live_sha == entry.pre_sha256 {
            continue; // this file was never published
        }
        // restore only files that carry the transaction's post-image; any
        // OTHER content means someone edited after the crash - refuse
        if live_sha != entry.post_sha256 {
            return Err(Error::Workspace(format!(
                "recover: {} was modified after the crashed transaction; resolve manually",
                entry.rel_path
            )));
        }
        let pre = std::fs::read(dir.join(&entry.pre_image_file)).map_err(|source| Error::Io {
            context: format!("read pre-image for {}", entry.rel_path),
            source,
        })?;
        std::fs::write(&abs, &pre).map_err(|source| Error::Io {
            context: format!("restore {}", abs.display()),
            source,
        })?;
        restored += 1;
    }
    std::fs::remove_dir_all(&dir).map_err(|source| Error::Io {
        context: format!("remove journal {}", dir.display()),
        source,
    })?;
    Ok(Recovery::RolledBack {
        transaction_id: journal.transaction_id,
        files: restored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ws() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    fn pubfile(rel: &str, old: &[u8], new: &[u8]) -> FilePublication {
        FilePublication {
            rel_path: rel.into(),
            expected_live_sha256: sha256_hex(old),
            content: new.to_vec(),
        }
    }

    #[test]
    fn two_file_transaction_publishes_both() {
        let dir = ws();
        std::fs::write(dir.path().join("a.txt"), b"a1").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"b1").unwrap();
        publish_journal(
            dir.path(),
            "tx-1",
            &[
                pubfile("a.txt", b"a1", b"a2"),
                pubfile("b.txt", b"b1", b"b2"),
            ],
        )
        .unwrap();
        assert_eq!(std::fs::read(dir.path().join("a.txt")).unwrap(), b"a2");
        assert_eq!(std::fs::read(dir.path().join("b.txt")).unwrap(), b"b2");
        assert!(!journal_path(dir.path()).exists());
    }

    #[test]
    fn stale_second_file_changes_nothing() {
        let dir = ws();
        std::fs::write(dir.path().join("a.txt"), b"a1").unwrap();
        std::fs::write(dir.path().join("b.txt"), b"CHANGED").unwrap();
        let err = publish_journal(
            dir.path(),
            "tx-2",
            &[
                pubfile("a.txt", b"a1", b"a2"),
                pubfile("b.txt", b"b1", b"b2"),
            ],
        );
        assert!(err.is_err());
        assert_eq!(std::fs::read(dir.path().join("a.txt")).unwrap(), b"a1");
        assert_eq!(std::fs::read(dir.path().join("b.txt")).unwrap(), b"CHANGED");
    }

    #[test]
    fn recover_restores_committed_journal() {
        let dir = ws();
        std::fs::write(dir.path().join("a.txt"), b"a2").unwrap(); // post-image
        let jd = journal_dir(dir.path());
        std::fs::create_dir_all(&jd).unwrap();
        std::fs::write(jd.join("pre-0000.bin"), b"a1").unwrap();
        let journal = Journal {
            schema_version: "greppy.edit-journal.v1".into(),
            transaction_id: "tx-crash".into(),
            committed: true,
            entries: vec![JournalEntry {
                rel_path: "a.txt".into(),
                pre_image_file: "pre-0000.bin".into(),
                pre_sha256: sha256_hex(b"a1"),
                post_sha256: sha256_hex(b"a2"),
            }],
        };
        write_journal(dir.path(), &journal).unwrap();
        let out = recover(dir.path()).unwrap();
        assert_eq!(
            out,
            Recovery::RolledBack {
                transaction_id: "tx-crash".into(),
                files: 1
            }
        );
        assert_eq!(std::fs::read(dir.path().join("a.txt")).unwrap(), b"a1");
    }

    #[test]
    fn recover_refuses_foreign_edits() {
        let dir = ws();
        std::fs::write(dir.path().join("a.txt"), b"SOMEONE ELSE").unwrap();
        let jd = journal_dir(dir.path());
        std::fs::create_dir_all(&jd).unwrap();
        std::fs::write(jd.join("pre-0000.bin"), b"a1").unwrap();
        let journal = Journal {
            schema_version: "greppy.edit-journal.v1".into(),
            transaction_id: "tx-crash".into(),
            committed: true,
            entries: vec![JournalEntry {
                rel_path: "a.txt".into(),
                pre_image_file: "pre-0000.bin".into(),
                pre_sha256: sha256_hex(b"a1"),
                post_sha256: sha256_hex(b"a2"),
            }],
        };
        write_journal(dir.path(), &journal).unwrap();
        assert!(recover(dir.path()).is_err());
        assert_eq!(
            std::fs::read(dir.path().join("a.txt")).unwrap(),
            b"SOMEONE ELSE"
        );
    }

    #[test]
    fn nothing_to_recover() {
        let dir = ws();
        assert_eq!(recover(dir.path()).unwrap(), Recovery::NothingToRecover);
    }
}
