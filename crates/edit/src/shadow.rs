//! Shadow-worktree publish: apply and VALIDATE in an isolated copy of the
//! workspace, then journal-publish into the real workspace only if every
//! validator passed.
//!
//! The shadow is a plain recursive copy excluding `.git` (correct for
//! validators that build/test the working tree; the copy cost is the price
//! of running tests against not-yet-published edits). Validators run with
//! the shadow as cwd, argv-only, no shell.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use crate::certificate::ValidatorReport;
use crate::journal::{publish_journal, FilePublication};
use crate::plan::PlanValidator;
use greppy_core::{Error, Result};

/// Copy the workspace (minus `.git`) into a temp dir.
fn copy_workspace(workspace_root: &Path) -> Result<(tempfile::TempDir, PathBuf)> {
    let tmp = tempfile::Builder::new()
        .prefix("greppy-shadow.")
        .tempdir()
        .map_err(|source| Error::Io {
            context: "create shadow dir".into(),
            source,
        })?;
    let dst = tmp.path().join("ws");
    copy_dir(workspace_root, &dst)?;
    Ok((tmp, dst))
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst).map_err(|source| Error::Io {
        context: format!("create {}", dst.display()),
        source,
    })?;
    let entries = std::fs::read_dir(src).map_err(|source| Error::Io {
        context: format!("read {}", src.display()),
        source,
    })?;
    for entry in entries {
        let entry = entry.map_err(|source| Error::Io {
            context: "read dir entry".into(),
            source,
        })?;
        let name = entry.file_name();
        if name == ".git" || name == ".greppy-edit-journal" || name == ".greppy-edit.lock" {
            continue;
        }
        let src_path = entry.path();
        let dst_path = dst.join(&name);
        let file_type = entry.file_type().map_err(|source| Error::Io {
            context: format!("stat {}", src_path.display()),
            source,
        })?;
        if file_type.is_symlink() {
            continue; // shadow never follows symlinks
        }
        if file_type.is_dir() {
            copy_dir(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path).map_err(|source| Error::Io {
                context: format!("copy {}", src_path.display()),
                source,
            })?;
        }
    }
    Ok(())
}

/// Run one validator in `cwd`. argv-only; no shell interpretation.
pub fn run_validator(cwd: &Path, validator: &PlanValidator) -> Result<ValidatorReport> {
    let Some((program, args)) = validator.argv.split_first() else {
        return Err(Error::Invalid("validator argv is empty".into()));
    };
    let mut child = std::process::Command::new(program)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|source| Error::Io {
            context: format!("spawn validator {program}"),
            source,
        })?;
    let deadline = std::time::Instant::now() + Duration::from_secs(validator.timeout_seconds);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return Ok(ValidatorReport {
                    argv: validator.argv.clone(),
                    exit_code: status.code().unwrap_or(-1),
                    timed_out: false,
                })
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Ok(ValidatorReport {
                        argv: validator.argv.clone(),
                        exit_code: -1,
                        timed_out: true,
                    });
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(source) => {
                return Err(Error::Io {
                    context: "wait for validator".into(),
                    source,
                })
            }
        }
    }
}

/// Apply `publications` in a shadow copy, run `validators` there, and
/// journal-publish to the real workspace only when all pass. Returns the
/// validator reports and whether publication happened.
pub fn shadow_validate_and_publish(
    workspace_root: &Path,
    transaction_id: &str,
    publications: &[FilePublication],
    validators: &[PlanValidator],
    publish: bool,
) -> Result<(Vec<ValidatorReport>, bool)> {
    let (_tmp, shadow) = copy_workspace(workspace_root)?;
    for p in publications {
        let dst = shadow.join(&p.rel_path);
        if let Some(parent) = dst.parent() {
            std::fs::create_dir_all(parent).map_err(|source| Error::Io {
                context: format!("create {}", parent.display()),
                source,
            })?;
        }
        std::fs::write(&dst, &p.content).map_err(|source| Error::Io {
            context: format!("shadow write {}", dst.display()),
            source,
        })?;
    }
    let mut reports = Vec::new();
    let mut all_ok = true;
    for v in validators {
        let report = run_validator(&shadow, v)?;
        all_ok &= report.exit_code == 0 && !report.timed_out;
        reports.push(report);
    }
    if !all_ok || !publish {
        return Ok((reports, false));
    }
    publish_journal(workspace_root, transaction_id, publications)?;
    Ok((reports, true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256_hex;

    #[test]
    fn passing_validator_publishes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"v1").unwrap();
        let pubs = vec![FilePublication {
            rel_path: "a.txt".into(),
            expected_live_sha256: sha256_hex(b"v1"),
            content: b"v2".to_vec(),
        }];
        let validators = vec![PlanValidator {
            argv: vec!["true".into()],
            timeout_seconds: 10,
        }];
        let (reports, published) =
            shadow_validate_and_publish(dir.path(), "tx", &pubs, &validators, true).unwrap();
        assert!(published);
        assert_eq!(reports[0].exit_code, 0);
        assert_eq!(std::fs::read(dir.path().join("a.txt")).unwrap(), b"v2");
    }

    #[test]
    fn failing_validator_publishes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"v1").unwrap();
        let pubs = vec![FilePublication {
            rel_path: "a.txt".into(),
            expected_live_sha256: sha256_hex(b"v1"),
            content: b"v2".to_vec(),
        }];
        let validators = vec![PlanValidator {
            argv: vec!["false".into()],
            timeout_seconds: 10,
        }];
        let (reports, published) =
            shadow_validate_and_publish(dir.path(), "tx", &pubs, &validators, true).unwrap();
        assert!(!published);
        assert_ne!(reports[0].exit_code, 0);
        assert_eq!(std::fs::read(dir.path().join("a.txt")).unwrap(), b"v1");
    }

    #[test]
    fn validator_sees_the_edited_shadow_not_the_workspace() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"v1").unwrap();
        let pubs = vec![FilePublication {
            rel_path: "a.txt".into(),
            expected_live_sha256: sha256_hex(b"v1"),
            content: b"v2".to_vec(),
        }];
        // grep exits 0 only if the shadow copy contains the NEW content
        let validators = vec![PlanValidator {
            argv: vec!["grep".into(), "-q".into(), "v2".into(), "a.txt".into()],
            timeout_seconds: 10,
        }];
        let (_, published) =
            shadow_validate_and_publish(dir.path(), "tx", &pubs, &validators, true).unwrap();
        assert!(published);
    }

    #[test]
    fn timeout_kills_and_refuses() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), b"v1").unwrap();
        let pubs = vec![FilePublication {
            rel_path: "a.txt".into(),
            expected_live_sha256: sha256_hex(b"v1"),
            content: b"v2".to_vec(),
        }];
        let validators = vec![PlanValidator {
            argv: vec!["sleep".into(), "30".into()],
            timeout_seconds: 1,
        }];
        let (reports, published) =
            shadow_validate_and_publish(dir.path(), "tx", &pubs, &validators, true).unwrap();
        assert!(!published);
        assert!(reports[0].timed_out);
        assert_eq!(std::fs::read(dir.path().join("a.txt")).unwrap(), b"v1");
    }
}
