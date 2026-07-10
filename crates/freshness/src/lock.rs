//! Crash-safe one-writer locking backed by the operating system.
//!
//! Lock files live in greppy's non-evictable `locks/` namespace.  The kernel
//! releases the advisory lock when a process exits, so there is no PID-age
//! takeover and a legitimate long-running embedding build can never be
//! displaced after an arbitrary timeout.

use std::path::{Path, PathBuf};

use thiserror::Error;

use greppy_core::Error as CoreError;

#[derive(Debug, Error)]
pub enum LockError {
    #[error("io: {context}: {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },
    #[error("lock held by another writer; path: {path}")]
    Held { path: PathBuf },
}

/// RAII writer guard. Dropping the underlying file releases the OS lock.
#[derive(Debug)]
pub struct Lock {
    _inner: greppy_core::cache::FileLock,
    path: PathBuf,
}

impl Lock {
    pub fn path(&self) -> &Path {
        &self.path
    }
}

pub fn try_acquire(target: &Path) -> std::result::Result<Lock, LockError> {
    acquire_impl(target, true)
}

pub fn acquire(target: &Path) -> std::result::Result<Lock, LockError> {
    acquire_impl(target, false)
}

fn acquire_impl(target: &Path, nonblocking: bool) -> std::result::Result<Lock, LockError> {
    let name = writer_lock_name(target);
    let path = greppy_core::cache::locks_root().join(&name);
    match greppy_core::cache::acquire_named_lock(
        &name,
        greppy_core::cache::LockMode::Exclusive,
        nonblocking,
    )
    .map_err(|source| LockError::Io {
        context: format!("acquire writer lock for {}", target.display()),
        source,
    })? {
        Some(inner) => Ok(Lock {
            _inner: inner,
            path,
        }),
        None => Err(LockError::Held { path }),
    }
}

/// Compatibility alias.
pub fn try_lock(target: &Path) -> std::result::Result<Lock, LockError> {
    try_acquire(target)
}

pub fn lock_path_for(target: &Path) -> PathBuf {
    greppy_core::cache::locks_root().join(writer_lock_name(target))
}

fn writer_lock_name(target: &Path) -> String {
    let store_dir = target.parent().unwrap_or(target);
    let id = store_dir
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| s.len() == 16 && s.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(str::to_string)
        .unwrap_or_else(|| greppy_core::workspace::workspace_hash(store_dir));
    format!("workspace-{id}.writer")
}

impl From<LockError> for CoreError {
    fn from(e: LockError) -> Self {
        match e {
            LockError::Io { context, source } => greppy_core::Error::io(context, source),
            LockError::Held { path } => {
                greppy_core::Error::Lock(format!("held: {}", path.display()))
            }
        }
    }
}

pub fn with_lock<T, F>(target: &Path, f: F) -> std::result::Result<T, LockError>
where
    F: FnOnce() -> std::result::Result<T, LockError>,
{
    let _guard = acquire(target)?;
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV: Mutex<()> = Mutex::new(());

    fn tempdir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "greppy-os-lock-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn lock_acquire_contention_and_release() {
        let _env = ENV.lock().unwrap();
        let dir = tempdir();
        std::env::set_var("GREPPY_STORE_DIR", dir.join("data"));
        let target = dir.join("graph.db");
        let first = try_acquire(&target).unwrap();
        assert!(matches!(try_acquire(&target), Err(LockError::Held { .. })));
        drop(first);
        assert!(try_acquire(&target).is_ok());
        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn old_lock_file_without_live_os_lock_is_harmless() {
        let _env = ENV.lock().unwrap();
        let dir = tempdir();
        std::env::set_var("GREPPY_STORE_DIR", dir.join("data"));
        let target = dir.join("graph.db");
        let path = lock_path_for(&target);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"legacy pid and ancient timestamp").unwrap();
        assert!(try_acquire(&target).is_ok());
        std::env::remove_var("GREPPY_STORE_DIR");
        let _ = std::fs::remove_dir_all(dir);
    }
}
