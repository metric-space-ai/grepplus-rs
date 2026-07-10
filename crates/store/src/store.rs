//! Connection wrapper, open modes, and the high-level entry point.

use std::path::Path;

use rusqlite::Connection;

use crate::migrate;
use crate::store_error::{Error, Result};

/// Open-mode flags for [`Store::open`].
#[derive(Debug, Clone, Copy)]
pub struct OpenOptions {
    /// Open the database read-only. SQLite will refuse any write attempt.
    pub read_only: bool,
    /// Run `PRAGMA integrity_check` after opening a writable store.
    pub integrity_check: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        Self {
            read_only: false,
            integrity_check: true,
        }
    }
}

impl OpenOptions {
    pub const fn read_only() -> Self {
        Self {
            read_only: true,
            integrity_check: false,
        }
    }

    /// Writable query hotpath open: apply migrations and allow small writes,
    /// but skip the full DB integrity scan reserved for index writers.
    pub const fn query_writer() -> Self {
        Self {
            read_only: false,
            integrity_check: false,
        }
    }
}

/// Handle to an open graph store.
///
/// A `Store` owns a single `rusqlite::Connection`. It is **not** `Clone` —
/// cloning a connection across threads requires `Send + Sync`, which
/// `rusqlite::Connection` provides only behind a `Mutex`. We deliberately
/// keep the type single-threaded; a `StorePool` handles
/// concurrent reads.
pub struct Store {
    conn: Connection,
    // Shared lifecycle lease prevents GC from renaming/removing the workspace
    // directory for as long as this SQLite handle is alive. In-memory and
    // non-workspace test databases legitimately have no lease.
    _lifecycle: Option<greppy_core::cache::FileLock>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Store").finish_non_exhaustive()
    }
}

impl Store {
    /// Open an in-memory database and run migrations.
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        Self::from_connection(conn)
    }

    /// Open a database at `path`. Creates the file if it does not exist.
    pub fn open(path: &Path) -> Result<Self> {
        let opts = OpenOptions::default();
        Self::open_with(path, opts)
    }

    /// Open with explicit options.
    pub fn open_with(path: &Path, opts: OpenOptions) -> Result<Self> {
        let lifecycle = workspace_lifecycle_for_path(path).map_err(|e| Error::Io {
            context: format!("acquire lifecycle lease for {}", path.display()),
            source: e,
        })?;
        let conn = if opts.read_only {
            Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY).map_err(
                |e| Error::Io {
                    context: format!("open read-only {}", path.display()),
                    source: std::io::Error::other(e.to_string()),
                },
            )?
        } else {
            Connection::open(path).map_err(|e| Error::Io {
                context: format!("open {}", path.display()),
                source: std::io::Error::other(e.to_string()),
            })?
        };
        if let Some(parent) = path.parent() {
            greppy_core::cache::touch_last_used_dir(parent);
        }
        Self::from_connection_with_options_and_lease(conn, opts, lifecycle)
    }

    fn from_connection(conn: Connection) -> Result<Self> {
        Self::from_connection_with_options(conn, OpenOptions::default())
    }

    fn from_connection_with_options(conn: Connection, opts: OpenOptions) -> Result<Self> {
        Self::from_connection_with_options_and_lease(conn, opts, None)
    }

    fn from_connection_with_options_and_lease(
        conn: Connection,
        opts: OpenOptions,
        lifecycle: Option<greppy_core::cache::FileLock>,
    ) -> Result<Self> {
        // Performance pragmas for the WRITE path (i.e. `greppy index`).
        // Default SQLite is journal_mode=DELETE + synchronous=FULL, which
        // fsyncs on every transaction commit. The indexer commits once per
        // file (batching), so a 423-file repo paid ~423 fsyncs — the
        // dominant cost of cold indexing (measured: ~1.2 s of a 2.65 s
        // python_large index was fsync). WAL + synchronous=NORMAL is the
        // standard crash-safe bulk-write configuration: it fsyncs only at
        // checkpoints, not per commit, and WAL is atomic so a crash can never
        // corrupt the DB (worst case loses the last checkpoint, and the index
        // is a rebuildable cache anyway). temp_store=MEMORY keeps FTS merge
        // scratch off disk. Readers don't set these (they open read-only and
        // tolerate whatever the DB has).
        if !opts.read_only {
            // journal_mode returns a row; use query_row, not execute.
            let _: String = conn
                .query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))
                .unwrap_or_default();
            let _ = conn.execute_batch(
                "PRAGMA synchronous = NORMAL; PRAGMA temp_store = MEMORY; PRAGMA cache_size = -16000;",
            );
        }
        // Apply pending migrations up-front — but only on writers.
        // A read-only open against a DB whose persisted
        // `schema_version` is older than `CURRENT_VERSION` would
        // attempt to CREATE / ALTER tables on a read-only
        // connection and fail (this is what the
        // `freshness-probe` bench was tripping on 2026-06-29).
        // Readers tolerate whatever schema the DB has; the
        // `greppy index` writer upgrades on the next write.
        if !opts.read_only {
            migrate::migrate(&conn)?;
        }
        let s = Self {
            conn,
            _lifecycle: lifecycle,
        };
        // Verify integrity on WRITE opens (i.e. `greppy index`) only.
        // `PRAGMA integrity_check` is O(db-size) — hundreds of ms on a large
        // store — so running it on every READ-ONLY open (the query hotpath:
        // who-calls / find-usages / trace / the grep freshness gate) would make
        // every greppy invocation pay that scan. The writer verifies before
        // it mutates; a read-only query against a genuinely corrupt DB still
        // fails loudly at the offending statement (SQLite errors on a malformed
        // image), so it never silently returns wrong data. This keeps the
        // agent-facing query path fast (the token-efficiency benchmark showed
        // per-open integrity_check was the dominant query latency).
        if !opts.read_only && opts.integrity_check {
            s.integrity_check()?;
        }
        if !opts.read_only {
            // Evidence packs are intentionally ephemeral. Prune them on every
            // writable maintenance/open path so an actively used workspace
            // cannot retain expired payloads indefinitely.
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let _ = s
                .conn
                .execute("DELETE FROM expand_packs WHERE expires_at <= ?1", [now]);
        }
        Ok(s)
    }

    /// Returns the current schema version recorded in `schema_meta`.
    pub fn schema_version(&self) -> Result<u32> {
        let v: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'schema_version'",
                [],
                |row| row.get(0),
            )
            .ok();
        Ok(v.and_then(|s| s.parse().ok()).unwrap_or(0))
    }

    /// Run SQLite's built-in `PRAGMA integrity_check`. Returns `Ok(())`
    /// when the database reports `ok`, otherwise returns the diagnostic
    /// text as an error.
    pub fn integrity_check(&self) -> Result<()> {
        let rows = self.integrity_check_messages()?;
        match rows.as_slice() {
            [single] if single == "ok" => Ok(()),
            other => Err(Error::Store(format!("integrity_check reported: {other:?}"))),
        }
    }

    /// Return SQLite's raw `PRAGMA integrity_check` messages.
    ///
    /// Diagnostics use this instead of [`Store::integrity_check`] so they can
    /// report an unhealthy store without hiding the exact SQLite messages
    /// behind an early error.
    pub fn integrity_check_messages(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare("PRAGMA integrity_check")?;
        let rows: Vec<String> = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Borrow the underlying connection.
    ///
    /// Public so peer crates (search, freshness, …) can issue raw SQL
    /// without us wrapping every query in a typed method. Callers must
    /// treat the returned `&Connection` as read-only-by-convention; the
    /// store's own helpers (`insert_node`, `insert_edge`, …) own the
    /// write paths.
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Begin a write transaction. Rolls back on drop if neither
    /// `commit()` nor `rollback()` is called explicitly.
    pub fn transaction(&mut self) -> Result<Transaction<'_>> {
        let tx = self.conn.transaction()?;
        Ok(Transaction { tx })
    }
}

fn workspace_lifecycle_for_path(
    path: &Path,
) -> std::io::Result<Option<greppy_core::cache::FileLock>> {
    let Some(parent) = path.parent() else {
        return Ok(None);
    };
    let Ok(manifest) = greppy_core::cache::read_store_manifest(parent) else {
        return Ok(None);
    };
    greppy_core::cache::acquire_workspace_lifecycle(
        &manifest.canonical_root,
        greppy_core::cache::LockMode::Shared,
        false,
    )
    .inspect(|lease| {
        debug_assert!(
            lease.is_some(),
            "blocking lifecycle lock must return a guard"
        );
    })
}

/// A write transaction. Use `Store::transaction()` to acquire.
pub struct Transaction<'a> {
    tx: rusqlite::Transaction<'a>,
}

impl<'a> Transaction<'a> {
    pub fn commit(self) -> Result<()> {
        self.tx.commit().map_err(Error::Sqlite)
    }

    pub fn rollback(self) -> Result<()> {
        self.tx.rollback().map_err(Error::Sqlite)
    }

    /// Borrow the underlying rusqlite transaction. Crate-internal.
    pub(crate) fn raw(&self) -> &rusqlite::Transaction<'a> {
        &self.tx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_memory_creates_db_with_schema() {
        let s = Store::open_memory().unwrap();
        // Schema is at CURRENT_VERSION after migrations run.
        assert_eq!(s.schema_version().unwrap(), crate::migrate::CURRENT_VERSION);
    }

    #[test]
    fn integrity_check_passes_on_fresh_db() {
        let s = Store::open_memory().unwrap();
        s.integrity_check().unwrap();
    }

    #[test]
    fn open_persistent_path_round_trip() {
        let tmp = tempdir_via_env();
        let path = tmp.join("test.db");
        {
            let s = Store::open(&path).unwrap();
            assert_eq!(s.schema_version().unwrap(), crate::migrate::CURRENT_VERSION);
        }
        let s2 = Store::open(&path).unwrap();
        assert_eq!(
            s2.schema_version().unwrap(),
            crate::migrate::CURRENT_VERSION
        );
        s2.integrity_check().unwrap();
    }

    fn tempdir_via_env() -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "greppy-store-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let p = base.join(unique);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
