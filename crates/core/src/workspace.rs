//! Workspace / store locator shared across the drop-in wrapper and the
//! CLI dispatcher.
//!
//! Keeps `.greppy/graph.db` from polluting `grep -R` in the same
//! workspace, and ensures `index <path>` writes to the same location that
//! `search-graph` / `trace` / `search-code` / `semantic` read from.
//! The store directory is created mode 0700 and the DB file mode 0600;
//! symlinked store paths are refused.
//!
//! Rules:
//!
//! 1. The graph DB is **never** placed at `<repo>/.greppy/graph.db`.
//!    That path lives inside the workspace and trips every `grep -R .`
//!    over the SQLite file.
//! 2. Default store location:
//!    - `$XDG_CACHE_HOME/greppy/<ws-hash>/graph.db` if `XDG_CACHE_HOME`
//!      is set (or `$HOME/.cache/greppy/...` on Linux);
//!    - `$TMPDIR/greppy/<ws-hash>/graph.db` as fallback on Unix;
//!    - `%LOCALAPPDATA%/greppy/<ws-hash>/graph.db` on Windows
//!      (Tier 2 — not Tier 1 today; kept so the function compiles).
//! 3. Override via `GREPPY_STORE_DIR=/path/to/dir`; the workspace hash
//!    is still appended so different workspaces do not collide.
//! 4. `<ws-hash>` is the first 16 hex chars of
//!    `sha256(canonical_workspace_root)` — deterministic, not
//!    `DefaultHasher` (which uses a fixed stdlib key and is not stable
//!    across Rust versions).
//! 5. The store dir is created mode 0700; the DB file is created
//!    mode 0600. Existing dirs/files are chmod'd on every call (so a
//!    store created before this rule was applied is also tightened).
//!    Symlinks at either path are refused.
//!
//! Tier 1 today: macOS + Linux. Windows behaviour is documented but not
//! built/tested.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Canonicalise `path` lexically for the hash (resolve symlinks if
/// possible, otherwise absolute-path it). Best-effort: a path we cannot
/// resolve still gets a hash so writes succeed, just one the user cannot
/// reverse.
fn canonical_for_hash(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// Compute the workspace hash from a canonical workspace root.
///
/// The hash is the first 16 hex chars of `sha256(canonical_root)` —
/// deterministic across runs and Rust versions, unlike `DefaultHasher`.
pub fn workspace_hash(workspace_root: &Path) -> String {
    let canon = canonical_for_hash(workspace_root);
    let mut h = Sha256::new();
    h.update(canon.to_string_lossy().as_bytes());
    let digest = h.finalize();
    let hex = format!("{:x}", digest);
    hex.chars().take(16).collect()
}

/// Directory that holds this workspace's graph DB and sidecars, under
/// `GREPPY_STORE_DIR` or under the OS cache dir. The directory is
/// **not** created by this function — callers create it on first write
/// so we never leave an empty dir behind on read-only paths.
pub fn store_dir(workspace_root: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("GREPPY_STORE_DIR") {
        return PathBuf::from(p).join(workspace_hash(workspace_root));
    }

    // DATA directories, deliberately NOT cache directories (O7, 2026-07-06):
    // stores hold expensively-computed embeddings, and the OS is entitled to
    // purge cache locations at will — macOS cache maintenance wiped
    // `~/Library/Caches/greppy` mid-session repeatedly, silently degrading
    // semantic search. Our own TTL eviction keeps the footprint bounded, so
    // nothing is lost by moving to a data location.
    //
    // macOS:      ~/Library/Application Support/greppy/<ws-hash>
    // Linux XDG:  $XDG_DATA_HOME/greppy/<ws-hash>, or ~/.local/share/...
    // Windows:    %LOCALAPPDATA%\greppy\<ws-hash>  (not OS-purged)
    // Fallback:   $TMPDIR/greppy/<ws-hash>.
    //
    // A store still living at the pre-O7 cache location is MIGRATED
    // (renamed) to the data location once, so existing indexes and
    // embeddings survive the move.
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME") {
            let home = PathBuf::from(home);
            let hash = workspace_hash(workspace_root);
            let new = home
                .join("Library")
                .join("Application Support")
                .join("greppy")
                .join(&hash);
            let old = home
                .join("Library")
                .join("Caches")
                .join("greppy")
                .join(&hash);
            migrate_store_dir(&old, &new);
            return new;
        }
    }
    #[cfg(all(target_os = "linux", not(target_os = "android")))]
    {
        let data_base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share"))
            });
        if let Some(base) = data_base {
            let hash = workspace_hash(workspace_root);
            let new = base.join("greppy").join(&hash);
            let cache_base = std::env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")));
            if let Some(cb) = cache_base {
                migrate_store_dir(&cb.join("greppy").join(&hash), &new);
            }
            return new;
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            return PathBuf::from(local)
                .join("greppy")
                .join(workspace_hash(workspace_root));
        }
    }
    if let Some(tmp) = std::env::var_os("TMPDIR") {
        return PathBuf::from(tmp)
            .join("greppy")
            .join(workspace_hash(workspace_root));
    }
    // Last resort: a per-process tmp dir.
    std::env::temp_dir()
        .join("greppy")
        .join(workspace_hash(workspace_root))
}

/// One-time, best-effort migration of a store from the pre-O7 cache
/// location to the data location. A plain rename (same volume in practice);
/// silently a no-op when the old dir is absent, the new dir already exists,
/// or the rename fails (the caller then simply starts fresh at `new`).
fn migrate_store_dir(old: &Path, new: &Path) {
    if new.exists() || !old.exists() {
        return;
    }
    if let Some(parent) = new.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return;
        }
    }
    let _ = std::fs::rename(old, new);
}

/// Full path to the workspace's graph DB.
pub fn store_path(workspace_root: &Path) -> PathBuf {
    store_dir(workspace_root).join("graph.db")
}

/// Root directory that holds every workspace's per-hash store dir —
/// i.e. the `<cache>/greppy/` parent of `store_dir(<any-root>)`. This
/// is where [`cleanup_stale_stores`] scans for evictable stores.
///
/// Derived from `store_dir` of a throwaway path so the two never drift:
/// `store_dir` appends `<ws-hash>`, and this strips that last segment.
/// Returns `None` only in the degenerate case where `store_dir` has no
/// parent (it always does today).
pub fn store_cache_root() -> Option<PathBuf> {
    // Any path works: we only want the `<cache>/greppy/` prefix that
    // is common to every workspace's store dir.
    store_dir(Path::new("/")).parent().map(|p| p.to_path_buf())
}

/// Name of the per-store "last used" marker file. Written on every
/// query that opens the store to serve a read (via [`touch_lastused`]),
/// because a read-only query never bumps `graph.db`'s mtime, so the
/// mtime of the DB alone cannot tell a stale store from a recently used
/// one. [`cleanup_stale_stores`] evicts stores whose marker is older
/// than the TTL.
pub const LASTUSED_MARKER: &str = ".lastused";

/// Touch (create/overwrite) the `.lastused` marker inside `store_dir`
/// to record that the store was used *now*. Best-effort: any I/O error
/// is swallowed — failing to record a touch must never fail a query.
/// The write updates the file's mtime, which is what
/// [`cleanup_stale_stores`] reads.
pub fn touch_lastused(store_dir: &Path) {
    let marker = store_dir.join(LASTUSED_MARKER);
    // Overwrite the (tiny) file so its mtime advances to now. An empty
    // body is fine — only the mtime is read back.
    let _ = std::fs::write(&marker, b"");
}

/// Environment variable overriding the store eviction TTL, in **days**.
/// `0` disables eviction entirely. Absent/unparsable falls back to
/// [`STORE_TTL_DAYS_DEFAULT`].
pub const ENV_STORE_TTL_DAYS: &str = "GREPPY_STORE_TTL_DAYS";

/// Default store eviction TTL: **14 days**. A store dir whose
/// `.lastused` marker has not been touched within this window is
/// evicted on the next probabilistic cleanup pass.
pub const STORE_TTL_DAYS_DEFAULT: u64 = 14;

/// Resolve the store eviction TTL in **seconds** from
/// `GREPPY_STORE_TTL_DAYS` (default 14 days). Returns `0` when the env
/// var is `0`, which callers treat as "eviction disabled".
pub fn store_ttl_secs() -> u64 {
    let days = std::env::var(ENV_STORE_TTL_DAYS)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(STORE_TTL_DAYS_DEFAULT);
    days.saturating_mul(24 * 60 * 60)
}

/// Evict stale index stores under `cache_root` (the `<cache>/greppy/`
/// directory returned by [`store_cache_root`]).
///
/// Scans each immediate subdirectory `<cache_root>/<ws-hash>/` and
/// removes the whole store dir when its `.lastused` marker (see
/// [`touch_lastused`]) is older than `ttl_secs` — EXCEPT the `keep` dir
/// (the store the current invocation is using), which is never removed
/// even if its marker looks stale. A store dir with **no** marker at all
/// is treated as stale (legacy stores predating the marker, or a store
/// whose marker write failed) and is eligible for eviction once
/// old enough by dir mtime.
///
/// `ttl_secs == 0` disables eviction and returns `0` immediately.
///
/// Best-effort throughout: unreadable entries, un-removable dirs, and a
/// missing `cache_root` are skipped, never surfaced as errors. Returns
/// the number of store dirs removed.
pub fn cleanup_stale_stores(cache_root: &Path, ttl_secs: u64, keep: &Path) -> usize {
    use std::time::{Duration, SystemTime};

    if ttl_secs == 0 {
        return 0;
    }
    let read_dir = match std::fs::read_dir(cache_root) {
        Ok(rd) => rd,
        Err(_) => return 0, // cache root absent or unreadable: nothing to do
    };
    // Canonicalise `keep` so a comparison is robust to symlinks and
    // relative-vs-absolute forms. Fall back to the raw path if it does
    // not resolve (e.g. does not exist yet).
    let keep_canon = keep.canonicalize().unwrap_or_else(|_| keep.to_path_buf());
    let now = SystemTime::now();
    let cutoff = Duration::from_secs(ttl_secs);
    let mut removed = 0usize;

    for entry in read_dir.flatten() {
        let dir = entry.path();
        // Only consider directories (the per-workspace store dirs).
        if !dir.is_dir() {
            continue;
        }
        // Never evict the store the current invocation is serving from.
        let dir_canon = dir.canonicalize().unwrap_or_else(|_| dir.clone());
        if dir_canon == keep_canon {
            continue;
        }
        // Age = time since the store was last used. Prefer the dedicated
        // `.lastused` marker's mtime; if it is absent, fall back to the
        // store dir's own mtime so legacy stores can still be reaped.
        let marker = dir.join(LASTUSED_MARKER);
        let last_used = std::fs::metadata(&marker)
            .and_then(|m| m.modified())
            .or_else(|_| entry.metadata().and_then(|m| m.modified()));
        let last_used = match last_used {
            Ok(t) => t,
            Err(_) => continue, // cannot determine age: leave it alone
        };
        let age = match now.duration_since(last_used) {
            Ok(a) => a,
            Err(_) => continue, // clock skew / future mtime: leave it alone
        };
        if age > cutoff && std::fs::remove_dir_all(&dir).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// Return the project-identity string for `start`: the basename of the
/// canonical repo root (walking up looking for `.git`, `Cargo.toml`,
/// or `pyproject.toml`). Falls back to the basename of `start` itself,
/// then to `"default"` if the basename is empty.
///
/// This is the *one* function the CLI dispatcher and the indexer use
/// to derive the `project` column for the store.
/// Using it consistently means a user can `greppy index /path/to/repo`
/// from any cwd, then `greppy search-code Q` from a subdir, and the
/// project identity matches.
pub fn project_identity(start: &Path) -> String {
    let canonical = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
    let mut cur = canonical.clone();
    let mut found_marker = false;
    loop {
        if cur.join(".git").exists()
            || cur.join("Cargo.toml").exists()
            || cur.join("pyproject.toml").exists()
        {
            found_marker = true;
            break;
        }
        match cur.parent() {
            Some(p) if !p.as_os_str().is_empty() && p != cur => cur = p.to_path_buf(),
            _ => break,
        }
    }
    // When no marker exists in the chain, return the basename of the
    // original `start` (not the walked-up `cur`, which would be `/`
    // or `private`).
    let final_path = if found_marker { &cur } else { &canonical };
    final_path
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "default".to_string())
}

/// Mode for newly-created store/sidecar directories. The actual `mkdir`
/// lives in [`ensure_store_dir`] and the DB-mode chmod in
/// [`ensure_db_mode`] — both are wired from callers so the helper
/// isn't dead code.
#[deprecated(note = "use ensure_store_dir / ensure_db_mode instead")]
pub fn dir_mode_default() -> u32 {
    0o700
}

/// Create the store directory at `dir` (and any missing parents),
/// mode 0700. Refuses to operate on a symlink. Idempotent: an existing
/// directory is chmod'd to 0700 (so a store created before this
/// hardening is tightened).
pub fn ensure_store_dir(dir: &Path) -> std::io::Result<()> {
    if let Ok(md) = std::fs::symlink_metadata(dir) {
        if md.file_type().is_symlink() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("refusing symlink at store dir {}", dir.display()),
            ));
        }
        set_mode_700(dir)?;
        return Ok(());
    }
    std::fs::create_dir_all(dir)?;
    set_mode_700(dir)
}

/// Set the DB file at `path` to mode 0600. Refuses to operate on a
/// symlink. No-op if the file does not yet exist (the writer creates
/// it with the right mode via `OpenOptions::mode`).
pub fn ensure_db_mode(path: &Path) -> std::io::Result<()> {
    let md = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    if md.file_type().is_symlink() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("refusing symlink at db path {}", path.display()),
        ));
    }
    set_mode_600(path)
}

#[cfg(unix)]
fn set_mode_700(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
}

#[cfg(unix)]
fn set_mode_600(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
}

#[cfg(not(unix))]
fn set_mode_700(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(not(unix))]
fn set_mode_600(_path: &Path) -> std::io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_hash_is_deterministic_and_changes_with_path() {
        let h1 = workspace_hash(Path::new("/tmp/repo-a"));
        let h2 = workspace_hash(Path::new("/tmp/repo-a"));
        let h3 = workspace_hash(Path::new("/tmp/repo-b"));
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h1.len(), 16);
        assert!(h1.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn store_dir_never_returns_path_inside_workspace_root() {
        // The store directory must NOT be `<root>/.greppy/graph.db`.
        // We assert it is not under the workspace root.
        let tmp = tempdir_root("greppy-locator-test");
        let d = store_dir(&tmp);
        assert!(
            !d.starts_with(&tmp),
            "store_dir {d:?} must not be inside workspace root {tmp:?}"
        );
        // And it must include a workspace hash segment.
        let tail = d.file_name().unwrap().to_string_lossy().to_string();
        assert_eq!(tail.len(), 16, "store_dir tail must be a 16-char hex hash");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn store_path_is_store_dir_plus_graph_db() {
        let tmp = tempdir_root("greppy-store-path");
        let sp = store_path(&tmp);
        assert!(sp.ends_with("graph.db"));
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn ensure_store_dir_creates_with_0700() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempdir_root("greppy-ensure-store-dir");
        let new_dir = tmp.join("store");
        ensure_store_dir(&new_dir).unwrap();
        let md = std::fs::metadata(&new_dir).unwrap();
        assert_eq!(
            md.permissions().mode() & 0o777,
            0o700,
            "store dir must be 0700; got {:o}",
            md.permissions().mode() & 0o777
        );
        // Idempotent chmod of a pre-existing 0755 dir.
        std::fs::set_permissions(&new_dir, std::fs::Permissions::from_mode(0o755)).unwrap();
        ensure_store_dir(&new_dir).unwrap();
        let md = std::fs::metadata(&new_dir).unwrap();
        assert_eq!(
            md.permissions().mode() & 0o777,
            0o700,
            "ensure_store_dir must re-tighten to 0700 even when pre-existing"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn ensure_store_dir_refuses_symlink() {
        let tmp = tempdir_root("greppy-symlink-store-dir");
        let real = tmp.join("real");
        std::fs::create_dir_all(&real).unwrap();
        let link = tmp.join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let r = ensure_store_dir(&link);
        assert!(r.is_err(), "must refuse symlinked store dir");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn ensure_db_mode_chmods_to_0600_and_refuses_symlink() {
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempdir_root("greppy-ensure-db-mode");
        let db = tmp.join("graph.db");
        std::fs::write(&db, b"sqlite").unwrap();
        std::fs::set_permissions(&db, std::fs::Permissions::from_mode(0o644)).unwrap();
        ensure_db_mode(&db).unwrap();
        let md = std::fs::metadata(&db).unwrap();
        assert_eq!(
            md.permissions().mode() & 0o777,
            0o600,
            "DB must be 0600 after ensure_db_mode; got {:o}",
            md.permissions().mode() & 0o777
        );
        // Symlink refusal.
        let link = tmp.join("graph.db.link");
        std::os::unix::fs::symlink(&db, &link).unwrap();
        let r = ensure_db_mode(&link);
        assert!(r.is_err(), "must refuse symlinked DB");
        // Non-existent file: silent no-op (writer will create with the
        // right mode via OpenOptions::mode).
        let phantom = tmp.join("nope.db");
        ensure_db_mode(&phantom).unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn tempdir_root(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "{tag}-{}-{}",
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
    fn project_identity_walks_up_to_git_or_cargo() {
        let tmp = tempdir_root("greppy-projid-git");
        let root = tmp.join("repo");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        let nested = root.join("a/b/c");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(project_identity(&nested), "repo");
        let _ = std::fs::remove_dir_all(&tmp);

        let tmp = tempdir_root("greppy-projid-cargo");
        let root = tmp.join("rustproj");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"x\"\n").unwrap();
        let nested = root.join("src/deep");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(project_identity(&nested), "rustproj");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn project_identity_falls_back_to_basename_when_no_marker() {
        let tmp = tempdir_root("greppy-projid-nomarker");
        let naked = tmp.join("naked/dir");
        std::fs::create_dir_all(&naked).unwrap();
        assert_eq!(project_identity(&naked), "dir");
        let _ = std::fs::remove_dir_all(&tmp);
    }

    /// Back-date `path`'s mtime to `t` via `touch -t` (Tier-1 targets are
    /// Unix; no `filetime` dep needed). Mirrors the helper the sidecar
    /// cleanup tests use.
    fn set_mtime(path: &Path, t: std::time::SystemTime) {
        let secs = t
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let day_secs = secs.rem_euclid(86_400);
        let hour = day_secs / 3600;
        let minute = (day_secs % 3600) / 60;
        let second = day_secs % 60;
        let mut days = secs.div_euclid(86_400);
        let mut year: i64 = 1970;
        loop {
            let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
            let year_days = if leap { 366 } else { 365 };
            if days >= year_days {
                days -= year_days;
                year += 1;
            } else {
                break;
            }
        }
        let month_days = if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
            [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        } else {
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        };
        let mut month = 1;
        for &md in &month_days {
            if days >= md {
                days -= md;
                month += 1;
            } else {
                break;
            }
        }
        let day = days + 1;
        let stamp = format!("{year:04}{month:02}{day:02}{hour:02}{minute:02}.{second:02}");
        let _ = std::process::Command::new("touch")
            .arg("-t")
            .arg(&stamp)
            .arg(path)
            .status();
    }

    /// Write a `.lastused` marker inside `store_dir` and back-date its
    /// mtime by `age_secs` so we can drive `cleanup_stale_stores`
    /// deterministically without sleeping.
    fn write_marker_aged(store_dir: &Path, age_secs: u64) {
        let marker = store_dir.join(LASTUSED_MARKER);
        std::fs::write(&marker, b"").unwrap();
        let when = std::time::SystemTime::now() - std::time::Duration::from_secs(age_secs);
        set_mtime(&marker, when);
    }

    #[test]
    fn cleanup_stale_stores_removes_old_keeps_fresh_and_keep_dir() {
        let cache = tempdir_root("greppy-cleanup-stores");
        // An OLD store: marker aged well past the TTL → evicted. Use a
        // large back-date (10 days) so a timezone skew in the `touch -t`
        // helper cannot flip its side of the 1-day cutoff.
        let old = cache.join("aaaaaaaaaaaaaaaa");
        std::fs::create_dir_all(&old).unwrap();
        std::fs::write(old.join("graph.db"), b"sqlite").unwrap();
        write_marker_aged(&old, 10 * 86_400);
        // A FRESH store: marker written just now (not back-dated) →
        // survives. Leaving it at creation time avoids any TZ-skew
        // ambiguity around the cutoff.
        let fresh = cache.join("bbbbbbbbbbbbbbbb");
        std::fs::create_dir_all(&fresh).unwrap();
        std::fs::write(fresh.join("graph.db"), b"sqlite").unwrap();
        touch_lastused(&fresh);
        // The KEEP store: even though its marker is old, it is the store
        // the current invocation is using and must NOT be removed.
        let keep = cache.join("cccccccccccccccc");
        std::fs::create_dir_all(&keep).unwrap();
        std::fs::write(keep.join("graph.db"), b"sqlite").unwrap();
        write_marker_aged(&keep, 10 * 86_400);

        // TTL = 1 day: only the old, non-keep store is stale.
        let removed = cleanup_stale_stores(&cache, 86_400, &keep);
        assert_eq!(removed, 1, "exactly the old non-keep store is removed");
        assert!(!old.exists(), "old store must be evicted");
        assert!(fresh.exists(), "fresh store must survive");
        assert!(keep.exists(), "keep store must survive even when stale");

        // ttl_secs == 0 disables eviction entirely, even for a store
        // whose marker is far past.
        write_marker_aged(&fresh, 10 * 86_400);
        let removed0 = cleanup_stale_stores(&cache, 0, &keep);
        assert_eq!(removed0, 0, "ttl 0 disables eviction");
        assert!(fresh.exists(), "nothing removed when eviction disabled");

        let _ = std::fs::remove_dir_all(&cache);
    }

    #[test]
    fn store_ttl_secs_defaults_to_14_days() {
        // Guard against another test's env mutation leaking in.
        std::env::remove_var(ENV_STORE_TTL_DAYS);
        assert_eq!(store_ttl_secs(), STORE_TTL_DAYS_DEFAULT * 24 * 60 * 60);
        assert_eq!(store_ttl_secs(), 14 * 86_400);
    }

    #[test]
    fn touch_lastused_creates_recent_marker() {
        let dir = tempdir_root("greppy-touch-lastused");
        touch_lastused(&dir);
        let marker = dir.join(LASTUSED_MARKER);
        assert!(marker.exists(), ".lastused marker must be created");
        let mtime = std::fs::metadata(&marker).unwrap().modified().unwrap();
        let age = std::time::SystemTime::now().duration_since(mtime).unwrap();
        assert!(age.as_secs() < 60, "marker mtime must be recent");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
