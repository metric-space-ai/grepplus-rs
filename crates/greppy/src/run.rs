//! Drop-in grep runner with optional semantic augmentation.
//!
//! This module is shared between:
//! - `greppy-grep` (the dedicated drop-in binary at `crates/greppy/src/main.rs`)
//! - `greppy` (the unified CLI dispatcher at `crates/cli/src/lib.rs::dispatch_grep`)
//!
//! Both call [`run_with_optional_augment`] so the bare-flag form
//! (`greppy -R foo .`) and the explicit-subcommand form
//! (`greppy grep -R foo .`) get the same heuristic + freshness
//! behaviour.

use std::ffi::OsString;
use std::path::Path;

use greppy_core::error::Error;
use greppy_core::workspace as workspace_locator;
use greppy_freshness::FreshnessOutcome;
use greppy_store::OpenOptions;

use crate::heuristic::{classify, FreshnessGate, GrepArgs, Mode};
use crate::sidecar;

/// Run real grep, then if the heuristic + freshness gate allow, write
/// a sidecar (and optionally append one labelled line to stdout).
///
/// Returns the real-grep exit code (modulo signal handling).
///
/// Drop-in contract: when real `grep`
/// returned a non-zero exit code (no matches, or an error), no synthetic
/// semantic content is produced: no sidecar, no synthetic stdout line.
/// The exit code and stdout/stderr are returned byte-exactly as real grep
/// produced them.
pub fn run_with_optional_augment(
    real_grep: &Path,
    argv: &[String],
    args: &GrepArgs,
) -> Result<i32, Error> {
    let real_exit = crate::run_grep(real_grep, argv)?;
    touch_passthrough_store();

    // Real-grep miss/error must not trigger
    // any visible semantic output. A non-zero exit code skips the
    // augment entirely (no sidecar, no synthetic line); the gate is
    // consulted only on a real-grep match.
    if real_exit != 0 {
        return Ok(real_exit);
    }

    let gate = freshness_gate(args);
    let mode = classify(args, gate);

    if matches!(mode, Mode::Sidecar | Mode::VisibleAugment) {
        if let Ok(cwd) = std::env::current_dir() {
            let workspace_root = workspace_locator::resolve_workspace_root(&cwd);
            // Augment errors are non-fatal; we never let them bubble
            // up as a real-grep exit-code change.
            let _ = run_augment(args, mode, &workspace_root, real_grep, argv);
        }
    }

    Ok(real_exit)
}

/// `OsString` argv variant of [`run_with_optional_augment`].
///
/// The drop-in `greppy-grep` entrypoint forwards
/// the original `OsString` argv to real grep byte-for-byte so it can
/// never panic on argv it cannot UTF-8-decode. The `GrepArgs`
/// classifier still operates on a best-effort lossy view for the
/// augmentation decision ONLY — the bytes that reach real grep are the
/// untouched `OsString`s.
pub fn run_with_optional_augment_os(
    real_grep: &Path,
    argv: &[OsString],
    args: &GrepArgs,
) -> Result<i32, Error> {
    let real_exit = crate::run_grep_os(real_grep, argv)?;
    touch_passthrough_store();

    // Real-grep miss/error must not trigger
    // any visible semantic output.
    if real_exit != 0 {
        return Ok(real_exit);
    }

    let gate = freshness_gate(args);
    let mode = classify(args, gate);

    if matches!(mode, Mode::Sidecar | Mode::VisibleAugment) {
        if let Ok(cwd) = std::env::current_dir() {
            let workspace_root = workspace_locator::resolve_workspace_root(&cwd);
            // The original command string for the sidecar header is a
            // best-effort lossy rendering of the OsString argv; only the
            // forwarded argv (above) must be byte-exact.
            let argv_lossy: Vec<String> = argv
                .iter()
                .map(|a| a.to_string_lossy().into_owned())
                .collect();
            let _ = run_augment(args, mode, &workspace_root, real_grep, &argv_lossy);
        }
    }

    Ok(real_exit)
}

fn touch_passthrough_store() {
    let Ok(cwd) = std::env::current_dir() else {
        return;
    };
    let root = workspace_locator::resolve_workspace_root(&cwd);
    let dir = greppy_core::cache::workspace_store_dir(&root);
    if greppy_core::cache::read_store_manifest(&dir).is_ok() {
        greppy_core::cache::touch_last_used_dir(&dir);
    }
    let _ = sidecar::cleanup_expired(&root, sidecar::sidecar_ttl_secs());
}

/// Compute the freshness gate for the current invocation. Used by
/// both the drop-in binary and the CLI dispatcher. Returns `Strict`
/// if the graph is stale, the store is unreadable, or the budget is
/// exceeded. Only `FreshnessOutcome::Fresh` yields `FreshnessGate::Fresh`.
pub fn freshness_gate(args: &GrepArgs) -> FreshnessGate {
    if args.is_stdin_only() {
        return FreshnessGate::Strict;
    }
    let cwd = match std::env::current_dir() {
        Ok(c) => c,
        Err(_) => return FreshnessGate::Strict,
    };
    let workspace_root = workspace_locator::resolve_workspace_root(&cwd);
    // Read the graph DB from the platform-locator's path,
    // never from `<cwd>/.greppy/graph.db`.
    let store_path = workspace_locator::store_path(&workspace_root);
    let store = match greppy_store::Store::open_with(&store_path, OpenOptions::read_only()) {
        Ok(s) => s,
        Err(_) => return FreshnessGate::Strict,
    };
    let project = workspace_locator::project_identity(&workspace_root);
    let res = match greppy_freshness::check_files(
        &store,
        &workspace_root,
        &project,
        std::time::Duration::from_millis(200),
    ) {
        Ok(r) => r,
        Err(_) => return FreshnessGate::Strict,
    };
    match res.outcome {
        FreshnessOutcome::Fresh => FreshnessGate::Fresh,
        _ => FreshnessGate::Strict,
    }
}

fn latest_workspace_generation(store: &greppy_store::Store) -> Option<u64> {
    let conn = store.conn();
    conn.query_row(
        "SELECT graph_generation FROM workspace_state ORDER BY updated_at DESC LIMIT 1",
        [],
        |row| row.get::<_, i64>(0),
    )
    .ok()
    .map(|g| g as u64)
}

fn run_augment(
    args: &GrepArgs,
    mode: Mode,
    workspace_root: &Path,
    real_grep_path: &Path,
    argv: &[String],
) -> std::io::Result<()> {
    let Some(query) = args.pattern.as_deref() else {
        return Ok(());
    };

    // Read the same locator'd store the freshness gate used.
    let store_path = workspace_locator::store_path(workspace_root);
    let store = match greppy_store::Store::open_with(&store_path, OpenOptions::read_only()) {
        Ok(s) => s,
        Err(_) => return Ok(()),
    };
    let near = workspace_root.to_string_lossy().to_string();
    let hits = match greppy_search::semantic_query(&store, query, Some(&near), None, 10) {
        Ok(h) => h,
        Err(_) => return Ok(()),
    };
    if hits.is_empty() {
        return Ok(());
    }

    let generation = latest_workspace_generation(&store).unwrap_or(0);

    let original_cmd = format!("{} {}", real_grep_path.display(), argv[1..].join(" "));

    match mode {
        Mode::Strict => Ok(()),
        Mode::Sidecar => {
            sidecar::write_sidecar(workspace_root, query, &original_cmd, generation, &hits)?;
            Ok(())
        }
        Mode::VisibleAugment => {
            let sidecar_path =
                sidecar::write_sidecar(workspace_root, query, &original_cmd, generation, &hits)?;
            println!(
                "{}:1:<!-- NON_CANONICAL_CODE_HINT: {} -->",
                sidecar_path.display(),
                query
            );
            Ok(())
        }
    }
}
