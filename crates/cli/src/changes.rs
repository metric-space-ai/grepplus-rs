//! Read-only, parser-backed symbol summary for `greppy changes`.

use greppy_core::error::{Error, Result};
use greppy_search::{GraphQuery, ReachDirection};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static SNAPSHOT_NONCE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
struct ChangedFile {
    path: String,
    status: FileStatus,
    before: Option<Vec<u8>>,
    after: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FileStatus {
    Added,
    Modified,
    Deleted,
}

impl FileStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Modified => "modified",
            Self::Deleted => "deleted",
        }
    }

    fn marker(self) -> char {
        match self {
            Self::Added => 'A',
            Self::Modified => 'M',
            Self::Deleted => 'D',
        }
    }
}

#[derive(Debug, Clone)]
struct DefinitionSnapshot {
    label: String,
    name: String,
    qualified_name: String,
    start_line: i64,
    end_line: i64,
    signature: String,
    body: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DefinitionChangeKind {
    Added,
    Modified,
    Deleted,
}

impl DefinitionChangeKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Modified => "modified",
            Self::Deleted => "deleted",
        }
    }
}

#[derive(Debug, Clone)]
struct DefinitionChange {
    kind: DefinitionChangeKind,
    label: String,
    name: String,
    qualified_name: String,
    before_span: Option<(i64, i64)>,
    after_span: Option<(i64, i64)>,
    before_signature: Option<String>,
    after_signature: Option<String>,
    signature_changed: bool,
}

#[derive(Debug, Clone)]
struct FileSummary {
    path: String,
    status: FileStatus,
    parser_indexed: bool,
    definitions: Vec<DefinitionChange>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CallsiteImpact {
    changed_symbol: String,
    caller: String,
    file: String,
    line: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct KnownTestImpact {
    changed_symbol: String,
    test_symbol: String,
    file: String,
    line: i64,
    hops: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct UnknownImpact {
    path: String,
    reason: String,
}

struct ParsedSnapshot {
    definitions: BTreeMap<String, Vec<DefinitionSnapshot>>,
    indexed_paths: BTreeSet<String>,
}

struct SnapshotDir(PathBuf);

impl SnapshotDir {
    fn create(tag: &str) -> Result<Self> {
        let nonce = SNAPSHOT_NONCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "greppy-changes-{tag}-{}-{nonce}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(path.join(".git")).map_err(|error| {
            Error::io(format!("create snapshot root {}", path.display()), error)
        })?;
        Ok(Self(path))
    }
}

impl Drop for SnapshotDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub(super) fn run(base: Option<&str>, json_output: bool, root: Option<&str>) -> Result<i32> {
    let workspace = super::resolve_root(root)?;
    let requested_base = base.unwrap_or("HEAD");
    let base_oid = resolve_base(&workspace, requested_base)?;
    let changed_files = changed_files(&workspace, &base_oid)?;

    let before_inputs = changed_files
        .iter()
        .filter_map(|file| file.before.clone().map(|bytes| (file.path.clone(), bytes)))
        .collect::<Vec<_>>();
    let after_inputs = changed_files
        .iter()
        .filter_map(|file| file.after.clone().map(|bytes| (file.path.clone(), bytes)))
        .collect::<Vec<_>>();
    let before = parse_snapshot("before", &before_inputs)?;
    let after = parse_snapshot("after", &after_inputs)?;

    let files = summarize_files(&changed_files, &before, &after);
    let (callsites, known_tests, unknown_tests) = graph_impact(root, &files)?;

    if json_output {
        let value = json!({
            "schema_version": "greppy.changes.v1",
            "base": {
                "requested": requested_base,
                "oid": base_oid,
                "target": "working_tree_including_staged",
            },
            "files": files.iter().map(file_json).collect::<Vec<_>>(),
            "callsite_impact": callsites.iter().map(|impact| json!({
                "changed_symbol": impact.changed_symbol,
                "caller": impact.caller,
                "file": impact.file,
                "line": impact.line,
            })).collect::<Vec<_>>(),
            "tests": {
                "known_impacted": known_tests.iter().map(|impact| json!({
                    "changed_symbol": impact.changed_symbol,
                    "test_symbol": impact.test_symbol,
                    "file": impact.file,
                    "line": impact.line,
                    "hops": impact.hops,
                })).collect::<Vec<_>>(),
                "unknown_or_unindexed": unknown_tests.iter().map(|impact| json!({
                    "path": impact.path,
                    "reason": impact.reason,
                })).collect::<Vec<_>>(),
                "graph_depth": 2,
            },
        });
        // Successful edit journals are intentionally transient and are deleted
        // after publication. There is currently no persistent journal history
        // to prove provenance against, so the contract requires us to omit a
        // certification field rather than label ordinary edits speculatively.
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|error| Error::Parse(format!("serialize changes JSON: {error}")))?
        );
        return Ok(0);
    }

    render_text(
        requested_base,
        &base_oid,
        &files,
        &callsites,
        &known_tests,
        &unknown_tests,
    );
    Ok(0)
}

fn resolve_base(workspace: &Path, requested: &str) -> Result<String> {
    let revision = format!("{requested}^{{commit}}");
    let output = Command::new("git")
        .args(["rev-parse", "--verify", "--quiet", &revision])
        .current_dir(workspace)
        .output()
        .map_err(|error| Error::io("run git rev-parse for changes", error))?;
    if !output.status.success() {
        return Err(Error::Invalid(format!(
            "changes base is not a commit: {requested}"
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn changed_files(workspace: &Path, base_oid: &str) -> Result<Vec<ChangedFile>> {
    let output = Command::new("git")
        .args([
            "diff",
            "--name-status",
            "-z",
            "--no-renames",
            base_oid,
            "--",
        ])
        .current_dir(workspace)
        .output()
        .map_err(|error| Error::io("run git diff for changes", error))?;
    if !output.status.success() {
        return Err(Error::Invalid(format!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }

    let fields = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|field| !field.is_empty())
        .collect::<Vec<_>>();
    let mut statuses = BTreeMap::<String, FileStatus>::new();
    for pair in fields.chunks_exact(2) {
        let status = match pair[0].first().copied() {
            Some(b'A') => FileStatus::Added,
            Some(b'D') => FileStatus::Deleted,
            _ => FileStatus::Modified,
        };
        statuses.insert(String::from_utf8_lossy(pair[1]).replace('\\', "/"), status);
    }

    let untracked = Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard", "-z", "--"])
        .current_dir(workspace)
        .output()
        .map_err(|error| Error::io("list untracked files for changes", error))?;
    if untracked.status.success() {
        for raw in untracked.stdout.split(|byte| *byte == 0) {
            if !raw.is_empty() {
                statuses
                    .entry(String::from_utf8_lossy(raw).replace('\\', "/"))
                    .or_insert(FileStatus::Added);
            }
        }
    }

    statuses
        .into_iter()
        .map(|(path, status)| {
            let before = if status == FileStatus::Added {
                None
            } else {
                git_blob(workspace, base_oid, &path)?
            };
            let after = if status == FileStatus::Deleted {
                None
            } else {
                let absolute = workspace.join(&path);
                Some(std::fs::read(&absolute).map_err(|error| {
                    Error::io(format!("read changed file {}", absolute.display()), error)
                })?)
            };
            Ok(ChangedFile {
                path,
                status,
                before,
                after,
            })
        })
        .collect()
}

fn git_blob(workspace: &Path, base_oid: &str, path: &str) -> Result<Option<Vec<u8>>> {
    let object = format!("{base_oid}:{path}");
    let output = Command::new("git")
        .args(["show", &object])
        .current_dir(workspace)
        .output()
        .map_err(|error| Error::io(format!("read {path} at changes base"), error))?;
    if output.status.success() {
        Ok(Some(output.stdout))
    } else {
        Ok(None)
    }
}

fn parse_snapshot(tag: &str, files: &[(String, Vec<u8>)]) -> Result<ParsedSnapshot> {
    if files.is_empty() {
        return Ok(ParsedSnapshot {
            definitions: BTreeMap::new(),
            indexed_paths: BTreeSet::new(),
        });
    }
    let dir = SnapshotDir::create(tag)?;
    for (path, bytes) in files {
        let absolute = dir.0.join(path);
        if let Some(parent) = absolute.parent() {
            std::fs::create_dir_all(parent).map_err(|error| {
                Error::io(
                    format!("create snapshot parent {}", parent.display()),
                    error,
                )
            })?;
        }
        std::fs::write(&absolute, bytes).map_err(|error| {
            Error::io(format!("write snapshot file {}", absolute.display()), error)
        })?;
    }

    let mut store =
        greppy_store::Store::open_memory().map_err(|error| Error::Store(error.to_string()))?;
    greppy_indexer::index(&mut store, &dir.0, "changes-snapshot")?;
    let indexed_paths = store
        .list_file_states("changes-snapshot")
        .map_err(|error| Error::Store(error.to_string()))?
        .into_iter()
        .map(|state| state.rel_path.replace('\\', "/"))
        .collect::<BTreeSet<_>>();

    let mut definitions = BTreeMap::new();
    for (path, bytes) in files {
        let rows =
            greppy_search::symbols_in_file(&store, Some("changes-snapshot"), path, usize::MAX)?;
        let source = String::from_utf8_lossy(bytes);
        let mut snapshots = Vec::new();
        for row in rows {
            if matches!(row.label.as_str(), "Project" | "Folder" | "File")
                || (row.label == "Module" && row.qualified_name.ends_with("::__file__"))
            {
                continue;
            }
            let properties = store
                .get_node(row.id)
                .map_err(|error| Error::Store(error.to_string()))?
                .map(|node| node.properties)
                .unwrap_or(Value::Null);
            let body = source_span(&source, row.start_line, row.end_line);
            let signature = properties
                .get("signature")
                .and_then(Value::as_str)
                .map(normalize_space)
                .filter(|signature| !signature.is_empty())
                .unwrap_or_else(|| fallback_signature(&body));
            snapshots.push(DefinitionSnapshot {
                label: row.label,
                name: row.name,
                qualified_name: row.qualified_name,
                start_line: row.start_line,
                end_line: row.end_line,
                signature,
                body,
            });
        }
        definitions.insert(path.clone(), snapshots);
    }

    Ok(ParsedSnapshot {
        definitions,
        indexed_paths,
    })
}

fn source_span(source: &str, start_line: i64, end_line: i64) -> String {
    if start_line <= 0 || end_line < start_line {
        return String::new();
    }
    source
        .lines()
        .skip(start_line.saturating_sub(1) as usize)
        .take((end_line - start_line + 1) as usize)
        .collect::<Vec<_>>()
        .join("\n")
}

fn fallback_signature(body: &str) -> String {
    let mut parts = Vec::new();
    for line in body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(8)
    {
        parts.push(line);
        if line.contains('{') || line.ends_with(':') || line.ends_with(';') || line.contains("=>") {
            break;
        }
    }
    let joined = parts.join(" ");
    let signature = joined
        .split_once('{')
        .map_or(joined.as_str(), |(head, _)| head)
        .trim_end_matches(':')
        .trim();
    normalize_space(signature)
}

fn normalize_space(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn summarize_files(
    changed_files: &[ChangedFile],
    before: &ParsedSnapshot,
    after: &ParsedSnapshot,
) -> Vec<FileSummary> {
    changed_files
        .iter()
        .map(|file| {
            let before_defs = before
                .definitions
                .get(&file.path)
                .cloned()
                .unwrap_or_default();
            let after_defs = after
                .definitions
                .get(&file.path)
                .cloned()
                .unwrap_or_default();
            let definitions = compare_definitions(&before_defs, &after_defs);
            let has_before_definitions = before_defs.iter().any(|definition| {
                !(definition.label == "Module" && definition.qualified_name.ends_with("::__file__"))
            });
            let has_after_definitions = after_defs.iter().any(|definition| {
                !(definition.label == "Module" && definition.qualified_name.ends_with("::__file__"))
            });
            FileSummary {
                path: file.path.clone(),
                status: file.status,
                parser_indexed: (before.indexed_paths.contains(&file.path)
                    && has_before_definitions)
                    || (after.indexed_paths.contains(&file.path) && has_after_definitions),
                definitions,
            }
        })
        .collect()
}

fn compare_definitions(
    before: &[DefinitionSnapshot],
    after: &[DefinitionSnapshot],
) -> Vec<DefinitionChange> {
    let before_map = before
        .iter()
        .map(|definition| (definition_key(definition), definition))
        .collect::<BTreeMap<_, _>>();
    let after_map = after
        .iter()
        .map(|definition| (definition_key(definition), definition))
        .collect::<BTreeMap<_, _>>();
    let keys = before_map
        .keys()
        .chain(after_map.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut changes = Vec::new();
    for key in keys {
        match (before_map.get(&key), after_map.get(&key)) {
            (None, Some(after)) => changes.push(definition_change(
                DefinitionChangeKind::Added,
                None,
                Some(after),
            )),
            (Some(before), None) => changes.push(definition_change(
                DefinitionChangeKind::Deleted,
                Some(before),
                None,
            )),
            (Some(before), Some(after)) if before.body != after.body => changes.push(
                definition_change(DefinitionChangeKind::Modified, Some(before), Some(after)),
            ),
            _ => {}
        }
    }
    changes.sort_by(|left, right| {
        change_order(left.kind)
            .cmp(&change_order(right.kind))
            .then_with(|| left.qualified_name.cmp(&right.qualified_name))
    });
    changes
}

fn definition_key(definition: &DefinitionSnapshot) -> (String, String) {
    (definition.label.clone(), definition.qualified_name.clone())
}

fn definition_change(
    kind: DefinitionChangeKind,
    before: Option<&DefinitionSnapshot>,
    after: Option<&DefinitionSnapshot>,
) -> DefinitionChange {
    let representative = after.or(before).expect("definition change has one side");
    let before_signature = before.map(|definition| definition.signature.clone());
    let after_signature = after.map(|definition| definition.signature.clone());
    DefinitionChange {
        kind,
        label: representative.label.clone(),
        name: representative.name.clone(),
        qualified_name: representative.qualified_name.clone(),
        before_span: before.map(|definition| (definition.start_line, definition.end_line)),
        after_span: after.map(|definition| (definition.start_line, definition.end_line)),
        signature_changed: kind == DefinitionChangeKind::Modified
            && before_signature != after_signature,
        before_signature,
        after_signature,
    }
}

fn change_order(kind: DefinitionChangeKind) -> u8 {
    match kind {
        DefinitionChangeKind::Modified => 0,
        DefinitionChangeKind::Added => 1,
        DefinitionChangeKind::Deleted => 2,
    }
}

fn graph_impact(
    root: Option<&str>,
    files: &[FileSummary],
) -> Result<(
    Vec<CallsiteImpact>,
    Vec<KnownTestImpact>,
    Vec<UnknownImpact>,
)> {
    let store = super::open_default_store(root)?;
    let project = super::project_for(root)?;
    let mut callsites = BTreeSet::new();
    let mut known_tests = BTreeSet::new();
    let mut unknown = BTreeSet::new();

    for file in files {
        if !file.parser_indexed && (source_like(&file.path) || test_path(&file.path)) {
            unknown.insert(UnknownImpact {
                path: file.path.clone(),
                reason: "changed file has no parser-backed symbols; test impact is unknown".into(),
            });
        }
        for change in &file.definitions {
            let rows = resolve_active_nodes(&store, &project, &file.path, change)?;
            if rows.is_empty() {
                unknown.insert(UnknownImpact {
                    path: file.path.clone(),
                    reason: format!(
                        "changed symbol {} is absent from the workspace graph",
                        change.qualified_name
                    ),
                });
                continue;
            }
            for row in rows {
                for step in greppy_search::callers_of(&store, row.id)? {
                    if let Some(caller) = step.node {
                        callsites.insert(CallsiteImpact {
                            changed_symbol: change.qualified_name.clone(),
                            caller: caller.qualified_name,
                            file: caller.file_path,
                            line: caller.start_line,
                        });
                    }
                }
                for impact in greppy_search::impact_radius(
                    &store,
                    row.id,
                    ReachDirection::Incoming,
                    "CALLS",
                    2,
                    10_000,
                )? {
                    if is_test_node(
                        &impact.node.label,
                        &impact.node.file_path,
                        &impact.node.name,
                    ) {
                        known_tests.insert(KnownTestImpact {
                            changed_symbol: change.qualified_name.clone(),
                            test_symbol: impact.node.qualified_name,
                            file: impact.node.file_path,
                            line: impact.node.start_line,
                            hops: impact.hops,
                        });
                    }
                }
            }
        }
    }

    Ok((
        callsites.into_iter().collect(),
        known_tests.into_iter().collect(),
        unknown.into_iter().collect(),
    ))
}

fn resolve_active_nodes(
    store: &greppy_store::Store,
    project: &str,
    file: &str,
    change: &DefinitionChange,
) -> Result<Vec<greppy_search::graph::SearchGraphRow>> {
    let exact = greppy_search::search_graph(
        store,
        &GraphQuery::any()
            .with_project(project)
            .with_qualified_name(&change.qualified_name)
            .with_limit(32),
    )?;
    if !exact.is_empty() {
        return Ok(exact);
    }
    let mut fallback = GraphQuery::any()
        .with_project(project)
        .with_name(&change.name)
        .with_limit(128);
    fallback.file_path_exact = Some(file.to_string());
    greppy_search::search_graph(store, &fallback)
}

fn is_test_node(label: &str, file: &str, name: &str) -> bool {
    label.eq_ignore_ascii_case("test")
        || test_path(file)
        || name.starts_with("test_")
        || name.ends_with("_test")
}

fn test_path(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    lower
        .split('/')
        .any(|part| matches!(part, "test" | "tests" | "spec" | "specs" | "__tests__"))
        || lower
            .rsplit('/')
            .next()
            .is_some_and(|name| name.starts_with("test_") || name.contains("_test."))
}

fn source_like(path: &str) -> bool {
    let extension = Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(
        extension.as_str(),
        "rs" | "py"
            | "pyi"
            | "js"
            | "jsx"
            | "mjs"
            | "cjs"
            | "ts"
            | "tsx"
            | "mts"
            | "cts"
            | "go"
            | "java"
            | "kt"
            | "kts"
            | "scala"
            | "sc"
            | "swift"
            | "c"
            | "h"
            | "cc"
            | "cpp"
            | "cxx"
            | "hpp"
            | "hh"
            | "cs"
            | "rb"
            | "php"
            | "ex"
            | "exs"
            | "hs"
            | "lhs"
            | "ml"
            | "mli"
            | "lua"
            | "dart"
            | "zig"
            | "sh"
            | "bash"
            | "r"
    )
}

fn file_json(file: &FileSummary) -> Value {
    let by_kind = |kind| {
        file.definitions
            .iter()
            .filter(|change| change.kind == kind)
            .map(definition_json)
            .collect::<Vec<_>>()
    };
    json!({
        "path": file.path,
        "status": file.status.as_str(),
        "parser_indexed": file.parser_indexed,
        "definitions": {
            "modified": by_kind(DefinitionChangeKind::Modified),
            "added": by_kind(DefinitionChangeKind::Added),
            "deleted": by_kind(DefinitionChangeKind::Deleted),
        },
    })
}

fn definition_json(change: &DefinitionChange) -> Value {
    json!({
        "kind": change.label,
        "name": change.name,
        "qualified_name": change.qualified_name,
        "change": change.kind.as_str(),
        "before_span": change.before_span.map(|(start, end)| json!({"start_line": start, "end_line": end})),
        "after_span": change.after_span.map(|(start, end)| json!({"start_line": start, "end_line": end})),
        "signature_changed": change.signature_changed,
        "before_signature": change.before_signature,
        "after_signature": change.after_signature,
    })
}

fn render_text(
    requested_base: &str,
    base_oid: &str,
    files: &[FileSummary],
    callsites: &[CallsiteImpact],
    known_tests: &[KnownTestImpact],
    unknown_tests: &[UnknownImpact],
) {
    println!(
        "changes: {requested_base} ({}) -> working tree",
        &base_oid[..12.min(base_oid.len())]
    );
    if files.is_empty() {
        println!("(no changes)");
    }
    for file in files {
        println!("\n{} {}", file.status.marker(), file.path);
        if file.definitions.is_empty() {
            let note = if file.parser_indexed {
                "no changed definitions"
            } else {
                "unindexed or no parser-backed definitions"
            };
            println!("  {note}");
            continue;
        }
        for change in &file.definitions {
            let span = change.after_span.or(change.before_span);
            let location =
                span.map_or_else(String::new, |(start, end)| format!(" lines {start}-{end}"));
            println!(
                "  {:<8} {} [{}]{}",
                change.kind.as_str(),
                change.qualified_name,
                change.label,
                location
            );
            if change.signature_changed {
                println!(
                    "    signature: {} -> {}",
                    change.before_signature.as_deref().unwrap_or("?"),
                    change.after_signature.as_deref().unwrap_or("?")
                );
            }
        }
    }

    println!("\ncallsite impact (direct callers):");
    if callsites.is_empty() {
        println!("  (none known)");
    } else {
        for impact in callsites {
            println!(
                "  {} <- {} {}:{}",
                impact.changed_symbol, impact.caller, impact.file, impact.line
            );
        }
    }

    println!("\ntests (CALLS graph, depth <= 2):");
    println!("  known_impacted:");
    if known_tests.is_empty() {
        println!("    (none known)");
    } else {
        for impact in known_tests {
            println!(
                "    {} {}:{} ({} hops from {})",
                impact.test_symbol, impact.file, impact.line, impact.hops, impact.changed_symbol
            );
        }
    }
    println!("  unknown_or_unindexed:");
    if unknown_tests.is_empty() {
        println!("    (none)");
    } else {
        for impact in unknown_tests {
            println!("    {}: {}", impact.path, impact.reason);
        }
    }
}
