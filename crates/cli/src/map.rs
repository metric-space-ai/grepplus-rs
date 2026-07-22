//! Deterministic, read-only project orientation for `greppy map`.

use greppy_core::error::{Error, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

const COLLAPSED_DIRS: &[&str] = &[
    "vendor",
    "generated",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    "third_party",
];

#[derive(Debug, Clone)]
struct SourceFile {
    rel_path: String,
    language: Option<&'static str>,
}

#[derive(Debug, Clone)]
struct LanguageSummary {
    language: &'static str,
    files: usize,
    indexed_files: usize,
}

#[derive(Debug, Clone)]
struct ModuleSummary {
    path: String,
    files: usize,
    collapsed: bool,
}

#[derive(Debug, Clone)]
struct BuildCommand {
    command: String,
    source: String,
}

pub(super) fn run(path: Option<&str>, json_output: bool, root: Option<&str>) -> Result<i32> {
    let workspace = super::resolve_root(root)?;
    let scope = resolve_scope(&workspace, path)?;
    let scope_rel = relative_display(&workspace, &scope);
    let files = repository_files(&workspace, &scope)?;
    let indexed = indexed_paths(root).unwrap_or_default();

    let languages = language_summaries(&files, &indexed);
    let modules = module_summaries(&scope, &files, &workspace)?;
    let test_roots = test_roots(&files);
    let commands = build_commands(&scope, &files, &workspace);
    let large_subtrees = large_subtrees(&scope, &files, &workspace);
    let suggestions = suggestions(&scope_rel, &modules);

    if json_output {
        let value = json!({
            "schema_version": "greppy.map.v1",
            "root": workspace.to_string_lossy(),
            "path": scope_rel,
            "index": {
                "available": !indexed.is_empty(),
                "indexed_files_in_scope": files.iter().filter(|file| indexed.contains(&file.rel_path)).count(),
            },
            "languages": languages.iter().map(|item| json!({
                "language": item.language,
                "files": item.files,
                "indexed_files": item.indexed_files,
                "indexed": item.indexed_files == item.files,
            })).collect::<Vec<_>>(),
            "modules": modules.iter().map(|item| json!({
                "path": item.path,
                "files": item.files,
                "collapsed": item.collapsed,
            })).collect::<Vec<_>>(),
            "test_roots": test_roots,
            "commands": commands.iter().map(|item| json!({
                "command": item.command,
                "source": item.source,
            })).collect::<Vec<_>>(),
            "large_subtrees": large_subtrees.iter().map(|item| json!({
                "path": item.path,
                "files": item.files,
            })).collect::<Vec<_>>(),
            "suggestions": suggestions,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&value)
                .map_err(|error| Error::Parse(format!("serialize map JSON: {error}")))?
        );
        return Ok(0);
    }

    let lines = render_text(
        &workspace,
        &scope_rel,
        &languages,
        &modules,
        &test_roots,
        &commands,
        &large_subtrees,
        &suggestions,
    );
    debug_assert!(lines.len() <= 60, "map text exceeded one-screen budget");
    println!("{}", lines.join("\n"));
    Ok(0)
}

fn resolve_scope(workspace: &Path, path: Option<&str>) -> Result<PathBuf> {
    let supplied = Path::new(path.unwrap_or("."));
    let candidate = if supplied.is_absolute() {
        supplied.to_path_buf()
    } else {
        let cwd = std::env::current_dir().map_err(|error| Error::io("read map cwd", error))?;
        let from_cwd = cwd.join(supplied);
        if from_cwd.exists() {
            from_cwd
        } else {
            workspace.join(supplied)
        }
    };
    let canonical = candidate
        .canonicalize()
        .map_err(|error| Error::io(format!("resolve map path {}", candidate.display()), error))?;
    if !canonical.starts_with(workspace) {
        return Err(Error::Invalid(format!(
            "map path must be inside workspace {}",
            workspace.display()
        )));
    }
    if !canonical.is_dir() {
        return Err(Error::Invalid(format!(
            "map path must be a directory: {}",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn repository_files(workspace: &Path, scope: &Path) -> Result<Vec<SourceFile>> {
    let scope_rel = scope.strip_prefix(workspace).unwrap_or(Path::new("."));
    let pathspec = if scope_rel.as_os_str().is_empty() {
        "."
    } else {
        scope_rel.to_str().unwrap_or(".")
    };
    let output = Command::new("git")
        .args([
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            pathspec,
        ])
        .current_dir(workspace)
        .output();

    let mut paths = match output {
        Ok(output) if output.status.success() => output
            .stdout
            .split(|byte| *byte == 0)
            .filter(|raw| !raw.is_empty())
            .map(|raw| String::from_utf8_lossy(raw).replace('\\', "/"))
            .collect::<Vec<_>>(),
        _ => walk_files(workspace, scope)?,
    };
    paths.sort();
    paths.dedup();
    Ok(paths
        .into_iter()
        .map(|rel_path| SourceFile {
            language: language_for_path(&rel_path),
            rel_path,
        })
        .collect())
}

fn walk_files(workspace: &Path, scope: &Path) -> Result<Vec<String>> {
    fn visit(workspace: &Path, dir: &Path, output: &mut Vec<String>) -> Result<()> {
        let mut entries = std::fs::read_dir(dir)
            .map_err(|error| Error::io(format!("read map directory {}", dir.display()), error))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| Error::io(format!("read map directory {}", dir.display()), error))?;
        entries.sort_by_key(std::fs::DirEntry::file_name);
        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                if entry.file_name() == ".git" {
                    continue;
                }
                visit(workspace, &path, output)?;
            } else if path.is_file() {
                output.push(relative_display(workspace, &path));
            }
        }
        Ok(())
    }

    let mut output = Vec::new();
    visit(workspace, scope, &mut output)?;
    Ok(output)
}

fn indexed_paths(root: Option<&str>) -> Result<BTreeSet<String>> {
    let store = super::open_default_store(root)?;
    let project = super::project_for(root)?;
    let states = store
        .list_file_states(&project)
        .map_err(|error| Error::Store(error.to_string()))?;
    Ok(states
        .into_iter()
        .map(|state| state.rel_path.replace('\\', "/"))
        .collect())
}

fn language_summaries(files: &[SourceFile], indexed: &BTreeSet<String>) -> Vec<LanguageSummary> {
    let mut counts: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for file in files {
        let Some(language) = file.language else {
            continue;
        };
        let entry = counts.entry(language).or_default();
        entry.0 += 1;
        entry.1 += usize::from(indexed.contains(&file.rel_path));
    }
    let mut summaries = counts
        .into_iter()
        .map(|(language, (files, indexed_files))| LanguageSummary {
            language,
            files,
            indexed_files,
        })
        .collect::<Vec<_>>();
    summaries.sort_by(|left, right| {
        right
            .files
            .cmp(&left.files)
            .then_with(|| left.language.cmp(right.language))
    });
    summaries
}

fn module_summaries(
    scope: &Path,
    files: &[SourceFile],
    workspace: &Path,
) -> Result<Vec<ModuleSummary>> {
    let mut names = std::fs::read_dir(scope)
        .map_err(|error| Error::io(format!("read map modules {}", scope.display()), error))?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| Error::io(format!("read map modules {}", scope.display()), error))?;
    names.sort_by_key(std::fs::DirEntry::file_name);
    let scope_rel = relative_display(workspace, scope);
    let prefix = if scope_rel == "." {
        String::new()
    } else {
        format!("{scope_rel}/")
    };
    let mut output = Vec::new();
    for entry in names {
        if !entry.path().is_dir() || entry.file_name() == ".git" {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        let rel = format!("{prefix}{name}");
        let file_prefix = format!("{rel}/");
        let count = files
            .iter()
            .filter(|file| file.rel_path.starts_with(&file_prefix))
            .count();
        output.push(ModuleSummary {
            path: rel,
            files: count,
            collapsed: COLLAPSED_DIRS.contains(&name.as_str()),
        });
    }
    Ok(output)
}

fn test_roots(files: &[SourceFile]) -> Vec<String> {
    let mut roots = BTreeSet::new();
    for file in files {
        let parts = file.rel_path.split('/').collect::<Vec<_>>();
        for index in 0..parts.len().saturating_sub(1) {
            let part = parts[index].to_ascii_lowercase();
            if matches!(
                part.as_str(),
                "test" | "tests" | "spec" | "specs" | "__tests__"
            ) {
                roots.insert(parts[..=index].join("/"));
            }
        }
    }
    roots.into_iter().collect()
}

fn build_commands(scope: &Path, files: &[SourceFile], workspace: &Path) -> Vec<BuildCommand> {
    let scope_rel = relative_display(workspace, scope);
    let in_scope = |name: &str| {
        let path = if scope_rel == "." {
            name.to_string()
        } else {
            format!("{scope_rel}/{name}")
        };
        files.iter().any(|file| file.rel_path == path) || scope.join(name).is_file()
    };
    let mut commands = BTreeMap::<String, String>::new();
    if in_scope("Cargo.toml") {
        commands.insert("cargo test".into(), source_path(&scope_rel, "Cargo.toml"));
    }
    if in_scope("go.mod") {
        commands.insert("go test ./...".into(), source_path(&scope_rel, "go.mod"));
    }
    if in_scope("pytest.ini") || in_scope("pyproject.toml") {
        let source = if in_scope("pytest.ini") {
            "pytest.ini"
        } else {
            "pyproject.toml"
        };
        commands.insert("pytest".into(), source_path(&scope_rel, source));
    }
    if in_scope("tox.ini") {
        commands.insert("tox".into(), source_path(&scope_rel, "tox.ini"));
    }
    if in_scope("package.json") {
        add_package_commands(scope, &scope_rel, &mut commands);
    }
    for makefile in ["Makefile", "makefile", "GNUmakefile"] {
        if in_scope(makefile) {
            add_make_commands(scope, &scope_rel, makefile, &mut commands);
            break;
        }
    }
    commands
        .into_iter()
        .map(|(command, source)| BuildCommand { command, source })
        .collect()
}

fn add_package_commands(scope: &Path, scope_rel: &str, commands: &mut BTreeMap<String, String>) {
    let Ok(bytes) = std::fs::read(scope.join("package.json")) else {
        return;
    };
    let Ok(value) = serde_json::from_slice::<Value>(&bytes) else {
        return;
    };
    let Some(scripts) = value.get("scripts").and_then(Value::as_object) else {
        return;
    };
    for name in scripts.keys().filter(|name| name.starts_with("test")) {
        let command = if name == "test" {
            "npm test".to_string()
        } else {
            format!("npm run {name}")
        };
        commands.insert(command, source_path(scope_rel, "package.json"));
    }
}

fn add_make_commands(
    scope: &Path,
    scope_rel: &str,
    makefile: &str,
    commands: &mut BTreeMap<String, String>,
) {
    let Ok(content) = std::fs::read_to_string(scope.join(makefile)) else {
        return;
    };
    for target in ["test", "check"] {
        if content.lines().any(|line| {
            line.strip_prefix(target)
                .is_some_and(|rest| rest.starts_with(':') || rest.starts_with(" :"))
        }) {
            commands.insert(format!("make {target}"), source_path(scope_rel, makefile));
        }
    }
}

fn large_subtrees(scope: &Path, files: &[SourceFile], workspace: &Path) -> Vec<ModuleSummary> {
    let scope_rel = relative_display(workspace, scope);
    let prefix = if scope_rel == "." {
        String::new()
    } else {
        format!("{scope_rel}/")
    };
    let mut counts = BTreeMap::<String, usize>::new();
    for file in files {
        let Some(rest) = file.rel_path.strip_prefix(&prefix) else {
            continue;
        };
        let Some(first) = rest.split('/').next().filter(|_| rest.contains('/')) else {
            continue;
        };
        *counts.entry(format!("{prefix}{first}")).or_default() += 1;
    }
    let mut output = counts
        .into_iter()
        .map(|(path, files)| ModuleSummary {
            collapsed: path
                .rsplit('/')
                .next()
                .is_some_and(|name| COLLAPSED_DIRS.contains(&name)),
            path,
            files,
        })
        .collect::<Vec<_>>();
    output.sort_by(|left, right| {
        right
            .files
            .cmp(&left.files)
            .then_with(|| left.path.cmp(&right.path))
    });
    output.truncate(5);
    output
}

fn suggestions(scope_rel: &str, modules: &[ModuleSummary]) -> Vec<String> {
    let mut candidates = modules
        .iter()
        .filter(|module| !module.collapsed && module.files > 0)
        .map(|module| module.path.clone())
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        let left_count = modules
            .iter()
            .find(|item| item.path == *left)
            .map_or(0, |item| item.files);
        let right_count = modules
            .iter()
            .find(|item| item.path == *right)
            .map_or(0, |item| item.files);
        right_count.cmp(&left_count).then_with(|| left.cmp(right))
    });
    candidates.truncate(3);
    if candidates.len() < 2 {
        for fallback in ["src", "tests", "crates"] {
            let path = if scope_rel == "." {
                fallback.to_string()
            } else {
                format!("{scope_rel}/{fallback}")
            };
            if !candidates.contains(&path) {
                candidates.push(path);
            }
            if candidates.len() == 2 {
                break;
            }
        }
    }
    candidates
        .into_iter()
        .take(3)
        .map(|path| format!("greppy map {path}/"))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn render_text(
    workspace: &Path,
    scope_rel: &str,
    languages: &[LanguageSummary],
    modules: &[ModuleSummary],
    test_roots: &[String],
    commands: &[BuildCommand],
    large_subtrees: &[ModuleSummary],
    suggestions: &[String],
) -> Vec<String> {
    let mut lines = vec![
        format!("project: {}", workspace.display()),
        format!("scope: {scope_rel}"),
        String::new(),
        "languages:".into(),
    ];
    push_limited(&mut lines, languages, 10, |item| {
        let coverage = if item.indexed_files == item.files {
            "yes".to_string()
        } else if item.indexed_files == 0 {
            "no".to_string()
        } else {
            format!("partial {}/{}", item.indexed_files, item.files)
        };
        format!(
            "  {:<12} {:>5} files  indexed: {coverage}",
            item.language, item.files
        )
    });
    push_section(&mut lines, "modules:");
    push_limited(&mut lines, modules, 12, |item| {
        let marker = if item.collapsed { "  [collapsed]" } else { "" };
        format!("  {:<28} {:>5} files{marker}", item.path, item.files)
    });
    push_section(&mut lines, "test roots:");
    push_limited(&mut lines, test_roots, 6, |item| format!("  {item}"));
    push_section(&mut lines, "build/test commands:");
    push_limited(&mut lines, commands, 7, |item| {
        format!("  {}  ({})", item.command, item.source)
    });
    push_section(&mut lines, "largest subtrees:");
    push_limited(&mut lines, large_subtrees, 5, |item| {
        format!("  {:<28} {:>5} files", item.path, item.files)
    });
    lines.push(String::new());
    for suggestion in suggestions {
        lines.push(format!("try: {suggestion}"));
    }
    lines
}

fn push_section(lines: &mut Vec<String>, title: &str) {
    lines.push(String::new());
    lines.push(title.to_string());
}

fn push_limited<T>(
    lines: &mut Vec<String>,
    values: &[T],
    limit: usize,
    format: impl Fn(&T) -> String,
) {
    if values.is_empty() {
        lines.push("  (none detected)".into());
        return;
    }
    lines.extend(values.iter().take(limit).map(format));
    if values.len() > limit {
        lines.push(format!("  ... {} more", values.len() - limit));
    }
}

fn source_path(scope_rel: &str, name: &str) -> String {
    if scope_rel == "." {
        name.to_string()
    } else {
        format!("{scope_rel}/{name}")
    }
}

fn relative_display(workspace: &Path, path: &Path) -> String {
    path.strip_prefix(workspace)
        .ok()
        .filter(|path| !path.as_os_str().is_empty())
        .map(|path| path.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|| ".".into())
}

fn language_for_path(path: &str) -> Option<&'static str> {
    let file = path.rsplit('/').next().unwrap_or(path);
    let extension = file
        .rsplit_once('.')
        .map(|(_, extension)| extension.to_ascii_lowercase());
    match extension.as_deref() {
        Some("rs") => Some("rust"),
        Some("py") | Some("pyi") => Some("python"),
        Some("js") | Some("jsx") | Some("mjs") | Some("cjs") => Some("javascript"),
        Some("ts") | Some("tsx") | Some("mts") | Some("cts") => Some("typescript"),
        Some("go") => Some("go"),
        Some("java") => Some("java"),
        Some("kt") | Some("kts") => Some("kotlin"),
        Some("scala") | Some("sc") => Some("scala"),
        Some("swift") => Some("swift"),
        Some("c") | Some("h") => Some("c"),
        Some("cc") | Some("cpp") | Some("cxx") | Some("hpp") | Some("hh") => Some("cpp"),
        Some("cs") => Some("csharp"),
        Some("rb") => Some("ruby"),
        Some("php") => Some("php"),
        Some("ex") | Some("exs") => Some("elixir"),
        Some("hs") | Some("lhs") => Some("haskell"),
        Some("ml") | Some("mli") => Some("ocaml"),
        Some("lua") => Some("lua"),
        Some("dart") => Some("dart"),
        Some("zig") => Some("zig"),
        Some("sh") | Some("bash") | Some("zsh") => Some("shell"),
        Some("sql") => Some("sql"),
        _ => None,
    }
}
