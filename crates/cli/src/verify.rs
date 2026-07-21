use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const CACHE_MAGIC: &[u8] = b"greppy-verify-cache-v1\0";
const MIRROR_CANDIDATES: &[&str] = &[".tox", ".venv", "venv", ".nox", "node_modules", "target"];

#[derive(Debug, Clone)]
pub(crate) struct CommandRun {
    pub(crate) exit_code: Option<i32>,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
    pub(crate) timed_out: bool,
    pub(crate) spawn_error: Option<String>,
}

impl CommandRun {
    fn from_spawn_error(error: std::io::Error) -> Self {
        Self {
            exit_code: None,
            stdout: Vec::new(),
            stderr: Vec::new(),
            timed_out: false,
            spawn_error: Some(error.to_string()),
        }
    }

    pub(crate) fn combined_text(&self) -> String {
        let mut bytes = Vec::with_capacity(self.stdout.len() + self.stderr.len() + 1);
        bytes.extend_from_slice(&self.stdout);
        if !self.stdout.is_empty() && !self.stderr.is_empty() && !self.stdout.ends_with(b"\n") {
            bytes.push(b'\n');
        }
        bytes.extend_from_slice(&self.stderr);
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

pub(crate) fn repository_root(cwd: &Path) -> Result<PathBuf, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .map_err(|error| format!("cannot start git: {error}"))?;
    if !output.status.success() {
        return Err(first_output_line(
            &output.stderr,
            "not inside a git worktree",
        ));
    }
    let root = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if root.is_empty() {
        return Err("git returned an empty worktree root".into());
    }
    Ok(PathBuf::from(root))
}

pub(crate) fn resolve_revision(root: &Path, revision: &str) -> Result<String, String> {
    let expression = format!("{revision}^{{commit}}");
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "--verify"])
        .arg(expression)
        .output()
        .map_err(|error| format!("cannot start git: {error}"))?;
    if !output.status.success() {
        return Err(first_output_line(
            &output.stderr,
            &format!("baseline revision `{revision}` is not a commit"),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

/// Hash the index entries and live bytes of every tracked path. Untracked
/// files are deliberately excluded: this is a dirstate attestation, not a
/// repository archive hash.
pub(crate) fn workspace_digest(root: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-s", "-z"])
        .output()
        .map_err(|error| format!("cannot start git for workspace digest: {error}"))?;
    if !output.status.success() {
        return Err(first_output_line(
            &output.stderr,
            "git ls-files failed while computing workspace digest",
        ));
    }

    let mut hasher = Sha256::new();
    for record in output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|r| !r.is_empty())
    {
        hasher.update((record.len() as u64).to_le_bytes());
        hasher.update(record);
        let Some(tab) = record.iter().position(|byte| *byte == b'\t') else {
            return Err("unexpected git ls-files record without path".into());
        };
        let path_bytes = &record[tab + 1..];
        let relative = path_from_git_bytes(path_bytes)?;
        let path = root.join(relative);
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                hasher.update(b"symlink\0");
                let target = fs::read_link(&path)
                    .map_err(|error| format!("read symlink {}: {error}", path.display()))?;
                hasher.update(os_str_bytes(target.as_os_str()));
            }
            Ok(metadata) if metadata.is_file() => {
                hasher.update(b"file\0");
                let mut file = fs::File::open(&path)
                    .map_err(|error| format!("open tracked file {}: {error}", path.display()))?;
                let mut buffer = [0_u8; 64 * 1024];
                loop {
                    let read = file.read(&mut buffer).map_err(|error| {
                        format!("read tracked file {}: {error}", path.display())
                    })?;
                    if read == 0 {
                        break;
                    }
                    hasher.update(&buffer[..read]);
                }
            }
            Ok(_) => hasher.update(b"other\0"),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                hasher.update(b"missing\0")
            }
            Err(error) => {
                return Err(format!("stat tracked path {}: {error}", path.display()));
            }
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(unix)]
fn path_from_git_bytes(bytes: &[u8]) -> Result<PathBuf, String> {
    use std::os::unix::ffi::OsStrExt;
    Ok(PathBuf::from(OsStr::from_bytes(bytes)))
}

#[cfg(windows)]
fn path_from_git_bytes(bytes: &[u8]) -> Result<PathBuf, String> {
    String::from_utf8(bytes.to_vec())
        .map(PathBuf::from)
        .map_err(|_| "git emitted a non-UTF-8 tracked path on Windows".into())
}

#[cfg(unix)]
fn os_str_bytes(value: &OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    value.as_bytes().to_vec()
}

#[cfg(windows)]
fn os_str_bytes(value: &OsStr) -> Vec<u8> {
    value.to_string_lossy().as_bytes().to_vec()
}

pub(crate) fn run_command(argv: &[String], cwd: &Path, timeout: Duration) -> CommandRun {
    let Some(program) = argv.first() else {
        return CommandRun::from_spawn_error(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "test command is empty",
        ));
    };
    let mut child = match Command::new(program)
        .args(&argv[1..])
        .current_dir(cwd)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => return CommandRun::from_spawn_error(error),
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_reader = thread::spawn(move || read_pipe(stdout));
    let stderr_reader = thread::spawn(move || read_pipe(stderr));
    let started = Instant::now();
    let mut timed_out = false;
    let status: Option<ExitStatus> = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if started.elapsed() < timeout => thread::sleep(Duration::from_millis(20)),
            Ok(None) => {
                timed_out = true;
                let _ = child.kill();
                break child.wait().ok();
            }
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                let stdout = stdout_reader.join().unwrap_or_default();
                let stderr = stderr_reader.join().unwrap_or_default();
                return CommandRun {
                    exit_code: None,
                    stdout,
                    stderr,
                    timed_out: false,
                    spawn_error: Some(format!("wait for test command: {error}")),
                };
            }
        }
    };
    CommandRun {
        exit_code: status.and_then(|value| value.code()),
        stdout: stdout_reader.join().unwrap_or_default(),
        stderr: stderr_reader.join().unwrap_or_default(),
        timed_out,
        spawn_error: None,
    }
}

fn read_pipe<R: Read>(pipe: Option<R>) -> Vec<u8> {
    let mut bytes = Vec::new();
    if let Some(mut pipe) = pipe {
        let _ = pipe.read_to_end(&mut bytes);
    }
    bytes
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Framework {
    Pytest,
    GoTest,
    CargoTest,
    JestVitest,
    Unknown,
}

impl Framework {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Pytest => "pytest",
            Self::GoTest => "go-test",
            Self::CargoTest => "cargo-test",
            Self::JestVitest => "jest-vitest",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TestStatus {
    Passed,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TestObservation {
    pub(crate) status: TestStatus,
    pub(crate) evidence_line: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct InfrastructureFailure {
    pub(crate) signature: String,
    pub(crate) evidence_line: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ParsedRun {
    pub(crate) framework: Framework,
    pub(crate) tests: BTreeMap<String, TestObservation>,
    pub(crate) infrastructure: Option<InfrastructureFailure>,
    pub(crate) limitation: Option<String>,
}

pub(crate) fn parse_run(run: &CommandRun) -> ParsedRun {
    if let Some(error) = &run.spawn_error {
        return ParsedRun {
            framework: Framework::Unknown,
            tests: BTreeMap::new(),
            infrastructure: Some(InfrastructureFailure {
                signature: "test_process_spawn_error".into(),
                evidence_line: one_line(error),
            }),
            limitation: Some("no test framework output was produced".into()),
        };
    }
    if run.timed_out {
        return ParsedRun {
            framework: Framework::Unknown,
            tests: BTreeMap::new(),
            infrastructure: Some(InfrastructureFailure {
                signature: "timeout".into(),
                evidence_line: "test command exceeded its timeout".into(),
            }),
            limitation: Some("the timed-out run may have emitted only partial test output".into()),
        };
    }

    let text = strip_ansi(&run.combined_text());
    let lines: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    let framework = detect_framework(&lines);
    let tests = match framework {
        Framework::Pytest => parse_pytest(&lines),
        Framework::GoTest => parse_go_test(&lines),
        Framework::CargoTest => parse_cargo_test(&lines),
        Framework::JestVitest => parse_jest_vitest(&lines),
        Framework::Unknown => BTreeMap::new(),
    };
    let infrastructure = classify_infrastructure(run.exit_code, framework, &tests, &lines);
    let limitation = match framework {
        Framework::Unknown => Some(
            "unrecognized test framework: comparison is limited to command exit codes".into(),
        ),
        _ if tests.is_empty() => Some(format!(
            "{} was recognized, but this output did not expose individual test IDs; command-level evidence is used where possible",
            framework.as_str()
        )),
        _ => None,
    };
    ParsedRun {
        framework,
        tests,
        infrastructure,
        limitation,
    }
}

fn detect_framework(lines: &[String]) -> Framework {
    if lines.iter().any(|line| {
        line.starts_with("test result:")
            || (line.starts_with("running ") && line.ends_with(" tests"))
    }) {
        return Framework::CargoTest;
    }
    if lines.iter().any(|line| {
        line.starts_with("=== RUN")
            || line.starts_with("--- PASS:")
            || line.starts_with("--- FAIL:")
            || line.starts_with("ok\t")
            || line.starts_with("FAIL\t")
    }) {
        return Framework::GoTest;
    }
    if lines.iter().any(|line| {
        line.starts_with("Test Suites:")
            || line.starts_with("Tests:")
            || line.starts_with("PASS ")
            || line.starts_with("FAIL ")
            || line.contains(" Vitest ")
    }) {
        return Framework::JestVitest;
    }
    if lines.iter().any(|line| {
        line.contains("short test summary info")
            || line.contains("ERROR collecting")
            || line.starts_with("collected ")
            || ((line.contains(" passed") || line.contains(" failed"))
                && (line.contains(" in ") || line.contains("pytest")))
            || (line.contains("::")
                && (line.contains(" PASSED")
                    || line.contains(" FAILED")
                    || line.starts_with("FAILED ")
                    || line.starts_with("PASSED ")))
    }) {
        return Framework::Pytest;
    }
    Framework::Unknown
}

fn parse_cargo_test(lines: &[String]) -> BTreeMap<String, TestObservation> {
    let mut tests = BTreeMap::new();
    for line in lines {
        let Some(rest) = line.strip_prefix("test ") else {
            continue;
        };
        let Some((test_id, status)) = rest.rsplit_once(" ... ") else {
            continue;
        };
        let status = match status.split_whitespace().next() {
            Some("ok") => TestStatus::Passed,
            Some("FAILED") => TestStatus::Failed,
            _ => continue,
        };
        tests.insert(
            test_id.to_owned(),
            TestObservation {
                status,
                evidence_line: one_line(line),
            },
        );
    }
    tests
}

fn parse_pytest(lines: &[String]) -> BTreeMap<String, TestObservation> {
    let mut tests = BTreeMap::new();
    for line in lines {
        let status = if line.starts_with("FAILED ") || line.contains(" FAILED") {
            Some(TestStatus::Failed)
        } else if line.starts_with("PASSED ") || line.contains(" PASSED") {
            Some(TestStatus::Passed)
        } else {
            None
        };
        let Some(status) = status else { continue };
        let Some(test_id) = line
            .split_whitespace()
            .find(|token| token.contains("::"))
            .map(|token| token.trim_end_matches([':', ',']).to_owned())
        else {
            continue;
        };
        tests.insert(
            test_id,
            TestObservation {
                status,
                evidence_line: one_line(line),
            },
        );
    }
    tests
}

fn parse_go_test(lines: &[String]) -> BTreeMap<String, TestObservation> {
    let mut tests = BTreeMap::new();
    for line in lines {
        let (status, rest) = if let Some(rest) = line.strip_prefix("--- PASS: ") {
            (TestStatus::Passed, rest)
        } else if let Some(rest) = line.strip_prefix("--- FAIL: ") {
            (TestStatus::Failed, rest)
        } else {
            continue;
        };
        let test_id = rest.split_whitespace().next().unwrap_or(rest);
        tests.insert(
            test_id.to_owned(),
            TestObservation {
                status,
                evidence_line: one_line(line),
            },
        );
    }
    tests
}

fn parse_jest_vitest(lines: &[String]) -> BTreeMap<String, TestObservation> {
    let mut tests = BTreeMap::new();
    for line in lines {
        let (status, rest) = if let Some(rest) = line.strip_prefix("PASS ") {
            (TestStatus::Passed, rest)
        } else if let Some(rest) = line.strip_prefix("FAIL ") {
            (TestStatus::Failed, rest)
        } else if let Some(rest) = strip_test_glyph(line, &["✓", "✔"]) {
            (TestStatus::Passed, rest)
        } else if let Some(rest) = strip_test_glyph(line, &["✕", "×", "✗"]) {
            (TestStatus::Failed, rest)
        } else {
            continue;
        };
        let test_id = strip_javascript_duration(rest);
        if test_id.is_empty() {
            continue;
        }
        tests.insert(
            test_id.to_owned(),
            TestObservation {
                status,
                evidence_line: one_line(line),
            },
        );
    }
    tests
}

fn strip_test_glyph<'a>(line: &'a str, glyphs: &[&str]) -> Option<&'a str> {
    let trimmed = line.trim_start();
    glyphs
        .iter()
        .find_map(|glyph| trimmed.strip_prefix(glyph).map(str::trim))
}

fn strip_javascript_duration(value: &str) -> &str {
    let value = value.trim();
    if let Some(open) = value.rfind(" (") {
        let suffix = &value[open + 2..];
        if suffix.ends_with("ms)") && suffix[..suffix.len() - 3].trim().parse::<u64>().is_ok() {
            return value[..open].trim();
        }
    }
    value
}

fn classify_infrastructure(
    exit_code: Option<i32>,
    framework: Framework,
    tests: &BTreeMap<String, TestObservation>,
    lines: &[String],
) -> Option<InfrastructureFailure> {
    if exit_code == Some(0) {
        return None;
    }
    const HARD_SIGNATURES: &[(&str, &[&str])] = &[
        (
            "command_not_found",
            &[
                "command not found",
                "not recognized as an internal or external command",
            ],
        ),
        (
            "missing_file_or_interpreter",
            &["no such file or directory", "bad interpreter"],
        ),
        (
            "missing_test_runner",
            &[
                "no module named pytest",
                "no module named 'pytest'",
                "no module named unittest",
                "no module named nose",
            ],
        ),
        (
            "cannot_execute",
            &[
                "cannot execute binary file",
                "cannot execute: required file not found",
            ],
        ),
        (
            "compile_failure",
            &["error: could not compile", "[build failed]"],
        ),
        (
            "collection_failure",
            &[
                "error collecting",
                "importerror while importing test module",
                "modulenotfounderror",
                "test suite failed to run",
            ],
        ),
    ];
    for (signature, needles) in HARD_SIGNATURES {
        if let Some(line) = lines.iter().find(|line| {
            let lower = line.to_ascii_lowercase();
            needles.iter().any(|needle| lower.contains(needle))
        }) {
            return Some(InfrastructureFailure {
                signature: (*signature).into(),
                evidence_line: one_line(line),
            });
        }
    }
    if framework != Framework::Unknown
        && !tests.values().any(|test| test.status == TestStatus::Failed)
    {
        return Some(InfrastructureFailure {
            signature: "nonzero_without_assertion_failure".into(),
            evidence_line: lines
                .first()
                .map(|line| one_line(line))
                .unwrap_or_else(|| "nonzero exit without test-framework output".into()),
        });
    }
    None
}

fn strip_ansi(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut output = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == 0x1b && bytes.get(index + 1) == Some(&b'[') {
            index += 2;
            while index < bytes.len() {
                let byte = bytes[index];
                index += 1;
                if (0x40..=0x7e).contains(&byte) {
                    break;
                }
            }
        } else {
            output.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&output).into_owned()
}

fn one_line(value: &str) -> String {
    let compact = value.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut chars = compact.chars();
    let shortened: String = chars.by_ref().take(500).collect();
    if chars.next().is_some() {
        format!("{shortened}…")
    } else {
        shortened
    }
}

pub(crate) const EXIT_NEWLY_FAILED: i32 = 21;
pub(crate) const EXIT_INFRASTRUCTURE: i32 = 22;

#[derive(clap::Args, Debug)]
pub struct VerifyArgs {
    /// Committed revision used for the isolated baseline run.
    #[arg(long, default_value = "HEAD")]
    pub(crate) baseline: String,
    /// Emit the complete greppy.verify-report.v1 JSON document.
    #[arg(long)]
    pub(crate) json: bool,
    /// Per-run timeout in seconds.
    #[arg(long, default_value_t = 900, value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) timeout: u64,
    /// Ignore any cached baseline result.
    #[arg(long)]
    pub(crate) no_cache: bool,
    /// Test command and arguments. A `--` separator is required.
    #[arg(last = true, required = true, num_args = 1.., allow_hyphen_values = true)]
    pub(crate) test_command: Vec<String>,
}

#[derive(Debug, Clone)]
struct ReportCase {
    test_id: String,
    evidence_line: String,
}

#[derive(Debug)]
struct VerifyReport {
    command: Vec<String>,
    baseline_rev: String,
    baseline_exit: Option<i32>,
    after_exit: Option<i32>,
    workspace_digest_before: String,
    workspace_digest_after: String,
    framework: String,
    mirrored_paths: Vec<String>,
    baseline_cache_hit: bool,
    limitations: Vec<String>,
    newly_failed: Vec<ReportCase>,
    fixed: Vec<ReportCase>,
    preexisting_failed: Vec<ReportCase>,
    still_passing: Vec<ReportCase>,
    not_run_in_after: Vec<ReportCase>,
    infrastructure_error: Vec<ReportCase>,
    exit_code: i32,
}

pub(crate) fn run(args: VerifyArgs, root_override: Option<&str>) -> i32 {
    let start = match root_override {
        Some(root) => PathBuf::from(root),
        None => match std::env::current_dir() {
            Ok(path) => path,
            Err(error) => {
                return emit_early_infrastructure(
                    &args,
                    "workspace",
                    &format!("cannot resolve current directory: {error}"),
                );
            }
        },
    };
    let root = match repository_root(&start) {
        Ok(path) => path,
        Err(error) => return emit_early_infrastructure(&args, "workspace", &error),
    };
    let command_cwd = if root_override.is_some() {
        root.clone()
    } else {
        match start.canonicalize() {
            Ok(path) => path,
            Err(error) => {
                return emit_early_infrastructure(
                    &args,
                    "workspace",
                    &format!(
                        "cannot resolve command directory {}: {error}",
                        start.display()
                    ),
                );
            }
        }
    };
    let relative_cwd = match command_cwd.strip_prefix(&root) {
        Ok(path) => path.to_path_buf(),
        Err(_) => {
            return emit_early_infrastructure(
                &args,
                "workspace",
                "the command directory is outside the selected git worktree",
            );
        }
    };
    let digest_before = match workspace_digest(&root) {
        Ok(digest) => digest,
        Err(error) => return emit_early_infrastructure(&args, "workspace-digest-before", &error),
    };

    // Binding execution order: the user's current tree is tested first. Git
    // worktree setup begins only after this child process has completed.
    let after_run = run_command(
        &args.test_command,
        &command_cwd,
        Duration::from_secs(args.timeout),
    );
    let after_parsed = parse_run(&after_run);
    let mirrors = discover_mirrors(&root, &relative_cwd);
    let mut limitations = Vec::new();
    push_limitation(&mut limitations, after_parsed.limitation.clone());
    let mut setup_infrastructure = Vec::new();
    let mut baseline_cache_hit = false;
    let mut baseline_revision = args.baseline.clone();
    let mut baseline_run: Option<CommandRun> = None;

    match resolve_revision(&root, &args.baseline) {
        Ok(revision) => {
            baseline_revision = revision.clone();
            let key = cache_key(&revision, &args.test_command, &mirrors);
            let cache_path = greppy_core::workspace::store_dir(&root)
                .join("verify-cache")
                .join(format!("{key}.bin"));
            if !args.no_cache {
                baseline_run = read_cache(&cache_path);
                baseline_cache_hit = baseline_run.is_some();
            }
            if baseline_run.is_none() {
                match TemporaryWorktree::add(&root, &revision) {
                    Ok(worktree) => {
                        let baseline_cwd = worktree.path().join(&relative_cwd);
                        if let Err(error) = install_mirrors(worktree.path(), &mirrors) {
                            setup_infrastructure.push(ReportCase {
                                test_id: "baseline:environment-mirror".into(),
                                evidence_line: one_line(&error),
                            });
                        } else if !baseline_cwd.is_dir() {
                            setup_infrastructure.push(ReportCase {
                                test_id: "baseline:command-directory".into(),
                                evidence_line: format!(
                                    "baseline revision does not contain command directory {}",
                                    relative_cwd.display()
                                ),
                            });
                        } else {
                            let run = run_command(
                                &args.test_command,
                                &baseline_cwd,
                                Duration::from_secs(args.timeout),
                            );
                            if let Err(error) = write_cache(&cache_path, &run) {
                                limitations.push(format!("baseline cache write failed: {error}"));
                            }
                            baseline_run = Some(run);
                        }
                        if let Err(error) = worktree.cleanup() {
                            setup_infrastructure.push(ReportCase {
                                test_id: "baseline:worktree-cleanup".into(),
                                evidence_line: one_line(&error),
                            });
                        }
                    }
                    Err(error) => setup_infrastructure.push(ReportCase {
                        test_id: "baseline:worktree-setup".into(),
                        evidence_line: one_line(&error),
                    }),
                }
            }
        }
        Err(error) => setup_infrastructure.push(ReportCase {
            test_id: "baseline:revision".into(),
            evidence_line: one_line(&error),
        }),
    }

    let baseline_parsed = baseline_run.as_ref().map(parse_run);
    if let Some(parsed) = &baseline_parsed {
        push_limitation(&mut limitations, parsed.limitation.clone());
    }
    let digest_after = match workspace_digest(&root) {
        Ok(digest) => digest,
        Err(error) => {
            setup_infrastructure.push(ReportCase {
                test_id: "workspace-digest-after".into(),
                evidence_line: one_line(&error),
            });
            String::new()
        }
    };

    let framework = combined_framework(&after_parsed, baseline_parsed.as_ref());
    let mut report = VerifyReport {
        command: args.test_command,
        baseline_rev: baseline_revision,
        baseline_exit: baseline_run.as_ref().and_then(|run| run.exit_code),
        after_exit: after_run.exit_code,
        workspace_digest_before: digest_before,
        workspace_digest_after: digest_after,
        framework,
        mirrored_paths: mirrors
            .iter()
            .map(|mirror| mirror.relative.to_string_lossy().into_owned())
            .collect(),
        baseline_cache_hit,
        limitations,
        newly_failed: Vec::new(),
        fixed: Vec::new(),
        preexisting_failed: Vec::new(),
        still_passing: Vec::new(),
        not_run_in_after: Vec::new(),
        infrastructure_error: setup_infrastructure,
        exit_code: 0,
    };

    append_run_infrastructure("after", &after_parsed, &mut report.infrastructure_error);
    if let Some(parsed) = &baseline_parsed {
        append_run_infrastructure("baseline", parsed, &mut report.infrastructure_error);
        classify_tests(
            parsed,
            &after_parsed,
            baseline_run.as_ref(),
            &after_run,
            &mut report,
        );
    }
    if !report.infrastructure_error.is_empty() {
        report.exit_code = EXIT_INFRASTRUCTURE;
    } else if !report.newly_failed.is_empty() {
        report.exit_code = EXIT_NEWLY_FAILED;
    }
    emit_report(&report, args.json);
    report.exit_code
}

fn classify_tests(
    baseline: &ParsedRun,
    after: &ParsedRun,
    baseline_run: Option<&CommandRun>,
    after_run: &CommandRun,
    report: &mut VerifyReport,
) {
    let mut ids: BTreeSet<String> = baseline.tests.keys().cloned().collect();
    ids.extend(after.tests.keys().cloned());
    if ids.is_empty() && baseline.infrastructure.is_none() && after.infrastructure.is_none() {
        let baseline_status = command_status(baseline_run.and_then(|run| run.exit_code));
        let after_status = command_status(after_run.exit_code);
        classify_pair(
            "command",
            baseline_status.map(|status| TestObservation {
                status,
                evidence_line: command_evidence(baseline_run),
            }),
            after_status.map(|status| TestObservation {
                status,
                evidence_line: command_evidence(Some(after_run)),
            }),
            report,
        );
        return;
    }
    for id in ids {
        classify_pair(
            &id,
            baseline.tests.get(&id).cloned(),
            after.tests.get(&id).cloned(),
            report,
        );
    }
}

fn classify_pair(
    test_id: &str,
    baseline: Option<TestObservation>,
    after: Option<TestObservation>,
    report: &mut VerifyReport,
) {
    match (baseline, after) {
        (Some(base), Some(current)) => match (base.status, current.status) {
            (TestStatus::Passed, TestStatus::Failed) => report.newly_failed.push(ReportCase {
                test_id: test_id.into(),
                evidence_line: current.evidence_line,
            }),
            (TestStatus::Failed, TestStatus::Passed) => report.fixed.push(ReportCase {
                test_id: test_id.into(),
                evidence_line: current.evidence_line,
            }),
            (TestStatus::Failed, TestStatus::Failed) => {
                report.preexisting_failed.push(ReportCase {
                    test_id: test_id.into(),
                    evidence_line: current.evidence_line,
                });
            }
            (TestStatus::Passed, TestStatus::Passed) => report.still_passing.push(ReportCase {
                test_id: test_id.into(),
                evidence_line: current.evidence_line,
            }),
        },
        (None, Some(current)) if current.status == TestStatus::Failed => {
            report.newly_failed.push(ReportCase {
                test_id: test_id.into(),
                evidence_line: current.evidence_line,
            });
        }
        (None, Some(current)) => report.still_passing.push(ReportCase {
            test_id: test_id.into(),
            evidence_line: current.evidence_line,
        }),
        (Some(base), None) => report.not_run_in_after.push(ReportCase {
            test_id: test_id.into(),
            evidence_line: base.evidence_line,
        }),
        (None, None) => {}
    }
}

fn command_status(exit_code: Option<i32>) -> Option<TestStatus> {
    exit_code.map(|code| {
        if code == 0 {
            TestStatus::Passed
        } else {
            TestStatus::Failed
        }
    })
}

fn command_evidence(run: Option<&CommandRun>) -> String {
    let Some(run) = run else {
        return "command did not run".into();
    };
    let output = strip_ansi(&run.combined_text());
    output
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(one_line)
        .unwrap_or_else(|| format!("command exited {}", run.exit_code.unwrap_or(-1)))
}

fn combined_framework(after: &ParsedRun, baseline: Option<&ParsedRun>) -> String {
    match baseline.map(|run| run.framework) {
        Some(value) if value == after.framework => value.as_str().into(),
        Some(Framework::Unknown) | None => after.framework.as_str().into(),
        Some(value) if after.framework == Framework::Unknown => value.as_str().into(),
        Some(value) => format!("mixed:{}+{}", value.as_str(), after.framework.as_str()),
    }
}

fn append_run_infrastructure(label: &str, parsed: &ParsedRun, target: &mut Vec<ReportCase>) {
    if let Some(failure) = &parsed.infrastructure {
        target.push(ReportCase {
            test_id: format!("{label}:{}", failure.signature),
            evidence_line: failure.evidence_line.clone(),
        });
    }
}

fn push_limitation(limitations: &mut Vec<String>, value: Option<String>) {
    if let Some(value) = value {
        if !limitations.contains(&value) {
            limitations.push(value);
        }
    }
}

fn emit_early_infrastructure(args: &VerifyArgs, test_id: &str, evidence: &str) -> i32 {
    let report = VerifyReport {
        command: args.test_command.clone(),
        baseline_rev: args.baseline.clone(),
        baseline_exit: None,
        after_exit: None,
        workspace_digest_before: String::new(),
        workspace_digest_after: String::new(),
        framework: "unknown".into(),
        mirrored_paths: Vec::new(),
        baseline_cache_hit: false,
        limitations: vec![
            "verification could not start; no test framework output was produced".into(),
        ],
        newly_failed: Vec::new(),
        fixed: Vec::new(),
        preexisting_failed: Vec::new(),
        still_passing: Vec::new(),
        not_run_in_after: Vec::new(),
        infrastructure_error: vec![ReportCase {
            test_id: test_id.into(),
            evidence_line: one_line(evidence),
        }],
        exit_code: EXIT_INFRASTRUCTURE,
    };
    emit_report(&report, args.json);
    EXIT_INFRASTRUCTURE
}

fn emit_report(report: &VerifyReport, json: bool) {
    if json {
        println!("{}", report_json(report));
        return;
    }
    println!(
        "verify: newly_failed={} fixed={} preexisting_failed={} still_passing={} not_run_in_after={} infrastructure_error={}",
        report.newly_failed.len(),
        report.fixed.len(),
        report.preexisting_failed.len(),
        report.still_passing.len(),
        report.not_run_in_after.len(),
        report.infrastructure_error.len()
    );
    println!(
        "framework: {} | baseline={} exit={}{} | after_exit={}",
        report.framework,
        report.baseline_rev,
        display_exit(report.baseline_exit),
        if report.baseline_cache_hit {
            " (cache hit)"
        } else {
            ""
        },
        display_exit(report.after_exit)
    );
    println!(
        "workspace: before={} after={} unchanged={}",
        report.workspace_digest_before,
        report.workspace_digest_after,
        report.workspace_digest_before == report.workspace_digest_after
            && !report.workspace_digest_before.is_empty()
    );
    if report.mirrored_paths.is_empty() {
        println!("baseline mirrors: (none)");
    } else {
        println!("baseline mirrors: {}", report.mirrored_paths.join(", "));
    }
    for limitation in &report.limitations {
        println!("limitation: {limitation}");
    }
    for case in &report.newly_failed {
        println!("newly_failed: {} — {}", case.test_id, case.evidence_line);
    }
    for case in &report.infrastructure_error {
        println!(
            "infrastructure_error: {} — {}",
            case.test_id, case.evidence_line
        );
    }
    if !report.preexisting_failed.is_empty() {
        println!(
            "preexisting_failed: {}",
            report
                .preexisting_failed
                .iter()
                .map(|case| case.test_id.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

fn report_json(report: &VerifyReport) -> serde_json::Value {
    serde_json::json!({
        "schema_version": "greppy.verify-report.v1",
        "command": report.command,
        "baseline_rev": report.baseline_rev,
        "baseline_exit": report.baseline_exit,
        "after_exit": report.after_exit,
        "workspace_digest_before": report.workspace_digest_before,
        "workspace_digest_after": report.workspace_digest_after,
        "workspace_unchanged": report.workspace_digest_before == report.workspace_digest_after && !report.workspace_digest_before.is_empty(),
        "framework": report.framework,
        "mirrored_paths": report.mirrored_paths,
        "baseline_cache_hit": report.baseline_cache_hit,
        "limitations": report.limitations,
        "newly_failed": cases_json(&report.newly_failed),
        "fixed": cases_json(&report.fixed),
        "preexisting_failed": cases_json(&report.preexisting_failed),
        "still_passing": cases_json(&report.still_passing),
        "not_run_in_after": cases_json(&report.not_run_in_after),
        "infrastructure_error": cases_json(&report.infrastructure_error),
        "exit_code": report.exit_code,
    })
}

fn cases_json(cases: &[ReportCase]) -> Vec<serde_json::Value> {
    cases
        .iter()
        .map(|case| {
            serde_json::json!({
                "test_id": case.test_id,
                "evidence_line": case.evidence_line,
            })
        })
        .collect()
}

fn display_exit(exit: Option<i32>) -> String {
    exit.map_or_else(|| "not-run".into(), |code| code.to_string())
}

pub(crate) struct TemporaryWorktree {
    repository: PathBuf,
    path: PathBuf,
    active: bool,
}

impl TemporaryWorktree {
    pub(crate) fn add(repository: &Path, revision: &str) -> Result<Self, String> {
        let path = unique_worktree_path();
        let output = Command::new("git")
            .arg("-C")
            .arg(repository)
            .args(["worktree", "add", "--detach"])
            .arg(&path)
            .arg(revision)
            .output()
            .map_err(|error| format!("cannot start git worktree add: {error}"))?;
        if !output.status.success() {
            return Err(first_output_line(&output.stderr, "git worktree add failed"));
        }
        Ok(Self {
            repository: repository.to_path_buf(),
            path,
            active: true,
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn cleanup(mut self) -> Result<(), String> {
        let result = self.remove();
        self.active = false;
        result
    }

    fn remove(&mut self) -> Result<(), String> {
        let output = Command::new("git")
            .arg("-C")
            .arg(&self.repository)
            .args(["worktree", "remove", "--force"])
            .arg(&self.path)
            .output()
            .map_err(|error| format!("cannot start git worktree remove: {error}"))?;
        let remove_dir_result = if self.path.exists() {
            fs::remove_dir_all(&self.path)
        } else {
            Ok(())
        };
        let _ = Command::new("git")
            .arg("-C")
            .arg(&self.repository)
            .args(["worktree", "prune"])
            .status();
        if !output.status.success() {
            return Err(first_output_line(
                &output.stderr,
                "git worktree remove failed",
            ));
        }
        remove_dir_result.map_err(|error| {
            format!(
                "remove temporary worktree directory {}: {error}",
                self.path.display()
            )
        })
    }
}

impl Drop for TemporaryWorktree {
    fn drop(&mut self) {
        if self.active {
            let _ = self.remove();
        }
    }
}

fn unique_worktree_path() -> PathBuf {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!("greppy-verify-{}-{epoch}", std::process::id()))
}

#[derive(Debug, Clone)]
pub(crate) struct Mirror {
    pub(crate) relative: PathBuf,
    pub(crate) source: PathBuf,
}

pub(crate) fn discover_mirrors(root: &Path, relative_cwd: &Path) -> Vec<Mirror> {
    let mut candidates = BTreeSet::new();
    for name in MIRROR_CANDIDATES {
        candidates.insert(PathBuf::from(name));
        if !relative_cwd.as_os_str().is_empty() {
            candidates.insert(relative_cwd.join(name));
        }
    }
    candidates
        .into_iter()
        .filter_map(|relative| {
            let source = root.join(&relative);
            if !source.is_dir() || !is_gitignored(root, &relative) {
                return None;
            }
            Some(Mirror { relative, source })
        })
        .collect()
}

fn is_gitignored(root: &Path, relative: &Path) -> bool {
    Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["check-ignore", "-q", "--"])
        .arg(relative)
        .status()
        .is_ok_and(|status| status.success())
}

pub(crate) fn install_mirrors(worktree: &Path, mirrors: &[Mirror]) -> Result<(), String> {
    for mirror in mirrors {
        let destination = worktree.join(&mirror.relative);
        if destination.exists() || fs::symlink_metadata(&destination).is_ok() {
            continue;
        }
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create mirror parent {}: {error}", parent.display()))?;
        }
        create_dir_symlink(&mirror.source, &destination).map_err(|error| {
            format!(
                "mirror {} at {}: {error}",
                mirror.source.display(),
                destination.display()
            )
        })?;
    }
    Ok(())
}

#[cfg(unix)]
fn create_dir_symlink(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::os::unix::fs::symlink(source, destination)
}

#[cfg(windows)]
fn create_dir_symlink(source: &Path, destination: &Path) -> std::io::Result<()> {
    std::os::windows::fs::symlink_dir(source, destination)
}

pub(crate) fn cache_key(revision: &str, argv: &[String], mirrors: &[Mirror]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(revision.as_bytes());
    hasher.update([0]);
    for arg in argv {
        hasher.update(arg.as_bytes());
        hasher.update([0]);
    }
    for mirror in mirrors {
        hasher.update(os_str_bytes(mirror.relative.as_os_str()));
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

pub(crate) fn read_cache(path: &Path) -> Option<CommandRun> {
    let mut file = fs::File::open(path).ok()?;
    let mut magic = vec![0_u8; CACHE_MAGIC.len()];
    file.read_exact(&mut magic).ok()?;
    if magic != CACHE_MAGIC {
        return None;
    }
    let exit_code = read_i32(&mut file)?;
    let stdout = read_blob(&mut file)?;
    let stderr = read_blob(&mut file)?;
    Some(CommandRun {
        exit_code: if exit_code == i32::MIN {
            None
        } else {
            Some(exit_code)
        },
        stdout,
        stderr,
        timed_out: false,
        spawn_error: None,
    })
}

pub(crate) fn write_cache(path: &Path, run: &CommandRun) -> Result<(), String> {
    if run.timed_out || run.spawn_error.is_some() {
        return Ok(());
    }
    let parent = path
        .parent()
        .ok_or_else(|| format!("cache path {} has no parent", path.display()))?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("create verify cache {}: {error}", parent.display()))?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    let mut file = fs::File::create(&temporary)
        .map_err(|error| format!("create verify cache {}: {error}", temporary.display()))?;
    file.write_all(CACHE_MAGIC)
        .and_then(|_| file.write_all(&run.exit_code.unwrap_or(i32::MIN).to_le_bytes()))
        .and_then(|_| write_blob(&mut file, &run.stdout))
        .and_then(|_| write_blob(&mut file, &run.stderr))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("write verify cache {}: {error}", temporary.display()))?;
    fs::rename(&temporary, path)
        .map_err(|error| format!("publish verify cache {}: {error}", path.display()))
}

fn read_i32(reader: &mut impl Read) -> Option<i32> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).ok()?;
    Some(i32::from_le_bytes(bytes))
}

fn read_blob(reader: &mut impl Read) -> Option<Vec<u8>> {
    let mut length = [0_u8; 8];
    reader.read_exact(&mut length).ok()?;
    let length = u64::from_le_bytes(length);
    let length: usize = length.try_into().ok()?;
    if length > 256 * 1024 * 1024 {
        return None;
    }
    let mut bytes = vec![0_u8; length];
    reader.read_exact(&mut bytes).ok()?;
    Some(bytes)
}

fn write_blob(writer: &mut impl Write, bytes: &[u8]) -> std::io::Result<()> {
    writer.write_all(&(bytes.len() as u64).to_le_bytes())?;
    writer.write_all(bytes)
}

fn first_output_line(output: &[u8], fallback: &str) -> String {
    String::from_utf8_lossy(output)
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or(fallback)
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_directory(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "greppy-verify-unit-{label}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn command_timeout_is_reported() {
        #[cfg(unix)]
        let argv = vec!["sh".into(), "-c".into(), "sleep 2".into()];
        #[cfg(windows)]
        let argv = vec!["cmd".into(), "/C".into(), "ping -n 3 127.0.0.1 >NUL".into()];
        let run = run_command(&argv, Path::new("."), Duration::from_millis(20));
        assert!(run.timed_out);
    }

    #[test]
    fn workspace_digest_changes_with_tracked_bytes_not_untracked_bytes() {
        let root = temporary_directory("digest");
        assert!(Command::new("git")
            .arg("init")
            .arg("-q")
            .arg(&root)
            .status()
            .unwrap()
            .success());
        fs::write(root.join("tracked.txt"), "one").unwrap();
        assert!(Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(["add", "tracked.txt"])
            .status()
            .unwrap()
            .success());
        let first = workspace_digest(&root).unwrap();
        fs::write(root.join("untracked.txt"), "ignored by digest").unwrap();
        assert_eq!(workspace_digest(&root).unwrap(), first);
        fs::write(root.join("tracked.txt"), "two").unwrap();
        assert_ne!(workspace_digest(&root).unwrap(), first);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn parses_pytest_quiet_failure_ids() {
        let run = CommandRun {
            exit_code: Some(1),
            stdout: b"FAILED tests/test_math.py::test_add - assert 3 == 4\n1 failed, 1 passed in 0.02s\n".to_vec(),
            stderr: Vec::new(),
            timed_out: false,
            spawn_error: None,
        };
        let parsed = parse_run(&run);
        assert_eq!(parsed.framework, Framework::Pytest);
        assert_eq!(
            parsed.tests["tests/test_math.py::test_add"].status,
            TestStatus::Failed
        );
        assert!(parsed.infrastructure.is_none());
    }

    #[test]
    fn parses_cargo_go_and_javascript_results() {
        let cargo = CommandRun {
            exit_code: Some(1),
            stdout: b"running 2 tests\ntest math::adds ... ok\ntest math::breaks ... FAILED\ntest result: FAILED. 1 passed; 1 failed\n".to_vec(),
            stderr: Vec::new(), timed_out: false, spawn_error: None,
        };
        let parsed = parse_run(&cargo);
        assert_eq!(parsed.framework, Framework::CargoTest);
        assert_eq!(parsed.tests["math::breaks"].status, TestStatus::Failed);

        let go = CommandRun {
            exit_code: Some(1),
            stdout: b"=== RUN   TestAdd\n--- FAIL: TestAdd (0.00s)\nFAIL\texample/pkg\n".to_vec(),
            stderr: Vec::new(),
            timed_out: false,
            spawn_error: None,
        };
        let parsed = parse_run(&go);
        assert_eq!(parsed.framework, Framework::GoTest);
        assert_eq!(parsed.tests["TestAdd"].status, TestStatus::Failed);

        let js = CommandRun {
            exit_code: Some(1),
            stdout: "FAIL src/math.test.ts\n  ✕ adds numbers (3 ms)\nTest Suites: 1 failed\n"
                .as_bytes()
                .to_vec(),
            stderr: Vec::new(),
            timed_out: false,
            spawn_error: None,
        };
        let parsed = parse_run(&js);
        assert_eq!(parsed.framework, Framework::JestVitest);
        assert_eq!(parsed.tests["adds numbers"].status, TestStatus::Failed);
    }

    #[test]
    fn compile_and_collection_errors_are_infrastructure() {
        let cargo = CommandRun {
            exit_code: Some(101),
            stdout: Vec::new(),
            stderr: b"error: could not compile `fixture` due to 1 previous error\n".to_vec(),
            timed_out: false,
            spawn_error: None,
        };
        let parsed = parse_run(&cargo);
        assert_eq!(parsed.infrastructure.unwrap().signature, "compile_failure");

        let pytest = CommandRun {
            exit_code: Some(2),
            stdout: b"collected 0 items / 1 error\nERROR collecting tests/test_bad.py\n".to_vec(),
            stderr: Vec::new(),
            timed_out: false,
            spawn_error: None,
        };
        let parsed = parse_run(&pytest);
        assert_eq!(parsed.framework, Framework::Pytest);
        assert_eq!(
            parsed.infrastructure.unwrap().signature,
            "collection_failure"
        );
    }

    #[test]
    fn cache_round_trip_preserves_process_result() {
        let root = temporary_directory("cache");
        let path = root.join("entry");
        let run = CommandRun {
            exit_code: Some(7),
            stdout: b"out".to_vec(),
            stderr: b"err".to_vec(),
            timed_out: false,
            spawn_error: None,
        };
        write_cache(&path, &run).unwrap();
        let restored = read_cache(&path).unwrap();
        assert_eq!(restored.exit_code, Some(7));
        assert_eq!(restored.stdout, b"out");
        assert_eq!(restored.stderr, b"err");
        let _ = fs::remove_dir_all(root);
    }
}
