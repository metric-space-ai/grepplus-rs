//! ripgrep-style passthrough support.
//!
//! Many coding agents emit `rg`-flavoured invocations (`--smart-case`,
//! `-t rust`, `-g '!target'`, bare recursive search of the cwd). The
//! grep passthrough must not forward those blindly to real `grep`, where
//! they are usage errors or — worse — silently different semantics.
//!
//! Strategy, in order:
//! 1. If a real `ripgrep` binary exists on the system, delegate to it
//!    byte-exactly (same guarantee as the grep passthrough).
//! 2. Otherwise translate the common, safely-mappable flag subset to a
//!    real-`grep` invocation (`-E` engine, recursive by default, smart-case
//!    resolved against the pattern, `--glob`/`--type` mapped to
//!    `--include`/`--exclude`).
//! 3. Flags whose semantics grep cannot reproduce (`--files`, `--json`,
//!    `--replace`, `--multiline`, …) fail LOUDLY with the reason and the
//!    closest alternative — never a silently wrong search.
//!
//! Translated output is not byte-identical to ripgrep's, but matches
//! ripgrep's piped format for the common case: `rg PAT` piped prints
//! `path:line-text` without line numbers, exactly like `grep -r PAT .`.

use std::ffi::{OsStr, OsString};

use greppy_core::error::{Error, Result};

/// Discover a real `ripgrep` binary, if any.
///
/// Discovery order mirrors [`crate::discover_grep`]:
/// 1. `GREPPY_REAL_RG` env override. An empty value forces "no ripgrep"
///    (translation fallback) — used by tests and minimal images. A
///    non-empty value that is not an executable file is a config error.
/// 2. Well-known system paths.
/// 3. `which::which("rg")`, excluding the current executable and
///    `~/.greppy/shims/` so a shimmed PATH cannot recurse.
///
/// Returns `Ok(None)` when no ripgrep exists — the caller falls back to
/// grep translation; absence of ripgrep is not an error.
pub fn discover_ripgrep() -> Result<Option<std::path::PathBuf>> {
    if let Ok(p) = std::env::var("GREPPY_REAL_RG") {
        if p.is_empty() {
            return Ok(None);
        }
        let path = std::path::PathBuf::from(p);
        if path.is_file() {
            return Ok(Some(path));
        }
        return Err(Error::Config(format!(
            "GREPPY_REAL_RG={} is not an executable file",
            path.display()
        )));
    }

    for candidate in [
        "/opt/homebrew/bin/rg",
        "/usr/local/bin/rg",
        "/usr/bin/rg",
        "/bin/rg",
    ] {
        let p = std::path::PathBuf::from(candidate);
        if p.is_file() {
            return Ok(Some(p));
        }
    }

    let own_exe = std::env::current_exe().ok();
    let shim_dir =
        std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".greppy").join("shims"));
    let found = match which::which("rg") {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };
    if own_exe.as_ref().is_some_and(|own| own == &found) {
        return Ok(None);
    }
    if let Some(ref sd) = shim_dir {
        if found.starts_with(sd) {
            return Ok(None);
        }
    }
    Ok(Some(found))
}

/// Tokens that only exist in ripgrep's flag surface (or that agents only
/// ever mean in the ripgrep sense). Presence of any of these switches the
/// passthrough into rg mode.
///
/// Deliberately NOT detectors, because real grep owns them: `-s`
/// (grep: no-messages), `-T` (GNU grep: initial-tab), `--no-messages`,
/// `--files-without-match`, `-b`/`--byte-offset`. Combined clusters such
/// as `-RS` (BSD grep: recursive + follow) are also left alone — only the
/// standalone `-S` token is claimed for smart-case.
const RG_ONLY_EXACT: &[&str] = &[
    "--smart-case",
    "-S",
    "--files",
    "--type-list",
    "--type-add",
    "--type-clear",
    "--type-not",
    "--hidden",
    "--no-ignore",
    "--no-ignore-vcs",
    "--no-ignore-parent",
    "--no-ignore-dot",
    "--no-ignore-global",
    "--no-ignore-files",
    "--no-config",
    "--no-heading",
    "--heading",
    "--vimgrep",
    "--column",
    "--pcre2",
    "--multiline",
    "--multiline-dotall",
    "-uu",
    "-uuu",
    "--follow",
    "--iglob",
    "--glob",
    "-g",
    "-t",
    "--type",
    "--case-sensitive",
    "--passthru",
    "--count-matches",
    "--search-zip",
    "--crlf",
    "--trim",
    "--stats",
    "--json",
    "--sort",
    "--sortr",
    "--engine",
    "--threads",
    "--max-columns",
    "--max-columns-preview",
    "--max-depth",
    "--max-filesize",
    "--replace",
    "--one-file-system",
    "--pretty",
];

const RG_ONLY_PREFIX: &[&str] = &[
    "--glob=",
    "--iglob=",
    "--type=",
    "--type-not=",
    "--type-add=",
    "--max-columns=",
    "--max-depth=",
    "--max-filesize=",
    "--sort=",
    "--sortr=",
    "--engine=",
    "--threads=",
    "--replace=",
    "--colors=",
];

/// Heuristic: does this argv tail (no binary/placeholder token) look like a
/// ripgrep invocation rather than a grep one?
pub fn is_rg_style(args: &[OsString]) -> bool {
    for tok in args {
        if tok == "--" {
            return false;
        }
        let Some(s) = tok.to_str() else { continue };
        if RG_ONLY_EXACT.contains(&s) {
            return true;
        }
        if RG_ONLY_PREFIX.iter().any(|p| s.starts_with(p)) {
            return true;
        }
        // Attached type selector: `-tpy`, `-trust`. No grep dialect has a
        // `-t` flag at all, so any such token would be a grep usage error.
        if let Some(rest) = s.strip_prefix("-t") {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_alphanumeric()) {
                return true;
            }
        }
    }
    false
}

/// File-name globs for ripgrep's common `--type` names. Pragmatic subset:
/// the types agents actually pass. Unknown types refuse loudly (the caller
/// names the type and suggests `--glob` or installing ripgrep).
fn type_globs(name: &str) -> Option<&'static [&'static str]> {
    Some(match name {
        "rust" => &["*.rs"],
        "py" | "python" => &["*.py", "*.pyi"],
        "js" | "javascript" => &["*.js", "*.jsx", "*.mjs", "*.cjs"],
        "ts" | "typescript" => &["*.ts", "*.tsx"],
        "go" | "golang" => &["*.go"],
        "java" => &["*.java"],
        "c" => &["*.c", "*.h"],
        "cpp" | "cxx" => &["*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh", "*.hxx", "*.h"],
        "cs" | "csharp" => &["*.cs"],
        "rb" | "ruby" => &["*.rb", "*.gemspec", "Rakefile"],
        "php" => &["*.php"],
        "swift" => &["*.swift"],
        "kotlin" => &["*.kt", "*.kts"],
        "scala" => &["*.scala", "*.sbt"],
        "sh" | "shell" => &["*.sh", "*.bash", "*.zsh", "*.ksh"],
        "md" | "markdown" => &["*.md", "*.markdown"],
        "json" => &["*.json"],
        "yaml" => &["*.yaml", "*.yml"],
        "toml" => &["*.toml"],
        "html" => &["*.html", "*.htm"],
        "css" => &["*.css"],
        "sql" => &["*.sql"],
        "xml" => &["*.xml"],
        "txt" => &["*.txt"],
        "lua" => &["*.lua"],
        "haskell" | "hs" => &["*.hs"],
        "elixir" | "ex" => &["*.ex", "*.exs"],
        "erlang" => &["*.erl", "*.hrl"],
        "r" => &["*.r", "*.R"],
        "dart" => &["*.dart"],
        "zig" => &["*.zig"],
        "vue" => &["*.vue"],
        "svelte" => &["*.svelte"],
        "protobuf" | "proto" => &["*.proto"],
        "cmake" => &["CMakeLists.txt", "*.cmake"],
        "make" | "mk" => &["Makefile", "makefile", "GNUmakefile", "*.mk"],
        "docker" => &["Dockerfile", "*.dockerfile"],
        "tex" => &["*.tex", "*.sty", "*.cls", "*.bib"],
        _ => return None,
    })
}

fn refuse(flag: &str, hint: &str) -> String {
    format!(
        "greppy: ripgrep flag '{flag}' has no grep translation and no ripgrep \
         binary was found (checked GREPPY_REAL_RG, system paths, PATH). {hint}"
    )
}

const HINT_INSTALL: &str = "Install ripgrep, or restate the search with plain grep flags.";

/// State collected while walking the rg argv.
#[derive(Default)]
struct Translation {
    flags: Vec<OsString>,
    patterns: Vec<OsString>,
    pattern_files: Vec<OsString>,
    paths: Vec<OsString>,
    fixed_strings: bool,
    pcre: bool,
    smart_case: bool,
    explicit_icase: bool,
    follow: bool,
}

impl Translation {
    fn glob(&mut self, val: &OsStr) -> std::result::Result<(), String> {
        let Some(s) = val.to_str() else {
            return Err(refuse("--glob", HINT_INSTALL));
        };
        let (negated, body) = match s.strip_prefix('!') {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        if body.contains('/') {
            return Err(refuse(
                &format!("--glob {s}"),
                "grep --include/--exclude match basenames only; \
                 path-shaped globs need real ripgrep.",
            ));
        }
        if negated {
            self.flags.push(format!("--exclude={body}").into());
            self.flags.push(format!("--exclude-dir={body}").into());
        } else {
            self.flags.push(format!("--include={body}").into());
        }
        Ok(())
    }

    fn type_selector(&mut self, val: &OsStr, negated: bool) -> std::result::Result<(), String> {
        let name = val.to_string_lossy();
        let Some(globs) = type_globs(&name) else {
            return Err(refuse(
                &format!("--type{} {name}", if negated { "-not" } else { "" }),
                "Unknown type for the grep fallback; use --glob '*.EXT' or install ripgrep.",
            ));
        };
        for g in globs {
            if negated {
                self.flags.push(format!("--exclude={g}").into());
            } else {
                self.flags.push(format!("--include={g}").into());
            }
        }
        Ok(())
    }
}

/// Translate a ripgrep-style argv tail into a real-grep argv tail.
///
/// `stdin_piped` selects ripgrep's stdin mode: with no path arguments and
/// piped stdin, rg searches stdin (so the translation must NOT inject `-r .`).
///
/// Returns the grep arguments (no argv0), or a loud human+agent-readable
/// refusal naming the untranslatable flag and the closest alternative.
pub fn translate_to_grep(
    args: &[OsString],
    stdin_piped: bool,
) -> std::result::Result<Vec<OsString>, String> {
    let mut t = Translation::default();
    let mut i = 0;
    let mut positional_only = false;

    // Fetch the value for a long/short flag: attached (`--flag=v` already
    // split by the caller) or the next token.
    fn next_value<'a>(
        args: &'a [OsString],
        i: &mut usize,
        flag: &str,
    ) -> std::result::Result<&'a OsStr, String> {
        *i += 1;
        args.get(*i)
            .map(|v| v.as_os_str())
            .ok_or_else(|| format!("greppy: ripgrep flag '{flag}' is missing its value"))
    }

    while i < args.len() {
        let tok = &args[i];
        if positional_only {
            if t.patterns.is_empty() && t.pattern_files.is_empty() && t.paths.is_empty() {
                t.patterns.push(tok.clone());
            } else {
                t.paths.push(tok.clone());
            }
            i += 1;
            continue;
        }
        if tok == "--" {
            positional_only = true;
            i += 1;
            continue;
        }
        let lossy = tok.to_string_lossy();
        if lossy.starts_with("--") {
            let (name, attached) = match lossy.find('=') {
                Some(pos) => (
                    lossy[..pos].to_string(),
                    Some(OsString::from(&lossy[pos + 1..])),
                ),
                None => (lossy.to_string(), None),
            };
            let value = |i: &mut usize| -> std::result::Result<OsString, String> {
                match &attached {
                    Some(v) => Ok(v.clone()),
                    None => next_value(args, i, &name).map(|v| v.to_os_string()),
                }
            };
            match name.as_str() {
                // Identical or directly-mappable semantics.
                "--ignore-case" => t.explicit_icase = true,
                "--invert-match" => t.flags.push("-v".into()),
                "--word-regexp" => t.flags.push("-w".into()),
                "--line-regexp" => t.flags.push("-x".into()),
                "--count" => t.flags.push("-c".into()),
                "--files-with-matches" => t.flags.push("-l".into()),
                "--files-without-match" => t.flags.push("-L".into()),
                "--only-matching" => t.flags.push("-o".into()),
                "--quiet" => t.flags.push("-q".into()),
                "--line-number" => t.flags.push("-n".into()),
                "--with-filename" => t.flags.push("-H".into()),
                "--no-filename" => t.flags.push("-h".into()),
                "--text" => t.flags.push("-a".into()),
                "--null" => t.flags.push("--null".into()),
                "--byte-offset" => t.flags.push("-b".into()),
                "--line-buffered" => t.flags.push("--line-buffered".into()),
                "--no-messages" => t.flags.push("-s".into()),
                "--fixed-strings" => t.fixed_strings = true,
                "--pcre2" => t.pcre = true,
                "--smart-case" => t.smart_case = true,
                "--follow" => t.follow = true,
                "--max-count" => {
                    let v = value(&mut i)?;
                    t.flags.push("-m".into());
                    t.flags.push(v);
                }
                "--after-context" | "--before-context" | "--context" => {
                    let short = match name.as_str() {
                        "--after-context" => "-A",
                        "--before-context" => "-B",
                        _ => "-C",
                    };
                    let v = value(&mut i)?;
                    t.flags.push(short.into());
                    t.flags.push(v);
                }
                "--color" => {
                    let v = value(&mut i)?;
                    let mut f = OsString::from("--color=");
                    f.push(&v);
                    t.flags.push(f);
                }
                "--regexp" => {
                    let v = value(&mut i)?;
                    t.patterns.push(v);
                }
                "--file" => {
                    let v = value(&mut i)?;
                    t.pattern_files.push(v);
                }
                "--glob" | "--iglob" => {
                    let v = value(&mut i)?;
                    t.glob(&v)?;
                }
                "--type" => {
                    let v = value(&mut i)?;
                    t.type_selector(&v, false)?;
                }
                "--type-not" => {
                    let v = value(&mut i)?;
                    t.type_selector(&v, true)?;
                }
                // No grep concept, but dropping only widens the search or
                // changes cosmetics — never produces wrong matches.
                "--case-sensitive" | "--no-line-number" | "--heading" | "--no-heading"
                | "--hidden" | "--no-ignore" | "--no-ignore-vcs" | "--no-ignore-parent"
                | "--no-ignore-dot" | "--no-ignore-global" | "--no-ignore-files"
                | "--no-config" | "--no-require-git" | "--crlf" | "--trim" | "--stats"
                | "--binary" | "--no-mmap" | "--mmap" | "--pretty" | "--one-file-system"
                | "--block-buffered" | "--no-unicode" | "--unicode" => {}
                "--colors"
                | "--sort"
                | "--sortr"
                | "--threads"
                | "--max-columns"
                | "--max-filesize"
                | "--regex-size-limit"
                | "--dfa-size-limit"
                | "--ignore-file"
                | "--context-separator"
                | "--field-context-separator"
                | "--field-match-separator"
                | "--hyperlink-format" => {
                    let _ = value(&mut i)?; // cosmetic/perf: consume and drop
                }
                "--engine" => {
                    let v = value(&mut i)?;
                    if v == "pcre2" {
                        t.pcre = true;
                    }
                }
                // Semantics grep cannot reproduce: refuse loudly.
                "--files" => {
                    return Err(refuse(
                        "--files",
                        "List files with `find PATH -type f`, or install ripgrep.",
                    ));
                }
                "--replace" => {
                    return Err(refuse(
                        "--replace",
                        "grep never rewrites files; for guarded in-place rewrites \
                         use `greppy edit regex-cas`.",
                    ));
                }
                "--json" | "--vimgrep" | "--column" => {
                    return Err(refuse(&name, "This output format needs real ripgrep."));
                }
                "--multiline" | "--multiline-dotall" | "--search-zip" | "--encoding"
                | "--passthru" | "--count-matches" | "--max-depth" | "--type-add"
                | "--type-clear" | "--type-list" | "--pre" | "--pre-glob" | "--null-data" => {
                    return Err(refuse(&name, HINT_INSTALL));
                }
                other => {
                    return Err(refuse(other, HINT_INSTALL));
                }
            }
            i += 1;
            continue;
        }
        if lossy.starts_with('-') && lossy.len() > 1 {
            // Short flag cluster; value-takers consume the rest of the
            // token (attached form) or the next token.
            let cluster: Vec<char> = lossy[1..].chars().collect();
            let mut ci = 0;
            while ci < cluster.len() {
                let c = cluster[ci];
                let attached_rest = || -> Option<OsString> {
                    let rest: String = cluster[ci + 1..].iter().collect();
                    if rest.is_empty() {
                        None
                    } else {
                        Some(OsString::from(rest))
                    }
                };
                let takes_value = matches!(
                    c,
                    'e' | 'f' | 'g' | 't' | 'T' | 'A' | 'B' | 'C' | 'm' | 'M' | 'j' | 'r' | 'E'
                );
                if takes_value {
                    let val = match attached_rest() {
                        Some(v) => v,
                        None => next_value(args, &mut i, &format!("-{c}"))?.to_os_string(),
                    };
                    match c {
                        'e' => t.patterns.push(val),
                        'f' => t.pattern_files.push(val),
                        'g' => t.glob(&val)?,
                        't' => t.type_selector(&val, false)?,
                        'T' => t.type_selector(&val, true)?,
                        'A' | 'B' | 'C' | 'm' => {
                            t.flags.push(format!("-{c}").into());
                            t.flags.push(val);
                        }
                        'M' | 'j' => {} // max-columns / threads: drop
                        'r' => {
                            return Err(refuse(
                                "-r (--replace)",
                                "grep never rewrites files; for guarded in-place \
                                 rewrites use `greppy edit regex-cas`.",
                            ));
                        }
                        'E' => return Err(refuse("-E (--encoding)", HINT_INSTALL)),
                        _ => unreachable!(),
                    }
                    break; // value consumed the rest of the cluster
                }
                match c {
                    'i' => t.explicit_icase = true,
                    'v' | 'w' | 'x' | 'c' | 'l' | 'o' | 'q' | 'n' | 'H' | 'a' => {
                        t.flags.push(format!("-{c}").into());
                    }
                    'I' => t.flags.push("-h".into()),
                    'F' => t.fixed_strings = true,
                    'P' => t.pcre = true,
                    'S' => t.smart_case = true,
                    'L' => t.follow = true,
                    '0' => t.flags.push("--null".into()),
                    'N' | 's' | 'u' | 'p' => {} // cosmetic / already-default
                    'U' => return Err(refuse("-U (--multiline)", HINT_INSTALL)),
                    'z' => return Err(refuse("-z (--search-zip)", HINT_INSTALL)),
                    other => {
                        return Err(refuse(&format!("-{other}"), HINT_INSTALL));
                    }
                }
                ci += 1;
            }
            i += 1;
            continue;
        }
        // Positional: first is the pattern (unless -e/-f supplied one),
        // the rest are paths.
        if t.patterns.is_empty() && t.pattern_files.is_empty() && t.paths.is_empty() {
            t.patterns.push(tok.clone());
        } else {
            t.paths.push(tok.clone());
        }
        i += 1;
    }

    if t.patterns.is_empty() && t.pattern_files.is_empty() {
        return Err(
            "greppy: ripgrep-style invocation without a pattern (use PATTERN [PATH..])".to_string(),
        );
    }

    let mut out: Vec<OsString> = Vec::new();
    // Engine: ripgrep's default regex dialect is closest to ERE.
    if t.pcre {
        out.push("-P".into());
    } else if t.fixed_strings {
        out.push("-F".into());
    } else {
        out.push("-E".into());
    }
    // Smart case: insensitive iff no pattern contains an uppercase letter.
    // With pattern files we cannot inspect the patterns; stay sensitive.
    let any_upper = t
        .patterns
        .iter()
        .any(|p| p.to_string_lossy().chars().any(|c| c.is_uppercase()));
    if t.explicit_icase || (t.smart_case && t.pattern_files.is_empty() && !any_upper) {
        out.push("-i".into());
    }
    out.extend(t.flags);
    for p in &t.patterns {
        out.push("-e".into());
        out.push(p.clone());
    }
    for f in &t.pattern_files {
        out.push("-f".into());
        out.push(f.clone());
    }
    let stdin_mode = t.paths.is_empty() && stdin_piped;
    if !stdin_mode {
        out.push(if t.follow { "-R" } else { "-r" }.into());
        out.push("--".into());
        if t.paths.is_empty() {
            out.push(".".into());
        } else {
            out.extend(t.paths);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn os(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    fn strs(out: &[OsString]) -> Vec<String> {
        out.iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn detects_rg_only_flags() {
        assert!(is_rg_style(&os(&["--smart-case", "foo"])));
        assert!(is_rg_style(&os(&["-S", "foo"])));
        assert!(is_rg_style(&os(&["-tpy", "foo"])));
        assert!(is_rg_style(&os(&["--glob=*.rs", "foo"])));
        assert!(is_rg_style(&os(&["-g", "!target", "foo"])));
        assert!(is_rg_style(&os(&["--files"])));
    }

    #[test]
    fn leaves_grep_invocations_alone() {
        assert!(!is_rg_style(&os(&["-Rn", "foo", "."])));
        // BSD grep's combined recursive+follow cluster stays grep.
        assert!(!is_rg_style(&os(&["-RS", "foo", "."])));
        assert!(!is_rg_style(&os(&["-s", "foo", "f.txt"])));
        assert!(!is_rg_style(&os(&["--include=*.rs", "-r", "foo", "."])));
        // Everything after `--` is positional, never a flag.
        assert!(!is_rg_style(&os(&["-n", "foo", "--", "--smart-case"])));
    }

    #[test]
    fn smart_case_lowercase_pattern_becomes_insensitive() {
        let out = translate_to_grep(&os(&["--smart-case", "beta"]), false).unwrap();
        let s = strs(&out);
        assert_eq!(s, ["-E", "-i", "-e", "beta", "-r", "--", "."]);
    }

    #[test]
    fn smart_case_uppercase_pattern_stays_sensitive() {
        let out = translate_to_grep(&os(&["-S", "Beta"]), false).unwrap();
        let s = strs(&out);
        assert_eq!(s, ["-E", "-e", "Beta", "-r", "--", "."]);
    }

    #[test]
    fn type_selector_maps_to_includes() {
        let out = translate_to_grep(&os(&["-trust", "foo", "src"]), false).unwrap();
        let s = strs(&out);
        assert_eq!(s, ["-E", "--include=*.rs", "-e", "foo", "-r", "--", "src"]);
    }

    #[test]
    fn negated_glob_maps_to_excludes() {
        let out = translate_to_grep(&os(&["-g", "!target", "foo"]), false).unwrap();
        let s = strs(&out);
        assert!(s.contains(&"--exclude=target".to_string()));
        assert!(s.contains(&"--exclude-dir=target".to_string()));
    }

    #[test]
    fn path_shaped_glob_refuses() {
        let err = translate_to_grep(&os(&["-g", "src/**/*.rs", "foo"]), false).unwrap_err();
        assert!(err.contains("--glob"), "{err}");
    }

    #[test]
    fn files_flag_refuses_with_find_hint() {
        let err = translate_to_grep(&os(&["--files"]), false).unwrap_err();
        assert!(err.contains("find PATH -type f"), "{err}");
    }

    #[test]
    fn replace_refuses_with_edit_hint() {
        let err = translate_to_grep(&os(&["foo", "-r", "bar"]), false).unwrap_err();
        assert!(err.contains("greppy edit regex-cas"), "{err}");
    }

    #[test]
    fn cluster_with_attached_context_value() {
        let out = translate_to_grep(&os(&["-inA3", "foo", "src"]), false).unwrap();
        let s = strs(&out);
        assert_eq!(
            s,
            ["-E", "-i", "-n", "-A", "3", "-e", "foo", "-r", "--", "src"]
        );
    }

    #[test]
    fn fixed_strings_switches_engine() {
        let out = translate_to_grep(&os(&["-F", "a.b(c)", "src"]), false).unwrap();
        assert_eq!(strs(&out)[0], "-F");
    }

    #[test]
    fn stdin_mode_omits_recursion_and_path() {
        let out = translate_to_grep(&os(&["--smart-case", "beta"]), true).unwrap();
        let s = strs(&out);
        assert_eq!(s, ["-E", "-i", "-e", "beta"]);
    }

    #[test]
    fn follow_uses_capital_r() {
        let out = translate_to_grep(&os(&["-L", "foo", "src"]), false).unwrap();
        assert!(strs(&out).contains(&"-R".to_string()));
    }

    #[test]
    fn dashes_split_pattern_and_paths() {
        let out = translate_to_grep(&os(&["-n", "--", "-weird", "dir"]), false).unwrap();
        let s = strs(&out);
        assert_eq!(s, ["-E", "-n", "-e", "-weird", "-r", "--", "dir"]);
    }

    #[test]
    fn unknown_long_flag_refuses_loudly() {
        let err = translate_to_grep(&os(&["--frobnicate", "foo"]), false).unwrap_err();
        assert!(err.contains("--frobnicate"), "{err}");
    }

    #[test]
    fn missing_pattern_refuses() {
        let err = translate_to_grep(&os(&["--smart-case"]), false).unwrap_err();
        assert!(err.contains("without a pattern"), "{err}");
    }

    #[test]
    fn explicit_regexp_makes_first_positional_a_path() {
        let out = translate_to_grep(&os(&["-e", "foo", "src"]), false).unwrap();
        let s = strs(&out);
        assert_eq!(s, ["-E", "-e", "foo", "-r", "--", "src"]);
    }
}
