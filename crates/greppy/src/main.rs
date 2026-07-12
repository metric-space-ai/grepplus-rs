//! `greppy-grep` binary entry point.
//!
//! The original argv is forwarded as `OsString`, and stdin/stdout/stderr plus
//! the real grep exit status are preserved without index access or semantic
//! augmentation.

use std::process::ExitCode;

fn main() -> ExitCode {
    let _ = greppy_core::logging::init();
    let argv: Vec<std::ffi::OsString> = std::env::args_os().collect();

    let real = match greppy_grep::discover_grep() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("grep: {e}");
            return ExitCode::from(3);
        }
    };

    let exit = match greppy_grep::run_grep_os(&real, &argv) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("grep: {e}");
            return ExitCode::from(2);
        }
    };

    let normalized = if exit < 0 { 1 } else { exit as u8 };
    ExitCode::from(normalized)
}
