//! Build-time embedding of the EmbeddingGemma-300M Q4_K model.
//!
//! Owner rule: greppy works OUT OF THE BOX — semantic search must ALWAYS
//! work, so the model ships INSIDE the binary and the model files live IN THIS
//! REPO (`crates/cli/assets/embedded-model/`, tracked via Git LFS because the
//! GGUF is ~228 MB). No download, no external path, no env var is required.
//!
//! The `embedded-model` feature is ON BY DEFAULT (see Cargo.toml), so a plain
//! `cargo build` bakes the model. build.rs copies the in-repo assets into
//! `OUT_DIR` (where `lib.rs` `include_bytes!`s them) after verifying their
//! SHA256, and FAILS LOUDLY if an asset is missing or altered — a binary
//! without its model must not be buildable.
//!
//! An optional local override (`GREPPY_BUILD_MODEL_GGUF` /
//! `GREPPY_BUILD_MODEL_TOKENIZER`) exists only for offline model-refresh
//! work; it is never required and still SHA-checked.

use std::path::{Path, PathBuf};

const GGUF_NAME: &str = "embeddinggemma-300M-Q4_K.gguf";
const GGUF_SHA: &str = "53f7d1c0d5c84a81e46f3bea8e0f17c94f459ffbaa8b06f7f52f1f09e58996f2";
const TOK_NAME: &str = "tokenizer.json";
const TOK_SHA: &str = "6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e";

fn main() {
    println!("cargo:rerun-if-env-changed=GREPPY_BUILD_MODEL_GGUF");
    println!("cargo:rerun-if-env-changed=GREPPY_BUILD_MODEL_TOKENIZER");
    println!("cargo:rustc-env=GREPPY_EMBEDDED_GGUF_SHA={GGUF_SHA}");
    println!("cargo:rustc-env=GREPPY_EMBEDDED_TOK_SHA={TOK_SHA}");
    if std::env::var("CARGO_FEATURE_EMBEDDED_MODEL").is_err() {
        return;
    }
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let assets = manifest.join("assets").join("embedded-model");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    bake(
        &assets.join(GGUF_NAME),
        &out.join(GGUF_NAME),
        std::env::var("GREPPY_BUILD_MODEL_GGUF").ok(),
        GGUF_NAME,
        GGUF_SHA,
    );
    bake(
        &assets.join(TOK_NAME),
        &out.join(TOK_NAME),
        std::env::var("GREPPY_BUILD_MODEL_TOKENIZER").ok(),
        TOK_NAME,
        TOK_SHA,
    );
}

/// Copy the in-repo asset (`repo_asset`, or an explicit local `override_path`)
/// into `dest` in OUT_DIR, verifying SHA256 on both ends. Panics — with a
/// message that says exactly what is wrong — rather than baking an unverified
/// or absent model.
fn bake(repo_asset: &Path, dest: &Path, override_path: Option<String>, name: &str, want_sha: &str) {
    println!("cargo:rerun-if-changed={}", repo_asset.display());
    let src = match override_path {
        Some(p) => {
            let path = PathBuf::from(p);
            println!("cargo:rerun-if-changed={}", path.display());
            path
        }
        None => repo_asset.to_path_buf(),
    };
    assert!(
        src.exists(),
        "embedded model asset `{name}` not found at {}.\n\
         The model must live in the repo (Git LFS): run `git lfs install && git lfs pull`,\n\
         or set GREPPY_BUILD_MODEL_{} to a local copy. Refusing to build a binary without its model.",
        src.display(),
        if name == GGUF_NAME { "GGUF" } else { "TOKENIZER" },
    );
    if dest.exists() && sha256_file(dest) == want_sha {
        return; // already baked from a previous build
    }
    // A Git-LFS pointer file (a few hundred bytes) is not the real asset —
    // catch the common "forgot to lfs pull" case with a clear message.
    let got = sha256_file(&src);
    assert_eq!(
        got,
        want_sha,
        "embedded model `{name}` at {} has the wrong SHA256 (got {got}).\n\
         If this is a ~130-byte Git-LFS pointer, run `git lfs pull`. Refusing to bake an unverified model.",
        src.display(),
    );
    std::fs::copy(&src, dest)
        .unwrap_or_else(|e| panic!("copy {} -> {}: {e}", src.display(), dest.display()));
    let baked = sha256_file(dest);
    assert_eq!(
        baked,
        want_sha,
        "embedded model `{name}` copied to {} has the wrong SHA256 (got {baked}).\n\
         Refusing to bake an unverified model.",
        dest.display(),
    );
}

fn sha256_file(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut f = std::fs::File::open(path).expect("open model file for hashing");
    std::io::copy(&mut f, &mut hasher).expect("hash model file");
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
