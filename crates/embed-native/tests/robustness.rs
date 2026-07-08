use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use greppy_embed_native::{
    tokenizer::plan_tokenizer_path, DevicePreference, EmbedTask, EmbeddingGemma, GgufModel,
    LoadOptions, PromptTokenizer, TokenizerConfig, EMBEDDING_DIM,
};

fn main() {
    let tmp = temp_dir();
    std::fs::create_dir_all(&tmp).expect("create robustness temp dir");

    bad_gguf_inputs_return_errors_without_panicking(&tmp);
    bad_tokenizers_return_errors_without_panicking(&tmp);
    tokenizer_handles_empty_and_oversize_inputs_without_panicking();
    product_embedding_edges_do_not_panic();
}

fn bad_gguf_inputs_return_errors_without_panicking(tmp: &Path) {
    let missing = tmp.join("missing.gguf");
    expect_err_no_panic("missing GGUF", || GgufModel::open(&missing));

    let garbage = tmp.join("garbage.gguf");
    std::fs::write(&garbage, b"not a gguf").expect("write garbage GGUF");
    expect_err_no_panic("garbage GGUF", || GgufModel::open(&garbage));

    let truncated = tmp.join("truncated.gguf");
    std::fs::write(&truncated, b"GGUF\x03\0").expect("write truncated GGUF");
    expect_err_no_panic("truncated GGUF", || GgufModel::open(&truncated));
}

fn bad_tokenizers_return_errors_without_panicking(tmp: &Path) {
    let missing = tmp.join("missing-tokenizer.json");
    expect_err_no_panic("missing tokenizer", || {
        PromptTokenizer::from_file(&missing, TokenizerConfig::default())
    });

    let corrupt = tmp.join("corrupt-tokenizer.json");
    std::fs::write(&corrupt, b"{not valid tokenizer json").expect("write corrupt tokenizer");
    expect_err_no_panic("corrupt tokenizer", || {
        PromptTokenizer::from_file(&corrupt, TokenizerConfig::default())
    });
}

fn tokenizer_handles_empty_and_oversize_inputs_without_panicking() {
    let Some(tokenizer_path) = tokenizer_path() else {
        eprintln!("skipping tokenizer robustness: tokenizer.json not found");
        return;
    };
    let tokenizer = no_panic("load tokenizer", || {
        PromptTokenizer::from_file(
            &tokenizer_path,
            TokenizerConfig {
                max_length: 8,
                pad_token_id: 0,
            },
        )
    });

    let empty = no_panic("empty tokenizer batch", || {
        tokenizer.encode_prompts(Vec::<String>::new())
    });
    assert!(empty.is_empty(), "empty tokenizer batch should be handled");

    let oversize = "token ".repeat(200_000);
    let encoded = no_panic("oversize tokenizer input", || {
        tokenizer.encode_prompts([oversize.as_str()])
    });
    assert_eq!(encoded.batch_size(), 1);
    assert!(
        encoded.seq_len() <= 8,
        "oversize tokenizer input should be truncated to max length, got {}",
        encoded.seq_len()
    );
    let untruncated_len = no_panic("untruncated tokenizer length", || {
        tokenizer.token_len(&oversize)
    });
    assert!(
        untruncated_len > encoded.seq_len(),
        "token_len should measure without truncation: {untruncated_len} <= {}",
        encoded.seq_len()
    );
    assert_eq!(tokenizer.max_length(), 8);

    let whitespace = no_panic("whitespace tokenizer input", || {
        tokenizer.encode_prompts([" \n\t      "])
    });
    assert_eq!(whitespace.batch_size(), 1);
    assert!(whitespace.seq_len() > 0);
}

fn product_embedding_edges_do_not_panic() {
    let Some((gguf, tokenizer)) = model_paths() else {
        eprintln!("skipping product embedding robustness: EMBED_NATIVE_GGUF/model path not set");
        return;
    };

    let model = no_panic("load CPU model", || {
        EmbeddingGemma::load_gguf(
            &gguf,
            &tokenizer,
            LoadOptions {
                device: DevicePreference::Cpu,
                max_length: Some(16),
                tokenizer_cache_dir: None,
            },
        )
    });
    assert_eq!(model.backend_name(), "cpu");

    let empty = no_panic("empty product batch", || {
        model.embed_prompts(Vec::<String>::new())
    });
    assert!(
        empty.is_empty(),
        "empty product batch should return no embeddings"
    );

    let empty_string = no_panic("empty string embedding", || {
        model.embed_one(EmbedTask::CodeRetrievalQuery, "")
    });
    assert_embedding(&empty_string);

    let whitespace = no_panic("whitespace embedding", || {
        model.embed_one(EmbedTask::CodeRetrievalQuery, " \n\t      ")
    });
    assert_embedding(&whitespace);

    let oversize = "token ".repeat(200_000);
    let truncated = no_panic("oversize product embedding", || {
        model.embed_one(EmbedTask::CodeRetrievalQuery, &oversize)
    });
    assert_embedding(&truncated);

    #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
    {
        let fallback = no_panic("unavailable CUDA falls back to CPU", || {
            EmbeddingGemma::load_gguf(
                &gguf,
                &tokenizer,
                LoadOptions {
                    device: DevicePreference::Cuda,
                    max_length: Some(8),
                    tokenizer_cache_dir: None,
                },
            )
        });
        assert_eq!(fallback.backend_name(), "cpu");
    }
}

fn expect_err_no_panic<T>(label: &str, f: impl FnOnce() -> greppy_embed_native::Result<T>) {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(_)) => panic!("{label}: expected Err, got Ok"),
        Ok(Err(err)) => println!("{label}: {err}"),
        Err(_) => panic!("{label}: panicked"),
    }
}

fn no_panic<T>(label: &str, f: impl FnOnce() -> greppy_embed_native::Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        Ok(Err(err)) => panic!("{label}: returned Err: {err}"),
        Err(_) => panic!("{label}: panicked"),
    }
}

fn model_paths() -> Option<(PathBuf, PathBuf)> {
    let gguf = std::env::var_os("EMBED_NATIVE_GGUF")
        .map(PathBuf::from)
        .or_else(|| {
            let path =
                PathBuf::from(std::env::var("GREPPY_EMBEDDINGGEMMA_GGUF").unwrap_or_default());
            path.is_file().then_some(path)
        })?;
    let tokenizer = tokenizer_path()?;
    Some((gguf, tokenizer))
}

fn tokenizer_path() -> Option<PathBuf> {
    let path = std::env::var_os("EMBED_NATIVE_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(plan_tokenizer_path()));
    path.is_file().then_some(path)
}

fn temp_dir() -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "greppy-embed-native-robustness-{}-{nanos}",
        std::process::id()
    ))
}

fn assert_embedding(vector: &[f32]) {
    assert_eq!(vector.len(), EMBEDDING_DIM);
    assert!(
        vector.iter().all(|v| v.is_finite()),
        "embedding contains non-finite values"
    );
}
