use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use greppy_embed_native::{
    device_has_memory, estimated_gpu_memory, tokenizer::plan_tokenizer_path, BackendKind,
    DeviceInfo, DevicePreference, DeviceType, EmbedTask, EmbeddingGemma, GgufModel,
    InferenceBackendRegistry, InferenceModelKind, LoadOptions, PromptTokenizer, TokenizerConfig,
    EMBEDDING_DIM, GPU_MEMORY_SAFETY_MARGIN,
};

fn main() {
    let tmp = temp_dir();
    std::fs::create_dir_all(&tmp).expect("create robustness temp dir");

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    std::env::set_var("GREPPY_STORE_DIR", tmp.join("store"));

    bad_gguf_inputs_return_errors_without_panicking(&tmp);
    bad_tokenizers_return_errors_without_panicking(&tmp);
    tokenizer_handles_empty_and_oversize_inputs_without_panicking();
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    cuda_backend_cache_repairs_tampering(&tmp);
    backend_registry_and_memory_preflight_are_conservative();
    product_embedding_edges_do_not_panic();
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_backend_cache_repairs_tampering(tmp: &Path) {
    use std::os::unix::fs::{symlink, PermissionsExt};

    let path = greppy_embed_native::cuda::ffi::materialize_embedded_backend_for_diagnostics()
        .expect("materialize embedded CUDA backend");
    let expected_len = path.metadata().expect("CUDA backend metadata").len();

    std::fs::write(&path, b"truncated").expect("truncate CUDA backend cache");
    let repaired = greppy_embed_native::cuda::ffi::materialize_embedded_backend_for_diagnostics()
        .expect("repair truncated CUDA backend");
    assert_eq!(repaired.metadata().unwrap().len(), expected_len);

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644))
        .expect("weaken CUDA backend permissions");
    greppy_embed_native::cuda::ffi::materialize_embedded_backend_for_diagnostics()
        .expect("repair CUDA backend permissions");
    assert_eq!(path.metadata().unwrap().permissions().mode() & 0o077, 0);

    std::fs::remove_file(&path).expect("remove CUDA backend before symlink test");
    let attacker = tmp.join("attacker.so");
    std::fs::write(&attacker, b"not a CUDA backend").expect("write attacker file");
    symlink(&attacker, &path).expect("install malicious CUDA backend symlink");
    greppy_embed_native::cuda::ffi::materialize_embedded_backend_for_diagnostics()
        .expect("replace CUDA backend symlink");
    let metadata = std::fs::symlink_metadata(&path).unwrap();
    assert!(metadata.file_type().is_file());
    assert!(!metadata.file_type().is_symlink());
    assert_eq!(metadata.len(), expected_len);
    assert_eq!(std::fs::read(attacker).unwrap(), b"not a CUDA backend");

    let digest_dir = path.parent().expect("CUDA backend digest directory");
    std::fs::remove_file(&path).expect("remove CUDA backend before directory symlink test");
    std::fs::remove_dir(digest_dir).expect("remove CUDA backend digest directory");
    let redirected = tmp.join("redirected-cuda-cache");
    std::fs::create_dir(&redirected).expect("create redirected CUDA cache");
    symlink(&redirected, digest_dir).expect("install malicious cache-directory symlink");
    assert!(
        greppy_embed_native::cuda::ffi::materialize_embedded_backend_for_diagnostics().is_err(),
        "CUDA cache extraction must reject a symlinked digest directory"
    );
    std::fs::remove_file(digest_dir).expect("remove cache-directory symlink");
    greppy_embed_native::cuda::ffi::materialize_embedded_backend_for_diagnostics()
        .expect("repair CUDA backend after directory symlink rejection");
}

fn backend_registry_and_memory_preflight_are_conservative() {
    let registry = InferenceBackendRegistry::probe(DevicePreference::Cpu, true);
    assert_eq!(registry.selected_backend, Some(BackendKind::Cpu));
    assert_eq!(registry.selected_device_id.as_deref(), Some("cpu:0"));
    assert!(registry.is_satisfied());
    assert!(registry
        .selected_probe()
        .is_some_and(|probe| probe.backend_id.starts_with("greppy-cpu-")));
    let indexed = greppy_embed_native::InferencePolicy::from_selector(Some("cuda:1"), false)
        .expect("parse explicit CUDA device policy");
    assert_eq!(indexed.preference, DevicePreference::Cuda);
    assert_eq!(indexed.cuda_device_index, Some(1));
    assert!(greppy_embed_native::InferencePolicy::from_selector(Some("cuda:-1"), false).is_err());

    let required = estimated_gpu_memory(InferenceModelKind::Qwen35, 512 * 1024 * 1024);
    let exact = DeviceInfo {
        backend: BackendKind::Cuda,
        id: "cuda:0".into(),
        name: "test".into(),
        description: "test device".into(),
        device_type: DeviceType::DiscreteGpu,
        memory_free: Some(required + GPU_MEMORY_SAFETY_MARGIN),
        memory_total: Some(required + GPU_MEMORY_SAFETY_MARGIN),
        compute_capability: Some("8.6".into()),
        metal_family: None,
        capabilities: vec!["q4_k".into()],
        rejection_reason: None,
    };
    assert!(device_has_memory(&exact, required));
    let mut short = exact;
    short.memory_free = Some(required + GPU_MEMORY_SAFETY_MARGIN - 1);
    assert!(!device_has_memory(&short, required));

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    if std::env::var_os("GREPPY_REQUIRE_CUDA").is_some() {
        let cuda = InferenceBackendRegistry::probe(DevicePreference::Cuda, true);
        assert!(cuda.is_satisfied(), "CUDA probe failed: {cuda:#?}");
        let selected = cuda.selected_probe().expect("CUDA must be selected");
        assert_eq!(selected.backend, BackendKind::Cuda);
        assert_eq!(selected.abi_version, 1);
        assert!(selected
            .devices
            .iter()
            .any(|device| device.rejection_reason.is_none()));
        if selected.devices.len() > 1 {
            let policy = greppy_embed_native::InferencePolicy::from_selector(Some("cuda:1"), false)
                .expect("explicit CUDA policy");
            let explicit_registry = InferenceBackendRegistry::probe_policy(&policy, 0);
            assert_eq!(explicit_registry.selected_backend, Some(BackendKind::Cuda));
            assert_eq!(
                explicit_registry.selected_device_id.as_deref(),
                Some("cuda:1")
            );
            let explicit = greppy_embed_native::cuda::ffi::select_cuda_device(0, Some(1))
                .expect("explicit visible CUDA device index must be selectable");
            assert_eq!(explicit, 1);
        }
    }
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
        expect_err_no_panic("explicit unavailable CUDA fails", || {
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
