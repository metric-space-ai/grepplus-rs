use std::path::{Path, PathBuf};

use greppy_embed_native::MetalEmbeddingModel;
use serde::Deserialize;

const COSINE_TOLERANCE: f64 = 0.998;

#[derive(Debug, Deserialize)]
struct SingleGolden {
    id: usize,
    token_ids: Vec<u32>,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct BatchGolden {
    token_ids: Vec<Vec<u32>>,
    attention_mask: Vec<Vec<u32>>,
    embeddings: Vec<Vec<f32>>,
}

fn main() {
    let Some(gguf_path) = std::env::var_os("EMBED_NATIVE_GGUF") else {
        eprintln!("skipping Metal forward parity: EMBED_NATIVE_GGUF is unset");
        return;
    };

    let model = MetalEmbeddingModel::open(&gguf_path)
        .unwrap_or_else(|e| panic!("open GGUF {}: {e}", Path::new(&gguf_path).display()));

    verify_single(&model);
    verify_batch(&model);
}

fn verify_single(model: &MetalEmbeddingModel) {
    let singles: Vec<SingleGolden> = read_json("golden_single.json");
    let mut min_cos = 1.0f64;
    for golden in singles {
        let mask = vec![vec![1u32; golden.token_ids.len()]];
        let native = model
            .forward_tokens(std::slice::from_ref(&golden.token_ids), &mask)
            .unwrap_or_else(|e| panic!("Metal forward single[{}]: {e}", golden.id));
        let cos = cosine(&native[0], &golden.embedding);
        eprintln!("metal single id={} cosine={cos:.9}", golden.id);
        min_cos = min_cos.min(cos);
        assert!(
            cos >= COSINE_TOLERANCE,
            "single[{}] cosine {cos:.9} < {COSINE_TOLERANCE}",
            golden.id
        );
    }
    eprintln!("metal single min cosine={min_cos:.9}");
}

fn verify_batch(model: &MetalEmbeddingModel) {
    let golden: BatchGolden = read_json("golden_batch.json");
    let native = model
        .forward_tokens(&golden.token_ids, &golden.attention_mask)
        .unwrap_or_else(|e| panic!("Metal forward batch: {e}"));
    let mut min_cos = 1.0f64;
    for (idx, (native, expected)) in native.iter().zip(&golden.embeddings).enumerate() {
        let cos = cosine(native, expected);
        eprintln!("metal batch row={idx} cosine={cos:.9}");
        min_cos = min_cos.min(cos);
        assert!(
            cos >= COSINE_TOLERANCE,
            "batch[{idx}] cosine {cos:.9} < {COSINE_TOLERANCE}",
        );
    }
    eprintln!("metal batch min cosine={min_cos:.9}");
}

fn read_json<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("golden")
        .join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn cosine(lhs: &[f32], rhs: &[f32]) -> f64 {
    assert_eq!(lhs.len(), rhs.len(), "cosine length mismatch");
    let mut dot = 0.0f64;
    let mut lhs_norm = 0.0f64;
    let mut rhs_norm = 0.0f64;
    for (&l, &r) in lhs.iter().zip(rhs) {
        let l = f64::from(l);
        let r = f64::from(r);
        dot += l * r;
        lhs_norm += l * l;
        rhs_norm += r * r;
    }
    dot / (lhs_norm.sqrt() * rhs_norm.sqrt()).max(1.0e-12)
}
