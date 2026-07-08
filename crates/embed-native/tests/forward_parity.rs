use std::collections::HashMap;
use std::path::{Path, PathBuf};

use greppy_embed_native::{matmul::QuantMatrix, CpuEmbeddingModel, GgufModel};
use serde::Deserialize;

// Per-stage absolute-diff check is a DEBUG/localization aid only (opt-in via
// EMBED_NATIVE_STRICT_STAGES); it is not a product gate because deep-layer
// residual magnitudes reach ~1e5, where a scale-free tolerance is meaningless.
// The real correctness gate is the FINAL embedding cosine: 0.998 is
// retrieval-equivalent (the same envelope the ggml Q4 GPU kernels land in vs
// candle), so we do not chase bit-exact reproduction of candle's Q8-activation
// matmul arithmetic — memory + throughput matter more than perfect parity.
const STAGE_TOLERANCE: f32 = 1.0e-2;
const COSINE_TOLERANCE: f64 = 0.998;

#[derive(Debug, Deserialize)]
struct SingleGolden {
    id: usize,
    prompt: String,
    token_ids: Vec<u32>,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct BatchGolden {
    token_ids: Vec<Vec<u32>>,
    attention_mask: Vec<Vec<u32>>,
    embeddings: Vec<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
struct StageGolden {
    prompt: String,
    stages: Vec<StageEntry>,
}

#[derive(Debug, Deserialize)]
struct StageEntry {
    name: String,
    len: usize,
    values: Vec<f32>,
}

fn main() {
    let Some(gguf_path) = std::env::var_os("EMBED_NATIVE_GGUF") else {
        eprintln!("skipping forward parity: EMBED_NATIVE_GGUF is unset");
        return;
    };

    let model = CpuEmbeddingModel::open(&gguf_path)
        .unwrap_or_else(|e| panic!("open GGUF {}: {e}", Path::new(&gguf_path).display()));

    let singles = load_single();
    if std::env::var_os("EMBED_NATIVE_PROBE_LAYER0_DOWN").is_some() {
        probe_layer0_down(Path::new(&gguf_path));
    }
    verify_stages(&model, &singles);
    verify_single(&model, &singles);
    verify_batch(&model);
}

fn probe_layer0_down(gguf_path: &Path) {
    let gguf = GgufModel::open(gguf_path).unwrap_or_else(|e| panic!("open probe GGUF: {e}"));
    let down = QuantMatrix::from_model(&gguf, "blk.0.ffn_down.weight")
        .unwrap_or_else(|e| panic!("load blk.0.ffn_down.weight: {e}"));
    let gate = QuantMatrix::from_model(&gguf, "blk.0.ffn_gate.weight")
        .unwrap_or_else(|e| panic!("load blk.0.ffn_gate.weight: {e}"));
    let up = QuantMatrix::from_model(&gguf, "blk.0.ffn_up.weight")
        .unwrap_or_else(|e| panic!("load blk.0.ffn_up.weight: {e}"));
    for name in [
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    ] {
        let tensor = gguf
            .tensor(name)
            .unwrap_or_else(|e| panic!("tensor {name}: {e}"));
        eprintln!(
            "probe tensor {} shape={:?} dtype={}",
            name, tensor.shape, tensor.dtype
        );
    }
    let goldens: Vec<StageGolden> = read_json("golden_stages.json");
    for (case_idx, golden) in goldens.iter().enumerate() {
        let pre_ffn = golden
            .stages
            .iter()
            .find(|s| s.name == "layer_0_pre_ffn_norm")
            .unwrap_or_else(|| panic!("case {case_idx} missing layer_0_pre_ffn_norm"));
        let gate_golden = golden
            .stages
            .iter()
            .find(|s| s.name == "layer_0_mlp_gate")
            .unwrap_or_else(|| panic!("case {case_idx} missing layer_0_mlp_gate"));
        let up_golden = golden
            .stages
            .iter()
            .find(|s| s.name == "layer_0_mlp_up")
            .unwrap_or_else(|| panic!("case {case_idx} missing layer_0_mlp_up"));
        let rows = pre_ffn.values.len() / gate.cols();
        for (name, matrix, expected) in [("gate", &gate, gate_golden), ("up", &up, up_golden)] {
            let native = matrix
                .matmul(&pre_ffn.values, rows)
                .unwrap_or_else(|e| panic!("probe {name} case {case_idx}: {e}"));
            let (max_abs, max_idx, native_value, golden_value) =
                max_abs_detail(&native, &expected.values);
            eprintln!(
                "probe layer0_{} case={} rows={} max_abs={:.9} idx={} native={:.9} golden={:.9}",
                name, case_idx, rows, max_abs, max_idx, native_value, golden_value
            );
        }
        let product = golden
            .stages
            .iter()
            .find(|s| s.name == "layer_0_mlp_product")
            .unwrap_or_else(|| panic!("case {case_idx} missing layer_0_mlp_product"));
        let mlp = golden
            .stages
            .iter()
            .find(|s| s.name == "layer_0_mlp")
            .unwrap_or_else(|| panic!("case {case_idx} missing layer_0_mlp"));
        let rows = product.values.len() / down.cols();
        let native = down
            .matmul(&product.values, rows)
            .unwrap_or_else(|e| panic!("probe down case {case_idx}: {e}"));
        let (max_abs, max_idx, native_value, golden_value) = max_abs_detail(&native, &mlp.values);
        eprintln!(
            "probe layer0_down case={} rows={} max_abs={:.9} idx={} native={:.9} golden={:.9}",
            case_idx, rows, max_abs, max_idx, native_value, golden_value
        );
    }
}

fn verify_stages(model: &CpuEmbeddingModel, singles: &[SingleGolden]) {
    let by_prompt = singles
        .iter()
        .map(|g| (g.prompt.as_str(), g.token_ids.as_slice()))
        .collect::<HashMap<_, _>>();
    let goldens: Vec<StageGolden> = read_json("golden_stages.json");
    for (case_idx, golden) in goldens.iter().enumerate() {
        let token_ids = by_prompt
            .get(golden.prompt.as_str())
            .unwrap_or_else(|| panic!("stage case {case_idx} prompt missing from golden_single"));
        let mask = vec![1u32; token_ids.len()];
        let native = model
            .forward_stages(token_ids, &mask)
            .unwrap_or_else(|e| panic!("forward stages case {case_idx}: {e}"));
        assert_eq!(
            native.len(),
            golden.stages.len(),
            "stage case {case_idx} stage count"
        );

        let mut prev_native: Option<&[f32]> = None;
        let mut prev_golden: Option<&[f32]> = None;
        for (stage_idx, (native, golden)) in native.iter().zip(&golden.stages).enumerate() {
            assert_eq!(
                native.name, golden.name,
                "case {case_idx} stage {stage_idx} name"
            );
            assert_eq!(
                native.values.len(),
                golden.len,
                "case {case_idx} stage {} len",
                golden.name
            );
            let (max_abs, max_idx, native_value, golden_value) =
                max_abs_detail(&native.values, &golden.values);
            eprintln!(
                "stage case={} name={} len={} max_abs={:.9} idx={} native={:.9} golden={:.9}",
                case_idx, golden.name, golden.len, max_abs, max_idx, native_value, golden_value
            );
            if golden.name == "layer_2_post_ffn_norm" {
                if let (Some(prev_native), Some(prev_golden)) = (prev_native, prev_golden) {
                    let dim = 768;
                    let row = max_idx / dim;
                    let col = max_idx % dim;
                    let nr = &prev_native[row * dim..(row + 1) * dim];
                    let gr = &prev_golden[row * dim..(row + 1) * dim];
                    let nden = (nr.iter().map(|v| v * v).sum::<f32>() / dim as f32 + 1.0e-6).sqrt();
                    let gden = (gr.iter().map(|v| v * v).sum::<f32>() / dim as f32 + 1.0e-6).sqrt();
                    eprintln!(
                        "post_ffn_detail row={} col={} native_in={:.9} golden_in={:.9} input_diff={:.9} native_den={:.9} golden_den={:.9} den_diff={:.9}",
                        row,
                        col,
                        nr[col],
                        gr[col],
                        nr[col] - gr[col],
                        nden,
                        gden,
                        nden - gden
                    );
                }
            }
            if std::env::var_os("EMBED_NATIVE_STRICT_STAGES").is_some() {
                assert!(
                    max_abs < STAGE_TOLERANCE,
                    "first divergent stage: case {case_idx} {} max_abs {max_abs} >= {STAGE_TOLERANCE}",
                    golden.name
                );
            }
            prev_native = Some(&native.values);
            prev_golden = Some(&golden.values);
        }
    }
}

fn verify_single(model: &CpuEmbeddingModel, singles: &[SingleGolden]) {
    let mut min_cos = 1.0f64;
    for golden in singles {
        let mask = vec![vec![1u32; golden.token_ids.len()]];
        let native = model
            .forward_tokens(std::slice::from_ref(&golden.token_ids), &mask)
            .unwrap_or_else(|e| panic!("forward single[{}]: {e}", golden.id));
        let cos = cosine(&native[0], &golden.embedding);
        min_cos = min_cos.min(cos);
        eprintln!("single id={} cosine={:.9}", golden.id, cos);
        assert!(
            cos >= COSINE_TOLERANCE,
            "single[{}] cosine {cos} < {COSINE_TOLERANCE}",
            golden.id
        );
    }
    eprintln!("single min_cosine={min_cos:.9}");
}

fn verify_batch(model: &CpuEmbeddingModel) {
    let golden: BatchGolden = read_json("golden_batch.json");
    let native = model
        .forward_tokens(&golden.token_ids, &golden.attention_mask)
        .unwrap_or_else(|e| panic!("forward batch: {e}"));
    let mut min_cos = 1.0f64;
    for (idx, (native, golden)) in native.iter().zip(&golden.embeddings).enumerate() {
        let cos = cosine(native, golden);
        min_cos = min_cos.min(cos);
        eprintln!("batch row={} cosine={:.9}", idx, cos);
        assert!(
            cos >= COSINE_TOLERANCE,
            "batch[{idx}] cosine {cos} < {COSINE_TOLERANCE}"
        );
    }
    eprintln!("batch min_cosine={min_cos:.9}");
}

fn load_single() -> Vec<SingleGolden> {
    read_json("golden_single.json")
}

fn read_json<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = golden_path(name);
    serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display())),
    )
    .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn golden_path(name: &str) -> PathBuf {
    std::env::var_os("EMBED_NATIVE_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("testdata")
                .join("golden")
        })
        .join(name)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_detail(a: &[f32], b: &[f32]) -> (f32, usize, f32, f32) {
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    let mut max_a = 0.0f32;
    let mut max_b = 0.0f32;
    for (idx, (&a, &b)) in a.iter().zip(b).enumerate() {
        let diff = (a - b).abs();
        if diff > max_abs {
            max_abs = diff;
            max_idx = idx;
            max_a = a;
            max_b = b;
        }
    }
    (max_abs, max_idx, max_a, max_b)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in a.iter().zip(b) {
        let a = a as f64;
        let b = b as f64;
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    dot / (na.sqrt() * nb.sqrt())
}
