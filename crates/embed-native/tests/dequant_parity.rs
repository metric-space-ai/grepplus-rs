use std::path::Path;

use greppy_embed_native::{quant, GgufModel};
use serde::Deserialize;

const TOLERANCE: f32 = 1.0e-3;

#[derive(Debug, Deserialize)]
struct DequantGolden {
    name: String,
    dtype: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

fn main() {
    let Some(gguf_path) = std::env::var_os("EMBED_NATIVE_GGUF") else {
        eprintln!("skipping dequant parity: EMBED_NATIVE_GGUF is unset");
        return;
    };
    let golden_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("golden")
        .join("golden_dequant.json");
    let goldens: Vec<DequantGolden> = serde_json::from_str(
        &std::fs::read_to_string(&golden_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", golden_path.display())),
    )
    .unwrap_or_else(|e| panic!("parse {}: {e}", golden_path.display()));

    let model = GgufModel::open(&gguf_path)
        .unwrap_or_else(|e| panic!("open GGUF {}: {e}", Path::new(&gguf_path).display()));

    for golden in goldens {
        let tensor = model
            .tensor(&golden.name)
            .unwrap_or_else(|e| panic!("load tensor {}: {e}", golden.name));
        assert_eq!(tensor.shape, golden.shape, "{} shape mismatch", golden.name);
        let sample_len = golden.values.len();
        assert!(
            sample_len % tensor.dtype.block_size() == 0,
            "{} sample len {} must be divisible by block size {}",
            golden.name,
            sample_len,
            tensor.dtype.block_size()
        );
        let byte_len = sample_len / tensor.dtype.block_size() * tensor.dtype.type_size();
        let native = quant::dequantize(tensor.dtype, &tensor.raw_bytes[..byte_len], sample_len)
            .unwrap_or_else(|e| panic!("dequant {}: {e}", golden.name));
        let max_abs = native
            .iter()
            .zip(&golden.values)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "{} dtype={} native_dtype={} sample={} max_abs={:.9}",
            golden.name, golden.dtype, tensor.dtype, sample_len, max_abs
        );
        assert!(
            max_abs < TOLERANCE,
            "{} dequant max_abs {max_abs} >= {TOLERANCE}",
            golden.name
        );
    }
}
