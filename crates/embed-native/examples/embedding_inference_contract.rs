//! Exact-token EmbeddingGemma encoder samples for the performance contract.

use std::io::BufRead;
use std::time::Instant;

use greppy_embed_native::{
    CpuEmbeddingModel, DevicePreference, EmbedTask, GgufModel, PromptTokenizer, TokenizerConfig,
};
use serde::Deserialize;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use greppy_embed_native::CudaEmbeddingModel;
#[cfg(all(feature = "metal", target_os = "macos"))]
use greppy_embed_native::MetalEmbeddingModel;

const RAW_SCHEMA_VERSION: &str = "greppy.inference-performance.raw.v1";
const SEMANTICS: &str = "embeddinggemma_encoder_forward_v1";

#[derive(Deserialize)]
struct PromptCase {
    id: String,
    source: String,
}

enum Encoder {
    Cpu(CpuEmbeddingModel),
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalEmbeddingModel),
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    Cuda(CudaEmbeddingModel),
}

impl Encoder {
    fn load(path: &str, device: DevicePreference) -> Result<Self, Box<dyn std::error::Error>> {
        match device {
            DevicePreference::Cpu => Ok(Self::Cpu(CpuEmbeddingModel::open(path)?)),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            DevicePreference::Metal => Ok(Self::Metal(MetalEmbeddingModel::open(path)?)),
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            DevicePreference::Metal => Err("Metal is unavailable in this build/platform".into()),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            DevicePreference::Cuda => Ok(Self::Cuda(CudaEmbeddingModel::open(path)?)),
            #[cfg(not(all(feature = "cuda", any(target_os = "linux", target_os = "windows"))))]
            DevicePreference::Cuda => Err("CUDA is unavailable in this build/platform".into()),
            DevicePreference::Auto => {
                Err("benchmark device must be explicit: cpu, metal, or cuda".into())
            }
        }
    }

    fn forward(
        &self,
        token_ids: &[Vec<u32>],
        attention_mask: &[Vec<u32>],
    ) -> Result<Vec<Vec<f32>>, greppy_embed_native::Error> {
        match self {
            Self::Cpu(model) => model.forward_tokens(token_ids, attention_mask),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(model) => model.forward_tokens(token_ids, attention_mask),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Self::Cuda(model) => model.forward_tokens(token_ids, attention_mask),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Self::Metal(_) => "metal",
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Self::Cuda(_) => "cuda",
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let gguf_path = args.next().ok_or(usage())?;
    let tokenizer_path = args.next().ok_or(usage())?;
    let prompts_path = args.next().ok_or(usage())?;
    let device = args.next().ok_or(usage())?.parse::<DevicePreference>()?;
    let device_name = device.as_str();
    let samples = parse_count(args.next(), "SAMPLES", 5)?;
    let warmups = parse_count(args.next(), "WARMUPS", 1)?;
    if args.next().is_some() {
        return Err(usage().into());
    }

    let gguf = GgufModel::open(&gguf_path)?;
    let tokenizer =
        PromptTokenizer::from_file(&tokenizer_path, TokenizerConfig::from_gguf(&gguf)?)?;
    let cases = read_cases(&prompts_path)?;
    let prepared = cases
        .into_iter()
        .map(|case| {
            let prompt = EmbedTask::document_with_title(Some(&case.id), &case.source);
            let batch = tokenizer.encode_prompts([prompt])?;
            if batch.batch_size() != 1 || batch.seq_len() == 0 {
                return Err(greppy_embed_native::Error::Tokenizer(format!(
                    "{}: tokenizer did not return one non-empty sequence",
                    case.id
                )));
            }
            Ok((case.id, batch.token_ids, batch.attention_mask))
        })
        .collect::<Result<Vec<_>, greppy_embed_native::Error>>()?;
    let encoder = Encoder::load(&gguf_path, device)?;

    for (case_id, token_ids, attention_mask) in prepared {
        for _ in 0..warmups {
            std::hint::black_box(encoder.forward(&token_ids, &attention_mask)?);
        }
        for sample_index in 0..samples {
            let started = Instant::now();
            let embedding = encoder.forward(&token_ids, &attention_mask)?;
            let elapsed_ns = u64::try_from(started.elapsed().as_nanos())
                .map_err(|_| "sample duration does not fit u64 nanoseconds")?;
            std::hint::black_box(&embedding);
            println!(
                "{}",
                serde_json::json!({
                    "schema_version": RAW_SCHEMA_VERSION,
                    "model_family": "embeddinggemma",
                    "workload": "embedding_encoder",
                    "semantics": SEMANTICS,
                    "generation_path": "encoder",
                    "case_id": case_id,
                    "sample_index": sample_index,
                    "elapsed_ns": elapsed_ns,
                    "input_token_ids": token_ids[0],
                    "attention_mask": attention_mask[0],
                    "output_token_ids": [],
                    "output_limit": 0,
                    "backend": encoder.name(),
                    "device": device_name,
                })
            );
        }
    }
    Ok(())
}

fn read_cases(path: &str) -> Result<Vec<PromptCase>, Box<dyn std::error::Error>> {
    let reader = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut cases = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let case: PromptCase = serde_json::from_str(&line)
            .map_err(|error| format!("{}:{}: {error}", path, index + 1))?;
        if case.id.trim().is_empty() || case.source.trim().is_empty() {
            return Err(format!("{}:{}: id and source must be non-empty", path, index + 1).into());
        }
        cases.push(case);
    }
    if cases.is_empty() {
        return Err(format!("{path}: no prompt cases").into());
    }
    Ok(cases)
}

fn parse_count(
    value: Option<String>,
    name: &str,
    default: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let count = value
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|_| format!("{name} must be a positive integer"))?
        .unwrap_or(default);
    if count == 0 {
        return Err(format!("{name} must be positive").into());
    }
    Ok(count)
}

fn usage() -> &'static str {
    "usage: embedding_inference_contract MODEL.gguf TOKENIZER.json PROMPTS.jsonl DEVICE [SAMPLES] [WARMUPS]"
}
