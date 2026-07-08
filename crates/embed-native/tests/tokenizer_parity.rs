use std::path::{Path, PathBuf};

use greppy_embed_native::{
    tokenizer::plan_tokenizer_path, EmbedTask, PromptTokenizer, TokenizedBatch, TokenizerConfig,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct SingleGolden {
    id: usize,
    task: String,
    content: String,
    prompt: String,
    token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct BatchGolden {
    prompts: Vec<String>,
    token_ids: Vec<Vec<u32>>,
    attention_mask: Vec<Vec<u32>>,
}

fn main() {
    let Some(tokenizer_path) = tokenizer_path() else {
        eprintln!("skipping tokenizer parity: tokenizer.json not found");
        return;
    };
    let tokenizer = PromptTokenizer::from_file(&tokenizer_path, TokenizerConfig::default())
        .unwrap_or_else(|e| panic!("load tokenizer {}: {e}", tokenizer_path.display()));

    verify_single(&tokenizer);
    verify_batch(&tokenizer);
}

fn tokenizer_path() -> Option<PathBuf> {
    let path = std::env::var_os("EMBED_NATIVE_TOKENIZER")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(plan_tokenizer_path()));
    path.is_file().then_some(path)
}

fn verify_single(tokenizer: &PromptTokenizer) {
    let golden_path = golden_path("golden_single.json");
    let goldens: Vec<SingleGolden> = serde_json::from_str(
        &std::fs::read_to_string(&golden_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", golden_path.display())),
    )
    .unwrap_or_else(|e| panic!("parse {}: {e}", golden_path.display()));

    for golden in goldens {
        let task = task_from_golden_label(&golden.task)
            .unwrap_or_else(|| panic!("single[{}] unknown task {}", golden.id, golden.task));
        let prompt = task.prompt(&golden.content);
        assert_eq!(
            prompt, golden.prompt,
            "single[{}] prompt mismatch",
            golden.id
        );

        let native = tokenizer
            .encode_prompts([prompt])
            .unwrap_or_else(|e| panic!("tokenize single[{}]: {e}", golden.id));
        assert_eq!(native.batch_size(), 1, "single[{}] batch size", golden.id);
        assert_eq!(
            native.token_ids[0], golden.token_ids,
            "single[{}] token IDs",
            golden.id
        );
        assert!(
            native.attention_mask[0].iter().all(|&v| v == 1),
            "single[{}] should be unpadded",
            golden.id
        );
    }
}

fn verify_batch(tokenizer: &PromptTokenizer) {
    let golden_path = golden_path("golden_batch.json");
    let golden: BatchGolden = serde_json::from_str(
        &std::fs::read_to_string(&golden_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", golden_path.display())),
    )
    .unwrap_or_else(|e| panic!("parse {}: {e}", golden_path.display()));

    let native = tokenizer
        .encode_prompts(golden.prompts.iter().map(String::as_str))
        .unwrap_or_else(|e| panic!("tokenize batch: {e}"));
    assert_rectangular(&native);
    assert_eq!(native.token_ids, golden.token_ids, "batch token IDs");
    assert_eq!(
        native.attention_mask, golden.attention_mask,
        "batch attention mask"
    );
}

fn assert_rectangular(batch: &TokenizedBatch) {
    let seq_len = batch.seq_len();
    assert!(
        batch.token_ids.iter().all(|row| row.len() == seq_len),
        "token ID batch is ragged"
    );
    assert!(
        batch.attention_mask.iter().all(|row| row.len() == seq_len),
        "attention-mask batch is ragged"
    );
}

fn task_from_golden_label(label: &str) -> Option<EmbedTask> {
    match label {
        "doc" => Some(EmbedTask::RetrievalDocument),
        "code_query" => Some(EmbedTask::CodeRetrievalQuery),
        "query" => Some(EmbedTask::RetrievalQuery),
        "qa" => Some(EmbedTask::QuestionAnswering),
        "fact" => Some(EmbedTask::FactVerification),
        "classification" => Some(EmbedTask::Classification),
        "clustering" => Some(EmbedTask::Clustering),
        "similarity" => Some(EmbedTask::SentenceSimilarity),
        _ => None,
    }
}

fn golden_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("golden")
        .join(name)
}
