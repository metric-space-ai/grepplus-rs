//! Measure embedding-index wall-time: serial (batch=1) vs batched (batch=16).
//!
//! Usage:
//!   bench_embed_index <corpus_dir> --gguf <model.gguf> --tokenizer <tok.json>
//! Optional: --auto (GPU device via LoadOptions::auto), --batch <N>.
//!
//! Indexes the corpus ONCE into an in-memory store (nodes are identical for
//! both runs; embedding indexing does not mutate them), then times
//! `index_code_embeddings_for_project` at batch=1 and batch=N, reporting the
//! before/after seconds and speedup.

use std::path::PathBuf;
use std::time::Instant;

use greppy_embed_native::{EmbeddingGemma, LoadOptions};
use greppy_indexer::{
    index, index_code_embeddings_for_project, EmbeddingGemmaCodeProvider, EmbeddingIndexOptions,
};
use greppy_store::Store;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let corpus = PathBuf::from(args.remove(0));

    let mut gguf: Option<PathBuf> = None;
    let mut tokenizer: Option<PathBuf> = None;
    let mut auto = false;
    let mut batch = 16usize;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => {
                gguf = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--tokenizer" => {
                tokenizer = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--auto" => {
                auto = true;
                i += 1;
            }
            "--batch" => {
                batch = args[i + 1].parse()?;
                i += 2;
            }
            other => return Err(format!("unknown arg {other}").into()),
        }
    }

    let opts = if auto {
        LoadOptions::auto()
    } else {
        LoadOptions::default()
    };
    eprintln!("device: {:?}", opts.device);

    let load = Instant::now();
    let model = match (&gguf, &tokenizer) {
        (Some(g), Some(t)) => EmbeddingGemma::load_gguf(g, t, opts)?,
        _ => return Err("need --gguf <f> --tokenizer <f>".into()),
    };
    eprintln!("model loaded in {:.2}s", load.elapsed().as_secs_f64());

    // Build the graph once.
    let mut store = Store::open_memory()?;
    let t = Instant::now();
    let report = index(&mut store, &corpus, "bench")?;
    eprintln!(
        "graph indexed in {:.2}s (generation {})",
        t.elapsed().as_secs_f64(),
        report.graph_generation
    );

    // Time one embedding-index pass at a given batch size. Each pass writes
    // fresh vectors at a distinct generation so runs do not interfere.
    let run = |store: &mut Store, batch_size: usize, generation: u64| -> (f64, usize) {
        std::env::set_var("GREPPY_EMBED_BATCH", batch_size.to_string());
        let mut provider = EmbeddingGemmaCodeProvider::new("bench-model", &model);
        let start = Instant::now();
        let rep = index_code_embeddings_for_project(
            store,
            &corpus,
            "bench",
            &mut provider,
            EmbeddingIndexOptions::for_generation(generation),
        )
        .expect("embedding index");
        eprintln!(
            "  [batch={batch_size}] considered={} embedded={} skip_label={} skip_missing_file={} skip_span={} skip_oversize={}",
            rep.nodes_considered,
            rep.nodes_embedded,
            rep.nodes_skipped_non_definition,
            rep.nodes_skipped_missing_file,
            rep.nodes_skipped_invalid_span,
            rep.nodes_skipped_oversize,
        );
        (start.elapsed().as_secs_f64(), rep.nodes_embedded)
    };

    // Warm-up (JIT of Metal kernels / allocator / caches) so the timed runs
    // are steady-state.
    let (_, warm_n) = run(&mut store, batch, 100);
    eprintln!("warm-up embedded {warm_n} nodes");

    let (serial_s, n1) = run(&mut store, 1, 101);
    let (batched_s, nb) = run(&mut store, batch, 102);

    assert_eq!(n1, nb, "serial and batched embedded different node counts");
    println!("nodes_embedded = {n1}");
    println!("serial  (batch=1)  : {serial_s:.3}s");
    println!("batched (batch={batch}) : {batched_s:.3}s");
    println!("speedup            : {:.2}x", serial_s / batched_s);
    Ok(())
}
