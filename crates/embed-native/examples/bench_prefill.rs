use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use greppy_embed_native::{matmul::cpu_simd_backend, CpuEmbeddingModel};

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
use greppy_embed_native::CudaEmbeddingModel;
#[cfg(all(feature = "metal", target_os = "macos"))]
use greppy_embed_native::{MetalEmbeddingModel, MetalForwardProfile};

#[cfg(target_os = "macos")]
const DEFAULT_GGUF: &str = ""; // set via GREPPY_EMBEDDINGGEMMA_GGUF
#[cfg(target_os = "macos")]
const DEFAULT_LLAMA_BENCH: &str = ""; // set via LLAMA_BENCH_BIN

#[cfg(target_os = "linux")]
const DEFAULT_GGUF: &str = ""; // set via GREPPY_EMBEDDINGGEMMA_GGUF
#[cfg(target_os = "linux")]
const DEFAULT_LLAMA_BENCH: &str = ""; // set via LLAMA_BENCH_BIN
#[cfg(target_os = "windows")]
const DEFAULT_GGUF: &str = ""; // set via GREPPY_EMBEDDINGGEMMA_GGUF
#[cfg(target_os = "windows")]
const DEFAULT_LLAMA_BENCH: &str = ""; // set via LLAMA_BENCH_BIN

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if cpu_requested() {
        return run_cpu();
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    return run_cuda();

    #[cfg(all(feature = "metal", target_os = "macos"))]
    return run_metal();

    #[allow(unreachable_code)]
    run_cpu()
}

fn cpu_requested() -> bool {
    std::env::args().skip(1).any(|arg| arg == "--cpu")
        || std::env::var_os("EMBED_NATIVE_BENCH_CPU").is_some()
}

fn run_cpu() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = std::env::var("EMBED_NATIVE_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    let repeats = bench_repeats();
    let rounds = bench_rounds();
    let llama_bench = std::env::var("LLAMA_BENCH").unwrap_or_else(|_| DEFAULT_LLAMA_BENCH.into());

    println!("model: {gguf}");
    println!("native CPU: one fixed 512-token forward, warm-up + R={repeats}, rounds={rounds}");
    println!("native CPU SIMD backend: {}", cpu_simd_backend());

    let model = CpuEmbeddingModel::open(&gguf)?;
    let token_ids = vec![fixed_tokens_512()];
    let attention_mask = vec![vec![1u32; 512]];
    let _ = model.forward_tokens(&token_ids, &attention_mask)?;

    if std::env::var_os("EMBED_NATIVE_SKIP_LLAMA").is_some() {
        let (native_tps, secs) = bench_cpu_native(&model, &token_ids, &attention_mask, repeats)?;
        println!("native CPU pp512: {native_tps:.2} t/s ({secs:.6}s for {repeats} repeats)");
        return Ok(());
    }

    let mut native_values = Vec::with_capacity(rounds);
    let mut llama_values = Vec::with_capacity(rounds);
    for round in 0..rounds {
        println!();
        println!("cpu paired round {}/{}:", round + 1, rounds);
        let native_first = round % 2 == 0;
        if native_first {
            let (native_tps, secs) =
                bench_cpu_native(&model, &token_ids, &attention_mask, repeats)?;
            println!("native CPU pp512: {native_tps:.2} t/s ({secs:.6}s for {repeats} repeats)");
            native_values.push(native_tps);
            if let Some(tps) =
                run_llama_bench_once(&llama_bench, &gguf, repeats, LlamaBenchMode::Cpu)?
            {
                llama_values.push(tps);
                println!("round ratio native/llama: {:.3}x", native_tps / tps);
            }
        } else {
            let llama_tps =
                run_llama_bench_once(&llama_bench, &gguf, repeats, LlamaBenchMode::Cpu)?;
            let (native_tps, secs) =
                bench_cpu_native(&model, &token_ids, &attention_mask, repeats)?;
            println!("native CPU pp512: {native_tps:.2} t/s ({secs:.6}s for {repeats} repeats)");
            native_values.push(native_tps);
            if let Some(tps) = llama_tps {
                llama_values.push(tps);
                println!("round ratio native/llama: {:.3}x", native_tps / tps);
            }
        }
    }

    if let (Some(native), Some(llama)) = (median(&native_values), median(&llama_values)) {
        println!();
        println!(
            "cpu paired median: native {native:.2} t/s vs llama.cpp CPU {llama:.2} t/s, ratio {:.3}x",
            native / llama
        );
    }
    Ok(())
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
fn run_cuda() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = std::env::var("EMBED_NATIVE_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    let repeats = bench_repeats();
    let llama_bench = std::env::var("LLAMA_BENCH").unwrap_or_else(|_| DEFAULT_LLAMA_BENCH.into());

    println!("model: {gguf}");
    println!("native CUDA: one fixed 512-token forward, warm-up + R={repeats}");
    println!("native matmul path: ggml-cuda MMQ mul_mat_q + quantize_mmq_q8_1");

    let model = CudaEmbeddingModel::open(&gguf)?;
    let token_ids = vec![fixed_tokens_512()];
    let attention_mask = vec![vec![1u32; 512]];

    let _ = model.forward_tokens(&token_ids, &attention_mask)?;
    let (_, profile) = model.forward_tokens_profiled(&token_ids, &attention_mask)?;
    println!(
        "native profile one pp512: total={:.3} ms, cuda_mem_used_before={:.1} MiB, cuda_mem_used_after={:.1} MiB",
        profile.total_secs * 1000.0,
        profile.cuda_mem_used_before as f64 / 1024.0 / 1024.0,
        profile.cuda_mem_used_after as f64 / 1024.0 / 1024.0,
    );

    let t0 = Instant::now();
    for _ in 0..repeats {
        let _ = model.forward_tokens(&token_ids, &attention_mask)?;
    }
    let secs = t0.elapsed().as_secs_f64();
    let native_tps = 512.0 * repeats as f64 / secs;
    println!("native CUDA pp512: {native_tps:.2} t/s ({secs:.6}s for {repeats} repeats)");

    run_llama_bench(&llama_bench, &gguf, repeats, LlamaBenchMode::Cuda)?;
    Ok(())
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn run_metal() -> Result<(), Box<dyn std::error::Error>> {
    let gguf = std::env::var("EMBED_NATIVE_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.to_string());
    let repeats = bench_repeats();
    let llama_bench = std::env::var("LLAMA_BENCH").unwrap_or_else(|_| DEFAULT_LLAMA_BENCH.into());

    println!("model: {gguf}");
    println!("native Metal: one fixed 512-token forward, warm-up + R={repeats}");

    let model = MetalEmbeddingModel::open(&gguf)?;
    let token_ids = vec![fixed_tokens_512()];
    let attention_mask = vec![vec![1u32; 512]];

    let _ = model.forward_tokens(&token_ids, &attention_mask)?;
    if std::env::var_os("EMBED_NATIVE_SKIP_PROFILE").is_none() {
        let (_, profile) = model.forward_tokens_profiled(&token_ids, &attention_mask)?;
        print_metal_profile(&profile);
    }

    let secs = if std::env::var_os("EMBED_NATIVE_METAL_SERIAL_REPEATS").is_some() {
        let t0 = Instant::now();
        for _ in 0..repeats {
            let _ = model.forward_tokens(&token_ids, &attention_mask)?;
        }
        t0.elapsed().as_secs_f64()
    } else {
        let _ = model.forward_tokens_pipelined_repeated(
            &token_ids,
            &attention_mask,
            repeats.min(2).max(1),
        )?;
        let (_, secs) =
            model.forward_tokens_pipelined_repeated(&token_ids, &attention_mask, repeats)?;
        secs
    };
    let native_tps = 512.0 * repeats as f64 / secs;
    let mode = if std::env::var_os("EMBED_NATIVE_METAL_SERIAL_REPEATS").is_some() {
        "serial"
    } else {
        "pipelined"
    };
    println!("native Metal pp512 ({mode}): {native_tps:.2} t/s ({secs:.6}s for {repeats} repeats)");
    print_rss();

    if std::env::var_os("EMBED_NATIVE_SKIP_LLAMA").is_none() {
        run_llama_bench(&llama_bench, &gguf, repeats, LlamaBenchMode::GpuDefault)?;
    }
    Ok(())
}

fn bench_repeats() -> usize {
    std::env::var("EMBED_NATIVE_BENCH_REPEATS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(5)
        .max(5)
}

fn bench_rounds() -> usize {
    std::env::var("EMBED_NATIVE_BENCH_ROUNDS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3)
        .max(1)
}

fn bench_cpu_native(
    model: &CpuEmbeddingModel,
    token_ids: &[Vec<u32>],
    attention_mask: &[Vec<u32>],
    repeats: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let t0 = Instant::now();
    for _ in 0..repeats {
        let _ = model.forward_tokens(token_ids, attention_mask)?;
    }
    let secs = t0.elapsed().as_secs_f64();
    Ok((512.0 * repeats as f64 / secs, secs))
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum LlamaBenchMode {
    Cpu,
    GpuDefault,
    Cuda,
}

#[allow(dead_code)]
fn run_llama_bench(
    llama_bench: &str,
    gguf: &str,
    repeats: usize,
    mode: LlamaBenchMode,
) -> Result<(), Box<dyn std::error::Error>> {
    let _ = run_llama_bench_once(llama_bench, gguf, repeats, mode)?;
    Ok(())
}

fn run_llama_bench_once(
    llama_bench: &str,
    gguf: &str,
    repeats: usize,
    mode: LlamaBenchMode,
) -> Result<Option<f64>, Box<dyn std::error::Error>> {
    if Path::new(llama_bench).exists() {
        println!();
        println!("llama-bench reference:");
        let repeats_arg = repeats.to_string();
        let mut cmd = Command::new(llama_bench);
        cmd.args(["-m", gguf, "-p", "512", "-n", "0", "-r", &repeats_arg]);
        match mode {
            LlamaBenchMode::Cpu => {
                cmd.args(["-ngl", "0"]);
                cmd.env("CUDA_VISIBLE_DEVICES", "");
            }
            LlamaBenchMode::GpuDefault => {}
            LlamaBenchMode::Cuda => {
                cmd.args(["-ngl", "99"]);
            }
        }
        let output = cmd.output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        print!("{stdout}");
        eprint!("{stderr}");
        if !output.status.success() {
            eprintln!("llama-bench exited with {}", output.status);
        }
        Ok(parse_llama_tps(&stdout))
    } else {
        eprintln!(
            "llama-bench not found at {}",
            PathBuf::from(llama_bench).display()
        );
        Ok(None)
    }
}

fn parse_llama_tps(output: &str) -> Option<f64> {
    for line in output.lines() {
        if !line.contains("pp512") || !line.contains('|') {
            continue;
        }
        for cell in line.split('|').map(str::trim) {
            if let Some((value, _)) = cell.split_once('±') {
                if let Ok(tps) = value.trim().replace(',', "").parse::<f64>() {
                    return Some(tps);
                }
            }
        }
    }
    None
}

fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut values = values.to_vec();
    values.sort_by(|a, b| a.total_cmp(b));
    Some(values[values.len() / 2])
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn print_metal_profile(profile: &MetalForwardProfile) {
    println!(
        "native profile one pp512: total={:.3} ms, gpu_kernel={:.3} ms, gpu_elapsed={:.3} ms, cpu_non_gpu={:.3} ms",
        profile.total_secs * 1000.0,
        profile.metal_kernel_secs * 1000.0,
        profile.metal_gpu_secs * 1000.0,
        profile.cpu_non_gpu_secs() * 1000.0,
    );
    println!(
        "native profile host: prepare={:.3} ms, encode={:.3} ms, submit+wait={:.3} ms, sync_overhead={:.3} ms, read={:.3} ms",
        profile.cpu_prepare_secs * 1000.0,
        profile.cpu_encode_secs * 1000.0,
        profile.cpu_submit_wait_secs * 1000.0,
        profile.sync_overhead_secs() * 1000.0,
        profile.output_read_secs * 1000.0,
    );
    println!(
        "native profile counts: matmul_path={}, command_buffers={}, dispatches={}, buffer_allocs={}, buffer_alloc_bytes={:.1} MiB",
        profile.matmul_path,
        profile.command_buffers,
        profile.dispatches,
        profile.buffer_allocs,
        profile.buffer_alloc_bytes as f64 / 1024.0 / 1024.0,
    );
    if !profile.op_breakdown.is_empty() {
        let total_gpu: f64 = profile
            .op_breakdown
            .iter()
            .map(|op| op.metal_gpu_secs)
            .sum();
        println!("native Metal op breakdown:");
        for op in &profile.op_breakdown {
            let pct = if total_gpu > 0.0 {
                100.0 * op.metal_gpu_secs / total_gpu
            } else {
                0.0
            };
            println!(
                "  {:>18}: {:8.3} ms {:5.1}% dispatches={}",
                op.op_type,
                op.metal_gpu_secs * 1000.0,
                pct,
                op.dispatches,
            );
        }
    }
}

fn fixed_tokens_512() -> Vec<u32> {
    let mut ids = Vec::with_capacity(512);
    ids.push(2);
    ids.extend(std::iter::repeat_n(108u32, 510));
    ids.push(1);
    ids
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn print_rss() {
    let pid = std::process::id().to_string();
    if let Ok(output) = Command::new("ps").args(["-o", "rss=", "-p", &pid]).output() {
        if output.status.success() {
            let rss_kib = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<u64>()
                .unwrap_or(0);
            if rss_kib > 0 {
                println!(
                    "native peak/current RSS: {:.1} MiB",
                    rss_kib as f64 / 1024.0
                );
            }
        }
    }
}
