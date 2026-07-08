use std::path::Path;

use greppy_embed_native::matmul::QuantMatrix;
use greppy_embed_native::metal::ffi::global_device;
use greppy_embed_native::metal::ops::{self, GgmlType, GgmlUnaryOp, UnaryParams};
use greppy_embed_native::metal::weights::MetalWeights;
use greppy_embed_native::GgufModel;
use half::f16;

fn main() {
    let Some(gguf_path) = std::env::var_os("EMBED_NATIVE_GGUF") else {
        eprintln!("skipping Metal kernel parity: EMBED_NATIVE_GGUF is unset");
        return;
    };
    let gguf = GgufModel::open(&gguf_path)
        .unwrap_or_else(|e| panic!("open GGUF {}: {e}", Path::new(&gguf_path).display()));
    let dev = global_device().expect("Metal device");
    let weights = MetalWeights::load(dev, &gguf).expect("load Metal weights");

    verify_get_rows(&gguf, dev, &weights);
    verify_rms_norm(dev, &weights);
    verify_rope(dev);
    verify_flash_attn(dev);
    verify_q4_matmul(&gguf, dev, &weights);
    verify_f32_matmul(&gguf, dev, &weights);
    verify_l2_norm(dev);
}

fn verify_get_rows(
    gguf: &GgufModel,
    dev: &greppy_embed_native::metal::ffi::Device,
    weights: &MetalWeights,
) {
    let ids = vec![2u32, 108, 1, 0];
    let cpu = QuantMatrix::from_model(gguf, "token_embd.weight")
        .expect("cpu token matrix")
        .embedding_rows(&ids)
        .expect("cpu embedding");
    let ids_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&ids))
        .expect("ids buf");
    let out = dev
        .new_buffer(cpu.len() * std::mem::size_of::<f32>())
        .expect("out");
    let scaled = dev
        .new_buffer(cpu.len() * std::mem::size_of::<f32>())
        .expect("scaled");
    let tok = weights.require("token_embd.weight").expect("token tensor");
    let cb = dev.new_command_buffer().expect("cmd");
    let enc = cb.compute().expect("enc");
    assert!(ops::op_get_rows(
        &enc,
        dev,
        GgmlType::Q6_K,
        &tok.buffer,
        tok.offset,
        &ids_buf,
        &out,
        768,
        tok.nb[1],
        tok.nb[2],
        tok.nb[3],
        ids.len() as i32,
        1,
        1,
        4,
        (ids.len() * 4) as u64,
        (ids.len() * 4) as u64,
        768 * 4,
        (ids.len() * 768 * 4) as u64,
        (ids.len() * 768 * 4) as u64,
    ));
    assert!(ops::op_unary(
        &enc,
        dev,
        GgmlUnaryOp::Scale,
        GgmlType::F32,
        GgmlType::F32,
        &out,
        &scaled,
        768,
        ids.len() as i32,
        1,
        1,
        4,
        768 * 4,
        (ids.len() * 768 * 4) as u64,
        (ids.len() * 768 * 4) as u64,
        768,
        ids.len() as i32,
        1,
        1,
        4,
        768 * 4,
        (ids.len() * 768 * 4) as u64,
        (ids.len() * 768 * 4) as u64,
        UnaryParams {
            scale: (768.0f32).sqrt(),
            ..UnaryParams::default()
        },
    ));
    enc.end();
    cb.commit_and_wait().expect("commit");
    let mut gpu = vec![0.0f32; cpu.len()];
    let mut gpu_scaled = vec![0.0f32; cpu.len()];
    unsafe {
        out.read(0, &mut gpu);
        scaled.read(0, &mut gpu_scaled);
    }
    let raw_max_abs = max_abs(&gpu, &cpu);
    eprintln!("metal get_rows_q6_K raw max_abs={raw_max_abs:.9}");
    assert!(raw_max_abs < 1.0e-5, "get_rows raw max_abs {raw_max_abs}");
    let scaled_cpu = cpu
        .iter()
        .map(|v| *v * (768.0f32).sqrt())
        .collect::<Vec<_>>();
    let max_abs = max_abs(&gpu_scaled, &scaled_cpu);
    eprintln!("metal get_rows_q6_K+scale max_abs={max_abs:.9}");
    assert!(max_abs < 1.0e-3, "get_rows max_abs {max_abs}");
}

fn verify_q4_matmul(
    gguf: &GgufModel,
    dev: &greppy_embed_native::metal::ffi::Device,
    weights: &MetalWeights,
) {
    let rows = 9usize;
    let cols = 768usize;
    let lhs = (0..rows * cols)
        .map(|i| ((i % 127) as f32 - 63.0) / 64.0)
        .collect::<Vec<_>>();
    let cpu = QuantMatrix::from_model(gguf, "blk.0.attn_q.weight")
        .expect("cpu q")
        .matmul(&lhs, rows)
        .expect("cpu matmul");
    let lhs_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&lhs))
        .expect("lhs");
    let out = dev
        .new_buffer(cpu.len() * std::mem::size_of::<f32>())
        .expect("out");
    let w = weights.require("blk.0.attn_q.weight").expect("q tensor");
    let cb = dev.new_command_buffer().expect("cmd");
    let enc = cb.compute().expect("enc");
    assert!(ops::op_mul_mm(
        &enc,
        dev,
        GgmlType::Q4_K,
        GgmlType::F32,
        &w.buffer,
        w.offset,
        &lhs_buf,
        &out,
        cols as i32,
        768,
        1,
        1,
        w.nb[1],
        w.nb[2],
        w.nb[3],
        rows as i32,
        1,
        1,
        4,
        (cols * 4) as u64,
        (rows * cols * 4) as u64,
        (rows * cols * 4) as u64,
        768,
        rows as i32,
    ));
    enc.end();
    cb.commit_and_wait().expect("commit");
    let mut gpu = vec![0.0f32; cpu.len()];
    unsafe {
        out.read(0, &mut gpu);
    }
    let max_abs = max_abs(&gpu, &cpu);
    eprintln!("metal mul_mm_q4_K max_abs={max_abs:.9}");
    assert!(max_abs < 0.2, "mul_mm max_abs {max_abs}");
}

fn verify_f32_matmul(
    gguf: &GgufModel,
    dev: &greppy_embed_native::metal::ffi::Device,
    weights: &MetalWeights,
) {
    let rows = 6usize;
    let cols = 768usize;
    let lhs = (0..rows * cols)
        .map(|i| ((i % 223) as f32 - 111.0) / 91.0)
        .collect::<Vec<_>>();
    let cpu = QuantMatrix::from_model(gguf, "dense_2.weight")
        .expect("cpu dense_2")
        .matmul(&lhs, rows)
        .expect("cpu f32 matmul");
    let lhs_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&lhs))
        .expect("lhs f32");
    let out = dev
        .new_buffer(cpu.len() * std::mem::size_of::<f32>())
        .expect("out f32");
    let w = weights.require("dense_2.weight").expect("dense_2 tensor");
    let cb = dev.new_command_buffer().expect("cmd");
    let enc = cb.compute().expect("enc");
    assert!(ops::op_mul_mm(
        &enc,
        dev,
        GgmlType::F32,
        GgmlType::F32,
        &w.buffer,
        w.offset,
        &lhs_buf,
        &out,
        cols as i32,
        3072,
        1,
        1,
        w.nb[1],
        w.nb[2],
        w.nb[3],
        rows as i32,
        1,
        1,
        4,
        (cols * 4) as u64,
        (rows * cols * 4) as u64,
        (rows * cols * 4) as u64,
        3072,
        rows as i32,
    ));
    enc.end();
    cb.commit_and_wait().expect("commit");
    let mut gpu = vec![0.0f32; cpu.len()];
    unsafe {
        out.read(0, &mut gpu);
    }
    let max_abs = max_abs(&gpu, &cpu);
    eprintln!("metal mul_mm_f32 max_abs={max_abs:.9}");
    assert!(max_abs < 2.0e-3, "mul_mm_f32 max_abs {max_abs}");
}

fn verify_flash_attn(dev: &greppy_embed_native::metal::ffi::Device) {
    let batch = 2usize;
    let seq = 32usize;
    let heads = 3usize;
    let kv_heads = 1usize;
    let head_dim = 256usize;
    let hidden = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let q = (0..batch * seq * hidden)
        .map(|i| ((i % 127) as f32 - 63.0) / 128.0)
        .collect::<Vec<_>>();
    let k = (0..batch * seq * kv_width)
        .map(|i| ((i % 113) as f32 - 56.0) / 117.0)
        .collect::<Vec<_>>();
    let v = (0..batch * seq * kv_width)
        .map(|i| ((i % 109) as f32 - 54.0) / 101.0)
        .collect::<Vec<_>>();
    let mut mask_f32 = vec![0.0f32; batch * seq * seq];
    for b in 0..batch {
        for qs in 0..seq {
            for ks in 0..seq {
                if (b == 0 && ks + 5 >= seq) || (b == 1 && ks < 3) || (qs + 11 < ks) {
                    mask_f32[(b * seq + qs) * seq + ks] = -1.0e9;
                }
            }
        }
    }
    let mask_f16 = mask_f32
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect::<Vec<_>>();
    let cpu = flash_cpu(
        &q,
        &k,
        &v,
        Some(&mask_f32),
        batch,
        seq,
        heads,
        kv_heads,
        head_dim,
        256.0f32.powf(-0.5),
    );
    let q_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&q))
        .expect("q");
    let k_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&k))
        .expect("k");
    let v_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&v))
        .expect("v");
    let mask_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&mask_f16))
        .expect("mask");
    let out = dev
        .new_buffer(cpu.len() * std::mem::size_of::<f32>())
        .expect("out");
    let pad = dev.new_buffer(512 * 1024).expect("pad");
    let blk = dev.new_buffer(64 * 1024).expect("blk");
    let tmp = dev
        .new_buffer(seq * heads * batch * 32 * (head_dim + 2) * std::mem::size_of::<f32>())
        .expect("tmp");
    let cb = dev.new_command_buffer().expect("cmd");
    let enc = cb.compute().expect("enc");
    assert!(ops::op_flash_attn_ext(
        &enc,
        dev,
        GgmlType::F32,
        &q_buf,
        &k_buf,
        &v_buf,
        Some(&mask_buf),
        None,
        &pad,
        &blk,
        &tmp,
        &out,
        head_dim as i32,
        seq as i32,
        heads as i32,
        batch as i32,
        (hidden * 4) as u64,
        (head_dim * 4) as u64,
        (seq * hidden * 4) as u64,
        seq as i32,
        kv_heads as i32,
        batch as i32,
        4,
        (kv_width * 4) as u64,
        (head_dim * 4) as u64,
        (seq * kv_width * 4) as u64,
        head_dim as i32,
        4,
        (kv_width * 4) as u64,
        (head_dim * 4) as u64,
        (seq * kv_width * 4) as u64,
        seq as i32,
        seq as i32,
        1,
        batch as i32,
        (seq * 2) as u64,
        (seq * seq * 2) as u64,
        (seq * seq * 2) as u64,
        heads as i32,
        seq as i32,
        batch as i32,
        256.0f32.powf(-0.5),
        0.0,
        0.0,
    ));
    enc.end();
    cb.commit_and_wait().expect("commit");
    let mut gpu = vec![0.0f32; cpu.len()];
    unsafe {
        out.read(0, &mut gpu);
    }
    let max_abs = max_abs(&gpu, &cpu);
    eprintln!("metal flash_attn_ext_f32 max_abs={max_abs:.9}");
    assert!(max_abs < 2.0e-3, "flash_attn max_abs {max_abs}");
}

fn flash_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    batch: usize,
    seq: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let hidden = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let groups = heads / kv_heads;
    let mut out = vec![0.0f32; batch * seq * hidden];
    for b in 0..batch {
        for h in 0..heads {
            let kh = h / groups;
            for qs in 0..seq {
                let q_base = (b * seq + qs) * hidden + h * head_dim;
                let mut scores = vec![0.0f32; seq];
                for ks in 0..seq {
                    let k_base = (b * seq + ks) * kv_width + kh * head_dim;
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[k_base + d];
                    }
                    let mask_value = mask.map(|m| m[(b * seq + qs) * seq + ks]).unwrap_or(0.0);
                    scores[ks] = dot * scale + mask_value;
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denom = 0.0;
                for s in &mut scores {
                    *s = (*s - max).exp();
                    denom += *s;
                }
                for ks in 0..seq {
                    let p = scores[ks] / denom;
                    let v_base = (b * seq + ks) * kv_width + kh * head_dim;
                    let out_base = (b * seq + qs) * hidden + h * head_dim;
                    for d in 0..head_dim {
                        out[out_base + d] += p * v[v_base + d];
                    }
                }
            }
        }
    }
    out
}

fn verify_rope(dev: &greppy_embed_native::metal::ffi::Device) {
    let batch = 2usize;
    let seq = 7usize;
    let heads = 3usize;
    let head_dim = 256usize;
    let row_width = heads * head_dim;
    let src = (0..batch * seq * row_width)
        .map(|i| ((i % 257) as f32 - 128.0) / 37.0)
        .collect::<Vec<_>>();
    let pos = (0..seq as i32).collect::<Vec<_>>();
    let cpu = rope_cpu(&src, batch, seq, heads, head_dim, row_width, 10_000.0);
    let src_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&src))
        .expect("rope src");
    let pos_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&pos))
        .expect("rope pos");
    let out = dev
        .new_buffer(cpu.len() * std::mem::size_of::<f32>())
        .expect("rope out");
    let cb = dev.new_command_buffer().expect("cmd");
    let enc = cb.compute().expect("enc");
    assert!(ops::op_rope(
        &enc,
        dev,
        &ops::pipeline_name_rope_neox(GgmlType::F32),
        &src_buf,
        &pos_buf,
        None,
        &out,
        head_dim as i32,
        heads as i32,
        seq as i32,
        batch as i32,
        4,
        (head_dim * 4) as u64,
        (row_width * 4) as u64,
        (seq * row_width * 4) as u64,
        head_dim as i32,
        heads as i32,
        seq as i32,
        batch as i32,
        4,
        (head_dim * 4) as u64,
        (row_width * 4) as u64,
        (seq * row_width * 4) as u64,
        0,
        head_dim as i32,
        2048,
        10_000.0,
        1.0,
        0.0,
        1.0,
        32.0,
        1.0,
        0,
        0,
        0,
        0,
    ));
    enc.end();
    cb.commit_and_wait().expect("commit");
    let mut gpu = vec![0.0f32; cpu.len()];
    unsafe {
        out.read(0, &mut gpu);
    }
    let max_abs = max_abs(&gpu, &cpu);
    eprintln!("metal rope_neox max_abs={max_abs:.9}");
    assert!(max_abs < 1.0e-4, "rope max_abs {max_abs}");
}

fn rope_cpu(
    src: &[f32],
    batch: usize,
    seq: usize,
    heads: usize,
    head_dim: usize,
    row_width: usize,
    base: f64,
) -> Vec<f32> {
    let mut out = vec![0.0; src.len()];
    let half = head_dim / 2;
    for b in 0..batch {
        for s in 0..seq {
            for h in 0..heads {
                let row = (b * seq + s) * row_width + h * head_dim;
                for i in 0..half {
                    let theta = s as f32 / base.powf((2 * i) as f64 / head_dim as f64) as f32;
                    let (sin, cos) = theta.sin_cos();
                    let x0 = src[row + i];
                    let x1 = src[row + half + i];
                    out[row + i] = x0 * cos - x1 * sin;
                    out[row + half + i] = x0 * sin + x1 * cos;
                }
            }
        }
    }
    out
}

fn verify_rms_norm(dev: &greppy_embed_native::metal::ffi::Device, weights: &MetalWeights) {
    let rows = 5usize;
    let dim = 768usize;
    let src = (0..rows * dim)
        .map(|i| ((i % 251) as f32 - 125.0) / 32.0)
        .collect::<Vec<_>>();
    let w = weights.require("blk.0.attn_norm.weight").expect("norm");
    let mut weight = vec![0.0f32; dim];
    unsafe {
        w.buffer.read(w.offset, &mut weight);
    }
    let cpu = rms_norm_cpu(&src, &weight, rows, dim, 1.0e-6);
    let src_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&src))
        .expect("src");
    let out = dev
        .new_buffer(cpu.len() * std::mem::size_of::<f32>())
        .expect("out");
    let cb = dev.new_command_buffer().expect("cmd");
    let enc = cb.compute().expect("enc");
    assert!(ops::op_rms_norm_mul(
        &enc,
        dev,
        GgmlType::F32,
        &src_buf,
        &w.buffer,
        w.offset,
        &out,
        1.0e-6,
        dim as i32,
        rows as i32,
        1,
        1,
        (dim * 4) as u64,
        (rows * dim * 4) as u64,
        (rows * dim * 4) as u64,
        (dim * 4) as u64,
        (rows * dim * 4) as u64,
        (rows * dim * 4) as u64,
        1,
        1,
        1,
        (dim * 4) as u64,
        (dim * 4) as u64,
        (dim * 4) as u64,
    ));
    enc.end();
    cb.commit_and_wait().expect("commit");
    let mut gpu = vec![0.0f32; cpu.len()];
    unsafe {
        out.read(0, &mut gpu);
    }
    let max_abs = max_abs(&gpu, &cpu);
    eprintln!("metal rms_norm_mul max_abs={max_abs:.9}");
    assert!(max_abs < 1.0e-5, "rms_norm max_abs {max_abs}");
}

fn rms_norm_cpu(src: &[f32], weight: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; rows * dim];
    for r in 0..rows {
        let row = &src[r * dim..(r + 1) * dim];
        let denom = (row.iter().map(|v| v * v).sum::<f32>() / dim as f32 + eps).sqrt();
        for d in 0..dim {
            out[r * dim + d] = row[d] / denom * weight[d];
        }
    }
    out
}

fn verify_l2_norm(dev: &greppy_embed_native::metal::ffi::Device) {
    let rows = 6usize;
    let dim = 768usize;
    let src = (0..rows * dim)
        .map(|i| ((i % 191) as f32 - 95.0) / 17.0)
        .collect::<Vec<_>>();
    let cpu = l2_norm_cpu(&src, rows, dim, 1.0e-12);
    let src_buf = dev
        .new_buffer_from_slice(bytemuck::cast_slice(&src))
        .expect("l2 src");
    let out = dev
        .new_buffer(cpu.len() * std::mem::size_of::<f32>())
        .expect("l2 out");
    let cb = dev.new_command_buffer().expect("cmd");
    let enc = cb.compute().expect("enc");
    assert!(ops::op_l2_norm(
        &enc,
        dev,
        GgmlType::F32,
        &src_buf,
        &out,
        1.0e-12,
        dim as i32,
        rows as i32,
        1,
        1,
        4,
        (dim * 4) as u64,
        (rows * dim * 4) as u64,
        (rows * dim * 4) as u64,
        dim as i32,
        rows as i32,
        1,
        1,
        4,
        (dim * 4) as u64,
        (rows * dim * 4) as u64,
        (rows * dim * 4) as u64,
    ));
    enc.end();
    cb.commit_and_wait().expect("commit");
    let mut gpu = vec![0.0f32; cpu.len()];
    unsafe {
        out.read(0, &mut gpu);
    }
    let max_abs = max_abs(&gpu, &cpu);
    eprintln!("metal l2_norm max_abs={max_abs:.9}");
    assert!(max_abs < 1.0e-6, "l2_norm max_abs {max_abs}");
}

fn l2_norm_cpu(src: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; rows * dim];
    for r in 0..rows {
        let row = &src[r * dim..(r + 1) * dim];
        let denom = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(eps);
        for d in 0..dim {
            out[r * dim + d] = row[d] / denom;
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0, f32::max)
}
