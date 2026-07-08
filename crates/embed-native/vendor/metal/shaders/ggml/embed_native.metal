// greppy-embed-native mean-pool + dispatch shader.
// Copyright (c) 2026 The greppy-rs authors. MIT License.
// Companion to the vendored ggml Metal kernels in this directory
// (ggml-metal.metal), Copyright (c) 2023-2026 The ggml authors, MIT License —
// see ../../../LICENSE-ggml.

#include <metal_stdlib>
using namespace metal;

struct embed_native_kargs_mean_pool {
    int32_t batch;
    int32_t seq_len;
    int32_t hidden;
};

struct embed_native_kargs_scale {
    int32_t n;
    float scale;
};

struct embed_native_kargs_rms_norm_f16 {
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    int32_t ne03;
    uint64_t src_nb1;
    uint64_t src_nb2;
    uint64_t src_nb3;
    uint64_t dst_nb1;
    uint64_t dst_nb2;
    uint64_t dst_nb3;
    uint64_t add_nb1;
    uint64_t add_nb2;
    uint64_t add_nb3;
    float eps;
};

struct embed_native_kargs_geglu_f16 {
    int32_t rows;
    int32_t dim;
};

struct embed_native_kargs_rms_norm_rope {
    int32_t batch;
    int32_t seq_len;
    int32_t heads;
    int32_t head_dim;
    int32_t row_width;
    float eps;
    float freq_base;
    int32_t pad;
};

struct embed_native_kargs_post_attn_ffn_norm {
    int32_t rows;
    int32_t dim;
    float eps;
    int32_t pad;
};

kernel void embed_native_mean_pool_f32(
        constant embed_native_kargs_mean_pool & args [[buffer(0)]],
        device const float * hidden [[buffer(1)]],
        device const uint  * mask   [[buffer(2)]],
        device       float * dst    [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint b = gid.y;
    if (d >= (uint) args.hidden || b >= (uint) args.batch) {
        return;
    }

    float sum = 0.0f;
    float count = 0.0f;
    const uint row_base = b * args.seq_len;
    for (int s = 0; s < args.seq_len; ++s) {
        const float m = mask[row_base + s] == 0 ? 0.0f : 1.0f;
        count += m;
        sum += hidden[(row_base + s) * args.hidden + d] * m;
    }
    count = max(count, 1.0e-12f);
    dst[b * args.hidden + d] = sum / count;
}

kernel void embed_native_scale_f32_to_f16(
        constant embed_native_kargs_scale & args [[buffer(0)]],
        device const float * src [[buffer(1)]],
        device       half  * dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.n) {
        return;
    }
    dst[gid] = half(src[gid] * args.scale);
}

kernel void embed_native_scale_f32(
        constant embed_native_kargs_scale & args [[buffer(0)]],
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.n) {
        return;
    }
    dst[gid] = src[gid] * args.scale;
}

kernel void embed_native_mean_pool_f16_to_f32(
        constant embed_native_kargs_mean_pool & args [[buffer(0)]],
        device const half * hidden [[buffer(1)]],
        device const uint * mask   [[buffer(2)]],
        device      float * dst    [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    const uint b = gid.y;
    if (d >= (uint) args.hidden || b >= (uint) args.batch) {
        return;
    }

    float sum = 0.0f;
    float count = 0.0f;
    const uint row_base = b * args.seq_len;
    for (int s = 0; s < args.seq_len; ++s) {
        const float m = mask[row_base + s] == 0 ? 0.0f : 1.0f;
        count += m;
        sum += float(hidden[(row_base + s) * args.hidden + d]) * m;
    }
    count = max(count, 1.0e-12f);
    dst[b * args.hidden + d] = sum / count;
}

kernel void embed_native_rms_norm_mul_f16(
        constant embed_native_kargs_rms_norm_f16 & args [[buffer(0)]],
        device const char  * src    [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device       char  * dst    [[buffer(3)]],
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint tiisg [[thread_index_in_simdgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    device const half * x = (device const half *) (src + i03*args.src_nb3 + i02*args.src_nb2 + i01*args.src_nb1);

    float sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        const float v = float(x[i00]);
        sumf += v * v;
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = rsqrt(sumf / args.ne00 + args.eps);
    device half * y = (device half *) (dst + i03*args.dst_nb3 + i02*args.dst_nb2 + i01*args.dst_nb1);
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        y[i00] = half(float(x[i00]) * scale * weight[i00]);
    }
}

kernel void embed_native_rms_norm_mul_add_f16(
        constant embed_native_kargs_rms_norm_f16 & args [[buffer(0)]],
        device const char  * src    [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device const char  * add    [[buffer(3)]],
        device       char  * dst    [[buffer(4)]],
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint sgitg [[simdgroup_index_in_threadgroup]],
        uint tiisg [[thread_index_in_simdgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    device const half * x = (device const half *) (src + i03*args.src_nb3 + i02*args.src_nb2 + i01*args.src_nb1);

    float sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        const float v = float(x[i00]);
        sumf += v * v;
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float scale = rsqrt(sumf / args.ne00 + args.eps);
    device const half * a = (device const half *) (add + i03*args.add_nb3 + i02*args.add_nb2 + i01*args.add_nb1);
    device half * y = (device half *) (dst + i03*args.dst_nb3 + i02*args.dst_nb2 + i01*args.dst_nb1);
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        y[i00] = half(float(x[i00]) * scale * weight[i00] + float(a[i00]));
    }
}

kernel void embed_native_geglu_f16(
        constant embed_native_kargs_geglu_f16 & args [[buffer(0)]],
        device const half * gate [[buffer(1)]],
        device const half * up   [[buffer(2)]],
        device       half * dst  [[buffer(3)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= (uint) args.rows) {
        return;
    }
    const uint base = row * args.dim;
    constexpr float sqrt_2_over_pi = 0.7978845608028654f;
    constexpr float gelu_coef_a = 0.044715f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float x0 = float(gate[base + i]);
        const float x1 = float(up[base + i]);
        const float gelu = 0.5f * x0 * (1.0f + precise::tanh(sqrt_2_over_pi * x0 * (1.0f + gelu_coef_a * x0 * x0)));
        dst[base + i] = half(gelu * x1);
    }
}

kernel void embed_native_rms_norm_rope_neox_f32(
        constant embed_native_kargs_rms_norm_rope & args [[buffer(0)]],
        device const float * src    [[buffer(1)]],
        device const float * weight [[buffer(2)]],
        device       float * dst    [[buffer(3)]],
        threadgroup float * shmem   [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg   [[threads_per_threadgroup]]) {
    const uint pos = tgpig.x;
    const uint head = tgpig.y;
    const uint batch = tgpig.z;
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    if (batch >= (uint) args.batch || pos >= (uint) args.seq_len || head >= (uint) args.heads) {
        return;
    }

    const uint head_dim = (uint) args.head_dim;
    const uint half_dim = head_dim / 2;
    const uint64_t src_base = ((uint64_t) batch * args.seq_len + pos) * args.row_width + head * head_dim;
    const uint64_t dst_base = src_base;

    float sumf = 0.0f;
    for (uint i = tid; i < head_dim; i += nthreads) {
        const float v = src[src_base + i];
        sumf += v * v;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = nthreads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(shmem[0] / float(head_dim) + args.eps);
    for (uint i = tid; i < half_dim; i += nthreads) {
        const float x0 = src[src_base + i] * inv * weight[i];
        const float x1 = src[src_base + half_dim + i] * inv * weight[half_dim + i];
        const float theta = float(pos) * pow(args.freq_base, -2.0f * float(i) / float(head_dim));
        const float c = cos(theta);
        const float s = sin(theta);
        dst[dst_base + i] = x0 * c - x1 * s;
        dst[dst_base + half_dim + i] = x0 * s + x1 * c;
    }
}

kernel void embed_native_post_attn_ffn_norm_f32(
        constant embed_native_kargs_post_attn_ffn_norm & args [[buffer(0)]],
        device const float * attn_proj        [[buffer(1)]],
        device const float * residual         [[buffer(2)]],
        device const float * post_attn_weight [[buffer(3)]],
        device const float * ffn_weight       [[buffer(4)]],
        device       float * sa_out           [[buffer(5)]],
        device       float * ffn_norm         [[buffer(6)]],
        threadgroup float * shmem             [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= (uint) args.rows) {
        return;
    }
    const uint64_t base = (uint64_t) row * args.dim;

    float sumf = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float v = attn_proj[base + i];
        sumf += v * v;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float post_scale = rsqrt(shmem[0] / float(args.dim) + args.eps);

    sumf = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float y = attn_proj[base + i] * post_scale * post_attn_weight[i] + residual[base + i];
        sa_out[base + i] = y;
        sumf += y * y;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float ffn_scale = rsqrt(shmem[0] / float(args.dim) + args.eps);
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        ffn_norm[base + i] = sa_out[base + i] * ffn_scale * ffn_weight[i];
    }
}

kernel void embed_native_post_ffn_next_attn_norm_f32(
        constant embed_native_kargs_post_attn_ffn_norm & args [[buffer(0)]],
        device const float * ffn_down         [[buffer(1)]],
        device const float * residual         [[buffer(2)]],
        device const float * post_ffn_weight  [[buffer(3)]],
        device const float * next_attn_weight [[buffer(4)]],
        device       float * out_state        [[buffer(5)]],
        device       float * next_attn_norm   [[buffer(6)]],
        threadgroup float * shmem             [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if (row >= (uint) args.rows) {
        return;
    }
    const uint64_t base = (uint64_t) row * args.dim;

    float sumf = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float v = ffn_down[base + i];
        sumf += v * v;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float post_scale = rsqrt(shmem[0] / float(args.dim) + args.eps);

    sumf = 0.0f;
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        const float y = ffn_down[base + i] * post_scale * post_ffn_weight[i] + residual[base + i];
        out_state[base + i] = y;
        sumf += y * y;
    }
    shmem[tid] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shmem[tid] += shmem[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float next_scale = rsqrt(shmem[0] / float(args.dim) + args.eps);
    for (uint i = tid; i < (uint) args.dim; i += ntg) {
        next_attn_norm[base + i] = out_state[base + i] * next_scale * next_attn_weight[i];
    }
}
