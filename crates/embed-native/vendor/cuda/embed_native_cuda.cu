// Source-built CUDA backend for greppy-embed-native.
//
// Copyright (c) 2026 The greppy-rs authors. MIT License.
// Derivative work of ggml (https://github.com/ggml-org/ggml), Copyright (c)
// 2023-2026 The ggml authors, MIT License — see ../LICENSE-ggml. This file
// includes and dispatches the vendored ggml CUDA kernels in ggml-cuda/.
//
// The quantized matmul path intentionally routes through ggml CUDA MMQ:
// f32 activations are packed on-device with quantize_mmq_q8_1, then the
// vendored mul_mat_q<type, mmq_x> kernel consumes original GGUF Q blocks.

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "mmq.cuh"
#include "quantize.cuh"

#if defined(_WIN32)
#define GP_CUDA_EXPORT extern "C" __declspec(dllexport)
#else
#define GP_CUDA_EXPORT extern "C" __attribute__((visibility("default")))
#endif

#define GP_CUDA_DRIVER_ERROR_BASE 20000
#define GP_CUDA_UNSUPPORTED_ARCH 30001

extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "ggml_abort at %s:%d: ", file ? file : "<unknown>", line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

GP_CUDA_EXPORT const char * gp_cuda_error_string(int code) {
    static char driver_error_buf[256];
    if (code >= GP_CUDA_DRIVER_ERROR_BASE && code < GP_CUDA_DRIVER_ERROR_BASE + 10000) {
        const char * name = nullptr;
        const char * desc = nullptr;
        CUresult cu = (CUresult)(code - GP_CUDA_DRIVER_ERROR_BASE);
        cuGetErrorName(cu, &name);
        cuGetErrorString(cu, &desc);
        snprintf(driver_error_buf, sizeof(driver_error_buf), "%s%s%s",
                 name ? name : "CUDA driver error",
                 desc ? ": " : "",
                 desc ? desc : "");
        return driver_error_buf;
    }
    if (code == GP_CUDA_UNSUPPORTED_ARCH) {
        return "CUDA GPU compute capability is below sm_75; greppy native CUDA MMQ backend is unavailable";
    }
    if (code <= 999) {
        return cudaGetErrorString((cudaError_t) code);
    }
    switch (code) {
        case 10000 + CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case 10000 + CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case 10000 + CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case 10000 + CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case 10000 + CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case 10000 + CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case 10000 + CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case 10000 + CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        default: return "unknown CUDA/cuBLAS error";
    }
}

GP_CUDA_EXPORT int gp_cuda_init(int device, void ** stream_out, void ** blas_out) {
    CUresult cu = cuInit(0);
    if (cu != CUDA_SUCCESS) return GP_CUDA_DRIVER_ERROR_BASE + (int) cu;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) return (int) err;

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return (int) err;
    if (prop.major*10 + prop.minor < 75) {
        return GP_CUDA_UNSUPPORTED_ARCH;
    }

    cudaStream_t stream = nullptr;
    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) return (int) err;

    cublasHandle_t blas = nullptr;
    cublasStatus_t st = cublasCreate(&blas);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(stream);
        return 10000 + (int) st;
    }
    st = cublasSetStream(blas, stream);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(blas);
        cudaStreamDestroy(stream);
        return 10000 + (int) st;
    }
    cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH);
    *stream_out = (void *) stream;
    *blas_out = (void *) blas;
    return 0;
}

GP_CUDA_EXPORT int gp_cuda_destroy(void * stream, void * blas) {
    if (blas) {
        cublasDestroy((cublasHandle_t) blas);
    }
    if (stream) {
        cudaStreamDestroy((cudaStream_t) stream);
    }
    return 0;
}

GP_CUDA_EXPORT int gp_cuda_malloc(void ** ptr, size_t bytes) {
    return (int) cudaMalloc(ptr, bytes);
}

GP_CUDA_EXPORT int gp_cuda_free(void * ptr) {
    return (int) cudaFree(ptr);
}

GP_CUDA_EXPORT int gp_cuda_memcpy_h2d_async(void * dst, const void * src, size_t bytes, void * stream) {
    return (int) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t) stream);
}

GP_CUDA_EXPORT int gp_cuda_memcpy_d2h_async(void * dst, const void * src, size_t bytes, void * stream) {
    return (int) cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t) stream);
}

GP_CUDA_EXPORT int gp_cuda_memset_async(void * dst, int value, size_t bytes, void * stream) {
    return (int) cudaMemsetAsync(dst, value, bytes, (cudaStream_t) stream);
}

GP_CUDA_EXPORT int gp_cuda_stream_sync(void * stream) {
    return (int) cudaStreamSynchronize((cudaStream_t) stream);
}

GP_CUDA_EXPORT int gp_cuda_mem_get_info(size_t * free_out, size_t * total_out) {
    return (int) cudaMemGetInfo(free_out, total_out);
}

static __device__ __forceinline__ float gp_half_to_float(const half h) {
    return __half2float(h);
}

struct gp_block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};

__global__ void gp_embed_q6k_kernel(
        const gp_block_q6_K * __restrict__ weights,
        const uint32_t * __restrict__ ids,
        float * __restrict__ dst,
        int rows,
        int blocks_per_row,
        float scale) {
    const int row = blockIdx.x;
    const int block_col = blockIdx.y;
    const int t = threadIdx.x;
    if (row >= rows || t >= 256) return;

    const uint32_t token = ids[row];
    const gp_block_q6_K & b = weights[(int64_t) token*blocks_per_row + block_col];
    const int n = t < 128 ? 0 : 128;
    const int idx = n / 128;
    const uint8_t * ql = b.ql + 64*idx;
    const uint8_t * qh = b.qh + 32*idx;
    const int8_t * sc = b.scales + 8*idx;
    const int l = t - n;
    const int lane = l & 31;
    const int is = lane / 16;
    int q;
    int sidx;
    if (l < 32) {
        q = ((ql[lane] & 0x0f) | ((qh[lane] & 3) << 4)) - 32;
        sidx = is;
    } else if (l < 64) {
        q = ((ql[lane + 32] & 0x0f) | (((qh[lane] >> 2) & 3) << 4)) - 32;
        sidx = is + 2;
    } else if (l < 96) {
        q = ((ql[lane] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32;
        sidx = is + 4;
    } else {
        q = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32;
        sidx = is + 6;
    }
    const float d = gp_half_to_float(b.d);
    dst[(int64_t) row*blocks_per_row*256 + block_col*256 + t] =
        scale * d * (float) sc[sidx] * (float) q;
}

GP_CUDA_EXPORT int gp_embed_q6k(
        const void * weights,
        const uint32_t * ids,
        float * dst,
        int rows,
        int hidden,
        float scale,
        void * stream) {
    const int blocks_per_row = hidden / 256;
    gp_embed_q6k_kernel<<<dim3(rows, blocks_per_row, 1), 256, 0, (cudaStream_t) stream>>>(
        (const gp_block_q6_K *) weights, ids, dst, rows, blocks_per_row, scale);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_rms_norm_kernel(
        const float * __restrict__ src,
        const float * __restrict__ weight,
        const float * __restrict__ add,
        float * __restrict__ dst,
        int rows,
        int dim,
        float eps) {
    extern __shared__ float s[];
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = src[(int64_t) row*dim + i];
        sum += v*v;
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(s[0] / (float) dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const int64_t idx = (int64_t) row*dim + i;
        float v = src[idx] * inv * weight[i];
        if (add) v += add[idx];
        dst[idx] = v;
    }
}

GP_CUDA_EXPORT int gp_rms_norm(
        const float * src, const float * weight, float * dst,
        int rows, int dim, float eps, void * stream) {
    gp_rms_norm_kernel<<<rows, 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        src, weight, nullptr, dst, rows, dim, eps);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_rms_norm_add(
        const float * src, const float * add, const float * weight, float * dst,
        int rows, int dim, float eps, void * stream) {
    gp_rms_norm_kernel<<<rows, 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        src, weight, add, dst, rows, dim, eps);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_rms_norm_heads_kernel(
        const float * __restrict__ src,
        const float * __restrict__ weight,
        float * __restrict__ dst,
        int batch,
        int seq,
        int heads,
        int head_dim,
        int row_width,
        float eps) {
    extern __shared__ float s[];
    const int b = blockIdx.z;
    const int h = blockIdx.y;
    const int pos = blockIdx.x;
    float sum = 0.0f;
    const int64_t src_base = ((int64_t) b*seq + pos)*row_width + h*head_dim;
    const int64_t dst_base = ((int64_t) b*heads + h)*seq*head_dim + pos*head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = src[src_base + i];
        sum += v*v;
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(s[0] / (float) head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        dst[dst_base + i] = src[src_base + i] * inv * weight[i];
    }
}

GP_CUDA_EXPORT int gp_rms_norm_heads(
        const float * src, const float * weight, float * dst,
        int batch, int seq, int heads, int head_dim, int row_width,
        float eps, void * stream) {
    gp_rms_norm_heads_kernel<<<dim3(seq, heads, batch), 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        src, weight, dst, batch, seq, heads, head_dim, row_width, eps);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_split_heads_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int batch,
        int seq,
        int heads,
        int head_dim,
        int row_width) {
    const int idx = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    const int total = batch*seq*heads*head_dim;
    if (idx >= total) return;
    const int i = idx % head_dim;
    const int h = (idx / head_dim) % heads;
    const int pos = (idx / (head_dim*heads)) % seq;
    const int b = idx / (head_dim*heads*seq);
    dst[((int64_t)b*heads + h)*seq*head_dim + pos*head_dim + i] =
        src[((int64_t)b*seq + pos)*row_width + h*head_dim + i];
}

GP_CUDA_EXPORT int gp_split_heads(
        const float * src, float * dst,
        int batch, int seq, int heads, int head_dim, int row_width,
        void * stream) {
    const int total = batch*seq*heads*head_dim;
    gp_split_heads_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        src, dst, batch, seq, heads, head_dim, row_width);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_rope_neox_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int total_pairs,
        int seq,
        int head_dim,
        float base_freq) {
    const int idx = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    const int half = head_dim / 2;
    if (idx >= total_pairs) return;
    const int i = idx % half;
    const int pos = (idx / half) % seq;
    const int row = idx / half;
    const int64_t off = (int64_t) row*head_dim;
    const float x1 = src[off + i];
    const float x2 = src[off + half + i];
    const float inv = powf(base_freq, -2.0f*(float)i/(float)head_dim);
    const float f = (float)pos * inv;
    const float c = cosf(f);
    const float s = sinf(f);
    dst[off + i] = x1*c - x2*s;
    dst[off + half + i] = x2*c + x1*s;
}

GP_CUDA_EXPORT int gp_rope_neox(
        const float * src, float * dst,
        int batch, int seq, int heads, int head_dim,
        float base_freq, void * stream) {
    const int total_pairs = batch*heads*seq*(head_dim/2);
    gp_rope_neox_kernel<<<(total_pairs + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        src, dst, total_pairs, seq, head_dim, base_freq);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_attention_scores(
        void * blas,
        const float * q,
        const float * k,
        float * scores,
        int batch,
        int heads,
        int seq,
        int head_dim) {
    cublasHandle_t handle = (cublasHandle_t) blas;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (batch == 1) {
        cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            seq,
            seq,
            head_dim,
            &alpha,
            k,
            head_dim,
            0,
            q,
            head_dim,
            (long long) seq*head_dim,
            &beta,
            scores,
            seq,
            (long long) seq*seq,
            heads);
        return st == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + (int) st;
    }
    for (int b = 0; b < batch; ++b) {
        const float * kb = k + (int64_t)b*seq*head_dim;
        for (int h = 0; h < heads; ++h) {
            const float * qh = q + ((int64_t)b*heads + h)*seq*head_dim;
            float * sh = scores + ((int64_t)b*heads + h)*seq*seq;
            cublasStatus_t st = cublasSgemm(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                seq,
                seq,
                head_dim,
                &alpha,
                kb,
                head_dim,
                qh,
                head_dim,
                &beta,
                sh,
                seq);
            if (st != CUBLAS_STATUS_SUCCESS) return 10000 + (int) st;
        }
    }
    return 0;
}

__global__ void gp_softmax_mask_kernel(
        float * __restrict__ scores,
        const uint32_t * __restrict__ mask,
        int batch,
        int heads,
        int seq,
        int sliding_window,
        float scale) {
    extern __shared__ float s[];
    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    float local_max = -INFINITY;
    const int64_t base = ((int64_t)b*heads + h)*seq*seq + (int64_t)q*seq;
    for (int k = threadIdx.x; k < seq; k += blockDim.x) {
        const bool key_visible = mask[(int64_t)b*seq + k] != 0;
        const bool in_window = sliding_window <= 0 || abs(q - k) < sliding_window;
        const float v = (key_visible && in_window) ? scores[base + k] * scale : -1.0e9f;
        scores[base + k] = v;
        local_max = fmaxf(local_max, v);
    }
    s[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + stride]);
        __syncthreads();
    }
    const float max_v = s[0];
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < seq; k += blockDim.x) {
        const float v = expf(scores[base + k] - max_v);
        scores[base + k] = v;
        local_sum += v;
    }
    s[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = 1.0f / fmaxf(s[0], 1.0e-20f);
    for (int k = threadIdx.x; k < seq; k += blockDim.x) {
        scores[base + k] *= inv;
    }
}

GP_CUDA_EXPORT int gp_softmax_mask(
        float * scores,
        const uint32_t * mask,
        int batch,
        int heads,
        int seq,
        int sliding_window,
        float scale,
        void * stream) {
    gp_softmax_mask_kernel<<<dim3(seq, heads, batch), 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        scores, mask, batch, heads, seq, sliding_window, scale);
    return (int) cudaPeekAtLastError();
}

GP_CUDA_EXPORT int gp_attention_values(
        void * blas,
        const float * scores,
        const float * v,
        float * out,
        int batch,
        int heads,
        int seq,
        int head_dim) {
    cublasHandle_t handle = (cublasHandle_t) blas;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (batch == 1) {
        cublasStatus_t st = cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            head_dim,
            seq,
            seq,
            &alpha,
            v,
            head_dim,
            0,
            scores,
            seq,
            (long long) seq*seq,
            &beta,
            out,
            head_dim,
            (long long) seq*head_dim,
            heads);
        return st == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + (int) st;
    }
    for (int b = 0; b < batch; ++b) {
        const float * vb = v + (int64_t)b*seq*head_dim;
        for (int h = 0; h < heads; ++h) {
            const float * sh = scores + ((int64_t)b*heads + h)*seq*seq;
            float * oh = out + ((int64_t)b*heads + h)*seq*head_dim;
            cublasStatus_t st = cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                head_dim,
                seq,
                seq,
                &alpha,
                vb,
                head_dim,
                sh,
                seq,
                &beta,
                oh,
                head_dim);
            if (st != CUBLAS_STATUS_SUCCESS) return 10000 + (int) st;
        }
    }
    return 0;
}

__global__ void gp_merge_heads_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int batch,
        int seq,
        int heads,
        int head_dim) {
    const int idx = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    const int total = batch*seq*heads*head_dim;
    if (idx >= total) return;
    const int i = idx % head_dim;
    const int h = (idx / head_dim) % heads;
    const int pos = (idx / (head_dim*heads)) % seq;
    const int b = idx / (head_dim*heads*seq);
    dst[((int64_t)b*seq + pos)*heads*head_dim + h*head_dim + i] =
        src[((int64_t)b*heads + h)*seq*head_dim + pos*head_dim + i];
}

GP_CUDA_EXPORT int gp_merge_heads(
        const float * src, float * dst,
        int batch, int seq, int heads, int head_dim,
        void * stream) {
    const int total = batch*seq*heads*head_dim;
    gp_merge_heads_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        src, dst, batch, seq, heads, head_dim);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_geglu_kernel(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float * __restrict__ dst,
        int total) {
    const int i = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    if (i >= total) return;
    const float x = gate[i];
    const float x3 = x*x*x;
    const float gelu = 0.5f*x*(1.0f + tanhf(0.7978845608028654f*(x + 0.044715f*x3)));
    dst[i] = gelu * up[i];
}

GP_CUDA_EXPORT int gp_geglu(
        const float * gate,
        const float * up,
        float * dst,
        int total,
        void * stream) {
    gp_geglu_kernel<<<(total + 255)/256, 256, 0, (cudaStream_t) stream>>>(
        gate, up, dst, total);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_mean_pool_kernel(
        const float * __restrict__ hidden,
        const uint32_t * __restrict__ mask,
        float * __restrict__ dst,
        int batch,
        int seq,
        int hidden_dim) {
    const int h = (int)(blockIdx.x*blockDim.x + threadIdx.x);
    const int b = blockIdx.y;
    if (h >= hidden_dim) return;
    float sum = 0.0f;
    int count = 0;
    for (int s = 0; s < seq; ++s) {
        if (mask[(int64_t)b*seq + s] != 0) {
            sum += hidden[((int64_t)b*seq + s)*hidden_dim + h];
            count += 1;
        }
    }
    dst[(int64_t)b*hidden_dim + h] = sum / (float) max(count, 1);
}

GP_CUDA_EXPORT int gp_mean_pool(
        const float * hidden,
        const uint32_t * mask,
        float * dst,
        int batch,
        int seq,
        int hidden_dim,
        void * stream) {
    gp_mean_pool_kernel<<<dim3((hidden_dim + 255)/256, batch, 1), 256, 0, (cudaStream_t) stream>>>(
        hidden, mask, dst, batch, seq, hidden_dim);
    return (int) cudaPeekAtLastError();
}

__global__ void gp_l2_norm_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        int rows,
        int dim) {
    extern __shared__ float s[];
    const int row = blockIdx.x;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float v = src[(int64_t)row*dim + i];
        sum += v*v;
    }
    s[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(fmaxf(s[0], 1.0e-12f));
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[(int64_t)row*dim + i] = src[(int64_t)row*dim + i] * inv;
    }
}

GP_CUDA_EXPORT int gp_l2_norm(
        const float * src,
        float * dst,
        int rows,
        int dim,
        void * stream) {
    gp_l2_norm_kernel<<<rows, 256, 256*sizeof(float), (cudaStream_t) stream>>>(
        src, dst, rows, dim);
    return (int) cudaPeekAtLastError();
}

static int gp_blck_size(int dtype) {
    switch ((ggml_type) dtype) {
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q8_0:
            return 32;
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            return 256;
        default:
            return 0;
    }
}

template <ggml_type type, int mmq_x, bool use_stream_k>
static cudaError_t gp_launch_mmq_typed(
        const char * weights,
        const int * y_q8,
        float * dst,
        float * fixup,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        int64_t ncols_dst,
        int nsm,
        cudaStream_t stream) {
    constexpr int cc = 860;
    constexpr int warp_size = 32;
    constexpr int nwarps = 8;
    constexpr int mmq_y = 128;
    const int nbytes_shared = (int) mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size, nwarps);
    cudaFuncSetAttribute((const void *) mul_mat_q<type, mmq_x, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);
    cudaFuncSetAttribute((const void *) mul_mat_q<type, mmq_x, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);

    const dim3 block_dims(warp_size, nwarps, 1);
    const int64_t nrows_dst = nrows_x;
    const int64_t ncols_y = ncols_dst;
    const int64_t ncols_max = ncols_dst;
    const int nty = (int)((nrows_x + mmq_y - 1) / mmq_y);
    const int ntx = (int)((ncols_max + mmq_x - 1) / mmq_x);

    const int channel_ratio = 1;
    const int sample_ratio = 1;
    const int64_t zero = 0;
    constexpr bool need_check = true;

    if constexpr (!use_stream_k) {
        const dim3 grid(nty, 1, 1);
        mul_mat_q<type, mmq_x, need_check><<<grid, block_dims, nbytes_shared, stream>>>(
            weights, y_q8, nullptr, nullptr, dst, nullptr,
            ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, nrows_dst,
            channel_ratio, 1, zero, zero, zero,
            sample_ratio, 1, zero, zero, zero,
            ncols_max);
        return cudaPeekAtLastError();
    } else {
        const int tile_count = ntx*nty;
        const int grid_x = ncols_dst <= 128 ? nsm : (tile_count < nsm ? tile_count : nsm);
        const dim3 grid(grid_x, 1, 1);
        const bool fixup_needed = tile_count % grid_x != 0;
        if (fixup_needed) {
            cudaError_t err = cudaMemsetAsync(dst, 0, (size_t) nrows_x * (size_t) ncols_dst * sizeof(float), stream);
            if (err != cudaSuccess) return err;
        }
        float * tmp = fixup_needed ? fixup : nullptr;
        mul_mat_q<type, mmq_x, need_check><<<grid, block_dims, nbytes_shared, stream>>>(
            weights, y_q8, nullptr, nullptr, dst, tmp,
            ncols_x, nrows_x, ncols_dst, stride_row_x, ncols_y, nrows_dst,
            channel_ratio, 1, zero, zero, zero,
            sample_ratio, 1, zero, zero, zero,
            ncols_max);
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess || !fixup_needed) return err;
        mul_mat_q_stream_k_fixup<type, mmq_x, need_check><<<grid, block_dims, 0, stream>>>(
            nullptr, nullptr, dst, tmp,
            ncols_x, nrows_x, ncols_dst, nrows_dst,
            1, zero, 1, zero, ncols_max);
        return cudaPeekAtLastError();
    }
}

template <ggml_type type, int mmq_x>
static cudaError_t gp_launch_mmq_choose_stream(
        const char * weights,
        const int * y_q8,
        float * dst,
        float * fixup,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        int64_t ncols_dst,
        int nsm,
        bool use_stream_k,
        cudaStream_t stream) {
    if (use_stream_k) {
        return gp_launch_mmq_typed<type, mmq_x, true>(
            weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, stream);
    }
    return gp_launch_mmq_typed<type, mmq_x, false>(
        weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, stream);
}

template <ggml_type type>
static cudaError_t gp_launch_mmq_type(
        const char * weights,
        const int * y_q8,
        float * dst,
        float * fixup,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        int64_t ncols_dst,
        int nsm,
        bool use_stream_k,
        cudaStream_t stream) {
    int mmq_x_best = 8;
    int ntiles_x_best = INT_MAX;
    constexpr int cc = 860;
    constexpr int warp_size = 32;
    constexpr int nwarps = 8;
    constexpr int mmq_y = 128;
    int smpbo = 0;
    cudaDeviceGetAttribute(&smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    for (int mmq_x = 8; mmq_x <= 128 && ntiles_x_best > 1; mmq_x += 8) {
        const int granularity = mmq_get_granularity_host(mmq_x, cc);
        if (mmq_x % granularity != 0) continue;
        if ((int) mmq_get_nbytes_shared<type>(mmq_x, mmq_y, cc, warp_size, nwarps) > smpbo) continue;
        const int ntiles_x = (int)((ncols_dst + mmq_x - 1) / mmq_x);
        if (ntiles_x < ntiles_x_best) {
            mmq_x_best = mmq_x;
            ntiles_x_best = ntiles_x;
        }
    }
    switch (mmq_x_best) {
        case 8:   return gp_launch_mmq_choose_stream<type, 8>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 16:  return gp_launch_mmq_choose_stream<type, 16>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 24:  return gp_launch_mmq_choose_stream<type, 24>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 32:  return gp_launch_mmq_choose_stream<type, 32>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 40:  return gp_launch_mmq_choose_stream<type, 40>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 48:  return gp_launch_mmq_choose_stream<type, 48>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 56:  return gp_launch_mmq_choose_stream<type, 56>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 64:  return gp_launch_mmq_choose_stream<type, 64>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 72:  return gp_launch_mmq_choose_stream<type, 72>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 80:  return gp_launch_mmq_choose_stream<type, 80>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 88:  return gp_launch_mmq_choose_stream<type, 88>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 96:  return gp_launch_mmq_choose_stream<type, 96>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 104: return gp_launch_mmq_choose_stream<type, 104>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 112: return gp_launch_mmq_choose_stream<type, 112>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        case 120: return gp_launch_mmq_choose_stream<type, 120>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
        default:  return gp_launch_mmq_choose_stream<type, 128>(weights, y_q8, dst, fixup, ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
    }
}

GP_CUDA_EXPORT int gp_mmq_matmul(
        int dtype,
        const void * weights,
        const float * src,
        float * dst,
        void * q8_scratch,
        void * fixup_scratch,
        int64_t ncols_x,
        int64_t stride_row_x,
        int64_t nrows_x,
        int64_t ncols_dst,
        void * stream_ptr) {
    cudaStream_t stream = (cudaStream_t) stream_ptr;
    const int64_t ne0 = GGML_PAD(ncols_x, MATRIX_ROW_PADDING);
    const int64_t s01 = ncols_x;
    quantize_mmq_q8_1_cuda(
        src, nullptr, q8_scratch, (ggml_type) dtype,
        ncols_x, s01, 0, 0,
        ne0, ncols_dst, 1, 1,
        stream);
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) return (int) err;

    int nsm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    const bool use_stream_k = ncols_dst <= 128 || ncols_dst >= 512;
    switch ((ggml_type) dtype) {
        case GGML_TYPE_Q4_K:
            err = gp_launch_mmq_type<GGML_TYPE_Q4_K>(
                (const char *) weights, (const int *) q8_scratch, dst, (float *) fixup_scratch,
                ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
            break;
        case GGML_TYPE_Q6_K:
            err = gp_launch_mmq_type<GGML_TYPE_Q6_K>(
                (const char *) weights, (const int *) q8_scratch, dst, (float *) fixup_scratch,
                ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
            break;
        case GGML_TYPE_Q8_0:
            err = gp_launch_mmq_type<GGML_TYPE_Q8_0>(
                (const char *) weights, (const int *) q8_scratch, dst, (float *) fixup_scratch,
                ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
            break;
        case GGML_TYPE_Q5_0:
            err = gp_launch_mmq_type<GGML_TYPE_Q5_0>(
                (const char *) weights, (const int *) q8_scratch, dst, (float *) fixup_scratch,
                ncols_x, stride_row_x, nrows_x, ncols_dst, nsm, use_stream_k, stream);
            break;
        default:
            return (int) cudaErrorInvalidValue;
    }
    return (int) err;
}
