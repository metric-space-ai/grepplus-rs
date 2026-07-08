//! Minimal Rust-side tensor abstraction used by the Metal backend.
//!
//! Holds just enough metadata to feed the `op_*` dispatchers in
//! `src/metal/ops.rs`: shape (`ne[4]`), byte-strides (`nb[4]`), dtype,
//! plus a reference to the backing `Buffer` (our `objc2-metal`-backed
//! MTLBuffer wrapper) and an offset within it.
//!
//! This is NOT a port of ggml's full tensor graph. It is just the
//! flat-tensor view the dispatchers need. Graph-level concerns
//! (allocator, op scheduling, concurrency) are not reproduced.
//!
//! ref (for shape/stride conventions): `ggml.h::ggml_tensor` (ne[], nb[])

use crate::metal::ffi::Buffer;
use std::sync::Arc;

/// Narrow mirror of `enum ggml_type` (ggml.h). Only the subset we
/// care about for Qwen3.5 lives here; numbers match ggml so any
/// helper that reads a GGUF tensor type-code works.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Bf16 = 30,
}

impl GgmlType {
    /// Byte-exact from `ggml.c::ggml_type_name`.
    pub fn name(self) -> &'static str {
        use GgmlType::*;
        match self {
            F32 => "f32",
            F16 => "f16",
            Q4_0 => "q4_0",
            Q4_1 => "q4_1",
            Q5_0 => "q5_0",
            Q5_1 => "q5_1",
            Q8_0 => "q8_0",
            Q8_1 => "q8_1",
            Q2_K => "q2_K",
            Q3_K => "q3_K",
            Q4_K => "q4_K",
            Q5_K => "q5_K",
            Q6_K => "q6_K",
            Q8_K => "q8_K",
            I8 => "i8",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            F64 => "f64",
            Bf16 => "bf16",
        }
    }

    /// Decode from the raw u32 in GGUF tensor-info records.
    pub fn from_raw(raw: u32) -> Option<Self> {
        use GgmlType::*;
        Some(match raw {
            0 => F32,
            1 => F16,
            2 => Q4_0,
            3 => Q4_1,
            6 => Q5_0,
            7 => Q5_1,
            8 => Q8_0,
            9 => Q8_1,
            10 => Q2_K,
            11 => Q3_K,
            12 => Q4_K,
            13 => Q5_K,
            14 => Q6_K,
            15 => Q8_K,
            24 => I8,
            25 => I16,
            26 => I32,
            27 => I64,
            28 => F64,
            30 => Bf16,
            _ => return None,
        })
    }

    /// Block size in elements, byte-exact from `ggml.c::ggml_blck_size`.
    pub fn block_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 | F16 | Bf16 | I8 | I16 | I32 | I64 | F64 => 1,
            Q4_0 | Q4_1 | Q5_0 | Q5_1 | Q8_0 | Q8_1 => 32,
            Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | Q8_K => 256,
        }
    }

    /// Size of a single quant block in bytes, byte-exact from
    /// `ggml.c::ggml_type_size`. Used to compute stride-0 for
    /// contiguous tensors.
    pub fn type_size(self) -> usize {
        use GgmlType::*;
        match self {
            F32 => 4,
            F16 => 2,
            Bf16 => 2,
            F64 => 8,
            I8 => 1,
            I16 => 2,
            I32 => 4,
            I64 => 8,
            Q4_0 => 18, // 2 (delta f16) + 16 (packed nibbles)
            Q4_1 => 20, // 4 (delta+min f16) + 16
            Q5_0 => 22,
            Q5_1 => 24,
            Q8_0 => 34, // 2 + 32
            Q8_1 => 36,
            Q2_K => 82,
            Q3_K => 110,
            Q4_K => 144, // hs (2) + dmin (2) + scales (12) + qs (128)
            Q5_K => 176,
            Q6_K => 210,
            Q8_K => 292,
        }
    }
}

/// Flat tensor descriptor. Matches the ggml convention of reading
/// `ne[]`/`nb[]` off a `ggml_tensor*` — but without the graph.
#[derive(Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: GgmlType,
    /// Extents along each of the 4 supported tensor axes. `ne[0]`
    /// is the fastest-moving (contiguous) axis.
    pub ne: [i64; 4],
    /// Byte strides along each axis (per the `ggml_tensor::nb`
    /// convention — `nb[0]` equals `type_size() / block_size()`
    /// for a contiguous row-major tensor, `nb[i]` equals `nb[i-1] * ne[i-1]`).
    pub nb: [u64; 4],
    /// Backing GPU buffer. `Arc` because several tensors can share
    /// the same underlying weight-buffer slab (e.g. when we mmap a
    /// whole GGUF file into one MTLBuffer).
    pub buffer: Arc<Buffer>,
    /// Offset into `buffer` where this tensor's data starts.
    pub offset: usize,
}

impl Tensor {
    /// Number of elements in the tensor (product of `ne`).
    pub fn n_elements(&self) -> i64 {
        self.ne[0] * self.ne[1] * self.ne[2] * self.ne[3]
    }

    /// Total byte length of the tensor data, correctly accounting
    /// for the quant block size. Byte-exact to `ggml_nbytes`.
    pub fn nbytes(&self) -> usize {
        let bs = self.dtype.block_size();
        let ts = self.dtype.type_size();
        // Standard formula: (n_elements / block_size) * type_size,
        // which collapses to n_elements * type_size for unquantized.
        let n = self.n_elements() as usize;
        (n / bs) * ts
    }

    /// Contiguous row-major stride layout. Matches how GGUF tensors
    /// are serialized and how ggml initializes `tensor->nb`.
    pub fn make_contiguous_strides(dtype: GgmlType, ne: [i64; 4]) -> [u64; 4] {
        let bs = dtype.block_size() as u64;
        let ts = dtype.type_size() as u64;
        let mut nb = [0u64; 4];
        nb[0] = ts;
        nb[1] = nb[0] * (ne[0] as u64) / bs;
        nb[2] = nb[1] * (ne[1] as u64);
        nb[3] = nb[2] * (ne[2] as u64);
        nb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_size_q4_k_is_144() {
        // Q4_K block is 256 elements packed as:
        //   d (2) + dmin (2) + scales (12) + qs (128) = 144 bytes.
        // ref: ggml.c::ggml_type_size
        assert_eq!(GgmlType::Q4_K.type_size(), 144);
        assert_eq!(GgmlType::Q4_K.block_size(), 256);
    }

    #[test]
    fn type_size_bf16_is_2() {
        assert_eq!(GgmlType::Bf16.type_size(), 2);
        assert_eq!(GgmlType::Bf16.block_size(), 1);
    }

    #[test]
    fn contiguous_strides_f32_row_major() {
        // A 2-D f32 tensor [4, 8] (ne[0]=4, ne[1]=8): nb[0]=4,
        // nb[1]=16, nb[2]=128, nb[3]=128. ref: ggml_new_tensor_impl.
        let nb = Tensor::make_contiguous_strides(GgmlType::F32, [4, 8, 1, 1]);
        assert_eq!(nb, [4, 16, 128, 128]);
    }

    #[test]
    fn contiguous_strides_q4_k() {
        // Q4_K [256, 2]: nb[0] = 144, nb[1] = 144*256/256 = 144.
        let nb = Tensor::make_contiguous_strides(GgmlType::Q4_K, [256, 2, 1, 1]);
        assert_eq!(nb[0], 144);
        assert_eq!(nb[1], 144);
    }
}
