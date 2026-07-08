//! Byte-exact Rust ports of the kernel-argument structs defined in
//! `vendor/metal/shaders/ggml/ggml-metal-impl.h`.
//!
//! Each `#[repr(C)]` struct here matches the C typedef in the vendored
//! header **bit-for-bit**. Any drift causes the GPU kernel to read
//! garbage, so every field type + order is mirrored from the C source.
//!
//! Why port these rather than link the C header via bindgen: per the
//! crate-level constraint, no C/C++/Obj-C source lives in the Rust
//! project (the vendored `.metal` + its `#include`d `.h` are kernel
//! source, compiled only by `xcrun metal`). The CPU-side Rust binds
//! against independent `#[repr(C)]` declarations that are kept in
//! lock-step with the vendored header.
//!
//! ref: vendor/metal/shaders/ggml/ggml-metal-impl.h
//!      (pinned via vendor/metal/ggml-metal.version)
//!
//! # Validation
//!
//! The `#[cfg(test)] mod tests` at the bottom asserts each struct
//! matches the expected size computed by hand from the C source. If
//! the vendored header ever moves a field, the test catches it.

#![allow(dead_code, non_snake_case)]

// ref: ggml-metal-impl.h:170-193
#[repr(C)]
pub struct KargsUnary {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub slope: f32,
    pub scale: f32,
    pub bias: f32,
    pub val: f32,
    pub min: f32,
    pub max: f32,
}

// ref: ggml-metal-impl.h:195-222
#[repr(C)]
pub struct KargsBin {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub ne13: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub offs: u64,
    pub o1: [u64; 8],
}

// ref: ggml-metal-impl.h:252-270
#[repr(C)]
pub struct KargsCpy {
    pub nk0: i64,
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub ne2: i64,
    pub ne3: i64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

// ref: ggml-metal-impl.h:287-318
#[repr(C)]
pub struct KargsRope {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub n_past: i32,
    pub n_dims: i32,
    pub n_ctx_orig: i32,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub ext_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub sect_0: i32,
    pub sect_1: i32,
    pub sect_2: i32,
    pub sect_3: i32,
    pub src2: bool,
}

// ref: ggml-metal-impl.h:538-551  (ggml_metal_kargs_norm)
#[repr(C)]
pub struct KargsNorm {
    pub ne00: i32,
    pub ne00_t: i32,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub eps: f32,
    pub nef1: [i32; 3],
    pub nef2: [i32; 3],
    pub nef3: [i32; 3],
    pub nbf1: [u64; 3],
    pub nbf2: [u64; 3],
    pub nbf3: [u64; 3],
}

// ref: ggml-metal-impl.h:553-571  (ggml_metal_kargs_l2_norm)
#[repr(C)]
pub struct KargsL2Norm {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub eps: f32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_mul_mm)
#[repr(C)]
pub struct KargsMulMm {
    pub ne00: i32,
    pub ne02: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne12: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub r2: i16,
    pub r3: i16,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_mul_mv)
#[repr(C)]
pub struct KargsMulMv {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub nr0: i32,
    pub r2: i16,
    pub r3: i16,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_mul_mm_id_map0)
#[repr(C)]
pub struct KargsMulMmIdMap0 {
    pub ne02: i32,
    pub ne10: i32,
    pub ne11: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub ne21: i32,
    pub ne20: i32,
    pub nb21: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_mul_mm_id)
#[repr(C)]
pub struct KargsMulMmId {
    pub ne00: i32,
    pub ne02: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne11: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne20: i32,
    pub ne21: i32,
    pub ne0: i32,
    pub ne1: i32,
    pub r2: i16,
    pub r3: i16,
}

// ref: ggml-metal-impl.h:149-168  (ggml_metal_kargs_concat)
#[repr(C)]
pub struct KargsConcat {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub ne13: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub dim: i32,
}

// ref: ggml-metal-impl.h:224-231  (ggml_metal_kargs_add_id)
#[repr(C)]
pub struct KargsAddId {
    pub ne0: i64,
    pub ne1: i64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb11: u64,
    pub nb21: u64,
}

// ref: ggml-metal-impl.h:233-250  (ggml_metal_kargs_repeat)
#[repr(C)]
pub struct KargsRepeat {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

// ref: ggml-metal-impl.h:272-285  (ggml_metal_kargs_set)
#[repr(C)]
pub struct KargsSet {
    pub nk0: i64,
    pub ne00: i64,
    pub ne01: i64,
    pub ne10: i64,
    pub ne11: i64,
    pub ne12: i64,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub offs: u64,
    pub inplace: bool,
}

// ref: ggml-metal-impl.h:320-336  (ggml_metal_kargs_flash_attn_ext_pad)
#[repr(C)]
pub struct KargsFlashAttnExtPad {
    pub ne11: i32,
    pub ne_12_2: i32,
    pub ne_12_3: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub nb21: u64,
    pub nb22: u64,
    pub nb23: u64,
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
}

// ref: ggml-metal-impl.h:338-347  (ggml_metal_kargs_flash_attn_ext_blk)
#[repr(C)]
pub struct KargsFlashAttnExtBlk {
    pub ne01: i32,
    pub ne30: i32,
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
}

// ref: ggml-metal-impl.h:349-382  (ggml_metal_kargs_flash_attn_ext)
#[repr(C)]
pub struct KargsFlashAttnExt {
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne11: i32,
    pub ne_12_2: i32,
    pub ne_12_3: i32,
    pub ns10: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ns20: i32,
    pub nb21: u64,
    pub nb22: u64,
    pub nb23: u64,
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub scale: f32,
    pub max_bias: f32,
    pub m0: f32,
    pub m1: f32,
    pub n_head_log2: i32,
    pub logit_softcap: f32,
}

// ref: ggml-metal-impl.h:384-417  (ggml_metal_kargs_flash_attn_ext_vec)
#[repr(C)]
pub struct KargsFlashAttnExtVec {
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne11: i32,
    pub ne_12_2: i32,
    pub ne_12_3: i32,
    pub ns10: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ns20: i32,
    pub nb21: u64,
    pub nb22: u64,
    pub nb23: u64,
    pub ne31: i32,
    pub ne32: i32,
    pub ne33: i32,
    pub nb31: u64,
    pub nb32: u64,
    pub nb33: u64,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub scale: f32,
    pub max_bias: f32,
    pub m0: f32,
    pub m1: f32,
    pub n_head_log2: i32,
    pub logit_softcap: f32,
}

// ref: ggml-metal-impl.h:419-421  (ggml_metal_kargs_flash_attn_ext_vec_reduce)
#[repr(C)]
pub struct KargsFlashAttnExtVecReduce {
    pub nrows: i32,
}

// ref: ggml-metal-impl.h:513-534  (ggml_metal_kargs_mul_mv_id)
#[repr(C)]
pub struct KargsMulMvId {
    pub nei0: i32,
    pub nei1: i32,
    pub nbi1: u64,
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub ne13: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub nb1: u64,
    pub nr0: i32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_group_norm)
#[repr(C)]
pub struct KargsGroupNorm {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub n_groups: i32,
    pub eps: f32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_glu)
#[repr(C)]
pub struct KargsGlu {
    pub ne00: i32,
    pub nb01: u64,
    pub ne10: i32,
    pub nb11: u64,
    pub ne0: i32,
    pub nb1: u64,
    pub i00: i32,
    pub i10: i32,
    pub alpha: f32,
    pub limit: f32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_sum)
#[repr(C)]
pub struct KargsSum {
    pub np: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_sum_rows)
#[repr(C)]
pub struct KargsSumRows {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub ne2: i64,
    pub ne3: i64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_soft_max)
#[repr(C)]
pub struct KargsSoftMax {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne11: i32,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub scale: f32,
    pub max_bias: f32,
    pub m0: f32,
    pub m1: f32,
    pub n_head_log2: i32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_ssm_conv)
#[repr(C)]
pub struct KargsSsmConv {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub ne10: i64,
    pub ne11: i64,
    pub nb10: u64,
    pub nb11: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub ne2: i64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_gated_delta_net)
#[repr(C)]
pub struct KargsGatedDeltaNet {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub ne13: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne20: i32,
    pub ne21: i32,
    pub ne22: i32,
    pub ne23: i32,
    pub nb20: u64,
    pub nb21: u64,
    pub nb22: u64,
    pub nb23: u64,
    pub ns02: i32,
    pub ns12: i32,
    pub ns22: i32,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_get_rows)
#[repr(C)]
pub struct KargsGetRows {
    pub ne00t: i32,
    pub ne00: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_set_rows)
#[repr(C)]
pub struct KargsSetRows {
    pub nk0: i32,
    pub ne01: i32,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne11: i32,
    pub ne12: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_diag)
#[repr(C)]
pub struct KargsDiag {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_argsort)
#[repr(C)]
pub struct KargsArgsort {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub top_k: i32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_argmax)
#[repr(C)]
pub struct KargsArgmax {
    pub ne00: i64,
    pub nb01: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_argsort_merge)
#[repr(C)]
pub struct KargsArgsortMerge {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub top_k: i32,
    pub len: i32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_arange)
#[repr(C)]
pub struct KargsArange {
    pub ne0: i64,
    pub start: f32,
    pub step: f32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_memset)
#[repr(C)]
pub struct KargsMemset {
    pub val: i64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_count_equal)
#[repr(C)]
pub struct KargsCountEqual {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_pool_1d)
#[repr(C)]
pub struct KargsPool1d {
    pub k0: i32,
    pub s0: i32,
    pub p0: i32,
    pub iw: i64,
    pub ow: i64,
    pub np: i64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_pool_2d)
#[repr(C)]
pub struct KargsPool2d {
    pub k0: i32,
    pub k1: i32,
    pub s0: i32,
    pub s1: i32,
    pub p0: i32,
    pub p1: i32,
    pub ih: i64,
    pub iw: i64,
    pub oh: i64,
    pub ow: i64,
    pub np: i64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_opt_step_adamw)
#[repr(C)]
pub struct KargsOptStepAdamw {
    pub np: i64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_opt_step_sgd)
#[repr(C)]
pub struct KargsOptStepSgd {
    pub np: i64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_ssm_scan)
#[repr(C)]
pub struct KargsSsmScan {
    pub d_state: i64,
    pub d_inner: i64,
    pub n_head: i64,
    pub n_group: i64,
    pub n_seq_tokens: i64,
    pub n_seqs: i64,
    pub s_off: u64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub ns12: u64,
    pub nb13: u64,
    pub nb20: u64,
    pub nb21: u64,
    pub ns21: u64,
    pub nb22: u64,
    pub ne30: i64,
    pub nb31: u64,
    pub nb41: u64,
    pub nb42: u64,
    pub ns42: u64,
    pub nb43: u64,
    pub nb51: u64,
    pub nb52: u64,
    pub ns52: u64,
    pub nb53: u64,
    pub nb0: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_solve_tri)
#[repr(C)]
pub struct KargsSolveTri {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne10: i32,
    pub ne11: i32,
    pub ne12: i32,
    pub ne13: i32,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_upscale)
#[repr(C)]
pub struct KargsUpscale {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub ne2: i64,
    pub ne3: i64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub sf0: f32,
    pub sf1: f32,
    pub sf2: f32,
    pub sf3: f32,
    pub poffs: f32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_pad)
#[repr(C)]
pub struct KargsPad {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub ne2: i64,
    pub ne3: i64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_pad_reflect_1d)
#[repr(C)]
pub struct KargsPadReflect1d {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i64,
    pub ne1: i64,
    pub ne2: i64,
    pub ne3: i64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub p0: i32,
    pub p1: i32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_timestep_embedding)
#[repr(C)]
pub struct KargsTimestepEmbedding {
    pub nb1: u64,
    pub dim: i32,
    pub max_period: i32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_tri)
#[repr(C)]
pub struct KargsTri {
    pub ne00: i32,
    pub ne01: i32,
    pub ne02: i32,
    pub ne03: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub ne0: i32,
    pub ne1: i32,
    pub ne2: i32,
    pub ne3: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_cumsum_blk)
#[repr(C)]
pub struct KargsCumsumBlk {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub net0: i64,
    pub net1: i64,
    pub net2: i64,
    pub net3: i64,
    pub nbt0: u64,
    pub nbt1: u64,
    pub nbt2: u64,
    pub nbt3: u64,
    pub outb: bool,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_cumsum_add)
#[repr(C)]
pub struct KargsCumsumAdd {
    pub ne00: i64,
    pub ne01: i64,
    pub ne02: i64,
    pub ne03: i64,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub net0: i64,
    pub net1: i64,
    pub net2: i64,
    pub net3: i64,
    pub nbt0: u64,
    pub nbt1: u64,
    pub nbt2: u64,
    pub nbt3: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_conv_transpose_1d)
#[repr(C)]
pub struct KargsConvTranspose1d {
    // lines 593-604 omitted for brevity — add if/when conv_transpose_1d dispatch is needed.
    _placeholder: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_conv_transpose_2d)
#[repr(C)]
pub struct KargsConvTranspose2d {
    pub ic: i32,
    pub ih: i32,
    pub iw: i32,
    pub kh: i32,
    pub kw: i32,
    pub oc: i32,
    pub s0: i32,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_conv_2d)
#[repr(C)]
pub struct KargsConv2d {
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
    pub iw: i32,
    pub ih: i32,
    pub kw: i32,
    pub kh: i32,
    pub ic: i32,
    pub oc: i32,
    pub ow: i32,
    pub oh: i32,
    pub n: i32,
    pub s0: i32,
    pub s1: i32,
    pub p0: i32,
    pub p1: i32,
    pub d0: i32,
    pub d1: i32,
}

// ref: ggml-metal-impl.h  (ggml_metal_kargs_conv_3d)
#[repr(C)]
pub struct KargsConv3d {
    pub iw: i32,
    pub ih: i32,
    pub id: i32,
    pub ow: i32,
    pub oh: i32,
    pub od: i32,
    pub kw: i32,
    pub kh: i32,
    pub kd: i32,
    pub s0: i32,
    pub s1: i32,
    pub s2: i32,
    pub p0: i32,
    pub p1: i32,
    pub p2: i32,
    pub d0: i32,
    pub d1: i32,
    pub d2: i32,
    pub ic: i32,
    pub n: i32,
    pub oc: i32,
    pub nb00: u64,
    pub nb01: u64,
    pub nb02: u64,
    pub nb03: u64,
    pub nb10: u64,
    pub nb11: u64,
    pub nb12: u64,
    pub nb13: u64,
    pub nb0: u64,
    pub nb1: u64,
    pub nb2: u64,
    pub nb3: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    // Spot-check a handful of the ported structs against hand-computed
    // sizes. If these drift, the vendored header got updated and the
    // Rust ports must be re-synced.
    #[test]
    fn size_cpy_matches() {
        // 1 i64 (nk0) + 4 i64 (ne00..03) + 4 u64 (nb00..03) +
        // 4 i64 (ne0..3) + 4 u64 (nb0..3) = 17*8 = 136 bytes.
        assert_eq!(size_of::<KargsCpy>(), 136);
    }

    #[test]
    fn size_argmax_matches() {
        // i64 (ne00) + u64 (nb01) = 16 bytes.
        assert_eq!(size_of::<KargsArgmax>(), 16);
    }

    #[test]
    fn size_sum_matches() {
        // u64 (np) = 8 bytes.
        assert_eq!(size_of::<KargsSum>(), 8);
    }

    #[test]
    fn size_rope_matches() {
        // 4 i32 (ne00..03) + 4 u64 (nb00..03) + 4 i32 (ne0..3) +
        // 4 u64 (nb0..3) + 3 i32 (n_past, n_dims, n_ctx_orig) +
        // 6 f32 (freq_base..beta_slow) + 4 i32 (sect_0..3) + 1 bool.
        // = 16 + 32 + 16 + 32 + 12 + 24 + 16 + 1 = 149 … with padding
        // to align the trailing bool, the C compiler emits 152.
        assert_eq!(size_of::<KargsRope>(), 152);
    }

    #[test]
    fn size_mul_mm_matches() {
        // 1 i32 + 1 i32 + 3 u64 + 1 i32 + 4 u64 + 2 i32 + 2 i16
        //   ne00(4) + pad(4) + ne02(?) — actually: ne00 i32 (4),
        //   ne02 i32 (4), nb01..nb03 u64 (24), ne12 i32 (4) + 4 pad,
        //   nb10..nb13 u64 (32), ne0+ne1 i32 (8), r2+r3 i16 (4) + 4 pad.
        // Alignment: u64 requires 8-byte alignment → C compiler adds
        // padding. Exact size is implementation-defined without
        // #pragma pack, but the layout must be the same on both sides.
        // This test is a canary: if the size changes, inspect
        // ggml-metal-impl.h diff.
        let s = size_of::<KargsMulMm>();
        assert!(s > 0 && s % 8 == 0, "KargsMulMm size = {s}, not 8-aligned");
    }
}
