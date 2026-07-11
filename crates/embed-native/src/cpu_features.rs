//! Baseline-safe, process-wide CPU feature detection.
//!
//! This module must never contain `target_feature` code. Keeping the cold
//! detector out of optimized kernels prevents an old CPU from executing an
//! accelerated instruction before runtime dispatch has selected a variant.

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    pub sse42: bool,
    pub avx2: bool,
    pub avx_vnni: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub neon: bool,
    pub dotprod: bool,
    pub i8mm: bool,
}

pub fn detected() -> &'static CpuFeatures {
    static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
    FEATURES.get_or_init(detect_baseline)
}

#[inline(never)]
#[cold]
fn detect_baseline() -> CpuFeatures {
    let mut features = CpuFeatures::default();
    #[cfg(target_arch = "x86_64")]
    {
        features.sse42 = std::arch::is_x86_feature_detected!("sse4.2");
        features.avx2 = std::arch::is_x86_feature_detected!("avx2");
        features.avx_vnni = std::arch::is_x86_feature_detected!("avxvnni");
        features.fma = std::arch::is_x86_feature_detected!("fma");
        features.avx512f = std::arch::is_x86_feature_detected!("avx512f");
    }
    #[cfg(target_arch = "aarch64")]
    {
        features.neon = true;
        features.dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
        features.i8mm = std::arch::is_aarch64_feature_detected!("i8mm");
    }
    features
}

#[inline]
pub fn has_avx2() -> bool {
    detected().avx2
}

#[inline]
pub fn has_avx_vnni() -> bool {
    detected().avx_vnni
}

#[inline]
pub fn has_fma() -> bool {
    detected().fma
}

#[inline]
pub fn has_dotprod() -> bool {
    detected().dotprod
}

#[inline]
pub fn has_i8mm() -> bool {
    detected().i8mm
}
