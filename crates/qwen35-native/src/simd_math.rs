// The vector exponential below is derived from ggml/src/ggml-cpu/vec.h.
// ggml is MIT licensed; its license is preserved in
// crates/embed-native/vendor/LICENSE-ggml.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const EXP_ROUND: f32 = f32::from_bits(0x4b40_0000);
const LOG2_E: f32 = f32::from_bits(0x3fb8_aa3b);
const LN2_HIGH: f32 = f32::from_bits(0x3f31_7200);
const LN2_LOW: f32 = f32::from_bits(0x35bf_be8e);
const EXP_C1: f32 = f32::from_bits(0x3f7f_fff6);
const EXP_C3: f32 = f32::from_bits(0x3eff_fedb);
const EXP_C5: f32 = f32::from_bits(0x3e2a_af33);
const EXP_C7: f32 = f32::from_bits(0x3d2b_9f17);
const EXP_C9: f32 = f32::from_bits(0x3c07_2010);

pub(crate) fn silu_in_place(values: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe {
            silu_in_place_avx2(values);
        }
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        silu_in_place_neon(values);
        return;
    }

    #[allow(unreachable_code)]
    for value in values {
        *value = silu_scalar(*value);
    }
}

pub(crate) fn swiglu_in_place(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe {
            swiglu_in_place_avx2(gate, up);
        }
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        swiglu_in_place_neon(gate, up);
        return;
    }

    #[allow(unreachable_code)]
    for (gate, up) in gate.iter_mut().zip(up) {
        *gate = silu_scalar(*gate) * up;
    }
}

pub(crate) fn mul_silu_in_place(values: &mut [f32], gate: &[f32]) {
    debug_assert_eq!(values.len(), gate.len());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe {
            mul_silu_in_place_avx2(values, gate);
        }
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        mul_silu_in_place_neon(values, gate);
        return;
    }

    #[allow(unreachable_code)]
    for (value, gate) in values.iter_mut().zip(gate) {
        *value *= silu_scalar(*gate);
    }
}

pub(crate) fn mul_sigmoid_in_place(values: &mut [f32], gate: &[f32]) {
    debug_assert_eq!(values.len(), gate.len());

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe {
            mul_sigmoid_in_place_avx2(values, gate);
        }
        return;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        mul_sigmoid_in_place_neon(values, gate);
        return;
    }

    #[allow(unreachable_code)]
    for (value, gate) in values.iter_mut().zip(gate) {
        *value *= sigmoid_scalar(*gate);
    }
}

pub(crate) fn exp_sum_shifted_in_place(values: &mut [f32], shift: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        unsafe {
            return exp_sum_shifted_in_place_avx2(values, shift);
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe {
        return exp_sum_shifted_in_place_neon(values, shift);
    }

    #[allow(unreachable_code)]
    values
        .iter_mut()
        .map(|value| {
            *value = (*value - shift).exp();
            *value
        })
        .sum()
}

#[inline]
fn silu_scalar(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[inline]
fn sigmoid_scalar(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_approx_avx2(value: __m256) -> __m256 {
    let round = _mm256_set1_ps(EXP_ROUND);
    let z = _mm256_fmadd_ps(value, _mm256_set1_ps(LOG2_E), round);
    let n = _mm256_sub_ps(z, round);
    let b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(LN2_LOW),
        _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HIGH), value),
    );
    let exponent = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    let scale = _mm256_castsi256_ps(_mm256_add_epi32(
        exponent,
        _mm256_castps_si256(_mm256_set1_ps(1.0)),
    ));
    let overflow = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
        _mm256_set1_ps(126.0),
        _CMP_GT_OQ,
    ));
    let squared = _mm256_mul_ps(b, b);
    let polynomial = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(_mm256_set1_ps(EXP_C9), b, _mm256_set1_ps(EXP_C7)),
            squared,
            _mm256_fmadd_ps(_mm256_set1_ps(EXP_C5), b, _mm256_set1_ps(EXP_C3)),
        ),
        squared,
        _mm256_mul_ps(_mm256_set1_ps(EXP_C1), b),
    );
    if _mm256_movemask_ps(_mm256_castsi256_ps(overflow)) == 0 {
        return _mm256_fmadd_ps(polynomial, scale, scale);
    }
    let exponent_adjustment = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(0x8200_0000u32 as i32),
    );
    let scale1 = _mm256_castsi256_ps(_mm256_add_epi32(
        exponent_adjustment,
        _mm256_set1_epi32(0x7f00_0000),
    ));
    let scale2 = _mm256_castsi256_ps(_mm256_sub_epi32(exponent, exponent_adjustment));
    let extreme = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0), n),
        _mm256_set1_ps(192.0),
        _CMP_GT_OQ,
    ));
    _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(extreme), _mm256_mul_ps(scale1, scale1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(extreme),
            _mm256_or_ps(
                _mm256_and_ps(
                    _mm256_castsi256_ps(overflow),
                    _mm256_mul_ps(_mm256_fmadd_ps(scale2, polynomial, scale2), scale1),
                ),
                _mm256_andnot_ps(
                    _mm256_castsi256_ps(overflow),
                    _mm256_fmadd_ps(scale, polynomial, scale),
                ),
            ),
        ),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sigmoid_avx2(value: __m256) -> __m256 {
    let denominator = _mm256_add_ps(
        _mm256_set1_ps(1.0),
        exp_approx_avx2(_mm256_sub_ps(_mm256_setzero_ps(), value)),
    );
    _mm256_div_ps(_mm256_set1_ps(1.0), denominator)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_avx2(value: __m256) -> __m256 {
    _mm256_mul_ps(value, sigmoid_avx2(value))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn silu_in_place_avx2(values: &mut [f32]) {
    let vector_len = values.len() & !7;
    for idx in (0..vector_len).step_by(8) {
        let value = _mm256_loadu_ps(values.as_ptr().add(idx));
        _mm256_storeu_ps(values.as_mut_ptr().add(idx), silu_avx2(value));
    }
    for value in &mut values[vector_len..] {
        *value = silu_scalar(*value);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn swiglu_in_place_avx2(gate: &mut [f32], up: &[f32]) {
    let vector_len = gate.len() & !7;
    for idx in (0..vector_len).step_by(8) {
        let gate_value = _mm256_loadu_ps(gate.as_ptr().add(idx));
        let up_value = _mm256_loadu_ps(up.as_ptr().add(idx));
        _mm256_storeu_ps(
            gate.as_mut_ptr().add(idx),
            _mm256_mul_ps(silu_avx2(gate_value), up_value),
        );
    }
    for (gate, up) in gate[vector_len..].iter_mut().zip(&up[vector_len..]) {
        *gate = silu_scalar(*gate) * up;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_silu_in_place_avx2(values: &mut [f32], gate: &[f32]) {
    let vector_len = values.len() & !7;
    for idx in (0..vector_len).step_by(8) {
        let value = _mm256_loadu_ps(values.as_ptr().add(idx));
        let gate_value = _mm256_loadu_ps(gate.as_ptr().add(idx));
        _mm256_storeu_ps(
            values.as_mut_ptr().add(idx),
            _mm256_mul_ps(value, silu_avx2(gate_value)),
        );
    }
    for (value, gate) in values[vector_len..].iter_mut().zip(&gate[vector_len..]) {
        *value *= silu_scalar(*gate);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn mul_sigmoid_in_place_avx2(values: &mut [f32], gate: &[f32]) {
    let vector_len = values.len() & !7;
    for idx in (0..vector_len).step_by(8) {
        let value = _mm256_loadu_ps(values.as_ptr().add(idx));
        let gate_value = _mm256_loadu_ps(gate.as_ptr().add(idx));
        _mm256_storeu_ps(
            values.as_mut_ptr().add(idx),
            _mm256_mul_ps(value, sigmoid_avx2(gate_value)),
        );
    }
    for (value, gate) in values[vector_len..].iter_mut().zip(&gate[vector_len..]) {
        *value *= sigmoid_scalar(*gate);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn exp_sum_shifted_in_place_avx2(values: &mut [f32], shift: f32) -> f32 {
    let vector_len = values.len() & !7;
    let shift_vector = _mm256_set1_ps(shift);
    let mut sum = _mm256_setzero_ps();
    for idx in (0..vector_len).step_by(8) {
        let value = _mm256_loadu_ps(values.as_ptr().add(idx));
        let exponential = exp_approx_avx2(_mm256_sub_ps(value, shift_vector));
        _mm256_storeu_ps(values.as_mut_ptr().add(idx), exponential);
        sum = _mm256_add_ps(sum, exponential);
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    lanes.iter().sum::<f32>()
        + values[vector_len..]
            .iter_mut()
            .map(|value| {
                *value = (*value - shift).exp();
                *value
            })
            .sum::<f32>()
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn exp_approx_neon(value: float32x4_t) -> float32x4_t {
    let round = vdupq_n_f32(EXP_ROUND);
    let z = vfmaq_f32(round, value, vdupq_n_f32(LOG2_E));
    let n = vsubq_f32(z, round);
    let b = vfmsq_f32(
        vfmsq_f32(value, n, vdupq_n_f32(LN2_HIGH)),
        n,
        vdupq_n_f32(LN2_LOW),
    );
    let exponent = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    let scale = vreinterpretq_f32_u32(vaddq_u32(exponent, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
    let overflow = vcagtq_f32(n, vdupq_n_f32(126.0));
    let squared = vmulq_f32(b, b);
    let polynomial = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(EXP_C1), b),
        vfmaq_f32(
            vfmaq_f32(vdupq_n_f32(EXP_C3), vdupq_n_f32(EXP_C5), b),
            vfmaq_f32(vdupq_n_f32(EXP_C7), vdupq_n_f32(EXP_C9), b),
            squared,
        ),
        squared,
    );
    if vpaddd_u64(vreinterpretq_u64_u32(overflow)) == 0 {
        return vfmaq_f32(scale, polynomial, scale);
    }
    let exponent_adjustment = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x8200_0000));
    let scale1 = vreinterpretq_f32_u32(vaddq_u32(exponent_adjustment, vdupq_n_u32(0x7f00_0000)));
    let scale2 = vreinterpretq_f32_u32(vsubq_u32(exponent, exponent_adjustment));
    vbslq_f32(
        vcagtq_f32(n, vdupq_n_f32(192.0)),
        vmulq_f32(scale1, scale1),
        vbslq_f32(
            overflow,
            vmulq_f32(vfmaq_f32(scale2, scale2, polynomial), scale1),
            vfmaq_f32(scale, scale, polynomial),
        ),
    )
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn sigmoid_neon(value: float32x4_t) -> float32x4_t {
    let denominator = vaddq_f32(vdupq_n_f32(1.0), exp_approx_neon(vnegq_f32(value)));
    vdivq_f32(vdupq_n_f32(1.0), denominator)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn silu_neon(value: float32x4_t) -> float32x4_t {
    vmulq_f32(value, sigmoid_neon(value))
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn silu_in_place_neon(values: &mut [f32]) {
    let vector_len = values.len() & !3;
    for idx in (0..vector_len).step_by(4) {
        let value = vld1q_f32(values.as_ptr().add(idx));
        vst1q_f32(values.as_mut_ptr().add(idx), silu_neon(value));
    }
    for value in &mut values[vector_len..] {
        *value = silu_scalar(*value);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn swiglu_in_place_neon(gate: &mut [f32], up: &[f32]) {
    let vector_len = gate.len() & !3;
    for idx in (0..vector_len).step_by(4) {
        let gate_value = vld1q_f32(gate.as_ptr().add(idx));
        let up_value = vld1q_f32(up.as_ptr().add(idx));
        vst1q_f32(
            gate.as_mut_ptr().add(idx),
            vmulq_f32(silu_neon(gate_value), up_value),
        );
    }
    for (gate, up) in gate[vector_len..].iter_mut().zip(&up[vector_len..]) {
        *gate = silu_scalar(*gate) * up;
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn mul_silu_in_place_neon(values: &mut [f32], gate: &[f32]) {
    let vector_len = values.len() & !3;
    for idx in (0..vector_len).step_by(4) {
        let value = vld1q_f32(values.as_ptr().add(idx));
        let gate_value = vld1q_f32(gate.as_ptr().add(idx));
        vst1q_f32(
            values.as_mut_ptr().add(idx),
            vmulq_f32(value, silu_neon(gate_value)),
        );
    }
    for (value, gate) in values[vector_len..].iter_mut().zip(&gate[vector_len..]) {
        *value *= silu_scalar(*gate);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn mul_sigmoid_in_place_neon(values: &mut [f32], gate: &[f32]) {
    let vector_len = values.len() & !3;
    for idx in (0..vector_len).step_by(4) {
        let value = vld1q_f32(values.as_ptr().add(idx));
        let gate_value = vld1q_f32(gate.as_ptr().add(idx));
        vst1q_f32(
            values.as_mut_ptr().add(idx),
            vmulq_f32(value, sigmoid_neon(gate_value)),
        );
    }
    for (value, gate) in values[vector_len..].iter_mut().zip(&gate[vector_len..]) {
        *value *= sigmoid_scalar(*gate);
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
unsafe fn exp_sum_shifted_in_place_neon(values: &mut [f32], shift: f32) -> f32 {
    let vector_len = values.len() & !3;
    let shift_vector = vdupq_n_f32(shift);
    let mut sum = vdupq_n_f32(0.0);
    for idx in (0..vector_len).step_by(4) {
        let value = vld1q_f32(values.as_ptr().add(idx));
        let exponential = exp_approx_neon(vsubq_f32(value, shift_vector));
        vst1q_f32(values.as_mut_ptr().add(idx), exponential);
        sum = vaddq_f32(sum, exponential);
    }
    let mut lanes = [0.0f32; 4];
    vst1q_f32(lanes.as_mut_ptr(), sum);
    lanes.iter().sum::<f32>()
        + values[vector_len..]
            .iter_mut()
            .map(|value| {
                *value = (*value - shift).exp();
                *value
            })
            .sum::<f32>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values() -> Vec<f32> {
        (-161..=161).map(|value| value as f32 / 8.0).collect()
    }

    fn assert_close(actual: &[f32], expected: &[f32], operation: &str) {
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 3.0e-6 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "{operation} drift at {idx}: actual={actual:.8e} expected={expected:.8e} tolerance={tolerance:.8e}",
            );
        }
    }

    #[test]
    fn simd_silu_stays_close_to_scalar() {
        let mut actual = values();
        let expected = actual
            .iter()
            .map(|value| silu_scalar(*value))
            .collect::<Vec<_>>();
        silu_in_place(&mut actual);
        assert_close(&actual, &expected, "SiLU");
    }

    #[test]
    fn simd_gated_activations_stay_close_to_scalar() {
        let gate = values();
        let up = gate
            .iter()
            .enumerate()
            .map(|(idx, value)| value * 0.125 + idx as f32 / 97.0)
            .collect::<Vec<_>>();
        let mut swiglu = gate.clone();
        let expected_swiglu = gate
            .iter()
            .zip(&up)
            .map(|(gate, up)| silu_scalar(*gate) * up)
            .collect::<Vec<_>>();
        swiglu_in_place(&mut swiglu, &up);
        assert_close(&swiglu, &expected_swiglu, "SwiGLU");

        let mut silu_gate = up.clone();
        let expected_silu_gate = up
            .iter()
            .zip(&gate)
            .map(|(value, gate)| value * silu_scalar(*gate))
            .collect::<Vec<_>>();
        mul_silu_in_place(&mut silu_gate, &gate);
        assert_close(&silu_gate, &expected_silu_gate, "SiLU gate");

        let mut sigmoid_gate = up.clone();
        let expected_sigmoid_gate = up
            .iter()
            .zip(&gate)
            .map(|(value, gate)| value * sigmoid_scalar(*gate))
            .collect::<Vec<_>>();
        mul_sigmoid_in_place(&mut sigmoid_gate, &gate);
        assert_close(&sigmoid_gate, &expected_sigmoid_gate, "sigmoid gate");
    }

    #[test]
    fn simd_shifted_exp_sum_stays_close_to_scalar() {
        let mut actual = values();
        let shift = 3.25;
        let expected = actual
            .iter()
            .map(|value| (*value - shift).exp())
            .collect::<Vec<_>>();
        let expected_sum = expected.iter().sum::<f32>();
        let actual_sum = exp_sum_shifted_in_place(&mut actual, shift);
        assert_close(&actual, &expected, "shifted exp");
        let tolerance = 3.0e-6 * expected_sum.abs().max(1.0);
        assert!(
            (actual_sum - expected_sum).abs() <= tolerance,
            "shifted exp sum drift: actual={actual_sum:.8e} expected={expected_sum:.8e} tolerance={tolerance:.8e}",
        );
    }
}
