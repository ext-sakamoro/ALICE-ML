//! ARM NEON 4-wide Ternary Matvec Kernel
//!
//! Mirror of the AVX2 kernel in `ops::simd` for ARM Cortex-A76 (Raspberry Pi 5).
//! Processes 4 floats per cycle using NEON 128-bit SIMD.
//!
//! Author: Moroya Sakamoto

use super::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Build 4-element mask from bitfield (NEON Edition)
///
/// Converts 4 bits into 4 × 32-bit masks using only NEON instructions.
/// Zero branches, zero scalar-to-vector transitions.
///
/// Algorithm:
/// 1. Broadcast mask_bits to all 4 lanes
/// 2. AND with bit positions [1, 2, 4, 8]
/// 3. Compare equal → 0xFFFFFFFF where bit was set
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn build_mask_4(bits: u32, offset: usize) -> uint32x4_t {
    // Extract 4 bits and broadcast to all lanes
    let mask_bits = ((bits >> offset) & 0x0F) as u32;
    let broadcasted = vdupq_n_u32(mask_bits);

    // Bit positions for each lane: [1, 2, 4, 8]
    let bit_positions: [u32; 4] = [1, 2, 4, 8];
    let bit_pos_vec = vld1q_u32(bit_positions.as_ptr());

    // AND: isolate the bit for each lane
    let anded = vandq_u32(broadcasted, bit_pos_vec);

    // Compare equal: if bit was set, anded == bit_positions → 0xFFFFFFFF
    vceqq_u32(anded, bit_pos_vec)
}

/// Horizontal sum of NEON float32x4 vector
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn hsum_neon(v: float32x4_t) -> f32 {
    // vpaddq reduces pairs: [a,b,c,d] → [a+b, c+d, a+b, c+d]
    let pair = vpaddq_f32(v, v);
    // Then reduce again: [a+b+c+d, ...]
    vgetq_lane_f32(vpaddq_f32(pair, pair), 0)
}

/// NEON ternary matvec using masked blending (4-wide)
///
/// Processes 4 floats at a time with branchless mask operations.
/// Designed for ARM Cortex-A76 (Pi 5) NEON 128-bit SIMD.
#[cfg(target_arch = "aarch64")]
pub unsafe fn ternary_matvec_neon(
    input: &[f32],
    weights: &TernaryWeightKernel,
    output: &mut [f32],
) {
    let in_features = weights.in_features();
    let words_per_row = weights.words_per_row();
    let scale = weights.scale();

    for (row, out) in output.iter_mut().enumerate() {
        let mut acc_plus = vdupq_n_f32(0.0);
        let mut acc_minus = vdupq_n_f32(0.0);

        let row_offset = row * words_per_row;

        // Process 4 floats (128 bits) at a time
        let mut col = 0;
        while col + 4 <= in_features {
            let x_vec = vld1q_f32(input.as_ptr().add(col));

            // Get masks for these 4 weights
            let word_idx = col / 32;
            let bit_offset = col % 32;

            let plus_word = *weights.plus_bits().get_unchecked(row_offset + word_idx);
            let minus_word = *weights.minus_bits().get_unchecked(row_offset + word_idx);

            // Build 4-element masks
            let plus_mask = build_mask_4(plus_word, bit_offset);
            let minus_mask = build_mask_4(minus_word, bit_offset);

            // Blend: select x where mask is set, else 0
            let zeros = vdupq_n_f32(0.0);
            let plus_mask_f = vreinterpretq_f32_u32(plus_mask);
            let plus_selected = vbslq_f32(vreinterpretq_u32_f32(plus_mask_f), x_vec, zeros);
            let minus_mask_f = vreinterpretq_f32_u32(minus_mask);
            let minus_selected = vbslq_f32(vreinterpretq_u32_f32(minus_mask_f), x_vec, zeros);

            acc_plus = vaddq_f32(acc_plus, plus_selected);
            acc_minus = vaddq_f32(acc_minus, minus_selected);

            col += 4;
        }

        // Horizontal sum
        let mut sum_plus = hsum_neon(acc_plus);
        let mut sum_minus = hsum_neon(acc_minus);

        // Handle remaining elements (scalar path)
        while col < in_features {
            let word_idx = col / 32;
            let bit_pos = col % 32;
            let plus_word = *weights.plus_bits().get_unchecked(row_offset + word_idx);
            let minus_word = *weights.minus_bits().get_unchecked(row_offset + word_idx);

            if (plus_word >> bit_pos) & 1 != 0 {
                sum_plus += *input.get_unchecked(col);
            } else if (minus_word >> bit_pos) & 1 != 0 {
                sum_minus += *input.get_unchecked(col);
            }
            col += 1;
        }

        *out = (sum_plus - sum_minus) * scale;
    }
}

/// Dispatch to NEON on aarch64 (always available on ARMv8+)
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn ternary_matvec_dispatch(input: &[f32], weights: &TernaryWeightKernel, output: &mut [f32]) {
    // NEON is mandatory on aarch64 — no runtime detection needed
    unsafe { ternary_matvec_neon(input, weights, output) }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(target_arch = "aarch64")]
mod tests {
    use super::*;

    #[test]
    fn test_neon_basic_matvec() {
        let weights = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];

        unsafe { ternary_matvec_neon(&input, &weights, &mut output) };

        assert!(
            (output[0] - (-1.0)).abs() < 1e-6,
            "y[0] = 2-3 = -1, got {}",
            output[0]
        );
        assert!(
            (output[1] - 3.0).abs() < 1e-6,
            "y[1] = 0+3 = 3, got {}",
            output[1]
        );
    }

    #[test]
    fn test_neon_all_plus() {
        let weights = TernaryWeightKernel::from_ternary(&[1, 1, 1, 1], 1, 4);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 1];

        unsafe { ternary_matvec_neon(&input, &weights, &mut output) };

        assert!(
            (output[0] - 10.0).abs() < 1e-6,
            "all-plus sum should be 10, got {}",
            output[0]
        );
    }

    #[test]
    fn test_neon_vs_scalar_equivalence() {
        let values = vec![1i8, -1, 0, 1, -1, 1, 0, -1, 1, 0, -1, 1];
        let input = [1.0f32, 2.0, 3.0, 4.0];

        let kernel = TernaryWeightKernel::from_ternary(&values, 3, 4);

        let mut out_scalar = [0.0f32; 3];
        let mut out_neon = [0.0f32; 3];

        ternary_matvec_kernel(&input, &kernel, &mut out_scalar);
        unsafe { ternary_matvec_neon(&input, &kernel, &mut out_neon) };

        for i in 0..3 {
            assert!(
                (out_scalar[i] - out_neon[i]).abs() < 1e-6,
                "scalar vs neon mismatch at row {}: {} vs {}",
                i,
                out_scalar[i],
                out_neon[i]
            );
        }
    }

    #[test]
    fn test_neon_large_dimension() {
        // 64 input features → exercises both SIMD loop and remainder
        let values: Vec<i8> = (0..128).map(|i| (i % 3) as i8 - 1).collect();
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();

        let kernel = TernaryWeightKernel::from_ternary(&values, 2, 64);

        let mut out_scalar = [0.0f32; 2];
        let mut out_neon = [0.0f32; 2];

        ternary_matvec_kernel(&input, &kernel, &mut out_scalar);
        unsafe { ternary_matvec_neon(&input, &kernel, &mut out_neon) };

        for i in 0..2 {
            assert!(
                (out_scalar[i] - out_neon[i]).abs() < 1e-4,
                "large dim mismatch at row {}: {} vs {}",
                i,
                out_scalar[i],
                out_neon[i]
            );
        }
    }

    #[test]
    fn test_neon_with_scale() {
        let kernel = TernaryWeightKernel::from_ternary_scaled(&[1, 1, 1, 1], 1, 4, 0.5);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 1];

        unsafe { ternary_matvec_neon(&input, &kernel, &mut output) };

        assert!(
            (output[0] - 5.0).abs() < 1e-6,
            "10 * 0.5 = 5, got {}",
            output[0]
        );
    }

    #[test]
    fn test_neon_dispatch() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];

        ternary_matvec_dispatch(&input, &kernel, &mut output);

        assert!((output[0] - (-1.0)).abs() < 1e-6);
        assert!((output[1] - 3.0).abs() < 1e-6);
    }
}
