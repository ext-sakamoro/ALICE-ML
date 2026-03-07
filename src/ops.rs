//! Ternary Matrix Operations: The Heart of ALICE-ML (Supernova Edition)
//!
//! **The Revolution**: No multiplication. No allocation. Only addition and subtraction.
//!
//! All kernels use Destination Passing Style (DPS):
//! - Output buffer is pre-allocated from Arena
//! - Zero heap allocations in hot path
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::tensor::{QuantizedTensor, Tensor};
use crate::Ternary;

// ============================================================================
// TernaryWeight: Packed 2-bit representation
// ============================================================================

/// Packed ternary weights for a layer
///
/// Weights are stored as 2-bit values packed into bytes.
/// 4 weights per byte: `[w0:2bit][w1:2bit][w2:2bit][w3:2bit]`
#[derive(Clone, Debug)]
pub struct TernaryWeight {
    /// Packed 2-bit ternary values (4 weights per byte)
    packed: Vec<u8>,
    /// Number of output features (rows)
    out_features: usize,
    /// Number of input features (columns)
    in_features: usize,
    /// Scale factor applied after accumulation
    scale: f32,
}

impl TernaryWeight {
    /// Create from ternary values (-1, 0, +1)
    ///
    /// # Panics
    /// Panics if `out_features * in_features` overflows `usize` or does not equal `values.len()`.
    #[must_use]
    pub fn from_ternary(values: &[i8], out_features: usize, in_features: usize) -> Self {
        let total = out_features
            .checked_mul(in_features)
            .expect("dimension overflow: out_features * in_features");
        assert_eq!(values.len(), total);

        let packed_len = values.len().div_ceil(4);
        let mut packed = vec![0u8; packed_len];

        for (i, chunk) in values.chunks(4).enumerate() {
            let mut byte = 0u8;
            for (j, &v) in chunk.iter().enumerate() {
                let t = Ternary::from_i8(v);
                byte |= (t as u8) << (j * 2);
            }
            packed[i] = byte;
        }

        Self {
            packed,
            out_features,
            in_features,
            scale: 1.0,
        }
    }

    /// Create from packed bytes with scale
    ///
    /// # Panics
    /// Panics if `out_features * in_features` overflows `usize` or `packed` is shorter than
    /// the required packed length (`ceil(out_features * in_features / 4)` bytes).
    #[must_use]
    pub fn from_packed(
        packed: Vec<u8>,
        out_features: usize,
        in_features: usize,
        scale: f32,
    ) -> Self {
        let total = out_features
            .checked_mul(in_features)
            .expect("dimension overflow: out_features * in_features");
        let expected_packed = total.div_ceil(4);
        assert!(
            packed.len() >= expected_packed,
            "packed data too short: {} < {}",
            packed.len(),
            expected_packed
        );
        Self {
            packed,
            out_features,
            in_features,
            scale,
        }
    }

    /// Get weight at position
    #[inline]
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Ternary {
        let flat_idx = row * self.in_features + col;
        let byte_idx = flat_idx / 4;
        let bit_offset = (flat_idx % 4) * 2;
        let bits = (self.packed[byte_idx] >> bit_offset) & 0b11;
        Ternary::from_bits(bits)
    }

    /// Number of output features (rows)
    #[inline]
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// Number of input features (columns)
    #[inline]
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// Scale factor applied after accumulation
    #[inline]
    #[must_use]
    pub const fn scale(&self) -> f32 {
        self.scale
    }

    /// Raw packed bytes (4 ternary values per byte)
    #[inline]
    #[must_use]
    pub fn packed(&self) -> &[u8] {
        &self.packed
    }

    /// Memory footprint of packed weights in bytes
    #[inline]
    #[must_use]
    pub const fn memory_bytes(&self) -> usize {
        self.packed.len()
    }

    /// Compression ratio vs FP32 (e.g. 16x for ternary)
    #[inline]
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        if self.packed.is_empty() {
            return 0.0;
        }
        let fp32_size = self.out_features * self.in_features * 4;
        fp32_size as f32 / self.packed.len() as f32
    }
}

// Ternary::from_bits is defined in lib.rs

// ============================================================================
// TernaryWeightKernel: Bit-Parallel representation (Charred Edition)
// ============================================================================

/// Bit-Parallel Ternary Weights
///
/// Separate bitplanes for +1 and -1 weights.
/// 32 weights per u32, enabling true SIMD.
///
/// Cache-line aligned (64 bytes) so the hot fields land in a single L1 cache
/// line, eliminating cross-line fetches during tight inference loops.
#[repr(C, align(64))]
#[derive(Clone, Debug)]
pub struct TernaryWeightKernel {
    /// Bitplane for +1 weights (1 bit per weight, 32 per u32)
    plus_bits: Vec<u32>,
    /// Bitplane for -1 weights (1 bit per weight, 32 per u32)
    minus_bits: Vec<u32>,
    /// Number of output features (rows)
    out_features: usize,
    /// Number of input features (columns)
    in_features: usize,
    /// Scale factor applied after accumulation
    scale: f32,
    /// Number of u32 words per row: `(in_features + 31) / 32`
    words_per_row: usize,
}

impl TernaryWeightKernel {
    /// Create from ternary values
    ///
    /// # Panics
    /// Panics if `out_features * in_features` overflows `usize` or does not equal `values.len()`.
    #[must_use]
    pub fn from_ternary(values: &[i8], out_features: usize, in_features: usize) -> Self {
        Self::from_ternary_scaled(values, out_features, in_features, 1.0)
    }

    /// Create with custom scale
    ///
    /// # Panics
    /// Panics if `out_features * in_features` overflows `usize` or does not equal `values.len()`.
    #[must_use]
    pub fn from_ternary_scaled(
        values: &[i8],
        out_features: usize,
        in_features: usize,
        scale: f32,
    ) -> Self {
        let total = out_features
            .checked_mul(in_features)
            .expect("dimension overflow: out_features * in_features");
        assert_eq!(values.len(), total);

        let words_per_row = in_features.div_ceil(32);
        let total_words = out_features * words_per_row;

        let mut plus_bits = vec![0u32; total_words];
        let mut minus_bits = vec![0u32; total_words];

        for row in 0..out_features {
            for col in 0..in_features {
                let val = values[row * in_features + col];
                let word_idx = row * words_per_row + col / 32;
                let bit_pos = col % 32;

                match val {
                    1 => plus_bits[word_idx] |= 1u32 << bit_pos,
                    -1 => minus_bits[word_idx] |= 1u32 << bit_pos,
                    _ => {}
                }
            }
        }

        Self {
            plus_bits,
            minus_bits,
            out_features,
            in_features,
            scale,
            words_per_row,
        }
    }

    /// Convert from packed `TernaryWeight`
    #[must_use]
    pub fn from_packed_weight(weights: &TernaryWeight) -> Self {
        let out_features = weights.out_features();
        let in_features = weights.in_features();
        let words_per_row = in_features.div_ceil(32);
        let total_words = out_features * words_per_row;

        let mut plus_bits = vec![0u32; total_words];
        let mut minus_bits = vec![0u32; total_words];

        for row in 0..out_features {
            for col in 0..in_features {
                let t = weights.get(row, col);
                let word_idx = row * words_per_row + col / 32;
                let bit_pos = col % 32;

                match t {
                    Ternary::Plus => plus_bits[word_idx] |= 1u32 << bit_pos,
                    Ternary::Minus => minus_bits[word_idx] |= 1u32 << bit_pos,
                    Ternary::Zero => {}
                }
            }
        }

        Self {
            plus_bits,
            minus_bits,
            out_features,
            in_features,
            scale: weights.scale(),
            words_per_row,
        }
    }

    /// Number of output features (rows)
    #[inline]
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// Number of input features (columns)
    #[inline]
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// Scale factor applied after accumulation
    #[inline]
    #[must_use]
    pub const fn scale(&self) -> f32 {
        self.scale
    }

    /// Memory footprint of bit-parallel weights in bytes
    #[inline]
    #[must_use]
    pub const fn memory_bytes(&self) -> usize {
        (self.plus_bits.len() + self.minus_bits.len()) * 4
    }

    /// Compression ratio vs FP32 (e.g. 8x for bit-parallel)
    #[inline]
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        let mem = self.memory_bytes();
        if mem == 0 {
            return 0.0;
        }
        let fp32_size = self.out_features * self.in_features * 4;
        fp32_size as f32 / mem as f32
    }

    /// Raw +1 bitplane (1 bit per weight, 32 per u32)
    #[inline]
    #[must_use]
    pub fn plus_bits(&self) -> &[u32] {
        &self.plus_bits
    }

    /// Raw -1 bitplane (1 bit per weight, 32 per u32)
    #[inline]
    #[must_use]
    pub fn minus_bits(&self) -> &[u32] {
        &self.minus_bits
    }

    /// Number of u32 words per row
    #[inline]
    #[must_use]
    pub const fn words_per_row(&self) -> usize {
        self.words_per_row
    }
}

// ============================================================================
// DPS Kernels: ZERO ALLOCATION
// ============================================================================

/// Ternary Matrix-Vector multiplication (DPS)
///
/// Writes result directly to `output`. No allocation.
///
/// # Arguments
/// * `input` - Input vector (length = `in_features`)
/// * `weights` - Packed ternary weights
/// * `output` - Pre-allocated output buffer (length = `out_features`)
#[inline]
pub fn ternary_matvec(input: &[f32], weights: &TernaryWeight, output: &mut [f32]) {
    debug_assert_eq!(input.len(), weights.in_features());
    debug_assert_eq!(output.len(), weights.out_features());

    let in_features = weights.in_features();
    let scale = weights.scale();

    for (i, out) in output.iter_mut().enumerate() {
        let mut acc_plus = 0.0f32;
        let mut acc_minus = 0.0f32;

        // Incremental byte/bit tracking eliminates /4 and %4 per iteration
        let row_start = i * in_features;
        let mut byte_idx = row_start >> 2;
        let mut bit_offset = (row_start & 3) << 1;

        for val in input.iter().take(in_features) {
            let bits = (weights.packed()[byte_idx] >> bit_offset) & 0b11;

            match bits {
                0b01 => acc_plus += val,
                0b10 => acc_minus += val,
                _ => {}
            }

            bit_offset += 2;
            if bit_offset >= 8 {
                bit_offset = 0;
                byte_idx += 1;
            }
        }

        *out = (acc_plus - acc_minus) * scale;
    }
}

/// Ternary `MatMul` with batched input (DPS)
///
/// # Arguments
/// * `input` - Input tensor data (`batch_size` × `in_features`, row-major)
/// * `weights` - Ternary weights
/// * `output` - Pre-allocated output (`batch_size` × `out_features`)
/// * `batch_size` - Number of input vectors
#[inline]
pub fn ternary_matmul_batch(
    input: &[f32],
    weights: &TernaryWeight,
    output: &mut [f32],
    batch_size: usize,
) {
    let in_features = weights.in_features();
    let out_features = weights.out_features();

    debug_assert_eq!(input.len(), batch_size * in_features);
    debug_assert_eq!(output.len(), batch_size * out_features);

    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        output
            .par_chunks_mut(out_features)
            .zip(input.par_chunks(in_features))
            .for_each(|(y_slice, x_slice)| {
                ternary_matvec(x_slice, weights, y_slice);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for b in 0..batch_size {
            let x_start = b * in_features;
            let x_slice = &input[x_start..x_start + in_features];

            let y_start = b * out_features;
            let y_slice = &mut output[y_start..y_start + out_features];

            ternary_matvec(x_slice, weights, y_slice);
        }
    }
}

/// Bit-parallel ternary matvec (DPS)
#[inline]
pub fn ternary_matvec_kernel(input: &[f32], weights: &TernaryWeightKernel, output: &mut [f32]) {
    debug_assert_eq!(input.len(), weights.in_features());
    debug_assert_eq!(output.len(), weights.out_features());

    let in_features = weights.in_features;
    let words_per_row = weights.words_per_row;
    let scale = weights.scale;

    for (row, out) in output.iter_mut().enumerate() {
        let mut acc_plus = 0.0f32;
        let mut acc_minus = 0.0f32;

        let row_offset = row * words_per_row;

        for word_idx in 0..words_per_row {
            let plus_word = weights.plus_bits[row_offset + word_idx];
            let minus_word = weights.minus_bits[row_offset + word_idx];

            let base_col = word_idx * 32;
            let end_col = (base_col + 32).min(in_features);

            for (bit_pos, val) in input[base_col..end_col].iter().enumerate() {
                if (plus_word >> bit_pos) & 1 != 0 {
                    acc_plus += val;
                } else if (minus_word >> bit_pos) & 1 != 0 {
                    acc_minus += val;
                }
            }
        }

        *out = (acc_plus - acc_minus) * scale;
    }
}

/// Bit-parallel matvec with quantized INT8 input (DPS)
#[inline]
pub fn ternary_matvec_kernel_quantized(
    input: &QuantizedTensor,
    weights: &TernaryWeightKernel,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), weights.in_features());
    debug_assert_eq!(output.len(), weights.out_features());

    let x = input.data_i8();
    let x_scale = input.scale();
    let in_features = weights.in_features;
    let words_per_row = weights.words_per_row;
    let w_scale = weights.scale;

    for (row, out) in output.iter_mut().enumerate() {
        let mut acc_plus = 0i32;
        let mut acc_minus = 0i32;

        let row_offset = row * words_per_row;

        for word_idx in 0..words_per_row {
            let plus_word = weights.plus_bits[row_offset + word_idx];
            let minus_word = weights.minus_bits[row_offset + word_idx];

            let base_col = word_idx * 32;
            let end_col = (base_col + 32).min(in_features);

            for (bit_pos, val) in x[base_col..end_col].iter().enumerate() {
                if (plus_word >> bit_pos) & 1 != 0 {
                    acc_plus += *val as i32;
                } else if (minus_word >> bit_pos) & 1 != 0 {
                    acc_minus += *val as i32;
                }
            }
        }

        *out = (acc_plus - acc_minus) as f32 * w_scale * x_scale;
    }
}

// ============================================================================
// Legacy API (for compatibility with tests/benchmarks)
// ============================================================================

/// Legacy API: allocates output tensor
///
/// **WARNING**: Allocates! Use DPS version in production.
#[must_use]
pub fn ternary_matvec_alloc(input: &Tensor, weights: &TernaryWeight) -> crate::tensor::OwnedTensor {
    let mut output = vec![0.0f32; weights.out_features()];
    ternary_matvec(input.data(), weights, &mut output);
    crate::tensor::OwnedTensor::from_slice(&output, &[weights.out_features()])
}

/// Legacy API: batched matmul with allocation
///
/// # Panics
/// Panics if the input tensor has more than 2 dimensions, or if `in_features` does not match
/// `weights.in_features()`.
#[must_use]
pub fn ternary_matmul_alloc(input: &Tensor, weights: &TernaryWeight) -> crate::tensor::OwnedTensor {
    let shape = input.shape();

    if shape.len() == 1 {
        return ternary_matvec_alloc(input, weights);
    }

    let (batch_size, in_features) = if shape.len() == 2 {
        (shape[0], shape[1])
    } else {
        panic!(
            "ternary_matmul_alloc: only 1D and 2D inputs are supported, got ndim={}",
            input.ndim()
        );
    };

    assert_eq!(in_features, weights.in_features());

    let out_features = weights.out_features();
    let mut output = vec![0.0f32; batch_size * out_features];

    ternary_matmul_batch(input.data(), weights, &mut output, batch_size);

    if batch_size == 1 {
        crate::tensor::OwnedTensor::from_slice(&output, &[out_features])
    } else {
        crate::tensor::OwnedTensor::from_slice(&output, &[batch_size, out_features])
    }
}

// ============================================================================
// AVX2 SIMD Kernels (Supernova Edition)
// ============================================================================

#[cfg(all(target_arch = "x86_64", feature = "simd"))]
/// AVX2-accelerated ternary matrix operations.
pub mod simd {
    use super::{has_avx2, ternary_matvec_kernel, TernaryWeightKernel};
    use core::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_and_si256, _mm256_blendv_ps, _mm256_castps256_ps128,
        _mm256_castsi256_ps, _mm256_cmpeq_epi32, _mm256_extractf128_ps, _mm256_loadu_ps,
        _mm256_set1_epi32, _mm256_set1_ps, _mm256_setr_epi32, _mm256_setzero_ps, _mm_add_ps,
        _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    /// AVX2 ternary matvec using masked blending
    ///
    /// Processes 8 floats at a time with branchless mask operations.
    ///
    /// # Safety
    ///
    /// Caller must ensure the CPU supports AVX2 (checked via `has_avx2()`).
    /// `input.len()` must equal `weights.in_features()` and `output.len()` must
    /// equal `weights.out_features()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn ternary_matvec_avx2(
        input: &[f32],
        weights: &TernaryWeightKernel,
        output: &mut [f32],
    ) {
        let in_features = weights.in_features();
        let words_per_row = weights.words_per_row;
        let scale = weights.scale;
        let _scale_vec = _mm256_set1_ps(scale);

        for (row, out) in output.iter_mut().enumerate() {
            let mut acc_plus = _mm256_setzero_ps();
            let mut acc_minus = _mm256_setzero_ps();

            let row_offset = row * words_per_row;

            // Process 8 floats (256 bits) at a time
            let mut col = 0;
            while col + 8 <= in_features {
                let x_vec = _mm256_loadu_ps(input.as_ptr().add(col));

                // Get masks for these 8 weights
                let word_idx = col / 32;
                let bit_offset = col % 32;

                let plus_word = weights.plus_bits[row_offset + word_idx];
                let minus_word = weights.minus_bits[row_offset + word_idx];

                // Build 8-element masks
                let plus_mask = build_mask_8(plus_word, bit_offset);
                let minus_mask = build_mask_8(minus_word, bit_offset);

                // Blend: select x where mask is set, else 0
                let zeros = _mm256_setzero_ps();
                let plus_selected = _mm256_blendv_ps(zeros, x_vec, plus_mask);
                let minus_selected = _mm256_blendv_ps(zeros, x_vec, minus_mask);

                acc_plus = _mm256_add_ps(acc_plus, plus_selected);
                acc_minus = _mm256_add_ps(acc_minus, minus_selected);

                col += 8;
            }

            // Horizontal sum
            let mut sum_plus = hsum_avx(acc_plus);
            let mut sum_minus = hsum_avx(acc_minus);

            // Handle remaining elements
            while col < in_features {
                let word_idx = col / 32;
                let bit_pos = col % 32;
                let plus_word = weights.plus_bits[row_offset + word_idx];
                let minus_word = weights.minus_bits[row_offset + word_idx];

                if (plus_word >> bit_pos) & 1 != 0 {
                    sum_plus += input[col];
                } else if (minus_word >> bit_pos) & 1 != 0 {
                    sum_minus += input[col];
                }
                col += 1;
            }

            *out = (sum_plus - sum_minus) * scale;
        }
    }

    /// Build 8-element mask from bitfield (Plasma Edition - Pure Vector)
    ///
    /// Converts 8 bits into 8 x 32-bit masks using only SIMD instructions.
    /// Zero branches, zero scalar-to-vector transitions.
    ///
    /// Algorithm:
    /// 1. Broadcast `mask_bits` to all 8 lanes
    /// 2. AND with bit positions \[1,2,4,8,16,32,64,128\]
    /// 3. Compare equal -> `0xFFFFFFFF` where bit was set
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available on the current CPU.
    #[inline(always)]
    unsafe fn build_mask_8(bits: u32, offset: usize) -> __m256 {
        // Extract 8 bits and broadcast to all lanes
        let mask_bits = ((bits >> offset) & 0xFF) as i32;
        let broadcasted = _mm256_set1_epi32(mask_bits);

        // Bit positions for each lane: [1, 2, 4, 8, 16, 32, 64, 128]
        let bit_positions = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);

        // AND: isolate the bit for each lane
        let anded = _mm256_and_si256(broadcasted, bit_positions);

        // Compare equal: if bit was set, anded == bit_positions → 0xFFFFFFFF
        let cmp = _mm256_cmpeq_epi32(anded, bit_positions);

        // Reinterpret as float mask for blendv
        _mm256_castsi256_ps(cmp)
    }

    /// Horizontal sum of AVX vector.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available on the current CPU.
    #[inline(always)]
    unsafe fn hsum_avx(v: __m256) -> f32 {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let sum128 = _mm_add_ps(lo, hi);
        let hi64 = _mm_movehl_ps(sum128, sum128);
        let sum64 = _mm_add_ps(sum128, hi64);
        let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
        let sum32 = _mm_add_ss(sum64, hi32);
        _mm_cvtss_f32(sum32)
    }

    /// Dispatch to AVX2 if available
    #[inline]
    pub fn ternary_matvec_dispatch(
        input: &[f32],
        weights: &TernaryWeightKernel,
        output: &mut [f32],
    ) {
        if has_avx2() {
            unsafe { ternary_matvec_avx2(input, weights, output) };
        } else {
            super::ternary_matvec_kernel(input, weights, output);
        }
    }
}

// ============================================================================
// Cached AVX2 detection (amortize is_x86_feature_detected! overhead)
// ============================================================================

/// Returns `true` if AVX2 is available on the current CPU.
///
/// The result is cached in an `AtomicU8` after the first call, so subsequent
/// calls are a single relaxed load — no CPUID instruction, no branch misprediction.
///
/// Values: 0 = unknown, 1 = not available, 2 = available.
///
/// This function is `pub(crate)` so that [`tensor`] and other sibling modules
/// can share the same cached detection without duplicating the `AtomicU8`.
#[cfg(all(target_arch = "x86_64", feature = "simd", feature = "std"))]
#[inline]
pub(crate) fn has_avx2() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHED: AtomicU8 = AtomicU8::new(0); // 0=unknown, 1=no, 2=yes
    match CACHED.load(Ordering::Relaxed) {
        2 => true,
        1 => false,
        _ => {
            let result = is_x86_feature_detected!("avx2");
            CACHED.store(if result { 2 } else { 1 }, Ordering::Relaxed);
            result
        }
    }
}

/// Fallback when `std` is not available: always probe via `is_x86_feature_detected!`.
///
/// In `no_std` environments this macro resolves to an `__cpuidex` call;
/// caching requires `std::sync::atomic` which is not available there.
#[cfg(all(target_arch = "x86_64", feature = "simd", not(feature = "std")))]
#[inline]
pub(crate) fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

// ============================================================================
// Cross-platform SIMD dispatch
// ============================================================================

/// Dispatch to the best available SIMD kernel for the current platform.
///
/// - `x86_64` + AVX2: 8-wide SIMD
/// - aarch64 + NEON: 4-wide SIMD
/// - Fallback: scalar kernel
#[inline]
pub fn ternary_matvec_simd_dispatch(
    input: &[f32],
    weights: &TernaryWeightKernel,
    output: &mut [f32],
) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        simd::ternary_matvec_dispatch(input, weights, output);
        return;
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        crate::neon::ternary_matvec_dispatch(input, weights, output);
        return;
    }

    #[allow(unreachable_code)]
    {
        ternary_matvec_kernel(input, weights, output);
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Extract +1 mask from packed byte (used in tests and bitmask kernel)
#[inline(always)]
#[allow(dead_code)]
#[must_use]
pub const fn extract_plus_mask(byte: u8) -> u8 {
    let b0 = (byte & 0b0000_0011) == 0b01;
    let b1 = (byte & 0b0000_1100) == 0b0100;
    let b2 = (byte & 0b0011_0000) == 0b01_0000;
    let b3 = (byte & 0b1100_0000) == 0b0100_0000;
    (b0 as u8) | ((b1 as u8) << 1) | ((b2 as u8) << 2) | ((b3 as u8) << 3)
}

/// Extract -1 mask from packed byte
#[inline(always)]
#[allow(dead_code)]
#[must_use]
pub const fn extract_minus_mask(byte: u8) -> u8 {
    let b0 = (byte & 0b0000_0011) == 0b10;
    let b1 = (byte & 0b0000_1100) == 0b1000;
    let b2 = (byte & 0b0011_0000) == 0b10_0000;
    let b3 = (byte & 0b1100_0000) == 0b1000_0000;
    (b0 as u8) | ((b1 as u8) << 1) | ((b2 as u8) << 2) | ((b3 as u8) << 3)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::wildcard_imports)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_weight_creation() {
        let values = vec![1i8, -1, 0, 1, -1, 1, 0, -1];
        let w = TernaryWeight::from_ternary(&values, 2, 4);

        assert_eq!(w.out_features(), 2, "out_features should be 2");
        assert_eq!(w.in_features(), 4, "in_features should be 4");

        assert_eq!(w.get(0, 0), Ternary::Plus, "w[0,0] should be +1");
        assert_eq!(w.get(0, 1), Ternary::Minus, "w[0,1] should be -1");
        assert_eq!(w.get(0, 2), Ternary::Zero, "w[0,2] should be 0");
        assert_eq!(w.get(0, 3), Ternary::Plus, "w[0,3] should be +1");
    }

    #[test]
    fn test_ternary_matvec_dps() {
        // W = [[1, -1], [0, 1]]
        // x = [2, 3]
        // y[0] = 2 - 3 = -1
        // y[1] = 0 + 3 = 3
        let weights = TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);
        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];

        ternary_matvec(&input, &weights, &mut output);

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
    fn test_ternary_matvec_all_plus() {
        let weights = TernaryWeight::from_ternary(&[1, 1, 1, 1], 1, 4);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 1];

        ternary_matvec(&input, &weights, &mut output);

        assert!(
            (output[0] - 10.0).abs() < 1e-6,
            "all-plus sum should be 10, got {}",
            output[0]
        );
    }

    #[test]
    fn test_ternary_matvec_all_minus() {
        let weights = TernaryWeight::from_ternary(&[-1, -1, -1, -1], 1, 4);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 1];

        ternary_matvec(&input, &weights, &mut output);

        assert!(
            (output[0] - (-10.0)).abs() < 1e-6,
            "all-minus sum should be -10, got {}",
            output[0]
        );
    }

    #[test]
    fn test_ternary_matmul_batch_dps() {
        let weights = TernaryWeight::from_ternary(&[1, -1, 1, 1], 2, 2);
        let input = [1.0f32, 2.0, 3.0, 4.0]; // batch of 2
        let mut output = [0.0f32; 4]; // 2 batches × 2 outputs

        ternary_matmul_batch(&input, &weights, &mut output, 2);

        assert!(
            (output[0] - (-1.0)).abs() < 1e-6,
            "batch0 out0: 1-2 = -1, got {}",
            output[0]
        );
        assert!(
            (output[1] - 3.0).abs() < 1e-6,
            "batch0 out1: 1+2 = 3, got {}",
            output[1]
        );
        assert!(
            (output[2] - (-1.0)).abs() < 1e-6,
            "batch1 out0: 3-4 = -1, got {}",
            output[2]
        );
        assert!(
            (output[3] - 7.0).abs() < 1e-6,
            "batch1 out1: 3+4 = 7, got {}",
            output[3]
        );
    }

    #[test]
    fn test_kernel_matvec_dps() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];

        ternary_matvec_kernel(&input, &kernel, &mut output);

        assert!(
            (output[0] - (-1.0)).abs() < 1e-6,
            "kernel y[0] should be -1, got {}",
            output[0]
        );
        assert!(
            (output[1] - 3.0).abs() < 1e-6,
            "kernel y[1] should be 3, got {}",
            output[1]
        );
    }

    #[test]
    fn test_kernel_large_dimension() {
        let values: Vec<i8> = (0..128).map(|i| (i % 3) as i8 - 1).collect();
        let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let mut output = [0.0f32; 2];

        let kernel = TernaryWeightKernel::from_ternary(&values, 2, 64);
        ternary_matvec_kernel(&input, &kernel, &mut output);
        // Verify no panic on large dimension
    }

    #[test]
    fn test_kernel_quantized_dps() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let quantized = QuantizedTensor::from_f32_slice(&[2.0, 3.0], &[2]);
        let mut output = [0.0f32; 2];

        ternary_matvec_kernel_quantized(&quantized, &kernel, &mut output);

        assert!(
            (output[0] - (-1.0)).abs() < 0.1,
            "quantized y[0] should be ~-1, got {}",
            output[0]
        );
        assert!(
            (output[1] - 3.0).abs() < 0.1,
            "quantized y[1] should be ~3, got {}",
            output[1]
        );
    }

    #[test]
    fn test_kernel_vs_packed_equivalence() {
        let values = vec![1i8, -1, 0, 1, -1, 1, 0, -1, 1, 0, -1, 1];
        let input = [1.0f32, 2.0, 3.0, 4.0];

        let packed = TernaryWeight::from_ternary(&values, 3, 4);
        let kernel = TernaryWeightKernel::from_ternary(&values, 3, 4);

        let mut out_packed = [0.0f32; 3];
        let mut out_kernel = [0.0f32; 3];

        ternary_matvec(&input, &packed, &mut out_packed);
        ternary_matvec_kernel(&input, &kernel, &mut out_kernel);

        for i in 0..3 {
            assert!(
                (out_packed[i] - out_kernel[i]).abs() < 1e-6,
                "packed vs kernel mismatch at row {}: {} vs {}",
                i,
                out_packed[i],
                out_kernel[i]
            );
        }
    }

    #[test]
    fn test_compression_ratio() {
        let weights = TernaryWeight::from_ternary(&vec![0i8; 1024 * 1024], 1024, 1024);
        let ratio = weights.compression_ratio();
        assert!(
            (ratio - 16.0).abs() < 0.1,
            "1.58bit should compress 16x vs f32, got {:.1}x",
            ratio
        );
    }

    #[test]
    fn test_extract_masks() {
        let byte = 0b01_00_10_01u8;

        let plus = extract_plus_mask(byte);
        let minus = extract_minus_mask(byte);

        assert_eq!(plus & 0b0001, 1, "plus mask bit 0 should be set");
        assert_eq!(plus & 0b0010, 0, "plus mask bit 1 should be clear");
        assert_eq!(plus & 0b0100, 0, "plus mask bit 2 should be clear");
        assert_eq!(plus & 0b1000, 8, "plus mask bit 3 should be set");

        assert_eq!(minus & 0b0001, 0, "minus mask bit 0 should be clear");
        assert_eq!(minus & 0b0010, 2, "minus mask bit 1 should be set");
        assert_eq!(minus & 0b0100, 0, "minus mask bit 2 should be clear");
        assert_eq!(minus & 0b1000, 0, "minus mask bit 3 should be clear");
    }

    #[test]
    fn test_zero_allocation_inference() {
        // Full inference with stack arrays (zero heap allocation)
        let weights = TernaryWeight::from_ternary(&[1, -1, 1, 1, -1, 1, 0, -1, 1], 3, 3);

        let input_data = [1.0f32, 2.0, 3.0];
        let mut output_data = [0.0f32; 3];

        // Inference: ZERO ALLOCATION
        ternary_matvec(&input_data, &weights, &mut output_data);

        // row 0 = [1,-1,1]·[1,2,3] = 1-2+3 = 2
        assert!(
            (output_data[0] - 2.0).abs() < 1e-6,
            "row0 should be 2, got {}",
            output_data[0]
        );
    }

    #[test]
    fn test_kernel_with_scale() {
        let kernel = TernaryWeightKernel::from_ternary_scaled(&[1, 1, 1, 1], 1, 4, 0.5);
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 1];

        ternary_matvec_kernel(&input, &kernel, &mut output);

        assert!(
            (output[0] - 5.0).abs() < 1e-6,
            "10 * 0.5 = 5, got {}",
            output[0]
        );
    }

    // ============================================================================
    // SIMD and alignment enhancement tests
    // ============================================================================

    /// has_avx2() must be idempotent: calling it multiple times returns the same value
    /// and the cache stores the result on the first call.
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    fn test_has_avx2_idempotent() {
        let first = has_avx2();
        let second = has_avx2();
        let third = has_avx2();
        assert_eq!(first, second, "has_avx2() must return consistent results");
        assert_eq!(second, third, "has_avx2() must return consistent results");
    }

    /// TernaryWeightKernel must be 64-byte aligned.
    #[test]
    fn test_ternary_weight_kernel_alignment() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let ptr = &kernel as *const TernaryWeightKernel as usize;
        assert_eq!(
            ptr % 64,
            0,
            "TernaryWeightKernel must be 64-byte (cache-line) aligned, ptr=0x{:x}",
            ptr
        );
    }

    /// simd_dispatch must produce identical results to the scalar kernel.
    #[test]
    fn test_simd_dispatch_matches_scalar() {
        // Use a dimension > 32 to exercise multiple SIMD words and the remainder path
        let n_in = 40;
        let n_out = 3;
        let values: Vec<i8> = (0..n_in * n_out).map(|i| (i as i8 % 3) - 1).collect();
        let input: Vec<f32> = (0..n_in).map(|i| (i as f32).mul_add(0.1, -2.0)).collect();

        let kernel = TernaryWeightKernel::from_ternary(&values, n_out, n_in);

        let mut out_scalar = vec![0.0f32; n_out];
        let mut out_dispatch = vec![0.0f32; n_out];

        ternary_matvec_kernel(&input, &kernel, &mut out_scalar);
        ternary_matvec_simd_dispatch(&input, &kernel, &mut out_dispatch);

        for i in 0..n_out {
            assert!(
                (out_scalar[i] - out_dispatch[i]).abs() < 1e-4,
                "simd_dispatch mismatch at row {}: scalar={} dispatch={}",
                i,
                out_scalar[i],
                out_dispatch[i]
            );
        }
    }

    // from_packed_weight で変換したカーネルは from_ternary と同じ推論結果を返す
    #[test]
    fn test_from_packed_weight_matches_from_ternary() {
        // 3x4 の混合重みで packed → kernel 変換を検証する
        let values: Vec<i8> = vec![1, -1, 0, 1, -1, 1, 1, 0, 0, -1, 1, -1];
        let packed_weight = TernaryWeight::from_ternary(&values, 3, 4);
        let kernel_direct = TernaryWeightKernel::from_ternary(&values, 3, 4);
        let kernel_from_packed = TernaryWeightKernel::from_packed_weight(&packed_weight);

        let input = [1.0f32, 2.0, 3.0, 4.0];
        let mut out_direct = [0.0f32; 3];
        let mut out_from_packed = [0.0f32; 3];

        ternary_matvec_kernel(&input, &kernel_direct, &mut out_direct);
        ternary_matvec_kernel(&input, &kernel_from_packed, &mut out_from_packed);

        for i in 0..3 {
            assert!(
                (out_direct[i] - out_from_packed[i]).abs() < 1e-5,
                "from_packed_weight mismatch at row {}: direct={} packed={}",
                i,
                out_direct[i],
                out_from_packed[i]
            );
        }
    }

    // TernaryWeight::memory_bytes は ceil(out*in / 4) バイトを返す
    #[test]
    fn test_ternary_weight_memory_bytes() {
        // 4 weights → 1 byte
        let w4 = TernaryWeight::from_ternary(&[1, -1, 0, 1], 1, 4);
        assert_eq!(w4.memory_bytes(), 1, "4 weights should pack into 1 byte");

        // 5 weights → 2 bytes (ceiling)
        let w5 = TernaryWeight::from_ternary(&[1, -1, 0, 1, -1], 1, 5);
        assert_eq!(w5.memory_bytes(), 2, "5 weights should pack into 2 bytes");

        // 8 weights → 2 bytes
        let w8 = TernaryWeight::from_ternary(&[1, -1, 0, 1, -1, 1, 0, -1], 2, 4);
        assert_eq!(w8.memory_bytes(), 2, "8 weights should pack into 2 bytes");
    }

    // TernaryWeight::get が行をまたぐインデックスでも正しく値を返す
    #[test]
    fn test_ternary_weight_get_cross_row_boundary() {
        // 2 行 × 5 列: packed 境界 (インデックス 4 と 5) が行境界と一致しない
        let values: Vec<i8> = vec![1, -1, 0, 1, -1, 1, 0, -1, 1, 1];
        let w = TernaryWeight::from_ternary(&values, 2, 5);

        for row in 0..2 {
            for col in 0..5 {
                let got = w.get(row, col).to_i8();
                let expected = values[row * 5 + col];
                assert_eq!(
                    got, expected,
                    "get({},{}) returned {} expected {}",
                    row, col, got, expected
                );
            }
        }
    }
}
