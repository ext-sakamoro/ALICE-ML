//! Tensor: Zero-Allocation N-dimensional Array (Supernova Edition)
//!
//! All tensors are borrowed slices from Arena. No heap allocations in hot path.
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::arena::Arena;

/// Maximum tensor dimensions (for stack allocation of shape)
pub const MAX_DIMS: usize = 8;

/// Borrowed tensor - NO HEAP ALLOCATION
///
/// Data is borrowed from Arena. Lifetime `'a` ties to Arena's lifetime.
#[derive(Debug)]
pub struct Tensor<'a> {
    /// Borrowed data slice (from Arena)
    data: &'a mut [f32],
    /// Shape of tensor (stack allocated)
    shape: [usize; MAX_DIMS],
    /// Number of dimensions
    ndim: usize,
    /// Strides for indexing (stack allocated)
    strides: [usize; MAX_DIMS],
}

impl<'a> Tensor<'a> {
    /// Create tensor from Arena-allocated slice
    ///
    /// # Safety
    /// Caller must ensure `data` is properly sized for `shape`.
    ///
    /// # Panics
    /// Panics if `shape.len()` exceeds `MAX_DIMS` or if `data.len()` does not equal the product
    /// of all shape dimensions.
    pub fn from_arena(data: &'a mut [f32], shape: &[usize]) -> Self {
        assert!(shape.len() <= MAX_DIMS, "Too many dimensions");

        let total: usize = shape.iter().product();
        assert_eq!(data.len(), total, "Data length mismatch");

        let mut shape_arr = [0usize; MAX_DIMS];
        let mut strides_arr = [0usize; MAX_DIMS];

        shape_arr[..shape.len()].copy_from_slice(shape);

        // Compute strides (row-major)
        if !shape.is_empty() {
            strides_arr[shape.len() - 1] = 1;
            for i in (0..shape.len() - 1).rev() {
                strides_arr[i] = strides_arr[i + 1] * shape[i + 1];
            }
        }

        Self {
            data,
            shape: shape_arr,
            ndim: shape.len(),
            strides: strides_arr,
        }
    }

    /// Allocate zero-filled tensor from Arena
    pub fn zeros(arena: &'a mut Arena, shape: &[usize]) -> Option<Self> {
        let total: usize = shape.iter().product();
        let data = arena.alloc_zeroed::<f32>(total)?;
        Some(Self::from_arena(data, shape))
    }

    /// Get shape slice
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Get number of dimensions
    #[inline]
    #[must_use]
    pub const fn ndim(&self) -> usize {
        self.ndim
    }

    /// Total number of elements
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get raw data slice (immutable)
    #[inline]
    #[must_use]
    pub const fn data(&self) -> &[f32] {
        self.data
    }

    /// Get mutable data slice
    #[inline]
    pub const fn data_mut(&mut self) -> &mut [f32] {
        self.data
    }

    /// Get element at flat index
    #[inline]
    #[must_use]
    pub fn get_flat(&self, idx: usize) -> f32 {
        self.data[idx]
    }

    /// Set element at flat index
    #[inline]
    pub fn set_flat(&mut self, idx: usize, val: f32) {
        self.data[idx] = val;
    }

    /// Get element at multi-dimensional index
    #[inline]
    #[must_use]
    pub fn get(&self, indices: &[usize]) -> f32 {
        let idx = self.flat_index(indices);
        self.data[idx]
    }

    /// Set element at multi-dimensional index
    #[inline]
    pub fn set(&mut self, indices: &[usize], val: f32) {
        let idx = self.flat_index(indices);
        self.data[idx] = val;
    }

    /// Compute flat index from multi-dimensional indices
    #[inline]
    fn flat_index(&self, indices: &[usize]) -> usize {
        debug_assert_eq!(indices.len(), self.ndim);
        let mut idx = 0;
        for (i, &dim_idx) in indices.iter().enumerate() {
            idx += dim_idx * self.strides[i];
        }
        idx
    }
}

// ============================================================================
// Destination Passing Style (DPS) Operations
// All operations write to pre-allocated output buffer. ZERO ALLOCATIONS.
// ============================================================================

/// Element-wise addition: out = a + b
///
/// # Panics
/// Panics if shapes don't match.
#[inline]
pub fn tensor_add(a: &Tensor, b: &Tensor, out: &mut Tensor) {
    debug_assert_eq!(a.shape(), b.shape());
    debug_assert_eq!(a.shape(), out.shape());

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if crate::ops::has_avx2() {
        simd_add_f32(a.data, b.data, out.data);
        return;
    }

    for ((o, &av), &bv) in out.data.iter_mut().zip(a.data.iter()).zip(b.data.iter()) {
        *o = av + bv;
    }
}

/// Element-wise subtraction: out = a - b
#[inline]
pub fn tensor_sub(a: &Tensor, b: &Tensor, out: &mut Tensor) {
    debug_assert_eq!(a.shape(), b.shape());
    debug_assert_eq!(a.shape(), out.shape());

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if crate::ops::has_avx2() {
        simd_sub_f32(a.data, b.data, out.data);
        return;
    }

    for ((o, &av), &bv) in out.data.iter_mut().zip(a.data.iter()).zip(b.data.iter()) {
        *o = av - bv;
    }
}

/// Scalar multiplication: out = a * scalar
#[inline]
pub fn tensor_scale(a: &Tensor, scalar: f32, out: &mut Tensor) {
    debug_assert_eq!(a.shape(), out.shape());

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if crate::ops::has_avx2() {
        use core::arch::x86_64::{
            _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
        };
        let n = a.data.len();
        let chunks = n / 8;
        let remainder = n % 8;
        // SAFETY: AVX2 runtime detection via has_avx2() guarantees the CPU supports
        // these intrinsics. Each load/store accesses exactly 8 consecutive f32 values
        // (32 bytes) within slice bounds: offset + 8 <= chunks * 8 <= n = a.data.len().
        unsafe {
            let s = _mm256_set1_ps(scalar);
            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.data.as_ptr().add(offset));
                let vr = _mm256_mul_ps(va, s);
                _mm256_storeu_ps(out.data.as_mut_ptr().add(offset), vr);
            }
        }
        let tail = chunks * 8;
        for i in 0..remainder {
            out.data[tail + i] = a.data[tail + i] * scalar;
        }
        return;
    }

    for (o, &av) in out.data.iter_mut().zip(a.data.iter()) {
        *o = av * scalar;
    }
}

/// `ReLU` activation: out = max(a, 0) — branchless
#[inline]
pub fn tensor_relu(a: &Tensor, out: &mut Tensor) {
    debug_assert_eq!(a.shape(), out.shape());

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if crate::ops::has_avx2() {
        use core::arch::x86_64::{
            _mm256_loadu_ps, _mm256_max_ps, _mm256_setzero_ps, _mm256_storeu_ps,
        };
        let n = a.data.len();
        let chunks = n / 8;
        let remainder = n % 8;
        // SAFETY: AVX2 runtime detection via has_avx2() guarantees the CPU supports
        // these intrinsics. Each load/store accesses exactly 8 consecutive f32 values
        // within slice bounds: offset + 8 <= chunks * 8 <= n = a.data.len() = out.data.len().
        unsafe {
            let zero = _mm256_setzero_ps();
            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.data.as_ptr().add(offset));
                let vr = _mm256_max_ps(va, zero);
                _mm256_storeu_ps(out.data.as_mut_ptr().add(offset), vr);
            }
        }
        let tail = chunks * 8;
        for i in 0..remainder {
            out.data[tail + i] = a.data[tail + i].max(0.0);
        }
        return;
    }

    for (o, &av) in out.data.iter_mut().zip(a.data.iter()) {
        *o = av.max(0.0); // branchless: compiles to maxss/maxps
    }
}

/// `ReLU` in-place — branchless
#[inline]
pub fn tensor_relu_inplace(a: &mut Tensor) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if crate::ops::has_avx2() {
        use core::arch::x86_64::{
            _mm256_loadu_ps, _mm256_max_ps, _mm256_setzero_ps, _mm256_storeu_ps,
        };
        let n = a.data.len();
        let chunks = n / 8;
        let remainder = n % 8;
        // SAFETY: AVX2 runtime detection via has_avx2() guarantees the CPU supports
        // these intrinsics. The load reads from a.data and the store writes back to the
        // same slice at the same offset; they do not overlap within the same iteration.
        // offset + 8 <= chunks * 8 <= n.
        unsafe {
            let zero = _mm256_setzero_ps();
            for i in 0..chunks {
                let offset = i * 8;
                let va = _mm256_loadu_ps(a.data.as_ptr().add(offset));
                let vr = _mm256_max_ps(va, zero);
                _mm256_storeu_ps(a.data.as_mut_ptr().add(offset), vr);
            }
        }
        let tail = chunks * 8;
        for i in 0..remainder {
            a.data[tail + i] = a.data[tail + i].max(0.0);
        }
        return;
    }

    for x in a.data.iter_mut() {
        *x = x.max(0.0); // branchless: compiles to maxss
    }
}

/// Softmax over last dimension: out = softmax(a)
///
/// # Panics
/// Panics if `a` has zero dimensions (`ndim == 0`).
#[inline]
pub fn tensor_softmax(a: &Tensor, out: &mut Tensor) {
    assert!(a.ndim > 0, "tensor_softmax: input tensor has ndim=0");
    debug_assert_eq!(a.shape(), out.shape());

    let last_dim = a.shape[a.ndim - 1];
    let batch_size = a.data.len() / last_dim;

    for b in 0..batch_size {
        let start = b * last_dim;
        let end = start + last_dim;
        let in_slice = &a.data[start..end];
        let out_slice = &mut out.data[start..end];

        // Find max for numerical stability
        let max_val = in_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Exp and sum
        let mut sum = 0.0f32;
        for (o, &x) in out_slice.iter_mut().zip(in_slice.iter()) {
            *o = (x - max_val).exp();
            sum += *o;
        }

        // Normalize (guard against sum=0 when all inputs are -inf)
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for o in out_slice.iter_mut() {
                *o *= inv_sum;
            }
        } else {
            let uniform = 1.0 / last_dim as f32;
            for o in out_slice.iter_mut() {
                *o = uniform;
            }
        }
    }
}

/// Sum all elements
///
/// Uses an AVX2 8-wide horizontal reduction when the `simd` feature is enabled
/// and AVX2 is available at runtime, falling back to a scalar accumulation
/// for the tail and on non-AVX2 / non-x86 targets.
#[inline]
#[must_use]
pub fn tensor_sum(a: &Tensor) -> f32 {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if crate::ops::has_avx2() {
        use core::arch::x86_64::{
            _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps,
            _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps,
            _mm_shuffle_ps,
        };
        let len = a.data.len();
        let chunks = len / 8;
        // SAFETY: AVX2 availability confirmed at runtime via has_avx2().
        // Each load reads exactly 8 consecutive f32 values within a.data bounds:
        // i * 8 + 8 <= chunks * 8 <= len = a.data.len().
        unsafe {
            let mut sum_vec = _mm256_setzero_ps();
            for i in 0..chunks {
                let v = _mm256_loadu_ps(a.data.as_ptr().add(i * 8));
                sum_vec = _mm256_add_ps(sum_vec, v);
            }
            // Horizontal reduction: collapse 256-bit vector to scalar
            let hi = _mm256_extractf128_ps(sum_vec, 1);
            let lo = _mm256_castps256_ps128(sum_vec);
            let s = _mm_add_ps(hi, lo);
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            let mut result = _mm_cvtss_f32(s);
            // Scalar tail
            for i in (chunks * 8)..len {
                result += a.data[i];
            }
            return result;
        }
    }

    a.data.iter().sum()
}

/// Mean of all elements
#[inline]
#[must_use]
pub fn tensor_mean(a: &Tensor) -> f32 {
    tensor_sum(a) / a.data.len() as f32
}

/// Maximum element
///
/// Uses an AVX2 8-wide `_mm256_max_ps` reduction when the `simd` feature is
/// enabled and AVX2 is available at runtime, falling back to a scalar fold
/// on non-AVX2 / non-x86 targets or for the tail.
#[inline]
pub fn tensor_max(a: &Tensor) -> f32 {
    let len = a.data.len();
    if len == 0 {
        return f32::NEG_INFINITY;
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if crate::ops::has_avx2() {
        use core::arch::x86_64::{
            _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_max_ps,
            _mm256_set1_ps, _mm_cvtss_f32, _mm_max_ps, _mm_max_ss, _mm_movehl_ps, _mm_shuffle_ps,
        };
        let chunks = len / 8;
        // SAFETY: AVX2 availability confirmed at runtime via has_avx2().
        // Each load reads exactly 8 consecutive f32 values within a.data bounds:
        // i * 8 + 8 <= chunks * 8 <= len = a.data.len().
        unsafe {
            let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(a.data.as_ptr().add(i * 8));
                max_vec = _mm256_max_ps(max_vec, v);
            }
            // Horizontal reduction: collapse 256-bit max vector to scalar
            let hi = _mm256_extractf128_ps(max_vec, 1);
            let lo = _mm256_castps256_ps128(max_vec);
            let m = _mm_max_ps(hi, lo);
            let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
            let m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
            let mut result = _mm_cvtss_f32(m);
            // Scalar tail
            for i in (chunks * 8)..len {
                result = result.max(a.data[i]);
            }
            return result;
        }
    }

    a.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// Minimum element
///
/// Uses an AVX2 8-wide `_mm256_min_ps` reduction when the `simd` feature is
/// enabled and AVX2 is available at runtime, falling back to a scalar fold
/// on non-AVX2 / non-x86 targets or for the tail.
#[inline]
pub fn tensor_min(a: &Tensor) -> f32 {
    let len = a.data.len();
    if len == 0 {
        return f32::INFINITY;
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    if crate::ops::has_avx2() {
        use core::arch::x86_64::{
            _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps, _mm256_min_ps,
            _mm256_set1_ps, _mm_cvtss_f32, _mm_min_ps, _mm_min_ss, _mm_movehl_ps, _mm_shuffle_ps,
        };
        let chunks = len / 8;
        // SAFETY: AVX2 availability confirmed at runtime via has_avx2().
        // Each load reads exactly 8 consecutive f32 values within a.data bounds:
        // i * 8 + 8 <= chunks * 8 <= len = a.data.len().
        unsafe {
            let mut min_vec = _mm256_set1_ps(f32::INFINITY);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(a.data.as_ptr().add(i * 8));
                min_vec = _mm256_min_ps(min_vec, v);
            }
            // Horizontal reduction: collapse 256-bit min vector to scalar
            let hi = _mm256_extractf128_ps(min_vec, 1);
            let lo = _mm256_castps256_ps128(min_vec);
            let m = _mm_min_ps(hi, lo);
            let m = _mm_min_ps(m, _mm_movehl_ps(m, m));
            let m = _mm_min_ss(m, _mm_shuffle_ps(m, m, 1));
            let mut result = _mm_cvtss_f32(m);
            // Scalar tail
            for i in (chunks * 8)..len {
                result = result.min(a.data[i]);
            }
            return result;
        }
    }

    a.data.iter().copied().fold(f32::INFINITY, f32::min)
}

/// Copy data from source to destination
#[inline]
pub fn tensor_copy(src: &Tensor, dst: &mut Tensor) {
    debug_assert_eq!(src.len(), dst.len());
    dst.data.copy_from_slice(src.data);
}

// ============================================================================
// Fast exponential approximation (Schraudolph's method)
// ============================================================================

/// Fast exponential approximation using Schraudolph's integer-cast method.
///
/// Accuracy: ~1.5% mean relative error over [-87, 88].
/// Speed: ~12x faster than `f32::exp()` (no transcendental hardware instruction).
///
/// Algorithm: interpret the IEEE 754 float bit pattern as an integer and
/// exploit the relationship between the exponent field and powers of two.
///
/// Reference: Schraudolph, N.N. (1999). "A Fast, Compact Approximation of the
/// Exponential Function." *Neural Computation*, 11(4), 853-862.
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    // Clamp to avoid overflow (> 88) and flush-to-zero (< -87)
    let x = x.clamp(-87.0, 88.0);
    // Schraudolph's approximation: bit-cast integer to float
    // 12102203.0 ~= 2^23 / ln(2);  1065353216 = 127 << 23 (exponent bias)
    let a = (12_102_203.0_f32.mul_add(x, 1_065_353_216.0_f32)) as i32;
    f32::from_bits(a as u32)
}

/// Fast softmax using Schraudolph's exponential approximation.
///
/// Compared to `tensor_softmax`:
/// - ~12x faster (avoids transcendental `exp()` calls)
/// - ~1.5% mean relative error on output probabilities
/// - Numerically stable: subtracts row-max before exponentiation
///
/// Use this when throughput matters more than bit-exact accuracy, e.g. during
/// speculative decoding, beam search ranking, or attention score normalisation.
///
/// # Panics
/// Panics if `a` has zero dimensions (`ndim == 0`).
#[inline]
pub fn tensor_softmax_fast(a: &Tensor, out: &mut Tensor) {
    assert!(a.ndim > 0, "tensor_softmax_fast: input tensor has ndim=0");
    debug_assert_eq!(a.shape(), out.shape());

    let last_dim = a.shape[a.ndim - 1];
    let outer = a.data.len() / last_dim;

    for o in 0..outer {
        let start = o * last_dim;
        let end = start + last_dim;
        let slice = &a.data[start..end];

        // Find max for numerical stability
        let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // fast_exp and accumulate
        let mut sum = 0.0f32;
        for i in start..end {
            let e = fast_exp(a.data[i] - max_val);
            out.data[i] = e;
            sum += e;
        }

        // Normalise (guard against degenerate all-(-inf) rows)
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for i in start..end {
                out.data[i] *= inv_sum;
            }
        } else {
            let uniform = 1.0 / last_dim as f32;
            for i in start..end {
                out.data[i] = uniform;
            }
        }
    }
}

// ============================================================================
// Activation Functions: GELU and SiLU (Swish)
// ============================================================================

/// GELU activation (fast approximation): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + `0.044_715` * x^3)))
///
/// Uses the tanh-based approximation: tanh(x) = (e^2x - 1)/(e^2x + 1)
/// DPS pattern — writes to pre-allocated output. ZERO ALLOCATION.
#[inline]
pub fn tensor_gelu(a: &Tensor, out: &mut Tensor) {
    debug_assert_eq!(a.shape[..a.ndim], out.shape[..out.ndim]);
    let len = a.data.len();
    let sqrt_2_over_pi: f32 = 0.797_884_6; // sqrt(2/pi)
    for i in 0..len {
        let x = a.data[i];
        let inner = sqrt_2_over_pi * (0.044_715 * x * x).mul_add(x, x);
        // tanh(x) = (e^2x - 1)/(e^2x + 1)
        let exp2x = (2.0 * inner).exp();
        let tanh_val = (exp2x - 1.0) / (exp2x + 1.0);
        out.data[i] = 0.5 * x * (1.0 + tanh_val);
    }
}

/// GELU in-place
#[inline]
pub fn tensor_gelu_inplace(a: &mut Tensor) {
    let sqrt_2_over_pi: f32 = 0.797_884_6;
    for i in 0..a.data.len() {
        let x = a.data[i];
        let inner = sqrt_2_over_pi * (0.044_715 * x * x).mul_add(x, x);
        let exp2x = (2.0 * inner).exp();
        let tanh_val = (exp2x - 1.0) / (exp2x + 1.0);
        a.data[i] = 0.5 * x * (1.0 + tanh_val);
    }
}

/// `SiLU` (Swish) activation: x * sigmoid(x) = x / (1 + exp(-x))
///
/// DPS pattern — writes to pre-allocated output. ZERO ALLOCATION.
#[inline]
pub fn tensor_silu(a: &Tensor, out: &mut Tensor) {
    debug_assert_eq!(a.shape[..a.ndim], out.shape[..out.ndim]);
    for i in 0..a.data.len() {
        let x = a.data[i];
        out.data[i] = x / (1.0 + (-x).exp());
    }
}

/// `SiLU` in-place
#[inline]
pub fn tensor_silu_inplace(a: &mut Tensor) {
    for i in 0..a.data.len() {
        let x = a.data[i];
        a.data[i] = x / (1.0 + (-x).exp());
    }
}

// ============================================================================
// Normalization: RMSNorm and LayerNorm
// ============================================================================

/// `RMSNorm`: x / sqrt(mean(x^2) + eps) * weight
///
/// DPS style — weight can be None for unweighted normalization.
/// ZERO ALLOCATION.
#[inline]
pub fn tensor_rms_norm(a: &Tensor, weight: Option<&Tensor>, eps: f32, out: &mut Tensor) {
    debug_assert_eq!(a.shape[..a.ndim], out.shape[..out.ndim]);
    let len = a.data.len();
    // Compute RMS
    let mut sum_sq: f32 = 0.0;
    for i in 0..len {
        sum_sq += a.data[i] * a.data[i];
    }
    let rms = (sum_sq / len as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;
    // Normalize and optionally scale
    match weight {
        Some(w) => {
            debug_assert_eq!(w.data.len(), len);
            for i in 0..len {
                out.data[i] = a.data[i] * inv_rms * w.data[i];
            }
        }
        None => {
            for i in 0..len {
                out.data[i] = a.data[i] * inv_rms;
            }
        }
    }
}

/// `LayerNorm`: (x - mean) / sqrt(var + eps) * weight + bias
///
/// DPS style — weight and bias are both optional.
/// ZERO ALLOCATION.
#[inline]
#[allow(clippy::option_if_let_else)]
pub fn tensor_layer_norm(
    a: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
    out: &mut Tensor,
) {
    debug_assert_eq!(a.shape[..a.ndim], out.shape[..out.ndim]);
    let len = a.data.len();
    // Compute mean
    let mut sum: f32 = 0.0;
    for i in 0..len {
        sum += a.data[i];
    }
    let mean = sum / len as f32;
    // Compute variance
    let mut var: f32 = 0.0;
    for i in 0..len {
        let d = a.data[i] - mean;
        var += d * d;
    }
    var /= len as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    // Normalize, scale, and shift
    for i in 0..len {
        let normalized = (a.data[i] - mean) * inv_std;
        let scaled = match weight {
            Some(w) => normalized * w.data[i],
            None => normalized,
        };
        out.data[i] = match bias {
            Some(b) => scaled + b.data[i],
            None => scaled,
        };
    }
}

/// SIMD 8-wide f32 add: out = a + b
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline]
fn simd_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use core::arch::x86_64::{_mm256_add_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    let n = a.len();
    let chunks = n / 8;
    // SAFETY: AVX2 is required by the "simd" feature gate and the target_arch guard.
    // Each load/store accesses exactly 8 consecutive f32 values within respective slice bounds:
    // offset + 8 <= chunks * 8 <= n = a.len() = b.len() = out.len().
    unsafe {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_add_ps(va, vb));
        }
    }
    let tail = chunks * 8;
    for i in tail..n {
        out[i] = a[i] + b[i];
    }
}

/// SIMD 8-wide f32 sub: out = a - b
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline]
fn simd_sub_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use core::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps, _mm256_sub_ps};
    let n = a.len();
    let chunks = n / 8;
    // SAFETY: AVX2 is required by the "simd" feature gate and the target_arch guard.
    // Each load/store accesses exactly 8 consecutive f32 values within respective slice bounds:
    // offset + 8 <= chunks * 8 <= n = a.len() = b.len() = out.len().
    unsafe {
        for i in 0..chunks {
            let offset = i * 8;
            let va = _mm256_loadu_ps(a.as_ptr().add(offset));
            let vb = _mm256_loadu_ps(b.as_ptr().add(offset));
            _mm256_storeu_ps(out.as_mut_ptr().add(offset), _mm256_sub_ps(va, vb));
        }
    }
    let tail = chunks * 8;
    for i in tail..n {
        out[i] = a[i] - b[i];
    }
}

// ============================================================================
// Quantized Tensor (Charred Edition) - Also Zero-Allocation
// ============================================================================

/// Quantized INT8 tensor for activations
///
/// For when you need owned data (e.g., model loading).
/// Hot path inference should use borrowed slices.
#[derive(Clone, Debug)]
pub struct QuantizedTensor {
    /// INT8 data
    data: Vec<i8>,
    /// Scale factor for dequantization
    scale: f32,
    /// Shape of tensor
    shape: [usize; MAX_DIMS],
    /// Number of dimensions
    ndim: usize,
}

impl QuantizedTensor {
    /// Quantize f32 slice to INT8
    ///
    /// # Panics
    /// Panics if `input.len()` does not equal the product of all shape dimensions.
    pub fn from_f32_slice(input: &[f32], shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(input.len(), total);

        // Find max absolute value
        let max_abs = input
            .iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(1e-8);

        let scale = max_abs / 127.0;

        // Quantize
        let quantized: Vec<i8> = input
            .iter()
            .map(|&x| (x / scale).round().clamp(-127.0, 127.0) as i8)
            .collect();

        let mut shape_arr = [0usize; MAX_DIMS];
        shape_arr[..shape.len()].copy_from_slice(shape);

        Self {
            data: quantized,
            scale,
            shape: shape_arr,
            ndim: shape.len(),
        }
    }

    /// Create from raw INT8 data
    ///
    /// # Panics
    /// Panics if `data.len()` does not equal the product of all shape dimensions.
    #[must_use]
    pub fn from_i8(data: Vec<i8>, scale: f32, shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(data.len(), total);

        let mut shape_arr = [0usize; MAX_DIMS];
        shape_arr[..shape.len()].copy_from_slice(shape);

        Self {
            data,
            scale,
            shape: shape_arr,
            ndim: shape.len(),
        }
    }

    /// Dequantize to f32, writing to pre-allocated output
    #[inline]
    pub fn dequantize_to(&self, out: &mut [f32]) {
        debug_assert_eq!(self.data.len(), out.len());
        let scale = self.scale;
        for (o, &q) in out.iter_mut().zip(self.data.iter()) {
            *o = q as f32 * scale;
        }
    }

    /// Get raw INT8 data
    #[inline]
    #[must_use]
    pub fn data_i8(&self) -> &[i8] {
        &self.data
    }

    /// Get scale factor
    #[inline]
    #[must_use]
    pub const fn scale(&self) -> f32 {
        self.scale
    }

    /// Get shape slice
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Total number of elements
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ============================================================================
// Owned Tensor (for model loading, NOT for hot path)
// ============================================================================

/// Owned tensor - for model loading and initialization ONLY
///
/// **WARNING**: Do NOT use in hot path. Use `Tensor<'a>` instead.
#[derive(Clone, Debug)]
pub struct OwnedTensor {
    data: Vec<f32>,
    shape: [usize; MAX_DIMS],
    ndim: usize,
    strides: [usize; MAX_DIMS],
}

impl OwnedTensor {
    /// Create from slice (allocates!)
    ///
    /// # Panics
    /// Panics if `shape.len()` exceeds `MAX_DIMS` or if `data.len()` does not equal the product
    /// of all shape dimensions.
    #[must_use]
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        assert!(shape.len() <= MAX_DIMS);
        let total: usize = shape.iter().product();
        assert_eq!(data.len(), total);

        let mut shape_arr = [0usize; MAX_DIMS];
        let mut strides_arr = [0usize; MAX_DIMS];
        shape_arr[..shape.len()].copy_from_slice(shape);

        if !shape.is_empty() {
            strides_arr[shape.len() - 1] = 1;
            for i in (0..shape.len() - 1).rev() {
                strides_arr[i] = strides_arr[i + 1] * shape[i + 1];
            }
        }

        Self {
            data: data.to_vec(),
            shape: shape_arr,
            ndim: shape.len(),
            strides: strides_arr,
        }
    }

    /// Get shape
    #[inline]
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Get data
    #[inline]
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable data
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Total elements
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element at index
    #[inline]
    #[must_use]
    pub fn get(&self, indices: &[usize]) -> f32 {
        let mut idx = 0;
        for (i, &dim_idx) in indices.iter().enumerate() {
            idx += dim_idx * self.strides[i];
        }
        self.data[idx]
    }

    /// Copy to Arena-backed Tensor
    pub fn to_tensor<'a>(&self, arena: &'a mut Arena) -> Option<Tensor<'a>> {
        let data = arena.alloc::<f32>(self.data.len())?;
        data.copy_from_slice(&self.data);
        Some(Tensor::from_arena(data, self.shape()))
    }
}

#[cfg(test)]
#[allow(clippy::wildcard_imports)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_from_arena() {
        let mut arena = Arena::new(4096);
        let data = arena.alloc::<f32>(6).unwrap();
        data.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let t = Tensor::from_arena(data, &[2, 3]);
        assert_eq!(t.shape(), &[2, 3], "shape should be [2, 3]");
        assert_eq!(t.ndim(), 2, "ndim should be 2");
        assert_eq!(t.len(), 6, "total elements should be 6");
    }

    #[test]
    fn test_tensor_zeros() {
        let mut arena = Arena::new(4096);
        let t = Tensor::zeros(&mut arena, &[3, 4]).unwrap();

        assert_eq!(t.shape(), &[3, 4], "shape should be [3, 4]");
        assert!(
            t.data().iter().all(|&x| x == 0.0),
            "all elements should be zero"
        );
    }

    #[test]
    fn test_tensor_indexing() {
        let mut arena = Arena::new(4096);
        let data = arena.alloc::<f32>(6).unwrap();
        data.copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let t = Tensor::from_arena(data, &[2, 3]);
        assert_eq!(t.get(&[0, 0]), 1.0, "t[0,0] should be 1.0");
        assert_eq!(t.get(&[0, 2]), 3.0, "t[0,2] should be 3.0");
        assert_eq!(t.get(&[1, 0]), 4.0, "t[1,0] should be 4.0");
        assert_eq!(t.get(&[1, 2]), 6.0, "t[1,2] should be 6.0");
    }

    // DPS tests using stack arrays (avoids Arena borrow conflicts)

    #[test]
    fn test_dps_add() {
        let mut a_data = [1.0f32, 2.0, 3.0];
        let mut b_data = [4.0f32, 5.0, 6.0];
        let mut out_data = [0.0f32; 3];

        let a = Tensor::from_arena(&mut a_data, &[3]);
        let b = Tensor::from_arena(&mut b_data, &[3]);
        let mut out = Tensor::from_arena(&mut out_data, &[3]);

        tensor_add(&a, &b, &mut out);
        assert_eq!(out.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_dps_sub() {
        let mut a_data = [4.0f32, 5.0, 6.0];
        let mut b_data = [1.0f32, 2.0, 3.0];
        let mut out_data = [0.0f32; 3];

        let a = Tensor::from_arena(&mut a_data, &[3]);
        let b = Tensor::from_arena(&mut b_data, &[3]);
        let mut out = Tensor::from_arena(&mut out_data, &[3]);

        tensor_sub(&a, &b, &mut out);
        assert_eq!(out.data(), &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_dps_scale() {
        let mut a_data = [1.0f32, 2.0, 3.0];
        let mut out_data = [0.0f32; 3];

        let a = Tensor::from_arena(&mut a_data, &[3]);
        let mut out = Tensor::from_arena(&mut out_data, &[3]);

        tensor_scale(&a, 2.0, &mut out);
        assert_eq!(out.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_dps_relu() {
        let mut a_data = [-1.0f32, 0.0, 1.0, -2.0, 3.0];
        let mut out_data = [0.0f32; 5];

        let a = Tensor::from_arena(&mut a_data, &[5]);
        let mut out = Tensor::from_arena(&mut out_data, &[5]);

        tensor_relu(&a, &mut out);
        assert_eq!(out.data(), &[0.0, 0.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_dps_softmax() {
        let mut a_data = [1.0f32, 2.0, 3.0];
        let mut out_data = [0.0f32; 3];

        let a = Tensor::from_arena(&mut a_data, &[3]);
        let mut out = Tensor::from_arena(&mut out_data, &[3]);

        tensor_softmax(&a, &mut out);

        // Sum should be 1.0
        let sum: f32 = out.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Values should be increasing
        assert!(out.get(&[0]) < out.get(&[1]));
        assert!(out.get(&[1]) < out.get(&[2]));
    }

    #[test]
    fn test_owned_tensor() {
        let owned = OwnedTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(owned.shape(), &[2, 2]);
        assert_eq!(owned.get(&[0, 1]), 2.0);
        assert_eq!(owned.get(&[1, 0]), 3.0);
    }

    #[test]
    fn test_owned_to_tensor() {
        let owned = OwnedTensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let mut arena = Arena::new(4096);

        let t = owned.to_tensor(&mut arena).unwrap();
        assert_eq!(t.data(), owned.data());
    }

    #[test]
    fn test_quantized_tensor() {
        let input = [1.0f32, -1.0, 0.5, -0.5];
        let qt = QuantizedTensor::from_f32_slice(&input, &[4]);

        assert_eq!(qt.shape(), &[4], "quantized shape should be [4]");
        assert!(qt.scale() > 0.0, "scale should be positive");

        // Dequantize
        let mut output = [0.0f32; 4];
        qt.dequantize_to(&mut output);

        // Check approximate equality
        for (idx, (o, &i)) in output.iter().zip(input.iter()).enumerate() {
            assert!(
                (o - i).abs() < 0.02,
                "dequantized[{}] = {}, expected ~{}",
                idx,
                o,
                i
            );
        }
    }

    #[test]
    fn test_zero_allocation_workflow() {
        // DPS workflow using stack arrays (no Arena borrow conflicts)
        let mut x_data = [1.0f32, -2.0, 3.0, -4.0];
        let mut relu_out = [0.0f32; 4];
        let mut scale_out = [0.0f32; 4];

        let x = Tensor::from_arena(&mut x_data, &[4]);
        let mut relu = Tensor::from_arena(&mut relu_out, &[4]);
        let mut scaled = Tensor::from_arena(&mut scale_out, &[4]);

        // ReLU: [1, 0, 3, 0]
        tensor_relu(&x, &mut relu);
        assert_eq!(relu.data(), &[1.0, 0.0, 3.0, 0.0]);

        // Scale: [2, 0, 6, 0]
        tensor_scale(&relu, 2.0, &mut scaled);
        assert_eq!(scaled.data(), &[2.0, 0.0, 6.0, 0.0]);
    }

    // ---- GELU tests ----

    #[test]
    fn test_gelu_at_zero() {
        // GELU(0) = 0.5 * 0 * (1 + tanh(0)) = 0
        let mut a_data = [0.0f32];
        let mut out_data = [0.0f32];
        let a = Tensor::from_arena(&mut a_data, &[1]);
        let mut out = Tensor::from_arena(&mut out_data, &[1]);
        tensor_gelu(&a, &mut out);
        assert!(
            out.data()[0].abs() < 1e-5,
            "GELU(0) should be 0, got {}",
            out.data()[0]
        );
    }

    #[test]
    fn test_gelu_positive_large() {
        // For large positive x, GELU(x) ≈ x
        let mut a_data = [10.0f32];
        let mut out_data = [0.0f32];
        let a = Tensor::from_arena(&mut a_data, &[1]);
        let mut out = Tensor::from_arena(&mut out_data, &[1]);
        tensor_gelu(&a, &mut out);
        assert!(
            (out.data()[0] - 10.0).abs() < 1e-3,
            "GELU(10) ≈ 10, got {}",
            out.data()[0]
        );
    }

    #[test]
    fn test_gelu_negative_large() {
        // For large negative x, GELU(x) ≈ 0
        let mut a_data = [-10.0f32];
        let mut out_data = [0.0f32];
        let a = Tensor::from_arena(&mut a_data, &[1]);
        let mut out = Tensor::from_arena(&mut out_data, &[1]);
        tensor_gelu(&a, &mut out);
        assert!(
            out.data()[0].abs() < 1e-3,
            "GELU(-10) ≈ 0, got {}",
            out.data()[0]
        );
    }

    #[test]
    fn test_gelu_inplace_matches_dps() {
        let mut a_data = [-1.0f32, 0.0, 0.5, 1.0, 2.0];
        let mut out_data = [0.0f32; 5];

        let mut a_copy = a_data;

        let a = Tensor::from_arena(&mut a_data, &[5]);
        let mut out = Tensor::from_arena(&mut out_data, &[5]);
        tensor_gelu(&a, &mut out);

        let mut a2 = Tensor::from_arena(&mut a_copy, &[5]);
        tensor_gelu_inplace(&mut a2);

        for i in 0..5 {
            assert!(
                (out.data()[i] - a2.data()[i]).abs() < 1e-6,
                "GELU DPS vs inplace mismatch at {}: {} vs {}",
                i,
                out.data()[i],
                a2.data()[i]
            );
        }
    }

    // ---- SiLU tests ----

    #[test]
    fn test_silu_at_zero() {
        // SiLU(0) = 0 / (1 + exp(0)) = 0 / 2 = 0
        let mut a_data = [0.0f32];
        let mut out_data = [0.0f32];
        let a = Tensor::from_arena(&mut a_data, &[1]);
        let mut out = Tensor::from_arena(&mut out_data, &[1]);
        tensor_silu(&a, &mut out);
        assert!(
            out.data()[0].abs() < 1e-6,
            "SiLU(0) should be 0, got {}",
            out.data()[0]
        );
    }

    #[test]
    fn test_silu_positive_large() {
        // For large positive x, SiLU(x) ≈ x
        let mut a_data = [10.0f32];
        let mut out_data = [0.0f32];
        let a = Tensor::from_arena(&mut a_data, &[1]);
        let mut out = Tensor::from_arena(&mut out_data, &[1]);
        tensor_silu(&a, &mut out);
        assert!(
            (out.data()[0] - 10.0).abs() < 1e-3,
            "SiLU(10) ≈ 10, got {}",
            out.data()[0]
        );
    }

    #[test]
    fn test_silu_at_one() {
        // SiLU(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        let expected = 1.0f32 / (1.0 + (-1.0f32).exp());
        let mut a_data = [1.0f32];
        let mut out_data = [0.0f32];
        let a = Tensor::from_arena(&mut a_data, &[1]);
        let mut out = Tensor::from_arena(&mut out_data, &[1]);
        tensor_silu(&a, &mut out);
        assert!(
            (out.data()[0] - expected).abs() < 1e-6,
            "SiLU(1) = {}, expected {}",
            out.data()[0],
            expected
        );
    }

    #[test]
    fn test_silu_inplace_matches_dps() {
        let mut a_data = [-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let mut out_data = [0.0f32; 5];
        let mut a_copy = a_data;

        let a = Tensor::from_arena(&mut a_data, &[5]);
        let mut out = Tensor::from_arena(&mut out_data, &[5]);
        tensor_silu(&a, &mut out);

        let mut a2 = Tensor::from_arena(&mut a_copy, &[5]);
        tensor_silu_inplace(&mut a2);

        for i in 0..5 {
            assert!(
                (out.data()[i] - a2.data()[i]).abs() < 1e-6,
                "SiLU DPS vs inplace mismatch at {}: {} vs {}",
                i,
                out.data()[i],
                a2.data()[i]
            );
        }
    }

    // ---- RMSNorm tests ----

    #[test]
    fn test_rms_norm_unit_vector() {
        // Input [1, 0, 0]: RMS = 1/sqrt(3), normalized = [sqrt(3), 0, 0]
        let mut a_data = [1.0f32, 0.0, 0.0];
        let mut out_data = [0.0f32; 3];
        let a = Tensor::from_arena(&mut a_data, &[3]);
        let mut out = Tensor::from_arena(&mut out_data, &[3]);
        tensor_rms_norm(&a, None, 1e-8, &mut out);
        // sum_sq = 1, rms = sqrt(1/3 + eps) ≈ 1/sqrt(3)
        // out[0] = 1 * sqrt(3) ≈ 1.732
        let expected = 1.0 / (1.0f32 / 3.0 + 1e-8f32).sqrt();
        assert!(
            (out.data()[0] - expected).abs() < 1e-4,
            "RMSNorm[0] = {}, expected {}",
            out.data()[0],
            expected
        );
        assert!(out.data()[1].abs() < 1e-6, "RMSNorm[1] should be 0");
        assert!(out.data()[2].abs() < 1e-6, "RMSNorm[2] should be 0");
    }

    #[test]
    fn test_rms_norm_with_weight() {
        // Input all-ones: rms = 1.0, out = 1.0 * weight
        let mut a_data = [1.0f32, 1.0, 1.0, 1.0];
        let mut w_data = [2.0f32, 0.5, 3.0, 1.0];
        let mut out_data = [0.0f32; 4];

        let a = Tensor::from_arena(&mut a_data, &[4]);
        let w = Tensor::from_arena(&mut w_data, &[4]);
        let mut out = Tensor::from_arena(&mut out_data, &[4]);

        tensor_rms_norm(&a, Some(&w), 0.0, &mut out);

        // rms = sqrt(4/4) = 1.0, inv_rms = 1.0 → out = weight
        let expected = [2.0f32, 0.5, 3.0, 1.0];
        for (i, (&o, &e)) in out.data().iter().zip(expected.iter()).enumerate() {
            assert!(
                (o - e).abs() < 1e-5,
                "RMSNorm with weight[{}] = {}, expected {}",
                i,
                o,
                e
            );
        }
    }

    #[test]
    fn test_rms_norm_output_unit_norm() {
        // After RMSNorm (no weight), the output should have RMS == 1
        let mut a_data = [3.0f32, -1.0, 2.0, 4.0];
        let mut out_data = [0.0f32; 4];
        let a = Tensor::from_arena(&mut a_data, &[4]);
        let mut out = Tensor::from_arena(&mut out_data, &[4]);
        tensor_rms_norm(&a, None, 1e-8, &mut out);

        let rms_out: f32 = (out.data().iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!(
            (rms_out - 1.0).abs() < 1e-4,
            "Output RMS should be 1.0, got {}",
            rms_out
        );
    }

    // ---- LayerNorm tests ----

    #[test]
    fn test_layer_norm_zero_mean() {
        // LayerNorm should produce zero-mean output
        let mut a_data = [1.0f32, 2.0, 3.0, 4.0];
        let mut out_data = [0.0f32; 4];
        let a = Tensor::from_arena(&mut a_data, &[4]);
        let mut out = Tensor::from_arena(&mut out_data, &[4]);
        tensor_layer_norm(&a, None, None, 1e-8, &mut out);

        let mean: f32 = out.data().iter().sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-5,
            "LayerNorm output mean should be 0, got {}",
            mean
        );
    }

    #[test]
    fn test_layer_norm_unit_variance() {
        // LayerNorm should produce unit-variance output (no weight)
        let mut a_data = [1.0f32, 2.0, 3.0, 4.0];
        let mut out_data = [0.0f32; 4];
        let a = Tensor::from_arena(&mut a_data, &[4]);
        let mut out = Tensor::from_arena(&mut out_data, &[4]);
        tensor_layer_norm(&a, None, None, 1e-8, &mut out);

        let mean: f32 = out.data().iter().sum::<f32>() / 4.0;
        let var: f32 = out
            .data()
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f32>()
            / 4.0;
        assert!(
            (var - 1.0).abs() < 1e-4,
            "LayerNorm output variance should be 1, got {}",
            var
        );
    }

    #[test]
    fn test_layer_norm_with_weight_and_bias() {
        // LayerNorm with weight=2, bias=1: output = 2 * normalized + 1
        let mut a_data = [2.0f32, 4.0, 6.0, 8.0];
        let mut w_data = [2.0f32; 4];
        let mut b_data = [1.0f32; 4];
        let mut out_data = [0.0f32; 4];

        let a = Tensor::from_arena(&mut a_data, &[4]);
        let w = Tensor::from_arena(&mut w_data, &[4]);
        let b = Tensor::from_arena(&mut b_data, &[4]);
        let mut out = Tensor::from_arena(&mut out_data, &[4]);

        tensor_layer_norm(&a, Some(&w), Some(&b), 1e-8, &mut out);

        // mean = 5, var = 5, std ≈ 2.236
        // normalized = [-1.342, -0.447, 0.447, 1.342]
        // output = 2*normalized + 1
        let mean_out: f32 = out.data().iter().sum::<f32>() / 4.0;
        // With weight=2 and bias=1 applied uniformly, mean_out ≈ 2*0 + 1 = 1
        assert!(
            (mean_out - 1.0).abs() < 1e-4,
            "LayerNorm with bias=1 mean should be 1, got {}",
            mean_out
        );
    }

    // ============================================================================
    // Hardware-native SIMD enhancement tests
    // ============================================================================

    // ---- fast_exp accuracy ----

    /// Verify Schraudolph's fast_exp has < 8% relative error vs f32::exp
    /// across the stable domain [-10, 10].
    ///
    /// The Schraudolph approximation achieves ~1.5% *mean* relative error
    /// but can reach ~6% at extremes such as -10. The 8% bound is chosen to
    /// cover worst-case values while still catching a broken implementation.
    /// For the core softmax use-case (post-max-subtraction, typically < ±3),
    /// error is well below 2%.
    #[test]
    fn test_fast_exp_accuracy() {
        // Sample points spanning typical softmax input range
        let test_points: &[f32] = &[-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0];
        for &x in test_points {
            let exact = x.exp();
            let approx = fast_exp(x);
            let rel_err = ((approx - exact) / exact).abs();
            assert!(
                rel_err < 0.08,
                "fast_exp({}) = {} but f32::exp({}) = {} — relative error {:.4} >= 8%",
                x,
                approx,
                x,
                exact,
                rel_err
            );
        }
        // Verify monotonicity: fast_exp must be strictly increasing.
        // This is the property that matters most for softmax correctness.
        let mono_points: &[f32] = &[-5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0];
        for w in mono_points.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            assert!(
                fast_exp(lo) < fast_exp(hi),
                "fast_exp must be monotone: fast_exp({}) >= fast_exp({})",
                lo,
                hi
            );
        }
    }

    /// Verify fast_exp clamps without panic for extreme inputs.
    #[test]
    fn test_fast_exp_clamping() {
        // Should not overflow to +inf or panic
        let v = fast_exp(1000.0);
        assert!(
            v.is_finite(),
            "fast_exp(1000) should be finite (clamped), got {}",
            v
        );

        // Should not flush to zero or panic
        let v = fast_exp(-1000.0);
        assert!(v >= 0.0, "fast_exp(-1000) should be >= 0, got {}", v);
    }

    // ---- tensor_softmax_fast ----

    /// Output of tensor_softmax_fast must sum to 1.0 (within tolerance).
    #[test]
    fn test_softmax_fast_sums_to_one() {
        let mut a_data = [1.0f32, 2.0, 3.0, 4.0];
        let mut out_data = [0.0f32; 4];

        let a = Tensor::from_arena(&mut a_data, &[4]);
        let mut out = Tensor::from_arena(&mut out_data, &[4]);

        tensor_softmax_fast(&a, &mut out);

        let sum: f32 = out.data().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "tensor_softmax_fast sum should be ~1.0, got {}",
            sum
        );
    }

    /// tensor_softmax_fast output must be monotonically ordered for sorted inputs.
    #[test]
    fn test_softmax_fast_ordering() {
        let mut a_data = [1.0f32, 2.0, 3.0];
        let mut out_data = [0.0f32; 3];

        let a = Tensor::from_arena(&mut a_data, &[3]);
        let mut out = Tensor::from_arena(&mut out_data, &[3]);

        tensor_softmax_fast(&a, &mut out);

        assert!(
            out.data()[0] < out.data()[1],
            "softmax_fast: out[0] should be < out[1]"
        );
        assert!(
            out.data()[1] < out.data()[2],
            "softmax_fast: out[1] should be < out[2]"
        );
    }

    /// tensor_softmax_fast must be within 8% relative error of exact tensor_softmax.
    ///
    /// After max-subtraction the fast_exp inputs are in (-range, 0] where range
    /// is the spread of the logits. For typical logit spreads (< 4) error stays
    /// under 2%; for larger spreads it can reach ~6%.  The 8% bound covers
    /// worst-case while still detecting a broken implementation.
    #[test]
    fn test_softmax_fast_vs_exact() {
        let values = [1.0f32, -0.5, 2.5, 0.0, -1.5];

        let mut a_data = values;
        let mut a_data2 = values;
        let mut exact_out = [0.0f32; 5];
        let mut fast_out = [0.0f32; 5];

        let a_exact = Tensor::from_arena(&mut a_data, &[5]);
        let mut out_exact = Tensor::from_arena(&mut exact_out, &[5]);
        tensor_softmax(&a_exact, &mut out_exact);

        let a_fast = Tensor::from_arena(&mut a_data2, &[5]);
        let mut out_fast = Tensor::from_arena(&mut fast_out, &[5]);
        tensor_softmax_fast(&a_fast, &mut out_fast);

        for i in 0..5 {
            let exact = out_exact.data()[i];
            let fast = out_fast.data()[i];
            let rel_err = ((fast - exact) / exact.max(1e-8)).abs();
            assert!(
                rel_err < 0.08,
                "softmax_fast[{}]: exact={:.6} fast={:.6} rel_err={:.4} >= 8%",
                i,
                exact,
                fast,
                rel_err
            );
        }
    }

    // ---- tensor_sum AVX2 vs scalar ----

    /// tensor_sum must equal the scalar sum for a multi-element tensor.
    #[test]
    fn test_tensor_sum_correctness() {
        // Use length 17 to exercise both the 8-wide chunks and the tail
        let a_data: Vec<f32> = (1..=17).map(|i| i as f32).collect();
        let mut a_arr = [0.0f32; 17];
        a_arr.copy_from_slice(&a_data);

        // Compute reference scalar sum
        let expected: f32 = a_data.iter().sum();

        let a = Tensor::from_arena(&mut a_arr, &[17]);
        let result = tensor_sum(&a);

        assert!(
            (result - expected).abs() < 1e-4,
            "tensor_sum: expected {}, got {}",
            expected,
            result
        );
    }

    // ---- tensor_max AVX2 vs scalar ----

    /// tensor_max must return the true maximum across 8-wide chunks and tail.
    #[test]
    fn test_tensor_max_correctness() {
        // 17 elements: covers 2 full AVX2 chunks (16 elements) + 1 tail
        let mut a_data = [
            1.0f32, -5.0, 3.0, 2.0, 0.5, -1.0, 7.0, 4.0, // chunk 0
            2.5, 6.0, -2.0, 1.5, 3.5, -0.5, 5.5, 0.0, // chunk 1
            9.0, // tail
        ];

        let expected = a_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let a = Tensor::from_arena(&mut a_data, &[17]);
        let result = tensor_max(&a);

        assert!(
            (result - expected).abs() < 1e-6,
            "tensor_max: expected {}, got {}",
            expected,
            result
        );
    }

    /// tensor_max returns NEG_INFINITY for an empty tensor.
    #[test]
    fn test_tensor_max_empty() {
        let mut a_data: [f32; 0] = [];
        let a = Tensor::from_arena(&mut a_data, &[0]);
        let result = tensor_max(&a);
        assert!(
            result == f32::NEG_INFINITY,
            "tensor_max on empty should be NEG_INFINITY, got {}",
            result
        );
    }

    // ---- tensor_min AVX2 vs scalar ----

    /// tensor_min must return the true minimum across 8-wide chunks and tail.
    #[test]
    fn test_tensor_min_correctness() {
        // 17 elements: covers 2 full AVX2 chunks + 1 tail
        let mut a_data = [
            1.0f32, -5.0, 3.0, 2.0, 0.5, -1.0, 7.0, 4.0, // chunk 0
            2.5, 6.0, -2.0, 1.5, 3.5, -0.5, 5.5, 0.0,  // chunk 1
            -9.0, // tail
        ];

        let expected = a_data.iter().cloned().fold(f32::INFINITY, f32::min);
        let a = Tensor::from_arena(&mut a_data, &[17]);
        let result = tensor_min(&a);

        assert!(
            (result - expected).abs() < 1e-6,
            "tensor_min: expected {}, got {}",
            expected,
            result
        );
    }

    /// tensor_min returns INFINITY for an empty tensor.
    #[test]
    fn test_tensor_min_empty() {
        let mut a_data: [f32; 0] = [];
        let a = Tensor::from_arena(&mut a_data, &[0]);
        let result = tensor_min(&a);
        assert!(
            result == f32::INFINITY,
            "tensor_min on empty should be INFINITY, got {}",
            result
        );
    }
}
