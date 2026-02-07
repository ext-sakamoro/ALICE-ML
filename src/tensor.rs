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
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Get number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.ndim
    }

    /// Total number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get raw data slice (immutable)
    #[inline]
    pub fn data(&self) -> &[f32] {
        self.data
    }

    /// Get mutable data slice
    #[inline]
    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data
    }

    /// Get element at flat index
    #[inline]
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
    {
        simd_add_f32(a.data, b.data, out.data);
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    {
        for ((o, &av), &bv) in out.data.iter_mut().zip(a.data.iter()).zip(b.data.iter()) {
            *o = av + bv;
        }
    }
}

/// Element-wise subtraction: out = a - b
#[inline]
pub fn tensor_sub(a: &Tensor, b: &Tensor, out: &mut Tensor) {
    debug_assert_eq!(a.shape(), b.shape());
    debug_assert_eq!(a.shape(), out.shape());

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        simd_sub_f32(a.data, b.data, out.data);
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    {
        for ((o, &av), &bv) in out.data.iter_mut().zip(a.data.iter()).zip(b.data.iter()) {
            *o = av - bv;
        }
    }
}

/// Scalar multiplication: out = a * scalar
#[inline]
pub fn tensor_scale(a: &Tensor, scalar: f32, out: &mut Tensor) {
    debug_assert_eq!(a.shape(), out.shape());

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        use core::arch::x86_64::*;
        let n = a.data.len();
        let chunks = n / 8;
        let remainder = n % 8;
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
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    {
        for (o, &av) in out.data.iter_mut().zip(a.data.iter()) {
            *o = av * scalar;
        }
    }
}

/// ReLU activation: out = max(a, 0) — branchless
#[inline]
pub fn tensor_relu(a: &Tensor, out: &mut Tensor) {
    debug_assert_eq!(a.shape(), out.shape());

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        use core::arch::x86_64::*;
        let n = a.data.len();
        let chunks = n / 8;
        let remainder = n % 8;
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
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    {
        for (o, &av) in out.data.iter_mut().zip(a.data.iter()) {
            *o = av.max(0.0); // branchless: compiles to maxss/maxps
        }
    }
}

/// ReLU in-place — branchless
#[inline]
pub fn tensor_relu_inplace(a: &mut Tensor) {
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        use core::arch::x86_64::*;
        let n = a.data.len();
        let chunks = n / 8;
        let remainder = n % 8;
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
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "simd")))]
    {
        for x in a.data.iter_mut() {
            *x = x.max(0.0); // branchless: compiles to maxss
        }
    }
}

/// Softmax over last dimension: out = softmax(a)
#[inline]
pub fn tensor_softmax(a: &Tensor, out: &mut Tensor) {
    debug_assert_eq!(a.shape(), out.shape());

    let last_dim = a.shape[a.ndim - 1];
    let batch_size = a.data.len() / last_dim;

    for b in 0..batch_size {
        let start = b * last_dim;
        let end = start + last_dim;
        let in_slice = &a.data[start..end];
        let out_slice = &mut out.data[start..end];

        // Find max for numerical stability
        let max_val = in_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

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
#[inline]
pub fn tensor_sum(a: &Tensor) -> f32 {
    a.data.iter().sum()
}

/// Mean of all elements
#[inline]
pub fn tensor_mean(a: &Tensor) -> f32 {
    tensor_sum(a) / a.data.len() as f32
}

/// Maximum element
#[inline]
pub fn tensor_max(a: &Tensor) -> f32 {
    a.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

/// Minimum element
#[inline]
pub fn tensor_min(a: &Tensor) -> f32 {
    a.data.iter().cloned().fold(f32::INFINITY, f32::min)
}

/// Copy data from source to destination
#[inline]
pub fn tensor_copy(src: &Tensor, dst: &mut Tensor) {
    debug_assert_eq!(src.len(), dst.len());
    dst.data.copy_from_slice(src.data);
}

/// SIMD 8-wide f32 add: out = a + b
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[inline]
fn simd_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    use core::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
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
    use core::arch::x86_64::*;
    let n = a.len();
    let chunks = n / 8;
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
    pub fn from_f32_slice(input: &[f32], shape: &[usize]) -> Self {
        let total: usize = shape.iter().product();
        assert_eq!(input.len(), total);

        // Find max absolute value
        let max_abs = input.iter()
            .map(|x| x.abs())
            .fold(0.0f32, f32::max)
            .max(1e-8);

        let scale = max_abs / 127.0;

        // Quantize
        let quantized: Vec<i8> = input.iter()
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
    pub fn data_i8(&self) -> &[i8] {
        &self.data
    }

    /// Get scale factor
    #[inline]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Get shape slice
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Total number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
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
    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Get data
    #[inline]
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
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get element at index
    #[inline]
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
        assert!(t.data().iter().all(|&x| x == 0.0), "all elements should be zero");
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
            assert!((o - i).abs() < 0.02, "dequantized[{}] = {}, expected ~{}", idx, o, i);
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
}
