//! BitLinear layer — drop-in replacement for nn.Linear using 1.58-bit ternary weights.
//!
//! Combines TernaryWeightKernel + optional bias + RMSNorm into a single forward() call.
//!
//! Author: Moroya Sakamoto

use crate::ops::{ternary_matvec_kernel, TernaryWeightKernel};

/// A BitLinear layer: ternary weights + optional bias + optional pre-norm.
///
/// This is the primary layer abstraction in ALICE-ML. It encapsulates a
/// `TernaryWeightKernel` (bit-parallel, SIMD-ready) with an optional bias
/// vector and optional RMSNorm pre-normalization, matching the BitNet b1.58
/// paper's design.
///
/// # Example
///
/// ```rust
/// use alice_ml::{BitLinear};
/// use alice_ml::ops::TernaryWeightKernel;
///
/// let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
/// let layer = BitLinear::new(kernel, None, false);
///
/// let input = [2.0f32, 3.0];
/// let mut output = [0.0f32; 2];
/// layer.forward(&input, &mut output);
/// // output[0] = 2*1 + 3*(-1) = -1
/// // output[1] = 2*0 + 3*1   =  3
/// assert!((output[0] - (-1.0)).abs() < 1e-5);
/// assert!((output[1] - 3.0).abs() < 1e-5);
/// ```
pub struct BitLinear {
    /// Ternary weight kernel (bitplane format for SIMD).
    pub weights: TernaryWeightKernel,
    /// Optional bias vector (f32, length = out_features).
    pub bias: Option<Vec<f32>>,
    /// Input features dimension.
    pub in_features: usize,
    /// Output features dimension.
    pub out_features: usize,
    /// Whether to apply RMSNorm before matmul.
    pub pre_norm: bool,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
}

impl BitLinear {
    /// Create a new BitLinear layer.
    ///
    /// # Arguments
    /// * `weights` - Pre-built TernaryWeightKernel (bit-parallel format).
    /// * `bias` - Optional bias vector; length must equal `weights.out_features()`.
    /// * `pre_norm` - Apply RMSNorm to the input before the matmul (BitNet b1.58 style).
    pub fn new(weights: TernaryWeightKernel, bias: Option<Vec<f32>>, pre_norm: bool) -> Self {
        let in_features = weights.in_features();
        let out_features = weights.out_features();
        if let Some(ref b) = bias {
            assert_eq!(b.len(), out_features, "bias length must equal out_features");
        }
        Self {
            weights,
            bias,
            in_features,
            out_features,
            pre_norm,
            norm_eps: 1e-5,
        }
    }

    /// Forward pass: input (in_features,) → output (out_features,)
    ///
    /// Uses DPS pattern — caller provides the output buffer. ZERO HEAP ALLOCATION.
    ///
    /// If `pre_norm` is true, RMSNorm is applied to the input before the ternary
    /// matmul. This is equivalent to normalising the input first, but is computed
    /// efficiently by scaling the output by `inv_rms` after the matmul (linearity).
    ///
    /// # Panics
    /// Panics if `input.len() != in_features` or `output.len() != out_features`.
    #[inline]
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.in_features);
        assert_eq!(output.len(), self.out_features);

        if self.pre_norm {
            // Compute RMS of input inline (no allocation)
            let mut sum_sq: f32 = 0.0;
            for &x in input.iter() {
                sum_sq += x * x;
            }
            let rms = (sum_sq / input.len() as f32 + self.norm_eps).sqrt();
            let inv_rms = 1.0 / rms;

            // Ternary matvec with the raw input, then scale by inv_rms.
            // This is equivalent to normalising the input first because the
            // ternary matmul is linear: W * (x * inv_rms) = (W * x) * inv_rms.
            ternary_matvec_kernel(input, &self.weights, output);
            for o in output.iter_mut() {
                *o *= inv_rms;
            }
        } else {
            ternary_matvec_kernel(input, &self.weights, output);
        }

        // Add bias
        if let Some(ref bias) = self.bias {
            for (o, b) in output.iter_mut().zip(bias.iter()) {
                *o += b;
            }
        }
    }

    /// Memory footprint in bytes (weights + bias).
    pub fn memory_bytes(&self) -> usize {
        self.weights.memory_bytes() + self.bias.as_ref().map_or(0, |b| b.len() * 4)
    }

    /// Compression ratio vs an equivalent FP32 linear layer (weights + bias).
    pub fn compression_ratio(&self) -> f32 {
        let fp32_bytes = (self.in_features * self.out_features * 4
            + self.bias.as_ref().map_or(0, |b| b.len() * 4)) as f32;
        fp32_bytes / self.memory_bytes() as f32
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::TernaryWeightKernel;

    fn make_kernel_2x2() -> TernaryWeightKernel {
        // W = [[1, -1], [0, 1]]
        TernaryWeightKernel::from_ternary(&[1i8, -1, 0, 1], 2, 2)
    }

    // ---- forward() without bias ----

    #[test]
    fn test_forward_no_bias() {
        // W = [[1, -1], [0, 1]], x = [2, 3]
        // y[0] = 2*1 + 3*(-1) = -1
        // y[1] = 2*0 + 3*1   =  3
        let layer = BitLinear::new(make_kernel_2x2(), None, false);
        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];

        layer.forward(&input, &mut output);

        assert!(
            (output[0] - (-1.0)).abs() < 1e-5,
            "y[0] should be -1, got {}",
            output[0]
        );
        assert!(
            (output[1] - 3.0).abs() < 1e-5,
            "y[1] should be 3, got {}",
            output[1]
        );
    }

    // ---- forward() with bias ----

    #[test]
    fn test_forward_with_bias() {
        // Same W, bias = [10.0, -5.0]
        // y[0] = -1 + 10 = 9
        // y[1] =  3 - 5  = -2
        let bias = vec![10.0f32, -5.0];
        let layer = BitLinear::new(make_kernel_2x2(), Some(bias), false);
        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];

        layer.forward(&input, &mut output);

        assert!(
            (output[0] - 9.0).abs() < 1e-5,
            "y[0] with bias should be 9, got {}",
            output[0]
        );
        assert!(
            (output[1] - (-2.0)).abs() < 1e-5,
            "y[1] with bias should be -2, got {}",
            output[1]
        );
    }

    // ---- forward() with pre_norm ----

    #[test]
    fn test_forward_pre_norm_no_bias() {
        // With pre_norm=true, the layer normalises the input by its RMS first.
        // W = [[1, 1, 1, 1]], input = [3, 3, 3, 3]
        // rms = 3.0, inv_rms = 1/3 → normalised input = [1, 1, 1, 1]
        // y[0] = 1+1+1+1 = 4
        let kernel = TernaryWeightKernel::from_ternary(&[1i8, 1, 1, 1], 1, 4);
        let layer = BitLinear::new(kernel, None, true);

        let input = [3.0f32, 3.0, 3.0, 3.0];
        let mut output = [0.0f32; 1];

        layer.forward(&input, &mut output);

        // W * x = 12.0, then * inv_rms = 12.0 / 3.0 = 4.0
        assert!(
            (output[0] - 4.0).abs() < 1e-4,
            "pre_norm forward should produce 4.0, got {}",
            output[0]
        );
    }

    #[test]
    fn test_forward_pre_norm_with_bias() {
        // Same setup as above, add bias = [1.0]
        let kernel = TernaryWeightKernel::from_ternary(&[1i8, 1, 1, 1], 1, 4);
        let layer = BitLinear::new(kernel, Some(vec![1.0f32]), true);

        let input = [3.0f32, 3.0, 3.0, 3.0];
        let mut output = [0.0f32; 1];
        layer.forward(&input, &mut output);

        // 4.0 + 1.0 = 5.0
        assert!(
            (output[0] - 5.0).abs() < 1e-4,
            "pre_norm + bias forward should produce 5.0, got {}",
            output[0]
        );
    }

    // ---- pre_norm scale-invariance property ----

    #[test]
    fn test_pre_norm_scale_invariance() {
        // With pre_norm, scaling the input by a constant should not change the output.
        let kernel = TernaryWeightKernel::from_ternary(&[1i8, -1, 0, 1, -1, 1, 0, -1], 2, 4);
        let layer = BitLinear::new(kernel, None, true);

        let input1 = [1.0f32, 2.0, 3.0, 4.0];
        let input2: Vec<f32> = input1.iter().map(|&x| x * 5.0).collect();

        let mut out1 = [0.0f32; 2];
        let mut out2 = [0.0f32; 2];

        layer.forward(&input1, &mut out1);
        layer.forward(&input2, &mut out2);

        for i in 0..2 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-4,
                "pre_norm scale-invariance failed at [{}]: {} vs {}",
                i,
                out1[i],
                out2[i]
            );
        }
    }

    // ---- memory_bytes and compression_ratio ----

    #[test]
    fn test_memory_bytes_no_bias() {
        // 2x2 kernel: words_per_row = (2+31)/32 = 1
        // plus_bits: 2 words × 4 bytes = 8 bytes
        // minus_bits: 2 words × 4 bytes = 8 bytes
        // total = 16 bytes
        let layer = BitLinear::new(make_kernel_2x2(), None, false);
        assert_eq!(
            layer.memory_bytes(),
            16,
            "2x2 kernel should use 16 bytes, got {}",
            layer.memory_bytes()
        );
    }

    #[test]
    fn test_memory_bytes_with_bias() {
        let bias = vec![1.0f32, 2.0];
        let layer = BitLinear::new(make_kernel_2x2(), Some(bias), false);
        // 16 (weights) + 2*4 (bias) = 24
        assert_eq!(
            layer.memory_bytes(),
            24,
            "2x2 + bias should use 24 bytes, got {}",
            layer.memory_bytes()
        );
    }

    #[test]
    fn test_compression_ratio_no_bias() {
        // FP32 baseline: 2*2*4 = 16 bytes
        // Bit-parallel: 16 bytes (same in this tiny example)
        let layer = BitLinear::new(make_kernel_2x2(), None, false);
        let ratio = layer.compression_ratio();
        assert!(
            ratio > 0.0,
            "compression ratio should be positive, got {}",
            ratio
        );
    }

    #[test]
    fn test_compression_ratio_large_layer() {
        // 1024x1024 layer: FP32 = 4MB, bit-parallel = 2 * (1024 * 32) * 4 = 256KB → 16x
        let values: Vec<i8> = (0..1024 * 1024).map(|i| (i % 3) as i8 - 1).collect();
        let kernel = TernaryWeightKernel::from_ternary(&values, 1024, 1024);
        let layer = BitLinear::new(kernel, None, false);
        let ratio = layer.compression_ratio();
        // Expect roughly 16x compression (may vary slightly with alignment)
        assert!(
            ratio > 10.0,
            "large layer should achieve >10x compression, got {:.2}x",
            ratio
        );
    }

    // ---- larger matmul correctness ----

    #[test]
    fn test_forward_3x3() {
        // W = [[1,-1,0],[1,1,-1],[0,-1,1]], x = [1,2,3]
        // y[0] = 1 - 2 + 0     = -1
        // y[1] = 1 + 2 - 3     =  0
        // y[2] = 0 - 2 + 3     =  1
        let kernel = TernaryWeightKernel::from_ternary(&[1i8, -1, 0, 1, 1, -1, 0, -1, 1], 3, 3);
        let layer = BitLinear::new(kernel, None, false);
        let input = [1.0f32, 2.0, 3.0];
        let mut output = [0.0f32; 3];

        layer.forward(&input, &mut output);

        assert!(
            (output[0] - (-1.0)).abs() < 1e-5,
            "y[0] should be -1, got {}",
            output[0]
        );
        assert!(
            (output[1] - 0.0).abs() < 1e-5,
            "y[1] should be 0, got {}",
            output[1]
        );
        assert!(
            (output[2] - 1.0).abs() < 1e-5,
            "y[2] should be 1, got {}",
            output[2]
        );
    }
}
