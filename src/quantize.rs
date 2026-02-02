//! Quantization: FP32 → Ternary {-1, 0, +1}
//!
//! Based on BitNet b1.58 research: weights can be quantized to 1.58 bits
//! with minimal accuracy loss when using learned scaling factors.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use crate::ops::TernaryWeight;

/// Quantization statistics
#[derive(Clone, Debug, Default)]
pub struct QuantStats {
    /// Number of +1 weights
    pub plus_count: usize,
    /// Number of -1 weights
    pub minus_count: usize,
    /// Number of 0 weights
    pub zero_count: usize,
    /// Original weight range (min, max)
    pub original_range: (f32, f32),
    /// Learned scale factor
    pub scale: f32,
    /// Mean absolute error after quantization
    pub mae: f32,
}

impl QuantStats {
    /// Sparsity ratio (fraction of zero weights)
    pub fn sparsity(&self) -> f32 {
        let total = self.plus_count + self.minus_count + self.zero_count;
        if total == 0 {
            0.0
        } else {
            self.zero_count as f32 / total as f32
        }
    }

    /// Effective bits per weight (log2(non-zero states))
    pub fn effective_bits(&self) -> f32 {
        // If sparsity is high, effective bits are lower
        // Full ternary = log2(3) ≈ 1.58 bits
        let non_zero = self.plus_count + self.minus_count;
        let total = non_zero + self.zero_count;
        if total == 0 {
            return 0.0;
        }

        // Entropy-based calculation
        let p_plus = self.plus_count as f32 / total as f32;
        let p_minus = self.minus_count as f32 / total as f32;
        let p_zero = self.zero_count as f32 / total as f32;

        let mut entropy = 0.0f32;
        if p_plus > 0.0 {
            entropy -= p_plus * p_plus.log2();
        }
        if p_minus > 0.0 {
            entropy -= p_minus * p_minus.log2();
        }
        if p_zero > 0.0 {
            entropy -= p_zero * p_zero.log2();
        }

        entropy
    }
}

/// Quantize FP32 weights to Ternary
///
/// Algorithm (BitNet b1.58 style):
/// 1. Compute scale factor: γ = mean(|W|)
/// 2. Quantize: W_t = sign(W) * round(|W| / γ) clamped to {-1, 0, +1}
///
/// # Arguments
/// * `weights` - FP32 weight matrix (row-major)
/// * `out_features` - Number of output features (rows)
/// * `in_features` - Number of input features (columns)
///
/// # Returns
/// Tuple of (TernaryWeight, QuantStats)
pub fn quantize_to_ternary(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
) -> (TernaryWeight, QuantStats) {
    assert_eq!(weights.len(), out_features * in_features);

    // Compute scale (mean absolute value)
    let sum_abs: f32 = weights.iter().map(|w| w.abs()).sum();
    let scale = sum_abs / weights.len() as f32;
    let scale = scale.max(1e-8); // Prevent division by zero

    // Quantize
    let mut ternary_values = vec![0i8; weights.len()];
    let mut stats = QuantStats {
        scale,
        original_range: (f32::INFINITY, f32::NEG_INFINITY),
        ..Default::default()
    };

    let mut mae_sum = 0.0f32;

    for (i, &w) in weights.iter().enumerate() {
        // Track range
        stats.original_range.0 = stats.original_range.0.min(w);
        stats.original_range.1 = stats.original_range.1.max(w);

        // Quantize: round(w / scale) clamped to {-1, 0, +1}
        let scaled = w / scale;
        let quantized = scaled.round().clamp(-1.0, 1.0) as i8;

        ternary_values[i] = quantized;

        // Track counts
        match quantized {
            1 => stats.plus_count += 1,
            -1 => stats.minus_count += 1,
            _ => stats.zero_count += 1,
        }

        // Compute error
        let reconstructed = quantized as f32 * scale;
        mae_sum += (w - reconstructed).abs();
    }

    stats.mae = mae_sum / weights.len() as f32;

    let tw = TernaryWeight::from_ternary(&ternary_values, out_features, in_features);
    let mut tw = tw;
    // Set the scale (need to rebuild with scale)
    tw = TernaryWeight::from_packed(tw.packed().to_vec(), out_features, in_features, scale);

    (tw, stats)
}

/// Quantize with custom threshold (for sparse models)
///
/// Values with |w| < threshold * scale are set to 0.
pub fn quantize_to_ternary_sparse(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    threshold: f32,
) -> (TernaryWeight, QuantStats) {
    assert_eq!(weights.len(), out_features * in_features);

    // Compute scale
    let sum_abs: f32 = weights.iter().map(|w| w.abs()).sum();
    let scale = (sum_abs / weights.len() as f32).max(1e-8);

    let cutoff = threshold * scale;

    let mut ternary_values = vec![0i8; weights.len()];
    let mut stats = QuantStats {
        scale,
        original_range: (f32::INFINITY, f32::NEG_INFINITY),
        ..Default::default()
    };

    let mut mae_sum = 0.0f32;

    for (i, &w) in weights.iter().enumerate() {
        stats.original_range.0 = stats.original_range.0.min(w);
        stats.original_range.1 = stats.original_range.1.max(w);

        // Apply threshold for sparsity
        let quantized = if w.abs() < cutoff {
            0i8
        } else if w > 0.0 {
            1i8
        } else {
            -1i8
        };

        ternary_values[i] = quantized;

        match quantized {
            1 => stats.plus_count += 1,
            -1 => stats.minus_count += 1,
            _ => stats.zero_count += 1,
        }

        let reconstructed = quantized as f32 * scale;
        mae_sum += (w - reconstructed).abs();
    }

    stats.mae = mae_sum / weights.len() as f32;

    let tw = TernaryWeight::from_packed(
        TernaryWeight::from_ternary(&ternary_values, out_features, in_features).packed().to_vec(),
        out_features,
        in_features,
        scale,
    );

    (tw, stats)
}

/// Dequantize ternary weights back to FP32
///
/// For debugging and comparison purposes.
pub fn dequantize_from_ternary(weights: &TernaryWeight) -> Vec<f32> {
    let mut result = vec![0.0f32; weights.out_features() * weights.in_features()];
    let scale = weights.scale();

    for row in 0..weights.out_features() {
        for col in 0..weights.in_features() {
            let t = weights.get(row, col);
            result[row * weights.in_features() + col] = t.to_i8() as f32 * scale;
        }
    }

    result
}

/// Compute quantization error metrics
pub fn compute_quantization_error(original: &[f32], quantized: &TernaryWeight) -> QuantizationError {
    let dequantized = dequantize_from_ternary(quantized);

    assert_eq!(original.len(), dequantized.len());

    let mut mae = 0.0f32;
    let mut mse = 0.0f32;
    let mut max_error = 0.0f32;

    for (o, d) in original.iter().zip(dequantized.iter()) {
        let err = (o - d).abs();
        mae += err;
        mse += err * err;
        max_error = max_error.max(err);
    }

    let n = original.len() as f32;

    QuantizationError {
        mae: mae / n,
        mse: mse / n,
        rmse: (mse / n).sqrt(),
        max_error,
        snr: compute_snr(original, &dequantized),
    }
}

/// Quantization error metrics
#[derive(Clone, Debug)]
pub struct QuantizationError {
    /// Mean Absolute Error
    pub mae: f32,
    /// Mean Squared Error
    pub mse: f32,
    /// Root Mean Squared Error
    pub rmse: f32,
    /// Maximum absolute error
    pub max_error: f32,
    /// Signal-to-Noise Ratio (dB)
    pub snr: f32,
}

/// Compute Signal-to-Noise Ratio
fn compute_snr(original: &[f32], reconstructed: &[f32]) -> f32 {
    let signal_power: f32 = original.iter().map(|x| x * x).sum();
    let noise_power: f32 = original.iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum();

    if noise_power < 1e-10 {
        return f32::INFINITY;
    }

    10.0 * (signal_power / noise_power).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_simple() {
        // Weights: [1.0, -1.0, 0.5, -0.5]
        // Scale ≈ 0.75
        // Quantized: [1, -1, 1, -1] (rounded)
        let weights = vec![1.0f32, -1.0, 0.5, -0.5];
        let (tw, stats) = quantize_to_ternary(&weights, 1, 4);

        assert_eq!(tw.out_features(), 1);
        assert_eq!(tw.in_features(), 4);
        assert!(stats.scale > 0.0);

        // Check that we have both plus and minus
        assert!(stats.plus_count > 0);
        assert!(stats.minus_count > 0);
    }

    #[test]
    fn test_quantize_sparse() {
        // With high threshold, small weights become zero
        let weights = vec![1.0f32, -1.0, 0.1, -0.1, 0.01, -0.01];
        let (_tw, stats) = quantize_to_ternary_sparse(&weights, 1, 6, 0.5);

        // Small weights should be zeroed
        assert!(stats.zero_count > 0);
        assert!(stats.sparsity() > 0.0);
    }

    #[test]
    fn test_dequantize_roundtrip() {
        let weights = vec![1.0f32, -1.0, 0.0, 1.0];
        let (tw, _) = quantize_to_ternary(&weights, 2, 2);
        let dequantized = dequantize_from_ternary(&tw);

        // Should have same sign pattern
        for (o, d) in weights.iter().zip(dequantized.iter()) {
            if *o > 0.5 {
                assert!(*d > 0.0);
            } else if *o < -0.5 {
                assert!(*d < 0.0);
            }
        }
    }

    #[test]
    fn test_effective_bits() {
        // Equal distribution: log2(3) ≈ 1.58 bits
        let stats = QuantStats {
            plus_count: 100,
            minus_count: 100,
            zero_count: 100,
            ..Default::default()
        };

        let bits = stats.effective_bits();
        assert!((bits - 1.58).abs() < 0.1);
    }

    #[test]
    fn test_quantization_error() {
        let original = vec![1.0f32, -1.0, 0.5, -0.5];
        let (tw, _) = quantize_to_ternary(&original, 1, 4);
        let error = compute_quantization_error(&original, &tw);

        // Error should be bounded
        assert!(error.mae < 1.0);
        assert!(error.rmse < 1.0);
        assert!(error.snr > 0.0);
    }

    #[test]
    fn test_compression_achieved() {
        // 1000 weights
        let weights: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        let (tw, _) = quantize_to_ternary(&weights, 100, 10);

        // FP32: 1000 * 4 = 4000 bytes
        // Ternary: 1000 / 4 = 250 bytes
        let compression = tw.compression_ratio();
        assert!((compression - 16.0).abs() < 1.0);
    }
}
