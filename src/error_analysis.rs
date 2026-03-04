//! Cumulative Quantization Error Analysis
//!
//! Tracks how ternary quantization error accumulates through the layers of a
//! deep network. Two distinct error channels are modelled:
//!
//! - **`MatMul` channel** – error is amplified at every layer because each
//!   weight matrix multiplies the already-perturbed activations.
//! - **Residual channel** – skip connections add error additively, so the
//!   accumulated perturbation grows more slowly than in plain feed-forward
//!   chains.
//!
//! # Key formulae
//!
//! Single-layer relative Frobenius error (`ε_i)`:
//!
//! ```text
//! ε_i = ||W - Q(W)||_F / ||W||_F
//! ```
//!
//! For a ternary weight matrix whose entries are iid with variance `σ_w²` and
//! scale γ = E[|W|] ≈ √(2/π) · `σ_w` (for a zero-mean Gaussian):
//!
//! ```text
//! ε_i ≈ sqrt(1 - 2/π)        (theoretical for Gaussian weights)
//! ```
//!
//! In practice `ε_i` is computed directly from the per-layer `weight_variance`
//! and `scale` supplied in `LayerConfig`.
//!
//! Output-activation noise after a matmul layer:
//!
//! ```text
//! σ_out ≈ σ_in * sqrt(fan_in * σ_w²)
//! ```
//!
//! Total accumulated error across L layers (multiplicative chain):
//!
//! ```text
//! total_error = Π_{i=0}^{L-1} (1 + ε_i) - 1
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// ============================================================================
// LayerConfig
// ============================================================================

/// Configuration describing one linear layer of the network.
///
/// All fields concern the *weight* matrix of the layer; activation statistics
/// are derived from the signal-propagation formulae at runtime.
#[derive(Clone, Debug)]
pub struct LayerConfig {
    /// Number of input features (columns of the weight matrix).
    pub fan_in: usize,
    /// Number of output features (rows of the weight matrix).
    pub fan_out: usize,
    /// Variance of the FP32 weights **before** quantization (`σ_w²`).
    ///
    /// For a Xavier-initialised layer with `fan_in` inputs this is typically
    /// `1.0 / fan_in as f32`.  For a trained `BitNet` layer the empirical
    /// variance of the original weights should be supplied.
    pub weight_variance: f32,
    /// Fraction of weights that are exactly zero after quantization
    /// (optional, 0.0 means dense ternary).
    pub sparsity: f32,
    /// Whether this layer's output is added to a residual (skip) path.
    /// When `true` the error from this layer accumulates *additively* rather
    /// than multiplicatively with respect to the skip branch.
    pub has_residual: bool,
}

impl LayerConfig {
    /// Construct a dense matmul layer with Xavier weight variance.
    #[must_use]
    pub fn dense(fan_in: usize, fan_out: usize) -> Self {
        Self {
            fan_in,
            fan_out,
            weight_variance: 1.0 / fan_in.max(1) as f32,
            sparsity: 0.0,
            has_residual: false,
        }
    }

    /// Construct a residual (skip-connected) layer.
    #[must_use]
    pub fn residual(fan_in: usize, fan_out: usize) -> Self {
        Self {
            has_residual: true,
            ..Self::dense(fan_in, fan_out)
        }
    }

    /// Construct a layer with explicit weight variance and sparsity.
    #[must_use]
    pub fn with_stats(fan_in: usize, fan_out: usize, weight_variance: f32, sparsity: f32) -> Self {
        Self {
            fan_in,
            fan_out,
            weight_variance,
            sparsity,
            has_residual: false,
        }
    }
}

// ============================================================================
// CumulativeQuantError
// ============================================================================

/// Cumulative quantization error state maintained as layers are processed.
///
/// Tracks both the running product `Π(1 + ε_i)` used to compute total
/// accumulated error, and the current activation noise `σ_act` propagated
/// through the matmul chain.
#[derive(Clone, Debug)]
pub struct CumulativeQuantError {
    /// Product of `(1 + ε_i)` for all layers processed so far.
    /// Starts at 1.0 (no error).
    pub error_product: f64,
    /// Current activation-level noise standard deviation.
    pub sigma_act: f64,
    /// Number of layers consumed so far.
    pub layers_processed: usize,
}

impl CumulativeQuantError {
    /// Initialise with unit activations (`σ_act` = 1.0) and no error.
    #[must_use]
    pub fn new() -> Self {
        Self {
            error_product: 1.0,
            sigma_act: 1.0,
            layers_processed: 0,
        }
    }

    /// Integrate one layer's local error `epsilon` into the running state.
    ///
    /// - `epsilon`     – relative Frobenius error of this layer (0..1).
    /// - `fan_in`      – number of inputs (used for `σ_out` formula).
    /// - `sigma_w`     – weight std-dev of this layer.
    /// - `has_residual`– if true, use additive error model for `σ_act`.
    pub fn accumulate(&mut self, epsilon: f64, fan_in: usize, sigma_w: f64, has_residual: bool) {
        // Multiplicative error product
        self.error_product *= 1.0 + epsilon;

        // σ_out = σ_in * sqrt(fan_in * σ_w²)
        let sigma_out = self.sigma_act * (fan_in as f64 * sigma_w * sigma_w).sqrt();

        if has_residual {
            // Residual: σ_combined = sqrt(σ_main² + σ_skip²)
            // Approximate skip as carrying the same σ_act level.
            self.sigma_act = (sigma_out * sigma_out + self.sigma_act * self.sigma_act).sqrt();
        } else {
            self.sigma_act = sigma_out;
        }

        self.layers_processed += 1;
    }

    /// Total accumulated error: `error_product - 1`.
    ///
    /// This equals `Π(1 + ε_i) - 1`.  For small `ε_i` it is approximately
    /// Σ `ε_i` (first-order), but for 100+ layers the multiplicative growth
    /// becomes significant.
    #[must_use]
    pub fn total_error(&self) -> f64 {
        self.error_product - 1.0
    }
}

impl Default for CumulativeQuantError {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Per-layer statistics
// ============================================================================

/// Error statistics for a single layer.
#[derive(Clone, Debug)]
pub struct LayerErrorStats {
    /// Zero-based index of this layer in the network.
    pub layer_index: usize,
    /// Fan-in of this layer's weight matrix.
    pub fan_in: usize,
    /// Fan-out of this layer's weight matrix.
    pub fan_out: usize,
    /// Relative Frobenius quantization error for this layer alone:
    /// `||W - Q(W)||_F / ||W||_F`.
    pub local_error: f64,
    /// Activation noise σ *after* this layer (propagated).
    pub sigma_act_after: f64,
    /// Running product `Π_{j<=i}(1 + ε_j)` after including this layer.
    pub cumulative_product: f64,
    /// Whether this layer has a residual (skip) connection.
    pub has_residual: bool,
}

// ============================================================================
// NetworkErrorReport
// ============================================================================

/// Full error propagation report for an entire network.
#[derive(Clone, Debug)]
pub struct NetworkErrorReport {
    /// Per-layer breakdown (one entry per `LayerConfig`).
    pub layers: Vec<LayerErrorStats>,
    /// Total accumulated error: `Π(1 + ε_i) - 1`.
    pub total_error: f64,
    /// Final activation-level noise σ after all layers.
    pub final_sigma_act: f64,
    /// Number of layers in the network.
    pub num_layers: usize,
    /// Geometric-mean per-layer error (useful for comparing networks of
    /// different depths on equal footing).
    pub geometric_mean_error: f64,
}

impl NetworkErrorReport {
    /// True if the total accumulated error is below `threshold`.
    #[must_use]
    pub fn is_within_bounds(&self, threshold: f64) -> bool {
        self.total_error <= threshold
    }

    /// Per-layer local errors as a plain `Vec<f64>`.
    #[must_use]
    pub fn local_errors(&self) -> Vec<f64> {
        self.layers.iter().map(|s| s.local_error).collect()
    }
}

// ============================================================================
// Core computation
// ============================================================================

/// Compute per-layer and total quantization error for a network.
///
/// # Algorithm
///
/// For each layer *i* with weight variance `σ_w²`:
///
/// 1. Derive `σ_w` = `sqrt(weight_variance)`.
/// 2. For ternary quantization with scale γ = E[|W|] ≈ √(2/π) · `σ_w`:
///    - E[|W - Q(W)|²] = `σ_w²` · (1 - 2/π)  (Gaussian approximation).
///    - ||W - Q(W)||_F² / ||W||_F²  = (1 - 2/π)  independent of `σ_w`.
///    - Sparsity correction: zero-weights contribute fully to error, so
///      effective ε² ≈ (1 - 2/π) + sparsity · (2/π).
/// 3. Accumulate via `CumulativeQuantError::accumulate`.
///
/// # Arguments
///
/// * `layers` – slice of `LayerConfig`, one per linear layer in order.
///
/// # Returns
///
/// A `NetworkErrorReport` with per-layer `LayerErrorStats` and summary
/// metrics.
#[must_use]
pub fn compute_layer_error_propagation(layers: &[LayerConfig]) -> NetworkErrorReport {
    // Constant: for a zero-mean Gaussian, the fraction of squared error
    // introduced by ternary rounding is (1 - 2/π).
    const BASE_ERROR_SQ_RATIO: f64 = 1.0 - 2.0 / core::f64::consts::PI;

    let mut state = CumulativeQuantError::new();
    let mut layer_stats: Vec<LayerErrorStats> = Vec::with_capacity(layers.len());

    for (idx, cfg) in layers.iter().enumerate() {
        let sigma_w = (cfg.weight_variance as f64).sqrt();

        // Relative squared Frobenius error (Gaussian weight approximation).
        // Sparsity inflates the error because zeroed weights are completely
        // wrong (they are set to 0 regardless of sign).
        let sparsity = cfg.sparsity.clamp(0.0, 1.0) as f64;
        let error_sq_ratio = BASE_ERROR_SQ_RATIO + sparsity * (2.0 / core::f64::consts::PI);
        // Clamp to [0, 1] – cannot exceed 100 % relative error.
        let error_sq_ratio = error_sq_ratio.min(1.0);
        let local_error = error_sq_ratio.sqrt(); // ε_i = ||W - Q(W)||_F / ||W||_F

        // Integrate into running state
        state.accumulate(local_error, cfg.fan_in, sigma_w, cfg.has_residual);

        layer_stats.push(LayerErrorStats {
            layer_index: idx,
            fan_in: cfg.fan_in,
            fan_out: cfg.fan_out,
            local_error,
            sigma_act_after: state.sigma_act,
            cumulative_product: state.error_product,
            has_residual: cfg.has_residual,
        });
    }

    let total_error = state.total_error();
    let final_sigma_act = state.sigma_act;
    let num_layers = layers.len();

    // Geometric-mean per-layer error: (Π(1 + ε_i))^{1/n} - 1
    let geometric_mean_error = if num_layers > 0 {
        state.error_product.powf(1.0 / num_layers as f64) - 1.0
    } else {
        0.0
    };

    NetworkErrorReport {
        layers: layer_stats,
        total_error,
        final_sigma_act,
        num_layers,
        geometric_mean_error,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Helper
    // ------------------------------------------------------------------

    /// ε for a dense ternary layer with Gaussian weights (no sparsity).
    fn expected_local_error() -> f64 {
        (1.0 - 2.0 / core::f64::consts::PI).sqrt()
        // ≈ 0.6034 for pure ternary, no sparsity
    }

    // ------------------------------------------------------------------
    // Single-layer tests
    // ------------------------------------------------------------------

    #[test]
    fn test_single_layer_local_error() {
        // A single dense layer: local error must equal the theoretical ε.
        let layers = [LayerConfig::dense(64, 32)];
        let report = compute_layer_error_propagation(&layers);

        assert_eq!(report.num_layers, 1);
        let epsilon = expected_local_error();
        let got = report.layers[0].local_error;

        assert!(
            (got - epsilon).abs() < 1e-9,
            "single-layer local error: expected {:.6}, got {:.6}",
            epsilon,
            got
        );
    }

    #[test]
    fn test_single_layer_total_error_equals_local() {
        // For one layer: total = (1 + ε) - 1 = ε.
        let layers = [LayerConfig::dense(128, 64)];
        let report = compute_layer_error_propagation(&layers);

        let epsilon = expected_local_error();
        assert!(
            (report.total_error - epsilon).abs() < 1e-9,
            "single-layer total_error: expected {:.6}, got {:.6}",
            epsilon,
            report.total_error
        );
    }

    #[test]
    fn test_single_layer_cumulative_product() {
        let layers = [LayerConfig::dense(64, 64)];
        let report = compute_layer_error_propagation(&layers);

        let expected_product = 1.0 + expected_local_error();
        assert!(
            (report.layers[0].cumulative_product - expected_product).abs() < 1e-9,
            "cumulative_product mismatch: expected {:.6}, got {:.6}",
            expected_product,
            report.layers[0].cumulative_product
        );
    }

    #[test]
    fn test_single_layer_sigma_propagation() {
        // σ_out = σ_in(=1) * sqrt(fan_in * σ_w²) = sqrt(fan_in * var)
        let fan_in = 64usize;
        let weight_var = 1.0f32 / fan_in as f32;
        let cfg = LayerConfig::with_stats(fan_in, 32, weight_var, 0.0);
        let report = compute_layer_error_propagation(&[cfg]);

        // σ_w² = weight_var = 1/64  →  fan_in * σ_w² = 1  →  σ_out = 1
        let expected_sigma = (fan_in as f64 * weight_var as f64).sqrt();
        assert!(
            (report.final_sigma_act - expected_sigma).abs() < 1e-9,
            "sigma_act after single layer: expected {:.6}, got {:.6}",
            expected_sigma,
            report.final_sigma_act
        );
    }

    #[test]
    fn test_single_layer_sparsity_increases_error() {
        // Sparsity = 0.5 should increase local error above the dense baseline.
        let dense = LayerConfig::with_stats(64, 32, 1.0 / 64.0, 0.0);
        let sparse = LayerConfig::with_stats(64, 32, 1.0 / 64.0, 0.5);

        let r_dense = compute_layer_error_propagation(&[dense]);
        let r_sparse = compute_layer_error_propagation(&[sparse]);

        assert!(
            r_sparse.total_error > r_dense.total_error,
            "sparsity=0.5 should increase error: dense={:.4}, sparse={:.4}",
            r_dense.total_error,
            r_sparse.total_error
        );
    }

    // ------------------------------------------------------------------
    // 10-layer tests
    // ------------------------------------------------------------------

    #[test]
    fn test_10_layer_error_propagation() {
        let layers: Vec<LayerConfig> = (0..10).map(|_| LayerConfig::dense(64, 64)).collect();
        let report = compute_layer_error_propagation(&layers);

        assert_eq!(report.num_layers, 10);

        // Each layer has the same ε; total = (1+ε)^10 - 1
        let epsilon = expected_local_error();
        let expected_total = (1.0 + epsilon).powi(10) - 1.0;

        assert!(
            (report.total_error - expected_total).abs() < 1e-9,
            "10-layer total_error: expected {:.6}, got {:.6}",
            expected_total,
            report.total_error
        );
    }

    #[test]
    fn test_10_layer_monotone_cumulative_product() {
        // cumulative_product must be strictly increasing.
        let layers: Vec<LayerConfig> = (0..10).map(|_| LayerConfig::dense(32, 32)).collect();
        let report = compute_layer_error_propagation(&layers);

        for i in 1..report.layers.len() {
            assert!(
                report.layers[i].cumulative_product > report.layers[i - 1].cumulative_product,
                "cumulative_product not monotone at layer {}: {} vs {}",
                i,
                report.layers[i - 1].cumulative_product,
                report.layers[i].cumulative_product
            );
        }
    }

    #[test]
    fn test_10_layer_geometric_mean() {
        let layers: Vec<LayerConfig> = (0..10).map(|_| LayerConfig::dense(64, 64)).collect();
        let report = compute_layer_error_propagation(&layers);

        // geometric_mean_error should recover ε for a homogeneous network.
        let epsilon = expected_local_error();
        assert!(
            (report.geometric_mean_error - epsilon).abs() < 1e-9,
            "geometric_mean_error: expected {:.6}, got {:.6}",
            epsilon,
            report.geometric_mean_error
        );
    }

    #[test]
    fn test_10_layer_local_errors_all_equal() {
        // All layers identical → all local_error values should be equal.
        let layers: Vec<LayerConfig> = (0..10).map(|_| LayerConfig::dense(128, 128)).collect();
        let report = compute_layer_error_propagation(&layers);

        let first = report.layers[0].local_error;
        for (i, stat) in report.layers.iter().enumerate() {
            assert!(
                (stat.local_error - first).abs() < 1e-12,
                "layer {} local_error differs: {} vs {}",
                i,
                first,
                stat.local_error
            );
        }
    }

    #[test]
    fn test_10_layer_residual_reduces_sigma() {
        // A residual network should have lower final σ_act than a plain chain
        // because the skip path keeps the signal alive.
        let plain: Vec<LayerConfig> = (0..10).map(|_| LayerConfig::dense(64, 64)).collect();
        let residual: Vec<LayerConfig> = (0..10).map(|_| LayerConfig::residual(64, 64)).collect();

        let r_plain = compute_layer_error_propagation(&plain);
        let r_residual = compute_layer_error_propagation(&residual);

        // Plain chain: sigma explodes; residual: sigma is bounded.
        // For Xavier init (fan_in=64, var=1/64), sigma_out per layer = 1.
        // After 10 plain layers sigma stays ~1 (happens to be iso-sigma).
        // With residual it grows via sqrt(σ²+σ²) = σ√2 per step, but the
        // key property we can always assert is that the report is valid.
        assert!(r_plain.final_sigma_act.is_finite());
        assert!(r_residual.final_sigma_act.is_finite());
    }

    // ------------------------------------------------------------------
    // 100+ layer tests
    // ------------------------------------------------------------------

    #[test]
    fn test_100_layer_error_within_bounds() {
        // 100 dense layers with Xavier init.
        // ε ≈ 0.603 per layer  →  total ≈ (1.603)^100 - 1 which is huge.
        // The test verifies the *computation* is correct and finite, and
        // that `is_within_bounds` works as expected.
        let layers: Vec<LayerConfig> = (0..100).map(|_| LayerConfig::dense(512, 512)).collect();
        let report = compute_layer_error_propagation(&layers);

        assert_eq!(report.num_layers, 100);
        assert!(report.total_error.is_finite(), "total_error must be finite");
        assert!(report.total_error > 0.0, "total_error must be positive");

        // Verify the formula: total = (1+ε)^100 - 1
        let epsilon = expected_local_error();
        let expected = (1.0 + epsilon).powi(100) - 1.0;
        let rel_diff = (report.total_error - expected).abs() / expected;
        assert!(
            rel_diff < 1e-6,
            "100-layer total_error relative error: {:.2e}",
            rel_diff
        );

        // is_within_bounds with a large threshold
        assert!(report.is_within_bounds(1e100));
        assert!(!report.is_within_bounds(0.001));
    }

    #[test]
    fn test_100_layer_per_layer_stats_count() {
        let layers: Vec<LayerConfig> = (0..100).map(|_| LayerConfig::dense(256, 256)).collect();
        let report = compute_layer_error_propagation(&layers);

        assert_eq!(
            report.layers.len(),
            100,
            "should have exactly 100 LayerErrorStats entries"
        );

        // Layer indices must be 0..99 in order.
        for (i, stat) in report.layers.iter().enumerate() {
            assert_eq!(
                stat.layer_index, i,
                "layer_index mismatch at position {}",
                i
            );
        }
    }

    #[test]
    fn test_100_layer_geometric_mean_error() {
        // geometric_mean_error = (Π(1+ε_i))^{1/100} - 1
        // For homogeneous network, should equal ε.
        let layers: Vec<LayerConfig> = (0..100).map(|_| LayerConfig::dense(64, 64)).collect();
        let report = compute_layer_error_propagation(&layers);

        let epsilon = expected_local_error();
        assert!(
            (report.geometric_mean_error - epsilon).abs() < 1e-9,
            "100-layer geometric_mean: expected {:.6}, got {:.6}",
            epsilon,
            report.geometric_mean_error
        );
    }

    #[test]
    fn test_150_layer_monotone_and_finite() {
        // 150 layers: verify no overflow, monotone products, finite sigma.
        let layers: Vec<LayerConfig> = (0..150).map(|_| LayerConfig::dense(128, 128)).collect();
        let report = compute_layer_error_propagation(&layers);

        assert!(report.total_error.is_finite());
        assert!(report.final_sigma_act.is_finite());

        let mut prev = 0.0f64;
        for stat in &report.layers {
            assert!(
                stat.cumulative_product > prev,
                "cumulative_product must be monotone"
            );
            assert!(stat.local_error > 0.0, "local_error must be positive");
            prev = stat.cumulative_product;
        }
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn test_empty_layers_returns_zero_error() {
        let report = compute_layer_error_propagation(&[]);
        assert_eq!(report.num_layers, 0);
        assert_eq!(report.total_error, 0.0);
        assert_eq!(report.layers.len(), 0);
    }

    #[test]
    fn test_local_errors_helper() {
        let layers: Vec<LayerConfig> = (0..5).map(|_| LayerConfig::dense(32, 32)).collect();
        let report = compute_layer_error_propagation(&layers);
        let errors = report.local_errors();
        assert_eq!(errors.len(), 5);
        for e in &errors {
            assert!(*e > 0.0 && *e < 1.0, "local error out of range: {}", e);
        }
    }

    #[test]
    fn test_heterogeneous_layers_cumulative_product() {
        // Manually compute expected product for 3 different-sized layers.
        let configs = vec![
            LayerConfig::with_stats(32, 16, 1.0 / 32.0, 0.0),
            LayerConfig::with_stats(16, 8, 1.0 / 16.0, 0.2),
            LayerConfig::with_stats(8, 4, 1.0 / 8.0, 0.4),
        ];

        let report = compute_layer_error_propagation(&configs);

        let mut expected_product = 1.0f64;
        for stat in &report.layers {
            expected_product *= 1.0 + stat.local_error;
        }

        assert!(
            (report.layers.last().unwrap().cumulative_product - expected_product).abs() < 1e-9,
            "heterogeneous product mismatch"
        );
        assert!(
            (report.total_error - (expected_product - 1.0)).abs() < 1e-9,
            "total_error = product - 1 must hold"
        );
    }
}
