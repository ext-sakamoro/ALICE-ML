//! Closed-form oracles for the numeric laws of ALICE-ML
//!
//! Every test compares an implementation against a value that is known
//! independently of the code (a formula, an algebraic identity, a
//! constructive input), not against a stored output of the code itself
//! Where the crate offers several kernels for one law (packed / bit-parallel /
//! SIMD dispatch / INT8 input) they are all held to the same oracle, and the
//! "precision parameter" of this crate — the vector length relative to the
//! 4-lane NEON, 8-lane AVX2 and 32-bit word boundaries — is swept so that the
//! remainder paths are exercised, not just the aligned case
//!
//! Inputs are small integers and powers of two wherever a sum is compared, so
//! f32 accumulation is exact and the assertions are `==`, not tolerances

#![allow(clippy::cast_precision_loss)] // small integer counts → f32

use alice_ml::{
    dequantize_from_ternary, quantize_to_ternary, quantize_to_ternary_qat, tensor_layer_norm,
    tensor_rms_norm, tensor_softmax, tensor_softmax_fast, ternary_matmul_batch, ternary_matvec,
    ternary_matvec_kernel, ternary_matvec_kernel_quantized, ternary_matvec_simd_dispatch,
    QuantizedTensor, Tensor, TernaryWeight, TernaryWeightKernel,
};

/// Deterministic ternary pattern with all three symbols in every row:
/// `w[r][c] = [+1, -1, 0][(r + c) % 3]`
fn pattern(out_features: usize, in_features: usize) -> Vec<i8> {
    (0..out_features * in_features)
        .map(|i| {
            let (r, c) = (i / in_features, i % in_features);
            [1i8, -1, 0][(r + c) % 3]
        })
        .collect()
}

/// Closed form of the ternary matvec: `y[r] = scale · (Σ_{w=+1} x − Σ_{w=−1} x)`
fn oracle_matvec(
    values: &[i8],
    out_features: usize,
    in_features: usize,
    x: &[f32],
    scale: f32,
) -> Vec<f32> {
    (0..out_features)
        .map(|r| {
            let row = &values[r * in_features..(r + 1) * in_features];
            let acc: f32 = row.iter().zip(x).map(|(&w, &xi)| f32::from(w) * xi).sum();
            acc * scale
        })
        .collect()
}

/// Integer-valued inputs: `x_c = (c % 7) − 3`, exact in f32 for any sum here
fn int_input(n: usize) -> Vec<f32> {
    (0..n).map(|c| (c % 7) as f32 - 3.0).collect()
}

// ----------------------------------------------------------------------------
// Ternary matvec: four kernels, one law, every length 1..=70
// ----------------------------------------------------------------------------

#[test]
fn every_matvec_kernel_equals_the_closed_form_for_all_lengths() {
    for out_features in [1usize, 2, 3, 5] {
        for in_features in 1..=70usize {
            let values = pattern(out_features, in_features);
            let x = int_input(in_features);
            let expected = oracle_matvec(&values, out_features, in_features, &x, 1.0);

            let packed = TernaryWeight::from_ternary(&values, out_features, in_features);
            let mut y = vec![0.0f32; out_features];
            ternary_matvec(&x, &packed, &mut y);
            assert_eq!(y, expected, "ternary_matvec {out_features}x{in_features}");

            let kernel = TernaryWeightKernel::from_ternary(&values, out_features, in_features);
            let mut y = vec![0.0f32; out_features];
            ternary_matvec_kernel(&x, &kernel, &mut y);
            assert_eq!(
                y, expected,
                "ternary_matvec_kernel {out_features}x{in_features}"
            );

            let mut y = vec![0.0f32; out_features];
            ternary_matvec_simd_dispatch(&x, &kernel, &mut y);
            assert_eq!(y, expected, "simd_dispatch {out_features}x{in_features}");

            let from_packed = TernaryWeightKernel::from_packed_weight(&packed);
            let mut y = vec![0.0f32; out_features];
            ternary_matvec_kernel(&x, &from_packed, &mut y);
            assert_eq!(
                y, expected,
                "from_packed_weight {out_features}x{in_features}"
            );
        }
    }
}

#[test]
fn scale_is_a_pure_multiplier_of_the_matvec() {
    let (out_features, in_features) = (3usize, 37usize);
    let values = pattern(out_features, in_features);
    let x = int_input(in_features);
    for scale in [0.5f32, 2.0, 0.25, 8.0] {
        let expected = oracle_matvec(&values, out_features, in_features, &x, scale);
        let kernel =
            TernaryWeightKernel::from_ternary_scaled(&values, out_features, in_features, scale);
        let mut y = vec![0.0f32; out_features];
        ternary_matvec_kernel(&x, &kernel, &mut y);
        assert_eq!(y, expected, "scale {scale}");
        let mut y = vec![0.0f32; out_features];
        ternary_matvec_simd_dispatch(&x, &kernel, &mut y);
        assert_eq!(y, expected, "simd scale {scale}");
    }
}

#[test]
fn batched_matmul_is_the_matvec_of_every_row() {
    let (out_features, in_features, batch) = (4usize, 33usize, 5usize);
    let values = pattern(out_features, in_features);
    let packed = TernaryWeight::from_ternary(&values, out_features, in_features);
    let inputs: Vec<f32> = (0..batch)
        .flat_map(|b| {
            int_input(in_features)
                .into_iter()
                .map(move |v| v + b as f32)
        })
        .collect();
    let mut out = vec![0.0f32; batch * out_features];
    ternary_matmul_batch(&inputs, &packed, &mut out, batch);
    for b in 0..batch {
        let x = &inputs[b * in_features..(b + 1) * in_features];
        let expected = oracle_matvec(&values, out_features, in_features, x, 1.0);
        assert_eq!(
            &out[b * out_features..(b + 1) * out_features],
            &expected[..],
            "row {b}"
        );
    }
}

#[test]
fn int8_input_kernel_matches_the_integer_closed_form() {
    // x = ±127 · k / 127 exactly representable: pick x in {−3..3} and a scale of
    // 1/127 · max|x| so that the INT8 grid contains every input exactly
    for in_features in [1usize, 4, 7, 8, 9, 31, 32, 33, 64, 65] {
        let out_features = 3;
        let values = pattern(out_features, in_features);
        let x = int_input(in_features);
        let q = QuantizedTensor::from_f32_slice(&x, &[in_features]);
        // Dequantised grid: q_i · scale where scale = max|x| / 127 — the
        // reconstruction is exact only when max|x| · (x_i / max|x|) · 127 is an
        // integer, so compare against the oracle applied to the dequantised x
        let mut x_deq = vec![0.0f32; in_features];
        q.dequantize_to(&mut x_deq);
        let kernel = TernaryWeightKernel::from_ternary(&values, out_features, in_features);
        let mut y = vec![0.0f32; out_features];
        ternary_matvec_kernel_quantized(&q, &kernel, &mut y);
        let expected = oracle_matvec(&values, out_features, in_features, &x_deq, 1.0);
        for (a, b) in y.iter().zip(&expected) {
            assert!(
                (a - b).abs() <= 1e-5 * b.abs().max(1.0),
                "n={in_features}: {a} vs {b}"
            );
        }
    }
}

// ----------------------------------------------------------------------------
// Quantisation laws (BitNet b1.58 style: γ = mean|W|, q = clamp(round(W/γ)))
// ----------------------------------------------------------------------------

#[test]
fn ternary_valued_weights_quantise_to_themselves_with_scale_mean_abs() {
    // W ∈ {−a, 0, +a}: γ = a · (non-zero fraction), |W|/γ = 1/f ≥ 1 → every
    // non-zero weight becomes sign(W) and every zero stays zero, for any f
    for (out_features, in_features) in [(1usize, 3usize), (2, 6), (3, 33), (5, 70)] {
        let ternary = pattern(out_features, in_features);
        for a in [0.5f32, 1.0, 3.0, 1e-3] {
            let w: Vec<f32> = ternary.iter().map(|&t| f32::from(t) * a).collect();
            let (q, stats) = quantize_to_ternary(&w, out_features, in_features);
            let nonzero = ternary.iter().filter(|&&t| t != 0).count() as f32;
            let gamma = a * nonzero / (out_features * in_features) as f32;
            assert!(
                (stats.scale - gamma).abs() <= 1e-5 * gamma,
                "γ {} vs {gamma}",
                stats.scale
            );
            for (r, row) in ternary.chunks(in_features).enumerate() {
                for (c, &t) in row.iter().enumerate() {
                    assert_eq!(q.get(r, c).to_i8(), t, "({r},{c}) a={a}");
                }
            }
            // Reconstruction is sign(W)·γ, so the error per weight is exactly
            // |a − γ| on non-zeros and 0 on zeros
            let deq = dequantize_from_ternary(&q);
            for (d, &t) in deq.iter().zip(&ternary) {
                let expected = f32::from(t) * gamma;
                assert!((d - expected).abs() <= 1e-6 * a, "deq {d} vs {expected}");
            }
        }
    }
}

#[test]
fn quantisation_is_equivariant_under_positive_scaling() {
    // quantize(c·W) has the same ternary pattern as quantize(W) and scale c·γ
    let (out_features, in_features) = (4usize, 17usize);
    let w: Vec<f32> = (0..out_features * in_features)
        .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
        .collect();
    let (q0, s0) = quantize_to_ternary(&w, out_features, in_features);
    for c in [2.0f32, 0.125, 64.0] {
        let wc: Vec<f32> = w.iter().map(|v| v * c).collect();
        let (qc, sc) = quantize_to_ternary(&wc, out_features, in_features);
        assert_eq!(qc.packed(), q0.packed(), "pattern c={c}");
        assert!(
            c.mul_add(-s0.scale, sc.scale).abs() <= 1e-6 * sc.scale,
            "scale c={c}"
        );
    }
}

#[test]
fn threshold_of_the_rounding_law_is_half_gamma() {
    // q = round(|w|/γ): |w| slightly above γ/2 → ±1, slightly below → 0
    // Use quantize_to_ternary_qat (γ given, temperature 1) so γ is exact
    let gamma = 0.8f32;
    let w = [
        0.5 * gamma * 1.01,
        -0.5 * gamma * 1.01,
        0.5 * gamma * 0.99,
        -0.5 * gamma * 0.99,
    ];
    let (q, _) = quantize_to_ternary_qat(&w, 1, 4, gamma, 1.0);
    assert_eq!(q.get(0, 0).to_i8(), 1);
    assert_eq!(q.get(0, 1).to_i8(), -1);
    assert_eq!(q.get(0, 2).to_i8(), 0);
    assert_eq!(q.get(0, 3).to_i8(), 0);
    // temperature τ scales the threshold to τ·γ/2
    let tau = 2.0f32;
    let w = [0.5 * gamma * tau * 1.01, 0.5 * gamma * tau * 0.99];
    let (q, _) = quantize_to_ternary_qat(&w, 1, 2, gamma, tau);
    assert_eq!(q.get(0, 0).to_i8(), 1);
    assert_eq!(q.get(0, 1).to_i8(), 0);
}

#[test]
fn compression_ratio_is_the_bit_width_law() {
    // Packed: 2 bits per weight → 32 / 2 = 16× when n is a multiple of 4
    // Bit-parallel: 2 bit-planes of 32-bit words → 16× when n_in is a multiple of 32
    let n = 64usize;
    let values = pattern(1, n);
    let packed = TernaryWeight::from_ternary(&values, 1, n);
    assert_eq!(packed.memory_bytes(), n / 4);
    assert!((packed.compression_ratio() - 16.0).abs() < 1e-6);
    let kernel = TernaryWeightKernel::from_ternary(&values, 1, n);
    assert_eq!(kernel.memory_bytes(), 2 * (n / 32) * 4);
    assert!((kernel.compression_ratio() - 16.0).abs() < 1e-6);
    // Odd length: the packed form rounds up to whole bytes, the kernel to
    // whole words — both ratios are strictly below the ideal
    let n = 33usize;
    let values = pattern(1, n);
    let packed = TernaryWeight::from_ternary(&values, 1, n);
    assert_eq!(packed.memory_bytes(), n.div_ceil(4));
    assert!(packed.compression_ratio() < 16.0);
    let kernel = TernaryWeightKernel::from_ternary(&values, 1, n);
    assert_eq!(kernel.memory_bytes(), 2 * n.div_ceil(32) * 4);
    assert!(kernel.compression_ratio() < 16.0);
}

// ----------------------------------------------------------------------------
// Normalisation / softmax identities
// ----------------------------------------------------------------------------

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i as f32).mul_add(0.37, -2.0)).collect()
}

#[test]
fn softmax_sums_to_one_is_shift_invariant_and_matches_the_two_element_formula() {
    for n in [1usize, 2, 3, 7, 8, 9, 64, 65] {
        let mut a = ramp(n);
        let mut o = vec![0.0f32; n];
        tensor_softmax(
            &Tensor::from_arena(&mut a, &[n]),
            &mut Tensor::from_arena(&mut o, &[n]),
        );
        let sum: f32 = o.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "n={n} sum {sum}");
        assert!(o.iter().all(|&p| p > 0.0 && p <= 1.0));
        // shift invariance: softmax(x + c) = softmax(x)
        let mut shifted: Vec<f32> = a.iter().map(|v| v + 5.5).collect();
        let mut o2 = vec![0.0f32; n];
        tensor_softmax(
            &Tensor::from_arena(&mut shifted, &[n]),
            &mut Tensor::from_arena(&mut o2, &[n]),
        );
        for (p, q) in o.iter().zip(&o2) {
            assert!((p - q).abs() < 1e-6, "n={n}: {p} vs {q}");
        }
        // the fast variant (Schraudolph exp, max relative error ≈ 6 %) is
        // still a normalised distribution with the same ranking
        let mut o3 = vec![0.0f32; n];
        tensor_softmax_fast(
            &Tensor::from_arena(&mut a, &[n]),
            &mut Tensor::from_arena(&mut o3, &[n]),
        );
        let sum3: f32 = o3.iter().sum();
        assert!((sum3 - 1.0).abs() < 1e-5, "fast n={n} sum {sum3}");
        for (p, q) in o.iter().zip(&o3) {
            assert!((p - q).abs() <= 0.06 * p, "fast n={n}: {p} vs {q}");
        }
        for w in o3.windows(2) {
            assert!(
                w[0] <= w[1],
                "fast n={n}: ranking of an increasing input not preserved"
            );
        }
    }
    // two elements: softmax([0, d]) = [1/(1+e^d), e^d/(1+e^d)]
    for d in [-3.0f32, -0.5, 0.0, 0.5, 3.0] {
        let mut a = [0.0f32, d];
        let mut o = [0.0f32; 2];
        tensor_softmax(
            &Tensor::from_arena(&mut a, &[2]),
            &mut Tensor::from_arena(&mut o, &[2]),
        );
        let p1 = d.exp() / (1.0 + d.exp());
        assert!(
            (o[1] - p1).abs() < 1e-6 && (o[0] - (1.0 - p1)).abs() < 1e-6,
            "d={d}: {o:?}"
        );
    }
}

#[test]
fn layer_norm_output_has_zero_mean_unit_variance_and_is_affine_invariant() {
    for n in [2usize, 3, 8, 9, 33, 64, 65] {
        let mut a = ramp(n);
        let mut o = vec![0.0f32; n];
        tensor_layer_norm(
            &Tensor::from_arena(&mut a, &[n]),
            None,
            None,
            0.0,
            &mut Tensor::from_arena(&mut o, &[n]),
        );
        let mean = o.iter().sum::<f32>() / n as f32;
        let var = o.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        assert!(mean.abs() < 1e-5, "n={n} mean {mean}");
        assert!((var - 1.0).abs() < 1e-4, "n={n} var {var}");
        // x → αx + β (α > 0) leaves the output unchanged when eps = 0
        let mut ab: Vec<f32> = a.iter().map(|v| 3.0 * v - 7.0).collect();
        let mut o2 = vec![0.0f32; n];
        tensor_layer_norm(
            &Tensor::from_arena(&mut ab, &[n]),
            None,
            None,
            0.0,
            &mut Tensor::from_arena(&mut o2, &[n]),
        );
        for (p, q) in o.iter().zip(&o2) {
            assert!((p - q).abs() < 1e-4, "n={n}: {p} vs {q}");
        }
        // eps ≪ var must not move the result (precision-parameter independence)
        for eps in [1e-12f32, 1e-8, 1e-6] {
            let mut o3 = vec![0.0f32; n];
            tensor_layer_norm(
                &Tensor::from_arena(&mut a, &[n]),
                None,
                None,
                eps,
                &mut Tensor::from_arena(&mut o3, &[n]),
            );
            for (p, q) in o.iter().zip(&o3) {
                assert!((p - q).abs() < 1e-4, "n={n} eps={eps}: {p} vs {q}");
            }
        }
    }
}

#[test]
fn rms_norm_output_has_unit_rms_and_is_scale_invariant() {
    for n in [1usize, 2, 7, 8, 9, 32, 33, 65] {
        let mut a = ramp(n);
        let mut o = vec![0.0f32; n];
        tensor_rms_norm(
            &Tensor::from_arena(&mut a, &[n]),
            None,
            0.0,
            &mut Tensor::from_arena(&mut o, &[n]),
        );
        let rms = (o.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "n={n} rms {rms}");
        // direction preserved: out = x / rms(x) exactly
        let rms_in = (a.iter().map(|v| v * v).sum::<f32>() / n as f32).sqrt();
        for (x, y) in a.iter().zip(&o) {
            assert!(
                (y - x / rms_in).abs() < 1e-5,
                "n={n}: {y} vs {}",
                x / rms_in
            );
        }
        // x → αx (α > 0) leaves the output unchanged when eps = 0
        let mut ax: Vec<f32> = a.iter().map(|v| v * 0.125).collect();
        let mut o2 = vec![0.0f32; n];
        tensor_rms_norm(
            &Tensor::from_arena(&mut ax, &[n]),
            None,
            0.0,
            &mut Tensor::from_arena(&mut o2, &[n]),
        );
        for (p, q) in o.iter().zip(&o2) {
            assert!((p - q).abs() < 1e-5, "n={n}: {p} vs {q}");
        }
    }
}
