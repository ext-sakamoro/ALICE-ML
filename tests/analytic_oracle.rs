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
#![allow(clippy::float_cmp)] // integer-valued f32 results are compared exactly on purpose
#![allow(clippy::many_single_char_names, clippy::similar_names)]
#![allow(clippy::suboptimal_flops, clippy::identity_op)] // oracle formulas are written as the law reads

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
fn every_matvec_entry_point_rejects_mismatched_lengths() {
    use std::panic::catch_unwind;
    let kernel = TernaryWeightKernel::from_ternary(&[1; 64], 2, 32);
    let packed = TernaryWeight::from_ternary(&[1; 64], 2, 32);
    let short_in = [1.0f32; 8];
    let ok_in = [1.0f32; 32];
    let mut ok_out = [0.0f32; 2];
    assert!(catch_unwind(|| ternary_matvec_kernel(&short_in, &kernel, &mut [0.0; 2])).is_err());
    assert!(catch_unwind(|| ternary_matvec_kernel(&ok_in, &kernel, &mut [0.0; 1])).is_err());
    assert!(catch_unwind(|| ternary_matvec(&short_in, &packed, &mut [0.0; 2])).is_err());
    assert!(catch_unwind(|| ternary_matvec(&ok_in, &packed, &mut [0.0; 1])).is_err());
    let q_short = QuantizedTensor::from_f32_slice(&short_in, &[8]);
    assert!(
        catch_unwind(|| ternary_matvec_kernel_quantized(&q_short, &kernel, &mut [0.0; 2])).is_err()
    );
    // matching lengths run
    ternary_matvec_kernel(&ok_in, &kernel, &mut ok_out);
    assert_eq!(ok_out, [32.0, 32.0]);
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

// ----------------------------------------------------------------------------
// Element-wise / reduction ops: exact on integer inputs for every length across
// the 8-lane AVX2 boundary (the SIMD path has a chunk loop + remainder loop)
// ----------------------------------------------------------------------------

use alice_ml::{
    compute_quantization_error, quantize_to_ternary_sparse, tensor_add, tensor_copy, tensor_max,
    tensor_mean, tensor_min, tensor_relu, tensor_relu_inplace, tensor_scale, tensor_sub,
    tensor_sum, OwnedTensor, QuantStats,
};

#[test]
fn elementwise_and_reduction_ops_are_exact_on_integer_inputs_for_every_length() {
    for n in 1..=40usize {
        let a = int_input(n);
        let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32 - 2.0).collect();
        let (mut av, mut bv, mut out) = (a.clone(), b.clone(), vec![0.0f32; n]);
        let ta = Tensor::from_arena(&mut av, &[n]);
        let tb = Tensor::from_arena(&mut bv, &[n]);

        tensor_add(&ta, &tb, &mut Tensor::from_arena(&mut out, &[n]));
        assert_eq!(
            out,
            a.iter().zip(&b).map(|(x, y)| x + y).collect::<Vec<_>>(),
            "add n={n}"
        );
        tensor_sub(&ta, &tb, &mut Tensor::from_arena(&mut out, &[n]));
        assert_eq!(
            out,
            a.iter().zip(&b).map(|(x, y)| x - y).collect::<Vec<_>>(),
            "sub n={n}"
        );
        tensor_scale(&ta, -1.5, &mut Tensor::from_arena(&mut out, &[n]));
        assert_eq!(
            out,
            a.iter().map(|x| x * -1.5).collect::<Vec<_>>(),
            "scale n={n}"
        );
        tensor_relu(&ta, &mut Tensor::from_arena(&mut out, &[n]));
        assert_eq!(
            out,
            a.iter().map(|x| x.max(0.0)).collect::<Vec<_>>(),
            "relu n={n}"
        );
        let mut inplace = a.clone();
        tensor_relu_inplace(&mut Tensor::from_arena(&mut inplace, &[n]));
        assert_eq!(inplace, out, "relu_inplace n={n}");
        tensor_copy(&ta, &mut Tensor::from_arena(&mut out, &[n]));
        assert_eq!(out, a, "copy n={n}");

        // Σ, mean, max, min of (i % 7) − 3 are integers: exact
        let sum: f32 = a.iter().sum();
        assert_eq!(tensor_sum(&ta), sum, "sum n={n}");
        assert_eq!(tensor_mean(&ta), sum / n as f32, "mean n={n}");
        assert_eq!(
            tensor_max(&ta),
            a.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            "max n={n}"
        );
        assert_eq!(
            tensor_min(&ta),
            a.iter().copied().fold(f32::INFINITY, f32::min),
            "min n={n}"
        );
        // the extreme sits in the remainder lane for n % 8 != 0 when placed last
        let mut tail = a.clone();
        tail[n - 1] = 100.0;
        assert_eq!(
            tensor_max(&Tensor::from_arena(&mut tail, &[n])),
            100.0,
            "max tail n={n}"
        );
        tail[n - 1] = -100.0;
        assert_eq!(
            tensor_min(&Tensor::from_arena(&mut tail, &[n])),
            -100.0,
            "min tail n={n}"
        );
    }
    // empty tensor: identities of the reductions
    let mut e: Vec<f32> = Vec::new();
    let te = Tensor::from_arena(&mut e, &[0]);
    assert!(te.is_empty());
    assert_eq!(tensor_sum(&te), 0.0);
    assert_eq!(tensor_max(&te), f32::NEG_INFINITY);
    assert_eq!(tensor_min(&te), f32::INFINITY);
}

#[test]
fn tensor_accessors_and_owned_tensor_are_consistent() {
    let mut data = int_input(12);
    let mut t = Tensor::from_arena(&mut data, &[3, 4]);
    assert!(!t.is_empty());
    assert_eq!(t.len(), 12);
    assert_eq!(t.get(&[1, 2]), t.get_flat(6));
    assert_eq!(t.get_flat(6), (6 % 7) as f32 - 3.0);
    t.set(&[2, 3], 42.0);
    assert_eq!(t.get_flat(11), 42.0);
    t.set_flat(0, -7.0);
    assert_eq!(t.get(&[0, 0]), -7.0);
    assert_eq!(t.data()[0], -7.0);

    let owned = OwnedTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(owned.len(), 6);
    assert!(!owned.is_empty());
    assert_eq!(owned.shape(), &[2, 3]);
    assert_eq!(owned.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut owned = owned;
    owned.data_mut()[5] = 60.0;
    assert_eq!(owned.data()[5], 60.0);
    let empty = OwnedTensor::from_slice(&[], &[0]);
    assert!(empty.is_empty());
    assert_eq!(empty.len(), 0);
    let q = QuantizedTensor::from_f32_slice(&[1.0, -2.0], &[2]);
    assert!(!q.is_empty());
    assert_eq!(q.len(), 2);
    let q0 = QuantizedTensor::from_f32_slice(&[], &[0]);
    assert!(q0.is_empty());
}

#[test]
fn layer_norm_and_rms_norm_apply_weight_and_bias_per_element() {
    let n = 9usize;
    let mut a = ramp(n);
    let mut w: Vec<f32> = (0..n).map(|i| 0.5 + i as f32).collect();
    let mut b: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();
    let mut plain = vec![0.0f32; n];
    let mut scaled = vec![0.0f32; n];
    tensor_layer_norm(
        &Tensor::from_arena(&mut a, &[n]),
        None,
        None,
        0.0,
        &mut Tensor::from_arena(&mut plain, &[n]),
    );
    tensor_layer_norm(
        &Tensor::from_arena(&mut a, &[n]),
        Some(&Tensor::from_arena(&mut w, &[n])),
        Some(&Tensor::from_arena(&mut b, &[n])),
        0.0,
        &mut Tensor::from_arena(&mut scaled, &[n]),
    );
    for i in 0..n {
        assert!(
            (scaled[i] - (plain[i] * w[i] + b[i])).abs() < 1e-5,
            "layer_norm {i}"
        );
    }
    // a large eps shrinks the output: out = (x − mean) / sqrt(var + eps)
    let mut big = vec![0.0f32; n];
    tensor_layer_norm(
        &Tensor::from_arena(&mut a, &[n]),
        None,
        None,
        3.0,
        &mut Tensor::from_arena(&mut big, &[n]),
    );
    let mean = a.iter().sum::<f32>() / n as f32;
    let var = a.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    for i in 0..n {
        assert!(
            (big[i] - plain[i] * (var / (var + 3.0)).sqrt()).abs() < 1e-5,
            "eps {i}"
        );
    }
    tensor_rms_norm(
        &Tensor::from_arena(&mut a, &[n]),
        None,
        0.0,
        &mut Tensor::from_arena(&mut plain, &[n]),
    );
    tensor_rms_norm(
        &Tensor::from_arena(&mut a, &[n]),
        Some(&Tensor::from_arena(&mut w, &[n])),
        0.0,
        &mut Tensor::from_arena(&mut scaled, &[n]),
    );
    for i in 0..n {
        assert!((scaled[i] - plain[i] * w[i]).abs() < 1e-5, "rms_norm {i}");
    }
    tensor_rms_norm(
        &Tensor::from_arena(&mut a, &[n]),
        None,
        3.0,
        &mut Tensor::from_arena(&mut big, &[n]),
    );
    let ms = a.iter().map(|x| x * x).sum::<f32>() / n as f32;
    for i in 0..n {
        assert!(
            (big[i] - plain[i] * (ms / (ms + 3.0)).sqrt()).abs() < 1e-5,
            "rms eps {i}"
        );
    }
}

#[test]
fn softmax_rows_are_independent_and_all_minus_inf_rows_become_uniform() {
    // two rows of a [2, 4] tensor: each row sums to 1, each equals its own 1-D softmax
    let mut a = vec![0.0f32, 1.0, 2.0, 3.0, -1.0, -1.0, 5.0, 0.5];
    let mut o = vec![0.0f32; 8];
    let mut o_fast = vec![0.0f32; 8];
    tensor_softmax(
        &Tensor::from_arena(&mut a, &[2, 4]),
        &mut Tensor::from_arena(&mut o, &[2, 4]),
    );
    tensor_softmax_fast(
        &Tensor::from_arena(&mut a, &[2, 4]),
        &mut Tensor::from_arena(&mut o_fast, &[2, 4]),
    );
    for r in 0..2 {
        let row = &o[r * 4..(r + 1) * 4];
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let mut single = a[r * 4..(r + 1) * 4].to_vec();
        let mut single_o = vec![0.0f32; 4];
        tensor_softmax(
            &Tensor::from_arena(&mut single, &[4]),
            &mut Tensor::from_arena(&mut single_o, &[4]),
        );
        assert_eq!(row, &single_o[..], "row {r}");
        let fast = &o_fast[r * 4..(r + 1) * 4];
        assert!((fast.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        for (p, q) in row.iter().zip(fast) {
            assert!((p - q).abs() <= 0.06 * p, "fast row {r}: {p} vs {q}");
        }
    }
    // −inf everywhere: the fast variant falls back to a uniform row
    let mut neg = vec![f32::NEG_INFINITY; 4];
    let mut u = vec![0.0f32; 4];
    tensor_softmax_fast(
        &Tensor::from_arena(&mut neg, &[4]),
        &mut Tensor::from_arena(&mut u, &[4]),
    );
    assert_eq!(u, vec![0.25; 4]);
}

// ----------------------------------------------------------------------------
// Quantisation statistics and the sparse / QAT variants
// ----------------------------------------------------------------------------

#[test]
fn quant_stats_counts_range_mae_sparsity_and_entropy_follow_their_definitions() {
    // 12 weights: 4 clear positives, 3 clear negatives, 5 near zero
    let w = [
        2.0f32, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, 0.1, -0.1, 0.05, 0.0, 0.2,
    ];
    let (q, s) = quantize_to_ternary(&w, 3, 4);
    let gamma = w.iter().map(|x| x.abs()).sum::<f32>() / 12.0;
    assert!((s.scale - gamma).abs() < 1e-6);
    assert_eq!((s.plus_count, s.minus_count, s.zero_count), (4, 3, 5));
    assert_eq!(s.original_range, (-2.0, 2.0));
    assert!((s.sparsity() - 5.0 / 12.0).abs() < 1e-6);
    let mae: f32 = w
        .iter()
        .zip(dequantize_from_ternary(&q))
        .map(|(x, d)| (x - d).abs())
        .sum::<f32>()
        / 12.0;
    assert!((s.mae - mae).abs() < 1e-6);
    // Shannon entropy of the three symbol probabilities, in bits
    let h = |p: f32| if p > 0.0 { -p * p.log2() } else { 0.0 };
    let entropy = h(4.0 / 12.0) + h(3.0 / 12.0) + h(5.0 / 12.0);
    assert!(
        (s.effective_bits() - entropy).abs() < 1e-5,
        "{} vs {entropy}",
        s.effective_bits()
    );
    // uniform ternary → log2(3); a single symbol → 0 bits; empty → 0
    let uniform = QuantStats {
        plus_count: 5,
        minus_count: 5,
        zero_count: 5,
        ..Default::default()
    };
    assert!((uniform.effective_bits() - 3f32.log2()).abs() < 1e-6);
    let single = QuantStats {
        plus_count: 7,
        ..Default::default()
    };
    assert_eq!(single.effective_bits(), 0.0);
    assert_eq!(single.sparsity(), 0.0);
    assert_eq!(QuantStats::default().effective_bits(), 0.0);
    assert_eq!(QuantStats::default().sparsity(), 0.0);

    // error metrics from the dequantised weights, by definition
    let e = compute_quantization_error(&w, &q);
    let deq = dequantize_from_ternary(&q);
    let errs: Vec<f32> = w.iter().zip(&deq).map(|(x, d)| (x - d).abs()).collect();
    let mse = errs.iter().map(|e| e * e).sum::<f32>() / 12.0;
    assert!((e.mae - mae).abs() < 1e-6);
    assert!((e.mse - mse).abs() < 1e-6);
    assert!((e.rmse - mse.sqrt()).abs() < 1e-6);
    assert_eq!(e.max_error, errs.iter().copied().fold(0.0, f32::max));
    let signal = w.iter().map(|x| x * x).sum::<f32>();
    let noise = errs.iter().map(|e| e * e).sum::<f32>();
    assert!((e.snr - 10.0 * (signal / noise).log10()).abs() < 1e-4);
    // exact reconstruction (γ given as 1) → infinite SNR
    let exact = [1.0f32, -1.0, 0.0, 1.0];
    let (q, _) = quantize_to_ternary_qat(&exact, 1, 4, 1.0, 1.0);
    assert_eq!(compute_quantization_error(&exact, &q).snr, f32::INFINITY);
}

#[test]
fn sparse_quantisation_zeroes_exactly_the_weights_below_threshold_times_scale() {
    let w = [3.0f32, -3.0, 1.0, -1.0, 0.5, -0.5, 0.0, 2.0];
    let gamma = w.iter().map(|x| x.abs()).sum::<f32>() / 8.0; // 11/8 = 1.375
    for threshold in [0.0f32, 0.5, 0.8, 1.0, 1.5, 3.0] {
        let (q, s) = quantize_to_ternary_sparse(&w, 2, 4, threshold);
        let cutoff = threshold * gamma;
        let mut expected_plus = 0;
        let mut expected_minus = 0;
        let mut expected_zero = 0;
        for (i, &x) in w.iter().enumerate() {
            let t = if x.abs() < cutoff {
                0
            } else if x > 0.0 {
                1
            } else {
                -1
            };
            assert_eq!(
                q.get(i / 4, i % 4).to_i8(),
                t,
                "threshold {threshold} weight {x}"
            );
            match t {
                1 => expected_plus += 1,
                -1 => expected_minus += 1,
                _ => expected_zero += 1,
            }
        }
        assert_eq!(
            (s.plus_count, s.minus_count, s.zero_count),
            (expected_plus, expected_minus, expected_zero)
        );
        assert!((s.scale - gamma).abs() < 1e-6);
        assert_eq!(s.original_range, (-3.0, 3.0));
        let deq = dequantize_from_ternary(&q);
        let mae = w.iter().zip(&deq).map(|(x, d)| (x - d).abs()).sum::<f32>() / 8.0;
        assert!((s.mae - mae).abs() < 1e-6, "threshold {threshold}");
    }
    // threshold 0 keeps every non-zero weight; an exact 0.0 is 0 for x > 0.0 false and x < cutoff false → −1 branch? No: 0.0 < 0.0 is false and 0.0 > 0.0 is false, so the sign branch gives −1
    let (q, _) = quantize_to_ternary_sparse(&[0.0f32, 1.0], 1, 2, 0.0);
    assert_eq!(q.get(0, 0).to_i8(), -1);
}

#[test]
fn qat_quantisation_reports_stats_like_the_plain_law_with_the_given_scale() {
    let w = [0.9f32, -0.9, 0.3, -0.3, 0.0, 1.7, -1.2, 0.45];
    let (gamma, tau) = (1.0f32, 1.0f32);
    let (q, s) = quantize_to_ternary_qat(&w, 2, 4, gamma, tau);
    assert_eq!(s.scale, gamma);
    assert_eq!(s.original_range, (-1.2, 1.7));
    // q = round(w / γ / τ) clamped: 0.9 → 1, 0.3 → 0, 0.45 → 0, 1.7 → 1, −1.2 → −1
    let expected = [1i8, -1, 0, 0, 0, 1, -1, 0];
    for (i, &t) in expected.iter().enumerate() {
        assert_eq!(q.get(i / 4, i % 4).to_i8(), t, "weight {}", w[i]);
    }
    assert_eq!((s.plus_count, s.minus_count, s.zero_count), (2, 2, 4));
    let mae = w
        .iter()
        .zip(expected)
        .map(|(x, t)| (x - f32::from(t) * gamma).abs())
        .sum::<f32>()
        / 8.0;
    assert!((s.mae - mae).abs() < 1e-6, "{} vs {mae}", s.mae);
    // a temperature of 2 doubles the rounding threshold: 0.9 / 2 = 0.45 → 0
    let (q2, s2) = quantize_to_ternary_qat(&w, 2, 4, gamma, 2.0);
    assert_eq!(q2.get(0, 0).to_i8(), 0);
    assert_eq!(q2.get(1, 1).to_i8(), 1); // 1.7 / 2 = 0.85 → 1
    assert_eq!((s2.plus_count, s2.minus_count, s2.zero_count), (1, 1, 6));
}

// ----------------------------------------------------------------------------
// Bit-plane kernel accessors and the packed-byte mask helpers
// ----------------------------------------------------------------------------

#[test]
fn kernel_bit_planes_are_the_ternary_pattern_and_scale_is_stored() {
    let values = pattern(2, 40); // [+1, −1, 0] repeating, 2 words per row
    let k = TernaryWeightKernel::from_ternary_scaled(&values, 2, 40, 0.75);
    assert_eq!(k.scale(), 0.75);
    assert_eq!(k.words_per_row(), 2);
    assert_eq!(k.plus_bits().len(), 4);
    assert_eq!(k.minus_bits().len(), 4);
    for r in 0..2 {
        for c in 0..40 {
            let word = r * 2 + c / 32;
            let bit = c % 32;
            let plus = (k.plus_bits()[word] >> bit) & 1 == 1;
            let minus = (k.minus_bits()[word] >> bit) & 1 == 1;
            let v = values[r * 40 + c];
            assert_eq!((plus, minus), (v == 1, v == -1), "({r},{c})");
        }
    }
    // the packed 2-bit form: 01 = +1, 10 = −1, 00 = 0, four per byte, low first
    let packed = TernaryWeight::from_ternary(&[1, -1, 0, 1], 1, 4);
    assert_eq!(packed.packed(), &[0b0100_1001]);
    assert_eq!(alice_ml::ops::extract_plus_mask(0b0100_1001), 0b1001);
    assert_eq!(alice_ml::ops::extract_minus_mask(0b0100_1001), 0b0010);
    for byte in 0..=255u8 {
        let plus = alice_ml::ops::extract_plus_mask(byte);
        let minus = alice_ml::ops::extract_minus_mask(byte);
        for i in 0..4 {
            let field = (byte >> (2 * i)) & 0b11;
            assert_eq!(
                (plus >> i) & 1,
                u8::from(field == 0b01),
                "byte {byte:#010b} field {i}"
            );
            assert_eq!(
                (minus >> i) & 1,
                u8::from(field == 0b10),
                "byte {byte:#010b} field {i}"
            );
        }
    }
}

#[test]
fn matmul_alloc_matches_the_batched_kernel_for_1d_and_2d_inputs() {
    use alice_ml::{ternary_matmul_alloc, ternary_matvec_alloc};
    let (out_features, in_features) = (3usize, 33usize);
    let values = pattern(out_features, in_features);
    let packed = TernaryWeight::from_ternary(&values, out_features, in_features);
    let mut x = int_input(in_features);
    let expected = oracle_matvec(&values, out_features, in_features, &x, 1.0);
    let t1 = Tensor::from_arena(&mut x, &[in_features]);
    assert_eq!(ternary_matvec_alloc(&t1, &packed).data(), &expected[..]);
    let one = ternary_matmul_alloc(&t1, &packed);
    assert_eq!(one.shape(), &[out_features]);
    assert_eq!(one.data(), &expected[..]);
    // a [1, in] input keeps the 1-D output shape; [b, in] gives [b, out]
    let mut x1 = int_input(in_features);
    let one_row = ternary_matmul_alloc(&Tensor::from_arena(&mut x1, &[1, in_features]), &packed);
    assert_eq!(one_row.shape(), &[out_features]);
    assert_eq!(one_row.data(), &expected[..]);
    let batch = 4usize;
    let mut xb: Vec<f32> = (0..batch)
        .flat_map(|b| {
            int_input(in_features)
                .into_iter()
                .map(move |v| v + b as f32)
        })
        .collect();
    let out = ternary_matmul_alloc(&Tensor::from_arena(&mut xb, &[batch, in_features]), &packed);
    assert_eq!(out.shape(), &[batch, out_features]);
    for b in 0..batch {
        let xrow = &xb[b * in_features..(b + 1) * in_features];
        let e = oracle_matvec(&values, out_features, in_features, xrow, 1.0);
        assert_eq!(
            &out.data()[b * out_features..(b + 1) * out_features],
            &e[..],
            "row {b}"
        );
    }
}

// ----------------------------------------------------------------------------
// Ternary Llama-3 forward on a 4-dimensional toy model: every stage is driven
// through identity / zero projections so the logits have a closed form
// ----------------------------------------------------------------------------

#[cfg(feature = "safetensors")]
mod ternary_llama {
    use alice_ml::llama3_ternary::{Llama3TernaryConfig, Llama3TernaryModel};
    use alice_ml::model_io::ModelArchive;
    use alice_ml::TernaryWeight;

    const H: usize = 4; // hidden = vocab = intermediate
    const KV: usize = 2; // num_kv_heads (1) × head_dim (2)

    const fn config() -> Llama3TernaryConfig {
        Llama3TernaryConfig {
            vocab_size: H,
            hidden_dim: H,
            intermediate_dim: H,
            num_heads: 2,
            num_kv_heads: 1,
            num_layers: 1,
            max_seq_len: 16,
            head_dim: 2,
            rope_theta: 10_000.0,
            norm_eps: 0.0,
        }
    }

    fn zero(out: usize, inp: usize) -> TernaryWeight {
        TernaryWeight::from_ternary(&vec![0i8; out * inp], out, inp)
    }

    /// `out × inp` matrix with +1 on the diagonal (row i selects input i)
    fn identity(out: usize, inp: usize) -> TernaryWeight {
        let v: Vec<i8> = (0..out * inp)
            .map(|k| i8::from(k / inp == k % inp))
            .collect();
        TernaryWeight::from_ternary(&v, out, inp)
    }

    fn embedding() -> Vec<f32> {
        // token t → row t, rows distinct and non-degenerate
        (0..H * H)
            .map(|k| (k % H) as f32 + 1.0 + 4.0 * (k / H) as f32)
            .collect()
    }

    fn rms_norm(x: &[f32]) -> Vec<f32> {
        let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        x.iter().map(|v| v / ms.sqrt()).collect()
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn model(projs: Vec<TernaryWeight>) -> Llama3TernaryModel {
        Llama3TernaryModel::from_parts(
            config(),
            embedding(),
            vec![1.0; H],
            identity(H, H),
            vec![(vec![1.0; H], vec![1.0; H], projs)],
        )
    }

    fn assert_close(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < 1e-4, "{what}[{i}]: {x} vs {y}");
        }
    }

    #[test]
    fn zero_projections_leave_the_embedding_and_logits_are_its_rms_norm() {
        // attention and FFN contribute nothing → hidden = embedding row →
        // logits = output_proj(identity) · rms_norm(hidden)
        let projs = vec![
            zero(H, H),
            zero(KV, H),
            zero(KV, H),
            zero(H, H),
            zero(H, H),
            zero(H, H),
            zero(H, H),
        ];
        let mut m = model(projs);
        for token in 0..4u32 {
            m.clear_cache();
            let row = &embedding()[token as usize * H..(token as usize + 1) * H];
            assert_close(&m.forward(token), &rms_norm(row), &format!("token {token}"));
        }
        // the KV cache grows but zero projections make every position identical
        m.clear_cache();
        let first = m.forward(2);
        let second = m.forward(2);
        assert_close(
            &second,
            &first,
            "position independence with zero projections",
        );
    }

    #[test]
    fn identity_value_and_output_projections_add_the_normed_hidden_through_attention() {
        // q = k = 0 → uniform softmax; v = first KV components of the normed
        // hidden; both heads read kv head 0; o_proj = identity → hidden +=
        // [n0, n1, n0, n1]; FFN zero
        let projs = vec![
            zero(H, H),
            zero(KV, H),
            identity(KV, H),
            identity(H, H),
            zero(H, H),
            zero(H, H),
            zero(H, H),
        ];
        let mut m = model(projs);
        let token = 1u32;
        let row = &embedding()[H..2 * H];
        let n = rms_norm(row);
        let hidden: Vec<f32> = (0..H).map(|i| row[i] + n[i % 2]).collect();
        let expected = rms_norm(&hidden);
        assert_close(&m.forward(token), &expected, "one token");
        // a second identical token: two cached positions with identical K (= 0)
        // and identical V → the same average → the same logits
        assert_close(&m.forward(token), &expected, "two tokens");
        m.clear_cache();
        assert_close(&m.forward(token), &expected, "after clear_cache");
    }

    #[test]
    fn identity_ffn_adds_silu_gate_times_up_of_the_normed_hidden() {
        // attention zero; gate = up = down = identity → hidden += silu(n) ⊙ n
        let projs = vec![
            zero(H, H),
            zero(KV, H),
            zero(KV, H),
            zero(H, H),
            identity(H, H),
            identity(H, H),
            identity(H, H),
        ];
        let mut m = model(projs);
        let row = &embedding()[2 * H..3 * H];
        let n = rms_norm(row);
        let hidden: Vec<f32> = (0..H).map(|i| row[i] + silu(n[i]) * n[i]).collect();
        assert_close(&m.forward(2), &rms_norm(&hidden), "ffn");
    }

    #[test]
    fn memory_bytes_is_the_sum_of_its_parts_and_atml_holds_every_projection() {
        let projs = vec![
            zero(H, H),
            zero(KV, H),
            zero(KV, H),
            zero(H, H),
            zero(H, H),
            zero(H, H),
            zero(H, H),
        ];
        let expected_bytes = H * H * 4 // embedding
            + H * 4 // output norm
            + identity(H, H).memory_bytes()
            + 2 * H * 4 // two norm weights
            + projs.iter().map(TernaryWeight::memory_bytes).sum::<usize>();
        let m = model(projs);
        assert_eq!(m.memory_bytes(), expected_bytes);
        let atml = m.save_atml();
        let archive = ModelArchive::deserialize(&atml).expect("atml parses");
        // output_proj + 7 per layer
        assert!(archive.get_layer(7).is_some());
        assert!(archive.get_layer(8).is_none());
        let out = archive.get_layer(0).unwrap();
        assert_eq!((out.out_features, out.in_features), (H, H));
        assert_eq!(out.packed, identity(H, H).packed());
        let k = archive.get_layer(2).unwrap();
        assert_eq!((k.out_features, k.in_features), (KV, H));
    }

    #[test]
    #[should_panic(expected = "need 7 projections per layer")]
    fn from_parts_rejects_a_layer_with_fewer_than_seven_projections() {
        let _ = model(vec![zero(H, H); 6]);
    }
}

// ----------------------------------------------------------------------------
// safetensors: files are synthesised from the format definition
// ([u64 LE header length][JSON header][data]) so parsing is checked against the
// values that went in, and malformed headers are errors, never panics
// ----------------------------------------------------------------------------

#[cfg(feature = "safetensors")]
mod safetensors_format {
    use alice_ml::safetensors::{bf16_to_f32, fp16_to_f32, DType, SafetensorsFile};

    fn file(header: &str, data: &[u8]) -> Vec<u8> {
        let mut v = (header.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(header.as_bytes());
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn f32_f16_bf16_tensors_read_back_exactly_and_metadata_is_skipped() {
        let f32s = [1.5f32, -2.25, 0.0, 1e-3, 3.0e8];
        let f16s: [u16; 4] = [0x3C00, 0xC000, 0x0001, 0x7C00]; // 1.0, −2.0, smallest subnormal, +inf
        let bf16s: [u16; 3] = [0x3F80, 0xC040, 0x0000]; // 1.0, −3.0, 0.0
        let mut data = Vec::new();
        for v in f32s {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let f16_start = data.len();
        for v in f16s {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let bf16_start = data.len();
        for v in bf16s {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let header = format!(
            concat!(
                "{{\"__metadata__\":{{\"format\":\"pt\",\"nested\":{{\"a\":[1,2,{{\"b\":\"}}\"}}]}}}},",
                "\"a.weight\":{{\"dtype\":\"F32\",\"shape\":[5],\"data_offsets\":[0,{}]}},",
                "\"b.weight\":{{\"dtype\":\"F16\",\"shape\":[2,2],\"data_offsets\":[{},{}]}},",
                "\"c.bias\":{{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[{},{}]}},",
                "\"d.other\":{{\"dtype\":\"I8\",\"shape\":[2],\"data_offsets\":[0,2]}}}}"
            ),
            f16_start,
            f16_start,
            bf16_start,
            bf16_start,
            data.len()
        );
        let bytes = file(&header, &data);
        let sf = SafetensorsFile::parse(&bytes).expect("well-formed file parses");
        assert_eq!(sf.len(), 4);
        assert!(!sf.is_empty());
        let mut names = sf.tensor_names();
        names.sort_unstable();
        assert_eq!(names, vec!["a.weight", "b.weight", "c.bias", "d.other"]);

        let a = sf.tensor_desc("a.weight").unwrap();
        assert_eq!(
            (a.dtype, &a.shape[..], a.n_elements(), a.data_size()),
            (DType::F32, &[5][..], 5, 20)
        );
        assert_eq!(sf.tensor_to_f32("a.weight").unwrap(), f32s.to_vec());
        assert_eq!(sf.tensor_bytes("a.weight").unwrap(), &data[..20]);

        let b = sf.tensor_desc("b.weight").unwrap();
        assert_eq!((b.dtype, b.n_elements(), b.data_size()), (DType::F16, 4, 8));
        let expect_f16: Vec<f32> = f16s.iter().map(|&h| fp16_to_f32(h)).collect();
        assert_eq!(sf.tensor_to_f32("b.weight").unwrap(), expect_f16);
        assert_eq!(expect_f16[0], 1.0);
        assert_eq!(expect_f16[1], -2.0);
        assert_eq!(expect_f16[2], 2f32.powi(-24)); // smallest f16 subnormal
        assert!(expect_f16[3].is_infinite() && expect_f16[3] > 0.0);

        let c = sf.tensor_desc("c.bias").unwrap();
        assert_eq!((c.dtype, c.n_elements()), (DType::BF16, 3));
        assert_eq!(sf.tensor_to_f32("c.bias").unwrap(), vec![1.0, -3.0, 0.0]);
        assert_eq!(bf16_to_f32(0x3F80), 1.0);

        // unsupported dtype: descriptor is kept, conversion is refused
        let d = sf.tensor_desc("d.other").unwrap();
        assert_eq!((d.dtype, d.dtype.element_size()), (DType::Other, 0));
        assert!(sf.tensor_to_f32("d.other").is_none());
        assert!(sf.tensor_to_f32("missing").is_none());
        assert!(sf.tensor_desc("missing").is_none());
        assert_eq!(
            DType::F32.element_size()
                + DType::F16.element_size()
                + DType::BF16.element_size()
                + DType::F64.element_size(),
            4 + 2 + 2 + 8
        );
    }

    #[test]
    fn half_precision_conversions_follow_ieee_754() {
        // fp16: sign / 5-bit exponent (bias 15) / 10-bit mantissa
        for (bits, value) in [
            (0x0000u16, 0.0f32),
            (0x8000, -0.0),
            (0x3C00, 1.0),
            (0x3E00, 1.5),
            (0x4900, 10.0),
            (0x7BFF, 65504.0),                            // max finite
            (0x0400, 2f32.powi(-14)),                     // min normal
            (0x03FF, 2f32.powi(-14) * (1023.0 / 1024.0)), // max subnormal
            (0xFC00, f32::NEG_INFINITY),
        ] {
            let got = fp16_to_f32(bits);
            assert!(
                got == value && got.is_sign_negative() == value.is_sign_negative(),
                "fp16 {bits:#06x}: {got} vs {value}"
            );
        }
        assert!(fp16_to_f32(0x7E00).is_nan());
        // bf16 is the top half of an f32
        for bits in [0x3F80u16, 0xBF80, 0x4049, 0x0080, 0x7F80, 0x0000] {
            assert_eq!(
                bf16_to_f32(bits).to_bits(),
                u32::from(bits) << 16,
                "bf16 {bits:#06x}"
            );
        }
        assert!(bf16_to_f32(0x7FC0).is_nan());
    }

    #[test]
    fn malformed_files_are_rejected_without_panicking() {
        // shorter than the length prefix / header longer than the file
        assert!(SafetensorsFile::parse(&[0u8; 7]).is_none());
        assert!(SafetensorsFile::parse(&file("{}", &[])[..9]).is_none());
        let mut huge = 1_000_000u64.to_le_bytes().to_vec();
        huge.extend_from_slice(b"{}");
        assert!(SafetensorsFile::parse(&huge).is_none());
        // header must be an object
        assert!(SafetensorsFile::parse(&file("[]", &[])).is_none());
        assert!(SafetensorsFile::parse(&file("", &[])).is_none());
        // an empty object is a valid file with no tensors
        let bytes_empty = file("{}", &[]);
        let empty = SafetensorsFile::parse(&bytes_empty).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        // a tensor whose descriptor is not an object is skipped, not fatal
        let bytes_sf = file(
            "{\"x\":\"junk\",\"y\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
            &1.0f32.to_le_bytes(),
        );
        let sf = SafetensorsFile::parse(&bytes_sf).unwrap();
        assert_eq!(sf.len(), 1);
        assert_eq!(sf.tensor_to_f32("y").unwrap(), vec![1.0]);
        // data section shorter than the declared range → None
        let bytes_sf = file(
            "{\"y\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}",
            &[0u8; 8],
        );
        let sf = SafetensorsFile::parse(&bytes_sf).unwrap();
        assert!(sf.tensor_bytes("y").is_none());
        assert!(sf.tensor_to_f32("y").is_none());
        // shape × element size larger than the byte range → None, not an index panic
        let bytes_sf = file(
            "{\"y\":{\"dtype\":\"F32\",\"shape\":[1000],\"data_offsets\":[0,4]}}",
            &[0u8; 4],
        );
        let sf = SafetensorsFile::parse(&bytes_sf).unwrap();
        assert!(sf.tensor_bytes("y").is_some());
        assert!(sf.tensor_to_f32("y").is_none());
        // inverted offsets → None
        let bytes_sf = file(
            "{\"y\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,4]}}",
            &[0u8; 8],
        );
        let sf = SafetensorsFile::parse(&bytes_sf).unwrap();
        assert!(sf.tensor_bytes("y").is_none());
        // absurd shape does not overflow n_elements
        let bytes_sf = file("{\"y\":{\"dtype\":\"F32\",\"shape\":[4294967296,4294967296,4294967296],\"data_offsets\":[0,4]}}", &[0u8; 4]);
        let sf = SafetensorsFile::parse(&bytes_sf).unwrap();
        assert_eq!(sf.tensor_desc("y").unwrap().n_elements(), usize::MAX);
        assert!(sf.tensor_to_f32("y").is_none());
        // non-numeric shape entry is a parse failure
        assert!(SafetensorsFile::parse(&file(
            "{\"y\":{\"dtype\":\"F32\",\"shape\":[\"a\"],\"data_offsets\":[0,4]}}",
            &[0u8; 4]
        ))
        .is_none());
        // truncated header (unterminated object) is a parse failure
        assert!(SafetensorsFile::parse(&file("{\"y\":{\"dtype\":\"F32\"", &[])).is_none());
    }
}
