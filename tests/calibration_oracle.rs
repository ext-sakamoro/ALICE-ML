//! Closed-form oracles for `alice_ml::calibration`
//!
//! Every expectation is derived from the definition, not read off a run:
//! the Brier score of a uniform distribution over `k` classes is `1 - 1/k`, the
//! NLL of a uniform distribution is `ln k`, a perfectly calibrated set has an
//! ECE of exactly zero, and pool-adjacent-violators merges a violating pair into
//! their mean. Where a value is a dyadic fraction the assertion is `==`, not a
//! tolerance
//!
//! The fitting tests are held to the same standard. A set whose label counts
//! match `softmax(z)` exactly is calibrated at `T = 1` by construction, so
//! stretching its logits by `s` puts the optimum at exactly `T = s` — that is an
//! analytic statement about the maximum likelihood estimate, not a number read
//! off a run. The weaker claim (a fit is never worse than the point it started
//! from) is asserted separately, because it must hold for *any* input

#![allow(clippy::float_cmp)] // dyadic fractions are compared exactly on purpose
#![allow(clippy::cast_precision_loss)] // small counts → f32 / f64 (tests/analytic_oracle.rs と同じ慣行)

use alice_ml::calibration::{
    brier_binary, brier_multiclass, negative_log_likelihood, CalibrationError, IsotonicRegression,
    ReliabilityBins, TemperatureScaling, VectorScaling,
};

// ---------------------------------------------------------------------------
// Brier
// ---------------------------------------------------------------------------

#[test]
fn brier_binary_is_the_squared_error() {
    assert_eq!(brier_binary(1.0, true).unwrap(), 0.0);
    assert_eq!(brier_binary(0.0, false).unwrap(), 0.0);
    assert_eq!(brier_binary(0.0, true).unwrap(), 1.0);
    assert_eq!(brier_binary(1.0, false).unwrap(), 1.0);
    // 0.5 is exact in binary floating point, so this is an equality
    assert_eq!(brier_binary(0.5, true).unwrap(), 0.25);
    assert_eq!(brier_binary(0.5, false).unwrap(), 0.25);
    assert_eq!(brier_binary(0.25, false).unwrap(), 0.0625);

    assert_eq!(
        brier_binary(f32::NAN, true),
        Err(CalibrationError::NotAProbability)
    );
    assert_eq!(
        brier_binary(1.5, true),
        Err(CalibrationError::NotAProbability)
    );
}

#[test]
fn brier_multiclass_of_a_uniform_distribution_is_one_minus_one_over_k() {
    // sum_k (p_k - y_k)^2 with p_k = 1/k is (1 - 1/k)^2 + (k-1)(1/k)^2 = 1 - 1/k
    for k in [2_usize, 4, 8] {
        let p = 1.0_f32 / k as f32;
        let probs = vec![p; k];
        let expected = 1.0_f32 - p;
        assert_eq!(
            brier_multiclass(&probs, 0).unwrap(),
            expected,
            "uniform over {k}"
        );
        // どのラベルでも同じ (対称性)
        assert_eq!(brier_multiclass(&probs, k - 1).unwrap(), expected);
    }
}

#[test]
fn brier_multiclass_spans_zero_to_two() {
    let certain_right = [1.0_f32, 0.0, 0.0];
    assert_eq!(brier_multiclass(&certain_right, 0).unwrap(), 0.0);
    // 確信して外すと 2.0 — 置くべきでない所に 1、置くべき所に 0 で 2 回間違える
    assert_eq!(brier_multiclass(&certain_right, 1).unwrap(), 2.0);
}

#[test]
fn brier_multiclass_rejects_what_is_not_a_distribution() {
    assert_eq!(
        brier_multiclass(&[], 0),
        Err(CalibrationError::EmptyDistribution)
    );
    assert_eq!(
        brier_multiclass(&[0.5, 0.5], 2),
        Err(CalibrationError::LabelOutOfRange {
            label: 2,
            classes: 2
        })
    );
    assert!(
        brier_multiclass(&[0.5, 0.6], 0).is_ok(),
        "合計 1 でなくてもよい — Brier は各成分の二乗誤差の和"
    );
    assert_eq!(
        brier_multiclass(&[1.5, -0.5], 0),
        Err(CalibrationError::NotAProbability)
    );
}

// ---------------------------------------------------------------------------
// NLL
// ---------------------------------------------------------------------------

#[test]
fn nll_of_a_uniform_distribution_is_ln_k() {
    for k in [2_usize, 4, 10] {
        let probs = vec![1.0_f32 / k as f32; k];
        let got = negative_log_likelihood(&probs, 0).unwrap();
        let expected = (k as f32).ln();
        assert!(
            (got - expected).abs() < 1e-6,
            "uniform over {k}: got {got}, expected ln({k}) = {expected}"
        );
    }
}

#[test]
fn nll_is_zero_when_the_truth_was_certain() {
    assert_eq!(negative_log_likelihood(&[1.0, 0.0], 0).unwrap(), 0.0);
}

#[test]
fn nll_is_large_but_finite_when_the_truth_was_impossible() {
    // ln(0) は -inf だが、1 サンプルで平均が無限になるのは使い物にならない
    let got = negative_log_likelihood(&[0.0, 1.0], 0).unwrap();
    assert!(got.is_finite(), "got {got}");
    assert!(got > 25.0, "十分大きな罰でなければ意味がない: {got}");
}

// ---------------------------------------------------------------------------
// ECE / MCE
// ---------------------------------------------------------------------------

#[test]
fn a_perfectly_calibrated_set_has_exactly_zero_error() {
    let mut bins = ReliabilityBins::with_bins(10).unwrap();
    // 各 bin で「確信度 == 正解率」になるように積む 値は全て 2 の冪分数
    for (confidence, n, correct) in [(0.25_f32, 4_u32, 1_u32), (0.5, 2, 1), (0.75, 4, 3)] {
        for i in 0..n {
            bins.observe(confidence, i < correct).unwrap();
        }
    }
    assert_eq!(bins.expected_calibration_error(), 0.0);
    assert_eq!(bins.maximum_calibration_error(), 0.0);
    assert_eq!(bins.total(), 10);
}

#[test]
fn a_maximally_overconfident_set_has_error_one() {
    let mut bins = ReliabilityBins::with_bins(10).unwrap();
    for _ in 0..8 {
        bins.observe(1.0, false).unwrap();
    }
    assert_eq!(bins.expected_calibration_error(), 1.0);
    assert_eq!(bins.maximum_calibration_error(), 1.0);
}

#[test]
fn the_maximum_error_is_never_below_the_expected_one() {
    let mut bins = ReliabilityBins::with_bins(5).unwrap();
    for (c, ok) in [
        (0.1_f32, false),
        (0.3, true),
        (0.5, false),
        (0.7, true),
        (0.9, true),
        (0.9, false),
    ] {
        bins.observe(c, ok).unwrap();
    }
    assert!(bins.maximum_calibration_error() >= bins.expected_calibration_error());
}

#[test]
fn an_empty_set_has_no_error_rather_than_an_undefined_one() {
    let bins = ReliabilityBins::with_bins(10).unwrap();
    assert!(bins.is_empty());
    assert_eq!(bins.expected_calibration_error(), 0.0);
    assert_eq!(bins.maximum_calibration_error(), 0.0);
}

#[test]
fn confidence_one_lands_in_the_last_bin_rather_than_past_it() {
    let mut bins = ReliabilityBins::with_bins(4).unwrap();
    bins.observe(1.0, true).unwrap();
    let rows = bins.bins();
    assert_eq!(rows[3].count, 1, "bins: {rows:?}");
    assert_eq!(rows[0].count, 0);
}

#[test]
fn bin_count_is_bounded() {
    assert_eq!(
        ReliabilityBins::with_bins(0).err(),
        Some(CalibrationError::BinCountOutOfRange { got: 0 })
    );
    assert!(ReliabilityBins::with_bins(1).is_ok());
    assert!(ReliabilityBins::with_bins(ReliabilityBins::MAX_BINS).is_ok());
    assert!(ReliabilityBins::with_bins(ReliabilityBins::MAX_BINS + 1).is_err());
}

// ---------------------------------------------------------------------------
// Temperature scaling
// ---------------------------------------------------------------------------

#[test]
fn the_identity_temperature_is_a_plain_softmax() {
    // 等しい logits は温度に関係なく一様分布
    for k in [2_usize, 4] {
        let logits = vec![0.0_f32; k];
        let mut out = vec![0.0_f32; k];
        TemperatureScaling::IDENTITY
            .apply(&logits, &mut out)
            .unwrap();
        let expected = 1.0_f32 / k as f32;
        for p in &out {
            assert_eq!(*p, expected, "uniform over {k}: {out:?}");
        }
    }
    assert_eq!(TemperatureScaling::IDENTITY.temperature(), 1.0);
}

#[test]
fn a_higher_temperature_flattens_and_a_lower_one_sharpens() {
    let logits = [3.0_f32, 1.0, 0.0];
    let mut hot = [0.0_f32; 3];
    let mut warm = [0.0_f32; 3];
    let mut cold = [0.0_f32; 3];
    TemperatureScaling::try_new(4.0)
        .unwrap()
        .apply(&logits, &mut hot)
        .unwrap();
    TemperatureScaling::IDENTITY
        .apply(&logits, &mut warm)
        .unwrap();
    TemperatureScaling::try_new(0.25)
        .unwrap()
        .apply(&logits, &mut cold)
        .unwrap();

    assert!(hot[0] < warm[0], "熱いほど平ら: {hot:?} vs {warm:?}");
    assert!(cold[0] > warm[0], "冷たいほど尖る: {cold:?} vs {warm:?}");
    // 順位は温度で変わらない — これが温度スケーリングを使う理由
    for out in [&hot, &warm, &cold] {
        assert!(out[0] > out[1] && out[1] > out[2], "{out:?}");
    }
}

#[test]
fn temperature_is_validated() {
    assert_eq!(
        TemperatureScaling::try_new(0.0).err(),
        Some(CalibrationError::TemperatureOutOfRange)
    );
    assert_eq!(
        TemperatureScaling::try_new(-1.0).err(),
        Some(CalibrationError::TemperatureOutOfRange)
    );
    assert_eq!(
        TemperatureScaling::try_new(f32::INFINITY).err(),
        Some(CalibrationError::TemperatureOutOfRange)
    );
}

fn mean_nll(samples: &[(&[f32], usize)], scaler: TemperatureScaling) -> f64 {
    let mut acc = 0.0_f64;
    for (logits, label) in samples {
        let mut out = vec![0.0_f32; logits.len()];
        scaler.apply(logits, &mut out).unwrap();
        acc += f64::from(negative_log_likelihood(&out, *label).unwrap());
    }
    acc / samples.len() as f64
}

/// **厳密に校正済の集合を 3 倍に引き伸ばした**データ
///
/// 各 logit パターンについて、ラベルの出現回数を `softmax(z)` に一致させて
/// 割り当てる (= 経験分布が模型の分布と厳密に一致する) この集合は定義から
/// `T = 1` で校正済なので、logits を `s` 倍したものの最適温度は解析的に `s`
/// になる — `softmax(s*z/T) = softmax(z)` が `T = s` でのみ成り立つため
///
/// 最初はラベルを適当に振った集合でこれを試し、最適温度が 3 から大きく外れた
/// ここで閾値を緩めるのは「test を実装に合わせる」ことになる 誤っていたのは
/// oracle の前提 (= 元集合が校正済であること) の方だったので、集合を作り直した
fn calibrated_then_stretched(scale: f32) -> Vec<(Vec<f32>, usize)> {
    let ln2 = 2.0_f32.ln();
    let ln3 = 3.0_f32.ln();
    let ln6 = 6.0_f32.ln();

    // exp = [6, 3, 1] / 10 → p = [0.6, 0.3, 0.1] なので 6 / 3 / 1 件
    // exp = [2, 2, 1] /  5 → p = [0.4, 0.4, 0.2] なので 2 / 2 / 1 件
    let patterns: [([f32; 3], [usize; 3]); 2] =
        [([ln6, ln3, 0.0], [6, 3, 1]), ([ln2, ln2, 0.0], [2, 2, 1])];

    let mut out = Vec::new();
    for (z, counts) in patterns {
        let stretched: Vec<f32> = z.iter().map(|v| v * scale).collect();
        for (label, n) in counts.iter().enumerate() {
            for _ in 0..*n {
                out.push((stretched.clone(), label));
            }
        }
    }
    out
}

/// 3 倍に引き伸ばした = 過信した集合
fn overconfident_samples() -> Vec<(Vec<f32>, usize)> {
    calibrated_then_stretched(3.0)
}

#[test]
fn fitting_never_does_worse_than_the_identity() {
    let owned = overconfident_samples();
    let samples: Vec<(&[f32], usize)> = owned.iter().map(|(z, y)| (z.as_slice(), *y)).collect();

    let fitted = TemperatureScaling::fit(&samples).unwrap();
    let before = mean_nll(&samples, TemperatureScaling::IDENTITY);
    let after = mean_nll(&samples, fitted);
    assert!(
        after <= before,
        "最小化器が出発点より悪くなってはいけない: {after} vs {before} (T = {})",
        fitted.temperature()
    );
}

#[test]
fn fitting_recovers_the_scale_the_logits_were_stretched_by() {
    let owned = overconfident_samples();
    let samples: Vec<(&[f32], usize)> = owned.iter().map(|(z, y)| (z.as_slice(), *y)).collect();
    let fitted = TemperatureScaling::fit(&samples).unwrap();
    // 元集合が厳密に校正済なので、最適温度は解析的に引き伸ばし率そのもの
    assert!(
        (fitted.temperature() - 3.0).abs() < 0.01,
        "T = {} (期待値 3.0)",
        fitted.temperature()
    );
}

#[test]
fn fitting_is_reproducible() {
    let owned = overconfident_samples();
    let samples: Vec<(&[f32], usize)> = owned.iter().map(|(z, y)| (z.as_slice(), *y)).collect();
    let a = TemperatureScaling::fit(&samples).unwrap();
    let b = TemperatureScaling::fit(&samples).unwrap();
    assert_eq!(a.temperature(), b.temperature());
}

#[test]
fn fitting_needs_samples() {
    assert_eq!(
        TemperatureScaling::fit(&[]).err(),
        Some(CalibrationError::NoSamples)
    );
}

// ---------------------------------------------------------------------------
// Vector scaling
// ---------------------------------------------------------------------------

#[test]
fn vector_scaling_identity_is_a_plain_softmax() {
    let identity = VectorScaling::identity(3).unwrap();
    assert_eq!(identity.slopes(), &[1.0, 1.0, 1.0]);
    assert_eq!(identity.offsets(), &[0.0, 0.0, 0.0]);

    let logits = [0.0_f32, 0.0, 0.0];
    let mut out = [0.0_f32; 3];
    identity.apply(&logits, &mut out).unwrap();
    let mut reference = [0.0_f32; 3];
    TemperatureScaling::IDENTITY
        .apply(&logits, &mut reference)
        .unwrap();
    assert_eq!(out, reference);
}

#[test]
fn zero_steps_leaves_the_identity_untouched() {
    let owned = overconfident_samples();
    let samples: Vec<(&[f32], usize)> = owned.iter().map(|(z, y)| (z.as_slice(), *y)).collect();
    let fitted = VectorScaling::fit(&samples, 0, 0.1).unwrap();
    assert_eq!(fitted, VectorScaling::identity(3).unwrap());
}

#[test]
fn gradient_descent_reduces_the_loss() {
    let owned = overconfident_samples();
    let samples: Vec<(&[f32], usize)> = owned.iter().map(|(z, y)| (z.as_slice(), *y)).collect();

    let nll_of = |scaler: &VectorScaling| -> f64 {
        let mut acc = 0.0_f64;
        for (logits, label) in &samples {
            let mut out = vec![0.0_f32; logits.len()];
            scaler.apply(logits, &mut out).unwrap();
            acc += f64::from(negative_log_likelihood(&out, *label).unwrap());
        }
        acc / samples.len() as f64
    };

    let identity = VectorScaling::identity(3).unwrap();
    let fitted = VectorScaling::fit(&samples, 200, 0.1).unwrap();
    assert!(
        nll_of(&fitted) < nll_of(&identity),
        "{} vs {}",
        nll_of(&fitted),
        nll_of(&identity)
    );
}

#[test]
fn vector_scaling_checks_its_shapes() {
    let scaler = VectorScaling::identity(3).unwrap();
    let mut out = [0.0_f32; 3];
    assert_eq!(
        scaler.apply(&[0.0, 0.0], &mut out).err(),
        Some(CalibrationError::LengthMismatch {
            expected: 3,
            got: 2
        })
    );
    assert_eq!(
        VectorScaling::identity(0).err(),
        Some(CalibrationError::EmptyDistribution)
    );
}

// ---------------------------------------------------------------------------
// Isotonic regression (pool adjacent violators)
// ---------------------------------------------------------------------------

#[test]
fn a_violating_pair_pools_to_its_mean() {
    // 低いスコアが当たり、高いスコアが外れ = 単調性の違反
    // PAV は 2 つを 1 ブロックにまとめ、その平均 0.5 を置く
    let fit = IsotonicRegression::fit(&[(0.0, true), (1.0, false)]).unwrap();
    assert_eq!(fit.blocks(), 1);
    assert_eq!(fit.values(), &[0.5]);
    assert_eq!(fit.apply(0.0), 0.5);
    assert_eq!(fit.apply(1.0), 0.5);
}

#[test]
fn an_already_monotone_input_is_left_alone() {
    let fit = IsotonicRegression::fit(&[(0.0, false), (0.5, false), (1.0, true)]).unwrap();
    assert_eq!(fit.blocks(), 3);
    assert_eq!(fit.values(), &[0.0, 0.0, 1.0]);
    assert_eq!(fit.apply(0.0), 0.0);
    assert_eq!(fit.apply(1.0), 1.0);
}

#[test]
fn the_fitted_values_never_decrease() {
    let samples = [
        (0.1_f32, true),
        (0.2, false),
        (0.3, true),
        (0.4, false),
        (0.5, true),
        (0.6, true),
        (0.7, false),
        (0.8, true),
    ];
    let fit = IsotonicRegression::fit(&samples).unwrap();
    for pair in fit.values().windows(2) {
        assert!(pair[0] <= pair[1], "単調でない: {:?}", fit.values());
    }
    for v in fit.values() {
        assert!((0.0..=1.0).contains(v), "確率の範囲外: {v}");
    }
}

#[test]
fn all_true_fits_one_and_all_false_fits_zero() {
    let all_true = IsotonicRegression::fit(&[(0.1, true), (0.9, true)]).unwrap();
    assert_eq!(all_true.values(), &[1.0, 1.0]);
    let all_false = IsotonicRegression::fit(&[(0.1, false), (0.9, false)]).unwrap();
    assert_eq!(all_false.values(), &[0.0, 0.0]);
}

#[test]
fn outside_the_fitted_range_the_nearest_answer_is_held() {
    let fit = IsotonicRegression::fit(&[(0.2, false), (0.8, true)]).unwrap();
    assert_eq!(fit.apply(-5.0), 0.0, "下は最初のブロックの値");
    assert_eq!(fit.apply(5.0), 1.0, "上は最後のブロックの値");
}

#[test]
fn isotonic_needs_samples_and_finite_scores() {
    assert_eq!(
        IsotonicRegression::fit(&[]).err(),
        Some(CalibrationError::NoSamples)
    );
    assert_eq!(
        IsotonicRegression::fit(&[(f32::NAN, true)]).err(),
        Some(CalibrationError::NotAProbability)
    );
}
