//! Tie tests for every class of maximum / minimum / clamp / selection site (ALICE-Law G2).
//!
//! Each test goes through a public entry point and asserts output bits (`to_bits`) on the
//! inputs where the Law's rule decides the result: `max(+0, -0) = max(-0, +0) = +0`,
//! `min(+0, -0) = min(-0, +0) = -0`, a NaN against a number gives the number in both
//! operand orders, equal values keep the first index or the input order. A site rewritten
//! as a plain comparison (`if a > b { a } else { b }`, either operand order) turns at least
//! one of these red unless that rewrite is equivalent by IEEE for the operands the site can
//! see; the classes and the measured rewrites are listed in `alice-law-conformance.toml` G2.
//!
//! The AVX2 body of `tensor_max` / `tensor_min` / `tensor_relu` is reached by the inputs of
//! 8 elements or more when the `simd` feature is on and the CPU has AVX2 (CI's `x86_64`
//! runner); the same assertions then cover the lanes and the scalar tail.

use alice_ml::moe::MoeRouter;
use alice_ml::ops::TernaryWeight;
use alice_ml::quantize::{
    quantize_to_ternary, quantize_to_ternary_qat, quantize_to_ternary_sparse,
};
use alice_ml::speculative::SpeculativeDecoder;
use alice_ml::tensor::{tensor_max, tensor_min, tensor_relu, tensor_relu_inplace, Tensor};

const P0: f32 = 0.0;
const N0: f32 = -0.0;

const fn bits(x: f32) -> u32 {
    x.to_bits()
}

fn max_of_tensor(v: &[f32]) -> f32 {
    let mut d = v.to_vec();
    let n = d.len();
    tensor_max(&Tensor::from_arena(&mut d, &[n]))
}

fn min_of_tensor(v: &[f32]) -> f32 {
    let mut d = v.to_vec();
    let n = d.len();
    tensor_min(&Tensor::from_arena(&mut d, &[n]))
}

/// Inputs of every length 1..=24 where one position differs from the rest: the zero
/// tie and the NaN rule are decided at each lane, in the tail and in the scalar fold.
fn one_different(len: usize, at: usize, rest: f32, odd: f32) -> Vec<f32> {
    (0..len).map(|i| if i == at { odd } else { rest }).collect()
}

/// Reductions: `tensor_max` / `tensor_min` (scalar fold, AVX2 lanes and tail) and the
/// C ABI `am_ml_tensor_max` / `am_ml_tensor_min` (`law_math::max_of` / `min_of`).
fn check_reductions(name: &str, max_f: impl Fn(&[f32]) -> f32, min_f: impl Fn(&[f32]) -> f32) {
    let mut compared = 0;
    for (a, b) in [(P0, N0), (N0, P0)] {
        assert_eq!(bits(max_f(&[a, b])), bits(P0), "{name}: max({a:?}, {b:?})");
        assert_eq!(bits(min_f(&[a, b])), bits(N0), "{name}: min({a:?}, {b:?})");
        compared += 2;
    }
    for (a, b) in [(f32::NAN, 1.5), (1.5, f32::NAN)] {
        assert_eq!(bits(max_f(&[a, b])), bits(1.5), "{name}: max({a:?}, {b:?})");
        assert_eq!(bits(min_f(&[a, b])), bits(1.5), "{name}: min({a:?}, {b:?})");
        compared += 2;
    }
    assert_eq!(
        bits(max_f(&[2.0, 2.0])),
        bits(2.0),
        "{name}: max of equal values"
    );
    assert_eq!(
        bits(min_f(&[-2.0, -2.0])),
        bits(-2.0),
        "{name}: min of equal values"
    );
    for len in 1..=24 {
        for at in 0..len {
            // one +0 among -0s: the maximum is +0 wherever it sits
            assert_eq!(
                bits(max_f(&one_different(len, at, N0, P0))),
                bits(P0),
                "{name}: max len {len} +0 at {at}"
            );
            // one -0 among +0s: the minimum is -0 wherever it sits
            assert_eq!(
                bits(min_f(&one_different(len, at, P0, N0))),
                bits(N0),
                "{name}: min len {len} -0 at {at}"
            );
            // one NaN among 1.0s: skipped (a lone NaN is an empty reduction: -inf / +inf)
            let (want_max, want_min) = if len == 1 {
                (f32::NEG_INFINITY, f32::INFINITY)
            } else {
                (1.0, 1.0)
            };
            assert_eq!(
                bits(max_f(&one_different(len, at, 1.0, f32::NAN))),
                bits(want_max),
                "{name}: max len {len} NaN at {at}"
            );
            assert_eq!(
                bits(min_f(&one_different(len, at, 1.0, f32::NAN))),
                bits(want_min),
                "{name}: min len {len} NaN at {at}"
            );
            // one number among NaNs: the number
            assert_eq!(
                bits(max_f(&one_different(len, at, f32::NAN, -3.0))),
                bits(-3.0),
                "{name}: max len {len} -3 among NaN at {at}"
            );
            assert_eq!(
                bits(min_f(&one_different(len, at, f32::NAN, 3.0))),
                bits(3.0),
                "{name}: min len {len} 3 among NaN at {at}"
            );
            compared += 6;
        }
    }
    assert_eq!(compared, 8 + 6 * (24 * 25 / 2), "{name}: compared count");
}

#[test]
fn tensor_max_and_min_ties() {
    check_reductions("tensor", max_of_tensor, min_of_tensor);
}

#[cfg(feature = "ffi")]
#[test]
fn ffi_max_and_min_ties() {
    use alice_ml::ffi::{am_ml_tensor_max, am_ml_tensor_min};
    // SAFETY: valid pointers and lengths of live slices
    check_reductions(
        "ffi",
        |v| unsafe { am_ml_tensor_max(v.as_ptr(), v.len()) },
        |v| unsafe { am_ml_tensor_min(v.as_ptr(), v.len()) },
    );
}

/// `relu(x) = max(x, +0)`: `+0` for `-0`, `+0` and NaN, `x` otherwise.
fn relu_expected(x: f32) -> f32 {
    if x.is_nan() || x <= 0.0 {
        P0
    } else {
        x
    }
}

fn relu_inputs(len: usize) -> Vec<f32> {
    let pattern = [
        N0,
        f32::NAN,
        P0,
        -1.0,
        2.5,
        f32::NEG_INFINITY,
        -f32::NAN,
        1e-40,
    ];
    (0..len).map(|i| pattern[i % pattern.len()]).collect()
}

#[test]
fn relu_ties() {
    let mut compared = 0;
    for len in 1..=24 {
        let input = relu_inputs(len);
        let mut a = input.clone();
        let mut out = vec![7.0f32; len];
        {
            let at = Tensor::from_arena(&mut a, &[len]);
            let mut ot = Tensor::from_arena(&mut out, &[len]);
            tensor_relu(&at, &mut ot);
        }
        let mut inplace = input.clone();
        {
            let mut t = Tensor::from_arena(&mut inplace, &[len]);
            tensor_relu_inplace(&mut t);
        }
        for i in 0..len {
            let want = bits(relu_expected(input[i]));
            assert_eq!(
                bits(out[i]),
                want,
                "tensor_relu len {len} [{i}] = relu({:?})",
                input[i]
            );
            assert_eq!(
                bits(inplace[i]),
                want,
                "tensor_relu_inplace len {len} [{i}]"
            );
            compared += 2;
        }
    }
    assert_eq!(compared, 2 * (24 * 25 / 2));
}

#[cfg(feature = "ffi")]
#[test]
fn ffi_relu_and_softmax_ties() {
    use alice_ml::ffi::{am_ml_tensor_relu, am_ml_tensor_relu_inplace, am_ml_tensor_softmax};
    for len in 1..=24 {
        let input = relu_inputs(len);
        let mut out = vec![7.0f32; len];
        let mut inplace = input.clone();
        // SAFETY: valid pointers and lengths of live vectors
        unsafe {
            am_ml_tensor_relu(input.as_ptr(), out.as_mut_ptr(), len);
            am_ml_tensor_relu_inplace(inplace.as_mut_ptr(), len);
        }
        for i in 0..len {
            let want = bits(relu_expected(input[i]));
            assert_eq!(bits(out[i]), want, "am_ml_tensor_relu len {len} [{i}]");
            assert_eq!(
                bits(inplace[i]),
                want,
                "am_ml_tensor_relu_inplace len {len} [{i}]"
            );
        }
    }
    // the softmax shift is the maximum of the numbers (the NaN skipped): exp(1 - 1) = 1
    // stays, the sum is NaN so nothing is normalised; a shift by NaN would make it NaN
    for input in [[1.0f32, f32::NAN], [f32::NAN, 1.0]] {
        let mut out = [0.0f32; 2];
        // SAFETY: valid pointers and lengths
        unsafe { am_ml_tensor_softmax(input.as_ptr(), out.as_mut_ptr(), 2) };
        let at = input.iter().position(|x| !x.is_nan()).expect("one number");
        assert_eq!(bits(out[at]), bits(1.0), "softmax({input:?})[{at}]");
        assert!(out[1 - at].is_nan(), "softmax({input:?})[{}]", 1 - at);
    }
}

/// `original_range` of each ternary quantiser is `(min, max)` of the weights.
#[test]
fn quantizer_range_ties() {
    type Quantize = fn(&[f32]) -> (f32, f32);
    let quantizers: [(&str, Quantize); 3] = [
        ("ternary", |w| {
            quantize_to_ternary(w, 1, w.len()).1.original_range
        }),
        ("qat", |w| {
            quantize_to_ternary_qat(w, 1, w.len(), 1.0, 1.0)
                .1
                .original_range
        }),
        ("sparse", |w| {
            quantize_to_ternary_sparse(w, 1, w.len(), 0.5)
                .1
                .original_range
        }),
    ];
    let mut compared = 0;
    for (name, q) in quantizers {
        for w in [[N0, P0], [P0, N0]] {
            let (lo, hi) = q(&w);
            assert_eq!(
                (bits(lo), bits(hi)),
                (bits(N0), bits(P0)),
                "{name}: range of {w:?}"
            );
            compared += 1;
        }
        for w in [[f32::NAN, 0.75], [0.75, f32::NAN]] {
            let (lo, hi) = q(&w);
            assert_eq!(
                (bits(lo), bits(hi)),
                (bits(0.75), bits(0.75)),
                "{name}: range of {w:?}"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 12);
}

/// argmax: the first maximum, `+0` over an earlier `-0`, a NaN never chosen.
#[test]
fn speculative_argmax_ties() {
    let accepted = |draft: &[f32], verify: &[f32]| {
        SpeculativeDecoder::acceptance_mask(draft, verify, draft.len(), 1).acceptance[0]
    };
    // the argmax of [-0, +0] is 1 and of [+0, -0] is 0: they differ
    assert!(!accepted(&[N0, P0], &[P0, N0]));
    assert!(accepted(&[N0, P0], &[-1.0, 0.5]));
    // a NaN is never the argmax: [NaN, 1] and [0, 1] both pick index 1
    assert!(accepted(&[f32::NAN, 1.0], &[0.0, 1.0]));
    assert!(accepted(&[1.0, f32::NAN], &[1.0, 0.0]));
    // equal values: the first
    assert!(accepted(&[2.0, 2.0], &[2.0, 1.0]));
    assert!(!accepted(&[2.0, 2.0], &[1.0, 2.0]));
}

/// The router's softmax shift (the maximum of the numbers), the fast exp's clamp
/// (a NaN stays NaN) and the top-k order (NaN below every score, equal scores in expert
/// order).
#[test]
fn moe_router_ties() {
    // identity gate: score i = input i
    let identity = |n: usize| {
        let v: Vec<i8> = (0..n * n).map(|k| i8::from(k / n == k % n)).collect();
        TernaryWeight::from_ternary(&v, n, n)
    };
    let router = MoeRouter::new(identity(2), 2, 1);
    for (input, top) in [([f32::NAN, 1.0], 1usize), ([1.0, f32::NAN], 0)] {
        let mut gate = [0.0f32; 2];
        let choice = router.route(&input, &mut gate);
        assert_eq!(choice.len(), 1);
        assert_eq!(choice[0].expert_id, top, "route({input:?})");
        assert_eq!(
            bits(gate[top]),
            bits(1.0),
            "gate of the number, route({input:?})"
        );
        assert!(gate[1 - top].is_nan(), "gate of the NaN, route({input:?})");
    }
    let router = MoeRouter::new(identity(3), 3, 3);
    let mut gate = [0.0f32; 3];
    let ids: Vec<usize> = router
        .route(&[2.0, 2.0, 2.0], &mut gate)
        .iter()
        .map(|c| c.expert_id)
        .collect();
    assert_eq!(ids, [0, 1, 2], "equal scores keep the expert order");
    let ids: Vec<usize> = router
        .route(&[2.0, f32::NAN, 3.0], &mut gate)
        .iter()
        .map(|c| c.expert_id)
        .collect();
    assert_eq!(ids, [2, 0, 1], "the NaN score last");
    // ties among distinct values, past the length where an unstable sort stops behaving
    // like an insertion sort (20): equal scores keep the expert order (a stable sort)
    let n = 64;
    let router = MoeRouter::new(identity(n), n, n);
    let mut gate = vec![0.0f32; n];
    let scores: Vec<f32> = (0..n)
        .map(|i| [0.5f32, 2.0, 1.0][(i * 7 + i / 5) % 3])
        .collect();
    let ids: Vec<usize> = router
        .route(&scores, &mut gate)
        .iter()
        .map(|c| c.expert_id)
        .collect();
    let mut want: Vec<usize> = (0..n).collect();
    want.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    assert_eq!(
        ids, want,
        "{n} scores of three values: ties in expert order"
    );
}

/// The predicted class of a reliability observation is the argmax of the distribution:
/// the first of equal probabilities, `+0` over an earlier `-0`.
#[test]
fn reliability_argmax_ties() {
    use alice_ml::calibration::ReliabilityBins;
    let accuracy = |probs: &[f32], label: usize| {
        let mut bins = ReliabilityBins::with_bins(4).expect("bins");
        bins.observe_distribution(probs, label)
            .expect("a distribution");
        let b: Vec<_> = bins.bins().into_iter().filter(|b| b.count > 0).collect();
        assert_eq!(b.len(), 1, "{probs:?}");
        b[0].accuracy
    };
    // equal probabilities: the first index is the prediction
    assert_eq!(bits(accuracy(&[0.5, 0.5], 0)), bits(1.0));
    assert_eq!(bits(accuracy(&[0.5, 0.5], 1)), bits(0.0));
    // a -0 and a +0: +0 is the larger, wherever it sits
    assert_eq!(bits(accuracy(&[N0, P0], 1)), bits(1.0));
    assert_eq!(bits(accuracy(&[P0, N0], 0)), bits(1.0));
}

/// The public helpers themselves: `max_of` / `min_of` / `relu` / `argmax` / `order`.
#[test]
fn law_math_helper_ties() {
    use alice_ml::law_math::{argmax, max_of, min_of, order, relu};
    use core::cmp::Ordering::{Equal, Greater, Less};
    for v in [[N0, P0], [P0, N0]] {
        assert_eq!(bits(max_of(&v)), bits(P0), "max_of({v:?})");
        assert_eq!(bits(min_of(&v)), bits(N0), "min_of({v:?})");
    }
    for v in [[f32::NAN, 4.0], [4.0, f32::NAN]] {
        assert_eq!(bits(max_of(&v)), bits(4.0), "max_of({v:?})");
        assert_eq!(bits(min_of(&v)), bits(4.0), "min_of({v:?})");
    }
    assert_eq!(bits(max_of(&[])), bits(f32::NEG_INFINITY));
    assert_eq!(bits(min_of(&[f32::NAN])), bits(f32::INFINITY));
    for (x, want) in [
        (N0, P0),
        (P0, P0),
        (f32::NAN, P0),
        (-f32::NAN, P0),
        (-2.0, P0),
        (3.0, 3.0),
    ] {
        assert_eq!(bits(relu(x)), bits(want), "relu({x:?})");
    }
    assert_eq!(argmax(&[N0, P0]), 1);
    assert_eq!(argmax(&[P0, N0]), 0);
    assert_eq!(argmax(&[f32::NAN, -1.0]), 1);
    assert_eq!(argmax(&[5.0, 5.0, 1.0]), 0);
    assert_eq!(argmax(&[f32::NAN, f32::NAN]), 0);
    assert_eq!(argmax(&[]), 0);
    assert_eq!(order(N0, P0), Less);
    assert_eq!(order(P0, N0), Greater);
    assert_eq!(order(f32::NAN, f32::NEG_INFINITY), Less);
    assert_eq!(order(f32::NEG_INFINITY, -f32::NAN), Greater);
    assert_eq!(order(f32::NAN, -f32::NAN), Equal);
    assert_eq!(order(1.0, 1.0), Equal);
}
