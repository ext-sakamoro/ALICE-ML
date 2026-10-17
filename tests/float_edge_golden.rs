//! ALICE-Law G4: the signed-zero and exact-cancellation cases of the Law sites, pinned as
//! exact bit patterns (`to_bits`), run in debug and in release on `x86_64` and `aarch64`
//! (`.github/workflows/ci.yml` test job and `.github/workflows/release-profile.yml`).
//!
//! Each case is one Law site class: the zero-sign rule of `law_math::max` / `min` / `max64` /
//! `min64` and of the helpers built on them (`clamp`, `relu`, `max_of`, `min_of`, `argmax`,
//! `tensor_max`, `tensor_min`), and the exact cancellations a softmax shift makes (equal
//! logits give exactly `1 / n`, for small, large and signed-zero logits, in the exact and
//! fast softmax, through the identity temperature, and into the negative log-likelihood and
//! the cross entropy). A build whose codegen contracts, reassociates or reorders one of
//! these differs in at least one pinned bit.
//!
//! `ALICE_ML_PRINT_EDGE_BITS=1 cargo test --test float_edge_golden -- --nocapture` prints
//! the measured values instead of asserting them (used once to record the table).

use std::hint::black_box;

use alice_ml::calibration::{negative_log_likelihood, TemperatureScaling};
use alice_ml::law_math::{argmax, clamp, clamp64, max, max64, max_of, min, min64, min_of, relu};
use alice_ml::tensor::{tensor_max, tensor_min, tensor_softmax, tensor_softmax_fast, Tensor};
use alice_ml::training::cross_entropy_loss;

const NZ: f32 = -0.0;
const PZ: f32 = 0.0;

fn softmax(logits: &[f32], fast: bool) -> Vec<u32> {
    let n = logits.len();
    let mut input = black_box(logits.to_vec());
    let mut output = vec![0.0_f32; n];
    {
        let a = Tensor::from_arena(&mut input, &[1, n]);
        let mut out = Tensor::from_arena(&mut output, &[1, n]);
        if fast {
            tensor_softmax_fast(&a, &mut out);
        } else {
            tensor_softmax(&a, &mut out);
        }
    }
    output.iter().map(|x| x.to_bits()).collect()
}

fn reduce(values: &[f32], f: impl FnOnce(&Tensor) -> f32) -> f32 {
    let mut data = black_box(values.to_vec());
    let n = data.len();
    let t = Tensor::from_arena(&mut data, &[n]);
    f(&t)
}

fn temperature_identity(logits: &[f32]) -> Vec<u32> {
    let mut out = vec![0.0; logits.len()];
    TemperatureScaling::IDENTITY
        .apply(logits, &mut out)
        .expect("finite logits");
    out.iter().map(|x| x.to_bits()).collect()
}

fn cross_entropy_uniform(n: usize) -> u32 {
    let logits = black_box(vec![3.5_f32; n]);
    let mut targets = vec![0.0_f32; n];
    targets[0] = 1.0;
    let mut grad = vec![0.0_f32; n];
    cross_entropy_loss(&logits, &targets, &mut grad)
        .value
        .to_bits()
}

/// The clamp sites of a NaN input of either sign, quiet or signalling with a payload: the
/// canonical NaN, in the order of the pinned table
fn nan_cases() -> Vec<(&'static str, u64)> {
    let f = |x: f32| u64::from(x.to_bits());
    vec![
        (
            "clamp(+NaN, -1, 1)",
            f(clamp(
                black_box(f32::from_bits(0x7fc0_0000)),
                black_box(-1.0),
                black_box(1.0),
            )),
        ),
        (
            "clamp(-NaN, -1, 1)",
            f(clamp(
                black_box(f32::from_bits(0xffc0_0000)),
                black_box(-1.0),
                black_box(1.0),
            )),
        ),
        (
            "clamp(-sNaN payload, -1, 1)",
            f(clamp(
                black_box(f32::from_bits(0xff80_1234)),
                black_box(-1.0),
                black_box(1.0),
            )),
        ),
        (
            "clamp64(-NaN payload, -1, 1)",
            clamp64(
                black_box(f64::from_bits(0xfff8_0000_0000_5678)),
                black_box(-1.0),
                black_box(1.0),
            )
            .to_bits(),
        ),
    ]
}

/// `(case, measured bits as u64)`: f32 results widened, f64 results as they are, index results as the index
fn measured() -> Vec<(&'static str, u64)> {
    // every input goes through black_box: the release legs must run the emitted comparisons
    // and arithmetic, not a value the optimiser folded at compile time
    let (nz, pz) = (black_box(NZ), black_box(PZ));
    let f = |x: f32| u64::from(x.to_bits());
    let mut v: Vec<(&'static str, u64)> = vec![
        ("max(-0, +0)", f(max(nz, pz))),
        ("max(+0, -0)", f(max(pz, nz))),
        ("min(-0, +0)", f(min(nz, pz))),
        ("min(+0, -0)", f(min(pz, nz))),
        (
            "max64(-0, +0)",
            max64(black_box(-0.0), black_box(0.0)).to_bits(),
        ),
        (
            "max64(+0, -0)",
            max64(black_box(0.0), black_box(-0.0)).to_bits(),
        ),
        (
            "min64(-0, +0)",
            min64(black_box(-0.0), black_box(0.0)).to_bits(),
        ),
        (
            "min64(+0, -0)",
            min64(black_box(0.0), black_box(-0.0)).to_bits(),
        ),
        ("relu(-0)", f(relu(nz))),
        ("relu(+0)", f(relu(pz))),
        ("relu(-1)", f(relu(black_box(-1.0)))),
        (
            "clamp(-0, -1, 1)",
            f(clamp(nz, black_box(-1.0), black_box(1.0))),
        ),
        (
            "clamp(+0, -1, 1)",
            f(clamp(pz, black_box(-1.0), black_box(1.0))),
        ),
        (
            "clamp(-0, +0, 1)",
            f(clamp(nz, black_box(0.0), black_box(1.0))),
        ),
        ("clamp(+0, -1, -0)", f(clamp(pz, black_box(-1.0), nz))),
    ];
    v.extend(nan_cases());
    v.extend([
        ("max_of([-0, +0])", f(max_of(&[nz, pz]))),
        ("max_of([+0, -0])", f(max_of(&[pz, nz]))),
        ("min_of([-0, +0])", f(min_of(&[nz, pz]))),
        ("min_of([+0, -0])", f(min_of(&[pz, nz]))),
        ("argmax([-0, +0])", argmax(&[nz, pz]) as u64),
        ("argmax([+0, -0])", argmax(&[pz, nz]) as u64),
        ("argmax([1, 1])", argmax(&black_box([1.0, 1.0])) as u64),
        (
            "tensor_max([-0, +0, -0])",
            f(reduce(&[nz, pz, nz], tensor_max)),
        ),
        (
            "tensor_min([+0, -0, +0])",
            f(reduce(&[pz, nz, pz], tensor_min)),
        ),
        (
            "nll([0.25; 4], 0)",
            f(negative_log_likelihood(&black_box([0.25; 4]), 0).expect("distribution")),
        ),
        (
            "cross_entropy(equal logits, n = 4)",
            u64::from(cross_entropy_uniform(4)),
        ),
        (
            "cross_entropy(equal logits, n = 8)",
            u64::from(cross_entropy_uniform(8)),
        ),
    ]);
    let lists: [(&'static str, Vec<u32>); 9] = [
        ("softmax([5; 4])", softmax(&[5.0; 4], false)),
        ("softmax_fast([5; 4])", softmax(&[5.0; 4], true)),
        ("softmax([1e30; 4])", softmax(&[1e30; 4], false)),
        ("softmax_fast([1e30; 4])", softmax(&[1e30; 4], true)),
        ("softmax([-0, +0])", softmax(&[nz, pz], false)),
        ("softmax_fast([-0, +0])", softmax(&[nz, pz], true)),
        ("temperature(1)([7; 4])", temperature_identity(&[7.0; 4])),
        ("temperature(1)([-0, +0])", temperature_identity(&[nz, pz])),
        (
            "temperature(1)([1e30; 4])",
            temperature_identity(&[1e30; 4]),
        ),
    ];
    for (name, bits) in lists {
        // every element of an equal-logit softmax is the same exact value: pin it and the count
        assert!(
            bits.iter().all(|&b| b == bits[0]),
            "{name}: elements differ: {bits:08x?}"
        );
        v.push((name, (u64::from(bits[0]) << 8) | bits.len() as u64));
    }
    v
}

/// The bits measured on `aarch64` and `x86_64`, debug and release (`ALICE_ML_PRINT_EDGE_BITS=1`)
const EXPECTED: &[(&str, u64)] = &[
    ("max(-0, +0)", 0x0000_0000_0000_0000),
    ("max(+0, -0)", 0x0000_0000_0000_0000),
    ("min(-0, +0)", 0x0000_0000_8000_0000),
    ("min(+0, -0)", 0x0000_0000_8000_0000),
    ("max64(-0, +0)", 0x0000_0000_0000_0000),
    ("max64(+0, -0)", 0x0000_0000_0000_0000),
    ("min64(-0, +0)", 0x8000_0000_0000_0000),
    ("min64(+0, -0)", 0x8000_0000_0000_0000),
    ("relu(-0)", 0x0000_0000_0000_0000),
    ("relu(+0)", 0x0000_0000_0000_0000),
    ("relu(-1)", 0x0000_0000_0000_0000),
    ("clamp(-0, -1, 1)", 0x0000_0000_8000_0000),
    ("clamp(+0, -1, 1)", 0x0000_0000_0000_0000),
    ("clamp(-0, +0, 1)", 0x0000_0000_0000_0000),
    ("clamp(+0, -1, -0)", 0x0000_0000_8000_0000),
    ("clamp(+NaN, -1, 1)", 0x0000_0000_7fc0_0000),
    ("clamp(-NaN, -1, 1)", 0x0000_0000_7fc0_0000),
    ("clamp(-sNaN payload, -1, 1)", 0x0000_0000_7fc0_0000),
    ("clamp64(-NaN payload, -1, 1)", 0x7ff8_0000_0000_0000),
    ("max_of([-0, +0])", 0x0000_0000_0000_0000),
    ("max_of([+0, -0])", 0x0000_0000_0000_0000),
    ("min_of([-0, +0])", 0x0000_0000_8000_0000),
    ("min_of([+0, -0])", 0x0000_0000_8000_0000),
    ("argmax([-0, +0])", 0x0000_0000_0000_0001),
    ("argmax([+0, -0])", 0x0000_0000_0000_0000),
    ("argmax([1, 1])", 0x0000_0000_0000_0000),
    ("tensor_max([-0, +0, -0])", 0x0000_0000_0000_0000),
    ("tensor_min([+0, -0, +0])", 0x0000_0000_8000_0000),
    ("nll([0.25; 4], 0)", 0x0000_0000_3fb1_7218),
    ("cross_entropy(equal logits, n = 4)", 0x0000_0000_3fb1_7218),
    ("cross_entropy(equal logits, n = 8)", 0x0000_0000_4005_1591),
    ("softmax([5; 4])", 0x0000_003e_8000_0004),
    ("softmax_fast([5; 4])", 0x0000_003e_8000_0004),
    ("softmax([1e30; 4])", 0x0000_003e_8000_0004),
    ("softmax_fast([1e30; 4])", 0x0000_003e_8000_0004),
    ("softmax([-0, +0])", 0x0000_003f_0000_0002),
    ("softmax_fast([-0, +0])", 0x0000_003f_0000_0002),
    ("temperature(1)([7; 4])", 0x0000_003e_8000_0004),
    ("temperature(1)([-0, +0])", 0x0000_003f_0000_0002),
    ("temperature(1)([1e30; 4])", 0x0000_003e_8000_0004),
];

#[test]
fn law_sites_signed_zero_and_exact_cancellation_bits() {
    let got = measured();
    if std::env::var_os("ALICE_ML_PRINT_EDGE_BITS").is_some() {
        for (name, bits) in &got {
            println!(
                "    ({name:?}, 0x{:04x}_{:04x}_{:04x}_{:04x}),",
                bits >> 48,
                (bits >> 32) & 0xffff,
                (bits >> 16) & 0xffff,
                bits & 0xffff
            );
        }
        return;
    }
    assert_eq!(
        got.len(),
        EXPECTED.len(),
        "the case list and the pinned table differ in length"
    );
    assert!(!got.is_empty(), "compared nothing");
    let mut compared = 0;
    for ((name, bits), (want_name, want)) in got.iter().zip(EXPECTED) {
        assert_eq!(name, want_name, "case order");
        assert_eq!(
            *bits, *want,
            "{name}: got 0x{bits:016x}, pinned 0x{want:016x}"
        );
        compared += 1;
    }
    assert_eq!(compared, EXPECTED.len());
}

/// The pinned values are the Law's, not only the build's: the zero-sign rule and `1 / n`
#[test]
fn the_pinned_values_are_the_law() {
    let want: &[(&str, u64)] = &[
        ("max(-0, +0)", 0),
        ("max(+0, -0)", 0),
        ("min(-0, +0)", 0x8000_0000),
        ("min(+0, -0)", 0x8000_0000),
        ("max64(-0, +0)", 0),
        ("min64(+0, -0)", 0x8000_0000_0000_0000),
        ("relu(-0)", 0),
        ("clamp(-0, -1, 1)", 0x8000_0000),
        ("clamp(-0, +0, 1)", 0),
        ("clamp(-NaN, -1, 1)", 0x7fc0_0000),
        ("clamp64(-NaN payload, -1, 1)", 0x7ff8_0000_0000_0000),
        ("max_of([+0, -0])", 0),
        ("min_of([-0, +0])", 0x8000_0000),
        ("argmax([-0, +0])", 1),
        ("argmax([1, 1])", 0),
        ("softmax([5; 4])", (u64::from(0.25_f32.to_bits()) << 8) | 4),
        (
            "softmax([1e30; 4])",
            (u64::from(0.25_f32.to_bits()) << 8) | 4,
        ),
        ("softmax([-0, +0])", (u64::from(0.5_f32.to_bits()) << 8) | 2),
        (
            "temperature(1)([7; 4])",
            (u64::from(0.25_f32.to_bits()) << 8) | 4,
        ),
    ];
    let got = measured();
    let mut compared = 0;
    for (name, bits) in want {
        let (_, g) = got.iter().find(|(n, _)| n == name).expect("case present");
        assert_eq!(g, bits, "{name}");
        compared += 1;
    }
    assert!(compared > 0, "compared nothing");
}
