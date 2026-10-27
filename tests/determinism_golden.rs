//! Bit-pattern goldens of the float paths that take a maximum, a minimum or a
//! clamp (ALICE-Law G2), over a lattice of IEEE special values.
//!
//! Each test feeds the library function inputs built from [`lattice`]: both
//! signed zeros, quiet NaNs of both signs and a NaN with a payload, both
//! infinities, the smallest and largest subnormals, the smallest normal, the
//! largest finite values and a few ordinary values, in lengths that cover
//! the scalar tail and the 8-lane SIMD body of every kernel. It hashes the
//! output bits (SHA-256 over little-endian bit patterns, with a label per
//! input) and compares the digest with the one recorded here, so any change
//! of a single output bit on any path is a test failure, on every target and
//! feature set CI runs (the digests are the same everywhere: that is what a
//! Law path has to give).
//!
//! A NaN output is hashed as the canonical NaN: the sign and payload of a NaN
//! that IEEE arithmetic produces (`inf - inf`, an operation on a NaN input) are
//! left to the platform and the optimiser, and they differed here between
//! `aarch64` and `x86_64` and between the debug and release profiles while every
//! non-NaN bit agreed. That an output is NaN, and every other bit, is pinned.
//!
//! Measured before the maximum, minimum and clamp sites moved to
//! `alice_ml::law_math`: `max_and_min_over_the_lattice` had one digest on
//! `aarch64` and another on `x86_64` (`tensor_max(+0, -0)` gave `-0` and
//! `tensor_min(-0, +0)` gave `+0` on `x86_64`); every golden now has one digest on
//! aarch64-apple-darwin and x86_64-apple-darwin, debug and release, with and
//! without the `simd`, `neon` and `ffi` features.
//!
//! `ALICE_ML_GOLDEN_PRINT=1 cargo test --test determinism_golden -- --nocapture`
//! prints the digests instead of comparing them (used to record a moved
//! golden after an intended change, never to accept an unexplained one).

use alice_ml::calibration::{negative_log_likelihood, TemperatureScaling};
use alice_ml::error_analysis::{compute_layer_error_propagation, LayerConfig};
use alice_ml::quantize::{
    compute_quantization_error, quantize_to_ternary, quantize_to_ternary_qat,
    quantize_to_ternary_sparse,
};
use alice_ml::tensor::{
    tensor_max, tensor_min, tensor_relu, tensor_relu_inplace, tensor_softmax, tensor_softmax_fast,
    QuantizedTensor, Tensor,
};
use alice_ml::training::cross_entropy_loss;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;

/// The special values every golden is built from, as bit patterns.
const LATTICE_BITS: [u32; 24] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x7fc0_0000, // quiet NaN
    0xffc0_0000, // quiet NaN, sign set
    0x7fa0_0001, // quiet NaN with a payload
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x0000_0001, // smallest subnormal
    0x8000_0001, // -smallest subnormal
    0x007f_ffff, // largest subnormal
    0x0080_0000, // smallest normal
    0x3f80_0000, // 1.0
    0xbf80_0000, // -1.0
    0x3f00_0000, // 0.5
    0xbf00_0000, // -0.5
    0x4060_0000, // 3.5
    0xc010_0000, // -2.25
    0x7f7f_ffff, // f32::MAX
    0xff7f_ffff, // -f32::MAX
    0x322b_cc77, // 1e-8
    0x2edb_e6ff, // 1e-10
    0x42af_0000, // 87.5
    0x42b1_0000, // 88.5
    0xc2af_0000, // -87.5
];

fn lattice() -> Vec<f32> {
    LATTICE_BITS.iter().map(|&b| f32::from_bits(b)).collect()
}

/// A deterministic vector of `len` lattice values, starting at `start`.
fn vector(len: usize, start: usize) -> Vec<f32> {
    let l = lattice();
    (0..len).map(|i| l[(start + i * 7) % l.len()]).collect()
}

/// Lengths that reach the scalar tail and the 8-lane body (1..=33).
const LENGTHS: std::ops::RangeInclusive<usize> = 1..=33;

struct Golden(Sha256);

impl Golden {
    fn new(label: &str) -> Self {
        let mut h = Sha256::new();
        h.update(label.as_bytes());
        Self(h)
    }
    /// a length prefix (u32, little-endian), so two inputs cannot run together
    fn len(&mut self, n: usize) {
        self.0.update(
            u32::try_from(n)
                .expect("a golden input is short")
                .to_le_bytes(),
        );
    }
    fn tag(&mut self, t: &str) {
        self.len(t.len());
        self.0.update(t.as_bytes());
    }
    fn f32s(&mut self, xs: &[f32]) {
        self.len(xs.len());
        for &x in xs {
            self.f32(x);
        }
    }
    fn f32(&mut self, x: f32) {
        let bits = if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() };
        self.0.update(bits.to_le_bytes());
    }
    fn f64(&mut self, x: f64) {
        let bits = if x.is_nan() {
            0x7ff8_0000_0000_0000
        } else {
            x.to_bits()
        };
        self.0.update(bits.to_le_bytes());
    }
    fn usize(&mut self, x: usize) {
        self.0
            .update(u64::try_from(x).expect("a usize fits in u64").to_le_bytes());
    }
    fn bytes(&mut self, b: &[u8]) {
        self.len(b.len());
        self.0.update(b);
    }
    fn digest(self) -> String {
        self.0.finalize().iter().fold(String::new(), |mut acc, b| {
            let _ = write!(acc, "{b:02x}");
            acc
        })
    }
    /// The digest when it is not `expected`
    fn mismatch(self, expected: &str) -> Option<String> {
        let got = self.digest();
        (got != expected).then_some(got)
    }
    fn finish(self, name: &str, expected: &str) {
        if std::env::var_os("ALICE_ML_GOLDEN_PRINT").is_some() {
            println!("GOLDEN {name} {}", self.digest());
            return;
        }
        if let Some(got) = self.mismatch(expected) {
            panic!("{name}: output bits moved to {got}, recorded {expected}");
        }
    }
}

#[test]
fn checker_rejects_a_wrong_digest() {
    let recorded = |bits: &[f32]| {
        let mut h = Golden::new("canary");
        h.f32s(bits);
        h
    };
    let expected = recorded(&[1.0, -0.0]).digest();
    assert_eq!(recorded(&[1.0, -0.0]).mismatch(&expected), None);
    assert!(
        recorded(&[1.0, 0.0]).mismatch(&expected).is_some(),
        "a scenario whose bits moved (-0 to +0) must not match its recorded digest"
    );
}

fn with_tensor<R>(data: &mut [f32], f: impl FnOnce(&mut Tensor) -> R) -> R {
    let n = data.len();
    let mut t = Tensor::from_arena(data, &[n]);
    f(&mut t)
}

#[test]
fn relu_over_the_lattice() {
    let mut g = Golden::new("relu");
    for len in LENGTHS {
        for start in 0..3 {
            let mut input = vector(len, start);
            let mut out = vec![0.0f32; len];
            {
                let a = Tensor::from_arena(&mut input, &[len]);
                let mut o = Tensor::from_arena(&mut out, &[len]);
                tensor_relu(&a, &mut o);
            }
            g.tag("relu");
            g.f32s(&out);
            let mut inplace = vector(len, start);
            with_tensor(&mut inplace, tensor_relu_inplace);
            g.tag("relu_inplace");
            g.f32s(&inplace);
        }
    }
    g.finish("relu", GOLDEN_RELU);
}

#[test]
fn max_and_min_over_the_lattice() {
    let mut g = Golden::new("max_min");
    let l = lattice();
    // every ordered pair: the zero sign and NaN rule decide each result
    for &a in &l {
        for &b in &l {
            let mut v = [a, b];
            g.f32(with_tensor(&mut v, |t: &mut Tensor| tensor_max(t)));
            g.f32(with_tensor(&mut v, |t: &mut Tensor| tensor_min(t)));
        }
    }
    for len in LENGTHS {
        for start in 0..3 {
            let mut v = vector(len, start);
            g.f32(with_tensor(&mut v, |t: &mut Tensor| tensor_max(t)));
            g.f32(with_tensor(&mut v, |t: &mut Tensor| tensor_min(t)));
        }
    }
    g.finish("max_min", GOLDEN_MAX_MIN);
}

#[test]
fn softmax_over_the_lattice() {
    let mut g = Golden::new("softmax");
    for len in 1..=17 {
        for start in 0..3 {
            for fast in [false, true] {
                let mut input = vector(len, start);
                let mut out = vec![0.0f32; len];
                {
                    let a = Tensor::from_arena(&mut input, &[1, len]);
                    let mut o = Tensor::from_arena(&mut out, &[1, len]);
                    if fast {
                        tensor_softmax_fast(&a, &mut o);
                    } else {
                        tensor_softmax(&a, &mut o);
                    }
                }
                g.tag(if fast { "fast" } else { "exact" });
                g.f32s(&out);
            }
        }
    }
    g.finish("softmax", GOLDEN_SOFTMAX);
}

fn hash_quantized(
    g: &mut Golden,
    w: &alice_ml::ops::TernaryWeight,
    s: &alice_ml::quantize::QuantStats,
) {
    g.bytes(w.packed());
    g.f32(w.scale());
    g.usize(s.plus_count);
    g.usize(s.minus_count);
    g.usize(s.zero_count);
    g.f32(s.original_range.0);
    g.f32(s.original_range.1);
    g.f32(s.scale);
    g.f32(s.mae);
}

#[test]
fn ternary_quantization_over_the_lattice() {
    let mut g = Golden::new("quantize");
    let values = lattice();
    for start in 0..values.len() {
        let weights = vector(12, start);
        let (ternary, stats) = quantize_to_ternary(&weights, 3, 4);
        g.tag("plain");
        hash_quantized(&mut g, &ternary, &stats);
        let error = compute_quantization_error(&weights, &ternary);
        g.tag("error");
        for x in [error.mae, error.mse, error.rmse, error.max_error, error.snr] {
            g.f32(x);
        }
        let (scale, temperature, threshold) = (
            values[start],
            values[(start + 5) % values.len()],
            values[(start + 11) % values.len()],
        );
        let (ternary, stats) = quantize_to_ternary_qat(&weights, 3, 4, scale, temperature);
        g.tag("qat");
        hash_quantized(&mut g, &ternary, &stats);
        let (ternary, stats) = quantize_to_ternary_sparse(&weights, 3, 4, threshold);
        g.tag("sparse");
        hash_quantized(&mut g, &ternary, &stats);
    }
    g.finish("quantize", GOLDEN_QUANTIZE);
}

#[test]
fn int8_quantization_over_the_lattice() {
    let mut g = Golden::new("int8");
    for len in LENGTHS {
        for start in 0..3 {
            let input = vector(len, start);
            let q = QuantizedTensor::from_f32_slice(&input, &[len]);
            g.f32(q.scale());
            g.bytes(
                &q.data_i8()
                    .iter()
                    .map(|&x| x.to_le_bytes()[0])
                    .collect::<Vec<u8>>(),
            );
        }
    }
    g.finish("int8", GOLDEN_INT8);
}

#[test]
fn cross_entropy_over_the_lattice() {
    let mut g = Golden::new("cross_entropy");
    for len in 1..=17 {
        for start in 0..3 {
            let logits = vector(len, start);
            let mut targets = vec![0.0f32; len];
            targets[start % len] = 1.0;
            let mut grad = vec![0.0f32; len];
            let r = cross_entropy_loss(&logits, &targets, &mut grad);
            g.f32(r.value);
            g.f32s(&grad);
        }
    }
    g.finish("cross_entropy", GOLDEN_CROSS_ENTROPY);
}

/// A finite logit within ±50 (non-finite values become 0), through the crate's
/// Law clamp (the platform `f32::clamp` is banned).
fn bounded(x: f32) -> f32 {
    if x.is_finite() {
        alice_ml::law_math::clamp(x, -50.0, 50.0)
    } else {
        0.0
    }
}

#[test]
fn calibration_over_the_lattice() {
    let mut g = Golden::new("calibration");
    // probabilities in [0, 1] from the lattice (both zeros, subnormals, 1e-10 / 1e-8, 0.5, 1)
    let probs: Vec<f32> = lattice()
        .into_iter()
        .filter(|p| p.is_finite() && (0.0..=1.0).contains(p))
        .collect();
    assert!(probs.len() >= 8, "{probs:?}");
    for (i, &p) in probs.iter().enumerate() {
        let dist = [p, 1.0 - p];
        g.f32(negative_log_likelihood(&dist, 0).expect("a probability"));
        g.f32(negative_log_likelihood(&dist, 1).expect("a probability"));
        g.usize(i);
    }
    // temperature scaling: fit (mean NLL over a golden-section search) then apply
    let rows: Vec<Vec<f32>> = (0..12)
        .map(|s| vector(4, s).into_iter().map(bounded).collect())
        .collect();
    let samples: Vec<(&[f32], usize)> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| (r.as_slice(), i % 4))
        .collect();
    let ts = TemperatureScaling::fit(&samples).expect("fit");
    for r in &rows {
        let mut out = vec![0.0f32; r.len()];
        ts.apply(r, &mut out).expect("apply");
        g.f32s(&out);
    }
    g.finish("calibration", GOLDEN_CALIBRATION);
}

#[test]
fn error_propagation_over_the_lattice() {
    let mut g = Golden::new("error_propagation");
    let values = lattice();
    for &sparsity in &values {
        for &variance in &[1.0f32, 1e-8, 0.5] {
            let mut first = LayerConfig::dense(16, 16);
            first.sparsity = sparsity;
            first.weight_variance = variance;
            let mut second = LayerConfig::dense(16, 8);
            second.has_residual = true;
            let report = compute_layer_error_propagation(&[first, second]);
            g.f64(report.total_error);
            g.f64(report.final_sigma_act);
            g.f64(report.geometric_mean_error);
            for layer in &report.layers {
                g.f64(layer.local_error);
                g.f64(layer.sigma_act_after);
                g.f64(layer.cumulative_product);
            }
        }
    }
    g.finish("error_propagation", GOLDEN_ERROR_PROPAGATION);
}

#[cfg(feature = "ffi")]
#[test]
fn ffi_kernels_over_the_lattice() {
    use alice_ml::ffi::{
        am_ml_tensor_max, am_ml_tensor_min, am_ml_tensor_relu, am_ml_tensor_relu_inplace,
        am_ml_tensor_softmax,
    };
    let mut g = Golden::new("ffi");
    for len in LENGTHS {
        for start in 0..3 {
            let v = vector(len, start);
            // SAFETY: valid pointers and lengths of live vectors
            unsafe {
                g.f32(am_ml_tensor_max(v.as_ptr(), len));
                g.f32(am_ml_tensor_min(v.as_ptr(), len));
                let mut out = vec![0.0f32; len];
                am_ml_tensor_relu(v.as_ptr(), out.as_mut_ptr(), len);
                g.f32s(&out);
                let mut inplace = v.clone();
                am_ml_tensor_relu_inplace(inplace.as_mut_ptr(), len);
                g.f32s(&inplace);
                let mut sm = vec![0.0f32; len];
                am_ml_tensor_softmax(v.as_ptr(), sm.as_mut_ptr(), len);
                g.f32s(&sm);
            }
        }
    }
    g.finish("ffi", GOLDEN_FFI);
}

/// A one-layer model whose attention rotates by the position itself: `head_dim` 2, so the
/// single `RoPE` frequency is `1 / theta^0 = 1` and the angle at position `p` is `p`.
/// The value, output and query / key projections are identities and the FFN is zero, so the
/// logits at `p` depend on the rotation of the query at `p` against every earlier key.
#[cfg(feature = "safetensors")]
fn rope_model() -> alice_ml::llama3_ternary::Llama3TernaryModel {
    use alice_ml::llama3_ternary::{Llama3TernaryConfig, Llama3TernaryModel};
    use alice_ml::TernaryWeight;
    const H: usize = 4;
    const KV: usize = 2;
    let identity = |out: usize, inp: usize| {
        let v: Vec<i8> = (0..out * inp)
            .map(|k| i8::from(k / inp == k % inp))
            .collect();
        TernaryWeight::from_ternary(&v, out, inp)
    };
    let zero =
        |out: usize, inp: usize| TernaryWeight::from_ternary(&vec![0i8; out * inp], out, inp);
    let config = Llama3TernaryConfig {
        vocab_size: 4,
        hidden_dim: H,
        intermediate_dim: H,
        num_heads: 2,
        num_kv_heads: 1,
        num_layers: 1,
        max_seq_len: ROPE_POSITIONS,
        head_dim: 2,
        rope_theta: 500_000.0,
        norm_eps: 1e-5,
    };
    // row t is t·4 + 1 .. t·4 + 4: distinct, non-degenerate, built in integers
    let embedding: Vec<f32> = (0..4 * H)
        .map(|k| f32::from(u16::try_from(k % H + 1 + 4 * (k / H)).expect("small")))
        .collect();
    let projs = vec![
        identity(H, H),
        identity(KV, H),
        identity(KV, H),
        identity(H, H),
        zero(H, H),
        zero(H, H),
        zero(H, H),
    ];
    Llama3TernaryModel::from_parts(
        config,
        embedding,
        vec![1.0; H],
        identity(H, H),
        vec![(vec![1.0; H], vec![1.0; H], projs)],
    )
}

/// Positions the `RoPE` goldens run to: the last 64 below `2^13` and the first 64 from it
#[cfg(feature = "safetensors")]
const ROPE_POSITIONS: usize = 8192 + 64;

/// The logits at positions `8128..8192` and `8192..8256` of [`rope_model`] fed tokens
/// `p mod 4`. `alice-det-math`'s `sin_cos` reduces `|x| < 2^13` in single precision and larger
/// arguments in double precision (since 2d01c63f), so the first window pins the single-precision
/// path and the second the double-precision one.
#[cfg(feature = "safetensors")]
fn rope_windows() -> (Golden, Golden) {
    let mut model = rope_model();
    let mut below = Golden::new("rope_below_2_13");
    let mut above = Golden::new("rope_from_2_13");
    for p in 0..ROPE_POSITIONS {
        let logits = model.forward(u32::try_from(p % 4).expect("a token below 4"));
        if (ROPE_POSITIONS - 128..8192).contains(&p) {
            below.f32s(&logits);
        } else if p >= 8192 {
            above.f32s(&logits);
        }
    }
    (below, above)
}

#[cfg(feature = "safetensors")]
#[test]
fn rope_below_and_from_2_13() {
    let (g, above) = rope_windows();
    g.finish("rope_below_2_13", GOLDEN_ROPE_BELOW_2_13);
    let g = above;
    g.finish("rope_from_2_13", GOLDEN_ROPE_FROM_2_13);
}

// recorded digests (see the module documentation)
const GOLDEN_RELU: &str = "eb212a7bfd02a74cd9c30587ddd268e2d63610d92de8eb2364cf99d166fd43f3";
const GOLDEN_MAX_MIN: &str = "8a2bc26d6793bb71db4c654834f5e99b1a4c8d621dc338406390f7ccea7aed0b";
const GOLDEN_SOFTMAX: &str = "c316a92a504431af0544bd90cd4a0a533daaef9ddbb1844d24434f00b672f68b";
const GOLDEN_QUANTIZE: &str = "7bd392c8df625897f748e81f6e55c7913ce730c48c987296da923a0d3ab66467";
const GOLDEN_INT8: &str = "c1cf22da2aa5fbd92ca976133addf5025554ff1ae0f1ddc512bd554f68848d7a";
const GOLDEN_CROSS_ENTROPY: &str =
    "9f8a614a5ed5fd15708cbf64979538246d7a07e7ce7dcc8e3a1eef48edcf3942";
const GOLDEN_CALIBRATION: &str = "390365518b328c524fc18fff90190baac69856a975f144c3e52f4d66d6dd6f0d";
const GOLDEN_ERROR_PROPAGATION: &str =
    "cef6d92869c8990d068fa59673af0b272a59c35868f784533bd4951a91eaf52e";
#[cfg(feature = "ffi")]
const GOLDEN_FFI: &str = "738bd4170174a7112ce57923999bb3229da5e325f902601fc72a862ff9235ce0";
#[cfg(feature = "safetensors")]
const GOLDEN_ROPE_BELOW_2_13: &str =
    "e6097785b5a566fac3b64b8360242bf99ffd66f1d460d9c6f94112b9327bbefe";
#[cfg(feature = "safetensors")]
const GOLDEN_ROPE_FROM_2_13: &str =
    "4631d545e20dc6d6c4eeceb8c506ccef10400e44280e79cff33a0caef4854ad1";
