#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use alice_ml::{TernaryWeight, ternary_matvec};

#[derive(Debug, Arbitrary)]
struct MatvecInput {
    /// Ternary weight values (mapped to -1/0/+1)
    weights: Vec<i8>,
    /// Input activations
    input: Vec<f32>,
    /// Matrix dimensions (capped to prevent OOM)
    out_features: u8,
    in_features: u8,
}

// Fuzz ternary matvec with random dimensions, weights, and input.
// Primary goal: no panics, no OOB, no UB — even with extreme f32 values.
fuzz_target!(|input: MatvecInput| {
    let out_f = (input.out_features as usize).max(1).min(64);
    let in_f = (input.in_features as usize).max(1).min(64);

    // Clamp weights to ternary range and pad/truncate to exact size
    let total = out_f * in_f;
    let mut ternary: Vec<i8> = input.weights.iter()
        .map(|&w| match w {
            w if w > 0 => 1i8,
            w if w < 0 => -1i8,
            _ => 0i8,
        })
        .take(total)
        .collect();
    while ternary.len() < total {
        ternary.push(0);
    }

    let w = TernaryWeight::from_ternary(&ternary, out_f, in_f);

    // Prepare input — allow any finite f32 (overflow to ±inf is expected, not UB)
    let mut inp: Vec<f32> = input.input.iter()
        .copied()
        .filter(|v| v.is_finite())
        .take(in_f)
        .collect();
    while inp.len() < in_f {
        inp.push(0.0);
    }

    let mut output = vec![0.0f32; out_f];

    // Must not panic or OOB — that's the security invariant
    ternary_matvec(&inp, &w, &mut output);
});
