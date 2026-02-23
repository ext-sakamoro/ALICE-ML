#![no_main]
use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use alice_ml::{quantize_to_ternary, dequantize_from_ternary};

#[derive(Debug, Arbitrary)]
struct QuantInput {
    weights: Vec<f32>,
    out_features: u8,
    in_features: u8,
}

// Fuzz quantize-then-dequantize roundtrip.
// Primary goal: no panics, no OOB, no UB — even with extreme values.
fuzz_target!(|input: QuantInput| {
    let out_f = (input.out_features as usize).max(1).min(32);
    let in_f = (input.in_features as usize).max(1).min(32);
    let total = out_f * in_f;

    // Build finite weight vector of exact required size
    let mut weights: Vec<f32> = input.weights.iter()
        .copied()
        .filter(|v| v.is_finite())
        .take(total)
        .collect();
    while weights.len() < total {
        weights.push(0.0);
    }

    // Must not panic or OOB
    let (tw, _stats) = quantize_to_ternary(&weights, out_f, in_f);

    // Dequantize roundtrip — must not panic
    let reconstructed = dequantize_from_ternary(&tw);
    assert_eq!(reconstructed.len(), total);
});
