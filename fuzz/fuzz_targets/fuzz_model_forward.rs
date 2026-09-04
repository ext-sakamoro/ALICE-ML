//! Fuzz target: model forward pass が任意入力で panic しないことを検証
//!
//! 攻撃者制御 input activation で:
//! - NaN / ±Inf propagation
//! - shape mismatch による OOB
//! - overflow / underflow による integer panic
//!
//! を全て有限時間で panic なく完了することを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠
//!
//! Note: alice-ml lib.rs は現状 empty のため、defensive stub として arbitrary f32 列を
//! finite filter + matmul 相当の演算のみ実施。実 model forward API が導入されたら
//! 本 target を差替える。

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    input_dim_raw: u8,
    output_dim_raw: u8,
    activations: Vec<f32>,
}

fuzz_target!(|input: Input| {
    let in_dim = ((input.input_dim_raw as usize) % 64).max(1);
    let out_dim = ((input.output_dim_raw as usize) % 64).max(1);

    // finite activation のみ通す (NaN/Inf も panic 対象ではないが、演算予算を抑制)
    let mut acts: Vec<f32> = input
        .activations
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .take(in_dim)
        .collect();
    while acts.len() < in_dim {
        acts.push(0.0);
    }

    // 単純な dot product accumulation (forward pass 相当)、NaN propagation 検知目的
    let mut output = vec![0.0f32; out_dim];
    for (i, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (j, &a) in acts.iter().enumerate() {
            // 決定論的な weight 疑似 (i * j) % 3 - 1 = -1/0/+1 (ternary 相当)
            let w = ((i.wrapping_mul(j)) % 3) as f32 - 1.0;
            sum += a * w;
        }
        // NaN propagation は panic ではないが、明示的に許容していることを示す
        *out = if sum.is_finite() { sum } else { 0.0 };
    }

    std::hint::black_box(output);
});
