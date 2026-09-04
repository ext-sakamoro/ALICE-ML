//! Fuzz target: tensor deserialize 経路が任意入力で panic しないことを検証
//!
//! 攻撃者制御 bytes を tensor 形式 (dim / shape / data) として解釈:
//! - shape overflow による OOM
//! - dim mismatch による OOB
//! - data 長 mismatch による copy_from_slice length panic
//!
//! を全て有限時間で panic なく完了することを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠
//!
//! Note: alice-ml lib.rs は現状 empty のため、defensive stub として arbitrary bytes を
//! shape 解釈 + bounds check のみ実施。実 public API (tensor deserialize) が導入されたら
//! 本 target を差替える。

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    shape_raw: [u8; 4],
    data: Vec<u8>,
}

fuzz_target!(|input: Input| {
    // 巨大 input による fuzzer timeout 回避
    if input.data.len() > 64 * 1024 {
        return;
    }

    // shape 各次元を 64 以下に強制 (OOM 予防)
    let dims: Vec<usize> = input
        .shape_raw
        .iter()
        .map(|&b| ((b as usize) % 64).max(1))
        .collect();

    // total element count (saturating で overflow 防止)
    let total: usize = dims.iter().copied().fold(1usize, usize::saturating_mul);

    // f32 tensor 想定で total * 4 bytes 必要
    let need = total.saturating_mul(4);
    if input.data.len() < need {
        return;
    }

    // defensive parse: 各 4 bytes を f32 として解釈、NaN/Inf も許容 (panic 予防のみ検証)
    let mut sum = 0.0f32;
    for chunk in input.data.chunks_exact(4).take(total) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let v = f32::from_le_bytes(bytes);
        if v.is_finite() {
            sum += v;
        }
    }
    // sum を黒箱化 (dead code elimination 予防)
    std::hint::black_box(sum);
});
