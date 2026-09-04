//! Fuzz target: weights file parse が任意 bytes で panic しないことを検証
//!
//! 攻撃者制御 weights file (safetensors / bincode / raw f32 stream 想定) で:
//! - header 長 overflow による allocation panic
//! - shape metadata の integer overflow
//! - data section 長 mismatch による OOB
//!
//! を全て有限時間で panic なく完了することを保証する
//!
//! canonical CI template [[reference_alice_ci_canonical_template]] 準拠
//!
//! Note: alice-ml lib.rs は現状 empty のため、defensive stub として arbitrary bytes を
//! header + payload 分割 + bounds check のみ実施。実 weights load API (safetensors) が
//! 導入されたら本 target を差替える。

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Debug, Arbitrary)]
struct Input {
    header_len_raw: u16,
    payload: Vec<u8>,
}

fuzz_target!(|input: Input| {
    // 巨大 input による fuzzer timeout 回避
    if input.payload.len() > 128 * 1024 {
        return;
    }

    // header_len は payload の実長より大きくなる可能性がある (攻撃者は嘘 header を書ける)
    let claimed_header_len = (input.header_len_raw as usize) % 4096;

    // defensive: header_len が payload を超えたら early return (panic ではなく Err 相当)
    if claimed_header_len > input.payload.len() {
        return;
    }

    // header section を取り出す (bounds check 済)
    let header = &input.payload[..claimed_header_len];
    let data = &input.payload[claimed_header_len..];

    // header を UTF-8 として解釈 (safetensors JSON header 相当)、失敗しても panic せず None 相当
    let _header_str = std::str::from_utf8(header).ok();

    // data section を f32 stream として解釈
    let mut sum = 0.0f32;
    for chunk in data.chunks_exact(4).take(1024) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let v = f32::from_le_bytes(bytes);
        if v.is_finite() {
            sum += v;
        }
    }

    std::hint::black_box(sum);
});
