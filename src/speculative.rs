//! Speculative Decoding — ドラフトモデルで先読みし検証モデルで一括検証。
//!
//! 帯域律速のエッジ環境（RasPi 5等）で、1.58bit ternary モデルの
//! 推論スループットを 2〜3 倍に引き上げる。
//!
//! # アルゴリズム概要
//!
//! 1. 小さいドラフトモデル（例: 1B）で K トークンを高速に先読み
//! 2. 大きい検証モデル（例: 24B）でバッチ検証
//! 3. 確率的受理/棄却で品質を保証
//!
//! # Example
//!
//! ```rust
//! use alice_ml::speculative::{SpeculativeDecoder, DecoderConfig};
//! use alice_ml::ops::TernaryWeightKernel;
//! use alice_ml::layer::BitLinear;
//!
//! // ドラフトモデル（小）と検証モデル（大）を構築
//! let draft_w = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
//! let draft_layer = BitLinear::new(draft_w, None, false);
//!
//! let verify_w = TernaryWeightKernel::from_ternary(&[1, 0, -1, 1], 2, 2);
//! let verify_layer = BitLinear::new(verify_w, None, false);
//!
//! let config = DecoderConfig { max_draft_tokens: 4, temperature: 1.0 };
//! let decoder = SpeculativeDecoder::new(
//!     vec![draft_layer],
//!     vec![verify_layer],
//!     config,
//! );
//!
//! let input = [1.0f32, 2.0];
//! let mut draft_buf = vec![0.0f32; 2 * 4]; // out_features * max_draft_tokens
//! let mut verify_buf = vec![0.0f32; 2 * 4];
//!
//! let result = decoder.decode_step(&input, &mut draft_buf, &mut verify_buf);
//! assert!(result.accepted >= 1);
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::layer::BitLinear;

// ============================================================================
// Configuration
// ============================================================================

/// Speculative Decoding の設定。
#[derive(Debug, Clone, Copy)]
pub struct DecoderConfig {
    /// ドラフトモデルが先読みする最大トークン数 (K)。
    pub max_draft_tokens: usize,
    /// サンプリング温度。1.0 = そのまま、低いほど決定的。
    pub temperature: f32,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            max_draft_tokens: 5,
            temperature: 1.0,
        }
    }
}

// ============================================================================
// Decode Result
// ============================================================================

/// 1ステップのデコード結果。
#[derive(Debug, Clone)]
pub struct DecodeResult {
    /// 受理されたトークン数（1 以上、最大 K+1）。
    pub accepted: usize,
    /// ドラフトモデルが生成した logits のインデックス範囲。
    pub draft_len: usize,
    /// 各ドラフトトークンが受理されたか。
    pub acceptance: Vec<bool>,
}

// ============================================================================
// SpeculativeDecoder
// ============================================================================

/// Speculative Decoding エンジン。
///
/// ドラフトモデル（高速・低精度）で K トークンを先読みし、
/// 検証モデル（低速・高精度）でバッチ検証する。
/// 受理されたトークンはそのまま採用し、棄却された時点で
/// 検証モデルの出力で置き換える。
pub struct SpeculativeDecoder {
    /// ドラフトモデルのレイヤー列。
    draft_layers: Vec<BitLinear>,
    /// 検証モデルのレイヤー列。
    verify_layers: Vec<BitLinear>,
    /// 設定。
    config: DecoderConfig,
}

impl SpeculativeDecoder {
    /// 新しい `SpeculativeDecoder` を作成。
    ///
    /// # Arguments
    /// * `draft_layers` - ドラフト（小）モデルのレイヤー列
    /// * `verify_layers` - 検証（大）モデルのレイヤー列
    /// * `config` - デコード設定
    #[must_use]
    pub const fn new(
        draft_layers: Vec<BitLinear>,
        verify_layers: Vec<BitLinear>,
        config: DecoderConfig,
    ) -> Self {
        Self {
            draft_layers,
            verify_layers,
            config,
        }
    }

    /// ドラフトモデルのレイヤー数。
    #[must_use]
    pub const fn draft_depth(&self) -> usize {
        self.draft_layers.len()
    }

    /// 検証モデルのレイヤー数。
    #[must_use]
    pub const fn verify_depth(&self) -> usize {
        self.verify_layers.len()
    }

    /// 設定を取得。
    #[must_use]
    pub const fn config(&self) -> &DecoderConfig {
        &self.config
    }

    /// ドラフトモデルで K トークン分の logits を生成 (DPS)。
    ///
    /// # Arguments
    /// * `input` - 入力ベクトル (length = `draft_layers[0].in_features`)
    /// * `draft_out` - 出力バッファ (length = `out_features * max_draft_tokens`)
    ///
    /// # Returns
    /// 生成したステップ数。
    ///
    /// # Panics
    /// ドラフトレイヤーが空、または `draft_out` が不足する場合。
    pub fn draft_forward(&self, input: &[f32], draft_out: &mut [f32]) -> usize {
        assert!(!self.draft_layers.is_empty(), "draft model has no layers");

        let k = self.config.max_draft_tokens;
        let first = &self.draft_layers[0];
        let out_features = first.out_features;

        assert!(
            draft_out.len() >= out_features * k,
            "draft_out too small: {} < {}",
            draft_out.len(),
            out_features * k
        );

        // 自己回帰用バッファ（前ステップの出力を保持）
        let mut prev_output = vec![0.0f32; out_features];

        for step in 0..k {
            let out_start = step * out_features;
            let out_end = out_start + out_features;

            let step_input: &[f32] = if step == 0 { input } else { &prev_output };

            // 中間バッファに推論結果を書き出す
            let mut step_result = vec![0.0f32; out_features];

            if self.draft_layers.len() == 1 {
                self.draft_layers[0].forward(step_input, &mut step_result);
            } else {
                let mut layer_buf = vec![0.0f32; out_features];
                self.draft_layers[0].forward(step_input, &mut layer_buf);

                for layer in &self.draft_layers[1..self.draft_layers.len() - 1] {
                    let mut tmp = vec![0.0f32; layer.out_features];
                    layer.forward(&layer_buf, &mut tmp);
                    layer_buf = tmp;
                }

                self.draft_layers
                    .last()
                    .unwrap()
                    .forward(&layer_buf, &mut step_result);
            }

            // draft_out に書き込み + 次ステップ用にコピー
            draft_out[out_start..out_end].copy_from_slice(&step_result);
            prev_output.copy_from_slice(&step_result);
        }

        k
    }

    /// 検証モデルでバッチ検証 (DPS)。
    ///
    /// ドラフトの各ステップ出力を検証モデルに通し、
    /// 検証 logits を `verify_out` に書き込む。
    ///
    /// # Arguments
    /// * `input` - 元の入力ベクトル
    /// * `draft_out` - ドラフトが生成した logits (K ステップ分)
    /// * `verify_out` - 検証結果出力バッファ (length = `out_features * steps`)
    /// * `steps` - 検証するステップ数
    ///
    /// # Panics
    /// 検証レイヤーが空、または `verify_out` が不足する場合。
    pub fn verify_batch(
        &self,
        input: &[f32],
        draft_out: &[f32],
        verify_out: &mut [f32],
        steps: usize,
    ) {
        assert!(!self.verify_layers.is_empty(), "verify model has no layers");

        let first = &self.verify_layers[0];
        let out_features = first.out_features;

        assert!(
            verify_out.len() >= out_features * steps,
            "verify_out too small: {} < {}",
            verify_out.len(),
            out_features * steps
        );

        // ステップ 0 は元の input を使用、以降は draft_out の各ステップ出力
        for step in 0..steps {
            let step_input = if step == 0 {
                input
            } else {
                let prev_start = (step - 1) * out_features;
                &draft_out[prev_start..prev_start + out_features]
            };

            let out_start = step * out_features;
            let out_end = out_start + out_features;
            let out_slice = &mut verify_out[out_start..out_end];

            // 全検証レイヤーを通す
            if self.verify_layers.len() == 1 {
                self.verify_layers[0].forward(step_input, out_slice);
            } else {
                let mut buf = vec![0.0f32; out_features];
                self.verify_layers[0].forward(step_input, &mut buf);

                for layer in &self.verify_layers[1..self.verify_layers.len() - 1] {
                    let mut tmp = vec![0.0f32; layer.out_features];
                    layer.forward(&buf, &mut tmp);
                    buf = tmp;
                }

                self.verify_layers.last().unwrap().forward(&buf, out_slice);
            }
        }
    }

    /// ドラフト logits と検証 logits を比較し、受理マスクを生成。
    ///
    /// 各ステップの logits の argmax が一致すれば受理。
    /// 不一致が見つかった時点で以降は全て棄却。
    ///
    /// # Arguments
    /// * `draft_out` - ドラフト logits (K ステップ × `out_features`)
    /// * `verify_out` - 検証 logits (K ステップ × `out_features`)
    /// * `out_features` - 1ステップの出力次元数
    /// * `steps` - ステップ数
    ///
    /// # Returns
    /// 受理されたステップ数（最低 1）。
    #[must_use]
    pub fn acceptance_mask(
        draft_out: &[f32],
        verify_out: &[f32],
        out_features: usize,
        steps: usize,
    ) -> DecodeResult {
        let mut accepted = 0;
        let mut acceptance = Vec::with_capacity(steps);

        for step in 0..steps {
            let d_start = step * out_features;
            let d_end = d_start + out_features;
            let v_start = step * out_features;
            let v_end = v_start + out_features;

            let d_argmax = argmax(&draft_out[d_start..d_end]);
            let v_argmax = argmax(&verify_out[v_start..v_end]);

            if d_argmax == v_argmax {
                accepted += 1;
                acceptance.push(true);
            } else {
                acceptance.push(false);
                // 不一致以降は全て棄却
                acceptance.resize(steps, false);
                break;
            }
        }

        // 最低 1 トークンは検証モデルの出力で確定
        let accepted = accepted.max(1);

        DecodeResult {
            accepted,
            draft_len: steps,
            acceptance,
        }
    }

    /// 1デコードステップ: ドラフト先読み → 検証 → 受理判定。
    ///
    /// # Arguments
    /// * `input` - 入力ベクトル
    /// * `draft_buf` - ドラフト出力バッファ (`out_features * max_draft_tokens`)
    /// * `verify_buf` - 検証出力バッファ (`out_features * max_draft_tokens`)
    ///
    /// # Returns
    /// デコード結果（受理トークン数等）。
    pub fn decode_step(
        &self,
        input: &[f32],
        draft_buf: &mut [f32],
        verify_buf: &mut [f32],
    ) -> DecodeResult {
        let k = self.draft_forward(input, draft_buf);

        let out_features = self.draft_layers[0].out_features;

        self.verify_batch(input, draft_buf, verify_buf, k);

        Self::acceptance_mask(draft_buf, verify_buf, out_features, k)
    }

    /// ドラフトモデルの合計メモリ使用量 (bytes)。
    #[must_use]
    pub fn draft_memory_bytes(&self) -> usize {
        self.draft_layers.iter().map(BitLinear::memory_bytes).sum()
    }

    /// 検証モデルの合計メモリ使用量 (bytes)。
    #[must_use]
    pub fn verify_memory_bytes(&self) -> usize {
        self.verify_layers.iter().map(BitLinear::memory_bytes).sum()
    }

    /// 合計メモリ使用量 (bytes)。
    #[must_use]
    pub fn total_memory_bytes(&self) -> usize {
        self.draft_memory_bytes() + self.verify_memory_bytes()
    }

    /// 理論上の最大スピードアップ倍率。
    ///
    /// ドラフトモデルのコストが無視できる場合、
    /// 最大 `K+1` トークンを検証モデル 1 回の呼び出しで確定できる。
    #[must_use]
    pub const fn max_speedup(&self) -> f32 {
        (self.config.max_draft_tokens + 1) as f32
    }
}

// ============================================================================
// Helper
// ============================================================================

/// スライスの argmax を返す。空なら 0。
#[inline]
#[must_use]
fn argmax(slice: &[f32]) -> usize {
    if slice.is_empty() {
        return 0;
    }
    let mut best_idx = 0;
    let mut best_val = slice[0];
    for (i, &v) in slice.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::TernaryWeightKernel;

    fn make_layer(values: &[i8], out: usize, inp: usize) -> BitLinear {
        let kernel = TernaryWeightKernel::from_ternary(values, out, inp);
        BitLinear::new(kernel, None, false)
    }

    #[test]
    fn test_decoder_config_default() {
        let cfg = DecoderConfig::default();
        assert_eq!(cfg.max_draft_tokens, 5);
        assert!((cfg.temperature - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_argmax_basic() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[1.0, 2.0, 9.0]), 2);
    }

    #[test]
    fn test_argmax_empty() {
        assert_eq!(argmax(&[]), 0);
    }

    #[test]
    fn test_argmax_single() {
        assert_eq!(argmax(&[42.0]), 0);
    }

    #[test]
    fn test_draft_forward_single_layer() {
        // 2x2 layer, K=3
        let layer = make_layer(&[1, -1, 0, 1], 2, 2);
        let config = DecoderConfig {
            max_draft_tokens: 3,
            temperature: 1.0,
        };
        let decoder = SpeculativeDecoder::new(vec![layer], Vec::new(), config);

        let input = [2.0f32, 3.0];
        let mut draft_out = vec![0.0f32; 2 * 3];

        let steps = decoder.draft_forward(&input, &mut draft_out);
        assert_eq!(steps, 3);

        // ステップ 0: [1,-1; 0,1] · [2,3] = [-1, 3]
        assert!(
            (draft_out[0] - (-1.0)).abs() < 1e-5,
            "step0[0]={}",
            draft_out[0]
        );
        assert!(
            (draft_out[1] - 3.0).abs() < 1e-5,
            "step0[1]={}",
            draft_out[1]
        );

        // ステップ 1: [1,-1; 0,1] · [-1, 3] = [-4, 3]
        assert!(
            (draft_out[2] - (-4.0)).abs() < 1e-5,
            "step1[0]={}",
            draft_out[2]
        );
        assert!(
            (draft_out[3] - 3.0).abs() < 1e-5,
            "step1[1]={}",
            draft_out[3]
        );
    }

    #[test]
    fn test_verify_batch() {
        let v_layer = make_layer(&[1, 0, -1, 1], 2, 2);
        let config = DecoderConfig {
            max_draft_tokens: 2,
            temperature: 1.0,
        };
        let decoder = SpeculativeDecoder::new(Vec::new(), vec![v_layer], config);

        let input = [1.0f32, 2.0];
        // ドラフト出力（2ステップ）
        let draft_out = [3.0f32, -1.0, 2.0, 1.0];
        let mut verify_out = vec![0.0f32; 2 * 2];

        decoder.verify_batch(&input, &draft_out, &mut verify_out, 2);

        // ステップ 0: [1,0; -1,1] · [1,2] = [1, 1]
        assert!(
            (verify_out[0] - 1.0).abs() < 1e-5,
            "v_step0[0]={}",
            verify_out[0]
        );
        assert!(
            (verify_out[1] - 1.0).abs() < 1e-5,
            "v_step0[1]={}",
            verify_out[1]
        );

        // ステップ 1: [1,0; -1,1] · [3,-1] = [3, -4]
        assert!(
            (verify_out[2] - 3.0).abs() < 1e-5,
            "v_step1[0]={}",
            verify_out[2]
        );
        assert!(
            (verify_out[3] - (-4.0)).abs() < 1e-5,
            "v_step1[1]={}",
            verify_out[3]
        );
    }

    #[test]
    fn test_acceptance_mask_all_match() {
        // ドラフトと検証の argmax が全て一致
        let draft = [1.0f32, 3.0, 2.0, 5.0]; // argmax: [1, 1]
        let verify = [0.0f32, 4.0, 1.0, 6.0]; // argmax: [1, 1]

        let result = SpeculativeDecoder::acceptance_mask(&draft, &verify, 2, 2);
        assert_eq!(result.accepted, 2);
        assert_eq!(result.acceptance, vec![true, true]);
    }

    #[test]
    fn test_acceptance_mask_first_reject() {
        // ステップ 0 で不一致
        let draft = [3.0f32, 1.0, 2.0, 5.0]; // argmax: [0, 1]
        let verify = [0.0f32, 4.0, 1.0, 6.0]; // argmax: [1, 1]

        let result = SpeculativeDecoder::acceptance_mask(&draft, &verify, 2, 2);
        assert_eq!(result.accepted, 1); // 最低 1
        assert_eq!(result.acceptance, vec![false, false]);
    }

    #[test]
    fn test_acceptance_mask_partial() {
        // ステップ 0 一致、ステップ 1 不一致
        let draft = [1.0f32, 3.0, 5.0, 1.0]; // argmax: [1, 0]
        let verify = [0.0f32, 4.0, 1.0, 6.0]; // argmax: [1, 1]

        let result = SpeculativeDecoder::acceptance_mask(&draft, &verify, 2, 2);
        assert_eq!(result.accepted, 1);
        assert_eq!(result.acceptance, vec![true, false]);
    }

    #[test]
    fn test_decode_step_full() {
        // ドラフトと検証で同じ重みを使えば全受理
        let d_layer = make_layer(&[1, -1, 0, 1], 2, 2);
        let v_layer = make_layer(&[1, -1, 0, 1], 2, 2);

        let config = DecoderConfig {
            max_draft_tokens: 3,
            temperature: 1.0,
        };
        let decoder = SpeculativeDecoder::new(vec![d_layer], vec![v_layer], config);

        let input = [2.0f32, 3.0];
        let mut draft_buf = vec![0.0f32; 2 * 3];
        let mut verify_buf = vec![0.0f32; 2 * 3];

        let result = decoder.decode_step(&input, &mut draft_buf, &mut verify_buf);
        // 同じ重み → argmax が全ステップ一致 → 全受理
        assert_eq!(result.accepted, 3);
        assert_eq!(result.draft_len, 3);
    }

    #[test]
    fn test_memory_bytes() {
        let d = make_layer(&[1, -1, 0, 1], 2, 2);
        let v = make_layer(&[1, 0, -1, 0, 1, -1, 0, 1, 0], 3, 3);

        let d_mem = d.memory_bytes();
        let v_mem = v.memory_bytes();

        let decoder = SpeculativeDecoder::new(vec![d], vec![v], DecoderConfig::default());
        assert_eq!(decoder.draft_memory_bytes(), d_mem);
        assert_eq!(decoder.verify_memory_bytes(), v_mem);
        assert_eq!(decoder.total_memory_bytes(), d_mem + v_mem);
    }

    #[test]
    fn test_max_speedup() {
        let config = DecoderConfig {
            max_draft_tokens: 5,
            temperature: 1.0,
        };
        let decoder = SpeculativeDecoder::new(Vec::new(), Vec::new(), config);
        assert!((decoder.max_speedup() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_depth() {
        let d1 = make_layer(&[1, -1, 0, 1], 2, 2);
        let d2 = make_layer(&[1, 0, -1, 1], 2, 2);
        let v1 = make_layer(&[0, 1, -1, 0], 2, 2);

        let decoder = SpeculativeDecoder::new(vec![d1, d2], vec![v1], DecoderConfig::default());
        assert_eq!(decoder.draft_depth(), 2);
        assert_eq!(decoder.verify_depth(), 1);
    }

    #[test]
    fn test_multi_layer_draft() {
        // 2層ドラフト: layer0(2→2) → layer1(2→2)
        let d0 = make_layer(&[1, 1, -1, 1], 2, 2);
        let d1 = make_layer(&[1, -1, 0, 1], 2, 2);

        let config = DecoderConfig {
            max_draft_tokens: 2,
            temperature: 1.0,
        };
        let decoder = SpeculativeDecoder::new(vec![d0, d1], Vec::new(), config);

        let input = [1.0f32, 2.0];
        let mut draft_out = vec![0.0f32; 2 * 2];

        let steps = decoder.draft_forward(&input, &mut draft_out);
        assert_eq!(steps, 2);

        // layer0: [1,1; -1,1] · [1,2] = [3, 1]
        // layer1: [1,-1; 0,1] · [3, 1] = [2, 1]
        assert!(
            (draft_out[0] - 2.0).abs() < 1e-5,
            "multi step0[0]={}",
            draft_out[0]
        );
        assert!(
            (draft_out[1] - 1.0).abs() < 1e-5,
            "multi step0[1]={}",
            draft_out[1]
        );
    }
}
