//! Mixture-of-Experts (`MoE`) ルーター
//!
//! Transformer `MoE` レイヤーのエキスパート選択を実装。
//! Top-K gating で各トークンに最適なエキスパートを選択する。
//!
//! # アーキテクチャ
//!
//! ```text
//! input (hidden_dim)
//!   │
//!   ▼
//! gate_weights (hidden_dim × num_experts)  ← ternary_matvec
//!   │
//!   ▼
//! softmax → top_k selection
//!   │
//!   ▼
//! ExpertChoice { expert_id, weight }
//! ```
//!
//! # ALICE-Cache との統合
//!
//! `ExpertSelector::predict_next_experts()` が次トークンで必要になる
//! エキスパートを予測。ALICE-Cache の `batch_put` でプリフェッチに利用可能。

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::ops::{ternary_matvec, TernaryWeight};

/// エキスパート選択結果
#[derive(Debug, Clone, Copy)]
pub struct ExpertChoice {
    /// エキスパートID (`0..num_experts`)
    pub expert_id: usize,
    /// ゲート重み（softmax後）
    pub weight: f32,
}

/// `MoE` ゲートルーター
///
/// ternary重みでゲーティングスコアを計算し、Top-K選択を行う。
pub struct MoeRouter {
    /// ゲート重み: `hidden_dim` → `num_experts`
    gate_weights: TernaryWeight,
    /// エキスパート数
    num_experts: usize,
    /// Top-K (通常 1 or 2)
    top_k: usize,
}

impl MoeRouter {
    /// 新規ルーター作成
    ///
    /// # Panics
    ///
    /// `top_k > num_experts` の場合パニック。
    #[must_use]
    pub fn new(gate_weights: TernaryWeight, num_experts: usize, top_k: usize) -> Self {
        assert!(top_k <= num_experts, "top_k must be <= num_experts");
        Self {
            gate_weights,
            num_experts,
            top_k,
        }
    }

    /// 入力ベクトルからTop-Kエキスパートを選択
    ///
    /// `gate_buf` はゲーティングスコア用ワークバッファ（長さ = `num_experts`）。
    /// DPSパターンでヒープ確保を回避。
    ///
    /// # Panics
    ///
    /// `gate_buf.len() < num_experts` の場合パニック。
    pub fn route(&self, input: &[f32], gate_buf: &mut [f32]) -> Vec<ExpertChoice> {
        assert!(gate_buf.len() >= self.num_experts, "gate_buf too small");

        // ゲーティングスコア計算（ternary matvec）
        let out = &mut gate_buf[..self.num_experts];
        ternary_matvec(input, &self.gate_weights, out);

        // softmax（数値安定版）
        softmax_inplace(out);

        // Top-K選択
        top_k_select(out, self.top_k)
    }

    /// エキスパート数を返す
    #[must_use]
    pub const fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Top-K値を返す
    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }
}

/// エキスパート選択予測器
///
/// 直近のエキスパート選択パターンを記録し、
/// 次トークンで必要になるエキスパートを予測する。
/// ALICE-Cache のプリフェッチと組み合わせて使用。
pub struct ExpertSelector {
    /// エキスパート数
    num_experts: usize,
    /// 遷移カウント: `transitions[prev * num_experts + next]` = 出現回数
    transitions: Vec<u32>,
    /// 直前に選択されたエキスパート（最初のルーティング前はNone）
    last_expert: Option<usize>,
}

impl ExpertSelector {
    /// 新規セレクター作成
    #[must_use]
    pub fn new(num_experts: usize) -> Self {
        Self {
            num_experts,
            transitions: vec![0u32; num_experts * num_experts],
            last_expert: None,
        }
    }

    /// エキスパート選択を記録（遷移テーブル更新）
    pub fn record(&mut self, choices: &[ExpertChoice]) {
        for c in choices {
            if let Some(prev) = self.last_expert {
                if prev < self.num_experts && c.expert_id < self.num_experts {
                    let idx = prev * self.num_experts + c.expert_id;
                    self.transitions[idx] = self.transitions[idx].saturating_add(1);
                }
            }
            self.last_expert = Some(c.expert_id);
        }
    }

    /// 次トークンで使用されるエキスパートを予測
    ///
    /// 直前のエキスパートからの遷移確率が高い順にTop-N個を返す。
    /// `max_predictions` は返す予測数の上限。
    #[must_use]
    pub fn predict_next_experts(&self, max_predictions: usize) -> Vec<usize> {
        let Some(prev) = self.last_expert else {
            return Vec::new();
        };
        if prev >= self.num_experts {
            return Vec::new();
        }

        let base = prev * self.num_experts;
        let row = &self.transitions[base..base + self.num_experts];
        let mut indexed: Vec<(usize, u32)> = row.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.cmp(&a.1));

        indexed
            .iter()
            .filter(|(_, count)| *count > 0)
            .take(max_predictions)
            .map(|(id, _)| *id)
            .collect()
    }

    /// 遷移テーブルをリセット
    pub fn reset(&mut self) {
        self.transitions.iter_mut().for_each(|c| *c = 0);
        self.last_expert = None;
    }
}

/// softmax（数値安定版、in-place）
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    // max探索
    let mut max_val = x[0];
    for &v in &x[1..] {
        if v > max_val {
            max_val = v;
        }
    }
    // exp + sum
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = fast_exp(*v - max_val);
        sum += *v;
    }
    // 正規化
    if sum > 0.0 {
        let inv_sum = 1.0 / sum;
        for v in x.iter_mut() {
            *v *= inv_sum;
        }
    }
}

/// 高速exp近似（range reduction + 3次多項式）
#[inline(always)]
fn fast_exp(x: f32) -> f32 {
    let clamped = x.clamp(-87.0_f32, 88.0_f32);
    let val = clamped * core::f32::consts::LOG2_E;
    let ipart = floor_f32(val);
    let fpart = val - ipart;
    let n = ipart as i32;
    let p = fpart.mul_add(0.0558_f32, 0.2402_f32);
    let p = fpart.mul_add(p, core::f32::consts::LN_2);
    let p = fpart.mul_add(p, 1.0_f32);
    let exp_n = f32::from_bits(((n + 127) as u32) << 23);
    exp_n * p
}

/// `no_std` 互換 floor
#[inline(always)]
fn floor_f32(x: f32) -> f32 {
    let i = x as i32;
    let f = i as f32;
    if x < f {
        f - 1.0
    } else {
        f
    }
}

/// Top-K選択（部分ソート）
fn top_k_select(scores: &[f32], k: usize) -> Vec<ExpertChoice> {
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    // 上位K個を選ぶために降順ソート
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));

    indexed
        .iter()
        .take(k)
        .map(|&(id, weight)| ExpertChoice {
            expert_id: id,
            weight,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moe_router_basic() {
        // 4 experts, hidden_dim=4, top-2
        // ゲート重み: 4×4 identity-like
        let gate_data = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let gate_weights = TernaryWeight::from_ternary(&gate_data, 4, 4);
        let router = MoeRouter::new(gate_weights, 4, 2);

        // 入力 [10, 1, 1, 1] → expert 0 が最大スコア
        let input = [10.0_f32, 1.0, 1.0, 1.0];
        let mut buf = [0.0_f32; 4];
        let choices = router.route(&input, &mut buf);

        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0].expert_id, 0);
        assert!(choices[0].weight > choices[1].weight);
    }

    #[test]
    fn test_moe_router_top1() {
        let gate_data = [1, 0, 0, 1, 0, 0, 0, 0, 1];
        let gate_weights = TernaryWeight::from_ternary(&gate_data, 3, 3);
        let router = MoeRouter::new(gate_weights, 3, 1);

        let input = [0.0_f32, 0.0, 10.0];
        let mut buf = [0.0_f32; 3];
        let choices = router.route(&input, &mut buf);

        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].expert_id, 2);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut scores = [1.0_f32, 2.0, 3.0, 4.0];
        softmax_inplace(&mut scores);

        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "softmax sum = {sum}");

        // 単調性: scores[3] > scores[2] > scores[1] > scores[0]
        assert!(scores[3] > scores[2]);
        assert!(scores[2] > scores[1]);
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn test_softmax_empty() {
        let mut scores: [f32; 0] = [];
        softmax_inplace(&mut scores);
    }

    #[test]
    fn test_top_k_select() {
        let scores = [0.1_f32, 0.5, 0.3, 0.1];
        let result = top_k_select(&scores, 2);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].expert_id, 1);
        assert_eq!(result[1].expert_id, 2);
    }

    #[test]
    fn test_expert_selector_prediction() {
        let mut selector = ExpertSelector::new(4);

        // パターン学習: 0 → 1 → 2 → 0 → 1 → 2 ...
        for _ in 0..50 {
            selector.record(&[ExpertChoice {
                expert_id: 0,
                weight: 1.0,
            }]);
            selector.record(&[ExpertChoice {
                expert_id: 1,
                weight: 1.0,
            }]);
            selector.record(&[ExpertChoice {
                expert_id: 2,
                weight: 1.0,
            }]);
        }

        // last_expert = 2 → 次は 0 が最も多い
        let preds = selector.predict_next_experts(2);
        assert!(!preds.is_empty());
        assert_eq!(preds[0], 0);
    }

    #[test]
    fn test_expert_selector_no_history() {
        let selector = ExpertSelector::new(4);
        let preds = selector.predict_next_experts(2);
        assert!(preds.is_empty());
    }

    #[test]
    fn test_expert_selector_reset() {
        let mut selector = ExpertSelector::new(4);
        selector.record(&[ExpertChoice {
            expert_id: 0,
            weight: 1.0,
        }]);
        selector.record(&[ExpertChoice {
            expert_id: 1,
            weight: 1.0,
        }]);

        selector.reset();
        assert!(selector.predict_next_experts(2).is_empty());
    }

    #[test]
    fn test_fast_exp_accuracy() {
        let result = fast_exp(0.0);
        assert!((result - 1.0).abs() < 0.01);

        let result = fast_exp(1.0);
        assert!((result - core::f32::consts::E).abs() < 0.05);
    }

    #[test]
    fn test_moe_router_num_experts() {
        let gate_data = [1, 0, 0, 1];
        let gate_weights = TernaryWeight::from_ternary(&gate_data, 2, 2);
        let router = MoeRouter::new(gate_weights, 2, 1);
        assert_eq!(router.num_experts(), 2);
        assert_eq!(router.top_k(), 1);
    }

    #[test]
    fn test_expert_selector_saturating() {
        let mut selector = ExpertSelector::new(2);
        // u32::MAX回記録してもオーバーフローしない
        selector.last_expert = Some(0);
        selector.transitions[0 * 2 + 1] = u32::MAX;
        selector.record(&[ExpertChoice {
            expert_id: 1,
            weight: 1.0,
        }]);
        assert_eq!(selector.transitions[0 * 2 + 1], u32::MAX);
    }
}
