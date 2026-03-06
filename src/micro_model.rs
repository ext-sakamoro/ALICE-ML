//! L2 Cache-Resident Micro Model — CPU キャッシュに「住む」超小型ドラフトモデル。
//!
//! `RasPi 5` の L2 キャッシュ (512KB/コア) に全重みを常駐させ、
//! DRAM 帯域を一切消費せずにトークンを高速生成する。
//! Speculative Decoding のドラフトモデルとして使い、
//! 検証モデルの DRAM 転送を 1 回で済ませる。
//!
//! # Cache Budget
//!
//! | パラメータ数 | 1.58bit サイズ | 収まるキャッシュ |
//! |-------------|---------------|----------------|
//! | 1M          | 125 KB        | L2 (512KB)     |
//! | 4M          | 500 KB        | L2 ギリギリ     |
//! | 8M          | 1 MB          | L3 or 2×L2     |
//!
//! # Example
//!
//! ```rust
//! use alice_ml::micro_model::{MicroModel, MicroModelBuilder, CacheBudget};
//!
//! let budget = CacheBudget::l2_rpi5(); // 512 KB
//!
//! let model = MicroModelBuilder::new(64, 64, budget)
//!     .add_hidden(64)
//!     .add_hidden(64)
//!     .build_random(42);
//!
//! assert!(model.fits_in_budget());
//!
//! let input = vec![1.0f32; 64];
//! let mut output = vec![0.0f32; 64];
//! model.forward(&input, &mut output);
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::layer::BitLinear;
use crate::ops::TernaryWeightKernel;

// ============================================================================
// Cache Budget
// ============================================================================

/// キャッシュ予算。モデルの総メモリ使用量をこの範囲に収める。
#[derive(Debug, Clone, Copy)]
pub struct CacheBudget {
    /// 予算上限 (bytes)。
    pub max_bytes: usize,
    /// 予算名（デバッグ用）。
    pub name: &'static str,
}

impl CacheBudget {
    /// Raspberry Pi 5 L2 キャッシュ (512 KB per core, Cortex-A76)。
    #[must_use]
    pub const fn l2_rpi5() -> Self {
        Self {
            max_bytes: 512 * 1024,
            name: "L2-RasPi5",
        }
    }

    /// Raspberry Pi 5 L2 × 2 コア分 (1 MB)。
    #[must_use]
    pub const fn l2_rpi5_dual() -> Self {
        Self {
            max_bytes: 1024 * 1024,
            name: "L2x2-RasPi5",
        }
    }

    /// 汎用 L2 (256 KB)。
    #[must_use]
    pub const fn l2_256k() -> Self {
        Self {
            max_bytes: 256 * 1024,
            name: "L2-256K",
        }
    }

    /// カスタム予算。
    #[must_use]
    pub const fn custom(max_bytes: usize, name: &'static str) -> Self {
        Self { max_bytes, name }
    }
}

impl Default for CacheBudget {
    fn default() -> Self {
        Self::l2_rpi5()
    }
}

// ============================================================================
// MicroModel
// ============================================================================

/// L2 キャッシュ常駐マイクロモデル。
///
/// 全レイヤーの重みを `CacheBudget` 内に収め、推論時に DRAM アクセスを
/// 発生させない。`BitLinear` (ternary bit-parallel) レイヤーの列で構成。
///
/// `#[repr(C, align(64))]` の `TernaryWeightKernel` を内部で使用するため、
/// 重みは cache-line 境界に自然にアラインされる。
pub struct MicroModel {
    /// レイヤー列。
    layers: Vec<BitLinear>,
    /// キャッシュ予算。
    budget: CacheBudget,
    /// 入力特徴数。
    in_features: usize,
    /// 出力特徴数。
    out_features: usize,
}

impl MicroModel {
    /// レイヤー列とキャッシュ予算から直接構築。
    ///
    /// # Panics
    /// `layers` が空の場合。
    #[must_use]
    pub fn new(layers: Vec<BitLinear>, budget: CacheBudget) -> Self {
        assert!(!layers.is_empty(), "MicroModel requires at least one layer");
        let in_features = layers[0].in_features;
        let out_features = layers.last().unwrap().out_features;
        Self {
            layers,
            budget,
            in_features,
            out_features,
        }
    }

    /// 全レイヤーの合計メモリ使用量 (bytes)。
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(BitLinear::memory_bytes).sum()
    }

    /// 予算内に収まっているか。
    #[must_use]
    pub fn fits_in_budget(&self) -> bool {
        self.memory_bytes() <= self.budget.max_bytes
    }

    /// キャッシュ予算の使用率 (0.0〜1.0+)。1.0 超は予算超過。
    #[must_use]
    pub fn budget_utilization(&self) -> f32 {
        if self.budget.max_bytes == 0 {
            return f32::INFINITY;
        }
        self.memory_bytes() as f32 / self.budget.max_bytes as f32
    }

    /// 予算の残りバイト数。予算超過なら 0。
    #[must_use]
    pub fn budget_remaining(&self) -> usize {
        self.budget.max_bytes.saturating_sub(self.memory_bytes())
    }

    /// レイヤー数。
    #[must_use]
    pub const fn depth(&self) -> usize {
        self.layers.len()
    }

    /// 入力特徴数。
    #[must_use]
    pub const fn in_features(&self) -> usize {
        self.in_features
    }

    /// 出力特徴数。
    #[must_use]
    pub const fn out_features(&self) -> usize {
        self.out_features
    }

    /// キャッシュ予算。
    #[must_use]
    pub const fn budget(&self) -> &CacheBudget {
        &self.budget
    }

    /// 合計パラメータ数（三値重みの数）。
    #[must_use]
    pub fn param_count(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.in_features * l.out_features)
            .sum()
    }

    /// 推論 (DPS)。全レイヤーを順に通す。
    ///
    /// # Panics
    /// `input.len() != in_features` または `output.len() != out_features` の場合。
    pub fn forward(&self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), self.in_features);
        assert_eq!(output.len(), self.out_features);

        if self.layers.len() == 1 {
            self.layers[0].forward(input, output);
            return;
        }

        // 多層: 中間バッファを使って順伝播
        let mut buf = vec![0.0f32; self.layers[0].out_features];
        self.layers[0].forward(input, &mut buf);

        for layer in &self.layers[1..self.layers.len() - 1] {
            let mut tmp = vec![0.0f32; layer.out_features];
            layer.forward(&buf, &mut tmp);
            buf = tmp;
        }

        self.layers.last().unwrap().forward(&buf, output);
    }

    /// K ステップ自己回帰推論 (DPS)。
    ///
    /// `token_logits[step * out_features..(step+1) * out_features]` に
    /// 各ステップの logits を書き込む。
    ///
    /// # Panics
    /// `token_logits` が不足する場合。
    pub fn predict_tokens(
        &self,
        input: &[f32],
        token_logits: &mut [f32],
        steps: usize,
    ) -> usize {
        assert!(
            token_logits.len() >= self.out_features * steps,
            "token_logits too small: {} < {}",
            token_logits.len(),
            self.out_features * steps
        );

        let mut prev = vec![0.0f32; self.out_features];

        for step in 0..steps {
            let step_input: &[f32] = if step == 0 { input } else { &prev };
            let out_start = step * self.out_features;
            let out_end = out_start + self.out_features;

            let mut step_out = vec![0.0f32; self.out_features];
            self.forward(step_input, &mut step_out);

            token_logits[out_start..out_end].copy_from_slice(&step_out);
            prev.copy_from_slice(&step_out);
        }

        steps
    }

    /// レイヤー列への参照。
    #[must_use]
    pub fn layers(&self) -> &[BitLinear] {
        &self.layers
    }
}

// ============================================================================
// MicroModelBuilder
// ============================================================================

/// `MicroModel` のビルダー。
///
/// キャッシュ予算内でレイヤーを積み上げる。予算超過は `build_*()` で検出。
pub struct MicroModelBuilder {
    /// 入力特徴数。
    in_features: usize,
    /// 出力特徴数。
    out_features: usize,
    /// 中間層の特徴数リスト。
    hidden_dims: Vec<usize>,
    /// キャッシュ予算。
    budget: CacheBudget,
    /// `BitLinear` の `pre_norm` フラグ。
    pre_norm: bool,
}

impl MicroModelBuilder {
    /// 新しいビルダー。
    #[must_use]
    pub const fn new(in_features: usize, out_features: usize, budget: CacheBudget) -> Self {
        Self {
            in_features,
            out_features,
            hidden_dims: Vec::new(),
            budget,
            pre_norm: false,
        }
    }

    /// 中間層を追加。
    #[must_use]
    pub fn add_hidden(mut self, dim: usize) -> Self {
        self.hidden_dims.push(dim);
        self
    }

    /// `pre_norm` を設定。
    #[must_use]
    pub const fn with_pre_norm(mut self, pre_norm: bool) -> Self {
        self.pre_norm = pre_norm;
        self
    }

    /// ビルド前のメモリ見積もり (bytes)。
    ///
    /// `TernaryWeightKernel` の bit-parallel 形式でのサイズ。
    #[must_use]
    pub fn estimate_memory(&self) -> usize {
        let dims = self.layer_dims();
        let mut total = 0;
        for pair in dims.windows(2) {
            let (inp, out) = (pair[0], pair[1]);
            // bit-parallel: 2 bitplanes × words_per_row × out_features × 4 bytes/word
            let words_per_row = inp.div_ceil(32);
            total += 2 * words_per_row * out * 4;
        }
        total
    }

    /// 予算内に収まるかの事前チェック。
    #[must_use]
    pub fn will_fit(&self) -> bool {
        self.estimate_memory() <= self.budget.max_bytes
    }

    /// 決定的な擬似乱数で重みを生成してビルド。
    ///
    /// テスト・ベンチマーク用。本番ではトレーニング済み重みを使う。
    ///
    /// # Panics
    /// レイヤーが予算超過の場合はパニックしない（`fits_in_budget()` で確認）。
    #[must_use]
    pub fn build_random(self, seed: u64) -> MicroModel {
        let dims = self.layer_dims();
        let mut rng = Lcg64(seed);
        let mut layers = Vec::with_capacity(dims.len() - 1);

        for pair in dims.windows(2) {
            let (inp, out) = (pair[0], pair[1]);
            let total = inp * out;
            let values: Vec<i8> = (0..total)
                .map(|_| {
                    let r = rng.next() % 3;
                    r as i8 - 1 // {-1, 0, 1}
                })
                .collect();

            let kernel = TernaryWeightKernel::from_ternary(&values, out, inp);
            layers.push(BitLinear::new(kernel, None, self.pre_norm));
        }

        MicroModel::new(layers, self.budget)
    }

    /// 既存の `BitLinear` レイヤー列からビルド。
    ///
    /// # Panics
    /// `layers` が空の場合。
    #[must_use]
    pub fn build_from_layers(self, layers: Vec<BitLinear>) -> MicroModel {
        MicroModel::new(layers, self.budget)
    }

    /// レイヤーの次元列を返す。`[in, hidden0, hidden1, ..., out]`
    fn layer_dims(&self) -> Vec<usize> {
        let mut dims = Vec::with_capacity(self.hidden_dims.len() + 2);
        dims.push(self.in_features);
        dims.extend_from_slice(&self.hidden_dims);
        dims.push(self.out_features);
        dims
    }
}

// ============================================================================
// Simple LCG RNG (deterministic, no_std)
// ============================================================================

/// 軽量 LCG 乱数生成器。外部依存ゼロ。
struct Lcg64(u64);

impl Lcg64 {
    #[inline]
    const fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        self.0 >> 33
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- CacheBudget ----

    #[test]
    fn test_budget_l2_rpi5() {
        let b = CacheBudget::l2_rpi5();
        assert_eq!(b.max_bytes, 512 * 1024);
        assert_eq!(b.name, "L2-RasPi5");
    }

    #[test]
    fn test_budget_dual() {
        let b = CacheBudget::l2_rpi5_dual();
        assert_eq!(b.max_bytes, 1024 * 1024);
    }

    #[test]
    fn test_budget_256k() {
        let b = CacheBudget::l2_256k();
        assert_eq!(b.max_bytes, 256 * 1024);
    }

    #[test]
    fn test_budget_custom() {
        let b = CacheBudget::custom(1024, "test");
        assert_eq!(b.max_bytes, 1024);
        assert_eq!(b.name, "test");
    }

    #[test]
    fn test_budget_default() {
        let b = CacheBudget::default();
        assert_eq!(b.max_bytes, 512 * 1024);
    }

    // ---- MicroModelBuilder ----

    #[test]
    fn test_builder_estimate_memory() {
        let builder = MicroModelBuilder::new(64, 64, CacheBudget::l2_rpi5()).add_hidden(64);

        let est = builder.estimate_memory();
        // 2 layers: 64→64 each
        // words_per_row = ceil(64/32) = 2
        // per layer: 2 bitplanes × 2 words × 64 rows × 4 bytes = 1024
        // total: 2 × 1024 = 2048
        assert_eq!(est, 2048, "estimate={}", est);
    }

    #[test]
    fn test_builder_will_fit_true() {
        let builder = MicroModelBuilder::new(64, 64, CacheBudget::l2_rpi5()).add_hidden(64);

        assert!(builder.will_fit(), "64→64→64 should fit in 512KB");
    }

    #[test]
    fn test_builder_will_fit_false() {
        // 巨大レイヤー: 4096→4096 → 2 × ceil(4096/32) × 4096 × 4 = 4MB
        let builder =
            MicroModelBuilder::new(4096, 4096, CacheBudget::custom(1024, "tiny"));

        assert!(!builder.will_fit(), "4096→4096 should NOT fit in 1KB");
    }

    #[test]
    fn test_builder_build_random() {
        let model = MicroModelBuilder::new(32, 32, CacheBudget::l2_rpi5())
            .add_hidden(32)
            .build_random(42);

        assert_eq!(model.depth(), 2);
        assert_eq!(model.in_features(), 32);
        assert_eq!(model.out_features(), 32);
        assert!(model.fits_in_budget());
    }

    #[test]
    fn test_builder_with_pre_norm() {
        let model = MicroModelBuilder::new(16, 16, CacheBudget::l2_rpi5())
            .with_pre_norm(true)
            .build_random(1);

        assert_eq!(model.depth(), 1);
    }

    // ---- MicroModel ----

    #[test]
    fn test_model_forward_single_layer() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let layer = BitLinear::new(kernel, None, false);
        let model = MicroModel::new(vec![layer], CacheBudget::l2_rpi5());

        let input = [2.0f32, 3.0];
        let mut output = [0.0f32; 2];
        model.forward(&input, &mut output);

        // [1,-1; 0,1] · [2,3] = [-1, 3]
        assert!((output[0] - (-1.0)).abs() < 1e-5, "out[0]={}", output[0]);
        assert!((output[1] - 3.0).abs() < 1e-5, "out[1]={}", output[1]);
    }

    #[test]
    fn test_model_forward_multi_layer() {
        let k0 = TernaryWeightKernel::from_ternary(&[1, 1, -1, 1], 2, 2);
        let k1 = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let l0 = BitLinear::new(k0, None, false);
        let l1 = BitLinear::new(k1, None, false);
        let model = MicroModel::new(vec![l0, l1], CacheBudget::l2_rpi5());

        let input = [1.0f32, 2.0];
        let mut output = [0.0f32; 2];
        model.forward(&input, &mut output);

        // layer0: [1,1; -1,1] · [1,2] = [3, 1]
        // layer1: [1,-1; 0,1] · [3,1] = [2, 1]
        assert!((output[0] - 2.0).abs() < 1e-5, "out[0]={}", output[0]);
        assert!((output[1] - 1.0).abs() < 1e-5, "out[1]={}", output[1]);
    }

    #[test]
    fn test_model_predict_tokens() {
        let kernel = TernaryWeightKernel::from_ternary(&[1, -1, 0, 1], 2, 2);
        let layer = BitLinear::new(kernel, None, false);
        let model = MicroModel::new(vec![layer], CacheBudget::l2_rpi5());

        let input = [2.0f32, 3.0];
        let mut logits = vec![0.0f32; 2 * 3];

        let steps = model.predict_tokens(&input, &mut logits, 3);
        assert_eq!(steps, 3);

        // step0: [-1, 3]
        assert!((logits[0] - (-1.0)).abs() < 1e-5);
        assert!((logits[1] - 3.0).abs() < 1e-5);
        // step1: [1,-1;0,1]·[-1,3] = [-4, 3]
        assert!((logits[2] - (-4.0)).abs() < 1e-5);
        assert!((logits[3] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_model_fits_in_budget() {
        // 小さいモデル → 収まる
        let model = MicroModelBuilder::new(16, 16, CacheBudget::l2_rpi5())
            .add_hidden(16)
            .build_random(1);
        assert!(model.fits_in_budget());

        // 巨大モデル → 収まらない
        let big = MicroModelBuilder::new(2048, 2048, CacheBudget::custom(256, "tiny"))
            .build_random(1);
        assert!(!big.fits_in_budget());
    }

    #[test]
    fn test_model_budget_utilization() {
        let model = MicroModelBuilder::new(64, 64, CacheBudget::l2_rpi5())
            .add_hidden(64)
            .build_random(42);

        let util = model.budget_utilization();
        assert!(util > 0.0, "utilization should be positive");
        assert!(util < 1.0, "small model should be well under budget");
    }

    #[test]
    fn test_model_budget_remaining() {
        let model = MicroModelBuilder::new(16, 16, CacheBudget::custom(4096, "test"))
            .build_random(1);

        let remaining = model.budget_remaining();
        let used = model.memory_bytes();
        assert_eq!(remaining + used, 4096);
    }

    #[test]
    fn test_model_param_count() {
        // 2 layers: 32→64, 64→16 → 32*64 + 64*16 = 2048 + 1024 = 3072
        let model = MicroModelBuilder::new(32, 16, CacheBudget::l2_rpi5())
            .add_hidden(64)
            .build_random(1);

        assert_eq!(model.param_count(), 3072);
    }

    #[test]
    fn test_model_depth() {
        let model = MicroModelBuilder::new(16, 16, CacheBudget::l2_rpi5())
            .add_hidden(32)
            .add_hidden(64)
            .build_random(1);

        assert_eq!(model.depth(), 3); // 16→32, 32→64, 64→16
    }

    #[test]
    fn test_model_layers_ref() {
        let model = MicroModelBuilder::new(16, 16, CacheBudget::l2_rpi5())
            .add_hidden(16)
            .build_random(1);

        assert_eq!(model.layers().len(), 2);
    }

    // ---- L2 予算に実際に収まるか: 現実的なサイズ ----

    #[test]
    fn test_realistic_l2_model_4m_params() {
        // ~4M params: 6 layers of 512→512
        // Per layer: 2 × ceil(512/32) × 512 × 4 = 2 × 16 × 512 × 4 = 65536 bytes
        // Total: 6 × 65536 = 393216 bytes = 384 KB → L2 (512KB) に収まる
        let mut builder = MicroModelBuilder::new(512, 512, CacheBudget::l2_rpi5());
        for _ in 0..5 {
            builder = builder.add_hidden(512);
        }

        assert!(builder.will_fit(), "4M-param model should fit in L2 512KB");

        let model = builder.build_random(42);
        assert!(model.fits_in_budget());

        let params = model.param_count();
        // 6 layers × 512 × 512 = 1,572,864 ≈ 1.5M
        assert!(params > 1_000_000, "should have >1M params, got {}", params);
        assert!(
            model.memory_bytes() <= 512 * 1024,
            "memory {} > 512KB",
            model.memory_bytes()
        );
    }

    #[test]
    fn test_realistic_l2_model_inference() {
        // 512→256→512 model, realistic inference test
        let model = MicroModelBuilder::new(512, 512, CacheBudget::l2_rpi5())
            .add_hidden(256)
            .build_random(123);

        assert!(model.fits_in_budget());

        let input: Vec<f32> = (0..512).map(|i| (i as f32) * 0.001).collect();
        let mut output = vec![0.0f32; 512];

        model.forward(&input, &mut output);

        // 出力がゼロでないことを確認（推論が動いている）
        let sum: f32 = output.iter().map(|x| x.abs()).sum();
        assert!(sum > 0.0, "output should be non-zero");
    }

    // ---- LCG determinism ----

    #[test]
    fn test_deterministic_build() {
        let m1 = MicroModelBuilder::new(16, 16, CacheBudget::l2_rpi5()).build_random(42);
        let m2 = MicroModelBuilder::new(16, 16, CacheBudget::l2_rpi5()).build_random(42);

        let input = [1.0f32; 16];
        let mut out1 = vec![0.0f32; 16];
        let mut out2 = vec![0.0f32; 16];
        m1.forward(&input, &mut out1);
        m2.forward(&input, &mut out2);

        for i in 0..16 {
            assert!(
                (out1[i] - out2[i]).abs() < 1e-6,
                "determinism failed at [{}]: {} vs {}",
                i,
                out1[i],
                out2[i]
            );
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let m1 = MicroModelBuilder::new(16, 16, CacheBudget::l2_rpi5()).build_random(1);
        let m2 = MicroModelBuilder::new(16, 16, CacheBudget::l2_rpi5()).build_random(2);

        let input = [1.0f32; 16];
        let mut out1 = vec![0.0f32; 16];
        let mut out2 = vec![0.0f32; 16];
        m1.forward(&input, &mut out1);
        m2.forward(&input, &mut out2);

        let diff: f32 = out1
            .iter()
            .zip(out2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0, "different seeds should produce different output");
    }
}
