//! Training Loop — Loss 関数、最適化器、学習ステップ。
//!
//! 軽量な学習インフラストラクチャ。外部フレームワーク不要で
//! 小規模モデルの fine-tuning が可能。

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Loss 関数の種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossFunction {
    /// 平均二乗誤差 (回帰用)。
    Mse,
    /// クロスエントロピー (分類用)。
    CrossEntropy,
    /// 平均絶対誤差。
    Mae,
}

/// Loss 計算結果。
#[derive(Debug, Clone, Copy)]
pub struct LossResult {
    /// スカラー loss 値。
    pub value: f32,
}

/// MSE loss と勾配を計算。
///
/// `predictions` と `targets` は同じ長さ。
/// 勾配は `grad_out` に書き込まれる (DPS)。
///
/// # Panics
///
/// `predictions`, `targets`, `grad_out` の長さが異なる場合。
pub fn mse_loss(predictions: &[f32], targets: &[f32], grad_out: &mut [f32]) -> LossResult {
    assert_eq!(predictions.len(), targets.len());
    assert_eq!(predictions.len(), grad_out.len());
    let n = predictions.len();
    if n == 0 {
        return LossResult { value: 0.0 };
    }
    let rcp_n = 1.0 / n as f32;
    let mut sum = 0.0_f32;
    for i in 0..n {
        let diff = predictions[i] - targets[i];
        sum += diff * diff;
        grad_out[i] = 2.0 * diff * rcp_n;
    }
    LossResult { value: sum * rcp_n }
}

/// クロスエントロピー loss と勾配を計算。
///
/// `logits` は softmax 前の生出力。`targets` は one-hot ラベル。
///
/// # Panics
///
/// `logits`, `targets`, `grad_out` の長さが異なる場合。
pub fn cross_entropy_loss(logits: &[f32], targets: &[f32], grad_out: &mut [f32]) -> LossResult {
    assert_eq!(logits.len(), targets.len());
    assert_eq!(logits.len(), grad_out.len());
    let n = logits.len();
    if n == 0 {
        return LossResult { value: 0.0 };
    }

    // Numerically stable softmax
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0_f32;
    for &l in logits {
        exp_sum += (l - max_logit).exp();
    }
    let log_sum = max_logit + exp_sum.ln();

    let mut loss = 0.0_f32;
    for i in 0..n {
        let softmax_i = (logits[i] - log_sum).exp();
        loss -= targets[i] * (softmax_i + 1e-8).ln();
        grad_out[i] = softmax_i - targets[i];
    }

    LossResult { value: loss }
}

/// MAE loss と勾配を計算。
///
/// # Panics
///
/// `predictions`, `targets`, `grad_out` の長さが異なる場合。
pub fn mae_loss(predictions: &[f32], targets: &[f32], grad_out: &mut [f32]) -> LossResult {
    assert_eq!(predictions.len(), targets.len());
    assert_eq!(predictions.len(), grad_out.len());
    let n = predictions.len();
    if n == 0 {
        return LossResult { value: 0.0 };
    }
    let rcp_n = 1.0 / n as f32;
    let mut sum = 0.0_f32;
    for i in 0..n {
        let diff = predictions[i] - targets[i];
        sum += diff.abs();
        grad_out[i] = if diff > 0.0 {
            rcp_n
        } else if diff < 0.0 {
            -rcp_n
        } else {
            0.0
        };
    }
    LossResult { value: sum * rcp_n }
}

/// SGD 最適化器パラメータ。
#[derive(Debug, Clone)]
pub struct SgdConfig {
    /// 学習率。
    pub learning_rate: f32,
    /// モメンタム係数 (0 = モメンタムなし)。
    pub momentum: f32,
}

impl SgdConfig {
    /// デフォルト SGD (lr=0.01, momentum=0)。
    #[must_use]
    pub const fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
        }
    }
}

/// SGD 最適化器の状態。
#[derive(Debug, Clone)]
pub struct SgdState {
    /// 設定。
    pub config: SgdConfig,
    /// モメンタムバッファ。
    velocity: Vec<f32>,
}

impl SgdState {
    /// パラメータ数を指定して初期化。
    #[must_use]
    pub fn new(config: SgdConfig, param_count: usize) -> Self {
        Self {
            config,
            velocity: alloc_vec_zero(param_count),
        }
    }

    /// パラメータを更新。
    ///
    /// # Panics
    ///
    /// `params` と `grads` の長さが `velocity` と異なる場合。
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        assert_eq!(params.len(), self.velocity.len());

        let lr = self.config.learning_rate;
        let mom = self.config.momentum;

        for i in 0..params.len() {
            self.velocity[i] = mom.mul_add(self.velocity[i], grads[i]);
            params[i] -= lr * self.velocity[i];
        }
    }
}

/// Adam 最適化器パラメータ。
#[derive(Debug, Clone)]
pub struct AdamConfig {
    /// 学習率。
    pub learning_rate: f32,
    /// 一次モーメント減衰率。
    pub beta1: f32,
    /// 二次モーメント減衰率。
    pub beta2: f32,
    /// 数値安定性のための epsilon。
    pub epsilon: f32,
}

impl AdamConfig {
    /// デフォルト Adam (lr=0.001, β1=0.9, β2=0.999, ε=1e-8)。
    #[must_use]
    pub const fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Adam 最適化器の状態。
#[derive(Debug, Clone)]
pub struct AdamState {
    /// 設定。
    pub config: AdamConfig,
    /// ステップ数。
    step_count: u32,
    /// 一次モーメント。
    m: Vec<f32>,
    /// 二次モーメント。
    v: Vec<f32>,
}

impl AdamState {
    /// パラメータ数を指定して初期化。
    #[must_use]
    pub fn new(config: AdamConfig, param_count: usize) -> Self {
        Self {
            config,
            step_count: 0,
            m: alloc_vec_zero(param_count),
            v: alloc_vec_zero(param_count),
        }
    }

    /// パラメータを更新。
    ///
    /// # Panics
    ///
    /// `params` と `grads` の長さが異なる場合。
    pub fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len());
        assert_eq!(params.len(), self.m.len());

        self.step_count += 1;
        let t = self.step_count as f32;
        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let lr = self.config.learning_rate;
        let eps = self.config.epsilon;

        let bias_correction1 = 1.0 - b1.powf(t);
        let bias_correction2 = 1.0 - b2.powf(t);

        for i in 0..params.len() {
            self.m[i] = b1.mul_add(self.m[i], (1.0 - b1) * grads[i]);
            self.v[i] = b2.mul_add(self.v[i], (1.0 - b2) * grads[i] * grads[i]);

            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    /// 現在のステップ数。
    #[must_use]
    pub const fn step_count(&self) -> u32 {
        self.step_count
    }
}

/// ゼロ初期化された Vec を作成。
fn alloc_vec_zero(n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(n);
    v.resize(n, 0.0);
    v
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_loss_basic() {
        let pred = [1.0, 2.0, 3.0];
        let target = [1.0, 2.0, 3.0];
        let mut grad = [0.0_f32; 3];
        let loss = mse_loss(&pred, &target, &mut grad);
        assert!(loss.value.abs() < 1e-6);
        assert!(grad.iter().all(|&g| g.abs() < 1e-6));
    }

    #[test]
    fn mse_loss_nonzero() {
        let pred = [2.0, 4.0];
        let target = [1.0, 2.0];
        let mut grad = [0.0_f32; 2];
        let loss = mse_loss(&pred, &target, &mut grad);
        // MSE = ((1)^2 + (2)^2) / 2 = 2.5
        assert!((loss.value - 2.5).abs() < 1e-5);
    }

    #[test]
    fn mse_loss_empty() {
        let loss = mse_loss(&[], &[], &mut []);
        assert!(loss.value.abs() < 1e-6);
    }

    #[test]
    fn cross_entropy_basic() {
        let logits = [2.0, 1.0, 0.1];
        let targets = [1.0, 0.0, 0.0]; // one-hot for class 0
        let mut grad = [0.0_f32; 3];
        let loss = cross_entropy_loss(&logits, &targets, &mut grad);
        assert!(loss.value > 0.0);
        // 勾配はsoftmax - target
        assert!(grad[0] < 0.0); // softmax(2.0) < 1.0
    }

    #[test]
    fn cross_entropy_empty() {
        let loss = cross_entropy_loss(&[], &[], &mut []);
        assert!(loss.value.abs() < 1e-6);
    }

    #[test]
    fn mae_loss_basic() {
        let pred = [1.0, 3.0];
        let target = [2.0, 1.0];
        let mut grad = [0.0_f32; 2];
        let loss = mae_loss(&pred, &target, &mut grad);
        // MAE = (|1-2| + |3-1|) / 2 = 1.5
        assert!((loss.value - 1.5).abs() < 1e-5);
    }

    #[test]
    fn mae_loss_empty() {
        let loss = mae_loss(&[], &[], &mut []);
        assert!(loss.value.abs() < 1e-6);
    }

    #[test]
    fn sgd_step() {
        let config = SgdConfig::new(0.1);
        let mut state = SgdState::new(config, 2);
        let mut params = [1.0, 2.0];
        let grads = [0.5, -0.5];
        state.step(&mut params, &grads);
        // params[0] = 1.0 - 0.1*0.5 = 0.95
        // params[1] = 2.0 - 0.1*(-0.5) = 2.05
        assert!((params[0] - 0.95).abs() < 1e-6);
        assert!((params[1] - 2.05).abs() < 1e-6);
    }

    #[test]
    fn sgd_momentum() {
        let config = SgdConfig {
            learning_rate: 0.1,
            momentum: 0.9,
        };
        let mut state = SgdState::new(config, 1);
        let mut params = [0.0];

        // Step 1: v = 0.9*0 + 1.0 = 1.0, p = 0 - 0.1*1.0 = -0.1
        state.step(&mut params, &[1.0]);
        assert!((params[0] - (-0.1)).abs() < 1e-6);

        // Step 2: v = 0.9*1.0 + 1.0 = 1.9, p = -0.1 - 0.1*1.9 = -0.29
        state.step(&mut params, &[1.0]);
        assert!((params[0] - (-0.29)).abs() < 1e-5);
    }

    #[test]
    fn adam_step() {
        let config = AdamConfig::new(0.001);
        let mut state = AdamState::new(config, 2);
        let mut params = [1.0, 2.0];
        let grads = [0.1, -0.1];
        state.step(&mut params, &grads);
        assert_eq!(state.step_count(), 1);
        // パラメータが更新されたことだけ確認
        assert!((params[0] - 1.0).abs() > 1e-6);
        assert!((params[1] - 2.0).abs() > 1e-6);
    }

    #[test]
    fn adam_multiple_steps() {
        let config = AdamConfig::new(0.01);
        let mut state = AdamState::new(config, 1);
        let mut params = [5.0];
        // 何ステップか回して収束方向に動くことを確認
        for _ in 0..10 {
            state.step(&mut params, &[1.0]);
        }
        assert_eq!(state.step_count(), 10);
        assert!(params[0] < 5.0); // 正の勾配で減少方向
    }

    #[test]
    fn loss_function_eq() {
        assert_eq!(LossFunction::Mse, LossFunction::Mse);
        assert_ne!(LossFunction::Mse, LossFunction::CrossEntropy);
    }

    #[test]
    fn sgd_config_default() {
        let cfg = SgdConfig::new(0.01);
        assert!((cfg.learning_rate - 0.01).abs() < 1e-6);
        assert!((cfg.momentum).abs() < 1e-6);
    }

    #[test]
    fn adam_config_defaults() {
        let cfg = AdamConfig::new(0.001);
        assert!((cfg.beta1 - 0.9).abs() < 1e-6);
        assert!((cfg.beta2 - 0.999).abs() < 1e-6);
    }
}
