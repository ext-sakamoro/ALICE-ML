# ALICE-ML

**1.58bit 三値推論エンジン**

> "掛け算は高い。足し算だけでいい。"

[English](README.md) | 日本語

BitNet b1.58 研究に基づく革新的な推論エンジン。浮動小数点の乗算を完全に排除し、全ての行列演算を三値重み {-1, 0, +1} による加算と減算のみで実行します。

## 革命

従来のニューラルネットワーク推論には高コストな浮動小数点乗算が必要：

```
y = W · x  →  N×M 回の FP32 乗算
```

ALICE-ML は全ての乗算を排除：

```
y = Σ(x[i] where W=+1) - Σ(x[i] where W=-1)  →  加算のみ！
```

## 特徴

- **1.58bit 重み**: 三値量子化 {-1, 0, +1}、2bit パッキング
- **乗算ゼロ**: 行列演算は加算/減算のみ
- **16倍圧縮**: 4 bytes → 0.25 bytes/重み
- **AVX2 SIMD**: 8-wide `_mm256` カーネル（MatVec、add/sub/scale/ReLU）
- **Rayon 並列化**: バッチ MatMul を `--features parallel` で自動並列化
- **ブランチレス ReLU**: `_mm256_max_ps` / `f32::max(0.0)` — ホットパスでの分岐ゼロ
- **ゼロコピーロード**: mmap でモデルファイルを直接マッピング
- **依存なし**: Pure Rust、外部クレートゼロ（rayon はオプション）
- **no_std 対応**: ベアメタル / WASM で動作

## クイックスタート

```rust
use alice_ml::{Tensor, TernaryWeight, ternary_matmul, quantize_to_ternary};

// FP32重みを三値に量子化
let fp32_weights = vec![0.8, -0.9, 0.1, 0.7, -0.6, 0.2, -0.1, 0.9];
let (weights, stats) = quantize_to_ternary(&fp32_weights, 2, 4);
println!("圧縮率: {}x", weights.compression_ratio());  // 16x

// 入力を作成
let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);

// 推論：乗算なし！
let output = ternary_matmul(&input, &weights);
```

## 仕組み

### 重みパッキング

1バイトに4つの三値を格納：

```
Byte: [w3:2bit][w2:2bit][w1:2bit][w0:2bit]

エンコード:
  0b00 = 0   (ゼロ、スキップ)
  0b01 = +1  (入力を加算)
  0b10 = -1  (入力を減算)
```

### 三値 MatMul カーネル

```rust
// 各出力ニューロンについて:
for weight in row {
    match weight {
        +1 => acc_plus += input[j],   // 加算だけ！
        -1 => acc_minus += input[j],  // 加算だけ！
        0  => { }                      // スキップ（スパース性ボーナス）
    }
}
output[i] = (acc_plus - acc_minus) * scale;
```

## 性能

| 演算 | FP32 | ALICE-ML | 改善 |
|------|------|----------|------|
| MatMul (1024×1024) | 200万回乗算 | 0回乗算 | ∞ |
| メモリ | 4MB | 256KB | 16x |
| 消費電力* | 100W | ~12W | 8x |

*FPU 使用排除に基づく推定

## 量子化

BitNet b1.58 スタイルの量子化（学習済みスケーリング付き）：

```rust
use alice_ml::quantize_to_ternary;

let (weights, stats) = quantize_to_ternary(&fp32_weights, rows, cols);

println!("スケールファクター: {}", stats.scale);
println!("スパース率: {:.1}%", stats.sparsity() * 100.0);
println!("有効ビット数: {:.2}", stats.effective_bits());  // ~1.58
println!("MAE: {}", stats.mae);
```

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                      ALICE-ML                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │   Tensor    │───▶│   三値      │───▶│   出力     │  │
│  │  (f32/i8)   │    │   MatMul    │    │   (f32)    │  │
│  └─────────────┘    └─────────────┘    └────────────┘  │
│         │                  │                  │        │
│         ▼                  ▼                  ▼        │
│  ┌─────────────────────────────────────────────────┐   │
│  │           パックド重み (2-bit)                    │   │
│  │  00 = 0, 01 = +1, 10 = -1, 11 = 予約             │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                             │
│                          ▼                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │               SIMD カーネル                       │   │
│  │  • +1 マスク抽出 → 水平加算                       │   │
│  │  • -1 マスク抽出 → 水平減算                       │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## モジュール

- `tensor.rs` - N次元配列（Arenaアロケータ付き）
- `ops.rs` - 三値 MatMul カーネル（心臓部）
- `quantize.rs` - FP32 → 三値変換
- `arena.rs` - ゼロアロケーションメモリ管理

## ユースケース

1. **エッジ推論**: FPUが限られたデバイスでLLMを実行
2. **省エネ**: 電力を大量消費する乗算器を排除
3. **組み込みAI**: マイコン向け no_std 対応
4. **モデル圧縮**: 16倍小さいモデルファイル
5. **決定論的ゲームAI**: ネットワークゲーム向けビット精度推論（ALICE-Physics連携）

## ALICE-Physics 連携

ALICE-ML の三値重みは物理シミュレーションにおける**決定論的ニューラル推論**の鍵です。三値 {-1, 0, +1} 重みと [ALICE-Physics](../ALICE-Physics) の128bit固定小数点演算を組み合わせることで、推論パイプライン全体が純粋な加算/減算に集約され、全プラットフォームでビット精度の結果を保証します。

### なぜ 三値 + 固定小数点 = 決定論なのか

```
従来のNN:      y = W · x  → FP32乗算 → プラットフォーム依存の丸め
ALICE-ML+Fix128: y = Σ(±x) → Fix128加減算 → どこでもビット精度一致
```

三値重みにより浮動小数点乗算は一切発生しません。唯一の乗算はスケールファクター1回のみ（モデルロード時にFix128として事前計算）。これにより以下の用途に最適な推論エンジンとなります：

- **格闘ゲーム**: AI対戦相手付きフレームパーフェクトなロールバックネットコード
- **アクションゲーム**: 全クライアントで同一に動作するラグドールコントローラ
- **対戦型マルチプレイ**: AI制御エンティティからのデシンクゼロ

### 使用例

```rust
use alice_ml::{TernaryWeight, quantize_to_ternary};
use alice_physics::neural::*;
use alice_physics::Fix128;

// 学習済みモデルを三値に量子化
let (weights, stats) = quantize_to_ternary(&fp32_weights, out_features, in_features);

// 固定小数点に変換（ここから決定論的）
let fixed_weights = FixedTernaryWeight::from_ternary_weight(weights);

// ネットワークを構築
let mut network = DeterministicNetwork::new(
    vec![fixed_weights],
    vec![Activation::ReLU],
);

// 推論 — 全プラットフォームでビット精度一致
let input = vec![Fix128::from_int(1); in_features];
let output = network.forward(&input);
```

### 依存関係

```toml
[dependencies]
alice-physics = { path = "../ALICE-Physics", features = ["neural"] }
alice-ml = { path = "../ALICE-ML" }
```

## Cargo Features

| Feature | デフォルト | 説明 |
|---------|----------|------|
| `std` | Yes | 標準ライブラリサポート |
| `simd` | No | AVX2 8-wide SIMDカーネル（テンソル演算 + 三値MatVec） |
| `parallel` | No | Rayonによる並列バッチMatMul |

```bash
cargo build --release --features simd       # SIMDカーネル有効化
cargo build --release --features parallel   # Rayon並列化有効化
cargo build --release --features "simd,parallel"  # 両方
```

## 最適化

### SIMD カーネル (`--features simd`)

全テンソル演算がサイクルあたり8つの浮動小数点をベクトル化処理：

| 演算 | スカラー | SIMD (AVX2) |
|------|---------|-------------|
| `tensor_add` | 1 add/cycle | `_mm256_add_ps` — 8 adds/cycle |
| `tensor_sub` | 1 sub/cycle | `_mm256_sub_ps` — 8 subs/cycle |
| `tensor_scale` | 1 mul/cycle | `_mm256_mul_ps` — 8 muls/cycle |
| `tensor_relu` | 分岐 + cmp | `_mm256_max_ps` — 8-wide ブランチレス |
| `ternary_matvec` | パック2bitデコード | `_mm256_blendv_ps` マスクセレクト |

### 並列バッチ (`--features parallel`)

バッチMatMulがRayon `par_chunks_mut` により自動的にサンプル間で並列化：

```rust
// 64入力 × 1024特徴量のバッチ → CPUコア間で並列処理
ternary_matmul_batch(&inputs, &weights, &mut outputs, 64);
```

### ブランチレス

ReLUは `f32::max(0.0)` を使用し、`maxss`/`maxps` にコンパイルされます — 分岐予測ミスゼロ。

## ロードマップ

- [x] ALICE-Physics連携による固定小数点推論（決定論的ゲームAI）
- [x] AVX2 SIMDカーネル（テンソル演算 + 三値MatVec）
- [x] Rayon並列バッチMatMul
- [ ] NEON SIMDカーネル（ARM）
- [ ] BitLinearレイヤー（nn.Linearのドロップイン置換）
- [ ] PyTorchモデルからの知識蒸留
- [ ] 動的深度のためのアーリーイグジット
- [ ] `.aml` モデルフォーマット（mmapロード対応）

## 参考文献

- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453)
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) - 1.58bitのブレークスルー
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)

## ライセンス

AGPL-3.0

## 作者

Moroya Sakamoto ([@ext-sakamoro](https://github.com/ext-sakamoro))
