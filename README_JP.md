# ALICE-ML

[![crates.io](https://img.shields.io/crates/v/alice-ml.svg)](https://crates.io/crates/alice-ml)
[![docs.rs](https://img.shields.io/docsrs/alice-ml)](https://docs.rs/alice-ml)
[![MSRV](https://img.shields.io/crates/msrv/alice-ml)](#msrv)
[![CI](https://github.com/ext-sakamoro/ALICE-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-ML/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey)](LICENSE-Proprietary)

[English](README.md) | 日本語

1.58-bit の ternary (`BitNet` 方式) 推論 engine 重みを `{-1, 0, +1}` に量子化するので、
推論の核にある行列・ベクトル kernel は加算と減算だけで構成される — 浮動小数点の乗算も
実行時の allocation も無い (全 buffer は arena から、Destination Passing Style で書く)
`no_std` 対応、bare-metal / embedded target でも動く

## 向いている用途

ternary 量子化済みモデルの forward pass を CPU で走らせること — edge device、embedded
target、full-precision モデルの乗算主体の matmul が合わない場所 周辺 (QAT 学習支援、
確率 calibration、speculative decoding、weight streaming、C-ABI / Unity / UE5 / Python
binding) も本 crate に含む

## 向いていない用途

full-precision でのモデルの一から学習、ternary 推論と無関係な汎用テンソル演算 —
[`Tensor`](src/tensor.rs) は本 crate の kernel を支えるために在るのであって、単体の
ndarray 代替ではない GGUF 形式のモデル読込や full LLM runtime が要るなら
[ALICE-LLM](https://github.com/ext-sakamoro/ALICE-LLM) を使うこと 本 crate 自身のモデル
形式は `.atml` (`src/model_io.rs`、`src/safetensors.rs`)

## Install

```bash
cargo add alice-ml
```

optional な feature ([Features](#features) 参照) で FFI binding / SIMD kernel /
Rayon 並列 batch / ALICE-DB persistence / Python binding を追加する

## Example

```rust
use alice_ml::{TernaryWeight, ternary_matvec};

// Load weights (once, at model load)
let weights = TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);

// Stack-allocated buffers (or from Arena)
let input = [2.0f32, 3.0];
let mut output = [0.0f32; 2];

// Inference: ZERO ALLOCATION, NO MULTIPLICATION
ternary_matvec(&input, &weights, &mut output);

assert!((output[0] - (-1.0)).abs() < 1e-6);  // 2*1 + 3*(-1) = -1
assert!((output[1] - 3.0).abs() < 1e-6);     // 2*0 + 3*1 = 3
```

## 決定論的なもの・そうでないもの

ternary matvec kernel (scalar / SIMD / INT8 入力版) は超越関数を一度も呼ばない —
ternary 重みは `+input` / `-input` / `0` のどれかを選ぶだけなので、kernel は加算と減算の
和であり、全 kernel variant が同じ閉形式の値と突合されている
<!-- claim-test: every_matvec_kernel_equals_the_closed_form_for_all_lengths -->
(`tests/analytic_oracle.rs`) この部分は platform を跨ぐ再現性のために特別な算術
library を要らない

matvec kernel の外側では超越関数を呼んでいる — softmax、RoPE、Adam の bias
correction、calibration の温度 fitting、quantization の entropy と SNR 0.3.0 から、
production code (test を除く) のこれら全てを `f32`/`f64` 自身の
`exp`/`ln`/`powf`/`powi`/`sin_cos`/`log2`/`log10` ではなく
[`alice_det_math`](https://crates.io/crates/alice-det-math) 経由にした (platform の
libm は IEEE 754 が platform を跨ぐ bit 一致を要求していない、`clippy.toml` の
`clippy::disallowed_methods` lint がこれらの method への新規の直呼びを黙って通さない)

`sqrt` / `round` はこの対象外: どちらも IEEE 754 が正しい丸めを要求する演算なので、
library を足さずとも platform を跨いで既に bit 一致する `hypot` は `exp`/`ln` と同じ
platform libm 問題の実例 (Rust 自身の doc が精度を「unspecified」「platform 依存」と
明記) なので、同じ方法 (禁止 lint) で塞いだ: `f64` の `hypot` の唯一の呼び出し箇所は
`alice_det_math::sqrt64(a*a + b*b)` に書き換え済 (`alice_det_math` 0.5 に 64-bit の
`hypot` がまだ無いため)

`mul_add` は drift の原因ではない — Rust は IEEE 754 の `fusedMultiplyAdd` として
規定しており、target CPU に FMA 命令が有るか無いかに関わらず (無い target は正しく
丸められた software 実装に fallback する) 1 回丸め・platform を跨いで bit 一致する
それでも production code (test を除く) の `mul_add` 呼び出しは全て `a * b + c` に
書き換え済 (ffi.rs / layer.rs / llama3_ternary.rs / moe.rs / quantize.rs /
tensor.rs / training.rs)、理由はもっと単純で: `a * b + c` と `a.mul_add(b, c)` は
互いに丸め方が違う (2 回丸め vs 1 回丸め) ので、どちらか一方の書き方に揃えておけば、
どの結果も source だけから再現できる この crate はその 1 つに統一し、もう片方を
禁止する (`alice_det_math` 自身の consumer policy と同じ選択)
(layer/RMS norm が呼んでいたのも `sqrt` だけになり、`mul_add` が消えたことで platform
依存の演算を一切呼ばなくなった)

## 含まれるもの

| 領域 | Module |
|---|---|
| Core kernel | `ops`、`tensor`、`layer`、`arena`、`quantize`、`neon` |
| 学習・calibration | `training`、`calibration`、`error_analysis`、`dataset` |
| 推論 | `llama3_ternary`、`micro_model`、`moe`、`speculative`、`streaming` |
| I/O・persistence | `model_io`、`safetensors`、`db_bridge` |
| Binding | `ffi`、`python` |

各 module の 1 行説明と必要 feature は [`docs/MODULES.md`](docs/MODULES.md) を参照

## Features

<!-- readme-sync: features -->
| Feature | 有効化するもの |
|---|---|
| `std` (既定) | `f32`/`f64` の inherent method `no_std` (bare-metal/embedded) ではこの feature を外す |
| `ffi` | C-ABI 全体 (`src/ffi.rs`)、下記 Unity/UE5 binding が使う |
| `simd` | 手で調整した AVX2 kernel (`x86_64`) |
| `neon` | 手で調整した NEON kernel (`aarch64`) |
| `parallel` | Rayon 並列の batched matmul |
| `pyo3` | Python binding (`src/python.rs`) |
| `db` | [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) による学習 metric の persistence |
| `safetensors` | Safetensors parse + Llama-3 ternary 推論 (`llama3_ternary`) |

## Binding

- **C-ABI** (`--features ffi`): 公開 function は全て `catch_unwind` で guard 済み、
  panic は FFI 境界を越えて unwind せずエラーコードを返す (`am_ml_last_error` が
  message を保持) `src/ffi.rs` 参照
- **Unity** (C#): [`bindings/unity/AliceMl.cs`](bindings/unity/AliceMl.cs)、
  C-ABI への薄い P/Invoke wrapper
- **Unreal Engine 5** (C++): [`bindings/ue5/AliceMl.h`](bindings/ue5/AliceMl.h)
- **Python** (`--features pyo3`): `src/python.rs`、PyO3 + numpy で構築

## Examples

| Example | 内容 |
|---|---|
| [`elyza_ternary`](examples/elyza_ternary.rs) | ternary 量子化済み ELYZA-JP-8B モデルの読込と推論 |
| [`qwen_qat_test`](examples/qwen_qat_test.rs) | QAT 学習済み ternary 重みを `.atml` から読込み推論 |

実行は `cargo run --example <name> --features safetensors`

## MSRV

対応する最小 Rust version: **1.87** <!-- readme-sync: msrv --> CI の `msrv` job が
この版の toolchain で build する (default と native feature 全部、全 target、`no_std`
rlib) job の toolchain が `rust-version` と違えば `scripts/check_msrv.py` が落とす
`rust-toolchain.toml` は開発用の pin であり MSRV ではない

## Build と test

```bash
cargo test --features ffi,simd,parallel,safetensors,db
cargo bench --features simd  # criterion benchmark、benches/matmul.rs 参照
```

`scripts/preflight.sh` (push 前は `--quick`) が CI と同じ検査を走らせる
(bare-metal target 向け `no_std` build を含む)

## 関連 crate

- [ALICE-LLM](https://github.com/ext-sakamoro/ALICE-LLM) — GGUF 形式のモデル読込と
  full LLM runtime (本 crate の `.atml` とは別のモデル形式)
- [alice-det-math](https://crates.io/crates/alice-det-math) — ternary kernel の外側で
  本 crate が経由する、platform を跨ぐ決定論的な超越関数
- [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) — 学習 metric の persistence
  (feature `db`)

<!-- law-integration:begin law=0a5de013ca1268eac22fe44908138d09a431fce0 -->
## ALICE-Law との統合

ALICE-Law の `scripts/gen-law-readme-block.py` が ALICE-Law `0a5de013ca12` とこの repository の `alice-law-conformance.toml` から生成する 手で編集しない

| | |
|---|---|
| release train | alice-ml 0.2.0 (published)、Law への依存: なし |
| 算術 | train の alice-det-math 0.5.1、算術の識別子 `85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315` この crate の要求は `alice-det-math = "0.5.1"` |
| end to end 検査 | end to end 検査の対象外 |

### 適合 (G1-G14)

| gate | 状態 | 注記 |
|---|---|---|
| G1 no-libm-direct | 適合 | layers that read: clippy.toml bans all 25 of G1's banned methods for f32 and f64 (plus powi and mul_add for both widths, and the reasons are in the file; … |
| G2 ieee-max-min | 適合 | every float maximum, minimum, clamp, argmax and selection order of the library goes through src/law_math.rs. Residual: the AVX2 max / min intrinsics are not … |
| G3 no-outer-fneg-of-fma | 適合 | no fused multiply-add call in the library sources (clippy.toml also bans mul_add there): the syntax-tree check reads 37 files and counts 2 fma-family calls, … |
| G4 debug-release-arch-goldens | 適合 | the AVX2 lanes of law_math are not covered by this golden on a local x86_64 (Rosetta 2 has no AVX2); on CI they are taken when the runner reports avx2 and are … |
| G5 release-profile-test-leg | 適合 | the x86_64 leg is not claimed to cover the AVX2 lanes (taken only when the runner reports avx2, printed in the log); the hardware fma claim rests on the … |
| G6 semantics-id | 適合 | hex with alice-det-math 0.5.1: a547237044e8fdba9ab152f7af13c431f675e5382e4e8cb40aa8c83c46bb606b. The identifier attests the pinned kernels (relu, max_min, … |
| G7 handshake-carries-id | 適合 | the list of type-erasing boundaries that receive another Law-bearing crate's value is empty: the C ABI and the Python exports take the caller's numbers, the … |
| G8 persisted-semantics-id | 適合 | the identifier recorded is ML_SEMANTICS_ID (G6): it pins the quantiser whose scale and packed values the file stores, and the det-math generation. db_bridge … |
| G9 persisted-meaning-change-is-major | 適合 | definition: ML_SEMANTICS_ID ([[id]] in scripts/id_move.toml with PREVIOUS_ML_SEMANTICS_IDS) may move only with a breaking version, or keeping the released … |
| G10 no-run-varying-artifacts | 適合 | the ban covers what src/ can reach: the clock and RandomState through std, and nothing else, because no direct dependency offers an RNG or a UUID (rand is a … |
| G11 gate-self-tests | 未達 | unmet: clause (c). scripts/clippy_canary.py, scripts/test_clippy_canary.py, scripts/gen_clippy_canary_calls.py and scripts/test_gen_clippy_canary_calls.py are … |
| G12 public-repo-text-rules | 適合 | the workflow fetches ALICE-Law's scripts/public_text_check.py at the revision in .github/public-text-pin (779fa443f95bc0caca5ca2d5bdb9156051fd6143) and refuses … |
| G13 no-unspecified-order | 適合 | the batch computes each output row in one task and combines no float across tasks, so its bits do not depend on the thread count; the test is what keeps it so. … |
| G14 float-environment-invariants | 適合 | the static scan covers the files of this repository (.cargo/mutants.toml included) and the dynamic check covers the builds CI runs on ubuntu and macOS (the … |

### feature と保証の範囲

| feature | 保証 | 注記 |
|---|---|---|
| `db` | 範囲外 | ALICE-DB metrics sink; the rows are alice-db's own format with its own identifier, read through alice-db's Rust API (G7, in memory); this crate adds no … |
| `ffi` | 範囲外 | C ABI entry points (67 extern "C" fn); the max / min / relu / softmax kernels are pinned by the ffi golden and tests/law_ties.rs, the rest by the G1 lint only; … |
| `neon` | 範囲外 | NEON kernel on aarch64; no golden compares its bits with the scalar path across architectures (G4) |
| `parallel` | 範囲外 | rayon over output rows (ops::ternary_matmul_batch); the same bits at 1, 2 and 8 threads (tests/parallel_order.rs, G13); no G4 golden |
| `pyo3` | 範囲外 | linted by the clippy job's `Clippy (pyo3, lib)` step (G1); no test builds it (a Python interpreter is needed), so its maximum / minimum sites are covered only … |
| `safetensors` | 範囲外 | safetensors parser and Llama-3 ternary path; no Law gate beyond the G1 lint, no golden (G4) |
| `simd` | 範囲外 | AVX2 kernel on x86_64; the max / min / relu lanes are tie-tested and mutant-checked on CI's x86_64 runner (law-tie-mutants.yml), but no G4 golden spans the … |
| `std` | 範囲外 | default feature; G6 and G8 are met (0.4.0) but G4 and G5 are not (no signed-zero or cancellation golden on every architecture, no release-profile test leg), so … |
<!-- law-integration:end -->

## ライセンス

`LicenseRef-Proprietary` 条文は [LICENSE-Proprietary](LICENSE-Proprietary)

Enquiries: https://extoria.co.jp/
