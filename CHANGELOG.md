# Changelog

All notable changes to ALICE-ML will be documented in this file.

## [Unreleased]

### Fixed
- **`SafetensorsFile::tensor_to_f32` / `tensor_bytes` panicked on a header that disagrees with the data**: a `shape` larger than the byte range (`shape: [1000]`, `data_offsets: [0, 4]`) indexed past the slice, inverted `data_offsets` sliced `start > end`, and a hostile `shape` overflowed `n_elements` Both return `None` now, `n_elements` saturates, an unsupported dtype returns `None` before touching the bytes (`tests/analytic_oracle.rs` § safetensors_format)
- **`ternary_matvec_simd_dispatch` (safe, crate root export) と arch 別 `ternary_matvec_dispatch` は入力長を検証していなかった** AVX2 / NEON kernel は `input` を raw pointer で読むため、`input.len() < in_features` で out-of-bounds read (UB) `assert_eq!` で長さを確認して panic に変更 (`# Panics` doc、`simd_dispatch_rejects_short_input` test) scalar kernel は slice index で panic していた
- **`no_std` build が一度も通っていなかった** (`--no-default-features` で 65 error: `mul_add` / `sqrt` / `exp` / `vec!` 等) — `src/math.rs` の `FloatExt` trait (`libm` 委譲、`std` 時は不使用) と `alloc::vec` import で修正、`parallel` feature は `std` を要求するよう明示 host rlib + bare-metal `thumbv7em-none-eabihf` で build を確認 (`crate-type` に cdylib / staticlib を含むため `cargo check` では panic_handler / allocator 要求で落ちる、検証は `cargo rustc --crate-type rlib`)
- feature-gated module (`ffi` / `safetensors` / `llama3_ternary`) の clippy pedantic / nursery 38 件と rustdoc 未解決 link 2 件 (CI が `--features simd` のみで未 lint だった)
- `elyza_ternary` / `qwen_qat_test` example に `required-features = ["safetensors"]` (default build で unresolved import)

### Added
- `ffi`: 全 65 `extern "C"` 関数の本体を `catch_unwind` (`guarded`) で囲み、Rust panic を host process の abort ではなく sentinel 戻り値 (null / 0 / 0.0 / -1) にする `am_ml_last_error()` で thread-local の message (NUL 終端) を取得、`am_ml_clear_last_error()` で消去 UE5 / Unity / C から呼ぶ時に一つの bad input で editor が落ちない

- `#![deny(clippy::undocumented_unsafe_blocks)]`: 手書き `unsafe` block 212 個 (ffi 195 / neon 6 / tensor 8 / arena 2 / ops 1) 全てに `// SAFETY:` で不変条件を記載

### Changed
- `alice-db` は crates.io の `0.2.0-beta.2` (path dep 廃止) CI の "dependency stubs" (空 crate、`db` feature を何も compile しない偽 gate) と `alice-stubs` action を撤去、`db` を native feature set と feature-powerset に含めて実体で check
- `ternary_matvec` / `ternary_matvec_kernel` / `ternary_matvec_kernel_quantized` の長さ検証を `debug_assert_eq` → `assert_eq` (release でも SIMD dispatch と同じ `# Panics` 契約、scalar kernel は slice index で panic していたので UB は無かった)
- release profile の `panic = "abort"` を除去 (`catch_unwind` を無効化するため、理由は `Cargo.toml` の comment)
- `libm = "0.2"` を依存に追加 (`no_std` build のみ使用) `no_std` 時の `exp` / `sqrt` 等は platform libm と最終 ulp で異なりうる (ternary matvec の bit-exact 性は不変、`src/math.rs` doc 参照)
- `tests/analytic_oracle.rs`: 閉形式 oracle 32 本 — ternary matvec は packed / bit-parallel / SIMD dispatch / `from_packed_weight` / batch / INT8 入力の全 kernel を `y = γ·(Σ₊x − Σ₋x)` と `in_features = 1..=70` (NEON 4 lane / AVX2 8 lane / 32-bit word の端数 path 全部) で `==` 突合、quantize は `W ∈ {−a, 0, +a}` の自己再現 + γ = mean|W| + 正のスケール同変 + 丸め閾値 γ/2 (温度 τ で τγ/2)、圧縮率の bit 幅法則、softmax (Σ = 1 / shift 不変 / 2 要素閉形式 / fast 版は Schraudolph 6 % 内 + 順位保存)、layer norm (mean 0 / var 1 / affine 不変 / eps 独立)、rms norm (rms 1 / x/rms(x) 一致 / scale 不変) 初回 mutants (64.9 %) 後に 9 本追加: element-wise / reduction op を長さ 1..=40 で整数厳密 (AVX2 8 lane の端数 lane に極値を置く)、Tensor / OwnedTensor / QuantizedTensor accessor、layer / rms norm の weight・bias・eps 経路、softmax の行独立性と全 −inf 行の uniform fallback、`QuantStats` (count / range / mae / sparsity / Shannon entropy) と `compute_quantization_error` (mae / mse / rmse / max / SNR、完全一致で ∞)、sparse 量子化の閾値 = threshold·γ、QAT の stats と温度、bit-plane accessor と packed byte mask helper (全 256 byte)、`ternary_matmul_alloc` の 1D / [1, n] / [b, n] shape 2 回目 (74.4 %) 後に ternary Llama-3 forward 5 本: 4 次元 toy model (`from_parts`) で projection を zero / identity にして logits を閉形式 (zero → rms_norm(embedding)、identity v / o → hidden + [n0, n1, n0, n1]、identity FFN → hidden + silu(n)⊙n、KV cache 2 position と clear_cache で不変)、`memory_bytes` の総和、`save_atml` → `ModelArchive` の 8 layer、7 projection 未満で panic safetensors 3 本: format 定義から合成した file (F32 / F16 / BF16 / 未対応 dtype / nested `__metadata__`) の読み戻し、fp16 の IEEE 754 点 (max finite 65504 / min normal / max subnormal / ±inf / NaN) と bf16 = f32 上位 16 bit、malformed header 10 種が panic せず None training 3 本: MSE / MAE / cross-entropy の値と勾配 (one-hot で −log softmax、勾配和 0、1e-8 床で loss は 18.42 に cap)、SGD momentum の漸化式、Adam の初回 step = −lr·sign(g) と 20 step の漸化式突合 matvec 3 kernel の長さ契約 (catch_unwind 5 経路)
- `quality-deep.yml` (週次 / 手動 / PR 差分): cargo-mutants 16 shard、feature set = native (ffi / simd / parallel / safetensors / db)、test set = lib + `analytic_oracle`、`src/python.rs` / `src/math.rs` (no_std 専用 shim) / `src/neon.rs` (aarch64 専用) は除外 (`.cargo/mutants.toml`、理由は file 内)
- CI `neon` job (macos-latest = aarch64): NEON kernel を同じ oracle で clippy + test (今まで NEON path は CI で一度も走っていなかった)
- CI: clippy を `--all-targets` + full native feature set (`ffi,simd,parallel,safetensors`) で `-D warnings`、`no_std` job (host + thumbv7em + clippy-driver wrapper)、`feature-powerset` job (cargo-hack、std 固定 depth 2)、test / doc も full native feature set を追加、rust-cache

## [0.2.0] - 2026-03-06

### Added
- `micro_model` — L2 Cache-Resident Micro Model (`MicroModel`, `MicroModelBuilder`, `CacheBudget`)
  - RasPi 5 L2 (512KB) に全重みを常駐、~100 GB/s で推論
  - `CacheBudget` プリセット (l2_rpi5, l2_rpi5_dual, l2_256k, custom)
  - `build_random()` で決定的テストビルド (LCG64)
- `speculative` — `CacheResidentDecoder` (L2 ドラフト + DRAM 検証)
- `speculative` — `SpeculativeDecoder` (ドラフトモデル先読み + バッチ検証)
- `streaming` — `LayerStreamer` (オンデマンドレイヤーロード + LRU エビクション)
- FFI: 14 新関数 (`am_ml_micro_model_*` 8個, `am_ml_cache_decoder_*` 6個) → 合計 65 関数
- Unity C#: 14 新 DllImport + 2 RAII handles (`MicroModelHandle`, `CacheDecoderHandle`) → 合計 65 DllImport + 9 handles
- UE5 C++: 14 新 extern C + 2 RAII handles (`MicroModelPtr`, `CacheDecoderPtr`) → 合計 65 extern C + 9 handles
- 235 tests (205 core + 24 FFI + 6 doc-tests)

### Fixed
- `cargo clippy --all-features -- -W clippy::pedantic -W clippy::nursery` → 0 warnings (53 warnings 修正)
- `cargo fmt` → 0 diff
- `cargo doc --no-deps` → 0 warnings

## [0.1.1] - 2026-03-04

### Added
- `ffi` — C-ABI FFI 51 `extern "C"` functions (Arena/Weight/Kernel/Matvec/Tensor/BitLinear/Quantize/Version)
- Unity C# bindings — 51 DllImport + 7 RAII IDisposable handles (`bindings/unity/AliceMl.cs`)
- UE5 C++ bindings — 51 extern C + 7 RAII unique_ptr handles (`bindings/ue5/AliceMl.h`)
- FFI prefix: `am_ml_*`
- 115 tests (93 core + 20 FFI + 2 doc-tests)

### Fixed
- `cargo fmt` trailing spaces修正
- `tensor.rs` unused_mut修正

## [0.1.0] - 2026-02-23

### Added
- `arena` — Bump allocator for zero-allocation inference
- `tensor` — `Tensor` (borrowed), `OwnedTensor`, `QuantizedTensor` (INT8); DPS operations (relu, softmax, layer_norm, gelu, silu, rms_norm, etc.)
- `ops` — `TernaryWeight` (packed 2-bit), `TernaryWeightKernel` (bit-parallel SIMD); DPS kernels (matvec, matmul_batch, quantized variants)
- `layer` — `BitLinear` layer abstraction
- `quantize` — INT8 quantization/dequantization with absmax scaling
- `error_analysis` — Quantization error analysis and error budget tracking
- `neon` — ARM NEON SIMD kernels (feature-gated: `neon`)
- `python` — PyO3 + NumPy zero-copy bindings (feature-gated: `pyo3`)
- `db_bridge` — ALICE-DB training metrics persistence (feature-gated: `db`)
- Feature flags: `std`, `simd`, `neon`, `parallel`, `pyo3`, `db`
- `no_std` + `alloc` support
- 93 unit tests (100 with all features) + 2 doc-tests
- CI/CD (GitHub Actions: test, clippy pedantic, fmt, doc)
- `#[must_use]` on all public query functions
- `# Errors` doc sections on all `Result`-returning functions
