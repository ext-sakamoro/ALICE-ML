# Changelog

All notable changes to ALICE-ML will be documented in this file.

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
