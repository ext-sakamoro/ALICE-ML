# Changelog

All notable changes to ALICE-ML will be documented in this file.

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
