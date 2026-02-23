# Changelog

All notable changes to ALICE-ML will be documented in this file.

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
- 93 unit tests + 2 doc-tests
