# Modules

Every `pub mod` of `alice-ml`, one line each. See [docs.rs](https://docs.rs/alice-ml)
for the full API.

| Module | Description | Feature |
|---|---|---|
| `arena` | Arena allocator for zero-allocation inference | — |
| `calibration` | Probability calibration — measuring it, and fixing it | — |
| `dataset` | `Dataset` trait and batch iterators | — |
| `error_analysis` | Cumulative quantization error analysis | — |
| `layer` | `BitLinear` layer, a drop-in replacement for `nn.Linear` using 1.58-bit ternary weights | — |
| `model_io` | Model weight serialization/deserialization (`.atml`) | — |
| `ops` | Ternary matrix operations — the matvec/matmul kernels | — |
| `quantize` | FP32 → ternary `{-1, 0, +1}` quantization | — |
| `tensor` | Zero-allocation N-dimensional array | — |
| `training` | Training loop — loss functions, optimizers, training steps | — |
| `neon` | ARM NEON 4-wide ternary matvec kernel | `neon` (`aarch64` only) |
| `micro_model` | L2-cache-resident micro model for speculative decoding drafts | — |
| `moe` | Mixture-of-Experts router | — |
| `speculative` | Speculative decoding — a draft model proposes, a verifier model checks | — |
| `streaming` | Layer-by-layer on-demand weight loading | — |
| `db_bridge` | [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) bridge for model weight and training metric persistence | `db` |
| `safetensors` | Safetensors format parser (BF16/FP16/FP32) | `safetensors` |
| `llama3_ternary` | Llama-3 inference with 1.58-bit ternary weights | `safetensors` |
| `ffi` | C-ABI FFI surface | `ffi` |
| `python` | `PyO3` Python bindings | `pyo3` |
| `prelude` | Convenient re-export of the commonly-used types | — |
