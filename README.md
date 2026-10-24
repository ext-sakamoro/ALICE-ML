# ALICE-ML

[![crates.io](https://img.shields.io/crates/v/alice-ml.svg)](https://crates.io/crates/alice-ml)
[![docs.rs](https://img.shields.io/docsrs/alice-ml)](https://docs.rs/alice-ml)
[![MSRV](https://img.shields.io/crates/msrv/alice-ml)](#msrv)
[![CI](https://github.com/ext-sakamoro/ALICE-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-ML/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey)](LICENSE-Proprietary)

English | [日本語](README_JP.md)

A 1.58-bit ternary (`BitNet`-style) inference engine. Weights are packed to
`{-1, 0, +1}`, so the matrix-vector kernel at the core of inference is
addition and subtraction only — no floating-point multiplication, no runtime
allocation (every buffer comes from an arena, written in Destination Passing
Style). `no_std` compatible for bare-metal and embedded targets.

## What this is for

Running a ternary-quantized model's forward pass on CPU — edge devices,
embedded targets, or anywhere the multiply-heavy float matmul of a
full-precision model is the wrong trade. The crate also carries the
surrounding pieces a ternary deployment needs: quantization-aware training
support, probability calibration, speculative decoding, weight streaming, and
C-ABI / Unity / UE5 / Python bindings.

## What this is not for

Training a model from scratch at full precision, or general-purpose tensor
math unrelated to ternary inference — [`Tensor`](src/tensor.rs) exists to
support the kernels in this crate, not as a standalone ndarray replacement.
If you need GGUF-format model loading or a full LLM runtime, see
[ALICE-LLM](https://github.com/ext-sakamoro/ALICE-LLM) instead; this crate's
own model format is `.atml` (`src/model_io.rs`, `src/safetensors.rs`).

## Install

```bash
cargo add alice-ml
```

Optional features (see [Features](#features) below) add FFI bindings, SIMD
kernels, Rayon-parallel batching, ALICE-DB persistence, or Python bindings.

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

## What is, and is not, deterministic

The ternary matvec kernels (scalar, SIMD, and the INT8-input variant) never
call a transcendental function: a ternary weight selects `+input`, `-input`,
or `0`, so the kernel is a sum of additions and subtractions, and every
kernel variant is checked against the same closed-form value
<!-- claim-test: every_matvec_kernel_equals_the_closed_form_for_all_lengths -->
(`tests/analytic_oracle.rs`). That part needs no special arithmetic library to
be reproducible across platforms.

Outside the matvec kernels, this crate does call transcendental functions —
softmax, RoPE, Adam's bias correction, calibration's temperature fitting,
quantization's entropy and SNR. As of 0.3.0, every one of those (in non-test
code) is routed through
[`alice_det_math`](https://crates.io/crates/alice-det-math) rather than
`f32`/`f64`'s own platform-libm methods (not required by IEEE 754 to be
cross-platform bit-exact). The full set this crate bans this way, for both
`f32` and `f64`, is ALICE-Law's G1 ("no-libm-direct") gate's banned-method
list (`sin`/`cos`/`tan`/`sin_cos`/`asin`/`acos`/`atan`/`atan2`/`sinh`/`cosh`/
`tanh`/`asinh`/`acosh`/`atanh`/`exp`/`exp2`/`exp_m1`/`ln`/`log`/`log2`/
`log10`/`ln_1p`/`powf`/`cbrt`/`hypot`), plus `powi` and `mul_add` (see
below) -- a `clippy::disallowed_methods` lint on all of them, in
`clippy.toml`, keeps a new one from being added back silently;
`tests/g1_disallowed_methods_complete.rs` re-checks the list is still
complete.

`sqrt` and `round` are not routed this way: IEEE 754 requires both to be
correctly-rounded operations, so they are already cross-platform bit-exact
without a library on top. `hypot` is a real case of the same platform-libm
problem as `exp`/`ln` (Rust's own docs call its precision "unspecified" and
platform-dependent), so it is banned and routed the same way: the crate's one
`f64` `hypot` call site was rewritten to `alice_det_math::sqrt64(a*a + b*b)`,
since `alice_det_math` 0.5 does not yet provide a 64-bit `hypot`.

`powi` is G1's one "pending" verdict: whether it is bit-exact across targets
(a compiler-runtime routine, repeated multiplication with a target-dependent
order) is still being measured. G1 lets a crate ban it in the meantime; this
crate does, for both `f32` and `f64`. The one production call site
(`quantize.rs`'s squared error) is `f32`, routed through
`alice_det_math::powi`; `alice_det_math` 0.5 has no 64-bit `powi` yet, so
`f64::powi` is banned pre-emptively (no production call site today, but a
future one would otherwise go unrouted silently).

`mul_add` is not a drift source — Rust specifies it as IEEE 754
`fusedMultiplyAdd`, single-rounding and bit-exact across platforms regardless
of whether the target CPU has a hardware FMA instruction (targets without one
fall back to a correctly-rounded software implementation). It is still banned
and rewritten to plain `a * b + c` everywhere in non-test code (ffi.rs,
layer.rs, llama3_ternary.rs, moe.rs, quantize.rs, tensor.rs, training.rs), for
a simpler reason: `a * b + c` and `a.mul_add(b, c)` round differently from
each other (two roundings vs. one), so writing one form everywhere keeps
every result reproducible from the source alone. This crate picks one form
and bans the other, matching `alice_det_math`'s own consumer policy.
(Layer/RMS norm's only other operation is `sqrt`, so with `mul_add` gone
those two call nothing platform-dependent at all.)

## What's included

| Area | Modules |
|---|---|
| Core kernels | `ops`, `tensor`, `layer`, `arena`, `quantize`, `neon` |
| Training & calibration | `training`, `calibration`, `error_analysis`, `dataset` |
| Inference | `llama3_ternary`, `micro_model`, `moe`, `speculative`, `streaming` |
| I/O & persistence | `model_io`, `safetensors`, `db_bridge` |
| Bindings | `ffi`, `python` |

See [`docs/MODULES.md`](docs/MODULES.md) for every module with a one-line
description and the feature it needs.

## Features

<!-- readme-sync: features -->
| Feature | Enables |
|---|---|
| `std` (default) | `f32`/`f64` inherent methods; off for `no_std` (bare-metal/embedded) |
| `ffi` | The C-ABI surface (`src/ffi.rs`), for the Unity/UE5 bindings below |
| `simd` | Hand-tuned AVX2 kernels (`x86_64`) |
| `neon` | Hand-tuned NEON kernels (`aarch64`) |
| `parallel` | Rayon-parallel batched matmul |
| `pyo3` | Python bindings (`src/python.rs`) |
| `db` | [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB)-backed training metric persistence |
| `safetensors` | Safetensors parsing + Llama-3 ternary inference (`llama3_ternary`) |

## Bindings

- **C-ABI** (`--features ffi`): every exported function is `catch_unwind`-guarded;
  a panic returns an error code rather than unwinding across the FFI boundary
  (`am_ml_last_error` reports the message). See `src/ffi.rs`.
- **Unity** (C#): [`bindings/unity/AliceMl.cs`](bindings/unity/AliceMl.cs), a
  thin P/Invoke wrapper over the C-ABI.
- **Unreal Engine 5** (C++): [`bindings/ue5/AliceMl.h`](bindings/ue5/AliceMl.h).
- **Python** (`--features pyo3`): `src/python.rs`, built with PyO3 + numpy.

## Examples

| Example | What it shows |
|---|---|
| [`elyza_ternary`](examples/elyza_ternary.rs) | Loading and running a ternary-quantized ELYZA-JP-8B model |
| [`qwen_qat_test`](examples/qwen_qat_test.rs) | Loading QAT-trained ternary weights from `.atml` and running inference |

Run one with `cargo run --example <name> --features safetensors`.

## MSRV

Minimum supported Rust version: **1.87** <!-- readme-sync: msrv -->. The `msrv`
CI job builds the crate with exactly this toolchain (default and full native
feature sets, all targets, `no_std` rlib), and `scripts/check_msrv.py` fails if
the job's toolchain differs from `rust-version`. `rust-toolchain.toml` is a
development pin, not the MSRV.

## Build and test

```bash
cargo test --features ffi,simd,parallel,safetensors,db
cargo bench --features simd  # criterion benchmarks, see benches/matmul.rs
```

`scripts/preflight.sh` (`--quick` for a push-sized subset) runs the same
checks as CI, including the `no_std` build for a bare-metal target.

## Related crates

- [ALICE-LLM](https://github.com/ext-sakamoro/ALICE-LLM) — GGUF-format model
  loading and a full LLM runtime (a different model format than this crate's
  `.atml`)
- [alice-det-math](https://crates.io/crates/alice-det-math) — the
  cross-platform deterministic transcendental functions this crate routes
  through outside its ternary kernels
- [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) — training metric
  persistence (feature `db`)

<!-- law-integration:begin law=e75d038a6aaa51edc64aa014a24f0b2617e34acd -->
## ALICE-Law integration

Generated by ALICE-Law `scripts/gen-law-readme-block.py` from ALICE-Law `e75d038a6aaa` and this repository's `alice-law-conformance.toml`; do not edit it by hand.

| | |
|---|---|
| Release train | alice-ml 0.2.0 (published), law dependency: no |
| Arithmetic | alice-det-math 0.5.1 in the train, arithmetic identifier `85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315`; this crate requires `alice-det-math = "0.5.1"` |
| End to end check | not part of the end to end check |

### Conformance (G1-G14)

| gate | status | note |
|---|---|---|
| G1 no-libm-direct | pass | layers that read: clippy.toml bans all 25 of G1's banned methods for f32 and f64 (plus powi and mul_add for both widths, and the reasons are in the file; … |
| G2 ieee-max-min | pass | every float maximum, minimum, clamp, argmax and selection order of the library goes through src/law_math.rs. Residual: the AVX2 max / min intrinsics are not … |
| G3 no-outer-fneg-of-fma | pass | no fused multiply-add call in the library sources (clippy.toml also bans mul_add there): the syntax-tree check reads 37 files and counts 2 fma-family calls, … |
| G4 debug-release-arch-goldens | pass | the AVX2 lanes of law_math are not covered by this golden on a local x86_64 (Rosetta 2 has no AVX2); on CI they are taken when the runner reports avx2 and are … |
| G5 release-profile-test-leg | pass | the x86_64 leg is not claimed to cover the AVX2 lanes (taken only when the runner reports avx2, printed in the log); the hardware fma claim rests on the … |
| G6 semantics-id | pass | hex with alice-det-math 0.5.1: a547237044e8fdba9ab152f7af13c431f675e5382e4e8cb40aa8c83c46bb606b. The identifier attests the pinned kernels (relu, max_min, … |
| G7 handshake-carries-id | pass | the list of type-erasing boundaries that receive another Law-bearing crate's value is empty: the C ABI and the Python exports take the caller's numbers, the … |
| G8 persisted-semantics-id | pass | the identifier recorded is ML_SEMANTICS_ID (G6): it pins the quantiser whose scale and packed values the file stores, and the det-math generation. db_bridge … |
| G9 persisted-meaning-change-is-major | pass | definition: ML_SEMANTICS_ID ([[id]] in scripts/id_move.toml with PREVIOUS_ML_SEMANTICS_IDS) may move only with a breaking version, or keeping the released … |
| G10 no-run-varying-artifacts | pass | the ban covers what src/ can reach: the clock and RandomState through std, and nothing else, because no direct dependency offers an RNG or a UUID (rand is a … |
| G11 gate-self-tests | known unmet | unmet: clause (c). scripts/clippy_canary.py, scripts/test_clippy_canary.py, scripts/gen_clippy_canary_calls.py and scripts/test_gen_clippy_canary_calls.py are … |
| G12 public-repo-text-rules | pass | the workflow fetches ALICE-Law's scripts/public_text_check.py at the revision in .github/public-text-pin (779fa443f95bc0caca5ca2d5bdb9156051fd6143) and refuses … |
| G13 no-unspecified-order | pass | the batch computes each output row in one task and combines no float across tasks, so its bits do not depend on the thread count; the test is what keeps it so. … |
| G14 float-environment-invariants | pass | the static scan covers the files of this repository (.cargo/mutants.toml included) and the dynamic check covers the builds CI runs on ubuntu and macOS (the … |

### Features and the guarantee

| feature | guarantee | note |
|---|---|---|
| `db` | outside | ALICE-DB metrics sink; the rows are alice-db's own format with its own identifier, read through alice-db's Rust API (G7, in memory); this crate adds no … |
| `ffi` | outside | C ABI entry points (67 extern "C" fn); the max / min / relu / softmax kernels are pinned by the ffi golden and tests/law_ties.rs, the rest by the G1 lint only; … |
| `neon` | outside | NEON kernel on aarch64; no golden compares its bits with the scalar path across architectures (G4) |
| `parallel` | outside | rayon over output rows (ops::ternary_matmul_batch); the same bits at 1, 2 and 8 threads (tests/parallel_order.rs, G13); no G4 golden |
| `pyo3` | outside | linted by the clippy job's `Clippy (pyo3, lib)` step (G1); no test builds it (a Python interpreter is needed), so its maximum / minimum sites are covered only … |
| `safetensors` | outside | safetensors parser and Llama-3 ternary path; no Law gate beyond the G1 lint, no golden (G4) |
| `simd` | outside | AVX2 kernel on x86_64; the max / min / relu lanes are tie-tested and mutant-checked on CI's x86_64 runner (law-tie-mutants.yml), but no G4 golden spans the … |
| `std` | outside | default feature; G6 and G8 are met (0.4.0) but G4 and G5 are not (no signed-zero or cancellation golden on every architecture, no release-profile test leg), so … |
<!-- law-integration:end -->

## License

`LicenseRef-Proprietary`. The text is [LICENSE-Proprietary](LICENSE-Proprietary).

Enquiries: https://extoria.co.jp/
