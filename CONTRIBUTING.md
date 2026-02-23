# Contributing to ALICE-ML

## Build

```bash
cargo build
cargo build --features simd
cargo build --features parallel
```

## Test

```bash
cargo test
cargo test --features simd
```

## Lint

```bash
cargo clippy -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Design Constraints

- **1.58-bit ternary weights**: {-1, 0, +1} packed 4 per byte, eliminating floating-point multiplication.
- **Zero allocation inference**: Arena-based memory, Destination Passing Style (DPS) for all kernels.
- **AVX2 SIMD**: 8-wide branchless accumulation with `_mm256_blendv_ps` (feature-gated: `simd`).
- **ARM NEON**: 4-wide NEON kernels for aarch64 (feature-gated: `neon`).
- **INT8 quantized path**: absmax scaling for maximum throughput on quantized activations.
- **`no_std` + `alloc`**: runs on embedded targets without `std`.
