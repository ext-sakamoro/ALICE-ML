# Contributing to ALICE-ML

## Prerequisites

- Rust 1.70+ (stable)
- `clippy`, `rustfmt` コンポーネント (`rustup component add clippy rustfmt`)
- PyO3ビルドに Python 3.8+ (feature `pyo3` 使用時)

## Code Style

- `cargo fmt` 準拠（CI で `--check` 実行）
- `cargo clippy --all-features -- -W clippy::all -W clippy::pedantic` 警告ゼロ
- パブリック関数には `#[must_use]` を付与
- `Result` 返却関数には `# Errors` docセクション必須
- `no_std` 互換: `std` は feature gate 経由（`alloc` のみ）
- コード内コメント: 日本語

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
