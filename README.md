# ALICE-ML

**1.58bit Ternary Inference Engine**

> "Multiplication is expensive. Addition is all you need."

English | [日本語](README_JP.md)

A radical inference engine based on BitNet b1.58 research that eliminates floating-point multiplication entirely. All matrix operations use only addition and subtraction with ternary weights {-1, 0, +1}.

## The Revolution

Traditional neural network inference requires expensive floating-point multiplications:

```
y = W · x  →  N×M FP32 multiplications
```

ALICE-ML eliminates ALL multiplications:

```
y = Σ(x[i] where W=+1) - Σ(x[i] where W=-1)  →  Additions only!
```

## Features

- **1.58-bit Weights**: Ternary quantization {-1, 0, +1} stored as 2-bit packed
- **No Multiplication**: Matrix operations use only add/sub
- **16x Compression**: 4 bytes → 0.25 bytes per weight
- **AVX2 SIMD**: 8-wide `_mm256` kernels for MatVec, add/sub/scale/ReLU
- **Rayon Parallel**: Batch MatMul auto-parallelized with `--features parallel`
- **Branchless ReLU**: `_mm256_max_ps` / `f32::max(0.0)` — zero branches in hot path
- **Zero-Copy Loading**: mmap model files directly
- **No Dependencies**: Pure Rust, zero external crates (rayon optional)
- **no_std Compatible**: Runs on bare metal / WASM

## Quick Start

```rust
use alice_ml::{Tensor, TernaryWeight, ternary_matmul, quantize_to_ternary};

// Quantize FP32 weights to ternary
let fp32_weights = vec![0.8, -0.9, 0.1, 0.7, -0.6, 0.2, -0.1, 0.9];
let (weights, stats) = quantize_to_ternary(&fp32_weights, 2, 4);
println!("Compression: {}x", weights.compression_ratio());  // 16x

// Create input
let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]);

// Inference: NO MULTIPLICATION!
let output = ternary_matmul(&input, &weights);
```

## How It Works

### Weight Packing

4 ternary values packed into 1 byte:

```
Byte: [w3:2bit][w2:2bit][w1:2bit][w0:2bit]

Encoding:
  0b00 = 0   (zero, skip)
  0b01 = +1  (add input)
  0b10 = -1  (subtract input)
```

### Ternary MatMul Kernel

```rust
// For each output neuron:
for weight in row {
    match weight {
        +1 => acc_plus += input[j],   // Just addition!
        -1 => acc_minus += input[j],  // Just addition!
        0  => { }                      // Skip (sparsity bonus)
    }
}
output[i] = (acc_plus - acc_minus) * scale;
```

## Performance

| Operation | FP32 | ALICE-ML | Improvement |
|-----------|------|----------|-------------|
| MatMul (1024×1024) | 2M muls | 0 muls | ∞ |
| Memory | 4MB | 256KB | 16x |
| Power* | 100W | ~12W | 8x |

*Estimated based on elimination of FPU usage

## Quantization

BitNet b1.58 style quantization with learned scaling:

```rust
use alice_ml::quantize_to_ternary;

let (weights, stats) = quantize_to_ternary(&fp32_weights, rows, cols);

println!("Scale factor: {}", stats.scale);
println!("Sparsity: {:.1}%", stats.sparsity() * 100.0);
println!("Effective bits: {:.2}", stats.effective_bits());  // ~1.58
println!("MAE: {}", stats.mae);
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      ALICE-ML                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌────────────┐  │
│  │   Tensor    │───▶│  Ternary    │───▶│   Output   │  │
│  │  (f32/i8)   │    │   MatMul    │    │   (f32)    │  │
│  └─────────────┘    └─────────────┘    └────────────┘  │
│         │                  │                  │        │
│         ▼                  ▼                  ▼        │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Packed Weights (2-bit)              │   │
│  │  00 = 0, 01 = +1, 10 = -1, 11 = reserved        │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                             │
│                          ▼                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │                 SIMD Kernel                      │   │
│  │  • Extract +1 mask → horizontal add              │   │
│  │  • Extract -1 mask → horizontal sub              │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Modules

- `tensor.rs` - N-dimensional array with Arena allocator
- `ops.rs` - Ternary MatMul kernel (the heart)
- `quantize.rs` - FP32 → Ternary conversion
- `arena.rs` - Zero-allocation memory management

## Use Cases

1. **Edge Inference**: Run LLMs on devices with limited FPU
2. **Energy Efficiency**: Eliminate power-hungry multipliers
3. **Embedded AI**: no_std compatible for microcontrollers
4. **Model Compression**: 16x smaller model files
5. **Deterministic Game AI**: Bit-exact neural inference for networked games (via ALICE-Physics)

## Integration with ALICE-Physics

ALICE-ML's ternary weights are the key to **deterministic neural inference** in physics simulations. By combining ternary {-1, 0, +1} weights with [ALICE-Physics](../ALICE-Physics)' 128-bit fixed-point arithmetic, the entire inference pipeline reduces to pure addition/subtraction — guaranteeing bit-exact results across all platforms.

### Why Ternary + Fixed-Point = Determinism

```
Traditional NN:   y = W · x  → FP32 multiply → platform-dependent rounding
ALICE-ML + Fix128: y = Σ(±x)  → Fix128 add/sub → bit-exact everywhere
```

With ternary weights, there is no floating-point multiplication at all. The only multiply is a single scale factor (precomputed as Fix128 at model load time). This makes it the ideal inference engine for:

- **Fighting games**: Frame-perfect rollback netcode with AI opponents
- **Action games**: Ragdoll controllers that behave identically on all clients
- **Competitive multiplayer**: Zero desync from AI-controlled entities

### Usage

```rust
use alice_ml::{TernaryWeight, quantize_to_ternary};
use alice_physics::neural::*;
use alice_physics::Fix128;

// Quantize your trained model
let (weights, stats) = quantize_to_ternary(&fp32_weights, out_features, in_features);

// Convert to fixed-point (deterministic from this point forward)
let fixed_weights = FixedTernaryWeight::from_ternary_weight(weights);

// Build network
let mut network = DeterministicNetwork::new(
    vec![fixed_weights],
    vec![Activation::ReLU],
);

// Inference — bit-exact on every platform
let input = vec![Fix128::from_int(1); in_features];
let output = network.forward(&input);
```

### Requirements

```toml
[dependencies]
alice-physics = { path = "../ALICE-Physics", features = ["neural"] }
alice-ml = { path = "../ALICE-ML" }
```

## Cargo Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `simd` | No | AVX2 8-wide SIMD kernels (tensor ops + ternary MatVec) |
| `parallel` | No | Rayon parallel batch MatMul |

```bash
cargo build --release --features simd       # Enable SIMD kernels
cargo build --release --features parallel   # Enable Rayon batch parallelism
cargo build --release --features "simd,parallel"  # Both
```

## Optimizations

### SIMD Kernels (`--features simd`)

All tensor operations vectorized to process 8 floats per cycle:

| Operation | Scalar | SIMD (AVX2) |
|-----------|--------|-------------|
| `tensor_add` | 1 add/cycle | `_mm256_add_ps` — 8 adds/cycle |
| `tensor_sub` | 1 sub/cycle | `_mm256_sub_ps` — 8 subs/cycle |
| `tensor_scale` | 1 mul/cycle | `_mm256_mul_ps` — 8 muls/cycle |
| `tensor_relu` | branch + cmp | `_mm256_max_ps` — 8-wide branchless |
| `ternary_matvec` | Packed 2-bit decode | `_mm256_blendv_ps` mask select |

### Parallel Batch (`--features parallel`)

Batch MatMul auto-parallelizes across samples via Rayon `par_chunks_mut`:

```rust
// Batch of 64 inputs × 1024 features → parallel across CPU cores
ternary_matmul_batch(&inputs, &weights, &mut outputs, 64);
```

### Branchless Everything

ReLU uses `f32::max(0.0)` which compiles to `maxss`/`maxps` — no branch prediction misses.

## Roadmap

- [x] Fixed-point inference via ALICE-Physics integration (deterministic game AI)
- [x] AVX2 SIMD kernels (tensor ops + ternary MatVec)
- [x] Rayon parallel batch MatMul
- [ ] NEON SIMD kernels (ARM)
- [ ] BitLinear layer (drop-in nn.Linear replacement)
- [ ] Knowledge distillation from PyTorch models
- [ ] Early exit for dynamic depth
- [ ] `.aml` model format with mmap loading

## References

- [BitNet: Scaling 1-bit Transformers](https://arxiv.org/abs/2310.11453)
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) - The 1.58-bit breakthrough
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)

## License

AGPL-3.0

## Author

Moroya Sakamoto ([@ext-sakamoro](https://github.com/ext-sakamoro))
