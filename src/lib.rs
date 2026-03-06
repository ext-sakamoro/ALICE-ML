//! ALICE-ML: 1.58bit Ternary Inference Engine (Supernova Edition)
//!
//! > "Multiplication is expensive. Addition is all you need."
//! > "Allocation is a sin. Arena is salvation."
//!
//! A radical inference engine based on `BitNet` b1.58 research that eliminates:
//! - **Floating-point multiplication** → Only add/sub with ternary weights {-1, 0, +1}
//! - **Runtime allocation** → Arena-based memory, DPS (Destination Passing Style)
//!
//! # Core Innovation
//!
//! Traditional neural network inference:
//! ```text
//! y = W · x  (requires N×M multiplications + N allocations)
//! ```
//!
//! ALICE-ML Supernova:
//! ```text
//! ternary_matvec(input, weights, output)  // add/sub only, zero allocation
//! ```
//!
//! # Features
//!
//! - **1.58-bit Weights**: Ternary {-1, 0, +1} packed 4 per byte
//! - **Zero Allocation**: All buffers from Arena, DPS everywhere
//! - **AVX2 SIMD**: 8 floats per cycle with branchless masking
//! - **INT8 Activations**: Optional quantized path for maximum throughput
//! - **No Dependencies**: Pure Rust, `no_std` compatible
//!
//! # Example
//!
//! ```rust
//! use alice_ml::{TernaryWeight, ternary_matvec};
//!
//! // Load weights (once, at model load)
//! let weights = TernaryWeight::from_ternary(&[1, -1, 0, 1], 2, 2);
//!
//! // Stack-allocated buffers (or from Arena)
//! let input = [2.0f32, 3.0];
//! let mut output = [0.0f32; 2];
//!
//! // Inference: ZERO ALLOCATION, NO MULTIPLICATION
//! ternary_matvec(&input, &weights, &mut output);
//!
//! assert!((output[0] - (-1.0)).abs() < 1e-6);  // 2*1 + 3*(-1) = -1
//! assert!((output[1] - 3.0).abs() < 1e-6);     // 2*0 + 3*1 = 3
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                  ALICE-ML Supernova Edition                     │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌─────────┐    ┌──────────────┐    ┌─────────┐                │
//! │  │  Arena  │───▶│ Tensor<'a>   │───▶│ Output  │                │
//! │  │ (alloc) │    │ (borrowed)   │    │ (arena) │                │
//! │  └─────────┘    └──────────────┘    └─────────┘                │
//! │       │                │                  │                     │
//! │       ▼                ▼                  ▼                     │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │              Ternary Weights (2-bit packed)              │   │
//! │  │  TernaryWeight: 4 per byte  │  TernaryWeightKernel: SIMD │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │                          │                                      │
//! │                          ▼                                      │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │           DPS Kernels (Destination Passing Style)        │   │
//! │  │  ternary_matvec(in, weights, out) → writes to out        │   │
//! │  │  No return value, no allocation, just computation        │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! │                          │                                      │
//! │                          ▼                                      │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │                 AVX2 SIMD (if available)                 │   │
//! │  │  _mm256_blendv_ps → branchless 8-wide accumulation       │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

// テストコードではアサーション内のformat!やf32比較でpedantic警告が出るため抑制
#![cfg_attr(
    test,
    allow(
        clippy::float_cmp,
        clippy::uninlined_format_args,
        clippy::doc_markdown,
        clippy::cloned_instead_of_copied,
        clippy::borrow_as_ptr,
        clippy::ref_as_ptr,
    )
)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::too_many_lines
)]
#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod arena;
pub mod dataset;
pub mod error_analysis;
pub mod layer;
pub mod model_io;
pub mod ops;
pub mod quantize;
pub mod tensor;
pub mod training;

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub mod neon;

pub mod micro_model;
pub mod speculative;
pub mod streaming;

#[cfg(feature = "db")]
pub mod db_bridge;

#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "pyo3")]
#[allow(clippy::needless_pass_by_value)] // PyO3 requires pass-by-value for Python types
pub mod python;

// ============================================================================
// Core Re-exports
// ============================================================================

pub use arena::Arena;

// Tensor types
pub use tensor::{
    OwnedTensor,     // For model loading only
    QuantizedTensor, // INT8 quantized
    Tensor,          // Borrowed, zero-allocation
};

// DPS tensor operations
pub use tensor::{
    tensor_add, tensor_copy, tensor_gelu, tensor_gelu_inplace, tensor_layer_norm, tensor_max,
    tensor_mean, tensor_min, tensor_relu, tensor_relu_inplace, tensor_rms_norm, tensor_scale,
    tensor_silu, tensor_silu_inplace, tensor_softmax, tensor_softmax_fast, tensor_sub, tensor_sum,
};

// Layer abstractions
pub use layer::BitLinear;

// Weight types
pub use ops::{
    TernaryWeight,       // Packed 2-bit
    TernaryWeightKernel, // Bit-parallel for SIMD
};

// DPS kernels (the hot path)
pub use ops::{
    ternary_matmul_batch,            // Batched, DPS
    ternary_matvec,                  // Packed weights, DPS
    ternary_matvec_kernel,           // Bit-parallel, DPS
    ternary_matvec_kernel_quantized, // INT8 input, DPS
};

// Legacy API (allocates - for tests/benchmarks only)
pub use ops::{ternary_matmul_alloc, ternary_matvec_alloc};

// Cross-platform SIMD dispatch
pub use ops::ternary_matvec_simd_dispatch;

// Quantization
pub use quantize::{
    compute_quantization_error, dequantize_from_ternary, quantize_to_ternary,
    quantize_to_ternary_sparse, QuantStats, QuantizationError,
};

// Cumulative quantization error analysis
pub use error_analysis::{
    compute_layer_error_propagation, CumulativeQuantError, LayerConfig, LayerErrorStats,
    NetworkErrorReport,
};

// Micro Model (L2 cache-resident draft)
pub use micro_model::{CacheBudget, MicroModel, MicroModelBuilder};

// Speculative Decoding
pub use speculative::{CacheResidentDecoder, DecodeResult, DecoderConfig, SpeculativeDecoder};

// Weight Streaming
pub use streaming::{LayerStreamer, SlotState, StreamerConfig, StreamerStats};

// ============================================================================
// Ternary Encoding
// ============================================================================

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Ternary value encoding in 2 bits
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Ternary {
    /// Zero weight (no contribution)
    Zero = 0b00,
    /// Positive weight (+1: add input)
    Plus = 0b01,
    /// Negative weight (-1: subtract input)
    Minus = 0b10,
}

impl Ternary {
    /// Convert from i8 (-1, 0, +1)
    #[inline(always)]
    #[must_use]
    pub const fn from_i8(v: i8) -> Self {
        match v {
            1 => Self::Plus,
            -1 => Self::Minus,
            _ => Self::Zero,
        }
    }

    /// Convert to i8
    #[inline(always)]
    #[must_use]
    pub const fn to_i8(self) -> i8 {
        match self {
            Self::Zero => 0,
            Self::Plus => 1,
            Self::Minus => -1,
        }
    }

    /// Pack 4 ternary values into a single byte
    #[inline(always)]
    #[must_use]
    pub const fn pack4(t0: Self, t1: Self, t2: Self, t3: Self) -> u8 {
        (t0 as u8) | ((t1 as u8) << 2) | ((t2 as u8) << 4) | ((t3 as u8) << 6)
    }

    /// Unpack 4 ternary values from a byte
    #[inline(always)]
    #[must_use]
    pub const fn unpack4(byte: u8) -> [Self; 4] {
        [
            Self::from_bits(byte & 0b11),
            Self::from_bits((byte >> 2) & 0b11),
            Self::from_bits((byte >> 4) & 0b11),
            Self::from_bits((byte >> 6) & 0b11),
        ]
    }

    /// Convert from 2-bit encoding
    #[inline(always)]
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        match bits {
            0b01 => Self::Plus,
            0b10 => Self::Minus,
            _ => Self::Zero,
        }
    }
}

// ============================================================================
// Prelude
// ============================================================================

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::ops::{
        ternary_matvec, ternary_matvec_kernel, TernaryWeight, TernaryWeightKernel,
    };
    pub use crate::tensor::{OwnedTensor, QuantizedTensor, Tensor};
    pub use crate::Arena;
    pub use crate::Ternary;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_encoding() {
        assert_eq!(Ternary::from_i8(1), Ternary::Plus);
        assert_eq!(Ternary::from_i8(-1), Ternary::Minus);
        assert_eq!(Ternary::from_i8(0), Ternary::Zero);

        assert_eq!(Ternary::Plus.to_i8(), 1);
        assert_eq!(Ternary::Minus.to_i8(), -1);
        assert_eq!(Ternary::Zero.to_i8(), 0);
    }

    #[test]
    fn test_ternary_packing() {
        let packed = Ternary::pack4(Ternary::Plus, Ternary::Minus, Ternary::Zero, Ternary::Plus);
        assert_eq!(packed, 0b01_00_10_01);

        let unpacked = Ternary::unpack4(packed);
        assert_eq!(unpacked[0], Ternary::Plus);
        assert_eq!(unpacked[1], Ternary::Minus);
        assert_eq!(unpacked[2], Ternary::Zero);
        assert_eq!(unpacked[3], Ternary::Plus);
    }

    #[test]
    fn test_ternary_roundtrip() {
        for v in [-1i8, 0, 1] {
            let t = Ternary::from_i8(v);
            assert_eq!(t.to_i8(), v);
        }
    }

    #[test]
    fn test_full_inference_pipeline() {
        // Complete zero-allocation inference demo using stack arrays

        // 1. Load model weights (one-time allocation at startup)
        let weights = TernaryWeight::from_ternary(&[1, -1, 0, 1, -1, 1, 0, -1, 1], 3, 3);

        // 2. Stack-allocated input/output
        let input = [1.0f32, 2.0, 3.0];
        let mut output = [0.0f32; 3];

        // 3. Run inference (ZERO ALLOCATION)
        ternary_matvec(&input, &weights, &mut output);

        // 4. Check results
        // Row 0: [1, -1, 0] · [1, 2, 3] = 1 - 2 + 0 = -1
        // Row 1: [1, -1, 1] · [1, 2, 3] = 1 - 2 + 3 = 2
        // Row 2: [0, -1, 1] · [1, 2, 3] = 0 - 2 + 3 = 1
        assert!((output[0] - (-1.0)).abs() < 1e-6);
        assert!((output[1] - 2.0).abs() < 1e-6);
        assert!((output[2] - 1.0).abs() < 1e-6);
    }

    // Ternary::from_bits で全4ビットパターンを直接検証する
    #[test]
    fn test_ternary_from_bits_all_patterns() {
        // 0b01 → Plus, 0b10 → Minus, 0b00/0b11 → Zero
        assert_eq!(Ternary::from_bits(0b01), Ternary::Plus);
        assert_eq!(Ternary::from_bits(0b10), Ternary::Minus);
        assert_eq!(Ternary::from_bits(0b00), Ternary::Zero);
        // 0b11 は未定義パターン → Zero にフォールバックする
        assert_eq!(Ternary::from_bits(0b11), Ternary::Zero);

        // to_i8 との整合性
        assert_eq!(Ternary::from_bits(0b01).to_i8(), 1);
        assert_eq!(Ternary::from_bits(0b10).to_i8(), -1);
        assert_eq!(Ternary::from_bits(0b00).to_i8(), 0);
    }

    // pack4 → unpack4 の全組み合わせで可逆性を検証する
    #[test]
    fn test_ternary_pack_unpack_all_combos() {
        use Ternary::{Minus, Plus, Zero};
        let variants = [Plus, Minus, Zero];

        for &a in &variants {
            for &b in &variants {
                for &c in &variants {
                    for &d in &variants {
                        let byte = Ternary::pack4(a, b, c, d);
                        let unpacked = Ternary::unpack4(byte);
                        assert_eq!(
                            unpacked[0], a,
                            "pack4/unpack4 roundtrip failed at position 0"
                        );
                        assert_eq!(
                            unpacked[1], b,
                            "pack4/unpack4 roundtrip failed at position 1"
                        );
                        assert_eq!(
                            unpacked[2], c,
                            "pack4/unpack4 roundtrip failed at position 2"
                        );
                        assert_eq!(
                            unpacked[3], d,
                            "pack4/unpack4 roundtrip failed at position 3"
                        );
                    }
                }
            }
        }
    }
}
