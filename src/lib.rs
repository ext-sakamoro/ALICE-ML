//! ALICE-ML: 1.58bit Ternary Inference Engine (Supernova Edition)
//!
//! > "Multiplication is expensive. Addition is all you need."
//! > "Allocation is a sin. Arena is salvation."
//!
//! A radical inference engine based on BitNet b1.58 research that eliminates:
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
//! - **No Dependencies**: Pure Rust, no_std compatible
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

#![warn(missing_docs)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

pub mod tensor;
pub mod ops;
pub mod quantize;
pub mod arena;

#[cfg(feature = "pyo3")]
pub mod python;

// ============================================================================
// Core Re-exports
// ============================================================================

pub use arena::Arena;

// Tensor types
pub use tensor::{
    Tensor,           // Borrowed, zero-allocation
    OwnedTensor,      // For model loading only
    QuantizedTensor,  // INT8 quantized
};

// DPS tensor operations
pub use tensor::{
    tensor_add,
    tensor_sub,
    tensor_scale,
    tensor_relu,
    tensor_relu_inplace,
    tensor_softmax,
    tensor_sum,
    tensor_mean,
    tensor_max,
    tensor_min,
    tensor_copy,
};

// Weight types
pub use ops::{
    TernaryWeight,       // Packed 2-bit
    TernaryWeightKernel, // Bit-parallel for SIMD
};

// DPS kernels (the hot path)
pub use ops::{
    ternary_matvec,             // Packed weights, DPS
    ternary_matmul_batch,       // Batched, DPS
    ternary_matvec_kernel,      // Bit-parallel, DPS
    ternary_matvec_kernel_quantized,  // INT8 input, DPS
};

// Legacy API (allocates - for tests/benchmarks only)
pub use ops::{
    ternary_matvec_alloc,
    ternary_matmul_alloc,
};

// Quantization
pub use quantize::{
    quantize_to_ternary,
    quantize_to_ternary_sparse,
    dequantize_from_ternary,
    QuantStats,
    QuantizationError,
    compute_quantization_error,
};

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
    pub const fn from_i8(v: i8) -> Self {
        match v {
            1 => Ternary::Plus,
            -1 => Ternary::Minus,
            _ => Ternary::Zero,
        }
    }

    /// Convert to i8
    #[inline(always)]
    pub const fn to_i8(self) -> i8 {
        match self {
            Ternary::Zero => 0,
            Ternary::Plus => 1,
            Ternary::Minus => -1,
        }
    }

    /// Pack 4 ternary values into a single byte
    #[inline(always)]
    pub const fn pack4(t0: Ternary, t1: Ternary, t2: Ternary, t3: Ternary) -> u8 {
        (t0 as u8) | ((t1 as u8) << 2) | ((t2 as u8) << 4) | ((t3 as u8) << 6)
    }

    /// Unpack 4 ternary values from a byte
    #[inline(always)]
    pub const fn unpack4(byte: u8) -> [Ternary; 4] {
        [
            Self::from_bits(byte & 0b11),
            Self::from_bits((byte >> 2) & 0b11),
            Self::from_bits((byte >> 4) & 0b11),
            Self::from_bits((byte >> 6) & 0b11),
        ]
    }

    /// Convert from 2-bit encoding
    #[inline(always)]
    pub const fn from_bits(bits: u8) -> Self {
        match bits {
            0b01 => Ternary::Plus,
            0b10 => Ternary::Minus,
            _ => Ternary::Zero,
        }
    }
}

// ============================================================================
// Prelude
// ============================================================================

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::Arena;
    pub use crate::tensor::{Tensor, OwnedTensor, QuantizedTensor};
    pub use crate::ops::{
        TernaryWeight, TernaryWeightKernel,
        ternary_matvec, ternary_matvec_kernel,
    };
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
        let packed = Ternary::pack4(
            Ternary::Plus,
            Ternary::Minus,
            Ternary::Zero,
            Ternary::Plus,
        );
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
        let weights = TernaryWeight::from_ternary(
            &[1, -1, 0, 1, -1, 1, 0, -1, 1],
            3, 3
        );

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
}
