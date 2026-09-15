//! `no_std` 用の float 数学関数 shim
//!
//! `std` あり: `f32` / `f64` の inherent method (`sqrt` / `exp` / `mul_add` 等) を
//! そのまま使う (本 module は空)
//!
//! `std` なし (`--no-default-features`): core には float の超越関数が無いため、
//! 同名 method を [`FloatExt`] trait で提供し、実装は [`libm`] (pure Rust、
//! `no_std`) に委譲する 各 module は
//! `#[cfg(not(feature = "std"))] use crate::math::FloatExt;` で取り込む
//!
//! 精度注意: `libm` と platform libm (glibc / macOS / MSVC CRT) は最終 ulp で
//! 異なりうる ternary matvec (add / sub のみ) の bit-exact 性には影響しないが、
//! activation (`exp` / `sqrt` 経路) の値は std build と `no_std` build で
//! 完全一致を保証しない

#[cfg(not(feature = "std"))]
pub trait FloatExt: Sized {
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn round(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn hypot(self, other: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
}

#[cfg(not(feature = "std"))]
impl FloatExt for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::expf(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::logf(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2f(self)
    }
    #[inline]
    fn log10(self) -> Self {
        libm::log10f(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::roundf(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        libm::powf(self, n)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        // libm に powi は無い、std の powi と同じく pow(x, n as f32) で近似
        libm::powf(self, n as Self)
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypotf(self, other)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fmaf(self, a, b)
    }
}

#[cfg(not(feature = "std"))]
impl FloatExt for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn exp(self) -> Self {
        libm::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        libm::log(self)
    }
    #[inline]
    fn log2(self) -> Self {
        libm::log2(self)
    }
    #[inline]
    fn log10(self) -> Self {
        libm::log10(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::round(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        libm::pow(self, n)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        libm::pow(self, Self::from(n))
    }
    #[inline]
    fn hypot(self, other: Self) -> Self {
        libm::hypot(self, other)
    }
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        libm::fma(self, a, b)
    }
}
