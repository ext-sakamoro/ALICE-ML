//! `no_std` 用の float 数学関数 shim
//!
//! `std` あり: `f32` / `f64` の inherent method (`sqrt` 等) をそのまま使う
//! (本 module は空)
//!
//! `std` なし (`--no-default-features`): core には `sqrt` / `round` が無いため、
//! 同名 method を [`FloatExt`] trait で提供し、実装は [`libm`] (pure Rust、
//! `no_std`) に委譲する 各 module は
//! `#[cfg(not(feature = "std"))] use crate::math::FloatExt;` で取り込む
//!
//! これらは IEEE 754 が正しい丸めを要求する操作 (`sqrt`) か、融合丸めを伴わない
//! 操作 (`round`) なので、`libm` 経由でも std inherent method と bit-exact 性に
//! 差は出ない
//!
//! `hypot` / `mul_add` はここには無い — `hypot` は f64 側に det-math の提供が
//! 無く、既存の唯一の呼び出し元は `sqrt64(a*a + b*b)` へ書き換え済 `mul_add`
//! は drift の原因ではない (IEEE 754 `fusedMultiplyAdd` として規定され、FMA
//! 命令の有無に関わらずどの target でも 1 回丸め、platform を跨いで bit 一致)
//! が、`a * b + c` (2 回丸め) とは違う bit を返す、だから 1 つの書き方に
//! 揃えて、どの結果も source だけから再現できるようにする目的で crate 全体で
//! 禁止 (`clippy.toml` の `disallowed-methods`)、全呼び出しを `a * b + c` へ
//! 書き換え済
//!
//! `exp` / `ln` / `log2` / `log10` / `powf` / `powi` もここには無い — 超越関数は
//! `std` / `no_std` を問わず常に `alice_det_math` (`no_std` 対応) を直接呼ぶ (crate
//! 全体で 1 本の決定論経路、`std` build と `no_std` build の分岐が無い)

#[cfg(not(feature = "std"))]
pub trait FloatExt: Sized {
    fn sqrt(self) -> Self;
    fn round(self) -> Self;
}

#[cfg(not(feature = "std"))]
impl FloatExt for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrtf(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::roundf(self)
    }
}

#[cfg(not(feature = "std"))]
impl FloatExt for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        libm::sqrt(self)
    }
    #[inline]
    fn round(self) -> Self {
        libm::round(self)
    }
}
