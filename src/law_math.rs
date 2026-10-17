//! Maximum, minimum and clamp with one defined result on every target (ALICE-Law G2).
//!
//! `f32::max` / `f32::min` leave the sign of a zero result and the handling of NaN to
//! the platform, so the same input gives different bits on aarch64 and `x86_64`, and
//! between the debug and release profiles (measured by `tests/determinism_golden.rs`
//! before these helpers replaced them). Every float maximum, minimum and clamp in this
//! crate goes through this module instead:
//!
//! * [`max`] / [`min`] / [`max64`] / [`min64`] are `alice-det-math`'s: IEEE
//!   maximumNumber / minimumNumber with `-0.0 < +0.0`. A NaN operand yields the other
//!   operand, and two NaNs give the canonical NaN. Both are commutative and
//!   associative, so a reduction in any order (a SIMD tree, a scalar fold) gives the
//!   same bits.
//! * [`clamp`] / [`clamp64`] keep a NaN input as the canonical NaN (what `f32::clamp`
//!   returns for NaN) and take every other input through [`max`] and [`min`].
//! * [`max_of`] / [`min_of`] are every scalar reduction, [`relu`] is `max(x, +0)`,
//!   [`argmax`] is the first Law maximum and [`order`] the total order the Law implies
//!   (for a selection or a sort).
//! * `max_ps` / `min_ps` (`x86_64` with the `simd` feature, AVX2) are the same rule
//!   lane by lane, built from comparisons and blends; this module's tests check them
//!   lane for lane against [`max`] / [`min`] over every pair of special values, and
//!   `tests/law_ties.rs` through the public reductions.
//!
//! `tests/law_ties.rs` asserts the ties of every site class through the public API, and
//! `scripts/law_tie_mutants.py` rewrites each site as a plain comparison and requires a
//! test to fail (`scripts/law_tie_equivalents.tsv` lists the rewrites that are equivalent).
//!
//! The platform forms are banned by `clippy.toml` (disallowed-methods); this module is
//! the only place that implements a maximum.

pub use alice_det_math::{max, max64, min, min64};

/// The canonical NaN a Law site returns for a NaN input (positive, quiet, no payload:
/// `0x7fc0_0000`), written out because `f32::NAN`'s bit pattern is not guaranteed by `std`
const CANONICAL_NAN: f32 = f32::from_bits(0x7fc0_0000);

/// The `f64` canonical NaN (`0x7ff8_0000_0000_0000`), for the same reason
const CANONICAL_NAN64: f64 = f64::from_bits(0x7ff8_0000_0000_0000);

/// `x` limited to `[lo, hi]`: NaN for a NaN `x` (the canonical NaN), otherwise
/// `min(max(x, lo), hi)` with the zero-sign rule of [`max`] / [`min`].
///
/// `lo <= hi` is the caller's contract (as for `f32::clamp`, which panics otherwise;
/// this function does not check it).
#[inline]
#[must_use]
pub fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_nan() {
        CANONICAL_NAN
    } else {
        min(max(x, lo), hi)
    }
}

/// [`clamp`] for `f64`.
#[inline]
#[must_use]
pub fn clamp64(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() {
        CANONICAL_NAN64
    } else {
        min64(max64(x, lo), hi)
    }
}

/// The [`max`] of a slice.
///
/// `f32::NEG_INFINITY` for an empty slice and for one that is all NaN (NaN elements are
/// skipped), and `+0.0` wins over `-0.0` in either order. Every
/// scalar maximum reduction of the crate is this function (`tests/law_ties.rs` checks
/// the ties through each public entry point).
#[inline]
#[must_use]
pub fn max_of(xs: &[f32]) -> f32 {
    xs.iter().copied().fold(f32::NEG_INFINITY, max)
}

/// The [`min`] of a slice: `f32::INFINITY` for an empty slice and for one that is all
/// NaN (NaN elements are skipped), and `-0.0` wins over `+0.0` in either order.
#[inline]
#[must_use]
pub fn min_of(xs: &[f32]) -> f32 {
    xs.iter().copied().fold(f32::INFINITY, min)
}

/// The index of the [`max_of`] of a slice.
///
/// The first element equal to it bit for bit, so `+0.0` is chosen over an earlier
/// `-0.0`; NaN elements are never chosen; `0` for an empty slice or one that is all NaN.
#[inline]
#[must_use]
pub fn argmax(xs: &[f32]) -> usize {
    let m = max_of(xs);
    xs.iter()
        .position(|x| x.to_bits() == m.to_bits())
        .unwrap_or(0)
}

/// The order [`max`] and [`min`] imply, as a total order.
///
/// Every NaN below every number (all NaNs equal), and otherwise `f32::total_cmp`
/// (`-0.0 < +0.0`). A selection or a sort by it gives the same elements as reductions
/// by [`max`] / [`min`]; with a stable sort, equal values keep their input order.
#[inline]
#[must_use]
pub fn order(a: f32, b: f32) -> core::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => core::cmp::Ordering::Equal,
        (true, false) => core::cmp::Ordering::Less,
        (false, true) => core::cmp::Ordering::Greater,
        (false, false) => a.total_cmp(&b),
    }
}

/// `max(x, +0.0)`: `+0.0` for `-0.0`, for `+0.0` and for NaN.
#[inline]
#[must_use]
pub fn relu(x: f32) -> f32 {
    max(x, 0.0)
}

/// Lane-wise [`max`] of two AVX2 vectors.
///
/// # Safety
///
/// The CPU must support AVX2 (the callers check `has_avx2()` first).
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub unsafe fn max_ps(
    a: core::arch::x86_64::__m256,
    b: core::arch::x86_64::__m256,
) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::{
        _mm256_and_ps, _mm256_andnot_ps, _mm256_blendv_ps, _mm256_castps_si256,
        _mm256_castsi256_ps, _mm256_cmp_ps, _mm256_cmpgt_epi32, _mm256_or_ps, _mm256_set1_ps,
        _mm256_setzero_si256, _CMP_GT_OQ, _CMP_UNORD_Q,
    };
    // the scalar rule: r = b if a is NaN or b > a; else a if b is NaN or a > b or a is
    // not negative; else b (equal values with a negative: -0 against +0 gives +0)
    let a_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(a, a);
    let b_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b);
    let b_gt = _mm256_cmp_ps::<_CMP_GT_OQ>(b, a);
    let a_gt = _mm256_cmp_ps::<_CMP_GT_OQ>(a, b);
    // a's sign bit clear: the integer is not below zero
    let a_neg = _mm256_castsi256_ps(_mm256_cmpgt_epi32(
        _mm256_setzero_si256(),
        _mm256_castps_si256(a),
    ));
    let take_b_first = _mm256_or_ps(a_nan, b_gt);
    let take_a = _mm256_or_ps(
        _mm256_or_ps(b_nan, a_gt),
        _mm256_andnot_ps(a_neg, _mm256_set1_ps(f32::from_bits(u32::MAX))),
    );
    let r = _mm256_blendv_ps(_mm256_blendv_ps(b, a, take_a), b, take_b_first);
    canonical_nan(r, _mm256_and_ps(a_nan, b_nan))
}

/// Lane-wise [`min`] of two AVX2 vectors.
///
/// # Safety
///
/// The CPU must support AVX2 (the callers check `has_avx2()` first).
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2")]
#[inline]
#[must_use]
pub unsafe fn min_ps(
    a: core::arch::x86_64::__m256,
    b: core::arch::x86_64::__m256,
) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::{
        _mm256_and_ps, _mm256_blendv_ps, _mm256_castps_si256, _mm256_castsi256_ps, _mm256_cmp_ps,
        _mm256_cmpgt_epi32, _mm256_or_ps, _mm256_setzero_si256, _CMP_LT_OQ, _CMP_UNORD_Q,
    };
    // the scalar rule: r = b if a is NaN or b < a; else a if b is NaN or a < b or a is
    // negative; else b (equal values with a positive: +0 against -0 gives -0)
    let a_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(a, a);
    let b_nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(b, b);
    let b_lt = _mm256_cmp_ps::<_CMP_LT_OQ>(b, a);
    let a_lt = _mm256_cmp_ps::<_CMP_LT_OQ>(a, b);
    let a_neg = _mm256_castsi256_ps(_mm256_cmpgt_epi32(
        _mm256_setzero_si256(),
        _mm256_castps_si256(a),
    ));
    let take_b_first = _mm256_or_ps(a_nan, b_lt);
    let take_a = _mm256_or_ps(_mm256_or_ps(b_nan, a_lt), a_neg);
    let r = _mm256_blendv_ps(_mm256_blendv_ps(b, a, take_a), b, take_b_first);
    canonical_nan(r, _mm256_and_ps(a_nan, b_nan))
}

/// Lanes where `both_nan` is set become the canonical NaN.
#[cfg(all(target_arch = "x86_64", feature = "simd"))]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn canonical_nan(
    r: core::arch::x86_64::__m256,
    both_nan: core::arch::x86_64::__m256,
) -> core::arch::x86_64::__m256 {
    use core::arch::x86_64::{_mm256_blendv_ps, _mm256_set1_ps};
    _mm256_blendv_ps(r, _mm256_set1_ps(f32::NAN), both_nan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    fn specials() -> Vec<f32> {
        [
            0x0000_0000u32,
            0x8000_0000,
            0x7fc0_0000,
            0xffc0_0000,
            0x7fa0_0001,
            0x7f80_0000,
            0xff80_0000,
            0x0000_0001,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
        ]
        .iter()
        .map(|&b| f32::from_bits(b))
        .collect()
    }

    #[test]
    fn clamp_keeps_nan_and_orders_zeros() {
        assert_eq!(
            clamp(f32::from_bits(0x7fa0_0001), -1.0, 1.0).to_bits(),
            f32::NAN.to_bits()
        );
        assert_eq!(clamp(-0.0, 0.0, 1.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(clamp(0.0, -1.0, -0.0).to_bits(), (-0.0f32).to_bits());
        assert_eq!(clamp(5.0, -1.0, 1.0), 1.0);
        assert_eq!(clamp(-5.0, -1.0, 1.0), -1.0);
        assert_eq!(
            clamp64(f64::from_bits(0x7ff4_0000_0000_0001), -1.0, 1.0).to_bits(),
            f64::NAN.to_bits()
        );
        assert_eq!(clamp64(-0.0, 0.0, 1.0).to_bits(), 0.0f64.to_bits());
    }

    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[test]
    fn avx2_lanes_equal_the_scalar_helpers_over_every_pair() {
        use core::arch::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};
        if !crate::ops::has_avx2() {
            // CI's x86_64 runners have AVX2 and set ALICE_ML_REQUIRE_AVX2, so the
            // lanes are checked there; elsewhere (an emulated x86_64 without AVX2)
            // the test says so instead of passing silently
            assert!(
                std::env::var_os("ALICE_ML_REQUIRE_AVX2").is_none(),
                "ALICE_ML_REQUIRE_AVX2 is set but this CPU has no AVX2: the lane test did not run"
            );
            eprintln!("law_math: no AVX2 on this CPU, the AVX2 lane test did not run");
            return;
        }
        let s = specials();
        let mut checked = 0;
        for &a in &s {
            for chunk in s.chunks(8) {
                let mut bs = [0.0f32; 8];
                bs[..chunk.len()].copy_from_slice(chunk);
                let av = [a; 8];
                let (mut mx, mut mn) = ([0.0f32; 8], [0.0f32; 8]);
                // SAFETY: AVX2 checked above; 8-element arrays
                unsafe {
                    let va = _mm256_loadu_ps(av.as_ptr());
                    let vb = _mm256_loadu_ps(bs.as_ptr());
                    _mm256_storeu_ps(mx.as_mut_ptr(), max_ps(va, vb));
                    _mm256_storeu_ps(mn.as_mut_ptr(), min_ps(va, vb));
                }
                for i in 0..chunk.len() {
                    assert_eq!(
                        mx[i].to_bits(),
                        max(a, bs[i]).to_bits(),
                        "max({a:?}, {:?})",
                        bs[i]
                    );
                    assert_eq!(
                        mn[i].to_bits(),
                        min(a, bs[i]).to_bits(),
                        "min({a:?}, {:?})",
                        bs[i]
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, s.len() * s.len());
        println!("law_math: AVX2 lanes compared: {checked} pairs for max and {checked} for min");
    }
}
