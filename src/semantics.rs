//! Content hash of the crate's float semantics: [`ML_SEMANTICS_ID`] (ALICE-Law G6).
//!
//! The determinism goldens of `tests/determinism_golden.rs` pin the output bits of the
//! crate's float kernels over a lattice of special values, one SHA-256 digest per
//! golden. This module folds those digests, plus the semantics identifier of
//! `alice-det-math` (the transcendental functions and the maximum / minimum the kernels
//! call), into one 32-byte value. Two builds with the same identifier give the same bits
//! on every pinned kernel; a build that changes any of them has a different value.
//!
//! # Hex form (with `alice-det-math` 0.5.2 at 39507d1b; other generations in [`ML_SEMANTICS_BY_DET_MATH`])
//!
//! ```text
//! 8464fe594013020b37c94d790f15167244074f3892a99d3c9030bd66ba4ed982
//! ```
//!
//! # Entries
//!
//! [`ML_SEMANTICS_PINS`] has one entry per golden of `tests/determinism_golden.rs`,
//! named by the label the golden hashes first (`relu`, `max_min`, `softmax`,
//! `quantize`, `int8`, `cross_entropy`, `calibration`, `error_propagation`, `ffi`),
//! with the digest the golden records, plus one entry `"alice-det-math"` holding
//! `alice_det_math::SEMANTICS_ID`. The crate version is not an entry: a release that
//! moves no golden keeps the identifier.
//!
//! A kernel no golden pins is not covered: **the identifier changes when a golden is
//! added or re-recorded**, and it does not notice a change on a path that has no
//! golden.
//!
//! # Fold
//!
//! The procedure of the `alice-det-math` `SEMANTICS_ID` fold: sort the entries by
//! name (ascending byte order), reject an empty table, a repeated name, an empty name or
//! a name that is not ASCII, then SHA-256 over, for each entry in that order, the name
//! length as 4 bytes big-endian (`u32`), the name bytes and the 32 digest bytes. The
//! length prefix keeps two different tables from producing the same byte stream; the
//! sort makes the value independent of the order the table is written in.
//!
//! # Independent of cargo features
//!
//! The table and the identifier are constants no `cfg` gates. The `ffi` golden runs
//! only with the `ffi` feature, and its entry is always present, so every feature set
//! reports the same identifier.
//!
//! # Checked by
//!
//! `tests/ml_semantics_id.rs` reads every digest from `tests/determinism_golden.rs`
//! (the source, not a copy), checks that the entry names are exactly the goldens plus
//! `"alice-det-math"`, that the det-math entry is `alice_det_math::SEMANTICS_ID`, and
//! recomputes every row of [`ML_SEMANTICS_BY_DET_MATH`]. Re-recording order after an
//! intended change of a kernel: the golden first, then its entry here, then the rows.

/// Decodes 64 hex digits into 32 bytes at compile time; anything else is a compile
/// error.
const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "a digest is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// The `(name, digest)` entries [`ML_SEMANTICS_ID`] folds: one per golden of
/// `tests/determinism_golden.rs`, plus `"alice-det-math"`.
pub const ML_SEMANTICS_PINS: &[(&str, [u8; 32])] = &[
    (
        "relu",
        hex32("eb212a7bfd02a74cd9c30587ddd268e2d63610d92de8eb2364cf99d166fd43f3"),
    ),
    (
        "max_min",
        hex32("8a2bc26d6793bb71db4c654834f5e99b1a4c8d621dc338406390f7ccea7aed0b"),
    ),
    (
        "softmax",
        hex32("c316a92a504431af0544bd90cd4a0a533daaef9ddbb1844d24434f00b672f68b"),
    ),
    (
        "quantize",
        hex32("7bd392c8df625897f748e81f6e55c7913ce730c48c987296da923a0d3ab66467"),
    ),
    (
        "int8",
        hex32("c1cf22da2aa5fbd92ca976133addf5025554ff1ae0f1ddc512bd554f68848d7a"),
    ),
    (
        "cross_entropy",
        hex32("9f8a614a5ed5fd15708cbf64979538246d7a07e7ce7dcc8e3a1eef48edcf3942"),
    ),
    (
        "calibration",
        hex32("390365518b328c524fc18fff90190baac69856a975f144c3e52f4d66d6dd6f0d"),
    ),
    (
        "error_propagation",
        hex32("dbd40bac666f357893a8afffd7a69af8b06eb7abcf397bf2fba3a383148685ff"),
    ),
    (
        "ffi",
        hex32("738bd4170174a7112ce57923999bb3229da5e325f902601fc72a862ff9235ce0"),
    ),
    // the resolved alice-det-math's own identifier, not a copy: the identifier below is
    // looked up by it
    ("alice-det-math", alice_det_math::SEMANTICS_ID),
];

/// [`ML_SEMANTICS_ID`] for each `alice-det-math` generation this crate knows:
/// `(alice_det_math::SEMANTICS_ID, fold of ML_SEMANTICS_PINS with that det-math entry)`.
///
/// One row per `alice-det-math` generation this crate has been built with. A later
/// `alice-det-math` release that moves `SEMANTICS_ID` stops the compile in
/// [`semantics_id_for`] until its row is added, rather than reporting an identifier of
/// another generation.
pub const ML_SEMANTICS_BY_DET_MATH: &[([u8; 32], [u8; 32])] = &[
    // alice-det-math 0.5.1
    (
        hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
        hex32("a547237044e8fdba9ab152f7af13c431f675e5382e4e8cb40aa8c83c46bb606b"),
    ),
    // alice-det-math 0.5.2 at 39507d1b (its clamp follows the G2 order)
    (
        hex32("a2e46794a49b62730d8eb75d10e6dbfdd59d6562a985ae94c716fc637a57e3cf"),
        hex32("8464fe594013020b37c94d790f15167244074f3892a99d3c9030bd66ba4ed982"),
    ),
];

/// Content hash of the crate's float semantics: the fold of [`ML_SEMANTICS_PINS`]
/// described in the [module documentation](self), taken from
/// [`ML_SEMANTICS_BY_DET_MATH`] by the resolved `alice-det-math`.
///
/// Hex (with `alice-det-math` 0.5.2 at 39507d1b):
/// `8464fe594013020b37c94d790f15167244074f3892a99d3c9030bd66ba4ed982`.
pub const ML_SEMANTICS_ID: [u8; 32] = semantics_id_for(&alice_det_math::SEMANTICS_ID);

/// Earlier values of [`ML_SEMANTICS_ID`] under which every pinned kernel gives the same
/// bits as this build, newest first
///
/// `a5472370…` is the identifier with `alice-det-math` 0.5.1: the move to 0.5.2 at
/// 39507d1b (whose clamp follows the G2 order) left every golden of
/// `tests/determinism_golden.rs` unchanged, so only the det-math entry of the fold moved.
pub const PREVIOUS_ML_SEMANTICS_IDS: &[[u8; 32]] = &[hex32(
    "a547237044e8fdba9ab152f7af13c431f675e5382e4e8cb40aa8c83c46bb606b",
)];

/// The [`ML_SEMANTICS_ID`] of a build that resolved the `alice-det-math` with identifier
/// `det_math`.
///
/// # Panics
///
/// When [`ML_SEMANTICS_BY_DET_MATH`] has no row for `det_math` (in a const context this
/// stops the compile).
#[must_use]
pub const fn semantics_id_for(det_math: &[u8; 32]) -> [u8; 32] {
    let mut i = 0;
    while i < ML_SEMANTICS_BY_DET_MATH.len() {
        if same32(&ML_SEMANTICS_BY_DET_MATH[i].0, det_math) {
            return ML_SEMANTICS_BY_DET_MATH[i].1;
        }
        i += 1;
    }
    panic!("alice-det-math SEMANTICS_ID has no row in ML_SEMANTICS_BY_DET_MATH: add its fold");
}

const fn same32(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 0;
    while i < 32 {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
