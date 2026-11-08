//! Content hash of the crate's float semantics: [`ML_SEMANTICS_ID`] (ALICE-Law G6).
//!
//! The determinism goldens of `tests/determinism_golden.rs` pin the output bits of the
//! crate's float kernels over a lattice of special values, one SHA-256 digest per
//! golden. This module folds those digests, plus the identifier of each `alice-det-math`
//! function the kernels call ([`DET_MATH_PINS`]), into one 32-byte value. Two builds with the same identifier give the same bits
//! on every pinned kernel; a build that changes any of them has a different value.
//!
//! # Hex form
//!
//! ```text
//! 2c7e07c50c30b70d48c2540189e2431c7754c13c01b92056b7ca18075644dc27
//! ```
//!
//! # Entries
//!
//! [`ML_SEMANTICS_PINS`] has one entry per golden of `tests/determinism_golden.rs`,
//! named by the label the golden hashes first (`relu`, `max_min`, `softmax`,
//! `quantize`, `int8`, `cross_entropy`, `calibration`, `error_propagation`, `ffi`),
//! with the digest the golden records, plus one entry per `alice-det-math` function the
//! crate calls ([`DET_MATH_FUNCTIONS`]), named `det-math <function>`, holding that
//! function's identifier ([`DET_MATH_PINS`], compared with `alice_det_math::function_pin`
//! at compile time). A function of `alice-det-math` the crate does not call moves nothing.
//! The crate version is not an entry: a release that moves no golden keeps the identifier.
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
//! (the source, not a copy), checks that the entry names are exactly the goldens, that
//! each [`DET_MATH_PINS`] entry is the resolved `alice-det-math`'s, and recomputes the
//! fold. Re-recording order after an intended change of a kernel: the golden first, then
//! its entry here, then the identifier.

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
/// `tests/determinism_golden.rs`.
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
        hex32("cef6d92869c8990d068fa59673af0b272a59c35868f784533bd4951a91eaf52e"),
    ),
    (
        "ffi",
        hex32("738bd4170174a7112ce57923999bb3229da5e325f902601fc72a862ff9235ce0"),
    ),
    (
        "rope_below_2_13",
        hex32("e6097785b5a566fac3b64b8360242bf99ffd66f1d460d9c6f94112b9327bbefe"),
    ),
    (
        "rope_from_2_13",
        hex32("4631d545e20dc6d6c4eeceb8c506ccef10400e44280e79cff33a0caef4854ad1"),
    ),
];

/// The `alice-det-math` functions this crate calls.
///
/// `ALICE-Law`'s `scripts/fold_set_check.py` compares the list with the calls in `src/`.
pub const DET_MATH_FUNCTIONS: &[&str] = &[
    "exp", "exp64", "ln", "ln64", "log10", "log2", "max", "max64", "min", "min64", "powf",
    "powf64", "powi", "sin_cos", "sqrt64",
];

/// The identifier of each function of [`DET_MATH_FUNCTIONS`], in that order.
///
/// As recorded when [`ML_SEMANTICS_ID`] was folded. The `const _` below compares each with
/// `alice_det_math::function_pin` at compile time, so a resolved `alice-det-math` whose
/// function computes other bits stops the compile instead of reporting this identifier.
pub const DET_MATH_PINS: &[(&str, [u8; 32])] = &[
    (
        "exp",
        hex32("606918f2d637f46f9a792a29b3ee31a991ef3196064456caa7b3a7ca81700a63"),
    ),
    (
        "exp64",
        hex32("f87502b37c757ae5ea59a3cc991d02c3fe19ff3c75c9edd294216055372bf2a8"),
    ),
    (
        "ln",
        hex32("6fe18bc0fba8b7e3e871c4cbe555ddb40a460be7b3c7690303ee30accdcb7291"),
    ),
    (
        "ln64",
        hex32("51c11488d360bb972761d2cecf192c173249f846f8127ef950e580b99a590cdc"),
    ),
    (
        "log10",
        hex32("0b2c5480b6b590cac0b897ba0fcd2295abc4ed53ec2f55d8c47265f9d97a786e"),
    ),
    (
        "log2",
        hex32("ce762397a3dcf55dd2ef35dfe1984aa7a173d9a2c944895914b0697b19864056"),
    ),
    (
        "max",
        hex32("8bd32fcb313f7fb3d7a699fc3075273de7e5a4e083912d8dc7f4cc4016fc2286"),
    ),
    (
        "max64",
        hex32("aac586eacc9acd69cf5988ffe736283c71fad8146027e86034d3e9b881db28b6"),
    ),
    (
        "min",
        hex32("2939e83813b26c868ec3113fb1c25102f16f28d8f24c343a727d4b1bec5904fe"),
    ),
    (
        "min64",
        hex32("1bef418073152f5e1e76f41718aa29b2517de8966a8c8ea8c1c2b5ba92fab3cb"),
    ),
    (
        "powf",
        hex32("0f1f85adcfa801b4afbbf858cf4c30f4ed0abdd3b7163d71aa9aa2b6031d9117"),
    ),
    (
        "powf64",
        hex32("b3b442523e6ddfcc344c5923864caa7d342d0fe0757f0ab1371149cd41a6ad2e"),
    ),
    (
        "powi",
        hex32("d3d76ba59208f7b0151ccde9b63043f538bc6b7737d1f16ff02d918f5806be45"),
    ),
    (
        "sin_cos",
        hex32("025acd2cdc900eff683e39ef1e71de115dafe8da8a1a4b02847b41d75cbd130f"),
    ),
    (
        "sqrt64",
        hex32("051b24b99798891af42895e889ecb5add515496fc771ba5637a830f8f645988d"),
    ),
];

const _: () = alice_det_math::assert_function_pins(DET_MATH_PINS);

/// Content hash of the crate's float semantics: the fold of [`ML_SEMANTICS_PINS`] and, under
/// `det-math <function>`, [`DET_MATH_PINS`], described in the [module documentation](self).
pub const ML_SEMANTICS_ID: [u8; 32] =
    hex32("2c7e07c50c30b70d48c2540189e2431c7754c13c01b92056b7ca18075644dc27");

/// Earlier values of [`ML_SEMANTICS_ID`] under which every pinned kernel gives the same
/// bits as this build, newest first
///
/// `14e0f043…` is the identifier with `alice-det-math` 0.5.2 at 2d01c63f folded as a whole: no
/// golden moved when the functions were folded one by one, so data stamped with it reads back
/// as the same bits. `8464fe59…` (0.5.2 at 39507d1b) and `a5472370…` (0.5.1) do not name these
/// bits (`error_propagation` and the `RoPE` path moved with 2d01c63f); data stamped with them
/// reads as a mismatch. `tests/ml_semantics_id.rs` shows which pins moved.
pub const PREVIOUS_ML_SEMANTICS_IDS: &[[u8; 32]] = &[hex32(
    "14e0f0435562e1fa77ab0ce071c85c422fd3f64122aaaf2e73d110264b5356b6",
)];
