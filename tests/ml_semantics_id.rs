//! `ML_SEMANTICS_ID` (ALICE-Law G6) against its sources: the entries are exactly the
//! goldens of `tests/determinism_golden.rs` (read from the source, not a copy), each
//! `DET_MATH_PINS` entry is the resolved `alice-det-math`'s, and the identifier is the fold
//! of both (the alice-det-math pins under `det-math <function>`).

use alice_ml::semantics::{
    DET_MATH_FUNCTIONS, DET_MATH_PINS, ML_SEMANTICS_ID, ML_SEMANTICS_PINS,
    PREVIOUS_ML_SEMANTICS_IDS,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn hex(b: &[u8]) -> String {
    b.iter().fold(String::new(), |mut s, x| {
        use std::fmt::Write as _;
        let _ = write!(s, "{x:02x}");
        s
    })
}

/// The fold of the module documentation: sorted by name, `u32` big-endian name length,
/// name bytes, 32 digest bytes.
fn fold(entries: &[(&str, [u8; 32])]) -> [u8; 32] {
    assert!(!entries.is_empty(), "an empty table");
    let mut sorted: Vec<_> = entries.to_vec();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    for w in sorted.windows(2) {
        assert_ne!(w[0].0, w[1].0, "a repeated name");
    }
    let mut h = Sha256::new();
    for (name, digest) in &sorted {
        assert!(!name.is_empty() && name.is_ascii(), "name {name:?}");
        h.update(
            u32::try_from(name.len())
                .expect("a short name")
                .to_be_bytes(),
        );
        h.update(name.as_bytes());
        h.update(digest);
    }
    h.finalize().into()
}

/// Every entry the identifier folds: [`ML_SEMANTICS_PINS`] and, under `det-math <function>`,
/// [`DET_MATH_PINS`]
fn all_entries() -> Vec<(String, [u8; 32])> {
    ML_SEMANTICS_PINS
        .iter()
        .map(|(n, d)| ((*n).to_owned(), *d))
        .chain(
            DET_MATH_PINS
                .iter()
                .map(|(n, d)| (format!("det-math {n}"), *d)),
        )
        .collect()
}

fn fold_all(entries: &[(String, [u8; 32])]) -> [u8; 32] {
    let borrowed: Vec<(&str, [u8; 32])> = entries.iter().map(|(n, d)| (n.as_str(), *d)).collect();
    fold(&borrowed)
}

/// `(label, digest)` of every golden, read from the source of the golden tests.
fn goldens_from_source() -> BTreeMap<String, String> {
    let src = include_str!("determinism_golden.rs");
    let mut consts = BTreeMap::new();
    for chunk in src.split("const GOLDEN_").skip(1) {
        let name: String = chunk
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let digest: String = chunk.split('"').nth(1).expect("a quoted digest").to_owned();
        assert_eq!(digest.len(), 64, "GOLDEN_{name}");
        consts.insert(format!("GOLDEN_{name}"), digest);
    }
    let mut out = BTreeMap::new();
    for chunk in src.split("g.finish(\"").skip(1) {
        let label: String = chunk.chars().take_while(|&c| c != '"').collect();
        let rest = &chunk[label.len() + 3..];
        let cname: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let digest = consts
            .get(&cname)
            .unwrap_or_else(|| panic!("{cname} not recorded"));
        assert!(
            out.insert(label, digest.clone()).is_none(),
            "a golden finishing twice"
        );
    }
    assert!(!out.is_empty(), "no golden read (compared nothing)");
    out
}

#[test]
fn entries_are_the_goldens() {
    let goldens = goldens_from_source();
    let mut pins: BTreeMap<String, String> = BTreeMap::new();
    for (name, digest) in ML_SEMANTICS_PINS {
        assert!(
            pins.insert((*name).to_owned(), hex(digest)).is_none(),
            "{name} twice"
        );
    }
    assert!(
        !pins.contains_key("alice-det-math"),
        "alice-det-math is folded per function (det-math <function>), not as a whole"
    );
    assert_eq!(
        pins, goldens,
        "ML_SEMANTICS_PINS against tests/determinism_golden.rs"
    );
    assert_eq!(goldens.len(), 11);
}

/// Each alice-det-math function this crate calls has one pin, in the list's order, and it is
/// the resolved alice-det-math's own identifier of that function
#[test]
fn the_det_math_pins_are_the_resolved_function_identifiers() {
    let names: Vec<&str> = DET_MATH_PINS.iter().map(|(n, _)| *n).collect();
    assert_eq!(names, DET_MATH_FUNCTIONS);
    assert!(!DET_MATH_PINS.is_empty(), "compared nothing");
    for (name, pin) in DET_MATH_PINS {
        assert_eq!(*pin, alice_det_math::function_pin(name), "{name}");
    }
}

#[test]
fn the_identifier_is_the_fold_of_the_goldens_and_the_det_math_pins() {
    assert_eq!(hex(&fold_all(&all_entries())), hex(&ML_SEMANTICS_ID));
    // an earlier identifier is never the current one
    assert!(!PREVIOUS_ML_SEMANTICS_IDS.contains(&ML_SEMANTICS_ID));
}

/// The previous identifier `14e0f043…` is these goldens with alice-det-math 0.5.2 at 2d01c63f
/// folded as a whole (`a2a11b9b…`): the goldens did not move when the functions were folded
/// one by one, so data stamped with it reads back as these bits
#[test]
fn the_previous_identifier_is_these_goldens_with_the_whole_det_math_identifier() {
    let mut entries: Vec<(&str, [u8; 32])> = ML_SEMANTICS_PINS.to_vec();
    let det = "a2a11b9b826227fa7a9ed8ab2c1ce63dbb07ad4c39aa2c46a363ddea6b47dbe9";
    entries.push((
        "alice-det-math",
        core::array::from_fn(|i| u8::from_str_radix(&det[2 * i..2 * i + 2], 16).expect("hex")),
    ));
    let previous = fold(&entries);
    assert_eq!(
        hex(&previous),
        "14e0f0435562e1fa77ab0ce071c85c422fd3f64122aaaf2e73d110264b5356b6"
    );
    assert_eq!(PREVIOUS_ML_SEMANTICS_IDS, [previous]);
}

#[test]
fn the_documented_hex_is_the_identifier() {
    assert_eq!(
        hex(&ML_SEMANTICS_ID),
        // the fold recomputed outside this toolchain from the documented procedure, which
        // also reproduces 8464fe59… from the frozen table below and 14e0f043… from the whole
        // det-math identifier
        "2c7e07c50c30b70d48c2540189e2431c7754c13c01b92056b7ca18075644dc27"
    );
}

/// The pins of the identifier with `alice-det-math` 0.5.2 at 39507d1b (`8464fe59…`), frozen as
/// they were: the nine goldens of that build and its det-math entry
const PINS_8464FE59: &[(&str, &str)] = &[
    (
        "relu",
        "eb212a7bfd02a74cd9c30587ddd268e2d63610d92de8eb2364cf99d166fd43f3",
    ),
    (
        "max_min",
        "8a2bc26d6793bb71db4c654834f5e99b1a4c8d621dc338406390f7ccea7aed0b",
    ),
    (
        "softmax",
        "c316a92a504431af0544bd90cd4a0a533daaef9ddbb1844d24434f00b672f68b",
    ),
    (
        "quantize",
        "7bd392c8df625897f748e81f6e55c7913ce730c48c987296da923a0d3ab66467",
    ),
    (
        "int8",
        "c1cf22da2aa5fbd92ca976133addf5025554ff1ae0f1ddc512bd554f68848d7a",
    ),
    (
        "cross_entropy",
        "9f8a614a5ed5fd15708cbf64979538246d7a07e7ce7dcc8e3a1eef48edcf3942",
    ),
    (
        "calibration",
        "390365518b328c524fc18fff90190baac69856a975f144c3e52f4d66d6dd6f0d",
    ),
    (
        "error_propagation",
        "dbd40bac666f357893a8afffd7a69af8b06eb7abcf397bf2fba3a383148685ff",
    ),
    (
        "ffi",
        "738bd4170174a7112ce57923999bb3229da5e325f902601fc72a862ff9235ce0",
    ),
    (
        "alice-det-math",
        "a2e46794a49b62730d8eb75d10e6dbfdd59d6562a985ae94c716fc637a57e3cf",
    ),
];

/// The `RoPE` windows under `alice-det-math` 0.5.2 at 39507d1b, recorded from that build with the
/// goldens of this file's `rope_below_and_from_2_13` (`tests/determinism_golden.rs`)
const ROPE_BELOW_2_13_AT_39507D1B: &str =
    "e6097785b5a566fac3b64b8360242bf99ffd66f1d460d9c6f94112b9327bbefe";
const ROPE_FROM_2_13_AT_39507D1B: &str =
    "35e0720cf3b999cf7bddee7c7cae115c3e8e2eb54bac4ffed12c18e105e9b061";

fn fold_hex(entries: &[(&str, &str)]) -> [u8; 32] {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
    };
    let owned: Vec<(&str, [u8; 32])> = entries.iter().map(|&(n, d)| (n, unhex(d))).collect();
    fold(&owned)
}

/// `8464fe59…` (alice-det-math 0.5.2 at 39507d1b) is no longer an identifier of this build, and
/// the reason is exactly this generation's changes, each visible in a pin: `error_propagation`
/// moved (`powf64` is fdlibm `e_pow`), and of the two `RoPE` windows added with it the one below
/// position `2^13` has the bits it had under 39507d1b while the one from `2^13` on moved
/// (`sin_cos` reduces those arguments in double precision). Every other pin is unchanged; a
/// further change to any kernel moves this set and fails here. `a5472370…` (0.5.1) relied on
/// the same goldens, so neither is in `PREVIOUS_ML_SEMANTICS_IDS` (which holds `14e0f043…`).
#[test]
fn the_8464fe59_identifier_is_rejected_because_exactly_these_pins_moved() {
    assert_eq!(
        hex(&fold_hex(PINS_8464FE59)),
        "8464fe594013020b37c94d790f15167244074f3892a99d3c9030bd66ba4ed982",
        "the frozen table does not fold to its identifier"
    );
    let now: BTreeMap<String, String> = ML_SEMANTICS_PINS
        .iter()
        .map(|(n, d)| ((*n).to_owned(), hex(d)))
        .collect();
    let mut moved = Vec::new();
    let mut compared = 0usize;
    for (name, digest) in PINS_8464FE59 {
        if *name == "alice-det-math" {
            continue;
        }
        compared += 1;
        if now.get(*name).map(String::as_str) != Some(*digest) {
            moved.push(*name);
        }
    }
    assert_eq!(compared, 9, "pins compared");
    assert_eq!(moved, ["error_propagation"]);
    assert_eq!(now["rope_below_2_13"], ROPE_BELOW_2_13_AT_39507D1B);
    assert_ne!(now["rope_from_2_13"], ROPE_FROM_2_13_AT_39507D1B);
    for id in [
        "8464fe594013020b37c94d790f15167244074f3892a99d3c9030bd66ba4ed982",
        "a547237044e8fdba9ab152f7af13c431f675e5382e4e8cb40aa8c83c46bb606b",
    ] {
        assert!(
            !PREVIOUS_ML_SEMANTICS_IDS.iter().any(|p| hex(p) == id),
            "{id} is kept as a previous identifier"
        );
    }
}

#[test]
fn the_fold_sees_every_entry_and_its_order_does_not_matter() {
    let all = all_entries();
    let base = fold_all(&all);
    // a moved golden or det-math pin moves the identifier
    for i in 0..all.len() {
        let mut moved = all.clone();
        moved[i].1[0] ^= 1;
        assert_ne!(fold_all(&moved), base, "entry {} is not folded", moved[i].0);
    }
    // a dropped entry moves it
    assert_ne!(fold_all(&all[1..]), base);
    // the order the table is written in does not
    let mut reversed = all;
    reversed.reverse();
    assert_eq!(fold_all(&reversed), base);
}
