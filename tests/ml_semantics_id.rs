//! `ML_SEMANTICS_ID` (ALICE-Law G6) against its sources: the entries are exactly the
//! goldens of `tests/determinism_golden.rs` (read from the source, not a copy) plus
//! `alice-det-math`, and every row of `ML_SEMANTICS_BY_DET_MATH` is the fold of the pins
//! with that row's det-math identifier.

use alice_ml::semantics::{
    semantics_id_for, ML_SEMANTICS_BY_DET_MATH, ML_SEMANTICS_ID, ML_SEMANTICS_PINS,
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
fn entries_are_the_goldens_plus_det_math() {
    let goldens = goldens_from_source();
    let mut pins: BTreeMap<String, String> = BTreeMap::new();
    for (name, digest) in ML_SEMANTICS_PINS {
        assert!(
            pins.insert((*name).to_owned(), hex(digest)).is_none(),
            "{name} twice"
        );
    }
    assert_eq!(
        pins.remove("alice-det-math").as_deref(),
        Some(hex(&alice_det_math::SEMANTICS_ID).as_str()),
        "the det-math entry is the resolved alice-det-math's identifier"
    );
    assert_eq!(
        pins, goldens,
        "ML_SEMANTICS_PINS against tests/determinism_golden.rs"
    );
    assert_eq!(goldens.len(), 9);
}

#[test]
fn every_row_is_the_fold_of_the_pins() {
    assert!(!ML_SEMANTICS_BY_DET_MATH.is_empty());
    for (det_math, id) in ML_SEMANTICS_BY_DET_MATH {
        let entries: Vec<(&str, [u8; 32])> = ML_SEMANTICS_PINS
            .iter()
            .map(|&(n, d)| {
                if n == "alice-det-math" {
                    (n, *det_math)
                } else {
                    (n, d)
                }
            })
            .collect();
        assert_eq!(
            hex(&fold(&entries)),
            hex(id),
            "row for det-math {}",
            hex(det_math)
        );
    }
    assert_eq!(
        ML_SEMANTICS_ID,
        semantics_id_for(&alice_det_math::SEMANTICS_ID)
    );
    // an earlier identifier is never the current one
    assert!(!PREVIOUS_ML_SEMANTICS_IDS.contains(&ML_SEMANTICS_ID));
    assert_eq!(hex(&ML_SEMANTICS_ID), hex(&fold(ML_SEMANTICS_PINS)));
}

#[test]
fn the_documented_hex_is_the_identifier() {
    assert_eq!(
        hex(&ML_SEMANTICS_ID),
        // the fold recomputed outside this toolchain from the documented procedure, which
        // also reproduces the previous a5472370… with alice-det-math 0.5.1's identifier
        "8464fe594013020b37c94d790f15167244074f3892a99d3c9030bd66ba4ed982"
    );
}

/// The identifier with `alice-det-math` 0.5.1 (`85e08846…`) is the fold of this build's pins
/// (every golden unchanged) with that det-math identifier, and it is kept as a previous one
#[test]
fn the_previous_identifier_is_the_fold_under_the_previous_det_math_identifier() {
    let det_math_0_5_1 = ML_SEMANTICS_BY_DET_MATH
        .iter()
        .map(|(d, _)| d)
        .find(|d| {
            hex(d.as_slice()) == "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"
        })
        .expect("a row for alice-det-math 0.5.1");
    let entries: Vec<(&str, [u8; 32])> = ML_SEMANTICS_PINS
        .iter()
        .map(|&(n, d)| {
            if n == "alice-det-math" {
                (n, *det_math_0_5_1)
            } else {
                (n, d)
            }
        })
        .collect();
    let previous = fold(&entries);
    assert_eq!(
        hex(&previous),
        "a547237044e8fdba9ab152f7af13c431f675e5382e4e8cb40aa8c83c46bb606b"
    );
    assert!(
        PREVIOUS_ML_SEMANTICS_IDS.contains(&previous),
        "the identifier with alice-det-math 0.5.1 is not kept as a previous one"
    );
}

#[test]
fn the_fold_sees_every_entry_and_its_order_does_not_matter() {
    let base = fold(ML_SEMANTICS_PINS);
    // a moved golden moves the identifier
    for i in 0..ML_SEMANTICS_PINS.len() {
        let mut moved: Vec<_> = ML_SEMANTICS_PINS.to_vec();
        moved[i].1[0] ^= 1;
        assert_ne!(fold(&moved), base, "entry {} is not folded", moved[i].0);
    }
    // a dropped entry moves it
    assert_ne!(fold(&ML_SEMANTICS_PINS[1..]), base);
    // the order the table is written in does not
    let mut reversed: Vec<_> = ML_SEMANTICS_PINS.to_vec();
    reversed.reverse();
    assert_eq!(fold(&reversed), base);
}
