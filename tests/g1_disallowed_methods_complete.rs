//! Guards `clippy.toml` against silently losing an entry from ALICE-Law's
//! G1 ("no-libm-direct") gate's "banned" method list.
//!
//! The list below is a snapshot of `conformance/gates.toml`'s G1 `methods`
//! entries with `verdict = "banned"` (ALICE-Law repo, re-checked by hand on
//! 2026-10-15, at the commit this crate's `clippy.toml` doc comment names).
//!
//! Known limitation, by design rather than an oversight: this test can only
//! confirm `clippy.toml` still matches the snapshot below, never that the
//! snapshot itself still matches ALICE-Law's live G1 list -- that would
//! need `cargo test` to reach another repo's checkout, which it can't do
//! here (this crate has no sibling-checkout CI wiring, unlike the Law-
//! dependent ALICE-* crates that do). If G1's banned list grows upstream,
//! re-reading `conformance/gates.toml` by hand and updating `G1_BANNED`
//! (and the length assertion below) is still a manual step.
const G1_BANNED: &[&str] = &[
    "sin", "cos", "tan", "sin_cos", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "exp", "exp2", "exp_m1", "ln", "log", "log2", "log10", "ln_1p",
    "powf", "cbrt", "hypot",
];

#[test]
fn clippy_toml_bans_every_g1_method_for_both_widths() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/clippy.toml");
    let text = std::fs::read_to_string(path).expect("clippy.toml must exist at the crate root");
    let parsed: toml::Value = text.parse().expect("clippy.toml must be valid TOML");

    let entries = parsed
        .get("disallowed-methods")
        .and_then(toml::Value::as_array)
        .expect("clippy.toml must have a disallowed-methods array");

    // Real TOML parsing (not a text search): a commented-out entry is not
    // in `entries` at all, so it can't produce a false "present".
    let banned_paths: std::collections::HashSet<&str> = entries
        .iter()
        .filter_map(|entry| entry.get("path").and_then(toml::Value::as_str))
        .collect();

    let mut missing = Vec::new();
    for method in G1_BANNED {
        for width in ["f32", "f64"] {
            let path = format!("{width}::{method}");
            if !banned_paths.contains(path.as_str()) {
                missing.push(path);
            }
        }
    }

    assert!(
        missing.is_empty(),
        "clippy.toml's disallowed-methods is missing {} G1-banned entr{}: {:?}",
        missing.len(),
        if missing.len() == 1 { "y" } else { "ies" },
        missing,
    );
}

#[test]
fn g1_banned_list_is_not_itself_empty() {
    // Catches someone accidentally clearing G1_BANNED above, which would
    // make the first test vacuously pass with 0 comparisons.
    assert_eq!(G1_BANNED.len(), 25, "expected exactly 25 G1-banned methods");
}

#[test]
fn clippy_toml_parses_as_an_array_of_inline_tables_not_a_flat_list() {
    // A mutant that replaces the array-of-tables with a flat array of
    // strings (or otherwise drops the `path` key) would make every
    // `entry.get("path")` above return `None`, silently shrinking
    // `banned_paths` to empty -- which the completeness test above would
    // (correctly) still fail on, but for the wrong, confusing reason. This
    // pins the shape independently so a `cargo test` failure here points
    // directly at "the array's shape changed," not "methods went missing."
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/clippy.toml");
    let text = std::fs::read_to_string(path).unwrap();
    let parsed: toml::Value = text.parse().unwrap();
    let entries = parsed["disallowed-methods"].as_array().unwrap();
    assert!(!entries.is_empty());
    for entry in entries {
        assert!(
            entry.get("path").and_then(toml::Value::as_str).is_some(),
            "a disallowed-methods entry has no string `path` key: {entry:?}"
        );
    }
}
