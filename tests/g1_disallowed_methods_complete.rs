//! Guards `clippy.toml` against silently losing an entry from ALICE-Law's
//! G1 ("no-libm-direct") gate's "banned" method list.
//!
//! The list below is a snapshot of `conformance/gates.toml`'s G1 `methods`
//! entries with `verdict = "banned"` (ALICE-Law repo, re-checked by hand on
//! 2026-10-15). If that list changes, update it here too -- this test
//! can't read another repo's checkout, so it is a pinned copy, not a live
//! query.
const G1_BANNED: &[&str] = &[
    "sin", "cos", "tan", "sin_cos", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh", "exp", "exp2", "exp_m1", "ln", "log", "log2", "log10", "ln_1p",
    "powf", "cbrt", "hypot",
];

#[test]
fn clippy_toml_bans_every_g1_method_for_both_widths() {
    let clippy_toml = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/clippy.toml"))
        .expect("clippy.toml must exist at the crate root");

    let mut missing = Vec::new();
    for method in G1_BANNED {
        for width in ["f32", "f64"] {
            let path = format!("\"{width}::{method}\"");
            if !clippy_toml.contains(&path) {
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
