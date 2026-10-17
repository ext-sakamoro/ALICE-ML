//! ALICE-Law G7: the crate's type-erasing boundaries, listed and classified.
//!
//! G7 asks every boundary that erases the Rust type and receives another Law-bearing
//! crate's value to carry and compare the producer's identifier. This test scans `src/`
//! (and the crate root for `build.rs` / a procedural-macro target) for every boundary
//! form the gate names and requires the counts below, each with the reason it receives no
//! other Law-bearing crate's value. A new boundary changes a count and fails here until it
//! is classified (or carries the identifier); a scan that reads no source fails too.

use std::path::{Path, PathBuf};

/// (boundary form, the substring it is found by, how many there are, why none of them
/// receives another Law-bearing crate's value)
const BOUNDARIES: &[(&str, &str, usize, &str)] = &[
    (
        "C ABI export",
        "extern \"C\" fn",
        67,
        "src/ffi.rs: raw f32 slices, sizes and this crate's own handles from a C caller; no other \
         Law-bearing crate's output is received (the values are the caller's inputs)",
    ),
    (
        "Python export",
        "#[pyfunction]",
        13,
        "src/python.rs: numpy f32 arrays from Python, read as plain numbers; the same reason as the C ABI",
    ),
    (
        "byte reader of this crate's own format",
        "pub fn read(data: &[u8], legacy: LegacyPolicy)",
        1,
        "src/model_io.rs ATML version 2: records and compares ML_SEMANTICS_ID (G8)",
    ),
    (
        "byte reader through ATML",
        "pub fn from_bytes(",
        1,
        "src/streaming.rs: reads through ModelArchive::read and returns its provenance or its error (G8)",
    ),
    (
        "byte reader of an external format",
        "pub fn parse(bytes: &'a [u8])",
        1,
        "src/safetensors.rs: a Hugging Face safetensors file, not a Law-bearing crate's output",
    ),
    (
        "dyn Any",
        "dyn core::any::Any",
        1,
        "src/ffi.rs: a caught panic's payload, turned into a message; no Law value",
    ),
    (
        "Law-bearing crate called in memory",
        "use alice_db::",
        1,
        "src/db_bridge.rs (db): alice-db's Rust API with concrete types, so the same crate version and \
         configuration (G7's in-memory rule); the stored rows are alice-db's own format, checked by its reader (G8)",
    ),
];

fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable") {
        let path = entry.expect("entry").path();
        if path.is_dir() {
            sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn code_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .filter(|l| !l.trim().is_empty())
}

#[test]
fn every_type_erasing_boundary_is_listed_and_classified() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    sources(&root.join("src"), &mut files);
    assert!(
        files.len() > 10,
        "read {} source files (compared nothing)",
        files.len()
    );
    let texts: Vec<String> = files
        .iter()
        .map(|f| std::fs::read_to_string(f).expect("source"))
        .collect();
    let mut problems = Vec::new();
    for (form, needle, want, reason) in BOUNDARIES {
        assert!(!reason.is_empty());
        let got: usize = texts
            .iter()
            .map(|t| code_lines(t).filter(|l| l.contains(needle)).count())
            .sum();
        if got != *want {
            problems.push(format!(
                "{form} (`{needle}`): {got} found, {want} classified"
            ));
        }
    }
    // forms with none: a new one fails until it is classified or carries the identifier
    for (form, needle) in [
        ("wasm export", "wasm_bindgen"),
        ("name-keyed registry of trait objects", "Box<dyn "),
    ] {
        let got: usize = texts
            .iter()
            .map(|t| code_lines(t).filter(|l| l.contains(needle)).count())
            .sum();
        if got != 0 {
            problems.push(format!("{form} (`{needle}`): {got} found, none classified"));
        }
    }
    // another Law-bearing crate's decoder called on bytes: a line naming an alice_* crate
    // (other than this one and alice-det-math, whose calls take and return numbers) with a
    // decode call
    let foreign_decode: usize = texts
        .iter()
        .map(|t| {
            code_lines(t)
                .filter(|l| {
                    let names_other = l
                        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                        .any(|w| {
                            w.starts_with("alice_") && w != "alice_ml" && w != "alice_det_math"
                        });
                    names_other
                        && [
                            "from_bytes",
                            "decode",
                            "deserialize",
                            "from_slice",
                            "read_from",
                        ]
                        .iter()
                        .any(|d| l.contains(d))
                })
                .count()
        })
        .sum();
    if foreign_decode != 0 {
        problems.push(format!(
            "another Law-bearing crate's decoder: {foreign_decode} found, none classified"
        ));
    }
    if root.join("build.rs").exists() {
        problems.push(
            "build.rs exists: a value computed at build time is a type-erasing boundary".into(),
        );
    }
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml");
    if manifest.contains("proc-macro") {
        problems.push("a procedural-macro target: its expansion is a type-erasing boundary".into());
    }
    assert!(
        problems.is_empty(),
        "G7 boundaries:\n{}",
        problems.join("\n")
    );
}
