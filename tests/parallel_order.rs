//! The one parallel float path (`ops::ternary_matmul_batch` under `parallel`, rayon) gives
//! the same bits at every thread count and the same bits as the serial row-by-row matvec
//! (ALICE-Law G13): each row is computed by one task, and no float is combined across
//! tasks.
//!
//! `every_parallel_site_is_the_one_this_file_tests` scans `src/` so that a new rayon site
//! cannot appear without a test of its own here.

#[cfg(feature = "parallel")]
use alice_ml::ops::{ternary_matmul_batch, ternary_matvec, TernaryWeight};

#[cfg(feature = "parallel")]
const SPECIALS: [f32; 12] = [
    0.0,
    -0.0,
    f32::NAN,
    1.0,
    -1.0,
    0.5,
    -2.25,
    3.5,
    f32::MAX,
    -f32::MAX,
    1e-40,
    f32::INFINITY,
];

#[cfg(feature = "parallel")]
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter()
        .map(|x| if x.is_nan() { 0x7fc0_0000 } else { x.to_bits() })
        .collect()
}

#[cfg(feature = "parallel")]
fn batch_in_pool(threads: usize, input: &[f32], w: &TernaryWeight, batch: usize) -> Vec<f32> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a pool");
    let mut out = vec![0.0f32; batch * w.out_features()];
    pool.install(|| ternary_matmul_batch(input, w, &mut out, batch));
    out
}

/// Finite values of very different magnitudes: a sum of them depends on the order it is
/// taken in, so any combination of partial results across tasks would move bits.
#[cfg(feature = "parallel")]
const ORDER_SENSITIVE: [f32; 7] = [1.0e7, 0.1, -1.0e7, 3.3, 1.0e-3, -0.7, 2.5e6];

#[cfg(feature = "parallel")]
fn check(name: &str, out_f: usize, in_f: usize, batch: usize, input: &[f32]) -> usize {
    let values: Vec<i8> = (0..out_f * in_f)
        .map(|k| [1i8, 0, -1, 1, -1][k % 5])
        .collect();
    let w = TernaryWeight::from_ternary(&values, out_f, in_f);
    let mut serial = vec![0.0f32; batch * out_f];
    for b in 0..batch {
        ternary_matvec(
            &input[b * in_f..(b + 1) * in_f],
            &w,
            &mut serial[b * out_f..(b + 1) * out_f],
        );
    }
    let want = bits(&serial);
    let mut compared = 0;
    for threads in [1, 2, 8] {
        assert_eq!(
            bits(&batch_in_pool(threads, input, &w, batch)),
            want,
            "{name}: {threads} threads"
        );
        compared += 1;
    }
    compared
}

#[cfg(feature = "parallel")]
#[test]
fn the_batch_has_the_same_bits_at_one_two_and_eight_threads() {
    let (out_f, in_f, batch) = (5, 1031, 23);
    let finite: Vec<f32> = (0..batch * in_f)
        .map(|k| ORDER_SENSITIVE[(k * 3 + k / 11) % ORDER_SENSITIVE.len()])
        .collect();
    let specials: Vec<f32> = (0..37 * 13)
        .map(|k| SPECIALS[(k * 5 + k / 7) % SPECIALS.len()])
        .collect();
    let compared = check("order-sensitive", out_f, in_f, batch, &finite)
        + check("specials", 7, 13, 37, &specials);
    assert_eq!(compared, 6);
}

/// Every rayon parallel iterator in `src/`: the file, the line's code, and the count. The
/// batch above is the only one; a new site fails here until it has a thread-count test.
#[test]
fn every_parallel_site_is_the_one_this_file_tests() {
    let mut found = Vec::new();
    let mut files = 0;
    let mut stack = vec![std::path::PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src"
    ))];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("src") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files += 1;
                let text = std::fs::read_to_string(&path).expect("source");
                for line in text.lines() {
                    let code = line.split("//").next().unwrap_or("");
                    if code.contains("rayon")
                        || code.contains(".par_")
                        || code.contains("into_par_iter")
                    {
                        let rel = path
                            .strip_prefix(env!("CARGO_MANIFEST_DIR"))
                            .expect("in the crate");
                        found.push(format!("{} {}", rel.display(), code.trim()));
                    }
                }
            }
        }
    }
    assert!(files > 10, "read {files} source files (compared nothing)");
    found.sort();
    assert_eq!(
        found,
        [
            "src/ops.rs .par_chunks_mut(out_features)",
            "src/ops.rs .zip(input.par_chunks(in_features))",
            "src/ops.rs use rayon::prelude::*;",
        ],
        "a parallel site in src/ other than ops::ternary_matmul_batch"
    );
}
