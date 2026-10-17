#!/usr/bin/env python3
"""Tests of scripts/law_tie_mutants.py: the rewrites, the AVX2 block selection and the verdicts."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import law_tie_mutants as ltm  # noqa: E402


def rewrite(line: str, v: str) -> str:
    m = ltm.CALL.search(line)
    assert m, line
    out = ltm.mutate(line, m, v)
    assert out is not None, line
    return out


class Rewrites(unittest.TestCase):
    def test_max_both_orders(self):
        self.assertEqual(rewrite("let s = crate::law_math::max(a, 1e-8);", "A"),
                         "let s = { let (x, y) = (a, 1e-8); if x > y { x } else { y } };")
        self.assertEqual(rewrite("let s = crate::law_math::max(a, 1e-8);", "B"),
                         "let s = { let (x, y) = (a, 1e-8); if y > x { y } else { x } };")

    def test_min64_nested_arguments(self):
        out = rewrite("m = crate::law_math::min64(f(x, (y, z)), 1.0);", "A")
        self.assertEqual(out, "m = { let (x, y) = (f(x, (y, z)), 1.0); if x < y { x } else { y } };")

    def test_fold_argument_is_a_closure(self):
        self.assertEqual(rewrite(".fold(f64::NEG_INFINITY, crate::law_math::max64);", "B"),
                         ".fold(f64::NEG_INFINITY, |x, y| if y > x { y } else { x });")

    def test_max_of_and_relu_and_argmax(self):
        self.assertIn("fold(f32::NEG_INFINITY, |x: f32, y: f32| if x > y { x } else { y })",
                      rewrite("let m = crate::law_math::max_of(&v);", "A"))
        self.assertIn("if 0.0 > x { 0.0 } else { x }", rewrite("*o = crate::law_math::relu(av);", "B"))
        self.assertIn("if v >= s[b]", rewrite("crate::law_math::argmax(slice)", "B"))

    def test_clamp_and_lanes(self):
        self.assertIn("if x < lo { lo } else if x > hi { hi } else { x }",
                      rewrite("crate::law_math::clamp(x, -1.0, 1.0)", "A"))
        self.assertEqual(rewrite("v = crate::law_math::max_ps(a, b);", "B"),
                         "v = core::arch::x86_64::_mm256_max_ps(b, a);")

    def test_unbalanced_call_is_an_error(self):
        with self.assertRaises(ValueError):
            rewrite("crate::law_math::max(a, b", "A")


class Avx2Blocks(unittest.TestCase):
    def test_lines_inside_the_block_only(self):
        text = "\n".join([
            "fn f() {",                                   # 1
            "    if crate::ops::has_avx2() {",            # 2
            "        a = crate::law_math::max_ps(a, b);",  # 3
            "        if x { y(); }",                      # 4
            "    }",                                      # 5
            "    c = crate::law_math::max(c, d);",         # 6
            "}",
        ])
        self.assertEqual(ltm.avx2_lines(text), {3, 4, 5})


class Verdicts(unittest.TestCase):
    def test_unlisted_survivor_fails(self):
        p = ltm.judge({"f\tl\tmaxA": "survived"}, {}, {"f\tl\tmaxA"})
        self.assertEqual(p, ["survived and not listed as equivalent: f\tl\tmaxA"])

    def test_listed_survivor_passes(self):
        self.assertEqual(ltm.judge({"f\tl\tmaxA": "survived"}, {"f\tl\tmaxA": "x > c is maximumNumber"}, {"f\tl\tmaxA"}), [])

    def test_listed_but_killed_fails(self):
        p = ltm.judge({"f\tl\tmaxA": "killed"}, {"f\tl\tmaxA": "r"}, {"f\tl\tmaxA"})
        self.assertEqual(p, ["listed as equivalent but killed: f\tl\tmaxA"])

    def test_stale_listing_fails_unless_the_tree_has_it(self):
        self.assertEqual(ltm.judge({}, {"f\tl\tmaxA": "r"}, set()),
                         ["listed equivalent no longer exists: f\tl\tmaxA"])
        # a listed mutant this run did not select (e.g. an AVX2 site under --skip-avx2) is fine
        self.assertEqual(ltm.judge({}, {"f\tl\tmaxA": "r"}, {"f\tl\tmaxA"}), [])

    def test_build_error_fails(self):
        self.assertEqual(ltm.judge({"f\tl\tmaxA": "build-error"}, {}, {"f\tl\tmaxA"}),
                         ["mutant does not compile: f\tl\tmaxA"])

    def test_no_mutant_selected_fails(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "src").mkdir()
            (Path(d) / "src" / "a.rs").write_text("fn f() {}\n", encoding="utf-8")
            self.assertEqual(ltm.main(["--repo", d, "--avx2-only", "--", "--lib"]), 2)

    def test_equivalents_need_a_reason(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "eq.tsv"
            p.write_text("f\tl\tmaxA\t\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                ltm.load_equivalents(p)


if __name__ == "__main__":
    unittest.main()
