#!/usr/bin/env python3
"""Tests for scripts/check_msrv.py.

The first case runs the checker against this repository; the others break
exactly one thing and assert it is reported.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_msrv as cm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CARGO = '[package]\nname = "x"\nversion = "0.1.0"\nrust-version = "1.87"\n'
WF = """name: CI
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - run: true

  msrv:
    runs-on: ubuntu-latest
    env:
      RUSTUP_TOOLCHAIN: "1.87"
    steps:
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: "1.87"

  fmt:
    runs-on: ubuntu-latest
    steps:
      - run: true
"""


class CheckMsrv(unittest.TestCase):
    def test_real_tree_is_consistent(self) -> None:
        with open(os.path.join(ROOT, "Cargo.toml"), encoding="utf-8") as f:
            cargo = f.read()
        with open(os.path.join(ROOT, ".github/workflows/ci.yml"), encoding="utf-8") as f:
            wf = f.read()
        errors, compared = cm.check(cargo, wf, None)
        self.assertEqual(errors, [])
        self.assertGreaterEqual(compared, 2)

    def test_fixture_passes(self) -> None:
        errors, compared = cm.check(CARGO, WF, "rustc 1.87.0 (17067e9ac 2025-05-09)")
        self.assertEqual(errors, [])
        self.assertEqual(compared, 3)

    def test_bump_rust_version_only_is_red(self) -> None:
        errors, _ = cm.check(CARGO.replace('"1.87"', '"1.98.1"'), WF, None)
        self.assertEqual(len(errors), 2)

    def test_bump_one_workflow_value_is_red(self) -> None:
        errors, _ = cm.check(CARGO, WF.replace('toolchain: "1.87"', 'toolchain: "1.90"'), None)
        self.assertEqual(len(errors), 1)
        errors, _ = cm.check(CARGO, WF.replace('RUSTUP_TOOLCHAIN: "1.87"', 'RUSTUP_TOOLCHAIN: "1.90"'), None)
        self.assertEqual(len(errors), 1)

    def test_wrong_active_rustc_is_red(self) -> None:
        errors, _ = cm.check(CARGO, WF, "rustc 1.98.1 (abc 2026-01-01)")
        self.assertEqual(len(errors), 1)

    def test_missing_job_or_values_is_red(self) -> None:
        errors, compared = cm.check(CARGO, WF.replace("  msrv:", "  other:"), None)
        self.assertEqual(compared, 0)
        self.assertTrue(errors)
        errors, _ = cm.check(CARGO, WF.replace('toolchain: "1.87"', "foo: bar"), None)
        self.assertTrue(errors)

    def test_missing_rust_version_is_red(self) -> None:
        errors, _ = cm.check(CARGO.replace('rust-version = "1.87"\n', ""), WF, None)
        self.assertTrue(errors)

    def test_rust_version_in_other_table_is_ignored(self) -> None:
        cargo = '[package]\nname = "x"\n\n[metadata]\nrust-version = "1.87"\n'
        errors, _ = cm.check(cargo, WF, None)
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
