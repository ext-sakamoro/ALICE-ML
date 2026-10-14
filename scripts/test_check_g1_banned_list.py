#!/usr/bin/env python3
"""Tests of scripts/check_g1_banned_list.py that need no network: the pin
format check, and the pure parsing/comparison functions, called directly
with in-memory TOML instead of a fetched file. The real fetch-and-compare
(does the live pin's content still match) runs in CI and in
scripts/preflight.sh.

usage: python3 scripts/test_check_g1_banned_list.py
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = ROOT / "scripts" / "check_g1_banned_list.py"

spec = importlib.util.spec_from_file_location("check_g1_banned_list", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
check_g1 = importlib.util.module_from_spec(spec)
sys.modules["check_g1_banned_list"] = check_g1
spec.loader.exec_module(check_g1)


FAKE_GATES_TOML = b"""
schema_version = 1

[[gate]]
id = "G1"
methods = [
    { name = "sin", verdict = "banned", reason = "x" },
    { name = "cos", verdict = "banned", reason = "x" },
    { name = "sqrt", verdict = "allowed", reason = "x" },
    { name = "powi", verdict = "pending", reason = "x" },
]

[[gate]]
id = "G2"
methods = []
"""


class G1BannedMethods(unittest.TestCase):
    def test_extracts_only_banned_from_the_named_gate(self) -> None:
        self.assertEqual(check_g1.g1_banned_methods(FAKE_GATES_TOML), ["cos", "sin"])

    def test_no_g1_gate_exits_2(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            check_g1.g1_banned_methods(b'[[gate]]\nid = "G2"\nmethods = []\n')
        self.assertEqual(cm.exception.code, 2)

    def test_malformed_toml_has_no_gate_key_exits_2(self) -> None:
        with self.assertRaises(SystemExit) as cm:
            check_g1.g1_banned_methods(b"schema_version = 1\n")
        self.assertEqual(cm.exception.code, 2)


class ClippyTomlBannedPaths(unittest.TestCase):
    def test_reads_path_from_each_entry(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            clippy_toml = Path(tmp) / "clippy.toml"
            clippy_toml.write_text(
                'disallowed-methods = [\n'
                '    { path = "f32::sin", reason = "x" },\n'
                '    { path = "f64::sin", reason = "x" },\n'
                ']\n'
            )
            original = check_g1.CLIPPY_TOML
            check_g1.CLIPPY_TOML = clippy_toml
            try:
                self.assertEqual(check_g1.clippy_toml_banned_paths(), {"f32::sin", "f64::sin"})
            finally:
                check_g1.CLIPPY_TOML = original

    def test_a_commented_out_entry_is_not_counted(self) -> None:
        # the whole point of parsing as TOML rather than a text search: a
        # commented-out line is not in the parsed array at all.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            clippy_toml = Path(tmp) / "clippy.toml"
            clippy_toml.write_text(
                'disallowed-methods = [\n'
                '    # { path = "f32::sin", reason = "x" },\n'
                '    { path = "f64::sin", reason = "x" },\n'
                ']\n'
            )
            original = check_g1.CLIPPY_TOML
            check_g1.CLIPPY_TOML = clippy_toml
            try:
                self.assertEqual(check_g1.clippy_toml_banned_paths(), {"f64::sin"})
            finally:
                check_g1.CLIPPY_TOML = original


class ReadPin(unittest.TestCase):
    def test_malformed_pin_exits_2(self) -> None:
        import tempfile

        bad_pins = [
            "",
            "rev abc\nsha256 " + "b" * 64 + "\n",
            "rev " + "a" * 40 + "\n",
            "sha256 " + "b" * 64 + "\n",
            "rev " + "A" * 40 + "\nsha256 " + "b" * 64 + "\n",
        ]
        for pin_text in bad_pins:
            with self.subTest(pin=pin_text):
                with tempfile.TemporaryDirectory() as tmp:
                    pin_path = Path(tmp) / "law-g1-pin"
                    pin_path.write_text(pin_text)
                    original = check_g1.PIN
                    check_g1.PIN = pin_path
                    try:
                        with self.assertRaises(SystemExit) as cm:
                            check_g1.read_pin()
                        self.assertEqual(cm.exception.code, 2)
                    finally:
                        check_g1.PIN = original

    def test_well_formed_pin_parses(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pin_path = Path(tmp) / "law-g1-pin"
            rev = "a" * 40
            sha = "b" * 64
            pin_path.write_text(f"rev {rev}\nsha256 {sha}\n")
            original = check_g1.PIN
            check_g1.PIN = pin_path
            try:
                self.assertEqual(check_g1.read_pin(), (rev, sha))
            finally:
                check_g1.PIN = original


if __name__ == "__main__":
    unittest.main()
