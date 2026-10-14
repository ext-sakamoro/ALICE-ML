#!/usr/bin/env python3
"""scripts/check_g1_banned_list.py — clippy.toml vs ALICE-Law's G1 gate.

Fetches ALICE-Law's conformance/gates.toml at the revision .github/law-g1-pin
names, refused unless its SHA-256 is the one written there (the file at a
named revision, byte-identical to it -- same shape as some other ALICE-*
repos' scripts/public_text.sh / .github/public-text-pin pattern uses for a
different fetched file). Reads G1 ("no-libm-direct")'s `methods` list for the
entries with verdict "banned", and checks clippy.toml's disallowed-methods
array (parsed as TOML, not text search, so a commented-out entry is correctly
absent) bans each one for both f32 and f64.

This can tell "clippy.toml still matches the pinned revision's G1 list" but
not "the pinned revision is still G1's current list" -- that's a human
decision, made by moving the pin (same division of labour as
that pattern: a change in ALICE-Law never changes this repository's
result by itself, only a commit here that bumps the pin and still passes at
the new revision does).

Exits 2 when the pin is malformed, the fetch fails, or G1's banned list
comes back empty (0 methods read is not "nothing wrong," it's "nothing was
compared").

usage: scripts/check_g1_banned_list.py
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

ROOT = Path(__file__).resolve().parent.parent
PIN = ROOT / ".github" / "law-g1-pin"
CLIPPY_TOML = ROOT / "clippy.toml"
LAW_REPO = "https://github.com/ext-sakamoro/ALICE-Law"


def read_pin() -> tuple[str, str]:
    text = PIN.read_text(encoding="utf-8")
    rev = sha = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "rev":
            rev = parts[1]
        elif len(parts) == 2 and parts[0] == "sha256":
            sha = parts[1]
    if rev is None or sha is None or not HEX40.match(rev) or not HEX64.match(sha):
        print(f"{PIN}: needs 'rev <40 hex>' and 'sha256 <64 hex>'", file=sys.stderr)
        sys.exit(2)
    return rev, sha


def fetch_gates_toml(rev: str, expected_sha256: str) -> bytes:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        subprocess.run(
            ["git", "-C", tmp, "fetch", "-q", "--depth", "1", LAW_REPO, rev],
            check=True,
        )
        data = subprocess.run(
            ["git", "-C", tmp, "show", "FETCH_HEAD:conformance/gates.toml"],
            check=True,
            capture_output=True,
        ).stdout
    got = hashlib.sha256(data).hexdigest()
    if got != expected_sha256:
        print(
            f"conformance/gates.toml at {rev} has sha256 {got}, {PIN} says {expected_sha256}",
            file=sys.stderr,
        )
        sys.exit(2)
    return data


def g1_banned_methods(gates_toml: bytes) -> list[str]:
    parsed = tomllib.loads(gates_toml.decode("utf-8"))
    gates = parsed.get("gate")
    if not isinstance(gates, list):
        print("conformance/gates.toml: no [[gate]] tables found", file=sys.stderr)
        sys.exit(2)
    for gate in gates:
        if gate.get("id") == "G1":
            methods = gate.get("methods", [])
            return sorted(m["name"] for m in methods if m.get("verdict") == "banned")
    print("conformance/gates.toml: no gate with id = \"G1\" found", file=sys.stderr)
    sys.exit(2)


def clippy_toml_banned_paths() -> set[str]:
    parsed = tomllib.loads(CLIPPY_TOML.read_text(encoding="utf-8"))
    entries = parsed.get("disallowed-methods", [])
    return {e["path"] for e in entries if "path" in e}


def main() -> int:
    rev, expected_sha256 = read_pin()
    gates_toml = fetch_gates_toml(rev, expected_sha256)
    banned = g1_banned_methods(gates_toml)
    if not banned:
        print("G1's banned method list came back empty -- nothing was compared", file=sys.stderr)
        return 2

    banned_paths = clippy_toml_banned_paths()
    missing = [
        f"{width}::{method}"
        for method in banned
        for width in ("f32", "f64")
        if f"{width}::{method}" not in banned_paths
    ]
    if missing:
        print(
            f"clippy.toml's disallowed-methods is missing {len(missing)} of G1's "
            f"{len(banned)} banned methods (pinned at {rev}): {missing}",
            file=sys.stderr,
        )
        return 1

    print(f"ok: clippy.toml bans all {len(banned)} of G1's banned methods for both widths (pinned at {rev})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
