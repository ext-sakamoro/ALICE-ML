#!/usr/bin/env python3
"""Check that the declared MSRV is the toolchain CI actually builds with.

`Cargo.toml` `rust-version` is the claim. The `msrv` job in
`.github/workflows/ci.yml` is the evidence. This script requires that

  * the manifest declares `rust-version`,
  * the `msrv` job exists and installs exactly that version (`toolchain:`)
    and selects it for cargo (`RUSTUP_TOOLCHAIN:`, which overrides
    `rust-toolchain.toml`), both written as the same `X.Y` / `X.Y.Z`,
  * with `--rustc`, the `rustc` that runs this script reports the same
    version, so a job that installed nothing (or fell back to the pinned
    development toolchain) fails instead of passing vacuously.

A comparison count of zero is a failure.

usage: scripts/check_msrv.py [--rustc]
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOB = "msrv"


def declared(cargo_toml: str) -> str | None:
    in_pkg = False
    for line in cargo_toml.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_pkg = s == "[package]"
            continue
        if in_pkg:
            m = re.match(r'rust-version\s*=\s*"([0-9]+\.[0-9]+(?:\.[0-9]+)?)"', s)
            if m:
                return m.group(1)
    return None


def job_block(workflow: str, job: str) -> str | None:
    m = re.search(rf"^  {re.escape(job)}:\n((?:(?:    .*)?\n)+)", workflow, re.M)
    return m.group(1) if m else None


def job_values(block: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"toolchain": [], "RUSTUP_TOOLCHAIN": []}
    for key in out:
        for m in re.finditer(rf'^\s*{key}:\s*"?([0-9][0-9.]*)"?\s*$', block, re.M):
            out[key].append(m.group(1))
    return out


def check(cargo_toml: str, workflow: str, rustc_version: str | None) -> tuple[list[str], int]:
    errors: list[str] = []
    compared = 0
    want = declared(cargo_toml)
    if want is None:
        return ["Cargo.toml: no rust-version"], 0
    block = job_block(workflow, JOB)
    if block is None:
        return [f"ci.yml: no `{JOB}` job"], 0
    vals = job_values(block)
    for key, found in vals.items():
        if not found:
            errors.append(f"ci.yml {JOB} job: no `{key}:` value")
        for v in found:
            compared += 1
            if v != want:
                errors.append(f"ci.yml {JOB} job: {key} is {v}, Cargo.toml rust-version is {want}")
    if rustc_version is not None:
        compared += 1
        m = re.match(r"rustc ([0-9]+)\.([0-9]+)\.([0-9]+)", rustc_version)
        if not m:
            errors.append(f"cannot parse `rustc --version`: {rustc_version!r}")
        else:
            got = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
            if got != want and got.rsplit(".", 1)[0] != want:
                errors.append(f"active rustc is {got}, Cargo.toml rust-version is {want}")
    if compared == 0:
        errors.append("compared nothing")
    return errors, compared


def main() -> int:
    with open(os.path.join(ROOT, "Cargo.toml"), encoding="utf-8") as f:
        cargo = f.read()
    with open(os.path.join(ROOT, ".github/workflows/ci.yml"), encoding="utf-8") as f:
        wf = f.read()
    rustc = None
    if "--rustc" in sys.argv[1:]:
        rustc = subprocess.run(["rustc", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    errors, compared = check(cargo, wf, rustc)
    for e in errors:
        print(f"check_msrv: {e}", file=sys.stderr)
    if errors:
        return 1
    print(f"check_msrv: ok ({compared} comparisons, rust-version {declared(cargo)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
