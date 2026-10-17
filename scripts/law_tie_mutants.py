#!/usr/bin/env python3
"""Comparison-rewrite mutants of every `crate::law_math` site (ALICE-Law G2, tie rule).

Each routed call is rewritten, one at a time, as the plain comparison that clippy cannot
see, and the given `cargo test` command runs on the mutated tree:

* variant A: `max(a, b)` -> `if a > b { a } else { b }` (`<` for min), a fold of `max` /
  `min` or `max_of` / `min_of` -> the same comparison as a closure, `relu(x)` ->
  `if x > 0.0 { x } else { 0.0 }`, `clamp` -> two comparisons, `argmax` -> the first
  strictly greater element, `order` -> `partial_cmp` with NaN as equal, `max_ps` /
  `min_ps` -> the bare `_mm256_max_ps` / `_mm256_min_ps`
* variant B: the operands swapped (`if b > a { b } else { a }`), `relu` ->
  `if 0.0 > x { 0.0 } else { x }`, `clamp` -> a comparison maximum then minimum,
  `argmax` -> `>=` (the last maximum), `order` -> `total_cmp` (a NaN's sign orders it),
  the intrinsics with the operands swapped

A mutant is killed when the tests fail. Some rewrites give the Law's result for every
operand the site can see (`x > c ? x : c` for a constant c > 0 is maximumNumber); they are
listed with a one-line reason in the equivalents file and must survive. The run fails when a
mutant survives that is not listed, when a listed mutant is killed or no longer exists, when
a mutant does not compile, and when no mutant ran.

usage: law_tie_mutants.py [--repo DIR] [--avx2-only] [--equivalents TSV] [--out TSV]
                          [--list] -- CARGO_TEST_ARGS...
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

CALL = re.compile(r"crate::law_math::(max_of|min_of|relu|argmax|order|max64|min64|max_ps|min_ps"
                  r"|clamp64|clamp|max|min)\b")
AVX2_OPEN = re.compile(r"if\s+crate::ops::has_avx2\(\)\s*\{")


def split_args(s: str, start: int) -> tuple[list[str], int]:
    """s[start] is '(': the top-level arguments and the index after the matching ')'"""
    depth, args, cur, i = 0, [], [], start
    while i < len(s):
        c = s[i]
        if c in "([{":
            depth += 1
            if depth > 1:
                cur.append(c)
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur).strip())
                return args, i + 1
            cur.append(c)
        elif c == "," and depth == 1:
            args.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    raise ValueError(f"unbalanced call: {s[start:]!r}")


def mutate(line: str, m: re.Match, v: str) -> str | None:
    """the line with the call at m rewritten as comparison variant v ('A' or 'B'), or None"""
    name, after = m.group(1), m.end()
    if after < len(line) and line[after] == "(":
        args, end = split_args(line, after)
        if name in ("max_of", "min_of"):
            (x,) = args
            init, op = ("NEG_INFINITY", ">") if name == "max_of" else ("INFINITY", "<")
            body = f"if x {op} y {{ x }} else {{ y }}" if v == "A" else f"if y {op} x {{ y }} else {{ x }}"
            rep = f"({x}).iter().copied().fold(f32::{init}, |x: f32, y: f32| {body})"
        elif name == "relu":
            (x,) = args
            body = "if x > 0.0 { x } else { 0.0 }" if v == "A" else "if 0.0 > x { 0.0 } else { x }"
            rep = f"{{ let x: f32 = {x}; {body} }}"
        elif name == "argmax":
            (x,) = args
            op = ">" if v == "A" else ">="
            rep = (f"{{ let s: &[f32] = {x}; let mut b = 0; for (i, &v) in s.iter().enumerate().skip(1) "
                   f"{{ if v {op} s[b] {{ b = i; }} }} b }}")
        elif name == "order":
            a, b = args
            body = "x.partial_cmp(&y).unwrap_or(core::cmp::Ordering::Equal)" if v == "A" else "x.total_cmp(&y)"
            rep = f"{{ let (x, y): (f32, f32) = ({a}, {b}); {body} }}"
        elif name in ("max", "max64", "min", "min64"):
            a, b = args
            op = ">" if name.startswith("max") else "<"
            body = f"if x {op} y {{ x }} else {{ y }}" if v == "A" else f"if y {op} x {{ y }} else {{ x }}"
            rep = f"{{ let (x, y) = ({a}, {b}); {body} }}"
        elif name in ("clamp", "clamp64"):
            x, lo, hi = args
            body = ("if x < lo { lo } else if x > hi { hi } else { x }" if v == "A"
                    else "let m = if x > lo { x } else { lo }; if m < hi { m } else { hi }")
            rep = f"{{ let (x, lo, hi) = ({x}, {lo}, {hi}); {body} }}"
        elif name in ("max_ps", "min_ps"):
            a, b = args
            ops = (a, b) if v == "A" else (b, a)
            rep = f"core::arch::x86_64::_mm256_{name}({ops[0]}, {ops[1]})"
        else:
            return None
        return line[:m.start()] + rep + line[end:]
    # a bare path: a fold argument
    if name in ("max", "max64", "min", "min64"):
        op = ">" if name.startswith("max") else "<"
        body = f"if x {op} y {{ x }} else {{ y }}" if v == "A" else f"if y {op} x {{ y }} else {{ x }}"
        return line[:m.start()] + f"|x, y| {body}" + line[after:]
    return None


# the helpers' own bodies in src/law_math.rs: (needle, variant A, variant B)
HELPERS = [
    ("    xs.iter().copied().fold(f32::NEG_INFINITY, max)",
     "    xs.iter().copied().fold(f32::NEG_INFINITY, |x: f32, y: f32| if x > y { x } else { y })",
     "    xs.iter().copied().fold(f32::NEG_INFINITY, |x: f32, y: f32| if y > x { y } else { x })"),
    ("    xs.iter().copied().fold(f32::INFINITY, min)",
     "    xs.iter().copied().fold(f32::INFINITY, |x: f32, y: f32| if x < y { x } else { y })",
     "    xs.iter().copied().fold(f32::INFINITY, |x: f32, y: f32| if y < x { y } else { x })"),
    ("    max(x, 0.0)\n}",
     "    if x > 0.0 { x } else { 0.0 }\n}",
     "    if 0.0 > x { 0.0 } else { x }\n}"),
    ("        .position(|x| x.to_bits() == m.to_bits())",
     "        .position(|&x| x == m)",
     "        .rposition(|x| x.to_bits() == m.to_bits())"),
    ("        (false, false) => a.total_cmp(&b),",
     "        (false, false) => a.partial_cmp(&b).unwrap_or(core::cmp::Ordering::Equal),",
     "        (false, false) => core::cmp::Ordering::Equal,"),
    ("        (true, false) => core::cmp::Ordering::Less,",
     "        (true, false) => core::cmp::Ordering::Greater,",
     "        (true, false) => core::cmp::Ordering::Equal,"),
]


def avx2_lines(text: str) -> set[int]:
    """1-based line numbers inside each `if crate::ops::has_avx2() {` block"""
    lines = text.split("\n")
    inside: set[int] = set()
    for i, line in enumerate(lines):
        if not AVX2_OPEN.search(line):
            continue
        depth = 0
        for j in range(i, len(lines)):
            code = lines[j].split("//", 1)[0]
            depth += code.count("{") - code.count("}")
            if j > i:
                inside.add(j + 1)
            if depth <= 0 and j > i:
                break
    return inside


def collect(repo: Path, avx2_only: bool, skip_avx2: bool = False) -> list[dict]:
    """every mutant: key, file, line number, the mutated text"""
    out = []
    for f in sorted((repo / "src").rglob("*.rs")):
        rel = f.relative_to(repo).as_posix()
        if rel == "src/law_math.rs":
            continue
        text = f.read_text(encoding="utf-8")
        blocks = avx2_lines(text)
        keep = blocks if avx2_only else None
        seen: dict[str, int] = {}
        lines = text.split("\n")
        for n, line in enumerate(lines, 1):
            if line.lstrip().startswith("//") or (keep is not None and n not in keep) or (skip_avx2 and n in blocks):
                continue
            for m in CALL.finditer(line):
                for v in ("A", "B"):
                    new = mutate(line, m, v)
                    if new is None:
                        continue
                    base = f"{rel}\t{line.strip()}\t{m.group(1)}{v}"
                    k = seen.get(base, 0)
                    seen[base] = k + 1
                    mutated = lines[:n - 1] + [new] + lines[n:]
                    out.append({"key": base if k == 0 else f"{base}#{k + 1}", "file": rel, "line": n,
                                "text": "\n".join(mutated)})
    if not avx2_only:
        lm = repo / "src/law_math.rs"
        orig = lm.read_text(encoding="utf-8")
        for needle, a, b in HELPERS:
            if orig.count(needle) != 1:
                raise SystemExit(f"law_tie_mutants: helper body not found once in src/law_math.rs: {needle!r}")
            for v, rep in (("A", a), ("B", b)):
                out.append({"key": f"src/law_math.rs\t{' '.join(needle.split())}\thelper{v}", "file": "src/law_math.rs",
                            "line": orig[:orig.index(needle)].count("\n") + 1, "text": orig.replace(needle, rep)})
    return out


def load_equivalents(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    eq = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 4 or not parts[3].strip():
            raise SystemExit(f"law_tie_mutants: {path}: want file<TAB>line<TAB>mutant<TAB>reason: {raw!r}")
        eq["\t".join(parts[:3])] = parts[3]
    return eq


def judge(results: dict[str, str], equivalents: dict[str, str], known: set[str]) -> list[str]:
    """problems: an unlisted survivor, a listed mutant killed, a listing no mutant of the tree
    has (`known`: every mutant key of the tree, whatever this run selected), a build error"""
    problems = []
    for key, verdict in sorted(results.items()):
        if verdict == "build-error":
            problems.append(f"mutant does not compile: {key}")
        elif verdict == "survived" and key not in equivalents:
            problems.append(f"survived and not listed as equivalent: {key}")
        elif verdict == "killed" and key in equivalents:
            problems.append(f"listed as equivalent but killed: {key}")
    for key in sorted(equivalents):
        if key not in known:
            problems.append(f"listed equivalent no longer exists: {key}")
    return problems


def run_one(repo: Path, file: str, text: str, cargo_args: list[str]) -> str:
    path = repo / file
    orig = path.read_text(encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    try:
        r = subprocess.run(["cargo", "test", *cargo_args], cwd=repo, capture_output=True, text=True)
    finally:
        path.write_text(orig, encoding="utf-8")
    if r.returncode == 0:
        return "survived"
    out = r.stdout + r.stderr
    return "build-error" if ("error[E" in out or "could not compile" in out) else "killed"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        print("law_tie_mutants: missing `-- CARGO_TEST_ARGS`", file=sys.stderr)
        return 2
    i = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=".")
    ap.add_argument("--avx2-only", action="store_true", help="only the sites inside `if crate::ops::has_avx2()` blocks")
    ap.add_argument("--skip-avx2", action="store_true", help="every site except those (on a target that never runs them)")
    ap.add_argument("--equivalents", default="scripts/law_tie_equivalents.tsv")
    ap.add_argument("--out")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv[:i])
    cargo_args = argv[i + 1:]
    repo = Path(a.repo)
    if a.avx2_only and a.skip_avx2:
        print("law_tie_mutants: --avx2-only and --skip-avx2 exclude each other", file=sys.stderr)
        return 2
    mutants = collect(repo, a.avx2_only, a.skip_avx2)
    if not mutants:
        print("law_tie_mutants: no mutant selected (compared nothing)", file=sys.stderr)
        return 2
    if a.list:
        for m in mutants:
            print(m["key"].replace("\t", " | "))
        print(f"law_tie_mutants: {len(mutants)} mutants")
        return 0
    equivalents = load_equivalents(repo / a.equivalents)
    results = {}
    for m in mutants:
        results[m["key"]] = run_one(repo, m["file"], m["text"], cargo_args)
        print(f"{results[m['key']]:<11} {m['file']}:{m['line']} {m['key'].rsplit(chr(9), 1)[1]}", flush=True)
    if a.out:
        Path(a.out).write_text("".join(f"{k}\t{v}\n" for k, v in sorted(results.items())), encoding="utf-8")
    known = {m["key"] for m in collect(repo, False)}
    problems = judge(results, equivalents, known)
    for p in problems:
        print(f"law_tie_mutants: {p}")
    killed = sum(v == "killed" for v in results.values())
    listed = sum(v == "survived" and k in equivalents for k, v in results.items())
    unlisted = sum(v == "survived" and k not in equivalents for k, v in results.items())
    print(f"law_tie_mutants: {len(results)} mutants, {killed} killed, {listed} survived as listed equivalents, "
          f"{unlisted} survived unlisted, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
