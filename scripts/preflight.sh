#!/usr/bin/env bash
# scripts/preflight.sh — local reproduction of the CI gates before `git push`.
# Every command is the one CI runs. A step this file does not cover is a step
# that can only fail remotely — when a workflow step is added, add it here in
# the same commit. Run `--quick` before every push.
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the test / bench suites)
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate roots invalidates only this repo's fingerprints.
relint() { git ls-files | grep -E '(^|/)src/(lib|main)\.rs$' | xargs -r touch; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
has_toolchain() { rustup toolchain list | grep -q "^$1"; }

# Steps CI runs that this file cannot reproduce locally (they can only fail remotely):
#   - security-audit.yml:audit:Install cargo-audit (needs network / runner-only)
#   - security-audit.yml:deny:Install cargo-deny (needs network / runner-only)
#   - security-audit.yml:coverage (job is continue-on-error: informational in CI)
#   - fuzz.yml:fuzz:Install cargo-fuzz (needs network / runner-only)
#   - fuzz.yml:fuzz:Build fuzz target (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) (continue-on-error)
#   - fuzz.yml:fuzz:Report crash (informational) (no cargo / grep)

need actionlint "brew install actionlint"
need cargo-audit "cargo install cargo-audit --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-hack "cargo install cargo-hack --locked"
need cargo-machete "cargo install cargo-machete --locked"

step "ci.yml / neon: Clippy (neon, all targets, pedantic + nursery)"
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo clippy --features "$NATIVE_FEATURES,neon" --all-targets -- -W clippy::pedantic -W clippy::nursery -D warnings )

step "ci.yml / no_std: Build (no_std, host)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo rustc --lib --no-default-features --crate-type rlib )

step "ci.yml / no_std: Build (no_std + simd, host)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo rustc --lib --no-default-features --features simd --crate-type rlib )

step "ci.yml / no_std: Build (no_std, bare-metal thumbv7em-none-eabihf)"
rustup target list --installed | grep -q '^thumbv7em-none-eabihf$' || rustup target add thumbv7em-none-eabihf
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo rustc --lib --no-default-features --crate-type rlib --target thumbv7em-none-eabihf )

step "ci.yml / no_std: Clippy (no_std, pedantic + nursery)"
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; RUSTC_WORKSPACE_WRAPPER="$(rustup which clippy-driver)" cargo rustc --lib --no-default-features --crate-type rlib -- -W clippy::pedantic -W clippy::nursery -D warnings )

step "ci.yml / clippy: Clippy (default features, all targets)"
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo clippy --all-targets -- -D warnings )

step "ci.yml / clippy: Clippy (full native feature set, all targets, pedantic + nursery)"
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo clippy --features "$NATIVE_FEATURES" --all-targets -- -W clippy::pedantic -W clippy::nursery -D warnings )

step "ci.yml / clippy: Clippy (pyo3, lib)"
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings"; cargo clippy --lib --features pyo3 -- -D warnings )

step "ci.yml / clippy: Clippy canary data (generator tests)"
python3 scripts/test_gen_clippy_canary_calls.py

step "ci.yml / clippy: Clippy canary data (generated from clippy.toml, no drift)"
python3 scripts/gen_clippy_canary_calls.py --check

step "ci.yml / clippy: Clippy canary (tests)"
python3 scripts/test_clippy_canary.py

step "ci.yml / clippy: Clippy canary (every planted banned call is flagged)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings"; python3 scripts/clippy_canary.py )

step "ci.yml / feature-powerset: Powerset (std + {ffi, simd, parallel, safetensors} depth 2)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo hack check --lib --feature-powerset --depth 2 --exclude-features pyo3,neon --features std )

step "ci.yml / fmt: run"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo fmt -- --check )

step "ci.yml / fmt: no-direct-rkyv guard (tests)"
python3 scripts/test_no_direct_rkyv.py

step "ci.yml / fmt: no-direct-rkyv guard"
python3 scripts/no_direct_rkyv.py

step "ci.yml / float-environment: run_tests.sh and build configuration guard (tests, then check)"
python3 scripts/test_run_tests.py
python3 scripts/test_build_config_guard.py
python3 scripts/build_config_guard.py
python3 scripts/test_build_flags_check.py

step "ci.yml / doc: Doc (full native feature set)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db" RUSTDOCFLAGS="-Dwarnings"; cargo doc --no-deps --features "$NATIVE_FEATURES" )

step "ci.yml / actionlint: actionlint"
actionlint .github/workflows/*.yml

step "ci.yml / docs: readme_sync (oracle + check)"
python3 scripts/test_readme_sync.py
python3 scripts/readme_sync.py --check

step "ci.yml / law-g1-banned-list: check_g1_banned_list (oracle + check)"
python3 scripts/test_check_g1_banned_list.py
python3 scripts/check_g1_banned_list.py

step "ci.yml / law-g1-banned-list: law_tie_mutants.py (tests)"
python3 scripts/test_law_tie_mutants.py

step "ci.yml / msrv: check_msrv (oracle + check)"
python3 scripts/test_check_msrv.py
python3 scripts/check_msrv.py

step "ci.yml / msrv: build with the declared rust-version"
MSRV="$(python3 -c 'import re;print(re.search(r"^rust-version\s*=\s*\"([0-9.]+)\"",open("Cargo.toml").read(),re.M).group(1))')"
rustup toolchain list | grep -q "^${MSRV}" || rustup toolchain install "$MSRV" --profile minimal
( export RUSTUP_TOOLCHAIN="$MSRV" RUSTFLAGS="" CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS="fallback" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"
  python3 scripts/check_msrv.py --rustc
  cargo check --all-targets
  cargo check --all-targets --features "$NATIVE_FEATURES"
  cargo test --no-run --lib --tests --features "$NATIVE_FEATURES"
  cargo rustc --lib --no-default-features --crate-type rlib )

step "security-audit.yml / deny: Run cargo deny check all"
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo deny --all-features check all )

step "security-audit.yml / unused-deps: cargo machete"
cargo machete

step "security-audit.yml / stub-guard: Detect panic!(STUB) / dbg!() in src/** (block)"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  # Block: panic!(STUB...) (explicit STUB marker) + dbg!() (debug residual)
  # comment 行 (// or /*) は除外
  hits=$(grep -rnE 'panic!\([^)]*STUB|dbg!\(' \
    src/ --include="*.rs" \
    --exclude-dir=bin \
    | grep -v ':[[:space:]]*//' \
    | grep -vE ':[[:space:]]*/\*' \
    || true)
  if [ -n "$hits" ]; then
    echo "❌ panic!(STUB) / dbg!() detected in src/ (production path):"
    echo "$hits"
    echo ""
    echo "Fix: use ? operator, Result<T, E>, or remove the marker."
    echo "  A stub must fail fast (todo!() / unimplemented!()), not return a silent Ok."
    exit 1
  fi
  echo "✓ No panic!(STUB) / dbg!() in src/ (excluding src/bin/)"
)

step "security-audit.yml / stub-guard: Detect todo!() / unimplemented!() in src/** (informational)"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  # Informational: todo!() / unimplemented!() は legitimate な fail-fast idiom
  # (未実装は必ず fail fast させる)
  hits=$(grep -rnE 'todo!\(|unimplemented!\(' \
    src/ --include="*.rs" \
    --exclude-dir=bin \
    | grep -v ':[[:space:]]*//' \
    | grep -vE ':[[:space:]]*/\*' \
    || true)
  if [ -n "$hits" ]; then
    count=$(echo "$hits" | wc -l | tr -d ' ')
    echo "::warning::todo!() / unimplemented!() found in src/ ($count markers, informational):"
    echo "$hits" | head -50
  else
    echo "✓ No todo!() / unimplemented!() in src/"
  fi
)

step "security-audit.yml / stub-guard: Detect TODO / FIXME / XXX / HACK (informational)"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  set -eo pipefail
  hits=$(grep -rnE 'TODO|FIXME|XXX|HACK' src/ --include="*.rs" || true)
  if [ -n "$hits" ]; then
    echo "::warning::TODO/FIXME/XXX/HACK found in src/ (informational, not blocking):"
    echo "$hits" | head -50
  else
    echo "✓ No TODO/FIXME/XXX/HACK in src/"
  fi
)

step "fuzz.yml / build every fuzz target (nightly; the replay needs the runner)"
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (cd fuzz && cargo +nightly fuzz build)
else
  echo "skip: nightly / cargo-fuzz not installed" >&2
fi

step "public-text.yml / public-text: Public text (ALICE-Law checker at the pin; tree, then the commits)"
python3 scripts/test_public_text.py
scripts/public_text.sh

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test / bench suites skipped)"; exit 0
fi

step "ci.yml / float-environment: Float environment (debug and release)"
scripts/run_tests.sh float_environment=2 -- cargo test --test float_environment
scripts/run_tests.sh float_environment=2 -- cargo test --release --test float_environment

step "ci.yml / float-environment: Effective rustc flags (default, full native feature set, no default features)"
python3 scripts/build_flags_check.py -- cargo check --release --lib
python3 scripts/build_flags_check.py -- cargo check --release --lib --features "ffi,simd,parallel,safetensors,db"
python3 scripts/build_flags_check.py -- cargo rustc --release --lib --no-default-features --crate-type rlib

step "ci.yml / test: Test (default)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo test --lib --tests )

step "ci.yml / test: Test (simd)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo test --lib --tests --features simd )

step "ci.yml / test: Test (full native feature set)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo test --lib --tests --features "$NATIVE_FEATURES" )

step "ci.yml / neon: Test (neon)"
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo test --lib --tests --features "$NATIVE_FEATURES,neon" )

step "security-audit.yml / audit: Run cargo audit"
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  # RUSTSEC-2026-0235 (rkyv 0.7 Rc/Arc archive validation): rkyv is reached only
  # through alice-db (db feature), whose archived types (SegmentMetadata / DataSegment /
  # ModelType / DataType) hold no Rc / Arc, so the advisory does not apply. This crate
  # neither depends on rkyv nor archives anything; scripts/no_direct_rkyv.py fails as soon
  # as either changes. The root fix is alice-db moving off rkyv 0.7 (pending); the entry
  # goes when that release is consumed.
  # (pyo3 0.29 / tokenizers 0.23 bump で 2025-0020 / 2026-0176 / 2026-0177 / number_prefix は解消済)
  cargo audit --deny yanked \
    --ignore RUSTSEC-2024-0436 \
    --ignore RUSTSEC-2025-0141 \
    --ignore RUSTSEC-2026-0192 \
    --ignore RUSTSEC-2026-0235
)

step "semver-checks.yml / semver-checks: semver gate (tests)"
python3 scripts/test_semver_gate.py

step "semver-checks.yml / semver-checks: semver gate (latest crates.io release, every consumer feature set)"
need cargo-semver-checks "cargo install cargo-semver-checks --version 0.50.0 --locked"
python3 scripts/semver_gate.py --config .github/semver-gate.toml --target-dir "${CARGO_TARGET_DIR:-target}/semver"

echo; echo "preflight OK"
