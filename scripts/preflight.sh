#!/usr/bin/env bash
# scripts/preflight.sh — local reproduction of the CI gates before `git push`.
# Every command is the one CI runs. A step this file does not cover is a step
# that can only fail remotely — when a workflow changes, regenerate this file
# in the same commit (checks no workflow runs go in scripts/preflight.local.sh).
# Run `--quick` before every push.
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the test / bench suites)
# generator rev 5d880c90b0eda591eac24285a4b3d36de84fe794
set -euo pipefail
cd "$(dirname "$0")/.."
# --- heavy slot (begin) ---
# ALICE_HEAVY_SLOTS (default 2, 1..16) heavy runs at a time on this machine. Slot k is a lock the kernel holds (flock
# on $ALICE_HEAVY_SLOT_DIR/lock.<k>, default $HOME/.cache/alice-heavy-slot) for as long as its holder lives. A small
# keeper process (perl) takes it and exits within a second of the holder's death, so a killed holder frees the slot
# and no child of the holder keeps it. Waiters queue in arrival order (FIFO): each keeper leaves a ticket in queue/
# (its start time and pid), and only the oldest live ticket may take a free slot; a ticket whose keeper is gone (a
# crash) is skipped and removed. A waiter prints its place and its wait once a minute. `info.<k>` only names the
# holder of slot k. A wait past ALICE_HEAVY_SLOT_TIMEOUT seconds (default 7200) fails; the holder's own nested runs
# (ALICE_HEAVY_SLOT_HELD) run inside its slot.
heavy_slot_dir() { printf '%s\n' "${ALICE_HEAVY_SLOT_DIR:-$HOME/.cache/alice-heavy-slot}"; }
heavy_slot_count() {
  case "${ALICE_HEAVY_SLOTS:-2}" in
    [1-9] | 1[0-6]) printf '%s\n' "${ALICE_HEAVY_SLOTS:-2}" ;;
    *) echo "heavy slot: ALICE_HEAVY_SLOTS must be an integer from 1 to 16 (got '${ALICE_HEAVY_SLOTS:-}')" >&2; return 1 ;;
  esac
}
# "label pid (tool)" of every held slot, joined with ", "
heavy_slot_holders() {
  local d=$1 f out=""
  for f in "$d"/info.*; do
    [ -s "$f" ] || continue
    out="${out:+$out, }$(sed -n 3p "$f") pid $(sed -n 1p "$f") ($(sed -n 4p "$f"))"
  done
  printf '%s\n' "${out:-nobody}"
}
heavy_slot_acquire() {
  local label=$1 tool=$2 d n k f line err limit="${ALICE_HEAVY_SLOT_TIMEOUT:-7200}"
  d=$(heavy_slot_dir)
  n=$(heavy_slot_count) || return 1
  if [ -n "${ALICE_HEAVY_SLOT_HELD:-}" ] && kill -0 "$ALICE_HEAVY_SLOT_HELD" 2>/dev/null; then
    for f in "$d"/info.*; do
      [ "$(sed -n 1p "$f" 2>/dev/null)" = "$ALICE_HEAVY_SLOT_HELD" ] && return 0
    done
  fi
  if ! err=$(mkdir -p "$d/queue" 2>&1); then
    echo "heavy slot: cannot create $d: ${err:-not writable}" >&2
    return 1
  fi
  for k in $(seq 0 $((n - 1))); do
    if ! err=$({ : >> "$d/lock.$k"; } 2>&1); then
      echo "heavy slot: cannot create $d: ${err:-not writable}" >&2
      return 1
    fi
  done
  exec 8< <(exec perl -MFcntl=:flock -MTime::HiRes=time,sleep -e '
    my ($dir, $n, $limit) = @ARGV; my $parent = getppid(); $| = 1;
    my $q = "$dir/queue";
    my $me = sprintf("%017.6f-%d", time, $$);
    my $ticket = "$q/$me";
    open(my $tf, ">", $ticket) or do { print "error $!\n"; exit 2 }; close $tf;
    END { unlink $ticket if defined $ticket }
    my @fh;
    for my $k (0 .. $n - 1) { open($fh[$k], ">>", "$dir/lock.$k") or do { print "error $!\n"; exit 2 } }
    print "pid $$\n";
    my $start = time; my $note = 0;
    while (1) {
      opendir(my $dh, $q) or do { print "error $!\n"; exit 2 };
      my @live;
      for my $t (sort grep { /^\d/ } readdir $dh) {
        my ($p) = $t =~ /-(\d+)$/;
        if ($p && ($p == $$ || kill(0, $p))) { push @live, $t } else { unlink "$q/$t" }
      }
      closedir $dh;
      my ($place) = grep { $live[$_] eq $me } 0 .. $#live;
      if (defined $place && $place == 0) {
        for my $k (0 .. $n - 1) {
          next unless flock($fh[$k], LOCK_EX | LOCK_NB);
          unlink $ticket; undef $ticket;
          print "held $k\n"; close(STDOUT);
          sleep 1 while getppid() == $parent;
          exit 0;
        }
      }
      my $w = int(time - $start);
      if ($w >= $limit) { print "timeout\n"; exit 1 }
      if ($w >= $note) { printf "wait %d %d\n", ($place // 0) + 1, $w; $note += 60 }
      sleep 0.2;
      exit 3 if getppid() != $parent;
    }
  ' "$d" "$n" "$limit")
  while IFS= read -r line <&8; do
    case "$line" in
      "pid "*) HEAVY_SLOT_KEEPER=${line#pid } ;;
      "wait "*)
        set -- $line
        case "$2" in 1) k=1st ;; 2) k=2nd ;; 3) k=3rd ;; *) k="$2th" ;; esac
        echo "heavy slot: waiting ($k in the queue, $3 s) for $(heavy_slot_holders "$d")" >&2
        ;;
      "held "*)
        exec 8<&-
        HEAVY_SLOT_INDEX=${line#held }
        printf '%s\n%s\n%s\n%s\n' "$$" "$(date +%s)" "$label" "$tool" > "$d/info.$HEAVY_SLOT_INDEX"
        ALICE_HEAVY_SLOT_HELD=$$
        export ALICE_HEAVY_SLOT_HELD
        return 0
        ;;
      timeout)
        exec 8<&-
        echo "heavy slot: waited ${limit}s for $(heavy_slot_holders "$d"); giving up (ALICE_HEAVY_SLOT_TIMEOUT=$limit)" >&2
        return 1
        ;;
      "error "*)
        exec 8<&-
        echo "heavy slot: cannot open $d/lock: ${line#error }" >&2
        return 1
        ;;
    esac
  done
  exec 8<&-
  echo "heavy slot: the lock keeper stopped (is perl on PATH?)" >&2
  return 1
}
heavy_slot_release() {
  local d
  d=$(heavy_slot_dir)
  if [ "${ALICE_HEAVY_SLOT_HELD:-}" = "$$" ] && [ -n "${HEAVY_SLOT_KEEPER:-}" ]; then
    # the info first: once the keeper is gone the next holder writes its own
    rm -f "$d/info.${HEAVY_SLOT_INDEX:-}"
    kill "$HEAVY_SLOT_KEEPER" 2>/dev/null || true
    HEAVY_SLOT_KEEPER=
  fi
}
# memory: `heavy_slot_watch <pid> &` sums the RSS of that process and its descendants about once a second
# (ALICE_HEAVY_SLOT_RSS_POLL) and past ALICE_HEAVY_SLOT_RSS_CAP_MIB (default 8192 = 8 GiB) kills the tree (itself
# aside) and prints `[heavy-slot] RSS cap <N>GiB exceeded by <pid> <cmd> (tree <M> MiB)`. It also sums the trees of
# every held slot; past ALICE_HEAVY_SLOT_RSS_TOTAL_MIB (default 20480 = 20 GiB) the watcher whose slot holds the
# largest tree kills its own tree and prints `[heavy-slot] RSS total cap ...` (every slot's watcher looks, so no
# coordinator is needed). It returns once the process is gone. A time limit does not bound memory (a test that loops
# while pushing fills the machine inside its timeout), this does. A `ps` that cannot list the tree turns the cap off
# with a note, not silently.
heavy_slot_cap_ok() {
  case "${ALICE_HEAVY_SLOT_RSS_CAP_MIB:-8192}" in
    '' | *[!0-9]* | 0) echo "heavy slot: ALICE_HEAVY_SLOT_RSS_CAP_MIB must be a positive integer" >&2; return 2 ;;
  esac
  case "${ALICE_HEAVY_SLOT_RSS_TOTAL_MIB:-20480}" in
    '' | *[!0-9]* | 0) echo "heavy slot: ALICE_HEAVY_SLOT_RSS_TOTAL_MIB must be a positive integer" >&2; return 2 ;;
  esac
}
# from a `ps -A -o pid=,ppid=,rss=,comm=` listing on stdin: "<pid> <rss KiB> <command>" for the process $1 and its
# descendants, leaving out $2 and its descendants
heavy_slot_tree_in() {
  awk -v root="$1" -v skip="${2:-}" '
    { pid[NR] = $1; ppid[NR] = $2; rss[NR] = $3; $1 = $2 = $3 = ""; cmd[NR] = substr($0, 4) }
    END {
      in_tree[root] = 1; grew = 1
      while (grew) {
        grew = 0
        for (i = 1; i <= NR; i++)
          if (!(pid[i] in in_tree) && pid[i] != skip && (ppid[i] in in_tree)) { in_tree[pid[i]] = 1; grew = 1 }
      }
      for (i = 1; i <= NR; i++) if (pid[i] in in_tree) print pid[i], rss[i], cmd[i]
    }'
}
heavy_slot_tree() { ps -A -o pid=,ppid=,rss=,comm= | heavy_slot_tree_in "$@"; }
# whether $1 runs (a zombie has ended: a watcher started by the process it watches outlives it until reaped)
heavy_slot_alive() {
  local st
  st=$(ps -o stat= -p "$1" 2>/dev/null) || return 1
  case "$st" in '' | *Z*) return 1 ;; esac
}
heavy_slot_watch() {
  local root=$1 cap="${ALICE_HEAVY_SLOT_RSS_CAP_MIB:-8192}" total_cap="${ALICE_HEAVY_SLOT_RSS_TOTAL_MIB:-20480}" \
    poll="${ALICE_HEAVY_SLOT_RSS_POLL:-1}" d self snap tree sum top f h hsum total largest largest_sum slots \
    largest_new largest_new_sum victim \
    peak=0 samples=0 noted=0
  d=$(heavy_slot_dir)
  self=$(exec sh -c 'echo "$PPID"')
  if kill -0 "$root" 2>/dev/null && ! heavy_slot_alive "$root"; then
    echo "[heavy-slot] note: ps does not show pid $root here, so the memory cap is off" >&2
    return 0
  fi
  while heavy_slot_alive "$root"; do
    snap=$(ps -A -o pid=,ppid=,rss=,comm= 2>/dev/null) || snap=""
    tree=$(printf '%s\n' "$snap" | heavy_slot_tree_in "$root" "$self" 2>/dev/null) || tree=""
    if [ -z "$tree" ]; then
      if [ "$noted" = 0 ] && heavy_slot_alive "$root"; then
        echo "[heavy-slot] note: ps does not list pid $root here, so the memory cap is off" >&2
        noted=1
      fi
    else
      sum=$(printf '%s\n' "$tree" | awk '{ s += $2 } END { printf "%d", s / 1024 }')
      samples=$((samples + 1))
      if [ "$sum" -gt "$peak" ]; then peak=$sum; fi
      if [ "$sum" -gt "$cap" ]; then
        top=$(printf '%s\n' "$tree" | sort -k2,2nr | head -1)
        # shellcheck disable=SC2046 # one pid per word
        kill -9 $(printf '%s\n' "$tree" | cut -d" " -f1) 2>/dev/null || true
        echo "[heavy-slot] RSS cap $(awk -v c="$cap" 'BEGIN { printf "%g", c / 1024 }')GiB exceeded by ${top%% *} $(printf '%s\n' "$top" | cut -d" " -f3-) (tree $sum MiB)" >&2
        return 137
      fi
      # every held slot's tree (by its holder), the total, the holder of the largest, and the largest of the
      # slots that have a watcher: `info` without a number is the one slot of a copy of this block from before
      # the slots were numbered (an older generated preflight), which has no watcher, so the watcher of the
      # largest numbered slot stops it when it is the largest
      total=0 largest="" largest_sum=-1 largest_new="" largest_new_sum=-1 slots=0
      for f in "$d"/info.* "$d/info"; do
        h=$(sed -n 1p "$f" 2>/dev/null) || continue
        [ -n "$h" ] && kill -0 "$h" 2>/dev/null || continue
        hsum=$(printf '%s\n' "$snap" | heavy_slot_tree_in "$h" | awk '{ s += $2 } END { printf "%d", s / 1024 }')
        total=$((total + hsum)) slots=$((slots + 1))
        if [ "$hsum" -gt "$largest_sum" ]; then largest=$h largest_sum=$hsum; fi
        if [ "$f" != "$d/info" ] && [ "$hsum" -gt "$largest_new_sum" ]; then largest_new=$h largest_new_sum=$hsum; fi
      done
      if [ "$total" -gt "$total_cap" ] && [ -n "$largest" ]; then
        victim=""
        if [ "$largest" = "${ALICE_HEAVY_SLOT_HELD:-}" ]; then
          victim=$tree
        elif [ "$largest" = "$(sed -n 1p "$d/info" 2>/dev/null)" ] && [ "$largest_new" = "${ALICE_HEAVY_SLOT_HELD:-}" ]; then
          victim=$(printf '%s\n' "$snap" | heavy_slot_tree_in "$largest")
        fi
        if [ -n "$victim" ]; then
          top=$(printf '%s\n' "$victim" | sort -k2,2nr | head -1)
          # shellcheck disable=SC2046 # one pid per word
          kill -9 $(printf '%s\n' "$victim" | cut -d" " -f1) 2>/dev/null || true
          echo "[heavy-slot] RSS total cap $(awk -v c="$total_cap" 'BEGIN { printf "%g", c / 1024 }')GiB exceeded ($total MiB in $slots slots), killed the largest tree ${top%% *} $(printf '%s\n' "$top" | cut -d" " -f3-) ($largest_sum MiB)" >&2
          [ "$largest" = "${ALICE_HEAVY_SLOT_HELD:-}" ] && return 137
        fi
      fi
    fi
    sleep "$poll"
  done
  if [ "${ALICE_HEAVY_SLOT_RSS_REPORT:-}" = 1 ]; then echo "[heavy-slot] RSS peak $peak MiB over $samples samples" >&2; fi
  return 0
}
# --- heavy slot (end) ---
heavy_slot_acquire "$(basename "$PWD")" preflight || exit 1
gh_files=$(mktemp -d)
gh_local=0
[ -z "${GITHUB_ENV:-}" ] && gh_local=1
export GITHUB_OUTPUT="${GITHUB_OUTPUT:-$gh_files/output}" GITHUB_STEP_SUMMARY="${GITHUB_STEP_SUMMARY:-$gh_files/summary}"
trap 'heavy_slot_release; rm -rf "$gh_files"' EXIT
# gh_step <job>: this job's $GITHUB_ENV / $GITHUB_PATH, with what its earlier steps wrote loaded
gh_step() {
  [ "$gh_local" = 1 ] || return 0
  export GITHUB_ENV="$gh_files/$1.env" GITHUB_PATH="$gh_files/$1.path"
  touch "$GITHUB_ENV" "$GITHUB_PATH"
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      '') ;;
      *'<<'*) echo "preflight: a multi-line \$GITHUB_ENV value is not reproduced here: $line" >&2; return 1 ;;
      [A-Za-z_]*=*) export "${line%%=*}=${line#*=}" ;;
      *) echo "preflight: not a NAME=value line in \$GITHUB_ENV: $line" >&2; return 1 ;;
    esac
  done < "$GITHUB_ENV"
  while IFS= read -r line || [ -n "$line" ]; do [ -n "$line" ] && PATH="$line:$PATH"; done < "$GITHUB_PATH"
  export PATH
}
heavy_slot_cap_ok || exit 2
heavy_slot_watch "$$" >/dev/null &

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
# `cargo clippy` reuses fresh `cargo check` artifacts and then lints nothing;
# touching the crate roots invalidates only this repo's fingerprints.
relint() { git ls-files | grep -E '(^|/)src/(lib|main)\.rs$' | xargs -r touch; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1 ($2)" >&2; exit 1; }; }
has_toolchain() { grep -q "^$1" <<<"$(rustup toolchain list)"; }

# Steps CI runs that this file cannot reproduce locally (they can only fail remotely):
#   - ci.yml:float-environment:run_tests.sh and build configuration guard (tests, then check) (no cargo / grep)
#   - ci.yml:msrv: uses actions/setup-python@v6 (an action this file cannot run)
#   - ci.yml:readme-sync: uses actions/setup-python@v6 (an action this file cannot run)
#   - ci.yml:law-g1-banned-list: uses actions/setup-python@v6 (an action this file cannot run)
#   - security-audit.yml:changes:decide (reads AFTER, BEFORE, CREATED, EVENT, FORCED, PR_BASE, PR_HEAD, a value only the runner has)
#   - security-audit.yml:audit:Install cargo-audit (needs network / runner-only)
#   - security-audit.yml:deny:Install cargo-deny (needs network / runner-only)
#   - security-audit.yml:gate:gate (reads NEEDS, a value only the runner has)
#   - fuzz.yml:changes:decide (reads AFTER, BEFORE, CREATED, EVENT, FORCED, PR_BASE, PR_HEAD, a value only the runner has)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_tensor_deserialize] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_tensor_deserialize] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_tensor_deserialize] (continue-on-error, runner-only (open-ended or unprovable-bounded fuzz run))
#   - fuzz.yml:fuzz:Report crash (informational) [target=fuzz_tensor_deserialize] (its `if:` runs on what happened earlier on the runner)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_model_forward] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_model_forward] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_model_forward] (continue-on-error, runner-only (open-ended or unprovable-bounded fuzz run))
#   - fuzz.yml:fuzz:Report crash (informational) [target=fuzz_model_forward] (its `if:` runs on what happened earlier on the runner)
#   - fuzz.yml:fuzz:Install cargo-fuzz [target=fuzz_weights_load] (needs network / runner-only)
#   - fuzz.yml:fuzz:Set fuzz duration [target=fuzz_weights_load] (no cargo / grep)
#   - fuzz.yml:fuzz:Run fuzz target (time-boxed) [target=fuzz_weights_load] (continue-on-error, runner-only (open-ended or unprovable-bounded fuzz run))
#   - fuzz.yml:fuzz:Report crash (informational) [target=fuzz_weights_load] (its `if:` runs on what happened earlier on the runner)
#   - fuzz.yml:gate:gate (reads NEEDS, a value only the runner has)

need actionlint "brew install actionlint"
need cargo-audit "cargo install cargo-audit --locked"
need cargo-deny "cargo install cargo-deny --locked"
need cargo-hack "cargo install cargo-hack --locked"
need cargo-machete "cargo install cargo-machete --locked"
has_toolchain 1.87 || { echo "missing toolchain 1.87 (rustup toolchain install 1.87)" >&2; exit 1; }

step "ci.yml / float-environment: Shared files equal ALICE-Law's table"
(
  gh_step ci.yml.float-environment
(
  export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"
  PY=$(command -v python3 || command -v python)
  "$PY" scripts/test_shared_files.py
  scripts/shared_files.sh
)
)

step "ci.yml / float-environment: G11 table matches the declared gates"
(
  gh_step ci.yml.float-environment
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; scripts/g11_table.sh )
)

step "ci.yml / float-environment: G11 self-check (ALICE-Law at the shared-files pin)"
(
  gh_step ci.yml.float-environment
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; scripts/g11_self_check.sh )
)

step "ci.yml / float-environment: Effective rustc flags (default, full native feature set, no default features)"
(
  gh_step ci.yml.float-environment
(
  export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"
  python3 scripts/build_flags_check.py -- cargo check --release --lib
  python3 scripts/build_flags_check.py -- cargo check --release --lib --features "$NATIVE_FEATURES"
  python3 scripts/build_flags_check.py -- cargo rustc --release --lib --no-default-features --crate-type rlib
)
)

step "ci.yml / neon: Clippy (neon, all targets, pedantic + nursery)"
(
  gh_step ci.yml.neon
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo clippy --features "$NATIVE_FEATURES,neon" --all-targets -- -W clippy::pedantic -W clippy::nursery -D warnings )
)

step "ci.yml / no_std: Build (no_std, host)"
(
  gh_step ci.yml.no_std
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo rustc --lib --no-default-features --crate-type rlib )
)

step "ci.yml / no_std: Build (no_std + simd, host)"
(
  gh_step ci.yml.no_std
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo rustc --lib --no-default-features --features simd --crate-type rlib )
)

step "ci.yml / no_std: Build (no_std, bare-metal thumbv7em-none-eabihf)"
(
  gh_step ci.yml.no_std
grep -q '^thumbv7em-none-eabihf$' <<<"$(rustup target list --installed)" || rustup target add thumbv7em-none-eabihf
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo rustc --lib --no-default-features --crate-type rlib --target thumbv7em-none-eabihf )
)

step "ci.yml / no_std: Clippy (no_std, pedantic + nursery)"
(
  gh_step ci.yml.no_std
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; RUSTC_WORKSPACE_WRAPPER="$(rustup which clippy-driver)" cargo rustc --lib --no-default-features --crate-type rlib -- -W clippy::pedantic -W clippy::nursery -D warnings )
)

step "ci.yml / clippy: Clippy (default features, all targets)"
(
  gh_step ci.yml.clippy
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo clippy --all-targets -- -D warnings )
)

step "ci.yml / clippy: Clippy (full native feature set, all targets, pedantic + nursery)"
(
  gh_step ci.yml.clippy
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo clippy --features "$NATIVE_FEATURES" --all-targets -- -W clippy::pedantic -W clippy::nursery -D warnings )
)

step "ci.yml / clippy: Clippy (pyo3, lib)"
(
  gh_step ci.yml.clippy
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo clippy --lib --features pyo3 -- -D warnings )
)

step "ci.yml / clippy: Clippy canary data (generator tests)"
(
  gh_step ci.yml.clippy
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/test_gen_clippy_canary_calls.py )
)

step "ci.yml / clippy: Clippy canary data (generated from clippy.toml, no drift)"
(
  gh_step ci.yml.clippy
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/gen_clippy_canary_calls.py --check )
)

step "ci.yml / clippy: Clippy canary (tests)"
(
  gh_step ci.yml.clippy
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/test_clippy_canary.py )
)

step "ci.yml / clippy: README ALICE-Law block preflight check (tests)"
(
  gh_step ci.yml.clippy
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/test_law_readme_block_check.py )
)

step "ci.yml / clippy: Clippy canary (every planted banned call is flagged)"
(
  gh_step ci.yml.clippy
relint
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/clippy_canary.py )
)

step "ci.yml / clippy: Identifier moves (tests)"
(
  gh_step ci.yml.clippy
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/test_id_move_check.py )
)

step "ci.yml / clippy: Gated test files are run"
(
  gh_step ci.yml.clippy
(
  export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"
  python3 scripts/test_gated_tests.py
  python3 scripts/gated_tests.py
)
)

step "ci.yml / clippy: Identifier moves (the release compared with)"
(
  gh_step ci.yml.clippy
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; echo "crate=$(python3 scripts/id_move_check.py --print-base)" >> "$GITHUB_OUTPUT" )
)

step "ci.yml / clippy: Identifier moves (against the published release)"
(
  gh_step ci.yml.clippy
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/id_move_check.py )
)

step "ci.yml / feature-powerset: Powerset (std + {ffi, simd, parallel, safetensors} depth 2)"
(
  gh_step ci.yml.feature-powerset
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo hack check --lib --feature-powerset --depth 2 --exclude-features pyo3,neon --features std )
)

step "ci.yml / msrv: check_msrv oracle (検査器自身の試験)"
(
  gh_step ci.yml.msrv
( export CARGO_TERM_COLOR="always" RUSTFLAGS="" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db" RUSTUP_TOOLCHAIN="1.87" CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS="fallback"; python3 scripts/test_check_msrv.py )
)

step "ci.yml / msrv: check_msrv (rust-version と toolchain の版が一致し、実 rustc がその版か)"
(
  gh_step ci.yml.msrv
( export CARGO_TERM_COLOR="always" RUSTFLAGS="" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db" RUSTUP_TOOLCHAIN="1.87" CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS="fallback"; python3 scripts/check_msrv.py --rustc )
)

step "ci.yml / msrv: Check (default + full native feature set, all targets)"
(
  gh_step ci.yml.msrv
(
  export CARGO_TERM_COLOR="always" RUSTFLAGS="" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db" RUSTUP_TOOLCHAIN="1.87" CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS="fallback"
  cargo +1.87 check --all-targets
  cargo +1.87 check --all-targets --features "$NATIVE_FEATURES"
)
)

step "ci.yml / msrv: Build tests (full native feature set)"
(
  gh_step ci.yml.msrv
( export CARGO_TERM_COLOR="always" RUSTFLAGS="" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db" RUSTUP_TOOLCHAIN="1.87" CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS="fallback"; cargo +1.87 test --no-run --lib --tests --features "$NATIVE_FEATURES" )
)

step "ci.yml / msrv: Build (no_std rlib)"
(
  gh_step ci.yml.msrv
( export CARGO_TERM_COLOR="always" RUSTFLAGS="" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db" RUSTUP_TOOLCHAIN="1.87" CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS="fallback"; cargo +1.87 rustc --lib --no-default-features --crate-type rlib )
)

step "ci.yml / fmt: run"
(
  gh_step ci.yml.fmt
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo fmt -- --check )
)

step "ci.yml / fmt: no-direct-rkyv guard (tests)"
(
  gh_step ci.yml.fmt
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/test_no_direct_rkyv.py )
)

step "ci.yml / fmt: no-direct-rkyv guard"
(
  gh_step ci.yml.fmt
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/no_direct_rkyv.py )
)

step "ci.yml / fmt: git consumer builds alice-ml alone"
(
  gh_step ci.yml.fmt
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/test_git_consumer.py )
)

step "ci.yml / doc: Doc (full native feature set)"
(
  gh_step ci.yml.doc
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db" RUSTDOCFLAGS="-Dwarnings"; cargo doc --no-deps --features "$NATIVE_FEATURES" )
)

step "ci.yml / actionlint: actionlint"
(
  gh_step ci.yml.actionlint
actionlint .github/workflows/*.yml
)

step "ci.yml / readme-sync: readme_sync oracle (検査器自身の試験)"
(
  gh_step ci.yml.readme-sync
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/test_readme_sync.py )
)

step "ci.yml / readme-sync: readme_sync (README / README_JP / docs/MODULES.md の check)"
(
  gh_step ci.yml.readme-sync
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/readme_sync.py --check )
)

step "ci.yml / law-g1-banned-list: check_g1_banned_list oracle (検査器自身の試験)"
(
  gh_step ci.yml.law-g1-banned-list
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/test_check_g1_banned_list.py )
)

step "ci.yml / law-g1-banned-list: law_tie_mutants.py (tests)"
(
  gh_step ci.yml.law-g1-banned-list
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/test_law_tie_mutants.py )
)

step "ci.yml / law-g1-banned-list: check_g1_banned_list (clippy.toml が .github/law-g1-pin の G1 一覧を全部 ban しているか)"
(
  gh_step ci.yml.law-g1-banned-list
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; python3 scripts/check_g1_banned_list.py )
)

step "security-audit.yml / deny: Run cargo deny check all"
(
  gh_step security-audit.yml.deny
( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo deny --all-features check all )
)

step "security-audit.yml / unused-deps: cargo machete"
(
  gh_step security-audit.yml.unused-deps
cargo machete
)

step "security-audit.yml / stub-guard: Detect panic!(STUB) / dbg!() in src/** (block)"
(
  gh_step security-audit.yml.stub-guard
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
)

step "security-audit.yml / stub-guard: Detect todo!() / unimplemented!() in src/** (informational)"
(
  gh_step security-audit.yml.stub-guard
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
)

step "security-audit.yml / stub-guard: Detect TODO / FIXME / XXX / HACK (informational)"
(
  gh_step security-audit.yml.stub-guard
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
)

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_tensor_deserialize]"
(
  gh_step fuzz.yml.fuzz
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_tensor_deserialize"
  )
else
  echo 'PREFLIGHT-SKIP: fuzz.yml / fuzz: Build fuzz target [target=fuzz_tensor_deserialize]: nightly / cargo-fuzz not installed' >&2
fi
)

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_model_forward]"
(
  gh_step fuzz.yml.fuzz
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_model_forward"
  )
else
  echo 'PREFLIGHT-SKIP: fuzz.yml / fuzz: Build fuzz target [target=fuzz_model_forward]: nightly / cargo-fuzz not installed' >&2
fi
)

step "fuzz.yml / fuzz: Build fuzz target [target=fuzz_weights_load]"
(
  gh_step fuzz.yml.fuzz
if has_toolchain nightly && cargo +nightly fuzz --version >/dev/null 2>&1; then
  (
    export CARGO_TERM_COLOR="always"
    cd fuzz
    cargo +nightly fuzz build "fuzz_weights_load"
  )
else
  echo 'PREFLIGHT-SKIP: fuzz.yml / fuzz: Build fuzz target [target=fuzz_weights_load]: nightly / cargo-fuzz not installed' >&2
fi
)

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick OK (test / bench suites skipped)"; exit 0
fi

step "ci.yml / test: Test (default)"
(
  gh_step ci.yml.test
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo test --lib --tests )
)

step "ci.yml / test: Test (simd)"
(
  gh_step ci.yml.test
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db" ALICE_ML_REQUIRE_AVX2="1"; cargo test --lib --tests --features simd )
)

step "ci.yml / test: Test (full native feature set)"
(
  gh_step ci.yml.test
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db" ALICE_ML_REQUIRE_AVX2="1"; cargo test --lib --tests --features "$NATIVE_FEATURES" )
)

step "ci.yml / test: AVX2 lane test of the Law max / min (count in the log)"
(
  gh_step ci.yml.test
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db" ALICE_ML_REQUIRE_AVX2="1"; cargo test --lib --features simd law_math::tests::avx2_lanes_equal_the_scalar_helpers_over_every_pair -- --exact --nocapture )
)

step "ci.yml / test: Thread-count independence of the parallel batch"
(
  gh_step ci.yml.test
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; scripts/run_tests.sh parallel_order=2 -- cargo test --features parallel --test parallel_order )
)

step "ci.yml / test: Law edge goldens (debug, full native feature set)"
(
  gh_step ci.yml.test
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; scripts/run_tests.sh float_edge_golden=2,determinism_golden,ml_semantics_id,atml_semantics,g7_boundaries -- cargo test --features "$NATIVE_FEATURES" --test float_edge_golden --test determinism_golden --test ml_semantics_id --test atml_semantics --test g7_boundaries )
)

step "ci.yml / test: No negated fma result (syntax tree, tests then check)"
(
  gh_step ci.yml.test
(
  export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"
  cargo fmt --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml -- --check
  cargo test --quiet --locked --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml
  cargo run --quiet --locked --release --manifest-path tools/no-outer-fneg-of-fma/Cargo.toml -- --root .
  # scripts/changed_paths.py decides whether g3-ir.yml's IR job runs; its own tests
  python3 scripts/test_changed_paths.py
)
)

step "ci.yml / float-environment: Float environment (debug and release)"
(
  gh_step ci.yml.float-environment
(
  export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"
  scripts/run_tests.sh float_environment=2 -- cargo test --test float_environment
  scripts/run_tests.sh float_environment=2 -- cargo test --release --test float_environment
)
)

step "ci.yml / neon: Test (neon)"
(
  gh_step ci.yml.neon
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; cargo test --lib --tests --features "$NATIVE_FEATURES,neon" )
)

step "ci.yml / neon: Law edge goldens (debug, aarch64, neon)"
(
  gh_step ci.yml.neon
( export CARGO_TERM_COLOR="always" RUSTFLAGS="-Dwarnings" NATIVE_FEATURES="ffi,simd,parallel,safetensors,db"; scripts/run_tests.sh float_edge_golden=2,determinism_golden -- cargo test --features "$NATIVE_FEATURES,neon" --test float_edge_golden --test determinism_golden )
)

step "security-audit.yml / audit: Run cargo audit"
(
  gh_step security-audit.yml.audit
(
  export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"
  # (pyo3 0.29 / tokenizers 0.23 bump で 2025-0020 / 2026-0176 / 2026-0177 / number_prefix は解消済)
  cargo audit --deny yanked \
    --ignore RUSTSEC-2024-0436 \
    --ignore RUSTSEC-2025-0141 \
    --ignore RUSTSEC-2026-0192
)
)

step "security-audit.yml / coverage: Run coverage"
(
  gh_step security-audit.yml.coverage
(
  log="$(mktemp)"
  set +e
  {
    ( export CARGO_TERM_COLOR="always" CARGO_NET_RETRY="5" CARGO_HTTP_MULTIPLEXING="false"; cargo llvm-cov --lib --summary-only 2>&1 )
  } 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  n=$(sed $'s/\033\[[0-9;]*m//g' "${log}" | awk '$1 == "TOTAL" && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ {n = $2 - $3} END {print n + 0}')
  rm -f "${log}"
  set -e
  if [ -z "${n}" ] || [ "${n}" -eq 0 ]; then echo 'security-audit.yml / coverage: Run coverage: cargo-llvm-cov compared nothing (continue-on-error in CI; a run with no count fails)' >&2; exit 1; fi
  echo 'PREFLIGHT-INFO: security-audit.yml / coverage: Run coverage: continue-on-error in CI, '"exit ${rc}, ${n} compared" >&2
)
)

echo; echo "preflight OK"
