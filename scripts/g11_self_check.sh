#!/usr/bin/env bash
# scripts/g11_self_check.sh — ALICE-Law's G11 self-check on this repository: the four G11 checks
# (g11_table_gen.py --check, g11_table_check.py, shared_files_check.py --repo and
# conformance_roster_check.py --check-manifest) in one command, with ALICE-Law checked out at the
# revision .github/shared-files-pin names. g11_self_check.py refuses to run unless the checkout is that
# revision and its table and checker match the pin's SHA-256, and it refuses a G11 pass while any own
# copy is off a pending row's target. It exits 2 when it compared nothing or cannot read what it needs,
# and that fails the run like a finding does. No GitHub API is used.
#
# usage: scripts/g11_self_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."

py=$(command -v python3 || command -v python || true)
if [[ -z "$py" ]]; then
  echo "no python3 or python on PATH" >&2
  exit 2
fi

pin=.github/shared-files-pin
name=$(awk '$1 == "name" { print $2 }' "$pin")
rev=$(awk '$1 == "rev" { print $2 }' "$pin")
if [[ ! "$name" =~ ^ALICE-[A-Za-z0-9-]+$ || ! "$rev" =~ ^[0-9a-f]{40}$ ]]; then
  echo "$pin: needs 'name ALICE-<repository>' and 'rev <40 hex>'" >&2
  exit 2
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git init -q "$tmp/law"
# the checkout keeps ALICE-Law's bytes: a CRLF conversion (core.autocrlf on Windows) makes the self-check refuse
# the checkout as changed and exit 2
git -C "$tmp/law" config core.autocrlf false
if ! git -C "$tmp/law" fetch -q --depth 1 https://github.com/ext-sakamoro/ALICE-Law "$rev"; then
  echo "ALICE-Law $rev: fetch failed" >&2
  exit 2
fi
git -C "$tmp/law" checkout -q --detach FETCH_HEAD
"$py" "$tmp/law/scripts/g11_self_check.py" --repo . --name "$name"
