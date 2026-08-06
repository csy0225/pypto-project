#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 5 ]; then
  echo \
    "usage: $0 <root> <baseline-parent> <candidate-parent> <overlay> <tag>" \
    >&2
  exit 2
fi

ROOT=$1
BASELINE_PARENT=$2
CANDIDATE_PARENT=$3
OVERLAY=$4
TAG=$5
OUT=$ROOT/sources/dfx-$TAG
OVERLAY_MANIFEST=$OVERLAY/OVERLAY_SHA256SUMS
VERIFIER=$(dirname "$0")/verify_exact_tree_manifest.py

test -d "$ROOT"
test -d "$ROOT/tmp"
test -d "$ROOT/sources"
test -d "$BASELINE_PARENT"
test -d "$CANDIDATE_PARENT"
test -d "$OVERLAY"
test -s "$BASELINE_PARENT/SOURCE_SHA256SUMS"
test -s "$CANDIDATE_PARENT/SOURCE_SHA256SUMS"
test -s "$OVERLAY_MANIFEST"
test -s "$VERIFIER"
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]]

verify_source() {
  local source=$1
  # Parent trees may contain stale, unmanifested pyc from historical probes.
  # Only the manifest projection is allowed to enter a frozen campaign source.
  (cd "$source" && sha256sum -c SOURCE_SHA256SUMS)
}

verify_output() {
  local output=$1
  test -d "$output/baseline"
  test -d "$output/candidate"
  python3 "$VERIFIER" --root "$output/baseline"
  python3 "$VERIFIER" --root "$output/candidate"
  test ! -w "$output/baseline"
  test ! -w "$output/candidate"
}

verify_source "$BASELINE_PARENT"
verify_source "$CANDIDATE_PARENT"
python3 "$VERIFIER" \
  --root "$OVERLAY" \
  --manifest OVERLAY_SHA256SUMS \
  --symlink-manifest -

if [ -e "$OUT" ]; then
  verify_output "$OUT"
  printf '%s\n' \
    "baseline=$OUT/baseline" \
    "candidate=$OUT/candidate"
  exit 0
fi

STAGING=$(
  mktemp -d "$ROOT/tmp/dfx-source-$TAG.staging.XXXXXXXX"
)
BUNDLE=$STAGING/dfx-$TAG
mkdir -p "$BUNDLE"

copy_overlay() {
  local destination=$1
  local relative
  while IFS= read -r -d '' relative; do
    install -D -m 0644 \
      "$OVERLAY/$relative" \
      "$destination/$relative"
  done < <(
    find "$OVERLAY" -type f \
      ! -name OVERLAY_SHA256SUMS \
      -printf '%P\0' \
      | sort -z
  )
}

prepare_one() {
  local role=$1
  local parent=$2
  local destination=$BUNDLE/$role
  local parent_manifest_sha
  local overlay_manifest_sha

  python3 "$VERIFIER" \
    --root "$parent" \
    --materialize "$destination"
  # A frozen parent may be read-only. Staging must be writable while adding
  # provenance and an explicitly approved overlay; the final tree is frozen
  # again after exact-tree verification.
  chmod -R u+w "$destination"
  cp "$parent/SOURCE_SHA256SUMS" \
    "$destination/PARENT_SOURCE_SHA256SUMS"
  copy_overlay "$destination"

  parent_manifest_sha=$(
    sha256sum "$parent/SOURCE_SHA256SUMS" | awk '{print $1}'
  )
  overlay_manifest_sha=$(
    sha256sum "$OVERLAY_MANIFEST" | awk '{print $1}'
  )
  printf '%s\n' \
    "schema=step3p5.five-layer-moe-dfx-source.v1" \
    "role=$role" \
    "authoritative_date=2026-08-06" \
    "parent_source=$parent" \
    "parent_source_manifest_sha256=$parent_manifest_sha" \
    "overlay=$OVERLAY" \
    "overlay_manifest_sha256=$overlay_manifest_sha" \
    > "$destination/DFX_SOURCE_PROVENANCE.txt"

  (
    cd "$destination"
    find . -type l -printf '%P\t%l\n' \
      | sort > SOURCE_SYMLINKS
    find . -type f ! -name SOURCE_SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      > SOURCE_SHA256SUMS
    python3 "$VERIFIER" --root .
  )
}

prepare_one baseline "$BASELINE_PARENT"
prepare_one candidate "$CANDIDATE_PARENT"

BASELINE_ANALYZER_SHA=$(
  sha256sum \
    "$BUNDLE/baseline/tools/step3p5/analyze_five_layer_moe_dfx.py" \
    | awk '{print $1}'
)
CANDIDATE_ANALYZER_SHA=$(
  sha256sum \
    "$BUNDLE/candidate/tools/step3p5/analyze_five_layer_moe_dfx.py" \
    | awk '{print $1}'
)
test "$BASELINE_ANALYZER_SHA" = "$CANDIDATE_ANALYZER_SHA"

mv "$BUNDLE" "$OUT"
chmod -R a-w "$OUT"
verify_output "$OUT"

printf '%s\n' \
  "baseline=$OUT/baseline" \
  "candidate=$OUT/candidate"
