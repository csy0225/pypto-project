#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 5 ]; then
  echo \
    "usage: $0 <root> <parent-source> <reference-overlay> <candidate-overlay> <tag>" \
    >&2
  exit 2
fi

ROOT=$1
PARENT=$2
REFERENCE_OVERLAY=$3
CANDIDATE_OVERLAY=$4
TAG=$5
OUT=$ROOT/sources/shared-experiment-$TAG
EXPECTED_REFERENCE_DECODE_SHA=65b0b8bf139aa40a5cf67317148dc16193ff22a81b394fbfe86e31ea05623e08
EXPECTED_CANDIDATE_DECODE_SHA=572ea2a2b0ceab8952cbd8d7c1f383351fb877e11b01b4d68a60697f4508576b

test -d "$ROOT"
test -d "$ROOT/tmp"
test -d "$ROOT/sources"
test -d "$PARENT"
test -d "$REFERENCE_OVERLAY"
test -d "$CANDIDATE_OVERLAY"
test -s "$PARENT/SOURCE_SHA256SUMS"
test -s "$REFERENCE_OVERLAY/OVERLAY_SHA256SUMS"
test -s "$CANDIDATE_OVERLAY/OVERLAY_SHA256SUMS"
[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]]

verify_source() {
  local source=$1
  (cd "$source" && sha256sum -c SOURCE_SHA256SUMS)
}

verify_output() {
  local output=$1
  test -d "$output/reference"
  test -d "$output/candidate"
  verify_source "$output/reference"
  verify_source "$output/candidate"
  test ! -w "$output/reference"
  test ! -w "$output/candidate"
  test "$(
    sha256sum "$output/reference/models/step3p5/decode_fwd.py" \
      | awk '{print $1}'
  )" = "$EXPECTED_REFERENCE_DECODE_SHA"
  test "$(
    sha256sum "$output/candidate/models/step3p5/decode_fwd.py" \
      | awk '{print $1}'
  )" = "$EXPECTED_CANDIDATE_DECODE_SHA"
  test "$(
    sha256sum \
      "$output/reference/tools/step3p5/analyze_five_layer_moe_dfx.py" \
      | awk '{print $1}'
  )" = "$(
    sha256sum \
      "$output/candidate/tools/step3p5/analyze_five_layer_moe_dfx.py" \
      | awk '{print $1}'
  )"
}

verify_source "$PARENT"
(cd "$REFERENCE_OVERLAY" && sha256sum -c OVERLAY_SHA256SUMS)
(cd "$CANDIDATE_OVERLAY" && sha256sum -c OVERLAY_SHA256SUMS)

if [ -e "$OUT" ]; then
  verify_output "$OUT"
  printf '%s\n' \
    "reference=$OUT/reference" \
    "candidate=$OUT/candidate"
  exit 0
fi

STAGING=$(
  mktemp -d "$ROOT/tmp/shared-source-$TAG.staging.XXXXXXXX"
)
BUNDLE=$STAGING/shared-experiment-$TAG
mkdir -p "$BUNDLE"

copy_overlay() {
  local overlay=$1
  local destination=$2
  local relative
  while IFS= read -r -d '' relative; do
    install -D -m 0644 \
      "$overlay/$relative" \
      "$destination/$relative"
  done < <(
    find "$overlay" -type f \
      ! -name OVERLAY_SHA256SUMS \
      -printf '%P\0' \
      | sort -z
  )
}

prepare_one() {
  local role=$1
  local overlay=$2
  local destination=$BUNDLE/$role
  local parent_manifest_sha
  local overlay_manifest_sha
  local decode_sha
  local analyzer_sha

  cp -a "$PARENT" "$destination"
  chmod -R u+w "$destination"
  cp "$PARENT/SOURCE_SHA256SUMS" \
    "$destination/PARENT_SOURCE_SHA256SUMS"
  copy_overlay "$overlay" "$destination"

  parent_manifest_sha=$(
    sha256sum "$PARENT/SOURCE_SHA256SUMS" | awk '{print $1}'
  )
  overlay_manifest_sha=$(
    sha256sum "$overlay/OVERLAY_SHA256SUMS" | awk '{print $1}'
  )
  decode_sha=$(
    sha256sum "$destination/models/step3p5/decode_fwd.py" \
      | awk '{print $1}'
  )
  analyzer_sha=$(
    sha256sum \
      "$destination/tools/step3p5/analyze_five_layer_moe_dfx.py" \
      | awk '{print $1}'
  )
  printf '%s\n' \
    "schema=step3p5.five-layer-moe-shared-experiment-source.v1" \
    "role=$role" \
    "authoritative_date=2026-08-05" \
    "parent_source=$PARENT" \
    "parent_source_manifest_sha256=$parent_manifest_sha" \
    "overlay=$overlay" \
    "overlay_manifest_sha256=$overlay_manifest_sha" \
    "decode_sha256=$decode_sha" \
    "analyzer_sha256=$analyzer_sha" \
    > "$destination/SHARED_EXPERIMENT_SOURCE_PROVENANCE.txt"

  (
    cd "$destination"
    find . -type l -printf '%P\t%l\n' \
      | sort > SOURCE_SYMLINKS
    find . -type f ! -name SOURCE_SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum \
      > SOURCE_SHA256SUMS
    sha256sum -c SOURCE_SHA256SUMS
  )
}

prepare_one reference "$REFERENCE_OVERLAY"
prepare_one candidate "$CANDIDATE_OVERLAY"

mv "$BUNDLE" "$OUT"
chmod -R a-w "$OUT"
verify_output "$OUT"

printf '%s\n' \
  "reference=$OUT/reference" \
  "candidate=$OUT/candidate"
