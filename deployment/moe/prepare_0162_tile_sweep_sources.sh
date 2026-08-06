#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <root> <base-source> <audit-script>" >&2
  exit 2
fi

ROOT=$1
BASE=$2
AUDIT=$3
OUT_ROOT=$ROOT/sources/tile

test -d "$ROOT"
test -d "$ROOT/tmp"
test -d "$BASE"
test -s "$BASE/SOURCE_SHA256SUMS"
test -s "$AUDIT"
mkdir -p "$OUT_ROOT"

verify_source() {
  local source=$1
  (cd "$source" && sha256sum -c SOURCE_SHA256SUMS >/dev/null)
}

variant_mm_n() {
  case "$1" in
    mm-n32-r8|mm-n32-r16|mm-n32-r32) printf '32\n' ;;
    mm-n64-r8|mm-n64-r16|mm-n64-r32) printf '64\n' ;;
    *) return 1 ;;
  esac
}

variant_recv_tile() {
  case "$1" in
    mm-n32-r8|mm-n64-r8) printf '8\n' ;;
    mm-n32-r16|mm-n64-r16) printf '16\n' ;;
    mm-n32-r32|mm-n64-r32) printf '32\n' ;;
    *) return 1 ;;
  esac
}

variant_axis() {
  case "$1" in
    mm-n64-r16) printf 'control\n' ;;
    mm-n32-r16) printf 'mm_n\n' ;;
    mm-n64-r8|mm-n64-r32) printf 'recv_tile\n' ;;
    mm-n32-r8|mm-n32-r32) printf 'cross_reserved\n' ;;
    *) return 1 ;;
  esac
}

verify_source "$BASE"
BASE_MANIFEST_SHA=$(sha256sum "$BASE/SOURCE_SHA256SUMS" | awk '{print $1}')

for variant in \
  mm-n32-r8 mm-n32-r16 mm-n32-r32 \
  mm-n64-r8 mm-n64-r16 mm-n64-r32; do
  OUT=$OUT_ROOT/$variant
  MM_N=$(variant_mm_n "$variant")
  RECV_TILE=$(variant_recv_tile "$variant")
  AXIS=$(variant_axis "$variant")

  if [ -e "$OUT" ]; then
    verify_source "$OUT"
    test -s "$OUT/MOE_TILE_SWEEP_PROFILE.txt"
    grep -Fx "variant=$variant" "$OUT/MOE_TILE_SWEEP_PROFILE.txt" >/dev/null
    grep -Fx "axis=$AXIS" "$OUT/MOE_TILE_SWEEP_PROFILE.txt" >/dev/null
    grep -Fx "ROUTED_GATE_MM_N_CHUNK=$MM_N" \
      "$OUT/MOE_TILE_SWEEP_PROFILE.txt" >/dev/null
    grep -Fx "RECV_TILE=$RECV_TILE" \
      "$OUT/MOE_TILE_SWEEP_PROFILE.txt" >/dev/null
    echo "[tile-source] verified existing $variant"
    continue
  fi

  STAGING=$(mktemp -d "$ROOT/tmp/tile-source-$variant.XXXXXXXX")
  DEST=$STAGING/$variant
  cp -a "$BASE" "$DEST"
  chmod -R u+w "$DEST"
  rm -f "$DEST/ACTIVATION_SWEEP_PROFILE.txt"
  DECODE=$DEST/models/step3p5/decode_fwd.py
  sed -i -E \
    "s/^ROUTED_GATE_MM_N_CHUNK = [0-9]+$/ROUTED_GATE_MM_N_CHUNK = $MM_N/" \
    "$DECODE"
  sed -i -E \
    "s/^RECV_TILE = [0-9]+$/RECV_TILE = $RECV_TILE/" \
    "$DECODE"
  cat > "$DEST/MOE_TILE_SWEEP_PROFILE.txt" <<EOF
schema=step3p5.moe.tile-sweep.v1
variant=$variant
axis=$AXIS
base_source_manifest_sha256=$BASE_MANIFEST_SHA
ROUTED_GATE_MM_K_CHUNK=512
ROUTED_GATE_MM_N_CHUNK=$MM_N
ROUTED_GATE_ACT_N_CHUNK=64
ROUTED_H_QUANT_N_CHUNK=64
ROUTED_DOWN_N_CHUNK=256
RECV_TILE=$RECV_TILE
RECV_SPECIAL_TILE=32
EOF
  (
    cd "$DEST"
    find . -type f ! -name SOURCE_SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > SOURCE_SHA256SUMS
    sha256sum -c SOURCE_SHA256SUMS >/dev/null
  )
  mv "$DEST" "$OUT"
  # This filesystem rejects renaming a read-only directory. Publish first,
  # then freeze the completed source tree in place.
  chmod -R a-w "$OUT"
  verify_source "$OUT"
  echo "[tile-source] created $variant"
done

python3 "$AUDIT" \
  --source-root "$OUT_ROOT" \
  --out "$ROOT/sources/tile_source_audit.json"
