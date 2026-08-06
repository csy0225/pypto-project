#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 4 ]; then
  echo \
    "usage: $0 <root> <image@digest> <source-dir> <variant>" \
    >&2
  exit 2
fi

ROOT=$1
IMG=$2
SOURCE=$3
VARIANT=$4
SCRIPTS=${CAMPAIGN_SCRIPTS:-$ROOT/scripts}
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
RUN=$ROOT/campaign/tile-compile/$VARIANT
CONTAINER="moe-tile-compile-${VARIANT//[^A-Za-z0-9]/-}"

[[ "$VARIANT" =~ ^mm-n(32|64)-r(8|16|32)$ ]]
test -d "$ROOT"
test -d "$SOURCE"
test -d "$SCRIPTS"
test -d "$CKPT"
test -s "$SOURCE/SOURCE_SHA256SUMS"
test -s "$SOURCE/MOE_TILE_SWEEP_PROFILE.txt"
test -x "$SCRIPTS/container_tile_compile_probe.sh"
test -s "$SCRIPTS/audit_tile_compile_resources.py"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256SUMS >/dev/null)

audit_run() {
  python3 "$SCRIPTS/audit_tile_compile_resources.py" \
    --build-root "$RUN/runtime/build_output" \
    --profile "$SOURCE/MOE_TILE_SWEEP_PROFILE.txt" \
    --source-manifest "$SOURCE/SOURCE_SHA256SUMS" \
    --out "$RUN/resource_audit.json"
}

SOURCE_MANIFEST_SHA=$(
  sha256sum "$SOURCE/SOURCE_SHA256SUMS" | awk '{print $1}'
)
if [ -e "$RUN" ]; then
  test -s "$RUN/runtime/compile_report.json"
  test "$(cat "$RUN/container.rc")" = 0
  test "$(cat "$RUN/tile_variant.txt")" = "$VARIANT"
  test "$(cat "$RUN/image_ref.txt")" = "$IMG"
  test "$(cat "$RUN/source_manifest_sha256.txt")" = "$SOURCE_MANIFEST_SHA"
  test "$(sha256sum \
    "$SCRIPTS/container_tile_compile_probe.sh" \
    "$SCRIPTS/audit_tile_compile_resources.py")" \
    = "$(cat "$RUN/scripts_sha256.txt")"
  audit_run
  python3 - "$RUN/resource_audit.json" "$VARIANT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["passed"] is True
assert report["variant"] == sys.argv[2]
PY
  echo "[tile-compile] verified existing $VARIANT"
  exit 0
fi

mkdir -p "$RUN/runtime"
printf '%s\n' "$IMG" > "$RUN/image_ref.txt"
printf '%s\n' "$VARIANT" > "$RUN/tile_variant.txt"
printf '%s\n' "$SOURCE_MANIFEST_SHA" \
  > "$RUN/source_manifest_sha256.txt"
sha256sum \
  "$SCRIPTS/container_tile_compile_probe.sh" \
  "$SCRIPTS/audit_tile_compile_resources.py" \
  > "$RUN/scripts_sha256.txt"
date -Ins > "$RUN/started_at.txt"

DEVS=()
for index in 0 1 2 3 4 5 6 7; do
  DEVS+=(--device "/dev/davinci${index}")
done

set +e
sudo -n "$NC" run --name "$CONTAINER" --rm \
  --net host --ipc host \
  --privileged --security-opt apparmor=unconfined \
  --env TILE_VARIANT="$VARIANT" \
  --env TILE_COMPILE_IMAGE_REF="$IMG" \
  "${DEVS[@]}" \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$CKPT":"$CKPT":ro \
  -v "$SOURCE":/workspace/pypto-lib:ro \
  -v "$SCRIPTS":/campaign-scripts:ro \
  -v "$RUN":/out \
  "$IMG" bash /campaign-scripts/container_tile_compile_probe.sh \
  2>&1 | tee "$RUN/container.log"
PIPE_RC=("${PIPESTATUS[@]}")
RC=${PIPE_RC[0]}
TEE_RC=${PIPE_RC[1]}
set -e

if [ "$RC" -eq 0 ] && [ "$TEE_RC" -ne 0 ]; then
  RC=24
fi
printf '%s\n' "$RC" > "$RUN/container.rc"
date -Ins > "$RUN/finished_at.txt"
if [ "$RC" -ne 0 ]; then
  exit "$RC"
fi
test -s "$RUN/runtime/compile_report.json"
audit_run
echo MOE_TILE_COMPILE_PROBE_0162=PASS
