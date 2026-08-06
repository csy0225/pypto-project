#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 4 ]; then
  echo \
    "usage: $0 <root> <image@digest> <candidate-route-source> <tag>" \
    >&2
  exit 2
fi

ROOT=$1
IMG=$2
SOURCE=$3
TAG=$4
SCRIPTS=${CAMPAIGN_SCRIPTS:-$ROOT/scripts}
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
RUN=$ROOT/campaign/route-compile/$TAG

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]]
test -d "$ROOT"
test -d "$SOURCE"
test -d "$SCRIPTS"
test -d "$CKPT"
test -s "$SOURCE/SOURCE_SHA256SUMS"
test -x "$SCRIPTS/container_five_layer_route_compile.sh"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256SUMS)

if [ -e "$RUN" ]; then
  test -s "$RUN/runtime/compile_report.json"
  test "$(cat "$RUN/container.rc")" = 0
  echo "[route-compile] verified existing $TAG"
  exit 0
fi

mkdir -p "$RUN/runtime"
SOURCE_MANIFEST_SHA=$(
  sha256sum "$SOURCE/SOURCE_SHA256SUMS" | awk '{print $1}'
)
printf '%s\n' "$IMG" > "$RUN/image_ref.txt"
printf '%s\n' "$SOURCE_MANIFEST_SHA" \
  > "$RUN/source_manifest_sha256.txt"
date -Ins > "$RUN/started_at.txt"

DEVS=()
for index in 0 1 2 3 4 5 6 7; do
  DEVS+=(--device "/dev/davinci${index}")
done

set +e
sudo -n "$NC" run --name "moe-route-compile-$TAG" --rm \
  --net host --ipc host \
  --privileged --security-opt apparmor=unconfined \
  --env ROUTE_COMPILE_IMAGE_REF="$IMG" \
  "${DEVS[@]}" \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$CKPT":"$CKPT":ro \
  -v "$SOURCE":/workspace/pypto-lib:ro \
  -v "$SCRIPTS":/campaign-scripts:ro \
  -v "$RUN":/out \
  "$IMG" bash /campaign-scripts/container_five_layer_route_compile.sh \
  2>&1 | tee "$RUN/container.log"
RC=${PIPESTATUS[0]}
set -e

printf '%s\n' "$RC" > "$RUN/container.rc"
date -Ins > "$RUN/finished_at.txt"
if [ "$RC" -ne 0 ]; then
  exit "$RC"
fi
test -s "$RUN/runtime/compile_report.json"
echo FIVE_LAYER_0162_ROUTE_COMPILE_ONLY=PASS
