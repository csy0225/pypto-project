#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 6 ]; then
  echo "usage: $0 <root> <image@digest> <source-dir> <source-kind> <bs> <normal|dfx>" >&2
  exit 2
fi

ROOT=$1
IMG=$2
SOURCE=$3
SOURCE_KIND=$4
BS=$5
MODE=$6
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
SCRIPTS=$ROOT/scripts
RUN_NAME="${SOURCE_KIND}-bs${BS}-${MODE}"
OUT=$ROOT/campaign/runs/$RUN_NAME
CONTAINER="moe64k-${SOURCE_KIND:0:1}-b${BS}-${MODE:0:1}"

test -d "$ROOT"
test -d "$SOURCE"
test -d "$CKPT"
test -x "$SCRIPTS/container_five_layer_case.sh"
test -s "$SCRIPTS/image_capability_probe.py"
if [ -e "$OUT" ]; then
  echo "refusing to overwrite $OUT" >&2
  exit 17
fi
mkdir -p "$OUT"

sudo -n "$NC" run --rm --net host \
  --security-opt apparmor=unconfined \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  "$IMG" bash -lc 'bash /workspace/pypto-image-audit.sh' \
  > "$OUT/image_audit.log" 2>&1
grep -Fx "IMAGE_IMMUTABLE_AUDIT=PASS" "$OUT/image_audit.log" >/dev/null

for index in 0 1 2 3 4 5 6 7; do
  if fuser "/dev/davinci${index}" >/dev/null 2>&1; then
    echo "device /dev/davinci${index} is busy" >&2
    fuser -v "/dev/davinci${index}" >&2 || true
    exit 18
  fi
done

DEVS=()
for index in 0 1 2 3 4 5 6 7; do
  DEVS+=(--device "/dev/davinci${index}")
done
NONCE=$(
  printf '%s:%s:%s:%s:%s' \
    "$IMG" "$SOURCE_KIND" "$BS" "$MODE" "$(date +%s%N)" \
    | sha256sum | awk '{print $1}'
)

date -Ins > "$OUT/started_at.txt"
printf '%s\n' "$IMG" > "$OUT/image_ref.txt"
printf '%s\n' "$NONCE" > "$OUT/run_nonce.txt"
sha256sum "$SCRIPTS/container_five_layer_case.sh" \
  "$SCRIPTS/image_capability_probe.py" \
  > "$OUT/container_script_sha256.txt"

CAPABILITY_ENV_ARGS=()
if [ "$MODE" = dfx ]; then
  CAPABILITY_ENV_ARGS=(
    --env PYPTO_REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN=1
  )
fi
set +e
sudo -n "$NC" run --name "$CONTAINER" --rm --net host --ipc host \
  --privileged --security-opt apparmor=unconfined \
  --env CASE_MODE="$MODE" \
  --env CASE_IMAGE_REF="$IMG" \
  --env ACTIVE_BATCH="$BS" \
  --env RUN_NONCE="$NONCE" \
  --env SOURCE_KIND="$SOURCE_KIND" \
  --env CASE_ITERS="${CASE_ITERS:-3}" \
  --env CASE_WARMUP="${CASE_WARMUP:-2}" \
  "${CAPABILITY_ENV_ARGS[@]}" \
  "${DEVS[@]}" \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$CKPT":"$CKPT":ro \
  -v "$SOURCE":/workspace/pypto-lib:ro \
  -v "$SCRIPTS":/campaign-scripts:ro \
  -v "$OUT":/out \
  "$IMG" bash /campaign-scripts/container_five_layer_case.sh \
  2>&1 | tee "$OUT/container.log"
RC=${PIPESTATUS[0]}
set -e

printf '%s\n' "$RC" > "$OUT/container.rc"
date -Ins > "$OUT/finished_at.txt"
if [ "$RC" -ne 0 ]; then
  exit "$RC"
fi
test -s "$OUT/capability_report.json"
for index in 0 1 2 3 4 5 6 7; do
  if fuser "/dev/davinci${index}" >/dev/null 2>&1; then
    echo "device /dev/davinci${index} remained busy after $RUN_NAME" >&2
    exit 19
  fi
done
echo FIVE_LAYER_0162_CASE=PASS
