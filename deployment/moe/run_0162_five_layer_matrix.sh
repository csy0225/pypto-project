#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

if [ "$#" -ne 7 ]; then
  echo "usage: $0 <root> <image@digest> <source-dir> <source-kind> <round> <normal|dfx> <batch>" >&2
  exit 2
fi

ROOT=$1
IMG=$2
SOURCE=$3
SOURCE_KIND=$4
ROUND=$5
MODE=$6
BATCH=$7
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
SCRIPTS=${CAMPAIGN_SCRIPTS:-$ROOT/scripts}
case "$SOURCE_KIND" in
  baseline|candidate) ;;
  *)
    echo "invalid source kind=$SOURCE_KIND" >&2
    exit 3
    ;;
esac
case "$BATCH" in
  1|2|4|7|8|16) ;;
  *)
    echo "invalid batch=$BATCH" >&2
    exit 3
    ;;
esac
RUN_NAME="${SOURCE_KIND}-r${ROUND}-${MODE}-bs${BATCH}-64k"
OUT=$ROOT/campaign/runs/$RUN_NAME
CONTAINER="moe64k-${SOURCE_KIND:0:1}-r${ROUND}-${MODE:0:1}-b${BATCH}"

test -d "$ROOT"
test -d "$SOURCE"
test -d "$CKPT"
test -d "$SCRIPTS"
test -x "$SCRIPTS/container_five_layer_matrix.sh"
test -s "$SCRIPTS/five_layer_matrix.py"
test -s "$SCRIPTS/validate_five_layer_case.py"
test -s "$SCRIPTS/verify_exact_tree_manifest.py"
test -s "$SCRIPTS/SCRIPTS_SHA256SUMS"
python3 "$SCRIPTS/verify_exact_tree_manifest.py" \
  --root "$SCRIPTS" \
  --manifest SCRIPTS_SHA256SUMS \
  --symlink-manifest -
python3 "$SCRIPTS/verify_exact_tree_manifest.py" --root "$SOURCE"
if [ -e "$OUT" ]; then
  echo "refusing to overwrite $OUT" >&2
  exit 17
fi
mkdir -p "$OUT"

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
    "$IMG" "$SOURCE_KIND" "$ROUND" "$MODE:$BATCH" "$(date +%s%N)" \
    | sha256sum | awk '{print $1}'
)

GOLDEN_ARGS=()
GOLDEN=${CAMPAIGN_GOLDEN_ROOT:-$ROOT/campaign/golden/heterogeneous-64k}
if [ "$MODE" = dfx ]; then
  test -d "$GOLDEN"
  GOLDEN_ARGS=(-v "$GOLDEN":/golden:ro)
fi
DFX_PROFILE_ARGS=()
if [ -n "${MOE_DFX_PROFILE:-}" ]; then
  DFX_PROFILE_ARGS=(
    --env PYPTO_MOE_DFX_PROFILE="$MOE_DFX_PROFILE"
  )
fi
date -Ins > "$OUT/started_at.txt"
printf '%s\n' "$IMG" > "$OUT/image_ref.txt"
printf '%s\n' "$NONCE" > "$OUT/run_nonce.txt"
npu-smi info > "$OUT/npu_smi_before.txt"
sha256sum \
  "$SCRIPTS/container_five_layer_matrix.sh" \
  "$SCRIPTS/five_layer_matrix.py" \
  "$SCRIPTS/validate_five_layer_case.py" \
  "$SCRIPTS/verify_exact_tree_manifest.py" \
  > "$OUT/matrix_scripts_sha256.txt"

set +e
sudo -n "$NC" run --name "$CONTAINER" --rm --net host --ipc host \
  --privileged --security-opt apparmor=unconfined \
  --env MATRIX_MODE="$MODE" \
  --env MATRIX_ROUND="$ROUND" \
  --env MATRIX_BATCH="$BATCH" \
  --env MATRIX_IMAGE_REF="$IMG" \
  --env RUN_NONCE="$NONCE" \
  --env SOURCE_KIND="$SOURCE_KIND" \
  --env MATRIX_ITERS="${MATRIX_ITERS:-20}" \
  --env MATRIX_WARMUP="${MATRIX_WARMUP:-3}" \
  "${DFX_PROFILE_ARGS[@]}" \
  "${DEVS[@]}" \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$CKPT":"$CKPT":ro \
  -v "$SOURCE":/workspace/pypto-lib:ro \
  -v "$SCRIPTS":/campaign-scripts:ro \
  -v "$OUT":/out \
  "${GOLDEN_ARGS[@]}" \
  "$IMG" bash /campaign-scripts/container_five_layer_matrix.sh \
  2>&1 | tee "$OUT/container.log"
RC=${PIPESTATUS[0]}
set -e

printf '%s\n' "$RC" > "$OUT/container.rc"
date -Ins > "$OUT/finished_at.txt"
npu-smi info > "$OUT/npu_smi_after.txt" || true
if [ "$RC" -ne 0 ]; then
  exit "$RC"
fi
python3 "$SCRIPTS/validate_five_layer_case.py" --run "$OUT"
for index in 0 1 2 3 4 5 6 7; do
  if fuser "/dev/davinci${index}" >/dev/null 2>&1; then
    echo "device /dev/davinci${index} remained busy after $RUN_NAME" >&2
    exit 19
  fi
done
echo FIVE_LAYER_0162_MATRIX=PASS
