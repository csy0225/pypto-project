#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 4 ]; then
  echo \
    "usage: $0 <experiment-root> <image@digest> <candidate-source> <golden-root>" \
    >&2
  exit 2
fi

ROOT=$1
IMG=$2
SOURCE=$3
GOLDEN=$4
SCRIPTS=$ROOT/scripts
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
RUN_NAME=candidate-shared-route-bs1-64k
RUN=$ROOT/campaign/route-sidecar/$RUN_NAME
VALIDATION=$RUN/runtime/route_artifact_validation.json
INPUT_TOKENS=6127
EXPECTED_DECODE_SHA=572ea2a2b0ceab8952cbd8d7c1f383351fb877e11b01b4d68a60697f4508576b

test -d "$ROOT"
test -d "$SOURCE"
test -d "$GOLDEN/bs1"
test -d "$CKPT"
test -s "$SOURCE/SOURCE_SHA256SUMS"
test -x "$SCRIPTS/container_five_layer_route.sh"
test -s "$SCRIPTS/validate_five_layer_route_case.py"
(cd "$SOURCE" && sha256sum -c SOURCE_SHA256SUMS)
test "$(
  sha256sum "$SOURCE/models/step3p5/decode_fwd.py" | awk '{print $1}'
)" = "$EXPECTED_DECODE_SHA"

if [ -e "$RUN" ]; then
  python3 -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True' \
    "$VALIDATION"
  echo "[shared-route] verified existing $RUN_NAME"
  exit 0
fi

for index in 0 1 2 3 4 5 6 7; do
  if fuser "/dev/davinci${index}" >/dev/null 2>&1; then
    echo "device /dev/davinci${index} is busy" >&2
    fuser -v "/dev/davinci${index}" >&2 || true
    exit 21
  fi
done

mkdir -p "$RUN/runtime"
SOURCE_MANIFEST_SHA=$(
  sha256sum "$SOURCE/SOURCE_SHA256SUMS" | awk '{print $1}'
)
NONCE=$(
  printf '%s:%s:%s:%s' \
    "$IMG" "$SOURCE_MANIFEST_SHA" 1 "$(date +%s%N)" \
    | sha256sum | awk '{print $1}'
)
printf '%s\n' "$IMG" > "$RUN/image_ref.txt"
printf '%s\n' "$SOURCE_MANIFEST_SHA" \
  > "$RUN/source_manifest_sha256.txt"
printf '%s\n' "$NONCE" > "$RUN/run_nonce.txt"
printf '%s\n' "$INPUT_TOKENS" > "$RUN/input_tokens.txt"
date -Ins > "$RUN/started_at.txt"
npu-smi info > "$RUN/npu_smi_before.txt"

DEVS=()
for index in 0 1 2 3 4 5 6 7; do
  DEVS+=(--device "/dev/davinci${index}")
done

set +e
sudo -n "$NC" run --name moe-shared-route-b1 --rm \
  --net host --ipc host \
  --privileged --security-opt apparmor=unconfined \
  --env ROUTE_BATCH=1 \
  --env ROUTE_IMAGE_REF="$IMG" \
  --env ROUTE_INPUT_TOKENS="$INPUT_TOKENS" \
  --env ROUTE_RUN_NONCE="$NONCE" \
  --env ROUTE_PROFILE=shared-split \
  "${DEVS[@]}" \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$CKPT":"$CKPT":ro \
  -v "$SOURCE":/workspace/pypto-lib:ro \
  -v "$SCRIPTS":/campaign-scripts:ro \
  -v "$GOLDEN":/golden:ro \
  -v "$RUN":/out \
  "$IMG" bash /campaign-scripts/container_five_layer_route.sh \
  2>&1 | tee "$RUN/container.log"
RC=${PIPESTATUS[0]}
set -e

printf '%s\n' "$RC" > "$RUN/container.rc"
date -Ins > "$RUN/finished_at.txt"
npu-smi info > "$RUN/npu_smi_after.txt" || true
if [ "$RC" -ne 0 ]; then
  exit "$RC"
fi

python3 -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True' \
  "$VALIDATION"
for index in 0 1 2 3 4 5 6 7; do
  if fuser "/dev/davinci${index}" >/dev/null 2>&1; then
    echo "device /dev/davinci${index} remained busy after $RUN_NAME" >&2
    exit 22
  fi
done
echo FIVE_LAYER_0162_SHARED_ROUTE_BS1=PASS
