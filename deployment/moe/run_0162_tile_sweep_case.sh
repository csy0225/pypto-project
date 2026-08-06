#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 5 ]; then
  echo \
    "usage: $0 <root> <image@digest> <source-dir> <variant> <batch>" \
    >&2
  exit 2
fi

ROOT=$1
IMG=$2
SOURCE=$3
VARIANT=$4
BATCH=$5
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
SCRIPTS=${CAMPAIGN_SCRIPTS:-$ROOT/scripts}
GOLDEN=$ROOT/campaign/golden/heterogeneous-64k/bs${BATCH}
OUT=$ROOT/campaign/tile/$VARIANT/bs${BATCH}
CONTAINER="moe-tile-${VARIANT//[^A-Za-z0-9]/-}-b${BATCH}"

test -d "$ROOT"
test -d "$SOURCE"
test -d "$SCRIPTS"
test -d "$GOLDEN"
test -x "$SCRIPTS/container_tile_sweep_case.sh"
test -s "$SCRIPTS/audit_tile_compile_resources.py"
[[ "$VARIANT" =~ ^mm-n(32|64)-r(8|16|32)$ ]]
case "$BATCH" in
  1|2|4|7|8|16) ;;
  *)
    echo "invalid batch=$BATCH" >&2
    exit 3
    ;;
esac
if [ -e "$OUT" ]; then
  test -s "$OUT/container.rc"
  test "$(cat "$OUT/container.rc")" = 0
  test -s "$OUT/runtime/five_layer_moe_report.json"
  test -s "$OUT/tile_variant.txt"
  test "$(cat "$OUT/tile_variant.txt")" = "$VARIANT"
  SOURCE_MANIFEST_SHA=$(
    sha256sum "$SOURCE/SOURCE_SHA256SUMS" | awk '{print $1}'
  )
  test "$(cat "$OUT/image_ref.txt")" = "$IMG"
  test "$(cat "$OUT/source_manifest_sha256.txt")" = \
    "$SOURCE_MANIFEST_SHA"
  test "$(sha256sum \
    "$SCRIPTS/container_tile_sweep_case.sh" \
    "$SCRIPTS/audit_tile_compile_resources.py" \
    "$SCRIPTS/run_0162_tile_sweep_case.sh")" \
    = "$(cat "$OUT/scripts_sha256.txt")"
  python3 "$SCRIPTS/audit_tile_compile_resources.py" \
    --build-root "$OUT/runtime/build_output" \
    --profile "$SOURCE/MOE_TILE_SWEEP_PROFILE.txt" \
    --source-manifest "$SOURCE/SOURCE_SHA256SUMS" \
    --out "$OUT/resource_audit.json"
  python3 - "$OUT/runtime/five_layer_moe_report.json" "$BATCH" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
batch = int(sys.argv[2])
assert report["comparisons"]["hidden_l3"]["exact"] is True
assert report["comparisons"]["hidden_l4"]["exact"] is True
assert report["workload"]["active_batch"] == batch
assert report["workload"]["context_len"] == 65536
assert report["workload"]["blocks_per_sequence"] == 512
PY
  echo "[tile-sweep] verified existing $VARIANT BS$BATCH"
  exit 0
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
    "$IMG" "$VARIANT" "$BATCH" "$(date +%s%N)" "$SOURCE" \
    | sha256sum | awk '{print $1}'
)

date -Ins > "$OUT/started_at.txt"
printf '%s\n' "$IMG" > "$OUT/image_ref.txt"
printf '%s\n' "$VARIANT" > "$OUT/tile_variant.txt"
printf '%s\n' "$NONCE" > "$OUT/run_nonce.txt"
sha256sum "$SOURCE/SOURCE_SHA256SUMS" | awk '{print $1}' \
  > "$OUT/source_manifest_sha256.txt"
sha256sum \
  "$SCRIPTS/container_tile_sweep_case.sh" \
  "$SCRIPTS/audit_tile_compile_resources.py" \
  "$SCRIPTS/run_0162_tile_sweep_case.sh" \
  > "$OUT/scripts_sha256.txt"

set +e
sudo -n "$NC" run --name "$CONTAINER" --rm --net host --ipc host \
  --privileged --security-opt apparmor=unconfined \
  --env ACTIVE_BATCH="$BATCH" \
  --env TILE_VARIANT="$VARIANT" \
  --env RUN_NONCE="$NONCE" \
  --env GOLDEN_DIR=/golden \
  --env MATRIX_IMAGE_REF="$IMG" \
  --env CASE_ITERS="${CASE_ITERS:-8}" \
  --env CASE_WARMUP="${CASE_WARMUP:-3}" \
  "${DEVS[@]}" \
  --device /dev/davinci_manager \
  --device /dev/hisi_hdc \
  --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp:/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp:ro \
  -v "$SOURCE":/workspace/pypto-lib:ro \
  -v "$SCRIPTS":/campaign-scripts:ro \
  -v "$GOLDEN":/golden:ro \
  -v "$OUT":/out \
  "$IMG" bash /campaign-scripts/container_tile_sweep_case.sh \
  2>&1 | tee "$OUT/container.log"
PIPE_RC=("${PIPESTATUS[@]}")
RC=${PIPE_RC[0]}
TEE_RC=${PIPE_RC[1]}
set -e

if [ "$RC" -eq 0 ] && [ "$TEE_RC" -ne 0 ]; then
  RC=24
fi
printf '%s\n' "$RC" > "$OUT/container.rc"
date -Ins > "$OUT/finished_at.txt"
if [ "$RC" -ne 0 ]; then
  exit "$RC"
fi
test -s "$OUT/runtime/five_layer_moe_report.json"
test -s "$OUT/runtime/hidden_l3.pt"
test -s "$OUT/runtime/hidden_l4.pt"
python3 "$SCRIPTS/audit_tile_compile_resources.py" \
  --build-root "$OUT/runtime/build_output" \
  --profile "$SOURCE/MOE_TILE_SWEEP_PROFILE.txt" \
  --source-manifest "$SOURCE/SOURCE_SHA256SUMS" \
  --out "$OUT/resource_audit.json"
python3 - "$OUT/runtime/five_layer_moe_report.json" "$VARIANT" "$BATCH" <<'PY'
import json
import sys

path, expected_variant, expected_batch = sys.argv[1:]
report = json.load(open(path, encoding="utf-8"))
assert report["comparisons"]["hidden_l3"]["exact"] is True
assert report["comparisons"]["hidden_l4"]["exact"] is True
assert report["workload"]["active_batch"] == int(expected_batch)
assert report["workload"]["context_len"] == 65536
assert report["workload"]["blocks_per_sequence"] == 512
assert expected_variant.startswith("mm-n")
PY
for index in 0 1 2 3 4 5 6 7; do
  if fuser "/dev/davinci${index}" >/dev/null 2>&1; then
    echo "device /dev/davinci${index} remained busy after $VARIANT BS$BATCH" >&2
    exit 19
  fi
done
echo MOE_TILE_SWEEP_0162_CASE=PASS
