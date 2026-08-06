#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

if [ "$#" -ne 3 ] && [ "$#" -ne 4 ]; then
  echo \
    "usage: $0 <root> <image@digest> <candidate-route-source> [matched-policy.json]" \
    >&2
  exit 2
fi

ROOT=$1
IMG=$2
SOURCE=$3
MATCHED_POLICY=${4:-}
SCRIPTS=${CAMPAIGN_SCRIPTS:-$ROOT/scripts}
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
GOLDEN=$ROOT/campaign/golden/heterogeneous-64k
SCRIPTS_MANIFEST=$SCRIPTS/SCRIPTS_SHA256SUMS
BATCHES=(1 2 4 7 8 16)
TOKENS=(6127 303 1207 19384 872 428 4231 2636 6178 410 1 2 3 4 5 6)
SIDECAR_DIR=${ROUTE_SIDECAR_DIR:-route-sidecar}
PROFILE=${ROUTE_PROFILE:-row16}
EXPECTED_DECODE_SHA256=
EXPECTED_SOURCE_ROLE=
MATCHED_POLICY_SHA=

[[ "$SIDECAR_DIR" =~ ^[A-Za-z0-9._-]+$ ]]
[[ "$PROFILE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "invalid ROUTE_PROFILE=$PROFILE" >&2
  exit 23
}

test -d "$ROOT"
test -d "$SOURCE"
test -d "$SCRIPTS"
test -d "$CKPT"
test -d "$GOLDEN"
test -s "$SOURCE/SOURCE_SHA256SUMS"
test -s "$SCRIPTS_MANIFEST"
test -x "$SCRIPTS/container_five_layer_route.sh"
test -s "$SCRIPTS/validate_five_layer_route_case.py"
test -s "$SCRIPTS/analyze_route_campaign.py"
test -s "$SCRIPTS/verify_exact_tree_manifest.py"
python3 "$SCRIPTS/verify_exact_tree_manifest.py" --root "$SOURCE"
python3 "$SCRIPTS/verify_exact_tree_manifest.py" \
  --root "$SCRIPTS" \
  --manifest SCRIPTS_SHA256SUMS \
  --symlink-manifest -

if [ -n "$MATCHED_POLICY" ]; then
  test -s "$MATCHED_POLICY"
  MATCHED_POLICY=$(realpath "$MATCHED_POLICY")
  mapfile -t POLICY_FIELDS < <(
    PYTHONPATH="$SCRIPTS" python3 - "$MATCHED_POLICY" <<'PY'
import sys

from validate_five_layer_case import load_matched_dfx_policy

policy = load_matched_dfx_policy(sys.argv[1])
candidate = policy["sources"]["candidate"]
print(policy["policy_sha256"])
print(policy["authority"]["image_ref"])
print(candidate["profile"])
print(candidate["decode_fwd_sha256"])
print(candidate["source_role"])
PY
  )
  test "${#POLICY_FIELDS[@]}" -eq 5
  MATCHED_POLICY_SHA=${POLICY_FIELDS[0]}
  test "${POLICY_FIELDS[1]}" = "$IMG"
  PROFILE=${POLICY_FIELDS[2]}
  EXPECTED_DECODE_SHA256=${POLICY_FIELDS[3]}
  EXPECTED_SOURCE_ROLE=${POLICY_FIELDS[4]}
  test \
    "$(sha256sum "$SOURCE/models/step3p5/decode_fwd.py" | awk '{print $1}')" \
    = "$EXPECTED_DECODE_SHA256"
else
  case "$PROFILE" in
    row16)
      EXPECTED_DECODE_SHA256=65b0b8bf139aa40a5cf67317148dc16193ff22a81b394fbfe86e31ea05623e08
      EXPECTED_SOURCE_ROLE=reference
      ;;
    shared-split)
      EXPECTED_DECODE_SHA256=572ea2a2b0ceab8952cbd8d7c1f383351fb877e11b01b4d68a60697f4508576b
      EXPECTED_SOURCE_ROLE=candidate
      ;;
    *)
      echo "generic ROUTE_PROFILE requires a matched policy" >&2
      exit 24
      ;;
  esac
fi

CORRECTNESS=$ROOT/campaign/matrix_correctness_report.json
python3 -c \
  'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True' \
  "$CORRECTNESS"

mkdir -p "$ROOT/campaign/$SIDECAR_DIR"
SPEC_STEM=${SIDECAR_DIR//-/_}
SPEC=$ROOT/campaign/${SPEC_STEM}_campaign_spec.txt
SOURCE_MANIFEST_SHA=$(sha256sum "$SOURCE/SOURCE_SHA256SUMS" | awk '{print $1}')
SCRIPTS_MANIFEST_SHA=$(sha256sum "$SCRIPTS_MANIFEST" | awk '{print $1}')
SPEC_CONTENT=$(
  printf '%s\n' \
    "image=$IMG" \
    "candidate_route_source=$SOURCE" \
    "candidate_route_source_manifest_sha256=$SOURCE_MANIFEST_SHA" \
    "campaign_scripts=$SCRIPTS" \
    "campaign_scripts_manifest_sha256=$SCRIPTS_MANIFEST_SHA" \
    "batches=1,2,4,7,8,16" \
    "context_len_per_sequence=65536" \
    "blocks_per_sequence=512" \
    "topk=8" \
    "golden=$GOLDEN" \
    "profile=$PROFILE" \
    "expected_decode_fwd_sha256=$EXPECTED_DECODE_SHA256" \
    "expected_source_role=$EXPECTED_SOURCE_ROLE" \
    "matched_policy=${MATCHED_POLICY:-none}" \
    "matched_policy_sha256=${MATCHED_POLICY_SHA:-none}"
)
if [ -e "$SPEC" ]; then
  test "$(cat "$SPEC")" = "$SPEC_CONTENT" || {
    echo "route campaign spec mismatch: $SPEC" >&2
    exit 20
  }
else
  printf '%s\n' "$SPEC_CONTENT" > "$SPEC"
fi

input_tokens() {
  local batch=$1
  local result=
  local index
  for ((index = 0; index < batch; index++)); do
    if [ -n "$result" ]; then
      result+=,
    fi
    result+=${TOKENS[$index]}
  done
  printf '%s\n' "$result"
}

for batch in "${BATCHES[@]}"; do
  RUN_NAME="candidate-route-bs${batch}-64k"
  RUN=$ROOT/campaign/$SIDECAR_DIR/$RUN_NAME
  VALIDATION=$RUN/runtime/route_artifact_validation.json
  if [ -e "$RUN" ]; then
    python3 - "$VALIDATION" "$PROFILE" "$EXPECTED_DECODE_SHA256" \
      "$EXPECTED_SOURCE_ROLE" <<'PY'
import json
import sys

path, profile, decode, role = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
assert value["passed"] is True
assert value["profile"] == profile
assert value["decode_fwd_sha256"] == decode
assert value["source_role"] == role
PY
    echo "[route-campaign] verified existing $RUN_NAME"
    continue
  fi

  for index in 0 1 2 3 4 5 6 7; do
    if fuser "/dev/davinci${index}" >/dev/null 2>&1; then
      echo "device /dev/davinci${index} is busy" >&2
      fuser -v "/dev/davinci${index}" >&2 || true
      exit 21
    fi
  done

  mkdir -p "$RUN/runtime"
  TOKENS_TEXT=$(input_tokens "$batch")
  NONCE=$(
    printf '%s:%s:%s:%s' \
      "$IMG" "$SOURCE_MANIFEST_SHA" "$batch" "$(date +%s%N)" \
      | sha256sum | awk '{print $1}'
  )
  printf '%s\n' "$IMG" > "$RUN/image_ref.txt"
  printf '%s\n' "$SOURCE_MANIFEST_SHA" > "$RUN/source_manifest_sha256.txt"
  printf '%s\n' "$NONCE" > "$RUN/run_nonce.txt"
  printf '%s\n' "$TOKENS_TEXT" > "$RUN/input_tokens.txt"
  date -Ins > "$RUN/started_at.txt"
  npu-smi info > "$RUN/npu_smi_before.txt"

  DEVS=()
  for index in 0 1 2 3 4 5 6 7; do
    DEVS+=(--device "/dev/davinci${index}")
  done
  CONTAINER="moe-route-b${batch}"
  set +e
  sudo -n "$NC" run --name "$CONTAINER" --rm --net host --ipc host \
    --privileged --security-opt apparmor=unconfined \
    --env ROUTE_BATCH="$batch" \
    --env ROUTE_IMAGE_REF="$IMG" \
    --env ROUTE_INPUT_TOKENS="$TOKENS_TEXT" \
    --env ROUTE_RUN_NONCE="$NONCE" \
    --env ROUTE_PROFILE="$PROFILE" \
    --env ROUTE_EXPECTED_DECODE_SHA256="$EXPECTED_DECODE_SHA256" \
    --env ROUTE_SOURCE_ROLE="$EXPECTED_SOURCE_ROLE" \
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
  echo "[route-campaign] pass $RUN_NAME"
done

if [ "$SIDECAR_DIR" = route-sidecar ]; then
  REPORT_OUT=$ROOT/campaign
else
  REPORT_OUT=$ROOT/campaign/$SIDECAR_DIR
fi
python3 "$SCRIPTS/analyze_route_campaign.py" \
  --campaign "$ROOT/campaign" \
  --sidecar-dir "$SIDECAR_DIR" \
  --expected-profile "$PROFILE" \
  --expected-decode-sha256 "$EXPECTED_DECODE_SHA256" \
  --expected-source-role "$EXPECTED_SOURCE_ROLE" \
  --expected-image-ref "$IMG" \
  --expected-source-manifest-sha256 "$SOURCE_MANIFEST_SHA" \
  --golden-root "$GOLDEN" \
  --out "$REPORT_OUT"
echo FIVE_LAYER_0162_ROUTE_CAMPAIGN=PASS
