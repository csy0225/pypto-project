#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

if [ "$#" -ne 4 ]; then
  echo "usage: $0 <root> <image@digest> <baseline-source> <candidate-source>" >&2
  exit 2
fi

ROOT=$1
IMG=$2
BASELINE_SOURCE=$3
CANDIDATE_SOURCE=$4
SCRIPTS=$ROOT/scripts
RUNNER=$SCRIPTS/run_0162_five_layer_matrix.sh
VALIDATOR=$SCRIPTS/validate_five_layer_case.py
SCRIPTS_MANIFEST=$SCRIPTS/SCRIPTS_SHA256SUMS
NC=/mnt/persist/k8s-install/containerd/bin/nerdctl
ITERS=${MATRIX_ITERS:-30}
WARMUP=${MATRIX_WARMUP:-5}
BATCHES=(1 2 4 7 8 16)

test -d "$ROOT"
test -d "$BASELINE_SOURCE"
test -d "$CANDIDATE_SOURCE"
test -x "$RUNNER"
test -s "$VALIDATOR"
test -s "$SCRIPTS_MANIFEST"
test -s "$SCRIPTS/verify_exact_tree_manifest.py"
[[ "$ITERS" =~ ^[1-9][0-9]*$ ]]
[[ "$WARMUP" =~ ^[0-9]+$ ]]
python3 "$SCRIPTS/verify_exact_tree_manifest.py" \
  --root "$SCRIPTS" \
  --manifest SCRIPTS_SHA256SUMS \
  --symlink-manifest -
SCRIPTS_MANIFEST_SHA=$(sha256sum "$SCRIPTS_MANIFEST" | awk '{print $1}')

mkdir -p "$ROOT/campaign"
SPEC=$ROOT/campaign/normal_campaign_spec.txt
SPEC_CONTENT=$(
  printf '%s\n' \
    "image=$IMG" \
    "baseline_source=$BASELINE_SOURCE" \
    "candidate_source=$CANDIDATE_SOURCE" \
    "campaign_scripts_manifest_sha256=$SCRIPTS_MANIFEST_SHA" \
    "batches=1,2,4,7,8,16" \
    "context_len_per_sequence=65536" \
    "warmup=$WARMUP" \
    "measured_iters=$ITERS" \
    "order=per-bs:baseline-r1,candidate-r1,baseline-r2,candidate-r2"
)
if [ -e "$SPEC" ]; then
  test "$(cat "$SPEC")" = "$SPEC_CONTENT" || {
    echo "campaign spec mismatch: $SPEC" >&2
    exit 20
  }
else
  printf '%s\n' "$SPEC_CONTENT" > "$SPEC"
fi

run_case() {
  local source_kind=$1
  local round_id=$2
  local batch=$3
  local source
  local run_name
  local run_dir
  if [ "$source_kind" = baseline ]; then
    source=$BASELINE_SOURCE
  else
    source=$CANDIDATE_SOURCE
  fi
  run_name="${source_kind}-r${round_id}-normal-bs${batch}-64k"
  run_dir=$ROOT/campaign/runs/$run_name
  if [ -e "$run_dir" ]; then
    if python3 "$VALIDATOR" --run "$run_dir"; then
      echo "[campaign] verified existing $run_name"
      return
    fi
    echo "[campaign] refusing incomplete or invalid existing run: $run_dir" >&2
    exit 21
  fi
  echo "[campaign] start $run_name"
  MATRIX_ITERS=$ITERS MATRIX_WARMUP=$WARMUP \
    bash "$RUNNER" \
      "$ROOT" "$IMG" "$source" "$source_kind" \
      "$round_id" normal "$batch"
  python3 "$VALIDATOR" --run "$run_dir"
  echo "[campaign] pass $run_name"
}

for batch in "${BATCHES[@]}"; do
  run_case baseline 1 "$batch"
  run_case candidate 1 "$batch"
  run_case baseline 2 "$batch"
  run_case candidate 2 "$batch"
done

CORRECTNESS=$ROOT/campaign/matrix_correctness_report.json
GOLDEN=$ROOT/campaign/golden/heterogeneous-64k
if [ -e "$GOLDEN" ]; then
  python3 -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p["passed"] is True' \
    "$CORRECTNESS"
  echo "[campaign] verified existing correctness report and golden"
else
  sudo -n "$NC" run --rm --net host \
    -v "$ROOT/campaign":/campaign \
    -v "$SCRIPTS":/campaign-scripts:ro \
    "$IMG" \
    /usr/local/python3.11.14/bin/python3 \
    /campaign-scripts/analyze_matrix_correctness.py \
    --campaign /campaign \
    2>&1 | tee "$ROOT/campaign/correctness.log"
fi

test -s "$CORRECTNESS"
test -d "$GOLDEN"
echo FIVE_LAYER_0162_NORMAL_CAMPAIGN=PASS
