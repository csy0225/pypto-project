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
SEAL_AUTHORITY=$ROOT/campaign/normal_seal_authority.json
SCRIPTS_MANIFEST=$SCRIPTS/SCRIPTS_SHA256SUMS
ROUND=${COUNTERBALANCE_ROUND:-3}
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
test -s "$SCRIPTS/create_normal_seal_authority.py"
[[ "$ROUND" =~ ^[1-9][0-9]*$ ]]
[[ "$ITERS" =~ ^[1-9][0-9]*$ ]]
[[ "$WARMUP" =~ ^[0-9]+$ ]]
python3 "$SCRIPTS/verify_exact_tree_manifest.py" \
  --root "$SCRIPTS" \
  --manifest SCRIPTS_SHA256SUMS \
  --symlink-manifest -
SCRIPTS_MANIFEST_SHA=$(sha256sum "$SCRIPTS_MANIFEST" | awk '{print $1}')

mkdir -p "$ROOT/campaign"
SPEC=$ROOT/campaign/normal_counterbalance_spec.txt
SPEC_CONTENT=$(
  printf '%s\n' \
    "image=$IMG" \
    "baseline_source=$BASELINE_SOURCE" \
    "candidate_source=$CANDIDATE_SOURCE" \
    "campaign_scripts_manifest_sha256=$SCRIPTS_MANIFEST_SHA" \
    "batches=1,2,4,7,8,16" \
    "context_len_per_sequence=65536" \
    "round=$ROUND" \
    "warmup=$WARMUP" \
    "measured_iters=$ITERS" \
    "order=per-bs:candidate-r${ROUND},baseline-r${ROUND}"
)
if [ -e "$SPEC" ]; then
  test "$(cat "$SPEC")" = "$SPEC_CONTENT" || {
    echo "counterbalance spec mismatch: $SPEC" >&2
    exit 20
  }
else
  printf '%s\n' "$SPEC_CONTENT" > "$SPEC"
fi

run_case() {
  local source_kind=$1
  local batch=$2
  local source
  local run_name
  local run_dir
  if [ "$source_kind" = baseline ]; then
    source=$BASELINE_SOURCE
  else
    source=$CANDIDATE_SOURCE
  fi
  run_name="${source_kind}-r${ROUND}-normal-bs${batch}-64k"
  run_dir=$ROOT/campaign/runs/$run_name
  if [ -e "$run_dir" ]; then
    if python3 "$VALIDATOR" --run "$run_dir"; then
      echo "[counterbalance] verified existing $run_name"
      return
    fi
    echo "[counterbalance] refusing incomplete or invalid run: $run_dir" >&2
    exit 21
  fi
  echo "[counterbalance] start $run_name"
  MATRIX_ITERS=$ITERS MATRIX_WARMUP=$WARMUP \
    bash "$RUNNER" \
      "$ROOT" "$IMG" "$source" "$source_kind" \
      "$ROUND" normal "$batch"
  python3 "$VALIDATOR" --run "$run_dir"
  echo "[counterbalance] pass $run_name"
}

for batch in "${BATCHES[@]}"; do
  run_case candidate "$batch"
  run_case baseline "$batch"
done

python3 "$SCRIPTS/create_normal_seal_authority.py" \
  --campaign "$ROOT/campaign" \
  --out "$SEAL_AUTHORITY"
echo FIVE_LAYER_0162_NORMAL_COUNTERBALANCE=PASS
