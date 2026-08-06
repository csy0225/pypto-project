#!/usr/bin/env bash
set -Eeuo pipefail

if [ "$#" -ne 5 ]; then
  echo \
    "usage: $0 <root> <candidate-source> <golden-root> <source-sidecar-dir> <target-sidecar-dir>" \
    >&2
  exit 2
fi

ROOT=$1
SOURCE=$2
GOLDEN=$3
SOURCE_DIR=$4
TARGET_DIR=$5
SCRIPTS=${CAMPAIGN_SCRIPTS:-$ROOT/scripts}
PYTHON_BIN=${PYTHON_BIN:-python3}

[[ "$SOURCE_DIR" =~ ^[A-Za-z0-9._/-]+$ ]]
[[ "$TARGET_DIR" =~ ^[A-Za-z0-9._/-]+$ ]]
test -d "$ROOT/campaign/$SOURCE_DIR"
test ! -e "$ROOT/campaign/$TARGET_DIR"
test -d "$SOURCE"
test -d "$GOLDEN"
test -s "$SCRIPTS/validate_five_layer_route_case.py"
test -s "$SCRIPTS/analyze_route_campaign.py"

for batch in 1 2 4 7 8 16; do
  source_run=$ROOT/campaign/$SOURCE_DIR/candidate-route-bs${batch}-64k
  target_run=$ROOT/campaign/$TARGET_DIR/candidate-route-bs${batch}-64k
  source_runtime=$source_run/runtime
  target_runtime=$target_run/runtime
  test -d "$source_runtime"
  mkdir -p "$target_runtime"
  while IFS= read -r -d '' file; do
    cp -p "$file" "$target_runtime/"
  done < <(
    find "$source_runtime" -maxdepth 1 -type f \
      ! -name route_artifact_validation.json \
      -print0 | sort -z
  )
  test -s "$source_run/image_ref.txt"
  test -s "$source_run/input_tokens.txt"
  image_ref=$(<"$source_run/image_ref.txt")
  input_tokens=$(<"$source_run/input_tokens.txt")
  "$PYTHON_BIN" "$SCRIPTS/validate_five_layer_route_case.py" \
    --runtime "$target_runtime" \
    --golden-dir "$GOLDEN/bs$batch" \
    --source-root "$SOURCE" \
    --image-ref "$image_ref" \
    --active-batch "$batch" \
    --input-tokens "$input_tokens" \
    --profile row16
done

"$PYTHON_BIN" "$SCRIPTS/analyze_route_campaign.py" \
  --campaign "$ROOT/campaign" \
  --sidecar-dir "$TARGET_DIR" \
  --expected-profile row16 \
  --out "$ROOT/campaign/$TARGET_DIR"
