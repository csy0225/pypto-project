#!/usr/bin/env bash
set -Eeuo pipefail

: "${ROUTE_BATCH:?missing ROUTE_BATCH}"
: "${ROUTE_IMAGE_REF:?missing ROUTE_IMAGE_REF}"
: "${ROUTE_INPUT_TOKENS:?missing ROUTE_INPUT_TOKENS}"
ROUTE_PROFILE=${ROUTE_PROFILE:-row16}
ROUTE_EXPECTED_DECODE_SHA256=${ROUTE_EXPECTED_DECODE_SHA256:-}
ROUTE_SOURCE_ROLE=${ROUTE_SOURCE_ROLE:-}
case "$ROUTE_BATCH" in
  1|2|4|7|8|16) ;;
  *)
    echo "invalid ROUTE_BATCH=$ROUTE_BATCH" >&2
    exit 31
    ;;
esac
[[ "$ROUTE_PROFILE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "invalid ROUTE_PROFILE=$ROUTE_PROFILE" >&2
  exit 32
}
if [ -n "$ROUTE_EXPECTED_DECODE_SHA256$ROUTE_SOURCE_ROLE" ]; then
  [[ "$ROUTE_EXPECTED_DECODE_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "invalid ROUTE_EXPECTED_DECODE_SHA256" >&2
    exit 33
  }
  [[ "$ROUTE_SOURCE_ROLE" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "invalid ROUTE_SOURCE_ROLE=$ROUTE_SOURCE_ROLE" >&2
    exit 34
  }
fi
ROUTE_VALIDATION_PROFILE_ARGS=()
if [ -n "$ROUTE_EXPECTED_DECODE_SHA256" ]; then
  ROUTE_VALIDATION_PROFILE_ARGS=(
    --expected-decode-sha256 "$ROUTE_EXPECTED_DECODE_SHA256"
    --source-role "$ROUTE_SOURCE_ROLE"
  )
fi

export PYTHONDONTWRITEBYTECODE=1
export SIMPLER_A2A3_FORCE_VMM_IPC=1
export PYPTO_LIVE_IPC_STRICT=1
export PYPTO_IPC_SESSION_NONCE="${ROUTE_RUN_NONCE:?missing ROUTE_RUN_NONCE}"
export PYPTO_IPC_LAUNCH_EPOCH
PYPTO_IPC_LAUNCH_EPOCH=$(
  /usr/local/python3.11.14/bin/python3 -c \
    'import time; print(f"{time.time():.9f}")'
)
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES

test ! -e /runtime-override
test "$PYPTO_IMAGE_PYPTO_COMMIT" = \
  8e92b46808f9f7c09b6431ad4691503f09c12ee5
test "$PYPTO_STEP3P5_ATTN_TASK_PROFILE" = a2a3
test "$PYPTO_REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN" = 1
test -d "/golden/bs$ROUTE_BATCH"

cd /workspace/pypto-lib
/usr/local/python3.11.14/bin/python3 \
  /campaign-scripts/verify_exact_tree_manifest.py \
  --root /workspace/pypto-lib \
  > /out/source_verify.log

/usr/local/python3.11.14/bin/python3 \
  -m tests.step3p5.harnesses._stage_five_layer_moe_route \
  --device 0,1,2,3,4,5,6,7 \
  --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --out /out/runtime \
  --golden-dir "/golden/bs$ROUTE_BATCH" \
  --num-blocks 512 \
  --context-len 65536 \
  --active-batch "$ROUTE_BATCH" \
  --input-tokens "$ROUTE_INPUT_TOKENS" \
  --warmup 1 \
  --platform a2a3 \
  --image-digest "$ROUTE_IMAGE_REF"

/usr/local/python3.11.14/bin/python3 \
  /campaign-scripts/validate_five_layer_route_case.py \
  --runtime /out/runtime \
  --golden-dir "/golden/bs$ROUTE_BATCH" \
  --source-root /workspace/pypto-lib \
  --image-ref "$ROUTE_IMAGE_REF" \
  --active-batch "$ROUTE_BATCH" \
  --input-tokens "$ROUTE_INPUT_TOKENS" \
  --profile "$ROUTE_PROFILE" \
  "${ROUTE_VALIDATION_PROFILE_ARGS[@]}"

test -s /out/runtime/route_artifact_validation.json
echo FIVE_LAYER_ROUTE_SIDECAR=PASS
