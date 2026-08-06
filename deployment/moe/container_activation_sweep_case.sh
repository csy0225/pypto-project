#!/usr/bin/env bash
set -Eeuo pipefail

: "${ACTIVE_BATCH:?missing ACTIVE_BATCH}"
: "${RUN_NONCE:?missing RUN_NONCE}"
: "${CASE_VARIANT:?missing CASE_VARIANT}"
: "${GOLDEN_DIR:?missing GOLDEN_DIR}"
: "${MATRIX_IMAGE_REF:?missing MATRIX_IMAGE_REF}"

case "$ACTIVE_BATCH" in
  1|2|4|7|8|16) ;;
  *)
    echo "invalid ACTIVE_BATCH=$ACTIVE_BATCH" >&2
    exit 31
    ;;
esac
[[ "$RUN_NONCE" =~ ^[0-9a-f]{64}$ ]]
[[ "$CASE_VARIANT" =~ ^[A-Za-z0-9._-]+$ ]]

export PYTHONDONTWRITEBYTECODE=1
export SIMPLER_A2A3_FORCE_VMM_IPC=1
export PYPTO_LIVE_IPC_STRICT=1
export PYPTO_IPC_SESSION_NONCE="$RUN_NONCE"
export PYPTO_IPC_LAUNCH_EPOCH
PYPTO_IPC_LAUNCH_EPOCH=$(
  /usr/local/python3.11.14/bin/python3 -c \
    'import time; print(f"{time.time():.9f}")'
)
export PYPTO_PROG_BUILD_DIR=/out/runtime/build_output
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES

test ! -e /runtime-override
test "$PYPTO_IMAGE_PYPTO_COMMIT" = \
  8e92b46808f9f7c09b6431ad4691503f09c12ee5
test "$PYPTO_STEP3P5_ATTN_TASK_PROFILE" = a2a3
test "$PYPTO_REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN" = 1

cd /workspace/pypto-lib
sha256sum -c SOURCE_SHA256SUMS > /out/source_verify.log

TOKENS=(
  6127 303 1207 19384 872 428 4231 2636
  6178 410 1 2 3 4 5 6
)
INPUT_TOKENS=$(
  IFS=,
  printf '%s' "${TOKENS[*]:0:$ACTIVE_BATCH}"
)

/usr/local/python3.11.14/bin/python3 \
  -m tests.step3p5.harnesses._stage_five_layer_moe \
  --device 0,1,2,3,4,5,6,7 \
  --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --out /out/runtime \
  --num-blocks 512 \
  --context-len 65536 \
  --active-batch "$ACTIVE_BATCH" \
  --input-tokens "$INPUT_TOKENS" \
  --iters "${CASE_ITERS:-12}" \
  --warmup "${CASE_WARMUP:-3}" \
  --golden-dir "$GOLDEN_DIR"

sha256sum \
  /out/runtime/hidden_l3.pt \
  /out/runtime/hidden_l4.pt \
  /out/runtime/five_layer_moe_report.json \
  > /out/output_sha256.txt
printf '%s\n' "$CASE_VARIANT" > /out/variant.txt
echo MOE_ACTIVATION_SWEEP_CASE=PASS
