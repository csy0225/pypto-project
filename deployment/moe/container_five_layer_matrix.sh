#!/usr/bin/env bash
set -Eeuo pipefail

: "${MATRIX_MODE:?missing MATRIX_MODE}"
: "${MATRIX_ROUND:?missing MATRIX_ROUND}"
: "${MATRIX_BATCH:?missing MATRIX_BATCH}"
: "${MATRIX_IMAGE_REF:?missing MATRIX_IMAGE_REF}"
: "${RUN_NONCE:?missing RUN_NONCE}"
: "${SOURCE_KIND:?missing SOURCE_KIND}"
valid_dfx_profile() {
  [[ "$1" =~ ^[[:alnum:]][[:alnum:]_.-]*$ ]]
}
case "$SOURCE_KIND" in
  baseline|candidate) ;;
  *)
    echo "invalid SOURCE_KIND=$SOURCE_KIND" >&2
    exit 30
    ;;
esac
if [ -n "${PYPTO_MOE_DFX_PROFILE:-}" ]; then
  valid_dfx_profile "$PYPTO_MOE_DFX_PROFILE" || {
    echo "invalid PYPTO_MOE_DFX_PROFILE=$PYPTO_MOE_DFX_PROFILE" >&2
    exit 33
  }
  export PYPTO_MOE_DFX_PROFILE
fi
case "$MATRIX_MODE" in
  normal|dfx) ;;
  *)
    echo "invalid MATRIX_MODE=$MATRIX_MODE" >&2
    exit 31
    ;;
esac
[[ "$RUN_NONCE" =~ ^[0-9a-f]{64}$ ]]
case "$MATRIX_BATCH" in
  1|2|4|7|8|16) ;;
  *)
    echo "invalid MATRIX_BATCH=$MATRIX_BATCH" >&2
    exit 32
    ;;
esac

export PYTHONDONTWRITEBYTECODE=1
export SIMPLER_A2A3_FORCE_VMM_IPC=1
export PYPTO_LIVE_IPC_STRICT=1
export PYPTO_IPC_SESSION_NONCE="$RUN_NONCE"
export PYPTO_IPC_LAUNCH_EPOCH
PYPTO_IPC_LAUNCH_EPOCH=$(
  /usr/local/python3.11.14/bin/python3 -c \
    'import time; print(f"{time.time():.9f}")'
)
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES

test ! -e /runtime-override
CAPABILITY_ARGS=(
  --output /out/capability_report.json
  --image-ref "$MATRIX_IMAGE_REF"
  --expected-pypto 8e92b46808f9f7c09b6431ad4691503f09c12ee5
  --expected-pypto-lib c9af5790d5fe450e14fd43c88099b87539089d17
  --expected-attn-profile a2a3
)
if [ "$MATRIX_MODE" = dfx ]; then
  CAPABILITY_ARGS+=(--require-l2-swimlane-reuse)
fi
/usr/local/python3.11.14/bin/python3 \
  /campaign-scripts/image_capability_probe.py \
  "${CAPABILITY_ARGS[@]}"

cd /workspace/pypto-lib
/usr/local/python3.11.14/bin/python3 \
  /campaign-scripts/verify_exact_tree_manifest.py \
  --root /workspace/pypto-lib \
  > /out/source_verify.log

ARGS=(
  --device 0,1,2,3,4,5,6,7
  --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
  --out /out/runtime
  --source-kind "$SOURCE_KIND"
  --round "$MATRIX_ROUND"
  --mode "$MATRIX_MODE"
  --batch "$MATRIX_BATCH"
  --iters "${MATRIX_ITERS:-20}"
  --warmup "${MATRIX_WARMUP:-3}"
)
if [ "$MATRIX_MODE" = dfx ]; then
  test -d /golden
  ARGS+=(--golden-root /golden)
  DFX_PROFILE=${PYPTO_MOE_DFX_PROFILE:-$SOURCE_KIND}
  valid_dfx_profile "$DFX_PROFILE" || {
    echo "invalid DFX profile=$DFX_PROFILE" >&2
    exit 34
  }
  ARGS+=(--dfx-profile "$DFX_PROFILE")
else
  test ! -e /golden
fi

/usr/local/python3.11.14/bin/python3 \
  /campaign-scripts/five_layer_matrix.py \
  "${ARGS[@]}"

test -s /out/runtime/matrix_report.json
test -s /out/capability_report.json
test -s "/out/runtime/bs${MATRIX_BATCH}/hidden_l3.pt"
test -s "/out/runtime/bs${MATRIX_BATCH}/hidden_l4.pt"
test -s "/out/runtime/bs${MATRIX_BATCH}/report.json"
if [ "$MATRIX_MODE" = dfx ]; then
  test -s \
    "/out/runtime/bs${MATRIX_BATCH}/dfx_analysis/moe_dfx_report.json"
  test -s \
    "/out/runtime/bs${MATRIX_BATCH}/dfx_analysis/moe_critical_path_report.md"
fi
echo FIVE_LAYER_MATRIX=PASS
