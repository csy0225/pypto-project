#!/usr/bin/env bash
set -Eeuo pipefail

: "${ACTIVE_BATCH:?missing ACTIVE_BATCH}"
: "${RUN_NONCE:?missing RUN_NONCE}"
: "${TILE_VARIANT:?missing TILE_VARIANT}"
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
[[ "$TILE_VARIANT" =~ ^mm-n(32|64)-r(8|16|32)$ ]]

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
/usr/local/python3.11.14/bin/python3 \
  /campaign-scripts/image_capability_probe.py \
  --output /out/capability_report.json \
  --image-ref "$MATRIX_IMAGE_REF" \
  --expected-pypto 8e92b46808f9f7c09b6431ad4691503f09c12ee5 \
  --expected-pypto-lib c9af5790d5fe450e14fd43c88099b87539089d17 \
  --expected-attn-profile a2a3

cd /workspace/pypto-lib
sha256sum -c SOURCE_SHA256SUMS > /out/source_verify.log
test -s MOE_TILE_SWEEP_PROFILE.txt
grep -Fx "variant=$TILE_VARIANT" MOE_TILE_SWEEP_PROFILE.txt >/dev/null

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
  --iters "${CASE_ITERS:-8}" \
  --warmup "${CASE_WARMUP:-3}" \
  --golden-dir "$GOLDEN_DIR"

sha256sum \
  /out/runtime/hidden_l3.pt \
  /out/runtime/hidden_l4.pt \
  /out/runtime/five_layer_moe_report.json \
  > /out/output_sha256.txt
test -s /out/capability_report.json
cp MOE_TILE_SWEEP_PROFILE.txt /out/MOE_TILE_SWEEP_PROFILE.txt
printf '%s\n' "$TILE_VARIANT" > /out/tile_variant.txt
echo MOE_TILE_SWEEP_CASE=PASS
