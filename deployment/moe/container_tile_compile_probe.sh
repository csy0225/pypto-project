#!/usr/bin/env bash
set -Eeuo pipefail

: "${TILE_VARIANT:?missing TILE_VARIANT}"
: "${TILE_COMPILE_IMAGE_REF:?missing TILE_COMPILE_IMAGE_REF}"

[[ "$TILE_VARIANT" =~ ^mm-n(32|64)-r(8|16|32)$ ]]

export PYTHONDONTWRITEBYTECODE=1
export SIMPLER_A2A3_FORCE_VMM_IPC=1
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES

test ! -e /runtime-override
test "$PYPTO_IMAGE_PYPTO_COMMIT" = \
  8e92b46808f9f7c09b6431ad4691503f09c12ee5
test "$PYPTO_STEP3P5_ATTN_TASK_PROFILE" = a2a3
test "$PYPTO_REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN" = 1

cd /workspace/pypto-lib
sha256sum -c SOURCE_SHA256SUMS > /out/source_verify.log
test -s MOE_TILE_SWEEP_PROFILE.txt
grep -Fx "variant=$TILE_VARIANT" MOE_TILE_SWEEP_PROFILE.txt >/dev/null

/usr/local/python3.11.14/bin/python3 \
  -m tests.step3p5.harnesses._stage_five_layer_moe \
  --device 0,1,2,3,4,5,6,7 \
  --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --out /out/runtime \
  --num-blocks 512 \
  --context-len 65536 \
  --active-batch 1 \
  --platform a2a3 \
  --compile-only

test -s /out/runtime/compile_report.json
cp MOE_TILE_SWEEP_PROFILE.txt /out/MOE_TILE_SWEEP_PROFILE.txt
echo MOE_TILE_COMPILE_PROBE=PASS
