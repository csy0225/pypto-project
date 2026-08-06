#!/usr/bin/env bash
set -Eeuo pipefail

: "${ROUTE_COMPILE_IMAGE_REF:?missing ROUTE_COMPILE_IMAGE_REF}"

export PYTHONDONTWRITEBYTECODE=1
export SIMPLER_A2A3_FORCE_VMM_IPC=1
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES

test ! -e /runtime-override
/usr/local/python3.11.14/bin/python3 \
  /campaign-scripts/image_capability_probe.py \
  --output /out/capability_report.json \
  --image-ref "$ROUTE_COMPILE_IMAGE_REF" \
  --expected-pypto 8e92b46808f9f7c09b6431ad4691503f09c12ee5 \
  --expected-pypto-lib c9af5790d5fe450e14fd43c88099b87539089d17 \
  --expected-attn-profile a2a3

cd /workspace/pypto-lib
sha256sum -c SOURCE_SHA256SUMS > /out/source_verify.log

/usr/local/python3.11.14/bin/python3 \
  -m tests.step3p5.harnesses._stage_five_layer_moe_route \
  --device 0,1,2,3,4,5,6,7 \
  --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp \
  --out /out/runtime \
  --num-blocks 512 \
  --context-len 65536 \
  --active-batch 1 \
  --platform a2a3 \
  --image-digest "$ROUTE_COMPILE_IMAGE_REF" \
  --compile-only

test -s /out/runtime/compile_report.json
test -s /out/capability_report.json
echo FIVE_LAYER_ROUTE_COMPILE_ONLY=PASS
