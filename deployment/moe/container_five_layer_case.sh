#!/usr/bin/env bash
set -Eeuo pipefail

: "${CASE_MODE:?missing CASE_MODE}"
: "${ACTIVE_BATCH:?missing ACTIVE_BATCH}"
: "${RUN_NONCE:?missing RUN_NONCE}"
: "${SOURCE_KIND:?missing SOURCE_KIND}"

case "$CASE_MODE" in
  normal|dfx) ;;
  *)
    echo "invalid CASE_MODE=$CASE_MODE" >&2
    exit 31
    ;;
esac
case "$ACTIVE_BATCH" in
  1|2|4|7|8|16) ;;
  *)
    echo "invalid ACTIVE_BATCH=$ACTIVE_BATCH" >&2
    exit 32
    ;;
esac
[[ "$RUN_NONCE" =~ ^[0-9a-f]{64}$ ]]

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
CAPABILITY_ARGS=(
  --output /out/capability_report.json
  --image-ref "${CASE_IMAGE_REF:?missing CASE_IMAGE_REF}"
  --expected-pypto 8e92b46808f9f7c09b6431ad4691503f09c12ee5
  --expected-pypto-lib 491267c45875e9b1e0071eed224e2e73526799e2
  --expected-attn-profile a2a3
)
if [ "$CASE_MODE" = dfx ]; then
  CAPABILITY_ARGS+=(--require-l2-swimlane-reuse)
fi
/usr/local/python3.11.14/bin/python3 \
  /campaign-scripts/image_capability_probe.py \
  "${CAPABILITY_ARGS[@]}"

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

COMMON_ARGS=(
  --device 0,1,2,3,4,5,6,7
  --ckpt /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
  --out /out/runtime
  --num-blocks 512
  --context-len 65536
  --active-batch "$ACTIVE_BATCH"
  --input-tokens "$INPUT_TOKENS"
  --iters "${CASE_ITERS:-3}"
  --warmup "${CASE_WARMUP:-2}"
)
if [ -n "${GOLDEN_DIR:-}" ]; then
  COMMON_ARGS+=(--golden-dir "$GOLDEN_DIR")
fi
if [ "$CASE_MODE" = dfx ]; then
  COMMON_ARGS+=(--dfx)
fi

cat > /out/case_contract.json <<EOF
{
  "active_batch": $ACTIVE_BATCH,
  "case_mode": "$CASE_MODE",
  "context_len_per_sequence": 65536,
  "image_pypto_commit": "$PYPTO_IMAGE_PYPTO_COMMIT",
  "input_tokens": "$INPUT_TOKENS",
  "num_blocks_per_sequence": 512,
  "run_nonce": "$RUN_NONCE",
  "source_kind": "$SOURCE_KIND",
  "runtime_override": false
}
EOF

/usr/local/python3.11.14/bin/python3 \
  -m tests.step3p5.harnesses._stage_five_layer_moe \
  "${COMMON_ARGS[@]}"

sha256sum \
  /out/runtime/hidden_l3.pt \
  /out/runtime/hidden_l4.pt \
  /out/runtime/five_layer_moe_report.json \
  > /out/output_sha256.txt
test -s /out/capability_report.json
if [ "$CASE_MODE" = dfx ]; then
  test -s /out/runtime/dfx_protocol_report.json
  test -s /out/runtime/dfx_analysis/moe_dfx_report.json
  test -s /out/runtime/dfx_analysis/moe_critical_path_report.md
fi
echo FIVE_LAYER_CASE=PASS
