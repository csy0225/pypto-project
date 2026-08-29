#!/usr/bin/env bash
# BS1 first-five-layer (L0-L4) l2_swimlane DFX capture for a candidate image.
#
# Instrumented run: onboard executes each kernel twice (dep_gen pass, then a
# timing-only pass reusing that dependency graph), so absolute times here are
# NOT a latency claim -- they are only usable for critical-path SHARE. Clean
# absolute latency comes from run_itl_gate.sh.
#
# Workpoint matches the 2026-08-11 K8 capture so shares are comparable:
# cards 0-7, active_batch=1, context_len=65536, num_blocks=512, warmup=3, iters=20.
#
# Two things a reader must know before interpreting the output:
#   * Pick the LOW-WAIT rank first. Makespan spread across ranks was 275x
#     (rank2 2.204 ms vs rank5 609.764 ms) because the slow ranks' tp_all_reduce
#     is spin absorbing rank skew, not compute. Only the low-wait rank is usable.
#   * container rc=1 is EXPECTED, not a new regression: analyze_five_layer_moe_dfx
#     fail-closes on rank0/1/3/6, where five early_dispatch tasks appear in
#     deps.json but are missing from the swimlane records. Pre-existing, see
#     design/performance/05-moe-optimization.md.
#
# H4 resident constants are enabled by default; set
# PYPTO_H4_RESIDENT=none for rollback or diagnostic comparisons.
# Usage: bash run_swimlane_gate.sh <image-id>
set -Eeuo pipefail
cd /tmp   # nerdctl resolves apparmor_parser relative to cwd

IMGID=${1:?need image id or digest}
H4_RESIDENT=${PYPTO_H4_RESIDENT:-all}
case "$H4_RESIDENT" in
  none|rope|gate|all) ;;
  *) echo "FAIL: PYPTO_H4_RESIDENT=$H4_RESIDENT is invalid" >&2; exit 2 ;;
esac
D=/mnt/persist/chensiyu/workspace/upgrade-20260821
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
OUT=$D/swimlane-$(date +%Y%m%d-%H%M%S)
NC="sudo -n /mnt/persist/k8s-install/containerd/bin/nerdctl"
DEVCSV=0,1,2,3,4,5,6,7
NONCE=$(printf 'swim:%s:%s' "$$" "$(date +%s%N)" | sha256sum | awk '{print $1}')

mkdir -p "$OUT/runtime" "$OUT/build_output"
DEVS=""; for i in 0 1 2 3 4 5 6 7; do DEVS="$DEVS --device /dev/davinci$i"; done

cat > "$OUT/run_contract.json" <<JSON
{"kind":"five_layer_l2_swimlane","image":"$IMGID","devices":"$DEVCSV",
 "active_batch":1,"context_len":65536,"num_blocks":512,"warmup":3,"iters":20,
 "h4_resident":"$H4_RESIDENT",
 "seed_token":6127,"instrumented":true,
 "absolute_latency_is_not_a_clean_claim":true,
 "digest_only":true,"source_overlay":false,"runtime_overlay":false,
 "ipc_session_nonce":"$NONCE","codegen_max_workers":1}
JSON
cat "$OUT/run_contract.json"
date -Is > "$OUT/started_at.txt"

set +e
# shellcheck disable=SC2086
$NC run --rm --net host --ipc host --privileged \
  --security-opt apparmor=unconfined \
  --env PYPTO_LIVE_IPC_STRICT=1 \
  --env PYPTO_H4_RESIDENT="$H4_RESIDENT" \
  --env PYPTO_CODEGEN_MAX_WORKERS=1 \
  --env PYPTO_RELEASE_CONTEXT_LEN=65536 \
  --env PYPTO_STEP3P5_MAX_SEQ=65536 \
  --env PYPTO_STEP3P5_ROPE_SEQ=65536 \
  --env SIMPLER_A2A3_FORCE_VMM_IPC=1 \
  --env PYPTO_IPC_SESSION_NONCE="$NONCE" \
  --env PYPTO_REQUIRE_L2_SWIMLANE_REUSE_DEP_GEN=1 \
  --env PYPTO_PROG_BUILD_DIR=/out/build_output \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONNOUSERSITE=1 \
  --env PTO2_RING_HEAP=4294967296 \
  --env PTO2_RING_TASK_WINDOW=131072 \
  --env PTO2_RING_DEP_POOL=131072 \
  --env PYPTO_STEP3P5_STORAGE_BATCH_CAPACITY=16 \
  --env PYPTO_DEVICES="$DEVCSV" \
  --env CKPT="$CKPT" \
  $DEVS --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v "$CKPT":"$CKPT":ro -v "$OUT":/out --shm-size 32g \
  "$IMGID" bash -lc '
    set -Eeuo pipefail
    cd /workspace/pypto-lib
    PYPTO_IPC_LAUNCH_EPOCH=$(python -c "import time; print(f\"{time.time():.9f}\")")
    export PYPTO_IPC_LAUNCH_EPOCH
    sha256sum models/step3p5/decode_fwd.py > /out/decode_fwd.container.sha256
    sha256sum /workspace/pypto/python/pypto/runtime/distributed_runner.py \
      > /out/distributed_runner.container.sha256
    git -C /workspace/pypto-lib rev-parse HEAD > /out/pypto_lib_head.txt 2>/dev/null || true
    git -C /workspace/pypto     rev-parse HEAD > /out/pypto_head.txt     2>/dev/null || true
    python -m tests.step3p5.harnesses._stage_five_layer_moe \
      --device "$PYPTO_DEVICES" --ckpt "$CKPT" --out /out/runtime \
      --num-blocks 512 --context-len 65536 --active-batch 1 \
      --seed-token 6127 --warmup 3 --iters 20 --dfx
    echo SWIMLANE_RUN_OK' \
  2>&1 | tee "$OUT/container.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$OUT/container.rc"
date -Is > "$OUT/finished_at.txt"

echo "===== container rc=$rc ====="
echo "===== l2_swimlane_records.json presence (the DFX gate's hard requirement) ====="
find "$OUT" -name "l2_swimlane_records.json" | sed 's|.*/dfx_outputs/||' | sort | head -20
echo "===== merged swimlane + critical path per rank ====="
find "$OUT" -name "critical_path_report.md" | sort | while read -r f; do
  r=$(echo "$f" | grep -o 'rank[0-7]')
  m=$(grep -m1 -iE "makespan" "$f" | tr -s ' ')
  echo "$r  $m"
done
echo "[gate] OUT=$OUT rc=$rc"
[ "$rc" = 0 ] && echo SWIMLANE_GATE_RC0 || echo SWIMLANE_GATE_RC_NONZERO
