#!/usr/bin/env bash
# ITL (inter-token latency) measurement for a candidate image, per canonical §5.
#
# Two clean (uninstrumented) runs on cards 0-7, inside the target image:
#   1. ctx=65536 with iters=1000 -- directly comparable to the R5 source-overlay
#      number (ctx-64K BS1 p50 26.329 ms @ITERS=1000) and to the K8 image
#      (32.14 ms @warmup=10/iters=100). A long run also maximises exposure to
#      probabilistic liveness defects, which one short run per config cannot buy.
#   2. the 4-point context curve at iters=20/warmup=3 -- comparable to the
#      2026-07-29 release-image curve (70.196 / 71.549 / 77.459 / 83.529 ms mean).
#
# PYPTO_PROG_BUILD_DIR is baked to /tmp/pypto_build_output in the image, so it is
# mounted out; with --rm an unmounted build dir loses every artifact.
# H4 resident constants are enabled by default for release measurements.
# Set PYPTO_H4_RESIDENT=none (or another supported mode) for rollback or diagnostics.
# Usage: bash run_itl_gate.sh <image-id>
set -Eeuo pipefail

# nerdctl resolves apparmor_parser relative to cwd; run from /tmp and pass
# apparmor=unconfined on every container.
cd /tmp

IMGID=${1:?need image id or digest}
H4_RESIDENT=${PYPTO_H4_RESIDENT:-all}
case "$H4_RESIDENT" in
  none|rope|gate|all) ;;
  *)
    echo "FAIL: PYPTO_H4_RESIDENT=$H4_RESIDENT is invalid" >&2
    exit 2
    ;;
esac
D=/mnt/persist/chensiyu/workspace/upgrade-20260821
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
OUT=$D/itl-$(date +%Y%m%d-%H%M%S)
NC="sudo -n /mnt/persist/k8s-install/containerd/bin/nerdctl"
H=/workspace/vllm-pypto

mkdir -p "$OUT"/{long,curve,build}
DEVS=""; for i in 0 1 2 3 4 5 6 7; do DEVS="$DEVS --device /dev/davinci$i"; done
COMMON="--rm --net host --ipc host --privileged --security-opt apparmor=unconfined $DEVS
  --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v $CKPT:$CKPT:ro -v $OUT:/out -v $OUT/build:/tmp/pypto_build_output --shm-size 32g"
COMMON="$COMMON --env PYPTO_H4_RESIDENT=$H4_RESIDENT"

{
  echo "schema=step3p5.itl-gate.v1"
  echo "host=$(hostname -f)"
  echo "image=$IMGID"
  echo "checkpoint=$CKPT"
  echo "host_devices=0,1,2,3,4,5,6,7"
  echo "platform=a2a3 active_batch=1 num_blocks=512 overrides=none"
  echo "h4_resident=$H4_RESIDENT"
  echo "started=$(date -Is)"
} > "$OUT/run_contract.txt"
cat "$OUT/run_contract.txt"

itl() {  # name  outdir  extra-args...
  local name=$1 sub=$2; shift 2
  echo "[itl] $name: $*"
  set +e
  # shellcheck disable=SC2086
  $NC run $COMMON "$IMGID" bash -lc "cd $H && \
    python -m tests.step3p5.harnesses._stage_main_hidden_only \
      --device 0,1,2,3,4,5,6,7 --dev 0 --ckpt $CKPT --out /out/$sub \
      --num-blocks 512 --active-batch 1 $*" > "$OUT/$name.log" 2>&1
  local rc=$?
  set -e
  echo "[itl] $name rc=$rc"
  grep -E "RESULT=|ITL|p50|tp_spread" "$OUT/$name.log" | tail -12 || true
  return $rc
}

set +e
itl long  long  --itl-context-lens 65536                  --itl-iters 1000 --itl-warmup 10
rc_long=$?
itl curve curve --itl-context-lens 1024,8192,32768,65536  --itl-iters 20   --itl-warmup 3
rc_curve=$?
set -e

echo "===== itl_report.json summaries ====="
for f in "$OUT"/long/itl_report.json "$OUT"/curve/itl_report.json; do
  [ -f "$f" ] || { echo "MISSING $f"; continue; }
  echo "--- $f"
  python3 -c "
import json
d = json.load(open('$f'))
print('num_blocks=%s active_batch=%s warmup=%s' % (
    d.get('num_blocks'), d.get('active_batch'), d.get('warmup')))
for p in d.get('results', []):
    print('ctx=%-6d iters=%-5d p50=%-9s mean=%-9s p99=%-9s min=%-9s max=%s' % (
        p['context_len'], p['iters'], p['itl_ms_p50'], p['itl_ms_mean'],
        p['itl_ms_p99'], p['itl_ms_min'], p['itl_ms_max']))
"
done

{ echo "finished=$(date -Is)"; echo "rc_long=$rc_long rc_curve=$rc_curve"; } >> "$OUT/run_contract.txt"
echo "[gate] OUT=$OUT"
gate_rc=0
[[ $rc_long -eq 0 && $rc_curve -eq 0 ]] || gate_rc=1
for report in "$OUT/long/itl_report.json" "$OUT/curve/itl_report.json"; do
  [[ -s $report ]] || gate_rc=1
done
if [[ $gate_rc -eq 0 ]]; then
  echo ITL_GATE_RC0
else
  echo ITL_GATE_RC_NONZERO >&2
fi
exit "$gate_rc"
