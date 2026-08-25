#!/usr/bin/env bash
# Release-grade precision acceptance (精度准出) for a candidate image.
#
# Two stages, both INSIDE the target image so the evidence is image-grade:
#   A. vanilla vLLM W8A8 oracle (torch_npu path, pypto bridge NOT engaged) serves
#      the same checkpoint and is stepped one token at a time with explicit ids;
#   B. the pypto whole-net holder replays that exact id sequence teacher-forced
#      and we count per-position top-1 agreement.
#
# Acceptance = N=128, ALIGNED >= 95%. A single-token argmax==303 liveness result
# is NOT a precision PASS.
#
# Both stages run on cards 0-7 sequentially, so only 0162-cards0-7.lock is held
# and codex keeps 8-15. Usage: bash run_precision_gate.sh <image-id> [N] [SEED]
set -Eeuo pipefail

# nerdctl resolves apparmor_parser relative to cwd, so any invocation from a
# directory that has no ./apparmor_parser but is also not / trips a fatal
# "resolves to executable in current directory". Run from /tmp and pass
# apparmor=unconfined on EVERY container -- a single missing flag is fatal.
cd /tmp

IMGID=${1:?need image id or digest}
N=${2:-128}
SEED=${3:-6127}
THRESHOLD=${THRESHOLD:-95}
H4_RESIDENT=${PYPTO_H4_RESIDENT:-}

case "$H4_RESIDENT" in
  ""|none|rope|gate|all) ;;
  *)
    echo "FAIL: PYPTO_H4_RESIDENT=$H4_RESIDENT is invalid" >&2
    exit 2
    ;;
esac

D=/mnt/persist/chensiyu/workspace/upgrade-20260821
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
R5=/mnt/persist/chensiyu/workspace/perf-2026q3/moe-routed-packed-fusion-20260815/r5-live-oracle-20260820-20260820-104357
MANIFEST=${MANIFEST:-$R5/checkpoint_identity.json}
FROZEN=${FROZEN:-$R5/oracle_ids.json}
OUT=$D/precision-$(date +%Y%m%d-%H%M%S)
NC="sudo -n /mnt/persist/k8s-install/containerd/bin/nerdctl"
SEC="--security-opt apparmor=unconfined"
ORACLE_CTR=pypto-upg-vanilla-oracle-8000
GATE=/workspace/vllm-pypto/tests/step3p5/ci/live_precision_gate.py
GEN=/workspace/vllm-pypto/tests/step3p5/ci/gen_vanilla_oracle.py

mkdir -p "$OUT/ipc" "$OUT/artifacts" "$OUT/build"
cp "$MANIFEST" "$OUT/checkpoint_identity.json"
DEVS=""; for i in 0 1 2 3 4 5 6 7; do DEVS="$DEVS --device /dev/davinci$i"; done
DEVOPTS="$DEVS --device /dev/davinci_manager --device /dev/hisi_hdc --device /dev/devmm_svm
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro"
# CPU-only client: no devices, but needs host net for :8000 and the tokenizer.
CLIENT="--rm --net host $SEC -v $CKPT:$CKPT:ro -v $OUT:/out"
H4_ENV=()
if [ -n "$H4_RESIDENT" ]; then
  H4_ENV=(--env "PYPTO_H4_RESIDENT=$H4_RESIDENT")
fi

{
  echo "schema=step3p5.precision-gate.v1"
  echo "host=$(hostname -f)"
  echo "image=$IMGID"
  echo "checkpoint=$CKPT"
  echo "manifest=$MANIFEST"
  echo "host_devices=0,1,2,3,4,5,6,7"
  echo "n=$N seed=$SEED threshold=$THRESHOLD"
  echo "h4_resident=${H4_RESIDENT:-none}"
  echo "started=$(date -Is)"
} > "$OUT/run_contract.txt"
cat "$OUT/run_contract.txt"

cleanup() { $NC rm -f "$ORACLE_CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# ---------------------------------------------------------------- stage A
echo "[stage A] starting vanilla vLLM oracle on cards 0-7, port 8000"
$NC rm -f "$ORACLE_CTR" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
$NC run -d --name "$ORACLE_CTR" --net host --ipc host --privileged $SEC $DEVOPTS \
  -v "$CKPT":"$CKPT":ro -v "$OUT":/out --shm-size 32g \
  "$IMGID" bash -lc "
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V1=1
  export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
  export HCCL_OP_EXPANSION_MODE=AIV HCCL_BUFFSIZE=512 TASK_QUEUE_ENABLE=0
  export VLLM_ASCEND_ENABLE_FLASHCOMM1=0 SHM_BARRIER=true
  export VLLM_ASCEND_ENABLE_PREFETCH_MLP=0
  unset CPU_AFFINITY_CONF VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE
  # The oracle must be the upstream torch_npu path. models/step3p5.py routes to
  # pypto only when PYPTO_STEP3P5_TAIL_ONLY is truthy (default '0'); unset it so
  # an inherited value cannot silently make the oracle the candidate itself.
  unset PYPTO_STEP3P5_TAIL_ONLY
  exec vllm serve $CKPT --trust-remote-code --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 --port 8000 --gpu-memory-utilization 0.90 \
    --no-enable-prefix-caching --served-model-name step3.5-flash \
    --quantization ascend --max-num-seqs 16 --async-scheduling \
    --max-num-batched-tokens 16384 --enable-expert-parallel --enforce-eager" \
  > "$OUT/oracle_container.id"

for _ in $(seq 1 150); do
  if curl -fsS --max-time 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then break; fi
  $NC inspect -f '{{.State.Running}}' "$ORACLE_CTR" 2>/dev/null | grep -q true || {
    echo "FAIL: oracle container died during startup"; $NC logs --tail 60 "$ORACLE_CTR"; exit 3; }
  sleep 10
done
curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models > "$OUT/oracle_models.json" || {
  echo "FAIL: oracle never became healthy"; $NC logs --tail 80 "$ORACLE_CTR"; exit 3; }
echo "[stage A] oracle healthy"

# The oracle client runs in a SEPARATE short-lived container, not `nerdctl exec`:
# the image declares WorkingDir=/workspace and containerd's exec path refuses to
# chdir into it ("chdir to cwd failed"), while `run` handles it fine.
# shellcheck disable=SC2086
$NC run $CLIENT "$IMGID" bash -lc \
  "python $GATE verify-checkpoint --checkpoint $CKPT \
     --manifest /out/checkpoint_identity.json --out /out/oracle_checkpoint_identity.json" \
  | tee "$OUT/oracle_verify.txt"

# shellcheck disable=SC2086
$NC run $CLIENT "$IMGID" bash -lc \
  "python $GEN --ckpt $CKPT --seed-token $SEED --n $N" > "$OUT/oracle.txt" 2>&1 || {
  echo "FAIL: oracle generation failed"; tail -30 "$OUT/oracle.txt"; exit 3; }
grep -E "^SEED=|^ORACLE_TEXT=" "$OUT/oracle.txt" || true

$NC logs "$ORACLE_CTR" > "$OUT/oracle_server.log" 2>&1 || true
$NC rm -f "$ORACLE_CTR" >/dev/null 2>&1 || true
echo "[stage A] oracle container stopped"

# Extract, then report the agreement PREFIX against an earlier frozen sequence.
# Not a binary match: a 128-step greedy self-continuation is chaotic, so two
# oracles from different image builds share a long prefix and then diverge for
# good (2026-07-27 vs 2026-08-20 agree on 0..48 and differ on 75 of the rest).
# An early divergence (< 8) means the checkpoint or serving config is wrong; a
# late one is expected and harmless.
# shellcheck disable=SC2086
$NC run $CLIENT "$IMGID" bash -lc \
  "python $GATE extract-oracle --log /out/oracle.txt --expected $N --out /out/oracle_ids.json"
if [ -f "$FROZEN" ]; then
  python3 -c "
import json
def ids(p):
    d = json.load(open(p))
    return d['oracle_ids'] if isinstance(d, dict) else d
new, old = ids('$OUT/oracle_ids.json'), ids('$FROZEN')
k = 0
while k < min(len(new), len(old)) and new[k] == old[k]:
    k += 1
print('ORACLE_AGREEMENT_PREFIX=%d vs %s' % (k, '$FROZEN'))
" | tee "$OUT/oracle_prefix.txt"
fi

# ---------------------------------------------------------------- stage B
echo "[stage B] pypto teacher-forced replay, $N steps"
set +e
# shellcheck disable=SC2086
$NC run --rm --net host --ipc host --privileged $SEC $DEVOPTS \
  "${H4_ENV[@]}" \
  -v "$CKPT":"$CKPT":ro -v "$OUT":/out --shm-size 32g \
  -v "$OUT/ipc":/tmp/n1_ci -v "$OUT/artifacts":/tmp/n1_artifacts \
  -v "$OUT/build":/tmp/pypto_build_output \
  "$IMGID" bash -lc "cd /workspace/vllm-pypto && \
    python $GATE verify-checkpoint --checkpoint $CKPT \
      --manifest /out/checkpoint_identity.json --out /out/pypto_checkpoint_identity.json && \
    python $GATE compare-checkpoint --oracle /out/oracle_checkpoint_identity.json \
      --pypto /out/pypto_checkpoint_identity.json && \
    read -r -a A <<< \"\$(python $GATE render-args --oracle-json /out/oracle_ids.json --expected $N)\" && \
    python -m tests.step3p5.harnesses._stage_main_hidden_only \
      --device 0,1,2,3,4,5,6,7 --dev 0 --out /out --ckpt $CKPT \
      --seed-token $SEED --teacher-forced --steps $N --num-blocks 32 \"\${A[@]}\"" \
  > "$OUT/pypto.log" 2>&1
rc=$?
set -e
echo "[stage B] rc=$rc"
grep -E "RESULT=|TEACHER_FORCED_MATCH" "$OUT/pypto.log" | tail -3 || true

# ---------------------------------------------------------------- verdict
set +e
# shellcheck disable=SC2086
$NC run $CLIENT "$IMGID" bash -lc \
  "python $GATE validate-result --log /out/pypto.log --oracle-json /out/oracle_ids.json \
     --expected $N --threshold $THRESHOLD --seed $SEED --release" \
  | tee "$OUT/verdict.txt"
grc=${PIPESTATUS[0]}
set -e

{ echo "finished=$(date -Is)"; echo "stage_b_rc=$rc gate_rc=$grc"; } >> "$OUT/run_contract.txt"
echo "[gate] OUT=$OUT"
gate_rc=0
[[ $rc -eq 0 && $grc -eq 0 ]] || gate_rc=1
[[ -s "$OUT/verdict.txt" ]] || gate_rc=1
if [[ $gate_rc -eq 0 ]]; then
  echo PRECISION_GATE_PASS
else
  echo PRECISION_GATE_FAIL >&2
fi
exit "$gate_rc"
