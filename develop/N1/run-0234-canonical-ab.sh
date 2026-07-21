#!/usr/bin/env bash
# Run one isolated 0234 canonical case with an explicitly selected CANN root
# and runtime-binary directory. The caller must first stage the four binaries.
set -euo pipefail

WS=${WS:-/data/chensiyu/hw_project/pypto/workspace}
CANN_ROOT=${CANN_ROOT:?set CANN_ROOT}
CANN_SETENV=${CANN_SETENV:-$CANN_ROOT/set_env.sh}
RUNTIME_STAGE=${RUNTIME_STAGE:?set RUNTIME_STAGE}
CASE_NAME=${CASE_NAME:?set CASE_NAME}
PY=${PY:-$WS/.venv311/bin/python}
MOE_LAYERS=${MOE_LAYERS:-${P_FAITHFUL_MOE_LAYERS:-1}}
DBG_STAGE=${DBG_STAGE:-${P_DBG_STAGE:-3}}
DEVICE_BASE=${DEVICE_BASE:-0}
EXPECTED_ARGMAX=${EXPECTED_ARGMAX:-}
EXPECTED_MAX_NEXT=${EXPECTED_MAX_NEXT:-}
CKPT=${CKPT:-/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp}

if [[ -z "$EXPECTED_ARGMAX" ]]; then
  if [[ "$MOE_LAYERS" == 1 && "$DBG_STAGE" == 3 ]]; then
    EXPECTED_ARGMAX=27527
    EXPECTED_MAX_NEXT=${EXPECTED_MAX_NEXT:-892.0000}
  elif [[ "$MOE_LAYERS" == 42 && "$DBG_STAGE" == 0 ]]; then
    EXPECTED_ARGMAX=303
    EXPECTED_MAX_NEXT=${EXPECTED_MAX_NEXT:-264192.0000}
  fi
fi

# Long-lived tmux panes on 0234 have historically sourced beta.1, non-GA, and
# ascend-toolkit environments in the same shell. Re-exec exactly once with a
# minimal environment so this A/B changes only the explicitly selected CANN
# root and runtime-binary stage.
if [[ ${PYPTO_CLEAN_ENV:-0} != 1 ]]; then
  exec env -i \
    HOME="${HOME:-/root}" \
    USER="${USER:-root}" \
    LOGNAME="${LOGNAME:-root}" \
    SHELL=/bin/bash \
    TERM="${TERM:-xterm}" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    WS="$WS" \
    CANN_ROOT="$CANN_ROOT" \
    CANN_SETENV="$CANN_SETENV" \
    RUNTIME_STAGE="$RUNTIME_STAGE" \
    CASE_NAME="$CASE_NAME" \
    PY="$PY" \
    MOE_LAYERS="$MOE_LAYERS" \
    DBG_STAGE="$DBG_STAGE" \
    DEVICE_BASE="$DEVICE_BASE" \
    EXPECTED_ARGMAX="$EXPECTED_ARGMAX" \
    EXPECTED_MAX_NEXT="$EXPECTED_MAX_NEXT" \
    CKPT="$CKPT" \
    PYPTO_CLEAN_ENV=1 \
    bash --noprofile --norc "$0"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="/tmp/n1_canon_0234_${CASE_NAME}_${STAMP}"
LOGDIR="$WS/logs_n1/canon_0234_${CASE_NAME}_${STAMP}"
BACKUP="$WS/runtime-active-backup-${CASE_NAME}-${STAMP}"

active_host="$WS/pypto/runtime/build/lib/a2a3/onboard/tensormap_and_ringbuffer/libhost_runtime.so"
active_aicpu="$WS/pypto/runtime/build/lib/a2a3/onboard/tensormap_and_ringbuffer/libaicpu_kernel.so"
active_dispatch="$WS/pypto/runtime/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so"
active_task="$WS/pypto/runtime/build/cp311-cp311-linux_x86_64/python/bindings/_task_interface.cpython-311-x86_64-linux-gnu.so"

stage_host="$RUNTIME_STAGE/lib/a2a3/onboard/tensormap_and_ringbuffer/libhost_runtime.so"
stage_aicpu="$RUNTIME_STAGE/lib/a2a3/onboard/tensormap_and_ringbuffer/libaicpu_kernel.so"
stage_dispatch="$RUNTIME_STAGE/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so"
stage_task="$RUNTIME_STAGE/cp311-cp311-linux_x86_64/python/bindings/_task_interface.cpython-311-x86_64-linux-gnu.so"

for file in \
  "$active_host" "$active_aicpu" "$active_dispatch" "$active_task" \
  "$stage_host" "$stage_aicpu" "$stage_dispatch" "$stage_task" \
  "$CANN_SETENV" "$CANN_ROOT/opp/version.info" "$PY" \
  "$CKPT/config.json"; do
  if [[ ! -f "$file" ]]; then
    echo "required file is missing: $file" >&2
    exit 2
  fi
done

# Never replace a loaded runtime image. A fresh exporter/worker pool is part of
# the canonical object, not an optional cleanup detail.
if pgrep -af 'tests\.step3p5\._stage_whole_faithful_real_ipc' >"/tmp/n1_preflight_${CASE_NAME}.txt"; then
  cat "/tmp/n1_preflight_${CASE_NAME}.txt" >&2
  echo "refusing to replace runtime binaries while N1 exporters/workers exist" >&2
  exit 3
fi

mkdir -p "$LOGDIR" "$OUT" \
  "$BACKUP/lib/a2a3/onboard/tensormap_and_ringbuffer" \
  "$BACKUP/lib/a2a3/dispatcher" \
  "$BACKUP/cp311-cp311-linux_x86_64/python/bindings"

backup_complete=0
runtime_installed=0
restore_verified=0

restore_runtime() {
  if [[ "$backup_complete" == 1 && "$runtime_installed" == 1 ]]; then
    cp -f "$BACKUP/lib/a2a3/onboard/tensormap_and_ringbuffer/libhost_runtime.so" "$active_host"
    cp -f "$BACKUP/lib/a2a3/onboard/tensormap_and_ringbuffer/libaicpu_kernel.so" "$active_aicpu"
    cp -f "$BACKUP/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so" "$active_dispatch"
    cp -f "$BACKUP/cp311-cp311-linux_x86_64/python/bindings/_task_interface.cpython-311-x86_64-linux-gnu.so" "$active_task"
    runtime_installed=0
    if \
      cmp -s "$BACKUP/lib/a2a3/onboard/tensormap_and_ringbuffer/libhost_runtime.so" "$active_host" &&
      cmp -s "$BACKUP/lib/a2a3/onboard/tensormap_and_ringbuffer/libaicpu_kernel.so" "$active_aicpu" &&
      cmp -s "$BACKUP/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so" "$active_dispatch" &&
      cmp -s "$BACKUP/cp311-cp311-linux_x86_64/python/bindings/_task_interface.cpython-311-x86_64-linux-gnu.so" "$active_task"; then
      restore_verified=1
    fi
  fi
}

pids=()
cleanup() {
  touch "$OUT/STOP" 2>/dev/null || true
  local deadline=$((SECONDS + 45))
  local alive=1
  while [[ "$alive" == 1 && "$SECONDS" -lt "$deadline" ]]; do
    alive=0
    for pid in "${pids[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
      fi
    done
    [[ "$alive" == 0 ]] || sleep 1
  done
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
  restore_runtime
  {
    echo "ACTIVE_RUNTIME_RESTORED=$restore_verified"
    sha256sum "$active_host" "$active_aicpu" "$active_dispatch" "$active_task"
  } >>"$LOGDIR/result.txt" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Preserve the exact active build, then install the selected immutable stage.
cp -f "$active_host" "$BACKUP/lib/a2a3/onboard/tensormap_and_ringbuffer/"
cp -f "$active_aicpu" "$BACKUP/lib/a2a3/onboard/tensormap_and_ringbuffer/"
cp -f "$active_dispatch" "$BACKUP/lib/a2a3/dispatcher/"
cp -f "$active_task" "$BACKUP/cp311-cp311-linux_x86_64/python/bindings/"
backup_complete=1
cp -f "$stage_host" "$active_host"
cp -f "$stage_aicpu" "$active_aicpu"
cp -f "$stage_dispatch" "$active_dispatch"
cp -f "$stage_task" "$active_task"
runtime_installed=1

export LD_LIBRARY_PATH=
export PYTHONPATH=
export CMAKE_PREFIX_PATH=
# shellcheck disable=SC1090
source "$CANN_SETENV"
# 0234 does not expose the same driver-install metadata as 0162.  The wrapper
# therefore appends devlib, while the 0162 stable environment does not.  Drop
# only that entry to keep the loader search path matched; do not alter CANN or
# driver files.
clean_ld=""
IFS=: read -r -a ld_parts <<<"${LD_LIBRARY_PATH:-}"
for part in "${ld_parts[@]}"; do
  [[ "$part" == "$CANN_ROOT/devlib" ]] && continue
  [[ -n "$part" ]] || continue
  clean_ld="${clean_ld:+${clean_ld}:}${part}"
done
export LD_LIBRARY_PATH="$clean_ld"
unset clean_ld ld_parts part

export REPO="$WS/pypto"
export PTOAS_ROOT="$WS/ptoas-bin"
export PYPTO_PROG_BUILD_DIR="$WS/build_output"
export PATH="$(dirname "$PY"):$WS/ptoas-bin/bin:${PATH:-}"
export LD_LIBRARY_PATH="$WS/ptoas-bin/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PTO_ISA_ROOT="$WS/pto-isa"
export PYTHONPATH="$WS/pypto/python:$WS/pypto/runtime/python:$WS/pypto-lib"

# Canonical environment: remove inherited debug/scheduler experiments.
unset \
  PTO2_SERIAL_ORCH_SCHED \
  PTO2_SCHEDULER_TIMEOUT_MS \
  PTO2_OP_EXECUTE_TIMEOUT_US \
  PTO2_STREAM_SYNC_TIMEOUT_MS \
  ASCEND_RUNTIME_OPTIONS \
  ASCEND_GLOBAL_LOG_LEVEL \
  ASCEND_PROCESS_LOG_PATH \
  SIMPLER_COMM_NO_HCCL \
  SIMPLER_ENABLE_PTO_SDMA_WORKSPACE \
  ASCEND_VISIBLE_DEVICES \
  VLLM_USE_V1 \
  VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE \
  LD_PRELOAD
export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072
export P_FAITHFUL_MOE_LAYERS="$MOE_LAYERS"
export P_DBG_STAGE="$DBG_STAGE"

selected_cann=$(readlink -f "$CANN_ROOT")
for value in "$PATH" "${LD_LIBRARY_PATH:-}" "${PYTHONPATH:-}" "${CMAKE_PREFIX_PATH:-}"; do
  sanitized=${value//"$selected_cann"/}
  case "$sanitized" in
    *cann900-nonga*|*ascend-toolkit/latest*|*/cann-9.0.0-beta.1*)
      echo "unexpected secondary CANN path after clean activation: $value" >&2
      echo "selected CANN is: $selected_cann" >&2
      exit 4
      ;;
  esac
done

{
  echo "CASE=$CASE_NAME"
  echo "HOST=$(hostname)"
  date -Is
  echo "CANN_ROOT=$(readlink -f "$CANN_ROOT")"
  echo "CANN_SETENV=$CANN_SETENV"
  echo "PY=$PY"
  echo "MOE_LAYERS=$MOE_LAYERS"
  echo "DBG_STAGE=$DBG_STAGE"
  echo "DEVICE_BASE=$DEVICE_BASE"
  echo "EXPECTED_ARGMAX=$EXPECTED_ARGMAX"
  echo "EXPECTED_MAX_NEXT=$EXPECTED_MAX_NEXT"
  sha256sum "$CANN_SETENV" "$CANN_ROOT/opp/version.info" "$PY"
  sha256sum "$active_host" "$active_aicpu" "$active_dispatch" "$active_task"
  sha256sum "$stage_host" "$stage_aicpu" "$stage_dispatch" "$stage_task"
  sha256sum \
    "$CKPT/config.json" \
    "$CKPT/quant_model_description.json" \
    "$CKPT/quant_model_weights.safetensors.index.json"
  for repo in pypto-lib pypto pypto/runtime pto-isa PTOAS; do
    printf "%s " "$repo"
    # The 0234 tmux user is root while several shared worktrees are owned by
    # chensiyu. Avoid mutating root's global safe.directory config; use the
    # one-shot override for this manifest-only read.
    git -c safe.directory="$WS/$repo" -C "$WS/$repo" rev-parse HEAD
    git -c safe.directory="$WS/$repo" -C "$WS/$repo" status --short
  done
  env | grep -E '^(PTO2_|SIMPLER_|ASCEND_|PTO_ISA|P_FAITHFUL|P_DBG|PYTHONPATH|LD_LIBRARY_PATH|PATH=|CMAKE_PREFIX_PATH)' | sort
  "$PY" - <<'PY'
import _task_interface
import pypto
import simpler
import simpler_setup
import sys
print("python", sys.executable)
print("version", sys.version)
print("pypto", pypto.__file__)
print("simpler", simpler.__file__)
print("simpler_setup", simpler_setup.__file__)
print("_task_interface", _task_interface.__file__)
print("sys.path", *sys.path, sep="\n  ")
PY
  echo "=== LOADER RESOLUTION ==="
  ldd "$active_host" | grep -E 'cann|ascend|hccl|not found' || true
} >"$LOGDIR/manifest.txt" 2>&1

cd "$WS/pypto-lib"

for rank in $(seq 0 7); do
  "$PY" -m tests.step3p5._stage_whole_faithful_real_ipc \
    --export-rank "$rank" \
    --dev "$((DEVICE_BASE + rank))" \
    --kv-ipc \
    --out "$OUT" \
    --ckpt "$CKPT" \
    >"$LOGDIR/exp_${rank}.log" 2>&1 &
  pids+=("$!")
done

deadline=$((SECONDS + 360))
while [ "$(find "$OUT" -maxdepth 1 -name 'ready.rank*' | wc -l)" -lt 8 ]; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "exporter readiness timeout" | tee -a "$LOGDIR/result.txt"
    exit 1
  fi
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rc=0
      wait "$pid" || rc=$?
      echo "exporter pid=$pid exited before ready rc=$rc" | tee -a "$LOGDIR/result.txt"
      exit 1
    fi
  done
  sleep 10
done

set +e
timeout --signal=INT --kill-after=30s "${WORKER_TIMEOUT:-900}s" \
  "$PY" -m tests.step3p5._stage_whole_faithful_real_ipc \
  --device "$(
    seq "$DEVICE_BASE" "$((DEVICE_BASE + 7))" | paste -sd, -
  )" \
  --reuse-exporters \
  --kv-ipc \
  --hidden-token 6127 \
  --out "$OUT" \
  --ckpt "$CKPT" \
  >"$LOGDIR/worker.log" 2>&1
worker_rc=$?
set -e

{
  echo "worker_rc=$worker_rc"
  grep -aE 'RUN done|argmax=|TOP5|RESULT=|507018|running-stalled|orch_error|sched_error|Traceback|Error|nan|NaN' \
    "$LOGDIR/worker.log" | tail -40
  echo "LOGDIR=$LOGDIR"
} | tee "$LOGDIR/result.txt"

# Success is semantic, not merely process-level.  P1/S3 has its own stable
# diagnostic fingerprint (argmax 27527); argmax 303 is reserved for P42/S0.
semantic_rc=0
if [[ "$worker_rc" -ne 0 ]]; then
  semantic_rc="$worker_rc"
elif ! grep -aq '\[worker\] RUN done' "$LOGDIR/worker.log"; then
  echo "SEMANTIC_FAIL=missing_RUN_done" | tee -a "$LOGDIR/result.txt"
  semantic_rc=10
elif grep -a '\[worker\] RUN done' "$LOGDIR/worker.log" |
     grep -qE '(^|[^A-Za-z])(nan|NaN|inf|Inf)([^A-Za-z]|$)'; then
  echo "SEMANTIC_FAIL=non_finite_output" | tee -a "$LOGDIR/result.txt"
  semantic_rc=11
elif [[ -n "$EXPECTED_ARGMAX" ]] &&
     ! grep -aqE "argmax=${EXPECTED_ARGMAX}([^0-9]|$)" "$LOGDIR/worker.log"; then
  echo "SEMANTIC_FAIL=argmax_not_${EXPECTED_ARGMAX}" | tee -a "$LOGDIR/result.txt"
  semantic_rc=12
elif [[ -n "$EXPECTED_MAX_NEXT" ]] &&
     ! grep -aqE "max\\|next_hidden\\|=${EXPECTED_MAX_NEXT}([^0-9]|$)" "$LOGDIR/worker.log"; then
  echo "SEMANTIC_FAIL=max_next_hidden_not_${EXPECTED_MAX_NEXT}" | tee -a "$LOGDIR/result.txt"
  semantic_rc=13
else
  echo "SEMANTIC_PASS=finite_argmax_${EXPECTED_ARGMAX:-unchecked}" | tee -a "$LOGDIR/result.txt"
fi

exit "$semantic_rc"
