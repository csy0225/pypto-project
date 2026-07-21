#!/usr/bin/env bash
# Run the current canonical source on the 0162 stable machine with the same
# diagnostic cut used by the 0234 clean-environment test:
#   P_FAITHFUL_MOE_LAYERS=1, P_DBG_STAGE=3, hidden token 6127.
#
# Safety:
# - never reuses or removes an existing OUT directory;
# - never replaces runtime binaries or modifies CANN links;
# - refuses to run beside an existing N1 exporter/worker;
# - asks exporters to stop and waits for them before exit.
set -euo pipefail

WS=${WS:-/data/chensiyu/hw_project/pypto/workspace}
CKPT=${CKPT:-/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp}
CANN_HOME=${CANN_HOME:-/usr/local/Ascend/cann}
PY=${PY:-$WS/.venv311/bin/python}
MOE_LAYERS=${MOE_LAYERS:-${P_FAITHFUL_MOE_LAYERS:-1}}
DBG_STAGE=${DBG_STAGE:-${P_DBG_STAGE:-3}}
DEVICE_BASE=${DEVICE_BASE:-8}

# Re-enter once without the long-lived login-shell environment. This makes the
# manifest comparable with run-0234-env-only-p1-s3-ab.sh while retaining the
# stable machine's own CANN, Python, runtime, driver, and physical device IDs.
if [[ ${PYPTO_CLEAN_ENV:-0} != 1 ]]; then
  exec env -i \
    HOME="${HOME:-/home/infra}" \
    USER="${USER:-infra}" \
    LOGNAME="${LOGNAME:-infra}" \
    SHELL=/bin/bash \
    TERM="${TERM:-xterm}" \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    WS="$WS" \
    CKPT="$CKPT" \
    CANN_HOME="$CANN_HOME" \
    PY="$PY" \
    MOE_LAYERS="$MOE_LAYERS" \
    DBG_STAGE="$DBG_STAGE" \
    DEVICE_BASE="$DEVICE_BASE" \
    PYPTO_CLEAN_ENV=1 \
    bash --noprofile --norc "$0"
fi

STAMP=$(date +%Y%m%d_%H%M%S)
CASE="matched_clean_p${MOE_LAYERS}_s${DBG_STAGE}_0162_${STAMP}"
OUT="/tmp/n1_${CASE}"
LOG_ROOT="$WS/logs_n1"
LOGDIR="$LOG_ROOT/$CASE"

RUNTIME_FILES=(
  "$WS/pypto/runtime/build/lib/a2a3/onboard/tensormap_and_ringbuffer/libhost_runtime.so"
  "$WS/pypto/runtime/build/lib/a2a3/onboard/tensormap_and_ringbuffer/libaicpu_kernel.so"
  "$WS/pypto/runtime/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so"
  "$WS/pypto/runtime/build/cp311-cp311-linux_x86_64/python/bindings/_task_interface.cpython-311-x86_64-linux-gnu.so"
)
SOURCE_FILES=(
  "$WS/pypto-lib/models/step3p5/decode_layer.py"
  "$WS/pypto-lib/models/step3p5/moe.py"
  "$WS/pypto-lib/tools/step3p5/_gen_faithful_real.py"
)

for file in \
  "$PY" \
  "$CANN_HOME/set_env.sh" \
  "$CANN_HOME/opp/version.info" \
  "$CKPT/config.json" \
  "$CKPT/quant_model_description.json" \
  "$CKPT/quant_model_weights.safetensors.index.json" \
  "${RUNTIME_FILES[@]}" \
  "${SOURCE_FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "required file is missing: $file" >&2
    exit 3
  fi
done

if [[ -e "$OUT" ]]; then
  echo "refusing to reuse existing OUT=$OUT" >&2
  exit 4
fi
if pgrep -af '[t]ests\.step3p5\._stage_whole_faithful_real_ipc|[c]hip_process' \
    >"/tmp/n1_${CASE}_preflight.txt"; then
  cat "/tmp/n1_${CASE}_preflight.txt" >&2
  echo "refusing to start while another N1 exporter/worker exists" >&2
  exit 5
fi

mkdir -p "$OUT" "$LOGDIR"
sha256sum "${RUNTIME_FILES[@]}" >"$LOGDIR/runtime.before.sha256"
sha256sum "${SOURCE_FILES[@]}" >"$LOGDIR/source.before.sha256"

pids=()
cleanup() {
  touch "$OUT/STOP" 2>/dev/null || true

  local deadline=$((SECONDS + 60))
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
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in "${pids[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done

  sha256sum "${RUNTIME_FILES[@]}" >"$LOGDIR/runtime.after.sha256" 2>/dev/null || true
  sha256sum "${SOURCE_FILES[@]}" >"$LOGDIR/source.after.sha256" 2>/dev/null || true
  if cmp -s "$LOGDIR/runtime.before.sha256" "$LOGDIR/runtime.after.sha256"; then
    echo "ACTIVE_RUNTIME_UNCHANGED=1" >>"$LOGDIR/result.txt"
  else
    echo "ACTIVE_RUNTIME_UNCHANGED=0" >>"$LOGDIR/result.txt"
  fi
  if cmp -s "$LOGDIR/source.before.sha256" "$LOGDIR/source.after.sha256"; then
    echo "SOURCE_UNCHANGED=1" >>"$LOGDIR/result.txt"
  else
    echo "SOURCE_UNCHANGED=0" >>"$LOGDIR/result.txt"
  fi
}
trap cleanup EXIT INT TERM

export LD_LIBRARY_PATH=
export PYTHONPATH=
export CMAKE_PREFIX_PATH=
# shellcheck disable=SC1090
source "$CANN_HOME/set_env.sh"

export PTOAS_ROOT="$WS/ptoas-bin"
export REPO="$WS/pypto"
export PYPTO_PROG_BUILD_DIR="$WS/build_output"
export PTO_ISA_ROOT="$WS/pto-isa"
export PATH="$WS/.venv311/bin:$WS/ptoas-bin/bin:${PATH:-}"
export LD_LIBRARY_PATH="$WS/ptoas-bin/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WS/pypto/python:$WS/pypto/runtime/python:$WS/pypto-lib"

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

{
  echo "CASE=$CASE"
  echo "HOST=$(hostname)"
  date -Is
  echo "WS=$(readlink -f "$WS")"
  echo "PY=$PY"
  echo "CANN_HOME=$(readlink -f "$CANN_HOME")"
  sha256sum "$PY" "$WS/.venv311/pyvenv.cfg"
  sha256sum "$CANN_HOME/set_env.sh" "$CANN_HOME/opp/version.info"
  cat "$LOGDIR/runtime.before.sha256"
  cat "$LOGDIR/source.before.sha256"
  sha256sum \
    "$CKPT/config.json" \
    "$CKPT/quant_model_description.json" \
    "$CKPT/quant_model_weights.safetensors.index.json"
  echo "=== ENV ==="
  env | sort
  echo "=== PYTHON ==="
  "$PY" - <<'PY'
import site
import sys

import numpy
import pypto
import safetensors
import simpler
import torch

print("python", sys.executable)
print("version", sys.version)
print("prefix", sys.prefix)
print("base_prefix", sys.base_prefix)
print("ENABLE_USER_SITE", site.ENABLE_USER_SITE)
for name, mod in (
    ("torch", torch),
    ("numpy", numpy),
    ("safetensors", safetensors),
    ("pypto", pypto),
    ("simpler", simpler),
):
    print(name, getattr(mod, "__version__", "NO_VERSION"), mod.__file__)
print("sys.path", *sys.path, sep="\n  ")
PY
} >"$LOGDIR/manifest.txt" 2>&1

cd "$WS/pypto-lib"
for rank in $(seq 0 7); do
  dev=$((8 + rank))
  "$PY" -m tests.step3p5._stage_whole_faithful_real_ipc \
    --export-rank "$rank" \
    --dev "$((DEVICE_BASE + rank))" \
    --kv-ipc \
    --out "$OUT" \
    --ckpt "$CKPT" \
    >"$LOGDIR/exp_${rank}.log" 2>&1 &
  pids+=("$!")
done

deadline=$((SECONDS + 900))
while [[ $(find "$OUT" -maxdepth 1 -name 'ready.rank*' | wc -l) -lt 8 ]]; do
  if [[ "$SECONDS" -ge "$deadline" ]]; then
    echo "exporter_readiness_timeout=1" | tee "$LOGDIR/result.txt"
    exit 6
  fi
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rc=0
      wait "$pid" || rc=$?
      echo "exporter_exit_before_ready=$rc" | tee "$LOGDIR/result.txt"
      exit 7
    fi
  done
  sleep 10
done
echo "exporters_ready=8" | tee "$LOGDIR/result.txt"

set +e
timeout --signal=INT --kill-after=30s 900s \
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
  grep -aE \
    'RUN done|argmax=|TOP5|DUMPED|RESULT=|Traceback|Error|Exception|507018|running-stalled|orch_error|sched_error|nan|NaN|inf|Inf' \
    "$LOGDIR/worker.log" | tail -80 || true
  echo "LOGDIR=$LOGDIR"
  echo "OUT=$OUT"
} | tee -a "$LOGDIR/result.txt"

exit "$worker_rc"
