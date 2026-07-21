#!/usr/bin/env bash
# Compare 0162-style clean environment with the 0234 RJob communication
# environment without replacing active runtime binaries or modifying source.
#
# Usage:
#   MODE=clean ./develop/N1/run-0234-env-only-p1-s3-ab.sh
#   MODE=rjob_hccl ./develop/N1/run-0234-env-only-p1-s3-ab.sh
#   MODE=rjob_full ./develop/N1/run-0234-env-only-p1-s3-ab.sh
set -euo pipefail

MODE=${MODE:?set MODE=clean, MODE=rjob_hccl, or MODE=rjob_full}
case "$MODE" in
  clean|rjob_hccl|rjob_full) ;;
  *)
    echo "unsupported MODE=$MODE" >&2
    exit 2
    ;;
esac

WS=${WS:-/data/chensiyu/hw_project/pypto/workspace/n1-live-exactcann-stableruntime-20260717}
PY_ROOT=${PY_ROOT:-/data/chensiyu/hw_project/pypto/workspace/python0162-exact-20260717}
CANN_HOME=${CANN_HOME:-/data/chensiyu/cann900-0162-exact-20260717}
CANN_SETENV=${CANN_SETENV:-/data/chensiyu/hw_project/pypto/workspace/logs_n1/env_ab_scripts_20260717/cann-wrapper-set_env.sh}
CKPT=${CKPT:-/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp}

# The RJob login shell contains beta.1/8.5.1/ATB/vLLM paths. Re-exec once from
# an empty environment, then add back only the variables selected by MODE.
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
    MODE="$MODE" \
    WS="$WS" \
    PY_ROOT="$PY_ROOT" \
    CANN_HOME="$CANN_HOME" \
    CANN_SETENV="$CANN_SETENV" \
    CKPT="$CKPT" \
    PYPTO_CLEAN_ENV=1 \
    bash --noprofile --norc "$0"
fi

PY="$PY_ROOT/.venv311/bin/python"
STAMP=$(date +%Y%m%d_%H%M%S)
CASE="envonly_${MODE}_currentruntime_p1_s3_0234_${STAMP}"
OUT="/tmp/n1_${CASE}"
LOG_ROOT=/data/chensiyu/hw_project/pypto/workspace/logs_n1
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
  "$PY_ROOT/.venv311/pyvenv.cfg" \
  "$CANN_SETENV" \
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

if pgrep -af '[t]ests\.step3p5\._stage_whole_faithful_real_ipc|[c]hip_process' \
    >"/tmp/n1_${CASE}_preflight.txt"; then
  cat "/tmp/n1_${CASE}_preflight.txt" >&2
  echo "refusing to start while another N1 exporter/worker exists" >&2
  exit 4
fi

mkdir -p "$LOGDIR" "$OUT"
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
# This is a copied wrapper around the exact 0162 CANN tree. It differs from the
# original set_env.sh only in the absolute installation root embedded in it.
# shellcheck disable=SC1090
source "$CANN_SETENV"

# 0162 finds the driver and therefore does not append CANN devlib. The 0234
# container lacks driver discovery metadata, so remove only that extra entry.
clean_ld=""
IFS=: read -r -a ld_parts <<<"${LD_LIBRARY_PATH:-}"
for part in "${ld_parts[@]}"; do
  [[ "$part" == "$CANN_HOME/devlib" ]] && continue
  [[ -n "$part" ]] || continue
  clean_ld="${clean_ld:+${clean_ld}:}${part}"
done
export LD_LIBRARY_PATH="$clean_ld"
unset clean_ld ld_parts part

export PTOAS_ROOT="$WS/ptoas-bin"
export REPO="$WS/pypto"
export PYPTO_PROG_BUILD_DIR="$WS/build_output"
export PTO_ISA_ROOT="$WS/pto-isa"
export PATH="$PY_ROOT/.venv311/bin:$WS/ptoas-bin/bin:${PATH:-}"
export LD_LIBRARY_PATH="$WS/ptoas-bin/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$WS/pypto/python:$WS/pypto/runtime/python:$WS/pypto-lib"

# Remove previous scheduler/debug/vLLM co-tenancy experiments in both modes.
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
  P_FILL_BATCH \
  VLLM_USE_V1 \
  VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE

export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072
export P_FAITHFUL_MOE_LAYERS=1
export P_DBG_STAGE=3

if [[ "$MODE" == rjob_hccl || "$MODE" == rjob_full ]]; then
  # Compute/communication variables inherited by the 0234 RJob. Do not add
  # cloud credentials or unrelated platform variables.
  export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export CPU_AFFINITY_CONF=1
  export HCCL_CONNECT_TIMEOUT=3600
  export HCCL_INTRA_PCIE_ENABLE=0
  export HCCL_INTRA_ROCE_ENABLE=1
  export HCCL_OP_EXPANSION_MODE=AIV
  export HCCL_RDMA_RETRY_CNT=7
  export HCCL_RDMA_SL=4
  export HCCL_RDMA_TC=144
  export HCCL_RDMA_TIMEOUT=18
  export HCCL_WHITELIST_DISABLE=1
  export LCCL_DETERMINISTIC=0
  export LCCL_PARALLEL=0
  export OMP_NUM_THREADS=1
  export SOC_VERSION=ascend910b1
  export TASK_QUEUE_ENABLE=0
else
  unset \
    ASCEND_VISIBLE_DEVICES \
    CPU_AFFINITY_CONF \
    HCCL_CONNECT_TIMEOUT \
    HCCL_INTRA_PCIE_ENABLE \
    HCCL_INTRA_ROCE_ENABLE \
    HCCL_OP_EXPANSION_MODE \
    HCCL_RDMA_RETRY_CNT \
    HCCL_RDMA_SL \
    HCCL_RDMA_TC \
    HCCL_RDMA_TIMEOUT \
    HCCL_WHITELIST_DISABLE \
    LCCL_DETERMINISTIC \
    LCCL_PARALLEL \
    OMP_NUM_THREADS \
    SOC_VERSION \
    TASK_QUEUE_ENABLE
fi

if [[ "$MODE" == rjob_full ]]; then
  # Reproduce the remaining compute-relevant state of the long-lived 0234
  # shell while keeping the exact 0162 GA CANN tree first in every search
  # path. This tests whether trailing beta/8.5/ATB paths or unrelated runtime
  # switches explain the numerical failure without replacing any artifacts.
  export ASCEND_NNAL_ENV_SET=true
  export ASCEND_RUNTIME_OPTIONS=
  export ASCEND_TOOLKIT_ENV_SET=true
  export ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest

  export ATB_COMPARE_TILING_EVERY_KERNEL=0
  export ATB_HOME_PATH=/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
  export ATB_MATMUL_SHUFFLE_K_ENABLE=1
  export ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT=5
  export ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT=1
  export ATB_SHARE_MEMORY_NAME_SUFFIX=
  export ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE=0
  export ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE=0
  export ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE=0
  export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=1

  export NCCL_IB_DISABLE=0
  export NCCL_IB_FIFO_TC=192
  export NCCL_IB_PCI_RELAXED_ORDERING=1
  export NCCL_IB_QPS_PER_CONNECTION=8
  export NCCL_IB_RETRY_CNT=7
  export NCCL_IB_TC=186
  export NCCL_IB_TIMEOUT=21
  export NCCL_NVLS_ENABLE=0
  export NCCL_PXN_DISABLE=1
  export NCCL_RAS_ADDR=0.0.0.0:28028
  export NCCL_RAS_ENABLE=1
  export NCCL_SET_THREAD_NAME=1

  export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
  export VLLM_USE_V1=1
  export VLLM_VERSION=0.19.0
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2

  export PATH="$PATH:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/bin:/usr/local/Ascend/cann-9.0.0-beta.1/tools/bishengir/bin:/usr/local/Ascend/cann-9.0.0-beta.1/bin:/usr/local/Ascend/cann-9.0.0-beta.1/tools/ccec_compiler/bin:/usr/local/Ascend/cann-9.0.0-beta.1/tools/profiler/bin:/usr/local/Ascend/cann-9.0.0-beta.1/tools/ascend_system_advisor/asys:/usr/local/Ascend/cann-9.0.0-beta.1/tools/show_kernel_debug_data:/usr/local/Ascend/cann-9.0.0-beta.1/tools/msobjdump:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/Ascend/ascend-toolkit/latest/tools/ccec_compiler/bin:/usr/local/python3.11.14/bin"
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1/tests/atbopstest:/usr/local/python3.11.14/lib"
  export PYTHONPATH="$PYTHONPATH:$CANN_HOME/python/site-packages:$CANN_HOME/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/cann-9.0.0-beta.1/python/site-packages:/usr/local/Ascend/cann-9.0.0-beta.1/opp/built-in/op_impl/ai_core/tbe:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"
  export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/usr/local/Ascend/cann-9.0.0-beta.1/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-9.0.0-beta.1/lib64/cmake:/usr/local/Ascend/cann-8.5.1/toolkit/tools/tikicpulib/lib/cmake:/usr/local/Ascend/cann-8.5.1/lib64/cmake"
else
  unset \
    ASCEND_NNAL_ENV_SET \
    ASCEND_TOOLKIT_ENV_SET \
    ASCEND_TOOLKIT_LATEST_HOME \
    ATB_COMPARE_TILING_EVERY_KERNEL \
    ATB_HOME_PATH \
    ATB_MATMUL_SHUFFLE_K_ENABLE \
    ATB_OPSRUNNER_KERNEL_CACHE_GLOABL_COUNT \
    ATB_OPSRUNNER_KERNEL_CACHE_LOCAL_COUNT \
    ATB_SHARE_MEMORY_NAME_SUFFIX \
    ATB_STREAM_SYNC_EVERY_KERNEL_ENABLE \
    ATB_STREAM_SYNC_EVERY_OPERATION_ENABLE \
    ATB_STREAM_SYNC_EVERY_RUNNER_ENABLE \
    ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE \
    NCCL_IB_DISABLE \
    NCCL_IB_FIFO_TC \
    NCCL_IB_PCI_RELAXED_ORDERING \
    NCCL_IB_QPS_PER_CONNECTION \
    NCCL_IB_RETRY_CNT \
    NCCL_IB_TC \
    NCCL_IB_TIMEOUT \
    NCCL_NVLS_ENABLE \
    NCCL_PXN_DISABLE \
    NCCL_RAS_ADDR \
    NCCL_RAS_ENABLE \
    NCCL_SET_THREAD_NAME \
    VLLM_VERSION \
    LD_PRELOAD
fi

{
  echo "CASE=$CASE"
  echo "MODE=$MODE"
  echo "HOST=$(hostname)"
  date -Is
  echo "WS=$(readlink -f "$WS")"
  echo "PY=$PY"
  echo "CANN_HOME=$CANN_HOME"
  echo "CANN_SETENV=$CANN_SETENV"
  sha256sum "$PY" "$PY_ROOT/.venv311/pyvenv.cfg"
  sha256sum "$CANN_SETENV" "$CANN_HOME/opp/version.info"
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
  "$PY" -m tests.step3p5._stage_whole_faithful_real_ipc \
    --export-rank "$rank" \
    --dev "$rank" \
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
    exit 5
  fi
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rc=0
      wait "$pid" || rc=$?
      echo "exporter_exit_before_ready=$rc" | tee "$LOGDIR/result.txt"
      exit 6
    fi
  done
  sleep 10
done
echo "exporters_ready=8" | tee "$LOGDIR/result.txt"

set +e
timeout --signal=INT --kill-after=30s 900s \
  "$PY" -m tests.step3p5._stage_whole_faithful_real_ipc \
    --device 0,1,2,3,4,5,6,7 \
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
