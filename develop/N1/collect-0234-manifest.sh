#!/usr/bin/env bash
# Collect the exact 0234 execution manifest through the existing tmux login.
# This script is read-only: it does not rebuild, replace, or clean artifacts.
set -euo pipefail

WS=${WS:-/data/chensiyu/hw_project/pypto/workspace}
CANN_ROOT=${CANN_ROOT:-/usr/local/Ascend/cann}
CANN_SETENV=${CANN_SETENV:-$CANN_ROOT/set_env.sh}
PY_ROOT=${PY_ROOT:-$WS}
PY=${PY:-$PY_ROOT/.venv311/bin/python}

# Do not let a long-lived tmux shell leak another CANN/Python installation into
# the manifest. Re-exec once with an explicit, minimal environment.
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
    PY_ROOT="$PY_ROOT" \
    PY="$PY" \
    PYPTO_CLEAN_ENV=1 \
    bash --noprofile --norc "$0"
fi

echo "HOST=$(hostname)"
date -Is

echo "=== CANN ==="
readlink -f "$CANN_ROOT" 2>&1 || true
sha256sum \
  "$CANN_SETENV" \
  "$CANN_ROOT/opp/version.info" 2>&1 || true

echo "=== SOURCE PINS AND STATUS ==="
for repo in pypto-lib pypto pypto/runtime pto-isa PTOAS; do
  path="$WS/$repo"
  printf "%s " "$path"
  # The RJob shell runs as root while the shared worktrees are owned by the
  # developer account. Keep the audit read-only and avoid mutating root's
  # global safe.directory list.
  git -c safe.directory="$path" -C "$path" rev-parse HEAD 2>&1 || true
  git -c safe.directory="$path" -C "$path" status --short 2>&1 || true
done

echo "=== RELEASE SOURCE HASHES ==="
sha256sum \
  "$WS/pypto-lib/models/step3p5/decode_layer.py" \
  "$WS/pypto-lib/models/step3p5/moe.py" \
  "$WS/pypto-lib/tools/step3p5/_gen_faithful_real.py" \
  "$WS/pypto/python/pypto/runtime/device_tensor.py" \
  "$WS/pypto/python/pypto/runtime/distributed_runner.py" \
  "$WS/pypto/runtime/python/simpler/worker.py" \
  "$WS/pypto/runtime/src/a2a3/platform/onboard/host/comm_hccl.cpp" 2>&1 || true

echo "=== ACTIVE RUNTIME BINARIES ==="
runtime_files=(
  "$WS/pypto/runtime/build/lib/a2a3/onboard/tensormap_and_ringbuffer/libhost_runtime.so"
  "$WS/pypto/runtime/build/lib/a2a3/onboard/tensormap_and_ringbuffer/libaicpu_kernel.so"
  "$WS/pypto/runtime/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so"
  "$WS/pypto/runtime/build/cp311-cp311-linux_x86_64/python/bindings/_task_interface.cpython-311-x86_64-linux-gnu.so"
)
for file in "${runtime_files[@]}"; do
  ls -l --time-style=full-iso "$file" 2>&1 || true
  sha256sum "$file" 2>&1 || true
done

echo "=== PYTHON ==="
export LD_LIBRARY_PATH=
export PYTHONPATH=
export CMAKE_PREFIX_PATH=
# shellcheck disable=SC1090
source "$CANN_SETENV"
# Use the explicitly selected virtualenv rather than the long-lived shell's
# Python or the workspace default.  This is intentionally an absolute
# interpreter invocation: it makes the manifest prove which interpreter and
# site-packages supplied the imported extension modules.
export PTO_ISA_ROOT="$WS/pto-isa"
export PYTHONPATH="$WS/pypto/python:$WS/pypto/runtime/python:$WS/pypto-lib${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - <<'PY'
import importlib
import os
import sys

import numpy
import pypto
import safetensors
import simpler
import simpler_setup
import torch
import _task_interface

print("python", sys.executable)
print("version", sys.version)
print("prefix", sys.prefix)
print("base_prefix", sys.base_prefix)
print("torch", torch.__version__, torch.__file__)
print("numpy", numpy.__version__, numpy.__file__)
print("safetensors", safetensors.__version__, safetensors.__file__)
for name, module in (
    ("_task_interface", _task_interface),
    ("pypto", pypto),
    ("simpler", simpler),
    ("simpler_setup", simpler_setup),
):
    print(name, getattr(module, "__file__", "<no __file__>"))
print("sys.path", *sys.path, sep="\n  ")

for name in ("_task_interface", "pypto", "simpler", "simpler_setup"):
    module = importlib.import_module(name)
    print("module_spec", name, module.__spec__)
PY

echo "=== DYNAMIC LOADER RESOLUTION ==="
task_so="$WS/pypto/runtime/build/cp311-cp311-linux_x86_64/python/bindings/_task_interface.cpython-311-x86_64-linux-gnu.so"
host_so="$WS/pypto/runtime/build/lib/a2a3/onboard/tensormap_and_ringbuffer/libhost_runtime.so"
dispatch_so="$WS/pypto/runtime/build/lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so"
for so in "$task_so" "$host_so" "$dispatch_so"; do
  echo "--- $so"
  ldd "$so" 2>&1 | grep -E 'cann|ascend|hccl|simpler|not found' || true
done

echo "=== PTOAS ==="
export LD_LIBRARY_PATH="$WS/ptoas-bin/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$WS/ptoas-bin/bin/ptoas" --version 2>&1 | head -3
sha256sum "$WS/ptoas-bin/bin/ptoas"

echo "=== RELEVANT ENVIRONMENT ==="
env | grep -E '^(PTO2_|SIMPLER_|ASCEND_|PTO_ISA|P_FAITHFUL|PYTHONPATH|LD_LIBRARY_PATH)' | sort

echo "=== PATH PURITY ==="
printf 'PATH=%s\n' "$PATH"
printf 'CMAKE_PREFIX_PATH=%s\n' "${CMAKE_PREFIX_PATH:-}"
printf 'ASCEND_HOME_PATH=%s\n' "${ASCEND_HOME_PATH:-}"

# The selected CANN may itself be the explicitly requested non-GA tree. Reject
# a second CANN installation, rather than rejecting a path merely because its
# directory name contains "cann900-nonga".
selected_cann=$(readlink -f "$CANN_ROOT")
for value in "$PATH" "${LD_LIBRARY_PATH:-}" "${PYTHONPATH:-}" "${CMAKE_PREFIX_PATH:-}"; do
  sanitized=${value//"$selected_cann"/}
  case "$sanitized" in
    *cann900-nonga*|*ascend-toolkit/latest*|*/cann-9.0.0-beta.1*)
      echo "ERROR: unexpected secondary CANN path in clean manifest: $value" >&2
      echo "ERROR: selected CANN is: $selected_cann" >&2
      exit 2
      ;;
  esac
done

echo "=== DEVICE ==="
npu-smi info 2>&1 | head -45 || true
