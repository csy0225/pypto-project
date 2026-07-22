#!/usr/bin/env bash
# pypto 运行时环境激活 —— 每个新 shell `source` 一次。
#
# 用法:
#   WS=/your/workspace source activate-pypto.sh    # 指定 workspace
#   source activate-pypto.sh                        # 用默认 WS
#
# 做三件套 + canonical 运行时参数。故意用 `source`（不是 bash 执行），
# 因为要把 env 注入当前 shell。不 set -e/-u（避免 source CANN set_env.sh
# 时因未定义变量杀掉用户 shell）。

: "${WS:=/data/chensiyu/hw_project/pypto/workspace}"
export WS

# 1) CANN（activate.sh 不做这步；少了会报 ASCEND_HOME_PATH not set）
_cann_env=/usr/local/Ascend/cann/set_env.sh
if [ -f "$_cann_env" ]; then
  # shellcheck disable=SC1090
  source "$_cann_env"
else
  echo "[activate-pypto][warn] $_cann_env 不存在——CANN 未激活（多卡会挂）" >&2
fi

# 2) workspace venv + PTOAS/PATH/LD_LIBRARY_PATH
if [ -f "$WS/activate.sh" ]; then
  # shellcheck disable=SC1090
  source "$WS/activate.sh"
else
  echo "[activate-pypto][warn] $WS/activate.sh 不存在" >&2
fi
# python not in PATH 兜底（machine-recovery 记录的坑）
if [ -f "$WS/.venv311/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$WS/.venv311/bin/activate"
fi

# 3) pto-isa + canonical 运行时参数（对齐 N1-STABLE-ENV §6.1）
export PTO_ISA_ROOT="$WS/pto-isa"
export PYTHONPATH="$WS/pypto/python:$WS/pypto-lib:${PYTHONPATH:-}"
export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072
export P_FAITHFUL_MOE_LAYERS=42

echo "[activate-pypto] WS=$WS"
echo "  python           = $(command -v python || echo MISSING)"
echo "  ASCEND_HOME_PATH = ${ASCEND_HOME_PATH:-UNSET}"
echo "  PTO_ISA_ROOT     = $PTO_ISA_ROOT"
echo "  ptoas            = $(command -v ptoas || echo 'not-on-PATH（先确认 ptoas-bin 已解到 $WS/ptoas-bin）')"
