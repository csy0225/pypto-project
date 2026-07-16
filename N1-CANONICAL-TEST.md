# N=1 Whole-Net Canonical Test（唯一准出标准）

> **HARD RULE**：N=1 step3p5 whole-net 的“精度正确 / 无 stall / 可发布”结论，
> 只能由本文定义的真实权重、真实 token、完整 P42 测试给出。禁止用随机输入、
> `RUN_CLEAN`、P1/P20、中间态、compile-only 或自造 harness 替代。

## 1. 被测对象与固定组合

```text
program = whole_decode_faithful_real
branch = feat/whole-net-n1-fusion
release commit = 0e7a0fddc90c4f2348f1d59e015fb817a0877a02
machine = gpu-a910x-0162
devices = 8,9,10,11,12,13,14,15
P_FAITHFUL_MOE_LAYERS = 42
dispatch = fixed-slot pull
combine = pull
weights = native W8A8 IPC
KV = IPC
token = 6127
golden argmax = 303
```

当前 whole-net 是 **dispatch pull + combine pull**。PUSH 探针或历史 push
版本不是当前准出对象。

## 2. 输入、batch 与精度标准

- `--hidden-token 6127`：读取真实 `embed_tokens.weight[6127]`。
- `ctx=1`、rope position-0 identity。
- `BATCH=16`，但只有 row0 是有效 token；row1..15 是 padding。
- 只拼接并检查 `logits_shard_out[r, 0]`。
- 唯一精度 PASS：

```text
argmax(full_logits) == 303
```

以下均不构成 PASS：

- 只有 `RESULT=REAL_WEIGHT_IPC_RUN_CLEAN`；
- P1/P20 或 `P_FAITHFUL_MOE_LAYERS=0`；
- 随机 hidden；
- BF16 dequant 权重回退；
- 只编译、不执行；
- 非 303 的任意 argmax。

## 3. 0162 标准命令

### 3.1 环境

```bash
source /usr/local/Ascend/cann/set_env.sh
source /data/chensiyu/hw_project/pypto/workspace/activate.sh
export PTO_ISA_ROOT=/data/chensiyu/hw_project/pypto/workspace/pto-isa
export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072
export P_FAITHFUL_MOE_LAYERS=42

cd /data/chensiyu/hw_project/pypto/workspace/pypto-lib
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
OUT=/tmp/n1_weight_ipc
```

注意：截至 2026-07-16，0162 上可用 checkpoint 是上述 `/data/chensiyu/...`
路径；旧 `/mnt/hw910test-jfs/...` 默认路径在本机不可用。

### 3.2 fresh exporter pool

```bash
rm -f "$OUT"/ready.rank* "$OUT"/STOP 2>/dev/null

for r in $(seq 0 7); do
  dev=$((8 + r))
  python -m tests.step3p5._stage_whole_faithful_real_ipc \
    --export-rank "$r" \
    --dev "$dev" \
    --kv-ipc \
    --out "$OUT" \
    --ckpt "$CKPT" \
    > "/data/chensiyu/hw_project/pypto/workspace/logs_n1/exp_${r}.log" 2>&1 &
done

while [ "$(find "$OUT" -maxdepth 1 -name 'ready.rank*' | wc -l)" -lt 8 ]; do
  sleep 15
done
```

### 3.3 canonical P42 worker

```bash
python -m tests.step3p5._stage_whole_faithful_real_ipc \
  --device 8,9,10,11,12,13,14,15 \
  --reuse-exporters \
  --kv-ipc \
  --hidden-token 6127 \
  --ckpt "$CKPT"
```

每次必须同时检查：

```text
process rc = 0
存在 [worker] RUN done
argmax=303
```

稳定性发布 gate 为同一份最终源码、fresh exporter pool 下连续 20 次全部满足。

### 3.4 清理

```bash
touch "$OUT/STOP"
rm -f "$OUT"/ready.rank* "$OUT"/STOP
```

测试结束后确认设备 8–15 无残留运行进程。

## 4. 当前 release 证据（2026-07-16）

20-run 目录：

```text
/data/chensiyu/hw_project/pypto/workspace/logs_n1/signal512/
  signal512_p42_20_20260716_220004
```

结果：

```text
FINAL pass=20/20 rc=0
每次 argmax=303
TOP5=[303, 9592, 768, 1043, 410]
runtime min/mean/max = 2.53 / 2.5685 / 2.62 s
```

20 次数值指纹一致：

```text
max|next_hidden|=264192.0000
row0|next_hidden|=588.0000
max|h_mid|=294.0000
max|logits|=14.0506
```

最终整理后 smoke：

```text
log = .../signal512_final_smoke_20260716_230225
runtime = 2.57s
argmax = 303
FINAL_SMOKE = PASS
```

20-run 和最终 smoke 的 dmesg 时间窗均未新增：

```text
devmm/page fault
illegal VA / illegal instruction
DMA/UB fault
507018
running-stalled
stranded CQE
```

## 5. 发布前静态与生成器 gate

```bash
python -m py_compile \
  models/step3p5/decode_layer.py \
  models/step3p5/moe.py \
  tools/step3p5/_gen_faithful_real.py

git diff --check
```

还必须执行真实 generator round-trip：

1. 从当前 `decode_layer.py` 剥离
   `_build_whole_decode_faithful_real_program` 和 binding；
2. 运行 `tools/step3p5/_gen_faithful_real.py`；
3. 与剥离前文件做字节比较。

当前 release 的要求与结果：

```text
PRECOMMIT_ROUNDTRIP=PASS
ROUNDTRIP_CMP_RC=0
```

这项检查用于防止 generator 把 `inverse_map` 从 dispatch 边界移回 combine，
或恢复 self `remote_load`、旧 count snapshot 等未验证结构。

## 6. 结论边界

可以声明：

> release commit `0e7a0fdd` 在 0162 上完成 native-W8A8、KV-IPC、
> dispatch-pull + combine-pull 的 canonical P42 20/20，全部
> `argmax=303`，无 stall、无精度漂移。

根因表述必须保持严谨：

- 32B → 512B control-signal physical isolation 是最终最小布局 A/B 变量；
- 20/20 相对历史随机 stall 是强因果证据；
- 不得把某个 PUSH/TPUT、某个 signal bit 或某个历史 kernel 位置写成已经由硬件层
  单独证明的唯一根因。
