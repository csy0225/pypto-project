# N=1 Whole-Net Canonical Test（唯一准出标准）

> **HARD RULE**：N=1 step3p5 whole-net 的“精度正确 / 无 stall / 可发布”结论，
> 只能由真实权重、真实 token 的多步测试给出。首 token `argmax==303` 只用于
> smoke/liveness，不能替代多步精度 gate。

## 1. 当前唯一被测对象

```text
program = models.step3p5.decode_fwd:whole_decode_step3p5
branch  = stepfun/develop
devices = 8,9,10,11,12,13,14,15
dispatch = fixed-slot pull
combine = pull
weights = native W8A8 IPC
KV = resident IPC + in-place InOut
```

当前 holder、sidecar、harness 和 CI 均直接使用上述 canonical symbol。
历史 unroll Main、rollback selector、自定义 Main module/name 参数和旧
compatibility alias 已删除，不得重新引入第二个 Main 产品入口。

Canonical Main 是一个 N=1 `@pl.program`，输出 45 层后的
pre-final-norm BF16 `next_hidden`。final RMSNorm、LM head、sampling、
accept/reject 由 vLLM 或 standalone host 负责。

## 2. 精度和 liveness gate

### 2.1 vanilla 原始精度

使用 live vanilla vLLM W8A8 oracle 做 teacher-forced 多步 decode，seed=6127，
要求：

```text
N >= 128
ALIGNED >= 95%
```

单 token、随机输入、compile-only、`RUN_CLEAN`、P1/P20、截断 MoE 和 BF16
fallback 都不能作为 precision PASS。

当前已知的 2026-07-26 N=256 结果：

```text
vanilla raw alignment = 240/256 = 93.75%
结论 = raw precision gate 未通过
```

该结果必须和 replacement equivalence 分开报告，不能把 replacement PASS
改写成 vanilla precision PASS。

### 2.2 替换等价性

canonical-only 清理前后必须逐 token、逐 hidden 对比。已有 N=256 证据：

```text
token exact = 256/256
hidden exact = 256/256
max_abs_diff = 0.0
TP spread = 0.0
```

该 gate 只证明代码清理没有改变 canonical 数学实现，不等价于完整
Main+MTP serving 已无条件平替。

### 2.3 首 token smoke/liveness

固定首 token `6127`，检查 canonical 输出的首 token `303`，并同时确认：

```text
process rc = 0
hidden finite
TP spread = 0
无 507018 / running-stalled / timeout
无残余 exporter process
```

首 token smoke 不能替代 2.1 的多步 precision gate。

## 3. 0162 环境和命令

```bash
source /usr/local/Ascend/cann/set_env.sh
source /data/chensiyu/pypto/workspace/activate.sh
export PTO_ISA_ROOT=/data/chensiyu/pypto/workspace/pto-isa
export PTO2_RING_HEAP=4294967296
export PTO2_RING_TASK_WINDOW=131072
export PTO2_RING_DEP_POOL=131072

cd /data/chensiyu/pypto/workspace/pypto-lib
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
OUT=/data/chensiyu/pypto/workspace/logs_n1_0162/main-canonical

python -m tests.step3p5.harnesses._stage_main_hidden_only \
  --device 8,9,10,11,12,13,14,15 \
  --out "$OUT" \
  --ckpt "$CKPT" \
  --steps 8
```

需要验证多步 KV metadata：

```text
seq_lens
positions
block_table
slot_mapping
resident K/V history
```

B3 额外使用 `--kv-probe` 检查 45 层、K/V、slot 0/1/2；相邻 invocation
必须证明历史 row 不被误写，padding row 必须来自 allocator-owned reserve。

## 4. MTP hidden-only 门

MTP 使用：

```text
models.step3p5.mtp_hidden_fwd:MTP_LAYER_HIDDEN_PROGRAMS
```

MTP 只输出 raw hidden，不输出 token、logits 或 acceptance state。MTP 的
三套 signal backing 是独立、非 layer-stacked、非跨 epoch 复用的 compact
allocation，保持 `[TP_WORLD_SIZE,1] INT32` 和 `tp_size * 4`；不能套用
canonical stacked/reused control signal 的 512B stride。

## 5. 512B control-signal 口径

DeepSeek v4 中的 512B 主要用于 data tile、L2 cache line 和 MTE 性能对齐，
不是通用 control-signal/window ABI。

step3p5 只对 canonical 中同时满足以下条件的 control signal slot 做
512B 物理隔离：

1. 被 `notify` / `wait` / `AtomicAdd` 使用；
2. 在同一个 backing buffer 中按 layer/slot 堆叠，或跨 `moe_epoch` 复用。

这类 slot 使用：

```text
COMM_CONTROL_SIGNAL_BYTES = 512
COMM_SIGNAL_STRIDE_I32 = 128
formal/window/slice shape = [128,1] INT32
```

逻辑通信 loop 仍只访问前 `n_ranks` 行。普通 data window、独立 signal、
MTP signal 不得为了“512B 对齐”机械扩容。

## 6. 镜像内 canonical-only 审计

发布结论必须来自目标 immutable image 内，不能用裸机 editable checkout
代替。至少检查：

```bash
set -e
test -e /workspace/pypto-lib/models/step3p5/decode_fwd.py
test ! -e /workspace/pypto-lib/models/step3p5/decode_layer_single_chip_hidden.py
test ! -e /workspace/pypto-lib/models/step3p5_opt
grep -q "whole_decode_step3p5" \
  /workspace/pypto-lib/models/step3p5/decode_fwd.py
! grep -RqsE \
  "whole_decode_opt|WholeDecodeOpt|baseline-main|baseline_main" \
  /workspace/pypto-lib/models/step3p5 \
  /workspace/pypto-lib/tools/step3p5 \
  /workspace/pypto-lib/tests/step3p5
```

此外必须执行：

```bash
pytest -q \
  tests/step3p5/unit/test_main_hidden_only_contract.py \
  tests/step3p5/unit/test_mtp_hidden_only_contract.py \
  tests/step3p5/unit/test_performance_bc_contract.py
```

## 7. 结论边界

在 B3/C1/G1 的镜像 compile、设备 liveness、KV row-diff 和多步 precision
证据全部完成前：

```text
B3 = IN PROGRESS
C1 = IN PROGRESS
C3 = IN PROGRESS
G1 = IN PROGRESS
overall release = NO-GO
```

C3 只有在合法 orchestration/SPMD fan-out、explicit join、non-aliasing
task ABI 和设备 swimlane 证据齐全后才能改为 DONE；禁止把 `pl.parallel`
放入 `FunctionType.InCore` 作为伪完成。
