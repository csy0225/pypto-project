# Step3p5 Whole-Net Canonical Test（唯一准出标准）

> **HARD RULE**：精度正确、无 stall、性能和可发布是不同 gate。真实权重、真实 token
> 的多步测试不能被首 token smoke、compile-only、源码挂载或旧镜像结果替代。
> 当前状态先读 [`../STATUS.md`](../STATUS.md)。

## 1. 当前唯一产品对象

```text
program = models.step3p5.decode_fwd:whole_decode_step3p5
branch  = stepfun/develop
weights = native W8A8 IPC
KV      = resident per-layer KV + runtime metadata
```

历史 unroll Main、rollback selector、自定义 Main module/name 参数、
`models/step3p5_opt` 和 compatibility alias 均不得恢复。

当前源码 tip：

```text
pypto-lib 491267c45875e9b1e0071eed224e2e73526799e2
pypto     8e92b46808f9f7c09b6431ad4691503f09c12ee5
```

历史 `c9af5790` image manifest `sha256:3eb694e…` 已通过其源码层级的 BS1×64K
ITL/DFX partial gate；当前 `491267c4` 尚无 immutable image。
最后一个完成全量 production matrix 的回退基线仍是 Wave5，镜像内源码为
pypto-lib `7099476b` / pypto `defa97c5`。历史 R1/R2 已 supersede。
**源码 tip、latest-source partial gate 与 full-matrix release 必须分开报告。**

## 2. 精度 gate

### 2.1 Vanilla raw alignment

使用同一 live vanilla W8A8 oracle 做 teacher-forced 多步 decode：

```text
N >= 128
ALIGNED >= 95%
all hidden finite
TP spread = 0
```

Wave5 在固定 oracle 上的三轮结果均为：

```text
123/128 = 96.09375%
miss = [2,8,13,22,82]
TP spread = 0
```

单 token、随机输入、compile-only、截断层数和 BF16 fallback 都不能作为 precision PASS。

### 2.2 Revision/replacement equivalence

若验证代码清理、入口替换或镜像重建，必须在相同输入上逐 token、逐 hidden 对比
明确的 baseline，并报告：

```text
token exact count
hidden exact count / max_abs_diff
finite
TP spread
```

该 gate 只证明两版本等价，不自动证明 vanilla raw alignment 或完整 serving 平替。

### 2.3 MTP

MTP 必须使用与 Main 配对的同代输入/oracle，单独报告 token、hidden、finite 和 TP
spread。缺少配对 oracle 时只能标 `SKIPPED_MISSING_ORACLE`，不能借用旧 N1 artifact。

## 3. Liveness gate

每轮同时确认：

```text
process rc = 0
无 507018 / running-stalled / timeout
无残余 exporter/chip/build 进程
hidden finite
TP spread = 0
```

首 token `6127 -> 303` 只用于 smoke/liveness，不能替代 §2 的多步 gate。

## 4. Immutable image gate

发布结论必须来自目标 manifest digest，且只挂 driver(ro)、checkpoint(ro)、output(rw)：

1. 核对 manifest/config digest 和五仓 exact pin。
2. `IMAGE_GIT_CREDENTIAL_AUDIT=PASS`。
3. `IMAGE_WORKTREE_CLEAN_AUDIT=PASS`。
4. 默认 Main/retired symbol audit PASS。
5. smoke、Main/MTP compile、数值和 TP spread PASS。
6. 不允许宿主源码挂载、runtime overlay 或借用其它 digest 的结果。

最小 symbol audit：

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

## 5. Performance/DFX gate

性能必须固定 workload、profile、digest、warmup 和统计轮数。至少报告：

```text
batch / context / active rows
attention profile and cleared overrides
min / mean / p50 / p99 / max
correctness / TP spread
specific rank swimlane path
manifest/config digest
```

prepared-worker swimlane 必须先生成依赖图，再在同 worker 上进行 timing-only dispatch；
当前 pypto `8e92b468` 使用 `l2_swimlane_reuse_dep_gen`。没有最终
`l2_swimlane_records.json` 就不能宣称 DFX gate 完成。

## 6. 2026-08-08 当前准出状态

```text
current source 491267c4  = NO IMMUTABLE IMAGE YET
historical c9af image    = BS1×64K ATTENTION/ITL/DFX PARTIAL PASS ON 0162
Wave5                    = FULL PRODUCTION MATRIX PASS ON 0162
R1                       = REVOKED
R2                       = NEVER PUBLISHED / SUPERSEDED
```

当前 digest、ITL、swimlane 路径和剩余 full-matrix gate 见
[`../benchmark/2026-08-06-attention-taskmajor-canonical.md`](../benchmark/2026-08-06-attention-taskmajor-canonical.md)
与 [`../planning/handoff.md`](../planning/handoff.md)。
