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

当前 canonical SRC tip：

```text
pypto-lib a745ab659c68afca01de37870e29ccb9648d7c87
pypto     655c7bda7b0a0b495a3387b2570ea68c4a857a40
```

当前 release-admitted r12 仍 bake `e6c7d8ec/14de90fd`；SRC tip 与 release IMG 必须分开。

当前 release-admitted immutable image：

```text
tag      stepfun-upgrade-20260826-r12
manifest sha256:ba42fd19b3af0144a835e95a4a6925ed89ea700624f696b221e93a54e6eb805d
config   sha256:b36f0cec3a8b64e5e17e273c63d69694730bd8b904e69c2806c3d73a5233f08f
```

r12 final contract 为 `1844/1844 PASS`；当前直接回退是 r11 manifest
`sha256:401ead7d…a67b12`。Wave5 只保留为历史完整 production-matrix 对账，
不再是当前直接回退。**SRC、source-overlay 性能证据、IMG immutable gate 与
历史 full-matrix 证据必须分开报告。**

## 2. 精度 gate

### 2.1 Vanilla raw alignment

使用同一 live vanilla W8A8 oracle 做 teacher-forced 多步 decode：

```text
N >= 128
ALIGNED >= 95%
all hidden finite
TP spread = 0
```

当前 r12 在配对 oracle 上，H4 `all/none` 两个模式均为：

```text
126/128 = 98.4375%
miss = [20,69]
TP spread = 0
```

Wave5 的历史三轮结果均为 `123/128`、miss `[2,8,13,22,82]`，只用于旧矩阵
对账，不能替代 r12 当前 gate。

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
当前 r12 的 BS1/BS16 token 均为 `[6178,410,303]`，三层 hidden pass rate
均为 `1.0`、max abs diff `0`。

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

若声明 prepared-worker whole-swimlane，必须先生成依赖图，再在同 worker 上进行
timing-only dispatch；没有最终 chip/swimlane records 就不能宣称 whole-swimlane
完成。当前 r12 只通过 8/8 `deps.json` 的 dep-only DFX，hidden/token exact，
**不声明 whole-swimlane**。

r12 的 whole-step 性能数字来自 r11 immutable digest 上两文件 source-overlay A/B/A，
不是 r12 immutable-image 性能复测；正式合同仍为 serial 8-rank independent submit。
当前 r15 local candidate 的 reset matched A/B/A 为 `21.617/20.516/21.257 ms`，
正式收益 `0.921 ms / 4.296%`；历史 `20.973/20.172 ms` 与其合同不同，不能横比。

## 6. 2026-09-02 当前准出状态

```text
canonical source 655c7bda / a745ab659 = REMOTE stepfun/develop EXACT
r15 local candidate                    = AUDIT/H4/EXTENDED PASS, NOT PUBLISHED
r15 performance                        = RESET FIX PASS, HISTORICAL BEST NOT PROVEN
r12 immutable image                    = RELEASE-ADMITTED, 1844/1844 PASS
r11 immutable image                    = DIRECT ROLLBACK, 20/20 PASS
r12 DFX                                = DEP-ONLY PASS, NOT WHOLE-SWIMLANE
Wave5                                  = HISTORICAL FULL PRODUCTION-MATRIX EVIDENCE
```

当前 digest、ITL、swimlane 路径和剩余 full-matrix gate 见
[`../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md`](../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md)、
[`../benchmark/2026-09-02-k8-historical-performance-reconciliation.md`](../benchmark/2026-09-02-k8-historical-performance-reconciliation.md)
与 [`../planning/handoff.md`](../planning/handoff.md)。
