# Step3p5 Attention/Vec 收尾、Wave3/Wave4 immutable 验证（2026-08-02—03）

## 0. 2026-08-03 最终覆盖：Wave4 immutable candidate

本文原始 §1--§10 记录 `pypto-lib@76d96bdb` 的 2026-08-02 clean canonical
candidate，作为历史基线保留。当前源码和最新 immutable 候选已前进到：

```text
pypto-lib stepfun/develop
  d7e1381be0236d6e068cd4d86aa815ea693ea5c7

pypto stepfun/develop
  defa97c526fec7e8f032dbbfcc39c820add02bf7
```

`d58b6be7` 在 canonical `tp_all_reduce` 的 final local copy 后增加第三波
completion barrier，确保所有 rank 完成通信 window 的最终本地读取后，任何 rank
才可返回并复用 window；`d7e1381b` 让 two-layer harness 使用同一三波协议，并通过
AST contract 防止再次漂移。focused contracts 为 `28 passed`。

镜像演进必须分开理解：

| 阶段 | pypto-lib | 结果 | 定位 |
|---|---|---|---|
| 历史 clean canonical | `76d96bdb` | N=128 三轮均 `121/128`；64K p50 `50.563 ms` | 历史失败基线 |
| Wave3 | `d58b6be7` | immutable `124/128`，miss `[2,8,13,82]`，TP spread 全零 | canonical 三波修复；harness 尚未对齐 |
| **Wave4** | **`d7e1381b`** | Run1 `122/128`、step2 spread=`2.0`；Run2 `123/128`、spread 全零 | 最新 immutable candidate |

Wave4 镜像：

```text
hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260802-attn-final-wave4

manifest:
  sha256:8125c678779c332d196b3d770242659d9a86185e0a8d96d89681647b00c864ab

config/image ID:
  sha256:c340001f791bd4666310b2f1755daba5492fec8c65f126888d46ed4366131c92
```

Wave4 通过 config/worktree/credential/canonical-only/CANN runtime/optimization
symbol/PTOAS ldd audit、smoke、compile-only、64K ITL 与 DFX。验证只挂载
driver(ro)、checkpoint(ro)、output(rw)，无宿主源码；只使用 cards `0--7`，未操作
cards `8--15` 或 PID `2045390--2045397`。

64K、bs=1、512 blocks、warmup=3、20 measured iterations：

```text
min  = 48.316 ms
mean = 50.529 ms
p50  = 50.204 ms
p99  = 56.355 ms
max  = 56.355 ms
```

其他已测点：active-batch=1/context=1 p50 `43.273 ms`；
active-batch=16/context=1 p50 `112.773 ms`。

DFX 的 LOW-WAIT reference 仍为 rank2：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_wave4_20260802_immutable/itl64k/build_output/
  WholeDecodeStep3p5_20260803_000049/dfx_outputs/rank2/d0/
    critical_path_report.md
    merged_swimlane_20260803_000143.json
```

rank2 makespan `38.504 ms`，`tp_all_reduce` critical-path compute `2.125 ms`。
rank0/1/3--7 的超长 all-reduce 主要吸收 kernel 内自旋等待，不能当真实计算量；
全 rank 摘要在 `itl64k/low_wait_reference.json`。

同一固定 oracle（sha256
`c9b2c72121880e9c605ae70d1cf85c0d4fc8815b180598bc76f7e293551dd947`）的
Wave4 两轮 raw token gate 都达到 `>=95%`：

| run | aligned | miss | TP spread |
|---|---:|---|---|
| 1 | `122/128=95.3125%` | `[2,8,13,22,82,93]` | step2=`2.0` |
| 2 | `123/128=96.09375%` | `[2,8,13,22,82]` | 全零 |

两轮 hidden 均 finite。**因此 raw token gate 已通过，但正式 release gate 仍未关闭**：
Run1 违反 TP-spread 合同，当前证据尚不足以证明跨运行稳定。准确状态为：

```text
Wave4 immutable candidate
raw token gate PASS
formal release gate pending TP-spread stability
```

不要通过无预先限定的重复运行挑选“最好一轮”，也不要把 Wave4 写成正式 release。
后续只应做预先定义的稳定性/根因实验；如果需要新镜像，必须再次完成 immutable
audit、N=128、ITL 与 DFX gate。

## 1. 结论摘要（历史 `76d96bdb` clean candidate）

本轮完成了 attention 阶段源码、0162 真机验证、文档 double-check、镜像构建与
immutable 性能/DFX 采集：

1. `pypto-lib` 与 `pypto` 源码已合入各自 `stepfun/develop`；
2. clean canonical candidate 的 image audit、smoke、64K ITL、DFX 均通过；
3. 64K hidden-only ITL p50 为 **50.563 ms**；
4. 但 fresh-oracle N=128 三轮均为 **121/128=94.53125%**，低于 `>=95%`
   raw gate，因此镜像尚不能正式发布。

准确状态：

```text
source integration        PASS
image audit/smoke         PASS
immutable 64K ITL/DFX     PASS
N=128 raw precision gate  FAIL
release                   BLOCKED
```

## 2. 源码 pin 与实现范围

```text
pypto
  branch: stepfun/develop
  commit: defa97c526fec7e8f032dbbfcc39c820add02bf7

pypto-lib
  branch: stepfun/develop
  commit: 76d96bdbeac280f12ecf626b1bbd722b9278719e
```

`defa97c5` 修复 workload-derived 动态 SPMD launch bound 在 orchestration codegen
中的变量重命名/声明问题。`76d96bdb` 包含本轮 attention/Vec 产品实现与回归 harness。

### 2.1 task 粒度与跨架构原则

当前实现不是 fixed-24-core SPMD：

```text
logical_tasks(row, stage) = ceil(actual_work(row) / grain(stage))
total_tasks(stage)        = sum(logical_tasks(active rows))
```

runtime 再将 logical tasks 映射到目标架构的物理 AIC/AIV 与一个或多个 wave。
`5--10 us/task` 只是 calibration 搜索起点，不是硬目标。最终 profile 联合比较：

```text
task duration + stage span + wave/core-wait + packing + tail
+ dispatch + reduction/finalize dependency tail + batch16
```

A2A3 当前 profile：

```text
Full QK blocks/task                    = 22
Full block-softmax blocks/task         = 12
Full SV+segment recurrence blocks/task = 16
Full online reduce fan-in              = 8
Full/SWA out-proj matmul N              = 64
Full/SWA out-proj tiles/task            = 3
Full/SWA vector N                       = 128
Full/SWA out-proj cast fusion           = 1
TP all-reduce transfer chunk            = 512
```

### 2.2 Full/SWA kernel graph

Full 当前图：

```text
full_qk_matmul
  -> full_softmax
  -> full_sv_matmul                  # SV + segment-local recurrence
  -> full_online_softmax_reduce
  -> full_online_softmax_finalize
  -> full_out_proj_matmul_{aic,aiv}  # cast fused
```

默认 decode graph 当前不存在独立的：

```text
full_online_softmax_pass_a
full_online_softmax_pass_b
full_online_softmax_pass_c
full_out_proj_cast
swa_out_proj_cast
```

源码仍保留 `FUSE_CAST=0` 的 decode fallback 分支；prefill 路径的
`prefill_full_out_proj_cast` / `prefill_swa_out_proj_cast` 也仍然存在。

`reduce/finalize` 是跨 task 的 RAW/liveness 边界：机械并入所有 SV task 会产生
concurrent-writer race，或退化为单 task 串行整行。SWA window 最多 4 个 KV blocks，
保持 row-oriented 高密度 task，不复制 Full 的层次归约。

### 2.3 batch16

`BATCH=16` 只是 storage capacity，不是永久 logical batch。已验证：

- active-batch=16、ctx=1：active hidden 全 finite/nonzero，TP spread=0；
- 16-row 异构 context：task 数按每行真实 workload 的 `ceil` 求和；
- uniform batch16/64K online grain 单轮：grain16 `5.5590 ms`、grain24
  `5.5494 ms`、grain32 `5.6126 ms`。

16 与 24 只差约 0.17%，不足以把 batch-aware 分支硬编码进模型语义。

### 2.4 all-reduce 与 Vec 融合决策

保留：

- reduce-scatter + push all-gather TP all-reduce；
- dense RMSNorm direct BF16 reread；
- dense down-proj cast fusion；
- Full/SWA out-proj cast fusion。

不合入：

- producer 直接写 AR window；
- AR final copy + residual；
- residual + RMS stats；
- RMSNorm + projection；
- gate/up + SiLU cast。

原因是无稳定收益、资源粒度不匹配或破坏既有 split/ownership。通信 chunk
`512 columns` 与 residual Vec grain `128/256 columns` 必须独立校准。

## 3. PyPTO 架构与最小改动

- Orchestration 负责 runtime scalar、logical task DAG 与 dependency；
- InCore task 只执行自己的 tile/segment，不递归 submit；
- runtime 负责 logical task 到物理核心/wave 的映射；
- task grain 属于 architecture profile，不进入模型数学语义；
- 未新增 app-side persistent worker/work-stealing queue。

因此用户提出的“按任务数量切分、每核拉取任务”的核心效果已经由 logical task
scheduler 实现。真正的 device-side persistent queue 需要修改 runtime/scheduler ABI，
本轮没有必要扩大改动面。

## 4. 镜像演进

### 4.1 v1：compile failed

```text
stepfun-develop-20260802-attn-final
```

包含 attention 源码，但 pypto 仍为 `1f704616`；动态 SPMD launch-bound 变量在
immutable orchestration codegen 中未声明，构建/验证失败。

### 4.2 v2：代码正确但 image config 不 canonical

```text
stepfun-develop-20260802-attn-final-v2
```

加入 `pypto@defa97c5` 后代码可执行，历史运行曾得到 `123/128`；但 image config
仍携带旧 CANN 8.5.1 字符串，不能作为 clean canonical release 或精度证据。

### 4.3 clean canonical candidate

```text
tag:
  hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260802-attn-final-canonical

manifest:
  sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d

config/image ID:
  sha256:c7f612a2562e932908d2a0d9ffadd1a1bd155c70bff0e82c24be32ef6b9f79ea
```

完整 pins：

```text
pypto      defa97c526fec7e8f032dbbfcc39c820add02bf7
pypto-lib  76d96bdbeac280f12ecf626b1bbd722b9278719e
pto-isa    ecb6c303f797749f811a494742c3c08156aacabb
PTOAS      fc8c6caee561914b4fb991dfc8427bb63194269e
simpler    e2efebcbd190302609c0775d2984f409f5f42c76
ptoas-bin  v0.50
vLLM       1b3e538c35999e62b6d24e0651b3a85b7d16c826
```

Dockerfile 显式重置 `PATH/PYTHONPATH/CMAKE_PREFIX_PATH`，避免 base image 中
CANN 8.5.1 的 inherited config 污染。

## 5. immutable 验证约束与 audit

容器只挂载：

```text
host driver  -> container driver  (read-only)
checkpoint   -> same path         (read-only)
output       -> result directory  (read-write)
```

没有挂载宿主源码。0162 只使用 cards `0--7`；cards `8--15` 上 PID
`2045390--2045397` 始终存活且未被操作。测试完成后 cards `0--7` 无残留进程。

结果：

```text
IMAGE_CONFIG_CANN_851_AUDIT=PASS
IMAGE_WORKTREE_CLEAN_AUDIT=PASS
IMAGE_GIT_CREDENTIAL_AUDIT=PASS
CANONICAL_ONLY_AUDIT=PASS
CANN_851_RUNTIME_AUDIT=PASS
EXPECTED_OPTIMIZATION_SYMBOL_AUDIT=PASS
PTOAS_LDD_AUDIT=PASS
[smoke] PASS
```

证据：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_canonical_20260802/
    image_config_audit.log
    audit_smoke_final.log
```

## 6. 64K ITL

条件：

```text
active_batch = 1
context      = 65536
num_blocks   = 512
block_size   = 128
warmup       = 3
iters        = 20
```

结果：

| metric | value |
|---|---:|
| min | 49.213 ms |
| mean | 50.568 ms |
| p50 | **50.563 ms** |
| p99 | 52.537 ms |
| max | 52.537 ms |

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_canonical_20260802/itl64k/itl_report.json
```

此前 source-tree 的 `68.058 ms` 只保留为历史开发数据；最终发布判断只认上述
immutable canonical image 结果。

## 7. DFX 与 LOW-WAIT reference 更正

各 rank：

| rank | makespan | TP all-reduce critical-path compute |
|---:|---:|---:|
| 0 | 58.930 ms | 21.949 ms |
| 1 | 127.720 ms | 90.748 ms |
| 2 | **38.924 ms** | **2.049 ms** |
| 3 | 60.665 ms | 23.776 ms |
| 4 | 370.826 ms | 333.843 ms |
| 5 | 381.410 ms | 344.553 ms |
| 6 | 373.030 ms | 336.079 ms |
| 7 | 373.719 ms | 336.794 ms |

rank4--7 的巨大 makespan 主要是 `tp_all_reduce` 内部自旋等待被计入 compute。
所以本轮正确 LOW-WAIT reference 是 **rank2**；rank5 可观察完整 collective wait
span，但不能称为 LOW-WAIT。

精确路径：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_canonical_20260802/itl64k/build_output/
  WholeDecodeStep3p5_20260802_162729/dfx_outputs/rank2/d0/
    critical_path_report.md
    merged_swimlane_20260802_162823.json
```

## 8. N=128 raw precision blocker

fresh oracle：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  fresh_vanilla_oracle_20260731/oracle_ids.json

sha256:
  c9b2c72121880e9c605ae70d1cf85c0d4fc8815b180598bc76f7e293551dd947
```

三轮结果：

| run | aligned | miss | TP spread |
|---|---:|---|---|
| 1 | 121/128 | `[2,8,13,15,22,82,93]` | none |
| 2 | 121/128 | `[2,8,15,22,29,82,93]` | step39=`0.953125` |
| 3 | 121/128 | `[2,8,13,14,22,82,93]` | step68=`1.1875`、step70=`3.25` |

所有 hidden finite。

报告：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_canonical_20260802/
    n128_teacher_forced/main_hidden_only_report.json
    n128_teacher_forced_rerun2/main_hidden_only_report.json
    n128_teacher_forced_rerun3/main_hidden_only_report.json
```

miss 集合和 TP spread 跨轮变化，现象更像 collective/runtime 非确定性，但尚无
根因闭环。当前规则：

- 不修改 fresh oracle；
- 不借用 v2 的 `123/128`；
- 不无限重跑直到偶然通过；
- 不在未定位根因时修改已经推送的产品 commit。

## 9. pto-isa 路径说明

```text
/workspace/pto-isa
```

是 Dockerfile 按 `ecb6c303...` 显式 checkout 的外部源码 pin。

```text
/workspace/pypto/runtime/build/pto-isa
```

是 `python -m simpler_setup.build_runtimes --platforms a2a3` 在 runtime build
目录生成/克隆的构建树。二者职责不同；release manifest 记录前者，后者用于
runtime 构建，不应被误认为第二个产品 pin。

## 10. 发布判定

Attention 阶段的实现与性能工作可以收尾，但镜像发布不能收尾：

```text
pypto / pypto-lib source merged      YES
clean canonical image built/pushed   YES
audit/smoke/ITL/DFX                  PASS
N=128 raw gate                       FAIL
formal release                       NO
```

正式发布前必须闭环 raw precision/TP spread 根因，并用新的 clean immutable image
重新完成全套 gate。
