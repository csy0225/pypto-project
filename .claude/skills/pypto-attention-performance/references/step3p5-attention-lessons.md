# Step3p5 Attention 性能优化案例与可迁移经验

本文件是 `pypto-attention-performance` 的按需 reference。它记录 Step3p5 在
0162/A2A3 上的实测案例，用于说明方法如何落地；参数和性能数字只适用于明确的
源码、设备与 workload，不是跨架构默认值。

## 目录

1. [适用范围与当前结论](#1-适用范围与当前结论)
2. [最终 workload 与任务模型](#2-最终-workload-与任务模型)
3. [Full 与 SWA 的最终任务图](#3-full-与-swa-的最终任务图)
4. [task-grain 校准经验](#4-task-grain-校准经验)
5. [online softmax 的融合边界](#5-online-softmax-的融合边界)
6. [out-proj cast 与通算融合经验](#6-out-proj-cast-与通算融合经验)
7. [TP all-reduce 性能与稳定性经验](#7-tp-all-reduce-性能与稳定性经验)
8. [DFX 解释经验](#8-dfx-解释经验)
9. [验证与收尾证据](#9-验证与收尾证据)
10. [负面结果清单](#10-负面结果清单)

## 1. 适用范围与当前结论

权威对象：

```text
pypto-lib perf/attn-rope-taskmajor-lifetime-20260805
  1ea76e0f2d3e6c132198dc6214034968daeaf2f2
  base stepfun/develop@56b3d477953ab1e2df87213aef3a536c64051dcc

pypto stepfun/develop
  defa97c526fec7e8f032dbbfcc39c820add02bf7

machine/profile
  0162 / Ascend A2A3 / PYPTO_STEP3P5_ATTN_TASK_PROFILE=a2a3
```

当前可以下的结论：

> 针对当前 Step3p5、0162/A2A3 profile，attention/Vec 优化本体以及 TP
> all-reduce 稳定性已经收尾；当前证据不支持新的 canonical attention 候选。

这不表示其它架构或未来 backend 没有空间。换架构后必须重新做 task-grain、
tile、resource mapping、batch 和 immutable gate。

## 2. 最终 workload 与任务模型

最终实现不固定 24 个物理核心：

```text
logical_tasks(row, stage)
  = ceil(actual_work(row) / architecture_profile_grain(stage))

total_tasks(stage)
  = sum(logical_tasks(active rows))
```

runtime 再将 logical tasks 映射到 AIC/AIV wave。必须区分：

```text
storage block/tile
logical task grain
physical AIC/AIV mapping
active workload
static capacity
```

A2A3 资源按：

```text
24 AIC
48 AIV
```

分别分析，不能合并成 72 个同质核心。

当前 shape 重点：

| 参数 | Full | SWA |
|---|---:|---:|
| TP-local Q heads | 8 | 12 |
| Q physical pad | 16 | 24/32 |
| head dim | 128 | 128 |
| KV storage block | 128 tokens | 128 tokens |
| 最大有效 KV blocks | context 决定；64K 为 512 | window 512：aligned=4，unaligned≤5 |
| storage batch capacity | 默认 16；编译时可配置，已验证 32 | 默认 16；编译时可配置，已验证 32 |

默认 `BATCH=16` 是 capacity；实际 logical tasks 由 active rows 和每行真实
`seq_len` 推导。batch32 验证使用 compile-time capacity=32，不是在 capacity=16
的二进制上越界运行。

profile 选择是显式的：

```text
portable (默认): QK/softmax/online=22/12/16，reduce fan-in=8，uniform O(1)=0
a2a3:            QK/softmax/online=22/16/22，reduce fan-in=8，四项 uniform O(1)=1
```

`--platform a2a3` 不会隐式选择 attention profile。做 A2A3 release A/B 时必须设置
`PYPTO_STEP3P5_ATTN_TASK_PROFILE=a2a3`，并清除 QK、softmax、online、reduce
fan-in 及四个 uniform-O(1) 单项 override；否则单项环境变量优先级更高，会把
“profile A/B”污染成混合配置。

## 3. Full 与 SWA 的最终任务图

### 3.1 Full

```text
full_rope_q ───────────────┐
                           ├─> full_qk_matmul
full_rope_kv_cache ────────┘
-> full_softmax
-> full_sv_matmul
   # SV + segment-local online recurrence
-> full_online_softmax_reduce
-> full_online_softmax_finalize
-> full_out_proj_matmul_{aic,aiv}
   # FP32 accumulator -> BF16 cast fused
```

默认 decode graph 不再生成：

```text
full_online_softmax_pass_a
full_online_softmax_pass_b
full_online_softmax_pass_c
full_out_proj_cast
```

历史 Pass-A 已并入 `full_sv_matmul`。`reduce` 和 `finalize` 仍是跨 task
partial 的必要 RAW/liveness 边界。

两个 RoPE producer 都按 `spmd(active_rows)` 一次提交，QK 显式依赖两个
TaskId。DFX 应同时检查 logical blocks 随 workload 增长、producer
`invocation_count=1`；不能只看 QK/SV 是否已经 task-major。

### 3.2 SWA

SWA 的 512-token window 在 aligned 边界覆盖 4 个 KV blocks、unaligned
边界最多覆盖 5 个，保持 row-oriented 高密度任务：

```text
swa_rope_q ───────────────┐
                         ├─> swa_qk_matmul
swa_rope_kv_cache ───────┘
-> swa_softmax
-> swa_sv_matmul
-> swa_online_softmax
-> swa_out_proj_matmul_{aic,aiv}
   # cast fused
```

代表性 `swa_online_softmax` 仅约 `2.9–3.2 us`。复制 Full 的层次归约会增加
scratch、dispatch 和依赖，没有收益证据。

## 4. task-grain 校准经验

### 4.1 QK / block-softmax

64K、batch1 的校准历史：

```text
QK/softmax/online grain 16/12/16:
  QK/SV 约 32 个 AIC logical tasks
  QK/SV task body 接近 10 us
  进入 2 waves

QK/softmax/online grain 24/12/24:
  QK/SV 约 22 个 AIC logical tasks
  task body 约 15–17 us
  保持 1 wave
```

`24/12/24` 的 device makespan 曾优于 `16/12/16`，证明“单 task 越接近
5–10 us 越好”不成立。随后 QK/online 的 22 与 24 在重复稳定性中已接近
噪声，block-softmax 又独立比较 12 与 16。显式 A2A3 profile 最终选择
`22/16/22`，不能把旧的三项联动 sweep 当成最终 profile。

### 4.2 SV + segment recurrence

Pass-A 并入 SV 后，目标函数变成整条 online chain，而不是旧 standalone SV：

```text
SV segment work
+ reduce fan-in
+ final row merge
+ normalize/store
```

早期 online grain 扫描中：

| grain | task p50 | tasks/waves | 主要结果 |
|---:|---:|---:|---|
| 6 | 9.16 us | 86 / 2 | 任务更短但 reduction tail 更长 |
| 14 | 约 17 us | 37 / 1 | 与 grain16 接近 |
| 16 | 约 19 us | 32 / 1（当时 Pass-A/AIV 口径） | 五轮 reference median 最优约 0.7 us |

该结果促成 portable fallback 使用 online grain=16。完成 TaskId chain、task-major
uniform O(1) mapping 和 per-request-64K 多 batch 复测后，0162 的显式 A2A3
profile 使用 online grain=22、reduce fan-in=8。这个结论来自整条依赖链，不可用
旧 QK/SV standalone sweep 或单轮最低值机械覆盖。

### 4.3 batch 与异构 context

异构两行 `65536 / 8192` 的 task 数等于逐 row `ceil` 求和，而不是按最大
context 给两行铺满。最终要求的 batch matrix 使用**每个 request 都为 65536**：

| active batch | 每 request context | QK / softmax / online tasks | fresh-process 两层 p50 |
|---:|---:|---:|---:|
| 1 | 65536 | 24 / 32 / 24 | 3.5837 ms |
| 2 | 65536 | 48 / 64 / 48 | 5.0313 ms |
| 4 | 65536 | 96 / 128 / 96 | 3.9684 ms |
| 8 | 65536 | 192 / 256 / 192 | 4.2073 ms |
| 16 | 65536 | 384 / 512 / 384 | 4.7132 ms |
| 7 | 65536 | 168 / 224 / 168 | 4.1188 ms |

六个独立进程都与最终 matrix bitwise exact replacement，finite、TP spread=0、
逐迭代 `unique_count=1`。另有 linear/reverse 20-iteration matrix 覆盖输入交替、
inactive-row、Q publication 与 device KV slot/readback。bs2 的 host p50 会受
collective 到达状态影响，不能据此增加 batch-aware attention 数学分支。

经验：

- capacity 不等于 workload；
- 多 batch 长 context 必须明确“每 request 64K”，不能用“总 context 64K”替代；
- batch16 必须验证；capacity 扩到 32 后还要重新检查 task-count、tile、RoPE
  grid dependency 和内存边界；
- 服务 workload 改变时，应补多轮 profile，再生成独立 architecture/workload profile。

## 5. online softmax 的融合边界

可以融合：

```text
每个 SV logical task 私有的 segment-local (m,l,o) recurrence
```

仍需保留：

```text
write-disjoint group reduction
per-row final merge / normalize / BF16 store
```

机械删除 reduce/finalize 会产生两类问题：

1. 多个并发 SV task 写同一个 row，形成 concurrent-writer race；
2. 退化成一个 task 串行处理整行，丢失 context parallelism。

因此“图上还有两个 kernel”不是继续融合的充分理由。

## 6. out-proj cast 与通算融合经验

### 6.1 保留的融合

Full/SWA decode out-proj：

```text
FP32 matmul accumulator
-> same mixed task AIV cast
-> BF16 partial output
```

当前 A2A3 profile：

```text
matmul N tile      = 64
tiles/task         = 3
vector N           = 128
cast fusion        = 1
```

cast fusion 删除了 standalone GM round-trip，correctness 通过；整网 latency
收益处于中性/噪声级，因此它是数据流清理和当前 profile 选择，不是跨架构定律。

out-proj task 沿输出 N tiles 切分，但 batch 外层仍按静态
`BATCH_TILE=16` capacity 执行；不要把 Full context stages 的
workload-derived mapping 外推成“整个 attention 已无 inactive-row work”。

### 6.2 未合入的融合

以下方向可运行或局部正确，但没有稳定收益：

```text
all-reduce + residual add
residual add + RMS statistics
RMSNorm + projection
gate/up + SiLU
producer direct-write all-reduce window
```

典型原因：

- 通信 chunk 512 直接继承给 residual Vec 后，并行度下降；
- mixed kernel 破坏原 projection split；
- focused span 有数微秒收益，但 whole-net wall 不变；
- 只减少 kernel 数，没有消除关键路径成本。

保留的 Vec 改动：

```text
dense RMSNorm direct BF16 tile reread
dense down-projection cast fusion
Full/SWA out-proj cast fusion
```

共同点是删除已确认的 FP32/BF16 GM materialize，并保持 write-disjoint、
现有 task graph 和资源 split。

## 7. TP all-reduce 性能与稳定性经验

协议演进：

```text
one-phase symmetric mesh
-> reduce-scatter + push all-gather
-> final-read lifetime close
-> source-publication TPUT
```

最终状态机：

```text
local source
-> self-target synchronous/drained TPUT
-> Wave 1 source publication
-> rank-owned reduce-scatter
-> write-disjoint push all-gather
-> Wave 2 result publication
-> final local copy
-> Wave 3 lifetime close
```

必须保持：

- peer 顺序固定 `0..tp_size-1`；
- rank-owned、write-disjoint chunk；
- 单一 FP32 accumulator；
- 最终只做一次 BF16 cast/store；
- notify 覆盖真实 data publication；
- reuse wait 覆盖最后一个 semantic consumer。

两层历史对比中，单次 all-reduce 从约 `81–83 us` 降到约 `35 us`。随后
Wave5 解决间歇性 TP spread。准确结论是：

> 0162 证据支持 source publication/lifetime ordering 是关键边界；self-target
> TPUT 是当前最小稳定性修复，但没有跨所有硬件的唯一根因证明。

性能方法上的经验：

1. 先把 collective 自身的算法、ownership 和 publication 做对；
2. 再评估通算融合；
3. 通信粒度和相邻 Vec task grain 必须解耦。

## 8. DFX 解释经验

`tp_all_reduce` 内 spin wait 会落在 kernel duration 中。因此：

- `LOW-WAIT REFERENCE` 只是 heuristic；
- 不能把长 span 全部算作通信算术；
- 每轮都要重新选择 reference，并检查所有 rank；
- dep generation 与 swimlane 分开采；
- DFX 必须在 warmup 后采；
- host wall-clock 与 device makespan 分开解释。

两层 harness 的价值是降低 whole-net 噪声，但不能替代 canonical：

```text
focused harness -> 参数筛选与机制证明
canonical whole-net -> 数值、跨 rank、端到端和发布判断
```

最终每请求 64K DFX：

| bs | LOW-WAIT rank | makespan | TP all-reduce span-sum | QK / softmax / SV |
|---:|---|---:|---:|---:|
| 1 | rank2 | 687.2 us | 168.44 us | 16.00 / 17.36 / 31.54 us |
| 7 | rank1 | 1143.8 us | 178.96 us | 142.76 / 79.32 / 168.54 us |
| 16 | rank2 | 1642.8 us | 180.06 us | 311.12 / 161.88 / 379.52 us |

```text
/mnt/persist/chensiyu/workspace/attn-opt/out/
rope_taskmajor_final_dfx_perrow64k_linear_bs{1,7,16}_s16_early_20260806/
```

每个 RoPE producer 都是一次 invocation，logical blocks 随 active batch 为
`1/7/16`。producer early-resolve 将 Full producer→QK 的全 rank 中位 gap 从
`3.46/4.74/7.53 us` 降到 `0.93/3.46/5.06 us`，SWA 从
`1.84/4.06/6.11 us` 降到 `0.77/1.99/4.14 us`。其它 rank 的数百毫秒
all-reduce 主要是 peer-arrival spin wait，不能解释为通信算术时间。

## 9. 验证与收尾证据

最终源码验证：

```text
py_compile / git diff --check / differential Ruff = PASS
skill quick validation = PASS
pytest = 254 passed, 3 skipped
```

compile-only 不能只看编译返回码。harness 会读取真实生成的
`orchestration/chip_orch.cpp`，验证 Full/SWA 每个 stage 的动态 launch bound、
launch/scalar SSA 一致性、TaskId publication 和完整依赖链；最终 checker 返回空错误
列表。这样可阻止“Python 图有依赖、lowering 后依赖丢失”的假 PASS。

最终真实设备门禁还包括：

```text
linear/reverse matrix:
  bs=1/2/4/8/16/7, every request context=65536
  alternating/inactive/Q-publication/device-KV audits PASS

fresh-process:
  six batches bitwise exact replacement, bad_ratio=0

SWA direct oracle:
  bs1 / ctx65535 / reverse table / five covered blocks PASS
```

此前 canonical whole-net release gate：

```text
Main N=128:
  3/3 runs = 123/128
  miss = [2,8,13,22,82]
  hidden finite = true
  tp_spread_max = 0.0

Main active-batch=16:
  8/8 exact
  active rank rows = 128
  TP spread = 0

MTP batch1/batch16:
  tokens = [6178,410,303]
  max_abs_diff = 0
  TP spread = 0

all-reduce stress:
  128/128 PASS
  256/256 PASS

ITL:
  batch1/context65536 p50 = 49.796 ms
  batch16/context1 p50 = 112.827 ms
```

收尾判断依据不是“想不到新点”，而是：

1. attention 主要串行轴、task grain、online reduction、out-proj cast 已完成；
2. 多个更激进 fusion 候选已被真实 A/B 否定；
3. batch16、异构 context、MTP、TP consistency 已覆盖；
4. 当前关键路径没有新的、被证据支持的最小 attention 候选。

## 10. 负面结果清单

| 尝试 | 结果 | 可迁移经验 |
|---|---|---|
| 固定 24 lane context split | 长 context 有收益，但短 context 回退 | 找到并行轴后仍要改成 workload-derived tasks |
| QK+softmax+PV per-block fusion | 64K ITL 约 +7% | 减 GM 不一定抵消 boxed tile/Vec 成本 |
| fused softmax 切 8 rows | PTOAS 拒绝 | cube boxed rows 必须满足 fractal 合法粒度 |
| 去 bias 的 fusion 计时 | 仍约 +5.2% | 回退主要是 16-row softmax，非 bias 本身 |
| runtime active tensor 首维 | dynamic-shape 风险高 | “只算 active”和“只分配 active”是两件事 |
| SWA 复制 Full 层次归约 | 无收益依据 | 短 window 优先 task 密度 |
| Full/SWA RoPE packed staging | bs12 独立 40 轮 `unique_count=40` 且明显错误 | 早期速度不能覆盖逐迭代稳定性；最终图隔离 A/B 前保持 NO-GO |
| AR + residual，grain 512 | 明显变慢 | 通信 chunk 不能代替 Vec grain |
| RMSNorm + projection | focused 小幅变快，破坏 split | 局部收益不足以扩大 mixed-kernel 风险 |
| gate/up + SiLU fusion | task 少但 batch16/wall 无稳定收益 | kernel count 不是目标函数 |

详细的统一设计与历史实验现已合并到：

```text
design/performance/04-attention-optimization.md
```
