# Step3p5 SWA RMSNorm → QKV 调度间隙优化（2026-08-12）

> **范围。** 本文只记录 `swa_moe_chip_orch_swa_rmsnorm_zc` 完成到
> `swa_qkv_proj` 开始之间的调度间隙优化。RMSNorm 本体的 8 核、strict
> `<5 us` 结论不变；packed QKV / pre-RoPE 前端相对 `f9065261` 的 I6
> 整体性能 NO-GO 也不被本文覆盖。
>
> **验证形态。** 设备验证仍是在固定 immutable K8 镜像上的只读
> `pypto-lib` source overlay；镜像内 `pypto-lib` 仍为 `cb96747e`，没有构建
> 包含本文提交的新镜像。

## 1. 结论

`fa58b5cf` 的五层 raw L2 swimlane 证实：RMSNorm 虽然已经约 `4–5 us`，
但完成后到首个直接消费者 `swa_qkv_proj` 的 useful Worker slice 还要再等约
`5 us`。这不是 RMS compute，而是另一段 scheduler completion/dispatch bubble。

最终方案让 QKV Worker setup 在 RMS 完成前就开始：Worker gap p50 从
`+4.77 us` 变为 `-1.78 us`（负值表示重叠）。RMS raw kernel 结束到 QKV raw
kernel 开始仍有 `2.64 us` p50 residual，但已经不是另一段约 `5 us` 的 Worker
调度等待：

| 指标 | baseline `fa58b5cf` | candidate `e5e26f9f` |
|---|---:|---:|
| QKV Worker start − RMS Worker end p50 | `+4.77 us` | **`-1.78 us`** |
| QKV raw-kernel start − RMS raw-kernel end min | `4.60 us` | `2.08 us` |
| raw-kernel residual p50 | `5.00 us` | **`2.64 us`** |
| raw-kernel residual mean | `5.01 us` | `2.58 us` |
| raw-kernel residual max | `5.48 us` | **`3.16 us`** |
| RMS raw-kernel span min/p50/max | — | `3.92 / 4.10 / 4.50 us` |
| RMS Worker span min/p50/max | — | `4.16 / 4.35 / 4.78 us` |

raw-kernel residual p50 减少 `2.36 us`，约 `47.2%`；更直接的调度证据是
QKV Worker setup 已提前 `1.50–2.06 us` 与 RMS 重叠。五层 L3/L4 hidden
byte-exact、finite、TP spread=`0`。整网 BS1/ctx64K A/B/A 的 candidate 位于
baseline bracket 内，没有引入可测回退。

最终源码状态：

```text
base
fa58b5cffe41b30d3f8d94482230867ee34b9e84

commits
18d1b5197acf4829b171bfa144eb06e5b0cacfdf
  perf(step3p5): pre-stage SWA projections behind RMSNorm
e5e26f9f5bf9184f97a4684ae7e865f1a8b0d228
  perf(step3p5): prioritize QKV prestage after SWA RMSNorm

remote
csy0225/pypto-lib:stepfun/develop = e5e26f9f
```

## 2. 为什么不是只给 `swa_qkv_proj` 开 `allow_early_resolve`

`allow_early_resolve` 是 **producer 属性**，表示 runtime 可以在该 producer
执行期间预解析、预驻留它的直接 consumers；consumer 仍要等 producer completion
和数据可见性条件满足后才真正执行。

因此两个 flag 优化的是不同的边：

```text
RMS producer allow_early_resolve
  -> 优化 RMS → QKV / head-gate

QKV producer allow_early_resolve
  -> 优化 QKV → split/QKNorm/RoPE
```

`fa58b5cf` 中 `swa_qkv_proj` 本来就已有 `allow_early_resolve=True`，所以
QKV → `swa_qkv_split_qknorm_rope` 的 useful gap 已约 `1.96 us` p50；它无法反向
缩短 QKV 自己等待 RMS completion 的 Worker gap。本文保留 QKV 的 flag，同时给
RMS producer 开 early resolve，从而同时优化两跳。

## 3. 最终实现

### 3.1 RMS producer 允许 consumer 预驻留

`swa_rmsnorm_zc` 的 8 个 logical tasks 保持原 workload-derived grain：

```text
BATCH=16
rows_per_task=2
logical tasks=8
```

只在该 producer 上增加 `allow_early_resolve=True`。TensorMap 数据依赖、RMS
归约顺序、输出 dtype/layout 和 consumer completion gate 均未改变。

### 3.2 只优先预置关键路径 QKV

RMS 的直接 consumer 不只有 14-slice packed QKV projection，还有 8-block
head-gate logits。若两组都同时 speculative prestage，head-gate 会占用调度资源，
推迟 RMS completion polling，gap 下降不稳定。

最终方案复用本来就必须存在、且早于 RMS 完成的 `swa_attn_out_zero` TaskId，
给 `swa_head_gate_logits_mm` 增加一条显式、未标记 early 的依赖：

```text
swa_attn_out_zero
  -> swa_head_gate_logits_mm       normal dispatch

swa_rmsnorm_zc (early producer)
  -> swa_qkv_proj                  critical prestage
```

这没有新增数学依赖或改变 head-gate 数据流；作用只是把非关键 head-gate 从 RMS
producer 的 speculative fanout 中隔离，让 QKV 在 RMS 执行期间优先驻留。QKV 自身
继续保留 `allow_early_resolve=True`，用于下一跳 pre-RoPE consumer。

## 4. 测试与五层 DFX

目标 unit contract：

```text
22 passed
```

权威五层工作点：

```text
layers       L0 Full dense / L1-L2 SWA dense / L3 SWA MoE / L4 Full MoE
active batch 1
context      65536 per active sequence
num_blocks   512
devices      8,9,10,11,12,13,14,15
warmup/iters 3/3
source       e5e26f9f (clean frozen snapshot)
```

设备结果：

```text
launch rc                         0
dep-gen preserved after swim     true
rank count                        8
L3 exact / finite / TP spread    true / true / 0
L4 exact / finite / TP spread    true / true / 0
RMS raw-kernel span min/p50/max   3.92 / 4.10 / 4.50 us
RMS Worker span min/p50/max       4.16 / 4.35 / 4.78 us
QKV Worker gap p50                -1.78 us (setup overlaps RMS)
QKV raw-kernel residual min/p50/max
                                  2.08 / 2.64 / 3.16 us
```

candidate container 的 `rc=1` 仍来自既有 canonical analyzer 限制：部分
zero-local-token routed early-dispatch task 在 deps 中存在、raw AICore swim
中没有记录。外层 runner、raw DFX、dep-gen、五层精度和本文 gap 分析均完整；
不得把该限制写成 canonical structural PASS。

## 5. 整网 A/B/A

工作点为 BS1、ctx65536、512 blocks、warmup/measured=`10/100`：

| arm | p50 | hidden SHA / token |
|---|---:|---|
| A1 baseline `fa58b5cf` | `30.992 ms` | exact / `14371` |
| B candidate `e5e26f9f` | `30.997 ms` | exact / `14371` |
| A2 baseline `fa58b5cf` | `31.136 ms` | exact / `14371` |

```text
baseline center       31.064 ms
half-range floor       0.072 ms
candidate delta       -0.067 ms / -0.216%
delta / floor          0.931x
performance verdict   WITHIN_BASELINE_BRACKET
precision gate        PASS
```

三臂 hidden SHA256 全等：

```text
567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e
```

因此本文只声明“QKV setup 已与 RMS 重叠，raw-kernel residual 约减半，且相对
`fa58b5cf` 无整网回退”，不把 `-0.067 ms` 写成显著整网收益。

## 6. 与 I6 NO-GO 的边界

I6 的权威比较是 `f9065261 → fa58b5cf`：整网回退 `+4.233%`，fresh 五层
QKV/pre-RoPE strict `<46 us` 为 `39/40`。本文 A/B/A 比较的是
`fa58b5cf → e5e26f9f`，只能证明新的调度修复没有进一步回退，并改善了 L3
SWA RMS→QKV gap。

所以当前同时成立：

```text
I6 packed QKV / pre-RoPE 相对 f9065261      NO-GO
I7 RMS→QKV critical prestage 相对 fa58b5cf   GO / MERGED
new immutable image                          NOT BUILT
production release qualification             NOT CLAIMED
```

## 7. 权威 artifact

```text
five-layer DFX:
/mnt/persist/chensiyu/workspace/perf-2026q3/
  rms-proj-critical-prestage-validation-20260812-r3/

raw dfx outputs:
five_layer/candidate_dfx/runtime/build_output/
  FiveLayerMoe_20260812_054213/dfx_outputs/

whole A/B/A:
/mnt/persist/chensiyu/workspace/perf-2026q3/
  attn-mix-device-gate-20260811/out/
  aba-bs1-ctx64k-20260812-134604/
```

核心 hash：

```text
ALL_RANKS_swimlane_bundle.tar.gz
2f58af78bc2f1d8121426aca1e531fb76ada88072263a1ed97d5dfb7936cf083

RMS_QKV_GAP_REPORT.json
ed7ebc7193340877c2cd193eb7ec2903b17325398e607bc52f27a0fe64e90ceb

RMS_QKV_GAP_REPORT.md
20179124939fce67fa50ff44129ca1b2dc494dd52932aa208b49e11b9c7d1762

ABA_RESULT.json
247fe3cb67fe81053f43932655521225087e60ccee53345fed3c93a7f2a5d4dc

dfx_protocol_report.json
4c4b7beefd786a0754493e62ecaa5981438e92a9415cdb8c308d1c1df1fe8d51

five_layer_moe_report.json
506189d1274b70f0fd00ecc414c45cace8476f7a2fabed82ec605ca7ec4f081b
```

五层 all-ranks swimlane delivery：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  rms-proj-critical-prestage-validation-20260812-r3/delivery/
  ALL_RANKS_swimlane_bundle.tar.gz
```

bundle 含 56 项，即 8 ranks ×
`CPM_observed/CPM_static/critical_path/deps/l2/merged/name_map`。
