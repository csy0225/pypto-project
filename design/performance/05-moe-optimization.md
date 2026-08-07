# 05 · MoE 优化专项：L0–L4 focused network

> **状态（2026-08-07）**：L0–L4 focused MoE 产品实现 `7928a275` 已合入
> `pypto-lib stepfun/develop@63814d4ae62718b3c0721834878e4b4af4e7ac1b`。
> 0162 上基于 `c9af5790` 的六档 BS=`1,2,4,7,8,16` formal normal campaign
> 已完成 36/36 fresh
> process run、完整 `hidden_l3/hidden_l4` golden、精度 finalize 和三轮
> counterbalance；每个 sequence 都独立使用 `context_len=65536`。
>
> **SWA 精度修复**：`63814d4a` 将 sliding-window score mask 从 `pl.cmp`
> predicate 转换路径改为 typed INT32 数值区间 mask。0162 使用 pre-fix image
> substrate + `63814d4a` 精确 source overlay 的 N=128 回归为
> `127/128=99.21875%`、TP spread=0；唯一 miss 为
> `step94 expected=478 actual=320`。
>
> **当前发布边界（NO-GO）**：matched-source whole-net 1-step×2、2-step×2
> 已 8/8 sealed PASS，source-overlay N=128 也已过线；但没有 immutable image
> 包含 `63814d4a`。SWA 源码变化后，`c9af5790` 的六档 golden 与性能不能自动
> 升级为最终证据。按用户决定，镜像发布推迟到统一 release commit 确定后；
> final image 六档回归、formal matched-source DFX、route-aware reanalysis 和
> all-rank swimlane 仍待完成。本文只覆盖五层裁剪网络，不把结果外推为 45 层
> whole-net、prefill、L43/L44 specialization 或其它机器。
>
> **pre-fix formal campaign 根目录**：
> `/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-formal-c9af-20260806-v2`
>
> **pre-fix authoritative normal 报告**：
> `.../campaign/matrix_correctness_report.json` 和
> `.../campaign/final-r123/matrix_performance_report.json`

---

## 0. 结论

五层拓扑固定为：

```text
L0  Full Attention + Dense MLP
L1  SWA            + Dense MLP
L2  SWA            + Dense MLP
L3  SWA            + MoE
L4  Full Attention + MoE   （直接消费真实 hidden_l3）
```

本轮最终落地：

1. routed fused gate/up 拆为独立 `expert_gate_mm`、`expert_up_mm` 和
   `expert_gate_up_act`；
2. 普通 routed expert 的 receive row tile `32 → 16`；
3. gate/up cube tile 使用 `K=512, N=64`；
4. routed down 使用 `N=256`；
5. L43/L44 specialization 明确保留原 `row=32, down N=128`；
6. 保留 `combine_scatter → combine_wait → combine_reduce` 的真实依赖。

`c9af5790` pre-fix formal normal 结果：

| BS | Baseline median-round p50 | Candidate median-round p50 | reduction |
|---:|---:|---:|---:|
| 1 | `14.4086 ms` | `13.0882 ms` | `9.164%` |
| 2 | `14.2412 ms` | `13.9802 ms` | `1.833%` |
| 4 | `14.9853 ms` | `14.4572 ms` | `3.524%` |
| 7 | `16.1734 ms` | `15.1925 ms` | `6.065%` |
| 8 | `15.8162 ms` | `15.7330 ms` | `0.526%` |
| 16 | `19.6457 ms` | `17.3643 ms` | `11.613%` |

六档均满足 `hidden_l3/hidden_l4` BF16 bit-exact、finite、TP spread=0，且
`performance_non_regression_all_batches=true`。早期诊断 DFX 中 gate/up task 已从约
`144 µs` 降至 `12.7–12.9 µs`；combine wait 仍暴露远端 producer/route-skew 尾部，
但 formal 64K DFX 未完成前不发布最终 wait 或 swimlane 结论。由于
`63814d4a` 修改了 L1–L3 会经过的 SWA mask，这些数据只作为 pre-fix 对照，
最终 immutable image 必须重新生成 golden 并重跑 A/B。

---

## 1. 范围与对象冻结

### 1.1 唯一调优对象

focused harness 只组合 canonical 的前五个物理层。计算 body 不复制，直接复用
`models.step3p5.decode_fwd` 中的 attention、dense MLP、router、dispatch、routed/shared
expert、combine 和 TP all-reduce 函数。

关键合同由 card-free AST 测试固定：

- 1 次 `full_chip_orch`；
- 2 次 `swa_chip_orch`；
- 1 次 `swa_moe_chip_orch`；
- 1 次 `full_moe_chip_orch`；
- 返回完整 `hidden_l3` 和 `hidden_l4`；
- L4 第一个输入必须是实际 `hidden_l3`，不能用独立输入或 mock。

### 1.2 代码与分支

```text
repo:
  /data/chensiyu/hw_project/pypto/workspace/pypto-lib-moe-final-20260806
development branch:
  moe-opt
base:
  stepfun/develop@56b3d477953ab1e2df87213aef3a536c64051dcc
product commit:
  7928a2751930b04c866788a396a7337b62c6d32f
published branch/current remote tip:
  stepfun/develop
  63814d4ae62718b3c0721834878e4b4af4e7ac1b
candidate decode_fwd.py SHA256:
  7884da7c33a2d338fd36097676a5fafbcc2795c845409868ff0ce40cbb2bc2f9
current attention_swa.py SHA256:
  c451b4cc1462cf2b7b960798d6242936962427940f68757bf7eea6679c109623
```

实现及验证文件：

```text
models/step3p5/decode_fwd.py
tests/step3p5/harnesses/_five_layer_moe_program.py
tests/step3p5/harnesses/_stage_five_layer_moe.py
tests/step3p5/unit/test_five_layer_moe_contract.py
tests/step3p5/unit/test_five_layer_moe_dfx_analysis.py
tools/step3p5/analyze_five_layer_moe_dfx.py
tools/step3p5/five_layer_moe_holder.py
```

`decode_fwd.py` 中普通 `_expert_routed` 是 canonical 共享 helper，因此产品源码影响面
不只两个 focused 层；但本轮设备证据只覆盖 L3/L4。L43/L44 走独立 specialization，
本次保持原 tile。本文不把 focused PASS 写成 whole-net release PASS。

### 1.3 0162 执行 substrate

```text
machine:
  gpu-a910x-0162.host.platform.shaipower.com
devices:
  8,9,10,11,12,13,14,15
checkpoint:
  /data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
image:
  hub.i.basemind.com/stepcast/vllm-pypto@sha256:cab89668164cf85dc75e4f3ac53ef77ef4b8653767c7d147c5113cdee6a9d88c
image pypto-lib:
  c9af5790d5fe450e14fd43c88099b87539089d17
image PyPTO:
  8e92b46808f9f7c09b6431ad4691503f09c12ee5
attention profile:
  a2a3
l2_swimlane_reuse_dep_gen:
  available and required for formal DFX
```

formal normal A/B 使用同一 digest-pinned image；baseline/candidate 只切换冻结 source
tree。镜像审计要求 PyPTO=`8e92b468`、attention=`a2a3` 且
`l2_swimlane_reuse_dep_gen` 存在，避免再次回退到旧 `defa97c5`/portable substrate。
该 digest 不包含 `63814d4a`，因此本节只描述 pre-fix substrate；最终镜像尚未构建。

---

## 2. Golden 与精度合同

### 2.1 六档 64K golden（pre-fix）

```text
/mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-formal-c9af-20260806-v2/campaign/golden/heterogeneous-64k/bs{1,2,4,7,8,16}/
```

| BS | `manifest.json` SHA256 |
|---:|---|
| 1 | `5c8ee88e8f5faf966604cb503f9476dc7c6e9cb7d28954de2fc95ccfa175dd3c` |
| 2 | `acce36c926f7649bafae027bdc94a224639567301983f58d8b67f316d31f23c5` |
| 4 | `68041f5b9e5203fdb7e5be8a0c26d0bb40a166de47b6d9fd039323b93ba93c9f` |
| 7 | `81c8feebed6c2e56d434b47db95cc7d1df18a34e57467e053177971865a5f7f8` |
| 8 | `36a8a74a3df5d1141487235fb7a5ac2a2dbead74d4b9e84788e49890eab980fa` |
| 16 | `55f676ae7e014606561e43afc44d305bec72661ab3e2f7eefc0160b8a318bf72` |

每档 golden 都来自独立 fresh-process baseline run，并满足：

```text
shape = [8, active_batch, 4096]
dtype = torch.bfloat16
finite = true
nonzero active rank/rows = 8 * active_batch
max TP spread = 0.0
context_len_per_sequence = 65536
blocks_per_sequence = 512
```

candidate 对 baseline 的 L3、L4 均为 `max_abs=0`、`bad_ratio=0`、bit-exact，
并通过 batch-extension invariance。
精度比较使用完整 hidden state，不以 token argmax 代替中间层精度。
这些 golden 绑定 `c9af5790` 的 SWA 行为；`63814d4a` 之后必须在最终 immutable
image 上重新 dump，不能原样作为 final golden。

---

## 3. MoE 数据流与依赖

```text
attention hidden
  → deferred RMSNorm + per-token INT8 quant/scale
  → sigmoid router + bias + top-k + route weight
  → expert-lane dispatch push / arrival / gather
  → local routed gate INT8×INT8
  → local routed up INT8×INT8
  → dequant + BF16 round + SiLU/SwiGLU activation
  → per-token W8A8 requant
  → routed down INT8×INT8 + route-weight epilogue
  → combine scatter
  → combine arrival wait
  → stable TOPK-order FP32 reduction
  → shared expert merge
  → residual BF16 output
```

本轮不改变：

- router、top-k、route index/weight；
- dispatch/combine 通信方向、expert lane ownership 和 epoch；
- INT32 accumulator、scale 应用、BF16 round、W8A8 requant；
- route weight 在 routed down epilogue 的位置；
- TOPK 固定顺序 FP32 combine；
- shared expert 和 TP all-reduce 数学；
- active-token bound 与 inactive rows 屏蔽。

`combine_wait` 的显式依赖必须保留：

```python
with pl.spmd(...) as combine_scatter_tid:
    ... put payload ...
    ... notify peer ...

with pl.at(
    ...,
    deps=[combine_scatter_tid],
) as combine_wait_tid:
    ... wait all peer epochs ...
```

曾删除这条结构依赖的候选在设备上出现 `S1:running-stalled`，因此它不是可用的
“隐藏 wait”优化。未来若要 overlap，必须重新拆分 payload publication、notify 和
local consumer，并为每条真实 RAW/lifetime 边建立新协议；不能直接删 dependency。

---

## 4. Baseline critical path

Baseline 普通 routed expert 使用：

```text
receive row tile = 32
fused gate/up K = 64
fused gate/up N = 64
down N = 128
```

### 4.1 L3（SWA + MoE）

```text
receive tiles by rank = [0, 4, 0, 8, 4, 8, 0, 8]
fused gate/up AIC p50 ≈ 143.5 µs
max combine wait       = 1597.6 µs
shared expert max span = 222.2 µs
combine scatter max    ≈ 141.1 µs
```

### 4.2 L4（Full Attention + MoE）

```text
receive tiles by rank = [4, 4, 4, 0, 8, 0, 8, 4]
fused gate/up AIC p50 ≈ 144.1 µs
max combine wait       = 1590.3 µs
shared expert max span = 225.2 µs
combine scatter max    ≈ 147.1 µs
```

### 4.3 根因分解

1. **gate/up task 过大**：约 144 µs，远高于本 workload 期望的
   `10–30 µs/task`，一个 fused task 同时绑定两次 GEMM 和 vector epilogue，无法独立
   校准 cube 粒度。
2. **route skew 形成跨 rank 尾部**：部分 rank 有 8 个 receive tiles，部分为 0；
   zero-local-work rank 很早进入 combine wait，却必须等远端重负载 rank。
3. **“未用满核心”有两类**：routed active rank 的任务波次与粒度不足以稳定填满资源；
   shared expert 更明确地只有一个 mixed task，峰值仅 `1 AIC + 2 AIV`。
4. **combine scatter 有 route-heavy block**：最大 slice 约 140–145 µs，单 expert block
   内串行遍历 source/slot，存在长尾。
5. **DFX makespan 不能当 clean latency**：TP all-reduce 自旋和 collector 插桩可将某些
   rank 的 makespan 放大到数百 ms。端到端收益只能引用无插桩的 repeated A/B。

---

## 5. 优化设计与落地

### 5.1 Gate/up stage split

原 fused task：

```text
expert_gate_up:
  gate GEMM + up GEMM + dequant + activation + BF16 store
```

改为：

```text
expert_gate_mm   → gate INT32 GM scratch
expert_up_mm     → up INT32 GM scratch
(gate, up scratch) → expert_gate_up_act → BF16 activation
```

cube 与 vector epilogue 解耦后，可独立选择 gate/up 的 K/N tile，并暴露更多
write-disjoint logical tasks。最终参数：

```text
ROUTED_GATE_MM_K_CHUNK  = 512
ROUTED_GATE_MM_N_CHUNK  = 64
ROUTED_GATE_ACT_N_CHUNK = 64
RECV_TILE               = 16
```

### 5.2 Down tile

```text
ROUTED_DOWN_N_CHUNK = 256
```

在 row=16 下，candidate down AIC p50 约 `15.6–21.6 µs`。该选择与 gate/up split
共同验证；不能把最终 11.58% 收益写成 “down-N256-only”。单独只改 down N=256 的
历史候选 p50 为 `11.9023 ms`，收益仅约 `2.26%`。

### 5.3 Specialization safety

L43/L44 的 SwiGLU7 specialized helper 没有被 focused L0–L4 真机覆盖，因此显式冻结：

```text
RECV_SPECIAL_TILE            = 32
ROUTED_SPECIAL_DOWN_N_CHUNK  = 128
special gate/up              = 原 fused K64/N64
```

这避免普通 helper 的 tile 常量静默改变未验证 specialization。

### 5.4 Memory 代价

stage split 新增两个 INT32 GM scratch：

```text
gate_i32: [4608, 1280] × 4 B = 22.5 MiB/rank
up_i32:   [4608, 1280] × 4 B = 22.5 MiB/rank
合计                              45.0 MiB/rank
```

这是用 GM 容量换 task grain 和并行度的明确成本。片上 memory report 无 overflow：

| Kernel | Mat | Left | Right | Acc | Vec |
|---|---:|---:|---:|---:|---:|
| `expert_gate_mm` / `expert_up_mm` | `264/512 KiB` | `8/64 KiB` | `32/64 KiB` | `4/128 KiB` | `32/184 KiB` |
| `expert_down` | `65/512 KiB` | `1/64 KiB` | `16/64 KiB` | `16/128 KiB` | `81.1/184 KiB` |
| `expert_gate_up_act` | — | — | — | — | `16.1/184 KiB` |

baseline/candidate `perf_hints.log` 均为 62 行，其中 `decode_fwd.py` 22 条，未见
resource overflow。

---

## 6. 早期诊断 DFX 结果（非最终 formal DFX）

本节保留 2026-08-04 的短 workload 诊断数据，用于说明 gate/up task-grain 选择和
combine wait 的解释方法。它不是六档独立 64K formal DFX，也不能作为最终 DFX 报告
或 all-rank swimlane 发布路径。

### 6.1 L3

```text
receive tiles by rank = [0, 8, 0, 16, 8, 16, 0, 16]
expert_gate AIC p50   = 12.9 µs
expert_up AIC p50     = 12.7 µs
expert_down AIC p50   ≈ 15.6–21.1 µs（随 rank/workload）
active-rank peak      通常达到 24 AIC / 48 AIV
max combine wait      = 1059.8 µs
shared expert max     = 210.2 µs，仍仅 1 AIC + 2 AIV
combine scatter max   = 140.8 µs
```

### 6.2 L4

```text
receive tiles by rank = [8, 8, 8, 0, 16, 0, 16, 8]
expert_gate AIC p50   = 12.8 µs
expert_up AIC p50     = 12.8 µs
expert_down AIC p50   ≈ 18.1–21.6 µs（随 rank/workload）
active-rank peak      通常达到 24 AIC / 48 AIV
max combine wait      = 1063.9 µs
shared expert max     = 219.4 µs，仍仅 1 AIC + 2 AIV
combine scatter max   = 142.6 µs
```

row tile 减半使 receive tile 数翻倍是预期，不表示路由 token 翻倍。决定性信号是
单 gate/up task 降至 10–30 µs、活跃 rank 的 AIC/AIV 并行度提高，以及无插桩 p50
下降。

### 6.3 Combine wait 的正确解释

- L3 最长 wait 在 `rank2`，其本地 receive tile 为 0；
- L4 最长 wait 在 `rank3`，其本地 receive tile 为 0；
- wait 的显式 scatter dependency 存在；
- DFX 没有跨 rank common-clock anchor；
- 因此可证明的是“远端 producer / route skew 尾部仍在”，不能证明 wait 指令本身慢，
  也不能从不同 rank 的归一化 timestamp 算出精确网络延迟。

---

## 7. 性能 A/B

### 7.1 pre-fix 六档独立 64K

每个 run 都是 fresh process，三轮顺序为：

```text
r1/r2: per-BS baseline → candidate
r3:    per-BS candidate → baseline
warmup = 5
measured_iters = 30
context_len_per_sequence = 65536
active_total_context_tokens = BS * 65536
```

| BS | Baseline p50 | Candidate p50 | p50 reduction | Baseline mean | Candidate mean |
|---:|---:|---:|---:|---:|---:|
| 1 | `14.4086` | `13.0882` | `9.164%` | `14.4392` | `13.2242` |
| 2 | `14.2412` | `13.9802` | `1.833%` | `14.6349` | `14.0110` |
| 4 | `14.9853` | `14.4572` | `3.524%` | `15.0584` | `14.5889` |
| 7 | `16.1734` | `15.1925` | `6.065%` | `16.2990` | `15.4190` |
| 8 | `15.8162` | `15.7330` | `0.526%` | `16.3686` | `16.0016` |
| 16 | `19.6457` | `17.3643` | `11.613%` | `19.7844` | `17.6998` |

报告：

```text
.../campaign/matrix_correctness_report.json
  SHA256 451a47d152fc0b0af1b9a21200011af5c0cfa6a884b6996ca057286c032f3368
.../campaign/final-r123/matrix_performance_report.json
  SHA256 d238baa1524dc9c7fe3f703f596211689f5d904a9d209589c09d12df722e6875
.../campaign/normal_seal_authority.json
  SHA256 875804ddbb81b4f15a907e41e454ed3004aca3b56075063431edef5efc70c531
```

报告的 `measurement_integrity_passed`、
`hidden_hash_exact_across_selected_rounds` 和
`performance_non_regression_all_batches` 均为 `true`。
该结论只对 `c9af5790` pre-fix source/image 组合成立，不是 `63814d4a` 最终准出。

### 7.2 Whole-net matched-source A/B

同镜像、同 checkpoint、同设备、只切换冻结 `decode_fwd.py` 的 BS1 teacher-forced
冒烟位于：

```text
.../whole-net-matched-ab-20260807T024525Z/
```

baseline/candidate 的 1-step 各两轮均输出 `303`，2-step 各两轮均输出
`303,1207`；共 8/8 run 通过，publication seal=`PASS`：

```text
.../whole-net-matched-ab-20260807T024525Z/publication_seal_report.json
SHA256 c0a03127a706ac3d987369573dda3a8b20f725ee0efee47af78a8c4892704338
```

该 matched-source A/B 关闭了先前 1/2-step smoke 的不完整状态，但两步输出仍不能
替代 N=128、ALIGNED≥95% 多步精度门。

### 7.3 `63814d4a` source-overlay N=128 精度

SWA 定位显示问题来自 `pl.cmp` predicate 到数值 mask 的转换路径。最终修复使用
显式 INT32 区间算术生成 `{0,1}` mask，再 cast 到 score dtype；不改 MoE tile、
route、combine 或 reduction 数学。

0162 cards `0–7`、fresh container、teacher-forced N=128：

```text
/mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-precision-fix-20260807-v2/runs/Lmask-v1-n128/summary.json
SHA256 7f91dcdb3b30bf7beeac9b6994c55b42c8154438bc63b9025bd9cb21a2479503
token exact: 127/128 = 99.21875%
unique miss: step94 expected=478 actual=320
hidden_tp_spread_max: 0.0
```

该 source tree 的关键 SHA256 与 `stepfun/develop@63814d4a` 一致，其中
`attention_swa.py=c451b4cc…`。该结果通过 source-level 精度门，但不是
immutable-image 回归，也没有证明性能无回退。

### 7.4 早期短 workload 诊断

2026-08-04 的 context=1 repeated-token A/B 为 `12.1777→10.7677 ms`
（`-11.58%`）。它帮助选择 task grain，但不再作为最终性能准出；性能准出以 7.1
的六档 64K counterbalanced campaign 为准。

---

## 8. 否决候选

| 候选 | 结果 | 决策 |
|---|---|---|
| fused gate `N32/K64` | exact，p50 `14.429 ms` | 慢于 baseline，NO-GO |
| fused gate `N32/K512` | exact，p50 `13.756 ms` | 仍回退，NO-GO |
| fused gate `N64/K512` | Vec overflow | compile/resource NO-GO |
| split gate/up `N128/K512` | Mat `532480 > 524288 B` | compile NO-GO |
| down `N256` only | p50 `11.9023 ms`，约 `2.26%` | 不足以解释最终收益 |
| 删除 scatter→wait dependency | `S1:running-stalled` | correctness/liveness NO-GO |

最终候选不是“tile 越小越好”或“强制每 task 10 µs”。选择依据是 task duration、
stage span、wave、核心利用率、片上资源、hidden 精度和无插桩 wall p50 的联合结果。

---

## 9. 剩余关键路径与下一轮方案

按收益/风险排序：

1. **Shared expert 并行化**：当前每层一个 mixed task、最多 `1 AIC + 2 AIV`、
   `190–219 µs`。候选应拆五个 32-channel gate/up tile，再拆 write-disjoint down
   feature tile；避免生成一个宽 `[BATCH,160]` Vec tile。
2. **Combine scatter route-heavy block**：按 `(expert, source-rank)` 建 write-disjoint
   scatter block，保持每 expert lane/peer 的 payload-before-notify 和 epoch 计数。
3. **Route-skew tail 调度**：优化 heavy rank 的 producer tail，而不是让 zero-work rank
   忙等做无意义计算；必须用第二组异构 token 验证，不能只对 repeated route 特化。
4. **受控 overlap 实验**：只有在拆出独立 publication/notify 协议并证明 RAW/lifetime
   后，才评估 local scatter 与 remote arrival wait 的 overlap。禁止再次直接删除依赖。
5. **GM scratch 生命周期/容量**：whole-net 集成前审计 45 MiB/rank scratch 是否复用、
   是否与其它层峰值叠加；focused PASS 不替代整网 HBM gate。

停止条件仍是：精度 bit-exact、TP spread=0、无 stall、memory 合法，且在同口径
repeated/heterogeneous A/B 中取得稳定收益。

---

## 10. 当前产物索引

```text
/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-formal-c9af-20260806-v2/
├── campaign/
│   ├── runs/                    # 36 sealed normal runs
│   ├── golden/heterogeneous-64k/
│   ├── matrix_correctness_report.json
│   ├── normal_seal_authority.json
│   └── final-r123/matrix_performance_report.{json,md}
├── sources/dfx-c9af-v2/{baseline,candidate}/
└── whole-net-matched-ab-20260807T024525Z/

/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-precision-fix-20260807-v2/
└── runs/Lmask-v1-n128/          # 63814d4a source-overlay N=128
```

### DFX 报告

```text
PENDING:
  formal matched-source DFX 12 runs
  route-aware publication reanalysis
  fixed final JSON/Markdown report path
```

### Candidate merged swimlane

```text
PENDING:
  rank{0..7}/d0/merged_swimlane_*.json
```

### Provenance

```text
campaign/normal_seal_authority.json
  SHA256 875804ddbb81b4f15a907e41e454ed3004aca3b56075063431edef5efc70c531
campaign/matrix_correctness_report.json
  SHA256 451a47d152fc0b0af1b9a21200011af5c0cfa6a884b6996ca057286c032f3368
campaign/final-r123/matrix_performance_report.json
  SHA256 d238baa1524dc9c7fe3f703f596211689f5d904a9d209589c09d12df722e6875
whole-net-matched-ab-20260807T024525Z/publication_seal_report.json
  SHA256 c0a03127a706ac3d987369573dda3a8b20f725ee0efee47af78a8c4892704338
../moe-precision-fix-20260807-v2/runs/Lmask-v1-n128/summary.json
  SHA256 7f91dcdb3b30bf7beeac9b6994c55b42c8154438bc63b9025bd9cb21a2479503
```

所有临时脚本位于 `/mnt/persist/chensiyu/workspace/moe-opt/tmp`。当前 normal
evidence 已增加 fail-closed publication seal，且 36/36 fresh-process run 已 seal。
不得改写旧 evidence 或把历史 `container.rc=1` 升级为 PASS。SWA mask 已变化，
所以上述 `c9af5790` normal evidence 只保留为 pre-fix 对照。统一 release commit
确定并构建 immutable image 后，需重跑六档 golden/A/B、N=128、formal DFX 和
all-rank swimlane，再在本节补唯一 final 路径和 SHA256。
