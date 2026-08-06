# 05 · MoE 优化专项：L0–L4 focused network

> **状态（2026-08-06）**：L0–L4 focused MoE 产品实现已合入
> `pypto-lib stepfun/develop@7928a2751930b04c866788a396a7337b62c6d32f`。
> 0162 上六档 BS=`1,2,4,7,8,16` 的 formal normal campaign 已完成 36/36 fresh
> process run、完整 `hidden_l3/hidden_l4` golden、精度 finalize 和三轮
> counterbalance；每个 sequence 都独立使用 `context_len=65536`。
>
> **当前发布边界**：formal matched-source DFX、route-aware publication gate 和最终
> all-rank swimlane 尚未完成，不能把 2026-08-04 的 context=1 诊断 DFX 路径写成最终
> 64K DFX。本文只覆盖五层裁剪网络，不把结果外推为 45 层 whole-net、prefill、
> L43/L44 specialization 或其它机器的发布结论。
>
> **formal campaign 根目录**：
> `/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-formal-act-n64-20260806-v1`
>
> **当前 authoritative normal 报告**：
> `.../campaign/matrix_correctness_report.json` 和
> `.../campaign/matrix_performance_report.json`

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

当前 formal normal 结果：

| BS | Baseline median-round p50 | Candidate median-round p50 | reduction |
|---:|---:|---:|---:|
| 1 | `12.9437 ms` | `12.9385 ms` | `0.04%` |
| 2 | `13.7780 ms` | `12.8646 ms` | `6.629%` |
| 4 | `15.2757 ms` | `13.4254 ms` | `12.113%` |
| 7 | `16.1510 ms` | `15.5611 ms` | `3.652%` |
| 8 | `15.8222 ms` | `14.3619 ms` | `9.229%` |
| 16 | `18.8267 ms` | `16.7303 ms` | `11.135%` |

六档均满足 `hidden_l3/hidden_l4` BF16 bit-exact、finite、TP spread=0，且
`performance_non_regression_all_batches=true`。早期诊断 DFX 中 gate/up task 已从约
`144 µs` 降至 `12.7–12.9 µs`；combine wait 仍暴露远端 producer/route-skew 尾部，
但 formal 64K DFX 未完成前不发布最终 wait 或 swimlane 结论。

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
branch:
  moe-opt
base:
  stepfun/develop@56b3d477953ab1e2df87213aef3a536c64051dcc
final commit:
  7928a2751930b04c866788a396a7337b62c6d32f
candidate decode_fwd.py SHA256:
  7884da7c33a2d338fd36097676a5fafbcc2795c845409868ff0ce40cbb2bc2f9
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
  hub.i.basemind.com/stepcast/vllm-pypto@sha256:b43e704ae878283575b77178501371bdb47848c4db97b2db6dbc3d7007a4995d
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

---

## 2. Golden 与精度合同

### 2.1 六档 64K golden

```text
/mnt/persist/chensiyu/workspace/moe-opt/tmp/
  moe-formal-act-n64-20260806-v1/campaign/golden/heterogeneous-64k/bs{1,2,4,7,8,16}/
```

| BS | `manifest.json` SHA256 |
|---:|---|
| 1 | `b169691fa0703c890f7a41523dec2a39226c33fd57f34fcaea74cd3aba8dcd22` |
| 2 | `f1ac1026c41f2a372e0224193c124d481c88c7c305d90e0afe52ca770fd0bdbc` |
| 4 | `bdee4f56a9283be06f1a5a0402596a287a2af8b7526797480f0e05653923727d` |
| 7 | `1aa70db3de8d2ae564fff691b4b9b609e5ad3e6f330f2033bb7265e305aa9338` |
| 8 | `e5b1b38e88c3c67397005c63151e8a1e261079b9a3188a4efeed89cda934a7f9` |
| 16 | `fcb04b11570b1395e35fadee6b42885691c2dac1e71d3b0c7d5e2e2648c36c09` |

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

### 7.1 主准出：六档独立 64K

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
| 1 | `12.9437` | `12.9385` | `0.04%` | `13.4412` | `12.9286` |
| 2 | `13.7780` | `12.8646` | `6.629%` | `14.0678` | `13.1146` |
| 4 | `15.2757` | `13.4254` | `12.113%` | `15.3179` | `14.3530` |
| 7 | `16.1510` | `15.5611` | `3.652%` | `16.6213` | `14.6357` |
| 8 | `15.8222` | `14.3619` | `9.229%` | `15.7558` | `14.3855` |
| 16 | `18.8267` | `16.7303` | `11.135%` | `18.9856` | `16.7510` |

报告：

```text
.../campaign/matrix_correctness_report.json
  SHA256 451a47d152fc0b0af1b9a21200011af5c0cfa6a884b6996ca057286c032f3368
.../campaign/matrix_performance_report.json
  SHA256 44bacd4980f656fd4ccd777b4515dd8c8d58b1edb6ccbc0feceed177dfc5a17b
```

### 7.2 早期短 workload 诊断

2026-08-04 的 context=1 repeated-token A/B 为 `12.1777→10.7677 ms`
（`-11.58%`）。它帮助选择 task grain，但不再作为最终性能准出；最终准出以 7.1
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
/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-formal-act-n64-20260806-v1/
├── authority/
├── campaign/
│   ├── runs/                    # 36 normal runs; formal DFX待补
│   ├── golden/heterogeneous-64k/
│   ├── matrix_correctness_report.json
│   └── matrix_performance_report.json
├── sources/dfx-formal-act-n64-v1/{baseline,candidate}/
├── overlays/{dfx,route}/
└── tmp/
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
authority/final_selection_report.json
authority/matched_source_audit.json
authority/normal_seal_authority_v1.json
  SHA256 16ac43432d0462e34bb939b11fb71e146cb2b9c2b068d9c3c5eec9901faa54be
campaign/matrix_correctness_report.json
campaign/matrix_performance_report.json
tmp/final-scripts-63d7526d/SCRIPTS_SHA256SUMS
  SHA256 63d7526dc7bcda7e5d3c0404112ddae95f4020dce6fe498dd0e08d5b137722af
```

所有临时脚本位于 `/mnt/persist/chensiyu/workspace/moe-opt/tmp`。当前 normal
evidence 已增加 fail-closed publication seal：seal 必须绑定外部钉住的旧证据 manifest，
旧 validation 必须存在且为普通文件，JSON 使用递归类型严格比较；0162 已完成
36/36 idempotent seal PASS。formal DFX 和 all-rank swimlane 完成后，再在本节补
唯一最终路径和 SHA256。
