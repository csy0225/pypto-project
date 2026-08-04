# 05 · MoE 优化专项：L0–L4 focused network

> **状态（2026-08-04）**：L0–L4 focused MoE 优化已在 0162 完成实现、精度、
> 性能和 DFX 验证。本文只覆盖五层裁剪网络，不把结果外推为 45 层 whole-net、
> 64K、prefill、L43/L44 specialization 或其它机器的发布结论。
>
> **最终 DFX**：
> `/mnt/persist/chensiyu/workspace/moe-opt/l0-l4/final/dfx/candidate/`
>
> **最终 8-rank merged swimlane**：
> `/mnt/persist/chensiyu/workspace/moe-opt/l0-l4/final/dfx/candidate/swimlane/rank{0..7}/d0/merged_swimlane_*.json`
>
> **最终产物根目录**：
> `/mnt/persist/chensiyu/workspace/moe-opt/l0-l4/final`

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

主要结果：

| 指标 | Baseline | Candidate | 结果 |
|---|---:|---:|---:|
| repeated-token p50，5 warmup + 30 measured | `12.1777 ms` | `10.7677 ms` | **-11.58%** |
| repeated-token mean | `12.8490 ms` | `10.8234 ms` | `-15.76%` |
| L3 fused gate/up → split gate、up AIC p50 | `143.5 µs` | `12.9 / 12.7 µs` | 进入目标 `10–30 µs/task` |
| L4 fused gate/up → split gate、up AIC p50 | `144.1 µs` | `12.8 / 12.8 µs` | 进入目标 `10–30 µs/task` |
| L3 max combine wait | `1597.6 µs` | `1059.8 µs` | routed producer 尾部缩短 |
| L4 max combine wait | `1590.3 µs` | `1063.9 µs` | routed producer 尾部缩短 |
| `hidden_l3` / `hidden_l4` | — | BF16 bit-exact | PASS |

`combine_wait` 仍长，但它不是约 1 ms 的 wait 算术：最长 wait 出现在本地
routed receive tile 为 0 的 rank，表示该 rank 正在等待远端 routed compute / scatter
完成。各 rank 的 DFX 时钟独立归一化，没有共同外部时钟锚点，禁止直接跨 rank
相减 timestamp 后宣布远端到达延迟。

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
  /data/chensiyu/hw_project/pypto/workspace/pypto-lib-moe-opt
branch:
  moe-opt
base:
  stepfun/develop@7099476b7c4f13112b159e237e7a64344803caf0
final commit:
  505e2c6b8d7c015e2e75a6799cf2b9f335db5543
candidate decode_fwd.py SHA256:
  2475c77c18273868b2d0bf56cd93c6c70819aafb9ce7fc49db05e6f26ca2b447
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
  hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260803-attn-final-wave5
manifest:
  sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32
```

正式 A/B 使用相同的 Wave5 image 和相同的 **comm-only force-VMM-IPC runtime
override**：

```text
libhost_runtime.so SHA256:
  ee81f95ddebdee2acd23221abdef8e3ced3a0dcb8761a13154dde633be546695
```

该 override 只用于 focused IPC 通信接线，baseline/candidate 完全一致；不能与
STATUS.md 中“不挂载 override 的 immutable whole-net release gate”混为同一个对象。

---

## 2. Golden 与精度合同

### 2.1 Repeated-token golden

输入为 16 个 token `6127`：

```text
/mnt/persist/chensiyu/workspace/moe-opt/l0-l4/final/golden/repeated/
```

| 文件 | SHA256 |
|---|---|
| `hidden_l3.pt` | `4f2cbcfc8557347030ad390327941db33555327f96e76aacff2e406f5dc98cb7` |
| `hidden_l4.pt` | `1e179669c2f66bf41713996fdf506409987b1d1efad5c6bd4b2e40de223cf3f1` |

### 2.2 Heterogeneous-token golden

```text
input = 6127,303,1207,19384,872,428,4231,2636,
        6178,410,1,2,3,4,5,6
```

```text
/mnt/persist/chensiyu/workspace/moe-opt/l0-l4/final/golden/heterogeneous/
```

| 文件 | SHA256 |
|---|---|
| `hidden_l3.pt` | `8f893262c7da7611aa45ad32b62293ee33952cd60f09162976badacd0da0377b` |
| `hidden_l4.pt` | `cdb38438ead8d1df7c389a4cfc047c1a875f5114c1fafefd7157c5004dbca748` |

两组 golden 均满足：

```text
shape = [8, 16, 4096]
dtype = torch.bfloat16
finite = true
nonzero active rank/rows = 128 / 128
max TP spread = 0.0
```

candidate 对 baseline 的 L3、L4 均为 `max_abs=0`、`bad_ratio=0`、bit-exact。
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

## 6. Candidate DFX 结果

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

### 7.1 主准出：repeated-token

双方均为同一 substrate、`active_batch=16`、`context_len=1`、16 个 token `6127`、
`5 warmup + 30 measured`：

```text
baseline p50 = 12.177705764770508 ms
candidate p50 = 10.767698287963867 ms
absolute delta = -1.410007476806641 ms
relative delta = -11.5786%

baseline mean = 12.848957379659018 ms
candidate mean = 10.823408762613932 ms
```

两边 `hidden_l3`、`hidden_l4` bit-exact。该组是最终性能结论。

### 7.2 异构 token guardrail

```text
baseline p50  = 14.313459396362305 ms  (5 warmup + 30 measured)
candidate p50 = 11.263608932495117 ms  (3 warmup + 5 measured)
descriptive delta = -21.31%
```

candidate 只测 5 次，采样口径不对称，因此这里只作为异构 route correctness 和趋势
证据，不作为主性能准出数字。L3/L4 仍 bit-exact、finite、TP spread=0。

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

## 10. 最终产物索引

```text
/mnt/persist/chensiyu/workspace/moe-opt/l0-l4/final/
├── source/{baseline,candidate}/
├── golden/{repeated,heterogeneous}/
├── baseline/{perf,dfx}/
├── candidate/{perf,dfx}/
├── dfx/{baseline,candidate}/
└── manifests/
```

### DFX 报告

```text
baseline:
  .../final/dfx/baseline/moe_dfx_report.json
  .../final/dfx/baseline/moe_critical_path_report.md
candidate:
  .../final/dfx/candidate/moe_dfx_report.json
  .../final/dfx/candidate/moe_critical_path_report.md
```

### Candidate merged swimlane

```text
rank0/d0/merged_swimlane_20260804_145912.json
rank1/d0/merged_swimlane_20260804_145913.json
rank2/d0/merged_swimlane_20260804_145913.json
rank3/d0/merged_swimlane_20260804_145914.json
rank4/d0/merged_swimlane_20260804_145915.json
rank5/d0/merged_swimlane_20260804_145916.json
rank6/d0/merged_swimlane_20260804_145916.json
rank7/d0/merged_swimlane_20260804_145917.json
```

### Provenance

```text
.../final/manifests/final_summary.json
.../final/manifests/SHA256SUMS
.../final/manifests/files.tsv
```

真机结束后 0162 cards `8–15` 无 MoE container/device process 残留。所有临时脚本
位于 `/mnt/persist/chensiyu/workspace/moe-opt/tmp`；最终目录采用唯一路径和拒绝覆盖
策略，未通过删除重建来更新产物。
