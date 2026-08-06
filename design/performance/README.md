# Performance 性能优化专项

> **2026-08-03 current-source override（优先于下方历史快照）**：当前 attention/Vec
> 与 TP all-reduce stability 源码为
> `pypto-lib stepfun/develop@7099476b7c4f13112b159e237e7a64344803caf0`，
> 配套 pypto 为 `stepfun/develop@defa97c526fec7e8f032dbbfcc39c820add02bf7`。
> A1/B1/B2/C1/C2/C3/C4/D1/D2/G1/H1/I1/I2 已完成；B3 仍在进行。
> Wave5 在 0162 release-qualified：Main N=128×3 均 `123/128` 且 TP spread=0，
> 64K immutable ITL p50 为 `49.796 ms`。下方 2026-07-27 状态、65 ms 分账和
> 旧依赖关系保留为
> 历史分析，不能覆盖 [`task-tracking.md`](task-tracking.md) 的当前状态。
>
> **2026-08-06 L0–L4 focused MoE 当前状态**：调优对象严格裁剪为
> `L0 Full+dense → L1/L2 SWA+dense → L3 SWA+MoE → L4 Full+MoE`，且 L4
> 消费真实 L3 输出。最终将 routed fused gate/up 拆为独立 gate、up 和 activation，
> 使用 `row=16, K=512, N=64`，并将 down 设为 `N=256`；L43/L44 specialization
> 保持原配置。产品实现已合入
> `pypto-lib stepfun/develop@7928a2751930b04c866788a396a7337b62c6d32f`。
> 0162 cards `8–15` 已完成 BS=`1,2,4,7,8,16`、每 sequence 独立 64K 的三轮
> counterbalanced normal campaign；六档 `hidden_l3/hidden_l4` 均 BF16 bit-exact、
> finite、TP spread=0，candidate p50 reduction 分别为
> `0.04/6.629/12.113/3.652/9.229/11.135%`。formal matched-source DFX、
> route-aware publication gate 和最终 all-rank swimlane 尚待完成；证据根目录为
> `/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-formal-act-n64-20260806-v1`。
> 该结论不是 45 层 whole-net 或 L43/L44 发布结论；详见
> [`05-moe-optimization.md`](05-moe-optimization.md)。

> **2026-07-27 当前目标快照**：已完成并保留完成态的优化为 **A1、B1、B2**。
> historical fixed-slot pull C2仅作为回归基线；目标通信架构已改为直接迁移
> V4-Flash expert-lane dispatch push/gather + combine scatter/wait/token reduce。
> 当前没有“旧版与新版同环境完整 serving latency”数据，不把设计预期写成实测加速。

> step3p5 decode 整网性能优化。对照 `origin/main` 上 deepseek **v4-flash** 的 mega-kernel 范式
> (`models/deepseek/v4-flash/decode_fwd.py`)，把当前"慢"的 whole-net 实现改造到位。
>
> **目标**：每个优化点都是**独立可落地的子任务**，便于团队并行认领。所有子任务的
> 状态在 [`task-tracking.md`](task-tracking.md) 里跟踪。

---

## 顶层审计方法

本专项禁止局部shape/name比较。任何“step3p5独有/必须保留”判断，都必须沿完整链路核对：

```text
producer → 数学变换/quant/route-map → transport/window
→ consumer → rounding/reduction/placement → lifetime/reuse/allocator
```

差异必须分类为：能力/算法、数学语义、layout/shape、host/allocator集成、backend/profile workaround。若V4-Flash已有同构能力，layout/shape不同只能称参数化或存储适配，不能称架构差异。执行任务前先核对current source、调用链和工作树diff；旧任务描述、历史设计和probe不能覆盖current source事实。

本次反例作为通用审计提示：INT8+scale、owner max、`BATCH=16` capacity与runtime logical batch、route-weight placement都必须核对到最终consumer和lifetime；这些例子不构成固定shape或实现模板。

## 文档结构

| 文档 | 层级 | 用途 |
|------|------|------|
| [`01-system-design.md`](01-system-design.md) | HLD | 现状瓶颈实证 + mega-kernel 目标架构 + 四条并行主线 + 收益模型 |
| [`02-detailed-design.md`](02-detailed-design.md) | LLD | 每个子任务的 file:line、接口、算法步骤、验证口径、落地边界 |
| [`03-tp-allreduce-algorithm-comparison.md`](03-tp-allreduce-algorithm-comparison.md) | 专项 | TP all-reduce 算法对比（C4 落地依据 + §5 就地修正记录） |
| [`04-attention-optimization.md`](04-attention-optimization.md) | 专项 | attention 单一设计入口：历史实验、负面结果、最终 workload-derived task/tile profile、Full/SWA/online-softmax/out-proj 与 release 边界 |
| [`05-moe-optimization.md`](05-moe-optimization.md) | 专项 | L0–L4 focused MoE：五层/双 hidden 合同、gate/up critical path、combine wait 解释、最终 split tile、golden、0162 A/B/DFX/swimlane |
| [`task-tracking.md`](task-tracking.md) | 跟踪 | 看板式任务跟踪记录（状态 / owner / 更新时间 / 阻塞） |
| [`user_prompt.md`](user_prompt.md) | 提示词 | 复制即用的推进/回归提示词（以 skill + 本目录为单一入口） |

---

## 一图速览：优化点主表

> 详细展开见 [`02-detailed-design.md`](02-detailed-design.md)；实时状态见 [`task-tracking.md`](task-tracking.md)。
> V4-Flash示例shape与standalone probe shape均非产品硬约束；实际window/tensor使用可配置capacity上界，并由模型上界、ownership、对齐和consumer ABI推导。runtime active batch/token才是每次调用的逻辑范围；默认16不定义产品batch。

| ID | 优化点 | Track | 优先级 | 收益 | 依赖 | 工作量 |
|----|--------|-------|--------|------|------|--------|
| **PERF-A1** | whole-net decode baseline + DFX 采集（l2_swimlane/PMU/perf_hints/mem-occupancy） | A 可观测性 | **P0** | 把盲调变有数，回归基线 | 无 | S (~1d) |
| **PERF-B1** | resident 权重池 + opt zero-copy view ABI | B Mega-kernel | **P0** | ✅ 复用 canonical resident pool，为 B2 提供 dynamic slice；相对复制 opt attention buckets，避免约 `0.94 GiB/rank` 额外设备副本 | 无 | M (~3d) |
| **PERF-B2** | 45 层 unroll → 单 `pl.range` 循环 + per-layer `pl.slice` | B Mega-kernel | **P1** | ✅ 主体源码 `31,686→4,772`（约 -84.94%）；MoE loop body `40→1`；N=256 replacement exact | B1 | XL (多周) |
| **PERF-B3** | KV pool `resident` + in-place 更新 | B Mega-kernel | P2 | 省 KV 每步 D2H 往返 | B1 | S (~2d) |
| **PERF-C1** | shared window set + 单调 `moe_epoch` + `WaitCmp.Ge` | C MoE 通信 | **P0** | 复用MoE通信window；arrival/completion按真实DAG闭合，不绑定pull双波 | 无 | M (~1w) |
| **PERF-C2** | 迁移V4-Flash dispatch/combine数据流 | C MoE 通信 | **P1** | expert-lane metadata/push/arrival/gather + scatter/wait/token FP32 reduce | C1 | L (~1-2w) |
| **PERF-C3** | expert-lane SPMD + whole-net调度适配 | C MoE 通信 | P2 | write-disjoint lane ownership、真实远端arrival依赖和42层复用闭环 | C1, C2 | L (~1-2w) |
| **PERF-D1** | 对齐 V4-Flash deferred-norm + INT8 activation/scale producer | D INT8-native | **P1** | 复用V4已有INT8数据流并减少重复norm/quant pass | 无 | M (~1w) |
| **PERF-D2** | 对齐V4-Flash routed expert INT8×INT8/requant/W2 epilogue | D INT8-native | **P1** | 统一INT8 cube、requant、route-weight placement与expert lane输出 | D1 | L (~2w) |
| **PERF-E1** | 按V4-Flash复用data window的LM-head seam | E LM head | P2 | 复用MoE/LM-head data storage并保持独立completion counter | C1 | M (~1w) |
| **PERF-F1** | 对齐V4-Flash dependency/early-resolve调度语义 | F L1/L0 微调 | P2 | 暴露可验证的QR/KV overlap机会 | A1 | S (~2d) |
| **PERF-F2** | 按V4-Flash data tile/pipeline做MTE性能调优 | F L1/L0 微调 | P2 | 依perf_hints消MTE停顿；与control signal 512B隔离 | A1 | S (~3d) |
| **PERF-F3** | 复用V4-Flash deferred-norm/quant producer | F L1/L0 微调 | P2 | 统一norm/amax/scale producer，避免重复fusion架构 | D1 | S (~2d) |
| **PERF-G1** | experts/feature调度轴 + runtime dynamic active batch/token | G 调度轴/动态batch | **P1** | capacity与逻辑batch解耦；attention/MoE/KV不处理inactive rows | 与B2/C2/C3协同 | L (~2w) |
| **PERF-H1** | retained window 清零：host 搬零 → device `aclrtMemset` | H host per-step | **P0** | ✅ 清零 `21.50→2.21 ms`、ITL p50 `85.02→65.55 ms`（−22.9%）、每步 H2D `244.7 MiB→0`；语义等价 | A1 | S (~1d) |
| **PERF-H2** | per-rank 视图重建 hoist 到 `prepare()`（= 跨卡起跑阶梯病根） | H host per-step | P1 | submit 3.49 ms 的大部分；起跑阶梯实测 2.914 ms。**v4-flash 同形状**，属 codegen 通用改进 | H1 | M (~1w) |
| **PERF-H3** | DFX run 第一 barrier 假长条（观测性，非性能） | H host per-step | P2 | 让 swimlane 可信——该假长条曾把 `tp_all_reduce` 误判成 74.1% wall | A1 | S (~2d) |
| **PERF-J1** | L0–L4 routed gate/up stage split + task-grain tuning | J MoE compute | **P0** | ✅ gate/up AIC p50 `≈144→12.7–12.9 µs`；focused p50 `12.1777→10.7677 ms`（-11.58%）；两套 L3/L4 hidden bit-exact | A1, C1–C3, D1–D2, G1, I2 | M |

优先级：**P0** 零/低风险且解锁其它项，先做；**P1** 收益大的主体；**P2** 微调/收尾。
工作量：S ≤ 3d，M ≈ 1w，L ≈ 2w，XL 多周。

---

## 第二维度：优化落在栈的哪一层

Track A–J 是**按 workstream 分工**（谁认领）。但同一个 ITL 数字是被不同**层**吃掉的，
调优手段、度量工具、回归口径都不同层不通用。下表是同一批子任务按**层**重新切一遍。
层级命名沿用 simpler 的 L0–L6 模型（`simpler/docs/hierarchical_level_runtime.md`）与本仓既有用法
（Track F 早已叫「intra-kernel L1/L0 微调」）。

| 层 | 是什么 | 谁在这一层被消耗 | 度量工具 | 子任务 |
|---|---|---|---|---|
| **L3 · host / CPU** | host 进程的 Python + nanobind + mailbox 往返；`Worker(level=3)` 编排、window 生命周期、per-step 参数打包 | **host CPU 时间**、PCIe H2D | `_dispatch_prepared` / `_reset_persistent_domains` 计时探针；`[STRACE]` | **H1 ✅** · **H2** · H3 |
| **L2 · AICPU 调度** | 片上 orchestrator/scheduler：task 图形状、依赖链深度、`early_dispatch`、每 task 的 `block_num` | **task 派发/完成延迟**、核空转 | l2_swimlane（`deps.json` + 记录）、scope_stats | F1 · （新）**任务图瘦身**：1086-hop 关键路径、878/1542 个 `block_num=1` task |
| **cross-chip · 通信** | 跨卡 collective 与 rendezvous：`tp_all_reduce`、EP dispatch/combine、signal window 与 epoch | **等待时间**（非带宽） | swimlane 上 `*_wait` / `tp_all_reduce` 时长分布；per-rank 对比 | C1 ✅ · C2 ✅ · C3 ✅ · C4 ✅ · E1 · （新）**combine_wait 13.4 ms** |
| **L1 · kernel 内数据流** | 单 task 内的 MTE/tiling/L1-UB 驻留、量化链、norm 融合 | **MTE 停顿**、cube/vec 利用率 | `perf_hints.log`（PH001 tile 粒度）、PMU exec counter | D1 ✅ · D2 ✅ · F2 · F3 |
| **L0 · 核内流水** | 一个 AICore 内 cube/vector/MTE 的 pipeline 重叠 | **单 task 时长** | l0_swimlane（`simpler_setup.tools.l0_swimlane`） | （暂无立项；`expert_gate_up_aiv` 与 `aic` 同耗时是候选线索） |
| **结构 / codegen** | program 形态本身：层展开 vs `pl.range`、权重 resident、调度轴、动态 batch | 以上各层的**上界** | 源码体量、编译产物、IR | B1 ✅ · B2 ✅ · B3 · G1 ✅ |
| **可观测性** | 让上面每一层可测且可信 | —— | —— | A1 ✅ · H3 |
| **MoE compute** | routed expert 的 expert/feature/tile 调度与 W8A8 cube/vec pipeline | **L0–L4 focused graph 的 L3/L4 gate/up/down** | focused clean A/B + all-rank DFX/swimlane + memory | **J1 ✅：gate/up split，row16/K512/N64，down N256** |

### 当前 ITL 65 ms 按层分账（ctx=64k / bs=16，实测）

| 层 | 时间 | 占比 | 下一步 |
|---|---|---|---|
| **L2 + L1 + L0**（device 执行，`drain`） | **59.15 ms** | **91%** | 其中 `combine_wait` 13.4 ms 属 cross-chip；余下是真算力 |
| L3 · host | 3.49 ms（`submit`）+ 2.20 ms（`clear`） | 8.8% | H2 收 submit；clear 已收完 |

> **H1 之前**这张表是 host 24.9 ms / device 59.9 ms（host 占 29%）。H1 落地后 host 压到 5.7 ms，
> **device 执行首次成为主导项** —— 这也意味着后续重心从 L3 移到 cross-chip 与 L2/L1。


---

## 依赖图

```
A1 (baseline) ─── 独立，最先做 ; 解锁 F2
B1 (resident pool 的 zero-copy view ABI) ──► B2 (pl.range 主体) ──► B3 核验
C1 (shared windows + epoch) ──► C2 (V4-Flash数据流迁移) ──► C3 (expert-lane whole-net适配)
C1 ────────────────────────────────────────────────────────────► E1
D1 (gate quant) ──► D2 (expert INT8) ; F3 随 D1
F1 独立 ; F2 需 A1
```

## 四条可并行主线（建议按 owner 分配）

| 主线 | 子任务链 | 说明 |
|------|---------|------|
| **① 结构线** | A1 → B1 → B2/B3 | current B2 已在保留 per-layer window 的前提下完成 |
| **② 通信线** | C1 → C2 → C3；C1 → E1 | 直接迁移V4-Flash通信数据流；historical pull只作回归对照 |
| **③ 数值线** | D1 → D2 → F3 | INT8-native / HBM，完全独立 |
| **④ 微调线** | F1、E1、F2 | 重叠 & L1/L0，A1/C1 出结果后启动 |

**当前收口**：A1/B1/B2 已交付；B3、C1/C2/C3、D1/D2、E1、
F1/F2/F3、G1仍是后续工作。historical pull C2不再计为目标完成态；不得把表中设计收益当作已测性能提升。

**历史推进顺序**曾把 C1 视为 B2 前置；实际 current source path 通过保留
per-layer communication stack，先完成了 **A1 → B1 → B2**，C1 留作独立
窗口/HBM 优化。后续 agent 不应再因 C1 未完成而把 B2 标回 blocked。

---

## 验证标准（所有子任务通用）

**当前 release 的两个 gate 必须分开**：

1. vanilla raw alignment：0162 canonical-only 发布镜像 N=256 为
   `240/256=93.75%`，低于历史 `>=95%` raw gate，**未通过**；
2. replacement equivalence：canonical-only 清理前后 token/hidden
   `256/256` exact，hidden `max_abs_diff=0`、TP spread `0.0`，**通过**。

驱动：`pypto-lib/tests/step3p5/ci/{LIVE_PRECISION_AB.md, run_live_precision_ab.sh}`；
两个 gate 不得混写。

> **stall / deadlock 是独立于精度的 liveness 检查**（不是精度判据）：`RUN_CLEAN` +
> 隔离探针 `_probe_barrier_scale.py` 判定是否 hang。任何"跑通/stall"结论用它，"精度不回退"用上面的多步 L3。

---

## 相关

- **回归 runbook（做完每个子任务按它回归）**：`.claude/skills/pypto-perf-regression/SKILL.md`
- 整网集成设计：[`../whole-net/README.md`](../whole-net/README.md)
- vLLM 共驻：[`../vllm-pypto/README.md`](../vllm-pypto/README.md)
- pypto kernel 坑：`pypto-lib/docs/known-pypto-pitfalls.md`
- perf 调优 playbook：`pypto-lib/docs/performance-tuning.md`
- 参考实现：`origin/main:models/deepseek/v4-flash/`（`git show` 读取）
