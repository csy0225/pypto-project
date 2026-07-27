# Performance 性能优化专项

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

优先级：**P0** 零/低风险且解锁其它项，先做；**P1** 收益大的主体；**P2** 微调/收尾。
工作量：S ≤ 3d，M ≈ 1w，L ≈ 2w，XL 多周。

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

**历史推进顺序**曾把 C1 视为 B2 前置；实际 current release 通过保留
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
