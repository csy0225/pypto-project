# 01 — System Design (HLD)：step3p5 decode 性能优化

> **2026-08-05 current-source override（优先于下方历史正文）**：Attention/Vec
> 收口源码为
> `pypto-lib stepfun/develop@91c7f46ee949045e2fce807276412b48d8121763`，
> 配套 PyPTO（动态 SPMD + immutable swimlane）为
> `pypto stepfun/develop@8e92b46808f9f7c09b6431ad4691503f09c12ee5`。
> 当前实现采用 workload-derived logical tasks + runtime wave mapping，Full SV 合并
> segment-local recurrence，Full/SWA out-proj cast 默认融合。Wave5 immutable
> audit/smoke/Main+MTP compile、Main N=128×3、Main batch16、MTP batch1/batch16×2、
> 64K/batch16 ITL/DFX 已通过；Main N=128 三轮均 `123/128` 且 TP spread=0，
> 仍是最后一个 **0162 release-qualified** 镜像。R1 已废弃，R2 build 已暂停，
> 尚无新 digest/ITL/DFX。下方较早的主线表和收益表是
> 设计历史；I1/I2 当前状态以 [`task-tracking.md`](task-tracking.md) 和
> [`04-attention-optimization.md`](04-attention-optimization.md) §12 为准；
> attention 当前 task/tile 设计见同文 §13。

> **2026-07-26 交付收益覆盖**：本专项当前已实际落地并完成 device
> regression 的优化为 **B1、B2、C2**，另有 A1 可观测性基线。本文 §5
> 的“预期”不能代替已测结果；已落地项的前后对比见下表，性能数字只在有
> 实测证据时标为实测，其余明确标为结构/容量收益。

> 上层设计：**为什么慢**、**目标架构长什么样**、**怎么拆成可并行的主线**。
> file:line、接口签名、算法步骤在 [`02-detailed-design.md`](02-detailed-design.md)。

---

## 1. 背景 & 目标

本专项的历史起点是 `whole_decode_faithful_real`
（historical `pypto-lib/models/step3p5/decode_layer.py`，**31,686 行**）的 45 层展开实现；
current source path 已切到 `models/step3p5/decode_fwd.py:whole_decode_step3p5`
（4,772 行）。下文 §2–§4 保留历史诊断和目标架构，current 实际交付及收益
以 §5 为准。

- **模型规模**（`config.py`）：`HIDDEN=4096`、`NUM_HIDDEN_LAYERS=45`、`VOCAB=128896`、
  `MOE_NUM_EXPERTS=288`（`TOP_K=8`，per-rank `LOCAL=36`）、`MOE_INTERMEDIATE=1280`、
  `TP=EP=8`、`HEAD_DIM=128`。45 层 = 3 dense + 42 MoE，full/swa attention 交错。
- **成功判据**：单步 decode 延迟下降（A1 建 baseline 后量化）；whole-net stall 消除；
  精度不回退（金标准 argmax==303）；HBM 占用下降（利于与 vLLM 共驻）。

---

## 2. 现状瓶颈实证（诊断）

> 证据来自 `feat/whole-net-n1-fusion` HEAD 的实际代码走查，file:line 见 LLD。

### 2.1 结构层面

| 瓶颈 | 现状 | 参考做法（v4-flash） |
|------|------|---------------------|
| **层展开** | 45 层**完全 unroll**：`_gen_faithful_real.py` 逐层 emit `self.*_chip_orch(...)`（3 dense 直排 + 42 MoE 在 Python for 里逐层 emit），产出约 31,686 行 | 单 `@pl.jit`：首/尾层显式，中间层 `pl.range(L)` 循环，权重 `pl.slice` 按 dynamic scalar 层号切 |
| **comm 窗口** | 每 MoE 层 emit 16 个 `_L{pos}` 窗口 → 42 层 ≈ **672 窗口合入一个 ~766MB comm domain** | `l3_decode_fwd` **一次性**分配 8 窗口，靠 `moe_epoch` 计数复用跨 43 层 |
| **权重残留** | 已经**做对**：经 IPC 一次性 H2D 常驻（`whole_decode_holder`），`rt.run` 复用 args。但布局非 leading-dim stacked，未打 `resident="stacked"` 标志 | `[N_RANKS, L*dim, ...]` stacked + `spec.resident="stacked"`，运行时上传一次跨 dispatch 复用 |

### 2.2 通信协议层面

- MoE dispatch/combine 都是 **push**（`remote_store`/`tensor.put`），peer 循环全是
  `pl.range(N_RANKS)` 顺序 barrier；**无 epoch 概念**（`grep epoch` 无命中）。
- 已定位：push dispatch（`_dispatch_push`, func_id 28）是 8 卡随机 stall 的根因
  （跨 die MTE3 写完成竞争，见 memory `n1_a2_primitive_exists_not_missing`）。

### 2.3 数值 / HBM 层面

- MoE 走 **BF16-dequant**（临时路径），weight IPC pool **~47GB/rank** → 与 vLLM 共驻 OOM。
- 参考 v4-flash 是 **INT8-native W8A8**：dispatch-side activation 量化 + INT8×INT8 cube + requant。

### 2.4 可观测性层面

- whole-net **无 perf 埋点**：`docs/step3p5` / `docs/performance-tuning.md` 无 whole-net 延迟数据，
  `perf_hints.log` / `memory_after_AllocateMemoryAddr.txt` 未针对该 program 采集 → **盲调**。

### 2.5 已经做对、别回退

1. 权重经 IPC 一次性 H2D 常驻，`rt.run` 复用 args；
2. LM head 已 inline 进同一 program（非 decoupled，但可用）；
3. KV = BF16 1 head/rank；
4. MoE/dense matmul 已用 `pl.pipeline` K-loop；
5. per-layer 独立 hidden buffer（`h_moe_L{pos}` / `resid_hold_L{pos}`）修了旧 2-buffer ping-pong 的 WAR/WAW 竞争。

---

## 3. Mega-kernel 目标架构

参照 `v4-flash/decode_fwd.py`，目标形态：

```
decode_fwd  (单 @pl.jit, auto_scope=False)
  ├─ 首层显式 (attention + moe scope)
  ├─ pl.range(中间层)                      ← 循环体，不 unroll
  │    ├─ layer_idx = 动态 scalar
  │    ├─ w_l = pl.slice(w_stacked, [dim], [layer_idx*dim, 0])   ← 每层切权重
  │    ├─ with pl.scope(): attention(...)
  │    └─ with pl.scope(): moe(..., layer_idx, nt, my_rank, moe_epoch)  ← 单调 epoch
  ├─ 尾层显式
  └─ hc/head + rms_norm → logits

l3_decode_fwd (@pl.jit.host)
  ├─ 一次性 pld.alloc_window_buffer × 8       ← 全 43 层共享
  ├─ for r in range(world): decode_fwd(..., device=r)   ← 权重 resident="stacked"
  └─ LM head 4 段 decoupled worker（复用 recv_x_buf）
```

三个支柱缺一不可：
1. **共享窗口 + epoch**（PERF-C1）：`moe_epoch` 单调递增，wait 用 `WaitCmp.Ge` 对
   AtomicAdd 计数器 → 旧 epoch 的 notify 永不误触发新 epoch → **一套窗口跨全部层安全 drain**。
   这是 unroll → `pl.range` 的**前置**（否则 SSA 窗口复用会撞 RAW-only-v1 非别名约束）。
2. **stacked + resident 权重**（PERF-B1）：`[N_RANKS, L*dim, ...]` 上传一次；层内 `pl.slice`。
3. **`pl.range` 循环体**（PERF-B2）：原设计希望同时依赖 C1/B1；实际
   current source path 保留 per-layer communication stack，只依赖 B1 的
   zero-copy view ABI 即完成了 loop-form replacement。

---

## 4. 四条可并行主线

| 主线 | 子任务 | 前置 | 可独立启动？ |
|------|--------|------|-------------|
| **① 结构线** | A1 → B1 → B2 → B3 | — | ✅ A1/B1/B2 已完成，B3 待独立核验 |
| **② 通信线** | C2；C1 → C3/E1 | C3/E1 依赖 C1 | ✅ C2 已完成；C1 独立待办 |
| **③ 数值线** | D1 → D2 → F3 | — | ✅ 完全独立于 ①② |
| **④ 微调线** | F1、E1、F2 | A1（F2）、C1（E1） | 🔶 baseline/窗口出结果后 |

**current 路径校正**：历史设计把 C1 当作 B2 的硬前置，但 current B2 已通过
保留 per-layer communication stack 独立完成。C1 现在只负责窗口复用/HBM
收敛及其后续 C3/E1，不得再据此把 B2 标为未完成。`D` 线仍与结构线独立。

---

## 5. 已落地优化与前后对比（2026-07-26）

| 项目 | 改造前 | 当前实现 | 已确认收益/证据 |
|------|--------|----------|----------------|
| **A1 可观测性** | whole-net 缺统一的多步 token/hidden、TP spread、per-layer dump 证据，性能只能盲调 | holder/harness 支持 256-step report、per-layer hidden、hidden finite、TP spread 和边界 step 检查 | 回归从“跑通”变成可审计数据；当前 N=256 检查 step127/128/255，TP spread `0.0` |
| **B1 resident 权重池 + opt zero-copy view ABI** | 0724 baseline 已经是每 rank 一次 IPC import、`prepare()` 内常驻并跨 step 复用；但 opt 缺少从 canonical FULL/SWA 桶构造 MoE 专用连续 view 的 ABI，若另建 opt 桶会产生额外设备副本 | 保留原 consolidated IPC pool；`Wsub()` 对 FULL `1:11`、SWA `2:32` 做 outermost contiguous slice，再以 `StackedDeviceTensor` 跨 rank 绑定，整个过程不 materialize 新权重 | **结构/容量收益**：B2 可直接消费 dynamic leading-dim view；相对复制 10 个 FULL + 30 个 SWA attention buckets，按当前 shape 估算避免约 `965 MiB≈0.94 GiB/rank` 额外设备副本。0724 baseline 本来就无 per-step 全量权重 H2D，因此不再虚写“24 GiB/rank/step H2D 消除” |
| **B2 45 层 loop-form** | 历史 faithful whole-net `decode_layer.py` `31,686` 行，层结构展开，MoE 主体重复 40 份 | canonical `decode_fwd.py` 以一个 `whole_chip_orch` 承载 L0、L1/L2 `pl.range(2)`、L3-L42 MoE `pl.range(40)`，L43/L44 保留必要 specialization；当前文件 `4,772` 行 | 主体源码体量 `31,686→4,772`，约 **84.94% 减少**；MoE 主体 `40→1` 个 runtime loop body；N=256 canonical-only 清理前后 token/hidden 均 `256/256` exact，hidden `max_abs_diff=0`，TP spread `0.0`。未单独证明 latency 提升 |
| **C2 dispatch/combine（历史 pull 基线；current 为 V4-Flash expert-lane push/gather/scatter）** | dispatch 由 source rank push/remote-store 到 peer，存在跨 die 写完成竞争和随机 stall 风险 | fixed-slot 对齐 `moe.py`：目标 rank 按对称槽位 `remote_load` pull；combine 也按固定槽 pull back | 消除原 push dispatch 的随机 stall 路径；0162 当前 N=256 无 stall、TP spread `0.0`。数据量不变，收益是 liveness/可重复性，不宣称带宽减少 |

### 5.1 当前可量化的收益边界

| 指标 | 改造前/历史基线 | 改造后/当前结果 | 口径 |
|------|----------------|----------------|------|
| 主体源码体量 | `31,686` lines (`3af13f4f` historical `models/step3p5/decode_layer.py`) | `4,772` lines (`53eb7212` `models/step3p5/decode_fwd.py`) | 静态结构指标，约 `84.94%` 降低；不是编译时延直接测量 |
| MoE loop 主体 | 40 个物理 MoE layer sites | 1 个 `pl.range(40)` loop body + 2 个尾部 specializations | 降低 IR/调度图重复；没有把“IR 行数”冒充 compiler wall-clock |
| loop-form attention bucket 绑定 | 若为 loop-form 的 10 FULL + 30 SWA bucket 另做 materialization，会重复约 `965 MiB/rank` 权重 | 直接从 canonical resident pool 建 `Wsub()` zero-copy view | 约 `0.94 GiB/rank` 避免的额外设备副本是 shape 推导值；不是 wall-clock 或实际 H2D 采样 |
| MoE communication window | 历史设计约 `766 MB` per-layer window domain | **当前 B2/C2 source path 仍是 per-layer stack**；C1 尚未交付 | 不能把 C1 设计目标 `766 MB→十几 MB` 写成当前收益 |
| vanilla raw alignment | current canonical-only 与清理前 canonical 各 `240/256=93.75%` | 两者相同 | raw `>=95%` 未通过；说明差异不是 B2 或兼容入口清理引入 |
| replacement equivalence | — | canonical-only 清理前后 token `256/256` exact；hidden `256/256` exact；`max_abs_diff=0` | B2 replacement 与 canonical-only cleanup regression 均 PASS |
| 256-step warm runtime | 历史 loop-form artifact warm mean `0.8057s/step`；baseline `0.7363s/step` | 最终镜像 canonical warm mean `0.8088s/step` | harness `run_sec`，不是完整 serving ITL；未与历史 31k-line implementation 做同环境 A/B，不能据此宣称 B2 加速 |

### 5.2 未完成优化，不得提前计入收益

- **C1 单 window + `moe_epoch` + `WaitCmp.Ge`**：当前 source path 仍按
  per-layer stack 传 window；预计的 `~766 MB→十几 MB` 是设计目标，不是本次
  B2 release 的实测收益。
- **D1/D2 INT8-native routed expert**：当前 source path 不能把设计中的
  `47.6 GiB/rank→~24 GiB/rank` 写成已交付收益。
- **C3 peer fan-out、E1 LM-head overlap、F1/F2/G1**：尚未有新的 device
  latency/PMU 对比，保留为待办。

## 6. 收益模型（已区分实测、结构收益与设计目标）

| 维度 | 现状 | 优化后预期 | 主要贡献项 |
|------|------|-----------|-----------|
| 编译期 / IR 体量 | 历史 `31,686` 行主体 | 当前 canonical Main `4,772` 行；C1 窗口仍未收敛 | B2 已落地；C1 未完成 |
| 多卡稳定性 | push dispatch 有随机 stall 风险 | current pull path，0162 N=256 无 stall | C2 已落地；不把 C1 预期写入当前结果 |
| HBM / rank | canonical resident pool 已存在；若为 opt 复制 attention buckets 会再占约 `0.94 GiB/rank` | B1 用 zero-copy view 避免该增量；C1/D2 的大项仍未交付 | B1 已落地；C1、D1/D2 未完成 |
| 单步延迟 | 历史数据不足 | 最终镜像 canonical harness warm mean `0.8088s/step`，未有同环境旧 31k-line A/B | 仅记录，不宣称加速 |
| 数据搬运 | 权重非 stacked 管理 | 一次 IPC resident + `StackedDeviceTensor` sub-view | B1 已落地；KV B3 尚未单独闭环 |

---

## 6. 风险 & 约束

- **单 `@pl.program` 硬约束**：用户口径禁止 multi-program。current B2 通过
  保留 per-layer window 避开了跨层窗口复用问题；若后续要把 42 套窗口收敛成
  一套，仍必须实现 C1 epoch/`WaitCmp.Ge` 协议。
- **B2 是 XL 且高风险**：层型交错（full/swa × dense/moe）比 v4-flash 规整层型复杂，需按层型分桶循环。
- **精度不可退**：每步落地都过金标准（argmax==303），INT8-native（D 线）尤其要逐层 detail 对齐。
- **substrate 漂移**：5 仓 + 2 分支 + 2 机器 + CANN 版本，落地前 pin 好组件（见 CLAUDE.md 版本表）。

---

## 相关

- LLD（每子任务展开）：[`02-detailed-design.md`](02-detailed-design.md)
- 任务跟踪：[`task-tracking.md`](task-tracking.md)
- 整网 HLD：[`../whole-net/01-system-design.md`](../whole-net/01-system-design.md)
