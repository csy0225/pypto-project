# 01 — System Design (HLD)：step3p5 decode 性能优化

> 上层设计：**为什么慢**、**目标架构长什么样**、**怎么拆成可并行的主线**。
> file:line、接口签名、算法步骤在 [`02-detailed-design.md`](02-detailed-design.md)。

---

## 1. 背景 & 目标

step3p5 整网 decode 目前以 `whole_decode_faithful_real`（`pypto-lib/models/step3p5/decode_layer.py`，
**31,636 行**）跑通并对齐精度（argmax==303），但**性能差**。本专项把它对照
`origin/main:models/deepseek/v4-flash/decode_fwd.py` 的 mega-kernel 范式重构。

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
| **层展开** | 45 层**完全 unroll**：`_gen_faithful_real.py` 逐层 emit `self.*_chip_orch(...)`（3 dense 直排 + 42 MoE 在 Python for 里逐层 emit），产出 31,636 行 | 单 `@pl.jit`：首/尾层显式，中间层 `pl.range(L)` 循环，权重 `pl.slice` 按 dynamic scalar 层号切 |
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
3. **`pl.range` 循环体**（PERF-B2）：靠上面两者，把 31,636 行折成数百行。

---

## 4. 四条可并行主线

| 主线 | 子任务 | 前置 | 可独立启动？ |
|------|--------|------|-------------|
| **① 结构线** | A1 → C1 → B1 → B2 → B3 | — | ✅ A1/C1/B1 均可立即起 |
| **② 通信线** | C2 → C3 | C1 | 🔶 C1 落地后 |
| **③ 数值线** | D1 → D2 → F3 | — | ✅ 完全独立于 ①② |
| **④ 微调线** | F1、E1、F2 | A1（F2）、C1（E1） | 🔶 baseline/窗口出结果后 |

**关键路径**：`C1`（既修 stall 又解锁 B2）与 `A1`（解锁一切调优）是全局瓶颈解锁点，
应最先并行推进。`D` 线（INT8-native）与结构线**零耦合**，可由独立 owner 全程并行。

---

## 5. 收益模型（定性，A1 出数后补定量）

| 维度 | 现状 | 优化后预期 | 主要贡献项 |
|------|------|-----------|-----------|
| 编译期 / IR 体量 | 31,636 行、~766MB comm domain | 数百行、8 窗口 | B2 + C1 |
| 多卡稳定性 | push dispatch 随机 stall | RUN_CLEAN 稳定 | C1 + C2 |
| HBM / rank | ~47GB（BF16-dequant） | ~6GB（INT8-native） | D1 + D2 |
| 单步延迟 | 待 A1 量化 | peer fan-out + 重叠 + INT8 cube | C3 + F1 + D2 |
| 数据搬运 | KV/权重每 dispatch | 上传一次常驻 | B1 + B3 |

---

## 6. 风险 & 约束

- **单 `@pl.program` 硬约束**：用户口径禁止 multi-program（v4-flash 靠 multi-program 规避
  跨层窗口别名；step3p5 必须在**单 program 内**用 epoch 方案解决 → C1 是硬前置）。
- **B2 是 XL 且高风险**：层型交错（full/swa × dense/moe）比 v4-flash 规整层型复杂，需按层型分桶循环。
- **精度不可退**：每步落地都过金标准（argmax==303），INT8-native（D 线）尤其要逐层 detail 对齐。
- **substrate 漂移**：5 仓 + 2 分支 + 2 机器 + CANN 版本，落地前 pin 好组件（见 CLAUDE.md 版本表）。

---

## 相关

- LLD（每子任务展开）：[`02-detailed-design.md`](02-detailed-design.md)
- 任务跟踪：[`task-tracking.md`](task-tracking.md)
- 整网 HLD：[`../whole-net/01-system-design.md`](../whole-net/01-system-design.md)
