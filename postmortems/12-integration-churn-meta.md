# 专项：集成反复推翻（meta）—— 为什么"之前 ready 的"又被推翻重做

| 字段 | 值 |
|------|----|
| **子系统** | process / verification-bar（meta，跨多个事故） |
| **error signature** | 无单一 error；签名 = "上个 session 判 ready，下个 session 推翻重来" |
| **首次出现** | 2026-07-13（归纳记录；案例横跨 2026-06 ~ 2026-07） |
| **状态** | 🟡 缓解（对策已落地，但 BF16→INT8-native 迁移 + DeepSeek 对齐仍在进行） |
| **相关 skill / doc** | [`06-gate-topk-deadlock.md`](06-gate-topk-deadlock.md)、[`11-8001-bridge-live-ops.md`](11-8001-bridge-live-ops.md)、`pypto-lib/docs/known-pypto-pitfalls.md`；always-loaded 版见 memory `feedback_integration_churn_root_causes` |

## 1. 背景（Background）

2026-07-13 记录。用户观察：每次集成都重复遇到同样的问题，之前判定 ready，后续 session 又推翻结果重来。本文不是某一个 error 的事故复盘，而是**归纳这一类反复 churn 的根因与对策** —— 让后续 session 不再把临时结论当定论、不再在弱验证口径上宣布 ready、不再建在临时地基上堆集成。

涉及的事故横跨 2026-06 ~ 2026-07，案例在 §2 列。always-loaded 版（memory）见 `feedback_integration_churn_root_causes`。

## 2. 现象（Symptom）

同一件事反复 解决 → 推翻 → 重做。下表每行都是一个"之前判 ready、后续被推翻"的具体案例：

- **Blocker B**：先定"IPC-VA 冲突"（写进 doc 传了几个 session）→ device 证伪 = gate_topk mrgsort 挂死（详见 [`06-gate-topk-deadlock.md`](06-gate-topk-deadlock.md)）。
- **G5b 根因**：先"SWA multi-entry kernel bug" → `--golden-fill-batch` 证伪 = seq_len=0 pad 行污染。
- **gap-5**：partial-tile → 证伪；scale-tail-zero → 证伪；quant-scope → 证伪（3 次）。
- **KV bridge**："纯 reshape" → 更正"非纯，3 障碍"。
- **HBM**："24G+47G=OOM" → 更正"64GB/卡，误判"。
- **head-gate**：bypass ↔ worker gate_r ↔ on-device 来回（详见 [`11-8001-bridge-live-ops.md`](11-8001-bridge-live-ops.md) §head-gate）。

## 3. 根因（Root Cause）

设计 + 流程，约各一半。

### 根因 1（流程，最主要）：「ready」验证口径太弱、没强制阶梯

真出口 = **live-token-exact-device**；途中有 compile-OK / offline / synthetic / single-config / 单卡 多个更弱的 bar。在**较弱 bar 宣布 ready** → 下个 session 在更强 bar 上推翻。**"声明 bar" 与 "真 bar" 的每个 gap 都是未来推翻点。**

本 session 亲历：offline golden 按 kernel 布局注入 KV pass → live 立即推翻。

### 根因 2（流程）：根因靠"看着对的假设"而非"决定性隔离实验"

假设被写进 doc / memory **当事实传递** → 后人基于错假设做 → 直到有人补一个**能证伪**它的控制实验才翻案。

对策：**声明 root cause 前，先设计能证伪它的实验**（fill-batch / dispatch-cut bisect / golden 对拍）。

### 根因 3（设计）：建在"明知临时"的地基上

BF16-dequant 是 bring-up 捷径、真目标 INT8-native → BF16 上所有**精度结论都是暂定、注定被 INT8-native 推翻**（今日 L17 attn 0.25 即此）。

### 根因 4（设计）：每个 step3p5-vs-DeepSeek divergence 都是潜在重复 bug

seq_len=0 pad = 静态 BATCH=16 padding 背离 DeepSeek 动态 T；DeepSeek 无 pad 行故永不踩。每个"和 DeepSeek 不一样"都埋雷。

### 根因 5（环境）：底座漂移

5 仓 + driver/CANN + 2 分支（stepfun/develop vs n1-live）+ 2 机器（0234 down → 0162）。升级栈曾 drop Phase-16 SDMA-OFF patch → 重踩 507899；rebase 引入 SplitIncoreOrch 回归。一组合上 ready 不迁移到另一组合 → 每 session re-derive。

## 4. 如何解决（Fix）

可落地的对策（已开始执行）：

- doc / memory **区分「假设」vs「隔离证明的事实」**；错的**大声 CORRECTED / SUPERSEDED** 撤回（本仓 postmortems 模板里 §5「走过的弯路」就是这条的落点）。
- **声明 root cause 前先跑证伪实验**。
- **"ready" 只认 live-token-exact-device**；compile / offline / synthetic 一律标 `provisional`。
- **别在 BF16-dequant 上堆 live 集成** → 直接 INT8-native（消临时地基）。
- **能对齐 DeepSeek 就对齐**；必须不一样的写清"为什么 + 验证口径"。
- **pin 单一可复现底座**（分支 / 机器 / 版本组合）。

## 5. 走过的弯路（Detours / What We Got Wrong）

本节是本文的实质 —— 列每个被推翻的"ready"声明，及**证伪它的实验**。

- ❌ **Blocker B = "IPC-VA 冲突"**（写进 doc 传了几个 session）→ 证伪：device 上 MoE comm window `0x12c041...` 低于 IPC pool `0x12c1c0...`，无 overlap；IPC map 48-key aligned / in-pool。真因 = gate_topk mrgsort format2-on-unsorted 挂死（[`06-gate-topk-deadlock.md`](06-gate-topk-deadlock.md)）。证伪实验 = device-level VA 范围比对 + dispatch-cut bisect。
- ❌ **G5b = "SWA multi-entry kernel bug"** → 证伪：`--golden-fill-batch`（16 行全 active）L0-L4 全 pass 1.0（full / swa / MoE）；L1-alone golden input 也 1.0。真因 = seq_len=0 pad 行产生 NaN 污染 active 行。证伪实验 = fill-batch 控制组。
- ❌ **gap-5 = "partial-tile 处理 bug"** → 证伪：控制组 int8-copy PASS。真因 = `cast→int8→cube` 误编译（`infer_tile_memory_space_pass.cpp:55-56` INT8 cube A-operand fractal=32 layout 未推导）。
- ❌ **gap-5 = "scale-tail-zero"** → 证伪：tail 不零也复现。
- ❌ **gap-5 = "quant-scope"** → 证伪：scope 调整无效。gap-5 共 3 次证伪。
- ❌ **KV bridge = "纯 reshape"** → 更正：3 障碍（MAX_SEQ sizing 667GB blowup / k_cache dim 是 KV_CACHE_ROWS_DYN baked 非 MAX_SEQ / 整池 feed 15GB dummy OOM）+ per-layer feed 路径。
- ❌ **HBM = "24G+47G=OOM"** → 更正：64GB/卡，TP=8 sharded 后 vLLM 3GB + pypto 6GB + KV ≈10GB/card ≪ 64GB。误判来源 = 把聚合非分片当分片。
- ❌ **head-gate 反复方向**：bypass ↔ worker gate_r ↔ on-device 来回。中段误判"KV 来源（self-KV vs 真-KV-IPC）是 live 乱码根因" → 证伪：真-KV-IPC 打通后 self-KV 和 真-KV-IPC **都乱码**。真因 = head-gate `matmul_acc` 小 N=16 丢 K 维累加（[`11-8001-bridge-live-ops.md`](11-8001-bridge-live-ops.md) §head-gate 终局）。
- ❌ **L2 attention residual 路径**（M4 determinism）：先怀疑 hidden-state pingpong / fence-gap / attention-residual → 证伪：device-disproven。真因 = dense L2 REUSED L1 comm/signal windows → premature Ge(1) barrier via leftover AtomicAdd signal → racy attention output。修 = distinct `l2_*` windows。

## 6. 如何避免（Prevention）

- **声明 root cause 前先跑证伪实验**（fill-batch / dispatch-cut bisect / golden 对拍 / VA 范围比对）。假设写进 doc 时标"假设"，事实标"隔离证明"。
- **"ready" 只认 live-token-exact-device**。compile / offline / synthetic / single-config / 单卡一律标 `provisional`。声明 bar 与真 bar 的每个 gap 都是未来推翻点。
- **别在 BF16-dequant 上堆 live 集成** → 直接 INT8-native（消临时地基）。
- **能对齐 DeepSeek 就对齐**（addr-align / padding / shape / dtype / layout / static-vs-dynamic T）；必须不一样的写清"为什么 + 验证口径"。每个 step3p5-vs-DeepSeek divergence 是潜在重复 bug —— 问"为什么 DeepSeek 不爆"，别 deep-dive codegen。
- **pin 单一可复现底座**（分支 / 机器 / 版本组合）；底座变更时 re-derive ready 状态，别默认迁移。
- **早期识别信号**：某个"ready"结论只在 compile / offline / synthetic / 单卡 / 单配置下成立 → 立刻标 `provisional`，别当定论传。
- 相关约束落点：本仓 [`06-gate-topk-deadlock.md`](06-gate-topk-deadlock.md)、[`11-8001-bridge-live-ops.md`](11-8001-bridge-live-ops.md)、`pypto-lib/docs/known-pypto-pitfalls.md`、`pypto-lib/docs/dev-workflow-gotchas.md`、memory `feedback_integration_churn_root_causes` + `feedback_align_deepseek_architecture_first`。
