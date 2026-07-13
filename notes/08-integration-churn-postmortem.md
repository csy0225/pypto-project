# 集成反复推翻复盘：为什么"之前 ready 的"又被推翻重做

> 2026-07-13 记录。用户观察：每次集成都重复遇到同样的问题，之前判定 ready，
> 后续 session 又推翻结果重来。本文归纳根因与对策。always-loaded 版见 memory
> `feedback_integration_churn_root_causes`。

## 现象（同一件事反复 解决→推翻→重做）
- **Blocker B**：先定"IPC-VA 冲突"（写进 doc 传了几个 session）→ device 证伪 = gate_topk mrgsort 挂死。
- **G5b 根因**：先"SWA multi-entry kernel bug" → `--golden-fill-batch` 证伪 = seq_len=0 pad 行污染。
- **gap-5**：partial-tile → 证伪；scale-tail-zero → 证伪；quant-scope → 证伪（3 次）。
- **KV bridge**："纯 reshape" → 更正"非纯，3 障碍"。**HBM**："24G+47G=OOM" → 更正"64GB/卡，误判"。
- **head-gate**：bypass ↔ worker gate_r ↔ on-device 来回。

## 根因（设计 + 流程，约各一半）
1. **"ready" 验证口径太弱、没强制阶梯（流程，最主要）**。真出口 = **live-token-exact-device**；
   途中有 compile-OK / offline / synthetic / single-config / 单卡 多个更弱的 bar。
   在**较弱 bar 宣布 ready** → 下个 session 在更强 bar 上推翻。**"声明 bar" 与 "真 bar" 的每个 gap 都是未来推翻点。**
   （本 session 亲历：offline golden 按 kernel 布局注入 KV pass → live 立即推翻。）
2. **根因靠"看着对的假设"而非"决定性隔离实验"（流程）**。假设被写进 doc/memory **当事实传递**
   → 后人基于错假设做 → 直到有人补一个**能证伪**它的控制实验才翻案。
   对策：**声明 root cause 前，先设计能证伪它的实验**（fill-batch / dispatch-cut bisect / golden 对拍）。
3. **建在"明知临时"的地基上（设计）**。BF16-dequant 是 bring-up 捷径、真目标 INT8-native
   → BF16 上所有**精度结论都是暂定、注定被 INT8-native 推翻**（今日 L17 attn 0.25 即此）。
4. **每个 step3p5-vs-DeepSeek divergence 都是潜在重复 bug（设计）**。seq_len=0 pad = 静态 BATCH=16
   padding 背离 DeepSeek 动态 T；DeepSeek 无 pad 行故永不踩。每个"和 DeepSeek 不一样"都埋雷。
5. **底座漂移（环境）**。5 仓 + driver/CANN + 2 分支(stepfun/develop vs n1-live) + 2 机器(0234 down→0162)。
   升级栈曾 drop Phase-16 SDMA-OFF patch → 重踩 507899；rebase 引入 SplitIncoreOrch 回归。
   一组合上 ready 不迁移到另一组合 → 每 session re-derive。

## 对策（可落地）
- doc/memory **区分「假设」vs「隔离证明的事实」**；错的**大声 CORRECTED/SUPERSEDED** 撤回。
- **声明 root cause 前先跑证伪实验**。
- **"ready" 只认 live-token-exact-device**；compile/offline/synthetic 一律标 `provisional`。
- **别在 BF16-dequant 上堆 live 集成** → 直接 INT8-native（消临时地基）。
- **能对齐 DeepSeek 就对齐**；必须不一样的写清"为什么 + 验证口径"。
- **pin 单一可复现底座**（分支/机器/版本组合）。
</content>
