# 活跃 Blocker

阻塞项目进展的 open issue 的 SSOT。每条：**症状 / 根因 / 当前状态 /
解除条件 / 链接**。

Blocker 解决时，**删掉本文件里这一节**，到
[`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)
"Resolved blockers" 段补一条 post-mortem。

**最后检视**：2026-06-24。

---

## 1. head_gate × 1 旁路 —— 跟 vLLM 原生精度对齐

**严重度**：🟡 精度 —— gate Phase 21 L1（per-layer hidden_states）严格
对齐。**不**阻塞 v0.1 / v0.2 功能 bring-up；只是"精度验证全绿"准出条件
的一部分。

**症状**：`attention_full.py:658-690` 和 `attention_swa.py` mirror 用
`attn_out_gated = attn_out`（× 1 identity），不是
`attn_out_gated = attn_out * sigmoid(head_gate_logits)`。每层 attention
输出大致是上游期望值的 2 倍（`sigmoid` 平均输出 ~0.5）。

**根因**：pypto kernel 没法表达 head_gate 操作而不触发
`pl.row_expand_mul([N, K], [N, 1])` 在 1 列 FP32 操作数上 — 这会撞 AIV
32-byte 行对齐限制。这是 pto-isa 硬限制，model 侧无干净绕路。

分类参考：`pypto-lib/docs/known-pypto-pitfalls.md` §1。

**跟踪**：TASK-L（pto-isa 上游 — 用 cube-matmul 配 block-diag R 矩阵
构造）。在 backlog 里跟踪。

**解除条件**（任一，按优先级）：

A. 上游 pto-isa 落 `[N, 1]` slice 32-byte 静态对齐 reject（§1 doc 提到）
   **同时**我们在 attention_full / attention_swa 用 cube-matmul × block-
   diag R 构造表达 head_gate，避免 intra-UB `[N, 1]` Vec tile。
B. Phase 21 §2.7 标定 —— patch 上游 vLLM `Step3p5Attention` 也走 × 1
   identity（语义上丢掉 gate）。失去 ~2× attention scaling 对生产意义不
   利，但允许 L1 ratio_allclose 在两个（同样降级的）实现之间通过。
C. 拓宽 Phase 21 L1 容忍区间，吸收 attention-output-only 路径 ~50%
   magnitude 差。less rigorous；记录差距即可。

**估时**：
- 路径 A：周（上游 gate）
- 路径 B：1-2 天（vLLM 侧 patch + 重跑）
- 路径 C：0.5 天（tolerance 配置改 + 重 baseline）

**Owner**：TASK-L 上游；项目侧决策待定。

---

## 2. Prefill MoE L1 overflow (TASK-29)

**严重度**：🟢 Deferred —— gate Phase 17（完整 prompt processing
e2e），**Phase 22 decode-only perf 不需要**。

**症状**：`models/step3p5/prefill_moe.py` 编译时 L1 buffer overflow
（~5 MB > 限）在 `moe_gate_up` MLP。Prefill MoE 层编译不过。

**根因**：Prefill 在很宽的 SEQ 维度上跑（如 SEQ=4096 vs decode BATCH=16），
decode UB 装得下的 MoE kernel 结构到 prefill 会爆 L1。

**跟踪**：TASK-29 in backlog。

**解除条件**：重设计 prefill_moe，加 multi-step gate_up chunking。~1-2
周专门工作。

**Phase 22 decode-only perf 的绕路**：用合成数据预填 KV cache 到目标 input
length，跳 prefill，测 decode-only TPS / ITL。详见
[`phases/22-perf-baseline.md`](phases/22-perf-baseline.md) "Prefill
workaround"。

**Owner**：未指派。

---

## 5. 机器 0234 driver+firmware 升级

**严重度**：🟢 基础设施 —— 备用部署机。**不**阻塞 0162 上的 Phase 2 工作。

**症状**：0234 driver `25.5.1` / firmware `7.8.0.6.201` /
CANN `9.0.0-beta.1`。`support_shmem_map_exbus=0` cap 还在因为
driver+firmware 都低于 Phase 16 minimum（`25.5.2` / `7.8.0.7.220`）。
跨卡 `aclrtIpcMemImportByKey` 返回 507899。0234 上跑多卡 e2e 不可能。

**根因**：标准 Phase 16 部署需求还没在 0234 上应用。

**解除条件**：按 [`deployment/machine-recovery.md`](deployment/machine-recovery.md)
跑升级。两个 `.run` 包已 stage 在 0162 `/mnt/persist/ascend-staging/`：

```
Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run
Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run
```

scp 到 0234，停 kubelet，跑 driver `--upgrade --quiet`，重启，搞定。

CANN 在 0234 上**已经是正确版本** —— **千万不要**跑可能 revert 到 GA
的集群自动化（升级前把 beta.1 install 备份到 persistent storage）。

**估时**：~2 小时 wallclock（含重启）。

**Owner**：未指派。

---

## 6. (Deferred) MTP 集成进 decode_fwd

**严重度**：🟢 Deferred —— speculative decoding 吞吐倍率。**不在** Phase 2
关键路径上。

**症状**：3 个 MTP 层有 kernel（`models/step3p5/mtp.py`）但没拼进
`decode_fwd`。vLLM 的 MTP 路径期望 1 main token + N speculative tokens
+ verification accept/reject，accept rate 高时给 ~3× 吞吐。

**根因**：没建过；Phase 1 期间为了聚焦关键的 45-layer dense+MoE 路径
deferred 掉。

**解除条件**：Phase 23 设计（TBD）—— 把 MTP 拼到 `decode_fwd` 输出阶段；
跟 vLLM speculative decoding pipeline 集成。

**估时**：Phase 22 baseline 出来后 2-4 周。

**Owner**：未指派，deferred。

---

## 怎么加新 blocker

1. 在最 deferred 项的位置之前插一节，选对严重度图标。
2. 顺序编号（不复用老编号）。
3. 从新节链回去症状第一次出现的地方（某 phase doc / `archive/milestones-
   2026-Q2.md` 里的某次 session log 等）。
4. 在 [`STATUS.md`](STATUS.md) "硬 Blocker" 表加一行。
5. 如果 gate 某个具体 phase，从那个 phase doc 的 "Risks" 段链过来。

## 4. Final e2e precision prerequisites

**严重度**：🔴 Critical —— gate 最终验收“端到端精度正确且无阻塞”。

**当前预检命令**：

```bash
cd <pypto-lib>
python tools/step3p5/e2e_precision_readiness.py --batch 2
```

**2026-06-24 结果**：host 级 smoke 全绿，但最终 e2e 精度仍被以下前置条件阻塞：

1. 0162 未挂载默认真实权重目录 `/mnt/chensiyu-jfs/multi-hardware/models/step3p5_flash_release_hf_mtp3_bf16`。
2. 当前环境未发现 vLLM / stepcast 原生 Step3p5 模型代码或 Python package。
3. `Step3p5DecodeFwd.host_orch` 仍是 final RMS + LM head skeleton，尚未 wire 45 层 per-layer program。
4. head_gate 当前在 PyPTO 侧是 ×1 bypass；vLLM parity 需要同策略 patch 或明确接受 L1 差异。
5. MoE 8 卡 ST runtime 已 PASS，但还缺 MoE golden 精度验证。

**解除条件**：真实权重 + vLLM oracle 可见；`decode_fwd` 45 层接线完成；能导出同一 decode step 的 hidden/KV/cache/slot 输入；8 rank logits shard concat 后与 vLLM logits/top-k 对齐。

