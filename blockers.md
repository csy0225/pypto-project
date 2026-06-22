# 活跃 Blocker

阻塞项目进展的 open issue 的 SSOT。每条：**症状 / 根因 / 当前状态 /
解除条件 / 链接**。

Blocker 解决时，**删掉本文件里这一节**，到
[`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)
"Resolved blockers" 段补一条 post-mortem。

**最后检视**：2026-06-22。

---

## 1. Barrier `tp_all_reduce` UB overflow

**严重度**：🔴 Critical —— gate Phase 22.3（多卡 dense）和所有 v0.3+ tier。

**症状**：pypto 编译在 `AllocateMemoryAddr` pass 报：

```
Verification failed after 'AllocateMemoryAddr':
  Function 'tp_all_reduce': Vec buffer usage (655360 bytes)
  exceeds platform limit (188416 bytes)
  Location: <pypto-lib>/models/step3p5/decode_layer.py:487
```

**根因**：pypto 编译器把 `for peer in pl.range(group_size=8)` unroll
展开，因为 `group_size` 是 factory closure 里捕获的 Python int；展开
后每次迭代的 `recv` / `recv_fp32` / loop-carried `acc` 被当成不同的
SSA 值，没有 UB 复用。UB 成本 = `7 × ~80 KB ≈ 560 KB`，远超
A2A3 的 184 KB Vec UB 限。

分类参考：`pypto-lib/docs/known-pypto-pitfalls.md` §7。

**WIP 在哪**：`csy0225/pypto-lib` 分支
`wip/step3p5-barrier-allreduce-20260622` HEAD `b5bb6ee`。分支保留*意图*
（把 ring all_reduce 换成 barrier-style，mirror
`pypto/tests/st/distributed/test_l3_allreduce.py`），但 dense ST device 0
编译时触发 UB overflow。

**解除条件**（任一）：

A. 把 `acc` 不 carry，改成在 `local` 上 in-place store/reload（每次
   迭代 UB 工作集 ≈ 144 KB，能装下 184 KB）。详见
   `pypto-lib/docs/known-pypto-pitfalls.md` §7 "avoidance recipe B"。
B. peer 循环边界改成 runtime-dynamic 用 `pld.nranks(ctx)`（mirror
   canonical test）。同 doc recipe A。
C. A + B 一起，保险。

**估时**：~3-5 天的 pypto-lib 工作 + dense ST device 0 回归检查。无
上游依赖。

**Owner**：未指派。

---

## 2. MoE device runtime 507018

**严重度**：🔴 Critical —— gate Phase 22 v1.0（全 pypto MoE）。Phase 2
v0.1-v0.3（mixed-mode MoE）**不**依赖此 blocker。

**症状**：6 个 MoE variant 编译干净（smoke 6/6 PASS at canonical TP=8
per-rank widths）但 device runtime 5 秒内 fault：

```
[ERROR] sync_run_streams: aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018
[ERROR] orch_error_code=2 sched_error_code=0 runtime_status=-2
RuntimeError: run_prepared failed with code 507018
```

host plog（`~/ascend/log/run/plog/plog-*.log`）只能看到清晰 init 然后
unrecoverable stream sync timeout。device log（`device-*_*.log`）也只看到
init 段。**没有具体 task_id / kernel_name / fault address 落到 host log**
（不像 Phase 15 dense 暴露了 `tslot:6` + `errcode 0x800`）。

**根因假设**：MoE 专属路径（gate_topk → dispatch EP-a2a → routed expert
MLP → combine EP-a2a → shared expert）某个 task 触发 AICore/AICPU
fault。CLAUDE.md 老 memory 归类「same family as simpler#1023 zero-shape
view」是过时的 — dense ST 通过说明 simpler#1023 已修。真因在 MoE 专属
kernel 里。

**复现器**：`gpu-a910x-0162`，2026-06-22：

```bash
cd <pypto-lib>
python -m tests.step3p5.test_decode_layer_moe_st \
    --variant full_silu_silu -p a2a3 -d 0
# 5 秒内 runtime fault
```

**解除条件**：dispatch-cut bisect 工具定位。两条路：

A. **加 `P19_DISPATCH_LIMIT` env hook**（仿 Phase 15
   `P15_DISPATCH_LIMIT`）。host_orch 只跑前 N 个 task。二分定位是哪个
   task 触发 fault。然后查该 task 的 IR / generated kernel / runtime
   trace 进一步定位。

B. **开 DFX swimlane + dep-graph dump**：
   `PYPTO_DISTRIBUTED_DEP_GEN=1` + `PYPTO_DISTRIBUTED_L2_SWIMLANE=1`
   （pypto `03136bf6` 加的 env hook）。看 fault 前最后一个完成的 task。

**估时**：1-2 周（深度 upstream-touching debug；可能要发 simpler 上游
issue 如果根因在 runtime）。

**Owner**：未指派。

---

## 3. head_gate × 1 旁路 —— 跟 vLLM 原生精度对齐

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

## 4. Prefill MoE L1 overflow (TASK-29)

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
