# 接力上下文（Handoff）

> **T6 = 纯指针 + "现在接什么"。最后更新：2026-08-21。预算 ≤90 行。**
>
> 开工前必读 → [`../postmortems/LESSONS.md`](../postmortems/LESSONS.md)　当前真相 → [`../STATUS.md`](../STATUS.md)
> 落地台账 → [`../progress/landed.md`](../progress/landed.md)　未决 → [`../blockers.md`](../blockers.md)　流水 → [`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
>
> ⚠ 历史 `fa58b5cf` NO-GO 与 `e5e26f9f` 中间态**不要复制回当前结论**。

## 现在接什么（按优先级，只有三条）

### 1. 基于 `pypto-lib@69ad31e4` 构建 immutable candidate image ← 最高优先

TP all-reduce single-row selector 的**代码与 source-overlay 门都已完成**，
这个专题只剩镜像级 qualification。做法与准出清单见 [`../STATUS.md`](../STATUS.md) §6 第 1–5 条。
**分账铁律**：source-overlay 与 image gate 不得混写；新镜像闭环前不得写
production / release-qualified。

### 2. ★★ 新性能主线：H4 host `bind.args` = `6.12 ms` ≈ ITL 的 23%

纯 host 侧参数绑定，与 `runner_run` 加性（对照臂 `5.87 ms`）——
**ROI 上界 `6.12 ms` ≈ 9.9× 地板**，比 dispatch 域任何 small-op 融合大一到两个数量级。
这是 dispatch 线关闭时从同一份 STRACE 里顺手捡到的副产品。

已按 swimlane 排序立项：**H4（P0）** + **H5（P1，`early_dispatch` swim record 缺失
⇒ 8 卡里只有 rank2 可分析，是一切 device 侧 cross-rank 结论的前置）**；`K10` 降到 H4 之后。
★ **强怀疑 `bind.args ≡ H2`**（未证实）—— H4a/H4b 就是去证它，**两步都不占卡**。
实施顺序（H4a 归因 → H4b 判定 ≡H2 → H4c cheap gate 手改 `host_orch.py` hoist →
H4d 改 overlap → H4e 整机锁 A/B/A）与风险说明：
[`../design/performance/09-swimlane-derived-next-optimizations.md`](../design/performance/09-swimlane-derived-next-optimizations.md)。
看板：[`../design/performance/task-tracking.md`](../design/performance/task-tracking.md)。

### 3. vLLM-Ascend MoE Trace 对齐 → P1 `routed_gmm1_swiglu_quant`

0162 clean worktree `…/develop-worktrees/vllm-moe-trace-align-20260812`（branch
`perf/vllm-moe-trace-align-20260812`，HEAD `9ca01d24`）。K8 digest `076af8…` 上 source-overlay
whole compile `COMPILE_OK 11.4s`（**只代表编译兼容**；该容器 PATH 无 `pytest`、`rc=127`，
单测未运行 —— 不能据此推断镜像内无 pytest package）。

**P1 目标** = regular L3–L42 的 `routed_gmm1_swiglu_quant` primitive + `(expert, source-rank)`
combine data-work bundle。约束：down 与 L43/L44 specialization 保持独立；completion 必须选
per-expert aggregator 以保留 36 credits，或版本化 `N_COMPLETIONS` 并重验 epoch/wraparound；
生成码必须证明 payload 后、credit 前有显式 `PIPE_ALL`/release seam（**不能假设 notify 自带
release** —— 见 [`../blockers.md`](../blockers.md) UPSTREAM-NOTIFY-FENCE）。
⚠ 捕获路径是 local routing + grouped experts + TP AllReduce，**不是** PyPTO 的 EP8
dispatch/combine —— 不要照搬路由/通信拓扑。
完整三列表与验收门：[`../benchmark/2026-08-12-vllm-ascend-decode-moe-trace-gap.md`](../benchmark/2026-08-12-vllm-ascend-decode-moe-trace-gap.md)。

---

## 已关闭：MoE dispatch 域小算子融合线（2026-08-21 收盘）

**不要重启这条线。** R6–R9 八个候选 + 结构修复候选全部 NO-GO；`orch` p50 降 74% 而
`device_wall` 没降 ⇒ orchestrator 从不在关键路径 ⇒ 整类改动 ROI 上界 = 0。
生产继续用 **R5**，`decode_fwd.py` 不做任何改动。
完整复盘（11 条已撤回主张 + 9 条铁律）→ [`../postmortems/16-dispatch-fusion-orch-decouple.md`](../postmortems/16-dispatch-fusion-orch-decouple.md)。

## 已退休、不得恢复的分支

`a791071` attention-inline Ring（实质 A/A，未命中 production canonical）、`b4d45b3` K6b
dynamic-valid-shape（dynamic publish 位于已知 notify-fence seam，无独立 rank-skew/zero-gap/
多 epoch safety proof）。二者只作 focused 历史证据。
其余 NO-GO 台账 → [`../progress/landed.md`](../progress/landed.md)「已否决，不要重试」。

## 现有实现的正确性硬约束（改 TP all-reduce 前逐条核对）

1. `active_rows` 必须在所有 TP rank 上一致，否则 selector 分叉**会死锁**。
2. 固定 peer 顺序、单 FP32 accumulator、一次最终 BF16 cast —— 不得改变。
3. 两波与三波**不能复用未清零的同一 signal slot**；exact two-layer mirror 必须继续与
   canonical body AST 一致。
4. `TP_ALL_REDUCE_OWNED_CHUNK = HIDDEN // TP_WORLD_SIZE = 512` 必须与 `TP_ALL_REDUCE_CHUNK`
   的 staging/final-copy transfer grain **保持解耦**。
5. **仓外**调用 `dense_mlp_body_tp`（含 `pl.inline(dense_mlp_body_tp._func)`）的代码，升级时
   必须补 `mlp_layer_idx` 之后新增的 `num_tokens: pl.Scalar[pl.INT32]` 实参。
6. 本实现**不等于**修复 notify fence；未来合并波次或把 payload store 与自己的 credit 拉近，
   仍被 `UPSTREAM-NOTIFY-FENCE` 阻塞。

设计入口：[`03-tp-allreduce-algorithm-comparison`](../design/performance/03-tp-allreduce-algorithm-comparison.md)、[`04-tp-allreduce-ring-refactor`](../design/vllm-pypto/04-tp-allreduce-ring-refactor.md)。

## 机器与操作约束

**每次启动前重新检查锁、container、`fuser` 与 NPU process**（非 root `fuser` 不可信 ——
须 `sudo -n fuser` + `npu-smi info -t proc-mem` 双查、fail-closed），不能沿用旧 session 的空闲结论。
锁与卡分配见 [`../STATUS.md`](../STATUS.md) §7。

**禁止**：在本地项目仓创建或修改 pypto-lib 产品代码（0162 是唯一执行主机）· 用未持锁的
device matrix 作性能数据 · 把 focused regular-call kernel-duration pooled mean
（如 `38.325 → 22.667 µs/call`）当作 strict critical-tail 或完整源码 A/B/A ·
用 host 独立检查覆盖 canonical structural fail-closed · 把 source-overlay 数据写成
immutable-image 结果。
