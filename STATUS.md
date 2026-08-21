# 实时状态（STATUS）

> **T1 = 只写此刻为真。** 每条都要带 sha / digest / 门结论。
> **禁止**：历史快照、"曾写…已撤回"、campaign 叙事、"我这次发现"。
> 那些属于 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)（开发流水）、
> [`postmortems/`](postmortems/README.md)（定案复盘）或 [`benchmark/`](benchmark/)（证据）。
>
> - 哪些是**确定落地**的（含全部镜像 digest + pin 历史）→ [`progress/landed.md`](progress/landed.md)
> - 未决 → [`blockers.md`](blockers.md)　接力 → [`planning/handoff.md`](planning/handoff.md)
> - **开工前必读**教训索引 → [`postmortems/LESSONS.md`](postmortems/LESSONS.md)
>
> **最后更新：2026-08-21。预算 ≤130 行 —— 超了就是有东西该下沉。**

## 0. Agent 判定当前状态的强制顺序

1. 读取本文件和 [`planning/handoff.md`](planning/handoff.md) 的日期与状态。
2. 用 GitHub 远端 `refs/heads/stepfun/develop` 核对 commit；**不得用本地同名分支、
   worktree 名称或历史 N1 文档推断当前 tip**。
3. 区分"当前源码 tip（SRC）"和"最新 release-qualified 镜像（IMG）"。源码前进不代表新镜像已准出。
4. 镜像只认 manifest digest、明确 pin 和 immutable gate；禁止借用旧镜像数据。
5. `develop/N1/`、旧 phase、旧 benchmark 和 hang-debug case study 都是历史证据，
   不能作为当前 checkout、构建 pin 或发布状态。

## 1. 当前 SRC（源码 tip）

| 项 | 值 |
|---|---|
| pypto-lib | `csy0225/pypto-lib:stepfun/develop@69ad31e4`（0162 指定 checkout clean、与远端对齐） |
| pypto | `csy0225/pypto:stepfun/develop@1c048a74` |
| pto-isa / PTOAS(src) / simpler / ptoas-bin | `ecb6c303` / `fc8c6cae` / `e2efebcb` / `v0.50`（= canonical pin 集） |
| Main 入口 | `models.step3p5.decode_fwd:whole_decode_step3p5` |
| `decode_fwd.py` sha256（tip） | `a17ae27440a4ff0e62f7fe8b6dc2d5548217ef617b0ddbccb927fda648600d01` |
| **MoE 生产路径** | **R5**（`decode_fwd.py` sha `67b73589…`，ctx-64K BS1 p50 `26.329 ms`@ITERS=1000）。R6–R9 dispatch 融合线 **NO-GO** |

`69ad31e4` = TP all-reduce small-message selector：Main 在 rank-uniform `active_rows == 1` 时走
静态 8 KiB 两波 one-shot mesh，其他行数与 MTP 走静态三波 reduce-scatter + push all-gather
fallback；ownership 固定 `HIDDEN // TP_WORLD_SIZE`，与 transfer chunk 解耦。
⚠ **ABI 变更**：`dense_mlp_body_tp` 在 `mlp_layer_idx` 后新增 `num_tokens` 实参 ——
仓外直接调用或 inline 该 body 的代码升级时必须同步更新。

**这些都只过了 source-overlay 门，没有包含它们的 immutable image。**
逐项门结论 + 证据链接：[`progress/landed.md`](progress/landed.md) 表 B。

## 2. 当前 IMG（镜像）

| 角色 | 镜像 | digest | 边界 |
|---|---|---|---|
| **完整 release-qualified 回退基线** | `…:stepfun-develop-20260803-attn-final-wave5` | `4acc77cdce…67b32` | 仅对 0162 完整准出。64K p50 `49.796 ms` |
| **最新镜像**（部分准出） | `…:stepfun-develop-20260811-k8-selective` | `076af8a167…c47f3` | pypto-lib=`cb96747e`，**不含** tip `69ad31e4`。64K bs1 p50 `32.14 ms`。⚠ Main batch16 / MTP batch1+16 / 六档独立 64K golden-A/B / formal matched-source DFX **未重跑** ⇒ 不是完整 production release-qualified |

其余镜像（含 pre-fix evidence 与其 digest）全部在 [`progress/landed.md`](progress/landed.md) 表 A。
**不得**把旧镜像的 golden / 性能 / DFX 升级为当前 tip 的准出结论。

## 3. 权威门口径（任何 A/B 都以此为精度门）

```text
hidden_sha256  567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e
tail token     14371                     (ctx=65536, bs=1)
N=128 逐 token >=95%  (冻结 vanilla vLLM W8A8 greedy oracle, seed 6127, sha c9b2c721…dd947)
```

- 单 token `argmax==303` 只是首 token 冒烟 / liveness，**不是精度准出**。
- **绝对值不可横比**：bs=1 用 `blocks=512`、bs=8 用 `blocks=4096`，编译期容量不同；
  bs=16 ctx-64K 物理不可行（16 GiB 单次 `rtMalloc` → `207001`）。
- bs=1 A/B 检测地板 `0.634 ms`。
- canonical 验收口径：[`reference/canonical-test.md`](reference/canonical-test.md)。

## 4. Attention 当前判断

- SWA tail-window mask 已改显式 typed INT32 数值区间（`63814d4a`），替掉 `pl.cmp` predicate
  转换路径。**source-level N=128 `127/128` PASS，镜像级未重跑** ⇒ 不能宣称 SWA 修复无性能回退。
- Full/SWA 核心计算的可避免调度 bubble 已闭环；logical task 按 workload 与 architecture
  profile 推导，**不固定 24 个物理核**。A2A3 blocks-per-task profile `22/16/22`、reduce fan-in=8
  **不是跨架构常量**，换架构必须重新 sweep。
- Full Pass-A 已并入 SV；Full/SWA out-proj cast 均融合。
- 设计入口：[`design/performance/04-attention-optimization.md`](design/performance/04-attention-optimization.md)。

## 5. MoE 当前判断

- `7928a275` / `cd19fe6b`（active-route scheduling）/ `491267c4`（route/precision release harness）
  均为 `stepfun/develop@69ad31e4` 的祖先。
- J1 保持 🟦 / NO-GO：source-overlay N=128 已过，但当前 tip 对应的 final immutable image 精度、
  六档 64K golden/A/B、formal matched-source DFX 12 runs 与 route-aware reanalysis 都未完成。
- canonical structural analyzer 仍 `FAIL_CLOSED`：零本地 routed-token 的 early-dispatch task
  缺 AICore swim record（既有限制，不是本轮回归）。
- 设计入口：[`design/performance/05-moe-optimization.md`](design/performance/05-moe-optimization.md)。

## 6. 当前下一步（详细接力面见 [`planning/handoff.md`](planning/handoff.md)）

1. **基于 `pypto-lib@69ad31e4` 构建 immutable candidate image**，固定 manifest/config 与所有 pin。
   不得把 source-overlay 数据当作新镜像数据。
2. 新镜像上重跑同口径 Whole A/B/A、Main/MTP compile、Main N=128、多 batch 与 canonical
   structural analyzer；zero-token raw-swim 限制未解决时继续 fail-closed。
3. 新镜像上重跑 BS `1/2/4/7/8/16` × 每请求独立 64K、L3/L4 golden 与 counterbalanced A/B。
4. 为最终 image/source 重新生成 matched source policy（candidate 绑最终 commit 源码哈希，
   baseline 从选定 immutable control source 独立计算）；**不得沿用历史
   `baseline=3553664c` / `candidate=7884da7c`**。
5. 用 `pypto-image-verify` + `pypto-perf-regression` 对最终 image 执行标准回归；
   若提升为完整 production release，按 Wave5 同口径补 Main N=128×3、Main batch16、MTP batch1/16。
6. BS16×每请求 64K 必须先过 runtime-memory 容量门禁；不能把 OOM 或两层数据写成整网性能。
7. ★★ **新性能主线候选**：`bind.args` = `6.12 ms` ≈ **ITL 的 23%**（纯 host 侧参数绑定，与
   `runner_run` 加性）—— 量级比 dispatch 域任何 small-op 融合大一到两个数量级。
   **已按 swimlane 立项为 `H4`（P0，上界 `6.12 ms` ≈ 9.9× 地板 `0.616 ms`）+ `H5`
   （P1，`early_dispatch` swim record 缺失 ⇒ 8 卡里只有 rank2 可分析，是 device 侧一切
   cross-rank 结论的前置）**；`K10` 降为 H4 之后（其上界 `0.45–0.53 ms` 低于近期 bracket
   地板，须紧 bracket 才可判）。
   ★ **强怀疑 `bind.args ≡ H2`**（H2 根因里 `~92 make_tensor_arg`/`add_tensor` 就是参数
   绑定，每 step `8 rank × 183 = 1464` 次 Python 级操作）⇒ 这条"最大新线索"可能早有设计。
   ★ 附一条**可复用否决判据**：关键路径 `front-gap = 0.000 ms` 且 stall 100% 为 data-wait
   ⇒ 方法论 `01-task-granularity` + `02-runtime-overhead` 两章 ROI 上界 = 0
   （**本可在 dispatch 融合线的 6 天 / 357 run 之前就否决它**）。
   排序全文：[`design/performance/09-swimlane-derived-next-optimizations.md`](design/performance/09-swimlane-derived-next-optimizations.md)；
   看板：[`design/performance/task-tracking.md`](design/performance/task-tracking.md)。

## 7. 其它项目级 active work

真实 vLLM live front、paged-KV / dynamic batch、同代 Main→MTP absolute gate 和 3-way HBM
仍未闭环；属 serving 集成，不改变本轮 attention/MoE 的准出顺序。

## 8. 机器状态口径

0162 是本轮验证机（driver `25.5.2` / firmware `7.8.0.7.220` / CANN `9.0.0-beta.1`）。
**每次作业前必须重新检查卡占用**，不能沿用旧 session 的空闲结论。
锁：Claude 用 `0162-cards0-7.lock`（dev0–7），codex 用 `0162-cards8-15.lock`（dev8–15），
`0162-full-machine-perf.lock` **仅 A/B/A 发布门**。诊断 / 相对分析可并发；A/B/A 必须持整机锁串行、两半都空。

## 9. Blocker 摘要（一条一行，详见 [`blockers.md`](blockers.md)）

| # | Blocker | 严重度 | gate 什么 |
|---|---|---|---|
| UPSTREAM-NOTIFY-FENCE | pypto `MakeNotifyCodegenPTO` 把 `dcci`(invalidate-only) 排在 payload drain 之前；最小修复 = 一条 pre-CMO `pipe_barrier(PIPE_ALL)`（device 已证），Wave2 单点代价 `0.405 µs/call` | 🔴 correctness | 一切"把 payload store 与它自己的 credit 拉近"的 AR 优化 |
| N1-S-0234 | 0234 同步 pypto-lib 后 whole-net stall（对象未确认，未独立复核） | 🔴 | 取得 SSH 后核对三仓/runtime/环境重跑 canonical |
| N1-L | Phase 28 live：per-layer KV + 3-way HBM + live token-exact A/B | 🔴 | live single-handoff |
| DEPLOY-REPRO | 已发布镜像内工作树曾 dirty，部分记录 pin 不可复现 | 🟡 | 镜像级可复现性审计 |
| Phase 20 backend | production backend 未接入，真实 vLLM 请求未走 PyPTO runner | 🟡 功能 | live serving |
| Prefill MoE L1 overflow | TASK-29 | 🟡 功能/性能 | 真实 PyPTO NPU prefill kernel |
| head_gate 语义 | 历史 ×1 旁路已由 on-device gate 取代；小 N `matmul_acc` 上游未修，gate 仍在 worker 侧预算 | 🟡 精度 | 在线 backend L1 parity |
| MTP 集成进 decode | — | 🟢 Deferred | speculative 吞吐 |

**已定案转复盘**：MoE dispatch 域融合线（`ORCH-SCALAR-READ-VS-CROSSRANK-WAIT`）
⇒ [`postmortems/16-dispatch-fusion-orch-decouple.md`](postmortems/16-dispatch-fusion-orch-decouple.md)。
