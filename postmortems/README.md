# 事故复盘 Postmortems

工程专项复盘索引。每份复盘对应**一个**可命名的工程/部署事故（一个 error
signature 或一类根因）。目标是沉淀——让后来者不重复踩坑。

## 三个索引，三个轴（先选对轴）

| 页 | 索引轴 | 什么时候看 |
|---|---|---|
| [`LESSONS.md`](LESSONS.md) | **动作**（你正要做 X → 先记住 Y） | **开工前**扫一遍 —— `CLAUDE.md` 铁律 §0 强制必读 |
| [`CASEBOOK.md`](CASEBOOK.md) | **现象**（你看到了 X → 背景/过程/处置） | **撞上了**再查。含**仍在生效的绕路台账**（🩹 标记 + 移除代价）—— 删一段看似冗余的代码前先查它 |
| 下表复盘 | **error signature / 一类根因** | 要完整证据链、消融矩阵、走过的弯路 |

## 复盘清单

| 编号 | 专项 | error signature | 子系统 | 状态 |
|------|------|-----------------|--------|------|
| [`01`](01-multirank-ipc-507899-507018.md) | 多卡 IPC `aclrtIpcMemImportByKey` / simpler init | `507899` / `507018` | deployment | ✅ 已解 |
| [`02`](02-0234-l3-ipc-pid-validation.md) | 0234 L3 allreduce 跨卡 IPC PID 校验 | `207006` / `507899` | deployment | ✅ 已解 |
| [`03`](03-hccl-cotenancy.md) | pypto worker 与 vLLM 同卡 HCCL control comm 冲突 | `HcclCommInitRootInfo failed: 7` | vllm-pypto | ✅ 已解 |
| [`04`](04-tmov-vec-lhs-matmul.md) | Vec-LHS 矩阵乘 `pto.tmov` 编译失败 | `pto.tmov ... supported tmov address-space pair` | codegen | ✅ 已解 |
| [`05`](05-splitincoreorch-swiglu-l43-l44.md) | swiglu MoE L43/L44 `SplitIncoreOrch` precondition | `InCore ScopeStmt found in non-InCore function` | codegen | 🟡 缓解 |
| [`06`](06-gate-topk-deadlock.md) | EpTpMoE 8 卡 real-W8A8 `gate_topk` 死锁 | `507018` + `sched_error_code=100` | whole-net | ✅ 已解（死锁段） |
| [`07`](07-whole-net-scheduler-timeout.md) | 整网 scheduler timeout（per-layer comm window alias） | `507018` / `orch_error=8 TENSOR_WAIT` | whole-net | ✅ 已解（deterministic 段） |
| [`08`](08-multiprogram-coprepare-deadlock.md) | 多程序 co-prepare 死锁（N≥6 program 墙） | `SCOPE_DEADLOCK` / `SCHEDULER_TIMEOUT` | whole-net | ✅ 已解（裁定单 `@pl.program`） |
| [`09`](09-attention-multiposition-corruption.md) | attention 多 position 乱码（rope-q-pack + head-gate `matmul_acc`） | `rot_q_hi` band corrupt / logits ~20× 偏小 | codegen | ✅ 已解（model-side） |
| [`10`](10-gap5-attention-quant-scope.md) | gap-5 in-kernel `cast(→INT8)` 喂 cube A-operand 误编译 | ~98% wrong, no fault | codegen | 🟡 缓解 |
| [`11`](11-8001-bridge-live-ops.md) | 8001 PyPTO bridge live 运维 | HCCL binary conflict / PID ns / exbus leak / `507899` | vllm-pypto | ✅ 已解 |
| [`12`](12-integration-churn-meta.md) | 集成反复推翻（meta） | —（流程级） | meta | 🟡 缓解 |
| [`13`](13-tp-allreduce-pull-notify-race.md) | TP all-reduce pull-form all-gather 跨方向发布 race | `hidden_tp_spread != 0` | whole-net | ✅ 已解 |
| [`14`](14-image-dirty-worktree-unreproducible-pins.md) | 镜像 dirty worktree 导致 pin 不可复现 | 相同 pin、行为不同 | deployment | ✅ 已解（主路径） |
| [`15`](15-tp-allreduce-source-publication-lifetime.md) | TP all-reduce source publication / lifetime 边界 | 间歇性 `hidden_tp_spread != 0` | whole-net | ✅ 已解（0162） |
| [`16`](16-dispatch-fusion-orch-decouple.md) | MoE dispatch 域小算子融合整线 NO-GO（`orch` 阻塞读是承重节流；orchestrator 不在关键路径） | `orch_error=8 TENSOR_WAIT` / `orch_error=2 HEAP_RING_DEADLOCK` | whole-net / perf | ✅ 已定案（**负结论**） |
| [`17`](17-upgrade-ipc-buffer-provenance.md) | 升级后 IPC interior slice 缺 Buffer provenance | `raw-pointer DeviceTensor cannot be dispatched` | whole-net / runtime | ✅ 已解 |
| [`18`](18-upgrade-itl-fixed-cost-runtime-contract.md) | 升级 ITL 固定 host 开销与 H4 运行合同混写 | 同一工作点 `47.993 / 27.812 / 22.253 ms` | whole-net / perf / deployment | 🟡 发布门已解，deployment env 待接线 |

## 必读教训索引

**开工前先读** [`LESSONS.md`](LESSONS.md) —— 从本目录复盘的「如何避免」段提炼的
触发式索引（你正要做 X → 先记住 Y）。它是 `CLAUDE.md` 铁律 §0 的强制必读项。

## 模板与新增

- **模板**：[`TEMPLATE.md`](TEMPLATE.md) —— 六段结构（背景 / 现象 / 根因 / 解决 /
  走过的弯路 / 如何避免）。
- **新增一份复盘**：复制 `TEMPLATE.md` 为 `NN-<short-name>.md`（NN 接上一编号），
  填完字段 + 六段，然后在上表加一行。每份复盘只对应一个 error signature 或一类根因，
  不要合并多个不相关的事故。
- **不够大、不值得单独成篇的单点坑** → 写进 [`CASEBOOK.md`](CASEBOOK.md)（每条 ≤14 行），
  不要为它新开一份复盘；也不要塞进已有复盘的正文。
- **任何一条值得开工前就记住** → 再回填 [`LESSONS.md`](LESSONS.md) 一行。

## 相关

- 当前活跃阻塞：[`../blockers.md`](../blockers.md)
- 部署 runbook（事故预防）：[`../deployment/`](../deployment/README.md)
- 整网架构背景：[`../design/whole-net/`](../design/whole-net/README.md)
- 已解 blocker 的 session 流水：[`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
