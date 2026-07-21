# 事故复盘 Postmortems

工程专项复盘索引。每份复盘对应**一个**可命名的工程/部署事故（一个 error
signature 或一类根因）。目标是沉淀——让后来者不重复踩坑。

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

## 模板与新增

- **模板**：[`TEMPLATE.md`](TEMPLATE.md) —— 五段结构（背景 / 现象 / 根因 / 解决 /
  走过的弯路 / 如何避免）。
- **新增一份复盘**：复制 `TEMPLATE.md` 为 `NN-<short-name>.md`（NN 接上一编号），
  填完字段 + 五段，然后在上表加一行。每份复盘只对应一个 error signature 或一类根因，
  不要合并多个不相关的事故。

## 相关

- 当前活跃阻塞：[`../blockers.md`](../blockers.md)
- 部署 runbook（事故预防）：[`../deployment/`](../deployment/README.md)
- 整网架构背景：[`../design/whole-net/`](../design/whole-net/README.md)
- 已解 blocker 的 session 流水：[`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
