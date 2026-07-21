# 实时状态（STATUS）

> **只放当前真相**：当前 phase、组件 pin、活跃 blocker、机器状态。
> 每日流水在 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)；
> 整体规划在 [`planning/roadmap.md`](planning/roadmap.md)；接力面在
> [`planning/handoff.md`](planning/handoff.md)。
> **最后更新：2026-07-18。**

## 阶段跟踪

| 阶段 | 标题 | 状态 | 详情 |
|-----:|------|------|------|
| **1** | pypto kernel 原型 | ✅ 已完成 | [`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md) |
| **2** | vLLM Ascend 后端集成 | 🟡 进行中 | 见下 |

### Phase 2 sub-phases

| Sub-phase | 范围 | 状态 | 文档 |
|-----------|------|------|------|
| **20** | vLLM monkey-patch e2e（整模型 patch `Step3p5Model.forward`） | 🟡 待实现；dump-based 精度已清，production backend 未接 | [`design/vllm-pypto/02-detailed-design.md`](design/vllm-pypto/02-detailed-design.md) |
| **21** | 与 vLLM 原生精度对比 harness（L1/L2/L3） | ✅ dump-based 闭环；在线 gate 待 Phase 20 | [`archive/completed-phases/21-precision-validation.md`](archive/completed-phases/21-precision-validation.md) |
| **22/26** | Perf baseline + 优化；TP=8 多卡 | 📐 设计已落；gate 见 roadmap | [`archive/completed-phases/22-perf-baseline.md`](archive/completed-phases/22-perf-baseline.md) |
| **27** | N=1 单 `@pl.program` whole-net standalone | ✅ canonical P42 20/20 `argmax=303`（2026-07-18 single-submit 合入三仓 `stepfun/develop`） | [`planning/phases/27-n1-whole-net-fusion.md`](planning/phases/27-n1-whole-net-fusion.md) |
| **28** | N=1 whole-net → vLLM live single-handoff | 🟡 standalone gate 已清；per-layer KV、3-way HBM、live token-exact A/B 待完成 | [`planning/phases/28-n1-live-integration.md`](planning/phases/28-n1-live-integration.md) |

> 交付分级 / 到 v1.0 的规划见 [`planning/roadmap.md`](planning/roadmap.md)。
> **口径提醒**：dump-based 精度闭环 ≠ 真实 vLLM 请求已走 PyPTO NPU runner；
> production backend（Phase 20）仍未完成。

## 组件 Pin Snapshot（最新）

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS(src) | simpler | ptoas-bin |
|------|------|-------|-----------|---------|-----------|---------|-----------|
| 2026-07-18 | N=1 single-submit 合入三仓 `stepfun/develop` + 干净回归 20/20 | `9ec303f6` | `e1513d22` | `ecb6c303` | `72ada0a1` | `c7fdc574` | v0.45(见 stable SSOT) |
| 2026-07-17 | N=1 stable env freeze（SSOT `develop/N1/N1-STABLE-ENV-0162-20260717.md`） | `n1fusion-base:e277de9f` | `feat/whole-net-n1-fusion:0e7a0fdd` | `ecb6c303` | `72ada0a1` | `n1fusion-base:36957c6b` | v0.45 |
| 2026-06-22 | Phase 2 设计落地；建项目跟踪仓 | `stepfun/develop:b00c8b23` | `stepfun/develop:b918e60` | `e25732f0` | `da011a3d` | `a6e06406` | v0.45 |

> 完整 pin 历史见 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)。

## 当前 Blocker / Deferred（摘要，详见 [`blockers.md`](blockers.md)）

| # | Blocker | 严重度 | gate 什么 | 详情 |
|--:|---------|--------|-----------|------|
| N1-S-0234 | 0234 同步 pypto-lib 后 whole-net stall（完整对象未确认） | 🔴 Active / 未独立复核 | 取得 SSH 后核对三仓/runtime/环境重跑 canonical | [`blockers.md`](blockers.md) |
| N1-L | Phase 28 live：per-layer KV + 3-way HBM + live token-exact A/B | 🔴 Active | live single-handoff | [`planning/phases/28-n1-live-integration.md`](planning/phases/28-n1-live-integration.md) |
| 1 | Phase 20 production backend 未接入 | 🟡 功能 | 真实 vLLM 请求走 PyPTO runner | [`design/vllm-pypto/`](design/vllm-pypto/) |
| 2 | Prefill MoE L1 overflow（TASK-29） | 🟡 功能/性能 | 真实 PyPTO NPU prefill kernel | [`blockers.md`](blockers.md) |
| 3 | head_gate 语义（历史 ×1 旁路已由 on-device gate 取代） | 🟡 精度 | 在线 backend L1 parity | [`postmortems/09-attention-multiposition-corruption.md`](postmortems/09-attention-multiposition-corruption.md) |
| 5 | MTP 集成进 decode | 🟢 Deferred | speculative 吞吐 | [`blockers.md`](blockers.md) |

> 已解 blocker 转为专项复盘：[`postmortems/`](postmortems/)（如 507899/507018、
> co-tenancy、tmov、gate_topk、gap-5、scheduler-timeout 等）。

## 机器状态

**`gpu-a910x-0162`（Phase 16 验证机，当前主力）**：driver 25.5.2 ✅ / firmware
7.8.0.7.220 ✅ / CANN 9.0.0 ✅；simpler L3 allreduce、前端 smoke、dense/SWA/MoE
ST、N=1 canonical P42 20/20 均 PASS。唯一 stable 环境记录见
[`develop/N1/N1-STABLE-ENV-0162-20260717.md`](develop/N1/N1-STABLE-ENV-0162-20260717.md)。

**`gpu-a910x-0234`**：三剑合璧已齐（driver 25.5.2 / firmware 7.8.0.7.220 / CANN
9.0.0-beta.1）。2026-07-16 起 SSH `Permission denied`，不可达——既不能标 poisoned
也不能标已验证。恢复步骤见 [`deployment/machine-recovery.md`](deployment/machine-recovery.md)。
