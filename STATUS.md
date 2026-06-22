# 实时状态

pypto step3p5 项目的实时状态板。**任何 phase / sub-task / blocker 状态
变化都更新这里**。历史细节查 [`archive/`](archive/)。

**最后更新**：2026-06-22

---

## 阶段跟踪

| 阶段 | 标题 | 状态 | 详情 |
|-----:|------|------|------|
| **1** | **pypto kernel 原型** | ✅ **已完成** | [`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md) |
| **2** | **vLLM Ascend 后端集成** | 🟡 **进行中**（设计已落） | 见下 |

### Phase 2 sub-phases

| Sub-phase | 范围 | 状态 | 文档 | 估时 |
|-----------|------|------|------|------|
| **2.0（Phase 20）** | vLLM monkey-patch e2e — 整模型 patch `Step3p5Model.forward`；单卡 TP=1；mixed-mode MoE | 📐 设计已落；**任务 1.1-1.9 未启动** | [`phases/20-vllm-backend-monkey-patch.md`](phases/20-vllm-backend-monkey-patch.md) | 3-4 周 |
| **2.1（Phase 21）** | 与 vLLM 原生精度对比 harness；L1/L2/L3 三层 | 📐 设计已落；gate Phase 20 | [`phases/21-precision-validation.md`](phases/21-precision-validation.md) | 3-4 周 |
| **2.2（Phase 22）** | Perf baseline + 两轮优化；TP=8 多卡 | 📐 设计已落；gate Phase 21 + 2 个硬 blocker | [`phases/22-perf-baseline.md`](phases/22-perf-baseline.md) | 6-8 周 |

**到 v1.0 production decode 的总目标**：自 2026-06-22 起约 12-16 周
（含 gate 任务的并行投入）。

---

## Phase 2 交付物分级（跟踪现在到了哪个 sub-version）

| Tier | 能跑什么 | 需要 Phase 2 哪几部分 | 需要清掉哪些 blocker |
|------|----------|----------------------|----------------------|
| **v0.1** | 单卡 dense + mixed-mode MoE 走 vLLM | Phase 20 | 无 |
| **v0.2** | 单卡 45 层 mixed-mode（dense pypto + MoE vLLM eager） | Phase 20 | 无 |
| **v0.3** | TP=8 多卡 dense + mixed-mode MoE | Phase 20 + Phase 22.1-3 | barrier all_reduce UB 修复 |
| **v1.0** | TP=8 / EP=8 全 pypto MoE + perf 数发布 | Phase 20-22 全完 | barrier all_reduce + MoE 507018 |

**当前**:超出 v0.1（Phase 1 全过）。**v0.1 入场无 gate** — Phase 20 实现
可立刻启动。

---

## 立即可做的下一步（按优先级）

1. **Phase 20.1**：`config_align.py` — 校验 vLLM `hf_config` 与 pypto
   `config.py` 各项常量。1 天，无依赖。最便宜入手点。
2. **Phase 20.2**：`weight_translate.py` — vLLM `nn.Module` → pypto
   bundle dict。5 天，Phase 20 核心工作量。
3. **并行做 — gate 1**：写 UB-friendly barrier all_reduce 重写
   （`acc` carry → 在 `local` 上 in-place store/reload）。目标 Phase 22
   多卡入场。详见 [`blockers.md`](blockers.md) §1。
4. **并行做 — gate 2**：写 `P19_DISPATCH_LIMIT` dispatch-cut bisect 工具
   解 MoE 507018。目标 Phase 22 v1.0 入场。详见
   [`blockers.md`](blockers.md) §2。

---

## 组件 Pin Snapshot（最新一行）

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS（src） | simpler（submodule） | ptoas-bin |
|------|------|-------|-----------|---------|--------------|---------------------|-----------|
| 2026-06-22 | Phase 2 设计落地；建项目跟踪仓 | `stepfun/develop:b00c8b23` | `stepfun/develop:9c4773f`（已撤回误置 docs） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `a6e06406` | `v0.45` |

历史 pin snapshot 见 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)。

---

## 硬 Blocker（gate Phase 22）

| # | Blocker | 严重度 | gate 什么 | Owner | 详情 |
|--:|---------|--------|-----------|-------|------|
| 1 | barrier `tp_all_reduce` UB overflow（`pl.range(constant)` 展开 unroll，624KB > 184KB UB 限） | 🔴 Critical | Phase 22.3 多卡 dense / v0.3+ | 未指派 | [`blockers.md`](blockers.md) §1 |
| 2 | MoE device runtime 507018（kernel 内 AICPU/AICore fault，host log 无元数据） | 🔴 Critical | Phase 22 v1.0 全 pypto MoE | 未指派 | [`blockers.md`](blockers.md) §2 |
| 3 | head_gate × 1 旁路 — vLLM 原生语义偏离（sigmoid gate 用 identity 替代） | 🟡 精度 | Phase 21 L1 layer-级 parity | TASK-L（pto-isa 上游） | [`blockers.md`](blockers.md) §3 |
| 4 | Prefill MoE L1 overflow（TASK-29） | 🟢 Deferred | Phase 17 prefill e2e（Phase 22 decode-only 不需要） | 未指派 | [`blockers.md`](blockers.md) §4 |
| 5 | 0234 driver+firmware 升级未做 | 🟢 基础设施 | 备用部署机 | 未指派 | [`blockers.md`](blockers.md) §5 |

---

## `gpu-a910x-0162`（Phase 16 验证机）目前已确认能跑

| 组件 | 验证 | 备注 |
|------|------|------|
| driver 25.5.2 | ✅ 2026-06-22 | `npu-smi info -t board -i 0` 报上 |
| firmware 7.8.0.7.220 | ✅（chip flash） | 跨重启持久 |
| CANN 9.0.0-beta.1 | ✅ symlink `/usr/local/Ascend/cann-9.0.0-beta.1` → NVMe | NOT GA — 见 [`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md) |
| simpler L3 allreduce_distributed -d 0-1 | ✅ 2026-06-22 | `max\|out-expected\|=0` 双卡 golden match |
| pypto-lib 前端 smoke rc=0 | ✅ 2026-06-22 | 4 个 program builder + 8 个 layer-idx variant |
| Phase 19 ST-1 full dense @ device 0 | ✅ 7.93s（ratio_allclose PASS） | 保 TP=8 per-rank slice 宽度 |
| Phase 19 ST-2 swa dense @ device 0 | ✅ 14.85s（ratio_allclose PASS） | 同上 |
| Phase 19 MoE 6 variants smoke compile | ✅ 6/6 PASS | TP=8 per-rank slice 路径 |
| Phase 19 MoE device runtime | ⏸ 5 秒内 507018 fault | blocker §2 |
| Phase 15 单卡 e2e | ✅ rc=0，20 tasks complete | head_gate ×1 旁路 + TP=1 patch 路径 |

---

## `gpu-a910x-0234` 当前状态

未升级。driver `25.5.1` / firmware `7.8.0.6.201` / CANN `9.0.0-beta.1`。
多卡 e2e 因 driver shmem-exbus cap 缺口而被卡，必须先升 driver+firmware。
`.run` 包已 stage 在 0162 `/mnt/persist/ascend-staging/` —— 升级 runbook
见 [`deployment/machine-recovery.md`](deployment/machine-recovery.md)。
