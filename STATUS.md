# 实时状态

pypto step3p5 项目的实时状态板。**任何 phase / sub-task / blocker 状态
变化都更新这里**。历史细节查 [`archive/`](archive/)。

**最后更新**：2026-06-24

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
| **v0.3** | TP=8 多卡 dense + mixed-mode MoE | Phase 20 + Phase 22.1-3 | ✅ kernel blocker 已清；待 vLLM harness |
| **v1.0** | TP=8 / EP=8 全 pypto MoE + perf 数发布 | Phase 20-22 全完 | 待整网精度 + perf 优化（split task 融合） |

**当前**：Decode 阶段 kernel/ST 已推进到 TP=8/EP=8 MoE runtime PASS；dense full/swa 单卡精度 ST PASS；MoE 8 卡 ST 目前只验证 runtime（validation skipped，尚未做 golden 精度）。**下一步可启动 Phase 20/21 的整网端到端精度对齐 harness**，但还不能宣称整网精度已通过。

---

## 立即可做的下一步（按优先级）

1. **Phase 20.1**：`config_align.py` — 校验 vLLM `hf_config` 与 pypto `config.py` 常量。
2. **Phase 20.2**：`weight_translate.py` — vLLM `nn.Module` → pypto bundle dict。
3. **Phase 21 入场准备**：先跑整网 decode-only 端到端精度对齐（L1 hidden / L2 logits），确认 head_gate ×1 旁路的可接受策略。
4. **后续性能优化**：当前 MoE dispatch 采用 split task 保正确性；恢复/融合成非 split task 作为 Phase 22 perf 优化项，不阻塞精度 harness。

---

## 组件 Pin Snapshot（最新一行）

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS（src） | simpler（submodule） | ptoas-bin |
|------|------|-------|-----------|---------|--------------|---------------------|-----------|
| 2026-06-22 | Phase 2 设计落地；建项目跟踪仓 | `stepfun/develop:b00c8b23` | `stepfun/develop:9c4773f`（已撤回误置 docs） | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `a6e06406` | `v0.45` |

历史 pin snapshot 见 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)。

---

## 当前 Blocker / Deferred Items

| # | Blocker | 严重度 | gate 什么 | Owner | 详情 |
|--:|---------|--------|-----------|-------|------|
| 1 | head_gate × 1 旁路 — vLLM 原生语义偏离（sigmoid gate 用 identity 替代） | 🟡 精度 | Phase 21 L1 layer-级 parity | TASK-L（pto-isa 上游） | [`blockers.md`](blockers.md) §1 |
| 2 | Prefill MoE L1 overflow（TASK-29） | 🟢 Deferred | Phase 17 prefill e2e（Phase 22 decode-only 不需要） | 未指派 | [`blockers.md`](blockers.md) §2 |
| 3 | 0234 driver+firmware 升级未做 | 🟢 基础设施 | 备用部署机 | 未指派 | [`blockers.md`](blockers.md) §3 |

---

## `gpu-a910x-0162`（Phase 16 验证机）目前已确认能跑

| 组件 | 验证 | 备注 |
|------|------|------|
| driver 25.5.2 | ✅ 2026-06-22 | `npu-smi info -t board -i 0` 报上 |
| firmware 7.8.0.7.220 | ✅（chip flash） | 跨重启持久 |
| CANN 9.0.0 non-GA/non-beta | ✅ `/usr/local/Ascend/cann` → `/mnt/persist/Ascend/cann-9.0.0/cann-9.0.0` | 2026-06-24 已重装并重编译 pypto/runtime |
| simpler L3 allreduce_distributed -d 0-1 | ✅ 2026-06-24 | 1 passed / 1 skipped（pytest harness） |
| pypto-lib 前端 smoke rc=0 | ✅ 2026-06-24 | `_smoke_program_build` 通过 |
| Decode dense full ST @ device 0 | ✅ 8.54s（ratio_allclose PASS，2026-06-24） | CANN 9.0.0 non-GA 重编译后验证 |
| Decode dense SWA ST @ device 0 | ✅ 15.61s（ratio_allclose PASS，2026-06-24） | CANN 9.0.0 non-GA 重编译后验证 |
| Phase 19 MoE 6 variants smoke compile | ✅ 6/6 PASS | TP=8 per-rank slice 路径 |
| Decode MoE full_silu_silu ST @ 8 cards | ✅ runtime PASS 26.51s（validation skipped，2026-06-24） | `tile_valid > 0` guard + split EP dispatch；需继续补 golden 精度 |
| Phase 15 单卡 e2e | ✅ rc=0，20 tasks complete | head_gate ×1 旁路 + TP=1 patch 路径 |

---

## `gpu-a910x-0234` 当前状态

未升级。driver `25.5.1` / firmware `7.8.0.6.201` / CANN `9.0.0-beta.1`。
多卡 e2e 因 driver shmem-exbus cap 缺口而被卡，必须先升 driver+firmware。
`.run` 包已 stage 在 0162 `/mnt/persist/ascend-staging/` —— 升级 runbook
见 [`deployment/machine-recovery.md`](deployment/machine-recovery.md)。
