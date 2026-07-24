# 任务跟踪记录 — Performance 专项

> step3p5 decode 性能优化的**单一事实源**。每个子任务的状态、owner、更新时间、阻塞在此维护。
> 设计详情见 [`02-detailed-design.md`](02-detailed-design.md)；改状态时同步更新本表的「最后更新」列 + 底部更新日志。
>
> **状态图例**：⬜ TODO ｜ 🟦 IN PROGRESS ｜ ✅ DONE ｜ ⛔ BLOCKED ｜ ⏸ PAUSED

---

## 看板（按 Track）

### Track A — 可观测性
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| A1 | whole-net baseline + DFX 采集 | P0 | ✅ | claude | — | 逐层 DFX 拿到：routed-expert 占 90.7%（PMU cube_int8 88.6%）；见 benchmark/2026-07-24 | 2026-07-24 |

### Track B — Mega-kernel 结构
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| B1 | 权重 stacking + `resident="stacked"` | P0 | ⬜ | — | — | — | 2026-07-24 |
| B2 | 45 层 unroll → `pl.range` 循环 | P1 | ⬜ | — | B1, C1 | 等 B1+C1 | 2026-07-24 |
| B3 | KV pool `resident` + in-place | P2 | ⬜ | — | B1 | 等 B1 | 2026-07-24 |

### Track C — MoE 通信协议
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| C1 | 单 window set + `moe_epoch` + `WaitCmp.Ge` | P0 | ⬜ | — | — | — | 2026-07-24 |
| C2 | dispatch push → pull（fixed-slot） | P1 | ⬜ | — | C1 | 等 C1 | 2026-07-24 |
| C3 | peer loop → `pl.spmd`/`pl.parallel` | P2 | ⬜ | — | C1 | 等 C1 | 2026-07-24 |

### Track D — INT8-native W8A8（gap-5）
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| D1 | gate deferred-norm + dispatch-side INT8 量化 | P1 | ⬜ | — | — | — | 2026-07-24 |
| D2 | routed expert INT8×INT8 + requant | P1 | ⬜ | — | D1 | 等 D1 | 2026-07-24 |

### Track E — LM head
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| E1 | LM head 4 段 decoupled + 复用 `recv_x_buf` | P2 | ⬜ | — | C1 | 等 C1 | 2026-07-24 |

### Track F — intra-kernel L1/L0 微调
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| F1 | attention `late_dep` deferral + `allow_early_resolve` | P2 | ⬜ | — | A1 | — | 2026-07-24 |
| F2 | matmul pipeline stage + MTE 512B 对齐 | P2 | ⬜ | — | A1 | 等 A1 | 2026-07-24 |
| F3 | RMSNorm+quant fused deferred-norm | P2 | ⬜ | — | D1 | 等 D1 | 2026-07-24 |

### Track G — 调度轴 / 动态 batch
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| G1 | 调度轴 batch→experts/feature + dynamic active-token | P1 | ⬜ | — | 与 B2 协同 | — | 2026-07-24 |

---

## 进度汇总

| 状态 | 数量 |
|------|------|
| ⬜ TODO | 12 |
| 🟦 IN PROGRESS | 0 |
| ✅ DONE | 1 |
| ⛔ BLOCKED | 0 |
| **合计** | **13** |

**关键路径解锁点**：`A1` + `C1`（先做）→ 解锁 `B2`（mega-kernel 主体）与 `F2`。
**可立即并行认领（无前置）**：`A1`、`B1`、`C1`、`D1`。

---

## 认领指南

1. 从「可立即并行认领」或依赖已 ✅ 的任务里挑。
2. 在对应行填 **Owner**，状态改 🟦，更新「最后更新」。
3. 落地前读 [`02-detailed-design.md`](02-detailed-design.md) 对应卡片（file:line + 算法步骤 + 验证口径）。
4. 完成：**多步 decode 逐 token** vs vanilla vLLM W8A8，seed=6127 / N=128 ≥95% ALIGNED（`pypto-lib/tests/step3p5/ci/LIVE_PRECISION_AB.md`）→ 状态改 ✅ → 在下方「更新日志」记一行（做了什么 / commit / 验证结果）。多步已含第一个 token；stall 用 `_probe_barrier_scale.py`（liveness，独立）。
5. 遇阻：状态改 ⛔，「阻塞」列写原因，必要时在 [`../../blockers.md`](../../blockers.md) 登记。

---

## 更新日志

| 日期 | ID | 变更 | 备注 |
|------|----|----|------|
| 2026-07-24 | — | 专项建档，12 个子任务初始化为 TODO | 对照 v4-flash `decode_fwd.py` 拆分；HLD/LLD 见同目录 |
| 2026-07-24 | — | 验证标准改为**单一多步 decode**（N=128 ≥95% vs vanilla）；删除单步/单 token 单列（多步已含首 token） | 采纳用户口径；stall 用探针独立判定 |
| 2026-07-24 | G1 | 新增：调度轴 batch→experts/feature + dynamic active-token（对齐 DeepSeek） | 源自 batch/SPMD 分歧调研 |
| 2026-07-24 | A1 | 🟦 代码接线：holder.run() 的 N1_DFX 扩到 swim/l2/pmu（+N1_PMU）+ perf-baseline.md 骨架 | b-csy-develop 无 NPU；device 采集待 0162/镜像 |
| 2026-07-24 | A1 | ✅ 完成：镜像(stepfun-develop-20260724, cards 8-15)跑 DFX，逐层拆解 = routed-expert 90.7% / PMU cube_int8 88.6%；结果并进 benchmark/2026-07-24-step3p5-decode-perlayer-dfx.md；DFX 接线进 stepfun/develop | 基底改为镜像(feat 分支已废)；total-step 基线复用 benchmark 590ms |
