# 任务跟踪记录 — Performance 专项

> step3p5 decode 性能优化的**单一事实源**。每个子任务的状态、owner、更新时间、阻塞在此维护。
> 设计详情见 [`02-detailed-design.md`](02-detailed-design.md)；改状态时同步更新本表的「最后更新」列 + 底部更新日志。
>
> **状态图例**：⬜ TODO ｜ 🟦 IN PROGRESS ｜ ✅ DONE ｜ ⛔ BLOCKED ｜ ⏸ PAUSED

---

> **⚠ 2026-07-24 base 校正**：本专项**唯一正确 base = 最新 `stepfun/develop @ bc5eecb1`**（fork csy0225；origin 无此分支）。
> LIVE 整网 = **手写维护的 `models/step3p5/decode_layer_single_chip_hidden.py`**（hidden-only；45× unroll `whole_chip_orch`）。
> commit `759c23e8 "prune to single-chip vllm integration"` 已**删除** `decode_layer.py`(−31686)、旧 generator（`_gen_faithful_real.py`/`_gen_single_chip_real.py`）、`decode_fwd/mtp/step3p5_decode`、canonical §5 round-trip 工作流。
> **本文档与 [`02-detailed-design.md`](02-detailed-design.md) 中所有 `decode_layer.py` / `_gen_*` / generator / round-trip / `3af13f4f` 引用均已失效**，落点一律改为 `decode_layer_single_chip_hidden.py`。
> 对账：A1 / C2 / B1 / SwiGLU-per-layer 已随 single_chip_hidden 交付；B/C 唯一确定剩余 = **C1**（+ 用户批准的 B2）。

## 看板（按 Track）

### Track A — 可观测性
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| A1 | whole-net baseline + DFX 采集 | P0 | ✅ | claude | — | 逐层 DFX 拿到：routed-expert 占 90.7%（PMU cube_int8 88.6%）；见 benchmark/2026-07-24 | 2026-07-24 |

### Track B — Mega-kernel 结构
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| B1 | 权重 stacking + `resident="stacked"` | P0 | ✅ | — | — | single_chip_hidden 已交付：`StackedDeviceTensor`+一次 IPC+`child_memory` 省 24GB/rank/step H2D；层内 `pl.slice([mlp_layer_idx*…])` dynamic-offset 已设备验证（commit 8b4bf3fa）。`resident=` IR 属性 codegen 不读=纯文档，不加 | 2026-07-24 |
| B2 | 45 层 unroll → `pl.range` 循环 | P1 | 🟦 | b1-weights | C1 | 用户批准。现状：`decode_layer_single_chip_hidden.py::whole_chip_orch` 45× unroll，层 body 已参数化（norm/attn/mlp layer_idx scalar）。真难点=45 层用 6 个 `*_chip_orch`（SwiGLU-per-layer 特化），pl.range 无法按 runtime scalar 静态选方法→需 enum 参数化统一。收益=编译期 IR/调度边（间接），非直接 decode 计算。C1 落后再折循环 | 2026-07-24 |
| B3 | KV pool `resident` + in-place | P2 | ⬜ | — | B1 | 旁证已大部分交付（KV IPC resident + `add_inout`，whole_decode_holder.py `build_stacked_kv_pool`）。待核验每 step 只写 1 行。C1/B2 后核验 | 2026-07-24 |

### Track C — MoE 通信协议
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| C1 | 单 window set + `moe_epoch` + `WaitCmp.Ge` | P0 | 🟦 | c1-comm | — | **B/C 唯一确定剩余 P0**。落点 `decode_layer_single_chip_hidden.py`（非 generator）：16 MoE stack(4857-4914)缩单套 + 45 unroll 站点 slice offset→`[0,0]` + moe_epoch(1→42) + wait expected 常量(701/737/798/1323/2316/2385/2486)→moe_epoch(Ge) + anchor read。**风险=撤 shipped 的 per-layer 隔离信号(0e627a4e)**，设备 6 轮 liveness gate 先验；退路 double-buffer。730MB→~17MB | 2026-07-24 |
| C2 | dispatch push → pull（fixed-slot） | P1 | ✅ | — | — | 已在 single_chip_hidden 完成（commit 42ac1ffd；dispatch=pull `remote_load`/TGET+AtomicAdd/Ge，combine=pull）。死代码 `_push_routed_y_to_sources`(含 NotifyOp.Set) C1 顺手清 | 2026-07-24 |
| C3 | peer loop → `pl.spmd`/`pl.parallel` | P2 | ⬜ | — | C1 | 等 C1。dispatch/combine peer 循环 fan-out | 2026-07-24 |

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
| ⬜ TODO | 9 |
| 🟦 IN PROGRESS | 2 |
| ✅ DONE | 3 |
| ⛔ BLOCKED | 0 |
| **合计** | **14** |

**base 校正后关键路径**：A1/C2/B1 已 ✅（single_chip_hidden 交付）。当前推进：**C1（🟦 c1-comm）→ 设备 6 轮 liveness gate → 多步精度 → B2（🟦 b1-weights，折循环+6-method enum 统一）**。
**下一波可认领**：C3/B3（C1 后）、D1（独立 INT8-native）、F1（独立）。

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
| 2026-07-24 | — | **base 校正**：唯一正确 base = 最新 `stepfun/develop @ bc5eecb1`（fork csy0225）。旧文档基于 pre-prune `3af13f4f`/faithful_real/generator，均失效 | commit 759c23e8 删 decode_layer.py+generator+§5 round-trip；LIVE=手写 decode_layer_single_chip_hidden.py |
| 2026-07-24 | C2 | ✅ 对账确认：single_chip_hidden dispatch/combine 已 pull（commit 42ac1ffd），非本 session 新增 | canonical release 已过多步精度 bar |
| 2026-07-24 | B1 | ✅ 对账确认：StackedDeviceTensor+IPC+child_memory+dynamic-offset slice(8b4bf3fa) 已交付；resident= IR 属性纯文档不加 | 剩 dynamic-offset 归 B2 |
| 2026-07-24 | C1 | 🟦 c1-comm 启动实现（bc5eecb1/single_chip_hidden）：单窗口+moe_epoch+Ge+anchor。撤隔离信号=device-risky revert，设备 6 轮 liveness gate 先验 | 本机无 pypto，验证全在 0162 |
| 2026-07-24 | B2 | 🟦 用户批准。现状 45× unroll，body 已参数化；真难点=6 个 *_chip_orch enum 统一。C1 后折循环 | 收益=编译期 IR/调度边（间接） |
