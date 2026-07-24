# Performance 性能优化专项

> step3p5 decode 整网性能优化。对照 `origin/main` 上 deepseek **v4-flash** 的 mega-kernel 范式
> (`models/deepseek/v4-flash/decode_fwd.py`)，把当前"慢"的 whole-net 实现改造到位。
>
> **目标**：每个优化点都是**独立可落地的子任务**，便于团队并行认领。所有子任务的
> 状态在 [`task-tracking.md`](task-tracking.md) 里跟踪。

---

## 文档结构

| 文档 | 层级 | 用途 |
|------|------|------|
| [`01-system-design.md`](01-system-design.md) | HLD | 现状瓶颈实证 + mega-kernel 目标架构 + 四条并行主线 + 收益模型 |
| [`02-detailed-design.md`](02-detailed-design.md) | LLD | 每个子任务的 file:line、接口、算法步骤、验证口径、落地边界 |
| [`task-tracking.md`](task-tracking.md) | 跟踪 | 看板式任务跟踪记录（状态 / owner / 更新时间 / 阻塞） |
| [`user_prompt.md`](user_prompt.md) | 提示词 | 复制即用的推进/回归提示词（以 skill + 本目录为单一入口） |

---

## 一图速览：优化点主表

> 详细展开见 [`02-detailed-design.md`](02-detailed-design.md)；实时状态见 [`task-tracking.md`](task-tracking.md)。

| ID | 优化点 | Track | 优先级 | 收益 | 依赖 | 工作量 |
|----|--------|-------|--------|------|------|--------|
| **PERF-A1** | whole-net decode baseline + DFX 采集（l2_swimlane/PMU/perf_hints/mem-occupancy） | A 可观测性 | **P0** | 把盲调变有数，回归基线 | 无 | S (~1d) |
| **PERF-B1** | 权重 leading-dim stacking + `resident="stacked"` 一次性上传 | B Mega-kernel | **P0** | 去掉每 dispatch H2D，铺垫 B2 | 无 | M (~3d) |
| **PERF-B2** | 45 层 unroll → 单 `pl.range` 循环 + per-layer `pl.slice` | B Mega-kernel | **P1** | 31,636 行→数百行，编译崩塌、跨层复用 | B1, C1 | XL (多周) |
| **PERF-B3** | KV pool `resident` + in-place 更新 | B Mega-kernel | P2 | 省 KV 每步 D2H 往返 | B1 | S (~2d) |
| **PERF-C1** | 单 window set + `moe_epoch` 单调计数 + `WaitCmp.Ge` | C MoE 通信 | **P0** | 干掉 ~766MB per-layer 窗口 + 修 whole-net stall 根因 | 无 | M (~1w) |
| **PERF-C2** | dispatch push→pull（fixed-slot，对齐 moe.py） | C MoE 通信 | **P1** | 消除 A2 随机 507018 stall | C1 | L (~1-2w) |
| **PERF-C3** | peer loop `pl.range(N_RANKS)` → `pl.spmd`/`pl.parallel` | C MoE 通信 | P2 | peer 通信顺序→fan-out | C1 | S (~3d) |
| **PERF-D1** | gate deferred-norm + dispatch-side INT8 量化 | D INT8-native | **P1** | 省一遍 x 全量 pass；activation INT8 化 | 无 | M (~1w) |
| **PERF-D2** | routed expert INT8×INT8 + requant 链（gap-5） | D INT8-native | **P1** | 47GB/rank→~6GB/rank；cube 吃 INT8 | D1 | L (~2w) |
| **PERF-E1** | LM head 拆 4 段 decoupled worker + 复用 `recv_x_buf` | E LM head | P2 | publish 与末层 combine 重叠 + 省 HBM | C1 | M (~1w) |
| **PERF-F1** | attention `late_dep=task_dummy(deps)` 延迟 + `allow_early_resolve` | F L1/L0 微调 | P2 | kv_proj 落后 qr_proj 一拍重叠 | A1 | S (~2d) |
| **PERF-F2** | matmul pipeline stage 调优 + MTE 512B 对齐 | F L1/L0 微调 | P2 | 依 perf_hints 消 MTE 停顿 | A1 | S (~3d) |
| **PERF-F3** | RMSNorm+quant fused deferred-norm（dense/attn 复用） | F L1/L0 微调 | P2 | 融合一遍 norm pass | D1 | S (~2d) |
| **PERF-G1** | 调度轴 batch→experts/feature + dynamic active-token（对齐 DeepSeek） | G 调度轴/动态batch | **P1** | decode 核占用↑ + 不算 padding token | 与 B2 协同 | L (~2w) |

优先级：**P0** 零/低风险且解锁其它项，先做；**P1** 收益大的主体；**P2** 微调/收尾。
工作量：S ≤ 3d，M ≈ 1w，L ≈ 2w，XL 多周。

---

## 依赖图

```
A1 (baseline) ─── 独立，最先做 ; 解锁 F2
C1 (epoch窗口) ─┬─► B2 (pl.range 主体) ◄── B1 (stacked+resident)
                ├─► C2 (pull dispatch) ─► C3 (spmd peer)
                └─► E1 (decoupled LM head)
B1 ──► B3 (KV resident)
D1 (gate quant) ──► D2 (expert INT8) ; F3 随 D1
F1 独立 ; F2 需 A1
```

## 四条可并行主线（建议按 owner 分配）

| 主线 | 子任务链 | 说明 |
|------|---------|------|
| **① 结构线** | A1 → C1 → B1 → B2/B3 | mega-kernel 骨架，收益最大 |
| **② 通信线** | C2 → C3 | 修 stall，可与结构线并行（共享 C1 前置） |
| **③ 数值线** | D1 → D2 → F3 | INT8-native / HBM，完全独立 |
| **④ 微调线** | F1、E1、F2 | 重叠 & L1/L0，A1/C1 出结果后启动 |

**推进顺序建议**：先 **A1 + C1**（零/低风险、C1 同时修 stall 又解锁 B2）→ 铺 **B1 / D1** 两条独立线 → 汇聚到 **B2** mega-kernel。

---

## 验证标准（所有子任务通用）

**唯一精度准出 = 多步 decode 逐 token 精度**（多步已覆盖第一个 token，不再单列单步/单 token 测试）：
teacher-forced 对比 live vanilla vLLM W8A8 oracle，seed=6127 / N=128 → **ALIGNED ≥ 95%**
（当前 baseline 124/128=96.9%，miss 均为 vanilla 自身 near-tie）。
驱动：`pypto-lib/tests/step3p5/ci/{LIVE_PRECISION_AB.md, run_live_precision_ab.sh}`（`stepfun/develop`）。

> **stall / deadlock 是独立于精度的 liveness 检查**（不是精度判据）：`RUN_CLEAN` +
> 隔离探针 `_probe_barrier_scale.py` 判定是否 hang。任何"跑通/stall"结论用它，"精度不回退"用上面的多步 L3。

---

## 相关

- **回归 runbook（做完每个子任务按它回归）**：`.claude/skills/pypto-perf-regression/SKILL.md`
- 整网集成设计：[`../whole-net/README.md`](../whole-net/README.md)
- vLLM 共驻：[`../vllm-pypto/README.md`](../vllm-pypto/README.md)
- pypto kernel 坑：`pypto-lib/docs/known-pypto-pitfalls.md`
- perf 调优 playbook：`pypto-lib/docs/performance-tuning.md`
- 参考实现：`origin/main:models/deepseek/v4-flash/`（`git show` 读取）
