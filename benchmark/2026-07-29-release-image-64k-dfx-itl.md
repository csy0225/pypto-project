# 2026-07-29 · 发布镜像 64k 实测：DFX 拆解 + ITL 端到端

| 字段 | 值 |
|------|----|
| **镜像** | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260729-allreduce-push`（digest `sha256:7924925f…`） |
| **pin** | pypto `6933b1aa` / pypto-lib `cfbdcce8` / pto-isa `ecb6c303` / PTOAS `fc8c6cae` / simpler `8459d60f` / ptoas-bin `v0.50` |
| **机器** | gpu-a910x-0162，cards 8–15（0–7 为 vanilla oracle，未动） |
| **被测** | `models.step3p5.decode_fwd:whole_decode_step3p5`（45 层 → hidden，**无 lm_head**），TP=8、W8A8 |
| **工作点** | `--num-blocks 512`、ctx=65536、active_batch=1（另有 batch 扫描见 §2.2） |
| **KV** | harness 的 **dummy KV**（非 vLLM device-KV）——见 §3 口径边界 |
| **artifacts** | 0162 `/mnt/persist/chensiyu/workspace/benchmark/2026-07-29-v3-64k/`（1.3 GB，含 8 rank swimlane/pmu/scope）；rollup CSV 在 [`data/2026-07-29_release_image_64k/`](data/2026-07-29_release_image_64k) |

---

# 第一部分 · DFX 实测拆解

采集分三次独立跑（swimlane / pmu / scope 互相扰动计时，不能同跑）。下面数字取 rank0。

## 1.1 采集规模

| 项 | 值 |
|---|---|
| clock | 50 MHz |
| cores | 72（24 aic + 48 aiv） |
| aicore tasks | 31568 |
| **instrumented span** | **435.205 ms** |
| sum of task busy（跨核求和） | 1065.575 ms（aic 217.491 = 20.4%，aiv 848.084 = 79.6%） |

> ⚠ **`435 ms` 不是延迟**。swimlane 打点会让 onboard 把 kernel 跑两遍（dep_gen + timing pass）
> 并额外落盘，所以 instrumented span ≈ 真实单步的 5 倍。**同一工作点未插桩的真实 ITL 是
> `83.3 ms`（见第二部分）**。DFX 只用来看**占比**，绝对值一律以第二部分为准。

## 1.2 按 stage 拆（**两个口径结论一致**）

`wall` = union-of-intervals（时间轴被占用的比例，映射延迟）；`busy` = 各核 busy 求和（映射算力）。

| stage | wall(ms) | **wall % span** | busy(ms) | busy % | tasks |
|---|---|---|---|---|---|
| **attention core**（softmax / qk / sv / rope / qk_norm） | 426.038 | **97.89%** | 746.607 | 70.07% | 17697 |
| attention o_proj / resid | 17.162 | 3.94% | 174.666 | 16.39% | 8244 |
| dense MLP | 10.133 | 2.33% | 34.198 | 3.21% | 1602 |
| other | 9.811 | 2.25% | 29.281 | 2.75% | 838 |
| **TP all-reduce** | 7.989 | **1.84%** | 24.879 | 2.33% | 1547 |
| MoE gate/quant | 7.430 | 1.71% | 35.474 | 3.33% | 1085 |
| **MoE routed expert** | 1.955 | **0.45%** | 10.533 | 0.99% | 323 |
| MoE EP comm | 1.457 | 0.33% | 7.928 | 0.74% | 162 |
| rmsnorm | 1.371 | 0.32% | 2.008 | 0.19% | 70 |

（wall 列相加 > 100%：不同 stage 在不同核上并发，各自的 union 会重叠。）

## 1.3 Top kernel

| kernel | count | busy(ms) | busy % | avg(µs) | wall union(ms) | **wall % span** |
|---|---|---|---|---|---|---|
| **`full_softmax`** | 5281 | 491.351 | 46.11% | 93.0 | 408.838 | **93.94%** |
| `full_sv_matmul` | 5197 | 111.542 | 10.47% | 21.5 | 29.425 | 6.76% |
| `full_online_softmax` | 3640 | 75.659 | 7.10% | 20.8 | 18.547 | 4.26% |
| `full_out_proj_matmul_aic` | 2318 | 53.898 | 5.06% | 23.3 | 12.133 | 2.79% |
| `full_out_proj_matmul_aiv` | 1843 | 43.226 | 4.06% | 23.5 | 10.821 | 2.49% |
| `full_qk_matmul` | 2432 | 39.292 | 3.69% | 16.2 | 15.168 | 3.49% |
| `full_out_proj_cast` | 2098 | 37.218 | 3.49% | 17.7 | 9.566 | 2.20% |
| `tp_all_reduce` | 1547 | 24.879 | 2.33% | 16.1 | 7.989 | 1.84% |

全 306 个 kernel 见 [`kernel_busy_wall_rank0.csv`](data/2026-07-29_release_image_64k/kernel_busy_wall_rank0.csv)。

## 1.4 PMU（rank0，event_type=1）

| exec counter | 计数 | 占比 |
|---|---|---|
| `cube_int8_exec` | 15875200 | 46.36% |
| `vec_fp32_exec` | 8795341 | 25.68% |
| `cube_fp16_exec` | 8765810 | 25.60% |
| `vec_misc_exec` | 691612 | 2.02% |
| `vec_int32_exec` | 69290 | 0.20% |
| `vec_fp16_128lane_exec` | 47744 | 0.14% |

`pmu_total_cycles` 合计 `1.349e9`（31568 行）。

## 1.5 Runtime 资源余量（scope_stats，8 rank 取最大）

| 资源 | 容量 | 峰值占用 | 占比 |
|---|---|---|---|
| **ring heap** | 4 GiB / ring | 3.43 GiB | **79.9%** ⚠ |
| task window | 131072 | 1285 | 0.98% |
| dep pool | 131072 | 1508 | 1.15% |
| tensormap | 65536 | 173 | 0.26% |

`dropped=0`、`fatal=false`。**ring heap 只剩约 20% 余量**是本次唯一偏紧的资源；
task window / dep pool / tensormap 都在 1% 量级，不是瓶颈。

## 1.6 结论：64k 下的优化优先级与 0724 相反

[`2026-07-24-step3p5-decode-perlayer-dfx.md`](2026-07-24-step3p5-decode-perlayer-dfx.md) 的结论是
「wall 口径 `tp_all_reduce` 占 74.1%、attention wall+busy 均 <1%、优先攻 C 系通信」。
**本次在真实 ctx=65536 下测得的是反过来的**：

| | 0724（`--steps 1`，实际 ctx≈1） | 本次（ctx=65536） |
|---|---|---|
| `tp_all_reduce` wall | **74.1%** | **1.84%** |
| attention wall | **<1%** | **97.89%**（`full_softmax` 单个 93.94%） |
| routed expert busy | **90.7%** | 0.99% |

**原因**：0724 那次用 `--steps 1`，`--num-blocks 512` 只是把 KV 池开到 64k 容量，
**实际 decode 位置在 ctx≈1**，attention 几乎不干活，于是 MoE/通信显得占满。
attention 的工作量随 context 线性增长，到 64k 就压倒一切。
所以 0724 的「C 系优先、attention 低 ROI」**只适用于短 context**，不能当作 64k 的指导。

> **次要混淆项**：两次采集镜像不同（`20260724` vs 本发布镜像），中间落了 C/D/G + C4。
> 但 attention 随 context 线性增长是结构性的，ctx 是主因；要彻底坐实，可在**同一镜像**上
> 做 `ctx=1` vs `ctx=65536` 的 A/B。

**因此 64k 的下一个优化目标是 `full_softmax`**（占 span 93.9%、5281 次、平均 93 µs），
而不是 all-reduce 或 routed expert。C4 已经把 all-reduce 从 mesh 降到 reduce-scatter+push，
但它在 64k 只值 1.84% span —— 天花板已经很低。

---

# 第二部分 · ITL 端到端实测

`_stage_main_hidden_only --itl-context-lens …`，iters=20 / warmup=3，未插桩。

## 2.1 context 曲线（active_batch=1，`--num-blocks 512`）

| context | p50 | mean | p99 | min |
|---|---|---|---|---|
| 1024 | 70.177 ms | 70.196 ms | 70.597 ms | 69.898 ms |
| 8192 | 71.450 ms | 71.549 ms | 72.699 ms | 71.167 ms |
| 32768 | 77.522 ms | 77.459 ms | 78.116 ms | 77.043 ms |
| **65536** | **83.349 ms** | 83.529 ms | 84.658 ms | 82.902 ms |

数据：[`itl_curve_bs1.csv`](data/2026-07-29_release_image_64k/itl_curve_bs1.csv)。
1024→65536（context ×64）只涨 **18.8%** —— 单步仍由 context 无关的固定开销主导，
与 §1.5 的低核占用一致（延迟/依赖受限，不是算力受限）。

## 2.2 active_batch 扫描（ctx=65536）

每行各占自己的 paged sequence，所以 `--num-blocks = 128 × active_batch`。

| active_batch | num_blocks | dummy KV(K+V)/rank | p50 | mean | p99 |
|---|---|---|---|---|---|
| 1 | 512 | 1.45 GiB | 84.141 ms | 84.125 ms | 84.327 ms |
| 2 | 1024 | 2.85 GiB | 87.754 ms | 87.565 ms | 87.818 ms |
| 4 | 2048 | 5.67 GiB | 104.023 ms | 104.505 ms | 106.760 ms |
| 8 | 4096 | 11.29 GiB | 145.073 ms | 144.812 ms | 145.963 ms |
| **16** | 8192 | 22.54 GiB | — | — | — **device HBM OOM** |

数据：[`itl_active_batch_64k.csv`](data/2026-07-29_release_image_64k/itl_active_batch_64k.csv)。

- **bs=16 @ 64k 跑不起来**：`out of memory` → `simpler_init failed with code -1`。
  bs=8 峰值已到 ≈36 GiB/rank，bs=16 的 KV 再涨 ≈11 GiB，超出单卡可用。
  **不是 host shm 不足**（`--shm-size` 从 32g 提到 400g 行为不变）。
- bs=1→8 吞吐提升明显（延迟 ×1.72 换 batch ×8 ≈ **4.6× 吞吐**）。
- **口径**：harness 的 `BATCH=16` 是 **batch_capacity**（padding 容量），
  `active_batch` 才是真正参与计算的行数。历史 report 里的 `"batch": 16` 只表示 capacity。
  该字段已拆成 `batch_capacity` / `active_batch`（`pypto-lib@cc850ee5`）。
- bs=1 这里是 `84.141 ms`、§2.1 是 `83.349 ms` —— 同一工作点两次独立跑，差 ~1%，可作重复性参考。

---

## 3. 口径边界（引用本文数字前必读）

1. **无 lm_head**：被测是 45 层 → hidden。
2. **dummy KV，非 vLLM device-KV**：KV 由 harness 零填充构造，不是在线 paged KV。
3. **与 0723 的 `≈590 ms/step` 差约 7 倍且基准未对齐**。0723 记的是
   「64k device-KV ≈590 ms raw `rt.run`，含 lm_head」，本文是 83 ms hidden-only + dummy KV。
   候选差异（**均未验证**）：device-KV vs dummy-KV、含/不含 lm_head、
   期间落地的 C/D/G + C4。**在对齐之前，两个数字不得互为 before/after 引用。**
   注：0723 那份 benchmark 文档（`2026-07-23-step3p5-decode-64k-itl.md`）在本仓中不存在，
   只有其他文档对它的引用 —— 这条溯源本身也是待补项。
4. **DFX 的 435 ms 不是延迟**，见 §1.1。
5. 本文全部数据为**单侧实测**，不是与 baseline 镜像的 A/B。C4 的 A/B 只在
   ctx ≤ 4096 做过（[`2026-07-28-tp-allreduce-push.md`](2026-07-28-tp-allreduce-push.md) §2），
   **64k A/B 仍未采**。

## 4. 复现

```bash
# 镜像内，cards 8-15。三种 DFX 分开跑（互相扰动计时）
python -m tests.step3p5.harnesses._stage_main_hidden_only \
  --device 8,9,10,11,12,13,14,15 --ckpt $CKPT --out /out/dfx \
  --num-blocks 512 --itl-context-lens 65536 --dfx swim     # 再 --dfx pmu --pmu 1 / --dfx scope

# ITL context 曲线
... --num-blocks 512 --itl-context-lens 1024,8192,32768,65536

# active_batch 扫描（每档 num_blocks = 128 * active_batch）
... --num-blocks 4096 --itl-context-lens 65536 --active-batch 8

# ⚠ 镜像 ENV PYPTO_PROG_BUILD_DIR=/tmp/pypto_build_output，需 -v 挂到宿主，否则 --rm 丢 artifacts
```
