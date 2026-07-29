# 2026-07-29 · 发布镜像 64k 实测：DFX 拆解 + ITL 端到端

| 字段 | 值 |
|------|----|
| **镜像** | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260729-allreduce-push`（digest `sha256:7924925f…`） |
| **pin** | pypto `6933b1aa` / pypto-lib `cfbdcce8` / pto-isa `ecb6c303` / PTOAS `fc8c6cae` / simpler `8459d60f` / ptoas-bin `v0.50` |
| **机器** | gpu-a910x-0162，cards 8–15（0–7 为 vanilla oracle，未动） |
| **被测** | `models.step3p5.decode_fwd:whole_decode_step3p5`（45 层 → hidden，**无 lm_head**），TP=8、W8A8 |
| **工作点** | `--num-blocks 512`、ctx=65536、active_batch=1（另有 batch 扫描见 §2.2） |
| **KV** | `WholeDecodeHolder(kv_ipc=True)` —— **真 device KV via IPC**，`block_table` 在 `--num-blocks 512` 下寻址满 64k，attention 真实遍历 64k KV；**KV 内容未 prefill**（attention 计算量与 KV 内容无关） |
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

## 1.2 按 stage 拆（⚠ 占比读法见 §1.6，**不能直接当延迟占比**）

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

（wall 列相加 > 100%：不同 stage 在不同核上并发，各自的 union 会重叠。**attention 的份额被插桩放大**，见 §1.6(b)。）

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

## 1.6 怎么读上面的占比（**不能直接当延迟占比**）

### (a) 0724 的占比是 ctx≈1 的，不适用 64k

[`2026-07-24-step3p5-decode-perlayer-dfx.md`](2026-07-24-step3p5-decode-perlayer-dfx.md) 的结论是
「wall 口径 `tp_all_reduce` 占 74.1%、attention wall+busy 均 <1%、优先攻 C 系通信」。
在真实 ctx=65536 下重测，**顺序完全反过来**：

| | 0724（`--steps 1`，实际 ctx≈1） | 本次（ctx=65536） |
|---|---|---|
| `tp_all_reduce` wall | **74.1%** | **1.84%** |
| attention wall | **<1%** | 97.89% |
| routed expert busy | **90.7%** | 0.99% |

**原因**：0724 用 `--steps 1`，`--num-blocks 512` 只把 KV **池容量**开到 64k，
**实际 decode 位置在 ctx≈1**，attention 几乎不干活，于是 MoE/通信显得占满。
所以 0724 的「C 系优先、attention 低 ROI」**只适用短 context**。
这一条与插桩无关（两边都是插桩后的 span 占比），可以放心引用。

### (b) ⚠ 但 attention 的 97.89% 被插桩放大了，**不等于占延迟 97.89%**

拿 §2.1 的 ITL 曲线做一致性检查，会发现矛盾：

- 若 attention 真占 64k 实际延迟的 97.89%（= 81.8 ms），那 ctx=1024 时 attention 只剩
  ≈1.28 ms，整步应当 ≈3 ms；
- 但 ctx=1024 实测是 **70.2 ms**。**差 23 倍，说明 97.89% 不能当延迟占比读。**

原因是 instrumented span `435.205 ms` 是真实单步的 **5.21×**，而 **attention 占了全部 task 数的
56.1%（17697/31568）**——per-task 的打点开销按 task 数分摊，task 最多的 stage 吃到的插桩放大
也最多。所以插桩后的 span 占比对 attention 系统性偏高。

### (c) 现在能确定和不能确定的

**能确定**：
- 64k 单步 `83.5 ms` 中，**随 context 变化的部分 ≤13.3 ms（16%）**，其余 **≈70 ms（84%）
  与 context 无关**（§2.1 的硬约束）。
- `tp_all_reduce` 在 64k 已经很小（span 1.84% / busy 2.33%），**C 系继续优化的天花板很低**。
- routed expert 在 64k 也很小（busy 0.99%）——**D/F 系对 64k 单 token 延迟同样低 ROI**。
- `full_softmax` 是**单个 kernel 里 busy 最大的**（46.11%，5281 次 × 93 µs）；busy 对 span
  膨胀不敏感（是各核 busy 求和），所以它确实是最大的单点消耗者，只是**份额没到 93.9%**。

**不能确定**：那 **≈70 ms 的固定 floor 由什么构成**。当前这份 DFX 回答不了——插桩开销
本身就占了 span 的 4/5，把真实结构盖住了。

**下一步该怎么测**（而不是直接开工优化）：
1. **同一镜像做 `ctx=1024` vs `ctx=65536` 的 DFX A/B**：两次插桩开销相当，**相减**即可把
   context-dependent 部分（attention）与固定 floor 分离，不受插桩污染。
2. 固定 floor 若确认是 host↔device glue（`holder.run()` 含 `[8,16,4096]` D2H + metadata/python），
   那它属于集成层而非 kernel，优化路径完全不同。§3.1 显示这个 floor 从 0723 的 ≈635 ms 塌到
   ≈70 ms，**说明它历史上就是主项、且是可以被压下去的**。

> 结论一句话：**先把那 70 ms 的 floor 拆开，再决定动谁**。
> 「下一步优化 `full_softmax`」是当前数据下最强的候选，但**尚未被证明**是延迟主项。

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

**这条曲线给出一个硬约束**：context ×64（1024→65536）只涨 `+13.3 ms`（mean 70.196→83.529，
+18.8%）。attention 的工作量随 context 线性增长，所以**随 context 变化的那部分最多只占 64k
单步的 16.0%**；剩下 **≈70 ms（84%）是与 context 无关的固定开销**。
这一条会在 §1.6 用来校正 DFX 的占比读数。

## 2.2 active_batch 扫描（ctx=65536）

每行各占自己的 paged sequence，所以 `--num-blocks = 128 × active_batch`。

| active_batch | num_blocks | KV pool(K+V)/rank | p50 | mean | p99 |
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
2. **KV 内容未 prefill**：KV 内存是真的（`kv_ipc=True`，block_table 寻址满 64k），
   但内容不是真实 prefill 出来的。attention 计算量与 KV 内容无关，所以对计时无影响；
   但**不能**据此声称在线 serving 精度。
3. **与 0723 记录的 `654 ms/step` 是同口径可比的**（见 §3.1）。
4. **与 `≈590 ms raw rt.run` 不同口径**：那个数是 `..._single_chip`（**含 lm_head**）+
   只计 raw `rt.run()`，与本文 hidden-only + `holder.run()`（含 host glue）不可直接比。
   且其来源文档 `2026-07-23-step3p5-decode-64k-itl.md` **在本仓中不存在**，
   只有其他文档对它的引用 —— 溯源是待补项。
5. **DFX 的 435 ms 不是延迟**，见 §1.1；**DFX 的占比也不能直接当延迟占比**，见 §1.6。
6. 本文全部数据为**单侧实测**，不是与 baseline 镜像的 A/B。C4 的 A/B 只在
   ctx ≤ 4096 做过（[`2026-07-28-tp-allreduce-push.md`](2026-07-28-tp-allreduce-push.md) §2），
   **64k A/B 仍未采** —— 但 §3.1 说明为什么现在很值得采。

## 3.1 与 0723 记录的 `654 ms/step` 对比：**同口径，floor 塌了 7.9×**

[`deployment/docker/README.md`](../deployment/docker/README.md) §6 记录的 2026-07-23 实测与本文
**用的是同一个 harness、同一批 flag、同一台机器、同样 8 卡**：
`_stage_main_hidden_only --num-blocks 512 --itl-context-lens … --itl-iters 20 --itl-warmup 3`，
holder 都是 `kv_ipc=True` + hidden-only，计时都是 `holder.run()`。**唯一变量是代码。**

| context | 2026-07-23 (mean) | 本次 (mean) | 倍数 |
|---|---|---|---|
| 1024 | 635.3 ms | 70.196 ms | 9.05× |
| 32768 | 646.7 ms | 77.459 ms | 8.35× |
| **65536** | **654.0 ms** | **83.529 ms** | **7.83×** |
| 1k→64k 增量 | **+18.7 ms** | **+13.3 ms** | 相当 |

**关键观察：随 context 增长的那部分（+18.7 vs +13.3 ms）两次差不多，塌掉的是与 context
无关的固定 floor（约 635 → 70 ms）。** 所以这 7.9× 不是 attention/KV 变快，而是**每步的固定
开销消失了** —— 与期间落地的 B2 loop-form Main、C/D/G、C4 这类结构性改动相符。

⚠ **这是两次不同代码的观测，不是受控 A/B**，不能把 7.9× 归给某一个具体 commit。
要归因就跑 §2b：拿 `stepfun-develop-20260726-step3p5-only` 镜像用同一条 ITL 命令测一遍。
**（因此把 §2b 的预期从"C4 收益 <1%"上调 —— 那条只说 C4 自己；0726→本发布之间还含
C/D/G，A/B 会一并显现，值得采。）**

同时，0723 那份数据得出的「1k→64k 仅 +19 ms → 整网 decode **计算受限**，非 attention/KV 受限」
在**结论层面已被本次 DFX 推翻**：64k 下 attention 占 device span 97.9%。
「近平坦」是对的（+13 ms），但把平坦解释成"计算受限、attention 不重要"是错的 ——
真实原因是当时有一个巨大的、与 context 无关的固定开销把 attention 的增长淹没了。

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
