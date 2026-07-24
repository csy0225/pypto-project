# step3p5 decode 逐层/逐kernel DFX 拆解（PERF-A1）

**日期**：2026-07-24 ｜ **机器**：gpu-a910x-0162（cards 8-15；0-7 为 vanilla oracle 未动）
**镜像**：`hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260724`
（pypto `ca21ab5f` / pypto-lib `fd26b1be` / ptoas v0.50）
**被测**：`whole_decode_faithful_real_single_chip_hidden_only`（45 层 → hidden，**无 lm_head**），
TP=8、W8A8、active batch=1、`--num-blocks 512`、`--steps 1`。
**采集**：`_stage_main_hidden_only --dfx swim` / `--dfx pmu`（PERF-A1 新接线，见文末代码）。
运行健康：rc=0、`MAIN_HIDDEN_ONLY_..._TOKEN_EXACT`、`hidden_finite=true`、TP `spread=0.0`。

> 绝对单步延迟以权威 benchmark 为准：[`2026-07-23-step3p5-decode-64k-itl.md`](2026-07-23-step3p5-decode-64k-itl.md)
> （64k device-KV **≈590 ms/step** raw `rt.run`，含 lm_head）。本文补的是**那 590ms 摊到哪些 kernel**。

---

## 1. 结论（两个都对、但答的是不同问题）

**必须区分"算力花在哪"和"墙上时间花在哪"——本次数据两者结论相反：**

| 口径 | 谁占大头 | 含义 |
|------|---------|------|
| **算力 / FLOPs**（busy-µs 按核求和；PMU exec） | **routed-expert 90.8%**（`cube_int8_exec` 88.6%） | 计算量几乎全在 MoE routed expert 的 INT8 矩阵乘 |
| **墙上时间 / 延迟**（union-of-intervals，时间轴占用） | **`tp_all_reduce` 74.1%**（expert 仅 14.8%，dispatch 13.6%） | 一个 decode step 的**耗时**主要花在 TP all-reduce 通信上 |

**为什么相反**：expert 矩阵乘是**计算密集**（~25 核并行、墙上时间短，busy/wall 并行度=24.7）；`tp_all_reduce` 是**少核、长时延**（178 次调用、每次 ~9ms 级、在时间轴上跨 74%）= **通信/延迟-bound**。→ **对"降 decode 延迟"而言，通信（tp_all_reduce + dispatch）才是主瓶颈，不是 expert 算力。** 我第一版只报了通信的 busy 份额（6%），**低估了它的 wall 份额（74%）——已更正。**

> ⚠⚠ **强 caveat（未定论，需 clean-run 确认）**：本数据来自 **DFX 插桩的 swim run**，makespan≈1120ms ≫ clean 590ms（benchmark 2026-07-23）。绝对 ms 不可信；**份额**是信号，但通信/等待类算子恰是插桩最易放大的。union-of-intervals 家族份额相加 >100%（家族间时间轴重叠）。**下结论/动手前必须用未插桩的 clean run 复核通信 wall 份额**（falsify-before-assert）。

## 1b. 墙上时间 vs 算力（rank0，数据见 [`data/2026-07-24_step3p5_perlayer_dfx/wall_vs_busy_rank0.csv`](data/2026-07-24_step3p5_perlayer_dfx/wall_vs_busy_rank0.csv)）

| 家族 | wall 份额 | ~ms of 590ms* | busy 份额 |
|------|----------|---------------|-----------|
| comm.tp_all_reduce | **74.1%** | ~437 | 6.0% |
| moe.expert_routed | 14.8% | ~87 | **90.8%** |
| moe.dispatch | 13.6% | ~80 | 1.0% |
| (other/orch) | 7.6% | ~45 | 0.6% |
| moe.gate/topk | 0.8% | ~5 | 0.2% |
| attn.o_proj/head_gate | 0.6% | ~3 | 1.1% |
| 其余（shared/rope/qkv/rmsnorm/flash/combine/dense） | 各 <0.6% | 各 <4 | 各 <0.2% |

\* ms 列 = wall 份额 × 590ms，仅**量级示意**（份额来自插桩 run，且份额相加>100%）。真实 ms 待 clean-run 复核。

## 2. Swimlane 家族占比 —— 算力/FLOPs 视角（rank0，busy-µs 按核求和）

> 这是"算力花在哪"（≠ 墙上时间；wall 视角见 §1b）。全表 CSV：[`data/2026-07-24_step3p5_perlayer_dfx/family_rollup_rank0.csv`](data/2026-07-24_step3p5_perlayer_dfx/family_rollup_rank0.csv)。

| 家族 | busy 占比 |
|------|----------|
| **moe.expert_routed（gate_up + down）** | **90.7%** |
| comm.tp_all_reduce | 6.0% |
| attn.o_proj / head_gate | 1.1% |
| moe.dispatch | 1.0% |
| moe.gate/topk | 0.2% |
| moe.shared | 0.1% |
| attn.qkv_proj | 0.1% |
| attn.flash/core | 0.1% |
| attn.rope / rmsnorm / dense_mlp / combine | 各 ≈0.0% |

## 3. Top kernel（rank0，busy-µs）

| 占比 | 次数 | kernel |
|------|------|--------|
| 35.4% | 46240 | `swa_moe...expert_gate_up_aiv_spmd`（**向量**：dequant/SwiGLU/requant） |
| 17.7% | 23120 | `swa_moe...expert_gate_up_aic_spmd`（**cube**：INT8 matmul） |
| 12.2% | 15920 | `full_moe...expert_gate_up_aiv_spmd` |
| 7.4% | 73984 | `swa_moe...expert_down_aiv_spmd` |
| 6.1% | 7960 | `full_moe...expert_gate_up_aic_spmd` |
| 6.0% | 178 | `tp_all_reduce` |
| 3.6% | 36992 | `swa_moe...expert_down_aic_spmd` |
| … | | 其余每项 <2.5%（dispatch / gate_matmul / sh_mlp / out_proj / rope …） |

> swa_moe > full_moe 只是因为层数多（30 swa-MoE vs 10 full-MoE）。

## 4. PMU（rank0，exec 单元活动）

| exec counter | 占比 |
|--------------|------|
| **cube_int8_exec** | **88.6%** |
| vec_fp32_exec | 7.7% |
| cube_fp16_exec | 2.1% |
| vec_misc / vec_int32 / vec_fp16 | 合计 <1.7% |

total_cycles 按 core_type：cube 核 67.4% / 向量核 32.6%。

## 5. 关键解读（swimlane × pmu 交叉）

- **cube 侧**：`cube_int8_exec` 独占 88.6% 的 exec 活动 → routed-expert 的 **INT8 matmul 是真正的 FLOP 大头**，且已是 INT8（cube 效率 OK）。
- **向量侧的悖论**：`expert_gate_up_aiv`（向量）busy-µs（35.4%）≈ `aic`（cube，17.7%）的 2×，**但**向量 exec counter 很低（vec_fp32 仅 7.7%）→ 这些向量任务**墙上时间长、exec 密度低 = 访存/stall-bound**（围着 INT8 matmul 的 dequant/SwiGLU/requant 在等数据），不是算力瓶颈。

## 6. 对优化专项的指向（ties to design/performance）

> 按 **wall（延迟）** 优先，不是 busy。前提：先用 clean run 复核 §1 caveat 里的通信 wall 份额。

| 观察（wall 视角） | 指向的优化项 | 预期 |
|------|-------------|------|
| **`tp_all_reduce` 占 wall 74%** — 少核长时延，通信-bound | **C 系最高优先**：**C1**（单 window set + epoch，减同步/等待）、**C3**（peer loop→spmd 并发）、**C2**（push→pull 减跨 die 写等待） | 直接砍 decode 延迟主项 |
| **dispatch 占 wall 13.6%** | **C2/C3**（dispatch 通信并发化 / pull） | 次大延迟项 |
| routed-expert 占 **busy 90.8%** 但 wall 仅 15% | **D2/F 系**（INT8 expert dequant/SwiGLU 融合）——影响**吞吐/算力**、对单 token **延迟**收益有限 | 提吞吐，非降延迟首选 |
| expert 向量半 stall-bound（busy 高 exec 低） | F2（pipeline stage / MTE 512B）、融合 dequant | 提 aiv 利用率（吞吐） |
| active batch=1 但 expert 跑 padded 容量（**待证伪**） | PERF-G1（dynamic active-token） | 主要利好吞吐；wall 上 expert 本就只 15% |
| attention（qkv/flash/rope/rmsnorm）wall+busy 均 <1% | N=1 低 ROI，暂不动（F1 缓做） | — |

**一句话修正**：之前"优化 ROI 全在 expert"是**算力视角**；就**单 token decode 延迟**而言，**通信（C 系）才是首要**，expert 优化（D/F）主要提吞吐。两者都要做，但降延迟先攻 C。

## 7. 复现

```bash
# 镜像内，cards 8-15；--dfx swim 收 l2_swimlane，--dfx pmu 收 pmu.csv（分开跑，各自扰动计时）
python -m tests.step3p5.harnesses._stage_main_hidden_only \
  --device 8,9,10,11,12,13,14,15 --ckpt $CKPT --out /tmp/n1_dfx \
  --num-blocks 512 --steps 1 --dfx swim   # 再 --dfx pmu --pmu 1
# artifact: {PYPTO_PROG_BUILD_DIR}/<prog>_*/dfx_outputs/rankN/d0/{merged_swimlane_*.json, l2_swimlane_records.json, pmu.csv}
# 注意：镜像 ENV PYPTO_PROG_BUILD_DIR=/tmp/pypto_build_output，需 -v 挂到宿主否则 --rm 丢失
```

**代码接线**（PERF-A1，待进 stepfun/develop → 下个镜像）：
- `tools/step3p5/whole_decode_holder.py::run()`：`N1_DFX` 扩到 `swim`/`l2`（`enable_l2_swimlane`）+ `pmu`（`enable_pmu=int(N1_PMU)`）。
- `tests/step3p5/harnesses/_stage_main_hidden_only.py`：新增 `--dfx` / `--pmu`，设 `N1_DFX`/`N1_PMU` 环境变量。

## 8. 原始数据 / artifacts

- **committed 派生数据**（本仓，机器可读，backing 上面表格）：[`data/2026-07-24_step3p5_perlayer_dfx/`](data/2026-07-24_step3p5_perlayer_dfx/)
  - `kernel_busy_us_rank0.csv`（190 个 kernel 全量 busy-µs）
  - `family_rollup_rank0.csv`（家族 busy 份额）
  - `wall_vs_busy_rank0.csv`（**wall vs busy 对照**，§1b/§6 依据）
  - `pmu_rollup_rank0.txt`（cube/vec exec + core_type cycles）
- **原始 trace（未入仓，太大）**：rank0 `merged_swimlane` 366MB / 全 8 rank 1.3GB。
  归档：`gpu-a910x-0162:/data/chensiyu/perf_a1/raw_dfx_traces_20260724.tar.gz`（68MB gz）；
  在线目录 `gpu-a910x-0162:/data/chensiyu/perf_a1/build_output/`。
  可视化：把 `merged_swimlane_*.json` 拖进 https://ui.perfetto.dev/ 。
  > 未入 git：单文件 366MB 会撑爆仓库（本 `benchmark/data/` 约定是 KB 级摘要）。需要原始 JSON 时从 0162 归档取。
