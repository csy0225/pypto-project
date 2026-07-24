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

## 1. 结论（一句话）

**decode 时间几乎全在 routed-expert 的 INT8 矩阵乘上（swimlane busy 占比 90.7%，PMU `cube_int8_exec` 占 exec 单元 88.6%）；attention / KV / tp_all_reduce / dispatch / combine / shared / dense 合计 <10%。** 优化 ROI 全在 MoE routed expert，attention/通信在 N=1 是次要项。

## 2. Swimlane 家族占比（rank0，busy-µs 按核求和）

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

| 观察 | 指向的优化项 | 预期 |
|------|-------------|------|
| routed-expert 占 ~90% | **PERF-D2**（INT8-native expert，已是 INT8 → 重点在 dequant/requant 链效率）+ **F 系**（融合 expert 的向量 dequant/SwiGLU，削 stall-bound 的 aiv 半） | 直接砍最大头 |
| expert 向量半 stall-bound（busy 高、exec 低） | **F2**（pipeline stage / MTE 512B）+ 融合 dequant 到 matmul 出口 | 提 aiv 利用率 |
| active batch=1 但 expert 在跑 padded 容量（**待证伪确认**：single_chip expert 是否按 padded `RECV_MAX` 迭代而非动态 nt） | **PERF-G1**（dynamic active-token）——若确为 padded，砍到真实路由数收益可能最大 | 需先在 IR 里确认 expert token 循环边界 |
| attention/KV/comm/dispatch/combine 各 <7% | 这些项在 N=1 decode 是**低 ROI**；attention 微调（F1）先不做 | 别过早优化 |

> ⚠ G1 那条是**假设**，动手前要在 `expert_routed` / `moe` 的 IR 里确认 expert 的 token 循环是 padded 容量还是动态 nt（falsify-before-assert）。

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

本次 artifact 留存：`gpu-a910x-0162:/data/chensiyu/perf_a1/build_output/`。
