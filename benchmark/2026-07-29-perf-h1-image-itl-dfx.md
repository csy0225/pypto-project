# 2026-07-29 · PERF-H1 自包含镜像实测：ITL 端到端 + DFX 拆解

| 字段 | 值 |
|------|----|
| **镜像** | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260729-perf-h1`（registry digest `sha256:b4e8c8a457a5…`） |
| **pin** | pypto `1f704616` / pypto-lib `4513007d` / pto-isa `ecb6c303` / PTOAS `fc8c6cae` / simpler `e2efebcb` / ptoas-bin `v0.50` |
| **基线** | C4 发布镜像 `stepfun-develop-20260729-allreduce-push`（pypto `6933b1aa` / simpler `8459d60f`），digest `sha256:7924925f…` |
| **变更** | PERF-H1：retained CommDomain window 清零从 per-step host H2D reset 改为 device `aclrtMemset`（`_CTRL_MEMSET` + `broadcast_control_all` 8 卡并行）。**runtime-only，不动 kernel / 数值** |
| **机器** | gpu-a910x-0162，cards 8–15（0–7 未动） |
| **被测** | `models.step3p5.decode_fwd:whole_decode_step3p5`（45 层 → hidden，无 lm_head），TP=8、W8A8、`kv_ipc=True` |
| **工作点** | `--num-blocks 512`、active_batch=1；ITL iters=20 / warmup=3；DFX ctx=65536 |
| **artifacts** | 0162 `/mnt/persist/chensiyu/workspace/benchmark/2026-07-29-perf-h1/`（`dfx/itl_curve/itl_report.json`、`dfx/{swim,pmu,scope}` + `dfx/dfx_rollup.json`、`n256_*`、`whole_net_ci_v4`） |

---

## 1. ITL 端到端（headline）—— device-memset 全曲线降 23–27%

`_stage_main_hidden_only --num-blocks 512 --itl-context-lens 1024,8192,32768,65536 --itl-iters 20 --itl-warmup 3`，未插桩。

| context | **H1 p50** | H1 mean | C4 发布镜像 p50 | Δ p50 |
|--------:|:-:|:-:|:-:|:-:|
| 1024 | **50.919 ms** | 50.984 | 70.177 | **−27.4%** |
| 8192 | **51.980 ms** | 51.995 | 71.450 | **−27.2%** |
| 32768 | **58.034 ms** | 58.272 | 77.522 | **−25.1%** |
| **65536** | **64.105 ms** | 64.215 | 83.349 | **−23.1%** |

- 与 PERF-H1 host-workspace A/B（`85.02→65.55 ms`，[`2026-07-29-host-window-memset.md`](2026-07-29-host-window-memset.md)）一致：镜像内 64k p50 `64.1 ms`。
- context ×64（1024→65536）增量 `+13.2 ms`（+25.9%），与 C4 曲线 (`+13.3 ms`) 相当 —— **随 context 变化的部分不变，塌下去的是 context 无关的固定 floor**（正是 memset 消掉的 per-step host reset 21.5→2.2 ms）。

---

## 2. 精度 / 等价回归（同镜像）

| gate | 结果 | 判定 |
|------|------|------|
| 整网 CI（Main 45 层 8 步 + MTP45/46/47 single & batch16） | `ok=true`，Main token `303,1207,19384,872,428,6127,4231,2636` exact；MTP token `6178,410,303` exact；`hidden_tp_spread=0` | **PASS** |
| N=256 teacher-forced，H1 vs C4 发布镜像 | **token 256/256 exact**（含 step127/128/255 跨 block 边界），全步 finite | **PASS** |
| N=256 raw-hidden byte-exact | H1-vs-C4 max_abs_diff 44、H1a-vs-H1b max_abs_diff 34（同量级） | 非 gate：run-to-run 抖动 = C4 push all-reduce 归约顺序，**非 H1 回归** |

> raw-hidden 逐 run 抖动是 PERF-C4（push all-gather，postmortems/13）引入的浮点归约顺序不定，在 C4 和 H1 上同时存在；argmax（token）不受影响，256/256 稳定。PERF-H1 的 device-memset 对数值零影响。

---

## 3. DFX 拆解（ctx=65536，rank0；`dfx/dfx_rollup.json`）

> 采集分 swim / pmu / scope 三次独立跑（互相扰动计时）。**DFX span ≠ 延迟**（插桩放大 + 多 iter 累加），只用于看结构/占比；绝对延迟一律以 §1 为准。

### 3.1 PMU（exec counter，31568 行，event_type=1）

| exec counter | 计数 | 占比 |
|---|---|---|
| `cube_int8_exec` | 15 875 200 | **46.35%** |
| `vec_fp32_exec` | 8 795 341 | 25.68% |
| `cube_fp16_exec` | 8 769 876 | 25.61% |
| `vec_misc_exec` | 691 612 | 2.02% |
| `vec_int32_exec` | 69 290 | 0.20% |
| `vec_fp16_128lane_exec` | 47 744 | 0.14% |

`total_cycles=1.293e9`。与 C4 发布镜像（`cube_int8 46.36%` / `vec_fp32 25.68%` / `cube_fp16 25.60%`，`1.349e9`）**逐项一致** —— 证实 memset 不改计算 profile，W8A8 INT8 cube 仍是主力。

### 3.2 Runtime 资源余量（scope_stats，8 rank 取峰值）

| 资源 | 容量 | 峰值 | 占比 |
|---|---|---|---|
| **ring heap** | 4 GiB / ring | 3.431 GiB | **79.9%** ⚠ |
| task window | 131072 | 1285 | 0.98% |
| dep pool | 131072 | 1495 | 1.14% |
| tensormap | 65536 | 173 | 0.26% |

`dropped=0`、`fatal=false`。与 C4 一致：**ring heap ~20% 余量仍是唯一偏紧资源**（memset 不改任务图，符合预期）。

### 3.3 Swimlane（span 96.109 ms，DFX warmup 已丢弃冷启动 run）

harness 现在**在 ITL warmup 期间关闭 DFX**（`_stage_main_hidden_only._run_itl` pop `N1_DFX`/`N1_PMU`，pypto-lib `80eb8a9e`），只在 warm 稳态 run 上采集，所以 swimlane 不再被冷启动 skew 污染。效果：

| 采集方式 | swim span | top `tp_all_reduce` |
|---|---|---|
| warmup=1、采集含冷启动（旧） | 373.878 ms | 636.86 ms（2×318.43，第一个 barrier 吸收 rank 启动 skew） |
| **warmup=3、DFX 丢弃 warmup（新，本文）** | **96.109 ms** | **81.55 ms（2×40.77，DFX 每 pass re-sync 残留）** |

- 每步真实 `tp_all_reduce` 中位数 **0.051 ms**；顶部 2 条 40.77 ms 是 **DFX 双 pass（dep_gen+timing）每次 instrument run 的首 barrier re-sync 残留**，非 steady-state（un-instrumented ITL p99 仅 66.05 ms 已证实每步无此开销）。
- 剩余 per-kernel 真实 busy 由 `*_expert_gate_up_aiv_spmd`（MoE，各 ~14–16 ms 累加）与 attention 组成。
- **口径**：冷启动 skew 是分布式多进程首次同步的一次性成本（干净测约 2.9 ms，见 [`2026-07-29-host-window-memset.md`](2026-07-29-host-window-memset.md) H2），**每 session 一次、非每 token**；serving 侧由 vLLM 启动 warmup 吸收。要干净归因 attention vs floor 仍需 ctx=1024 vs 65536 的 DFX A/B 相减（见 [`2026-07-29-release-image-64k-dfx-itl.md §1.6`](2026-07-29-release-image-64k-dfx-itl.md)）。

---

## 4. 复现

```bash
IMG=hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260729-perf-h1
CKPT=/data/chensiyu/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp
# ITL 曲线（cards 8-15，PYPTO_PROG_BUILD_DIR 需挂宿主否则 --rm 丢 DFX 产物）
python -m tests.step3p5.harnesses._stage_main_hidden_only \
  --device 8,9,10,11,12,13,14,15 --ckpt $CKPT --out /out \
  --num-blocks 512 --itl-context-lens 1024,8192,32768,65536 --itl-iters 20 --itl-warmup 3
# DFX（三段分开）: 追加 --itl-context-lens 65536 --dfx swim / --dfx pmu --pmu 1 / --dfx scope
# 整网 CI + MTP（MTP 需挂 oracle 目录 + --mtp-oracle-dir，见 whole_net_ci_v4）
python -m tests.step3p5.ci.run_whole_network_ci --ckpt $CKPT \
  --devices 8,9,10,11,12,13,14,15 --mtp-oracle-dir /oracle --out /tmp/n1_ci --artifact-dir /out
```

## 5. 口径边界

1. **无 lm_head**：被测 45 层 → hidden。
2. **KV 内容未 prefill**（`kv_ipc=True`，block_table 寻址满 64k，但内容非真实 prefill）——计时无影响，不能据此声称在线 serving 精度。
3. **DFX span 不是延迟**，占比也不能直接当延迟占比（见 §3.3）。
4. **live N=128 vanilla-raw 精度门本轮未跑**（oracle 未起）；因 H1 与 C4 token 256/256 一致，其 vanilla-raw 与 C4 的 `240/256` 等价。
5. MTP gate 为自洽 kernel 正确性校验（喂 oracle 配对输入）；MTP 端到端在线对齐属 Phase 28。
