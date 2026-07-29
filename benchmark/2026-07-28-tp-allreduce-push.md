# 2026-07-28 · TP all-reduce: onephase mesh → reduce-scatter + push all-gather

| 字段 | 值 |
|------|----|
| **子任务** | [`task-tracking.md`](../design/performance/task-tracking.md) `C4` |
| **被测对象** | `models.step3p5.decode_fwd:whole_decode_step3p5`（canonical Main，45 层 × 2 = 90 个 all-reduce 调用点） |
| **机器 / 镜像** | 0162，cards 8–15。§2/§3 的数据采自 `step3p5-b404a3c9-ci-final-20260728`（⚠ 该 candidate 工作树 dirty，带未提交 simpler 补丁，见 [`postmortems/14`](../postmortems/14-image-dirty-worktree-unreproducible-pins.md)）；§2b 采自已发布镜像 `stepfun-develop-20260729-allreduce-push` |
| **对照方式** | 同镜像、同卡、背靠背；唯一变量 = 挂载 `decode_fwd.py` + `dense_mlp.py`（baseline 用镜像内原文件，md5 `4d671a5e`；改后 md5 `653f29cd`） |
| **代码** | `csy0225/pypto-lib` `perf/step3p5-bc-20260726` @ `408b343d` |

## 1. 结论

> **⚠ 适用范围（2026-07-29 更正）**：下面 §2 的 −3.6%/−3.9% 是 **ctx ≤ 4096** 工作点的数字
> （`--num-blocks 32` 上限 4096）。项目的**权威 ITL 工作点是 64k**（`--num-blocks 512`）。
> all-reduce 的绝对开销基本不随 context 变化（只搬 `[16,4096]` hidden），而 64k 下单步分母更大，
> 因此 **64k 的相对收益必然显著小于 −3.8%**。已由 64k DFX 坐实：`tp_all_reduce` 在 ctx=65536
> 只占 device span **1.84%**（见 §2c 指向的文档），**所以 §2 的数字不得当作发布收益引用**。

整网单步 decode 延迟：**ctx ≤ 4096 降低约 3.6–3.9%**；64k 见 §2b。无精度回归
（bit-identical 设计 + 设备验证）。

## 2. ITL @ ctx ≤ 4096（inter-token latency，`_stage_main_hidden_only --itl-context-lens`，iters=20 / warmup=3）

| context | 指标 | baseline (onephase mesh) | twophase + push AG | 变化 |
|---|---|---|---|---|
| 1024 | p50 | 68.397 ms | **65.910 ms** | **−3.64%** |
| 1024 | mean | 68.437 ms | 65.856 ms | −3.77% |
| 1024 | p99 | 68.849 ms | 66.202 ms | −3.84% |
| 4096 | p50 | 69.195 ms | **66.510 ms** | **−3.88%** |
| 4096 | mean | 69.215 ms | 66.575 ms | −3.81% |
| 4096 | p99 | 70.160 ms | 67.570 ms | −3.69% |

工件：`0162:/mnt/persist/chensiyu/ar-twophase/itl-{base,twophase}/itl/itl_report.json`

## 2b. ITL @ 权威工作点（`--num-blocks 512`，ctx 1024/8192/32768/65536）

**仍未采集。** 计划对比双方均为已发布 immutable 镜像（无挂载 overlay）：
`stepfun-develop-20260726-step3p5-only`（pypto-lib `53eb7212`，无本改动）
vs `stepfun-develop-20260729-allreduce-push`（pypto-lib `cfbdcce8`）。

发布镜像单侧的 64k 数据已有（见 §2c），缺的是 baseline 侧。
按 §2c 的 DFX 拆解，`tp_all_reduce` 在 64k 只占 span 1.84%，
所以这个 A/B 的预期收益是**小于 1%**，优先级不高。

## 2c. 64k 的单侧实测与 DFX 拆解

发布镜像在 ctx=65536 的 ITL 曲线、active_batch 扫描（bs≤8 可跑、bs=16 撞 device HBM）
和 DFX 逐 kernel 拆解，见独立文档
[`2026-07-29-release-image-64k-dfx-itl.md`](2026-07-29-release-image-64k-dfx-itl.md)。
**那些是单侧实测，不是本文的 A/B。** 关键一条：64k 下 `full_softmax` 占 span 93.9%，
而本项优化的 `tp_all_reduce` 只占 1.84% —— C4 在 64k 的天花板本就很低。

## 3. 8-step 真实 decode 的 per-step wallclock（旁证，step 1–7 均值）

| 轮次 | 均值 |
|---|---|
| baseline | 68.74 ms |
| twophase — CI 轮 | 66.87 ms |
| twophase — repeat 1 | 67.86 ms |
| twophase — repeat 2 | 67.06 ms |
| twophase — repeat 3 | 67.09 ms |

与 ITL 同向，量级一致。

## 4. 通信量变化（P=8，`[BATCH=16, HIDDEN=4096]` BF16 = 128 KB）

| | onephase mesh | twophase + push |
|---|---|---|
| 每卡远程传输次数 | `8 chunk × 7 peer = 56`（全 pull） | `2 × (P−1) = 14`（7 pull + 7 push） |
| 每卡远程字节 | `7 × N = 896 KB` | `1.75 × N = 224 KB` |
| barrier | 2 | 3 |
| barrier 后是否还有远端读 | 否 | 否 |

## 5. 为什么整步只快 ~3.8%，而设计文档 §4 的微基准显示单次 all-reduce p50 −35%

单步 decode 约 67 ms，90 次 all-reduce 只占其中一小部分；微基准测的是**通信本身**的 device 侧耗时，
两个数不矛盾。按 A1 的逐层 DFX，routed-expert 仍占 ~90.7%，是后续优化的主战场。

## 6. 精度 / liveness 回归

| 检查 | 结果 |
|---|---|
| whole-network CI（`run_whole_network_ci --skip-mtp`） | **PASS** |
| Main 8-step token | `303, 1207, 19384, 872, 428, 6127, 4231, 2636` 全对 |
| `hidden_tp_spread` | **3 次独立重复 + CI 轮，共 32 个 decode step 全为 `0.0`** |

`tp_spread` 是本次的关键 gate：改动过程中出现过 token 全对但 8 卡不一致的中间版本
（token 只从 rank0 采样，掩盖了 rank 间分歧）。根因与完整弯路见
[`postmortems/13-tp-allreduce-pull-notify-race.md`](../postmortems/13-tp-allreduce-pull-notify-race.md)。
