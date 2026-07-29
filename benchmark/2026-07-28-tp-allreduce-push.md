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
> （`--num-blocks 32` 上限 4096）。项目的**权威 ITL 工作点是 64k**
> （`--num-blocks 512`，见 [`2026-07-24-…-perlayer-dfx.md`](2026-07-24-step3p5-decode-perlayer-dfx.md)
> 引用的 `≈590 ms/step`）。all-reduce 的绝对开销基本不随 context 变化（只搬 `[16,4096]` hidden），
> 而 64k 下单步分母大 8~9 倍，因此 **64k 的相对收益必然显著小于 −3.8%**。§2 的数字不得直接
> 当作发布收益引用；64k 曲线见 §2b。

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

对比双方均为 **已发布 immutable 镜像**（无挂载 overlay）：
`stepfun-develop-20260726-step3p5-only`（pypto-lib `53eb7212`，无本改动）
vs `stepfun-develop-20260729-allreduce-push`（pypto-lib `cfbdcce8`）。

<!-- PENDING: 待 64k A/B 实测数据 -->

## 2c. ctx=65536 的 active-batch 扫描（发布镜像单侧实测，非 A/B）

`_stage_main_hidden_only --itl-context-lens 65536 --active-batch N --num-blocks <128*N>`，
iters=20 / warmup=3，cards 8–15，`--shm-size 400g`。ITL 的 `--active-batch` 支持已入库
（`csy0225/pypto-lib stepfun/develop@cc850ee5`，纯测试改动，在发布镜像之后）——
本节数据是把该 harness 挂载进发布镜像跑出来的。**这一节只是发布镜像自身的 batch 标定，
不是与 baseline 的对比**（baseline 侧 64k 数据未采）。

| active_batch | scheduler blocks | dummy KV (K+V) / rank | p50 | mean | p99 |
|---|---|---|---|---|---|
| 1 | 512 | 1.45 GiB | 84.141 ms | 84.125 ms | 84.327 ms |
| 2 | 1024 | 2.85 GiB | 87.754 ms | 87.565 ms | 87.818 ms |
| 4 | 2048 | 5.67 GiB | 104.023 ms | 104.505 ms | 106.760 ms |
| 8 | 4096 | 11.29 GiB | 145.073 ms | 144.812 ms | 145.963 ms |
| 16 | 8192 | 22.54 GiB | — | — | — （device HBM OOM，见下） |

- **bs=16 @ 64k 在本配置下跑不起来**：`out of memory` → `simpler_init failed with code -1`。
  峰值 HBM/rank 在 bs=8 已到 `≈36 GiB`，bs=16 的 KV 再涨 `≈11 GiB`，超出单卡可用。
  **不是** host shm 不足（已把 `--shm-size` 从 32g 提到 400g，行为不变）。
- **口径提醒**：harness 里的 `BATCH=16` 是 **batch_capacity**（padding 容量），
  `active_batch` 才是真正参与计算的行数。历史报告里出现的 `"batch": 16` 字段
  只表示 capacity，不代表跑了 16 个活跃 row —— 该字段已改名为
  `batch_capacity` / `active_batch` 两项以消除歧义。
- **未对齐项**：本节 bs=1 的 `84 ms` 与 [`2026-07-24-step3p5-decode-perlayer-dfx.md`](2026-07-24-step3p5-decode-perlayer-dfx.md)
  引用的 64k `≈590 ms/step` 相差约 7 倍，两者测量基准（KV 来源 / 是否含 host glue /
  是否 raw `rt.run`）尚未对齐。**在对齐之前两个数字都不得互相当作 before/after 引用。**

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
