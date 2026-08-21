# 专项：MoE dispatch 域小算子融合 —— 整条线 NO-GO（`orch` 阻塞读是承重节流，且 orchestrator 不在关键路径）

| 字段 | 值 |
|------|----|
| **子系统** | whole-net / performance |
| **error signature** | `orch_error_code=8 TENSOR_WAIT_TIMEOUT` + `S1:running-stalled`（概率性）；修复候选 `orch_error_code=2 HEAP_RING_DEADLOCK` |
| **首次出现** | 2026-08-17（`code -8` 首次复现），2026-08-21 定案 |
| **状态** | ✅ 已定案（**负结论**：该方向关闭，生产不做改动） |
| **相关 skill / doc** | [`LESSONS.md`](LESSONS.md)、[`12-integration-churn-meta.md`](12-integration-churn-meta.md)（流程根因 6–11）、[`../design/performance/07-hardware-scheduler-performance.md`](../design/performance/07-hardware-scheduler-performance.md) §9 |

## 1. 背景（Background）

step3p5 W8A8 MoE decode（TP=8/EP=8、单 `@pl.program`、ctx-64K BS1）的性能主线上，
DFX 显示 MoE dispatch 域有一串 pure-control 小算子（`dispatch_count_publish` /
`dispatch_meta` / `dispatch_wait` / `dispatch_push` / `dispatch_gather`），
看起来"融合掉几个就能省调度延迟"。2026-08-15 ~ 08-20 codex 在 0162 上跑了 6 天、
357 个 run 目录、8 个候选变体（per-expert / forceipc / per-block-ready / dual-mix /
fence-phase / local-self-phase / retire-gated / split-comm-boundary），**0 落地**。
2026-08-21 接手后一天内定案。生产基线是 **R5**（`decode_fwd.py` sha `67b73589…`）。

## 2. 现象（Symptom）

最后那个候选 R9 相对 R5，`decode_fwd.py` 的**全部差异只有一行**：

```diff
 with pl.spmd(scan_blocks, name_hint="dispatch_gather",
-    deps=[wait_tid, meta_collect_tid],
+    deps=[dispatch_push_tid, wait_tid, meta_collect_tid],
     predicate=(local_route_count[0] > 0), allow_early_resolve=True)
```

在**生产配置**（`MAX_SEQ=ROPE_SEQ=65536`、`--num-blocks 512`）下概率性挂死：

```
orch_error_code=8 TENSOR_WAIT_TIMEOUT   sched_error_code=100   runtime_status=-8
sub_class=S1:running-stalled   completed=546/1744 running=1 orch_done=0
stuck_task_id=12884902515 -> ring=3 local=627 = swa_moe_chip_orch_combine_wait
FATAL(code=8): Timeout (750000000 cycles): producer (ring=3, local=933) not completed
                                                    ^ = swa_moe_chip_orch_dispatch_meta
```

**是概率性的，不是配置性的**：生产配置 3 次挂 2 次（`inv=95` HANG / PASS p50 `26.615 ms` /
`inv≈224` HANG）；小配置（ctx 4096 / 32 blk）单步与 N=128 各 1 次 PASS，
但只跑过 2 次，**无法区分**「概率与配置有关」和「无关」。

## 3. 根因（Root Cause）

**已确立**：一个 **orchestration 级的阻塞标量读**落在了一个**体内含跨卡 wait** 的 task 输出上。

```
orch: scatter_blocks = pl.read(local_route_count,[1])        # decode_fwd.py:2611
    -> AICPU get_tensor_data -> wait_for_tensor_ready(读路径)  # pto_runtime2.cpp:221
    -> 自旋等 producer ring3/local933 = dispatch_meta 完成
dispatch_meta 体内: pld.system.wait(meta_arrived[src] >= moe_epoch)   # 跨卡
```

⇒ orchestrator 的前进被耦合到跨卡进度上，任何上游跨卡停滞被放大成**全 rank orchestrator
冻结**（**deadlock amplifier**）。按 AICPU pid 计数，**8 个 orchestrator 全部**阻塞在同一
producer（`grep -ho "AICPU([0-9]*," log | sort | uniq -c` → 8 个 distinct pid；
`producer (ring=3, local=933)` 16 次无一例外）。该耦合在 **R5 里逐字节相同**。

**关键区分**：`predicate=` **不**阻塞 orchestrator（`pto_orchestrator.cpp:966-985` 只算
地址不等待）；只有**动态 spmd grid 定尺**才产生阻塞 `get_tensor_data`。

**未确立（不得当作已知）**：`combine_wait`(627) 为何完不成；因果方向（orch 先卡 vs 计算先停）；
那一行 diff 是"引发"还是只"改变时序"。⇒ amplifier 是结构性主张，**不是**已证的 root cause。

### ★ 反直觉的第二层根因：那个阻塞读同时是**承重的 run-ahead 流控阀**

按"消掉 amplifier"出的结构修复候选（`combine_scatter` 的 grid 改静态常量、上界下沉进
kernel 体）**无卡 codegen 门完全达标**（`local_route_count` 上的 orchestrator 阻塞读
**4 → 0**），但 **device 门三臂全挂**：

| 臂 | sched 超时 | 死在 | 签名 |
|---|---|---|---|
| WORKERS=8 | 默认 | `inv=10` | `orch_error_code=2` **HEAP_RING_DEADLOCK** |
| WORKERS=1 | 默认 | `inv=1` | `S1` + **`orch_done=1`**（看门狗误报，见 §5） |
| WORKERS=1 | **45 000 ms** | `inv=357` | **HEAP_RING_DEADLOCK**（与 w8 同签名） |

机制（`RUNTIME_LOGIC.md`，a2a3 `host_build_graph/docs/`）：**§4.4** ring 耗尽时
orchestrator 阻塞 ⇒ ring 就是对 run-ahead 的背压；**§4.5** ring 太小会因 **scope 引用**
死锁（`fanout_count` 含一个只在 `scope_end()` 释放的 scope 引用，而 `scope_end()` 由
orchestrator 调用，orchestrator 又在等 ring 空间）。
那个每层一次的阻塞读原本把 run-ahead 限制在**一层内**；删掉 ⇒ `orch_done=1`
（一次提交完整张 1744-task 图）⇒ **调度体制被改变**，run-ahead 无界 ⇒ ring 饱和。

**不是容量账**：runner 本来就是 4 GiB heap / 131072 slot；加容量只把失败从 `inv=10`
推到 `inv=357`。⚠ **但机制只到"观测"为止**：单次 invocation 1744 task vs 131072 slot
⇒ **单次填不满 ring（差 75 倍）**，而失败发生在 `inv=10` / `inv=357`
⇒ **资源必然跨 invocation 累积**，真缺陷可能是**回收/泄漏**被这个改动暴露。
（此项已无需追究，见下。）

### ★★ 定案依据：orchestrator 从不在关键路径上 ⇒ 整类改动 ROI 上界 = 0

只解析**已有 artifact** 的 runtime STRACE 嵌套 span（不占卡、不改码、不加锁）：

```text
simpler_run ⊃ { bind, runner_run }
runner_run  ⊃ device_wall ⊃ graph_build ⊃ sched ⊃ orch
```

每 invocation 一组、`clk=dev`、`dur` 单位 ns；丢弃 warmup `inv<10`、8 rank 汇池：

| p50 span | R5（有阻塞读） | w1（去掉阻塞读） | Δ |
|---|---:|---:|---:|
| `device_wall` | 17466.93 µs | 17910.32 µs | **+443.4** |
| `graph_build` | 17457.59 | 17900.76 | +443.2 |
| `sched` | 17439.98 | 17883.85 | +443.9 |
| **`orch`** | **17279.28** | **4443.18** | **−12836.1（−74.3%）** |
| 样本 | n=8008 | n=2776 | |

**`orch` 缩了 12.8 ms / −74%，`device_wall` 一点没降、反而略升。**
⇒ orchestrator 与 device 并发跑、两种情况下都提前完成 ⇒ **不在关键路径上**
⇒ **去掉阻塞读的 ITL ROI = 0（甚至微负）**。

诚实边界：非 A/B/A bracket，两轮不同 run 各取一臂，w1 那轮最终挂了，
所以 +443 µs 可能混有走向失败途中的退化。但结论不依赖这一点 —— `orch` 降 12.8 ms 而
`device_wall` 没降，**量级差 29×**，噪声解释覆盖不了。

## 4. 如何解决（Fix）

**不修。整条线关闭，生产继续用 R5，`decode_fwd.py` 不做任何改动。**

- R6-R9 融合候选：**NO-GO**（预登记规则 R1：生产配置任何 liveness 失败即 NO-GO；
  且匹配曝光后**也不快** —— R5 `ITERS=1000` p50 `26.329` 优于 R9 clean run `26.615`）。
- 结构修复候选（静态 grid）：**NO-GO**（device 门三臂全挂，删掉了承重节流）。
- 原计划的「本地节流」候选与「判 ring 耗尽机制」：**一并关闭**（ROI = 0，无换取价值）。

dispatch 域其余可融合点逐个都撞已文档化硬约束：

| 候选融合 | 撞到的硬约束 |
|---|---|
| `dispatch_meta` + `dispatch_wait` | orchestrator 阻塞读会等**更多**跨卡进度 ⇒ **加剧** amplifier |
| `dispatch_count_publish` + `dispatch_push` | task 边界**正是** self-row 可见性 fence（`decode_fwd.py:1740-1744`：remote_store-to-self 没有 peer notify 提供跨 task fence） |
| `dispatch_wait` 吸进 `dispatch_gather` | 违反「不要把跨卡 wait 吸进 compute kernel」——把 scheduler 能直接读出缺谁的 `S4` 退化成要人肉映射的 `S1` |
| `dispatch_push` / `dispatch_gather` 的 grid | 本来就由 **host** `num_tokens` 定尺（`:1810`/`:1817`）⇒ 无 orchestrator 阻塞读 |

**★★ 副产品（量级更大的新线索，已转性能主线）**：同一份 STRACE 里 R5 每 invocation
`simpler_run` p50 `26.45 ms`（与 ITL p50 `26.329 ms` 对得上），其中
**`bind.args` = `6.12 ms` ≈ ITL 的 23%** —— 纯 host 侧参数绑定、与 `runner_run` 加性
（w1 侧 `5.87 ms` 同量级）。**比本线任何可得收益大一到两个数量级。**
见 [`../design/performance/task-tracking.md`](../design/performance/task-tracking.md)。

## 5. 走过的弯路（Detours / What We Got Wrong）

- ❌ **「闭合死锁环」**（orch 卡 933 ⇒ 走不到 `combine_wait`(627) ⇒ 饿死 peer ⇒ 环闭合）
  → 证伪：**提交序 627 < 933**，orchestrator 推进到 933 必然早已提交 627；而已 RUNNING 的
  `combine_wait` 是先 notify 再 wait，发 notify **不需要 orchestrator**。⇒ 环没有闭合边。
- ❌ **「1 个元凶 rank + 7 个受害者」**（据 `sched_error_code` 分布 7×100 + 1×0）
  → 证伪：**AICPU pid 计数 = 8 个 distinct pid**，全部阻塞在同一 producer。
  少数派报告只说明**哪个子系统停了**（orch vs scheduler），**不说明哪个 rank 有责任**。
- ❌ **「w8 与 w1 不是同一根因」** → 证伪：抬高 `SIMPLER_SCHEDULER_TIMEOUT_MS` 到 45 s 后
  w1 跑到 `inv=357`，挂法与 w8 **完全相同**。**w1 从来不是 rendezvous 死锁，只是慢。**
- ❌ **按"跑了几轮"比较 liveness 风险** → 每轮 `ITERS` 不同、曝光差 9 倍。正确单位是
  **invocation-until-failure**：R9 曝光 429 / 2 事件；R5 曝光 1010 / 0 事件 ⇒ 强烈提示但非已证
  （只有 2 个事件，Poisson 下界代入后 p 可高到 ~50%）。
- ❌ **「run-ahead 无界 ⇒ ring 饱和」当已证机制** → 算术证伪：单次 invocation 填不满 ring。
- ❌ **预注册的「写路径 / consumers 未释放（fanout）」** → device 给的是**读路径**。
- ❌ **「predicated 任务漏放 fanout 引用」** → 源码证伪：predicate 失败的 task 走
  `dummy_ready_queue` 内联退休，`on_task_complete` + 延迟 `on_task_release` → `release_producer`。
- ❌ **「小配置门结构上看不见」** → R9 生产配置也有 1 次 PASS，是概率问题。
- ❌ **`dispatch_meta` + `dispatch_wait` 融合**（看似省一个小算子）→ 会让 orchestrator 的
  阻塞读等待**更多**跨卡进度，方向与解耦**相反**。
- ❌ **「要拿 ROI 必须先写一个能活下来的候选」** → 失败臂在挂前跑了几百个 invocation，
  稳态 span 早就够统计。**这一条差点又烧掉一轮 device 门。**

## 6. 如何避免（Prevention）

**铁律**：

1. **改任何 orchestration 级构造之前，先从现有日志读 `orch` 与 `device_wall` 的 p50。**
   若 `orch <= device_wall`，orchestrator 不是瓶颈，**整类改动的 ROI 上界就是 0**。
   这比"审立项前提"更硬 —— 它是可量化的否决门。
2. **不要**让 orchestration 级阻塞标量读（动态 spmd grid 定尺）指向一个**体内含跨卡 wait**
   的 producer。写新 kernel 时自检这一条。
3. **也不要直接删它** —— 必须换成一个**只依赖本地 device 进度**的节流，否则是把稀有的
   概率性跨卡死锁换成确定性 ring 死锁。
4. **`orch_done=1` 是免费的早期判据**：任何候选一旦让它出现，run-ahead 已无界，注定撞 ring
   —— 不必等 device 门跑完。
5. **`S1` 要先用 `SIMPLER_SCHEDULER_TIMEOUT_MS` 分「慢 vs 死」**（env-only、零改码）。
   ⚠ runtime 提示里的 env 名不可信：`error_names.h:172` 让你调 `PTO2_SCHEDULER_TIMEOUT_MS`，
   **该 env 不存在**；真名 `SIMPLER_SCHEDULER_TIMEOUT_MS`（`runtime_timeout_config.h:25`），
   受 `scheduler_timeout_us < op_execute_timeout_us` 约束（实测 op 超时 50000 ms）。
   同族 `PTO2_TENSOR_DATA_TIMEOUT_MS` 是编译期 constexpr、不可设。**一律 grep `getenv` 核对。**
6. **`HEAP_RING_DEADLOCK` 先算容量账**：单次 invocation 的 task 数 vs ring slot 数。
   若单次填不满，那不是 run-ahead 深度问题，是累积/回收问题
   （去读 `scope_end()` / `on_task_release` / `release_producer`）。
7. **抓 orchestrator 侧 FATAL 必须 `ASCEND_GLOBAL_LOG_LEVEL=3`**，否则点名 producer 的字符串
   被 `CheckLogLevel(AICPU, DLOG_ERROR)` 门掉。只设 `ASCEND_PROCESS_LOG_PATH` 会
   "建了目录、文件是空的" —— 这是把 6 天定位卡住的直接原因之一。
8. **grid-stride 的 grid 值与正确性无关**：`pl.range(worker, N, grid)` 在 worker 取遍
   `[0, grid)` 时并集恰为 `[0, N)`，任何 `>=1` 都全覆盖 ⇒ 它只是并行度/节流旋钮。
   别把节流问题误写成正确性问题。
9. **概率性 liveness 缺陷要用"一轮长跑"抓，不要"多跑几轮"**：一轮 11 分钟里 decode 只占
   约 3 秒，`ITERS` 100→1000 只多约 25 秒墙钟却把曝光放大 10 倍。

**早期识别信号**：

- `orch_error_code=8` + `producer (ring, local) not completed` → 查那个 producer 体内有没有跨卡 wait。
- 候选日志出现 `orch_done=1` 而基线没有 → 你删掉了一个节流阀。
- `S1` 在 `inv=1` 就报 → 先抬超时，别下"死锁"结论。

**落点**：本复盘 + [`LESSONS.md`](LESSONS.md) 流程/整网两段 +
[`../design/performance/07-hardware-scheduler-performance.md`](../design/performance/07-hardware-scheduler-performance.md) §9 +
[`12-integration-churn-meta.md`](12-integration-churn-meta.md) 根因 9/10/11。

**证据入口（0162）**：
`…/perf-2026q3/dispatch-orch-decouple-20260821/{FINDINGS.md, analysis-bin/orch_span_stats.py}`；
`…/perf-2026q3/dispatch-fusion-triage-20260821/MECHANISM.md` **v2**
sha256 `2e964264b6c7ae24d5681b71d5176d62e1350f1b115fde9a9c822880ee354f66`
（v1 `3f670f1a…` 声称环已闭合，**已被取代，不要引用**）。
复现器：`ITERS=1000 bash <triage>/runner/faultlog_r9.sh <R9-tree> <outdir>`（半机锁，~12 min，命中率约 2/3）。
