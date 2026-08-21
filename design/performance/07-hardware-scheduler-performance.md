# 硬件性能知识：AICore completion 到后继 task execute 的 L2 调度间隙

> **分类**：Hardware performance / L2 AICPU scheduler / completion、early dispatch、memory ordering  
> **案例**：step3p5 five-layer 中 22-block MIX out-proj → TP all-reduce  
> **更新时间**：2026-08-21（新增 §9：orchestrator run-ahead 与 ring 背压；§9.6 收盘量化 ⇒ 该方向关闭）  
> **状态**：early-dispatch 归因已有真机证据；completion microbatch=8 仍在验证，不能提前写成性能收益。
> §9 的 device 现象已实测，但"跨 invocation 累积"的具体机制（回收 vs run-ahead 深度）**未确立**
> —— 且 §9.6 已表明它不必确立：orchestrator 不在关键路径，整类改动 ROI 上界为 0。

本文记录可复用于其他多 block producer → 单一后继 task 的调度知识。它讨论的是
**all-reduce 开始执行前的空隙**，不是 all-reduce 算法自身的执行耗时。

## 1. 唯一主指标

在每个 `(arm, rank, layer)` 内定义：

```text
physical_last = max(该 out-proj 的全部 AICore record.end_time)
AR_start      = min(该 TP all-reduce AICore record.start_time)
gap           = AR_start - physical_last
```

本例逻辑上是 22-block MIX；当前 Level-1 DFX 中每次 occurrence 展开为 66 条
producer physical slice，必须取全部 66 条的最后结束点。目标 all-reduce 只有一条
AICore execute record。

测量纪律：

- 只能在**同一 rank、同一 layer、同一时钟域**内相减，禁止跨 rank 对时间戳；
- 起点必须是最后一个 producer 的**物理结束**，不能用 logical finish 或平均结束；
- 终点必须是 all-reduce 的 **AICore execute start**，不能用 AICPU dispatch/receive；
- dispatch 可以因 early dispatch 早于依赖满足，出现负的
  `physical_last → dispatch`，这不是错误；execute gap 不应为负；
- 使用直接配置的 Level-1 swimlane。Level-4 会引入额外观测与调度扰动，不作为主证据。

## 2. 空隙的硬件事件链

```text
22-block MIX 的最后一个物理 slice 结束
  → AICore 发布 FIN/COND
  → AICPU scheduler 轮询到对应 core
  → memory barrier 后读取 AICore 发布的 Normal-memory 状态
  → 聚合 block completion，完成 logical task/fan-in
  → 依赖满足，释放或下发 TP all-reduce
  → core/slot/sync-start 条件满足
  → TP all-reduce AICore execute start
```

因此 `gap` 至少包含四类成本：

| 分段 | 主要含义 | 可优化手段 |
|---|---|---|
| producer tail | 多 block 中最后一个 slice 才决定 logical completion | block 负载均衡、尾核治理 |
| FIN observation | AICPU 扫描 running core、读取 COND、执行内存屏障 | completion polling/batching |
| completion apply | slot transition、fanin、deferred release、依赖传播 | 状态机热路径瘦身 |
| release → execute | dispatch、doorbell、sync-start、core/slot 排队 | early dispatch、资源调度 |

必须先分段再改代码。只看到两条 kernel 之间有白缝，不能直接断言是 collective、AICPU
或 core 不够。

## 3. 已建立的 five-layer 证据

40 个 occurrence（8 ranks × 5 layers）的同源 DFX 分析：

| 指标 | baseline p50 | early-dispatch candidate p50 |
|---|---:|---:|
| producer physical last → final producer AICPU finish | 5.130 µs | 5.120 µs |
| final producer AICPU finish → AR start | 1.750 µs | 0.620 µs |
| producer physical last → AR start | 6.820 µs | 5.650 µs |

结论边界：

1. early dispatch 实际消掉的是 final finish 之后约 **1.13 µs**；
2. 总 gap 的 p50 改善约 **1.17 µs**；
3. producer 最后物理结束到 AICPU final finish 的约 **5.1 µs** 没有变化，已成为主要剩余项；
4. 所以后续主线应是 completion polling/aggregation，而不是继续扩大 early-dispatch
   预占窗口。

placement 证据也显示：

- 40/40 次都预放在 **IDLE running slot**，不是 pending slot；
- `receive → dependency-ready` 中位数为 **27.920 µs**；
- `ready → execute start` 中位数为 **5.650 µs**；
- 只在 1/40 次中，目标 exact core 同时属于该 producer 使用过的 core；
- 当前样本没有观察到 normal dispatch 或 exact-core competing work 被 gate window
  挤占，但其他 task readiness 覆盖不足，不能据此证明“绝无 victim”。

因此，producer-affine pending-slot 策略目前只是未证实候选，不能作为已定位根因。

## 4. Early dispatch 能做什么、不能做什么

Early dispatch 的作用是把后继 task 的下发、slot 获取和一部分准备工作提前到 producer
执行期间；真正的数据依赖仍由 doorbell/gate 保护。

- 它可以把后继放入 running/pending slot，等最后依赖完成后快速 release；
- 它不能让 all-reduce 在 22 个 producer block 全部完成前读取结果；
- `STAGING` task 尚未 rung，不可能 ACK/FIN，completion poll 应跳过这类 core；
- sync-start task 即使从 `STAGING` 变成 `DISPATCHED`，在 cohort rendezvous 完成前仍是
  gated；不能仅凭枚举值把它当作普通 pending task，否则可能等待一个永远不会来的 ACK；
- 更早预占 slot 可能缩短 release 后路径，也可能挤压正常 ready work，必须用 victim
  证据或 whole-network A/B/A 判断，不能只看 dispatch 提前了多少。

## 5. Completion microbatch=8 的设计知识

原热路径对每个 running core 依次执行：

```text
COND load → rmb → transition classify/apply
```

对于 22-block MIX 的密集 completion，多个 FIN 会在相近时间到达，逐 core 屏障和重复
DFX/timing 解析会直接落在最后一个 completion 的关键路径上。

候选实现按最多 8 个 core 分组：

1. 按 tracker bitmap 原顺序 probe，最多读取 8 个 COND；
2. 只用 scheduler-owned state 做 transition classification；
3. 没有 match 时不执行 `rmb()`；
4. 有 match 时对整组只执行一次共享 `rmb()`；
5. 每组只解析一次 task-timing records、采一次 counter；
6. 屏障后仍按原 probe 顺序逐项执行 ACK hook、completion、slot update 和 fan-in。

它利用的是**密集 completion 对屏障固定成本的摊薄**，并没有改变 task DAG 或
all-reduce 算法。

### 5.1 收益与代价

| 场景 | 预期 |
|---|---|
| 多核在同一轮密集 FIN | 多个 match 共用一次 barrier，最可能受益 |
| 整组没有 FIN | 只做 COND probe，不付 barrier |
| 单个稀疏 FIN 出现在组首 | apply 前最多再等 7 次串行 COND load，可能增加尾延迟 |
| DFX/task timing | group timestamp 是组内 FIN observation 的上界，不是逐 FIN 精确时刻 |

所以 batch=8 不是无条件越大越好。它是在 barrier 固定成本、COND MMIO 成本和 completion
密度之间取平衡；必须同时验证 dense fan-in 和 sparse task 的 p50/p99。

### 5.2 不能破坏的状态机与内存序

- `reg_load_acquire` 必须保留：CPU sim 中与 FIN release store 配对；
- 硬件上 COND 属 Device memory，AICore 随 FIN 发布的 task record/deferred slab 等属于
  Normal cacheable memory；所有这些 Normal-memory 读取必须位于共享 `rmb()` 之后；
- ACK-gated DFX hook 必须在对应 transition apply 前执行；
- dual-slot 的 running/pending Case 1–4、deferred completion 和 ring reuse 语义不变；
- gated sync-start promotion 必须先保留 promoted pointer，再更新 rendezvous；
- task timing 对同一 timing slot 取 `max(finish_cycle)`，避免多 block 后到记录被覆盖。

## 6. 因果隔离与 A/B/A 规范

completion batching 与 early dispatch 必须独立归因：

- 三个 arm 使用同一个冻结的 early-dispatch graph；
- A1/A2 只挂 baseline AICPU SO，B 只挂 candidate AICPU SO；
- host runtime、AICore object、模型、输入、镜像和图必须保持相同；
- 每个 arm 使用独立冷容器，并记录 `RuntimeBuilder().aicpu_path`、SHA256 和 `findmnt`；
- non-DFX 先验证 exact precision、TP spread=0 和稳态 ITL；
- DFX 再按 `(arm, rank, layer)` 计算本文 §1 的 execute gap。

建议准入门槛：

- 两个 baseline 的 p50 drift ≤ 0.5 µs；
- paired `median(|A2-A1|)` ≤ 0.75 µs；
- B p50 至少比 A1、A2 各低 1 µs；
- B p90 不得回退超过 0.5 µs 或 5%；
- 至少 80% 的 rank×layer cell 改善；
- 不允许缺 record、drop 或负 execute gap。

如果同时改变 graph early-dispatch 和 completion SO，即使结果变快也无法判断收益来自哪一层。

## 7. 结果解释决策树

| 观测 | 应解释为 |
|---|---|
| physical last → AICPU finish 下降，finish → start 不变 | completion 优化命中 |
| AICPU finish 下降，但总 execute gap 不降 | 瓶颈转移到 fan-in、core queue 或 sync-start |
| dispatch 明显提前，execute start 不变 | 只增加 gated residency，没有缩短关键路径 |
| five-layer gap 下降，whole ITL 无变化 | 局部收益低于噪声或不在 whole-network critical path |
| p50 下降但 p90/sparse 回退 | batching head-of-line 或 scheduler 干扰，需要减小 batch/自适应 |

不能用单次最优值或一张 swimlane 图宣布收益；需要 A/B/A、逐 cell 一致性和 whole-network
复核三者同时成立。

## 8. 当前状态与证据入口

截至 2026-08-13：

- early-dispatch 局部归因成立，但没有证据支持继续扩大 slot 预占；
- 当前直接目标是约 5.1 µs 的 producer completion observation/aggregation；
- completion microbatch=8 已通过源码 review 和无卡回归，硬件性能 A/B/A 尚未完成，
  因而本文不记录未经验证的加速数字。

0162 证据：

```text
/mnt/persist/chensiyu/workspace/tp-ar-early-dispatch-20260813/validation/analysis-global/
  early_dispatch_placement_analysis.md
  dfx_completion_and_gate_victim_analysis.md

/mnt/persist/chensiyu/workspace/runtime-worktrees/tp-ar-completion-batch-20260813/
  src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_completion.cpp
```

## 9. 第三个调度主体：orchestrator run-ahead 与 ring 背压（2026-08-21 新增）

上面 §1–§8 讨论的是 **L2 AICPU scheduler**（completion / slot / fan-in / execute）。
它不是唯一的调度主体。**orchestrator 是一个独立的 AICPU 主体**：它把 task 提交进
ring，两者可以各自独立停摆（同一次故障里出现过
`7×sched_error_code=100 + 1×sched_error_code=0`，后者就是 scheduler 没停、orchestrator 停了）。

### 9.1 ring 既是资源，也是背压机制

`RUNTIME_LOGIC.md`（a2a3 `host_build_graph/docs/`）：

- **§4.4** —— ring 耗尽时 **orchestrator 阻塞**。⇒ ring 就是对 orchestrator run-ahead 的背压。
- **§4.5** —— ring 太小会因 **scope 引用**死锁：`fanout_count` 含一个只在 `scope_end()`
  释放的 scope 引用，而 `scope_end()` 由 orchestrator 调用，orchestrator 又在等 ring 空间。

⇒ **run-ahead 不是"越深越好"的自由变量，它被 ring 容量与回收速度共同约束。**

### 9.2 什么会阻塞 orchestrator（决定可优化面）

| 构造 | 是否阻塞 orchestrator | 依据 |
|---|---|---|
| `predicate=` | **不阻塞** | `pto_orchestrator.cpp:966-985` 只计算 predicate 地址、不等待 |
| 动态 `pl.spmd` grid 定尺（trace 级 `pl.read`） | **阻塞** | → AICPU `get_tensor_data` → `wait_for_tensor_ready`（`pto_runtime2.cpp:221`）等 producer task 完成 |
| host 供给标量（如 `num_tokens`）定尺 grid | 不阻塞 | 值在 host，无 device producer |

⇒ 想减少 orchestrator 阻塞，只有**动态 grid 定尺**那一类读可动；改 predicate 无效。

### 9.3 这类读有**两个**作用，删掉会换来确定性故障（device 实测）

一个 orchestration 级阻塞标量读若其 producer **体内含跨卡 `pld.system.wait`**，
它就把 orchestrator 的前进耦合到跨卡进度上 ⇒ 上游任何跨卡停滞被放大成
**全 rank orchestrator 冻结**（**deadlock amplifier**）。step3p5 的实例是
`combine_scatter` 的 `scatter_blocks = pl.read(local_route_count,[1])` 撞上
`dispatch_meta` 的跨卡 `meta_arrived` wait。

**但它同时是那一层唯一的 run-ahead 节流。** 把它改成静态常量后（语义不变，
grid-stride 覆盖不依赖 grid 值），device 实测：

| 静态 grid | 死在 | 签名 |
|---|---|---|
| 8 | `inv=10` | `orch_error_code=2` **HEAP_RING_DEADLOCK** |
| 1 | `inv=1` | `S1:running-stalled` + **`orch_done=1`** —— 抬高 `SIMPLER_SCHEDULER_TIMEOUT_MS` 后变成 `inv=357` 的 **HEAP_RING_DEADLOCK**（原来只是慢，不是死锁） |

`orch_done=1` 表示 orchestrator **一次把整张 1744-task 图提交完** —— 基线做不到这件事。
⇒ **调度体制被改变了**，而不是"少了一次等待"。

**⚠ 机制未确立（重要）**：一次 invocation 只有 **1744** task，ring 是
**131072** slot + 4 GiB heap ⇒ **单次 invocation 无论 run-ahead 多深都填不满 ring（差 75 倍）**。
失败却发生在 `inv=10` / `inv=357` ⇒ **资源必然跨 invocation 累积**（每轮有一部分未回收），
两臂可能受限于不同资源（多 block 那臂吃 heap 字节，单 block 那臂吃 slot）。
⇒ 真缺陷可能是**回收 / 泄漏**被这个改动暴露，而不只是"提交太超前"。
待判定项：读 ring allocator 与 `scope_end()` / `on_task_release` / `release_producer` 路径。

### 9.4 设计规则

1. **不要**让 orchestration 级阻塞标量读（动态 grid 定尺）指向一个**体内含跨卡 wait**的
   producer。写新 kernel 时自检这一条。
2. **也不要直接删它** —— 必须换成一个**只依赖本地 device 进度**的节流
   （grid 定尺标量来自一个从不 wait 的纯本地 task），否则是把稀有的概率性跨卡死锁
   换成确定性 ring 死锁。
3. **`orch_done=1` 是免费的早期判据**：任何候选一旦让它出现，run-ahead 已无界，
   该候选注定撞 ring —— 不必等 device 门跑完。
4. **融合方向可能与解耦方向相反**：把两个 pure-control 小 task 合并（如
   `dispatch_meta` + `dispatch_wait`）会让 orchestrator 的阻塞读等待**更多**跨卡进度、
   **加剧** amplifier。**先解耦，再融合。**
5. 与 §1–§8 的关系：early dispatch 与 completion batching 改的是**task 何时执行**，
   **不改** orchestrator run-ahead。但任何会**提高** run-ahead 的改动（去掉 orchestration
   级阻塞读、让原本阻塞的构造变非阻塞）都必须同时对 ring 容量与回收速度做检查。

### 9.5 判别法与陷阱

- **`S1:running-stalled` 不区分「永不完成」与「比超时慢」。** 下"死锁"结论前先抬高
  `SIMPLER_SCHEDULER_TIMEOUT_MS`（env-only、零改码，runtime 自己文档化的判别法）。
  本例它推翻了一个错的双根因结论。
- ⚠ **runtime 错误提示里的 env 名不可信**：`error_names.h:172` 让你调
  `PTO2_SCHEDULER_TIMEOUT_MS`，**该 env 不存在**；真名
  `SIMPLER_SCHEDULER_TIMEOUT_MS`（`runtime_timeout_config.h:25`），受
  `scheduler_timeout_us < op_execute_timeout_us` 约束（实测 op 超时 50000 ms）。
  同族 `PTO2_TENSOR_DATA_TIMEOUT_MS` 是编译期 constexpr、不可设。**一律 grep `getenv` 核对。**
- 抓 orchestrator 侧 FATAL（点名 producer 的 `ring/local`）**必须**
  `ASCEND_GLOBAL_LOG_LEVEL=3`，否则该字符串被 `CheckLogLevel(AICPU, DLOG_ERROR)` 门掉。

### 9.6 收盘量化：orchestrator 不在关键路径（2026-08-21）

上面 §9.1–§9.5 说明**为什么不能删**那个阻塞读。剩下的问题是：**删掉它能赚多少？**
答案由已有 artifact 直接给出（不占卡、不改码）。runtime STRACE 发的是嵌套 span：

```text
simpler_run ⊃ { bind, runner_run }
runner_run  ⊃ device_wall ⊃ graph_build ⊃ sched ⊃ orch
```

每 invocation 一组，`clk=dev`，`dur` 单位 ns。丢弃 warmup `inv<10`、8 rank 汇池后：

| p50 span | R5（有阻塞读） | 候选 w1（无阻塞读） | Δ |
|---|---:|---:|---:|
| `device_wall` | 17466.93 µs | 17910.32 µs | **+443.4** |
| `graph_build` | 17457.59 | 17900.76 | +443.2 |
| `sched` | 17439.98 | 17883.85 | +443.9 |
| **`orch`** | **17279.28** | **4443.18** | **−12836.1（−74.3%）** |
| 样本 | n=8008 | n=2776 | |

**orchestrator 的 span 缩了 12.8 ms / −74%，device 总时间一点没降、反而略升。**
⇒ orchestrator 与 device 并发跑，两种情况下都提前完成 ⇒ **它从不在关键路径上**
⇒ **去掉这个阻塞读的 ITL ROI = 0（甚至微负）。**

诚实边界：这不是 A/B/A bracket，是两轮不同 run 各取一臂，且 w1 那轮最终以 ring 死锁失败，
所以 +443 µs 里可能混有走向失败途中的退化。但结论不依赖这一点 —— `orch` 降 12.8 ms 而
`device_wall` 没降，量级差 29×，噪声解释覆盖不了。

**⇒ 可复用的判据（比上面五条规则更省事）**：在动手改任何 orchestration 级构造之前，
先从现有日志里读 `orch` 与 `device_wall` 的 p50 比值。**若 `orch` 明显小于或仅略小于
`device_wall`，orchestrator 不是瓶颈，整类改动的 ROI 上界就是 0。**

**★★ 同一份数据暴露的更大项**：R5 每 invocation `simpler_run` p50 = `26.45 ms`
（与该配置 ITL p50 `26.329 ms` 对得上），其中 **`bind.args` = `6.12 ms` ≈ ITL 的 23%** ——
纯 host 侧参数绑定，且 `simpler_run ≈ bind + runner_run` 是加性串行（候选侧 `5.87 ms`，
同量级 ⇒ 不是偶发）。**这比 dispatch 域任何 small-op 融合的可得收益大一到两个数量级**，
且不碰 device 语义、不碰跨卡同步。见
[`task-tracking.md`](task-tracking.md) 的 host-side bind 线。

证据入口：`0162:…/dispatch-orch-decouple-20260821/{FINDINGS.md, analysis-bin/orch_span_stats.py}`；
§9.1–§9.5 的证据入口同目录 `FINDINGS.md`；血泪账见
[`../../postmortems/12-integration-churn-meta.md`](../../postmortems/12-integration-churn-meta.md) 根因 9/10。

