# 接力上下文（Handoff）

> **只描述下一位 agent 现在要接的工作。最后更新：2026-08-21。**
> 当前状态以 [`../STATUS.md`](../STATUS.md) 为准。
> 本文并列两条线：**§0 = MoE dispatch 域融合线（已关闭，负结论）**；
> **§1 起 = TP all-reduce single-row selector（已落地 source-overlay）**。
> 历史 `fa58b5cf` NO-GO 与 `e5e26f9f` 中间态不要复制回当前结论。

## 0. MoE dispatch 域小算子融合线：已关闭（2026-08-21 收盘）


> **dispatch 域小算子融合这条线已关闭（负结论已定稿）。下一位不要再写这个方向的候选。**
> 新线索见本节末「★★ 下一条性能主线：`bind.args`」。

三件事各自独立成立：

1. **R9 维持 NO-GO** —— liveness 不过，且**匹配曝光后也不快**。
2. **修复候选 `dispatch-orch-decouple-20260821`（静态 scatter grid）device 门三臂全挂** ⇒
   该方向 NO-GO。原因不是实现 bug：**那个阻塞读是承重节流**（见「★ 关键发现」）。
3. **#7 已量化完毕 ⇒ 整条线关闭。** 只用已有 artifact 解析 runtime STRACE 五层 span：

   | p50 span | R5（有阻塞读） | w1（去掉阻塞读） | Δ |
   |---|---:|---:|---:|
   | `device_wall` | 17466.93 µs | 17910.32 µs | **+443.4** |
   | `graph_build` | 17457.59 | 17900.76 | +443.2 |
   | `sched` | 17439.98 | 17883.85 | +443.9 |
   | **`orch`** | **17279.28** | **4443.18** | **−12836.1（−74.3%）** |
   | 样本 | n=8008 | n=2776 | |

   **orchestrator 的 span 缩了 12.8 ms / −74%，device 总时间一点没降、反而略升。
   ⇒ orchestrator 本来就与 device 并发、两种情况下都提前完成，不在关键路径上
   ⇒ 去掉阻塞读的 ITL ROI = 0（甚至微负）。**
   原计划的 #5（本地节流实现）与 #6（判 ring 耗尽机制）随之关闭 —— 它们只在
   准备继续该方向时才有价值。**生产继续用 R5，不做任何改动。**

   诚实边界：这不是 A/B/A bracket，是两轮不同 run 各取一臂；w1 那轮最终挂了，
   所以 +443 µs 里可能混有走向 ring 耗尽途中的退化。但结论不依赖这一点 ——
   `orch` 降 12.8 ms 而 `device_wall` 没降，量级差 29×，噪声解释覆盖不了。

**这条线留下的可复用产物 = 两条设计规则**（已写进
[`../design/performance/07-hardware-scheduler-performance.md`](../design/performance/07-hardware-scheduler-performance.md) §9）：
① 不要让 orchestration 级阻塞标量读（动态 grid 定尺）指向一个体内含跨卡 wait 的 producer；
② 也不要直接删它 —— 换成只依赖本地 device 进度的节流，否则把稀有的概率性跨卡死锁
换成确定性 ring 死锁。

> ⚠ **本节曾写"机制已闭合 / 完整死锁环"，已被同日对抗性复核撤回。** 见下方
> 「已确立 / 已撤回 / 未确立」。别再引用「环闭合」或「1 元凶 + 7 受害者」。
>
> ⚠ 本节还曾写"w8 与 w1 不是同一根因"，**也已撤回** —— 抬高 scheduler 超时后 w1 的挂法
> 与 w8 完全相同（都是 `HEAP_RING_DEADLOCK`）。**两臂一个机制。**
> ⚠ 但"移除节流 ⇒ run-ahead 无界 ⇒ ring 饱和"这个**机制本身仍未确立** ——
> 算术上单次 invocation 填不满 ring，见「机制缺口」。别把它当已证。
>
> ★ **本轮最重要的一条**：那个阻塞标量读**不只是缺陷，它还是一个承重的流控阀** ——
> 不能直接删。详见「★ 关键发现」。

### 候选到底有多大：一行

R9 相对生产基线 R5，`decode_fwd.py` 的**全部差异只有一行**：

```diff
 with pl.spmd(scan_blocks, name_hint="dispatch_gather",
-    deps=[wait_tid, meta_collect_tid],
+    deps=[dispatch_push_tid, wait_tid, meta_collect_tid],
     predicate=(local_route_count[0] > 0), allow_early_resolve=True)
```

`grep -c routed_nz` 三列 = `develop 0 / R5 19 / R9 19` —— NZ GMM 融合**在 R5 里就有了**。
所以「R6-R9 都捆了两个优化、无法归因」是**基线选错造成的假象**（旧记录已更正）。
教训：**特征矩阵与 `diff` 要相对你实际要发布的基线算，不是 `develop`。**

### hang 是概率性的（这一条推翻了昨天的写法）

| 配置 | 次数 | 结果 |
|---|---|---|
| 生产 ctx 65536 / 512 blk，10+**100** iters | #1 | **HANG** S1 @ `inv=95` |
| 生产 ctx 65536 / 512 blk，10+**100** iters | #2 | **PASS**，p50 `26.615 ms` |
| 生产 ctx 65536 / 512 blk，10+**1000** iters | #3 | **HANG** @ ~`inv=224`，抓到 FATAL |
| 小 ctx 4096 / 32 blk，单步 / N=128 | 各 #1 | PASS |

生产配置 **3 次挂 2 次**。所以"小配置门**结构上**看不见"**不成立**；小配置也只跑过 2 次，
无法区分「概率与配置有关」与「无关」。**⇒ 单次生产配置跑通不构成 liveness 门。**
⚠ **但"几次挂几次"是错的计量单位** —— 每轮 ITERS 不同、曝光差 9 倍。
正确口径见下方「把'3 次挂 2 次'换成**曝光**口径算一遍」。
便宜补法：**拉长单轮曝光**（`ITERS=1000`），别重复整轮 —— 一轮 11 分钟里 decode 只占
约 3 秒，其余全是 compile + weight load，所以 iters 100→1000 只多约 25 秒、曝光 ×10。
**这一招当场奏效**：iters=1000 那轮就挂了，并拿到了定位所需的 FATAL 字符串。

### 停点已兑现（根因欠债还上了）

```
stuck_task_id=12884902515 = ring 3 / local 627 = swa_moe_chip_orch_combine_wait
未满足谓词  pld.system.wait(combine_arrived[src], expected=moe_epoch*n_local_experts, Ge)
应由谁置位  rank src 自己的 combine_wait（同一 task 先 notify 全 peer、再 wait 全 peer）
```

`combine_wait` 是**对称 all-rank rendezvous**，所以停在里面的 rank 都在等别人 notify。
同一份日志里 8 个 rank 的 `sched_error_code` 分布 = **7×100 + 1×0**，那个 `0` 的
rank 不产出 `sub_class=` 行、只报 orch `TENSOR_WAIT_TIMEOUT` ⇒ **它的 orchestrator 卡住，
scheduler 没停**。

```bash
grep -E "PTO2 runtime failed: orch_error_code=" container.log | sed -E "s/.*failed: //" | sort | uniq -c
```

**可推广判据（已修正措辞）：少数派报告告诉你"哪个子系统停了"（orch vs scheduler），
不是"哪个 rank 有责任"。** 原先据此推出的"1 元凶 + 7 受害者"是错的 —— 按 AICPU pid 计数，
**8 个 orchestrator 全部阻塞在同一 producer 上**：

```bash
grep -ho "AICPU([0-9]*," container.log | sort | uniq -c            # => 8 个 distinct pid
grep -o "producer (ring=[0-9]*, local=[0-9]*)" container.log | sort | uniq -c   # => 16× ring3/local933
```

### 已确立 / 已撤回 / 未确立（2026-08-21，含同日自我更正）

`TENSOR_WAIT_TIMEOUT` 全 runtime 只有 2 个 raise 点，都在
`pto_runtime2.cpp::wait_for_tensor_ready`。device 给出的是**读路径**：

```
FATAL(code=8): Timeout (750000000 cycles): producer (ring=3, local=933) not completed
```

**已确立**：

```
orch: scatter_blocks = pl.read(local_route_count,[1])        # decode_fwd.py:2611（动态 grid 定尺）
    -> AICPU get_tensor_data -> wait_for_tensor_ready(读)     # pto_runtime2.cpp:221
    -> 等 producer ring3/local933 = dispatch_meta 完成
dispatch_meta 体内: pld.system.wait(meta_arrived[src] >= moe_epoch)     # 跨卡
8 个 rank 的 orchestrator 全部卡在这里（8 pid × 2 次 = 16 条，producer id 无一例外）
```

**❌ 已撤回**：「peer 卡在 627 → 饿死该 rank → 环闭合」。**提交序 627 < 933**：orchestrator
已推进到 933 的 rank 必然早已提交过 627，而已经 RUNNING 的 `combine_wait` 是**先 notify 再
wait**、发 notify 不需要 orchestrator ⇒ **环没有闭合边**。

**未确立**：627 究竟为何完不成（若 8 rank 都到了它的 wait，所有 notify 都已发出、每个 wait
都该通过 ⇒ **至少有 1 个 rank 不在那里，而那个 rank 恰好没产出 `sub_class` 行**——这是本案
**缺失的关键数据**）；因果方向（T1 orch 先卡无闭合边；T2 计算先停、orch 被放大更可信但未证）；
那一行 diff 是"引发"还是仅"改变时序"。

**仅存的结构性主张**：那个阻塞标量读把 orchestrator 的前进耦合到跨卡进度上，于是任何上游
跨卡停滞都会被放大成**全 rank orchestrator 冻结** ⇒ **deadlock amplifier，不是已证 root cause**。
该耦合在 R5 里逐字节相同。**设计规则**：不要让 orchestration 级阻塞标量读（动态 spmd grid
定尺）指向一个体内含跨卡 wait 的 producer。

**R5 对照（决定性）**：同配置 `ITERS=1000` 一轮长跑 **PASS**、无 FATAL、p50 `26.329 ms`
（`faultlog-R5-iter1000/`）。⇒ R9 3 次挂 2 次 vs R5 10× 曝光 1 次 1 过，**那一行对 hang 概率
不是中性的**（n=1，不证明 R5 绝对安全）。

#### ⚠ 把"3 次挂 2 次"换成**曝光**口径算一遍（对抗性自审，第二个口径错误）

"几次挂几次"是**错的计量单位** —— 每轮 `ITERS` 不同，曝光差 9 倍。按
**invocation-until-failure** 重算：

| 臂 | ITERS | 曝光（invocation） | 事件 |
|---|---|---:|---|
| R9 #1 | 10+100 | 95（挂在 inv=95） | 1 |
| R9 #2 | 10+100 | 110（跑完） | 0 |
| R9 #3 | 10+1000 | 224（挂在 ~inv=224） | 1 |
| **R9 合计** | | **429** | **2** |
| **R5** | 10+1000 | **1010** | **0** |

R9 hazard 点估计 ≈ `2/429` ≈ **1/215 每 invocation**。若 R5 同 hazard，
`P(1010 个 invocation 全过) = (1-1/215)^1010 ≈ e^-4.70 ≈ 0.009` ⇒ **p ≈ 1%**。
**比原来那句"n=1"强得多。**

**但只有 2 个事件，hazard 的 CI 很宽**：λ 的 95% 下界（Poisson，2 事件）约 `0.24/429`
≈ `1/1788`，代入得 `P ≈ e^-0.565 ≈ 0.57` ⇒ **p 可以高到 ~50%**。
⇒ **诚实结论：R5 更安全是「强烈提示」，不是「已证」。**
**便宜的收口**：再跑 1~2 轮 R5 `ITERS=1000`（各约 12 min，半机锁）。若再 0 事件，
合计曝光 2000~3000 个 invocation，即使取 hazard 下界也能把 p 压到 <5%。

**⇒ 由此得到一条可复用的计量纪律**（与"拉长单轮曝光"是同一条的分析面）：
**概率性缺陷的比较必须按曝光（invocation-until-failure）算，不能按"跑了几轮"算。**
按轮数算会同时**高估** R9 的坏（它的两个 100-iter 臂曝光很小）和**低估** R5 的证据。


**决定性的一步是一个环境变量**：那条 FATAL 由 AICPU `unified_log_error` 发出，被
`CheckLogLevel(AICPU, DLOG_ERROR)` 门住 ⇒ **必须 `ASCEND_GLOBAL_LOG_LEVEL=3`**。
之前所有 arm 只设了 `ASCEND_PROCESS_LOG_PATH`，目录建了、文件是空的、字符串全丢。
**成本为零。**

**已证伪的机制假设**（别重复走）：① 我**预注册**的"写路径 / consumers 未释放
（fanout）"——device 证伪，相关推理已撤回（`MECHANISM.prereg.md` sha `089587ca…`，
故意在取数前写好以便记分）；②「predicated 任务漏放 fanout 引用」——源码证伪
（predicate 失败走 `dummy_ready_queue` 内联退休，`on_task_release` → `release_producer`
逐个释放 fanin）；③「闭合死锁环 / 1 元凶」——见上；④「小配置门结构上看不见」——概率问题。
**仍成立**：predicated task 永不 early-dispatch（`pto_scheduler.h:872/938/968`）；
**predicate 本身不阻塞 orchestrator**（`pto_orchestrator.cpp:966-985` 只算地址不等待）
⇒ 会阻塞的**只有动态 grid 定尺**那一类读。这一条直接决定了修复方向。

### 已判定候选（⛔ NO-GO）：`dispatch-orch-decouple-20260821`（静态 scatter grid）

`0162:/mnt/persist/chensiyu/workspace/perf-2026q3/dispatch-orch-decouple-20260821/candidate`，
`decode_fwd.py` sha `c5d87e259c3aff264fe7debaeb6f978a63d735475405e1282a6ccdce5bf3b645`
（基线 = R5 `67b73589…`，全树 `diff -rq` 只有 `decode_fwd.py` 一处）。

改法：`combine_scatter` 的 grid 由 `pl.read(local_route_count,[1])` 改成静态常量
`COMBINE_SCATTER_WORKERS = 8`，active-expert 上界**下沉进 kernel 体**（device 侧读，不 gate
任务提交），worker 用 `pl.range(worker, n_local_experts, scatter_blocks)` grid-stride
——该形式 `dispatch_gather` 已在用，不引入未验证的动态 stop。

**⚠ 当初写的立项理由已被自己的 device 门证伪，抄录在此以免重犯**：
~~"它是唯一同时消掉 amplifier 又**不删任何同步点**的改法"~~ ——
**错在把"同步点"只理解成 `pld.system.wait` / notify。那个阻塞标量读本身就是一个同步点
（对 orchestrator run-ahead 的背压阀），改静态 grid 恰恰把它删了。**
仍然成立的那半句：**语义不变 ⇒ 精度应逐字节相同**，所以精度门自身是强正确性检查。

**已过：无卡 codegen 门**（`bin/run_compile_gate.sh`，cap16 + cap32 双档 `COMPILE_MATRIX_PASS`）。
**配对结构验证**（同门同镜像编译 R5 与候选，比 orchestrator 生成码）：

| | R5 基线 | 候选 |
|---|---:|---:|
| `get_tensor_data` 总数 | 11 | **7** |
| 打在 `ext_seq_lens` / `ext_num_tokens_per_owner`（host 张量，安全） | 7 | 7 |
| 打在 **`local_route_count`**（producer 含跨卡 wait） | **4** | **0** |
| `scatter_blocks` | `= active_expert_count_inline*` | `= static_cast<int64_t>(8)` |

⇒ 4 处（4 个 MoE variant 各一处）orchestrator 阻塞读**全部消除**，剩余全是 host 供给张量。

### ⛔ device 门：三臂全挂 ⇒ 该方向 NO-GO（且第三臂收敛了机制）

| 臂 | sha | sched 超时 | 结果 |
|---|---|---|---|
| WORKERS=8 | `c5d87e25…` | 默认 | `inv=10` 挂（9 步完成）；`orch_error_code=2` **HEAP_RING_DEADLOCK**；无 FATAL |
| WORKERS=1 | `75b1dd6c…` | 默认 | `inv=1` 挂；`sched_error_code=100` `S1:running-stalled` `completed=1551/1744` **`orch_done=1`**；卡 `ring3/local1446` = `swa_moe_chip_orch_combine_wait`；无 FATAL |
| WORKERS=1 | `75b1dd6c…` | **45 000 ms** | `inv=357` 挂；**HEAP_RING_DEADLOCK**（与 w8 同签名）；`sched_error_code=0`；无 FATAL |

**第三臂推翻了第二臂的读法**：w1 **从来不是 rendezvous 死锁，只是慢** —— 默认 scheduler
看门狗先响而已。抬高超时后它跑了 357 个 invocation，然后撞上**和 w8 完全相同**的 ring 死锁。
⇒ **两臂一个机制**（原写的"两者不是同一根因"**已撤回**），共同点就是「移除了节流」。

`(ring,local)` 映射可复用 `depgen-r9-nonce` 的 `deps.json`：三个候选 task 总数都是 **1744**
—— 改 `pl.spmd` grid 只改 `block_num`，**不改 task 数**，所以 local id 不移位。
顺带确认：基线 `combine_scatter` 在生产配置 depgen 里就是 `block_num=1`
⇒ **BS1 下 `active_expert_count == 1`**，所以 WORKERS=8 是实打实的 8× block。


### ★ 关键发现（比候选本身重要，且改写了修复方向）

**那个阻塞标量读不只是缺陷，它还是一个承重的流控阀。**
`RUNTIME_LOGIC.md`（a2a3 `host_build_graph/docs/`）：
- **§4.4** —「ring 耗尽时 orchestrator **阻塞**」⇒ ring 就是对 orchestrator run-ahead 的背压；
- **§4.5** — ring 太小会因 **scope 引用**死锁（`fanout_count` 含一个 scope 引用，只在
  `scope_end()` 释放，而 `scope_end()` 由 orchestrator 调用，orchestrator 又在等 ring 空间）。

原来那个每层一次的阻塞读把 orchestrator 的 run-ahead 限制在一层内。删掉它 ⇒
**`orch_done=1`（一次把全部 1744 task 提交完）** ⇒ **整个调度体制变了**。
w8 因「8× block/scatter task」比 w1 早 36× 撞墙（`inv=10` vs `inv=357`），
**不是两个不同的根因**。
⚠ 至于"run-ahead 无界 ⇒ ring 饱和 ⇒ §4.5 死锁"这条**因果链只是假设**，
算术上不成立，见下方「机制缺口」。**已确立的只有：删掉它 ⇒ `orch_done=1` ⇒ 撞 ring。**

**⚠ 这不是容量账，加容量修不了**：runner 本来就设了 `PTO2_RING_HEAP=4294967296` /
`PTO2_RING_TASK_WINDOW=131072`；w1 只是把失败从 `inv=10` 推到 `inv=357`。
⇒ 原计划的"加大 ring 容量"实验**已作废**。

#### ⚠ 机制缺口（对抗性自审 —— 已随方向关闭而**不再需要**回答）

`orch_done=1` 是**观测**；"run-ahead 无界 ⇒ ring 饱和"是**推测的机制，而且算术上不足以
解释数据**：

```
一次 invocation = 1744 task     ring task window = 131072 slot
=> 单次 invocation 无论 run-ahead 多深都填不满 ring（差 75 倍）
```

可是 w1 死在 `inv=357`、w8 死在 `inv=10` ⇒ **资源必然是跨 invocation 累积的**
（每轮有一部分 slot / heap 没被回收）。另外两臂很可能**受限于不同资源**
（w8 的 8× block 主要吃 heap 字节，w1 主要吃 slot），这也只有累积才解释得通。

⇒ **真正的缺陷可能是「回收 / 泄漏」被这个改动暴露，而不只是「提交得太超前」。**
**若是泄漏，本地节流未必能修。**

**因此：实现本地节流之前先做这件便宜事** —— 读 ring allocator 与
`scope_end()` / `on_task_release` / `release_producer` 路径（`pto_scheduler.h`、
`pto_orchestrator.cpp`），并查 runtime 是否每 invocation 打印 ring 占用
（错误提示提到过一条 `Ring buffer sizes:` 信息行），判定：

- **(a) run-ahead 深度** —— 那么"恢复一个每层节流"就是对症的；
- **(b) 每 invocation 未回收** —— 那么先要找到谁没释放，本地节流只是掩盖。

**别在没判定之前烧 device 门。**（这正是根因 9 的教训：候选的立项前提要先被审。）

> ⛔ **收盘：以上判定（原 #6）不必再做。** #7 表明 orchestrator 不在关键路径 ⇒ 无论是
> (a) 还是 (b)，都不存在值得换取的收益。保留本节仅作为**下次遇到
> `HEAP_RING_DEADLOCK` 的检查清单**：先算「单次 invocation task 数 vs ring 容量」，
> 若单次填不满，那就不是 run-ahead 深度问题、是累积/回收问题。

**修正后的设计规则（取代 MECHANISM.md v2 里更简单那条）**：orchestration 级阻塞标量读若其
producer 含跨卡 wait，是隐患；**但不能直接删** —— 必须换成一个**只依赖本地 device 进度**的
节流，否则是把稀有的概率性跨卡死锁换成确定性 ring 死锁。
**免费早期判据：只要日志出现 `orch_done=1`，run-ahead 已无界，该候选注定撞 ring。**

### 曾经的下一个候选 = 本地节流（⛔ 已随方向关闭，**不要实现**）

> ⛔ **2026-08-21 收盘：#7 量化完毕 —— orchestrator 不在关键路径，去掉阻塞读 ROI = 0。
> ⇒ 本候选（#5）与「判 ring 耗尽机制」（#6）一并关闭。**
> 下面保留设计与三条前置核对，仅作为**知识**：如果将来因别的原因需要动这个阻塞读，
> 这是唯一安全的替换形态（换节流，而不是删节流）。

保留一个阻塞 grid 读，但指向**纯本地** task 产出的标量。`dispatch_count_publish`
（`decode_fwd.py:1745`）已存在且纯本地（remote_store + notify，**从不 wait**），
其本地输出 `self_meta`（`:1739`，`pl.write(self_meta,[e],…)`）是独立张量、producer 唯一
⇒ orchestrator 节流在本地 device 进度上，不再挂在 `dispatch_meta` 的跨卡 rendezvous 上。

**grid 值与正确性完全无关**（这条纠正了我先前写的"需要 ≥ 真实 active-expert 数的上界"）：
候选 kernel 体是 grid-stride `pl.range(worker, n_local_experts, scatter_blocks)`，worker 取遍
`[0, scatter_blocks)` 时并集恰为 `[0, n_local_experts)`，**任何 `>=1` 都全覆盖**。
实测佐证：w1 以 `grid=1` 跑过 357 invocation、无 credit 错误（`combine_wait` 的
`value=n_local_experts` / `expected=moe_epoch*n_local_experts` 与 grid 无关）。
⇒ **语义不变、精度应逐字节相同**，精度门自身即强正确性检查。

⚠ 诚实边界：**任何 task 都传递地依赖 attention 的 `tp_all_reduce`**，所以本地节流不能让
orchestrator 对一切跨卡停滞免疫。它消掉的是**特定**耦合 —— 不再等一个*体内自旋在 MoE epoch
计数器上*的 task，而是等一个纯本地计算 task ⇒ 退化成"落后于 device 进度"这一正常背压。

#### 实现前必须先核对的三件事（都很便宜，别跳）

1. **`self_meta` 是否只有一个 writer？** 它在 `decode_fwd.py:1739` `pl.create_tensor`、
   在 `dispatch_count_publish` 体内 `:1788-1792` 被写。**但我没有验证后面是否还有别的 task
   写它**。`wait_for_tensor_ready` 是**按张量**取 producer 的（这正是 `local_route_count`
   即使被 `dispatch_count_publish` 先写过 `[0]/[1]`，orchestrator 仍要等 `dispatch_meta` 的
   原因 —— 最后的 writer 才是 producer）。⇒ **先 grep 全文 `self_meta`**；若有第二个 writer，
   这个候选的整个前提就没了，必须换一个专用的 1 元素张量。
2. **`self_meta` 不在 combine 函数签名里**（签名见 `:2596-2603`，只有 `recv_meta_local` /
   `local_route_count` 等）⇒ 需要**穿参 + 改调用点**，diff 不再是"单点"。
   这削弱了隔离性，A/B 归因时要声明。
3. **`pl.range` 的 step 变成动态量**。静态 grid 那版是 `pl.range(worker, 36, 8)`（常量 step，
   编译期可展开成约 5 次）；本地节流版 step 来自 `pl.read` ⇒ 不可展开。
   这**避开**了 `known-pypto-pitfalls §7`（`pl.range(常量)` 展开且不复用 SSA buffer → UB 溢出），
   但换成了未验证的动态 step 形式 ⇒ **无卡 codegen 门必须先确认它能编过**，
   并顺带确认 `get_tensor_data` 落在 `self_meta`、**不再**落在 `local_route_count`。

#### 反向复核的五条（reviewer 角色，2026-08-21）

1. **前提未确立** —— ring 耗尽是「回收/泄漏」还是「run-ahead 深度」未判（见上）。
   若是泄漏，本地节流只是把累积变慢。**#6 先做。**
2. **有一条未核实的事实依赖** —— `self_meta` 必须只有一个 writer（producer 按张量取
   **最后一个** writer）。有第二个 writer 则整个设计作废。
3. **不再是单点 diff** —— 需要穿参改调用点，削弱了 R9 那种"教科书级隔离"属性，
   A/B 归因要声明。
4. **引入未验证的 codegen 形式** —— 动态 step `pl.range`。
5. **★ 收益从未被测量过** —— 去掉 orchestrator 阻塞的 ITL ROI 是 **0 个数据点**，
   而五层 DFX 指向的瓶颈是 **WAIT**、不是 small-op dispatch latency。
   同时 **R5 是已发布且能跑的基线**。
   ⇒ **当前计划等于「拿一个没被证明存在的收益，去换一个可能给工作基线引入概率性 hang 的风险」。**

**⇒ 因此把顺序倒过来：先量化，再改码。** #7 = 只用 0162 已有 artifact（不占卡、不改码）
解析 runtime STRACE 的嵌套 span（`simpler_run ⊃ {bind, runner_run}`、
`runner_run ⊃ device_wall ⊃ graph_build ⊃ sched ⊃ orch`，每 invocation 一组、`clk=dev`、
`dur` 单位 ns），比较 R5（有阻塞读）与 w1（无阻塞读）的 `orch` 与 `device_wall`。
工具：`analysis-bin/orch_span_stats.py`（丢弃 warmup `inv<10`，8 rank 汇池）。

#### ✅ #7 的答案（2026-08-21）：orchestrator 不在关键路径 ⇒ 整条线关闭

p50 表见本节开头。要点：**`orch` 降 12836 µs（−74.3%），`device_wall` 反升 443 µs。**
⇒ orchestrator 与 device 并发、两种情况下都提前完成 ⇒ 去掉阻塞读 **ITL ROI = 0（微负）**。
⇒ **#5 / #6 关闭，生产继续 R5 不动。** 落地口径与第五条反向复核完全一致：
「收益从未被测量过」这条一旦被测量，答案是"没有收益"。

保留下来的价值 = 一条负结果 + 两条可复用设计规则：
「不要让动态 spmd grid 定尺标量指向一个体内含跨卡 wait 的 producer」
+ 「这类读同时是承重节流，不能直接删」。



### dispatch 域还剩什么可融合 —— 逐个都撞已文档化的硬约束

| 候选融合 | 撞到的硬约束 |
|---|---|
| `dispatch_meta` + `dispatch_wait` | orchestrator 阻塞读会等**更多**跨卡进度 ⇒ 加剧 amplifier |
| `dispatch_count_publish` + `dispatch_push` | task 边界**正是** self-row 可见性 fence（`:1740-1744` 注释） |
| `dispatch_wait` 吸进 `dispatch_gather` | 违反「不要把跨卡 wait 吸进 compute kernel」（`S4` 会退化成 `S1`） |
| `dispatch_push` / `dispatch_gather` 的 grid | 本来就由 **host** `num_tokens` 定尺（`:1810`/`:1817`）⇒ 无 orchestrator 阻塞读 |

⇒ 叠加 DFX 那条「收益被 `dispatch_wait` 吸走（`2.18→12.56 µs`）、瓶颈是 **WAIT** 而非
small-op latency」，再叠加 #7 的量化（orchestrator 不在关键路径）：
**dispatch 域小算子融合终态关闭，不留"先换节流再谈融合"的后路。**

### 一个可复用的判别法（本轮新得）

**`SIMPLER_SCHEDULER_TIMEOUT_MS` 能把 `S1` 分成「真死锁」与「只是慢」** ——
`S1:running-stalled` 只说明看门狗到点，不区分"永不完成"与"比超时慢"。env-only、零改码，
是 runtime 自己文档化的判别法。本例它当场推翻了一个错的双根因结论。
runner：`gate-bin/run_gate_schedms.sh`（`SCHED_MS=45000`）。


### 这两件便宜事已经做完了（结论已并入上文，不要重跑）

1. ✅ **w1 + `SIMPLER_SCHEDULER_TIMEOUT_MS=45000`**（`gate-w1-sched45s/`）：判出 w1 是
   **只是慢**、不是死锁 —— 跑到 `inv=357` 才挂，且挂法与 w8 相同。
2. ⛔ **w8 + 加大 ring 容量**：**作废，不必跑**。runner 本来就已经是
   `PTO2_RING_HEAP=4294967296` / `PTO2_RING_TASK_WINDOW=131072`；w1 已经证明加容量只是把
   失败从 `inv=10` 推到 `inv=357`。无界 run-ahead 下任何有限 ring 终会饱和。

**仍然未测的唯一一项** ✅ **已测**（#7，2026-08-21）：去掉 orchestrator 阻塞的 ITL ROI = **0（微负）**。
方法不是"跑一个能活下来的候选"，而是**从两个已失败臂的 STRACE span 里直接读**——
失败臂在挂之前跑过几百个 invocation，`orch` / `device_wall` 的稳态 p50 早就够统计了。
**教训：不要为了拿一个数去写新候选，先看现有日志里是不是已经有那个数。**


### ★★ 下一条性能主线（本轮副产品，量级远大于 dispatch 域融合）

同一份 STRACE 里，R5 每 invocation：

| span | p50 | 占 ITL |
|---|---:|---:|
| `simpler_run` | 26.45 ms | ≈ ITL p50 26.329 ms ✅ 对得上 |
| `bind.args` | **6.12 ms** | **≈ 23%** |
| `runner_run` | ≈ 20.3 ms | — |

`simpler_run ≈ bind + runner_run`（加性、host 侧串行）。w1 侧同量级（5.87 ms）⇒ 不是偶发。

**`bind.args` 约 6.1 ms/step、约 ITL 的 23%，纯 host 侧参数绑定** ——
比 dispatch 域任何 small-op 融合的可得收益**大一到两个数量级**，且不碰 device 语义、
不碰跨卡同步、不动 `@pl.program` 结构。**⇒ 建议作为下一条主线先去核实**
（先确认它是否真在 ITL 关键路径上、能否与 device 执行重叠或跨 step 缓存）。


⚠ **runtime 提示里的 env 名不可信**：`error_names.h:172` 让你调
`PTO2_SCHEDULER_TIMEOUT_MS`，**该 env 不存在**；真实变量是 **`SIMPLER_SCHEDULER_TIMEOUT_MS`**
（`runtime_timeout_config.h:25`，读于 `:165` 与 `device_runner_base.cpp:90`），
且受 `scheduler_timeout_us < op_execute_timeout_us` 约束（实测 op 超时 50000 ms，故用 45000）。
同族：`PTO2_TENSOR_DATA_TIMEOUT_MS` 是编译期 constexpr、根本不可设。
**一律去源码 grep `getenv` 核对。**

**判定规则（预登记，不变）**：liveness 失败或 `hidden_byte_exact=false` ⇒ NO-GO。两者皆过再上
A/B/A 计时门（**整机锁、两半皆空**）；p50 单次读数**不作为收益记账**，只作是否值得上 A/B/A 的筛子。

**权威记录**：`0162:…/dispatch-orch-decouple-20260821/FINDINGS.md`（含全部签名、映射、
源码引用与"明确未确立"清单）。

**诊断复现器（半机锁，~12 min，命中率约 2/3）**：

```bash
B=/mnt/persist/chensiyu/workspace/perf-2026q3/dispatch-fusion-triage-20260821
R9=/mnt/persist/chensiyu/workspace/perf-2026q3/moe-routed-packed-fusion-20260815/dispatch-split-comm-boundary-r9-20260820-225253
ITERS=1000 bash $B/runner/faultlog_r9.sh "$R9" "$B/<new-out-dir>"
```

`(ring, local)` → kernel 名的工具：`$B/depgen-bin/lookup_task.py <deps.json> <kernel_config.py> <ring> <local> [radius]`
（`deps.json` 由带 `PYPTO_DISTRIBUTED_DEP_GEN=1` 的**生产配置** run 产出，见 `depgen-r9-nonce/`）。

**⚠ 未验证的风险**：`ASCEND_SLOG_PRINT_TO_STDOUT=1` 的日志量可能扰动时序；若连续几轮都
PASS，先关掉它、只留 `ASCEND_PROCESS_LOG_PATH` 文件落盘（`ASCEND_GLOBAL_LOG_LEVEL=3` 必须保留）。

### 前提问题：这条线可能反而是有价值的

`DFX_R5_R7_COMPARISON`（五层）显示 dispatch 域收益**不传播** ——
`dispatch_count_start_to_gather_end` `40.08→36.74`（−3.34 µs），但
`dispatch_count_start_to_gmm1_start` `78.05→78.32`（**+0.27 µs**）；`dispatch_meta` 7.12
消失而 `dispatch_wait` `2.18→12.56`（+10.38）把收益吸走 ⇒ 瓶颈像 **WAIT** 而非 small-op 延迟。
**但有一个方向相反的数据点**：R9 跑通那次整网 p50 `26.615 ms`，R5 两次是 `27.478` /
`27.757 ms`。**非配对单次比较、不满足 A/B/A bracket、不能当收益记账**，但足以说明那条 DFX
推断（R5-vs-R7、两臂都是五层）对 R9 只是弱证据。
⇒ **修掉 liveness 隐患后重估这条线，而不是直接废弃。**

**两个 runner 硬要求**（踩过）：① ITL/faultlog arm **必须 `--privileged`**，否则
24.86 GiB VMM weight pool 的 `aclrtMalloc` 直接 `rc=107002`（卡是空的，**不是 OOM**）；
② `PYPTO_DEVICES` 选的是**物理**卡号，privileged 下 `--device /dev/davinci*` 是摆设
—— 老 runner 里 `devices=8-15` 的 lock 注释是错的，实际跑在 0-7，所以**诊断只需
`0162-cards0-7.lock`**（只有 A/B/A 计时门才取整机锁）。

**一条旧记录更正**：`_route_anchor` / `_routed_anchor` / `dump_phys` 在 R5 与 R9 都各
出现一次，是 baseline 既有残留，不是 R9 引入，不在任何 R9 清理范围内。

流程教训已写进 [`../postmortems/12-integration-churn-meta.md`](../postmortems/12-integration-churn-meta.md) §3 根因 6/7/8 + §5 + §6。

## 1. 当前判定

TP all-reduce 单行优化已完成源码落地和 source-overlay 最终门：

```text
pypto-lib stepfun/develop
  HEAD    69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
  parent  9ca01d243e534949287fa769e5be35031ebc4be7
  tree    e26d762cb8c4abd49a1546e7db2beddeb6480e14
```

- GitHub remote 与 0162 指定 checkout 对齐、clean；
- Main BS1 使用静态 `1×4096` 两波 one-shot mesh；
- Main 多行与 MTP 使用静态三波 fallback；
- Whole A/B/A：`31.065 / 29.912 / 30.999 ms`；
- candidate：`-1.120 ms / -3.609%`；
- precision/per-iteration gate：PASS；
- immutable image：**未构建**。

## 2. 源码位置与最终实现

0162 指定 checkout：

```text
/mnt/persist/chensiyu/workspace/develop/pypto-lib
branch  stepfun/develop
HEAD    69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
origin  69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
status  clean
```

最终 selector：

```text
Main active_rows == 1
  static 1×4096 self-TPUT
  -> Wave 1 publication
  -> fixed rank-order full-row remote loads
  -> one FP32 accumulator / one final BF16 cast
  -> Wave 2 completion

Main active_rows != 1
  static three-wave reduce-scatter + push-all-gather fallback

MTP
  shared ABI, three calls pass static BATCH, always static fallback
```

ownership 必须保持：

```text
TP_ALL_REDUCE_OWNED_CHUNK = HIDDEN // TP_WORLD_SIZE = 512
```

它与 `TP_ALL_REDUCE_CHUNK` 的 staging/final-copy transfer grain 解耦。

源码兼容性提醒：`dense_mlp_body_tp` 在 `mlp_layer_idx` 后新增了
`num_tokens: pl.Scalar[pl.INT32]`。仓内 Main 已传运行时 `num_tokens`，MTP 已传
静态 `BATCH`。仓外的直接调用方以及
`pl.inline(dense_mlp_body_tp._func)` 调用方升级时必须同步补该位置实参；这是源码
调用 ABI 变化，不能沿用旧参数表。

## 3. 最终验证

```text
canonical/two-layer AST       FINAL_STATIC_SELECTOR_CONTRACT_PASS
unit                           365 passed, 7 skipped
ruff / diff-check              PASS
Whole compile default          PASS
Whole compile chunk=256        PASS
MTP 3/3 default                PASS
MTP 3/3 chunk=256              PASS
8-card rows 1/3/16             PASS
```

rows `1/3/16` 覆盖 single-row smallmesh 与两档 multi-row fallback；该 device
matrix 未持性能锁，只是功能证据。

Whole BS1 / ctx64K / 512 blocks / warmup 10 / measured 100：

```text
A1 9ca static fallback p50     31.065 ms
B  final-tree smallmesh p50    29.912 ms
A2 9ca static fallback p50     30.999 ms
baseline center                31.032 ms
candidate delta                -1.120 ms / -3.609%
performance                    IMPROVEMENT_BEYOND_BRACKET
precision/per-iteration        PASS / PASS
```

B 臂 `b67afe77` 与最终 landing `69ad31e` 的 Git tree 相同。三臂 hidden SHA：

```text
567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e
```

三臂 finite、TP spread=0、tail token `14371` exact。

five-layer 只声明 L3/L4 exact、finite、TP spread=0，并提取了 regular-call
kernel-duration pooled mean；
既有 zero-token canonical structural analyzer 仍 fail-closed。

## 4. 权威产物

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  tp-allreduce-hccl-smallmesh-validation-20260812/final-static-fallback/

whole-aba/out/final-aba-bs1-ctx64k-20260812-174433/ABA_RESULT.json
  sha256 383caa23124c7da42d676ef642bc8b488344349564fd4131efa560c6b5ea3757
```

固定验证镜像：

```text
manifest sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3
config   sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
```

镜像内 `pypto-lib` 仍为 `cb96747e`；`69ad31e` 是 read-only source overlay，
runtime 无 overlay。

## 5. 已退休的分支

### `a791071` attention-inline Ring

该实验没有命中 production canonical `WholeDecodeStep3p5.tp_all_reduce`，实质为
A/A；compile OK 不能推断 device correctness 或性能。不得继续扩展或恢复。

### `b4d45b3` K6b dynamic-valid-shape

动态 publish/final-copy 虽能过部分 codegen，但 self-TPUT/remote-load 仍受静态
shape 约束。dynamic publish 位于已知 notify-fence seam；现有设备运行未复现错误，
但没有针对该 seam 的独立 rank-skew/zero-gap/多 epoch safety proof。该分支只保留为
focused 历史证据，不得写成“可进产品、不必上卡”。

## 6. 正确性硬约束

1. `active_rows` 必须在所有 TP rank 上一致，否则 selector 分叉会死锁；
2. 固定 peer 顺序、单 FP32 accumulator、一次 BF16 cast 不得改变；
3. 两波与三波不能复用未清零的同一 signal slot；
4. exact two-layer mirror 必须继续与 canonical body AST 一致；
5. 本实现不等于修复 notify fence；未来合并波次或把 payload store 与自己的
   credit 拉近，仍受 `UPSTREAM-NOTIFY-FENCE` 阻塞；
6. 仓外 `dense_mlp_body_tp` 调用点必须传新增的 `num_tokens` 实参。

## 7. 下一步

1. 基于 `pypto-lib@69ad31e` 构建 immutable candidate image；
2. 固定 manifest/config 和所有组件 pin；
3. 在新镜像上重跑 Whole A/B/A、Main N=128、多 batch、MTP、canonical
   structural analyzer；
4. source-overlay 与 image gate 分账；新镜像闭环前不得写
   production/release-qualified；
5. 不再启动 Ring 或 dynamic-valid-shape 产品化，除非出现新的独立证据。

## 8. 机器与操作约束

后续启动前重新检查锁、container、`fuser` 与 NPU process，不能沿用旧 session
的空闲结论。

禁止事项：

- 在本地项目仓创建或修改 pypto-lib 产品代码；
- 用未持锁的 device matrix 作为性能数据；
- 把 focused regular-call kernel-duration pooled mean
  `38.325 → 22.667 µs/call` 当作 strict critical-tail 或最终完整源码 A/B/A；
- 用 host 独立检查覆盖 canonical structural fail-closed；
- 把 source-overlay 数据写成 immutable-image 结果。

## 9. vLLM-Ascend MoE Trace 对齐

0162 已建立独立 clean worktree：

```text
/mnt/persist/chensiyu/workspace/develop-worktrees/
  vllm-moe-trace-align-20260812

branch  perf/vllm-moe-trace-align-20260812
HEAD    9ca01d243e534949287fa769e5be35031ebc4be7
```

K8 digest `076af8…` 上 source-overlay whole compile 为
`COMPILE_OK 11.4s`；只代表编译兼容。本次容器 PATH 找不到 `pytest`
命令、`rc=127`，单测未运行；不能据此推断镜像内完全没有 pytest package。

Trace 结论：

```text
68 structural decode replays = 34 + 34
45 Main layers; zero-based L3-L44 are 42 MoE
Main p50              18.067 ms
Main + MTP3 p50       27.138 ms
MoE core p50             229.75 us
Main collectives      91 AR, 0 AG, 0 A2A, 0 MC2
```

捕获路径是 local routing + grouped experts + TP AllReduce，不是 PyPTO 的
EP8 dispatch/combine。不要直接照搬路由/通信拓扑。本轮没有产品 kernel commit；
整体阶段已按 input/router/route organization/routed GMM1/down/combine/
shared-global merge/residual 对齐；regular GMM1+SwiGLU+requant 是唯一
优先进入 P1 的 fusion candidate，需先过 UB/codegen/精度/ROI feasibility
gates；其它阶段保持语义边界、只调粒度。
P1 目标为 regular L3–L42 的 `routed_gmm1_swiglu_quant` primitive，加
`(expert, source-rank)` combine data-work bundle。down 和 L43/L44
specialization 保持独立；completion 必须选择 per-expert aggregator 以保留
36 credits，或版本化 `N_COMPLETIONS` 并重验 epoch/wraparound。生成码还必须
证明 payload 后、credit 前有显式 `PIPE_ALL`/release seam，不能假设 notify
自带 release。

完整三列表与验收门：
[`../benchmark/2026-08-12-vllm-ascend-decode-moe-trace-gap.md`](../benchmark/2026-08-12-vllm-ascend-decode-moe-trace-gap.md)。

