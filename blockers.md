# 活跃 Blocker

阻塞项目进展的 **open** issue 的 SSOT。每条：症状 / 根因 / 当前状态 / 解除条件 / 链接。

**协议**：blocker 解决时，**删掉本文件这一节** → 到 [`postmortems/`](postmortems/) 建一篇
五段复盘（模板 [`postmortems/TEMPLATE.md`](postmortems/TEMPLATE.md)）+ 更新
[`STATUS.md`](STATUS.md) blocker 摘要。已解问题不留在本文件。

**已解 blocker 的复盘去向**：见 [`postmortems/README.md`](postmortems/README.md)。
如 507899/507018、co-tenancy(G4)、tmov Vec-LHS、gate_topk、多程序 co-prepare 死锁、
gap-5、scheduler-timeout、attention 乱码、G5b import_ipc、swa_moe const-fold 等均已归档。

**最后检视**：2026-08-21。

---

## 🔴 ACTIVE — ORCH-SCALAR-READ-VS-CROSSRANK-WAIT：动态 grid 定尺的阻塞标量读撞上跨卡 wait（R9 概率性 S1）

> **⚠ 2026-08-21 二次更正：本条曾写"机制已闭合 / 完整死锁环"，那个结论已被我自己的对抗性
> 复核推翻，见下方「已撤回」。** 当前诚实口径：**停点、阻塞分支、阻塞对象三者已确立**；
> **闭合的死锁环未确立**。仍然成立的结构性主张只有一条 —— 那个阻塞标量读把 orchestrator
> 的前进耦合到跨卡进度上，因此任何上游跨卡停滞都会被**放大**成全 rank 的 orchestrator 冻结
> （**deadlock amplifier**，不等于 root cause）。该耦合在 R5 里逐字节相同。

**候选规模（先说清，避免被当成大改）**：R9 相对生产基线 R5，`decode_fwd.py` 的**全部差异
只有一行** —— `dispatch_gather` 的 `deps` 多了一个 `dispatch_push_tid`：

```diff
 with pl.spmd(scan_blocks, name_hint="dispatch_gather",
-    deps=[wait_tid, meta_collect_tid],
+    deps=[dispatch_push_tid, wait_tid, meta_collect_tid],
     predicate=(local_route_count[0] > 0), allow_early_resolve=True)
```

**症状**：R9（`dispatch-split-comm-boundary-r9-20260820-225253`，`decode_fwd.py`
sha `dd0e9cea…`）在**生产配置**（`MAX_SEQ=ROPE_SEQ=65536`、`--num-blocks 512`、
BS1、ctx 64K）整网跑到 `inv=95`（110 次调用中）挂死：

```
orch_error_code=8 TENSOR_WAIT_TIMEOUT   sched_error_code=100  runtime_status=-8
sub_class=S1:running-stalled (detail=1)
completed=546/1744 running=1 ready=0 waiting=30 orch_done=0
stuck_task_id=12884902515   ->  ring=3 local=627
stuck_core=24 / 26 / 27 / 28   （7 个 rank 同时停在同一 task => 跨卡）
```

⚠ **不是每次都挂**：同配置复跑一次 PASS（p50 `26.615 ms`）。详见下方「重要更正」。

**小配置下未复现**：同一棵树在 `MAX_SEQ=4096 / --num-blocks 32` 下整网单步 PASS、
N=128 replacement-equivalence PASS（256/256 tensor 与 R5 逐字节相等）。生产配置确实是
另一张 task 图（1744 vs 1702 task），但**不能**据此断定"小配置门结构上看不见"——
既然生产配置本身 2 次里只挂 1 次，小配置那 2 次全过完全可以是概率所致（见下方更正）。

**根因**：**已定位（2026-08-21）—— 见下方「完整死锁环」。本质缺陷是结构性的、R5 也潜伏。**

*停点已兑现成 kernel 与谓词*（用生产配置 + `PYPTO_DISTRIBUTED_DEP_GEN=1` 的 `deps.json` 解析）：

```
stuck_task_id=12884902515  =  ring 3 / local 627
                           =  swa_moe_chip_orch_combine_wait   (block_num=1, early=True)
```

未满足的自旋谓词（`decode_fwd.py` `combine_wait` 体内）：

```python
pld.system.wait(signal=combine_arrived, offsets=[src, 0],
                expected=pl.cast(moe_epoch * n_local_experts, pl.INT32),
                cmp=pld.WaitCmp.Ge)      # 对每个 src != my_rank
```

**该谓词应由谁置位**：rank `src` **自己的 `combine_wait`**——同一个 task 先 notify 全部 peer
再 wait 全部 peer，所以 `combine_wait` 是一个**对称 all-rank rendezvous**：

```python
for peer in ...: if peer != my_rank: pld.system.notify(combine_arrived, peer,
                     offsets=[my_rank,0], value=n_local_experts, op=AtomicAdd)   # 先 notify
for src  in ...: if src  != my_rank: pld.system.wait(...)                        # 再 wait
```

⇒ **停在 `combine_wait` 里的 rank 都在等别人 notify。** 但"因此存在一个唯一元凶 rank"这一步
**已撤回**——见下方「已撤回」。

*曾被当作元凶识别的少数派报告*——8 个 rank 的 `sched_error_code` 分布是
**7×`100` + 1×`0`**；那 1 个 `sched_error_code=0` 的 rank **不产出 `sub_class=` 行**，
只报 orch `TENSOR_WAIT_TIMEOUT`，即**它的 scheduler 没停、是 orchestrator 卡住**：

```bash
grep -E "PTO2 runtime failed: orch_error_code=" container.log | sed -E "s/.*failed: //" | sort | uniq -c
#   7 orch_error_code=8 sched_error_code=100 runtime_status=-8
#   1 orch_error_code=8 sched_error_code=0   runtime_status=-8
```

**这个差异说明的是「哪个子系统停了」，不是「哪个 rank 有责任」。** 少数派报告仍是有用的技巧
（它把 orchestrator 与 scheduler 区分开了），但据此推出"1 个元凶 + 7 个受害者"是错的：
按 AICPU pid 计数，**8 个 orchestrator 全部阻塞**（见下）。

`TENSOR_WAIT_TIMEOUT` 在 runtime 里只有 **2 个** raise 点，都在
`pto_runtime2.cpp::wait_for_tensor_ready`：① 读路径「producer 未 COMPLETED」；
② 写路径「consumers of producer not done」。**2026-08-21 已拿到 device 答案 = ① 读路径**：

```
FATAL(code=8): Timeout (750000000 cycles): producer (ring=3, local=933) not completed
```

`ring3/local933` = **`swa_moe_chip_orch_dispatch_meta`**（`deps.json` + `kernel_config.py` 解析）。

### 已确立（device + 源码三方吻合）

1. orchestrator 需要 `combine_scatter` 的**动态 spmd grid 大小**：
   `scatter_blocks = pl.read(local_route_count, [1])`（`decode_fwd.py:2611`）。
   orchestration 级 `pl.read` 在 AICPU 上就是 `get_tensor_data()`。
2. `get_tensor_data` → `wait_for_tensor_ready(..., wait_for_consumers=false)`
   （`pto_runtime2.cpp:221`）→ 自旋等该 tensor 的 **producer** 到 COMPLETED，超时即报
   `producer (ring=%d, local=%d) not completed`（`:128`）。**读路径已由 device 证实。**
3. `local_route_count` 的 producer 正是 **`dispatch_meta`**（它执行
   `pl.write(local_route_count, [0]/[1], …)`）—— 与 FATAL 指名的 `ring3/local933` **吻合**。
4. `dispatch_meta` 体内**跨卡自旋**：
   `pld.system.wait(signal=meta_arrived, offsets=[src,0], expected=moe_epoch, cmp=Ge)`。
5. **8 个 rank 的 orchestrator 全部**卡在同一个 producer 上。按 AICPU pid 计数：

```bash
# 8 个不同 pid × 每个 2 次发射 = 16 条，producer id 无一例外
grep -ho "AICPU([0-9]*," container.log | sort | uniq -c   # => 8 个 distinct pid
grep -o "producer (ring=[0-9]*, local=[0-9]*)" container.log | sort | uniq -c
#  16 producer (ring=3, local=933)
```

6. `ring3/local627` = `swa_moe_chip_orch_combine_wait`，是一个**对称
   notify-all-then-wait-all rendezvous**；7 个 rank 的 scheduler S1 停在这里。

### ❌ 已撤回（曾写在本文件里，现已证伪）

> "闭合环：orch 卡在 933 使该 rank 走不到 `combine_wait`(627)，从而饿死 peer，peer 又让 orch
> 继续卡住。"

**为什么不成立**：提交序上 **627 < 933**。一个 orchestrator 已经推进到 933 的 rank，
**必然早已提交过 627**；而一个已经在 RUNNING 的 `combine_wait` 是先 notify 再 wait 的，
它发 notify **不需要 orchestrator**。⇒ **所断言的环没有闭合边。**

同时撤回：「1 个元凶 + 7 个受害者」。8 个 orch 全部阻塞（事实 5），不存在单一元凶 rank。

### ⚠ 未确立（不得当作已知）

- **`combine_wait`(627) 为什么完不成**。若 8 个 rank 都到了它的 wait，则所有 notify 都已发出、
  每个 wait 都该通过。所以**至少有 1 个 rank 不在那里**——而那个 rank 恰好没产出 `sub_class` 行。
  **这个 rank 的真实停点是本案缺失的关键数据。**
- **因果方向**。T1（orch 先卡）无闭合边；T2（计算先停、orch 随后被放大）更可信但**未证**。
- **那一行 diff 究竟是"引发"还是只"改变时序"**，未定。

### 仍然成立的唯一结构性主张

**一个 orchestration 级的阻塞标量读（动态 spmd grid 定尺）落在了一个「体内含跨卡
`pld.system.wait`」的 task 的输出上。** 这把 orchestrator 的前进耦合到跨卡进度上，于是任何
上游跨卡停滞都会被放大成**全 rank 的 orchestrator 冻结**，而 orchestrator 恰恰是唯一能提交
peer 正在等的更早层任务的角色。

⇒ 它是 **deadlock amplifier**，**不是**已证的 root cause。该耦合在 R5 里逐字节相同。

**设计规则（可复用）**：不要让一个 orchestration 级的阻塞标量读（如动态 spmd grid 定尺）
指向一个体内含跨卡 wait 的 producer。

### R5 对照：同配置 10+1000 iters 一轮长跑 **PASS**

`faultlog-R5-iter1000/`：`HARNESS_RC=0`、1000 iters 全过、**无任何 `FATAL(code=`**、
p50 `26.329 ms`。⇒ 与 R9 形成不对称：**R9 生产配置 3 次挂 2 次，R5 在 10× 曝光下 1 次 1 过。**
（n=1，不证明 R5 绝对安全；但足以说明那一行 diff 对 hang 概率**不是**中性的。）

**顺带推翻上一轮留下的"R9 可能更快"猜测**：匹配 `ITERS=1000` 后 R5 p50 `26.329` **优于**
R9 clean run 的 `26.615`。此前拿来对比的 `27.478/27.757` 是 `ITERS=100` 的臂，**不可比**。
⇒ R9 既不活也不快，出局无悬念。

### 修复方向

按"攻击结构耦合"而非"消症状"排序：

1. **⛔ 已试并 NO-GO（2026-08-21）— 直接去掉数据相关的 grid 尺寸**：
   `combine_scatter` 的 grid 改静态常量 `COMBINE_SCATTER_WORKERS`，active-expert 上界下沉进
   kernel 体（device 侧读，不 gate 提交）。候选
   `0162:…/perf-2026q3/dispatch-orch-decouple-20260821/`，`candidate`(WORKERS=8,
   sha `c5d87e25…`) 与 `candidate-w1`(WORKERS=1, sha `75b1dd6c…`)，相对 R5 全树只改
   `decode_fwd.py` 一处。

   **无卡 codegen 门过了，而且结构上完全达标**（同门同镜像编译 R5 与候选做配对）：

   | | R5 | 候选 |
   |---|---:|---:|
   | orchestrator `get_tensor_data` 总数 | 11 | **7** |
   | 打在 host 张量 `ext_seq_lens`/`ext_num_tokens_per_owner`（不可能等 peer） | 7 | 7 |
   | 打在 **`local_route_count`**（producer 含跨卡 wait） | **4** | **0** |
   | `scatter_blocks` | `= active_expert_count_inline*` | `= static_cast<int64_t>(8)` |

   **device 门三臂全挂，且第三臂把前两臂收敛成同一个机制**：

   | 臂 | sched 超时 | 死在 | 签名 |
   |---|---|---|---|
   | WORKERS=8 | 默认 | `inv=10`（9 步完成） | `orch_error_code=2` **HEAP_RING_DEADLOCK**，无 FATAL |
   | WORKERS=1 | 默认 | `inv=1` | `sched_error_code=100` `S1` `completed=1551/1744` **`orch_done=1`**，卡 `ring3/local1446` = `combine_wait`，无 FATAL |
   | WORKERS=1 | **45 000 ms** | `inv=357` | **HEAP_RING_DEADLOCK**（与 w8 同签名），`sched_error_code=0` |

   ⇒ **w1 从来不是 rendezvous 死锁，只是慢** —— 默认 scheduler 超时先响而已；抬高超时后它
   跑了 357 个 invocation，然后撞上**和 w8 完全相同**的 ring 死锁。
   **原先"两者不是同一根因"的中间结论已撤回：两臂一个机制。**

   **⇒ 关键发现（比候选本身重要，且改写了修复方向）**：那个阻塞标量读**不只是缺陷，它还是
   一个承重的流控阀**。`RUNTIME_LOGIC.md §4.4`「ring 耗尽时 orchestrator 阻塞」——
   ring 就是对 orchestrator run-ahead 的背压；`§4.5` ring 太小会因
   **scope 引用**（`fanout_count` 含一个 scope 引用，只在 `scope_end()` 释放，而
   `scope_end()` 由 orchestrator 调用，orchestrator 又在等 ring 空间）而死锁。
   原来那个每层一次的阻塞读把 orchestrator 的 run-ahead 限在**一层内**。删掉它 ⇒
   **`orch_done=1`（一次提交完全部 1744 task）** ⇒ 整个调度体制改变，run-ahead 无界
   ⇒ ring 饱和 ⇒ §4.5 死锁。

   **⚠ 这不是容量账，加容量修不了**：runner 本来就设了
   `PTO2_RING_HEAP=4294967296` / `PTO2_RING_TASK_WINDOW=131072`；w1 只是把失败从
   `inv=10` 推到 `inv=357`。**无界 run-ahead 下任何有限 ring 终会饱和。**
   ⇒ 原计划的"加大 ring 容量确认是否纯容量账"实验**已作废，不必再跑**。

   **⚠ 但机制只到"观测"为止 —— 对抗性自审补一条缺口**：`orch_done=1` 是观测，
   "run-ahead 无界 ⇒ ring 饱和"**算术上不足以解释数据**：一次 invocation 只有
   **1744** task，ring 有 **131072** slot ⇒ **单次 invocation 再深的 run-ahead 也填不满
   ring（差 75 倍）**。而 w1 死在 `inv=357`、w8 死在 `inv=10`
   ⇒ **资源必然跨 invocation 累积**（每轮有一部分 slot/heap 没回收）；两臂很可能
   **受限于不同资源**（w8 的 8× block 主要吃 heap，w1 主要吃 slot）。
   ⇒ **真正的缺陷可能是「回收 / 泄漏」被这个改动暴露，而不只是「提交得太超前」。**
   **若是泄漏，方向 2（本地节流）未必能修。**

   **⇒ 实现方向 2 之前必须先判定 (a) run-ahead 深度 还是 (b) 每 invocation 未回收**：
   读 ring allocator 与 `scope_end()` / `on_task_release` / `release_producer` 路径，
   并查 runtime 是否每 invocation 打印 ring 占用（错误提示提到过一条 `Ring buffer sizes:`
   信息行）。**别在没判定之前烧 device 门**（根因 9 的教训：候选的立项前提要先被审）。

   **修正后的设计规则（取代 MECHANISM.md v2 里那条更简单的）**：orchestration 级阻塞标量读
   若其 producer 含跨卡 wait，是隐患；但**不能直接删** —— 必须换成一个**只依赖本地 device
   进度**的节流，否则等于把一个稀有的概率性跨卡死锁换成确定性的 ring 死锁。

2. **本地节流（⛔ 2026-08-21 已关闭，不要实现；设计保留为知识）**：保留一个阻塞 grid 读，但指向一个**纯本地** task
   产出的标量。`dispatch_count_publish` 已存在且纯本地（它 remote_store + notify，
   **从不 wait**），其本地输出 `self_meta`（`decode_fwd.py:1739`，`pl.write(self_meta, [e], …)`）
   是一个独立张量、producer 唯一 ⇒ orchestrator 于是节流在**本地 device 进度**上，不再挂在
   `dispatch_meta` 的跨卡 rendezvous 上。

   ⚠ **修正一条我自己写错的要求**：先前写"该读只需提供 **≥ 真实 active-expert 数**的上界"，
   **那个要求是多余的**。候选 kernel 体用的是 grid-stride
   `for plan_slot in pl.range(worker, n_local_experts, scatter_blocks)` —— worker 取遍
   `[0, scatter_blocks)` 时并集恰为 `[0, n_local_experts)`，**对任何 `scatter_blocks >= 1`
   都全覆盖**。所以 **grid 值与正确性完全无关**，只是并行度/节流旋钮。
   实测佐证：w1 以 `grid=1` 跑过 357 个 invocation，无 credit 错误
   （`combine_wait` 的 `value=n_local_experts` / `expected=moe_epoch*n_local_experts`
   与 grid 无关，已源码 + device 双证）。⇒ 该候选**语义不变、精度应逐字节相同**。

   ⚠ 仍要诚实的一点：**任何 task 都传递地依赖 attention 里的 `tp_all_reduce`**，所以本地节流
   并不能让 orchestrator 对一切跨卡停滞免疫。它消掉的是**特定**耦合 —— orchestrator 不再等一个
   *体内自旋在 MoE epoch 计数器上*的 task，而是等一个纯本地计算 task ⇒ 退化为"落后于 device
   进度"这一正常背压语义。
3. 限制 orchestrator run-ahead（若 runtime 提供旋钮，比模型侧假读更干净）。
4. 回退那一行（= R5）。恢复已发布基线，耦合仍在。

⚠ **反向教训（R9 亲身示范）**：把 `dispatch_meta` 与 `dispatch_wait` **合并**成一个控制 task
（看似省一个小算子）会让 orchestrator 的阻塞读等待**更多**跨卡进度，**加剧** amplifier。
**融合方向与解耦方向在这一对上是相反的** —— 必须先做方向 2，融合才安全。

### dispatch 域还剩什么可融合 —— 逐个都撞已文档化的硬约束

| 候选融合 | 撞到的硬约束 |
|---|---|
| `dispatch_meta` + `dispatch_wait` | orchestrator 阻塞读会等**更多**跨卡进度 ⇒ 加剧 amplifier（见上） |
| `dispatch_count_publish` + `dispatch_push` | task 边界**正是** self-row 可见性 fence（`decode_fwd.py:1740-1744` 注释：remote_store-to-self 没有 peer notify 提供跨 task 可见性 fence） |
| `dispatch_wait` 吸进 `dispatch_gather` | 违反「不要把跨卡 wait 吸进 compute kernel」—— 把 scheduler 能直接读出缺谁的 `S4` 退化成要人肉映射的 `S1` |
| `dispatch_push` / `dispatch_gather` 的 grid | 本来就由 **host** `num_tokens` 定尺（`:1810` / `:1817`）⇒ 不产生 orchestrator 阻塞读，无耦合可解 |

⇒ 叠加 DFX 那条「收益被 `dispatch_wait` 吸走（`2.18→12.56 µs`）、瓶颈是 **WAIT** 而非
small-op latency」：**dispatch 域小算子融合这条线在当前结构下已穷尽。**
而 #7（见下方「⇒ 整条线关闭的最终依据」）进一步表明**换节流也没有收益** ——
orchestrator 根本不在关键路径上。⇒ **这条线终态关闭，不留"先换节流再谈融合"的后路。**

### ★ 一个可复用的判别法：用 `SIMPLER_SCHEDULER_TIMEOUT_MS` 分开「真死锁」与「只是慢」

`S1:running-stalled` 只说明 scheduler 的看门狗到点了，**不区分**"永不完成"与"完成得比超时慢"。
抬高 `SIMPLER_SCHEDULER_TIMEOUT_MS` 是 runtime 自己文档化的判别法：**env-only、零改码**。
本例它直接推翻了一个错的双根因结论（w1 默认超时下看着像 rendezvous 死锁，抬到 45 s 后
跑了 357 个 invocation ⇒ 只是慢）。**下次看到 `S1` 且怀疑"慢 vs 死"，先花一轮跑这个。**

⚠ **另一个 runtime 提示不可信**：`error_names.h:172` 让你调 `PTO2_SCHEDULER_TIMEOUT_MS`,
**该 env 不存在**；真实变量是 **`SIMPLER_SCHEDULER_TIMEOUT_MS`**
（`runtime_timeout_config.h:25`，读于 `:165` 与 `device_runner_base.cpp:90`），且受
`scheduler_timeout_us < op_execute_timeout_us` 约束（实测 op 超时 50000 ms，故用 45000）。
同族：`PTO2_TENSOR_DATA_TIMEOUT_MS` 是编译期 constexpr、根本不可设。
**runtime 错误提示里的 env 名一律去源码 grep `getenv` 核对。**

任何选择都必须过：生产 ctx 65536 / 512 blocks、`ITERS>=1000`（**一轮长跑**，不是多轮短跑）、
至少重复 3 次，外加 A/B/A 计时门。

**权威机制报告**：`0162:/mnt/persist/chensiyu/workspace/perf-2026q3/dispatch-fusion-triage-20260821/MECHANISM.md`
（**v2**）sha256 `2e964264b6c7ae24d5681b71d5176d62e1350f1b115fde9a9c822880ee354f66`；
v1（sha `3f670f1a…`）声称环已闭合，**已被 v2 取代，不要再引用**。
预注册判别（**在拿到 FATAL 之前写的**，我的预测**被证伪**）`MECHANISM.prereg.md`
sha256 `089587cab9162dd841969a4ddeaa6f433381b0507365a3238e5e7ccf5f81f960`。

**已证伪的机制假设**（不要据此出候选）：① 我预注册的「写路径 / consumers 未释放
（fanout）」——device 给的是读路径，**该推理已撤回**；②「predicated 任务漏放 fanout
引用」——源码证伪：predicate 失败的 task 走 `dummy_ready_queue` 内联退休，
`on_task_complete` + 延迟 `on_task_release` → `release_producer` 逐个释放 fanin；
③「闭合死锁环 / 1 元凶 + 7 受害者」——被 pid 计数与提交序证伪（见上「已撤回」）；
④「小配置门结构上看不见」——R9 生产配置也有 1 次 PASS，是概率问题。
仍然成立的事实：**predicated task 永不 early-dispatch**（`pto_scheduler.h:872/938/968`）；
**predicate 本身不阻塞 orchestrator**（`pto_orchestrator.cpp:966-985` 只算地址不等待）
—— 所以阻塞的只有**动态 grid 定尺**那一类读。

**当前状态（2026-08-21 收盘）**：R6-R9 整条 dispatch 融合线维持 **NO-GO**（依据预登记规则 R1：
生产配置任何 liveness 失败即 NO-GO），生产继续用 R5。**方向 1（静态 grid）device 门三臂全挂
⇒ 也 NO-GO**。**方向 2（本地节流）与「判 ring 耗尽机制」一并关闭** —— 见下方
「⇒ 整条线关闭的最终依据」。这条 blocker 从"待修"降级为**已定案的负结论 + 两条设计规则**。

**⚠ 2026-08-21 重要更正 —— hang 是概率性的，不是"只在生产配置"**：

| 配置 | 次数 | 结果 |
|---|---|---|
| 生产 ctx 65536 / 512 blk，10+**100** iters | #1（A/B/A arm B） | **HANG** S1 @ `inv=95` |
| 生产 ctx 65536 / 512 blk，10+**100** iters | #2（faultlog） | **PASS**，p50 `26.615 ms` |
| 生产 ctx 65536 / 512 blk，10+**1000** iters | #3（faultlog，开 AICPU 日志） | **HANG** @ ~`inv=224`，**抓到 FATAL** |
| 小 ctx 4096 / 32 blk，整网单步 | #1 | PASS |
| 小 ctx 4096 / 32 blk，N=128 | #1 | PASS（256/256 逐字节等于 R5） |

⇒ 生产配置 **3 次里挂 2 次**。所以原先写的"小配置门**结构上**看不见"**不成立**；
支持的读法是"R9 有概率性 hang"。而小配置只跑过 2 次，**无法区分**「概率与配置有关」和
「概率与配置无关」——两者都兼容当前数据，别再把任何一种写成已证。

**这条更正改变了门的设计**：概率性失败意味着**单次生产配置跑通也不构成 liveness 门**。
但有一个便宜得多的办法——**提高单次曝光而不是重复整轮**：一轮 11 分钟里 decode 只占
约 3 秒（110 × 27 ms），其余全是 compile + weight load，所以 `--itl-iters` 从 100 提到
1000 只多约 25 秒墙钟却把曝光放大约 10 倍。**这一招当场奏效** —— iters=1000 那轮就挂了，
并拿到了定位所需的 FATAL 字符串。⇒ **概率性 liveness 缺陷要用"一轮长跑"抓，不要"多跑几轮"。**

**顺带一个与 DFX 推断相反的数据点**：R9 那次跑通的整网 p50 = `26.615 ms`
（min 25.984 / mean 27.161），而 R5 两次分别是 `27.478` / `27.757 ms`。**这是非配对单次
比较、不满足 A/B/A bracket 口径，不能当收益记账**，但它与下方"五层 DFX 显示收益不传播"
的推断方向相反，说明那条推断（R5-vs-R7、且两臂都是五层）对 R9 只是弱证据。
⇒ **这条线的想法可能是有价值的，值得在修掉 liveness 隐患后重估，而不是直接废弃。**

**⇒ 整条线关闭的最终依据（2026-08-21，不占卡、只解析已有 artifact）**：

比较 R5（有阻塞读）与 w1（去掉阻塞读）的 runtime STRACE 嵌套 span
（`runner_run ⊃ device_wall ⊃ graph_build ⊃ sched ⊃ orch`，每 invocation 一组，
丢弃 warmup `inv<10`，8 rank 汇池）：

| p50 span | R5 | w1 | Δ |
|---|---:|---:|---:|
| `device_wall` | 17466.93 µs | 17910.32 µs | **+443.4** |
| `graph_build` | 17457.59 | 17900.76 | +443.2 |
| `sched` | 17439.98 | 17883.85 | +443.9 |
| **`orch`** | **17279.28** | **4443.18** | **−12836.1（−74.3%）** |
| 样本 | n=8008 | n=2776 | |

**orchestrator 的 span 缩了 12.8 ms / −74%，device 总时间一点没降、反而略升。
⇒ orchestrator 与 device 并发、两种情况下都提前完成，不在关键路径上
⇒ 去掉那个阻塞读的 ITL ROI = 0（甚至微负）。**

诚实边界：非 A/B/A bracket，两轮不同 run 各取一臂，w1 那轮最终挂了。但结论不依赖这些 ——
`orch` 降 12.8 ms 而 `device_wall` 没降，量级差 29×。
工具：`analysis-bin/orch_span_stats.py`。权威记录：同目录 `FINDINGS.md`。

**⇒ 因此不再有"下一步修复"。** 本节剩下的内容（方向 2 设计、三条前置核对）只作为知识保留：
如果将来因别的原因必须动这个阻塞读，**换节流而不是删节流**是唯一安全形态。
⚠ 免费早期判据仍然有效：只要日志出现 `orch_done=1`，run-ahead 已无界，该候选注定撞 ring。

**★★ 本轮副产品（更大的线索，已转入性能主线）**：同一份 STRACE 里 R5 每 invocation
`simpler_run` p50 = 26.45 ms（与 ITL p50 26.329 ms 对得上），其中 **`bind.args` = 6.12 ms
≈ ITL 的 23%**，纯 host 侧参数绑定、与 `runner_run` 加性（w1 侧 5.87 ms，同量级）。
**比 dispatch 域任何 small-op 融合的可得收益大一到两个数量级**，且不碰 device 语义与跨卡同步。

**复现器（半机锁，~12 min，命中率约 2/3）**：
```bash
B=/mnt/persist/chensiyu/workspace/perf-2026q3/dispatch-fusion-triage-20260821
ITERS=1000 bash $B/runner/faultlog_r9.sh <R9-tree> "$B/<new-out-dir>"
```
—— 单臂、只取 `0162-cards0-7.lock`（诊断不需要整机锁）、`ASCEND_GLOBAL_LOG_LEVEL=3`
+ `ASCEND_SLOG_PRINT_TO_STDOUT=1` 抓 AICPU FATAL、`ITERS` 可调曝光。
**注意**：前两次 arm 没开 `ASCEND_GLOBAL_LOG_LEVEL`，所以 `ascend/` 目录是空的、
FATAL 全丢了 —— 这是把 6 天定位卡住的直接原因之一。

**旧的 A/B/A 复现器（整机锁，仅用于计时门）**：
`0162:同目录/runner/aba_itl.sh <R5-tree> <R9-tree> <outdir>`。证据：`aba-r5-r9/VERDICT.md`。

**解除条件**：已定案 —— 不再需要"修复后过 liveness 门"。**R6-R9 整条线判 NO-GO 为终态**，
生产继续用 R5。本条保留仅为可复用知识（两条设计规则 + `orch_done=1` 判据 + 超时判别法）。

**附带前提问题**（独立于 hang，可能让整条线本来就不值得做）：`DFX_R5_R7_COMPARISON`
（五层）显示 dispatch 域收益**不传播** —— `dispatch_count_start_to_gather_end`
`40.08→36.74`（−3.34 µs）但 `dispatch_count_start_to_gmm1_start` `78.05→78.32`
（**+0.27 µs**），`dispatch_meta` 7.12 消失而 `dispatch_wait` `2.18→12.56`（+10.38）
把收益吸走。瓶颈像是 **WAIT** 而不是 small-op 调度延迟。

---

## 🔴 ACTIVE — UPSTREAM-NOTIFY-FENCE：notify 的 cache-invalidate 排在 payload drain 之前

**症状**：`pld.tile.remote_store(...)` 紧接 `pld.system.notify(...)` 时，接收方读到的
payload 部分或全部丢失（受损区域恰好为 `0`，偶有残留碎片），且哪些 rank 受损随时序变化。

**根因**（device 已证，非推测）：pypto `src/backend/common/pto_ops_distributed.cpp` 的
`MakeNotifyCodegenPTO` 生成的前导顺序是

```c
TSTORE(peer_window, tile);                 // payload
dcci((__gm__ void*)0, ENTIRE_DATA_CACHE);  // invalidate-only，无 writeback
pipe_barrier(PIPE_MTE3); dsb(DSB_DDR); pipe_barrier(PIPE_MTE2);
pto::comm::TNOTIFY(peer_signal, ...);      // credit
```

`dcci` 抢在 payload `TSTORE` 还没从流水排空时就执行，把在途的 store 丢掉；现成的
`pipe_barrier(PIPE_MTE3)` 排在 invalidate **之后**，对此毫无作用。根因是不对称：
`MakeRemoteStoreCodegenPTO` 只发 `pto.tstore`、不发任何屏障，而 `MakePutCodegenPTO`
给 tput 夹了两个 `pipe_barrier(PIPE_ALL)`（注释写成 "WORKAROUND for PTOAS#872"）。

**最小修复 = 一条 `pipe_barrier(PIPE_ALL)`**，插在 `cacheinvalid` 之前。消融（同一插入点，
每臂相对 baseline 的 kernel diff 恰好一行，32 KiB / ring_up / warmup=0 / epochs=64）：

| `dcci` 之前插入 | exact |
|---|---|
| （无） | **False** |
| `pipe_barrier(PIPE_MTE3)` | **False** ← **纯 reorder 不够** |
| `dsb(DSB_DDR)` | **False** |
| `pipe_barrier(PIPE_MTE3) + dsb(DSB_DDR)` | **False** ← 组合也不行（codex 指出的 gap，已闭合） |
| `pipe_barrier(PIPE_ALL)` | **True**（64/64） |
| `PIPE_ALL + dsb` | **True**（64/64） |
| 同样两条放 `TNOTIFY` 之后（安慰剂） | **False** |
| payload 与 notify 之间插一次本地 GM store（纯 MTE3 流量、无屏障） | **False** |

⇒ 必须是 `PIPE_ALL` 这种**全流水等待**；MTE3 级屏障、DDR 屏障、两者组合、纯 MTE3 流量
都不行。**消融矩阵已闭合**：`PIPE_ALL` 是必需的，不能用更便宜的屏障替代，所以下面那个
`0.405 µs/call` 的 Wave2 代价也不能再压低。
**注意**：不能声称已排除「credit 超车未完成的 store」这一解释 —— `dsb` 不等待 MTE3
store 完成、`PIPE_ALL` 才等，所以「dsb 无效 / PIPE_ALL 有效」同时兼容「invalidate 破坏」
与「credit 超车」；两者修复相同，故不影响结论。

**修复在生产形状上的实测代价**（后处理注入生成后的 parent kernel，不动 codex 生成器）：
全部 3 个 notify site = **+1.250 µs/call**（half-range 0.150，`aba-20260811-100050`）；
只 Wave2 一个 site = **+0.405 µs/call**（half-range 0.005，`aba-20260811-100328`）。
约 `0.417 µs/site`、`0.060 µs/PIPE_ALL` —— 比 K2a 的 pipe-specific barrier（`0.0033 µs`）
贵约 18 倍，**不能按 K2a 外推成「免费」**。

**证据**：ring 探针（每 rank 仅 1 次 payload store + 1 次 notify，row offset 0，列槽）。
`aba-20260811-013946`：`ring_up` / `ring_down` 无 fix 都 `exact=False`，
补 `pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)` 后都 `exact=True`（**方向被排除**），
kernel diff 恰好只有这两行。payload 扫描（无 fix）`16/32/64/128 KiB` **全部** `exact=False`；
`16 KiB + fix` `64/64` epoch exact（`aba-20260811-015235`、`-014938`）。
权威报告 `0162:/mnt/persist/chensiyu/workspace/p2-k5-rhrd-20260810/CLAUDE-NOTIFY-FENCE-DEFECT.md`
sha256 `a34817832550b9c68c907a58774403802d79c1926e8aa085b658ff0aafc9f21b`。
codex 独立复核报告 `0162:同目录/CODEX-NOTIFY-FENCE-INDEPENDENT-REVIEW-20260811.md`
sha256 `37fae3aba51a555189c9da05633d88d0ab7810e28d72df9681f9fcfa11d472ac`。

**解除条件**：① 上游 pypto 在 `MakeNotifyCodegenPTO` 的 `pto.cmo.cacheinvalid` **之前**补
`pto.barrier <PIPE_ALL>` —— 即**把 put 路径已有的那条屏障对齐到 notify 路径**，不引入新概念；
② 或在产品 AR 里显式插入等效屏障并过 A/B/A + 精度门。

**安慰剂对照（已排除「两条指令只是扰动时序」）**：`--fence-late` 把**完全相同的两条指令**
放到 `TNOTIFY` **之后**（指令数/开销一致，但不再夹在 store 与 invalidate 之间）。
`aba-20260811-015951`（32 KiB）：baseline `exact=False`、placebo `exact=False`、
真 fix `exact=True`（`64/64`）；两个变体的 kernel diff 都恰好是同样那两行，只差插入位置
5 行 ⇒ **原因是顺序，不是时序**。

**生产暴露面（两 agent 对账后的口径，不要写成「生产正在损坏」也不要写成「生产是安全的」）**：
读 `A1_parent`（production 形状三波 AR）生成码确认 Wave2/Wave3 的 notify 前导与被证伪的形状
**逐字节相同**，且 Wave2 前面正是 `remote_store`。我曾据「探针失败近乎确定性」反推
「生产必被某个结构性因素保护」——**该推论已撤回**：它默认了失败率与结构无关，而这一点未验证。
codex 独立复核给出更保守的结论，我接受：**生产 Wave2 没有可证明的安全机制，只是当前调度
没有触发它；是否正在损坏未知。** 已否证四个候选保护机制：① 纯 MTE3 流量（`--drain-store`
仍 `exact=False`）；② MTE3 级屏障（消融已证不够，含 MTE3+DSB 组合）；③ store-loop 自带的
MTE3 屏障（codex 自撤：最多是间距/背压，不是正确性屏障）；④ **rank 到达 skew**——受控 skew
实验（`aba-20260811-120907`）里未加 fence 的 `P_parent_sk10000` 在 **78% epoch（50/64）**
被换成最后到达者，仍 `exact=True` `64/64`，故「生产靠 rank 时序错开而安全」被否证。
**Wave3 slack 假设已被结构性否证**：Wave3 位于 consumer read **之后**，不可能解释 Wave2 的安全。

**两 agent 一致的工程建议**：不要继续争论生产是否暴露，直接以 `0.405 µs/call`（Wave2 单点）
把这个不可证明的安全条件消掉；它同时也是删波次 / 合并波次类优化的前置条件。

**当前状态**：缺陷已定位，最小修复（一条 `pipe_barrier(PIPE_ALL)`）已在 device 上验证，
代价已量化，消融矩阵已闭合；尚未提上游、尚未进产品代码。

**硬约束（现在就生效）**：**任何把「payload store 与它自己的 credit」拉近的改动 —— 合并波次、
按 peer 融合 store+notify、单 peer 交换 —— 都会进入探针的近确定性失败区间，必须先落 fence**。
合并 Wave1+Wave2（bench 约 `5.6 µs/call`）属此类。

**注意两处已修正的表述**：① **删 Wave3 不属此类**（Wave2 notify 位置不变，不把 payload 与 credit
拉近），故它本不依赖 fence；② 但**删 Wave3 已在生产整网上被否决** —— byte-exact 却 ITL
`+1.72 ms/step`（`+5.08%`，73× floor），bench 的 `−5.92 µs/call` 符号相反，原因是
`critical_tail` 对「删同步点」结构性失明（详见 `design/performance/task-tracking.md`
2026-08-11 K9 行）。⇒ **合并 Wave1+Wave2 同属「删同步点」，即使落了 fence 也必须先过整网
ITL 才能计入收益**，不能再按 bench 的 `5.6 µs/call` 记账。

---

## 🟡 ACTIVE — DEPLOY-REPRO：镜像内 git 工作树 dirty，pin 集不足以复现验证环境（部分已解）

**症状**：按 `deployment/docker/build.sh` + 记录 pin 全新构建的镜像，整网 CI 在
`_reset_persistent_domains → orch.copy_to` 报
`device pointer 0x… is not a live allocation on worker 0 (… or an interior pointer)`；
而 0728 的 candidate 镜像同样 pin 却能跑。

**根因**：0728 三个 candidate 镜像
（`step3p5-b404a3c9-{,ci-cleanup-,ci-final-}20260728`）都带一份**未提交**的 simpler
补丁（span-aware child provenance，`worker.py +133/−19` + `orchestrator.py +2/−2`），
`build.sh` 严格按 pin clone 拿不到。已发布的 0726 镜像不含该补丁。
详见 [`postmortems/14`](postmortems/14-image-dirty-worktree-unreproducible-pins.md)。

**已解部分**：补丁已逐字节入库为 `csy0225/simpler@8459d60f`（`stepfun/develop`），并由
`csy0225/pypto@6933b1aa` 的 `runtime` submodule gitlink 固定（Dockerfile 的显式 checkout
会被 `git submodule update` 覆盖，只改 spec 无效）；`img_regress.sh` 已增加
`IMAGE_WORKTREE_CLEAN_AUDIT`（五仓逐个查 dirty）；已发布镜像
`stepfun-develop-20260729-allreduce-push` 在该基线上回归全绿。

**仍 open 的部分**：
1. 本地 `workspace/pypto/runtime` 工作树里还有一版**更新的** WIP（`worker.py +222/−47`，
   给 `exact_malloc_live` 加 `malloc_size` 上界，另带 119 行单测），**设备上从未验证**，
   未入库。需独立验证后再决定是否前进 pin，否则它会重复制造同类不可复现状态。
2. 0728 三个 candidate 的既有验证结论（C/D/G 的 BS1/2/16、N=256、Main 8-step）
   都建立在未入库补丁上，需在 `8459d60f` 基线上复核后才能作为可复现证据引用。
3. 其余四仓（pypto / pto-isa / PTOAS / pypto-lib）是否也曾以 dirty 工作树参与过
   历史镜像构建，尚未回溯核查。

**解除条件**：① 本地 WIP 增量完成设备验证并入库或明确废弃；② 0728 的 C/D/G 结论在
`8459d60f` 基线上复现；③ 历史镜像的 dirty 回溯完成。

---

## 🔴 ACTIVE — Phase 28 live serving：per-layer KV bridge + 3-way HBM / redundant weights

**当前代码边界（2026-08-06）**：
`pypto-lib stepfun/develop@c9af5790d5fe450e14fd43c88099b87539089d17`
与 `pypto stepfun/develop@8e92b46808f9f7c09b6431ad4691503f09c12ee5`
只保留 `models.step3p5.decode_fwd:whole_decode_step3p5`。C/D/G、BS1 correctness、
Attention/Vec I1 与 Wave5 TP all-reduce stability 已合入；Wave5 在 0162
完整 release-qualified。最新源码 canonical image manifest
`sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479`
已完成 BS1×64K ITL/DFX gate；旧 R2 已 supersede。本节只描述 Phase 28 live serving
缺口，不再把 BS1 的旧
`6127` 结果视为当前代码状态，也不把 0162 的结论外推到其它机器。
`models/step3p5_opt`
package、`whole_decode_opt` 和 `WholeDecodeOpt` 已删除。0726 已发布镜像内
canonical-only N=256 与清理前 canonical 镜像 token/hidden `256/256` exact、
`max_abs_diff=0`、TP spread `0.0`，所以 compatibility removal regression 已关闭。
对同一 vanilla oracle raw 为 `240/256=93.75%`，低于历史 95% raw gate；不能写成
raw PASS，也不能外推成完整 Main+MTP serving 已平替。
2026-07-27 已删除 retired 0724 unroll source、rollback selector 和自定义
Main module/name 参数；后续 blocker 定位只允许 canonical。

**当前 live serving blocker**：

> **2026-07-22 更新（device 0162, stepfun/develop `a632c42e` = hidden-only 集成）**：
> per-layer KV bridge **已接线并可跑 multi-step**——`_stage_main_hidden_only --steps 8`
> 用 per-step `block_table/slot_mapping/seq_lens` 常驻 decode。**多步精度 = NORMAL,
> pypto == vanilla vLLM 逐 token 一致**。已 device 定论：重启 vanilla W8A8 oracle
> （containerd/k8s 容器,`sudo nsenter -t <sleep-infinity-pid> bash /logs/start_8000_oracle.sh`,
> cards 0-7,port 8000）并查它自己对相同 bare-token context 的下一 token 分布——
> `[6127]→303`、`[6127,303]→1207`、`[6127,303,1207]→`**`6127`**（北京 -2.8685;
> 19384 题目是 vanilla 的 #2 -2.9935）。**vanilla step2 自己就输出 6127,与 pypto 一致**;
> harness `DEFAULT_ORACLE_TOKENS[2]=19384` 是**过时/不同 setup 生成的常量**(step2 是
> near-tie,只有它对 BOS/template setup 敏感;step0/1 margin 大所以任何 setup 都对)。
> teacher-forced 8-step = 7/8(唯一 miss 就是这个 stale-oracle 的 step2)。严格自回归
> harness 显示 2/8 纯属"一次翻转污染后续输入"的 chain artifact,非精度问题。
> **结论:多decode精度 blocker 已解决,整网 forward 数值忠实、逐 token 对齐 vanilla。**
> 历史"near-tie/未完全正常"表述作废。详见 memory `n1_multidecode_neartie_faithful_a632c42e`。

> **2026-08-04 更新（device 0162 cards 0-7, 分支 `feat/vllm-live-front-wiring`, 镜像
> `vllm-pypto:wave5-local`）——co-resident 整条 round-trip WIRING 已在设备上打通到
> decode ABI；下一墙 = live prefill**：
>
> 本 session 把 tail-only vLLM（`--load-format pypto` + `PYPTO_STEP3P5_TAIL_ONLY=1`,
> kept=3/skipped=109539）与常驻 whole-net sidecar **同卡 0-7 co-resident** 跑通到
> 真实请求进入 gate。解决/验证：
> 1. **block_table ABI 修复**：live vLLM 默认 `max-model-len`(~262144)→ block table
>    宽 2048 → flat `16×2048=32768`，而 compiled `BLOCK_TABLE_FLAT_DYN(BTF)=512`
>    (`USER_BATCH_DYN=16 × ceil(MAX_SEQ 4096/128)=32`)。**修法：vLLM launcher 固定
>    `--max-model-len 4096`** 让 block table 宽=32 → flat 512 = BTF。（`whole_decode_holder.py:536`
>    的 `table.numel()!=BTF` 校验。）
> 2. **G4 co-tenancy NO_HCCL 补丁不在任何发布镜像里**：release/Wave 镜像都为 standalone
>    canonical（cards 8-15）构建，`comm_hccl.cpp` 无 `SIMPLER_COMM_NO_HCCL` gate（image
>    `.so` grep=0），故 env flag 空转、sidecar `comm_init` 撞 `HcclCommInitRootInfo
>    failed: 7`。**修法：从 git `878f3742` 重建 patch，在 wave5 镜像内 patch
>    `src/a2a3/platform/onboard/host/comm_hccl.cpp`（5 处 anchor）+ `build_runtimes
>    --platforms a2a3` 重编，把 patched `libhost_runtime.so`（host_build_graph +
>    tensormap_and_ringbuffer）mount 进 sidecar**。重建后 gate_count=1。
>    → device 验证：sidecar 常驻、weight+KV IPC 零拷贝导入、`whole_decode_step3p5`
>    在 8 chip **co-resident 编译+prewarm+run**（`simpler_run` device_wall spans，
>    ~50ms），**0 次 HcclCommInitRootInfo failed、0 次 507018、无 card poison / 无
>    force-reset**（co-tenancy hazard 未触发）。
> 3. **下一墙 = live prefill（H4，非 wiring 缺陷）**：真实请求首个 forward 是 **prefill**
>    (`AscendAttentionState.PrefillNoCache`, `prompt_token_ids_len=1, num_computed_tokens=0`)；
>    `whole_decode_step3p5` 是 **decode-only**，gate（`vllm_monkey_patch.py`
>    `classify_decode_gate`）在 `extract_pypto_decode_plan` 上正确 fail-closed
>    (`DecodeMetadataError: unsupported attention state PrefillNoCache`)，EngineCore 退出。
>    要端到端出 token，必须先 prefill 填 KV 再 decode——需要 wire prefill program/bridge
>    或 prefill KV-fill 路径。跟踪见 phase 28 H4。
>
> 未推送：分支 `feat/vllm-live-front-wiring` 3 个 fix commit（`a9573180` load_format
> coerce、`c9af2a6a` MTP profile no-op hoist、`d35a71bf` KVPOOL MTP-optional）待用户
> 授权后 HTTP/1.1+PAT push。NO_HCCL patch 目前只在镜像内重建（`nohccl_patch.py` +
> `build_nohccl.sh` 在 0162 `live-front-wiring/patches/`），未 bake 进发布镜像。

1. **live per-layer paged-KV bridge**：standalone canonical path 已有
   resident per-layer KV 并完成多步回归；缺口是从真实 vLLM paged KV pool
   导入 per-layer BF16 slice，并按请求/step 传
   `block_table`/`slot_mapping`/`seq_lens` 与 dynamic batch metadata。
2. **HBM 容量有两个独立口径**：
   - live 3-way：vLLM W8A8 常驻权重 + exporter whole-net INT8 IPC 权重 +
     runtime working set 同时存在时，0162 报 `207001`；需消除重复权重或做等价
     in-place/shared-weight 方案；
   - standalone bs16×每请求64K：KV pool `22.541 GiB/卡`、weight pool
     `24.857 GiB/卡`，prewarm 前约 `52,013 MiB/卡`，再申请约 16 GiB pooled
     static arena 时 `207001`。该口径没有 live 重复权重，也尚无有效整网 ITL；
     ring-heap/task-window 容量 A/B 已暂停，不能与 live 3-way 根因混写。
3. **独立 live front + 同代 MTP absolute gate**：sidecar 默认 canonical Main wiring 已完成，
   但真实 online request 接管、current Main 输出进入 MTP 后的 absolute
   token/hidden oracle 尚未闭环。

**解除条件**：完成 live paged-KV/dynamic batch 接线 + device 验证；解决重复权重与
live HBM 预算；完成独立 live-front A/B 和同代 MTP absolute gate。详见
[`planning/phases/28-live-integration.md`](planning/phases/28-live-integration.md)、
[`design/vllm-pypto/02-detailed-design.md`](design/vllm-pypto/02-detailed-design.md)。

> **历史定位结论降级**：旧文档把 PUSH/TPUT/某 stuck kernel/signal bit 写成唯一硬件根因
> 的结论已撤下（详见 [`postmortems/12-integration-churn-meta.md`](postmortems/12-integration-churn-meta.md)）。

---

## 🟡 Phase 20 production backend 未接入（功能）

**症状**：BF16/W8A8 decode 与 W8A8 prefill 的结论来自 vLLM eager detail dump + PyPTO
reference/detail/final-logits replay——证明数值路径可对齐，但**不是** production backend。

**根因**：`Step3p5DecodeFwd`/prefill runner、`Step3p5Model.forward` monkey-patch、runtime
weight bundle 注入、KV/block_table/slot_mapping ABI 尚未接成在线请求路径。

**解除条件**：① `config_align.py` 校验 vLLM `hf_config` vs PyPTO constants；②
`weight_translate.py` 支持 vLLM module → PyPTO bundle；③ runner 接入 vLLM 请求路径
（decode-only 能返回 token）；④ Phase 21 在线 L1/L2/L3 gate 通过。详见
[`design/vllm-pypto/`](design/vllm-pypto/)。**Owner**：未指派。

---

## 🟡 Prefill MoE L1 overflow（TASK-29）

**症状**：`models/step3p5/prefill_moe.py` 编译时 `moe_gate_up` L1 buffer overflow
（~5MB > 限）。prefill MoE 层编译不过。**根因**：prefill 在宽 SEQ 维（如 4096 vs decode
BATCH=16）上跑，decode UB 装得下的 MoE 结构到 prefill 爆 L1。

**解除条件**：重设计 prefill_moe，加 multi-step gate_up chunking（~1-2 周）。**decode-only
perf 绕路**：合成数据预填 KV 到目标 length，跳 prefill 测 decode-only TPS/ITL（见
[`archive/completed-phases/22-perf-baseline.md`](archive/completed-phases/22-perf-baseline.md)）。
gate Phase 17。**Owner**：未指派。

---

## 🟡 head_gate 剩余：L1 A/B 暴露的整网 MoE NaN

**已解部分**：`matmul_acc N=16` codegen bug 已修，on-device head-gate 已在
`attention_full/swa` Scope 1.f 恢复（gate_r 承载 layer-independent block-diag R）——
详见 [`postmortems/09-attention-multiposition-corruption.md`](postmortems/09-attention-multiposition-corruption.md)。

**剩余**：L1 ctx=1 A/B（tid 6127 → 期望 303）曾 `logits=nan`。bisect 定位 NaN 在 42 层
INT8 W8A8 routed-MoE（单层即复现），非 attention——属 gap-5 territory，见
[`postmortems/10-gap5-attention-quant-scope.md`](postmortems/10-gap5-attention-quant-scope.md)。
**解除条件**：per-op MoE dump 定位首个 NaN 算子 → 修 → 重跑 L1。**Owner**：TASK-L 上游。

---

## 🟢 Deferred — MTP 集成进 decode

3 个 MTP 层有 kernel（`models/step3p5/mtp.py`）但没拼进 decode。speculative decoding
吞吐倍率，**不在 Phase 2 关键路径**。gate Phase 22 baseline 后 2-4 周。**Owner**：未指派。

---

## 🔴 Final e2e precision prerequisites

**gate**：最终验收"端到端精度正确且无阻塞"。预检
`python tools/step3p5/e2e_precision_readiness.py --batch 2`（host smoke 全绿）。
**剩余前置**：① 真实权重目录挂载；② vLLM/stepcast 原生 Step3p5 代码可见；③ live backend
接入（Phase 20）；④ head_gate vLLM parity 策略。**解除条件**：真实权重 + vLLM oracle 可见 +
`decode_fwd` 接线完成 + 8 rank logits shard concat 对齐 vLLM top-k。

---

## 怎么加新 blocker

1. 按严重度（🔴 Critical / 🟡 功能 / 🟢 Deferred）插入。
2. 写 症状 / 根因 / 当前状态 / 解除条件 / 链接。
3. 链回症状首次出现处（phase doc 或 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)）。
4. 在 [`STATUS.md`](STATUS.md) blocker 摘要表加一行。
5. 解决后 → 删本节 + 建 [`postmortems/`](postmortems/) 复盘。
