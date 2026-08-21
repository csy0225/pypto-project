# 专项：集成反复推翻（meta）—— 为什么"之前 ready 的"又被推翻重做

| 字段 | 值 |
|------|----|
| **子系统** | process / verification-bar（meta，跨多个事故） |
| **error signature** | 无单一 error；签名 = "上个 session 判 ready，下个 session 推翻重来" / "同一个 error 复现几十次仍在换姿势重试" |
| **首次出现** | 2026-07-13（归纳记录；案例横跨 2026-06 ~ 2026-08） |
| **状态** | 🟡 缓解（对策已落地，但 BF16→INT8-native 迁移 + DeepSeek 对齐仍在进行） |
| **相关 skill / doc** | [`06-gate-topk-deadlock.md`](06-gate-topk-deadlock.md)、[`07-whole-net-scheduler-timeout.md`](07-whole-net-scheduler-timeout.md)、[`11-8001-bridge-live-ops.md`](11-8001-bridge-live-ops.md)、`pypto-lib/docs/known-pypto-pitfalls.md`、`.claude/skills/pypto-whole-net-hang-debug/`；always-loaded 版见 memory `feedback_integration_churn_root_causes`、`pto2_failure_decode_discipline`、`cheap_gate_and_single_optimization_per_candidate` |

## 1. 背景（Background）

2026-07-13 记录。用户观察：每次集成都重复遇到同样的问题，之前判定 ready，后续 session 又推翻结果重来。本文不是某一个 error 的事故复盘，而是**归纳这一类反复 churn 的根因与对策** —— 让后续 session 不再把临时结论当定论、不再在弱验证口径上宣布 ready、不再建在临时地基上堆集成。

涉及的事故横跨 2026-06 ~ 2026-08，案例在 §2 列。always-loaded 版（memory）见 `feedback_integration_churn_root_causes`。2026-08 的 dispatch 融合 campaign（§3 根因 6/7/8）是这一类的**新形态**：不是跨 session 推翻，而是**同一 session 内在没定位的情况下反复出候选**。

## 2. 现象（Symptom）

同一件事反复 解决 → 推翻 → 重做。下表每行都是一个"之前判 ready、后续被推翻"的具体案例：

- **Blocker B**：先定"IPC-VA 冲突"（写进 doc 传了几个 session）→ device 证伪 = gate_topk mrgsort 挂死（详见 [`06-gate-topk-deadlock.md`](06-gate-topk-deadlock.md)）。
- **G5b 根因**：先"SWA multi-entry kernel bug" → `--golden-fill-batch` 证伪 = seq_len=0 pad 行污染。
- **gap-5**：partial-tile → 证伪；scale-tail-zero → 证伪；quant-scope → 证伪（3 次）。
- **KV bridge**："纯 reshape" → 更正"非纯，3 障碍"。
- **HBM**："24G+47G=OOM" → 更正"64GB/卡，误判"。
- **head-gate**：bypass ↔ worker gate_r ↔ on-device 来回（详见 [`11-8001-bridge-live-ops.md`](11-8001-bridge-live-ops.md) §head-gate）。
- **dispatch 域小算子融合**（2026-08-15 ~ 08-21，新增案例）：6 天、`357` 个 run 目录、**同一个 `code -8` 复现 46 次**（08-17 23:44 ~ 08-20 22:02），无落地。这一轮不是"上个 session 判 ready 被推翻"，而是**同一个 session 内在错误的层反复重试**——形态不同、根因同族（§3 根因 6/7/8）。**最刺眼的一点**：最后那个候选（R9）相对生产基线 R5，
  `decode_fwd.py` 的**全部差异只有一行**（`dispatch_gather` 的 `deps` 多一个 `dispatch_push_tid`）。
  一行的改动烧了 6 天。**最终结局（2026-08-21 收盘）**：这条线以**负结论**关闭 ——
  接手后又花一天走了"结构修复候选 → device 门三臂全挂 → 发现那个阻塞读是承重节流"，
  最后才用一个不占卡的日志解析发现 **orchestrator 从不在关键路径上、整类改动 ROI 上界为 0**
  （§3 根因 11）。**那个否决门本来在第一天就能做。**

## 3. 根因（Root Cause）

设计 + 流程，约各一半。

### 根因 1（流程，最主要）：「ready」验证口径太弱、没强制阶梯

真出口 = **live-token-exact-device**；途中有 compile-OK / offline / synthetic / single-config / 单卡 多个更弱的 bar。在**较弱 bar 宣布 ready** → 下个 session 在更强 bar 上推翻。**"声明 bar" 与 "真 bar" 的每个 gap 都是未来推翻点。**

本 session 亲历：offline golden 按 kernel 布局注入 KV pass → live 立即推翻。

### 根因 2（流程）：根因靠"看着对的假设"而非"决定性隔离实验"

假设被写进 doc / memory **当事实传递** → 后人基于错假设做 → 直到有人补一个**能证伪**它的控制实验才翻案。

对策：**声明 root cause 前，先设计能证伪它的实验**（fill-batch / dispatch-cut bisect / golden 对拍）。

### 根因 3（设计）：建在"明知临时"的地基上

BF16-dequant 是 bring-up 捷径、真目标 INT8-native → BF16 上所有**精度结论都是暂定、注定被 INT8-native 推翻**（今日 L17 attn 0.25 即此）。

### 根因 4（设计）：每个 step3p5-vs-DeepSeek divergence 都是潜在重复 bug

seq_len=0 pad = 静态 BATCH=16 padding 背离 DeepSeek 动态 T；DeepSeek 无 pad 行故永不踩。每个"和 DeepSeek 不一样"都埋雷。

### 根因 5（环境）：底座漂移

5 仓 + driver/CANN + 2 分支（stepfun/develop vs n1-live）+ 2 机器（0234 down → 0162）。升级栈曾 drop Phase-16 SDMA-OFF patch → 重踩 507899；rebase 引入 SplitIncoreOrch 回归。一组合上 ready 不迁移到另一组合 → 每 session re-derive。

> 下面 3 条由 2026-08 dispatch 融合 campaign 归纳（6 天 / 357 run / 0 落地）。与根因 1、2 同族但**是新的失效模式**，不是它们的重述。

### 根因 6（流程）：`stuck_task_id` 从没被兑现成"哪个 kernel 在等什么"就出下一个候选

`RuntimeError: run failed with code -8` 的 `rc` 就是 `-(PTO2 error code)`，而同一份 host log 里 runtime **已经给出**分类：

```
PTO2 runtime failed: orch_error_code=8 sched_error_code=100 runtime_status=-8
PTO2 scheduler timeout sub_class=S1:running-stalled (detail=1) \
  completed=339/1702 running=1 ready=0 waiting=29 stuck_task_id=12884902272 stuck_core=28
```

`sub_class` 把故障**定位**了，且 `S1` / `S4` 互斥：

| sub_class | 语义 | 首先该看什么 |
|---|---|---|
| `S1:running-stalled` | 有 task 已上核（`running=1`）但永不完成 | **该 kernel 内部的自旋谓词**，以及**谁应该满足它** |
| `S4:dependency-deadlock` | 只剩 WAIT（`running=0`），fanin 永不满足 | 依赖图接线 / credit 计数 |

⚠ **注意这里有个容易过度解读的地方**（本 session 初稿就犯了）：`S1` 说的是**停在哪**，不是**该改哪**。一个 kernel 如果体内自旋等对端 notify / signal cell，那么**接线错**也会表现成 `S1` —— 停点在 kernel 体内，修点却在该发 notify 的一侧。所以不能从 `S1` 直接推出"改依赖图一定是错方向"。[`07-whole-net-scheduler-timeout.md`](07-whole-net-scheduler-timeout.md) 正是同签名、根因在 comm window 复用的先例，它让"往 comm / 依赖方向查"成为**合理**的第一直觉。

**真正的流程失效不是"改错层"，而是"从没把 `stuck_task_id` 兑现成具体 kernel + 具体自旋谓词就出下一个候选"**：46 次 `-8`（08-17 23:44 ~ 08-20 22:02）之后，8 个候选变体（per-expert / forceipc / per-block-ready / dual-mix / fence-phase / local-self-phase / retire-gated / split-comm-boundary）里没有任何一个 evidence 目录记录过 ring3/local384 是哪个 kernel、它在等哪个条件、该条件本应由谁置位。**没有这个映射，任何候选都是猜。**

由此推出一条设计约束：**不要把跨卡 wait 吸进 compute kernel** —— 那会把一个 scheduler 可诊断的 `S4`（能直接读出 fanin 缺谁）退化成只能靠人肉映射的 `S1`。

#### 决定性的一步竟然是一个环境变量（2026-08-21）

把 `stuck_task_id` 兑现成 kernel + 谓词只解开了**受害者**那一半。真正闭合机制的那条信息，
runtime **一直在打印**，只是没落盘：

```
FATAL(code=8): Timeout (750000000 cycles): producer (ring=3, local=933) not completed
```

它由 AICPU 侧 `orch.report_fatal` → `unified_log_error` 发出，而
`unified_log_error` 被 `is_log_enable_error()` 门住，该标志来自
`CheckLogLevel(AICPU, DLOG_ERROR)` —— **即必须设 `ASCEND_GLOBAL_LOG_LEVEL=3`**。
之前所有 arm 都只设了 `ASCEND_PROCESS_LOG_PATH`，于是目录建了、文件是空的、这条字符串全丢。

`TENSOR_WAIT_TIMEOUT` 全 runtime 只有 **2 个** raise 点（都在
`pto_runtime2.cpp::wait_for_tensor_ready`），二者含义相反、修法也相反：

| 分支 | 调用者 | 消息 | 含义 |
|---|---|---|---|
| 读 | `get_tensor_data` | `producer (ring, local) not completed` | orch 在等某 producer 完成（本例就是这一支） |
| 写 | `set_tensor_data` | `consumers of producer (ring, local) not done` | `fanout_refcount < fanout_count`，某 consumer 没释放 |

⇒ **判据加一条：整网 hang 的 arm 一律带 `ASCEND_GLOBAL_LOG_LEVEL=3`。**
没有它，`orch_error_code=8` 只是个分类，不含地址；有它，直接点名 producer 的 ring/local。
**成本为零，价值是 6 天。**

#### 另一半：orch 与 scheduler 是两个独立的停摆主体

同一份日志里 8 个 rank 的 `sched_error_code` 分布是 **7×`100` + 1×`0`**。
`sched_error_code=0` 那个 rank **不产出 `sub_class=` 行** —— 它的 scheduler 没停，是
**orchestrator** 卡住。所以：

- 只 grep `sub_class=` 会**只看到受害者**（本例 7 个 rank 全停在同一个对称 barrier）。
- 少数派签名 = "**没有** `sub_class` 行 + `sched_error_code=0` + orch `TENSOR_WAIT_TIMEOUT`"。

**可推广判据（2026-08-21 修正）：对称 barrier 里，少数派报告说明「哪个子系统停了」
（orchestrator 还是 L2 scheduler），去查它 —— 但它 <ins>不</ins>说明「哪个 rank 有责任」。**

⚠ 本文档此处原写"**例外那个 rank 才是元凶**"，**已撤回**：后续按 AICPU pid 计数
（`grep -ho "AICPU([0-9]*," log | sort | uniq -c`）发现 **8 个 distinct pid 全部**阻塞在
**同一个** producer，不存在"1 个元凶 + 7 个受害者"。见
[`16-dispatch-fusion-orch-decouple.md`](16-dispatch-fusion-orch-decouple.md) §3/§5。

### 根因 7（流程）：便宜门"看不见"目标 bug → 反复跑绿 = 假绿

R9 反复跑绿的 device gate（`device-gates/bs1`、`bs2`、`route-bs1`）全是五层 harness（`five_layer_*`）。整网是 **1702 task** 的图，五层 harness **装不下**这个 bug —— 于是"门绿了"和"bug 还在"同时成立。

**判据**：这一族 S1 hang 有两个子类，**门必须两个都覆盖**——(a) 首次 invocation 就挂（观测到 `completed=339/1702`，`--steps 1` 即复现，分钟级）；(b) 跑到中途才挂（R7 观测到 `inv=105` / 104 条 step record，即确定性停在 step 104，只有 N=128 看得见）。所以正确的门是便宜→贵**逐级加压**，每级都要能看见它负责的那一类：

```
compile gate (不占卡)
  -> 整网单步 liveness (半机锁, 分钟级)   <-- 抓 (a) 类 S1/S4
  -> hidden sha256 byte-exact
  -> N=128 多步精度 (半机锁)             <-- 抓 (b) 类；精度门不需要整机锁
  -> A/B/A ITL (整机锁, 串行)            <-- 只有计时门需要整机锁
```

配套的经济学错误：N=128 是**精度**门，却被挂在**整机锁**下串行（R7 的 run.sh 同时抢 FULL + cards8-15），白白把一个 13 分钟的半机任务变成需要等两半都空的整机任务。**锁的粒度要跟门的性质对齐：只有计时门需要整机锁。**

#### 加强版（2026-08-21 亲历，比上面写的更严重）

"整网"**不等于**"生产配置"。同一个 `@pl.program` 在 `MAX_SEQ=4096 / --num-blocks 32`
和 `MAX_SEQ=ROPE_SEQ=65536 / --num-blocks 512` 下是**两张不同的 task 图**（task 总数
`1702` vs `1744`），所以小配置的整网门**照样会假绿**：

| 门 | 配置 | R9 结果 |
|---|---|---|
| 整网单步 liveness | ctx 4096 / 32 blocks | **PASS**（6127→303，tp_spread 0） |
| N=128 replacement-equivalence | ctx 4096 / 32 blocks | **PASS**（256/256 tensor 与 R5 逐字节相等） |
| 整网 ITL A/B/A arm B | **ctx 65536 / 512 blocks（生产）** | **HANG** `S1` `completed=546/1744` ring3/local627 |

我先用前两个门得出"R9 语义中性且无 hang"，被第三个门直接推翻。

⚠ **诚实标注证据强度 —— 已由后续实验解决，见下**。初稿写的是：生产配置 R9 只跑了 1 次
（挂），小配置各跑了 1 次（过），所以"小配置门**结构上**看不见它"是**推断**，竞争解释是
"R9 有一定概率挂"。当日就跑了区分实验。

#### 区分实验的结果（2026-08-21 当日）：**是概率性的**

| 配置 | 次数 | 结果 |
|---|---|---|
| 生产 ctx 65536 / 512 blk，10+100 iters | #1 | **HANG** S1 @ `inv=95` |
| 生产 ctx 65536 / 512 blk，10+100 iters | #2 | **PASS**，p50 `26.615 ms` |
| 生产 ctx 65536 / 512 blk，10+**1000** iters | #3 | **HANG** @ ~`inv=224`，抓到 FATAL |
| 小 ctx 4096 / 32 blk，单步 | #1 | PASS |
| 小 ctx 4096 / 32 blk，N=128 | #1 | PASS |

⇒ 生产配置 **3 次挂 2 次**。所以「小配置**结构上**看不见」**被否证**；支持的读法是
「R9 有概率性 hang」。而小配置只跑过 2 次，**无法区分**「概率与配置有关」与「概率与配置无关」，
两者都兼容当前数据 —— 别把任何一种写成已证。

**这个结果改变了门的设计**（比原判据更强的一条）：概率性失败意味着
**"生产配置跑通一次"也不构成 liveness 门**。原判据「每个门都要在生产 ctx + 生产 block 数
再跑一遍」是必要但不充分的。

**便宜的补法 —— 提高单次曝光，而不是重复整轮**：一轮 11 分钟里 decode 只占约 3 秒
（110 × 27 ms），其余全是 compile + weight load。所以 `--itl-iters` 100 → 1000 只多约
25 秒墙钟，却把曝光放大约 10 倍。**概率性 liveness 缺陷要用"长跑一轮"抓，不要用"多跑几轮"抓。**

判据修正为：**每个 liveness / 精度门都必须在生产 ctx 与生产 KV block 数下跑，并且
单轮曝光要拉长（`--itl-iters` 量级 1000 而非 100）**；小配置只能当更快的预筛。
前半句在"结构"和"概率"两种解释下都成立；后半句是概率解释被证实后新增的。

### 根因 8（流程）：特征矩阵算错了基线 —— "捆了两个优化"其实是基线选错的假象

> ⚠ **本节已于 2026-08-21 大幅更正。初稿断言"从来没有一个 dispatch-only 候选"，
> 该断言对 `develop` 成立、对 **R5（真正的生产基线）不成立**，因此对操作性问题是错的。**

特征矩阵（`grep -c routed_nz`）：

| 基线 / 候选 | `routed_nz_*` 引用 |
|---|---:|
| `develop` | **0** |
| **R5（生产基线）** | **19** |
| R9 | **19** |

⇒ NZ routed GMM1+SwiGLU+quant 融合**在 R5 里已经落地了**。所以：

- **相对 `develop`**：R6-R9 确实各带两个优化（dispatch 域重排 + NZ GMM 融合），无法归因。
- **相对 R5**：R9 的 `decode_fwd.py` 与 R5 的**全部差异只有一行** ——
  `dispatch_gather` 的 `deps` 多了一个 `dispatch_push_tid`：

  ```diff
   with pl.spmd(
       scan_blocks, name_hint="dispatch_gather",
  -    deps=[wait_tid, meta_collect_tid],
  +    deps=[dispatch_push_tid, wait_tid, meta_collect_tid],
       predicate=(local_route_count[0] > 0),
       allow_early_resolve=True,
   ) as dispatch_gather_tid:
  ```

  这是一个**教科书级隔离候选**：单行、dispatch-only、归因毫无歧义。

**因此真正的教训不是"捆了两个优化"，而是「特征矩阵必须相对*你实际要对比/发布的那个基线*来算」。**
R5 是生产基线（`planning/handoff.md`），所以操作基线是 R5，而相对 R5 归因是干净的。
初稿把基线取成 `develop` → 得出"无法归因" → 反而**掩盖**了真正的失效点。

**这条更正把责任重新压回根因 6 和 7，而不是减轻它们**：既然 R9-vs-R5 本来就是一行、
可完美归因，那么 6 天没定位下来就**只能**由「`stuck_task_id` 从没被兑现成 kernel + 谓词」
（根因 6）和「门看不见生产配置」（根因 7）来解释。一行的改动跑了 6 天 —— 这不是归因难度问题。

`code -8` 最早出现的目录名 `dfx-dispatch-gather-gmm-fused-20260817-final` 仍是有用线索
（`dispatch_gather` 与 GMM 融合的第一个 run），但它**不能**再被当作"候选捆了两个优化"的证据。

> 下面 3 条由 2026-08-21 的**修复候选**（不是原 campaign 的候选）归纳。它们解释的是
> **"定位对了之后仍然做废一轮"**，是根因 1/2 之外的新失效模式。

### 根因 9（流程/设计）：候选的**立项理由**没被审过 —— 把"看似多余的同步"当纯缺陷删掉

定位闭合之后，我出了一个"消掉 amplifier"的候选（`combine_scatter` 的动态 grid 改静态常量），
立项理由写的是：**"它是唯一同时消掉 amplifier 又不删任何同步点的改法"**。
device 门三臂全挂，`orch_error_code=2 HEAP_RING_DEADLOCK`。

**错在把"同步点"只理解成 `pld.system.wait` / `notify`。** 那个 orchestration 级阻塞标量读
**本身就是一个同步点** —— 它是对 orchestrator run-ahead 的背压阀（`RUNTIME_LOGIC.md`
§4.4「ring 耗尽时 orchestrator 阻塞」，§4.5 ring 太小会因 scope 引用死锁）。
每层一次的阻塞读把 run-ahead 限在一层内；改静态 grid 把它删了 ⇒ `orch_done=1`
（一次提交完整张 1744-task 图）⇒ run-ahead 无界 ⇒ ring 饱和 ⇒ §4.5 死锁。
**加容量修不了**：ring 已经是 4 GiB heap / 131072 slots，加容量只把失败从 `inv=10` 推到 `inv=357`。

**可推广判据（出候选前问自己）**：我要删的这个东西，**除了我认定的那个坏作用之外，
还在承担什么？** 尤其是任何形如"阻塞 / 等待 / 串行化"的构造 —— 它多半同时是某种
**流控**。删之前先找它的背压语义，找不到再删。
**免费早期判据**：日志里一旦出现 `orch_done=1`，说明 run-ahead 已无界，该候选注定撞 ring。

**对应的设计规则**：orchestration 级阻塞标量读若其 producer 含跨卡 wait，是隐患，
**但不能直接删** —— 必须换成一个**只依赖本地 device 进度**的节流，否则是把一个稀有的
概率性跨卡死锁换成确定性 ring 死锁。

### 根因 10（流程）：看门狗超时会把"慢"伪装成"死"，于是编出多余的第二个根因

同一个候选的两个 worker 数各自以**不同签名**挂：w8 报 `HEAP_RING_DEADLOCK`，
w1 报 `sched_error_code=100` `S1:running-stalled` @ `combine_wait`。我据此写下
**"两者不是同一根因"**，并准备分头查两条线。

**这是错的。** `S1:running-stalled` 只说明 **scheduler 的看门狗到点了**，它
**不区分**「永不完成」与「完成得比超时慢」。抬高 `SIMPLER_SCHEDULER_TIMEOUT_MS`
（env-only、零改码、runtime 自己文档化的判别法）之后，w1 跑了 **357 个 invocation**，
最后挂的是**和 w8 完全相同**的 `HEAP_RING_DEADLOCK` —— **它从来不是死锁，只是慢。**
两臂一个机制，第二条线根本不存在。

**可推广判据**：
1. **看到 `S1` 且要下"死锁"结论前，先花一轮抬高 `SIMPLER_SCHEDULER_TIMEOUT_MS`。**
2. **"两个候选以不同方式挂"这种话，在超时被抬高之前都不可信** —— 看门狗会把最慢的那条
   路径伪装成故障点，而不同配置的"最慢路径"往往不同。
3. ⚠ **runtime 错误提示里的 env 名不可信**：`error_names.h:172` 让你调
   `PTO2_SCHEDULER_TIMEOUT_MS`，**该 env 不存在**；真名是
   `SIMPLER_SCHEDULER_TIMEOUT_MS`（`runtime_timeout_config.h:25`），且受
   `scheduler_timeout_us < op_execute_timeout_us` 约束（实测 op 超时 50000 ms，故用 45000）。
   同族 `PTO2_TENSOR_DATA_TIMEOUT_MS` 是编译期 constexpr、根本不可设。
   **一律去源码 grep `getenv` 核对。**

### 根因 11（流程）：为了拿一个数去写新候选，而那个数已经躺在失败 run 的日志里

整条 dispatch-orch-decouple 线的立项前提是"orchestrator 阻塞值钱"。这个前提
**从头到尾没有一个数据点**。我当时的判断是：「三个失败臂都没跑到测量段 ⇒ 要拿 ROI
必须先写一个**能活下来**的候选。」于是"下一步"就变成了再写一个候选、再烧一轮 device 门。

**这是错的。** 失败臂在挂掉之前跑了几百个 invocation（`inv=357` / `inv=224`），
而 runtime 的 STRACE 每个 invocation 都发一组嵌套 span：

```text
simpler_run ⊃ { bind, runner_run }
runner_run  ⊃ device_wall ⊃ graph_build ⊃ sched ⊃ orch
```

⇒ **失败的 run 也能给出可靠的稳态计时。** 去掉 warmup 后样本 n=2776 / n=8008，
一个 40 行 stdlib 脚本、不占卡、不加锁、不改码，当场给出答案：
**`orch` p50 `17279 → 4443 µs`（−74%）而 `device_wall` `17467 → 17910 µs`（反升 +443）
⇒ orchestrator 从不在关键路径上 ⇒ 整条线的 ROI 上界 = 0。**

**可推广判据**：

1. **在决定"必须再跑一轮 / 再写一个候选"之前，先穷举现有 artifact 里已经有什么。**
   失败的日志不只有失败签名。
2. **改任何 orchestration 级构造之前，先比 `orch` 与 `device_wall` 的 p50。**
   若 `orch <= device_wall`，orchestrator 不是瓶颈，整类改动没有收益可言 ——
   这比根因 9 的"审立项前提"更硬：它是一个**可量化的**否决门。
3. **顺手读全部 span 层级。** 本例副产品：`bind.args = 6.12 ms ≈ ITL 的 23%`
   （纯 host 侧参数绑定）—— **比追了整天的那个项大一到两个数量级。**

⇒ 与根因 9 合起来是同一个病的两面：**根因 9 = 候选的立项前提没被审；
根因 11 = 那个前提其实很便宜就能被量化，只是没人去量。**

## 4. 如何解决（Fix）

可落地的对策（已开始执行）：

- doc / memory **区分「假设」vs「隔离证明的事实」**；错的**大声 CORRECTED / SUPERSEDED** 撤回（本仓 postmortems 模板里 §5「走过的弯路」就是这条的落点）。
- **声明 root cause 前先跑证伪实验**。
- **"ready" 只认 live-token-exact-device**；compile / offline / synthetic 一律标 `provisional`。
- **别在 BF16-dequant 上堆 live 集成** → 直接 INT8-native（消临时地基）。
- **能对齐 DeepSeek 就对齐**；必须不一样的写清"为什么 + 验证口径"。
- **pin 单一可复现底座**（分支 / 机器 / 版本组合）。
- **拿到 `run failed with code -N` 先解码再动手**：`grep -E "orch_error_code=|sub_class=|stuck_task_id="`，把 `sub_class` 和 ring/local（`stuck_task_id = (ring<<32)|local`）写进结论，再决定改 kernel 还是改依赖图。
- **整网 hang 的 arm 一律带 `ASCEND_GLOBAL_LOG_LEVEL=3`**（只设 `ASCEND_PROCESS_LOG_PATH` 不够 —— AICPU 的 `unified_log_error` 被 `CheckLogLevel(AICPU, DLOG_ERROR)` 门住，不开就只建目录不写文件）。开了才拿得到 `FATAL(code=8): … producer (ring=R, local=L) not completed`，直接点名 producer。**成本为零。**
- **`(ring, local)` → kernel 名**：跑一次带 `PYPTO_DISTRIBUTED_DEP_GEN=1` 的**生产配置** run 拿 `deps.json`，再用 `deps.json::tasks[].kernel_ids` → `func_id` → `kernel_config.py::KERNELS` 反查。工具：`0162:…/dispatch-fusion-triage-20260821/depgen-bin/lookup_task.py`（同时打印邻域 + 前驱 + 后继）。
- **概率性 liveness 缺陷用"一轮长跑"抓，不要"多跑几轮"**：一轮里 decode 只占几秒，compile + weight load 占十几分钟，所以把 `--itl-iters` 从 100 提到 1000 只多约 25 秒却让曝光 ×10。
- **便宜门必须能看见目标 bug 才算门**：五层 harness 不能替整网门。可复用 runner 在 0162 `perf-2026q3/dispatch-fusion-triage-20260821/runner/`（`wholenet_liveness.sh` 自动输出 `VERDICT` ∈ OK / HANG_S1 / DEP_DEADLOCK_S4 / TENSOR_WAIT_TIMEOUT / SCHED_TIMEOUT / FAIL_OTHER，并抓 `STUCK`；`n128_replacement_gate.sh` 只取半机锁）。
- **特征矩阵要相对"你实际要发布的那个基线"算**：`grep -c <新特征> candidate/ <生产基线>/ develop/` 三列一起数。只对 `develop` 算会把"其实是一行的隔离候选"误判成"捆了两个优化"（根因 8 初稿就这么错的）。真捆了两个才拆。
- **先 `diff` 再推理**：出结论前跑一次 `diff <生产基线>/…/decode_fwd.py <候选>/…/decode_fwd.py`。R9-vs-R5 一行的事实，一条 `diff` 就能拿到，却在 6 天里没人拿。
- **锁粒度跟门性质对齐**：compile / liveness / 精度门 → 半机锁；只有 A/B/A 计时门取整机锁。

## 5. 走过的弯路（Detours / What We Got Wrong）

本节是本文的实质 —— 列每个被推翻的"ready"声明，及**证伪它的实验**。

- ❌ **Blocker B = "IPC-VA 冲突"**（写进 doc 传了几个 session）→ 证伪：device 上 MoE comm window `0x12c041...` 低于 IPC pool `0x12c1c0...`，无 overlap；IPC map 48-key aligned / in-pool。真因 = gate_topk mrgsort format2-on-unsorted 挂死（[`06-gate-topk-deadlock.md`](06-gate-topk-deadlock.md)）。证伪实验 = device-level VA 范围比对 + dispatch-cut bisect。
- ❌ **G5b = "SWA multi-entry kernel bug"** → 证伪：`--golden-fill-batch`（16 行全 active）L0-L4 全 pass 1.0（full / swa / MoE）；L1-alone golden input 也 1.0。真因 = seq_len=0 pad 行产生 NaN 污染 active 行。证伪实验 = fill-batch 控制组。
- ❌ **gap-5 = "partial-tile 处理 bug"** → 证伪：控制组 int8-copy PASS。真因 = `cast→int8→cube` 误编译（`infer_tile_memory_space_pass.cpp:55-56` INT8 cube A-operand fractal=32 layout 未推导）。
- ❌ **gap-5 = "scale-tail-zero"** → 证伪：tail 不零也复现。
- ❌ **gap-5 = "quant-scope"** → 证伪：scope 调整无效。gap-5 共 3 次证伪。
- ❌ **KV bridge = "纯 reshape"** → 更正：3 障碍（MAX_SEQ sizing 667GB blowup / k_cache dim 是 KV_CACHE_ROWS_DYN baked 非 MAX_SEQ / 整池 feed 15GB dummy OOM）+ per-layer feed 路径。
- ❌ **HBM = "24G+47G=OOM"** → 更正：64GB/卡，TP=8 sharded 后 vLLM 3GB + pypto 6GB + KV ≈10GB/card ≪ 64GB。误判来源 = 把聚合非分片当分片。
- ❌ **head-gate 反复方向**：bypass ↔ worker gate_r ↔ on-device 来回。中段误判"KV 来源（self-KV vs 真-KV-IPC）是 live 乱码根因" → 证伪：真-KV-IPC 打通后 self-KV 和 真-KV-IPC **都乱码**。真因 = head-gate `matmul_acc` 小 N=16 丢 K 维累加（[`11-8001-bridge-live-ops.md`](11-8001-bridge-live-ops.md) §head-gate 终局）。
- ❌ **L2 attention residual 路径**（M4 determinism）：先怀疑 hidden-state pingpong / fence-gap / attention-residual → 证伪：device-disproven。真因 = dense L2 REUSED L1 comm/signal windows → premature Ge(1) barrier via leftover AtomicAdd signal → racy attention output。修 = distinct `l2_*` windows。

**2026-08 dispatch 融合 campaign（新增）**：

- ❌ **"`code -8` 是依赖图 / 调度层问题"**（隐含假设，从未明说，因此从未被检验）→ 证伪：每份日志自带 `sub_class=S1:running-stalled`，语义是 task 已上核永不完成 = kernel 体内自旋。8 轮改依赖图 / 调度的候选全部无效正是这条的代价。证伪实验 = **读日志**（`grep orch_error_code=|sub_class=|stuck_task_id=`）——本例的教训是"最便宜的证伪实验是把已有输出读完"。
- ❌ **"R9 的三个 device gate 全绿 ⇒ R9 没引入 hang"** → 证伪：那三个门都是 `five_layer_*` harness，装不下整网 1702-task 图；换整网单步门后 R9 才第一次被真正测到（结果 PASS，见下）。
- ❌ **"整网单步就足以证明无 hang"**（本 session 我自己先犯的）→ 自我证伪：R7 的 hang 记录到 `inv=105` / 104 条 step record，即**确定性地卡在 step 104**，单步只是早期过滤器、不是无 hang 的证明。**更正**：单步门用来快速抓 first-invocation 类 S1；step-104 类必须 N=128 才能看见。
- ❌ **"step-104 hang 可能就是那个已知的 ~1/3 概率残留随机 stall"**（本 session 我自己先犯的）→ 自我证伪：(a) [`reference/cache-line-signal-isolation.md`](../reference/cache-line-signal-isolation.md) 记 512B signal isolation 后 20/20 PASS；(b) 同 harness / 同配置下 R5 基线跑满 128 步；(c) R9 的 N=128 也跑满 128 步且 256/256 tensor 与 R5 逐字节相等。**更正**：step-104 归因于 R7 的改动，不是底座随机性。**残留缺口（诚实标注）**：没有重跑 R7 第二次，所以"R7 的 step-104 是确定性的"这一点是从 46 次 `-8` 复现 + R5/R9 各一次干净跑推出的**强推断**，不是同一候选的重复实验证明。要彻底钉死需要 R7 复跑 2 次。
- ❌ **"N=128 太贵所以只能最后跑"** → 证伪：R7 那次 N=128 实际只用 ~7 分钟；贵的不是 run 而是**它被挂在整机锁下串行**。更正后本 session 的 N=128 replacement-equivalence 门只取半机锁、13 分钟出结论。
- ❌ **"R9 语义中性且无 hang"**（本 session 我自己的结论，同日被自己的下一个门推翻）→ 证伪：ctx 4096/32-block 的整网单步门与 N=128 门都 PASS，但 **ctx 65536/512-block 的生产配置 arm 直接 S1 hang**（`completed=546/1744` ring3/local627，多 rank core 24/26/28）。**更正**：R9 = NO-GO，理由是 liveness 而不是 ROI。教训见 §3 根因 7「加强版」——小配置整网门同样会假绿。
- ❌ **"R6-R9 四个候选都捆了两个优化，从来没有 dispatch-only 候选"**（本 session 我自己写进 §3 根因 8 的）→ 自我证伪：`grep -c routed_nz` 三列 = `develop 0 / **R5 19** / R9 19`，NZ GMM 融合在 R5 里就有了；`diff R5 R9` 的 `decode_fwd.py` **只差一行**（`dispatch_gather` 的 `deps` 多一个 `dispatch_push_tid`）。**更正**：相对生产基线 R5，R9 是教科书级隔离候选，归因毫无歧义；见改写后的 §3 根因 8。**这条更正加重而非减轻根因 6/7 的责任** —— 一行的改动烧了 6 天，只能由"没兑现 `stuck_task_id`"+"门看不见生产配置"解释。
- ❌ **"7 个 rank 都停在 `combine_wait` ⇒ `combine_wait` 是根因"** → 证伪：`combine_wait` 是**对称 rendezvous**（先 notify 全部 peer、再 wait 全部 peer），所以停在它里面的 rank 全是**受害者**。同一份日志的 8 个 rank，`sched_error_code` 分布是 **7×100 + 1×0** —— 那个 `sched_error_code=0`（无 scheduler stall，只有 orch `TENSOR_WAIT_TIMEOUT`，且**不产出 `sub_class=` 行**）的 rank 说明的是**它的 orchestrator 停了、scheduler 没停**。⚠ **半截更正（2026-08-21）**：这条当时进一步推成"那个 rank 才是元凶"，**已撤回** —— AICPU pid 计数显示 8 个 distinct pid 全部阻塞在同一 producer。**可推广判据只到这一层：少数派报告说明「哪个子系统停了」，不说明「哪个 rank 有责任」。** 见 [`16-dispatch-fusion-orch-decouple.md`](16-dispatch-fusion-orch-decouple.md) §5。
- ❌ **"predicated 任务漏放 fanout 引用，把 producer 永久钉在非 CONSUMED"**（本 session 的机制假设，动手改之前先证伪）→ 源码证伪：predicate 失败的 task 走 `dummy_ready_queue` 内联退休，路径是 `on_task_complete` + 延迟 `on_task_release` → `for_each_fanin_slot_state` → `release_producer` 逐个释放 fanin（`scheduler_dispatch.cpp` dummy drain + `pto_scheduler.h` `on_task_release`）。所以这个机制不成立，**不要**据此出候选。仍成立的事实：predicated task **永不 early-dispatch**（`pto_scheduler.h:872/938/968`），只在 ready 点解析。
- ❌ **我预注册的"是写路径 / consumers 未释放（fanout）"** → device 证伪：FATAL 是**读路径** `producer (ring=3, local=933) not completed`。我把这个预测**在拿到数据之前**写进了 `0162:…/dispatch-fusion-triage-20260821/MECHANISM.prereg.md`（sha256 `089587ca…`），正是为了让它可被记分；结果是**错的**，相关 fanout 推理已撤回。**这条本身就是方法论的正面样本**：先写预测、再取数据，错了就是错了，不用事后合理化。真实机制见改写后的 [`blockers.md`](../blockers.md) ORCH-SCALAR-READ-VS-CROSSRANK-WAIT。
- ❌ **"R9 的生产配置 hang 是确定性的"**（前一版 blocker 的隐含读法）→ 证伪：生产配置 **3 次挂 2 次**（100 iters 挂 / 100 iters 过 / 1000 iters 挂）。**更正**：概率性。**并且这改变了门**——单次生产配置跑通不构成 liveness 门；便宜补法是**拉长单轮曝光**（`ITERS=1000`，只多约 25 秒墙钟、曝光 ×10），不是重复整轮。这一招当场奏效。

## 6. 如何避免（Prevention）

- **声明 root cause 前先跑证伪实验**（fill-batch / dispatch-cut bisect / golden 对拍 / VA 范围比对）。假设写进 doc 时标"假设"，事实标"隔离证明"。
- **"ready" 只认 live-token-exact-device**。compile / offline / synthetic / single-config / 单卡一律标 `provisional`。声明 bar 与真 bar 的每个 gap 都是未来推翻点。
- **别在 BF16-dequant 上堆 live 集成** → 直接 INT8-native（消临时地基）。
- **能对齐 DeepSeek 就对齐**（addr-align / padding / shape / dtype / layout / static-vs-dynamic T）；必须不一样的写清"为什么 + 验证口径"。每个 step3p5-vs-DeepSeek divergence 是潜在重复 bug —— 问"为什么 DeepSeek 不爆"，别 deep-dive codegen。
- **pin 单一可复现底座**（分支 / 机器 / 版本组合）；底座变更时 re-derive ready 状态，别默认迁移。
- **早期识别信号**：某个"ready"结论只在 compile / offline / synthetic / 单卡 / 单配置下成立 → 立刻标 `provisional`，别当定论传。
- **同一个 error 复现第 3 次就停手**：不要"换姿势重试"。第 3 次时强制回答三个问题——(1) runtime 自己说了什么（`sub_class` / `stuck_*`）？(2) 我改的层和它指的层是同一层吗？(3) 我这次跑绿的门能看见这个 bug 吗？三问有一个答不上就先补诊断，不要再出候选。
- **接手别人的候选先做特征矩阵 + 先跑 `diff`**：`grep -c <新特征> candidate/ <生产基线>/ develop/` **三列**一起数（漏掉生产基线那一列就会把一行的隔离候选误判成"捆了两个"，见 §3 根因 8 的更正），然后对 `decode_fwd.py` 直接 `diff` 生产基线与候选。真捆了两个才拆。
- **多 rank 故障先看少数派报告，但只读到"哪个子系统"这一层**：8 个 rank 的 `sched_error_code` / `sub_class` 分布若是"多数一致 + 一个例外"，例外那个告诉你**停摆主体是 orchestrator 还是 L2 scheduler**（多数派通常是对称 barrier 的受害者签名）。命令：`grep -E "PTO2 runtime failed: orch_error_code=" log | sed -E "s/.*failed: //" | sort | uniq -c`。⚠ **它不告诉你哪个 rank 有责任** —— 要判责任用 AICPU pid 计数：`grep -ho "AICPU([0-9]*," log | sort | uniq -c`（曾据少数派推出"1 元凶 + 7 受害者"，被 8 个 distinct pid 全阻塞在同一 producer 证伪，见 [`16`](16-dispatch-fusion-orch-decouple.md) §5）。
- **锁粒度跟门性质对齐**：只有 A/B/A 计时门需要整机锁；compile / liveness / 精度门一律半机。错配会把分钟级任务变成需要等两半都空的任务。
- 相关约束落点：本仓 [`06-gate-topk-deadlock.md`](06-gate-topk-deadlock.md)、[`11-8001-bridge-live-ops.md`](11-8001-bridge-live-ops.md)、`pypto-lib/docs/known-pypto-pitfalls.md`、`pypto-lib/docs/dev-workflow-gotchas.md`、memory `feedback_integration_churn_root_causes` + `feedback_align_deepseek_architecture_first`。
