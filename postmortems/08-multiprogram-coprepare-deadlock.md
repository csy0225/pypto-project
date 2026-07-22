# 专项：多程序 co-prepare 死锁（N≥6 program 墙，SCOPE_DEADLOCK / SCHEDULER_TIMEOUT）

| 字段 | 值 |
|------|----|
| **子系统** | whole-net / runtime |
| **error signature** | `code -1` (SCOPE_DEADLOCK) / `sched_error_code=100` (SCHEDULER_TIMEOUT) / `HandleTaskTimeout ... Split kernel TaskMapSize=0` / `507018` on first dispatch |
| **首次出现** | 2026-07-08 |
| **状态** | ✅ 已解（架构层面裁定：永久放弃多程序路径，整网收敛到单 `@pl.program`） |
| **相关 skill / doc** | [`../design/whole-net/01-system-design.md`](../design/whole-net/01-system-design.md) · [`.claude/skills/pypto-whole-net-hang-debug/`](../.claude/skills/pypto-whole-net-hang-debug/) · sibling `07-whole-net-scheduler-timeout.md` |

## 1. 背景（Background）

事故发生在 N1 整网 decode 集成早期。当时 whole-decode worker 的设计还是**每层独立 `@pl.program`**（Option-C：TP-attention 程序 + `moe_block` 程序，按层 `select_decode_layer(li)` 分发），目标是让 `DistributedWorker`（simpler `#1706`）在 load 时把全部 45 层所需的 distinct programs **co-prepare** 到一个 resident worker，serving 时每 token 顺序 `rt.run` 各层程序、层间 residual 在 host 侧串。

机器：`gpu-a910x-0162` cards 8-15，TP=8。harness：`_stage_whole_decode_run.py --worker`。

按 `select_decode_layer` 层表 dedup 后，真实 distinct 程序数 ≈ 7-8（`full_dense` / `swa_dense` / `attn_full` / `attn_swa` / `moe_silu` / `moe_swiglu7_silu` / `moe_swiglu16`，harness `moeblk_cache` 按 `kind` 多编译一遍 silu 会记成 8）。

## 2. 现象（Symptom）

prepare **≤5 个程序**时全部 rc=0、residual 串接正确（30.4→53.5→64.0）；prepare **≥6 个程序**就开始死锁。3 次 device 尝试的错误签名：

```text
# 尝试 1：默认 ring
sched_error_code=100    # SCHEDULER_TIMEOUT, dev8

# 尝试 2：PTO2_RING_* env raise（16GB / 524288）
code -1                 # SCOPE_DEADLOCK, dev14

# 尝试 3：per-dispatch RunConfig(ring_task_window=2^20, ring_heap=16GB, ring_dep_pool=2^20)
code -1                 # SCOPE_DEADLOCK, dev8
```

**关键现象**：即使 `task_window=65536`（2^16，生产推荐值，arena ≈ 4GB）把 **PREPARE 阶段**修好（`[worker] PREPARE OK`），**第一个 dispatch step**（L0 `full_dense`，standalone 已验证程序）仍立即 fault：

```text
# device AICPU log
HandleTaskTimeout ... Split kernel TaskMapSize=0
event id[0..63] 全 value 0    # AICPU 空转 28s，任务从未到达 device

# host
taskTimeout=28s → AICore 507018 / sched_error_code=100 / runtime_status=-100
→ 8 卡 poison + aclrtResetDeviceForce
```

N 阈值实测（据 session log `blockers.md` 续5/续6）：

```text
N=5 distinct programs  -> PREPARE OK + dispatch OK + rc=0
N=6 distinct programs  -> SCOPE_DEADLOCK
N=7 distinct programs  -> PREPARE OK + dispatch OK + rc=0   # 之前从没测过 7
N=8 distinct programs  -> PREPARE OK + DISPATCH 首步 507018
```

## 3. 根因（Root Cause）

**根因轴 = distinct `@pl.program` 个数 N**（不是程序内容、不是 kernel bug、不是 ring sizing）。

L0 `full_dense` 单独 device PASS、N≤5 全部 PASS → 排除 kernel bug。`task_window=65536` 修好 PREPARE 后 DISPATCH 仍挂 → 排除 ring sizing。func_id 上限 1024（8 程序远不到）、`MAX_REGISTERED_CALLABLE_IDS=64` 远够 → 排除容量上限。

sw-analyst 对 simpler runtime `worker.py:1971-2121` 的代码定位（据 session log 续5）：N≥threshold 的 dispatch wedge = **co-prepare fork-then-prewarm 协议的结构性 race**。首次 `run()` 后 N 个程序的 pre-warm `_CTRL_PREPARE` fan-out 与 dispatch 的 `TASK_READY` race → chip child 没推进 → task 不到 AICPU → `TaskMapSize=0` → 60s timeout → `507018`。**这不是可调 limit，是协议结构 bug**，根 fix 要在上游 simpler runtime 的 pre-warm loop 加 per-chip prepare-ack barrier + timeout。

DeepSeek V4 的对照（续3 定位）：DeepSeek 根本不 fuse attention+MoE，而是用 `decode_front`(attention) + `decode_back`(MoE) 两个独立 `@pl.program` 按层反复 `rt.run`，per-layer 权重走 runtime tensor 参数（不 bake 进编译）→ N=block-type 数=2~6，永远 <6 墙。**DeepSeek 绕开了这堵墙，是因为它的 distinct program 数本来就少**。

step3p5 的 fault 来自把 TP-attention + EP-MoE fuse 进一个 `chip_orch`（`decode_layer.py:2629-2860`，11 个 TP+EP 混合 comm window），或每层一个 distinct program（N 线性爆到 ~87）→ 两种结构都撞 N≥6 墙。

## 4. 如何解决（Fix）

**用户裁定（2026-07-14，覆盖之前所有"program 个数 N 三档"权衡）**：

> "多程序从来不考虑…实现不了是代码bug"。

据此，**多程序路径（per-layer / Option-C 多程序 / DistributedWorker co-prepare）永久排除**。生产形态收敛到 **N=1 整网单 `@pl.program`**（`WholeDecodeFaithfulReal`），45 层 host_orch 源码 unroll 进同一个 program。N=1 若跑不通（A2 collective `507018` / S1 死锁）= collective handshake **代码 bug**，修它，不换路径。

落点：

- builder：`decode_layer.py:24786` `_build_whole_decode_faithful_real_program`
- `@pl.program` 类：`decode_layer.py:24897` `WholeDecodeFaithfulReal`
- 模块 binding：`decode_layer.py:31636`
- 生成器：`tools/step3p5/_gen_faithful_real.py`（~1908 行，文本生成 builder，按 45 层 layer 表 source-unroll）
- resident holder：`whole_decode_holder.py`（`build()` 编译 / `__enter__` `compiled.prepare()` 常驻 + `import_weights_all` + `import_kv_all` / `run()` 每 step 一次 `self.rt.run(self.compiled, *self._args_list)`）

N=1 单程序跑通后，衍生出的新 deterministic bug（跨层 comm window alias）由 sibling [`07-whole-net-scheduler-timeout.md`](07-whole-net-scheduler-timeout.md) 记录并修复（per-layer `_L{pos}` comm window）。

**前期阶段性缓解（已被裁定作废，仅作历史记录）**：

- `distributed_runner.py:724` INIT call_config 注入 `RunConfig(ring_task_window=2^16, ring_heap=2^32, ring_dep_pool=2^16)`（env `PTO2_WD_INIT_RING_TASK_WINDOW=65536` 可调）—— 修 PREPARE 阶段的 ring sizing，但**修不了 DISPATCH wedge**。
- `batched co-prepare`（≤5/批，host 侧跨批串 residual）—— sw-analyst 代码确认 `create→prepare≤5→run→close→recreate` 可行（chip child 在 close 时 reaped），但**对 serving 不可行**（prepare 是 load 时一次性 + chip 常驻，不能 per-token re-prepare，也不能在 8 卡上跑两套 8-chip worker）。

## 5. 走过的弯路（Detours / What We Got Wrong）

2026-07-08 续1-6 的调查线程，按时间列出被证伪的假设与无效尝试。

- ❌ **per-dispatch `RunConfig(ring_*)` 能解 co-prepare 死锁** → 证伪：`PTO2_RING_*` env 本 build 疑似不读取（`runtime_maker.cpp:261` call_config 的 `ring_task_window` 优先于 env）；per-dispatch RunConfig 在 line 930 太晚，共享 ring 在 `_w.init()`（line 709）时已按 `CallConfig()` 默认分配。耗尽的是 prepare-time 的**共享 worker ring**，不是 per-dispatch ring。
- ❌ **`task_window=2^20` 越大越好** → 证伪：`rtMalloc failed: 207001 (size=6871947759)` ≈ 64 GiB "pooled static arena" OOM（arena ≈ task_window × ~65536 B/task；2^20×64KB≈64GB 顶爆 64GB 卡）。正解是生产推荐值 `2^16=65536`（arena ≈ 4GB）。
- ❌ **fused swa_moe 的 507018 是 attention kernel bug** → 证伪：`_stage_whole_decode_run.py --layers 1`（swa_dense = attention + dense MLP，无 MoE）device 8 卡 PASS 21.99s；moe_block 单独 dispatch ✓；但 attention+MoE fused ✗。fault 在 MoE-fusion host_orch dispatch（`TaskMapSize=0`），不在 kernel/对齐（无 `0x800/errcode`）。
- ❌ **"fused 是对的结构，只是 attention 重写不够好"** → 证伪：DeepSeek V4 根本不 fuse attention+MoE（`models/deepseek/v4/moe_ep.py:175-234` decode host_orch 只有 MoE；attention 是 `@pl.jit.inline` sub-kernel 无 host_orch；V3_2 拆成 `decode_front` + `decode_back` 两个 program）。step3p5 把 TP-attention + EP-MoE fuse 进一个 `chip_orch` 是**错的结构**。
- ❌ **distinct program 数是 8** → 证伪：`select_moe_block(li)` 是 attention-agnostic，对 L3/L4/L5/L8 返回**同一个 program 对象**（`id()` 实测相同）—— silu moe_block 是一个共享程序。真实 distinct = 7。harness `moeblk_cache` 按 `kind`（含 swa/full）做 key 把同一个 silu 程序编译两遍 → 多算 1 个。
- ❌ **N=6 就挂说明 N=7 也挂** → 证伪：`_probe_Nsweep_v0.py` 实测 N=7 `PREPARE OK -> DISPATCH OK -> SUCCESS`，clean finalize，无 507018。墙在 6~8 之间，N=7 通。但这个发现已无意义，因为用户裁定走 N=1。
- ❌ **comm-domain window O(N) 池撑爆 / tensormap 分区 / fork-prewarm race / state-select 四个候选** → 4-agent file:line 全部否定：comm window 同名 `comm_d0` per-dispatch 分配即释放（`aclrtMemset` 清零）；tensormap/ring/arena per-Worker 无 program-id 字段；fork-prewarm 阻塞 ack-barrier；per-program state 按 `id(program)` keyed dict。没有"O(N) 固定池"这种东西。
- ❌ **SWA attention 的 `valid_shape` Var symbolize 是整网 blocker** → 证伪（部分）：这是 fused swa_moe 的**编译**墙（EP lowering 把 `valid_len` symbolize 成 free Var → `tile.create got Var`），不是 dispatch 墙。`pl.tile.load` 把 scores 搬进 UB tile 后 `row_max` 输出变 const → 编译过。但 fused 路径本身是错的结构，修编译也白搭。

## 6. 如何避免（Prevention）

**铁律**：

1. **整网只用一个 `@pl.program`**。这是 [`../design/whole-net/01-system-design.md`](../design/whole-net/01-system-design.md) §1 的生产形态约束。禁止退回 per-layer 多 program、禁止 Option-C 多程序、禁止 DistributedWorker co-prepare 多个 whole-net 程序。
2. **多程序是 simpler runtime 的结构 bug，不是 step3p5 的设计选项**。如果未来有人提"用 N 个 program 分摊编译/复用"，引用 2026-07-14 用户裁定和本复盘拒绝。
3. **N=1 若跑不通，修 collective handshake 代码**，不换路径。典型 successor 问题见 [`07-whole-net-scheduler-timeout.md`](07-whole-net-scheduler-timeout.md)（per-layer comm window alias）和 [`06-gate-topk-deadlock.md`](06-gate-topk-deadlock.md)（gate_topk mrgsort）。
4. **不要用 `batched co-prepare` 作 serving 方案**。prepare 是 load 时一次性 + chip 常驻，不能 per-token re-prepare，也不能在 8 卡上跑两套 8-chip worker。只能作 validation-only 权宜。
5. **`task_window=65536`（2^16）是生产推荐值**，不要盲目调大（2^20 → 64GB arena OOM）也不要用默认 16384（N≥6 SCOPE_DEADLOCK）。

**早期识别信号**：

- `SCOPE_DEADLOCK` (code -1) + `SCHEDULER_TIMEOUT` (sched=100) + `HandleTaskTimeout ... TaskMapSize=0` + AICPU 空转 → 先数 distinct program 个数 N。N≥6 几乎一定是 co-prepare 墙。
- L0 standalone PASS、N≤5 PASS、N≥6 第一次 dispatch 就挂 → 排除 kernel bug，直接怀疑 co-prepare 协议。
- `PREPARE OK` 后 DISPATCH 仍挂 → 不是 ring sizing，是 host→AICPU 派发 wedge，调 `task_window` 无效。

**落点**：

- 本复盘 + `design/whole-net/01-system-design.md` §1（唯一生产形态 = 单 `@pl.program`）。
- skill `.claude/skills/pypto-whole-net-hang-debug/`（orchestrator code 表：`code -1` = scope deadlock / `sched=100` = scheduler timeout）。
- `blockers.md` 2026-07-08 节（⛔ 作废标记，保留仅为历史）。
- memory `blocker1_coprepare_wall_overcounting_N7.md` / `feedback_align_deepseek_architecture_first.md`。
