# 专项：整网 scheduler timeout（507018 / orch_error=8 TENSOR_WAIT，per-layer comm window alias）

| 字段 | 值 |
|------|----|
| **子系统** | whole-net |
| **error signature** | `507018` mid-exec + `orch_error_code=8` (TENSOR_WAIT) / `sched_error_code=100` (SCHEDULER_TIMEOUT) |
| **首次出现** | 2026-07-10 |
| **状态** | ✅ 已解（deterministic 部分；后续 07-16 残余随机 stall 是另一类问题，见 `../reference/cache-line-signal-isolation.md` / `12-integration-churn-meta.md`） |
| **相关 skill / doc** | [`../design/whole-net/02-detailed-design.md`](../design/whole-net/02-detailed-design.md) §6 · [`.claude/skills/pypto-whole-net-hang-debug/`](../.claude/skills/pypto-whole-net-hang-debug/) · [`../reference/cache-line-signal-isolation.md`](../reference/cache-line-signal-isolation.md) · sibling `06-gate-topk-deadlock.md` · `08-multiprogram-coprepare-deadlock.md` |

## 1. 背景（Background）

事故发生在 N1 整网 decode bring-up 阶段。生产形态已经裁定为**单个 whole-net `@pl.program`**（`WholeDecodeFaithfulReal`，builder 在 `decode_layer.py:24786` `_build_whole_decode_faithful_real_program`，binding `:31636`），把 45 层 decode（3 dense/swa 前缀 + 42 MoE）host_orch 源码 unroll 进同一个 program，TP=8 在 cards 8-15 上跑。目标是让单个 resident holder（`whole_decode_holder.py`）一次 `rt.run` 把整网 45 层 decode 跑完，对接 vLLM live serving。

机器：`gpu-a910x-0162`（driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1 三剑合璧已就绪，见 [`../deployment/phase16-three-pillars.md`](../deployment/phase16-three-pillars.md)）。

触发命令（据 session log）：

```bash
# staging harness（whole_decode_holder.py 路径，faithful real weights）
_stage_whole_faithful_device.py --tp 8 --dev-offset 8 --layers 0-44
# 或单步复现：
_stage_whole_decode_run.py --worker --tp 8 --dev-offset 8
```

## 2. 现象（Symptom）

单 MoE 层（`P_FAITHFUL_MOE_LAYERS=1`，即 3 dense + 1 MoE）能 RUN_CLEAN；把 MoE 层数加到 2（`P_FAITHFUL_MOE_LAYERS=2`）就**确定性卡死**，错误签名：

```text
aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018
orch_error_code=8        # TENSOR_WAIT_TIMEOUT
sched_error_code=100     # SCHEDULER_TIMEOUT
runtime_status=-100
# 8 卡 poison → aclrtResetDeviceForce(0..7)
```

`orch_error_code=8` = tensor-wait timeout（见 `.claude/skills/pypto-whole-net-hang-debug/SKILL.md` §3.1 orchestrator code 表），即某个 tensor 的依赖 fanin 永不满足；`sched_error_code=100` = scheduler timeout，AICPU 空转。卡死位置在**第二层 MoE 进入时**（layer-4 进入对 layer-3 通信对象的访问），不是 compile、不是 prepare、不是首个 dispatch。

关键对照（dispatch-cut bisect）：

```text
K=1  (1 个 MoE 层，复用 1 套 comm window)  -> RUN_CLEAN
K=2  (2 个 MoE 层，复用 1 套 comm window)  -> 确定性 507018 / orch_error=8
K=42 (完整 42 MoE 层，复用 1 套 comm window) -> 确定性卡死
K=2  (2 个 MoE 层，各用独立 comm window)    -> RUN_CLEAN
K=42 (完整 42 MoE 层，各用独立 comm window) -> RUN_CLEAN 45.89s（8 卡）
```

## 3. 根因（Root Cause）

42 个 MoE 层在同一个 `@pl.program` 的 host_orch 里**复用同一套 communication window**（`combine_done_buf` / `recv_x` / `pub_counts` / `routed_y_window_buf` / `attn_tmp_buf` 以及对应 signal SSA）。

生成的 `host_orch.py` 明确显示 layer-3 和 layer-4 的 `chip_orch` 引用了**同一个** SSA buffer，例如 `combine_done_buf__ssa_v0` 在两层间被 alias。这违反 pypto runtime 的 **RAW-only-v1** 依赖模型（`ADR-013` / P3）：仍可能被上一层远端 rank 读取的中间 buffer，不能在下一层直接复用为另一个逻辑对象，否则 SSA/lifetime 描述与硬件实际访问不一致。

机制：上一层（layer-3）的 cross-rank reader（`pld.tile.remote_load` / `pld.system.wait`）在 GM-NoC 上还在飞行，下一层（layer-4）重新写同一 signal/payload 地址 → 下一层 wait 的 fanin 被旧 generation 提前满足或被新写入覆盖 → tensor 依赖永不达成 → `orch_error=8 TENSOR_WAIT` → 60s timeout → `507018`。

证据链：

1. **dispatch-cut bisect**（`P_FAITHFUL_MOE_LAYERS` 旋钮，`decode_layer.py:24856` `_FAITHFUL_MOE_LAYERS = int(os.environ.get('P_FAITHFUL_MOE_LAYERS','42'))`）：K=1 clean / K=2 stall，把失败边界从"整网"缩到"第二层 MoE 进入"。
2. **生成 `host_orch.py` 静态审计**：layer-3 与 layer-4 的 `chip_orch` 引用同名 SSA `combine_done_buf__ssa_v0` / `recv_x_buf__ssa_v0` / `pub_counts_buf__ssa_v0` / `routed_y_window_buf__ssa_v0` —— 直接显示跨层 alias。
3. **决定性对照**（见 `.claude/skills/pypto-whole-net-hang-debug/references/n1-causal-chains.md` §4.17.1）：保持其他变量不变，只把 comm window 从"共享"改成"per-layer distinct"，K=2 和 K=42 都从确定性卡死变成 RUN_CLEAN。这组对照排除了"层数变大所以概率变差"的弱解释，坐实 alias 是独立、确定性的架构 bug。

## 4. 如何解决（Fix）

落地修复 = **在同一个 `@pl.program` 内给每层分配独立的 comm window**，命名加 `_L{pos}` 前缀：

- `attn_tmp_buf_L0` / `pub_counts_buf_L0` / `recv_x_buf_L0` / `combine_done_buf_L0` / `routed_y_window_buf_L0` …，`decode_layer.py:27761+`，`_L1`…`_L41` 重复（dense 前缀层用 `l0_/l1_/l2_`）。
- 生成器 `tools/step3p5/_gen_faithful_real.py`（~1908 行文本生成 builder）按 layer pos 展开时，每层 emit 一组独立 buffer alloc + 独立 signal。
- 改动后中间张量（`h_moe_L{pos}` / `h_mid` 等）也遵循 **write-once per layer**，不跨层复用 SSA（`02-detailed-design.md` §3 不变量）。

配套修复（与 per-layer window 同期落地，都是 deterministic 阻塞链上的必要环节）：

- **512B signal 物理隔离**：`COMM_CONTROL_SIGNAL_BYTES=512`（`decode_layer.py:24895`），逻辑仍是 `[tp,1]` INT32，但物理独占一条 512B L2 cache line，避免 signal 与相邻 payload 共线导致 false-sharing / atomic 串行化。详见 [`../reference/cache-line-signal-isolation.md`](../reference/cache-line-signal-isolation.md)。
- **`tp_all_reduce` 两波完成 barrier**（`decode_layer.py:24905-24980`，四相：stage-in → notify(Ge 1) → own-load + remote_load + FP32 tadd → completion barrier(Ge 2)）。第 ④ 波由 `tools/step3p5/_add_allreduce_completion_wave.py` 注入 —— 单波在 ≥41 层会挂；两波后 deterministic collective 路径通（但残余随机 stall 仍需 512B 隔离，见 §5）。

**关键坑**：改生成器加 `_L{pos}` 后，**死的旧 alloc 必须删干净**。生成器是文本拼接，旧的共享 alloc 残留会让 `MaterializeCommDomainScopes` pass 报错（comm domain 里同名 buffer 冲突）。流程：

```bash
# 1. strip 旧 builder 副本
python tools/step3p5/_strip_real_builder.py
# 2. regenerate
python tools/step3p5/_gen_faithful_real.py
# 3. byte-compare roundtrip gate（02-detailed-design.md §9 不变量 5）
```

验证：完整 45 层 TP=8 cards 8-15 `RUN_CLEAN 45.89s`，无 `507018`、无 `aclrtResetDeviceForce`。

**适用边界**：本修复只解决"跨层 comm window alias"这个 deterministic bug。2026-07-16 出现的残余随机 stall（同 signature `507018 / orch_error=8`，但发生在已经 per-layer distinct 的版本上，20/20 复跑约 1/3 概率挂）是**另一类问题**，与 512B signal physical isolation 强关联但未获唯一硬件证明 —— 见 [`../reference/cache-line-signal-isolation.md`](../reference/cache-line-signal-isolation.md) 和 skill `n1-causal-chains.md` §4.17.6。勿把两者混为一谈。

## 5. 走过的弯路（Detours / What We Got Wrong）

按时间列出被证伪的假设。每条都附证伪实验。

- ❌ **comm-window byte-cap 假设**（"window 太小，N 层撑爆"）→ 证伪：device 上把 allreduce window 从 64MB 扩到 2GB（offset 2GB），隔离 8-rank allreduce 仍 clean；同时 co-resident 24GB IPC pool 也 clean（allocator 保持 VA disjoint）。`winSize`/`windowsIn` 都是 uint64，无字节上限。卡死与容量无关。见 memory `n1_comm_window_bytecap_refuted.md`。
- ❌ **IPC VA-collision 假设**（"MoE comm window 与 IPC pool VA 重叠"）→ 证伪：device log 显示 MoE comm window `0x12c041...` 在 IPC pool `0x12c1c0...` 之下，无重叠；IPC map 48-key 全部 in-pool 对齐。曾长期怀疑并驱动 Blocker B 调查，后被 `gate_topk` 的 exact TASK 证据推翻（`func_id=3 -> gate_topk`，见 sibling `06-gate-topk-deadlock.md`）。
- ❌ **"shrink pool" / "avoid overlap" 两种修法**（基于上一条错误假设）→ 无效：既没解决问题，也没改变复现率。pool 大小与 window 重叠都不是因。
- ❌ **poison 级联误判**（"某次 reset 没清干净，卡毒扩散到全 8 卡"）→ 证伪：reboot 后 + 全 fresh pool 仍复现；隔离 PULL-探针 `N_COLL=2/340` 都 CLEAN，证明卡健康。把"结构 bug"误判成"环境 poison"浪费了多轮 device 重试。
- ❌ **single-wave `tp_all_reduce` 是根因**（早期假设）→ 证伪：加两波 completion barrier 后仍约 2/3 stall；single-wave 只解释 ≥41 层挂的 collective 协议缺陷，不解释 K=2 就挂的 alias bug。两波是必要但不充分。
- ❌ **"层数变大所以概率变差"**（弱解释）→ 证伪：K=1 clean / K=2 确定性挂，不是概率退化；生成 `host_orch.py` 直接显示 layer-3/layer-4 引用同一 SSA。
- ❌ **用不同 buffer 名替代实际 SSA 审计**（"我改了名字应该没事"）→ 证伪：必须看生成的 `combine_done_buf__ssa_v0` 实际 physical offset 和 lifetime ledger，符号名换了 SSA 没换等于没修。

## 6. 如何避免（Prevention）

**铁律 / 检查项**：

1. **整网单 `@pl.program` 内，每层 comm window 必须 distinct**（`_L{pos}` 前缀），层间不复用 SSA。这是 `02-detailed-design.md` §9 不变量 1，写生成器时强制。
2. **死的旧 alloc 必须删**。改生成器后跑 `_strip_real_builder.py` → regen → byte-compare roundtrip，否则 `MaterializeCommDomainScopes` 报错。
3. **中间张量 write-once per layer**（`h_moe_L{pos}` / `h_mid`），不跨层 stash/WAW。
4. **signal 512B 物理隔离**（`COMM_CONTROL_SIGNAL_BYTES=512`），与 payload 不共 L2 cache line。
5. **`tp_all_reduce` 保留两波完成 barrier**（stage-in → notify(Ge 1) → accumulate → completion(Ge 2)）。
6. **bisect 先于假设**：看到 `507018 / orch_error=8` 先跑 `P_FAITHFUL_MOE_LAYERS` dispatch-cut（K=1/K=2/K=42），把失败边界缩到具体层再下结论。
7. **生成 `host_orch.py` 静态审计**先于 device 重试：grep `combine_done_buf__ssa_v0` / `recv_x_buf__ssa_v0` 在多层间是否同名，比再跑一次 device 便宜得多。

**早期识别信号**：

- K=1 clean、K=2 确定性挂 → 几乎一定是跨层 alias 或 lifetime bug，不是容量、不是概率、不是 poison。
- `orch_error_code=8`（TENSOR_WAIT）+ fanin 永不满足 → 依赖 DAG 问题，查 producer/consumer lifetime，不要先调 ring/dep-pool 容量。
- 生成 `host_orch.py` 里两层 `chip_orch` 引用同名 SSA → 直接坐实 alias。

**落点**：

- 本复盘 + `02-detailed-design.md` §6/§9 不变量。
- skill `.claude/skills/pypto-whole-net-hang-debug/`（特别是 `references/n1-causal-chains.md` §4.17.1 / §4.17.4 / §4.17.6）。
- `pypto-lib/docs/known-pypto-pitfalls.md`（RAW-only-v1 non-aliasing 约束）。
- 顶层 `ADR-013` / P3（RAW-only-v1 依赖模型）。

**与 sibling 复盘的边界**：

- 本文档 = deterministic 的跨层 comm window alias（K=2 就挂，per-layer window 修好就 clean）。
- `08-multiprogram-coprepare-deadlock.md` = 多 `@pl.program` co-prepare 墙（N≥6），是本整网被迫走"单 `@pl.program`"的直接原因。
- `06-gate-topk-deadlock.md` = `gate_topk` mrgsort 死锁（同 `507018` signature 但不同 task）。
- 残余随机 stall（07-16，20/20 约 1/3 挂）= 512B signal isolation 关联问题，未单独编号，见 `cache-line-signal-isolation.md`。
