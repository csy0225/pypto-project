# NEXT SESSION — N=1 整网 decode 集成 + 端到端精度对齐 vs vLLM

> **✅ 续11（2026-07-14 本 session，device 验证）— A2 死锁修复 + argmax=303 复现 3/3；残余 logit 抖动已定位但 reference 修法在流水线里死锁（已回退）。**
> **两个 root-cause 修复（对齐框架/moe.py，无 work-around，均在生成的 decode_layer.py，未 commit）**：
> 1. **A2 collective 死锁 = 手写 `tp_all_reduce` 缺完成波（single-wave）**。wiki `S1:running-stalled` = collective kernel 挂死（非容量）。框架 `pld.tensor.allreduce` 是 two-wave。修：全 17 份 `tp_all_reduce` 加 Phase-4 完成波（`notify+wait≥2`，transform `tools/step3p5/_add_allreduce_completion_wave.py`）。**device：P=42 确定性 RUN_CLEAN、无 507018、不靠 logging（7 次干净）。多 session 的硬 blocker 解除。**
> 2. **argmax 非确定 = combine zero-vs-push 竞态**（丢了 moe.py `pub_route_barrier`）。修：`combine_done` 上补 zero-done 波（`tools/step3p5/_fix_combine_zero_push_race.py`）。**device：argmax=303==vLLM golden 复现 3/3（修前 303/20/303）。→ L3 greedy top-1 准出线达标。**
>
> **残余 = L2 logit 分布抖动（top-303 logit 8-14，margin 最薄 0.9）**：device 二分 P=0(dense) 逐 bit 相同、P=1(单 MoE 层)抖 → 单 MoE 层引入、跨 42 层累积。**根因（reference 确认）= 整网丢了 moe.py `_serialize_after_shared`**（强制 shared-expert tp_all_reduce 先于 routed dispatch/combine，防 collective 重叠）。**但插入它 device 实测 P=42 死锁 3/3（507018）**——它把 42 层流水 schedule 过约束（moe.py 只在单块非流水里安全）。**已回退。**
> ⟹ **可靠结论：残余抖动 与 A2 死锁 是同一「流水化 collective 调度脆弱性」的两面——不能靠加 per-layer 串行化消除抖动而不重新引入死锁。**
> **⭐ 残余抖动已 vector-diff 精确定位（device，本 session）**：P=1 逐 stage row0 向量 ×2 offline L1/cos 对比（`logs_n1/_vec_diff.py` + harness `N1_DUMP_DIR`）：**resid_hold（attention 残差）L1=0 cos=1.0 完全确定**、**moe_out（combine 输出）L1=113 cos=0.990 DIVERGE** → **抖动源 = COMBINE（`moe_out=_weighted_gather_and_add(routed_y_buf,w,sh_y)`），不是 attention**（推翻早前 attention-residual 假设）。max-based 探测看不到（各 stage max 全同），必须 vector-diff。已有的 combine 修复（zero-done+self-fence+completion-wave）修好了 argmax(303) 但残留 ~1% moe_out 向量竞态。**下 session 精确步骤**：dump routed_y_buf(gather 后) 向量 vs local_routed_y 向量 ×2，区分 (a) INT8 routed 计算 vs (b) 跨卡 routed_y_buf gather；再做非串行化修复。routing 由确定的 post_norm 派生（gate/dispatch 确定），竞态在 INT8 routed 计算或 gather 读序，不在 routing。**从不翻 greedy argmax（303 稳定）→ L3 准出线已达标，属 L2-cos 细化。**
> 下 session 抖动三选一：(a) 接受 argmax=303 3/3 greedy-exact（准出线已达标）+ 记为 known jitter；(b) 上面 vector-diff 精确定位后做非串行化修复；(c) 深入 pypto-runtime collective 调度器。**勿再试 `_serialize_after_shared`（device 证明死锁）；勿再跑 max-based 探测（看不到向量抖动）。**
> 详见 memory `n1_m4_accuracy_gap_converged_direction_drift`（续11 段）。两个 solid 修复未 commit（root-owned .git → tmux commit，b-csy push）。
>
> **⛔⛔ 续10（2026-07-14）— 多程序永久排除；N=1 单程序唯一方案；实现不了是代码 bug（用户裁定，已在 SKILL §H·§D / blockers.md / notes/07 标注）。**
> ⟹ **(1) 多程序 + resident-DeviceTensor 永久出局**——任何 session 不得再提议 pivot 多程序、不得把 A2 说成"N=1 固有所以要转多程序"。**(2)** N=1 整网单 `@pl.program`（`WholeDecodeFaithfulReal`）= 唯一生产形态。**(3)** A2 collective 死锁 = **N=1 collective handshake 的代码 bug**，只能定位+修，不是框架墙。（据此把续9 底部"三条可选(B)评估 multi-program"删除、SKILL §H·§D / blockers.md / notes/07 均已标注。）
>
> **A2 权威 wiki 根因（[Device-Error-Codes_zh](https://github.com/hw-native-sys/simpler/wiki/Device-Error-Codes_zh)，续10 分析）**：本次 P=42 挂死签名 = `sched=100 sub_class=S1:running-stalled running=1 waiting=1 completed=39/137 orch_done=0 stuck_task_id=(1,23) stuck_core=26` + `orch_error=8 TENSOR_WAIT` + `507018/507014`，**8 卡同一 stuck task**。wiki：**S1:running-stalled = "任务在核上跑但永不完成 → AICore kernel 挂死或过久"**（不是容量！容量死锁是另外的 orch 码 1/2/3/4/11，我们一个都没有）。⟹ **推翻正向 agent 的头号嫌疑"ring-1 容量耗尽"**，也印证之前 RING/dep_pool/task_window knob 全无效。真因 = **per-layer `tp_all_reduce` collective kernel（decode_layer.py:628-674，task (1,23)）在 41+ 流水深度间歇挂死**：`notify(AtomicAdd,+1)`/`wait(Ge, expected=1)` 握手在深度处丢/乱序一个 notify → wait 永久自旋（全 rank 同 task = 跨 rank 互等；logging 能躲开 = 时序/内存序竞态）。wiki 的 S1 补救"调大 PTO2_SCHEDULER_TIMEOUT_MS" = **work-around（掩盖，禁用）**。修法 = root-cause 该 handshake（bound 在飞 collective signal 状态 / 确认 per-layer tmp_window+signal_window distinct+reset+fence），让 P=42 **不靠 logging 也确定性跑通**。
>
> **续9（2026-07-14，pushed pypto-lib `7294e26`）— 精度主因之一定位并修复：dense L2 `attn_layer_idx` bug。**
> 手法：搭**精确 dense torch golden**（`tools/step3p5/_dense_golden_ctx1.py`；dense 层纯 BF16、ctx=1⟹attn_out=value，无需 vLLM）+ 逐层比对（`_cmp_vec.py`，看 row0 cos 不看 max）。device P=0（真 W8A8+双 IPC 8 卡）：L0+L1(h_mid)==golden **cos=1.0**，L2(next_hidden) **cos=0.931 DIVERGE**。真因：`WholeDecodeFaithfulReal`（decode_layer.py:24438，非 21918 那个 unused 的 WholeDecodeFaithful）dense L2 调用 `swa_chip_orch` 误传 `attn_layer_idx=1`，而 wq/wk/wv/w_g 已 call-site 预切（swa_wq[r,1]）；`attention_swa.py:226 layer_hidden_base=attn_layer_idx*HIDDEN` 对预切单层权重再 K-偏移 → 越界读到相邻 layer-3 权重。L1+全部 42 MoE 层都传 0。**修复 L2→0（+ generator `_gen_faithful_real.py::_emit_l2`），device P=0 复验 cos 0.931→0.999999 MATCH。** 该 bug 污染所有 42 MoE 层输入。
> **遗留（下一步主线）**：P=42 全网 A2 507018 collective deadlock 现已判定为 **intermittent 时序竞态**（device：P=41 无 logging STALL、开 `ASCEND_GLOBAL_LOG_LEVEL=1` 后 CLEAN；说明非确定性结构死锁）。**已用 logging 时序扰动让 P=42 跑通一次 → `--hidden-token 6127` argmax=303（TOP5 [303,410,1176,525,3163] logit 13.63，303 为自信 top）== vLLM golden ⟹ ✅ M4 token-exact 达成、L2 修复即精度主因坐实。** confirm run 复现 STALL（竞态间歇）。**A2 竞态本身仍需修**（robustness，非精度）：已 device 二分阈值=41；两 agent 排除 address-reuse（MemoryReuse 跳过 Orchestration + AllocateMemoryAddr bump 无 wrap）、next_hidden_out 多 producer（generator L910 compile-time gate 只 1 个 writer）、DAG 定长溢出（各 pool 远未满、溢出会报别的码）。**主嫌 = `MaterializeCommDomainScopes`(pass 38) 的 per-layer `CommDomainScopeStmt` scope-release 与 peer rank IPC remote_load 抢跑 → Ge(N) 信号不满足 → TENSOR_WAIT**。决定性下一步：dump pass-38 IR 看 layer-N/N+1 的 CommDomainScopeStmt 是否 NEST（重叠→竞态）；若是，scope close 处加 fence/barrier 确保 peer 读完再 release。临时 unblock：开 `ASCEND_GLOBAL_LOG_LEVEL=1`。
> memory `n1_m4_accuracy_gap_converged_direction_drift`（含 L2 根因 + 全部已证伪项）。
>
> 直接把最底部 code block 当第一条消息粘贴。自包含。**权威当前状态 = 底部 code block 的「当前状态（续9）」段**（✅ M4 token-exact argmax=303==vLLM 达成、L2 attn_layer_idx bug 已修；唯一遗留 = A2 运行时时序竞态 robustness）。⚠ 本文件"续8/续7/续6/续5/M3/M3b"等历史段落中所有"M4 精度未达成 / argmax=993 / 303 不在 top-5 / NaN / 幅值爆炸"结论均为 L2 修复前的旧状态，**已被续9更新，勿据此行动**。
> **运行环境：0234 机器，通过本地 tmux `pypto-ascend-0:0` 登陆**（8 卡 0-7；781GB RAM；driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1）。
> 编辑机 `b-csy-develop`（**无 python，有 npu-smi，能直连 github**；NFS 与 0234 共享，编辑即时可见）。分支 `pypto-lib feat/whole-net-n1-fusion`。

---

## 🎯 总目标（north star，勿只盯单步）

**完成 step3p5 整网 decode 阶段的集成 + 端到端精度对齐 vs vLLM**（真 W8A8 + IPC，token-exact）。
本 track = N=1 整网融合（offline）：`whole_decode_faithful_real` —— **全 45 层（42 MoE + 3 dense/swa）内联进一个 `@pl.program`**，真 W8A8 权重+KV 经 **IPC** 加载，harness `tests/step3p5/_stage_whole_faithful_real_ipc.py`，分支 `feat/whole-net-n1-fusion`。
> ⚠ **别和另一个 track 搞混**：`NEXT-SESSION.md` 是 **G5b / per-layer 逐层 golden + live vLLM** track（harness `_stage_whole_decode_run.py`，本分支无此文件）。目标/harness/分支都不同，但 L2/live 阶段可复用其 KV-bridge / co-tenancy 机器。

### 里程碑路线图（当前进度）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 单算子 probe | matmul_acc N=16 / head-gate / gate_topk vs torch | ✅ PASS |
| M1 功能 bring-up | 42 MoE 真 W8A8 + 权重+KV 双 IPC 8 卡 dispatch-clean | ✅（Blocker B 解） |
| M2 per-layer gate_r | monolithic 整网自算逐层 head-gate（on-device，token-exact-capable） | ✅（路径 a） |
| **M3 单层 MoE 数值正确** | **NaN 修掉 → finite** | **✅（2026-07-13 续5：A1 device-proven P=1，NaN 消）** |
| **M3b 单层 MoE 幅值正确** | **next_hidden 合理幅值** | **✅（2026-07-13 续5：A1 —— `_expert_routed` byte-align moe.py，flat-pre-quant INT8；P=1 `local_routed_y` 3.99e11→1.41，next_hidden=502 finite）** |
| **A2 确定性（原 M4 竞态部分）** | **P=0/1/42 跨 run 确定** | **✅（2026-07-14 续8：真因=dense L2 复用 L1 通信/信号窗口，修为独立 l2_* 窗口。argmax P=0/1/42 稳定、无 507018。续6 的"字节上限"、续7 的"hidden 乒乓/相消"均 device 证伪）。⚠ P=42 next_hidden 幅值仍抖 256-416（argmax 鲁棒、残余待查）** |
| M4 L1 ctx=1 token-exact | 全 42 层放开，`--hidden-token 6127` → **argmax=303** vs vLLM | **✅ 达成（2026-07-14 续9，device）** —— 根因 = dense L2 误传 `attn_layer_idx=1` 越界读 layer-3 权重、污染全部 42 MoE 层输入；修 L2→`attn_layer_idx=0`（generator `_gen_faithful_real.py::_emit_l2` + decode_layer.py，pushed 7294e26），P=42 → **argmax=303 == vLLM golden**（303 自信 top）。⚠ 需靠 `ASCEND_GLOBAL_LOG_LEVEL=1` 时序扰动跑通（A2 race，见下）。**勿再查旧的 argmax=993/top-5——已解决。** |
| M4.A2 P=42 稳定跑通 | 不靠 logging 扰动即 RUN_CLEAN | ⛔ **A2 collective deadlock = 运行时动态时序竞态**（Heisenbug，阈值=41 融合层，所有静态假设已 device/代码证伪；robustness 非精度）。下个 session 主项之一。 |
| M5 L2 多 token / decode-step | vLLM→whole-net KV bridge 或 live A/B（8001 vs 8000），多 token token-exact | ⏸（gated on A2；需 port G5b 机器） |
| M6 整网 decode 集成落地 | 接入 serving 路径（live single-handoff），端到端精度双过准出 | ⏸（"完成后端替换"的关键剩余里程碑；gated on A2——不能在 Heisenbug 上建可靠 serving） |

**判据**：L1 per-layer hidden `ratio_allclose(atol=0.04)` / L2 logits cos≥0.999+topK overlap≥4/5 / L3 greedy top-1≥95%。**oracle = vLLM eager dump，synthetic golden 会 stale。**

## ⭐⭐⭐ 本 session（2026-07-13 续5）— A1 INT8 routed FIXED (device-proven)；A2 P=42 collective deadlock 诊断（所有 runtime knob 无效）

> ⚠⚠ **续6 更正（务必先读）**：本 续5/续5b 段的核心结论 **"deadlock 是 pool-bytes/offset；1A 缩 recv_x 是确定修法"** 已被 **device 证伪**（memory `n1_comm_window_bytecap_refuted`：standalone 8-rank allreduce 在 64MB→2GB 窗口全 PASS，24GB co-resident pool 也不复现）。真机制 = **逐层 window/buffer 别名竞态**（P3/ADR-013），非字节上限、非 VA 碰撞。1A 确已实现且让 **P=42 RUN_CLEAN**（有用、保留、对齐 moe.py），但那是"改了 footprint 让别名从 stall 转成 M4 数值竞态"，**不是消除别名**。续5b 的"666MB stall vs 186MB clean"是**在 47GB 权重池共存下**测的，standalone 无权重池时 666MB PASS。**以最底部「当前状态（续6）」+ 任务清单为准。**

> 详见 memory `n1_a1_int8_routed_fixed_p42_windowstall`（续6 段）+ `n1_comm_window_bytecap_refuted`。device 实测为准。

- **✅ A1 (M3/M3b 真解，device-proven)**：把 whole-net `_expert_routed` 换成 **byte-faithful moe.py device-PASS 版本**（新 generator 常量 `FRESH_EXPERT_ROUTED`，`tools/step3p5/_gen_faithful_real.py`）。输入量化改成对**整个 `local_recv_max=1024` recv buffer 的 FLAT pre-quant**（32 个满 RECV_TILE tile，无 partial-tile → 避开 gap-5 miscompile；镜像 moe.py `_quant_moe_input`），`lrx_scale` **非-padded `[1,1024]`** + `[1,RECV_TILE]` row-slice→reshape 读取（ccec ND2ND-safe），bare-slice `routed_h_quant`，`fillpad(zero)` gated。旧 generator `_pre_quant`/repoint string-transform 删除。**device P=1（真 W8A8+双 IPC，`P_DBG_STAGE=3`）：`local_routed_y` 3.99e11 → 1.41 ✓，next_hidden=502 finite，RUN_CLEAN**。前 8 次 UPDATE 的 M3b「handoff/别名」定位被 A1 覆盖（真因就是 `_expert_routed` in-expert partial-tile INT8 quant）。
- **✅ 多层 bisect device-clean**：P=1 ✓、**P=20 ✓**（next_hidden=560）、**P=31 ✓**（next_hidden=2736）。
- **⛔ A2 卡点 = P=42 TP-allreduce COLLECTIVE deadlock**：P=42 ~18s 后 `507018 orch_error=8 (TENSOR_WAIT_TIMEOUT) completed=38 running=1 waiting=2 stuck_core=AIV`；device stall-diag 显示 stuck kernel = `aiv_reduce_scatter`/`aiv_broadcast`/`hccl_aiv_sync`（= TP all_reduce），全 rank 同 task 挂（collective 互等）。**bisect 阈值 31<T≤42（cumulative depth，非特定层——`completed=38` 停在 pos~35-37 regular MoE 层，未到 special L43/L44）**。
- **全部 RULED OUT（device 实测，勿再试）**：① arena exhaustion —— comm-domain window pool 自动扩容且 alloc 成功（P=20 186MB / P=42 391MB 都成功）；② per-layer window aliasing —— generator `_host_orch` 本就 per-layer distinct（`*_buf_L{pos}`）；③ **rotating window pool（K=8 reuse）—— 编译失败 `ConvertToSSA Error 6`**（跨层复用 window buffer = 多个 SSA def；这正是 per-layer-distinct 存在的原因）；④ task_window 131072→65536、dep_pool 131072→524288(4×)、heap 4→8GB —— 全部同样 stall。**没有 runtime knob 能修。**
- **🎯 下一步 = 1A full-moe.py dispatch-side INT8**（evidence-based，最优先）：deadlock 与 pool size 相关（clean ≤290MB@P=31 / stall 391MB@P=42）。当前 A1 保留 **BF16 push-dispatch** → `recv_x_buf`=8MB/层。改 **INT8 recv_x**（moe.py Option A：dispatch 前量化 post_norm→INT8+scale，push INT8 + 并行 `recv_scale`/`send_scale` 窗口，`_expert_routed` 直接读 pre-quantized INT8、去掉 flat pre-quant）→ recv_x=4MB/层，pool **391→~223MB < P=31-clean(290MB)** → 大概率解 deadlock，且更贴 moe.py。改动面：base builder `_dispatch_push`/`_dispatch_stage`/dispatch window decls + `_expert_routed` + generator（模板见 moe.py `chip_orch` L2154-2185）。**备选**：FUSE→SPLIT（SKILL §H，pre-FUSE 有 N=42 clean）。
- **✅ 续5b FALSIFICATION（bytes CONFIRMED / count REFUTED）—— 1A 是确定解**：把 `recv_x_buf` ×4 膨胀（diag）跑 **P=20**（只 20 层 = 同样低 collective count）→ pool `win_size=698690816`(**666MB**) → **507018 STALL**（正常 P=20@186MB CLEAN）。同层数、同 collective count、pool 变大就 stall ⟹ **deadlock 是 pool-bytes/offset，不是 collective-count/depth**。∴ **1A（recv_x→INT8，pool 391→223MB<290MB）是确定修法；FUSE→SPLIT 不必做**（count 理论已排除）。recv_x 是唯一够大的 lever（8MB/层，必须缩它；无非-INT8 捷径能到 <290MB）。已 revert 膨胀，tree 回 clean A1。**surgery 前先 `cp _gen_faithful_real.py{,.bak}`（generator 无备份）**。
- **代码状态**：`feat/whole-net-n1-fusion` 工作区（NFS 0234），**未 commit**。已 regen 回 per-layer-distinct（compiling A1，P=1/20/31 device-clean，P=42 stall）。`FRESH_EXPERT_ROUTED` 在 generator。`.bak.a1pre`/`.bak.a1pre2` 备份。8 个 hold-mode exporter 仍 READY（`--reuse-exporters`）。



## ⭐⭐ 本 session（2026-07-13）— M3 NaN 根因找到并修复；前 8 次 UPDATE 的 MoE-INT8 判断是错的

**M3 NaN 真根因 = `attn_only_orch` 的 Out 未写回（不是 MoE/INT8/gap-5）。**
- `full_attn_only_orch`/`swa_attn_only_orch`（generator `_gen_faithful_real.py` FRESH_*_ATTN_ONLY）里 `resid3_out = pl.create_tensor(...)` **遮蔽了 `resid3_out` pl.Out 参数** → attention 写进新建的局部 tensor，`h_mid_out[rd]` **从没被写** → base `chip_orch` 读到未初始化的 h_mid → NaN。dummy-0 权重把它掩盖了（所以 dummy 跑 DISPATCH_CLEAN；真权重才暴露）。
- **修复**（已落 real-builder 两个 orch 定义 + generator）：把 attention 结果 `pl.assemble` 进 `resid3_out` Out（`with pl.at(CORE_GROUP): for _co: resid3_out = pl.assemble(resid3_out, slice(resid_a,...))`）。**device 验证 P=1：NaN→finite，`h_mid 0.0→448`，`next_hidden nan→1.05e12`，logits/argmax 有限。**
- **定位手法**（有用，记下来）：harness 打印 `max|h_mid|`（attn_only 输出，chip_orch 不覆写它）+ 编译期 `P_MOE_PASSTHROUGH` 旋钮（chip_orch 直接返回它的 `resid1`=h_mid 输入）。`h_mid`(harness)=448 但 `PASSTHROUGH`(chip_orch 读到的)=6.31e11 → **chip_orch 读到的 h_mid ≠ attn_only 写的**。

**M3b 新阻塞 = MoE 输出幅值 1e11–1e12（不是 INT8 数学问题）。**
- NaN 修完后 P=1 `next_hidden`≈1e11–1e12（有限但巨大），argmax≠303。
- bisect 旋钮（都是 device-if，已证会 runtime-select）：`P_MOE_PASSTHROUGH`(返回 resid1)=6.31e11；`P_MOE_NORM_ONLY`(返回 post_norm)=4.23e11；`P_MOE_SHARED_ONLY`(resid+shared)=2.86e11。→ chip_orch 读到 huge h_mid 输入 → 经残差加流进 next_hidden + 压垮 rmsnorm。幅值爆炸对 shared(BF16)+routed(INT8) **共同** → 是被坏输入带的，**不是 MoE compute**。
- **把 attn_only+chip_orch 的两个 per-rank 循环合成一个循环（pos=0）→ 幅值仍 1.99e11 未修** → 是 **buffer 别名**，不是循环顺序。`h_mid_out`/`next_hidden_out` 两 buffer 乒乓被全部 45 层的 orch 复用 → chip_orch 读到被别名的 huge 值。这正是 memory `n1_whole_net_scheduler_timeout_fixed_perlayer_windows`（per-layer distinct buffers）的领域；文档之前"aliasing REFUTED"只覆盖了 3 层 dense P=0 乒乓，**从没测过 MoE 层内 attn_only→chip_orch 这段 handoff**。
- **下 session M3b 修法**：给 attn_only→chip_orch 的 handoff **一个专用 buffer**（不复用共享 h_mid_out；per-MoE-layer distinct，或至少破掉别名）让 chip_orch 读到 attn_only 的真实输出。可能需要 DCE 的 `CommDomainScopeStmt` 修复（见 memory）。修完 P=1 幅值→合理 → 放开 42 层 → L1 argmax→303。

**代码状态**（pypto-lib `feat/whole-net-n1-fusion` 工作区，**未 commit**，在 NFS 0234）：
- **保留**（真修复）：attn_only writeback（两个 orch 定义 + generator FRESH_*_ATTN_ONLY）；in-expert INT8 quant padding-mask（`_expert_routed` amax/quant 循环把未初始化的 RECV_TILE padding 行 mask 到 tile_valid，两份 copy —— 正确且对 partial tile 有用，但**不是** NaN 根因）。harness `_stage_whole_faithful_real_ipc.py` +1 行 `max|h_mid|` 打印（保留）。
- **commit 前要清**（诊断脚手架，默认 0 inert）：3 个 module global + `if _MOE_{SHARED_ONLY,PASSTHROUGH,NORM_ONLY} > 0:` gated 提前 return（在 10 份 chip_orch copy 里各一个）；host_orch pos=0 的 merged-loop（中性，没修好）。
- 8 个 hold-mode exporter 之前在 0234 上跑（真 W8A8 + KV IPC）。

### M3b 深入（续）— 三个 ordering 修法全失败 → 是 pypto 跨-orch 依赖墙；正解 = FUSE

- 追加验证：**chip_orch 在 attn_only 写 h_mid_out 之前就读它**（读到未初始化内存 → 跨 run 不确定：NaN 或 ~1e11-1e12；h_mid 也在 448-504 跳）。三个 ordering 修法**都没让 chip_orch 读到 attn_only 的真实输出**：
  1. 合并两个 `for r` 循环成一个 → 仍 1.99e11。
  2. 捕获 attn_only 返回值传给 chip_orch 当 current_hidden（强制 data-dep）→ PASSTHROUGH 仍 3.05e11（harness h_mid=394）。
  3. attn_only 改成把 Out 直接传给 inline attention（对齐 dense 的 `dense_mlp_inline(h0_out)` 写法，"C1"）→ 回退成 `next_hidden=nan`。
- **对比**：dense 层内 attention+MLP 在**一个** `swa/full_chip_orch` 里做（无层内 split），所以 dense L1→L2 的 h_mid handoff 正常（P=0→502）。MoE 层被拆成 attn_only+chip_orch 两个 orch，跨-orch 依赖 pypto 没排上序。**这正是 SKILL H 说的 N=1 inline 墙**（"whole-decode worker 用 multi-program + resident DeviceTensor，不 inline 45 层 body"）。
- **正解（下 session）= FUSE：把 attention 塞进 MoE chip_orch**（chip_orch 对 current_hidden 调 attention_inline → resid1 局部 → post_norm+MoE；镜像 dense full_chip_orch），消掉层内 handoff。需两个 fused 变体（full/swa，因 11 full + 31 swa MoE 层）+ generator 重写 + regen；有编译墙风险（attention+全 MoE 一个 orch 可能过大）。**或**转 SKILL 推荐的 multi-program resident-DeviceTensor（非本 N=1 track）。
- **当前最佳工作态 = attn_only assemble-writeback 修复（finite ~1e12，NaN 已消）；C1 已回退。** decode_layer.py 工作区仍带诊断旋钮 + merged-loop/capture-pass pos=0，commit 前清。

### ⭐⭐ 本 session（2026-07-13 续2）— FUSE 落地 + device 复诊：推翻 M3b「handoff/resid1」定位，真相是 MoE 层 nondeterministic ~1e11 + FUSE 引入 intra-orch 别名

> 详见 memory `n1_m3b_fuse_handoff_fixed_residual_clobbered`。本段以 device 实测为准，**覆盖上面 M3/M3b/UPDATE1-10 的定位**（多为 truncated-orch 或不可靠旋钮所致的误判）。

- **实现了 FUSE（正解路径 1）**：generator `tools/step3p5/_gen_faithful_real.py` 把 attention 折进 MoE orch（`full_moe_chip_orch`/`swa_moe_chip_orch` = `attention_{full,swa}_inline→resid1` 局部 + MoE body B/C/D），`_host_orch` 层间 `next_hidden`↔`h_mid` ping-pong + parity lm_head。**P=1 与 P=42 都 a2a3sim COMPILE OK（无编译墙）**——推翻文档「FUSE 可能过大撞编译墙」的担忧（orchestration 级融合，InCore kernel 不变，buffer 不爆）。
- **可靠 device 基线（host 级 `P_FAITHFUL_MOE_LAYERS` gate，可信）**：
  - **P=0（3 dense 层）CLEAN + 确定**：`next_hidden=448 h_mid=294 argmax=27527`。dense 路径（含 on-device head-gate）正确。
  - **P=1（+1 MoE 层）nondeterministic ~1e11–1e12**（跨 run 1.7e11/4.5e11/3.2e11/2.9e11）。MoE 层是 garbage 源。
- **FUSE 有问题（SKILL 应验）**：complete orch 上连 `P_FUSE_ATTN_ONLY`（只 attention 提前 return）都 =1.39e12，而 dense attention（L1/L2 同 `attention_swa_inline`）干净 → **MoE InCore kernels 无视 Python 提前 return 仍被排进 DAG，经 intra-orch 别名污染 attention/输出 buffer**。正是 SKILL §H「别把 TP-attn+EP-MoE 塞一个 chip_orch」。**强证据 FUSE 是错方向。**
- **两个陷阱（记下勿再踩）**：
  1. **generator `_be` truncation bug（已修）**：`GEN_STRIP_KNOBS=0` 保留旋钮时，`index("            return next_hidden_out\n")` 会 substring-命中 16-空格的 `NORM_ONLY` return（offset 4）而非 D 的 12-空格 return → moe_body 被截断在 NORM_ONLY，丢掉 shared/dispatch/routed/combine/D。**我这 session 所有带旋钮的 bisect（attn=502/norm=1.42/shared=0）跑在 truncated 坏 orch 上 → 全部作废。** 已改用 `name_hint="moe_residual_add"` 锚定。
  2. **orch 内 `if X>0: return` 旋钮不可靠**：pypto orchestration 仍会排掉 return 之后的 InCore kernels（建整张 DAG），所以 `P_FUSE_ATTN_ONLY`/`P_MOE_NORM_ONLY`/`P_MOE_SHARED_ONLY` 无法干净隔离 stage。**只有 host 级 `P_FAITHFUL_MOE_LAYERS`（`if pos<N` 门整层）可信。**
- **本 session 试过但没解决 ~1e11 的两个修复**（原理可能仍对，但非充分）：`_zero_routed_y_buf`（InCore，对齐 moe.py，data window 非自动清零 → combine gather 读 garbage；保留）；residual-protection（stash `resid1→next_hidden_out` Out + D 读它，基于已作废的 resid1-corruption 说，**回退候选**）。
- **代码状态**：`feat/whole-net-n1-fusion` 工作区**未 commit**，含 FUSE + generator `_be` 修复 + zero-init + residual-stash + 诊断旋钮。**FUSE 状态存疑，未 push。**

### ⭐⭐⭐ 本 session（2026-07-13 续4）— E2 op-dump 定位真因 = INT8 routed-expert kernel；FUSE 无罪，架构约束未违反

> 详见 memory `n1_m3b_fuse_handoff_fixed_residual_clobbered` UPDATE续4。**覆盖上面所有 FUSE-别名 / 回退-split 的结论。**

- **架构核查（`pypto_top_level_documents/`）**：FUSE（attn+EP-MoE 一个 orch）**不违反任何硬不变量**。「不 fuse」是软选择（DeepSeek 的）；「live 中间量不别名」是**编译器/`IMemoryManager` 的保证**（ADR-013）。⟹ ~1e11 是**实现 bug**，用户判断对。
- **E1（填满 batch 真输入）**：仍 nondeterministic 1e11 → **input-agnostic**（排除 degenerate 零行 / INT8-amax-除 0）。
- **heap 4→8GB**：无变化 → 非 GM ring-heap 溢出。
- **E2（可靠 op 级 `pl.Out` dump，写进独立 `dbg_out` buffer，非 orch 内 return 旋钮）device P=1**：

  | stage | 内容 | max\|dbg\| |
  |---|---|---|
  | 1 post_norm | MoE 输入 | **1.45 ✓** |
  | 5 local_routed_x | dispatch 输出=routed 输入 | **1.45 ✓** |
  | 3 local_routed_y | **routed expert 输出** | **3.99e11 ✗** |
  | 2 sh_y | shared expert | **1.62 ✓** |
  | 4 moe_out | combine 后 | 1.95e11（继承 routed garbage） |

- **结论（device 证据、可靠）**：~1e11 **完全出自 `_expert_routed`（INT8 routed-expert 计算）**。post_norm / dispatch 输出 / shared 全干净。
  - **推翻 FUSE-别名**（fused orch 中间量全干净）→ **FUSE 可行，别回退别改 split**。
  - **推翻 collective**（dispatch 输出干净）。
  - **推翻 degenerate 输入**（E1）。
- **真因 = gap-5**：whole-net 内联的是**旧 in-expert INT8 quant**（`routed_x_quant`/`routed_h_quant`：per-token amax→INT8 cast→INT8 cube→INT32→dequant），**device 误编译**（nondeterministic 1e11）；moe.py 有**已验证的 dispatch-side quant**（Option A）。对应 memory `n1_w8a8_int8_kernel_done_wholenet_inlined_decoupled`(A5) + `gap5_int8_math_correct_pivot_dispatch_side` + `gap5_int8_cube_fractal_32_partial_tile`。

### ⛔ 修正后的下一步（按优先级）

1. **[真修，memory「A5」] 把 whole-net 的 routed 路径对齐 moe.py 的 dispatch-side INT8 quant**：post_norm→INT8 在 gate/dispatch 做，dispatch INT8 `recv_x` + per-token scale，`_expert_routed` 去掉 in-expert input-quant。从 moe.py 当前 `EpTpMoE` 重生 whole-net 的 gate/dispatch/expert_routed step（generator 重生）。**保留 FUSE**。修完 P=1 routed 输出应 ~O(1) → 放开 42 层 → L1 A/B（tid 6127 → argmax 303）。
2. commit 前清诊断脚手架（`_MOE_*`/`_FUSE_ATTN_ONLY`/`_DBG_STAGE` module ints + fused-orch dump 块 + `dbg_out` plumbing + harness `P_FILL_BATCH`/`max|dbg|` + residual-stash）；保留 generator `_be` 修复 + `_zero_routed_y_buf`。
3. 权重/scale 前 session 已验有限，非源头。





## ⛔ 用户硬约束（不可违背，勿走弯路）

- **必须用 IPC 共享显存机制**做端到端，**KV cache 和权重都走 IPC**。**不许 H2D 绕路**、不许换非-IPC 方案。（权重+KV IPC 已 device 跑通。）
- **必须用真实权重加载**（真 W8A8 checkpoint，非 dummy）。**真权重调试，不走其他弯路。**
- 遇到问题只能**解决它**，不能绕开（work-around）。诊断脚手架只能定位、不能进产品路径。
- **correctness 和 speed 都要**：既要跑出正确结果、也要推进到底完成目标；别用"correctness"当借口停在半路，也别为"快"造出错误的精度数字。
- **不能只盯一个子目标**——总目标是整网 decode 集成 + 端到端精度对齐；单步（如 MoE NaN）修完要立刻推进下一里程碑。
- **对齐 DeepSeek/Qwen**：遇问题先看 DeepSeek v4/Qwen 实现 + 历史开发文档，尽量对齐；step3p5-vs-DeepSeek 差异必须论证（只在"性能更好"时保留）。
- **架构优先**：coding 前先系统分析 + 整体设计。**严格遵守 SKILL.md**（`pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md`）；不满足约束可能是设计不合理需重设计。
- **⚠ 历史文档可能 stale，先核对当前代码再下结论**。

## ✅ 已解决并推送（pypto-lib `f07da3b`）

- **M1 Blocker B（gate_topk mrgsort）+ 权重+KV 双 IPC 8 卡 device-clean**（`4bede85`/`c61046b`/`b92031f`）。gate_topk 修复对齐 DeepSeek format1 链，device 数值验证 PASS（vs torch.topk）。
- **M2 整网 per-layer gate_r 结构性阻塞已解 —— on-device head-gate（路径 a）**：
  - `matmul_acc N=16` 丢 K 累加的 codegen bug（当年把 head-gate 移 worker 的原因）**现栈已修**（`_probe_matmul_acc_n16` + full-chain `_probe_head_gate_full` device PASS）。
  - `attention_full.py`/`attention_swa.py` **Scope 1.f 恢复 on-device gate**：`gate_logits = normed_all @ w_g`（K-chunk matmul_acc N=16）→ `sigmoid` → `gate_exp = gate_score @ R`（N-chunk）；Scope 3.a o_proj 乘 `gate_exp`（2 scope 分离控 UB）。
  - `gate_r` 槽改承载 **block-diag R 常量**（`R[h,h*HEAD_DIM+d]=1` 实头；**layer-independent → 喂一次全 45 层通用**），harness 填 R（实头=HQ//HEAD_DIM：full=8/swa=12）。→ monolithic 整网每层从自己 `normed_all` 自算 gate → **token-exact-capable**（不再需 per-layer dispatch / resident-DeviceTensor）。
  - 对齐 vLLM `modeling_step3p5` L489（g_proj on **post-input-layernorm** hidden，非 raw）+ L527-531。`whole_decode_faithful_real` **TP=8 COMPILE OK**（attention inline 从 `._func` 重导，无需 regen）。
  - memory `n1_head_gate_ondevice_restored_l1_nan` / `step3p5_head_gate_uses_normed_hidden`。

## ⛔ M3 当前阻塞：单层 INT8 W8A8 routed-MoE 的**有效行**计算产生 NaN

> ⚠ **本节已 SUPERSEDED（2026-07-13）**：M3 NaN 已修，真根因是 `attn_only_orch` 的 Out 未写回（非 INT8/gap-5）——见顶部「本 session（2026-07-13）」段。本节保留作历史 bisect 记录，勿再据此往 INT8 方向查。当前阻塞是 M3b（幅值）。

**L1 ctx=1 A/B 现状**：pypto worker `--hidden-token 6127 --kv-ipc` **RUN_CLEAN ~3.6s 但 `next_hidden=nan / logits=nan / argmax=0`**（vLLM golden：tid=6127「北京」→ next=**303**「，」）。

**已完成的 bisect（旋钮 `P_FAITHFUL_MOE_LAYERS`，`decode_layer.py:19182`，默认 42，emit N 层 MoE）**：
- `P_FAITHFUL_MOE_LAYERS=0`（仅 3 dense/swa attention 层）→ **FINITE**（`next_hidden=502.0 / logits=9.03 / argmax=27527`）→ **attention 路径 + on-device head-gate 确认干净**。
- `P_FAITHFUL_MOE_LAYERS=1`（3 attention + **1 MoE**）→ **NaN** → **单个 INT8 MoE 层就复现**，非跨层累积。MoE 输入 post_attention_layernorm 归一化（O(1)，非 502），不是输入幅度问题。

**已排除**：
- **不是 gate/head-gate**（bisect MoE=0 干净）。
- **不是 A-operand fractal-32 padding**（gap-5 经典坑）：试过把 `routed_x_quant`/`routed_h_quant` 的 `x_i8`/`h_i8` padding 行 [tile_valid:RECV_TILE] 用 `fillpad(set_validshape(cast_tile,tile_valid,K), PadValue.zero)` 置零（对 cast **TILE**，非 tensor —— set_validshape 只吃 TileType）→ 编译+跑通**仍 NaN** → 已 revert（tree clean f07da3b）。amax 是 per-row（`row_max` over K → `[1,RECV_TILE]`），有效行 scale 干净，padding 置零证实无效。
- **不是 stale 权重格式**：exporter `int8_routed=True` → INT8 pool 25.35GiB；`w_g` padding zero-pad（`weight_loader._slice_g_proj` L594）；KV pool `torch.zeros`（`pypto_weight_ipc.export_from_checkpoint` L394）。

**关键结构事实**：whole-net 内联 MoE 是 **INT8-native in-expert quant**（`decode_layer.py` `routed_x_quant`/`routed_h_quant` scope：`xe_amax→scale→cast INT8 trunc→matmul out_dtype=INT32→dequant col/row_expand_mul`），`_quant_moe_input` 次数 **decode_layer.py=0 vs moe.py=2** —— 即 whole-net 内联 MoE 与 **standalone-validated `moe.py` INT8 kernel（dispatch-side quant, Option A）DECOUPLED**（旧 in-expert 路子）。只有 **2** 处 `routed_x_quant`（base + real builder），routed expert body 是共享 inline（**非 42 份**）→ 定位/修改是**单点**。

**剩余嫌疑（有效行计算，按可疑度）**：
1. **routed INT8 gate/up/down dequant**：`gate_2d = col_expand_mul(row_expand_mul(cast(gate_acc,FP32), x_scale_dq), wg_scale_row)`（`decode_layer.py:~20113`）—— 查 IPC 里 `moe_w_gate_r_scale`/`moe_w_up_r_scale`/`moe_w_down_r_scale` 是否含 0/inf/NaN（weight_loader dequant / exporter INT8 scale 逻辑）。
2. **combine** routing weight × routed_y + shared。
3. **shared expert**（BF16 swiglu；layer-0=full_moe_silu_silu 用 silu）。
4. **dispatch a2a**（comm windows / CSR）。

**M3 攻克手法（下 session，用户已批准集中攻）**：
- **手法 A（推荐，最快隔离 权重 vs 代码）**：把真 IPC INT8 routed 权重+scale 喂进 **standalone `moe.py` MoE-block harness**（已 device-validated 的 kernel）跑单 MoE block。standalone **finite** → whole-net 内联副本 bug（→ 手法 C）；standalone **也 NaN** → 权重/scale bug（→ 查 weight_loader/exporter INT8 scale dequant）。
- **手法 B（per-op dump 仪表化）**：real builder routed expert 加中间 Out（gate_2d / h_bf16 / routed_y / shared_out）经 host_orch + harness 拉回，逐 stage 看谁先 NaN。monolithic 无中间输出，必须显式加 Out。
- **手法 C（根治，A5 大改）**：把 `moe.py` 的 `_quant_moe_input` + dispatch-side INT8 recv（Option A，已 validated）经 `tools/step3p5/_gen_faithful_real.py` regen 应用到 whole-net 内联 MoE（替换 decoupled 旧副本）。工作量大但根治。
- **收尾即推进 M4**：修完 `P_FAITHFUL_MOE_LAYERS=1` finite → 逐步放开 2/4/…/42 → 全量 L1 A/B（tid 6127 期望 **argmax=303**）→ 进 M5（多 token / KV bridge / live A/B）。

### ⭐ 本 session 更新（2026-07-12 续）— NaN 可靠定位到 MoE，attention 干净；两个 confound 已记

- **数据全排除**：36288 个 routed INT8 `*_scale` 全有限（`tests/step3p5/_diag_check_w8a8_scales.py`，min1.6e-4/max1.5e-2）；shared expert 权重确为 BF16（ckpt 无 `_scale`）且有限；layer-3 attn q/k/v/o_proj 也是 BF16 有限（与有限的 layer-1 同构）。
- **决定性编译期测试**：新增 host-orch 编译期开关 `P_L3_ATTN_ONLY`（跳过 L3 的 MoE `chip_orch`，把 attention 输出 `h_mid` 直接喂 lm_head）→ `logits=0.0000 argmax=0` **有限、无 NaN** → **attention 输出有限，NaN 由 MoE(chip_orch) 引入**（重新确认原 INT8-MoE 定位）。
- **⚠ 两个 confound（勿再踩）**：
  1. pypto 在 `@pl.function` 体内对 module-int 的 `if` 会变成 **device-if（两分支都 trace）** → combine 旁路开关 `P_MOE_ROUTED_OFF`/`P_MOE_SHARED_OFF`/`P_MOE_OUT_OFF` **全不可靠**（都误报 NaN）。**可靠 gating = host-orch 层比较式 `if`**（像 `_FAITHFUL_MOE_LAYERS` 的 `if X>0:`），且**不能用三元 IfExp**（被 reject）。
  2. `chip_orch` 读且**覆写 `h_mid_out`** → MoE 一跑就读不到干净 attention 输出，必须编译期跳过 MoE 才能读。`h_mid` 确实 copy-back（P=0 显示 294）。
- **次要异常（可能是触发点）**：L3 `h_mid≈0.0000`（本应 ~502，attention_swa L802-808 确有残差加法），且"L2 输出"幅值跨 run 不一致（P=0→448 vs _L3_ATTN_ONLY→502）→ 疑 **整网 buffer 数据流/aliasing** 把退化的 ~0 输入喂给 MoE，INT8 MoE 在其上出 NaN。
- **代码状态**：诊断脚手架（decode_layer.py 616 行 device-if 开关）**已 revert 回 f07da3b clean**；保留 `_diag_check_w8a8_scales.py` + harness 1 行 `h_mid` 上报。
- **下一步（修复，不再重新定位）**：① 可靠子阶段隔离——把 `sh_y`/`routed_y` 作 **copied-back Out** 串出 chip_orch→host_orch→harness（**不用 device-if**）确认 shared vs routed；或 ② 直接上 gap-5 根治方案 **dispatch-side INT8 quant**（对齐 moe.py `_quant_moe_input`）；③ 并查 h_mid≈0 / buffer 幅值异常（可能才是真触发点）。重建 `P_L3_ATTN_ONLY` 记法见 memory `n1_head_gate_ondevice_restored_l1_nan` UPDATE5。

### ⭐ 本 session 收尾（2026-07-12 续2）— 根因假设=hidden-state 乒乓 SSA 别名；per-layer 修复撞 pypto DCE 墙

- **根因假设（对齐先例 `n1_whole_net_scheduler_timeout_fixed_perlayer_windows`）**：整网 `whole_decode_faithful_real` 把 hidden state 在**仅 2 个 `pl.Out`**（`next_hidden_out`/`h_mid_out`）之间乒乓、跨 45 层各写 ~44 次 → 与 comm-window 别名同一类 P3/ADR-013 违反，但发生在 hidden state。真权重下 L3-attn 读到 stale SSA 版本 → `h_mid≈0` → MoE 退化输入 → NaN；dummy 全零权重时不可见（所以之前 DISPATCH_CLEAN 没发现）。
- **修复尝试 + 受阻**：改 `_gen_faithful_real.py` `_host_orch`，给每个 sub-orch 输出独立 slot（先 4D `hchain[tp,87,B,H]`，后 3D-flattened `hchain[tp*87,B,H]` 用 `hchain[r*87+slot]`）。**两种都编译失败** `pypto.InternalError: Unhandled ScopeStmt subtype in DCE: CommDomainScopeStmt`（P=1 和 P=42 都失败）。根因：**单个 pl.Out 参数在 RUNTIME 偏移 `r*87+slot` 处写入** → pypto DCE 无法静态匹配写/读偏移 → 认定带 comm-domain 的写是 dead → DCE 其 `CommDomainScopeStmt`（不支持）。旧乒乓能编只因偏移是裸 `[r]`（可静态匹配）。
- **唯一 DCE-safe 的 per-layer 形式** = **N 个独立 3D `pl.Out` 参数，各在 `[r]` 处写一次**（像旧的 2 个但要 ~87 个）——大改、有风险（87 参数签名 + harness alloc + lm_head 的 slot 依 `_FAITHFUL` 需编译期 if/elif，因为把 Out 参数别名到局部 `_lm_in` 会被 SSA reject）。
- **代码状态**：generator + decode_layer + harness **全 revert 回 clean f07da3b**（`git diff` 空）。**别名假设仍未证实**（所有单参数修复都编不过）。
- **下 session（三选一，建议先做最便宜的确认）**：
  - **(最便宜确认)** 只给 **L2→L3-attn 这一处**加 1 个独立 3D 参数 `hs_test`（L2 写 hs_test、L3-attn 读 hs_test），P=1 跑；若 `next_hidden` 变 finite → 别名确认 → 再做 87-参数全量版。（DCE-safe。⚠ 直接改生成后的 decode_layer.py 有大量重复签名/调用，建议改 generator 再 regen。）
  - **(全量根治)** N 个独立 3D 参数（generator 生成 87 个 + harness）。
  - **(上游)** 修 pypto DCE pass 让它处理 `CommDomainScopeStmt`（pypto core，较深）。
  - 若最终证实**不是**别名 → 转查 attention-compute 或 KV 喂给 MoE 的路径。
- 详见 memory `n1_head_gate_ondevice_restored_l1_nan` UPDATE6/UPDATE7 + `feedback_no_over_localize_just_fix`。

### ⭐ 本 session 收尾（2026-07-12 续3）— 两假设 REFUTED（gate + aliasing）；pypto DCE bug 已修；嫌疑收窄

用户建议下用两个 agent（DCE 排查 + 反向质疑）从别的角度看，结果：
- **GATE REFUTED**：加编译期 `P_GATE_BYPASS`（whole-net `gate_step` 写死路由 experts0-7/均权，绕过真 `_gate` sort/mrgsort），device P=1 **仍 `logits=nan`** → 真 gate 不是 NaN 源（反向 agent 的头号嫌疑被否）。（编译期 `if X>0:` device-if 确实 runtime-select 正常，gate bypass 跑了。）
- **ALIASING REFUTED**（反向 agent）：P=0 用**同一个 2-buffer 乒乓** L0→L1→L2 却 finite（argmax 27527）；若乒乓别名 dense 层就会暴露。`h_mid=0` 也不是别名的症状（别名给 stale ~502）。**放弃 87-参数 buffer 重写方案。**
- **pypto DCE bug 已定位+修复（独立价值）**：`pypto/src/ir/transforms/utils/dead_code_elimination.cpp` 两处 ScopeStmt-rebuild dispatch 缺 `CommDomainScopeStmt` case（~L290 + ~L462）→ `Unhandled ScopeStmt subtype in DCE`。已补（仿 RuntimeScopeStmt）。**未 commit、未 rebuild（inert）**，值得 commit+上游（解锁任何 computed-offset Out 写进 comm-domain orch）。
- **NaN 是真的**（logits=nan），在 `chip_orch` 内、gate 下游 → **dispatch / routed-INT8 experts / shared-BF16 / combine**。standalone moe.py MoE-block 曾 device-验证 PASS 但用 `--bypass-gate` + 合成输入 + H2D 权重；whole-net 差在真 attention-输出输入 + IPC INT8 权重。
- **待破的矛盾**：`P_L3_ATTN_ONLY`（跳 chip_orch、lm_head 读 attn 输出）→ `logits=0.0000`（finite 且为 0）→ attention 输出 ≈ 0（device-可靠，经 lm_head 非 host readback）。但解析上 MoE(0-输入)=0（routed amax floor 1e-4 / shared BF16 / combine 求和）应 finite 0 而非 NaN。⇒ 要么 (a) attention 确实输出 ~0（zeroed-KV ctx=1；dense L1/L2 非零只因 dense-MLP 加幅值，attention 部分本就 ~0），MoE(~0) 撞真 codegen/div NaN（rsqrt(0) in norm，或 in-kernel INT8 quant partial-tile gap-5）；要么 (b) attention 输出是非零 garbage 而 h_mid=0 仍误导。
- **下一步**：gate bypass 下（已证 device-if 可靠）重跑 routed-off / shared-off 干净拆分 shared-BF16 vs routed-INT8；若 routed → 攻 gap-5 in-kernel INT8 quant（`routed_x_quant`/`routed_h_quant`）。detail: memory `n1_head_gate_ondevice_restored_l1_nan` UPDATE8。

## 🎯 精度对齐三档（对应 M1/M4/M5）

- **L0 单算子 probe（M0 ✅）**：`_probe_matmul_acc_n16`、`_probe_head_gate_full`、`_probe_gate_sort` 全 PASS。范式见 🐞。
- **L1 ctx=1 单 token A/B（M4，脚手架已建可跑）**：`_stage_whole_faithful_real_ipc.py --hidden-token <id>` 把 `embed(token)` 灌进 `current_hidden` row0 + pos-0 identity rope（cos=1/sin=0），seq_lens=ones（16 行全 ctx=1，避开 seq_len=0 NaN）。vLLM 1-token prompt 首 token = argmax(logits(pos0))，等价 ctx=1 self-attn，**不需 KV bridge**。**当前卡 M3 MoE NaN**，修完即通。
- **L2 整网多 token / decode-step golden（M5，终极）**：多 token 需 vLLM→whole-net KV bridge（分页池→整网 flat KV，memory `g5b_kv_bridge_not_pure_reshape` / `g5b_kv_is_bf16_not_int8` / `g5b_import_ipc_facade_missing`）或 live A/B（8001 pypto vs 8000 vanilla，co-tenancy `SIMPLER_COMM_NO_HCCL=1`，memory `project_g4_cotenancy_hccl_conflict`）。**这套是 G5b track 机器（0162 working tree），本分支需 port。**

## 🖥 环境 / vLLM oracle 启动（本 session 验证可用）

- **三件套激活**（每 fresh shell，`activate.sh` 不带 CANN env）：
  `source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh && export PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib`（`WS=/data/chensiyu/hw_project/pypto/workspace`）。
- **vLLM W8A8 oracle（0234 可跑）**：**⚠ 先 `unset PYTHONPATH` 再 `source cann/set_env.sh`**（顺序关键——先 source 再 unset 会把 acl 抹掉 → `ModuleNotFoundError: acl`），**不 export pypto PYTHONPATH**，再：
  `vllm serve <W8A8ckpt> --served-model-name step3p5 --trust-remote-code --quantization ascend --tensor-parallel-size 8 --enable-expert-parallel --enforce-eager --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85`
  → 占 8 卡 0-7，~5-6min load。
  - **⚠⚠ vLLM 跑在 0234，查询必须在 tmux 0234 里 `curl localhost:8000`**。**从 b-csy-develop 的 Bash `curl localhost:8000` 命中的是它自己的 nginx → 404**（本 session 踩过，别误判 vLLM 挂了）。
  - 1-token golden：`_l1_ab_vllm.py --word 北京`（取首 token 作 tid，查 vLLM greedy next-token id+text）。本 session：tid=6127→303。
  - vLLM 与 pypto **同卡** → 先 vLLM 出 golden 再 kill 腾卡跑 pypto（offline A/B）。kill：`pkill -f "[v]llm serve"; pkill -9 -f "[E]ngineCore"`。
- **W8A8 ckpt** = `/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`（arch `Step3p5ForCausalLM`，45 层，embed 在 shard 00048 非量化）。
- **8 卡 pypto env**：`PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
- **exporters（IPC 权重/KV）—— bisect 高效设置**：默认 worker mode 自己 launch 8 exporter + 跑完写 STOP（每次 15min 冷载）。**多次 bisect 用 hold-mode**：先手动起 8 个常驻 `--export-rank r --dev r --kv-ipc`（一次 15min 冷载，写 `ready.rank{r}`+`pypto_weight.key.rank{r}` 后 hold 等 STOP），再多次 `--reuse-exporters` worker 秒级 attach（每次只 compile+run ~5min）。收尾 `touch /tmp/n1_weight_ipc/STOP` 或 `pkill -f export-rank` 释放卡。
  - **⚠ `/tmp` 每机独立**：exporter 写 0234 的 `/tmp/n1_weight_ipc/`，**从 b-csy-develop `ls /tmp/...` 看不到**（本 session 误以为 0 ready）。查 ready keys 必须在 **tmux 0234**：`ls /tmp/n1_weight_ipc/ready.rank* | wc -l`。

## 🐞 Debug 方式（累积经验）

- **数字 device error 先查 [wiki Device-Error-Codes_zh](https://github.com/hw-native-sys/simpler/wiki/Device-Error-Codes_zh)**。`507018` 是泛化 host 码，看 `orch_error_code`/`sched_error_code`/`sub_class` 定真因。
- **device stall 快照**（定位卡住 kernel/核，本 session 定位 gate_topk 用过）：harness `logging.getLogger("simpler").setLevel(15)` + `export ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_PROCESS_LOG_PATH=<预建目录>` → 读 `<dir>/debug/device-*/device-*.log` 的 `log_stall_diagnostics`（`TASK state=RUNNING kernels=[...] running_on=[core=X]`）。`orch_error=8`=TENSOR_WAIT_TIMEOUT，`sched=100`=running-stalled。root-owned，tmux(root) grep。
- **NaN/精度 bisect（本 session 定位 MoE 用的）**：`P_FAITHFUL_MOE_LAYERS=N` 层数二分（0=纯 attention / 1=单 MoE / …/42）隔离 attention-vs-MoE、单层-vs-累积。monolithic 无中间输出 → 要 op 级须显式加 Out dump（手法 B）。
- **单算子精度 probe 范式**（隔离验证）：module-level `@pl.jit` + `from golden import TensorSpec, run_jit, ratio_allclose` + torch golden_fn（见 `_probe_head_gate_full.py`/`_probe_matmul_acc_n16.py`）。⚠ `@pl.jit` 的 shape/const 必须 module-level；per-K-chunk cast 避免 `[T,HIDDEN]` FP32 UB overflow（256KB>184KB）；2-scope 分离让 K-loop buffer 先释放再做 N-chunk expand；小 N=16 输出 matmul_acc 现栈已正确累加 K。
- **launch 前**：`pkill -f '[_]stage_whole'` + `rm -f /tmp/n1_weight_ipc/{STOP,ready.rank*,pypto_weight.*}`（在 0234）；**禁 `-9` 强杀 device 进程 / `npu-smi set -t reset`**（netboot 机锁死全卡）；`npu-smi info -t usages -i <c>` 确认 HBM<10%。stale pyc：改 models 后 `find models/step3p5 -name '*.py' -exec touch {} +`。
- **每次 device run 慢**：exporter 全 ckpt load ~15min（用 hold+reuse 省重复冷载）；compile 42 层 ~4min，少层更快。

## ⭐⭐ 铁律（勿再踩）

1. **单卡 ST/UT shape**：`apply_perrank_patch()`（保 TP=8 per-rank slice），不用 `apply_tp1_patch()`。gate_matmul 单卡 unsliced 会 Mat/Vec 溢出——验 gate 用隔离 probe。
2. **gap-5 坑**：in-kernel `pl.cast(bf16/fp32,INT8)` 喂 cube 可能静默错；照抄 DeepSeek cast 链 + create_tensor 位置 + scope。**但本 track 的 MoE NaN 已证不是 A-operand padding**（经典 gap-5 fix 试过无效），是有效行计算。
3. **ccec ND2ND**：scale slice 必须 contiguous row-slice + reshape；a2a3sim compile 过 ≠ device ccec-clean（必须真 device 跑）。
4. **push**：PAT `/data/chensiyu/secrets/github.env` + `git -c http.version=HTTP/1.1`，输出屏蔽 token。**github 从 0234 连不通**（直连 130s 超时；proxy `deploy.i.shaipower.com/httpproxy` 返回的 proxy 需 auth → 407）；**从 b-csy-develop 的 Bash 能直连 github**（NFS 共享同一 repo；commit 在 tmux(root) 做完后 `git push` 从 b-csy-develop 跑）。pypto-lib `.git/objects` root-owned → commit 走 tmux(root)，push 可从 b-csy-develop（读 objects OK）。pypto-project chensiyu-owned 可直接。跨仓 push 同步 STATUS pin。
5. **文档 stale 风险**：核对当前代码再下结论（head-gate 就栽在 stale 注释）。
6. **别只盯单步**：M3 修完立刻推进 M4→M5→M6，总目标是整网 decode 集成 + 端到端精度对齐。

## 本 session commits

pypto-lib `feat/whole-net-n1-fusion`：**未 commit**，工作区含真修复 + 诊断脚手架。
- **真修复（保留）**：`attn_only_orch`(full+swa) 把 attention 结果 assemble 进 `resid3_out` Out（修 NaN，两 orch 定义 + generator `_gen_faithful_real.py` FRESH_*_ATTN_ONLY）；`_expert_routed` in-expert INT8 quant 的 amax/quant 循环把未初始化 RECV_TILE padding 行 mask 到 tile_valid（对 partial tile 正确，非 NaN 根因）；harness `_stage_whole_faithful_real_ipc.py` +1 行 `max|h_mid|` 打印。
- **诊断脚手架（commit 前清）**：3 个 module global + `if _MOE_{SHARED_ONLY,PASSTHROUGH,NORM_ONLY} > 0:` gated 提前 return（10 份 chip_orch copy 各一，默认 0 inert）；host_orch pos=0 merged-loop + capture-pass（`hmid_L0`，未修好）；generator `_host_orch` 也含 merged+capture-pass。
pypto-project `main`：`6f5256d`（M3 NaN 根因修复 + M3b 定位）/`c28b7ac`（M3b 三个 ordering 修法失败 + FUSE 正解）。
memory `n1_head_gate_ondevice_restored_l1_nan`：UPDATE9（根因修复，推翻 UPDATE1-8 的 INT8 判断）+ UPDATE10（M3b 跨-orch 依赖墙 + FUSE 正解）。

---

```
继续 step3p5 **N=1 整网 decode 集成 + 端到端精度对齐 vs vLLM**（总目标，未完成，跨多 session）。分支
feat/whole-net-n1-fusion；harness tests/step3p5/_stage_whole_faithful_real_ipc.py；0234 tmux
pypto-ascend-0:0（8 卡）；真 W8A8 + 权重/KV 双 IPC。⚠ 汇报避免误导性词语：未 device 验证的不能称
"已完成/已修复"；P=42 未通过就是未通过。

【当前状态（device 实测，2026-07-14 续9）—— ✅ M4 token-exact 精度已达成，以本段为准；下方"续8"及更早为历史】

★★★ **精度目标达成（device 验证）**：整网 P=42（全 45 层、真 W8A8 原生 INT8 + 权重/KV 双 IPC、8 卡）`--hidden-token 6127` → **argmax=303 == vLLM golden**（TOP5=[303,410,1176,525,3163]，logit 13.63，303 自信 top）。**精度不对齐的根因已修（无 work-around）**：dense L2 调 swa_chip_orch 误传 `attn_layer_idx=1`——权重已在 call-site 预切（swa_wq[r,1]），而 attention_swa.py:226 `layer_hidden_base=attn_layer_idx*HIDDEN` 又对预切单层权重做 K-偏移 → 越界读到相邻 layer-3 权重 → 污染全部 42 个 MoE 层输入。修 L2 → `attn_layer_idx=0`（生成 decode_layer.py 两个 L2 分支 + generator `_gen_faithful_real.py::_emit_l2`；L1 与全部 42 MoE 层本就传 0）。device 复验 L2 next_hidden cos 0.931→0.999999。已 push pypto-lib `7294e26` / pypto-project `b20cda2`。**⚠ 下个 session 勿再查"argmax=993 / 303 不在 top-5"——那是 L2 修复前的旧状态，已解决；也勿再据"续8 M4 精度未达成"行动。**

★★ **唯一遗留 = A2 collective deadlock（robustness，非精度）**：P=42 间歇 507018（orch_error=8 TENSOR_WAIT），device 二分阈值=41 融合层。**这次靠 `ASCEND_GLOBAL_LOG_LEVEL=1` 时序扰动让 P=42 跑通一次拿到 303；logging 是临时 unblock、不是修复，生产 live serving 不能靠它。** 静态假设已全部 device/代码证伪：RING knob、fresh-exporter、`PTO2_SERIAL_ORCH_SCHED=1`、MemoryReuse(跳 Orchestration)、next_hidden_out 多 producer(generator 编译期 gate 只 1 writer)、DAG 定长溢出、CommDomainScopeStmt 提前 release(实为嵌套、全开到底)、ISA fence(TWait 有 dcci 读失效 + TNotify 有 st_atomic+dsb+pipe_barrier)、VA 碰撞(comm 池单个 224MB `[0x12c0416,0x12c04f6)`，权重池 `0x12c1c0` 上方~5.9GB 不重叠、42 层 windows 装得下不 wrap)。**定论（续10 修正）= A2 是 N=1 collective handshake 的代码 bug（wiki S1:running-stalled = kernel 挂死，非容量）；多程序永久排除，实现不了就修代码。** 静态分析到头，下一步 = 动态 device 抓取 stuck kernel (1,23) 的 signal 状态（观测会扰动，需最小化 + 多跑）+ 修 handshake。全证据链见 memory `n1_m4_accuracy_gap_converged_direction_drift`（含续10 wiki 分析 + 用户裁定）。

★ **下个 session 主线（续10 用户裁定，非可选）= 把 A2 当代码 bug 修**：(1) 动态 device DFX 抓 stuck kernel (1,23) 在各 rank 的 notify/wait signal 状态，定位丢/乱序的 notify（wiki S1 diag：定位卡住 kernel）；(2) root-cause 修 `tp_all_reduce` 流水握手（per-layer distinct tmp_window+signal_window reset+fence / bound 在飞 signal），P=42 **不靠 logging 也确定性 RUN_CLEAN**；(3) 再复验 --hidden-token 6127 → argmax=303 + cos/top5。**禁 work-around（禁调大 timeout、禁 logging 依赖、禁 multi-program）。** M5/M6 gated on 此修。

【历史（device 实测，2026-07-14 续8）—— M4 确定性已解；此段关于"M4 精度未达成/argmax=993"的部分已被上面续9更新，勿再据此行动】

★★ **M4 确定性 SOLVED（device 验证，本 session 最大成果）**：根因 = 整网 host_orch 只给 3 个 dense 层分配 l0_*/l1_* 两套通信/信号窗口，**第 3 个 dense 层 L2 复用了 L1 的 l1_attn_sig/l1_mlp_sig**。信号窗 AtomicAdd+Ge(1) 且层间不清零 → L1 残留信号=1 让 L2 的 Ge(1) wait 提前通过 → L2 all_reduce 在 peer 写完前 remote_load 读竞态数据 → attention 输出间歇破坏 → argmax 翻 + 偶发 507018。**修复 = 给 L2 独立 l2_* 窗口**（generator `_host_orch` + 生成的 decode_layer.py 都改；对齐 per-layer-distinct 原则，MoE 层本就如此，只 dense L2 漏了）。device 验证：P=0 next_hidden=23.5×4（bit-同）、P=1 argmax=102706×5 且 next_hidden=2.06×5（bit-同）、P=42 argmax=993×6（跨 l2fix+rbias 两批稳定）——**但 P=42 next_hidden row0 幅值仍抖（256~416）：argmax 鲁棒/稳定，但非 bit-identical，残余幅值非确定性尚未查清（可能是深层 MoE 的另一处，argmax 暂鲁棒）**。这些验证 run 无 507018（修前的间歇 507018 是同一竞态、偶发）。

★★ **续6/续7 的错误理论已被 device 证伪（勿再据此查）**：① **不是 fence-gap** —— pld.system.notify 确实发 dsb(DDR)+pipe_barrier（pto-isa a2a3 TNotify.hpp:50-60），排干 DMA；remote_store→pto.tstore、tensor.put→pto.comm.tput 都被 notify barrier 排干。② **不是通信基础设施** —— 独立 C++ allreduce_distributed 8卡 8/8 max|diff|=0.000、DSL test_l3_allreduce 8/8 golden。③ **不是 hidden-state 乒乓/attention residual racy**（那是 L2 窗口复用的表象，真因是窗口复用）。④ **不是残差相消 bug**：续8 用 device 向量证伪——`cos(moe_out, resid_hold)=−0.25`（轻微，非 −1），max-abs 的"10× 塌陷"是**假象**（"看 row0 不看 max"陷阱），残差 add 自洽（nh=resid+moe 精确、moe≈shared+routed）。

★ **golden 已 live vLLM 重验 = 303**（tid=6127「北京」→ 303「，」，top logprob −3.38 → 分布极平，top prob 仅 3.4%）。判据不应只看 exact-argmax，应同时看 cos≥0.999 + top-5 overlap。

★ **router_bias BF16 修复已落**（对齐 moe.py:485-490；整网 gate 10 份 copy 原缺 BF16-round，bias~4.79 的 ~0.015 舍入定 top-8 尾部）。但**非精度主因**：P=42 argmax 仍 993，303 不在 pypto top-5（pypto 自信 logit 34.9 vs 2nd 29.8）→ 系统性偏差，非平分布噪声。

★ **精度 gap（993≠303）现状**：已排除 fence-gap/通信/section-D sign-buffer/router_bias/确定性(argmax)/残差相消。字节对齐**部分核查**：所有中间 tensor 的**静态 shape 全 512B 对齐**（moe_out/sh_y/post_norm [16,4096]、routed_y_buf [128,4096]、K_CHUNK=256→512B 行）；device 向量 cos 平滑未见明显 tile-边界 garbage——但**逐 tile garbage / 运行时对齐未专门穷尽验证**（下 session 可在 golden 对拍时顺带查）。残差流逐层减小/非单调（device row0: L2=23.5→resid_hold=15.75(attn 与 h 反相关 −0.45)→nh(L3)=2.08→L4=160→…→L44；⚠ P=42 幅值抖 256-416）。**是否为 bug 无法从 device 向量单独判定，唯一可靠工具 = vLLM 逐层 golden 对拍。**

【下一 session 任务（按依赖）】
**T-CORR（M4 收尾主线）= vLLM 逐层 golden 对拍**：(1) 搭 vLLM 逐层 dump（tid=6127 单 token；beijing_1tok golden dump **不在 0234**，需重 dump —— vLLM dump instrumentation，端口/hook 见 tools/step3p5/collect_w8a8_prefill_golden.py + `.enable` marker；18-token prefill 的 row0=pos0=tid6127 也等价 ctx=1 golden）。(2) device **L2/L3 向量已 dump**（非全 42 层）在 0234 本地 `/tmp/n1_vec/`（⚠ /tmp 每机独立、重启即失，需要时重跑）：P0_nh=h_d2(L2 输出), P1_S5=resid_hold, P1_S4=moe_out, P1_S2=sh_y, P1_S3=routed, P1_S1=post_norm, P1_nh=nh(L3)；harness 已加 `N1_DUMP_DIR` env（存 row0 向量）+ top5 打印。要全 42 层需扫 P=0..42。(3) 逐层对拍 cos/ratio_allclose(atol=0.04)，**首个超阈层=出错层**（residual 逐层减小的 dense→MoE 边界 + shared-expert 是嫌疑，但**不预设**——以 golden 对拍为准）。(4) 定位后修，再 P=42 --hidden-token 6127 → argmax=303（+ cos/top5 判据）。
**T2（correctness，可能需要）= L43/L44 特殊 swiglu limit**：config SWIGLU_LIMITS[43]=[44]=7.0(routed)、SWIGLU_LIMITS_SHARED[44]=16.0；整网 baked silu 全 42 层。moe.py 用**编译期 baked 3 特化**（非 runtime scalar），shared clamp 窄 N-chunk（宽 tile miscompile）。swiglu clamp 在**共享方法** _expert_routed(FRESH_EXPERT_ROUTED)/expert_shared，per-layer 需生成 baked 变体 + host_orch 派发（L43=swa/routed7、L44=full/routed7+shared16）。**只 2/42 层，平分布下大概率非主导——先做 T-CORR 逐层定位再决定是否/何时上 T2。**
**T-MULTI（M5/M6）= 多 token / KV bridge / live A/B**（8001 vs 8000），gated on T-CORR。

【代码状态（未 commit，工作区 feat/whole-net-n1-fusion on 0234 NFS）】
- decode_layer.py：**L2 窗口修复（确定性，已验证）** + **router_bias BF16（10 gate copy）** + combine remote_store→tensor.put（对齐 moe.py）。generator `_gen_faithful_real.py`：L2 窗口修复（regen 用）。harness `_stage_whole_faithful_real_ipc.py`：N1_DUMP_DIR 向量 dump + top5 打印。
- 备份：`decode_layer.py.bak.l2fix_determinism_solved`、`_gen_faithful_real.py.bak.l2fix_determinism_solved`。
- ⚠ 未 push/commit；确定性修复值得优先 commit 保全（root-owned .git → tmux commit，b-csy push）。
- memory `n1_m4_determinism_solved_l2_window_reuse` 记确定性根因 + 纠正。

【当前状态（device 实测，2026-07-13 续7，多已被续8 证伪 —— 仅存作历史，勿据此往"竞态/相消"方向查）】
⚠⚠ **续6 的"next_hidden 爆 5.6e5 / 每 MoE 层 ×2 指数增长 / hidden-state 乒乓 buffer 竞态"结论 = 粗粒度 dump 假象，作废。**
- **纠正**：`max|next_hidden|` 是对**整个 buffer**取 max，混入了**垃圾 padding 行（ctx=1 只有 row0 是有效 token，row1-15 是 padding）+ 未用 expert 行（local_recv_max 里没被路由的槽，data window 不自动清零）**。按 **row0（唯一有效 token）**看，valid token **全程有界**：post_norm≈1.15、moe_out≈15、next_hidden≈34-46。**没有指数爆炸。**
- **M4 真实症状（row0 有效 token，device）**：valid token 仍**非确定**但幅值小——next_hidden row0 在 31-46 跳（~30%），argmax 乱跳（27527/47645/36711…，golden=303）。
- **精确定位（row0）**：racy 的是 **attention 残差**（resid_hold = next_hidden − moe_out，跳 16-31 ~2×）；**MoE body 相对稳**（moe_out row0 14.8-15.5 ~5%，post_norm row0 ~1.15）。⟹ **竞态在 MoE-block 层的融合 attention，不在 MoE body/combine。** P=0（dense-only）argmax=27527 两次一致 → **tail(lm_head) 干净**。
- **⛔ 根因未最终锁定（下 session 重点）**：融合 orch 里的 attention 用的是和 dense **同一个 kernel**（dense L0/1/2 确定），但融进 MoE orch 后 attention 残差 racy。**为什么？** MoE body 在数据依赖上是 attention 之后（post_norm 读 resid1）——理论上不该 race attention。候选（未证实）：attn all_reduce 的 `attn_tmp_window` 与 MoE 的 comm window 在 **IPC 跨 rank VA 重合**被别的 rank 的 remote_store 踩；或 KV write/read；或融合引入的其他别名。**必须找到真因，不许 un-fuse 绕过（用户硬约束：留 N=1 融合方案）。**
- **本 session 已落地 3 个真修复（都在 generator `_gen_faithful_real.py`，已 regen，未 commit；都是真 bug 但非 M4 根因）**：
  (1) per-layer distinct hidden buffer `h_moe_L{pos}`（去 2-buffer 乒乓；h_mid 194560→294 稳定）；
  (2) write-once `resid_hold_L{pos}`（去 fused-orch stash→residual_add 对 next_hidden_out 的 WAW）；
  (3) combine DMA-drain fence（`_push_routed_y_to_sources` 加 self-notify AtomicAdd+0，对齐 moe.py:1753；moe.py:1840 明说"data window 不自动清零，只 signal window 自动清零"）。
  另加诊断：harness 打印 row0|next_hidden|/row0|dbg|；generator 加 `P_DBG_STAGE=5` dump resid_hold（attention 输出）。
- **可靠方法学（务必遵守，避免重蹈弯路）**：
  ① **`P_FUSE_ATTN_ONLY` 是 NO-OP**（pypto 建整张 DAG，`moe_residual_add` 覆盖 FUSE 写）——不能用它 bisect；只有 host 级 `P_FAITHFUL_MOE_LAYERS` 可靠。
  ② **看 row0（有效 token），不看 max**（max 被 padding/未用行污染，误导成"爆炸"）。
  ③ **golden 比对：直接 dump tensor 的有效行，比 mean/variance/L1-norm，不比 max。** oracle = vLLM（L1 ctx=1: tid=6127→argmax=303）。
  ④ 确定性检查 = 同配 ×2/×3 比 row0 + argmax（device 有间歇 507018，需多跑）。
- **swiglu limit（T2）没实现**：整网 baked silu 全 42 层；config `SWIGLU_LIMITS[43]=[44]=7.0`(routed)、`SWIGLU_LIMITS_SHARED[44]=16.0`。P=42 在 L43/L44 本来就偏——与竞态两件独立的事，M4 达 argmax=303 需要它。
- **下一步（按依赖）**：① 直接 dump resid_hold(attention 输出)有效行 mean/var/L1-norm ×3 确认 attention racy（stage-5 已加）；② 找 attention racy 真因（IPC 跨 rank window 重合 / KV / 融合别名）——**留 N=1 融合，修根因不绕过**；③ 补 swiglu limit(T2)；④ P=42 --hidden-token 6127 → argmax=303。
- M5 多 token / M6 live vLLM 集成 均未开始（gated on M4）。

【历史（续6，M4 部分已被上面证伪）】
- M0 probe ✅ / M1 双 IPC 8 卡 dispatch-clean ✅ / M2 on-device head-gate ✅。
- **A1 单层 MoE INT8 数值 ✅** device-verified（_expert_routed 对齐 moe.py，原生 W8A8 无 BF16-dequant）。
- **A2 = 1A dispatch-side INT8 已实现，P=42 device RUN_CLEAN**：`_gen_faithful_real.py` 加
  `FRESH_QUANT_MOE_INPUT`（对齐 moe.py `_quant_moe_input`）+ span-scoped 改 dispatch 链（recv_x
  BF16→INT8 + per-token recv_scale 贯穿 push）。P=42 RUN_CLEAN，comm pool win_size=224MB（原 391MB）；
  smoke a2a3sim COMPILE OK；P=1/P=10/P=31/P=42 全 RUN_CLEAN。
  ⚠⚠ **更正文档早前错误描述**：A2 stall 的"comm-window pool **字节上限 ~290-390MB**"结论 **已被 device 证伪**
  （见 memory `n1_comm_window_bytecap_refuted`：standalone 8-rank allreduce 在 64MB→2GB 窗口全 PASS；
  co-resident 24GB pool 也不复现；winSize/windowsIn 都是 uint64 无 cap）。真相：P=42 507018 是**程序结构性
  竞态 / 逐层 window-signal 别名**（RAW-only-v1 P3/ADR-013），不是字节上限、也不是 VA 碰撞（两者都已证伪）。
  ∴ 1A 缩 recv_x 让 P=42 从"挂死"变"能跑"很可能是**改了 footprint 让别名从 stall 转成数值竞态**，不是真正
  消除别名。（1A 本身仍值得保留——对齐 moe.py、原生 W8A8；但别把它当"修好了 P=42"。）
  ⚠ regen 坑：`pregen_real` 是**陈旧 base**（缺 gate_topk mrgsort 修复 + `_zero_routed_y_buf`）；正确 regen =
  从 clean-A1（`decode_layer.py.bak.pre1a_gen`）**原地 strip real builder** 保留最新 base（strip 前 assert
  `pl.mrgsort(srt, block_len=256)` + `_zero_routed_y_buf` 在 base 里），**不要 cp pregen_real**。
- **⛔ M4 token-exact 未达成；root cause 已 device 确认 = hidden-state 乒乓 buffer 竞态/别名（与 A2 同一病根类）**：
  P=42 argmax=993（应 303），next_hidden 爆 5.6e5，**每 MoE 层 ×~2 指数增长**。证伪链（都 device）：
  ① post_norm(深层 L12)=4.66=O(1) → RMSNorm 正常，MoE 输入有界；
  ② **P_FUSE_ATTN_ONLY=1（跳 MoE body）仍爆** P=10=374784 → 排除 MoE / head-gate / routed-INT8（1A 洗清）；
  ③ dense P=0=502 有界（每乒乓 buffer 写 ≤2 次，低于别名阈值）；
  ④ **P=10 同配置重跑 251904 vs 430080 → 非确定** ⟹ 物理 buffer 复用竞态（非确定性 compute bug）。
  定位：整网只用 **2 个 `pl.Out`（next_hidden_out/h_mid_out）在 45 层乒乓、各写 ~22 次** → pypto 单值
  producer_index（RAW-only-v1）分不清多 producer → 后层读 stale/raced 版本 → ×2 复合放大。与
  `n1_whole_net_scheduler_timeout_fixed_perlayer_windows`（comm-window 别名，per-layer distinct 修）同类；
  也与 `n1_comm_window_bytecap_refuted` 的"逐层 window/signal 别名"结论收敛。
- M5 多 token / M6 live vLLM 集成 均未开始（gated on M4）。

【下一步任务清单（下个 session，按依赖顺序）】
**T1（M4 主修）= hidden state per-layer distinct buffer**：`_host_orch` 现用 2 个 program-Out 乒乓，改成
  **每层独立 hidden buffer**（每层写自己的、读上一层的，任何 buffer 不写 >1 次；最后一层写 program Out 供
  harness readback）。镜像 comm-window `_L{pos}` 先例；**用静态命名 `h_buf_L{pos}`**（大概率绕开先前
  computed-offset `hchain[r*87+slot]` 撞的 pypto DCE `Unhandled ScopeStmt subtype: CommDomainScopeStmt`
  ——该 DCE 补丁已在 `dead_code_elimination.cpp` 定位但**未 rebuild**；静态命名免 rebuild）。
  - **T1a 先最便宜证伪**（agent 建议，做全量前必做）：只给 L2→L3 一处加一个独立 3D `pl.Out hs_test`，P=1 跑；
    若 next_hidden 变 finite + **跨 run deterministic** → 别名坐实 → 再做全量 T1b。
  - **T1b 全量**：generator `_host_orch` 45 层各独立 buffer + harness alloc。验证：P=1 跨 run deterministic
    （不再 502/462 跳）→ P=10 有界（非 2.5e5）→ P=42。
**T2（token-exact correctness gap，M4 必修）= L43/L44 特殊 swiglu**：config `SWIGLU_LIMITS` 7.0@L43/44、
  `SWIGLU_LIMITS_SHARED` 16.0@L44；whole-net 现 baked **silu_silu 全 42 层**（`routed_lim=0/shared_lim=0`）。
  修法 = runtime per-layer limit（always `pl.minimum(x, limit)`；常规层 limit≈1e30 等价 silu，L43/44 传 7/16），
  host_orch 传 per-layer 值。（T1 修完 P=42 有界后，这是达成 argmax=303 的下一必修项。）
**T3 M4 验收**：P=42 `--hidden-token 6127` → argmax=**303**（vLLM golden）。依赖 T1+T2。
**T4 M5/M6**：多 token / KV bridge / live A/B（8001 vs 8000），复用 G5b track 机器（0162）。gated on T3。

【代码状态（未 commit，工作区 feat/whole-net-n1-fusion on 0234 NFS）】
- 1A（A2 版）已生成于 `models/step3p5/decode_layer.py`；generator `tools/step3p5/_gen_faithful_real.py` = 1A。
- 备份：`decode_layer.py.bak.1a_A2_generated`（1A 生成结果）、`_gen_faithful_real.py.bak.1a_A2solved_*`（1A generator）、
  `_gen_faithful_real.py.bak.pre1a`（clean-A1 generator）、`decode_layer.py.bak.pre1a_gen`（clean-A1 = 正确 base+旧 real）。
- 8 hold-mode exporter 常驻 READY（--reuse-exporters）。⚠ 一次 507018 会毒化整卡（clean-A1 P=20/186MB 本该干净
  却也 stall）；下 session 若遇连续 stall，先 `touch /tmp/n1_weight_ipc/STOP` 重启 exporter 或重启机再验。

【硬约束（不可违背）】
- IPC(KV+权重) + 真实 W8A8 权重；**禁 BF16-dequant 历史版本**；禁 H2D 绕路。
- 遇问题只解决不绕开；诊断脚手架不进产品路径。correctness+speed 都要（别用 correctness 当停摆借口，
  也别造错误精度数字）。
- 对齐 DeepSeek/Qwen + moe.py；step3p5-vs-DeepSeek 差异必须论证（只在性能更好时保留）。
- 架构优先、先设计后编码；严格遵守 pypto-dev-constraints SKILL；分清模型每层交接边界；避免陷入局部错误反复。
- 历史文档可能 stale，先核对当前代码。ready 的口径 = live-token-exact-device（compile/offline/synthetic
  均不算 done）。汇报用词精确、不误导。

【环境】三件套：source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh && export
PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib:$PYTHONPATH（append，否则丢
acl）。8 卡 env：PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072。
W8A8 ckpt=/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp。8 hold-mode
exporter 常驻 READY（--reuse-exporters；/tmp 每机独立，查 ready 在 tmux 0234：
ls /tmp/n1_weight_ipc/ready.rank*|wc -l）。改 models 后 find models/step3p5 -name '*.pyc' -delete。
禁 -9 强杀 device / npu-smi reset（netboot 锁卡）。push 走 b-csy + HTTP/1.1 + PAT
/data/chensiyu/secrets/github.env（0234 连不通 github）。
```

---

## 📋 详细任务清单 — 收敛到 INT8-native token-exact（2026-07-13, G5b + N=1 两 track 汇合）

> 来源：G5b track（`NEXT-SESSION.md`，harness `_stage_whole_decode_run.py`）诊断出 whole-decode 剩余
> token-garbage 根因 = **BF16-dequant 在大残差幅值下精度退化**（L17 attn 起，control 证实单调随幅值），
> 与 N=1 track（本文，M3b gap-5）**汇合到同一修法：INT8-native（与 vLLM W8A8 对齐）**。
> 贯穿纪律见 `notes/08-integration-churn-postmortem.md` + memory `feedback_integration_churn_root_causes`：
> **ready 只认 live-token-exact-device；声明 root cause 前先跑证伪实验；对齐 DeepSeek；pin 单一底座。**
> 每条任务标了「验证口径(bar)」——只有过 bar 才算 done，否则标 `provisional`。

### Phase A — INT8-native whole-net 数值正确 + P=42 全链跑通（N=1 主线，当前卡 A2）
- [x] **A1 单层 MoE INT8 数值** — ✅ **device-verified（2026-07-13 续5）**：`_expert_routed` 对齐 moe.py device-PASS 版（flat-pre-quant INT8，原生 W8A8、无 BF16-dequant）。**bar 达成**：P=1 op-dump `local_routed_y` 3.99e11→1.41 finite；P=1/P=20/P=31 全链 device RUN_CLEAN。（clean-A1 在工作区，未 commit。）
- [x] **A2 P=42 全链能跑 = 1A dispatch-side INT8** — ✅ **device RUN_CLEAN（续6）**：1A 已实现（`FRESH_QUANT_MOE_INPUT` + span-scoped dispatch INT8 recv_x + recv_scale）；P=42 RUN_CLEAN，pool win_size=224MB。⚠⚠ **更正**：早前"根因 = comm-domain window pool 字节上限 ~290-390MB"**已被 device 证伪**（`n1_comm_window_bytecap_refuted`：64MB→2GB 窗口全 PASS，24GB co-resident 也不复现；续5b 的 666MB stall 是**权重池共存**下测的）。真机制 = **逐层 window/buffer 别名竞态**（P3/ADR-013）。1A 让 P=42 从 stall 变"能跑"是改了 footprint（stall→数值竞态），**不是消除别名**。**（不做 FUSE→SPLIT。）**
- [ ] **A2b 真消除逐层别名（= M4 的 T1）** — ⛔：整网 hidden state 用 2-buffer 乒乓（各写 ~22 次）→ M4 数值竞态。修法见底部任务清单 T1（per-layer distinct hidden buffer）。
- [ ] **A3 M4 L1 token-exact**：P=42 `--hidden-token 6127`（L1 ctx=1）→ **argmax=303** vs vLLM。**bar**：greedy top-1 命中 303。依赖 A2。

### Phase B — 解 G5b 精度 open question（falsify-before-assert；memory g5b_swa... 续¹¹h）
- [x] **B1 查 attention 是否 W8A8** — ✅ DONE 2026-07-13（non-device）：W8A8 ckpt weight-index 显示 **q/k/v/o_proj `weight_scale#=0`（BF16，未量化）**，gate/up/down/experts `weight_scale#>0`（INT8）→ **step3p5 W8A8 只量化 MoE/MLP，attention 是 BF16**（vLLM 也 BF16）。∴ **BF16 attention 大幅值不内在退化**（vLLM 在 L17 正常为证）→ **L17-alone 0.25 是 mid-start bootstrap artifact，非真 bug**；唯一 quant-path 差异 = BF16-dequant MoE → **INT8-MoE-only 是正确且充分的修法，attention 不需 INT8**。（复现器 `/tmp/_ckq.py`；memory 续¹¹i。）
- [ ] **B2 INT8-native full-chain golden 对拍**：在 A2 的 INT8-native whole-net 上，跑 **full-chain-from-L0**（非 mid-start，避 bootstrap artifact）逐层 out vs vLLM golden。**bar**：L0-44 逐层 `ratio_allclose(atol=0.04)` 全过，尤其 **L17 恢复**（BF16 时 attn 0.25 / chain NaN）。→ 坐实 INT8-MoE 是否足够修 L17。依赖 A2 + B1。
- [ ] **B3（条件）attention INT8/精度**：若 B2 显示 L17 仍退化 且 B1 显示 attn 是 W8A8 → 给 attention 上 INT8（per-token act-quant，仿 vLLM/moe.py），或残差流关键处 FP32 累加。**bar**：B2 重跑 L17 过。依赖 B2。

### Phase C — G5b live 基建移植到 INT8-native + live A/B（M5/M6）
- [ ] **C1 移植 G5b live fixes 到 INT8-native whole-decode**：从 `NEXT-SESSION.md` track（`_stage_whole_decode_run.py` stepfun/develop）移植 —— (a) `_feed_meta` **seq_len=0 pad 行 sanitize**（seq_len=1 + 非冲突 scratch slot；已 device 验证消 NaN）；(b) 容器后端 prefill 检测 **`num_prefill_tokens>0`**（原 seq_len==0 误判把 prefill 路由到 decode kernel）；(c) `SIMPLER_COMM_NO_HCCL=1` co-tenancy；(d) `--serve` sidecar loop + AF_UNIX。**bar**：INT8-native sidecar serve listening + import KV + co-resident live 8001 无 crash。依赖 A3。
- [ ] **C2 M5 真 KV bridge**：vLLM 真 KV 经 IPC 接入 whole-net（复用 G5b KvIpcMap/`_CTRL_IMPORT_IPC`/build_stacked_kv）。**bar**：单层 active-row(row0) paged-index 数值对拍 vLLM decode dump 过。依赖 C1。
- [ ] **C3 M6 live A/B token-exact**：8001(INT8 pypto mode=full) vs 8000(vanilla) **3-prompt greedy(temp=0)**。runbook 见 `NEXT-SESSION.md`（停 8001 pkill EngineCore + 清 card8-15 zombie；起 8001 `/logs/start_8001_full.sh`；起 sidecar `SIMPLER_COMM_NO_HCCL=1 WD_RING_HEAP=1GB`；SIGTERM 停 sidecar）。**bar（真出口）**：L3 greedy **top-1 ≥ 95%** 对 8000（L1 hidden atol=0.04 / L2 logits cos≥0.999+topK≥4/5 辅证）。依赖 C2。

### Phase D — 准出 + perf
- [ ] **D1 精度双过**：L1/L2/L3 全过（vLLM eager dump oracle，非 synthetic）。
- [ ] **D2 perf baseline**：去掉每步 MoE 权重 copy（常驻权重 / G2 weight-IPC 重叠）；现 ~120s/token → 目标基线。

### ⚠ 关键依赖 / 汇合点 / 已排除
- **A1(gap5) 是全链 gating**：INT8-native whole-net 不 finite 前，B/C 都无法验证。
- **两 track 汇合 = N=1 INT8-native compute（A/B）+ G5b live 基建（C）**。避免同时改 n1-live（跨 session 冲突 = 反复推翻根因之一）。
- **已排除（device 验证，别重查）**：G5b 原始 NaN=seq_len=0 pad（已修）、rope/qk_norm/head-gate/KV-layout/full-attn kernel、Blocker B=gate_topk（非 IPC-VA）、KV 池 offset（regular）。
- **底座**：N=1 = 0234 tmux `pypto-ascend-0:0` + `feat/whole-net-n1-fusion`（moe.py INT8 cd3ef0d 在此分支）；G5b live 基建 = 0162 + stepfun/develop。moe.py 的 `select_moe_block(w8a8_native=True)`/EpTpMoEW8A8 在 `feat/whole-net-n1-fusion`，n1-live/stepfun-develop 都没有。
