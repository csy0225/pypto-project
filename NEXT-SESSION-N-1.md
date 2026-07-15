# NEXT SESSION — N=1 整网 decode 集成 + 端到端精度对齐 vs vLLM

> **🔒 测试唯一事实源（焊死）**：任何"跑通 / 精度 / stall"验证一律用
> [`N1-CANONICAL-TEST.md`](N1-CANONICAL-TEST.md) 的唯一测试 + 命令。禁止自造 harness / 换随机输入 /
> 把 `RUN_CLEAN` 当通过。要点：被测程序唯一 = `whole_decode_faithful_real`；精度金标准 = 真 token
> **6127** + 真 W8A8 权重（IPC）→ `argmax == 303`（`RUN_CLEAN ≠ PASS`）；slow-vs-deadlock 用隔离探针
> `_probe_barrier_scale.py` PUSH + 三个 `PTO2_*` 超时；不可 work around；下结论前两 agent 对抗。
>
> **【续15（2026-07-15，device）三超时诊断结果 — 坐实"真 hang 非慢"】** 按续14 第 1 步执行：隔离探针
> `PUSH=1 CHAIN=1 N_COLL=42`（`_probe_barrier_scale.py`，忠实复现 `_dispatch_push`/`remote_store` 的
> `remote_store→dsb(DSB_DDR)→TNOTIFY` 时序，与整网 func28 生成 cpp 同构）：
> - **baseline（默认超时 sched10s/op45s/stream50s）：5/5 STALL**，`dt≈65s`，`sub_class=S1 sched_error_code=100 state=RUNNING kernels=[aiv0:0]`。
> - **抬 10×（sched=100000ms/op=450000000us/stream=500000ms，过排序校验）：attempt-1 仍 STALL，`dt≈475s`**（≈op 450s 预算跑满），同一 S1 签名。`dt` 从 65s→475s 证明超时确实生效，**但仍不完成** → **真 hang / 死锁，非"慢"**。多等 7× 时间也救不回 → 抬超时不是修法。
> - ⟹ **续14 的定性成立**：`_dispatch_push` 跨卡 PUSH 写完成间歇不触发；抬超时坐实真 hang。**下一步落地 dispatch push→pull**（续14"修复计划"）。
> - **重要澄清（golden）**：`argmax=303` 是**真 token 6127 + 真权重**跑出的（`_stage_whole_faithful_real_ipc --hidden-token 6127`，非随机输入的 `_weights` harness），日志复现 76 次、vLLM golden=303；但存在非确定性尾巴（clean 有时出 993/45114），故金标准必须 `argmax==303` 而非 `RUN_CLEAN`。
> - **dsb 说明**：生成 cpp（探针 `reduce_step.cpp` L114/128-130、整网 `_dispatch_push.cpp` L283-285）**确有** `dsb(DSB_DDR)` 夹在 `remote_store` 与 `TNOTIFY` 之间——**不是"缺 dsb"**。为何已有 dsb 盖不住时序问题见 [`N1-PUSH-DSB-TIMING.md`](N1-PUSH-DSB-TIMING.md)（给专家看的时序图）。
>
> **【续15 定案数据】** baseline 默认超时 **5/5 STALL**（~65s）；抬 10×（sched=100000ms/op=450000000us/stream=500000ms）**3/3 STALL**（~473–476s，op 预算跑满仍不完成），全同 S1/`sched_error_code=100`/`aiv0:0`。⟹ **真 hang 已坐实，抬超时不是修法。下 session 主线 = 落地 dispatch push→pull（见下）。**

> **【续16（2026-07-15，device 环境已就绪 + 设计落地 + 评审通过 + pull 方法码已入库，wiring 未接线）】**
> 本 session 完成 dispatch push→pull 的**设计 + 双 agent 对抗评审 + 可复用模板确认 + FRESH pull 方法码落地（inert）**；剩下的是**机械 wiring + regen + device 验 argmax==303**（较大、且 0234 无本地 compile-check，留作接续）。**未 device 验证——不得声称已完成。**
> - **环境就绪**：0234 容器活着（tmux `pypto-ascend-0:1`，`brainctl rjob launch` PID 见 win1），env 三件套已激活，**8 个 hold-mode exporter 8/8 READY**（`--reuse-exporters` 秒级 attach，省 15min 冷载）。canonical §1 golden 命令可直接跑。
> - **设计（validated）**：push（`_dispatch_push` = `pld.tensor.put`/`remote_store` TPUT MTE3 远端写，func28 S1 挂）→ pull（`pld.tile.remote_load` TGET MTE2，完成本地可观测；tp_all_reduce/`ep_all_to_all`(collectives.py:455)/barrier 探针 PUSH=0 均 device-clean）。source 各卡把自己 token 按 (dst,loc_e) bucket 序打进**自己**的 peer-readable `send_x/send_scale/send_route` window；Set/Ge rendezvous；dst 用 `remote_load` 按 `off_s`（**完全由已发布的 pub_counts 逆推，无新增跨卡发布**）gather 回本地 recv_x，复现**逐字节相同**的 expert-major CSR（loc_e 外/src 升序/cursor 内）→ `_dispatch_stage`/`_expert_routed`/combine **不动**。
> - **双 agent 对抗评审（canonical rule 5）= PASS**：Agent A（correctness）逐条 re-derive off_s + CSR 逐字节相同 + barrier soundness 全 PASS（唯一 caveat：`send_route` padding 列 1..idx_pad-1 若不 zero-init 与 push 不逐字节同，但 stage 只读 col0，correctness 无影响；FRESH 已用 `pl.tile.full(value=0)` idx_tile zero-init）。Agent B（feasibility vs SKILL）：无 hard blocker；remote_load 在 `pl.range(n=runtime)`+runtime offset OK（`collectives.py:561` 先例）；self-window `pl.store` OK；send_x 静态首维 128 避开 §3/§4；§7 unroll UB 风险 → gather 嵌套镜像**已 device-clean 的 combine `_push_routed_y_to_sources`**（loc_e/src/row、per-row [1,HIDDEN]）即安全；只有 L25472 那份 `_dispatch_push` 被 `WholeDecodeFaithfulReal` 用；建议 real-only transform 保 BF16 base+其 INT8 transform 不动。
> - **落地物**：`tools/step3p5/_gen_faithful_real.py` 新增 **`FRESH_DISPATCH_PULL_INT8`** 常量（`_dispatch_pack_publish` + `_dispatch_pull` 两个 InCore 方法，INT8 pull，已 py_compile 过）+ **WIRING PLAN** 注释（5 步机械接线）。**inert（未被 splice 引用，generator 输出不变，tree 仍可跑）**。备份 `_gen_faithful_real.py.bak.pre_pulldispatch_20260715_123346` + `decode_layer.py.bak.pre_pulldispatch_20260715_123346`。
> - **下一步（接续，按 WIRING PLAN）**：(1) `_host_orch` 加 `send_x_buf/send_scale_buf/send_route_buf` per-layer + 复用 `count_done_buf` 作 `pack_done_sig`；(2) MoE fused orch sig 加 send_* 参数（real-only transform）；(3) splice FRESH pull 方法 + 改 `dispatch_step`（pack_publish→pull→stage）+ 改 `decode_layer.py:~26623` 调用点；(4) strip real builder → regen → py_compile → **device 跑 canonical §1，多跑几次，判据 argmax==303 且不 stall**；(5) combine push（`_push_routed_y_to_sources` TPUT，抖动源非挂死源）之后同法转 pull。
> - **纪律**：不 work around；ready 只认 live-token-exact-device（argmax==303）；改前两 agent 对抗（本 session 已做设计评审，wiring 落码后仍需 device 复验）。

> **【续16b（2026-07-15，device 验证 — dispatch pull 已接线+编译+跑通；hang 按预测移到 combine push）】**
> **dispatch push→pull 已完整接线并 device 跑通**（wiring 全落地 generator + regen-clean + py_compile + 8 卡 device run）。device 证据：pull 版**编译通过全流程**（39 kernels 注册、IPC 权重导入、KV bound、"built 46 args, running..."），**跑过 dispatch**，最终 stall 在 **func 36**（`completed=78/81 running=1 waiting=2 orch_error=8 TENSOR_WAIT sub_class=S1`，8 卡同一 stuck，stuck_task_id 低位=26）——**orch 末端（78/81），func 36 > dispatch 的 28/29**。
> - **定位（by elimination）**：`_dispatch_pull` 若挂则 stuck func = 28/29（实测已完成，只在 RUNNING 采样出现）；tp_all_reduce 是 PULL（device-proven clean）；唯一剩下的"末端+带 WAIT 的跨卡 PUSH（TPUT）"= **combine `_push_routed_y_to_sources`**（仍 `pld.tensor.put`，即 WIRING PLAN 第 5 步/续14 预言的第二个 push 点）。⟹ **dispatch pull 生效，hang 按预测移到 combine push。**（func 36 精确名未拿到——build dir finalize 时被清；靠 completed=78/81+func>dispatch+WAIT+唯一剩余 push 三重旁证，非拍脑袋。下轮可设 keep-build 拿 kernel_config 坐实。）
> - **结论**：**push→pull 的 dispatch 半已完成、device 证明有效**；到 argmax==303 还需**把 combine 也转 pull**（第 5 步，现为 critical-path）。combine 是天然 gather：expert holder 把 routed 输出写进 peer-readable window，source 用 `inverse_map[t,k]`（generator 有 def=dead code，需激活）算 (dst_rank,dst_row) 后 `remote_load` 回本地 `routed_y_buf[r_route]` → `_weighted_gather_and_add`。
> - **落地物**：generator + decode_layer.py 现为 pull-dispatch 版（regen-clean，py_compile 过）。checkpoint 备份 `models/step3p5/decode_layer.py.PULLDISPATCH_WORKS_20260715_051112` + `tools/step3p5/_gen_faithful_real.py.PULLDISPATCH_20260715_051112`；known-good push 版在 `decode_layer.py.GOODKEEP`。新增 `tools/step3p5/_strip_real_builder.py`（regen helper）。修复过一个 wiring bug：base dispatch_step 的独立 `self._dispatch_publish(...)` 调用需删（`_dispatch_pack_publish` 已含 publish）——已在 generator `_step_edits` 处理。
> - **下一步（combine→pull）**：(1) 新增 peer-readable `routed_src_buf[local_recv_max,HIDDEN]` window（expert holder 写 routed 输出）；(2) 激活 `_build_inverse_map` 或 pub_counts 逆推 dst_row；(3) rendezvous barrier；(4) source `remote_load(routed_src_buf, peer=dst, [dst_row,0])`→本地 `routed_y_buf[r_route]`；(5) 删 `_push_routed_y_to_sources` 的 tensor.put + combine_done push；(6) regen → device 验 argmax==303（多跑）。exporters 8/8 常驻 READY（--reuse-exporters）。

> **【续16c（2026-07-15，device — 全 push→pull 已实现+编译+跑通；仍 507018，kernel_config 坐实 stuck=`_dispatch_pull`；纠正续16b 的未证实映射】**
> **combine→pull 也已完整落地**（generator FRESH_COMBINE_PULL：`_stage_routed_src`+`_pull_routed_y`，激活了 dead 的 `_build_inverse_map`；window `routed_src_buf` 全线程接；base `_push_routed_y_to_sources` 留死。regen-clean + py_compile 过 + 8 卡 device 编译通过全 codegen + 跑）。踩坑并修：① `_pull_routed_y` local-vs-remote device-if 让 `tile` 两种 Tile 类型 → SSA reject → 改成**统一 remote_load**（peer=dst 可==my_rank，dispatch_pull 已证自读 OK）；② `_build_inverse_map` 读 `pub_counts`(DistributedTensor) 必须在 **InCore** 上下文（放进 `_pull_routed_y`，不能在 Inline `combine_step` 里，否则 codegen `tensor.read must be TensorType`）。
> - **⚠ 纠正续16b**：续16b 说"hang 移到 combine（func36）"是**未证实的 func 映射猜测**（run3 build dir 被清、拿不到 kernel_config，只凭 completed=78/81+旁证瞎猜）——**这正是文档反复警告的坑**。本轮 run7 **build_output 没被清**（`build_output/WholeDecodeFaithfulReal_20260715_053555/next_levels/full_moe_chip_orch/kernel_config.py` 在），坐实 **func 28 = `_dispatch_pull`**（func27=_dispatch_pack_publish/28=_dispatch_pull/29=_dispatch_stage/35=_zero_routed_y_buf）。full-pull device stuck = **`_dispatch_pull`**（`completed=39/41 orch_error=8 sub_class=S1`，8 卡 aiv0:28），**不是 combine**。⟹ dispatch pull 自身会挂（或 racy）。
> - **待定（正在 device 判 racy vs deterministic）**：full-pull 已跑 2/2 STALL；正在多跑 3 次（run8-10）看是否偶发出 argmax==303（原 push 版 ~34% clean）。若从不 clean → pull 单独不足以消除 507018，root cause 比 push-vs-pull 更深（单 program whole-net 里多 collective 交织的调度/完成 race，接近续13 论点）；若偶发 clean → pull 降低了挂率、dispatch_pull 时序仍需收紧。
> - **落地物**：full-pull checkpoint `decode_layer.py.FULLPULL_20260715_054234` + `_gen_faithful_real.py.FULLPULL_20260715_054234`；dispatch-only-pull `PULLDISPATCH_WORKS_20260715_051112`；known-good push `GOODKEEP`。kernel_config 存 `logs_n1/full_moe_kernel_config_053555.py`。
> - **拿 kernel_config 的正道（下轮 func 映射必用）**：harness 打印 `compile OK => <output_dir>`；`build_output/WholeDecodeFaithfulReal_*/next_levels/<orch>/kernel_config.py` 里 `func_id→name`；device log `kernels=[aiv0:N]` 的 N 查此表。**别再靠 completed 比例/位置猜 func。**

> **【续16d（2026-07-15，device — 全 push→pull 仍 507018；6/6 STALL；pull 当前是 regression 非 fix；重估 root cause】**
> **全 push→pull（dispatch+combine 都 pull，含 self=local/peer=remote 对齐 ep_all_to_all 的修正）device 跑 6 次（run6-11）全 STALL（0% clean）**，而 known-good **push 版是 ~34% clean**。⟹ **as-implemented 的 pull 改写目前比 push 更差（regression），未达 argmax==303。**
> - **关键观察**：run11 同一次挂里不同核 stuck 点不同（core24 `completed=39/41`、core26 `completed=67/70`，都 `orch_error=8 S1 running-stalled`）——**不是单个确定 kernel 的 bug，是整网多 collective 交织在调度器下 stall**。self-`remote_load`→local 的修正（对齐 ep_all_to_all，理应更对）**没有改变挂的行为** ⟹ self-remote_load 不是（唯一）根因。
> - **重估（诚实，可能推翻文档"pull 修 stall"假设）**：把两个跨 die push 都转 pull **没有**消除 507018，说明 root cause **比 dispatch/combine 的 TPUT push 更深**——更像 **续13 的论点**（N=1 单 program whole-net 里大量 collective 交织、缺可靠 in-kernel 跨 die 同步/确定派发序 → racy/scheduler stall），而非"某个 push 原语脆弱"。隔离探针 PULL N=42×2 clean 是**孤立**的；整网 42 层 dispatch-pull+combine-pull+tp_all_reduce+shared 交织后仍 stall。**注意**：这与续14"dispatch push 是唯一挂点、pull 修好"直接冲突——续14 可能也基于未证实的 func 映射（同续16b 的坑）。
> - **下一步（需重新定方向，勿盲目继续 pull 细修）**：(1) run11 开 ASCEND log 拿 stuck func（kernel_config 在 build_output 持久，别猜）——确认是 dispatch_pull / combine _pull_routed_y / tp_all_reduce 哪个，且是否跨 run 漂移（漂移=racy 调度，非单 kernel）；(2) 若确认是"多 collective 交织调度 race"→ 这是 runtime/scheduler 层问题，考虑上报 simpler runtime team（要确定性跨 rank 派发序）或重估 N=1 vs 多 program（但用户续10 禁多 program）；(3) 若定位到具体 pull kernel 的确定 bug（如 pub_counts AtomicAdd 未落地→off_s 越界→remote_load 越界 hang）→ 修那个。**先拿 stuck func 再定方向，勿再瞎修。**
> - **回退保险**：`decode_layer.py.GOODKEEP` = known-good push（~34% clean，argmax==303 可复现）。full-pull 版在 `decode_layer.py.FULLPULL_20260715_054234` + 当前 tree + generator（含 self=local 修正，未存新 checkpoint——tree 即最新）。要回 push baseline：`cp decode_layer.py.GOODKEEP decode_layer.py`。

> **【续16e（2026-07-15，device — run12 开 log，kernel_config 坐实 stuck func）】**
> run12（ASCEND log，build `WholeDecodeFaithfulReal_20260715_055407` 持久）device stall kernels（RUNNING 采样）：**aiv0:37/38/39（513× 各）+ aiv0:28/29（77× 各）**。kernel_config 映射：**28=`_dispatch_pull`、29=`_dispatch_stage`、36=`_stage_routed_src`、37=`_pull_routed_y`、38=`moe_combine`(=_weighted_gather_and_add)、39=`moe_residual_add`**。多个 orch 实例卡在不同深度（completed 39/74、103/106、107/110、111/114；stuck_task_id 23/27）。
> - **坐实**：hang 集中在**我新加的 pull collectives + combine/residual 段**（dispatch_pull 28 + combine _pull_routed_y 37 + combine 38 + residual 39），跨层多点。不是 scheduler 全局玄学，是**我的 pull 实现里的 hang**——但同时出现在 dispatch-pull 与 combine-pull（共享 Set/Ge rendezvous + remote_load gather 模式）+ 下游 combine/residual，且 0% clean。
> - **未 root-cause（诚实）**：Set/Ge rendezvous（对齐 ep_all_to_all）+ remote_load（self=local/peer=remote 已修）+ pub_counts-derived offset（数学已验证逐字节等价 push CSR）逐条 reasoning 都没找到死锁点。需 device 级探针：dump pack_done/combine_done signal 窗口值 + 检查 remote_load/barrier 是否真的完成 + pub_counts AtomicAdd 是否在 pull 读前落地（若没落地→off_s/n 越界→remote_load 越界可能 hang）。**下轮先做隔离探针（仿 _probe_barrier_scale 但用 dispatch-shape 的 pull gather + pub_counts）判定，再改代码。勿再整网瞎试（6 次 device run 已烧很多）。**
> - **也要复核 push baseline 是否真 34%**：本 session 没重跑 push GOODKEEP 确认它现在还 ~34% clean（环境可能漂）；下轮先 `cp GOODKEEP` 复跑 3 次确认 baseline，再对比 pull。

> **【续16f（2026-07-15，device — 抬 10× 超时 = 真 deadlock 非 slow；收窄到 pull-gather 并发/资源）】**
> run13 全 push→pull + **抬 10× 超时**（`PTO2_SCHEDULER_TIMEOUT_MS=100000 PTO2_OP_EXECUTE_TIMEOUT_US=450000000 PTO2_STREAM_SYNC_TIMEOUT_MS=500000`，过排序校验）：**~8min（op 450s 预算耗尽）后仍 507018** ⟹ **真 deadlock，不是 slow**（同续15 对 push 的结论；抬超时救不回 = 完成事件永不触发，非时间不够）。∴ **"per-row remote_load 太多导致慢"被排除，chunk-为-提速 不是修法。**
> - **reasoning 排除**：`_dispatch_pull`/`_pull_routed_y` 的 Set/Ge rendezvous 是**对称全屏障**（所有 rank 都 Set+wait，在 data-dependent 分支外）→ 单独不死锁；gather 是**只读已 staged 的 peer 数据**（peer 在 rendezvous 前 task1 已 stage 到自己 HBM），gather 期间无跨 rank 等待 → 逻辑上不该死锁。self=local/peer=remote 已对齐 ep_all_to_all。**逐条 reasoning 找不到逻辑死锁点。**
> - **最可疑（未证实）= 高 read-count 的并发跨卡 remote_load 资源环**：隔离探针 PULL N=42×2 clean，但那是**每 collective 每 peer 1 次** remote_load；`_dispatch_pull` 每层最多 **1024 次** per-row remote_load（8 rank 同时互读）。运行时 peer-access 读路径在高并发/高计数下可能资源死锁（runtime 层，非我 DSL 逻辑）。**这不是 slow（抬超时无效），是资源 deadlock。**
> - **下轮决定性诊断（务必先做，勿再整网瞎试）**：写隔离探针（仿 `_probe_barrier_scale`，但复刻 dispatch-shape 的 pull-gather：rendezvous + 每 (bucket) 多次 remote_load，用合成 pub_counts，无 MoE/权重/exporter），扫 remote_load 计数（1→64→256→1024）找 deadlock onset。若高计数 deadlock → (a) chunk：把每 (loc_e,s) bucket 的 n 连续行合成**一次** remote_load `[t_rows, HIDDEN]`（collectives.py:209 证明多行静态 shape OK；n 动态则按固定 tile 分块+mask），把 per-layer remote_load 从 ~1024 降到 ~非空 bucket 数；(b) 若仍 deadlock → runtime 层跨卡并发读资源问题，上报 simpler team。
> - **状态**：pull 改写完整实现+编译+跑通，但 **device 死锁（0% clean，regression vs push 34%），未达 argmax==303**。root cause 收窄到"pull-gather 高并发 remote_load 资源死锁（疑）"，未坐实。**下轮先隔离探针坐实，再决定 chunk vs 上报 runtime。** GOODKEEP=push 回退基线。
> - **⭐ 最关键洞察（改方向）**：`remote_load` shape 必须**静态**（tp_all_reduce `t_rows` int / ep_all_to_all `[1,d_cols]`）→ 动态 bucket 不能合成一次读。但 **ep_all_to_all 就是同样的 per-row pull，且在 standalone moe.py（device-validated）里干净**——而 standalone 是**每层单独 program**（层间有 `rt.run()` drain 边界）。我的整网是**42 层全在一个 `@pl.program`**、层间无 drain。⟹ **死锁大概率是"单 program 里 42 层 collective 交织、无层间 drain 边界"（续13 论点），不是 push-vs-pull**——这解释了为何转 pull 没用（push 34% clean 是 racy 能完成，pull 0% 是我引入的某个确定性死锁 + 单 program 交织放大）。**下轮 decisive test**：隔离探针跑 N=42 层的 dispatch_pull-pattern（合成 pub_counts、per-row remote_load、rendezvous），看单 program 42 层交织是否必死锁；若是 → 这是 runtime/scheduler 层（单 program 缺 drain），不是我 DSL 能修，需上报 simpler runtime team 或（用户已禁的）多 program；若否 → 我 pull 代码里有确定性 bug（继续 bisect 层数 P_FAITHFUL_MOE_LAYERS=1→2→…找 onset）。**先做 P_FAITHFUL_MOE_LAYERS bisect（cheap，改 env 不改码）：若 P=1 single MoE 层 pull 就死锁 → 我代码 bug；若 P=1 clean、大 P 才死锁 → 单 program 交织。**

> **【续21（2026-07-15，device — pull 实现已提交 GitHub + env 开关 + 挂率修正 + serial-gate 死路）】**
> - **✅ 提交推送**：pull 实现推到 fork `csy0225/pypto-lib feat/whole-net-n1-fusion`（`fc5a269..5b058ca`，6 文件 +1933/-179）：`_gen_faithful_real.py`（best combo pull/push generator）+ `decode_layer.py`（pull 版）+ `_patch_moepy_dispatch.py` + `_patch_combine_pull.py` + `_strip_real_builder.py` + `_regen_mode.sh`。
> - **✅ env 开关 `_regen_mode.sh`**：`N1_DISPATCH/N1_COMBINE={push|pull}` 一条命令切 push+push(GOODKEEP) / pull+push(默认 validated) / pull+pull。3 种模式 regen+py_compile 全过。pypto 每次 run 都重编（~4min），regen 开关与运行时开关成本等价、更干净。
> - **⚠ 挂率修正（推翻"原语无关、白做"的过强说法）**：独立 run clean 率 push+push ~34% vs pull-dispatch ~70-80%（v3/rep1/rep4/fresh_p42 clean，rep2 stall）→ **pull-dispatch 明显降挂率**（续14 的 push-primitive 脆弱性真实存在、pull 缓解了）。但**残余随机挂仍在**（续13 交织）。∴ **两个归因各对一半**：pull 缓解 push-primitive + 残余 = 交织。样本小 + poison 干扰，非严格统计。
> - **⛔ serial-gate 死路**：`PTO2_SERIAL_ORCH_SCHED=1`（`wait_for_orchestration_done_before_dispatch`，runtime_maker.cpp:611 / aicpu_executor.cpp:793）——假设"默认 orch/dispatch 流水→任务在 DAG 建全前被派发→竞态"，enable 后串行化。**device 实测 SER1 fresh 仍 STALL → 不修**。⟹ 残余挂不是 orch/dispatch 流水，是**执行期 collective 交织**（更深）。runtime knob 排除一个。
> - **结论不变**：pull dispatch 改造完成+验证（argmax==303）；残余交织挂 = runtime/scheduler 层（非 DSL/非 orch-gate）；建议上报 runtime team 或 retry-based serving。

---

> **【续20（2026-07-15，device — combine-pull 实验：无单层 bug、不降挂率 → 残余=交织；最佳组合=dispatch-pull+combine-push argmax==303】**
> 用户追问：combine 切 pull 是否代码 bug？挂在哪层？据此把 combine 也做成 **fixed-slot pull（dispatch 的对称反向）**：`_stage_routed_src` 反向 re-pack（expert-major→peer-major routed_src_buf[src*128+within]，pub_counts LOCAL）；`_pull_routed_y` = AtomicAdd barrier + source 本地重算 within（cursor replay）+ compound-scalar offset `my_rank*128+within` remote_load（无跨卡 pub_counts 读）。patch `tools/step3p5/_patch_combine_pull.py`（在 dispatch patch 后跑）。device 实测：
> - **combine-pull P=1 = RUN_CLEAN** → **无单层代码 bug**（offset/order 数学已静态复核一致）。
> - **combine-pull P=42 = 随机**（cprep2=303/cprep4=303 clean；cprep1=STALL；cprep5=COMMINIT）→ clean 时出 303、随机挂，**与 combine-push 相当，不降挂率**。
> - ⟹ **combine push↔pull 不改变 P=42 挂率**。所有组合（push+push / pull-disp+push-comb / pull+pull）都 **P=1 clean / P=42 随机挂** → 残余 = **42 层单-program collective 交织**（续13），与原语无关。
> - **挂在哪层？不是固定层**：历史 device log（续16d 同 0234）一次 P=42 挂里不同核卡在不同深度（completed=39/41 vs 67/70）= racy 调度漂移，非确定单层 bug（与 P=1-clean 一致）。本 session fresh sweep（P=10/20/30/41）被**卡毒化全 COMMINIT** 阻塞（+device stall-diag 不落盘），未拿到 fresh onset。
> - **最佳组合（已回退到 tree）= dispatch-pull + combine-push**（`MOEPY_PULL_303_20260715_174703` checkpoint，argmax==303 已验 3/3 clean）。combine-pull WIP 在 `_patch_combine_pull.py`（可复现）。
> - **建议**：残余交织挂是 runtime/scheduler 层（确定性跨 rank 派发序 / 层间 drain）→ 上报 simpler runtime team；非 DSL 原语能修。**pull 改造（dispatch）已完成+验证。**
> - **环境 friction**：507018 挂毒化卡→下 run comm_init（~8min 自清，可靠性循环高估挂率）；AICPU device stall-diag 不落 `$HOME(/root on container)/ascend/log/debug/device-*/`（force-reset 抢在 flush 前）。

---

> **【续19（2026-07-15，device — ✅ pull dispatch 修好 + argmax==303 验证通过；残余 P=42 随机 stall = 更深的 42 层交织，非 pull-dispatch）】**
> 按续18 的 5 步把 dispatch port 成 moe.py fixed-slot pull（patch 脚本 `tools/step3p5/_patch_moepy_dispatch.py`，regen-clean，py_compile 过），combine revert 回 push（moe.py 组合）。**device 实测**：
> - **P=1 = RUN_CLEAN**（FULLPULL 原是确定性 STALL）→ **dispatch 确定性死锁修好**（坐实）。
> - **P=42 canonical golden = `argmax==303`==vLLM，RUN_CLEAN，3 次独立 clean run（v3+rep1+rep4）全 303** → **pull 改造正确性验证通过**（canonical §1 判据）。
> - **P=42 残余 = 随机 stall**（rep2/rep3 挂；rep4 在其后 clean+303 = 毒化已清）。一次 507018 会毒化下一 run（comm_init 失败 / 级联）→ 循环测的挂率被高估；真挂率≈push baseline(~34% clean)。**此残余不是 pull-dispatch**（P=1 确定性 clean；每个 clean 的 P=42 都出 303）——是**更深的 42 层单 program collective 交织**（续13 论点：一个 `@pl.program` 层间无 drain），push+push 的 GOODKEEP 同样 ~34% clean → **与 dispatch/combine 原语选择无关**。stuck-func 未抓到（本 session `ASCEND_PROCESS_LOG_PATH` device log 没落盘，是待解的抓取 friction；结论靠 P=1-clean + 3/3-clean→303 的强推断）。
> - **踩坑**：507018 stall 毒化卡 → 下一 run comm_init 失败（`_ensure_comm_base failed 8/8`），过一会自清（rep4 clean）。可靠性循环要在 run 之间做卡恢复，否则毒化级联。SSA 坑：`pl.load`→Tile[Mem.Vec] vs `remote_load`→plain Tile，self/peer 拷必须用不同变量名（`sxt/sst/srt` vs `xt/st/rt`）。
> - **落地物**：checkpoint `decode_layer.py.MOEPY_PULL_303_20260715_174703` + `_gen_faithful_real.py.MOEPY_PULL_303_*`（argmax==303 工作态）+ `_patch_moepy_dispatch.py`（可复核 patch）。GOODKEEP=push baseline。memory `n1_pull_dispatch_must_align_moepy_fixedslot` 有完整证据链。
> - **下一步（若要 P=42 每次都 clean 的鲁棒性）**：残余交织 stall —— (a) 把 combine 也 port 成 fixed-slot pull（对齐 dispatch）看挂率是否降；或 (b) 若确认是 runtime 层单-program 交织 → 上报 simpler runtime team（确定性跨 rank 派发序 / 层间 drain），非 DSL 原语能修。**pull dispatch 本身已完成验证。**

---

> **【续18（2026-07-15，goal-session — pull 死锁根因锁定 = 偏离 device-validated moe.py；修法定案 + 已入 memory；port 未落地）】**
> 用户重启 pull 方向（"push 握手问题，pull 能解决么？之前 pull 也有 bug，修好并验证，别重复造轮子"）。本 session **只做诊断 + 修法设计**（未 device 验证 argmax==303，不得声称完成）。
> - **device 复现**：恢复 FULLPULL（`decode_layer.py.FULLPULL_20260715_054234`）跑 canonical §1 `P_FAITHFUL_MOE_LAYERS=1` → **507018 STALL（确定性，单 MoE 层就挂）**。确定性 ≠ push 的随机 ~66% → 是 pull 实现的**确定性代码 bug**。
> - **根因（2 个对抗 agent + device-validated `moe.py ep_all_to_all` 对照，决定性）**：FULLPULL `_dispatch_pull`（generator `FRESH_DISPATCH_PULL_INT8`）把 device-proven 的**对称 fixed-slot a2a** 改成了会挂的**自造 fused CSR gather**，三处偏差：① barrier `Set`（moe.py 用 `AtomicAdd`）；② gather 上界运行时 `pl.range(n=读跨卡 pub_counts)`（moe.py 静态 `pl.range(T*TOPK=128)`）；③ remote_load offset 跨卡 `pub_counts` 逆推（moe.py compound-scalar `my_rank*128`/`peer*128` fixed-slot）。**静态上界要求 fixed-slot 打包——没有捷径。**
> - **排除的错误方向（勿再查）**：`Set` 单独不是 bug（`collectives.py:538` ep_all_to_all 也用 Set+Ge）；`pub_counts` 确实清零（`comm_hccl.cpp:1059 aclrtMemset`）。
> - **device-validated `moe.py` 组合 = pull-dispatch + push-combine**（moe.py combine 本身就是 push，只抖动不 stall、不翻 greedy argmax）。∴ 修法 = **从 GOODKEEP 出发，只把 dispatch 换 moe.py fixed-slot pull，combine 保持 push**（≠ FULLPULL 的 full-pull，那个 combine 也 pull 且 0% clean）。
> - **落地 5 步（从 FULLPULL 脚手架改，已有 send 窗口 + splice；memory `n1_pull_dispatch_must_align_moepy_fixedslot` 有 moe.py 行号）**：
>   (a) `_dispatch_pack_publish` 打包基址 `send_offsets_rank[r]`(prefix-sum) → **固定槽 `r*n_routes_per_rank`**（对齐 moe.py `_pack_send_payload`）；
>   (b) send_x/send_scale/send_route 窗口 `n_routes_per_rank`(128)→`local_recv_max`(1024)：generator sig + host_orch buf 尺寸 + window shape。**注意 routed_y_buf 保持 n_routes_per_rank（combine 窗口，128 正确）**；
>   (c) `_dispatch_pull`：barrier `Set→AtomicAdd` + 删 fused-CSR gather 换**静态 peer-major pull**（self 块 `pl.load` 本地拷、peer 块 `remote_load(send_x, peer, [my_rank*128+r,0])` 静态 `pl.range(128)` → recv_x[`peer*128+r`]，recv_x 变 **peer-major**）；
>   (d) `_dispatch_stage`：直拷 → **re-pack**（peer-major recv_x → expert-major local_routed_x，用 pub_counts，镜像 moe.py:960-993）；**需给 `_dispatch_stage` 加 `pub_counts` 参数**（base `decode_layer.py` + generator `_stage_edits`/`_scall` 同步）——最耦合的一步；
>   (e) **combine 回 push**：从 FULLPULL 删 `FRESH_COMBINE_PULL` splice + 恢复 base `_push_routed_y_to_sources` 调用（或直接从 GOODKEEP 起、只加 dispatch-pull）。
>   → regen → py_compile → device 验 P=1 clean（多跑）→ 放开 42 层 → **argmax==303**。
> - **备份/回退**：`decode_layer.py.GOODKEEP`（push baseline，~34% clean argmax==303）、`.FULLPULL_20260715_054234`（0% clean full-pull）、generator `.FULLPULL_20260715_054234` + `.bak.pre_pulldispatch_20260715_123346`。**本 session 收尾 tree+generator 均回 clean push baseline（GOODKEEP），exporters EXP=8 RDY=8 常驻。**
> - **纪律**：0234 上 a2a3sim 坏了无快速编译检查，每轮 = 整网 device ~5min；`_dispatch_stage` re-pack 加 pub_counts 是最易出错处，改完先 py_compile 再 device。

---

## ⭐ 下 SESSION 主线：落地 dispatch `push→pull`（用户 2026-07-15 拍板）

**背景已定案**：`_dispatch_push`（func28）跨卡 PUSH（`remote_store`/TPUT）写完成间歇不触发 →
S1 hang；抬超时 3/3 无效（续15）。修法 = 把 dispatch 的前向 scatter-by-push 改成
consumer 侧 gather-by-`remote_load`（PULL），因为**读的完成在本地可观测、写的完成在远端不可观测**
（原理 + 性能权衡见 [`N1-PUSH-DSB-TIMING.md`](N1-PUSH-DSB-TIMING.md)）。PULL pattern 已被
tp_all_reduce 与隔离探针 `PUSH=0`（`remote_load`）device 证明 N=42×2/N=2×4 全 clean。

**⚠ 一切验证走 [`N1-CANONICAL-TEST.md`](N1-CANONICAL-TEST.md)**：唯一程序 `whole_decode_faithful_real`、
真 token 6127、`argmax==303` 才算 PASS（`RUN_CLEAN≠PASS`）；改代码前先起两 agent 对抗审设计。

### 执行步骤（每步都能独立 device 验证）
1. **先 dispatch，后 combine**：device 确认 stuck = dispatch 的 `_dispatch_push`（stall 根因）。
   combine 的 push（`push_routed_y_to_sources`）是 residual jitter 来源（见 memory），作为第 2 步。
   两者都转 pull 才彻底，但先修 dispatch 解 stall。
2. **激活 inverse_map**（generator `tools/step3p5/_gen_faithful_real.py` + `decode_layer.py`）：
   `_build_inverse_map` 现有 **10 处 def / 0 处 call（dead）**。先接线——orch 里 alloc
   `inverse_map[BATCH,TOPK] INT32` 并 `self._build_inverse_map(...)`；`packed=inverse_map[t,k]`，
   `dst_rank=packed//LOCAL_RECV_MAX`、`dst_row=packed%LOCAL_RECV_MAX`。
3. **source 侧改本地 peer-readable window**：dispatch 输入 token 从"A `remote_store` 写到
   B.recv_x"改为"A 把自己的 token 写进**自己本地**的 `send_x` window（`pld.DistributedTensor`，
   peer-readable，纯本地写、本地 fence）"；`recv_scale`/`recv_r_route` 对应改 `send_scale`/`send_route`。
   +~7MB/layer（pool 自增，64GB HBM 够，见 [[n1_comm_window_bytecap_refuted]]）。
4. **read-前 barrier**：一道 notify/wait rendezvous，保证所有 source 已写完各自 send window。
5. **dest 侧 gather-by-pull**：dest rank 需"哪些 source 的哪些行给我"——把 combine push 里 dst 侧
   已读的源路由表（`src_route_table`/`pub_counts`）镜像成 peer-readable，dest 据此算 `(src_rank,src_row)`，
   `remote_load(send_x, peer=src_rank, offsets=[src_row,0])` 拉进 dest 本地 recv_x（scale/route 同理）。
   删掉 `_dispatch_push` 的 `remote_store` + count_done/data_done 的 push notify。
   ⚠ dispatch 是**前向 scatter**（源知道目的地），改 pull 是**反向 gather**（比 combine pull 难，
   combine 天然 gather）——这一步是主要工作量。
6. **compile → 8 卡 device → golden**：a2a3sim 在 0234 坏（g++-15 缺）→ 无快速 compile check，
   每轮 = 全 device run（~5min w/ `--reuse-exporters`）。PASS 判据：不再 stall（多跑几次）**且**
   `argmax==303`。先用隔离探针 `PUSH=0`（已 clean）锚定 pull pattern 正确，再上真程序。

### 注意 / 备份
- 改前备份：`_gen_faithful_real.py.bak.pre_pulldispatch_*` + `decode_layer.py.bak.pre_pulldispatch_*`。
- generator 会静态 unroll 42 层 → inverse_map/send window 要 per-layer distinct 命名（`_L{pos}`），
  否则撞 comm-domain 重名 alloc（同 whole-net per-layer window 规则）。
- **不可 work around**：不得用抬超时/塞假输入冒充通过；golden 必须 `argmax==303`。

---


> **⛔⛔⛔ 续14（2026-07-15，device 确认，权威当前状态——覆盖续13/续12 中与本段冲突的一切结论。本 session 收尾。用词严谨、勿误导。**
>
> **【一句话结论】** A2 挂死（507018 / S1:running-stalled）的**根因 = 跨卡 PUSH 原语 `pld.tile.remote_store` / `pld.tensor.put`（两者都 lower 到 TPUT = MTE3 跨 die 远端写）的写完成间歇不触发**（kernel 挂在 `wait_flag(MTE3)`）。**精确挂点（device 确认）= dispatch 的 `_dispatch_push`（func_id 28）**。修法 = 把 dispatch 的 push 改成 **pull**（用 `remote_load`/TGET = MTE2 读，已 device 证明可靠）。**本 session 完成：根因去-confound + 精确目标的 device 确认 + 修复方案设计 + 排除所有捷径。修复代码未实现（是较大的 dispatch a2a 重写，留下 session）。**
>
> **【⭐ 下 session 执行顺序（用户 2026-07-15 拍板，务必按此序）】**
> 1. **先试 3 个超时环境变量**（见下方表）——把 `SIMPLER_SCHEDULER_TIMEOUT_MS`/`SIMPLER_OP_EXECUTE_TIMEOUT_US`/`SIMPLER_STREAM_SYNC_TIMEOUT_MS` 调大（如 ×10），整网 P=42 跑几次，**观察状态有无改善 / 能发现什么**。判读：(a) 若 `_dispatch_push` 竟能完成、stall 消失 → 说明是"慢而非真 hang"，**改变定性**，可能有比 pull 更轻的修法，需重估；(b) 若仍在 `aiv0:28` 挂 → **坐实真 hang**，进第 2 步。⚠ 这一步是**诊断/观察**，不是修复；调大 timeout 当"修复"上线是 work-around（SKILL/wiki 禁）。
> 2. **看情况再落地 pull 修改**（下方"修复计划"）——只有第 1 步坐实真 hang（或虽有改善但仍不稳定）时才做 dispatch push→pull 大改。
>
> **【device 证据链（可复现，别再走弯路）】**
> - 隔离复现器 `tests/step3p5/_probe_barrier_scale.py`（本 session 新增，framework-only，无 MoE/权重/exporter；env 旋钮 `CHAIN`/`PUSH`/`PREBAR`）：
>   - **PULL**（`remote_load`/TGET，= tp_all_reduce 用的原语）：链式 `CHAIN=1 BARRIER_ONLY=0 N_COLL=42` ×2 + `N=2` ×4 **全 CLEAN，0 挂** → **tp_all_reduce 彻底洗清**。
>   - **PUSH**（`remote_store`/TPUT，= combine/dispatch 用的原语）：**随机挂**。⚠**去-confound 关键一步**：最初 PUSH 用动态首维窗口 `[NR,SIZE]`（`NR=pl.dynamic`，SKILL §3/§4 说动态首维跨函数丢父 stride），一度怀疑挂是这个 artifact；**改成静态窗口 `[8,SIZE]`（`_bscale_gen.py` 确认生成 `DistributedTensor[[8, SIZE]`，与整网 combine 静态窗口 `[128,HIDDEN]`、dispatch recv_x 同构）后仍随机挂：`CHAIN=1 PUSH=1 N_COLL=2` → 1clean/3STALL、`N=42` → STALL×2**。⟹ 动态窗口 artifact 排除，**`remote_store` 是真脆弱**。
> - **整网 P=42 抓 stall + 读 stuck kernel 名**（`--reuse-exporters` + `ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_PROCESS_LOG_PATH=<dir>`；harness 已设 `simpler` logger level 15）：device log `<dir>/debug/device-*/device-*.log` 8 卡同一 stuck task：`state=RUNNING fanin_refcount=6/6 kernels=[aic:-1 aiv0:28 aiv1:-1] running_on=[cores=[core=24/26/28(aiv0)]]` → **`aiv0:28` = func_id 28 = `_dispatch_push`**（查 `build_output/…/next_levels/full_moe_chip_orch/kernel_config.py`）。⚠**host 侧 `stuck_task_id=(1,23)` 的 23 是 runtime task_id、≠ func_id**——读 kernel 名必须看 device log 的 `kernels=[aiv0:N]` 再查 kernel_config 的 func_id N。**续13"(1,23)=tp_all_reduce"是没核实的错误映射。**
> - **clean run（~34%，A2 是随机 ~66% 挂）**：`--hidden-token 6127` → **argmax=303 == vLLM golden**。∴ 精度本身对，**唯一 blocker = 这个随机 push 挂**。
>
> **【被本段推翻/修正的旧结论——勿再据此行动】**
> - 续13「N=1 内没有可靠的 in-kernel 跨卡同步原语（dsb 只 drain 本地 / 核间 cache 非一致 / HcclBarrier 只 host 级…）」→ **证伪**。原语存在且有 fence：`pld.system.notify/wait` → `TNotify.hpp`/`TWait.hpp` 带 `dcci`+`dsb(DSB_DDR)`+`pipe_barrier(PIPE_ALL)`（grid_intrinsic.hpp:455 的"核间非一致"注释在 `grid_mock` 里，是说 mock 需要 dcci、不是说没 fence）。**PULL 路径 device N=42 全 clean** 即证。问题不是"没原语"，是**某个特定原语（TPUT 远端写）脆弱**。
> - 续12/续13「A2 = 手写 tp_all_reduce 单波 barrier 缺完成波；补 Phase-4 completion-wave 已修；argmax=303 复现 3/3；P=42 确定性 RUN_CLEAN」→ **证伪**。tp_all_reduce（PULL）被复现器洗清；真凶是 **dispatch 的 PUSH**（`_dispatch_push`），与 completion-wave 无关。argmax=303 只在 ~34% clean run 出现，不是 completion-wave 修好的。
> - 续13「stuck_task_id=(1,23)=tp_all_reduce」→ **修正**为 `_dispatch_push`（func_id 28）；task_id≠func_id。
> - 续12「残余抖动定位到 combine 跨卡 `routed_y_buf` gather」→ 部分成立但需正名：combine 的 push（`_push_routed_y_to_sources`）也是 TPUT（同类脆弱），是**抖动**源；但**挂死**在 **dispatch 的 push**。两者**同根（TPUT 远端写）、同修法（push→pull）**。
>
> **【本 session 试过并 device 实测失败的修复——已回退，勿重试】**
> 1. pto-isa `TPut.hpp` 单传输路径（`TPUT_IMPL` 281-289）补 `pipe_barrier(PIPE_ALL)+dsb(DSB_DDR)`：仍 4clean/3stall。挂在 `wait_flag(PIPE_MTE3…)` 本身，补在其后无用。**已回退**。
> 2. pre-push rendezvous barrier（push 前加一道全屏障，2nd signal window）：**更差 5/5 挂**。
> 3. `pld.tensor.put` 换 `remote_store`：**不是替代**——两者都 lower 到 TPUT（MTE3），`pld.tensor.put` 非 SDMA。`_dispatch_push` 现在 recv_x 用 `pld.tensor.put`、recv_r_route 用 `remote_store`，都是 TPUT。
> ⟹ **不是 fence/屏障顺序问题，是 TPUT 远端写完成本身间歇不触发。只有换成 pull（远端读）才可靠。**
>
> **【pypto 专家给的 3 个超时环境变量（第 1 步先试；定义在 runtime `runtimeout_config.h:23-25`）】**
>
> | 环境变量 | 单位 | 含义 | 默认 (onboard) |
> |---|---|---|---|
> | `SIMPLER_SCHEDULER_TIMEOUT_MS` | ms | AICPU 调度器无进展看门狗 | 10 s |
> | `SIMPLER_OP_EXECUTE_TIMEOUT_US` | μs | STARS op-execute 超时 | 45 s |
> | `SIMPLER_STREAM_SYNC_TIMEOUT_MS` | ms | host 侧 stream 同步超时 | 50 s |
>
> 试法：整网 P=42 把三个都调大（如 `SIMPLER_SCHEDULER_TIMEOUT_MS=100000 SIMPLER_OP_EXECUTE_TIMEOUT_US=450000000 SIMPLER_STREAM_SYNC_TIMEOUT_MS=500000`）跑 3-5 次，看 stall 率 / 是否完成 / device log 有无新信息。判读见上"执行顺序"第 1 步。
>
> **【修复计划（第 2 步、条件性）= dispatch `_dispatch_push` push→pull】**
> 现状 `_dispatch_push`（`decode_layer.py:2678`，生成自 generator base builder）：逐 (t,k) token 从跨卡可读 `pub_counts` 算 `dst_row`，然后 `pld.tensor.put`(recv_x) + `pld.tile.remote_store`(recv_r_route) **推**到 peer dst。改 pull（每步都要 device 验证 stall 率 66%→0 且 argmax=303）：
> 1. **加本地 pack**（无跨卡）：每卡按 bucket=`dst_rank*N_LOCAL_EXPERTS+loc_e` 顺序把自己的 token pack 进 `send_buf`（用 per-bucket cursor，逻辑见 `dispatch.py:pack_send_payload`）。
> 2. **`send_buf` 变 peer-readable `pld.DistributedTensor` window**（host_orch alloc，尺寸 `[BATCH*TOPK=128, HIDDEN]`，比 recv_x[1024] 小）；recv_x 变本地输出（不再是 push 目标）。
> 3. **一道 barrier**（所有卡 pack 完再拉）。
> 4. **dst pull**：`for loc_e: for src S: off = Σ_{(dst',e') 在 bucket 顺序里 < (my_rank,loc_e)} pub_counts[S*N_RANKS+dst', e']; n = pub_counts[S*N_RANKS+my_rank, loc_e]; for row in range(n): tile = pld.tile.remote_load(S.send_buf, peer=S, offsets=[off+row,0]); pl.store(tile,[recv_slot,0], recv_x_local); recv_slot += 1`。**关键简化：`off` 完全由已有的跨卡可读 `pub_counts` 重算，无需新增任何跨卡发布**（pack 顺序=bucket 顺序、bucket 计数=各卡已发布的 pub_counts，所以 dst 能算出每个 src 的 send_buf 偏移）。
> 5. **recv_r_route**：本地可重算（r_route 由 pull 顺序推）或同法 pull；**recv_scale**（INT8 dispatch 的 per-token scale）同法 pull。
> 6. 改在 generator `tools/step3p5/_gen_faithful_real.py` 的 base builder（`_dispatch_push`/`dispatch_step` + host_orch window alloc）→ regen。
> 7. **combine 的 push（`_push_routed_y_to_sources`）同类问题（TPUT），同法改 pull**（combine 侧有 `inverse_map[t,k]=dst_rank*LOCAL_RECV_MAX+dst_row`，但整网里 `_build_inverse_map` **定义了却 0 调用=dead code**，要 pull-combine 得先激活它、或同样用 pub_counts 逆推）。combine 的 push 目前是**抖动**源、dispatch 的 push 是**挂死**源；先修 dispatch（挂死），combine 顺带修（消抖动）。
> ⚠ 0234 上 a2a3sim 坏了（`g++-15` 缺）**无快速编译检查**，每轮只能整网 device 跑（~5min/轮，用常驻 exporter `--reuse-exporters`）。备份在 `pypto-lib/{tools/step3p5/_gen_faithful_real.py,models/step3p5/decode_layer.py}.bak.pre_pullcombine_*`。**改根因、不 work-around；诊断脚手架（复现器、旋钮）不进产品路径。**
>
> **【环境 / 恢复 runbook（本 session 踩坑记）】**
> - 0234 通过 `brainctl rjob launch --charged-group=hw910test --host-network=true --privileged=true --cpu 60 --gpu 8 --private-machine=group --positive-tags A910X,node/gpu-a910x-0234.host.platform.shaipower.com --volume /data/chensiyu:/data/chensiyu --volume /home/chensiyu:/home/chensiyu --mount juicefs…chensiyu-jfs --mount juicefs…hw910test-jfs --image hub.i.basemind.com/stepcast/stepcast:0.19.0-081dd47dd175-fbfe288fe1ee-2026.06.09-141938 --entrypoint "" -- bash`（在 b-csy 上跑，落进 tmux 窗口成 0234 容器 shell）。
> - **卡死 shell 恢复**：Ctrl-C 中断正挂死的 device run 会残留一个卡在 ACL 调用的进程（同续13），C-c/C-\/pdb 都唤不醒。`brainctl exec`/`brainctl stop` 对 replica/rjob **被 RBAC 挡**（ns=shai-core，kube 用户无权）。**可用恢复 = 在 b-csy 上 `kill -TERM <'brainctl rjob launch …0234' 客户端 PID>`**（干净结束交互 rjob → pod 回收 → 0234 卡释放，非孤儿；本 session 验证：kill 后新 worker 又调度回 0234），再用上面同样命令原样重开。exporter 会丢（barrier 复现器不需要 exporter；整网需重载 ~15min）。
> - 三件套激活（每 fresh shell）：`export WS=/data/chensiyu/hw_project/pypto/workspace; source /usr/local/Ascend/cann/set_env.sh; source $WS/activate.sh; export PTO_ISA_ROOT=$WS/pto-isa; export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib:$PYTHONPATH`。8 卡 env：`PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
> - exporter：常驻 hold-mode（`--export-rank r --dev r --kv-ipc --out /tmp/n1_weight_ipc` ×8，~15min 冷载写 `ready.rank{r}` 后 hold）→ worker `--reuse-exporters` 秒级 attach。收尾释放：`touch /tmp/n1_weight_ipc/STOP`。**本 session 收尾时已 STOP 释放 8 卡**。
>
> **【buffer 命名核查（此前用户问）= 干净】** generator 静态展开 `for pos: sfx=L{pos}`，comm/signal/hidden buffer 全逐层 distinct（每 MoE 层 12 window + 5 signal 1:1 collective + 84 个 `h_moe_L{pos}`/`resid_hold_L{pos}`），无复用，满足设计不变量 P3/ADR-013。**不是本 bug。**
>
> **【memory 索引】** `n1_a2_primitive_exists_not_missing`（全证据链 + 修复 recipe + 恢复 runbook）；`n1_m4_accuracy_gap_converged_direction_drift`（历史，其中"A2 已修/completion-wave"部分被本段作废）。`blockers.md` A2 段已同步。

> **⛔⛔ 续13（2026-07-14 goal-session-2，device 复验）——历史段落。其中「没有可靠 in-kernel 跨卡同步原语」「A2=tp_all_reduce 缺完成波已修」「stuck_task_id=(1,23)=tp_all_reduce」等结论均已被上面续14 device 证伪/修正，仅作历史保留，勿据此行动。用词严谨：A2 未修复。本段覆盖续11/续12 中一切"A2 已修 / argmax=303 复现 3/3 / P=42 确定性 RUN_CLEAN"的结论。**
>
> **【最重要 · 纠正，勿再被误导】** 续12 与 commit `fc5a269` message 声称"A2 collective 死锁已修（completion-wave），P=42 不靠 logging 确定性 RUN_CLEAN 7 次、argmax=303 复现 3/3"。**本 session device 复验推翻**：clean `fc5a269` tree（completion-wave + combine 修复都在、工作区干净、17+10 marker 确认）、warm healthy exporters、**不开任何 logging**：`P_FAITHFUL_MOE_LAYERS=42 --hidden-token 6127 -d0-7` → **run1 STALL 507018 / run2 CLEAN argmax=303 / run3 STALL** ≈ **~66% 挂**。⟹ fc5a269 的 completion-wave **并未可靠修复 A2**；续12/commit-msg 的"device 验证 3/3"是**不可复现的 premature-victory**。⚠⚠ **本 session 一个 adversarial agent 因为读了 commit message 的"device 验证"就误判"A2 已修"**——所以下 session **铁律**：任何 commit message / memory / 文档里"已修 / device 验证"的声明，**必须先在 device 上复验（多次、无 logging）再据此行动**。
>
> **【A2 可靠根因（DFX 工具 + 多轮正反 agent 交叉质疑 + device 坐实）】** A2 = **跨卡 collective 时序竞态（Heisenbug）**，每个 collective 从第一个就潜伏（符合"顺序执行、一开始就有时序问题"），不是"数量并发争抢"。
> - **非资源**：DFX `scope_stats`（42 scope×8 rank）task_window/heap/dep_pool/tensormap 全 <1% 容量、dropped=0。
> - **非阈值/非某层**：sweep P=20 0/3、P=35 1/3、P=41 1/4、P=42 2/3（stochastic）。
> - **非并发**：collective 数据依赖串行，任一时刻仅 1 个在跑。
> - **本质差别（N=1 vs 多 program）**：多 program（DeepSeek/Qwen）每次 `rt.run()` dispatch 之间有 **runtime execute() drain 完成边界** + 层间可调 host 级 `comm_barrier`（`HcclBarrier`，厂商正确同步）；**N=1 单 program 层间没有这道边界**，in-kernel 只有 racy 的 shmem `TNotify/TWait`（`dsb(DSB_DDR)` 只 drain 本地、不保证跨 HCCS 落 peer HBM；作者自注 `grid_intrinsic.hpp:454` 核间 cache 非一致）。**N=1 内没有可靠的 in-kernel 跨卡同步原语**：`pld.tensor.barrier` 也 lower 到同一 racy shmem；`SyncAll/ffts_cross_core_sync` 只跨核不跨 die；数据路径无 `HcclAllReduce`；`HcclBarrier` 是 host 级只能多 program 用。
>
> **【本 session A2 修复尝试 + 确切 device 结果（全部已回退到 clean fc5a269 基线；勿重试这些死路）】**
> - completion-wave（fc5a269 基线已含）：~66% 挂。
> - `TNotify.hpp` read-back（dcci+reload+dsb 跨 die publish，纯 header）：66%→~25%（改善非根治；**非单调=Heisenbug 签名**）。
> - notify `AtomicAdd→Set`（transform `_set_notify_tp_allreduce.py`）：更差 ~100%。
> - `Set + poll-until-landed`（TNotify Set 分支 spin 至 100M）：更差；**且 100M in-kernel spin 会把卡 near-hang——本 session 0234 shell 卡死就是它导致的。铁律：禁大 in-kernel spin。**
> - 框架 `pld.tensor.allreduce` intrinsic（transform `_swap_tp_allreduce_to_intrinsic.py`）：**UB 溢出**（一次 reduce 整个 [16,4096] 不 chunk，262144B>188416B），不能用。
> ⟹ **DSL/ISA 层修不掉（只是扰动 Heisenbug）。修法必须在 runtime 层，或走多 program。**
>
> **【下一步（需用户先定方向 + 恢复环境；按依赖）】**
> 1. **恢复环境**：0234 tmux `pypto-ascend-0:0` shell 被残留 device 进程卡死（poll-fix 的 100M spin 所致），从 b-csy 进不去（ssh 无权限；b-csy venv python 断链跑不了 pypto；0162 在 stepfun/develop 分支非本 track）。**需在 0234 手动 `pkill -f _stage_whole`（SIGTERM，勿 -9），可能需 card reset**，并查 exporter（EXP=8）/卡 HBM。
> 2. **定方向（三选一）**：(A) **上报 simpler/pto-isa runtime team**——要 in-kernel 可靠跨 die 完成/同步原语，或确定性跨 rank collective 派发序（`scheduler_cold_path.cpp:248` fanin-driven 会跨 rank 错位）；(B) **环境恢复后跑隔离复现器** `tests/step3p5/_probe_barrier_scale.py`（N 个纯 barrier collective、无 MoE、barrier-only vs full × N=20/42/84）精确判定 scheduler-ordering vs data-movement，再定 runtime 修法；(C) **重估 N=1 vs 多 program**（多 program 是框架对"多层多 collective"的原生答案 = 天生有 drain 边界 + 可用 HcclBarrier）。
>
> **【精度状态（勿误解 / 勿夸大）】** 当 P=42 偶尔跑通时 argmax=303==vLLM golden（L2 attn_layer_idx `7294e26` + INT8 routed + combine 有效）。但"跑通"是**间歇**的、**不可复现**，**不能宣称 M4 token-exact 达成**。唯一 blocker = A2 间歇挂死。M5/M6 serving 集成**完全未开始**。
>
> **【本 session 落地物】** 代码：无（所有 A2 实验已回退到 clean fc5a269）。新增待用脚手架（未 commit，留参考）：`tests/step3p5/_probe_barrier_scale.py`（隔离复现器）、`tests/step3p5/_stage_whole_faithful_real_ipc.py` 的 `N1_DFX=dep,scope` env-gated DFX hook、3 个 transform（`_swap_tp_allreduce_to_intrinsic.py`/`_set_notify_tp_allreduce.py` = 死路；`_add_allreduce_completion_wave.py` = fc5a269 已用）。文档：本续13 + `blockers.md`（A2 ACTIVE 段）+ memory `n1_m4_accuracy_gap_converged_direction_drift`（全证据链，含"OVERTURNED 续12"段）。
>
> **【准则（用户反复强调，作为硬规则，不可再犯）】**
> 1. **ready 只认 live-token-exact-device-reproducible**；任何"已修/device验证"声明先 device 复验再行动（本 session 血的教训：连 agent 都被 commit-msg 骗）。
> 2. **不可 work-around**：诊断脚手架不进产品路径；禁大 in-kernel spin；不用 BF16-dequant，坚持原生 W8A8。
> 3. **遇问题先查框架设计约束**（`pypto_top_level_documents` + `.claude/skills/pypto-dev-constraints`），没违反再查代码逻辑，**整体复查再 debug**。
> 4. **两个 agent 正反质疑得可靠结论**，且**必须交叉核对 agent 结论是否被 stale 文档误导**。
> 5. **先设计后编码；分清模型每层交接边界，避免陷入局部错误反复**。
> 6. **golden 标准要正确**（oracle=vLLM eager dump；synthetic 会 stale）；可先跑单 batch。
>
> ---
> （以下续11/续12 及更早段落中关于"A2 已修 / argmax=303 复现 3/3 / 确定性 RUN_CLEAN / 残余仅 combine 抖动"的结论**均已被上面续13 device 复验作废**，仅作历史保留，**勿据此行动**。）

> **✅ 续11（2026-07-14 本 session，device 验证）— A2 死锁修复 + argmax=303 复现 3/3；残余 logit 抖动已定位但 reference 修法在流水线里死锁（已回退）。**
> **两个 root-cause 修复（对齐框架/moe.py，无 work-around，均在生成的 decode_layer.py，未 commit）**：
> 1. **A2 collective 死锁 = 手写 `tp_all_reduce` 缺完成波（single-wave）**。wiki `S1:running-stalled` = collective kernel 挂死（非容量）。框架 `pld.tensor.allreduce` 是 two-wave。修：全 17 份 `tp_all_reduce` 加 Phase-4 完成波（`notify+wait≥2`，transform `tools/step3p5/_add_allreduce_completion_wave.py`）。**device：P=42 确定性 RUN_CLEAN、无 507018、不靠 logging（7 次干净）。多 session 的硬 blocker 解除。**
> 2. **argmax 非确定 = combine zero-vs-push 竞态**（丢了 moe.py `pub_route_barrier`）。修：`combine_done` 上补 zero-done 波（`tools/step3p5/_fix_combine_zero_push_race.py`）。**device：argmax=303==vLLM golden 复现 3/3（修前 303/20/303）。→ L3 greedy top-1 准出线达标。**
>
> **残余 = L2 logit 分布抖动（top-303 logit 8-14，margin 最薄 0.9）**：device 二分 P=0(dense) 逐 bit 相同、P=1(单 MoE 层)抖 → 单 MoE 层引入、跨 42 层累积。**根因（reference 确认）= 整网丢了 moe.py `_serialize_after_shared`**（强制 shared-expert tp_all_reduce 先于 routed dispatch/combine，防 collective 重叠）。**但插入它 device 实测 P=42 死锁 3/3（507018）**——它把 42 层流水 schedule 过约束（moe.py 只在单块非流水里安全）。**已回退。**
> ⟹ **可靠结论：残余抖动 与 A2 死锁 是同一「流水化 collective 调度脆弱性」的两面——不能靠加 per-layer 串行化消除抖动而不重新引入死锁。**
> **⭐ 残余抖动已 vector-diff 精确定位（device，本 session）**：P=1 逐 stage row0 向量 ×2 offline L1/cos 对比（`logs_n1/_vec_diff.py` + harness `N1_DUMP_DIR`）：**resid_hold（attention 残差）L1=0 cos=1.0 完全确定**、**moe_out（combine 输出）L1=113 cos=0.990 DIVERGE** → **抖动源 = COMBINE（`moe_out=_weighted_gather_and_add(routed_y_buf,w,sh_y)`），不是 attention**（推翻早前 attention-residual 假设）。max-based 探测看不到（各 stage max 全同），必须 vector-diff。已有的 combine 修复（zero-done+self-fence+completion-wave）修好了 argmax(303) 但残留 ~1% moe_out 向量竞态。**下 session 精确步骤**：dump routed_y_buf(gather 后) 向量 vs local_routed_y 向量 ×2，已判定=(b) 跨卡 routed_y_buf gather：local_routed_y(INT8 routed 输出) 向量 L1=0 完全确定、resid_hold L1=0、sh_y/gate/routing 确定，唯 moe_out DIVERGE ⟹ 抖动=combine 跨卡 gather 竞态(zero-done+self-fence+wave2 未完全排序)。下步修 routed_y_buf push/gather 排序(非串行化)。routing 由确定的 post_norm 派生（gate/dispatch 确定），竞态在 INT8 routed 计算或 gather 读序，不在 routing。**从不翻 greedy argmax（303 稳定）→ L3 准出线已达标，属 L2-cos 细化。**
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
> 直接把最底部 code block 当第一条消息粘贴。自包含。**权威当前状态 = 底部 code block 的「当前状态（续12）」段**（✅ A2 collective 死锁已修=tp_all_reduce 补完成波，P=42 不靠 logging 确定性 RUN_CLEAN；✅ argmax=303==vLLM 复现 3/3 greedy-exact；唯一遗留 = 残余 ~1% logit 抖动，vector-diff 定位到 combine 跨卡 routed_y_buf gather 运行时顺序问题，从不翻 argmax）。⚠ 本文件"续9/续8/续7/续6/续5/M3/M3b"等历史段落中所有"A2 靠 logging/intermittent race、argmax=993/303 不在 top-5、M4 精度未达成、NaN、幅值爆炸"结论均为旧状态，**已被续12更新，勿据此行动**。
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
| M4 L1 ctx=1 token-exact | 全 42 层放开，`--hidden-token 6127` → **argmax=303** vs vLLM | **✅ 达成且可复现（2026-07-14 续12，device）**：argmax=303==vLLM golden **复现 3/3、不靠 logging**。含 3 个修复：L2 attn_layer_idx（续9，pushed 7294e26）+ A2 完成波 + combine zero-vs-push（续12，pushed fc5a269）。**勿再查 argmax=993/top-5/靠 logging——旧状态，已解决。** |
| A2 P=42 确定性 RUN_CLEAN | 不靠 logging 即 RUN_CLEAN | **✅ 已修（2026-07-14 续12）**：真因=手写 `tp_all_reduce` 单波 barrier 缺完成波（wiki S1:running-stalled=kernel 挂死，非容量）；补 Phase-4 完成波（全 17 copy）。P=42 device 7 次 RUN_CLEAN 无 507018 无 logging。**续9/续10 的"intermittent race/Heisenbug/DFX 抓 (1,23)"已作废。** |
| M4.R 残余 logit 抖动 | P=42 logit 分布 bit-确定（L2-cos） | ⏸ **~1% 抖动，从不翻 greedy argmax**（L3 已达标）。vector-diff 定位=combine 跨卡 `routed_y_buf` push/gather 运行时 DMA 顺序问题（DSL barrier 逻辑正确、per-layer distinct）；非 DSL 可安全补。下 session 选做（方向 B）或作 known-issue。 |
| M5 L2 多 token / decode-step | vLLM→whole-net KV bridge 或 live A/B（8001 vs 8000），多 token token-exact | ⏸ 未开始（需 port G5b co-tenancy 基建到本分支；残余抖动不翻 greedy token，可在 greedy-exact 基线上先做） |
| M6 整网 decode 集成落地 | 接入 serving 路径（live single-handoff），端到端精度双过准出 | ⏸ 未开始（"完成后端替换"的关键剩余里程碑；A2 已解，不再是 blocker） |

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

【当前状态（device 实测，2026-07-14 续12）—— 权威当前状态，以本段为准；下方"续9/续8/续7"及更早全部为历史，勿据此行动】

★★★ **A2 collective 死锁已修复（本 session 最大成果，device 验证，已 commit+push）**：根因 = 手写 `tp_all_reduce`（decode_layer.py:628，17 份 copy）是**单波 barrier**（notify+wait≥1→读→return），缺框架 `pld.tensor.allreduce` 文档要求的**完成波**。按 wiki `S1:running-stalled`=collective kernel 挂死（非容量，容量是 orch 码 1/2/3/4，我们没有）。修 = 全 17 份加 **Phase-4 完成波**（同 signal_window，AtomicAdd+wait≥2；transform `tools/step3p5/_add_allreduce_completion_wave.py` 幂等）。**device：P=42 确定性 RUN_CLEAN、无 507018、不靠 logging（7 次干净）。续9/续10 说的"A2 靠 logging 扰动/intermittent race/需 DFX 抓 (1,23)"已作废——A2 是单波 barrier 代码 bug，已修。**

★★★ **greedy token-exact 达成且可复现（device，已 commit）**：`--hidden-token 6127` → **argmax=303 == vLLM golden 复现 3/3**（修前 303/**20**/303）。第 2 个修复 = combine **zero-vs-push 竞态**（丢了 moe.py `pub_route_barrier`；本地 `_zero_routed_y_buf` 与 peer `pld.tensor.put` 无 barrier 隔开）→ 补 `combine_done` 上的 zero-done 波（transform `tools/step3p5/_fix_combine_zero_push_race.py`）。**→ L3 greedy top-1 准出线达标。** （L2 attn_layer_idx 修复=续9，仍有效，是 pushed 7294e26 的一部分。）

★★ **唯一遗留 = 残余 L2 logit 分布抖动（~1%，从不翻 greedy argmax）**：device vector-diff（P=1 row0 向量 ×2 L1/cos，`logs_n1/_vec_diff.py`+`N1_DUMP_DIR`）**逐 stage 精确定位**：resid_hold(attention 残差) L1=0、local_routed_y(INT8 routed 输出) L1=0、sh_y/gate/routing 全确定，**唯 moe_out(combine 输出) L1=113 cos=0.990 DIVERGE** ⟹ **抖动源 = combine 跨卡 `routed_y_buf` push/gather**（MY buffer 被 peer push 填充，跨卡交付非确定）。**DSL barriers(zero-done+self-fence+wave2)+per-layer distinct windows 逻辑均正确、已核查** → 是**运行时跨卡 DMA 可见性/顺序问题**，非 DSL 可安全补的 bug。max-based 探测看不到（各 stage max 全同），必须 vector-diff。

★ **已证伪/勿再试**：`_serialize_after_shared`（moe.py 的 shared→dispatch 串行化）device 实测 P=42 **死锁 3/3**（过约束 42 层流水 schedule，moe.py 只在单块非流水安全）→ **已回退，勿再试**。recv_r_route→tensor.put：非干净镜像(tile vs tensor)+index 竞态会是大错非~1%，排除。max-based P_DBG_STAGE 探测：看不到向量抖动，勿再跑。

【下一 session 任务（按依赖；用户决定 A 或 B）】
**方向 A（推荐，若接受 greedy-exact 基线）**：M5/M6 serving 集成 —— 多 token / KV-bridge / live 8001(pypto) vs 8000(vanilla) A/B。需把 G5b co-tenancy 基建（`SIMPLER_COMM_NO_HCCL=1`、`--serve` sidecar、KV-IPC bridge）从 0162/stepfun-develop 移植到本分支（大工程，多 session）。残余抖动不翻 greedy token，作 known-issue 记录；跨卡 gather 运行时顺序问题上报 simpler/pypto runtime team。memory: `g5b_*`、`project_g4_cotenancy_hccl_conflict`。
**方向 B（若要先消残余抖动）**：device DFX 抓 combine `routed_y_buf` post-gather 的跨卡窗口状态（P=1 不挂死、无 poison 风险；需加 stage-6 dump routed_y_buf post-gather + 各 rank 窗口快照），定位 push-未落/gather-抢读/可见性 gap，再做**非串行化**运行时修复。**禁 `_serialize_after_shared`（死锁）、禁 speculative DSL 补丁。**
**T2（correctness，M5 时可能需要）= L43/L44 特殊 swiglu limit**：config SWIGLU_LIMITS[43]=[44]=7.0(routed)、SWIGLU_LIMITS_SHARED[44]=16.0；整网 baked silu 全 42 层。只 2/42 层、平分布非主导，多 token 精度时再上。

【代码状态（已 commit+push）】
- **pypto-lib `fc5a269`**（fork `feat/whole-net-n1-fusion`）：A2 完成波 + combine zero-vs-push 修复 + 2 个幂等 transform（`tools/step3p5/_add_allreduce_completion_wave.py`、`_fix_combine_zero_push_race.py`）。**这 2 个修复在生成的 decode_layer.py 里，未进 generator——regen 会丢，需重跑 transform（memory 有 recipe）。**
- **pypto-project `28a6ef3`**：全部定位 + 裁定文档。
- `_serialize_after_shared` **不在 tree**（已回退，勿重加）。0234 备份 `.bak.pre_a2wave_*`/`.bak.pre_combrace_*`/`.bak.pre_serialize_*`。
- exporters 健康常驻 EXP=8 ST=8（`--reuse-exporters` 秒级 attach，省 15min 冷载）。
- memory `n1_m4_accuracy_gap_converged_direction_drift`（续12 完整证据链：A2 修复 + 残余=combine gather 的 full vector-elimination）。

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

> **【续16g（2026-07-15，device — P=1 bisect 结论 + 用户拍板回退 push baseline）】**
> - **P=1 bisect**：`P_FAITHFUL_MOE_LAYERS=1`（3 dense + 1 MoE 层）pull 版 device **仍 507018 死锁** ⟹ **单 MoE 层就复现 = 我 pull collective 的确定性代码 bug，NOT 42 层单-program 交织**（推翻续16f 倾向）。好消息：单层可 P=1 快迭代。bug 在 `_dispatch_pull`(func28) 或 combine `_pull_routed_y`(func37)；reasoning 找不到（rendezvous 对称、gather 越界已排除），需 device log 拿 stuck func + `waiting=` 定位（带 log 的 P=1 未拿到 stall 诊断即被回退打断）。
> - **⛔ 用户拍板回退 push baseline（已执行）**：`cp decode_layer.py.GOODKEEP decode_layer.py` + `cp _gen_faithful_real.py.bak.pre_pulldispatch_20260715_123346 _gen_faithful_real.py` → decode_layer.py 现为 **push 版**（`_dispatch_push`×10 / `_dispatch_pull`×0，py_compile 过），generator 也回 pre-pull（regen 复现 push）。push baseline = fc5a269 known-good（~34% clean，argmax==303 可复现）。
> - **pull WIP 全保留可续**：`decode_layer.py.FULLPULL_20260715_054234` + `_gen_faithful_real.py.FULLPULL_20260715_054234`（全 push→pull）；`PULLDISPATCH_WORKS_20260715_051112`（仅 dispatch pull）；`logs_n1/full_moe_kernel_config_053555.py`（func 映射表）。
> - **接续起点（若重启 pull）**：恢复 FULLPULL 备份 → P=1 + ASCEND log 拿 stuck func（build_output kernel_config 持久映射，**别猜**）→ 二分 dispatch-pull vs combine-pull 哪半死锁（可暂只改 dispatch、combine 留 push）→ 修 rendezvous/handoff 确定性 bug → P=1 clean 后放开 42 层 → argmax==303。**勿 chunk-为-提速（抬超时证明 deadlock 非 slow）。**

> **【续17（2026-07-15，device — PUSH baseline 重确认 + 专家 put/dsb/notify 线索深调 + 3 处纠正）】**
> 用户拍板**回到 PUSH**（pull 是 regression，见续16d）。本 session 深调专家的 `A: put+pipe_mte3+dsb+notify / B: wait+load` 假设，device 复现并**坐实卡点**（记忆 `n1_a2_primitive_exists_not_missing` 已附完整证据）：
> - **PUSH baseline（GOODKEEP）device 复现 BOTH**：clean → `argmax=303==vLLM golden`（3.32s）；stall → `507018×8`（P=42，1/1 attempt 即挂）。**必须 `export P_FAITHFUL_MOE_LAYERS=42`**（shell 里陈旧 `=1` 会只跑 1 MoE 层 → 2.78s clean、`argmax=27527`=P=0 诊断值、push 几乎没跑，假通过）。
> - **stall device log（权威）**：`TASK …319 state=RUNNING fanin=6/6 kernels=[aiv0:28] core=26`（在核上自旋、输入齐备、永不完成）+ `TASK …320 state=WAIT kernels=[aiv0:29] missing_deps=1`；`completed=39/137`；**device-4/5/6/7 全同 …319/aiv0:28**。⟹ **func28=`_dispatch_push` 在核 kernel hang（S1），8 卡一致；func29=`_dispatch_stage` 等它**。是 in-kernel hang、**不是**调度器 fanin 死锁。
> - **专家 A/B 模式确实生成**（源核实）：put→`TPUT_IMPL`(TPut.hpp:281-289) 末尾 `wait_flag(PIPE_MTE3)`（写入排空）；notify→`TNOTIFY_IMPL` = `st_atomic`+`dsb(DSB_DDR)`+`pipe_barrier`。
> - **⚠ 纠正 1**：`PTO_ASSERT` 由 `_DEBUG` 门控（debug.h:32-36），release kernel 下 = `((void)0)` → `TWAIT` 的 100M 自旋保护**在 device 上被编译掉** → **barrier 的 wait 也能静默 S1-stall**。不能靠"无 assert ⟹ 非 barrier"推断，必须 device 定位。
> - **挂点子操作（强指向，未 PC 钉死）**：消去法 —— 干净的 PULL `tp_all_reduce`(N=42) 用**完全相同的 AtomicAdd notify + Ge TWAIT barrier** → barrier 可靠；PUSH vs PULL 唯一差别 = `remote_store`(跨die批量`TSTORE`)/`wait_flag(MTE3)` vs `remote_load` ⟹ 挂点 = **跨 die 批量 MTE3 写完成排空**。**佐证**：simpler 自己验证通过的 `allreduce_twophase` 所有跨 die 批量搬运都用 PULL(`TLOAD`)，唯一跨 die 写是标量 `st_atomic`，**从不做跨 die 批量 `TSTORE`**——正是整网 dispatch push 依赖、且挂死的原语。可靠跨 die 写 = 标量 atomic，非 MTE3 DMA 批量。
> - **⚠ 纠正 2**：`_probe_barrier_scale PUSH BARRIER_ONLY` 的 inv=5 stall = **重复调用 + Set-notify 无 inter-run reset → 跨 run rank 去同步** 的探针 artifact；inv=1（单次 42-collective）CLEAN。整网是**单次 `rt.run`**（harness:298），故 barrier-only 单跑干净，与"remote_store 才导致挂"一致。（已给探针 PUSH 分支加 `barrier_only`-gated `push_store` 隔离。）
> - **决定性下一实验（未跑）**：物理**删掉** dispatch 的 `put`/`remote_store`（InCore body 内 module-int `if` 会变 device-if → 不能 gate，必须删+regen），保留 barriers，P=42 跑几次 → clean ⟹ 坐实 TSTORE；stall ⟹ barrier。**这是整网单跑、真 AtomicAdd barrier、per-layer window，无探针 confound。** 然后定修法方向：跨 die 批量 MTE3 写完成是 hardware/runtime 级问题——请专家确认"跨 die 批量 MTE3 写是否被支持/可靠"，还是正确原语是标量/pull。此前 TPut.hpp 加 fence / pre-push barrier 均失败（续14）；抬 3× `PTO2_*` 超时 10× 仍 deadlock（续15）。
> - **运维**：stall 后 worker 残留（STAGE=9）→ SIGTERM `[_]stage_whole...hidden-token`（禁 -9）；exporters（EXP=8 RDY=8）能扛住 507018 reset；build_output 即使 stall 也被自动清（要 kernel_config 需 stall ~100s 窗口内抓，或复用续14 映射）。
> - **⚠⚠ 收尾更正（同 session）——撤回"barrier 非 write"，sub-op 仍 UNRESOLVED（stale-compile confound）**：试了 redirect 变体（dispatch write `peer=dst`→`peer=my_rank`，10 份，barrier 保留）。P=42 ×3：a1 STALL 同 func28/task319/completed=39；**a2 CLEAN `argmax=303`**；a3 clean(被 kill)。⚠ **a2=303 是红旗**——若 local-dispatch 真生效，42 层错路由应使 argmax≠303；出正好 golden 303 ⟹ **edit 很可能没进 device kernel（kernel-compile 缓存？）** ⟹ redirect 实验**不可信**，write-vs-barrier **仍未定**（消去法偏 write；redirect 偏 barrier 但被 confound）。**任何基于改源码的诊断前，硬前提 = 先证实 `decode_layer.py` 的改动真的进了 device kernel**（stall ~100s 窗口内抓 KEPT build 的 dispatch `.cpp` 核对，或找/清 ir.compile/ptoas kernel 缓存）。更干净的定位（不改数据写）：删掉 count_done/data_done barrier 的 notify+wait（非 write）→ func28 若不再挂 = barrier 坐实。GOODKEEP 已还原（改过的版本备份 `.bak.pre_skippush_diag_20260715`）。**⚠ exporters 被误杀**（`pkill -f '[_]stage_whole_faithful_real_ipc'` 漏了 `.*hidden-token` → 连 `--export-rank` 一起杀，EXP=0，stale ready.rank* 残留）；下 session `rm /tmp/n1_weight_ipc/{ready.rank*,STOP}` + 重起 8 exporter（~15min）。杀 worker 必须带 `.*hidden-token` 过滤。
> - **⚠⚠ 收尾 SETTLED（推翻上一条"a2=303 红旗/UNRESOLVED"）——stale-compile 排除；sub-op 偏 BARRIER**：`simpler_setup/kernel_compiler.py::compile_incore`(296-383) 每次 ccec **全新编译** kernel `.cpp` 到临时 `.o`、**无 cache-skip**；唯一缓存是 orchestration-`.so` 上传（按 content sha1 Build-ID，改内容即失效）→ **`decode_layer.py` 改动确实进 device kernel（LIVE）**。a2=303 **不是** stale-compile，而是 **TP-replicated-token**：`x`(post-norm) 在 8 卡 TP-复制、expert EP-切分，local-dispatch 下每卡留 token 算自己那份 expert shard，cross-die combine（未改）照常 gather 正确 top-K → 仍出 303。⟹ **a1 在 write=local 下仍挂 func28 ⟹ 挂点是 notify/wait BARRIER（count_done/data_done），非跨die批量 `TSTORE`**；**推翻消去法"write 是元凶"**。置信：edit-live + func28 挂 = 坐实；barrier-是挂点 = 偏向（n=1，需多跑几次坐实）。疑似机制：count_done/data_done 的跨die `st_atomic` notify 间歇丢/竞（可能跟在批量写突发后 / notify 投递竞态）；pull 的 tp_all_reduce 无此前置批量写故不挂。**修法从专家的 data-path put/dsb/notify 转向 barrier-handshake 健壮性**。干净坐实：只删 count_done+data_done 的 notify+wait（不动 write）→ func28 不再挂 = barrier 坐实。
> - **⚠⚠ TWO-WAVE barrier fix 已试 → 无效（device，同 session）**：假设强且框架对齐（dispatch count_done/data_done 是单波 `expected=1`，10/10；tp_all_reduce 有两波完成波、device-clean；`_add_allreduce_completion_wave.py` 明说单波在 ≥41 层挂 `507018/S1`=func28 签名）。给 count_done & data_done 各加两波完成波（2nd notify + `wait expected=2`，同 window，10 份，cd_exp2=10 dd_exp2=10，edit-live）。**device P=42 ×5：a1 CLEAN argmax=303；a2 STALL；a4 STALL → ~与 baseline 同的 ~66% 挂 → 没修好。** ⟹ write-locality 和 barrier-wave-count 都修不了 func28 → **收敛到续13：A2 是更深的 Heisenbug 时序竞态；DSL/ISA 层微调（fence/notify变体/barrier波数/写本地化）扰动但不根治。真修法 = runtime/scheduler 层（可靠 in-kernel 跨die完成 / 确定性跨rank派发序）或多program（用户禁）。** GOODKEEP 已还原（~34% clean baseline）；two-wave 备份 `.bak.pre_twowave_20260715`（框架更正确但非 fix，勿当 fix 上线）。**建议带证据上报 pypto/simpler runtime team**：func28 dispatch 在核挂 + two-wave 无效 + write-local 无效 = runtime 层跨die/调度时序竞态，非 DSL data-path(put/dsb/notify) 问题。
