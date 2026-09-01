# 任务跟踪记录 — Performance 专项

> step3p5 decode 性能优化的**单一事实源**。每个子任务的状态、owner、更新时间、阻塞在此维护。
> 设计详情见 [`02-detailed-design.md`](02-detailed-design.md)；改状态时同步更新本表的「最后更新」列 + 底部更新日志。
>
> **状态图例**：⬜ TODO ｜ 🟦 IN PROGRESS ｜ ✅ DONE ｜ ⛔ BLOCKED ｜ ⏸ PAUSED

---

> **✅ 2026-09-01 current override：H6 已进入 r12，H7 local-owner reset 回退已修复并推 SRC。**
> canonical SRC 为 pypto `655c7bda` / pypto-lib `a745ab659`；release IMG 仍是
> r12（历史 pins `14de90fd/e6c7d8ec`）。
> `sha256:ba42fd19…eb805d`。r11 digest 上两文件 source-overlay A/B/A：
> ITL `21.6805→21.115 ms`（`−2.608%`）、graph build `−44.429%`、
> graph→first runner `−47.936%`、serial rank submit envelope `−23.887%`；
> 三臂 hidden/token exact。正式合同仍为 `serial-eight-rank`、8 个独立 submit，
> 不是 native group-submit。r12 immutable Main/MTP/dep-only DFX 与
> `1844/1844` final contract PASS；未重采 r12 immutable 性能 A/B/A。
> `bind.args` 仅占候选 ITL `0.259%` 且 `no_clear_change`，不再优化。
> 2026-08-29 canonical deployment launcher 已默认注入 H4=`all`，`none` 可回退；
> source-default-all matched A/B/A 收益 `7.372 ms / 24.591%`，exact launcher 64K/1000
> p50 `20.973 ms`。matched candidate H4=`all` A/B/A `21.617/20.516/21.257 ms`，
> H4 与 5-case extended correctness 均 PASS；a745 `recv_meta` publication 仍待补齐。
> 证据见
> [`../../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md`](../../benchmark/2026-08-27-whole-step-host-graph-submit-r12-release.md)。
>
> **历史线索（已由上方 H4 收口取代）：host 侧 `bind.args` ≈ ITL 的 23%。**
> 从 R5 MoE 生产基线（`decode_fwd.py` sha `67b73589…`，ctx-64K BS1）的 runtime STRACE
> 嵌套 span 读出：每 invocation `simpler_run` p50 = **`26.45 ms`**（与该配置 ITL p50
> `26.329 ms` 对得上），其中 **`bind.args` = `6.12 ms`**、`runner_run` ≈ `20.3 ms`，
> 两者**加性串行**（`simpler_run ≈ bind + runner_run`）。对照臂 `bind.args = 5.87 ms`，
> 同量级 ⇒ 不是偶发、与 device 侧改动无关。
>
> ⇒ **纯 host 侧参数绑定开销，占 ITL 约 23%，比 dispatch 域任何 small-op 融合的可得收益
> 大一到两个数量级**，且不碰 device 语义、不碰跨卡同步、不动 `@pl.program` 结构。
>
> **当时的下一步**：① 确认它是否真在 ITL 关键路径上（能否与 device 执行重叠）；
> ② 看有多少是跨 step 不变、可缓存的；③ 定位这 `6.12 ms` 的构成。
> 证据：`0162:…/dispatch-orch-decouple-20260821/{FINDINGS.md, analysis-bin/orch_span_stats.py}`；
> 方法与判据见 [`07-hardware-scheduler-performance.md`](07-hardware-scheduler-performance.md) §9.6。
>
> **⇒ 2026-08-21 已立项为 `H4`（P0）+ `H5`（P1，观测性前置）**，并按 swimlane 重排了
> 整个候选池：见 [`09-swimlane-derived-next-optimizations.md`](09-swimlane-derived-next-optimizations.md)。
> ★ 该文给出**强怀疑 `bind.args ≡ H2`**（H2 根因里的 `~92 make_tensor_arg`/`add_tensor`
> 就是参数绑定）⇒ 这条"最大新线索"可能早有现成设计。
> ★ 另给出一条**可复用否决判据**：关键路径 `front-gap = 0.000 ms` 且 stall 100% 为
> data-wait ⇒ `01-task-granularity` + `02-runtime-overhead` 两章 ROI 上界 = 0
> （**本可在 dispatch 融合线的 6 天 / 357 run 之前就否决它**）。

> **⛔ 2026-08-21 MoE dispatch 域小算子融合线整体关闭（负结论已定稿）。**
> R6-R9 融合候选 NO-GO（概率性 hang + 匹配曝光后也不快）；结构修复候选
> `dispatch-orch-decouple-20260821` device 门三臂全挂（**那个 orchestration 级阻塞标量读
> 是承重的 run-ahead 节流阀，删掉即 `orch_done=1` ⇒ ring 死锁**）；最后量化表明
> **orchestrator 从不在关键路径上**（p50 `orch` `17279 → 4443 µs` 即 −74%，而
> `device_wall` `17467 → 17910 µs` 反升）⇒ **整类改动 ROI 上界 = 0**。
> 当时的 MoE 生产继续用 R5，不做改动。详见 [`../../blockers.md`](../../blockers.md)、
> [`07-hardware-scheduler-performance.md`](07-hardware-scheduler-performance.md) §9、
> [`../../archive/milestones-2026-Q2.md`](../../archive/milestones-2026-Q2.md) 2026-08-21 §E。

> **历史：2026-08-12 K11 已合入 single-row small-message selector；2026-08-24 已被 r9 IMG 收录。**
> 当时 `csy0225/pypto-lib:stepfun/develop@69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d`
> （parent `9ca01d2`）。Main 的 rank-uniform `active_rows == 1` 走静态 8 KiB
> 两波 one-shot mesh；其他行数与 MTP 保留静态三波 fallback；ownership 固定
> `HIDDEN // TP_WORLD_SIZE` 并与 transfer chunk 解耦。unit `365 passed, 7 skipped`；
> Main/MTP default+chunk256 compile、8 卡 rows `1/3/16` PASS。focused 历史
> K6b-vs-smallmesh regular-call kernel-duration pooled mean 为
> `38.325 → 22.667 µs/call`（-40.9%，不是 strict critical-tail）；Whole A/B/A
> `31.065 / 29.912 / 30.999 ms`，delta `-1.120 ms / -3.609%`，三臂
> precision/per-iteration PASS。旧 `a791071` ring 是 A/A，K6b dynamic-validshape
> 是未落地的中间方案。five-layer 只声明 L3/L4 exact、finite、TP spread=0；
> zero-token canonical structural fail-closed 在 2026-08-12 当时仍不变。该历史缺口已由 2026-08-24 r9
> 的 immutable image、8/8 chip swimlane 与 outer admission 关闭。

> **历史：2026-08-12 I7 已合入，RMS→QKV raw-kernel residual p50
> `5.00 → 2.64 us`。**
> `allow_early_resolve` 是 producer 属性；`swa_qkv_proj` 原有的 flag 只优化
> `QKV → split/QKNorm/RoPE`，不能反向优化 `RMS → QKV`。最终提交链
> `fa58b5cf → 18d1b519 → e5e26f9f`：RMS producer 开 early resolve，同时用
> 已存在的 `swa_attn_out_zero` TaskId 把非关键 head-gate speculative fanout
> 隔离为 normal dispatch，让 14-slice QKV 在 RMS 执行期间优先预驻留。
> fresh 五层 8-rank 中 RMS Worker span min/p50/max=`4.16/4.35/4.78 us`；
> QKV Worker gap p50=`-1.78 us`，表示 setup 已与 RMS 重叠；剩余 raw-kernel
> residual=`2.08/2.64/3.16 us`。L3/L4 byte-exact、finite、TP spread=0。
> 整网 BS1/ctx64K A/B/A p50=`30.992/30.997/31.136 ms`，candidate 相对
> baseline center `-0.067 ms / -0.216%`、`0.931×` floor，判定
> `WITHIN_BASELINE_BRACKET`；三臂 hidden SHA 全等、token `14371` exact。
> `e5e26f9f` 当时已 fast-forward push；当前远端已由后续提交前进到 `69ad31e4`。
> 该历史结果不覆盖下方 I6
> 相对 `f9065261` 的整体 NO-GO；仍是 source overlay，未构建新镜像。详见
> [`../../benchmark/2026-08-12-step3p5-rms-qkv-dispatch-gap.md`](../../benchmark/2026-08-12-step3p5-rms-qkv-dispatch-gap.md)。

> **历史：2026-08-12 I6 当时 NO-GO，精度通过，但整网 ITL 与 fresh 五层门失败。**
> 设备验证候选 worktree 位于 0162
> `/mnt/persist/chensiyu/workspace/develop-worktrees/qkv-prerope-mix`
> （branch=`perf/qkv-prerope-mix-20260811`，base=`f9065261`），已提交为
> `fa58b5cffe41b30d3f8d94482230867ee34b9e84`；在 I6 landing 时已
> fast-forward push 到 `stepfun/develop`，且 origin、0162 main/candidate 三者同
> commit、clean。该 Attention 快照随后由 I7 前进到 `e5e26f9f`；当前 tip 以
> 上方 `69ad31e4` 为准。Full/SWA 分别以
> 10/14 个 packed QKV projection blocks 替代独立 Q/K/V projection，随后各接一个
> `qkv_split_qknorm_rope` mixed kernel，图为
> `qkv_proj → qkv_split_qknorm_rope → attn_mix`。unit=`362 passed, 7 skipped`、
> whole compile
> `rc=0`、focused 三臂 `container.rc=0`，12 个 edge context、Full/SWA
> Q-publication 12/12、异构 context `[1,2816,2817]` 与 L3/L4 precision 均通过。
>
> 2026-08-12 最终 clean commit 的 BS1/ctx64K A/B/A：A1/A2 p50
> `31.787/31.905 ms`，baseline center `31.846 ms`；candidate `33.194 ms`，
> **回退 `+1.348 ms / +4.233%`**。三臂 hidden SHA 全等、finite、token
> `14371` exact，故 `PRECISION_GATE=PASS`，但性能判定
> `REGRESSION_BEYOND_BRACKET`。同日 fresh 五层 DFX 为 **39/40**，rank7/L0
> Full=`54.54 us`，超过 strict `<46 us` `8.54 us`。该点是约 `12 us`
> AICPU scheduler dispatch stall；kernel compute 正常，但权威端到端 span 必须计入。
> 2026-08-11 的 40/40、max `43.60 us` 仅保留为历史 capture，不能覆盖 fresh failure。
> 固定镜像内仍为 `pypto-lib@cb96747e`，全部是 **source overlay**；canonical
> analyzer 的 `rc=1` 仍来自零本地 routed-token early-dispatch record 缺失，不能写成
> canonical structural PASS、immutable-image seal 或 production-qualified。该 I6
> 阶段当时不得构建 release image，需先隔离 packed projection 与 fused epilogue；
> 该历史操作结论已被后续 `9ca01d2`/`69ad31e4` supersede，当前 image 下一步以上方
> K11 状态为准。

> **✅ 2026-08-11 Attention mix + SWA RMSNorm multicore 已完成源码集成**：
> 该轮落地 commit 为 `fa58b5cf`（其 parent `f9065261` 包含
> `21d928b9` Attention mix + `f9065261` RMSNorm multicore）；当前
> GitHub 与 0162 checkout 当时随后由 I7 fast-forward 到 `e5e26f9f`；当前远端
> 已前进到 `69ad31e4`。固定镜像 manifest `sha256:076af8a…` 内仍是
> `cb96747e`；全部新结果均为该 immutable 镜像上的只读 `/candidate`
> **source overlay**，不是新镜像验证。
>
> RMSNorm 按 storage capacity rows / 2 rows per task 产生 8 个 logical tasks
>（非 active-token-derived），设备每 rank 8 blocks/8 distinct cores；strict
> block max `4.46 us`、logical span max
> `4.90 us`，L3/L4 byte-exact。Attention 的 Full QK→softmax→SV→segment
> recurrence 已进入 `full_attn_mix`，SWA 每 active row 一个 `swa_attn_mix`，旧
> split family=0。combined unit `357 passed, 7 skipped`、whole compile、
> focused byte-exact/Q-publication/8-rank DFX 全 PASS。最终 BS1/ctx64K A/B/A：
> `32.222 / 31.790 / 32.330 ms`，candidate 相对 baseline center
> `-0.486 ms / -1.506%`，三臂 hidden byte-exact、token `14371`。
> 完整证据见
> [`../../benchmark/2026-08-11-step3p5-attention-mix-rmsnorm.md`](../../benchmark/2026-08-11-step3p5-attention-mix-rmsnorm.md)。
>
> **前五层 DFX limited delivery**：L0–L4、8-rank capture/precision/mixed
> inventory/RMS `<5 us` 均 PASS；LOW-WAIT rank2=`2.124 ms`，L3 RMSNorm
> 8 tasks/8 distinct cores，全 rank最坏 slice/span=`4.28/4.30 us`。但
> rank0/1/3/6 各有 5 个零本地 routed-token 的 early-dispatch task 无 AICore
> swim record，故 canonical structural analyzer=`FAIL_CLOSED`；不得宣称
> structural PASS 或 release-qualified。candidate container `rc=1` 来自该
> postprocess analyzer fail-closed。delivery seal SHA256 =
> `088cf05ffbff717fd6da9fcf443122da88c4c9373c41276e4f5ae8dbfa51eb94`；
> delivery report JSON SHA256 =
> `7bc5811da7cf543d3ddf812ee90e8297c3238ce0e7b24160899c19584fc29688`。

> **⚠ 2026-08-11 correctness 约束（device 已证，影响后续所有 AR 优化）**：pypto
> `MakeNotifyCodegenPTO` 生成的 notify 前导把 `dcci(ENTIRE_DATA_CACHE)`
> （**invalidate-only，无 writeback**）排在任何 drain 之前，credit 因此可能跑到 payload
> 前面。ring 探针（裸 `remote_store` 紧接自己的 `notify`）在 `16/32/64/128 KiB` 四档、
> 两个方向上均复现（`exact=False`，epoch 0 就坏）。**最小修复 = 一条
> `pipe_barrier(PIPE_ALL)` 插在 `cacheinvalid` 之前**；消融矩阵已闭合：`PIPE_MTE3` 单独不够
> （⇒ **纯 reorder 上游现成指令无效**）、`dsb(DSB_DDR)` 单独不够、**`PIPE_MTE3`+`dsb` 组合
> 也不够**、纯 MTE3 流量不够、两条指令放 `TNOTIFY` 之后的安慰剂也不够（⇒ **是顺序不是时序**）。
> ⇒ `PIPE_ALL` 是必需的，代价压不下去。
> **代价已量化**：Wave2 单点 `+0.405 µs/call`、全 3 点 `+1.250 µs/call`
> （约 `0.060 µs/PIPE_ALL`，**比 K2a 的 pipe-specific barrier 贵约 18 倍，不能当免费**）。
> **⚠ 生产暴露面口径（两 agent 对账后）**：Wave2/Wave3 的 notify 前导与被证伪的形状逐字节
> 相同。我原先「探针近确定性失败 ⇒ 生产必被某结构性因素保护」的推论**已撤回**（默认了失败率
> 与结构无关，未验证）。接受 codex 更保守的结论：**生产 Wave2 没有可证明的安全机制，只是当前
> 调度没触发；是否正在损坏未知。** 已否证四条候选：纯 MTE3 流量、MTE3 级屏障（含 +dsb）、
> store-loop 自带 MTE3 屏障（codex 自撤）、Wave3 slack（结构性否证 —— Wave3 在 consumer
> read 之后）。**既不得声称生产正在静默损坏，也不得声称生产是安全的。**
> ⇒ **硬约束：任何把「payload store 与它自己的 credit」拉近的改动（合并波次 /
> 按 peer 融合 store+notify / 单 peer 交换）都必须先落 fence 修复**。上游诉求可表述为
> 「把 put 路径已有的 `PIPE_ALL` 对齐到 notify 路径」。详见 [底部 2026-08-11 更新日志](#更新日志)。

> **⚠ 2026-08-11 方法论约束（bench 已两次误导，第二次错到符号）**：bench 的
> `critical_tail(epoch) = max(rank_exit) − max(rank_entry)` 把参考点重置到每次 AR 的
> **最后到达者**，因此「让部分 rank 在**下一次** AR 入口等更久」的代价被它按定义减掉。
> 删同步点恰好是把同步开销转成到达 skew —— 正是该指标被设计成不敏感的量。
> ⇒ **任何删除/合并波次的候选不得用 bench `critical_tail` 评估，必须上整网 ITL。**
> 实证：删 Wave3 bench `−5.92 µs/call`、整网 **`+35.76 µs/call`（`+1.72 ms/step`，73× floor）**，
> 且三臂 byte-exact ⇒ 精度论证对、性能结论反。同理 K5-C 的定律外推失效（预测 `−20 µs`、
> 实测 `−0.54 µs`）。**受影响的账目**：K9 的 `−4.92 µs/call` 已撤回；「合并 Wave1+Wave2」
> 的 bench `5.6 µs/call` 降级为未确认。K6b 的 `7.02 µs/call` 只保留为中间实验，
> dynamic-validshape 版本未落地。当前落地的 AR 收益以 K11 Whole A/B/A
> `-1.120 ms / -3.609%` 为准；focused `38.325 → 22.667 µs/call` 只是
> regular-call kernel-duration pooled mean 的机制证据，不是 strict critical-tail。K8
> 是独立的 host reset 优化，不能与 per-collective 数字混为同一口径。

> **✅ 2026-08-11 K8 已在整网上兑现（截至当时是第一个 A/B/A + 精度门双过，
> 也是当时最新的 immutable-image 优化）**：
> control-prefix 重排 + 单次 `47,616 B` 清零。**两个独立 bracket 一致**：v1
> `−1.7505 ms/step`（66× floor）、硬化后的 v3 `−1.7455 ms/step`（`−5.16%`，**89.5× floor**），
> 差 `0.005 ms`。两次都四臂/三臂 `hidden_sha256` 全等生产 baseline `567b206b…`、`token=14371`。
> 关键在**实现形态**：想法（只清必需的 46.5 KiB）没错，错的是第一版实现拆成 6 次
> `broadcast_control_all` ⇒ 固定开销反超字节节省，`+1.886 ms`。正解是让模型把 7 个
> control buffer 声明在最前面，借 simpler carve「严格顺序无 padding」的性质构成唯一连续
> 前缀 `[0, 47616)`，**一次 broadcast 即可，完全不动 simpler**。
> **顺带修正机制模型**：ITL/wall 放大率**分 regime** —— 额外 broadcast 的固定开销贵、
> 字节传输约 1:1。**但两者都只能定性用**（`1.95×` 按 mean 变成 `1.67×`，字节路径
> `1.017–1.066×`，broadcast 次数 1/2/3 未测）⇒ 禁止定量外推。详见
> [更新日志](#更新日志)。
> **状态：已落地**（用户授权后直改 pypto fork）。落地件 = 被测的 v3 两份文件，
> 逐字节等同：模型 `decode_fwd.py` sha `eb1f89bf…` →
> `pypto-lib stepfun/develop@cb96747e`（+11/−7）；runtime
> `distributed_runner_prefix_v3.py` sha `fe50c11f…` →
> `pypto stepfun/develop@1c048a74`（`distributed_runner.py` +174/−22，含 codex 的
> reset trace 仪表）。落地后已重跑全部门：AST 名字门、3 例 layout 门
> （whole-decode 走前缀 / five-layer 回退全清 / 名字变体回退全清）、
> `ALLOC_ORDER_OK 16 allocs, first 7 are control`、无卡 codegen 门 `VERDICT=PASS`。
> v1 已被 codex 复核否掉（会打断 five-layer / two-layer / multi-program 等其它
> persistent 路径），**不要**拿 v1 落地。

> **✅ 2026-08-11 position 混淆已排除（null control）**：连续两个 A/B/A 的中间臂都慢
> 约 1.7–1.9 ms（K9 `+1.72`、K8-selective `+1.89`），量级巧合到必须先排除「第 2 个 arm
> 系统性偏慢」这一 harness 层混淆。**null control 判定：不存在。** campaign
> `k8-selective-20260811/null-20260811-131548`，三臂 `A1_parent / B_null / A2_parent`
> **完全同源同 runtime**（`B_null` 就是 baseline）：`A1 33.889 / A2 33.922 / B_null 33.725`
> ⇒ floor `0.0165 ms`、**`B_null delta = −0.1805 ms`（中间臂略微更快，方向相反）**。
> ⇒ **K9 与 K8-selective 的 ITL 结论成立**，且若按 `−0.18 ms` 的中间臂偏置校正，真实效应
> **还要再大 0.18 ms**（K9 ≈ `+1.90`、K8-selective ≈ `+2.07 ms`）。此后所有 A/B/A 可沿用
> 三臂顺序，不需要随机化。

> **⚠ 2026-08-08 historical source override**：attention/Vec 当时的产品实现权威 pin 为
> `pypto-lib stepfun/develop@491267c45875e9b1e0071eed224e2e73526799e2` 与
> `pypto stepfun/develop@8e92b46808f9f7c09b6431ad4691503f09c12ee5`。Wave5
> 以 self-target TPUT 发布 source partial，并保持既有三波 lifetime；immutable
> audit/smoke/Main+MTP compile、Main N=128×3、Main batch16、MTP batch1/batch16×2、
> 64K/batch16 ITL/DFX 均通过，是最后一个完整 production release-qualified
> 回退基线。`63814d4a` 将 SWA mask 从 `pl.cmp` predicate 转换路径改为 typed
> INT32 数值区间 mask；0162 source-overlay N=128 为
> `127/128=99.21875%`、TP spread=0。当时没有包含该提交的 immutable image，
> 镜像发布已按用户决定推迟到统一 release commit 确定后。manifest
> `sha256:3eb694e…` 仅是 `c9af5790` pre-fix evidence；历史 R1/R2 已
> supersede。下方历史 override 不覆盖 I1/I2。
>
> **⚠ 2026-08-07 L0–L4 focused MoE historical override**：Track J 的产品实现已随
> `7928a275` 进入当时的 `491267c4`；范围只包含
> `L0 Full+dense / L1–L2 SWA+dense / L3 SWA+MoE / L4 Full+MoE`，L4 必须消费
> 真实 L3 输出。最终方案为 routed gate/up stage split，普通 expert 使用
> `row=16, K=512, N=64, down N=256`；L43/L44 specialization 保持原配置。
> pre-fix digest `sha256:cab8966…` 上 BS `1/2/4/7/8/16`、每请求独立 64K 的三轮
> normal A/B 已完成，36/36 fresh-process run 已 seal，六档 L3/L4 hidden hash
> exact 且性能无回退；matched-source whole-net 1-step×2、2-step×2 已 8/8
> sealed PASS，source-overlay N=128 已过线。SWA mask 变化后，旧 golden/性能
> 不能自动升级为 final evidence。剩余准出是最终 immutable image 上六档 64K、
> 双 hidden、N=128、formal matched-source DFX、route-aware reanalysis 和
> all-rank swimlane。
> 该结果不覆盖 whole-net、prefill 或 L43/L44 release gate。

> **历史 2026-07-28 release override**：active base =
> **`stepfun/develop @ 563fe62a`**。唯一 release Main 为
> `models.step3p5.decode_fwd:whole_decode_step3p5`。0724 unroll Main、
> rollback selector、自定义 Main module/name 参数和旧 opt compatibility
> package/aliases 均已删除；后续实现、合同和设备验收只允许 canonical。
> 2026-07-26 以前的 unroll/generator 记录仅是历史设计依据，不是可执行入口。
> 对账：A1 / C2 / B1 / SwiGLU-per-layer / B2 已随 current replacement 交付；
> C1 仍是独立通信优化项，不与 B2 replacement regression 混淆。

## 任务执行前的端到端同构核对

任务卡不是代码事实源。执行任何B3/C1/C2/C3/G1任务前，先读取current source、调用链和工作树diff，再对照`origin/main:models/deepseek/v4-flash/`。禁止用局部shape/name、单个primitive或旧任务描述判定“step3p5独有/必须保留”。

每个任务必须沿以下链路记录证据：

```text
producer → 数学变换/quant/route-map → transport/window
→ consumer → rounding/reduction/placement → lifetime/reuse/allocator
```

并将差异分为：能力/算法、数学语义、layout/shape、host/allocator集成、backend/profile workaround。V4-Flash已有同构能力时，layout/shape差异只能作为参数化或存储适配，不能升级为架构差异。

记录反例时使用通用规则而非固化shape：INT8+scale需核对scale和dequant；owner max需核对owner-vector到所有active consumer；`BATCH=16`需区分capacity与runtime logical batch；route-weight placement需核对route/weight到最终FP32 weighted reduction。旧任务描述、历史probe和状态表若落后于current source，先记录冲突，再按current source与最新用户决定更新合同。

## 看板（按 Track）

### Track A — 可观测性
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| A1 | whole-net baseline + DFX 采集 | P0 | ✅ | claude | — | 新增多步 report/per-layer hidden/TP spread/finite 检查；逐层 DFX：ctx≈1 下 routed-expert 占 90.7%（PMU cube_int8 88.6%）；⚠ **该占比只适用短 context** —— ctx=65536 实测 `tp_all_reduce` 仅占 span 1.84%、routed expert busy 0.99%（C/D/F 系对 64k 延迟均低 ROI）；attention 的 97.9% 含插桩放大不可当延迟占比，见 benchmark/2026-07-24 与 benchmark/2026-07-29 §1.6 | 2026-07-26 |

### Track B — Mega-kernel 结构
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| B1 | resident 权重池 + loop-form zero-copy view ABI | P0 | ✅ | — | — | 前：0724 已一次 IPC import/resident，但 loop-form Main 无 10 FULL + 30 SWA 连续 bucket ABI；后：`Wsub()` 从 loader pool 取 FULL `1:11` / SWA `2:32` zero-copy view，再跨 rank stacking。相对复制专用 bucket，按 shape 避免约 `0.94 GiB/rank` 额外设备副本；不再误写成 24 GiB/rank/step H2D。dynamic-offset probe PASS，N=256 replacement exact | 2026-07-26 |
| B2 | 45 层 unroll → `pl.range` 循环 | P1 | ✅ | b1-weights | — | 前：historical `decode_layer.py` 31,686 lines / 40 MoE layer sites；后：canonical `decode_fwd.py` 4,772 lines、MoE `pl.range(40)` 单 loop body + L43/L44 specialization，源码约 -84.94%。N=256 canonical-only 清理前后 token/hidden 均 256/256 exact；raw 同为 240/256，低于历史 raw gate | 2026-07-26 |
| B3 | KV pool `resident` + in-place | P2 | 🟦 | codex-bc | B1 | canonical source contract、全物理池 fail-closed row-diff/first-write-no-op 合同及 0162/0726 镜像 backend compile 已通过；历史 r4 在首个 invocation 的 AICPU/collective 初始化后 `507018 + S1:running-stalled`，0 个 invocation 完成，故连续 ≥6 次、45 层×8 rank K/V device row-diff 仍 NO-GO | 2026-07-27 |

### Track C — MoE 通信协议
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| C1 | shared window set + `moe_epoch` + `WaitCmp.Ge` | P0 | ✅ | codex-bc | — | V4-Flash-style shared EP windows、单调epoch与真实 arrival/completion lineage 已落地；固定 expert lane layout 保证 active-batch 扩展不改变 row0 | 2026-07-28 |
| C2 | 迁移 V4-Flash dispatch/combine 数据流 | P1 | ✅ | codex-bc | C1 | expert-lane dispatch metadata/push/arrival/gather 与 combine scatter/wait/token FP32 reduce 已迁移；固定物理 expert lane base 修复 BS1/2/16 的 batch-extension invariance | 2026-07-28 |
| C3 | expert-lane SPMD + whole-net 调度适配 | P2 | ✅ | codex-bc | C1, C2 | local-expert lane ownership、whole-net canonical scheduling、compile/lowered/设备回归已验证；最终镜像 Main 8-step 与 N=256 teacher-forced hidden 回归通过 | 2026-07-28 |
| C4 | TP all-reduce: onephase mesh → reduce-scatter + **push** all-gather | P0 | ✅ | claude | — | 按 [`03-tp-allreduce-algorithm-comparison.md`](03-tp-allreduce-algorithm-comparison.md) 落地，但**该文 §5 原方案（pull 形式 all-gather）落整网即 8 卡不一致**，已就地修正。根因 = pull-after-remote-notify 跨方向握手无序，见 [`postmortems/13`](../../postmortems/13-tp-allreduce-pull-notify-race.md)。每卡远程传输 56→14、远程字节 896KB→224KB；数值 bit-identical（canonical peer order + 单 FP32 累加器 + 每元素一次 BF16 store）。回归：whole-network CI `PASS` + 3 次独立重复共 32 step，token 全对、`hidden_tp_spread` 全 `0.0`；ITL p50 −3.6%(ctx1024) / −3.9%(ctx4096)，见 [`benchmark/2026-07-28-tp-allreduce-push.md`](../../benchmark/2026-07-28-tp-allreduce-push.md)。⚠ 收益仅适用 ctx ≤ 4096：64k DFX 显示 `tp_all_reduce` 只占 span 1.84%，见 [`benchmark/2026-07-29-release-image-64k-dfx-itl.md`](../../benchmark/2026-07-29-release-image-64k-dfx-itl.md) | 2026-07-28 |
| C5 | ~~TP all-reduce 并行扇出补齐（`pl.range`→`pl.parallel`）~~ | P2 | ❌ | claude | C4, I2 | **2026-08-05 实测证伪并关闭，收益恒为 0，不要重开。** 把三波 barrier 的 6 个 notify/wait 循环 + push all-gather 循环全改 `pl.parallel(group_size)`，在 wave5 镜像里 compile A/B：生成的 `kernels/aiv/tp_all_reduce.cpp`(431 行) 与 `ptoas/tp_all_reduce.cpp`(378 行) **逐字节相同**。原因 = 这些循环在生成码里**未被展开**（9 个真实 C `for`，`TNotify`×3/`TWait`×5 各在循环体内），而 **InCore 的 AIV codegen 不消费 `ForKind`**；`pl.parallel` 只在 Orchestration 层（切 task/block）有意义。连带推论：§2.4 把 `twophase_par` −35% 归因于 `pl.parallel` **站不住**，真实差异更可能是它的 AG 直接写 `out`（无 final copy、无 Wave3）这个结构差异。复现 `0162:/mnt/persist/chensiyu/workspace/ar-c5/compile_ab.sh {base,c5}`。产物：只保留一条守卫契约 `test_tp_all_reduce_keeps_reduce_scatter_accumulate_serial`，kernel 未改。见 §7.3② | 2026-08-05 |
| C5′ | self-target TPUT 加 `pipeline=True` | P2 | ⬜ | — | C4, I2 | C5 里唯一未被证伪的那一半，也是当前结构内**唯一**剩下的实现层杠杆：128 KB self-publish 走 ping-pong 双缓冲（`chunk_rows/cols` 已就位，`pipeline` 语法已合法）。⚠ 该 TPUT 的**同步 drain 正是 Wave5 修间歇 `tp_spread` race 的承重件**（`decode_fwd.py:263` + PTOAS#872），`pipeline` 在 chunk 间重叠可能重开 Wave4 blocker → **必须走 Wave5 同级稳定性验收**（Main N=128×3 + batch16 + MTP，全部 `tp_spread_max=0.0`），rc=0 不算过。见 §7.4 | 2026-08-05 |
| C6 | 消 Wave 3 + final local copy 多核化 | P2 | ⬜ | — | C4, I2 | 当前每次 all-reduce 的本地搬运（self-TPUT 128 KB + final copy 128 KB = 256 KB）**多于**跨卡搬运（224 KB）。⚠ **2026-08-05 纠错**：原「push all-gather 直接落输出、省掉 final copy」**不成立**——`local: pl.Tensor`（`decode_fwd.py:251`）不是 `pld.DistributedTensor`，peer 无法 `remote_store` 进去；final copy 只能换 `pl.spmd` 多核化（`pl.parallel` 已证伪无效），或把 `local` 提升成 window（动 90 调用点 ABI）。⚠ Wave3 是 **run 边界**护栏：层内 `win_off=(layer_idx+1)*BATCH`（`:3479`）已按层切故层间不复用，但下次 `rt.run()` 重用同 buffer——删前须证明跨 run 另有 fence。必须过 `hidden_tp_spread == 0` gate。见 §7.3③ / §7.4 | 2026-08-05 |
| C7 | atomic-add push 版 reduce-scatter（消除远程读 + 加法链） | P3 | ⬜ | — | C4, I2, **rebase origin/main** | `pld.tensor.put(..., atomic=AtomicType.Add)`（`tensor_ops.py:291-394`）推贡献到 peer → 1 barrier → 本地读自己那块 → push → 1 barrier：**零远程读**（`postmortems/13` 结论 = push+notify 可靠、pull-after-remote-notify 有 race）、消掉 7 级串行加法依赖链、3 barrier→2。⚠ **精度**：BF16 atomic add 破坏 C4 保住的 bit-identical，而 `hidden_tp_spread == 0` gate 依赖它；缓解 = FP32 comm window + 末尾一次 BF16 cast（RS 阶段窗口字节翻倍）。⚠ 依赖上游 `9776f276`（bf16 atomic-add on A2/A3）+ `5b17dfa6`，本地落后 `origin/main` 1213 commit 尚不具备。见 §7.4 / §7.5 | 2026-08-05 |

> **⚠ Track C 剩余项的共同天花板**：`tp_all_reduce` 在 ctx=65536 只占 device span **1.84%** / busy **2.33%**
> （avg **16.1 µs**/次，`benchmark/2026-07-29-release-image-64k-dfx-itl.md:62`），**干到 0 也只降 1.84%**。
> 且该 16 µs **含 barrier 自旋等待**（profiling 计入 kernel compute，`benchmark/2026-08-03-*.md:100`），
> clean run 跨卡起跑阶梯 2.914 ms 全被每步第一个 barrier 吸收。
> 「每次 40+ µs」来自 PMU（17×）/ swimlane（476×）放大 run，**不是 clean run 耗时**。
> **分段计时已做完（2026-08-05，§7.1.1）**：排除每步第一个 barrier 后，89 个 barrier × 8 rank 的
> `min over ranks`（自身搬运）稳定在 **39.6–44.7 µs**（p50 41.1），mean 51.1 → **自身搬运约占 80%**；
> swim 放大 ~2.5× 折回 clean 即 **16.1 µs 基本全是自身搬运**。480 KB ÷ 单核 ~14 GB/s ≈ 34 µs 与之吻合。
> p90 的 205 µs 长尾是 MoE 负载不均 skew 在此被吸收（straggler 在 8 rank 间轮换，无坏卡），
> 记在 `combine_wait` 账上。**结论：靶子合法但天花板仍是 1.84%**；`combine_wait`（13.4 ms = device 24%）
> 仍是 13× 大的目标。详见 §7.1 / §7.1.1 / §7.6。

### Track D — INT8-native W8A8（gap-5）
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| D1 | 对齐 V4-Flash deferred-norm + INT8 activation/scale producer | P1 | ✅ | codex-bc | — | deferred norm/amax/single per-token scale 与 INT8 dispatch ABI 保持；gate/top-k 与 producer lineage 已核验，BS1 正确性通过固定 expert lanes 修复，未回退 D1 | 2026-07-28 |
| D2 | 对齐 V4-Flash routed expert INT8×INT8/requant/W2 epilogue | P1 | ✅ | codex-bc | D1 | routed expert INT8×INT8/requant/W2 epilogue 与 route-weight placement 保持；combine 仅做 FP32 token reduction，固定-lane 设备回归通过 | 2026-07-28 |

### Track E — LM head
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| E1 | 按 V4-Flash 复用 data window 的 LM head seam | P2 | ⬜ | — | C1 | V4-Flash已有MoE/LM-head data storage复用与独立completion counter；step3p5只适配active rows、TP/vocab和tail ABI，不另造通信架构 | 2026-07-27 |

### Track F — intra-kernel L1/L0 微调
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| F1 | 对齐 V4-Flash dependency/early-resolve 调度语义 | P2 | ⬜ | — | A1 | V4-Flash已有等价依赖模式；`task_dummy`仅是当前frontend表达，不是硬合同 | 2026-07-27 |
| F2 | 按 V4-Flash data tile/pipeline 做 MTE 性能调优 | P2 | ⬜ | — | A1 | 512B只表示data tile/cache/MTE性能对齐，不得与control signal 512B ABI混淆；需A1数据驱动 | 2026-07-27 |
| F3 | 复用 V4-Flash deferred-norm/quant producer | P2 | ⬜ | — | D1 | 直接复用D1统一的norm/amax/scale producer，不再为dense/attention另造独立fusion架构 | 2026-07-27 |

### Track G — 调度轴 / 动态 batch
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| G1 | experts/feature调度轴 + runtime dynamic active batch/token | P1 | ✅ | codex-bc | 与B2/C2/C3协同 | active batch/token 已贯穿 attention、MoE、combine、KV；BS1/2/16 单步与 BS1 持续 4-step 通过，row0 batch-extension exact，最终镜像 Main 8-step PASS；N=256 hidden 全 finite、TP spread=0，teacher-forced token exact 241/256（沿用历史 live oracle，raw 95% 不宣称通过） | 2026-07-28 |

### Track H — host 侧 per-step 开销（L3 层）
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| H1 | retained window 清零：host 搬零 → device `aclrtMemset` | P0 | ✅ | claude | A1 | simpler `e2efebcb` + pypto `1f704616`（含 gitlink bump）。清零 `21.50→2.21 ms`、ITL p50 `85.02→65.55 ms`（−22.9%）、每步 H2D `244.7 MiB→0`、mailbox 往返 `248 串行→1 次广播`。同镜像 A/B `main_hidden_8step` 两边 `passed=True`，`main_hidden_only_report.json` 除 `run_sec` 外逐字段相同；单测 8/8。⚠ live N=128 精度门未跑（需 vanilla oracle 占 cards 0-7）；CI 整体 rc=1 但两边同样缺 MTP fixture，与本项无关。数据见 [`benchmark/2026-07-29-host-window-memset.md`](../../benchmark/2026-07-29-host-window-memset.md) | 2026-07-29 |
| H2 | per-rank 视图重建 hoist 到 `prepare()`（residual graph-build 候选） | P1 | ⬜ | — | H1, H6 | 2026-07-29 量化过起跑阶梯 `2.914 ms` 与 submit `3.49 ms`。H6 已用 prepared descriptor/signature cache 回收 graph/submit 主体，但候选 graph build 仍为 `2.2743 ms`；不能仅凭 `bind.args=0.259%` 宣布 H2 无 ROI。后续先重拆 residual graph-build，再决定是否 hoist；不得把 H2 重新表述成 bind 优化 | 2026-08-27 |
| **H4** | **8 个 step-invariant RoPE/gate-R 常量 device-resident** | **P0** | ✅ | — | — | r9 起 `PYPTO_H4_RESIDENT=all`：`bind.args` p50 `6.461 → 0.063 ms`；2026-08-29 r12 matched `none/default/none` 收益 `7.372 ms / 24.591%`，exact launcher 64K/1000 p50 `20.973 ms`。三个 canonical launcher 默认注入 `all`，额外 `99.64 MiB/rank`，`none` 可回退；image Config/代码默认不变 | 2026-08-29 |
| **H5** | **恢复 all-rank chip swimlane 与 outer admission** | **P1** | ✅ | — | A1 | r9 combined gate：8/8 rank `chip_swimlane_records.json`、deps/name-map/critical-path/merged 全齐；L3/L4 exact，analyzer `pass=true/blockers=[]`，recv_meta ready，outer admission `pass=true`。analyzer 自身保留 `PENDING_EXTERNAL_GATE` 是职责边界，不是 structural failure | 2026-08-24 |
| **H6** | **prepared TaskArgs descriptor/signature cache + whole-step graph/submit 收口** | **P0** | ✅ | codex | H4 | pypto `14de90fd`。r11 source-overlay A/B/A：ITL `−0.5655 ms / −2.608%`、graph build `−1.8183 ms / −44.429%`、graph→first runner `−1.4851 ms / −47.936%`、serial submit envelope `−0.2875 ms / −23.887%`；hidden/token exact。cache 有界、未知对象 fail-open，`free()` 与 proof window 互斥。baked 入 r12 并过 immutable release gate；设备门仍是 serial 8-rank independent submit | 2026-08-27 |
| **H7** | **local-owner persistent reset ABI 精确 profile（回退修复）** | **P0** | ✅ SRC + candidate gate | codex | H1, H6 | `655c7bda` 为 local-owner 4-control/4-data layout 增加显式顺序、byte-size、control prefix `46,080 B` 与 full window `11,842,560 B` pin；未知 layout 继续 full clear，变更 fail-closed。旧 `655+e6c7d8ec` B 臂 H4 p50 `21.562 ms`；当前 `r14a(14de+a745)` / `r14b(655+a745)` A/B/A `21.617/20.516/21.257 ms`，5-case extended admission v2 PASS。candidate IMG 尚待 a745 `recv_meta` publication，不宣称 release IMG | 2026-09-01 |
| H3 | DFX run 第一 barrier 假长条（观测性，非性能） | P2 | ⬜ | — | A1 | DFX run 里第一个 `tp_all_reduce` 被记成 115 ms(pmu)–379.9 ms(swim)，其余 89 次 39–366 µs，straggler 每次换人。**已排除 host 下发**（`_submit_chip` 在 DFX 下只多一次字符串拼接；clean run 下发等距 0.412 ms）。clean run 算术上界：非 device 时间共 5.7 ms → 380 ms 不可能存在。方向 = chip child 侧 collector 开销落在被 trace 区间内（注意 `orch._dfx_dispatch_idx` 每 request 重置，留下的是**最后一步**，非冷启动）。危害：曾使 `tp_all_reduce` 被误判成 74.1% wall | 2026-07-29 |

### Track I — Attention / Vec 收尾与 canonical 发布
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| I1 | workload-derived attention、Full 层次归约、out-proj cast、dense Vec 收尾 | P0 | ✅ | codex | A1, C4, H1 | `pypto-lib stepfun/develop@7099476b`（attention/Vec 内容自 `76d96bdb` 起保持）：logical task 按 active workload 推导；Full SV 合并 segment recurrence，只保留 reduce/finalize；Full/SWA out-proj cast 默认融合；dense RMS direct BF16 reread、dense down-proj cast fusion 保留；AR+residual、residual+RMS stats、RMS+projection、gate/up+SiLU 等无稳定收益方案不合入。active-batch=16/异构 context、source/compile/device/DFX 已完成。当前 task/tile 设计见 [`04-attention-optimization.md`](04-attention-optimization.md) §13 | 2026-08-03 |
| I2 | TP all-reduce immutable release stability gate | P0 | ✅ | codex | I1 | Wave3/4 先闭合 final-read lifetime 并对齐 harness AST；Wave5 `7099476b` 再以 self-target synchronous TPUT 发布 source partial，并同步 Main/MTP/harness/返回值 lineage。manifest `sha256:4acc77cd…`：audit/smoke/Main+MTP compile、Main N=128 预定义三轮均 `123/128` 且 spread=0、Main batch16、MTP batch1/batch16×2、64K/batch16 ITL/DFX 全 PASS；64K p50 `49.796 ms`。machine scope=`0162 release-qualified`。见 [`../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md`](../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md) | 2026-08-03 |
| I3 | `swa_moe_chip_orch_swa_rmsnorm_zc` storage-capacity-row-derived 多核化 | P0 | ✅ | codex | I1 | `rows_per_task=2`，`BATCH=16` 时 8 logical tasks（非 active-token-derived）；每 rank 8 blocks 映射 8 distinct cores。block max `4.46 us`、logical span max `4.90 us`，strict `<5 us` 双门 PASS；L3/L4 byte-exact、finite、TP spread=0。只接受 grain=2，其余 fail-closed。commit `f9065261` 已 push；验证为 K8 immutable image 上 source overlay，不是新镜像。 | 2026-08-11 |
| I4 | Full/SWA Attention mixed InCore task 集成 | P0 | ✅ | codex | I1, I3 | Full 的 QK→typed mask/softmax→SV→segment recurrence 合入 `full_attn_mix`，跨 segment reduce/finalize 保留；SWA 每 active row 一个 `swa_attn_mix`，旧 split family=0。focused exact/Q-publication/focused mixed-kernel 8-rank DFX PASS；BS1/ctx64K A/B/A candidate p50 `31.790 ms`，相对 baseline center `-1.506%`，三臂 hidden exact。commits `21d928b9..f9065261` 已 push；尚无新镜像。见 [`04-attention-optimization.md`](04-attention-optimization.md) §15。 | 2026-08-11 |
| I5 | 最终 commit 前五层 DFX swimlane limited delivery | P0 | ✅ | codex | I3, I4 | L0–L4、BS1、ctx64K、8-rank capture；L3/L4 exact、LOW-WAIT rank2 `2.124 ms`；mixed inventory 与 RMS 8-core `<5 us` PASS。candidate container `rc=1`：canonical analyzer 因 4 ranks 各缺 5 个零本地 routed-token early-dispatch AICore record 而 `FAIL_CLOSED`；只作 limited delivery，不是 structural/release seal。all-ranks bundle SHA `d6f689c7…`，report JSON `7bc5811d…`，delivery seal `088cf05f…`。 | 2026-08-11 |
| I6 | packed QKV projection + `split/QKNorm/RoPE` mixed epilogue strict `<46 us` | P0 | ⛔ NO-GO | codex | I4, I5 | 实现/精度正确：Full/SWA 为 10/14 个 packed projection blocks + 1 个 fused epilogue；unit、whole compile、focused correctness、Q-publication 及整网 hidden/token exact 均 PASS。最终 clean commit `fa58b5cf` 的 BS1/ctx64K A/B/A 却从 baseline center `31.846 ms` 回退到 `33.194 ms`（`+1.348 ms / +4.233%`）；fresh 五层 DFX 为 **39/40**，rank7/L0=`54.54 us`。异常点是约 12 us AICPU scheduler dispatch stall，inventory/dependency 仍 PASS。2026-08-11 的 40/40/max `43.60 us` 只作历史 capture。全部为 manifest `sha256:076af8a…` 上 source overlay；canonical zero-route record 限制仍在。阻塞：拆分定位 packed projection 与 fused epilogue，最终候选须重过 whole A/B/A 和 40/40 strict 门后才能构建镜像。fresh report JSON `0b5cbe20…`，ABA result `065f67c8…`。 | 2026-08-12 |
| I7 | SWA RMSNorm → QKV critical prestage | P0 | ✅ | codex | I3, I6 | 修复 RMS 本体 `<5 us` 后仍额外约 `5 us` 的 consumer dispatch bubble。RMS producer 开 `allow_early_resolve`；QKV producer 原 flag 保留用于下一跳。用已存在的 `swa_attn_out_zero` TaskId 将非关键 8-block head-gate 从 speculative fanout 隔离，优先预置 14-slice QKV。fresh 五层 RMS Worker span p50/max=`4.35/4.78 us`；QKV Worker gap p50=`-1.78 us`，setup 已与 RMS 重叠；raw-kernel residual p50/max=`2.64/3.16 us`（baseline `5.00/5.48 us`）。L3/L4 exact。整网 A/B/A=`30.992/30.997/31.136 ms`、precision PASS、`WITHIN_BASELINE_BRACKET`。commits `18d1b519..e5e26f9f` 已 FF push；source overlay，canonical zero-route limitation 与 I6 NO-GO 均不变。 | 2026-08-12 |

### Track J — MoE compute 优化
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| J1 | L0–L4 routed gate/up stage split + task-grain tuning | P0 | 🟦 | codex | A1, C1–C3, D1–D2, G1, I2 | 产品实现已被当前 `stepfun/develop@e6c7d8ec` 与 r11/r12 继承；r12 Main precision/MTP/dep-only DFX 已过门。r9 的 8/8 whole-swimlane 与旧 `c9af5790` 六档 normal A/B 仍只是历史 evidence；若 J1 要升级为“当前六档 campaign release”，仍需在 r12 对 BS `1/2/4/7/8/16` 重跑同口径 golden/A/B 与所需 DFX，不能跨镜像借证据。设计见 [`05-moe-optimization.md`](05-moe-optimization.md) | 2026-08-27 |
| J2 | gate fan-out 与 norm/quant 解耦（deferred `inv_rms`） | P0 | ✅ | claude | J1 | 已发布 `stepfun/develop@d13b2ca6`（单 commit FF，只改 `decode_fwd.py` +63/-35，sha `d392311c… -> 28080c53…`）。`gate_expert_fanout` 只写 raw FP32 logits，`inv_rms/sigmoid/bias` 尾巴搬进本来就等 `inv_rms` 的 `gate_topk`；算子顺序与数值语义不变，codegen 侧 `params_t70` 不再 `add_input(moe_inv_rms)`，task 数与 `block_num=9` 不变。0162 三臂 A/B/A：bs=1/64k/nb512 p50 `36.494 -> 33.849 ms`（**+7.25%**，地板 0.634）、bs=8/64k/nb4096 p50 `97.528 -> 91.722 ms`（**+5.95%**，地板 2.637）；bs=16 物理不可行（16 GiB 单次 rtMalloc → `207001`）。**两档三臂 hidden payload 各自 byte-exact**（bs=1 = N256 golden `567b206b…`、bs=8 `1fcd4fcc…`）→ 按项目口径 sha256 即准出。机理：MoE-only 段 15→14 hop、`norm_quant` 离开关键路径、链头 `81.8 -> 56.5 us`。数据见 [`../../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](../../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md) | 2026-08-10 |
| J3 | routed GMM active-worker dual-latch | P0 | 🟦 | codex | J1, J2 | `a745ab659` 已进入 canonical `stepfun/develop`：`A=min(active_local_experts,36)`，`G/H/Q=min(22,10A/5A/A)`；历史 source-overlay H4 收益 `0.931 ms / 4.4117%`，matched `655+a745` candidate A/B/A `21.617/20.516/21.257 ms`，hidden/token exact，5-case extended gate PASS。固定 22-participant 版本 `0.588 ms < 0.616 ms` 已 NO-GO。阻塞：exact `recv_meta` route sidecar、完整 publication gate 与 successor immutable image；r12 不含该候选。见 [`../../benchmark/2026-08-30-routed-gmm-active-worker-dual-latch.md`](../../benchmark/2026-08-30-routed-gmm-active-worker-dual-latch.md)、[`../../benchmark/2026-09-01-k8-local-owner-reset-regression.md`](../../benchmark/2026-09-01-k8-local-owner-reset-regression.md) | 2026-09-01 |

### Track K — TP all-reduce 二次优化（campaign 内部代号 P2；注意与本表「优先级」列的 P0/P2 无关）
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| K1 | AR critical-tail 口径确立 + notify/drain 分账（Phase 0，只插桩不改生产） | P0 | ✅ | codex（claude 独立复核 REPRODUCED） | I2 | headline 口径改为 `critical_tail = max(rank_exit) - max(rank_entry)`；T/P/T 三臂 half-range `0.06 µs`、插桩扰动 `0.36 µs`(<1%)、`latest_entry==last_exit 128/128`、`phase_sum_error_ticks {0,0}`。**critical-tail p50 `43.18 µs`**，校准分解 control `18.38 µs`(42.6%) / data-compute `24.82 µs`(57.5%)。最大可寻址项 = 三轮 notify control 合计 `16.51 µs`（不是 Wave2 publish completion `10.62 µs`）。pooled rank×epoch p50 `171.95 µs` 已显式降级为 **host 顺序提交 artifact**，不得当 collective latency。claude 补一条定律：`first_peer_control 0.700 µs` ≈ `marginal_per_peer 0.801 µs` → `cost(n)=a+b·n` 中 **a≈0、b≈0.80**，故 16.5 µs 是 21 次 notify 的边际成本而非每 wave 固定开销。上游侧补全五仓 branch 枚举，`PTOAS fix/issue711-tnotify-mte-drain` 已是生产 pin 祖先，是 **data-before-signal 正确性约束**、不可回退也不可摘。原件 `0162:/mnt/persist/chensiyu/workspace/p2-ar-diag-20260810/phase0-split-20260810-190834/`，复核 `0162:/mnt/persist/chensiyu/workspace/perf-2026q3/claude-verify-p2-phase0-20260810/`（`claude_verify_p2_phase0.py` sha256 `d8fccb90…`、`recompute.json` sha256 `dc6d3c01…`；**执行脚本只存在于 0162，本仓不留副本**） | 2026-08-10 |
| K2a | batched notify：`n stores + 1 barrier` vs `n stores + n barriers` 两点对比 | P0 | ❌ NO-GO | codex（claude 独立读原始 diag） | K1 | **已实测否决。** 三臂 A1 `4.46` / B `4.44` / A2 `4.46 µs`，`parent_center 4.460`、`half_range 0.000`、delta `+0.020 µs`(0.45%)、`output_exact=true`、`credit_errors` 全 0。反解每 barrier `0.0033 µs`，18 个可摘 barrier = `0.060 µs/call`，48 次 = `0.0029 ms` —— 比地板 `0.634 ms` 低两个数量级。**证实 K1 的 `a≈0` 定律**：`0.786 µs/notify` 几乎全部是 remote credit store 本身，`dsb`/`pipe_barrier(PIPE_ALL)` 份额≈0，所以 barrier 多重性批处理无价值，唯一杠杆是**减少 remote store 次数**（→ K5/RD 因此复活，K6 因此立项）。原件 `0162:/mnt/persist/chensiyu/workspace/p2-barrier-store-20260810/run-20260810-195604/` | 2026-08-10 |
| K2b | publisher release fence hoist（每 peer fence → 每 wave 一次）+ whole-cache invalidate 间接成本 | P0 | 🟦 | codex（claude 独立读原始 diag） | K1, K2a | 生产 notify 点位不是裸 TNOTIFY：`MakeNotifyCodegenPTO`（pypto `src/backend/common/pto_ops_distributed.cpp`）在每个 `pto.comm.tnotify` 前无条件 emit `pto.cmo.cacheinvalid all #gm` + `pto.fence.barrier_all #gm`。128 KiB 单档三臂（warmup 2 / measure 16，kernel sha256 A`8b264955` B`ec3ad385` C`f2be3406`，三臂 `output_exact=true`、`credit_errors`/`load_errors` 全 0）：`notify_us` p50 A per_peer_full `16.17` → B per_wave_full `13.78` → C per_wave_fence_only `13.56`，**A→B = `2.39 µs/call`**，落在预估 prefix 天花板 `2.70~3.13 µs` 内；C 只再省 `0.22 µs` → 收益几乎全来自 fence 提升、cacheinvalid 那一半贡献很小。**两条限定**：① 本轮只有单个 A 臂、无 A1/B/A2 bracketing，`2.39` 只能记作单臂差值，需补 bracketing 才能算「已测」；② claude 的「whole-cache invalidate 有间接成本、`transmission_factor>1`」假设 **as measured 不成立** —— 三臂 `post_minus_hot_us = cold_minus_hot_us = 0.0`，21/3/0 次 invalidate 下游 load 无差别。但 `cold_load ≈ hot_load ≈ 92.28 µs`（连显式 invalidate 都不让 cold 变慢）说明该 workset 是 GM-bound、探针可能结构上测不出 locality，此 null 一半属于仪器 → 不宣布假设死亡，但 K2b 维持在直接 prefix 天花板，不提高当前优先级。**落地约束**：`pld.system.notify` Python 签名（`system_ops.py:112`）无 fence/release 参数，K2b 改不了 `decode_fwd.py`，**必须上游 pypto 补丁** → 排在纯 model 侧的 K6 之后。原件 `0162:/mnt/persist/chensiyu/workspace/p2-k2b-release-cache-20260810/formal-128k-20260810/run-20260810-231938/`（另有 `bisect-20260810/` 记录多档 workset 触发 harness `code -100` 的定位过程，已绕开：128 KiB 单档即生产下游工作集） | 2026-08-10 |
| K3 | reduce→publish 按 chunk 流水 | P1 | ⬜ | — | K1 | 重叠上界 `min(reduce_owned 7.92, publish_completion 10.62) = 7.92 µs`，**单项越不过整机门 14 µs**，只能进 bundle；分块还会增加 TSTORE 启动数，实际收益可能显著低于上界 | 2026-08-10 |
| K4 | Wave3 双缓冲/epoch window + final-copy fusion | P1 | ⬜ | — | K1 | Wave3 `7.69 µs` 拆开为 notify control `5.51` + copy completion `2.18`；双缓冲打前者、copy fusion 打后者，**不得把 7.69 同时承诺给任一单项**。`self TPUT drain 3.92 µs` 只有让上游 producer 直接写 comm window 才可能删 | 2026-08-10 |
| K5-C | recursive halving RS + recursive doubling AG | P0 | ❌ device NO-GO | codex | K2a, K6a | C32′（FP32 RS + BF16 AG）在补齐正确性 fence 后 64 epoch byte-exact，但 critical-tail `42.63 µs` 对 parent `43.24/43.10 µs` 只改善 `0.54 µs/call`，远低于门。交易数虽从 35 降到 12，依赖深度却从 3 波增为 6 个串行 round；证明 K6 交易定律不能跨拓扑外推。不落地，不再作为 K11 后续路线 | 2026-08-11 |
| K6a | AR 行数按**编译期** storage capacity 特化（`BATCH = STORAGE_BATCH_CAPACITY`） | P0 | ✅ 收益已实测 | codex（claude 独立读原始 diag + 拟合） | K1 | **实测 `14.44 µs/call` = `0.693 ms`（48 次），80× half_range，首个单项越过 14 µs 门。** 十臂 A1/A2 bracket：`43.26 / 43.62` → center `43.440`、half_range `0.180 µs (0.41%)`，且与 K1 独立 campaign 的 `43.18 µs` 一致 → harness 复现基线。scan A（batch=tile=N，一次 N 行传输）p50 `29.00/30.10/32.86/36.00/43.26` for N=`1/2/4/8/16`。**落地不需改 kernel**：`config.py` 的 `BATCH = STORAGE_BATCH_CAPACITY = env("PYPTO_STEP3P5_STORAGE_BATCH_CAPACITY", 16)`，只服务 bs=1 的部署编译成 capacity=1 即可；唯一机械改动 `BATCH_TILE = 16` 需变 `min(BATCH, 16)`（否则 `pl.slice(..., [BATCH_TILE, ...])` 在 BATCH=1 越界）。**代价 = 放弃并发多行 batching，是部署配置取舍、不是纯优化。****副产物结论**：现有 bs=1 整网数字（34.3 ms 等）均在 capacity=16 下测得 → 一直背着这 `0.693 ms` 可避免的 AR 成本。**限定**：bs=16 收益 `0.18 µs`（=1× half_range=噪声），bs=8 只 `7.44 µs/call = 0.357 ms` 而 bs=8 地板 `2.637 ms` 需 `54.9 µs/call` → **K6a 在 bs=8 门下无望，只能作低 batch 收益**，且必须实测 bs=8/16 不回退。原件 `0162:/mnt/persist/chensiyu/workspace/p2-k6-active-rows-20260810/run-20260810-233142/`，复核 `perf-2026q3/claude-verify-k6-scan-20260810/`（`VERDICT.md` sha256 `cc589dbb…`） | 2026-08-10 |
| K6b | AR payload 按 runtime `active_tokens` 裁剪（dynamic valid shape） | P1 | ⏸ 历史中间方案（未落地） | codex | K6a | 历史 codegen 证明 remote-store 与 final local copy 可动态裁剪，但 remote-load 仍是静态 16 行；dynamic-destination TPUT 被 PTOAS 拒绝。把 valid shape 贯穿 reduction 还会触发 loop-phi dominance/convergence。dynamic publish 位于已知 notify-fence seam；现有设备运行未复现错误，但没有独立 rank-skew/zero-gap/多 epoch safety proof。该版本只作为 K11 的历史 focused 对照，不得再按旧建议进入产品实现；最终 `69ad31e4` 删除 dynamic DMA 依赖，改用静态 single-row selector + 静态三波 fallback。历史报告 `0162:.../p2-k6b-runtime-validshape-20260810/K6B-CODEGEN-GATE.md` | 2026-08-12 |
| K8 | persistent window 每步清零 `30.6 MiB` → 只清 `47,616 B` control 前缀 | **P0** | ✅ 已落地 + immutable image 已发布 | claude（codex landing review + reset 仪表） | K1 | **截至 2026-08-11 是首个「A/B/A + 精度门」双过的优化，也是当时最新的 immutable-image 优化。** 调研：每步 `_reset_persistent_domains` 清整个 `32,063,232 B` window，实际必需的只有 7 个 control counter 共 `47,616 B`（`0.1485%`）。实现两侧：模型侧 `decode_fwd.py`（+11/−7）把那 7 个 buffer 提到 16 个 alloc 的最前面，构成**唯一连续前缀** `[0, 47616)`；runtime `distributed_runner.py`（+174/−22，含 codex 的 `PYPTO_PERSISTENT_RESET_TRACE` / `reset_body_us` / `memset_all_us` 仪表）只 memset 该前缀，并带 **16-buffer 指纹 fail-closed 回退全清**（指纹不匹配或名字变体 → 全清），且**只对 WholeDecode 生效**（v3 硬化；v1 被 landing review 否掉）。**源码级 A/B/A 双 bracket**：ITL p50 `33.84 → 32.08 ms`（**−1.7455 ms / −5.16%**，89.5× floor）、`_reset_persistent_domains` body `2253 → 518 µs`、`hidden_sha256` `567b206b…` byte-exact + token `14371`。**镜像级复现**（manifest `sha256:076af8a1…c47f3`）：p50 `32.14 ms`（与候选臂差 `0.06 ms` ≪ floor）、reset body p50 `523.1 µs`、109/109 步 `k8_prefix_applied=true`、N=128 三轮 `123/128` 且与 Wave5 逐位相同。落地件 sha：`decode_fwd.py` `eb1f89bf…04fb5`、runtime `fe50c11f…39622e`。**衍生负结果**：天花板探针失败 —— 语义无效的臂不能界定性能上界（见更新日志）。数据见 [`../../benchmark/2026-08-11-k8-selective-window-zeroing.md`](../../benchmark/2026-08-11-k8-selective-window-zeroing.md)（源码级）+ [`../../benchmark/2026-08-11-k8-selective-window-zeroing-image.md`](../../benchmark/2026-08-11-k8-selective-window-zeroing-image.md)（镜像级） | 2026-08-11 |
| K9 | 删 Wave3（final-copy 波次） | P1 | ❌ 整网 NO-GO（实测否决，保留为负结果） | claude | K1 | bench 层面看似省，但整网 A/B/A **ITL `+1.72 ms/step`，符号与 bench 相反**；byte-exact 成立但性能回退 ⇒ **不落地**。价值在于确立「bench 收益不等于整网收益，符号都可能反」这条纪律，以及暴露中间臂偏置（见更新日志 2026-08-11 校正行） | 2026-08-11 |
| K10 | 去掉剩下那一次**阻塞** host control round | P1 | ⬜ | — | K8 | K8 之后 reset 路径仍有一次阻塞 host↔device control 往返，上界 **`0.45–0.53 ms/step`**。⚠ **两条口径纪律**：① **不得**表述为「4.4 ms 异步化 reset」；② **不得**引用已被否证的 no-reset 探针（语义无效臂）。实施顺序：① device 侧 zero prologue → ② request epoch/generation → ③ async / 双缓冲。评估只认整网 A/B/A + byte-exact | 2026-08-11 |
| K11 | HCCL selector 思路的 single-row one-shot mesh | **P0** | ✅ 已合入 + IMG | codex | K1, K6b | `69ad31e4` 实现已被当前 `e6c7d8ec` 与 r12 继承。rank-uniform `active_rows==1` 走静态 8 KiB 两波 mesh，其余 Main/MTP 走三波 fallback；旧 `a791071` ring 禁止恢复。r12 Main/MTP immutable gate PASS；不能把历史 focused pooled mean 当 strict ITL | 2026-08-27 |

**Track K 历史三波 parent 拟合（2026-08-10，已 supersede）**

历史静态三波 parent 曾拟合为：

```text
AR_cost ≈ 0.80 µs × N_remote_transactions + 0.94 µs/row × active_rows
```

K1 的 remote credit store（`0.8010 µs`）、K6 scan B 的 payload transfer
（`0.7616 µs`）以及 K6 scan A 的截距（实测 `28.442 µs`，35 次交易预测
`28.035 µs`）在**同一三波拓扑内**互相吻合。它解释了 K2a 批 barrier 为何无收益，
也促成了按 active rows 优化的方向。

该拟合不是跨拓扑性能模型：K5-C 把交易数从 35 降到 12，却因依赖深度变成六个
串行 round，设备实测只改善 `0.54 µs/call`，已经否证旧 path2 外推。K6b 的
dynamic-valid-shape 也未落地。旧 path1/path2 组合、`26.8 µs/call` 预测和据此给出的
实施顺序全部退休，不再指导产品实现。

当前产品结论只以 K11 为准：单行走静态两波 one-shot mesh，其他 Main 行数与 MTP
走静态三波 fallback；Whole A/B/A 已证明 `-1.120 ms / -3.609%`，后续只做
immutable-image qualification，不再恢复旧 Ring、K5-C 或 K6b 路线。

---

## 进度汇总

| 状态 | 数量 |
|------|------|
| ⬜ TODO | 12 |
| 🟦 IN PROGRESS | 4 |
| ✅ DONE | 26 |
| ⏸ PAUSED / SUPERSEDED | 1 |
| ❌ / ⛔ NO-GO | 5 |
| ⛔ BLOCKED | 0 |
| **合计** | **47** |

**base 校正后关键路径**：A1/B1/B2/C1/C2/C3/C4/D1/D2/G1/H1/**H4/H5/H6/H7**/I1/I2/I3/I4/I5/I7/J2/K1/K6a/**K8/K11** 已 ✅；K6b 已降为被 K11 supersede 的历史中间方案。
其中 **K8 是已通过整网 A/B/A + 精度门并发布 immutable image 的 host-reset 优化**
（ITL `33.84 → 32.08 ms` 源码级 / `32.14 ms` 镜像级，byte-exact + N=128 `123/128`）。
K11 已被后续 immutable image 收录；historical pull C2 仅作回归基线；C5、K2a、K5-C 与 **K9** 是四个实测否决的负结果（K9 整网
`+1.72 ms/step`，符号与 bench 相反）。当前 performance 看板进行中的是
**B3（KV resident/in-place 的连续多轮 row-diff/liveness 证据）**、**J1（formal DFX /
publication / swimlane 收尾）**、**J3（active-worker dual-latch canonical/IMG 准入）**、
**K2b（publisher release fence hoist，需上游 pypto 补丁）**。H4/H5/H6/H7 已收口；当前
performance 第一优先是为 `655+a745` 补 exact `recv_meta` 并完成 successor immutable image gate。
`bind.args` 继续优化已按 `0.259%` / `no_clear_change` 判为 NO-GO。
`K10`（上界 `0.45–0.53 ms/step`，K8 的直接后继）排在 deployment 接线之后 ——
**注意 K10 的上界低于近期整网 bracket 地板
`0.616 / 0.634 ms`，只有拿到紧 bracket（K8 曾达 `0.0195 ms`）才可判**。
排序依据见 [`09-swimlane-derived-next-optimizations.md`](09-swimlane-derived-next-optimizations.md)。
K5-C 已由设备实测判定 NO-GO，不再列为候选；K11 已被后续 r12 immutable gate 继承验证。
历史 I1 Attention/Vec 与 TP all-reduce stability 已在 0162 release-qualified；
I3/I4 的 `f9065261` 当前仅为 source-overlay GO；I6 的 `fa58b5cf` 为
source-overlay 实现/精度 PASS、性能 **NO-GO**；I7 的 `e5e26f9f` 只对
`fa58b5cf → e5e26f9f` 的 RMS→QKV 调度补丁判定 GO，不能覆盖 I6。J1 产品实现
和六档 64K 历史 normal gate 已完成；r12 Main/MTP/dep-only DFX 已闭环，但当前
六档 r12 campaign 尚未重跑，不能借旧镜像数据补齐。
其它机器和整网集成仍需独立 gate。

---

## 认领指南

1. 从「可立即并行认领」或依赖已 ✅ 的任务里挑。
2. 先核对current source、当前调用链和工作树diff，完成producer→数学变换→transport/window→consumer→rounding/reduction→lifetime审计。
3. 在对应行填 **Owner**，状态改 🟦，更新「最后更新」；阻塞/验证栏写明五类差异归属与证据等级。
4. 落地前读 [`02-detailed-design.md`](02-detailed-design.md) 对应卡片（file:line + 算法步骤 + 验证口径）。
5. 完成：**多步 decode 逐 token** vs vanilla vLLM W8A8，seed=6127 / N=128 ≥95% ALIGNED（`pypto-lib/tests/step3p5/ci/LIVE_PRECISION_AB.md`）→ 状态改 ✅ → 在下方「更新日志」记一行（做了什么 / commit / 验证结果）。多步已含第一个 token；stall 用 `_probe_barrier_scale.py`（liveness，独立）。
6. 遇阻：状态改 ⛔，「阻塞」列写原因，必要时在 [`../../blockers.md`](../../blockers.md) 登记。

---

## 更新日志

| 日期 | ID | 变更 | 备注 |
|------|----|----|------|
| 2026-08-30 | J3 | routed GMM active-worker dual-latch feature branch 完成 source-overlay GO | `a745ab6`（parent `e6c7d8ec`）已推送；H4 A/B/A `21.099/20.172/21.107 ms`，收益 `0.931 ms / 4.4117%`，hidden/token exact；whole compile、focused/full unit、五层结构 DFX 与 L3/L4 exact PASS。fixed-22 `0.588 ms` NO-GO。缺 exact `recv_meta`，publication `NOT_EVALUABLE`；待 canonical merge 与 IMG gate。 |
| 2026-08-27 | H6 / H2 | prepared TaskArgs cache 合入并发布 r12；H2 改为 residual graph-build 重画像 | pypto `14de90fd`；r11 source-overlay A/B/A：ITL `−2.608%`、graph build `−44.429%`、graph→first runner `−47.936%`、serial submit envelope `−23.887%`，hidden/token exact；r12 immutable Main/MTP/dep-only DFX 与 1844/1844 contract PASS。正式仍是 serial 8-rank，bind `0.259%` 不再优化；H2 是否仍有 ROI 需先拆候选剩余 `2.2743 ms` graph build |
| 2026-08-24 | H4 / H5 / K11 | upgrade r9 immutable admission + 五仓同步 | manifest `b637f00c…`、config `f6c8f72e…`；H4 all 令 `bind.args 6.461→0.063 ms`、64K/1000 `27.812→22.253 ms`，all/none 输出 parity；H5 8/8 chip records、L3/L4 exact、analyzer + outer admission PASS；Main/MTP liveness 与 precision `127/128` PASS。五仓 `stepfun/develop` 已同步。镜像未 bake H4 env，deployment 接线仍 open |
| 2026-08-12 | K11 | single-row small-message selector 合入 `stepfun/develop@69ad31e4` | `active_rows==1` 为静态 8 KiB 两波 one-shot mesh，其余 Main/MTP 为静态三波 fallback；ownership/transfer chunk 解耦。unit `365 passed, 7 skipped`；Main/MTP default+chunk256 compile、8 卡 rows `1/3/16` PASS。focused 历史 regular-call kernel-duration pooled mean `38.325→22.667 µs/call`（非 strict critical-tail）；Whole A/B/A `31.065/29.912/30.999 ms`，delta `-1.120 ms/-3.609%`，precision/per-iteration PASS。landing tree `e26d762c…`；未构建新 immutable image。 |
| 2026-08-12 | I7 | RMS→QKV critical prestage 已完成设备门并合入 `stepfun/develop` | baseline `fa58b5cf` 的 QKV raw-kernel start − RMS raw-kernel end 为 min/p50/max=`4.60/5.00/5.48 us`；最终 `e5e26f9f` 为 `2.08/2.64/3.16 us`，p50 减少 `2.36 us`（47.2%）。更直接的调度证据：QKV Worker gap p50=`+4.77 → -1.78 us`，setup 已与 RMS 重叠；candidate RMS Worker span=`4.16/4.35/4.78 us`。方案：RMS producer 开 early resolve；保留 QKV producer flag 优化下一跳；用已存在的 `swa_attn_out_zero` TaskId 把非关键 head-gate 隔离为 normal dispatch。target unit `22 passed`，五层 L3/L4 exact。整网 A/B/A=`30.992/30.997/31.136 ms`、hidden/token exact、`WITHIN_BASELINE_BRACKET`。commit 链 `18d1b519..e5e26f9f` 已 FF push；all-ranks bundle SHA `2f58af78…`。I6 NO-GO、canonical zero-route limitation 和新镜像未构建边界均不变。 |
| 2026-08-12 | I6 | post-merge 整网 ITL 与 fresh 五层 DFX 将状态改为 NO-GO | 最终 clean `fa58b5cf`、固定 manifest `sha256:076af8a…`、BS1/ctx64K/512 blocks。A/B/A p50=`31.787 / 33.194 / 31.905 ms`，candidate 相对 baseline center `31.846 ms` 回退 `+1.348 ms / +4.233%`；三臂 hidden SHA `567b206b…`、finite、token `14371` exact，精度 PASS。fresh 五层 8-rank swimlane 共 8 份，strict gate 39/40，rank7/L0=`54.54 us`；该点为约 12 us AICPU scheduler dispatch stall，kernel compute 与依赖正常。campaign=`0162:/mnt/persist/chensiyu/workspace/perf-2026q3/qkv-prerope-postmerge-validation-20260811-r1/`；fresh report JSON `0b5cbe20…`，whole ABA result `065f67c8…`。 |
| 2026-08-11 | I6 | packed QKV projection + split/QKNorm/RoPE 独立严格五层门完成并推送 | 0162 candidate worktree=`/mnt/persist/chensiyu/workspace/develop-worktrees/qkv-prerope-mix`，branch=`perf/qkv-prerope-mix-20260811`，base=`f9065261`；commit `fa58b5cffe41b30d3f8d94482230867ee34b9e84` 已 fast-forward push，origin、main/candidate 三者同 commit 且 clean。图收敛为 `qkv_proj → qkv_split_qknorm_rope → attn_mix`；Full/SWA inventory=`10+1 / 14+1`，旧 projection/norm/rope family=0。unit `362 passed, 7 skipped`，whole compile `rc=0`，focused 三臂 `container.rc=0`，12 edge、Q-publication 12/12、heterogeneous context 与 L3/L4 precision PASS。严格 merged Worker View **40/40 PASS**，global max `43.60 us`，margin `2.40 us`；分层范围 L0 `42.40–43.60`、L1 `39.10–41.60`、L2 `38.80–41.14`、L3 `39.30–41.16`、L4 `38.96–41.62 us`。证据根 `0162:/mnt/persist/chensiyu/workspace/perf-2026q3/qkv-prerope-final-device-gate-20260811-r1/`，report JSON SHA `e12e6bd2…`、MD `96467afb…`、seal `e2be0cad…`。固定镜像内仍为 `cb96747e`，只作 source-overlay Attention gate；canonical analyzer 的 known `rc=1` 不得升级为 structural/release PASS。 |
| 2026-08-11 | I5 | 最终 `f9065261` 前五层 8-rank DFX limited delivery 完成 | L0–L4、BS1、ctx64K source-overlay capture；L3/L4 baseline-candidate exact/finite/spread=0；LOW-WAIT rank2 `2.124 ms`。L3 RMSNorm 8 tasks/8 distinct cores，全 rank slice/span max `4.28/4.30 us`；L0/L4 Full 各 24 mixed blocks，L1/L2/L3 SWA 各 1，forbidden split=0。candidate container `rc=1`；canonical analyzer 对 rank0/1/3/6 各缺 5 个零本地 routed-token early-dispatch record 返回 `FAIL_CLOSED`，所以 status=`LIMITED_NOT_RELEASE_QUALIFIED`。all-ranks bundle SHA `d6f689c7…`，LOW-WAIT bundle `e0bb2cc2…`，seal `088cf05f…`。 |
| 2026-08-11 | I3 / I4 | SWA RMSNorm 多核与 Attention mixed kernel 已推送到 `stepfun/develop` | GitHub 与 0162 local develop 均为 `f9065261`、本地 clean。I3：8 logical tasks/8 distinct cores，block max `4.46 us`、logical span max `4.90 us`，strict `<5 us` 与 L3/L4 exact PASS。I4：`full_attn_mix`/`swa_attn_mix` 上线，旧 split family=0；unit `357 passed, 7 skipped`，whole compile、12 个 edge exact、12 个 Q-publication、focused mixed-kernel 8-rank DFX 全 PASS。BS1/ctx64K A/B/A=`32.222/31.790/32.330 ms`，delta `-1.506%`，precision PASS。全部为 manifest `sha256:076af8a…` 上 source overlay，不是新镜像；证据见 [`../../benchmark/2026-08-11-step3p5-attention-mix-rmsnorm.md`](../../benchmark/2026-08-11-step3p5-attention-mix-rmsnorm.md)。 |
| 2026-08-11 | K8 | **★★ K8 immutable image 已发布并在 0162 通过双精度门 + ITL；五层 swimlane 已采到（cross-rank 契约未过）** | `hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260811-k8-selective`，manifest `sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3`、config `sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39`，spec `deployment/docker/builds/stepfun-develop-20260811-k8-selective.env`。**第一个包含 K8 发布时 tip** `cb96747e`/`1c048a74` 的 immutable image。构建在 devbox（0162 无 buildkitd、github/proxy 均不通，结构上不能构建），**验证全部在 0162、digest-only、无 overlay**。<br>**落地件==被测件**：`distributed_runner.py` sha `fe50c11f…39622e`、`decode_fwd.py` sha `eb1f89bf…04fb5`，与落地行记录一致。<br>**精度两条独立证据都过**：① byte-exact `hidden_sha256` `567b206b…f03e` == 生产 baseline、token `14371`；② **N=128 预定义冻结 oracle 三轮 `123/128 = 96.09375%`、miss `[2,8,13,22,82]`、`tp_spread_max=0.0`，三轮一致且与 Wave5 逐位相同** ⇒ K8 未改 token 轨迹。oracle `0162:.../attn-opt/out/fresh_vanilla_oracle_20260731/oracle_ids.json` sha `c9b2c721…dd947`（与 Wave3 快照同 sha），离线无 live server，checkpoint 身份一致。<br>**性能（clean 非插桩）**：bs=1 ctx=65536 blocks=512 warmup=10 iters=100，p50 **`32.14 ms`**（min 31.766 / mean 32.467 / p99 37.644）。对 pre-K8 `33.84` = **−1.70 ms / −5.02%**；与 source-overlay 候选臂 `32.08` 差 `+0.06 ms` ≪ 地板 `0.634 ms` ⇒ **镜像复现 K8 收益**。<br>**K8 runtime 生效**：109/109 步 `k8_prefix_applied=true`、`k8_control_bytes=47616`（单段）、`reset_body_us` p50 `523.1 µs`（A/B/A 实测 518）。<br>**五层 BS1 swimlane**：LOW-WAIT=`rank2`，makespan `2.204 ms`、static CPM `1.825 ms`(82.8%)、observed 103 task（compute `1.788` 81.2% + stall `0.415` 18.8% 全 data-wait）、tiling exact；`tp_all_reduce` 占 **15.3%**（8 次）。⚠ 其余七 rank makespan `288~610 ms`、`tp_all_reduce` 占 99.4%+ 是**自旋吸收 skew 不是算力**（跨 rank 差 275×）。⚠ campaign `rc=1`：分析器 fail-closed 契约在 rank0/1/3/6 拒收（各 5 个 `early_dispatch=true` task 无 swimlane 记录）⇒ 记为**可用观测，非 sealed publication**。<br>**构建期修掉三个同族凭据坑**（凭据覆盖没生效→静默退化匿名 GitHub→proxy 401）：① submodule 覆盖 key 必须用**名字** `simpler` 而非路径 `runtime`；② pypto CMake 在无 secret 层 init `3rdparty/{libbacktrace,msgpack-c}`；③ simpler 另有**第二份** pto-isa（`runtime/build/pto-isa` 按 `pto_isa.pin=83d01313`，`PTO_ISA_ROOT` 对它无效），按上游支持的 warm-cache 预置。①③ 应落 sub-repo `dev-workflow-gotchas.md`。<br>⚠ **未跑**：Main batch16、MTP batch1/16、六档 64K golden/A/B、formal DFX；phase 2b 按用户要求跳过（启动 68 s 干净停容器，NPU 8-15 无残留）⇒ **不是完整 production release-qualified**，完整矩阵回退基线仍 Wave5。数据见 [`../../benchmark/2026-08-11-k8-selective-window-zeroing-image.md`](../../benchmark/2026-08-11-k8-selective-window-zeroing-image.md) |
| 2026-08-11 | K8 | **★★ K8 已落地：pypto `1c048a74` + pypto-lib `cb96747e`，两份文件逐字节等同被测件** | 用户授权「直改 pypto fork」+「精度正确则合入 `stepfun/develop`」后落地。<br>**pypto** `8e92b468 → 1c048a74`（2 commit，只改 `python/pypto/runtime/distributed_runner.py`，+174/−22）：codex 的 reset instrumentation（`PYPTO_PERSISTENT_RESET_TRACE`、`reset_body_us`、`memset_all_us`）+ K8 v3 选择性清零。<br>**pypto-lib** `27a43f6a → cb96747e`（1 commit，只改 `models/step3p5/decode_fwd.py`，+11/−7）：7 个 control buffer 提到 16 个 alloc 的最前面，构成唯一连续前缀 `[0, 47616)`。<br>**落地件 = 被测件（逐字节校验通过）**：`decode_fwd.py` sha `eb1f89bf7add419f2382836c1eab9a1c4b1f63f738923d47e771e4159f104fb5`、runtime sha `fe50c11fb76ec77789636de05e7376711c731d2b00db5033f0564c07a739622e`（见 v3 确认臂行的权威 sha）。<br>**落地后重跑的门**：AST dead-name 门、3 例 layout 门（whole-decode 走 47,616 B 前缀 / five-layer 指纹不匹配 → 回退全清 / 名字变体 → 回退全清）、`ALLOC_ORDER_OK 16 allocs, first 7 are control`、无卡 codegen 门 `VERDICT=PASS`。<br>**过程坑**：pypto-lib push 被 `--force-with-lease` 正确挡下（远端已前进到 codex 的 `27a43f6a`）→ 把 fork tip bundle 回 0162、在 `base-tree` rebase、重验 `decode_fwd.py` sha 未变、再 bundle 回来推。安全 tag `k8-prerebase-20260811` → `64898e13`。<br>**遗留**：落地版的 `trace_domains.append({...})` 仍无条件构建（只有写文件受 env 门），为保「落地==被测」没有顺手加门，已问 codex 是否另起一 commit 补。 |
| 2026-08-11 | K8 | **★★ K8 v3 确认臂 PASS：`−1.7455 ms/step`（`−5.16%`，89.5× floor），落地件已是被测件** | campaign `0162:.../k8-selective-20260811/v3-20260811-144622`，三臂 rc=0，`K8_V3_RESULT.json` sha256 `7bb0226326cfe77f9af2d2789673f81da99ab12b651a5c7495ba8e876561a045`。<br>`A1_parent 33.842 / A2_parent 33.803` ⇒ floor **`0.0195`**（本轮最小）；`B_reorder_prefix_v3 32.077` ⇒ **`−1.7455 ms`（`−5.16%`，89.5× floor）**，bias-corrected `−1.565 ms`。`reset_body 2260.3 / 2246.1 / 518.3 µs` ⇒ `−1734.9 µs`；`memset_all 2239.7 / 2222.1 / 476.4 µs`。<br>**两个独立 bracket 一致**：v1（`prefix-20260811-135232`）`−1.7505 ms`、v3 `−1.7455 ms`，**差 `0.005 ms` = 0.26× floor** ⇒ 硬化没有代价，效应可复现。<br>**两道门都 PASS**：`PRECISION_GATE=PASS`（三臂 sha 全 `567b206b…`、`token=14371`、`all_finite`、`matches_production_baseline_sha`）；`PREFIX_APPLIED_GATE=PASS`（B 的 trace `k8_prefix_applied=true` + `k8_control_bytes=47616`，两个 A 臂为 `None`）—— 后者专门用来挡「静默回退到全窗清零会长得像没收益」。<br>**⇒ 落地件锁定（就是被测的这两份，不许重写）**：模型侧 `src_k8reorder/models/step3p5/decode_fwd.py` sha `eb1f89bf7add419f2382836c1eab9a1c4b1f63f738923d47e771e4159f104fb5`；runtime 侧 `runtime/distributed_runner_prefix_v3.py` sha `fe50c11fb76ec77789636de05e7376711c731d2b00db5033f0564c07a739622e`。驱动 `bin/run_k8_v3_aba.sh` sha `c36bb537…`，离线门 `bin/claude_check_prefix_v3_layout.py` sha `9f81ee67…`。**已按用户授权推产品仓（见本表首行落地记录）。** |
| 2026-08-11 | K8 | **⚠ 方法论：天花板探针失败 —— 语义无效的臂不能用来界定性能上界** | 我想量出「reset 路径的全部成本」以判断「reset 异步化」值不值得开，做法是保留整个 reset 路径与埋点、**只跳过那一次 `memset_all`**（`distributed_runner_noreset.py` sha `487bc687…`，diff 恰好 4 行）。**这个探针作废。**<br>campaign `0162:.../k8-selective-20260811/ceiling-20260811-142404`。`A1_parent 33.776`（rc=0、sha `567b206b…`）；`B_noreset_ceiling` **`itl_p50 = 36.537 ms`**、`min 35.713`、`reset_body 19.96 µs`、`memset_all 0.15 µs` —— **比 baseline 慢 `+2.76 ms`**，尽管它把 2250 µs 的 reset 全省了。精度如预期 FAIL：`sha bd6eb03abb05…`、`token 81596`（期望 14371）、`finite=true`。<br>**⇒ 不许把 `+2.76 ms` 解释成 reset 路径的任何量。** 跳过清零后 control counter 带着上一步的陈旧值，wait 被提前满足 / rank 次序错乱，跑的是**另一个程序**；它的耗时不等于「正确程序减去被删的工作」。这与 K9 的发现同源：这些 window 的同步结构在给 rank 定速，破坏它会引入下游等待。<br>**方法论规则（新增）**：**ceiling probe 必须保持语义有效**，否则它既不是上界也不是下界。要界定 reset 的剩余机会，唯一有效的工具是**已测到的剩余 reset wall（K8 后 `530 µs`）+ 字节约 1:1 传到 ITL 的定性规律** ⇒ 预算 **`0.45–0.53 ms/step`**（与 codex 独立给出的 `~0.53 ms` 收敛）。**不要**再引用「no-reset 能省 X」这类数字。<br>**副产品（唯一可用的信息）**：control counter 确实是承载正确性的，且陈旧 counter 造成的是**变慢 + 变错**，不是只变错。<br>另记一个 harness 约束：arm runner 的容器脚本末尾硬断言 `token_exact`，所以**任何精度必然失败的臂都会 rc=1 并让 `set -e` 中止整个 campaign**（本次 C/A2 未跑）。以后要跑 known-bad 臂，必须先给 runner 加一个显式的 `allow_precision_fail` 开关，而不是靠读 rc=1 之前落盘的数据。 |
| 2026-08-11 | K8 | **★ K8 落地件硬化到 v3（codex landing review 抓到 v1 不能落地）** | **被测的 v1 不能原样进 `pypto.runtime.distributed_runner`**：它把 WholeDecode 的 16-buffer allowlist 应用于**每一个** persistent CommDomain，而仓内还有 `five_layer_moe_holder` / route holder / two-layer attention tests / multi-program persistent 等路径，buffer 名不同 ⇒ 会在 dispatch 之前撞 unknown-name raise。**已离线实证**：对一个合成的 `five_layer_*` domain，v1 抛 `unknown comm-window buffer base name 'five_layer_route_buf'`，v3 正常回退到全窗清零。<br>**v3 改法**（`runtime/distributed_runner_prefix_v3.py` sha `fe50c11fb76ec77789636de05e7376711c731d2b00db5033f0564c07a739622e`，applier `bin/claude_apply_k8_prefix_v3.py`）：① 用 base-name **多重集精确匹配** WholeDecode 的 16 个 buffer 才启用；② 其它 domain 保持原全窗清零，未识别的名字**不是错误**；③ WholeDecode 内部仍 fail-closed（control 出现在 data 之后 / control 字节数偏离 pinned 47,616 / carve 与 window 不符 ⇒ raise）；④ trace 增 `k8_prefix_applied`，发布门要求为 `true`，**否则「静默回退到全窗清零」会长得像「没收益」而不是像错误**。<br>**离线门**（`bin/claude_check_prefix_v3_layout.py` sha `9f81ee67…`，直接 exec 目标方法体 + 假 worker）三例全过：`CASE1` 重排后 applies=True / 47,616 B / 1 broadcast；`CASE2` 旧布局被拒；`CASE3` 无关 domain 回退到 3072 B 全窗、不 raise。v1 对同一门 FAIL（positive control）。<br>**确认臂在跑**：`v3-20260811-1446*`，三臂 `A1_parent / B_reorder_prefix_v3 / A2_parent`，因为**落地件必须是被测的那一份**。<br>**codex 同时收紧了我的机制口径（接受）**：`1.95×` 不是可复用的固定开销系数 —— 同一实验按 p50 是 `1.95×`、按 mean 是 `1.67×`，字节路径 `1.017×`（C vs parent）/ `1.066×`（C vs 同布局 B）；broadcast 次数 1/2/3 未测，ITL 与 reset 的分位数也不是逐迭代配对。⇒ **只保留两条定性规律**：多一次阻塞 broadcast 很贵；大块 memset 的字节收益约 1:1 传到 ITL。两因子模型禁止定量外推。<br>**下一候选的正确表述**（codex 建议，我同意）：不是「4.4 ms 异步 reset」，而是「**去掉剩下那一次阻塞 host control round**」，上界 `0.45–0.53 ms/step`。且 `memset_all` 与 task dispatch 共用 per-chip `mailbox_mu_`，control 请求在 dispatch 中途会等 `TASK_DONE`，**所以「把现有调用改异步」必须先有新的 runtime ordering/event + 双缓冲或 epoch 语义**。建议顺序：① device-side zero prologue（每 rank 一个任务清 7 个 control tensor，直接删掉 host broadcast）；② request epoch/generation 彻底免 reset；③ 最后才考虑双缓冲 + 异步 API。codex 报告 `0162:.../k8-selective-20260811/CODEX_K8_LANDING_REVIEW_20260811.md` sha256 `42f02957716fc0804627150b790e44b90248752341e8f262ca537ced88f4fdd7`。 |
| 2026-08-11 | K8 | **★★ K8 整网 GO：control-prefix 重排 + 单次 47,616 B 清零 = `−1.7505 ms/step`（`−5.18%`，66× floor），byte-exact** | 这是**本轮第一个在整网 ITL 上被证实的优化**，也是唯一一个已过 A/B/A + 精度门的候选（K6b 的 `7.02 µs/call` 仍只有 bench 证据）。<br>campaign `0162:/mnt/persist/chensiyu/workspace/k8-selective-20260811/prefix-20260811-135232`，四臂 rc=0，`K8_PREFIX_RESULT.json` sha256 `f3b81a91513597f32331e4972e6d8e2d44224dbe4e33e5dfc0dffdfb92300751`。<br>**四臂设计（B 臂专门隔离「重排本身」）**：`A1_parent`（`src_parent` + baseline runtime）→ `B_reorder_fullclear`（`src_k8reorder` + **baseline** runtime，即只重排、清零一字不动）→ `C_reorder_prefix`（`src_k8reorder` + `distributed_runner_prefix.py`）→ `A2_parent`。<br>**结果**：`A1 33.776 / A2 33.829` ⇒ floor `0.0265`；`B 33.879` ⇒ **`+0.0765 ms`（2.9× floor，落在历次 A 臂 33.776–33.933 的自然散布内）⇒ 重排本身中性**；`C 32.052` ⇒ **`−1.7505 ms` raw / `−1.570 ms` 经 `−0.1805` 中间臂偏置修正 / 66× floor**。`C` vs `B`（重排 held constant，纯前缀效应）= **`−1.827 ms`**。<br>`reset_body` `2254.1 / 2247.8 / 2245.1 / 530.4 µs`、`memset_all` `2233.0 / 2223.4 / 2221.9 / 496.9 µs` ⇒ C 把 reset wall 砍掉 **`1720.6 µs`**。C 臂 trace 自证 `k8_control_bytes=47616`、`k8_control_range_count=1`、`k8_control_ranges=[[0,47616]]` ⇒ **确实是一次 broadcast、47,616 B**。<br>**精度门 PASS**：四臂 `hidden_sha256` 全 `567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e`、`token=14371`、`all_finite` ⇒ **不清那 30 MiB data buffer 一个 bit 都没变**，与 codex 的 window 审计（每个 data buffer 每步先写后读、无跨步残留依赖）实测一致。<br>**⇒ 修正上一行的机制模型**：字节传输时间在 ITL 上的放大是 **`1750.5 / 1720.6 = 1.02×`**，**不是 1.95×**。1.95× 只适用于「额外 broadcast 的固定开销」，两个 regime 必须分开记账。上一行用 1.95× 外推字节部分（`−3403 µs`）是错的，那个 `+687 µs` 残差因此**不再是需要解释的物理量，而是模型外推错误的产物**；用两因子模型重算 selective 得 `+5×920.4 − 1720.6 = +2881 µs` vs 实测 `+1886 µs`，残差换了符号 ⇒ **两因子模型也只对到量级，不要再拿它做定量外推**。<br>**与 codex ballast 交叉验证**：codex 双 memset 探针给「整块清零 ≈ `1943 µs` wall」，减去其自带的一次固定开销 `472 µs` ⇒ 字节部分 `≈1471 µs`；本臂实测字节部分 `1720.6 µs`。同量级、差 ~17%，互为独立验证（不同机制、不同卡、不同臂序）。<br>**落地件（就是被测的那一份，不许重写）**：模型侧 `src_k8reorder/models/step3p5/decode_fwd.py` sha `eb1f89bf7add419f2382836c1eab9a1c4b1f63f738923d47e771e4159f104fb5`（16 个 alloc 数量不变，7 个 control 移到最前）；runtime 侧 `runtime/distributed_runner_prefix.py` sha `de1301234f533655818bd3cb6ef32c6df1eeecc3b34072c0db314823bf0338e9`（一次 `memset_all` 覆盖 `[0, control_bytes)`，精确 allowlist，control 出现在任何 data 之后即 raise）。驱动 `bin/run_k8_prefix_aba.sh` sha `0da3012d…`，布局检查器 `bin/claude_check_prefix_layout.py` sha `af37270f…`。<br>**fail-closed 已实证**：布局检查器对旧顺序返回 `REFUSES_OLD_LAYOUT: control 'dense_attn_signal_stack_buf' after 1 data buffer(s)` —— 「以后有人往 control 前面插 data buffer」会在取锁前被拦下，不会静默清错范围。三道 pre-flight 门（两次 AST 未定义名检查 + 布局检查 + 拒旧布局自证）全部在取锁前通过。<br>**未落地**：这是**生产代码改动（模型 + runtime 各一处）**，需用户授权才推。仍缺 codex 对「重排是否踩到隐藏假设」的独立复核（是否有 kernel/host 代码假设某 buffer 位于 window 起点或假设对齐 —— 我的论据是身份由返回指针决定、且 `moe_recv_x` 本来就落在非 512 对齐的 6,317,824）。 |
| 2026-08-11 | **★ K8 机制定量：一次阻塞 `broadcast_control_all` = `472 µs` host wall / `920 µs` ITL（放大 1.95×），与字节数无关** | 加性对照：保留全量清零不变，只多做 5 次对已归零 range 的幂等 memset ⇒ reset `+2361 µs`、ITL **`+4.602 ms`**（43.8× floor） | campaign `0162:/mnt/persist/chensiyu/workspace/k8-selective-20260811/bcast-20260811-133*`（dev0-7，三臂 rc=0）。arm `B_extra5bcast` runtime sha `515064e5…`（diff 16 行）：**全量 memset 一字不动**，其后追加 5 次对 `(393216,1536) (787968,1536) (6294528,21504) (6317312,512) (25487104,512)` 的阻塞 memset —— 这些 range 刚被全量清零覆盖，所以**幂等、零语义变化**，是纯 broadcast-count 探针。<br>结果：`A1 34.021 / A2 33.811`（floor `0.105`）/ `B 38.518` ⇒ **ITL `+4.602 ms`**；`reset_body 2240.7 / 2250.2 / 4606.9 µs` ⇒ **reset `+2361 µs`**。三臂 sha 全 `567b206b…`、`token=14371` ⇒ 精度 PASS（幂等改动本应如此，用作 harness 自证）。<br>**⇒ 单位成本**：每次额外 broadcast = **`472.3 µs` reset wall / `920.4 µs` ITL**，**放大 `1.95×`**。即阻塞式 `broadcast_control_all` 在 ITL 上的代价约为它 host wall 的两倍 —— 它不只占用 host 时间，还打断了本可掩盖 reset 的 host/device 重叠。**这是本轮最有用的机制常数**，且与字节数无关（这 5 次每次只清 0.5–21 KiB）。<br>**用它分解 K8-selective 的 `+1.886 ms`**：selective = 「+5 次 broadcast」+「−32 MiB 字节」。前者按上式 = `+5 × 920.4 = +4602 µs`；后者字节部分的 wall ≈ `2217 − 472 = 1745 µs`，若同样按 1.95× 放大 = `−3403 µs` ⇒ 预测净 **`+1199 µs`**，实测 `+1886 µs`，**残差 `+687 µs` 未归因**。⇒ **selective 的惩罚主要是 broadcast 次数（1.2 / 1.9 ms 可解释），不是「不清数据本身」**；残差 0.687 ms 可能来自「不清 30 MiB 的副作用」，也可能只是把 1.95× 从「小 range 固定开销」外推到「字节传输」这一步不成立。**不得把残差直接断言为副作用。**<br>**⇒ 结论：K8 的想法没被否证，是实现方式错了。** 唯一能兑现 K8 的实现是**一次 broadcast**。原以为需要改 simpler 的 ctrl 协议（`_encode_memset_payload` / `_handle_ctrl_memset` 支持每 device 多段），但**有更简单的等价做法**：carve 严格顺序无 padding ⇒ **让模型把 7 个 control buffer 声明在最前面**，它们就构成唯一连续前缀 `[0, 47616)`，一次 `memset_all` 即可，**完全不动 simpler**。已实现并离线验证：模型侧 `claude_apply_k8_reorder.py` → `src_k8reorder/…/decode_fwd.py` sha `eb1f89bf…`（16 个 alloc 数量不变、7 个 control 移到最前、断言移动后仍各出现一次且连续成前缀）；runtime 侧 `claude_apply_k8_prefix.py` → `distributed_runner_prefix.py` sha `de130123…`（v2 精确 allowlist + 「control 出现在任何 data 之后就 raise」+ `control_bytes==47616` pin + `offset==spec[1]==actual`）。**fail-closed 配对已证**：`claude_check_prefix_layout.py` 对新序输出 `1 range [[0,47616]]`，对**旧序直接 raise** `control 'dense_attn_signal_stack_buf' after 1 data buffer(s)` ⇒ 这个 runtime 不可能被错配到未重排的模型源码上而静默清错字节。<br>**正在跑 4 臂判定**：`A1_parent / B_reorder_fullclear（重排+全量清零，隔离「重排本身」）/ C_reorder_prefix（重排+前缀清零，候选）/ A2_parent`。预测区间很宽（若无「清零副作用」，reset wall 从 `2245 → ~472 µs`，按 1.95× 放大可达 `−3.4 ms`；若副作用主导则接近 0 或转正），所以这一臂的信息量最大。 |
| 2026-08-11 | **null control：position 混淆已排除，K9 与 K8-selective 的 ITL 结论成立** | 三臂完全同源同 runtime，中间臂 `−0.1805 ms`（略快，方向相反）⇒ 不存在「第 2 个 arm 偏慢」 | campaign `0162:/mnt/persist/chensiyu/workspace/k8-selective-20260811/null-20260811-131548`（dev0-7，三臂 rc=0）。`B_null` 用的就是 `distributed_runner_baseline.py` + `src_parent`，与两个 A 臂**逐字节相同**，所以任何差异只能是位置与噪声。<br>结果：`A1 33.889 / A2 33.922 / B_null 33.725` ⇒ floor `0.0165 ms`、**`B_null delta = −0.1805 ms`**（`−0.53%`）；`reset_body` `2260.9 / 2256.5 / 2247.8 µs`（delta `−10.9 µs`）；三臂 sha 全 `567b206b…`。<br>**为什么要做这一步**：K9 `B +1.72 ms` 与 K8-selective `B +1.89 ms` 都出现在中间臂，两个改动却完全无关（一个改 kernel Wave3、一个改 host memset 范围），量级巧合到不排除 harness 层混淆就不能下结论。**判定：混淆不存在**，且中间臂偏置是 **`−0.18 ms`（更快）** ⇒ 两个结论不仅成立，按偏置校正后**真实效应还要大 0.18 ms**（K9 ≈ `+1.90`、K8-selective ≈ `+2.07 ms`）。此后 A/B/A 可沿用固定三臂顺序，不需要随机化。<br>**副产物**：`floor` 在这台机上可小到 `0.0165 ms`，说明 100-iter p50 的重复性很好；`B_null` 的 `−0.18 ms` 是 10.9× floor，即这个小的「中间臂略快」本身是可测的系统效应，不是噪声 —— 记录下来作为以后所有中间臂读数的已知偏置。 |
| 2026-08-11 | **★ K8 selective 实测：byte 省了但被 broadcast 固定开销吃掉；发现 `memset_all` 每次 broadcast ≈342 µs** | 只清 6 段共 47,616 B（比全量少 673 倍字节），`memset_all` 却只从 `2214.5 → 2055.2 µs`；ITL 反而 `+1.89 ms`（**待 null control 判定是否 position 混淆**） | campaign `0162:/mnt/persist/chensiyu/workspace/k8-selective-20260811/aba-20260811-125724`（dev0-7，三臂 rc=0，**模型源码三臂完全相同，只有 runtime 文件不同** = 最干净的 matched-source）。候选 `runtime/distributed_runner_selective.py` sha `fcb71f83…`。<br>**设备确认选择性路径真的生效**（不靠离线干跑）：B 臂 trace 里 `k8_control_bytes=47616`、`k8_control_range_count=6`，与离线预测的 6 段逐字节一致。<br>**★ 主要发现 —— 成本不是字节主导，而是 broadcast 次数主导**：<br>`memset_all_us` p50：`A1 2214.5` / `A2 2226.0` / **`B_selective 2055.2`**；`reset_body` p50 `2236.3 / 2247.4 / 2086.4` ⇒ **reset 只省 `155.4 µs`**。解模型：设每次 `broadcast_control_all` 固定开销 `f`、32 MiB 数据成本 `c`，则 baseline `= f + c ≈ 2220`、selective `= 6f + ε ≈ 2055` ⇒ **`f ≈ 342 µs`、`c ≈ 1878 µs`**。该模型与 codex 的双 memset 臂**独立对上**（多做一次全量 memset 实测 `direct_second_memset = 1943 µs ≈ c`）。⇒ **字节省下的 `≈1878 µs` 几乎被 5 次额外 broadcast（`5×342 ≈ 1710 µs`）吃光**，净剩 155 µs。<br>**⇒ 这正是本实验设计要探测的失败模式，结论是「实现方式错了，不是想法错了」。** 正确实现 = **把 6 段合并进一次 broadcast**：扩 `_encode_memset_payload` / `_handle_ctrl_memset` 支持「每 device 多段」，则 `= 1f + ε ≈ 342 µs`，比 baseline 省 **`≈1878 µs/step`**，与 K8 的投影一致。这两个文件是**纯 Python 且可挂载**（`simpler/worker.py`），仍然不需要等上游。已进 #22 诉求第 2 条的配套项，并从「可选」升级为「必需」。<br>**精度门 PASS**：三臂 `hidden_sha256` 相同且 = 生产 baseline `567b206b…`、`token=14371`、`finite=true` ⇒ **不清 30 MiB data buffer 没有改变任何输出比特**，实证了 codex 的窗口审计（含 `moe_recv_meta_stack` 归 data 的论证）。**边界**：单次 100-iter byte-exact 不等于所有调度下安全，发布前需多 seed / 多 batch 重复。<br>**🟡 ITL 数字待判定**：`A1 33.776 / A2 33.933`（floor `0.0785`）/ `B 35.740` ⇒ `+1.8855 ms`（24× floor）。但 reset 明明**快了** 155 µs，ITL 却慢 1886 µs，多出的 ~2.04 ms 无解释；且这与 K9 中间臂的 `+1.72 ms` 量级巧合。**已启动 null control**（`A1 / B_null(与 A 完全同源同 runtime) / A2`）判定是否 position 混淆。若确认非混淆，则待查机制候选：不清零后 data buffer 含上一步陈旧值（byte-exact 说明不被数值消费，但 padded tile 上的算术输入从 0 变成任意 bit pattern 可能走慢路径），或全量 memset 原本起到把 window 留在 L2 的「预热」作用、去掉后下游读变冷。<br>**踩坑（记录给后续）**：v1 第一次上设备 `rc=1` `NameError: worker_ranges` —— 删了绑定但后面 trace dict 还在读它，`compile()` 看不出来。已加 AST 名字解析预检 `claude_check_undefined_names.py`（带正对照自测：对已知坏版本必须 flag、对 baseline 与修好版必须 pass），接进 driver 在**取锁前**跑。driver 的 `set -e` 正确中止并释放了锁。<br>**codex 独立复核（报告 sha `305c4a2a…`）**：分类语义正确、无遗漏 counter；但指出 v1 两处不够 fail-closed（substring 规则可能静默归类未来新 buffer；`offset > actual` 会接受未分类的 padded tail）。已生成硬化版 v2 `fb8e0a22…`：去 `__ssa_vN` 后缀后**两份精确 allowlist**、未知 base name 直接 raise、精确不变量 `offset == spec[1] == actual`、外加「7 个 control 全在 / `control_bytes==47616` pin / ranges 不重叠升序且长度和一致」四条断言。v2 离线验证发射**与 v1 逐字节相同的 6 段**，并自测注入未知 buffer 会被 raise。**落地的 artifact 必须是被测过的那一个**，故 v2 需一次确认性 A/B/A。 |
| 2026-08-11 | **★ K9 整网 NO-GO（byte-exact 但 ITL +1.72 ms/step）；`critical_tail` 对「删同步点」结构性失明** | 删 Wave3 在 bench 上 `−5.92 µs/call`，在生产整网上 **`+35.76 µs/call`（符号相反）**；精度门 PASS 反而证明源码论证是对的 | campaign `0162:/mnt/persist/chensiyu/workspace/k9-precision-20260811/aba-20260811-122545`（dev0-7，生产程序 ctx=65536 / bs=1 / warmup=10 / iters=100，三臂 rc=0，三臂 `other-half-lock-at-start` 均 `holder=none`）。<br>结果：`A1_parent p50 33.845`（min 33.310）/ `A2_parent 33.798`（min 33.304）/ **`B_k9 35.538`（min 34.986）** ⇒ floor（half-range）**0.0235 ms**、baseline `33.8215`、**delta `+1.7165 ms/step` = `+5.08%` = 73× floor**。折成 per-AR-call（48 次/step）= **`+35.76 µs/call`**，bench 给 `−5.92 µs/call` ⇒ **连符号都相反**，差约 41.7 µs/call。<br>**精度门 PASS 且强于要求**：三臂 `hidden_sha256` 相同且**等于生产 baseline 权威值** `567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e`（该值在 codex dev8-15 与我 dev0-7 两个半区各自独立复现）、`token=14371`、`finite=true`。⇒ **「Wave3 对窗口 lifetime 不是必需」的源码论证被实证是对的**；错的是「因此可以删」。<br>**混淆项已排除**：① 三臂 other half 全空闲；② 三臂用同一 runtime，`reset_body p50` 2232.1 vs 2239.7 µs（delta +7.6 µs = 噪声）⇒ ITL 差**不来自** reset 路径；③ 全树 `diff -rq` 只有 `decode_fwd.py` 不同（20 行）；④ **B 的 min 34.986 高于两个 A 臂的 p50**，整个分布平移，非尾部假象。<br>**★ 方法论结论（比这个候选本身更重要）**：`critical_tail(epoch) = max(rank_exit) − max(rank_entry)` 把参考点重置到每次 AR 的**最后到达者**，因此「让部分 rank 在**下一次** AR 入口等更久」的代价被它按定义减掉；而删同步点恰好是把同步开销转成到达 skew —— 正是该指标被设计成不敏感的量。⇒ **硬规则：任何删除/合并波次的候选都不能用 bench 的 `critical_tail` 评估，必须上整网 ITL。** 这也解释了 rank-skew 实验为何没预警：它同样只看 `critical_tail`，注入 191 µs skew 后读数几乎不动 —— 那不是「skew 无代价」，是**指标看不见 skew 的代价**。这是 bench 第二次误导（K5-C 只错量级，这次错到符号）。<br>**账目撤回**：**K9 的 `−4.92 µs/call` 从累计收益中删除**，已确认 AR 侧收益回落到只有 K6b `7.02 µs/call ≈ 0.337 ms/step`；K8（~1.95 ms/step）现在是剩余 AR 侧总量的 **5.8 倍**。**连带降级**：「合并 Wave1+Wave2」（bench 约 5.6 µs/call）属同一类，**未经整网 ITL 不得计入**；K6b 只缩字节、保留全部波次，不属此类，但鉴于 bench 已两次误导，落地前也应补整网确认。<br>**未解机制（不影响判定）**：`+35.76 µs/call` 量级接近整个 AR 成本（43 µs/call），偏大。两个候选未区分：① skew 跨 48 次串行 AR 累积，代价落在下游等待（注意 MoE EP dispatch/combine 的 data window 是**跨层共享的一套**，只靠 count-bounded read 保护，比 per-layer 的 AR 窗口更易受 skew 影响）；② 删 notify/wait 循环扰动 kernel 调度/UB/指令布局，与同步语义无关。两种机制下都是 NO-GO；下一步可用 DFX swimlane 看多出的时间落在哪。<br>provenance：`bin/run_k9_arm.sh` `d0f64578…`（= codex `run_k8_arm.sh` 只改 DEVICES/容器名 3 处 hunk）、`bin/run_k9_precision_aba.sh` `a70baf9b…`、`bin/claude_apply_k9_argv.py` `5b3c86ac…`、`src_parent/…/decode_fwd.py` `28080c53…`、`src_k9/…/decode_fwd.py` `47d12a08…`、`runtime/distributed_runner.py` `278b0c4b…`、`aba-20260811-122545/K9_PRECISION_RESULT.json`。 |
| 2026-08-11 | **K8 生产实测 GO（约 1.95 ms/step ≈ 5.75% ITL）—— 迄今最大单项，方法比我的 bench 外推更强** | codex 在**生产程序 + 生产形状**（ctx=65536 / bs=1 / 100 iters）上直接量到「一次 30.578 MiB 清零」的成本，不再依赖 bench 外推 | campaign `0162:/mnt/persist/chensiyu/workspace/perf-2026q3/k8-production-reset-20260811-115222`，`schema=k8.production-runtime-reset-ballast.v2`，**codex verdict=GO**，三臂 rc=0（dev8-15，本半区独占，每臂另存 `dev0-7-concurrency.txt`）。<br>**★ 方法替换（codex 的方法严格优于我的）**：我原来的 model-side `alloc_window_buffer(+32 MiB)` ballast 在生产程序上**编译期就被 `MaterializeCommDomainScopes` 拒**（allocated window buffer 无 `pld.tensor.window` materialisation = 非法 IR），该臂**从未上设备**（`B-ballast32.runner.rc=1`）。codex 换成**运行时加性对照**：模型源码与 CommDomain 尺寸**完全不变**，只让 `_reset_persistent_domains` 把同一个 per-rank `32,063,232 B` 的阻塞 `Worker.memset_all` **再做一遍**（幂等、不进任何 model kernel），trace 分别记录两次调用。⇒ 直接量「多清一遍生产尺寸」的真实成本，**不需要 ballast 存活、也不需要跨程序外推**。<br>**结果（p50，单位 µs 除注明）**：ITL `A1 34.083 / A2 33.971 / B 36.066 ms` ⇒ **floor（half-range）= 0.056 ms**、`itl_b_minus_parent = 2039`（**36.4× floor**）；in-runtime `reset_b_minus_parent = 1958`；直接计时新增那次调用 `direct_second_memset = 1943`。三个独立读数彼此收敛（`itl_minus_reset_delta = 81 µs`，即 ITL 差比 reset 差仅多 81 µs）。**正对照**：`primary_memset_b_drift = −6.995 µs` ⇒ 加了第二次 memset **没有改变第一次的成本**，故增量归因干净。**分解**：A1 `primary_memset p50 = 2217.9`、`reset_body p50 = 2238.5`、`reset_minus_primary p50 = 20.9` ⇒ **reset 体几乎全是 memset，外围开销仅 ~21 µs**。<br>**投影**（按 `removable_bytes=32,015,616 / full=32,063,232 = 99.8515%`，`retained_control_bytes=47,616`）：`projected_k8_savings` p50 = **1.941 ms**（按 direct call）/ **1.955 ms**（按 reset delta），两条路线差 0.7%。对 baseline ITL p50 `34.027 ms` ⇒ **≈ 5.75%**（ITL 级增量 2.039 ms = 6.0% 为上界）。<br>**与我 bench 外推对账（如实记录偏差）**：bench ballast 给 `52.4 µs/MiB → 1.602 ms`，生产给 `≈64 µs/MiB → 1.94–1.96 ms`，**bench 低了约 18%**。⇒ 以生产数为权威，bench 数降级为「同量级的独立交叉验证」，不再作为 headline。<br>**精度中性**：三臂 `hidden_sha256` 全为 `567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e`、`token=14371`、`finite=true`（memset 幂等，符合预期；这也给出了**生产 baseline 的权威 hidden sha256**，可作后续候选的 byte-exact 参照）。<br>**边界（不得夸大）**：这证明的是「reset 路径上一次生产尺寸清零值 ~1.95 ms」，**不是**「选择性清零的最终实现已验证」。落地仍是**上游 pypto 改动**且有实现约束：control 与 data buffer 在 window 里**交错**（`buffer_count=16`，`dense_attn_tmp_stack` 393,216 / `dense_attn_signal_stack` 1,536 / `moe_attn_tmp_stack` 5,505,024 / `moe_attn_signal_stack` 21,504 / `moe_recv_meta_stack` 1,280 …），**不能只把一次连续 memset 缩短**，需 buffer-level reset / 多 range 批量 memset / 独立 control domain。<br>**优先级结论**：K8 `1.95 ms/step` vs AR 侧全部已确认项合计 `11.94 µs/call ≈ 0.57 ms/step`（K6b 7.02 + K9 含 fence 4.92，按每 step 48 次 AR 折算）⇒ **K8 约为 AR 全部收益的 3.4 倍，且不受 14 µs/call 门约束**（它不是 per-AR-call 项）。 |
| 2026-08-11 | **K9 rank-skew / no-hang 门通过；skew 作为 #23「生产为何不触发」的候选解释被否证** | 受控 rank skew 下删 Wave3 仍 **64/64 byte-exact、无 hang**；同时未加 fence 的 baseline 在同一 skew 下**也**保持 exact ⇒ skew 不是生产安全的解释 | campaign `0162:/mnt/persist/chensiyu/workspace/p2-k5-rhrd-20260810/aba-20260811-120907`（dev0-7，7 臂，EPOCHS=64/WARMUP=8）。方法：注入器 `claude_k6_skew_inject.py` 在 `int64_t k6_entry = get_sys_cnt();` **之前**按 rank 插自旋（`if (v5 == 3) { spin TICKS }`），故 `critical_tail = max(exit) − max(entry)` 仍从**最后到达者**起算、不把自旋计入 AR 成本；**skew 同时施加到 baseline 与 now3 两臂**，这是本实验的设计要点 —— 只有 now3 崩才是 Wave3 有用的证据。<br>结果（`critical_tail_us` p50 / exact / epochs / failures）：`A1_parent 43.18` / `A2_parent 42.89` / `P_parent_sk2000 43.37` / `P_parent_sk10000 42.58` / `N_now3_sk2000 37.30` / `N_now3_sk10000 37.25` / `L_now3_ffk6w2_sk2000 38.30` —— **七臂全部 `exact=True`、`epochs=64`、`failures=0`、无 hang**。<br>**★ 有效性检验（否则这个实验是空的）**：每 epoch `wall_ms` 只涨到标称自旋的约 52%（标称 19.06 ns/tick，由 100000-tick 校准臂 `+1.906 ms` 反推 ⇒ sys_cnt ≈ 52.5 MHz），所以必须独立证明到达顺序真被改变。查 `latest_entry_rank_counts`（哪个 rank 是本 epoch 最后进 AR 的）：baseline `A1 1/64 (2%)`、`A2 0/64 (0%)`；**`P_parent_sk10000` rank3 = 50/64 (78%)**、**`N_now3_sk10000` rank3 = 38/64 (59%)** ⇒ **只有 sk10000 两臂构成有效 skew**（标称约 191 µs ≈ 43 µs AR 的 4.4×）；`*_sk2000` 三臂只有 `3/64 (5%)`、与 baseline 的 `0–2%` 无法区分 ⇒ **sk2000 臂是 inconclusive，不得当作支持证据**。<br>**结论一（K9）**：删 Wave3 在**有效** skew 下仍 byte-exact 且不 hang ⇒ K9 通过 codex 要求的 rank-skew/no-hang 门。剩余唯一门 = 整网精度 A/B/A。<br>**结论二（#23，负结果）**：`P_parent_sk10000` 是**未加 fence 的 baseline**，在 78% epoch 里到达顺序被改变仍保持 exact ⇒ **skew 不是「生产为何不触发 notify fence 缺陷」的解释**，这是被否证的第五条候选（前四条：纯 MTE3 流量、MTE3 级屏障含 +dsb、store-loop 自带 MTE3 屏障、Wave3 slack）。#23 的口径不变：**生产 Wave2 没有可证明的安全机制，是否正在损坏未知**。<br>provenance：`claude_k6_skew_inject.py`、`claude_patch_runner_skew.py`（`_sk<N>` → `--ticks N`，在 wave3 删除与 fence 注入**之后**运行）均在上述 campaign 同目录。 |
| 2026-08-11 | **K9 device 实测通过（−5.92 µs/call，byte-exact）** | 删 Wave3 在生产形状 parent 上实测省 **5.920 µs/call**；含 Wave2 fence 的实际落地组合省 **4.920 µs/call**；两臂都 64/64 byte-exact。**另发现 fence 代价不是常数** | campaign `aba-20260811-114721`（dev0-7，EPOCHS=64/WARMUP=8，**四臂的 `other-half-lock-at-start.txt` 全为 `holder=none`**，绝对计时未被并发污染）。实现方式：后处理注入器 `claude_k6_wave3_remove.py` 删掉生成码里 Wave3 的 notify 循环 + wait 循环（干跑：`removed_lines=380-419`、40 行、**diff 为纯删除 0 增 40 删**、剩 `2 notify / 2 wait`），**不动 codex 的生成器**；注入器对 notify/wait 计数、两循环相邻性、「纯删除」都有硬断言，生成器若改变波次结构会响亮失败而不是删错循环。<br>结果（`critical_tail_us` p50）：`A1_parent 43.16` / `A2_parent 43.12` ⇒ **floor（half-range）= 0.020 µs**；`B_now3 37.22` ⇒ **−5.920 µs/call**（floor 的 296 倍）；`C_now3_ffk6w2 38.22` ⇒ **−4.920 µs/call**。四臂全部 `output_exact=True`、`epochs_completed=64`、`failures=0`。<br>**新事实：notify fence 的代价依赖临界路径构成，不是常数。** 同一个 Wave2 site：3 波 parent 上 `+0.405 µs`（`-100328`），删掉 Wave3 后 `+1.000 µs`（= 5.920 − 4.920，远超 0.020 µs floor）。⇒ 引用 fence 代价时**必须带上它是在哪个 kernel 结构上量的**，不能把 `0.405` 当通用常数外推。<br>**累计账（截至此行）**：K6b 已确认 `7.02 µs/call`（不需上游）+ K9 含 fence `4.92 µs/call` ≈ **11.94 µs/call**，**仍未过 14 µs 门**；而 K8 约 `1.60 ms/step` 比这些大两个数量级 ⇒ **优先级应转向 K8**。<br>**K9 尚未过的门**：codex 要求的 **rank-skew / no-hang**（受控 skew 注入器 `claude_k6_skew_inject.py` 已写好并接进运行器 `_sk<N>` 路由，但 dev8-15 正跑 codex 的 K8 生产 campaign，运行器按方法论拒绝并发跑绝对计时，待卡空闲后执行）。该 skew 实验有第二重价值：**skew 是「生产为什么没触发 notify fence 缺陷」最可能的答案**，若未加 fence 的 baseline 在 skew 下失去 exact，则 #23 里「生产是否暴露」的未知将翻为「是」。 |
| 2026-08-11 | **K8 新线索（最大单项）/ K9 新候选 / 两个缺陷假设被否证** | 从 AR 转向 host 侧每步开销：**每个 decode step 把整块 30.58 MiB retained window 清零，其中只有 46.5 KiB 必需**，实测速率外推 **约 1.60 ms/step ≈ 2.4% ITL** | **① K8：per-step window 清零（device 实测 + 线性性对照）**。机制（读源码，非推测）：`pypto/python/pypto/runtime/distributed_runner.py:1350` 在**每个** persistent request（= 每个 decode step）前调 `_reset_persistent_domains`（docstring「Restore retained windows to the zero-filled fresh-allocation state」）；a2a3 上走 `worker.memset_all` → 各 chip 内 `aclrtMemset`（不过 PCIe），但 `broadcast_control_all` **阻塞**到 8 个 child 全完成；多个 `alloc_window_buffer` 被顺序 carve 进**同一个** domain window，所以是一次覆盖全量的 memset。生产每 rank window 实算 = **30.58 MiB**（`moe_recv_x` 18.00 + `moe_attn_tmp` 5.25 + `moe_sh_tmp` 5.25 + `moe_routed_y` 1.00 + dense tmp 2×0.38 + aux/route 2×0.14 + …），signal/arrived 类**仅 46.5 KiB = 0.15%**。**实测**（`aba-20260811-112710`，dev0-7，EPOCHS=32/WARMUP=8）：给 bench 的 retained window 加 ballast MiB（内核 `pld.window` 视图不变、永不读写这些字节），比较每 epoch `wall_ms` p50 —— `A1_bl0 2.735` / `A2_bl0 2.745`（**floor = half-range 0.005 ms**）、`B_bl32 4.416`（**+1.676 ms / 32 MiB = 52.39 µs/MiB**）、`D_bl128 9.493`（**+6.753 ms / 128 MiB = 52.76 µs/MiB**），四臂全部 `exact=True`。**线性性 4.03（理想 4.00）** ⇒ 无固定开销混淆、ballast 未被编译器丢弃（这正是正对照的设计目的：若 128 MiB 与 0 读数相同，则测量无意义，必须响亮失败而不是报「免费」）。外推生产 30.58 MiB = **1.602 ms/step**，只清 46.5 KiB = `0.002 ms` ⇒ **可省约 1.60 ms/step**，对照 ITL p50 `65.942/66.455 ms` ≈ **2.4%**。**边界（不得夸大）**：这是 bench 上按已验证线性速率**外推**，不是生产 ITL 实测；发布必须走 canonical 整网 A/B/A + 精度门。**前置耦合 —— 此前的说法已被 codex 源码否证，不要再引用**：我原写「全量归零正是让 C2 保持 latent 的原因，缩小 memset 前必须先修 C2」。**错**，三条反证：① AR 的输入是 `partial_attn_proj`，不是 `resid1_out`；② `resid1_out` 全 16 行由 `current_hidden` 初始化，holder 另行把 inactive rows 清零；③ AR 的全 16 行 TPUT 会覆盖已清零的 tmp window，所以 window memset **无法**掩盖本地未初始化行。⇒ **C2 可独立修，但不是 K8 的前置条件**。codex 逐窗口审计结论：**只有 signal/arrived counters 必须跨 request 清零**；TP tmp、`recv_meta/x/aux/route`、`routed_y` 都有完整的 producer-before-consumer 或 count-bounded read。另有一处收紧：46.5 KiB 是**分配量**，实际被访问的 counter 字节只有 **2,976 B = 2.90625 KiB**（512 B stride 绝大部分是 padding）。**落地约束（codex 提出）**：不能只把一次连续 memset 缩短 —— control 与 data buffer 当前在 window 里**交错**，需要 buffer-level reset / 多 range 批量 memset / 或独立 control domain。实现属**上游 pypto 改动**，已进 #22 诉求清单。**② K9：Wave3 候选删除（约 5.6 µs/call）**，源码级论证：注释称它守「window read lifetime」，但那些 final read 是**本地** `pl.load(tmp_window,…)`（无 `peer=`）；窗口**严格 per-layer**（`moe_attn_tmp_stack[layer_idx*BATCH]` / `moe_attn_signal_stack[layer_idx*COMM_SIGNAL_STRIDE_I32]`，attn 与 shared 各有独立 stack）⇒ 一次 program run 内每个 (layer,purpose) 槽只用一次、**run 内不存在窗口复用**；跨 run 由 host 侧结构性屏障覆盖（`orch._drain()` → 阻塞式 reset → 才 dispatch 下一 request）。codex 独立复核**未找到**「peer 尚未读完、同槽已被改写」的路径，并覆盖了 MTP 段（三个 AR 用独立且当前 non-persistent 的 fresh domain，不与 Main 别名）、L43/L44 显式特化层、`num_tokens` 与 bs=16 动态路径（不改变窗口偏移）。**但 codex 纠正了我的一条理由**：我原写「Wave2 本身已证明所有 peer 读完了我的 Wave1 窗口」——**Wave2 不能证明 final-read lifetime**；真正使 Wave3 冗余的是**窗口 one-shot**。**同时修正一处此前笼统表述**：删 Wave3 **不**把 payload store 与它自己的 credit 拉近（Wave2 notify 位置不变），故**单纯删 Wave3 不依赖 notify PIPE_ALL 修复**；**强制依赖 fence 的是「合并 Wave1+Wave2」**，两者必须分开验证。仍需的门：byte-exact、zero-gap 多 epoch、rank-skew / no-hang。codex 复核报告 `0162:/mnt/persist/chensiyu/workspace/perf-2026q3/codex-k8-k9-review-20260811/CODEX-K8-K9-INDEPENDENT-REVIEW.md` sha256 `2b317434719704c1fec7df7fa0b0abd9b1cd47d72bf54c1d6686323d77507200`（本轮未占卡）。**③ 两个缺陷假设被源码否证（负结果，避免重复排查）**：(a)「signal 槽复用 + `AtomicAdd` 累加 + 恒定阈值 `1/2/3` ⇒ 第二次 AR 的 wait 空转」—— 否证，槽是 per-layer per-purpose 的，循环体内两处同 `moe_sig_off` 是 full/swa **互斥分支**；(b)「跨 decode step 计数器累加 ⇒ 第 2 步起三波 wait 全部空转」—— 否证，每步 `_reset_persistent_domains` 归零，这也解释了为什么生产复刻 k6（不轮换槽、阈值恒定）能连跑 64 epoch `exact=True`。**④ fence 修复的精度确认**（此前只量了代价）：`aba-20260811-100050`（全 3 site）与 `-100328`（只 Wave2）三臂全部 `output_exact=True`、`epochs_completed=64`、`failures=0` ⇒ **fence 不引入精度回归**。provenance（0162 绝对路径 + sha256，均在 `/mnt/persist/chensiyu/workspace/p2-k5-rhrd-20260810/`）：`claude_k5_bench.py` `e3e09a7877c71d9c9281f023fe4a33b23f9928b8d0795b4276f5997e0ab580f4`、`claude_run_k5_aba.sh` `26b42b744fd3dee0f98e1e9a9d4cfb18ce7e0b6bcc893dca17fb050ef844daa2`、`claude_patch_bench_ballast2.py` `6bc4a613160401f5aebfa0ac6d220870732a424ef437c62528d1da383982a34d`、`claude_patch_runner_ballast.py` `9629c12e8d8269928303d202ad8a038ce592cee7e990829d2077bdf4c23ada34` |
| 2026-08-11 | **K5-C 否决 / 新增 correctness 缺陷** | C32′ device 实测只省 `0.54 µs/call`（远低于 14 µs 门）⇒ **统一定律对「深串行依赖链」外推失效**；同时定位到 **notify 的 cache-invalidate 排在 payload drain 之前**，生产 Wave2/Wave3 暴露 | **① notify fence 缺陷（device 已证）**：pypto `MakeNotifyCodegenPTO` 发的前导是 `dcci(ENTIRE_DATA_CACHE)`（**invalidate-only，无 writeback**）→ `barrier(MTE3)` → `dsb(DDR)` → `barrier(MTE2)` → `TNOTIFY`，**invalidate 排在任何 drain 之前**，credit 因此可能跑到 payload 前面（裸 `remote_store` 紧接自己的 `notify` 时 epoch 0 就坏）。在 `dcci` 前补屏障即修复；**最小集是一条 `pipe_barrier(PIPE_ALL)`，见 ⑤**。证据：ring 探针（单次交换、row offset 0、列槽），`aba-20260811-013946` ring_up/ring_down **无 fix 都 `exact=False`、有 fix 都 `exact=True`** ⇒ 方向被排除；payload 扫描（无 fix）`16/32/64/128 KiB` **全部 `exact=False`**，`16 KiB + fix` `64/64` epoch exact（`aba-20260811-015235`、`-014938`）。**② 生产暴露面**（读 `A1_parent` 生成码，非推测）：`MakePutCodegenPTO` 给 tput 夹的 `pipe_barrier(PIPE_ALL)`（注释写成「WORKAROUND for PTOAS#872」）**实际承载正确性**，而 `MakeRemoteStoreCodegenPTO` 什么屏障都不发 —— 这个不对称就是缺陷来源；Wave2/Wave3 的 notify 前导与被证伪的形状**逐字节相同**，且 Wave2 前面正是 `remote_store`。**但生产为什么安全仍无答案，见 ⑧**。**硬约束**：任何把「payload store 与它自己的 credit」拉近的改动（删 Wave3 ~5.6 µs/call、合并 Wave1+Wave2 ~5.6 µs/call、按 peer 融合 store+notify、单 peer 交换）都会进入探针的近确定性失败区间，**必须先落 fence**。**③ K5-C 否决**：column-slot C32′ + fence fix **64 epoch byte-exact**，但 `critical_tail` p50 `42.63` vs parent `43.24/43.10`（half-range `0.070`）⇒ **`−0.54 µs/call`**，远低于门（`aba-20260811-014431`；无 fix 对照组 `exact=False`）。**定律外推失效已证**：交易 `35→12`、bytes/row `30720→21504`，`0.80 µs×交易 + 0.94 µs/row×行` 预测约 `−20 µs`，实测 `−0.54 µs` —— 定律**不含依赖深度**（C32′ depth 6 串行往返 vs parent depth 3 且 7-peer 扇出并行）。**后续方案必须减交易数而不增深度**。作废数据点：`-014938` 的 `c256` 无效（copy 循环仍按 `CHUNK=512` 步进而 payload 只 256 列，尾部读到从未写过的窗口空间 —— harness bug，已改 `step=min(CHUNK,cols)` 后重跑）。权威报告 `0162:/mnt/persist/chensiyu/workspace/p2-k5-rhrd-20260810/CLAUDE-NOTIFY-FENCE-DEFECT.md` sha256 `a34817832550b9c68c907a58774403802d79c1926e8aa085b658ff0aafc9f21b`。**④ 安慰剂对照（已排除「两条指令只是扰动时序」）**：`--fence-late` 把**完全相同的两条指令**放到 `TNOTIFY` **之后**（指令数/开销一致，但不再夹在 store 与 invalidate 之间），`aba-20260811-015951`（32 KiB）baseline `exact=False`、placebo `exact=False`、真 fix `exact=True`（`64/64`），两个变体 kernel diff 都恰好同样两行、只差插入位置 5 行 ⇒ **原因是顺序不是时序**（这条不受 ⑦ 的收回影响）。**⑤ 消融（codex 指出两条指令不是最小集后补做）**：同一插入点、每臂 diff 恰好一行、32 KiB/ring_up/warmup=0/epochs=64 —— `pipe_barrier(PIPE_MTE3)` 单独 **False**（`-095638`，⇒ **纯 reorder 上游现成指令不够**）、`dsb(DSB_DDR)` 单独 **False**（`-095441`）、纯 MTE3 流量（`--drain-store`）**False**（`-100538`）、`pipe_barrier(PIPE_ALL)` 单独 **True** `64/64`（`-095441`）。⇒ **最小修复 = 一条 `pipe_barrier(PIPE_ALL)` 插在 `cacheinvalid` 之前**，且这正是 `MakePutCodegenPTO` 给 tput 已在发的那条屏障，故上游诉求可表述为「把 put 路径已有的 `PIPE_ALL` 对齐到 notify 路径」。**⑥ 修复代价已量化**：后处理注入生成后的 parent kernel（不动 codex 生成器）—— 全部 3 个 notify site `+1.250 µs/call`（half-range 0.150，`-100050`）、只 Wave2 一个 site `+0.405 µs/call`（half-range **0.005**，`-100328`），约 `0.417 µs/site` / `0.060 µs/PIPE_ALL`，**比 K2a 的 pipe-specific barrier（`0.0033 µs`）贵约 18 倍，不能外推成免费**。扣掉 Wave2 fence 后，删 Wave3 + 合并 Wave1+Wave2 的净收益约 `10.8 µs/call`，**单独仍不过 14 µs 门**。**⑦ 两处自我修正**：(a) 收回「机制确定不是晚到」—— `dsb` 不等 MTE3 store 完成、`PIPE_ALL` 才等，故「invalidate 破坏」与「credit 超车」两解释都兼容（修复相同，不影响结论）；(b) 「纯 MTE3 流量可替代屏障」已被 `--drain-store` 否证。**⑧ 消融矩阵闭合（补 codex 指出的 gap）**：此前只单测过 `PIPE_MTE3` 与 `dsb`，未测**组合**；补测 `pipe_barrier(PIPE_MTE3) + dsb(DSB_DDR)`（同一插入点、diff 恰好两行）→ `exact=False`、`bad_ranks=[0,4]`、epoch 0 就坏。⇒ **`PIPE_ALL` 是真必需的，`0.405 µs/call` 压不下去**。**⑨ 与 codex 独立复核对账**（codex 报告 `0162:同目录/CODEX-NOTIFY-FENCE-INDEPENDENT-REVIEW-20260811.md` sha256 `37fae3aba51a555189c9da05633d88d0ab7810e28d72df9681f9fcfa11d472ac`）：一致项 = pre-CMO `PIPE_ALL` 为最小修复、`dsb` 冗余、上游诉求表述、我 ⑦(a) 的收回（codex 独立得到同一结论）。两处撤回 = codex 自撤「store-loop 的 MTE3 屏障保证前六个 store 安全」（最多是间距/背压，不是正确性屏障）；我撤回 Wave3 slack 假设（**结构性否证：Wave3 位于 consumer read 之后，不可能解释 Wave2 的安全**）。唯一分歧已按更保守方向收口：**生产 Wave2 没有可证明的安全机制，只是当前调度没触发；是否正在损坏未知** —— 我原先「近确定性失败 ⇒ 必有结构性保护」的反推默认失败率与结构无关，该前提未验证，故撤回。**两 agent 一致的工程建议**：不要继续争论生产是否暴露，直接以 `0.405 µs/call`（Wave2 单点）消掉这个不可证明的安全条件；它同时是删波次 / 合并波次类优化的前置条件 |
| 2026-08-11 | K6b / K5-C | TPUT 被 PTOAS 否证（第三条上游诉求）；K5-C 变体集修正 + claude 补 C32′ | **TPUT 已测且被否证**（此前记「未测」）：codex 补齐 `shape=` + 两个 offsets 后 pypto 正确 emit `pto.comm.tput(... <?x4096xbf16> ...)`，但 **PTOAS 在 `tp_all_reduce.pto:51:3` 硬拒** `'pto.comm.tput' op expects dst to have a positive static shape` → **阻塞在 PTOAS 而非 pypto**。K6b 最终账：已确认可缩 `50.0% = 7.02 µs/call = 0.337 ms`（不需上游）；TPUT `26.7%` 需 PTOAS 改、remote_load `23.3%` 需 pypto 改。另记一条落地约束：dynamic valid shape 贯穿整个 reduction 会触发 **loop-phi dominance bug**，正确结构是只在 publish + final-copy 处用。**三条上游诉求**：`remote_load` 缺 `valid_shape=`、`notify` 缺 fence 参数、`pto.comm.tput` 拒 dynamic dst。**K5-C**：codex 修正了 claude 原 `26.8 µs/call` 预测的矛盾（224 KiB 与「仅 FP32 re-parenthesization」不能同时成立；原数用的是 BF16 payload = C16，含 3 次中间舍入）。claude 再补 **C32′（FP32 RS + BF16 AG）**：RS 结束时 owned shard 已是完整 FP32 和，cast 一次即与 parent 同一舍入契约，故 AG 只需搬 BF16 终值 —— `21,504 B/row` = parent 的 70.0%，预测 `20.1 µs`、**省 `23.4 µs/call` = `1.12 ms`**，**严格优于 codex 的 C32（字节少 25%、数值契约相同）**。三变体都远超 14 µs 门 ⇒ **筛选依据是精度代价而非门**。定律外推薄弱点：C32′ 的 12 交易分 6 个**串行** round，而 `0.80 µs/transaction` 拟合自 3-wave 与轮内可连发的拓扑，**未覆盖串行 round 间的握手延迟** —— 这是唯一可能翻车处，必须实测 |
| 2026-08-10 | K6a / K6b / K5 | K6 scan 出数：**统一定律成立**，K6a 成为首个越过 14 µs 门的单项；K5 从末位提到 P0 | **统一定律** `AR_cost ≈ 0.80 µs × N_remote_transactions + 0.94 µs/row × active_rows`，三条独立测量互证：K1 notify `0.8010 µs/store`、K6 scan B payload `0.7616 µs/transfer`（差 5%）、scan A 截距 `28.442` vs 交易计数预测 `35 × 0.8010 = 28.035`（差 **1.4%**）。回代 bs=1 `28.9`（实测 `29.00`）、bs=16 `43.0`（实测 `43.26/43.62`）。**K6a**：scan A p50 `29.00/30.10/32.86/36.00/43.26`(N=1/2/4/8/16)，bs=1 delta `14.44 µs/call` = 80× half_range = `0.693 ms` → **越门**；落地不需改 kernel（`STORAGE_BATCH_CAPACITY` 已是 env），只需 `BATCH_TILE = min(BATCH,16)`；代价是放弃并发 batching。副产物：现有 bs=1 整网数字一直背着这 `0.693 ms`。bs=8 只 `7.44 µs/call` 而门需 `54.9` → 低 batch 专项。**K6b**：`remote_load` 的 shape 决定 `pl.Tile` 类型 → 必须编译期常量，14 次 payload 传输动不了，估计仅 `~3.7 µs`；`set_validshape`（接受 runtime `Scalar[INDEX]`）是否被 DMA 遵守是决定性未知 → 一个臂定生死。**K5-C**：12 交易 vs 35、7168 列 vs 15360（流量也更少，原记的「加重流量」只适用变体 B），外推预测 bs=16 省 `26.8 µs/call = 1.29 ms` 且 **bs=16 也有效**（K6a 做不到）→ 若 K6b 死则为 P2 唯一能独立越门的候选。**已知不足**：scan B 多轮臂 `latest_entry_is_last_exit_rate` 掉到 `0.531~0.625`；scan B t=16 与 scan A N=16 用同一臂 `base16_a1`，共享点是恒等式、未独立交叉验证 |
| 2026-08-10 | K2a / K2b / K6 | K2a 实测否决；K2b 拿到 128 KiB 三臂单臂差值；K6 新立项并排到 K2b 之前 | **K2a NO-GO**：三臂 `4.46/4.44/4.46 µs`，half_range `0.000`，delta `+0.020 µs`；反解每 barrier `0.0033 µs`，18 个可摘 = `0.060 µs/call` = 48 次 `0.0029 ms`，比地板低两个数量级。这**证实了 K1 的 `a≈0`**：`0.786 µs/notify` 几乎全是 remote credit store，barrier 多重性无价值 → 唯一杠杆是**减少 store 次数** → K5/RD 复活（变体 C 21→6 store 且不加流量）、K6 立项。**K2b**：`notify_us` p50 A`16.17`→B`13.78`→C`13.56`，A→B `2.39 µs/call` 落在预估 `2.70~3.13` 内，C 只再省 `0.22` → 收益几乎全来自 fence 提升。两条限定：本轮无 A1/B/A2 bracketing 故只算单臂差值；claude 的 `transmission_factor>1` 假设 as measured 不成立（`post_minus_hot = cold_minus_hot = 0.0` 三臂全 0），但 `cold≈hot≈92.28 µs` 说明探针 GM-bound、可能结构上测不出 locality，null 一半属仪器。K2b 需上游 pypto 补丁（`pld.system.notify` 无 fence 参数）。**K6**：`attention_full.py:244 batch_padded = BATCH` 是静态别名 → bs=1 仍搬满 16 行（224 KiB 单向），而同文件已有 device-validated 的 `active_tokens` runtime 界可复用；纯 model 侧 + 保序，故优先级高于 K2b/K4。反证据需先排除：两次 43 µs 测量在 active rows 相差 16× 下取得 |
| 2026-08-10 | K1 | TP all-reduce 二次优化立项：critical-tail 口径确立 + notify/drain 分账，claude 独立复核 REPRODUCED | headline `critical_tail = max(rank_exit) − max(rank_entry)`，p50 `43.18 µs`；control `18.38` / data-compute `24.82`。**最大可寻址项是三轮 notify control `16.51 µs`**，不是 publish completion。pooled p50 `171.95 µs` 降级为 host 顺序提交 artifact。claude 补 `cost(n)=a+b·n`、a≈0 / b≈0.80 µs/notify → 决定性实验改为 barrier-vs-store 两点对比（K2）。`PTOAS fix/issue711-tnotify-mte-drain` 已在生产 pin 内，是 data-before-signal 正确性约束。原件 `phase0-split-20260810-190834/`，复核 `claude-verify-p2-phase0-20260810/` |
| 2026-08-10 | J2 | gate fan-out 与 norm/quant 解耦已发布，整网再拿约 6% | `pypto-lib stepfun/develop@d13b2ca6`（FF over `a31977fb`），`decode_fwd.py` SHA256 `28080c53…`。bs=1/64k/nb512 p50 `36.494 -> 33.849 ms`（+7.25%）、bs=8/64k/nb4096 p50 `97.528 -> 91.722 ms`（+5.95%）；两档三臂 hidden byte-exact。5 层 swimlane（bs=1、已发布代码）在 `perf-2026q3/swimlane-p1a-candidate-20260810-130154`，rank2 makespan `2.210 ms`、static CPM 81.7%、stall 19.5% 全 data-wait，`tp_all_reduce` 占 15.9%（8 次 on-path）。同轮三个 NO-GO 与两条新硬约束（UB per-kernel-per-core、`pl.pipeline` 可行性）见 [`../../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](../../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md) |
| 2026-08-08 | J1 | 当前源码 tip 与最小 pending build spec 冻结 | `stepfun/develop@491267c4` 已包含 `7928a275`、active-route scheduling、SWA mask 修复和 route/precision release harness；当前无包含该提交的 immutable image，下一步按 pending spec 构建并执行 0162 标准回归 |
| 2026-08-07 | I1/J1 | SWA mask 精度修复进入远端，状态文档冻结并推迟镜像发布 | `stepfun/develop@63814d4a`：typed INT32 数值区间 mask 替代 `pl.cmp` predicate 转换；0162 source-overlay N=128=`127/128=99.21875%`、唯一 miss `step94 478→320`、TP spread=0。当前无包含该提交的 immutable image；后续选统一 release commit 再走标准发布流程 |
| 2026-08-07 | J1 | `c9af5790` pre-fix 六档 normal A/B、双 hidden与 matched-source A/B 完成，J1 保持 🟦 | 36/36 focused run 已 seal；`normal_seal_authority.json` SHA256=`875804dd…`，r1/r2/r3 性能报告 SHA256=`d238baa1…`。whole-net 1-step×2、2-step×2 共 8/8 sealed PASS。SWA mask 已变化，final image 仍须重跑六档/精度并补 formal DFX/swimlane |
| 2026-08-06 | J1 | 产品实现并入最新 `stepfun/develop`，启动 canonical formal regression | `7928a275` 已是 `c9af5790` 祖先；新镜像必须绑定 `pypto@8e92b468`、A2/A3 profile 和 prepared swimlane capability。准出为 BS `1/2/4/7/8/16` 各自 64K、L3/L4 hidden golden、端到端精度与 all-rank DFX；完成前保持 🟦 |
| 2026-08-06 | J1 | 产品代码合入并完成六档独立 64K normal gate | `pypto-lib stepfun/develop@7928a275`（base=`56b3d477`），`decode_fwd.py` SHA256=`7884da7c…`；36/36 normal、correctness finalize、counterbalance PASS；六档 hidden bit-exact，p50 均 non-regression。formal DFX/publication/all-rank swimlane 待补，根目录 `/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-formal-act-n64-20260806-v1` |
| 2026-08-05 | C5 → ❌ + C5′ | 分段计时完成；C5 实测证伪并关闭 | **① 分段计时（§7.1.1）**：用 `benchmark/2026-07-29-v3-64k/dfx-swim` 全 8 rank swimlane，按**程序序**配对（`task_token_raw` 高 32 位是 ring_id、跨 rank 不一致，90 个 barrier 里只有 12 个 task_id 相同，**不能按 task_id join**）。排除每步第一个 barrier 后 89×8：`min over ranks`（自身搬运）**39.6–44.7 µs**（p50 41.1）、mean 51.1 → **自身搬运 ~80%**；straggler 在 8 rank 间轮换（rank2 21 / rank1 16 / rank3 6）**无坏卡**；480 KB ÷ 单核 ~14 GB/s ≈ 34 µs 对账吻合。⇒ clean 的 16.1 µs 基本全是自身搬运，**靶子合法**。**② C5 证伪**：6 个 notify/wait 循环 + push AG 循环改 `pl.parallel` 后，wave5 镜像 compile A/B 出**逐字节相同**的 `kernels/aiv/tp_all_reduce.cpp`(431 行) 与 `ptoas/tp_all_reduce.cpp`(378 行)。循环**未被展开**（9 个真实 C `for`，`TNotify`×3 / `TWait`×5 在循环体内），**InCore 的 AIV codegen 不消费 `ForKind`** → `pl.parallel` 在 InCore 是惰性注解，只有 Orchestration 层（切 task/block）才有意义。连带：§2.4 把 `twophase_par` −35% 归因于 `pl.parallel` **站不住**（真实差异更可能是它 AG 直接写 `out`、无 final copy、无 Wave3 的结构差异），引用前需重测。**产物**：3 个 kernel 文件全部回退，只留一条守卫契约 `test_tp_all_reduce_keeps_reduce_scatter_accumulate_serial`（+16 行）；in-image 全量 unit **218 passed / 4 skipped**，two-layer compile-only **RC=0**。复现：`0162:/mnt/persist/chensiyu/workspace/ar-c5/compile_ab.sh {base,c5}`、`0162:/tmp/ar_segment2.py`。⚠ **未做 device/ITL**：0162 有 `0162-full-machine-perf.lock`（今天 19:28），另一 campaign 今天在 0–7 与 8–15 两组卡都在跑，时序测量会互相污染 |
| 2026-08-05 | C5/C6/C7 | 复核 Wave5 后 TP all-reduce 残余瓶颈，立 3 个 ⬜ 候选并纠正两处口径 | 复核基准 `pypto-lib stepfun/develop@7099476b`。**口径纠正**：① 权威耗时 = DFX avg **16.1 µs**/次、span **1.84%**（「40+ µs」来自 PMU 17× / swimlane 476× 放大 run）；② 该 16 µs **含 barrier 自旋等待**（profiling 计入 kernel compute），clean run 跨卡 skew 2.914 ms 全被首个 barrier 吸收；③ vLLM-Ascend 的 ~10 µs 是 HCCL/SDMA 口径，不计入 AICore kernel 时间，**不可直接对比**。**算法已到下界**（224 KB/卡 = `2(P-1)/P × N`），残余全在实现层：单核 `block_num=1`（runtime 默认，`pto_submit_types.h:250-270`；48 AIV 用 1 个）、C4 落地版丢了微基准赢的并行扇出（只 final copy 用 `pl.parallel`）、本地搬运 256 KB > 跨卡 224 KB、零 compute/comm overlap。候选 C5（补并行扇出，~8 行）/ C6（消 Wave3+final copy）/ C7（atomic-add push RS，需 rebase 拿上游 `9776f276` bf16 atomic-add）。**多核化与 MC2 级融合明确不建议**（全仓零跨卡多核先例；Wave5 已立「不机械合入无稳定收益的 AR/residual/RMS 融合」）。详见 [`03-tp-allreduce-algorithm-comparison.md`](03-tp-allreduce-algorithm-comparison.md) §7 |
| 2026-08-05 | C5/C6/C7 纠错 | 复核 §7.3 三条落地前提：推翻两处、确认一处 | ① **推翻上一行「全仓零跨卡多核先例」**：V4-Flash `origin/main:models/deepseek/v4-flash/moe.py:203-274` 就在 `pl.spmd(N_LOCAL, "dispatch_push")` 里做跨卡 `pld.tensor.put`/`remote_store`/per-block `notify`，阈值 `expected=moe_epoch * N_LOCAL`。早期结论误读了工作树 `models/deepseek/v4/`——`SKILL.md §7` 已警告那不是指定 baseline（`origin/main` 只有 `v4-flash/`，工作树只有 `v4/`，不是同一份代码）。**今后引 baseline 必须 `git show origin/main:models/deepseek/v4-flash/…`，不读工作树。** ② **推翻 C6 的「push 直接落输出」**：`local: pl.Tensor`（`decode_fwd.py:251`）不是 `pld.DistributedTensor`，peer 无法 `remote_store` 进去 → final copy 只能多核化、删不掉；Wave3 是 **run 边界**的 window 复用护栏（层内 `win_off=(layer_idx+1)*BATCH` 已按层切，跨 `rt.run()` 复用同 buffer）。③ **确认阈值风险可归零**：notify 不折进 spmd、留在其后的 `pl.at(CORE_GROUP)` scope，则 `expected=1/2/3` 一行不改。**多核化否决理由改为量级**：16.1 µs / span 1.84% 的天花板 vs 90 调用点 × task 图膨胀（按 3.5–5.2 µs/task dispatch）+ §2.5 `onephase_par` 并行反而变慢的实测反例（215→277 µs）；同期 `combine_wait` 13.4 ms / 24% 是 13× 大的靶子。**下一步 = 先做分段计时**（把 16.1 µs 拆成自身搬运 vs peer 等待），别先改代码 |
| 2026-08-04 | J1 | 旧源码 campaign 完成候选选择，不作为当前发布完成态 | `moe-opt@505e2c6b`（base=`7099476b`），gate/up split + `row16/K512/N64/down-N256`；repeated p50 -11.58%；两套 L3/L4 hidden bit-exact。证据只用于选择最终实现，需在 2026-08-06 最新 pins 上重跑 formal gate |
| 2026-08-04 | J1 | 历史短 workload 诊断完成 | context=1 repeated p50 `12.1777→10.7677 ms`，gate/up AIC p50 `≈144→12.7–12.9 µs`；该证据用于 task-grain 选择，不再作为最终 64K DFX/swimlane 发布路径 |
| 2026-08-03 | J1 | 建立 focused MoE 专项并冻结五层、双 hidden 与 0162 验证合同 | 仅 L0–L4；L4 消费 L3；禁止用 whole-net/64K 或 mock hidden 替代；设计见 [`05-moe-optimization.md`](05-moe-optimization.md) |
| 2026-08-03 | I2 | Wave5 self-target TPUT source publication 关闭间歇性 TP spread，0162 release gate 转绿 | `pypto-lib@7099476b`；manifest `sha256:4acc77cd…`；Main N=128×3 均 `123/128(spread=0)`；Main batch16/MTP/64K+batch16 ITL/DFX PASS；64K p50 `49.796 ms` |
| 2026-08-03 | I2 | 三波 completion lifetime + harness AST 对齐进入 Wave4 immutable candidate；raw token gate 转绿，稳定性仍阻塞 | `pypto-lib@d7e1381b`；64K p50 `50.204 ms`；N=128 为 `122/128(spread=2.0)`、`123/128(spread=0)`；LOW-WAIT rank2 |
| 2026-08-02 | I1/I2 | Attention/Vec 产品实现收口并构建历史 clean canonical candidate；I1 完成，当时 I2 因 raw precision gate 阻塞 | 源码 `pypto-lib@76d96bdb` / `pypto@defa97c5`；64K p50 `50.563 ms`；N=128 三轮均 `121/128` |
| 2026-07-24 | — | 专项建档，12 个子任务初始化为 TODO | 对照 v4-flash `decode_fwd.py` 拆分；HLD/LLD 见同目录 |
| 2026-07-24 | — | 验证标准改为**单一多步 decode**（N=128 ≥95% vs vanilla）；删除单步/单 token 单列（多步已含首 token） | 采纳用户口径；stall 用探针独立判定 |
| 2026-07-24 | G1 | 新增：调度轴 batch→experts/feature + dynamic active-token（对齐 DeepSeek） | 源自 batch/SPMD 分歧调研 |
| 2026-07-24 | A1 | 🟦 代码接线：holder.run() 的 N1_DFX 扩到 swim/l2/pmu（+N1_PMU）+ perf-baseline.md 骨架 | b-csy-develop 无 NPU；device 采集待 0162/镜像 |
| 2026-07-24 | A1 | ✅ 完成：镜像(stepfun-develop-20260724, cards 8-15)跑 DFX，逐层拆解 = routed-expert 90.7% / PMU cube_int8 88.6%；结果并进 benchmark/2026-07-24-step3p5-decode-perlayer-dfx.md；DFX 接线进 stepfun/develop | 基底改为镜像(feat 分支已废)；total-step 基线复用 benchmark 590ms |
| 2026-07-24 | — | **历史 base 校正**：当时 base = `stepfun/develop @ bc5eecb1`（fork csy0225）。旧文档基于 pre-prune `3af13f4f`/faithful_real/generator，均失效 | 当时的手写 unroll Main 现已整体 retired |
| 2026-07-24 | C2 | 历史记录：single_chip_hidden dispatch/combine 曾切换为pull（commit `42ac1ffd`） | 2026-07-27新决定已覆盖该完成态；pull仅作迁移前回归基线 |
| 2026-07-24 | B1 | ✅ 对账确认：StackedDeviceTensor+IPC+child_memory+dynamic-offset slice(8b4bf3fa) 已交付；resident= IR 属性纯文档不加 | 剩 dynamic-offset 归 B2 |
| 2026-07-24 | C1 | ⬜ 设计/实验记录保留，但未进入 current release；单 window/epoch 仍待独立设备回归 | 不能把 730MB→~17MB 写成当前收益 |
| 2026-07-24 | B2 | 🟦 用户批准。现状 45× unroll，body 已参数化；真难点=6 个 *_chip_orch enum 统一。C1 后折循环 | 收益=编译期 IR/调度边（间接） |
| 2026-07-26 | B2 | ✅ current loop-form Main replacement 发布收口：canonical-only 清理前后 256-step token/hidden `256/256` exact，max hidden diff `0.0`，TP spread `0.0` | vanilla raw `240/256=93.75%` 由清理前 canonical 完全复现；raw 95% gate 与 replacement gate 分开记录 |
| 2026-07-26 | — | active release base 前进到 `stepfun/develop@53eb7212`；loop-form Main 正式迁入 `models/step3p5/decode_fwd.py`；`bc5eecb1` 降为历史设计 base | canonical symbol 固定为 `whole_decode_step3p5`；旧 compatibility package/aliases 后续删除 |
| 2026-07-26 | — | 补充历史优化收益前后对比并校正 B1 口径 | B1：zero-copy opt bucket view，避免约 0.94 GiB/rank额外副本；B2：源码31,686→4,772、MoE body 40→1；当时C2记录的push→pull结论已由2026-07-27 V4-Flash迁移决定覆盖 |
| 2026-07-26 | B3/C1/C3 | 历史记录：曾按current pull设计ready/read-complete双波并评估peer/route并行 | 2026-07-27新决定已覆盖：C1回归共享window+epoch，C2/C3直接迁移V4-Flash expert-lane数据流 |
| 2026-07-26 | G1 | 历史记录：holder接入`num_tokens_per_owner`并让部分MoE阶段按active rows运行 | 2026-07-27新决定已覆盖“固定16-row/KV padding reserve暂留”口径；目标是全路径runtime dynamic batch/token |
| 2026-07-27 | — | canonical-only 清理：删除 retired unroll Main、两份 Phase-2 single-layer decode draft、断链的 per-layer MoE socket worker 和零引用一次性 repro/golden；dense MLP 抽到 `models/step3p5/dense_mlp.py` 供 Main/MTP 共用；holder/sidecar/harness/CI 删除 rollback 与自定义 Main selector | 后续只以 `decode_fwd.py:whole_decode_step3p5` 为 base；历史 reference 加 retired 标识，不再作为执行指南 |
| 2026-07-27 | C1 | 收紧 512B 口径与合同：只对同 backing 中 stacked/reused 且参与 notify/wait/AtomicAdd 的 control slot 使用 512B physical stride；逻辑访问仍为前 `n_ranks` 个 INT32 | DeepSeek 的 512B 是 data tile/cache-line/MTE 口径，不是通用 signal/window ABI；MTP 独立 compact signal 保持 `tp_size*4` |
| 2026-07-27 | B3/C1/G1 | canonical 静态实现与 25 项合同测试通过；设备/镜像 gate 尚未完成 | C3 明确保持 IN PROGRESS，不能以 InCore `pl.parallel` 伪完成 |
| 2026-07-27 | C3 | 以 `origin/main:models/deepseek/v4-flash/` 重新审计：V4-Flash 使用 expert-lane `pl.spmd` push/gather 与 combine scatter/wait/reduce，不使用 peer slab/`spmd_submit`/`task_dummy` 候选拓扑 | 撤销 probe 反向驱动产品架构的口径；没有明确 step3p5 差异或性能证据时默认沿用 V4-Flash 语义，不新增 probe |
| 2026-07-27 | B3/C1/C3/G1 | 验收证据收敛：B3 canonical compile-only PASS；C1 current-source lowered signal chain PASS、无 runtime DAG 故整体 NO-GO；G1 current-source compile/lowered active-bound PASS、无 route telemetry。C3 standalone peer-SPMD 只证明候选 DSL 可编译，不再作为 canonical production architecture 或 NO-GO 依据 | pypto-lib 本地 commits `899c7ffa` / `dc523fed` / `efc511c3`；未 push。source/synthetic/compile/device 证据继续分级；canonical/all-reduce 保持基线 |
| 2026-07-27 | C1/C2/C3 | 文档/合同决定更新：C2/C3不再保留“先量化current pull再决定”的分支，直接迁移V4-Flash expert-lane dispatch push/gather与combine scatter/wait/token reduce；C1删除pull专属双波描述，仅保留shared windows、单调epoch与真实arrival/completion生命周期 | probe拓扑与shape不作为硬约束；本次仅改文档，不改canonical代码 |
| 2026-07-27 | G1 | 产品合同更新：`BATCH=16`不再是step3p5逻辑batch硬约束；runtime active batch/token贯穿attention、MoE、combine和KV写入 | 静态formal shape仅可作为可配置capacity上界；清理固定16 padding与永久KV reserve表述；本次仅改文档 |
| 2026-07-27 | 方法论 | 禁止局部shape/name比较；所有“step3p5独有/必须保留”判断改为producer→数学变换→transport/window→consumer→rounding/reduction→lifetime端到端审计 | 差异统一分为能力/算法、数学语义、layout/shape、host/allocator集成、backend/profile workaround；任务描述先与current source对账 |
| 2026-07-27 | C/D/G 设备回归 | 0162 镜像、cards 8-15、canonical `whole_decode_step3p5`：clean active-batch=2/8/16 单次通过；clean active-batch=1 复现 `output_token=6127`、TP spread `4.203125`，`N1_DFX=dep` 旧产物也异常；norm extent=2 实验恢复 spread=0 但 token 仍错误，已回退 | DFX 不是唯一根因；不能把 batch=2/8/16 泛化为全部动态 batch；C/D source/compile/lowered 仍保持 DeepSeek-first |
| 2026-07-28 | C4 | ✅ TP all-reduce 换 reduce-scatter + **push** all-gather。**修正 03 文档 §5**：其 twophase_par 结论来自独立微基准，pull 形式 all-gather 落整网导致 8 卡 `hidden_tp_spread` 2~58；根因=pull-after-remote-notify 跨方向握手无序（与 §5 判 ring 死刑同一上游缺口），改 push 后归零 | 回归=whole-network CI PASS + 3×8 step 全 `0.0`；顺带修掉 `dense_mlp.py` 与 2 处 MoE shared-expert 丢弃 `tp_all_reduce` 返回值（违反 dev-constraints §1.1 alias/ownership）|
| 2026-07-29 | A1/C4 | ✅ 发布镜像 64k 实测落档：ITL 曲线(1024→65536 只涨 18.8%，64k p50 83.3ms) + active_batch 扫描(bs≤8 OK，bs=16 撞 device HBM) + DFX 拆解。**优化方向修正**：0724 的「C 系通信优先」只适用 ctx≈1（那次 `--steps 1`，KV 池开到 64k 但 decode 位置在 ctx≈1）；64k 实测 `tp_all_reduce` 1.84% span、routed expert busy 0.99%，C 与 D/F 系对 64k 延迟均已低 ROI。**但下一个目标尚未确定**：ITL 曲线约束住 context-dependent 部分 ≤16%，≈70ms 固定 floor 的构成被插桩掩盖（span 膨胀 5.21×，attention 占 56% task 数被放大） | benchmark/2026-07-29-release-image-64k-dfx-itl.md + data/2026-07-29_release_image_64k/；下一步 = 同镜像 ctx=1024 vs 65536 的 DFX A/B 相减，拆开 70ms floor 后再定优化目标 |
| 2026-07-27 | C/D/G 约束清理 | canonical 注释改为 V4-Flash-style shared EP window + epoch lineage；512B 仅 stacked/reused control slot 的 step3p5 backend/profile 隔离，不是通用 DeepSeek signal ABI；未改 TP all-reduce、未恢复 pull | 旧 pull 只作历史回归基线；按 source/compile/lowered/device/runtime DAG 分级，不因 probe 编译越级 DONE |


### 2026-07-29 首次 host/device 分账：新增 Track H，H1 落地

A1 建完 baseline 后第一次把 ITL 拆成 host vs device，发现 **85 ms 里只有 ~55 ms 是 device 执行**，
约 25 ms 在 host，其中 `_reset_persistent_domains` 一项独占 **21.5 ms**（每步 248 次串行阻塞 mailbox
往返 + 244.7 MiB H2D，而其中只有 47,616 B 是语义上必须清零的信号计数器）。

- **H1 ✅**：改走 backend 给 fresh window 用的同一条 device 侧 `aclrtMemset`（新增 `_CTRL_MEMSET` 控制 op，
  `broadcast_control_all` 8 卡并行）。清零 `21.50→2.21 ms`，ITL p50 `85.02→65.55 ms`（**−22.9%**），
  每步 H2D 归零。语义等价：区间/填充值/「清完才下发」的 happens-before 全不变，sim 平台仍走原 host 路径，
  改动对镜像是纯新增（`+120/-0` + `+12/-0`）。commit：simpler `e2efebcb`、pypto `1f704616`（含 gitlink bump）。
- **证伪 4 条**（勿重复走）：① 「3 波 barrier 共用 signal cell + 每步清零 ⇒ notify 被抹 ⇒ 挂死」——清零是
  忙等同步的，8 卡清完才下发；② 「step3p5 关掉 `persistent=True` 就行」——实测 ITL **276.2 ms**，
  domain churn ≈ 212.7 ms，是它能省下的 10 倍；③ 「每步 12336 次串行 `_submit_chip`」——B2 之后只有 **8** 次；
  ④ 运行时**没有** multi-step/resident/replay 接口。
- **H2 / H3 立项**：跨卡起跑阶梯 clean run 实测 **2.914 ms**（每 rank +0.412 ms 等距），根因是生成
  `host_orch` 的 rank-major 循环；**v4-flash 同形状**，故列为 pypto codegen 通用改进。DFX run 的
  115–764 ms 假长条另立 H3（观测性），它曾使 `tp_all_reduce` 被误判成 74.1% wall。
- **README 新增第二维度**：把 A–H 按「优化落在栈的哪一层」（L3/host · L2/AICPU 调度 · cross-chip 通信 ·
  L1/kernel 内数据流 · L0/核内流水 · 结构 codegen · 可观测性）重切一遍，并给出 ITL 65 ms 的分层分账。
  H1 之后 host 从 29% 压到 8.8%，**device 执行首次成为主导项（91%）**，后续重心从 L3 移到 cross-chip 与 L2/L1。
- **待办**：live N=128 精度门（需 vanilla oracle 占 cards 0-7）。

### 2026-07-28 C/D/G 收口与镜像验证

- `b404a3c9`：恢复固定 expert-lane physical bases（`expert_recv_max = n_ranks * BATCH`），修复 BS1 batch-extension invariance；BS1/2/16 单步 token `6127→303`、TP spread `0`，BS1 持续 4-step `6127→303→1207→19384→872`，row0 hidden 与 BS2/BS16 bit-identical。
- `563fe62a`：镜像 CI 清理忽略 zombie-only process group，并增加 `--skip-mtp`；MTP oracle 缺失单独记录为外部依赖，不归因于 C/D/G。
- 最终本地 candidate 镜像 `hub.i.basemind.com/stepcast/vllm-pypto:step3p5-b404a3c9-ci-final-20260728`（产品 HEAD `b404a3c9` + 对应 `563fe62a` 的 CI 三文件 patch，未推 registry）：smoke PASS，Main 8-step `303,1207,19384,872,428,6127,4231,2636` exact，hidden 全 finite、8 rank active rows nonzero、TP spread `0`。
- 镜像内 N=256 teacher-forced 回归：`256/256` hidden finite、`256/256` TP spread `0`，token exact `241/256`；该结果沿用历史 live vanilla oracle 序列，raw 95% gate 不宣称通过。
- 本地 `stepfun/develop` 已快进到 `563fe62a`；代码已推送至 `csy0225/pypto-lib` 的 `stepfun/develop` 与 `perf/step3p5-bc-20260726`。

### 2026-07-27 顶层方向收口

本轮复核发现，之前存在“局部 shape/layout 差异 → step3p5 独有架构”的误判。后续所有任务必须先完成端到端同构审计：

```text
producer → 数学变换/quant/route-map → transport/window
→ consumer → rounding/reduction/placement → lifetime/reuse/allocator
```

因此：INT8 activation+scale、resident/InOut KV、shared window+epoch、owner max/active-token、fixed storage+dynamic valid rows、expert-lane fan-out、LM-head data-window reuse 均优先直接沿用 V4-Flash；只有 vLLM/IPC/allocator、45层 flat layout、step3p5 shape、canonical whole-net 和当前 backend/profile 才能作为局部适配。C2/C3直接迁移 V4-Flash push/gather/scatter/reduce；route weight在expert W2 epilogue完成缩放，combine只做FP32 token reduction。默认16只是容量实例，不是逻辑batch或永久padding/KV reserve合同。
