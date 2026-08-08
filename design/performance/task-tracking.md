# 任务跟踪记录 — Performance 专项

> step3p5 decode 性能优化的**单一事实源**。每个子任务的状态、owner、更新时间、阻塞在此维护。
> 设计详情见 [`02-detailed-design.md`](02-detailed-design.md)；改状态时同步更新本表的「最后更新」列 + 底部更新日志。
>
> **状态图例**：⬜ TODO ｜ 🟦 IN PROGRESS ｜ ✅ DONE ｜ ⛔ BLOCKED ｜ ⏸ PAUSED

---

> **⚠ 2026-08-08 current-source override**：attention/Vec 产品实现权威 pin 为
> `pypto-lib stepfun/develop@491267c45875e9b1e0071eed224e2e73526799e2` 与
> `pypto stepfun/develop@8e92b46808f9f7c09b6431ad4691503f09c12ee5`。Wave5
> 以 self-target TPUT 发布 source partial，并保持既有三波 lifetime；immutable
> audit/smoke/Main+MTP compile、Main N=128×3、Main batch16、MTP batch1/batch16×2、
> 64K/batch16 ITL/DFX 均通过，是最后一个完整 production release-qualified
> 回退基线。`63814d4a` 将 SWA mask 从 `pl.cmp` predicate 转换路径改为 typed
> INT32 数值区间 mask；0162 source-overlay N=128 为
> `127/128=99.21875%`、TP spread=0。当前没有包含该提交的 immutable image，
> 镜像发布已按用户决定推迟到统一 release commit 确定后。manifest
> `sha256:3eb694e…` 仅是 `c9af5790` pre-fix evidence；历史 R1/R2 已
> supersede。下方历史 override 不覆盖 I1/I2。
>
> **⚠ 2026-08-07 L0–L4 focused MoE override**：Track J 的产品实现已随
> `7928a275` 进入上述 `491267c4`；范围只包含
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
| H2 | per-rank 视图重建 hoist 到 `prepare()`（= 跨卡起跑阶梯病根） | P1 | ⬜ | — | H1 | 已量化未修：起跑阶梯 clean run 实测 **2.914 ms**（每 rank 等距 +0.412 ms，n=20），submit 阶段 3.49 ms。根因 = 生成 `host_orch.py` 把 per-rank 体（53 常量 slice + 38 `pl.reshape` + ~92 `make_tensor_arg`/`add_tensor` + 1 `_submit_chip`）包在 `for r__idx_v0 in range(0, world_size, 1)`。**v4-flash `decode_fwd.py:774` 完全同形状** → 属 pypto codegen 通用改进而非 step3p5 缺陷。红队复核修正：抹平阶梯只值 ~0.4 ms（关键路径是最后一个 rank），真收益来自减少 host 工作本身 | 2026-07-29 |
| H3 | DFX run 第一 barrier 假长条（观测性，非性能） | P2 | ⬜ | — | A1 | DFX run 里第一个 `tp_all_reduce` 被记成 115 ms(pmu)–379.9 ms(swim)，其余 89 次 39–366 µs，straggler 每次换人。**已排除 host 下发**（`_submit_chip` 在 DFX 下只多一次字符串拼接；clean run 下发等距 0.412 ms）。clean run 算术上界：非 device 时间共 5.7 ms → 380 ms 不可能存在。方向 = chip child 侧 collector 开销落在被 trace 区间内（注意 `orch._dfx_dispatch_idx` 每 request 重置，留下的是**最后一步**，非冷启动）。危害：曾使 `tp_all_reduce` 被误判成 74.1% wall | 2026-07-29 |

### Track I — Attention / Vec 收尾与 canonical 发布
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| I1 | workload-derived attention、Full 层次归约、out-proj cast、dense Vec 收尾 | P0 | ✅ | codex | A1, C4, H1 | `pypto-lib stepfun/develop@7099476b`（attention/Vec 内容自 `76d96bdb` 起保持）：logical task 按 active workload 推导；Full SV 合并 segment recurrence，只保留 reduce/finalize；Full/SWA out-proj cast 默认融合；dense RMS direct BF16 reread、dense down-proj cast fusion 保留；AR+residual、residual+RMS stats、RMS+projection、gate/up+SiLU 等无稳定收益方案不合入。active-batch=16/异构 context、source/compile/device/DFX 已完成。当前 task/tile 设计见 [`04-attention-optimization.md`](04-attention-optimization.md) §13 | 2026-08-03 |
| I2 | TP all-reduce immutable release stability gate | P0 | ✅ | codex | I1 | Wave3/4 先闭合 final-read lifetime 并对齐 harness AST；Wave5 `7099476b` 再以 self-target synchronous TPUT 发布 source partial，并同步 Main/MTP/harness/返回值 lineage。manifest `sha256:4acc77cd…`：audit/smoke/Main+MTP compile、Main N=128 预定义三轮均 `123/128` 且 spread=0、Main batch16、MTP batch1/batch16×2、64K/batch16 ITL/DFX 全 PASS；64K p50 `49.796 ms`。machine scope=`0162 release-qualified`。见 [`../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md`](../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md) | 2026-08-03 |

### Track J — MoE compute 优化
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| J1 | L0–L4 routed gate/up stage split + task-grain tuning | P0 | 🟦 | codex | A1, C1–C3, D1–D2, G1, I2 | 产品实现 `7928a275` 已进入 `stepfun/develop@491267c4`；当前 tip 还包含 active-route scheduling 和 route/precision release harness。普通 expert 为 `row16/K512/N64/down-N256`，保留 scatter→wait 真依赖与 L43/L44 specialization。旧 `c9af5790` 镜像 `sha256:cab8966…` 的三轮 normal A/B 已 seal：BS `1/2/4/7/8/16` 每请求独立 64K，L3/L4 hidden hash exact，p50 分别改善 `9.16/1.83/3.52/6.07/0.53/11.61%`；matched-source 8/8 run sealed PASS。旧 source-overlay N=128=`127/128`、spread=0。阻塞：`491267c4` immutable image，以及其上的六档 64K golden/A/B、N=128、formal DFX/reanalysis/all-rank swimlane。设计见 [`05-moe-optimization.md`](05-moe-optimization.md) | 2026-08-08 |

---

## 进度汇总

| 状态 | 数量 |
|------|------|
| ⬜ TODO | 9 |
| 🟦 IN PROGRESS | 2 |
| ✅ DONE | 13 |
| ❌ REFUTED | 1 |
| ⛔ BLOCKED | 0 |
| **合计** | **25** |

**base 校正后关键路径**：A1/B1/B2/C1/C2/C3/C4/D1/D2/G1/H1/I1/I2 已 ✅；
historical pull C2 仅作回归基线。当前 performance 看板只剩
**B3（KV resident/in-place 的连续多轮 row-diff/liveness 证据）**与
**J1（final immutable image 六档/precision/DFX/swimlane）**处于进行中。
Attention/Vec 与 TP all-reduce stability 已在 0162 release-qualified；J1 的
pre-fix focused normal PASS 和 source-overlay precision PASS 不能升级为最终
immutable-image、whole-net 或 L43/L44 release 结论。
其它机器仍需独立 gate。

2026-08-05 新增 **C5/C6/C7**（TP all-reduce 实现层残余开销）为 ⬜ 候选，
**均非关键路径**：共同天花板是 64K 下 span 1.84%。同日已完成两件事：
① **分段计时**（§7.1.1）——16.1 µs 里约 **80% 是自身搬运**，不是等 peer，靶子合法；
② **C5 实测证伪并关闭**（❌）——`pl.range`→`pl.parallel` 在 InCore 里出**逐字节相同**的
codegen，收益恒为 0；剩下的实现层杠杆只有 **C5′**（self-TPUT `pipeline=True`，需 Wave5
同级稳定性验收）和 **C6/C7**。`pl.parallel` 不能用来拆核，要拆核只有 `pl.spmd`。
详见 Track C 表下的说明。

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
| 2026-08-08 | J1 | 当前源码 tip 与最小 pending build spec 冻结 | `stepfun/develop@491267c4` 已包含 `7928a275`、active-route scheduling、SWA mask 修复和 route/precision release harness；当前无包含该提交的 immutable image，下一步按 pending spec 构建并执行 0162 标准回归 |
| 2026-08-07 | I1/J1 | SWA mask 精度修复进入远端，状态文档冻结并推迟镜像发布 | `stepfun/develop@63814d4a`：typed INT32 数值区间 mask 替代 `pl.cmp` predicate 转换；0162 source-overlay N=128=`127/128=99.21875%`、唯一 miss `step94 478→320`、TP spread=0。当前无包含该提交的 immutable image；后续选统一 release commit 再走标准发布流程 |
| 2026-08-07 | J1 | `c9af5790` pre-fix 六档 normal A/B、双 hidden与 matched-source A/B 完成，J1 保持 🟦 | 36/36 focused run 已 seal；`normal_seal_authority.json` SHA256=`875804dd…`，r1/r2/r3 性能报告 SHA256=`d238baa1…`。whole-net 1-step×2、2-step×2 共 8/8 sealed PASS。SWA mask 已变化，final image 仍须重跑六档/精度并补 formal DFX/swimlane |
| 2026-08-06 | J1 | 产品实现并入最新 `stepfun/develop`，启动 canonical formal regression | `7928a275` 已是 `c9af5790` 祖先；新镜像必须绑定 `pypto@8e92b468`、A2/A3 profile 和 prepared swimlane capability。准出为 BS `1/2/4/7/8/16` 各自 64K、L3/L4 hidden golden、端到端精度与 all-rank DFX；完成前保持 🟦 |
| 2026-08-05 | C5 → ❌ + C5′ | 分段计时完成；C5 实测证伪并关闭 | **① 分段计时（§7.1.1）**：用 `benchmark/2026-07-29-v3-64k/dfx-swim` 全 8 rank swimlane，按**程序序**配对（`task_token_raw` 高 32 位是 ring_id、跨 rank 不一致，90 个 barrier 里只有 12 个 task_id 相同，**不能按 task_id join**）。排除每步第一个 barrier 后 89×8：`min over ranks`（自身搬运）**39.6–44.7 µs**（p50 41.1）、mean 51.1 → **自身搬运 ~80%**；straggler 在 8 rank 间轮换（rank2 21 / rank1 16 / rank3 6）**无坏卡**；480 KB ÷ 单核 ~14 GB/s ≈ 34 µs 对账吻合。⇒ clean 的 16.1 µs 基本全是自身搬运，**靶子合法**。**② C5 证伪**：6 个 notify/wait 循环 + push AG 循环改 `pl.parallel` 后，wave5 镜像 compile A/B 出**逐字节相同**的 `kernels/aiv/tp_all_reduce.cpp`(431 行) 与 `ptoas/tp_all_reduce.cpp`(378 行)。循环**未被展开**（9 个真实 C `for`，`TNotify`×3 / `TWait`×5 在循环体内），**InCore 的 AIV codegen 不消费 `ForKind`** → `pl.parallel` 在 InCore 是惰性注解，只有 Orchestration 层（切 task/block）才有意义。连带：§2.4 把 `twophase_par` −35% 归因于 `pl.parallel` **站不住**（真实差异更可能是它 AG 直接写 `out`、无 final copy、无 Wave3 的结构差异），引用前需重测。**产物**：3 个 kernel 文件全部回退，只留一条守卫契约 `test_tp_all_reduce_keeps_reduce_scatter_accumulate_serial`（+16 行）；in-image 全量 unit **218 passed / 4 skipped**，two-layer compile-only **RC=0**。复现：`0162:/mnt/persist/chensiyu/workspace/ar-c5/compile_ab.sh {base,c5}`、`0162:/tmp/ar_segment2.py`。⚠ **未做 device/ITL**：0162 有 `0162-full-machine-perf.lock`（今天 19:28），另一 campaign 今天在 0–7 与 8–15 两组卡都在跑，时序测量会互相污染 |
| 2026-08-05 | C5/C6/C7 | 复核 Wave5 后 TP all-reduce 残余瓶颈，立 3 个 ⬜ 候选并纠正两处口径 | 复核基准 `pypto-lib stepfun/develop@7099476b`。**口径纠正**：① 权威耗时 = DFX avg **16.1 µs**/次、span **1.84%**（「40+ µs」来自 PMU 17× / swimlane 476× 放大 run）；② 该 16 µs **含 barrier 自旋等待**（profiling 计入 kernel compute），clean run 跨卡 skew 2.914 ms 全被首个 barrier 吸收；③ vLLM-Ascend 的 ~10 µs 是 HCCL/SDMA 口径，不计入 AICore kernel 时间，**不可直接对比**。**算法已到下界**（224 KB/卡 = `2(P-1)/P × N`），残余全在实现层：单核 `block_num=1`（runtime 默认，`pto_submit_types.h:250-270`；48 AIV 用 1 个）、C4 落地版丢了微基准赢的并行扇出（只 final copy 用 `pl.parallel`）、本地搬运 256 KB > 跨卡 224 KB、零 compute/comm overlap。候选 C5（补并行扇出，~8 行）/ C6（消 Wave3+final copy）/ C7（atomic-add push RS，需 rebase 拿上游 `9776f276` bf16 atomic-add）。**多核化与 MC2 级融合明确不建议**（全仓零跨卡多核先例；Wave5 已立「不机械合入无稳定收益的 AR/residual/RMS 融合」）。详见 [`03-tp-allreduce-algorithm-comparison.md`](03-tp-allreduce-algorithm-comparison.md) §7 |
| 2026-08-05 | C5/C6/C7 纠错 | 复核 §7.3 三条落地前提：推翻两处、确认一处 | ① **推翻上一行「全仓零跨卡多核先例」**：V4-Flash `origin/main:models/deepseek/v4-flash/moe.py:203-274` 就在 `pl.spmd(N_LOCAL, "dispatch_push")` 里做跨卡 `pld.tensor.put`/`remote_store`/per-block `notify`，阈值 `expected=moe_epoch * N_LOCAL`。早期结论误读了工作树 `models/deepseek/v4/`——`SKILL.md §7` 已警告那不是指定 baseline（`origin/main` 只有 `v4-flash/`，工作树只有 `v4/`，不是同一份代码）。**今后引 baseline 必须 `git show origin/main:models/deepseek/v4-flash/…`，不读工作树。** ② **推翻 C6 的「push 直接落输出」**：`local: pl.Tensor`（`decode_fwd.py:251`）不是 `pld.DistributedTensor`，peer 无法 `remote_store` 进去 → final copy 只能多核化、删不掉；Wave3 是 **run 边界**的 window 复用护栏（层内 `win_off=(layer_idx+1)*BATCH` 已按层切，跨 `rt.run()` 复用同 buffer）。③ **确认阈值风险可归零**：notify 不折进 spmd、留在其后的 `pl.at(CORE_GROUP)` scope，则 `expected=1/2/3` 一行不改。**多核化否决理由改为量级**：16.1 µs / span 1.84% 的天花板 vs 90 调用点 × task 图膨胀（按 3.5–5.2 µs/task dispatch）+ §2.5 `onephase_par` 并行反而变慢的实测反例（215→277 µs）；同期 `combine_wait` 13.4 ms / 24% 是 13× 大的靶子。**下一步 = 先做分段计时**（把 16.1 µs 拆成自身搬运 vs peer 等待），别先改代码 |
| 2026-08-04 | J1 | 旧源码 campaign 完成候选选择，不作为当前发布完成态 | `moe-opt@505e2c6b`（base=`7099476b`），gate/up split + `row16/K512/N64/down-N256`；repeated p50 -11.58%；两套 L3/L4 hidden bit-exact。证据只用于选择最终实现，需在 2026-08-06 最新 pins 上重跑 formal gate |
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
