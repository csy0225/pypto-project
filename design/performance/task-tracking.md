# 任务跟踪记录 — Performance 专项

> step3p5 decode 性能优化的**单一事实源**。每个子任务的状态、owner、更新时间、阻塞在此维护。
> 设计详情见 [`02-detailed-design.md`](02-detailed-design.md)；改状态时同步更新本表的「最后更新」列 + 底部更新日志。
>
> **状态图例**：⬜ TODO ｜ 🟦 IN PROGRESS ｜ ✅ DONE ｜ ⛔ BLOCKED ｜ ⏸ PAUSED

---

> **⚠ 2026-08-03 current-source override**：attention/Vec 产品实现权威 pin 为
> `pypto-lib stepfun/develop@7099476b7c4f13112b159e237e7a64344803caf0` 与
> `pypto stepfun/develop@defa97c526fec7e8f032dbbfcc39c820add02bf7`。Wave5
> 以 self-target TPUT 发布 source partial，并保持既有三波 lifetime；immutable
> audit/smoke/Main+MTP compile、Main N=128×3、Main batch16、MTP batch1/batch16×2、
> 64K/batch16 ITL/DFX 均通过，当前为 **0162 release-qualified**。其它机器/架构
> 未由本轮独立证明。下方历史 override 不覆盖 I1/I2。
>
> **⚠ 2026-08-06 L0–L4 focused MoE override**：Track J 产品实现已合入
> `pypto-lib stepfun/develop@7928a2751930b04c866788a396a7337b62c6d32f`，验证范围只包含
> `L0 Full+dense / L1–L2 SWA+dense / L3 SWA+MoE / L4 Full+MoE`，L4 必须消费
> 真实 L3 输出。最终方案为 routed gate/up stage split，普通 expert 使用
> `row=16, K=512, N=64, down N=256`；L43/L44 specialization 保持原配置。
> 0162 cards `8–15` 上已完成 BS=`1,2,4,7,8,16`、每 sequence 独立 64K 的
> 36 个 normal run；六档 L3/L4 hidden 均 bit-exact、finite、TP spread=0，p50
> reduction 为 `0.04/6.629/12.113/3.652/9.229/11.135%`。formal matched-source
> DFX、route-aware publication gate 和最终 8-rank swimlane 仍在进行，不能引用
> 2026-08-04 context=1 诊断路径作为最终 DFX。当前证据根目录为
> `/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-formal-act-n64-20260806-v1`。

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
| I1 | workload-derived attention、Full 层次归约、out-proj cast、dense Vec 收尾 | P0 | ✅ | codex | A1, C4, H1 | `pypto-lib stepfun/develop@7099476b`（attention/Vec 内容自 `76d96bdb` 起保持）：logical task 按 active workload 推导；Full SV 合并 segment recurrence，只保留 reduce/finalize；Full/SWA out-proj cast 默认融合；dense RMS direct BF16 reread、dense down-proj cast fusion 保留；AR+residual、residual+RMS stats、RMS+projection、gate/up+SiLU 等无稳定收益方案不合入。active-batch=16/异构 context、source/compile/device/DFX 已完成。最终 task/tile 设计已合并到 [`04-attention-optimization.md`](04-attention-optimization.md) §13 | 2026-08-03 |
| I2 | TP all-reduce immutable release stability gate | P0 | ✅ | codex | I1 | Wave3/4 先闭合 final-read lifetime 并对齐 harness AST；Wave5 `7099476b` 再以 self-target synchronous TPUT 发布 source partial，并同步 Main/MTP/harness/返回值 lineage。manifest `sha256:4acc77cd…`：audit/smoke/Main+MTP compile、Main N=128 预定义三轮均 `123/128` 且 spread=0、Main batch16、MTP batch1/batch16×2、64K/batch16 ITL/DFX 全 PASS；64K p50 `49.796 ms`。machine scope=`0162 release-qualified`。见 [`../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md`](../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md) | 2026-08-03 |

### Track J — MoE compute 优化
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| J1 | L0–L4 routed gate/up stage split + task-grain tuning | P0 | 🟦 | codex | A1, C1–C3, D1–D2, G1, I2 | 产品代码 `7928a275` 已合入 `stepfun/develop`：普通 expert 使用 `row16/K512/N64/down-N256`，保留 scatter→wait 真依赖与 L43/L44 原 specialization。0162 六档独立 64K normal/correctness/counterbalance PASS，L3/L4 hidden bit-exact；BS1/2/4/7/8/16 p50 reduction=`0.04/6.629/12.113/3.652/9.229/11.135%`。剩余 gate：formal matched-source DFX 12 runs、route-aware publication reanalysis、最终 all-rank swimlane 固定路径与 hash。设计见 [`05-moe-optimization.md`](05-moe-optimization.md) | 2026-08-06 |
| J2 | gate fan-out 与 norm/quant 解耦（deferred `inv_rms`） | P0 | ✅ | claude | J1 | 已发布 `stepfun/develop@d13b2ca6`（单 commit FF，只改 `decode_fwd.py` +63/-35，sha `d392311c… -> 28080c53…`）。`gate_expert_fanout` 只写 raw FP32 logits，`inv_rms/sigmoid/bias` 尾巴搬进本来就等 `inv_rms` 的 `gate_topk`；算子顺序与数值语义不变，codegen 侧 `params_t70` 不再 `add_input(moe_inv_rms)`，task 数与 `block_num=9` 不变。0162 三臂 A/B/A：bs=1/64k/nb512 p50 `36.494 -> 33.849 ms`（**+7.25%**，地板 0.634）、bs=8/64k/nb4096 p50 `97.528 -> 91.722 ms`（**+5.95%**，地板 2.637）；bs=16 物理不可行（16 GiB 单次 rtMalloc → `207001`）。**两档三臂 hidden payload 各自 byte-exact**（bs=1 = N256 golden `567b206b…`、bs=8 `1fcd4fcc…`）→ 按项目口径 sha256 即准出。机理：MoE-only 段 15→14 hop、`norm_quant` 离开关键路径、链头 `81.8 -> 56.5 us`。数据见 [`../../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](../../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md) | 2026-08-10 |

### Track K — TP all-reduce 二次优化（campaign 内部代号 P2；注意与本表「优先级」列的 P0/P2 无关）
| ID | 优化点 | 优先级 | 状态 | Owner | 依赖 | 阻塞 | 最后更新 |
|----|--------|--------|------|-------|------|------|----------|
| K1 | AR critical-tail 口径确立 + notify/drain 分账（Phase 0，只插桩不改生产） | P0 | ✅ | codex（claude 独立复核 REPRODUCED） | I2 | headline 口径改为 `critical_tail = max(rank_exit) - max(rank_entry)`；T/P/T 三臂 half-range `0.06 µs`、插桩扰动 `0.36 µs`(<1%)、`latest_entry==last_exit 128/128`、`phase_sum_error_ticks {0,0}`。**critical-tail p50 `43.18 µs`**，校准分解 control `18.38 µs`(42.6%) / data-compute `24.82 µs`(57.5%)。最大可寻址项 = 三轮 notify control 合计 `16.51 µs`（不是 Wave2 publish completion `10.62 µs`）。pooled rank×epoch p50 `171.95 µs` 已显式降级为 **host 顺序提交 artifact**，不得当 collective latency。claude 补一条定律：`first_peer_control 0.700 µs` ≈ `marginal_per_peer 0.801 µs` → `cost(n)=a+b·n` 中 **a≈0、b≈0.80**，故 16.5 µs 是 21 次 notify 的边际成本而非每 wave 固定开销。上游侧补全五仓 branch 枚举，`PTOAS fix/issue711-tnotify-mte-drain` 已是生产 pin 祖先，是 **data-before-signal 正确性约束**、不可回退也不可摘。原件 `0162:/mnt/persist/chensiyu/workspace/p2-ar-diag-20260810/phase0-split-20260810-190834/`，复核 `0162:/mnt/persist/chensiyu/workspace/perf-2026q3/claude-verify-p2-phase0-20260810/`（`claude_verify_p2_phase0.py` sha256 `d8fccb90…`、`recompute.json` sha256 `dc6d3c01…`；**执行脚本只存在于 0162，本仓不留副本**） | 2026-08-10 |
| K2a | batched notify：`n stores + 1 barrier` vs `n stores + n barriers` 两点对比 | P0 | ❌ NO-GO | codex（claude 独立读原始 diag） | K1 | **已实测否决。** 三臂 A1 `4.46` / B `4.44` / A2 `4.46 µs`，`parent_center 4.460`、`half_range 0.000`、delta `+0.020 µs`(0.45%)、`output_exact=true`、`credit_errors` 全 0。反解每 barrier `0.0033 µs`，18 个可摘 barrier = `0.060 µs/call`，48 次 = `0.0029 ms` —— 比地板 `0.634 ms` 低两个数量级。**证实 K1 的 `a≈0` 定律**：`0.786 µs/notify` 几乎全部是 remote credit store 本身，`dsb`/`pipe_barrier(PIPE_ALL)` 份额≈0，所以 barrier 多重性批处理无价值，唯一杠杆是**减少 remote store 次数**（→ K5/RD 因此复活，K6 因此立项）。原件 `0162:/mnt/persist/chensiyu/workspace/p2-barrier-store-20260810/run-20260810-195604/` | 2026-08-10 |
| K2b | publisher release fence hoist（每 peer fence → 每 wave 一次）+ whole-cache invalidate 间接成本 | P0 | 🟦 | codex（claude 独立读原始 diag） | K1, K2a | 生产 notify 点位不是裸 TNOTIFY：`MakeNotifyCodegenPTO`（pypto `src/backend/common/pto_ops_distributed.cpp`）在每个 `pto.comm.tnotify` 前无条件 emit `pto.cmo.cacheinvalid all #gm` + `pto.fence.barrier_all #gm`。128 KiB 单档三臂（warmup 2 / measure 16，kernel sha256 A`8b264955` B`ec3ad385` C`f2be3406`，三臂 `output_exact=true`、`credit_errors`/`load_errors` 全 0）：`notify_us` p50 A per_peer_full `16.17` → B per_wave_full `13.78` → C per_wave_fence_only `13.56`，**A→B = `2.39 µs/call`**，落在预估 prefix 天花板 `2.70~3.13 µs` 内；C 只再省 `0.22 µs` → 收益几乎全来自 fence 提升、cacheinvalid 那一半贡献很小。**两条限定**：① 本轮只有单个 A 臂、无 A1/B/A2 bracketing，`2.39` 只能记作单臂差值，需补 bracketing 才能算「已测」；② claude 的「whole-cache invalidate 有间接成本、`transmission_factor>1`」假设 **as measured 不成立** —— 三臂 `post_minus_hot_us = cold_minus_hot_us = 0.0`，21/3/0 次 invalidate 下游 load 无差别。但 `cold_load ≈ hot_load ≈ 92.28 µs`（连显式 invalidate 都不让 cold 变慢）说明该 workset 是 GM-bound、探针可能结构上测不出 locality，此 null 一半属于仪器 → 不宣布假设死亡，但 K2b 维持在直接 prefix 天花板、不升为 path1 首位。**落地约束**：`pld.system.notify` Python 签名（`system_ops.py:112`）无 fence/release 参数，K2b 改不了 `decode_fwd.py`，**必须上游 pypto 补丁** → 排在纯 model 侧的 K6 之后。原件 `0162:/mnt/persist/chensiyu/workspace/p2-k2b-release-cache-20260810/formal-128k-20260810/run-20260810-231938/`（另有 `bisect-20260810/` 记录多档 workset 触发 harness `code -100` 的定位过程，已绕开：128 KiB 单档即生产下游工作集） | 2026-08-10 |
| K3 | reduce→publish 按 chunk 流水 | P1 | ⬜ | — | K1 | 重叠上界 `min(reduce_owned 7.92, publish_completion 10.62) = 7.92 µs`，**单项越不过整机门 14 µs**，只能进 bundle；分块还会增加 TSTORE 启动数，实际收益可能显著低于上界 | 2026-08-10 |
| K4 | Wave3 双缓冲/epoch window + final-copy fusion | P1 | ⬜ | — | K1 | Wave3 `7.69 µs` 拆开为 notify control `5.51` + copy completion `2.18`；双缓冲打前者、copy fusion 打后者，**不得把 7.69 同时承诺给任一单项**。`self TPUT drain 3.92 µs` 只有让上游 producer 直接写 comm window 才可能删 | 2026-08-10 |
| K5-C | recursive halving RS + recursive doubling AG | **P0** | ⬜ DRAFT（未编译未跑） | codex 起草 / claude 补变体 | K2a, K6a | **被统一定律从末位提到 P0**：`a≈0` ⇒ 减少 remote 交易次数是最大杠杆，6 步 × (1 remote_store + 1 notify) = **12 次交易 vs parent 35**，且不需 remote_load 与 final copy。RS masks 4/2/1、payload 列宽 `2048/1024/512`；AG masks 1/2/4、列宽 `512/1024/2048`；六个 disjoint receive slot + `[epoch,step]` signal slot 防跨步/跨 epoch credit 复用。**codex 抓到并修正了 claude 原预测的一个矛盾**：「224 KiB」与「仅 FP32 re-parenthesization」不能同时成立 —— 原 `26.8 µs/call` 用的是 BF16 payload（C16），而 C16 每个 RS step 后要 cast 回 BF16 才能发送，**引入 3 次中间舍入，是超出括号改变的第二个数值变化**。**claude 再补一个严格更优的变体 C32′**：RS 结束时 owned shard 已是全 8 rank 的完整 FP32 和，此刻 cast 一次即与 parent 的 `BF16(FP32 sum)` 同一舍入契约，**故 AG 只需广播已定稿的 BF16 终值，AG 侧用 FP32 是纯流量浪费**。四变体（parent 每 rank 每行 `30,720 B`）：**C16** BF16/BF16 `14,336 B` = 46.7% → 预测 `16.6 µs`（省 `26.8`），数值变化 = 括号 + 3 次中间舍入；**C32**（codex）FP32/FP32 `28,672 B` = 93.3% → 预测 `23.6 µs`（省 `19.9`），仅括号；**C32′**（claude）FP32/BF16 `21,504 B` = 70.0% → 预测 `20.1 µs`（**省 `23.4 µs/call` = `1.12 ms`**），仅括号 —— **严格优于 C32：字节少 25%、数值契约相同**。三变体都远超 14 µs 门，**故门不是筛选依据，精度代价才是** → 建议运行序列改为 `A1 / C32′ / A2 / C32 / A3 / C16 / A4`，卡时间紧则 `A1 / C32′ / A2` 即可定去留。**定律外推的已知薄弱点（唯一可能翻车处）**：parent 的 21 次 notify 分 3 wave、wave 内 7 次可连发；C32′ 的 12 次交易分 **6 个串行 round**，每 round 必须等上一 round 数据到位。`0.80 µs/transaction` 是从 3-wave 拓扑与 scan B（轮内可连发）拟合的，**未覆盖「6 个串行 round」**；若 round 间有固定握手延迟则真实成本高于预测（`wave1_wait` 方差曾 `0.6 → 371 µs` 是此类风险的既有证据）。**精度代价**：括号改变 ⇒ 非 byte-exact ⇒ 准出升级为 live vanilla vLLM W8A8 oracle（N=128 逐 token ≥95%，需整机 + oracle 容器）。**2026-08-11 claude 无卡 codegen 复核（未上设备）**：draft 原本连编译都过不了，修两处后 **两变体 rc=0**；`rh_rd_bf16` payload `229,376 B`（224 KiB）、`rh_rd_fp32` `458,752 B`（448 KiB），均 6 payload store + 6 notify store。**两处修复都是真实 API 约束**：① `chip_orch` 必须全量类型标注（frontend 直接拒）；② ★ `notify` 的 value 与 `wait` 的 expected 必须显式 `pl.cast(1, pl.INT32)` —— 裸 `1` 发射成 `%c1_index : index`，PTOAS 拒 `invalid kind of type specified: expected builtin.integer, but found index`；后端 `MakeNotifyCodegenPTO` 明写契约「tnotify value 的 MLIR 类型必须与 signal 元素类型一致，且发射 value 自己的 ScalarType」。**未解之谜（如实记录）**：生产 `decode_fwd.py` 与 K6 generator 同样传裸 `1` 却得到 `%c1_i32`，机制未查明 → 规则写成「显式 cast 是稳健写法」而非「裸 1 一定是 INDEX」（属 dev-workflow gotcha，落点是 sub-repo `known-pypto-pitfalls.md`）。**三项新约束**：③ **静态 op 普查不可跨「循环 vs 全展开」比较** —— parent 用 `pl.range(group_size)` 故 3 静态 tnotify = **21 动态**，K5 全展开 6 静态 = **6 动态**，静态 `3<6` 而动态 `21>6` **方向相反**；动态交易数因此确认 parent `21+14=35` vs K5 `6+6=12`，**定律前提结构上成立**。④ ★ **UB 占用**：parent `96.0 KB`(52.2%) / bf16 `112.0 KB`(60.9%) / **fp32 `160.0 KB`(87.0%)**，峰值 `mem_vec_6` **128 KB** = RS step-0 的 `[16,2048]` FP32 送出 tile。⇒ 「K4 不得计入 RD 预算」除「同抽 notify 池」外**多一条硬理由：只剩 24 KB 余量，UB 装不下**；C32′ 的 UB 峰值与 C32 相同（峰值在 RS 侧，AG 改 BF16 不降峰值）故仍严格优于 C32；且 **C32/C32′ 不能外推到 batch>16**（FP32 RS step-0 在 batch=32 需 256 KB > 184 KB 限制），要分块而**分块增加交易次数、与「减交易」的收益来源直接对冲** —— 这是 K5-C 的结构性张力。⑤ whole-cache invalidate 从 21 降到 6，但 K2b 已实测其对下游 load 无可测差别，**不得记为额外收益**。**上设备前仍缺**：CPU 仿真 byte-exact 正确性对比（本轮 0 行为验证）、C32′ 变体尚未实现、`A1/C32′/A2` 性能序列 + 卡锁、6 receive slot 与 `[epoch,step]` 的跨 epoch 不复用未经设备验证。原件 `0162:/mnt/persist/chensiyu/workspace/p2-k5-rhrd-20260810/`：codex `CONTRACT.md` + `k5_generate_source.py`（sha `88b6a372…` **未被改动**）；claude `CLAUDE-C32-PRIME.md`（sha `554dcc4d…`）、`CLAUDE-CODEGEN-REVIEW.md`（sha `9f3a57f5…`）、`claude_k5_generate_source.py`（sha `0f0fd995…`）、`claude_codegen_gate.sh`、campaign `codegen-gate-20260811-001734` | 2026-08-11 |
| K6a | AR 行数按**编译期** storage capacity 特化（`BATCH = STORAGE_BATCH_CAPACITY`） | P0 | ✅ 收益已实测 | codex（claude 独立读原始 diag + 拟合） | K1 | **实测 `14.44 µs/call` = `0.693 ms`（48 次），80× half_range，首个单项越过 14 µs 门。** 十臂 A1/A2 bracket：`43.26 / 43.62` → center `43.440`、half_range `0.180 µs (0.41%)`，且与 K1 独立 campaign 的 `43.18 µs` 一致 → harness 复现基线。scan A（batch=tile=N，一次 N 行传输）p50 `29.00/30.10/32.86/36.00/43.26` for N=`1/2/4/8/16`。**落地不需改 kernel**：`config.py` 的 `BATCH = STORAGE_BATCH_CAPACITY = env("PYPTO_STEP3P5_STORAGE_BATCH_CAPACITY", 16)`，只服务 bs=1 的部署编译成 capacity=1 即可；唯一机械改动 `BATCH_TILE = 16` 需变 `min(BATCH, 16)`（否则 `pl.slice(..., [BATCH_TILE, ...])` 在 BATCH=1 越界）。**代价 = 放弃并发多行 batching，是部署配置取舍、不是纯优化。****副产物结论**：现有 bs=1 整网数字（34.3 ms 等）均在 capacity=16 下测得 → 一直背着这 `0.693 ms` 可避免的 AR 成本。**限定**：bs=16 收益 `0.18 µs`（=1× half_range=噪声），bs=8 只 `7.44 µs/call = 0.357 ms` 而 bs=8 地板 `2.637 ms` 需 `54.9 µs/call` → **K6a 在 bs=8 门下无望，只能作低 batch 收益**，且必须实测 bs=8/16 不回退。原件 `0162:/mnt/persist/chensiyu/workspace/p2-k6-active-rows-20260810/run-20260810-233142/`，复核 `perf-2026q3/claude-verify-k6-scan-20260810/`（`VERDICT.md` sha256 `cc589dbb…`） | 2026-08-10 |
| K6b | AR payload 按 **runtime** `active_tokens` 裁剪 | P1 | ✅ codegen 门 PASS，可进产品实现 | codex（claude 逐 op 读生成 MLIR） | K6a | **无卡 codegen 门通过。** 判据是生成 MLIR 中 `partition_tensor_view` 的行维、**不是**编译成功。原件 `0162:.../p2-k6b-runtime-validshape-20260810/codegen-gate-store-20260810-235314/.../ptoas/tp_all_reduce.pto`，类型分布 `10 × <?x512xbf16>`（动态）/ `2 × <16x512xbf16>`（remote_load）/ `4 × <16x4096xbf16>`（TPUT）。逐 op：**✅ `pld.tile.remote_store`**（`:157 tstore ins(tile_buf<rows=16, v_row=?>) outs(?x512)`）动态、tile 仍静态分配 rows=16 → **机制确认**；**✅ 本地 `pl.load`/`pl.store` 也跟着动态**（`:93 tload ?x512`、`:135 tstore ?x512`、`:200/:203` final copy 读写皆 `?x512`）→ **final copy 的 4096 列也可缩，这是未预测到的额外收获**；**⚠ `pld.tile.remote_load` 仍静态 16 行**（`:123 tload ins(16x512)`，extent 取自 `shape=` 参数而非 valid_shape）→ 需上游 `valid_shape=` kwarg；**❌ TPUT 已在独立臂中测过并被 PTOAS 否证**（不是未测）—— 主臂只传了 `chunk_rows/chunk_cols` 故 partition view 是全量常量（`:48 sizes=[%c16_index, %c4096_index]`）；codex 另起 `k6b_generate_source_runtime_tput.py` 补齐 `shape=` + 两个 offsets 后，**pypto 正确 emit 了 `pto.comm.tput(... !pto.partition_tensor_view<?x4096xbf16> ...)`，但 PTOAS 在 `tp_all_reduce.pto:51:3` 硬拒**：`'pto.comm.tput' op expects dst to have a positive static shape`（原始日志 `codegen-gate-20260810-235230/host.log:1`）。**⇒ 阻塞在 PTOAS 而非 pypto**，构成第三条上游诉求。**账（列数 TPUT 4096 / load 3584 / store 3584 / copy 4096 = 15360，斜率 `0.9357 µs/row`）**：已确认可缩 = store `23.3%` + copy `26.7%` = **50.0% → `7.02 µs/call` = `0.337 ms`，不需任何上游改动**；TPUT `26.7%`（需 PTOAS 改）+ remote_load `23.3%`（需 pypto 改）→ 两条上游都修好才到 `100% → 14.04 µs/call`。**★ 内部一致性检查通过**：四项全通推出的 `14.04` vs K6a 实测 `14.44 µs/call`，**差 2.8%** —— K6a 是「编译期把所有 shape 改小」语义上等价于四项全缩，**两条独立推导对上，验证了按 bytes 归因的分配模型**（此前标注为「估计」的方法现有实证支撑）。**建议**：① K6b 直接进产品实现，不必再上卡验证（机制已在 MLIR 层确认、总量已由 K6a 实测、归因已交叉验证），上卡复核与 bundle 一起做；② 补一臂定 TPUT（`shape=` + `dst_offsets`/`src_offsets` 三者必须同时给，见 `tensor_ops.py:367`），仍无卡；③ **三条上游诉求**（前两条 pypto、第三条 PTOAS，同属「把 DMA/fence 语义与静态类型解耦」）：`pld.tile.remote_load` 缺 `valid_shape=` kwarg（解锁 `3.27 µs/call`）、`pld.system.notify` 缺 fence/release 参数（解锁 K2b 的 `2.39 µs/call`）、`pto.comm.tput` 拒绝 dynamic dst shape（解锁 `3.75 µs/call`）。④ **落地时的实现约束**（codex 实测踩到）：把 dynamic valid shape 贯穿整个 reduction 会触发 **loop-phi dominance bug**；正确结构是 own/remote load 与 reduce loop 保持静态，只在 publish 处新建 `reduced_tile_valid = set_validshape(reduced_tile_raw, active_rows, 512)`，并只对 final-copy 的 load/store 用 dynamic valid shape。报告 `0162:.../p2-k6b-runtime-validshape-20260810/K6B-CODEGEN-GATE.md` sha256 `03baffd3…` | 2026-08-11 |

**★ Track K 统一定律（2026-08-10，K6 scan 拟合 + K1/K2a 交叉验证）**

```
AR_cost ≈ 0.80 µs × N_remote_transactions  +  0.94 µs/row × active_rows
```

三条独立测量互证：

| 来源 | 量 | 值 |
|------|----|----|
| K1 notify law | 每次 remote credit store 边际成本 | `0.8010 µs` |
| K6 scan B | 每次 payload 传输边际成本（`10.6628 µs/round ÷ 14 传输/round`） | `0.7616 µs`（差 5%） |
| K6 scan A | 截距（row-count 无关项）实测 vs 交易计数预测 | `28.442` vs `35 × 0.8010 = 28.035`（差 `−0.407 µs` = **1.4%**） |

其中 `35 = 21 notify credit store（7 peer × 3 wave）+ 14 payload（7 remote_load + 7 remote_store）`。
scan A 最小二乘 `cost = 28.442 + 0.9357 × rows`，残差 `−0.38/−0.21/+0.67/+0.07/−0.15`。
回代：bs=1 → `28.9`（实测 `29.00`）、bs=16 → `43.0`（实测 `43.26/43.62`）。

**这条定律同时解释了 K1 的 `a≈0` 和 K2a 的实测否决：成本是 per-remote-transaction 的固定
发起/完成开销，与单次传输大小几乎无关。** 因此只有两个杠杆 —— **减少 remote 交易次数**
与 **减少行数** —— 且二者作用在不同项上、**可加**。这也是 K5（减交易）被从末位提到前列、
K2a（批 barrier）被否决的共同原因。

**定律的外推限制**：scan B 多轮臂 `latest_entry_is_last_exit_rate` 从 `0.953` 掉到
`0.625/0.625/0.594/0.531`，headline 统计在多轮臂条件变差（效应量 10~160 µs 远大于
half_range 0.18 µs 故方向可信，但 `10.66 µs/round` 系数带此保留）；且 scan B 的 t=16 与
scan A 的 N=16 用**同一个臂** `base16_a1`，两条 scan 在共享点上是恒等式、**未独立交叉验证**。


（`13 × 48 = 0.624 ms < 0.634 ms` 越不过历史地板）。上门前重算
`projected_gain = delta_per_call × on_path_call_count(48) × transmission_factor(0..1)`，
并与**当次** A/B/A 的 contemporaneous parent half-range 比较 —— `0.634 ms` 不是永久常数。
P2 绝对天花板 = 48 次 on-path AR × 43 µs ≈ `2.20 ms` = 整网 25.8 ms 的 **8.5~9.7%**
（不是 5 层 swimlane rank2 的局部 15.9%）。standalone probe 相对灵敏度 `0.14%` 约为整机门
`2.5%` 的 18 倍，故 **microbenchmark 为主证据、整机只做 no-regression**。

**没有任何单项能独立越过 14 µs 门** —— 此判断已被 K6a 推翻：K6a 实测 `14.44 µs/call`
（bs=1）是首个越门的单项。其余仍成立：K2a 实测 `0.060` 已否决、K2b 单臂 `2.39`、
K3 上界 `7.92`、K4 `5.51 + 2.18`、self-TPUT `3.92`、K6b 估计 `~3.7`。
K5-C 按定律外推预测 `26.8 µs/call`（bs=16 也有效）但需实测。

**天花板不可相加**：K2b / K4 都从同一个 `16.51 µs` 三轮 notify control 池取水。
但 **K6a（row 项）与 K5-C（transaction 项）作用在定律的不同项上、可加**。
两条路径的精度代价不对称：

| 路径 | 组成 | 预估 `µs/call` | 48 次 | FP32 求和顺序 | 精度准出 |
|------|------|---------------|-------|--------------|---------|
| **path1（保序）** | K6a `14.44`(bs=1) + K4 `5.06` + K2b `2.70` + K3 `7.92` | `30.12` | `1.446 ms` | rank0→7 canonical 保留 | hidden payload sha256 **byte-exact**（更便宜且更严，不占 cards 0-7） |
| **path2（RD/RS）** | K5-C `26.8`（外推）+ K6a 可叠加 | `26.8+` | `1.286 ms+` | 括号改变 | live vanilla vLLM W8A8 oracle，N=128 逐 token ≥95%（需整机 + oracle 容器） |

path1 现在**仅靠 K6a 一项就越门**（bs=1），故先走 path1；path2 的 K5-C 因在
**bs=16 也有效**（path1 在 bs=16 只剩 K4/K2b/K3 的 `15.68 µs = 0.753 ms`，
而 bs=16 地板需 `54.9 µs/call`）而仍需实测，作为高 batch 档的唯一希望。

---

## 进度汇总

| 状态 | 数量 |
|------|------|
| ⬜ TODO | 10 |
| 🟦 IN PROGRESS | 3 |
| ✅ DONE | 17 |
| ❌ NO-GO（实测否决） | 1 |
| ⛔ BLOCKED | 0 |
| **合计** | **31** |

**base 校正后关键路径**：A1/B1/B2/C1/C2/C3/C4/D1/D2/G1/H1/I1/I2/J2/K1/K6a/K6b 已 ✅；
historical pull C2 仅作回归基线。当前 performance 看板进行中的是
**B3（KV resident/in-place 的连续多轮 row-diff/liveness 证据）**、**J1（formal DFX /
publication / swimlane 收尾）**、**K2b（publisher release fence hoist，需上游 pypto 补丁）**；下一优先是
**K5-C（recursive halving/doubling，唯一在 bs=16 也有效的候选）**。
Attention/Vec 与 TP all-reduce stability 已在 0162 release-qualified；J1 产品实现
和六档 64K normal gate 已完成，但 formal DFX/publication/swimlane 尚未完成，且只在
0162 的 L0–L4 focused graph validated，不能升级为 whole-net release 结论。
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
| 2026-08-11 | K6b / K5-C | TPUT 被 PTOAS 否证（第三条上游诉求）；K5-C 变体集修正 + claude 补 C32′ | **TPUT 已测且被否证**（此前记「未测」）：codex 补齐 `shape=` + 两个 offsets 后 pypto 正确 emit `pto.comm.tput(... <?x4096xbf16> ...)`，但 **PTOAS 在 `tp_all_reduce.pto:51:3` 硬拒** `'pto.comm.tput' op expects dst to have a positive static shape` → **阻塞在 PTOAS 而非 pypto**。K6b 最终账：已确认可缩 `50.0% = 7.02 µs/call = 0.337 ms`（不需上游）；TPUT `26.7%` 需 PTOAS 改、remote_load `23.3%` 需 pypto 改。另记一条落地约束：dynamic valid shape 贯穿整个 reduction 会触发 **loop-phi dominance bug**，正确结构是只在 publish + final-copy 处用。**三条上游诉求**：`remote_load` 缺 `valid_shape=`、`notify` 缺 fence 参数、`pto.comm.tput` 拒 dynamic dst。**K5-C**：codex 修正了 claude 原 `26.8 µs/call` 预测的矛盾（224 KiB 与「仅 FP32 re-parenthesization」不能同时成立；原数用的是 BF16 payload = C16，含 3 次中间舍入）。claude 再补 **C32′（FP32 RS + BF16 AG）**：RS 结束时 owned shard 已是完整 FP32 和，cast 一次即与 parent 同一舍入契约，故 AG 只需搬 BF16 终值 —— `21,504 B/row` = parent 的 70.0%，预测 `20.1 µs`、**省 `23.4 µs/call` = `1.12 ms`**，**严格优于 codex 的 C32（字节少 25%、数值契约相同）**。三变体都远超 14 µs 门 ⇒ **筛选依据是精度代价而非门**。定律外推薄弱点：C32′ 的 12 交易分 6 个**串行** round，而 `0.80 µs/transaction` 拟合自 3-wave 与轮内可连发的拓扑，**未覆盖串行 round 间的握手延迟** —— 这是唯一可能翻车处，必须实测 |
| 2026-08-10 | K6a / K6b / K5 | K6 scan 出数：**统一定律成立**，K6a 成为首个越过 14 µs 门的单项；K5 从末位提到 P0 | **统一定律** `AR_cost ≈ 0.80 µs × N_remote_transactions + 0.94 µs/row × active_rows`，三条独立测量互证：K1 notify `0.8010 µs/store`、K6 scan B payload `0.7616 µs/transfer`（差 5%）、scan A 截距 `28.442` vs 交易计数预测 `35 × 0.8010 = 28.035`（差 **1.4%**）。回代 bs=1 `28.9`（实测 `29.00`）、bs=16 `43.0`（实测 `43.26/43.62`）。**K6a**：scan A p50 `29.00/30.10/32.86/36.00/43.26`(N=1/2/4/8/16)，bs=1 delta `14.44 µs/call` = 80× half_range = `0.693 ms` → **越门**；落地不需改 kernel（`STORAGE_BATCH_CAPACITY` 已是 env），只需 `BATCH_TILE = min(BATCH,16)`；代价是放弃并发 batching。副产物：现有 bs=1 整网数字一直背着这 `0.693 ms`。bs=8 只 `7.44 µs/call` 而门需 `54.9` → 低 batch 专项。**K6b**：`remote_load` 的 shape 决定 `pl.Tile` 类型 → 必须编译期常量，14 次 payload 传输动不了，估计仅 `~3.7 µs`；`set_validshape`（接受 runtime `Scalar[INDEX]`）是否被 DMA 遵守是决定性未知 → 一个臂定生死。**K5-C**：12 交易 vs 35、7168 列 vs 15360（流量也更少，原记的「加重流量」只适用变体 B），外推预测 bs=16 省 `26.8 µs/call = 1.29 ms` 且 **bs=16 也有效**（K6a 做不到）→ 若 K6b 死则为 P2 唯一能独立越门的候选。**已知不足**：scan B 多轮臂 `latest_entry_is_last_exit_rate` 掉到 `0.531~0.625`；scan B t=16 与 scan A N=16 用同一臂 `base16_a1`，共享点是恒等式、未独立交叉验证 |
| 2026-08-10 | K2a / K2b / K6 | K2a 实测否决；K2b 拿到 128 KiB 三臂单臂差值；K6 新立项并排到 K2b 之前 | **K2a NO-GO**：三臂 `4.46/4.44/4.46 µs`，half_range `0.000`，delta `+0.020 µs`；反解每 barrier `0.0033 µs`，18 个可摘 = `0.060 µs/call` = 48 次 `0.0029 ms`，比地板低两个数量级。这**证实了 K1 的 `a≈0`**：`0.786 µs/notify` 几乎全是 remote credit store，barrier 多重性无价值 → 唯一杠杆是**减少 store 次数** → K5/RD 复活（变体 C 21→6 store 且不加流量）、K6 立项。**K2b**：`notify_us` p50 A`16.17`→B`13.78`→C`13.56`，A→B `2.39 µs/call` 落在预估 `2.70~3.13` 内，C 只再省 `0.22` → 收益几乎全来自 fence 提升。两条限定：本轮无 A1/B/A2 bracketing 故只算单臂差值；claude 的 `transmission_factor>1` 假设 as measured 不成立（`post_minus_hot = cold_minus_hot = 0.0` 三臂全 0），但 `cold≈hot≈92.28 µs` 说明探针 GM-bound、可能结构上测不出 locality，null 一半属仪器。K2b 需上游 pypto 补丁（`pld.system.notify` 无 fence 参数）。**K6**：`attention_full.py:244 batch_padded = BATCH` 是静态别名 → bs=1 仍搬满 16 行（224 KiB 单向），而同文件已有 device-validated 的 `active_tokens` runtime 界可复用；纯 model 侧 + 保序，故优先级高于 K2b/K4。反证据需先排除：两次 43 µs 测量在 active rows 相差 16× 下取得 |
| 2026-08-10 | K1 | TP all-reduce 二次优化立项：critical-tail 口径确立 + notify/drain 分账，claude 独立复核 REPRODUCED | headline `critical_tail = max(rank_exit) − max(rank_entry)`，p50 `43.18 µs`；control `18.38` / data-compute `24.82`。**最大可寻址项是三轮 notify control `16.51 µs`**，不是 publish completion。pooled p50 `171.95 µs` 降级为 host 顺序提交 artifact。claude 补 `cost(n)=a+b·n`、a≈0 / b≈0.80 µs/notify → 决定性实验改为 barrier-vs-store 两点对比（K2）。`PTOAS fix/issue711-tnotify-mte-drain` 已在生产 pin 内，是 data-before-signal 正确性约束。原件 `phase0-split-20260810-190834/`，复核 `claude-verify-p2-phase0-20260810/` |
| 2026-08-10 | J2 | gate fan-out 与 norm/quant 解耦已发布，整网再拿约 6% | `pypto-lib stepfun/develop@d13b2ca6`（FF over `a31977fb`），`decode_fwd.py` SHA256 `28080c53…`。bs=1/64k/nb512 p50 `36.494 -> 33.849 ms`（+7.25%）、bs=8/64k/nb4096 p50 `97.528 -> 91.722 ms`（+5.95%）；两档三臂 hidden byte-exact。5 层 swimlane（bs=1、已发布代码）在 `perf-2026q3/swimlane-p1a-candidate-20260810-130154`，rank2 makespan `2.210 ms`、static CPM 81.7%、stall 19.5% 全 data-wait，`tp_all_reduce` 占 15.9%（8 次 on-path）。同轮三个 NO-GO 与两条新硬约束（UB per-kernel-per-core、`pl.pipeline` 可行性）见 [`../../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](../../benchmark/2026-08-10-step3p5-p1a-gate-decouple.md) |
| 2026-08-06 | J1 | 产品代码合入并完成六档独立 64K normal gate | `pypto-lib stepfun/develop@7928a275`（base=`56b3d477`），`decode_fwd.py` SHA256=`7884da7c…`；36/36 normal、correctness finalize、counterbalance PASS；六档 hidden bit-exact，p50 均 non-regression。formal DFX/publication/all-rank swimlane 待补，根目录 `/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-formal-act-n64-20260806-v1` |
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
