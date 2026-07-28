# 任务跟踪记录 — Performance 专项

> step3p5 decode 性能优化的**单一事实源**。每个子任务的状态、owner、更新时间、阻塞在此维护。
> 设计详情见 [`02-detailed-design.md`](02-detailed-design.md)；改状态时同步更新本表的「最后更新」列 + 底部更新日志。
>
> **状态图例**：⬜ TODO ｜ 🟦 IN PROGRESS ｜ ✅ DONE ｜ ⛔ BLOCKED ｜ ⏸ PAUSED

---

> **⚠ 2026-07-28 release override**：active base =
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
| A1 | whole-net baseline + DFX 采集 | P0 | ✅ | claude | — | 新增多步 report/per-layer hidden/TP spread/finite 检查；逐层 DFX：routed-expert 占 90.7%（PMU cube_int8 88.6%）；见 benchmark/2026-07-24 | 2026-07-26 |

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

---

## 进度汇总

| 状态 | 数量 |
|------|------|
| ⬜ TODO | 4 |
| 🟦 IN PROGRESS | 2 |
| ✅ DONE | 8 |
| ⛔ BLOCKED | 0 |
| **合计** | **14** |

**base 校正后关键路径**：A1/B1/B2/C1/C2/C3/D1/D2/G1 已 ✅；historical pull C2 仅作回归基线。当前剩余：
**B3（KV resident/in-place 的连续多轮 row-diff/liveness 证据）与镜像发布后的远端同步**。
C/D/G 产品修复与最终镜像 Main 验证已收口；N=256 raw vanilla 仍按历史口径记录为 241/256 teacher-forced exact，不宣称 raw 95% PASS。

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
| 2026-07-27 | C/D/G 约束清理 | canonical 注释改为 V4-Flash-style shared EP window + epoch lineage；512B 仅 stacked/reused control slot 的 step3p5 backend/profile 隔离，不是通用 DeepSeek signal ABI；未改 TP all-reduce、未恢复 pull | 旧 pull 只作历史回归基线；按 source/compile/lowered/device/runtime DAG 分级，不因 probe 编译越级 DONE |


### 2026-07-28 C/D/G 收口与镜像验证

- `b404a3c9`：恢复固定 expert-lane physical bases（`expert_recv_max = n_ranks * BATCH`），修复 BS1 batch-extension invariance；BS1/2/16 单步 token `6127→303`、TP spread `0`，BS1 持续 4-step `6127→303→1207→19384→872`，row0 hidden 与 BS2/BS16 bit-identical。
- `563fe62a`：镜像 CI 清理忽略 zombie-only process group，并增加 `--skip-mtp`；MTP oracle 缺失单独记录为外部依赖，不归因于 C/D/G。
- 最终镜像 `hub.i.basemind.com/stepcast/vllm-pypto:step3p5-b404a3c9-ci-final-20260728`：smoke PASS，Main 8-step `303,1207,19384,872,428,6127,4231,2636` exact，hidden 全 finite、8 rank active rows nonzero、TP spread `0`。
- 镜像内 N=256 teacher-forced 回归：`256/256` hidden finite、`256/256` TP spread `0`，token exact `241/256`；该结果沿用历史 live vanilla oracle 序列，raw 95% gate 不宣称通过。
- 本地 `stepfun/develop` 已快进到 `563fe62a`；代码已推送至 `csy0225/pypto-lib` 的 `stepfun/develop` 与 `perf/step3p5-bc-20260726`。

### 2026-07-27 顶层方向收口

本轮复核发现，之前存在“局部 shape/layout 差异 → step3p5 独有架构”的误判。后续所有任务必须先完成端到端同构审计：

```text
producer → 数学变换/quant/route-map → transport/window
→ consumer → rounding/reduction/placement → lifetime/reuse/allocator
```

因此：INT8 activation+scale、resident/InOut KV、shared window+epoch、owner max/active-token、fixed storage+dynamic valid rows、expert-lane fan-out、LM-head data-window reuse 均优先直接沿用 V4-Flash；只有 vLLM/IPC/allocator、45层 flat layout、step3p5 shape、canonical whole-net 和当前 backend/profile 才能作为局部适配。C2/C3直接迁移 V4-Flash push/gather/scatter/reduce；route weight在expert W2 epilogue完成缩放，combine只做FP32 token reduction。默认16只是容量实例，不是逻辑batch或永久padding/KV reserve合同。
