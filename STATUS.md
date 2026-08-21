# 实时状态（STATUS）

> **只放当前真相**：当前 phase、组件 pin、活跃 blocker、机器状态。
> 每日流水在 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)；
> 整体规划在 [`planning/roadmap.md`](planning/roadmap.md)；接力面在
> [`planning/handoff.md`](planning/handoff.md)；镜像组合在
> [`deployment/version-matrix.md`](deployment/version-matrix.md)。
> **最后更新：2026-08-21。**
>
> ⚠ **本文件 §1 起是 TP all-reduce / QKV 线（2026-08-12 口径，tip `69ad31e4`）。**
> 2026-08-15 ~ 08-21 的 MoE `moe-routed-packed-fusion` 线（R5 发布 + R6-R9 dispatch
> 融合 campaign）**未回写到下面各节**，其当前口径全部在此：
>
> - MoE 生产继续用 **R5**（`decode_fwd.py` sha `67b73589…`，ctx-64K BS1 p50
>   `27.757 ms` @ITERS=100 / `26.329 ms` @ITERS=1000，`hidden_sha256=567b206b…`，
>   tail token `14371`）。
> - **R6-R9 dispatch 融合线 = NO-GO**（概率性 liveness hang + 匹配曝光后也不快）。
>   曾写"机制已闭合、完整死锁环"，**已被对抗性复核撤回**；现口径 = 停点/分支/阻塞对象
>   已确立、闭合环未确立，仅存 **deadlock amplifier** 一条结构性主张。
> - 结构修复候选 `dispatch-orch-decouple-20260821`（`combine_scatter` 静态 grid，
>   sha `c5d87e25…`）：无卡 codegen 门过（`local_route_count` 上的 orchestrator
>   阻塞读 4→0），**device 门三臂全挂 ⇒ 该方向也 NO-GO**。
>   ★ 原因反直觉：**那个阻塞标量读同时是承重的 run-ahead 流控阀** —— 删掉它就
>   `orch_done=1`（一次提交完整张 1744-task 图）⇒ run-ahead 无界 ⇒ ring 饱和 ⇒
>   `HEAP_RING_DEADLOCK`。**不是容量账**（ring 已 4 GiB / 131072 slots，加容量只把
>   失败从 `inv=10` 推到 `inv=357`）。
> - ⇒ **2026-08-21 收盘：整条线关闭。** 只解析已有 STRACE span（不占卡）比较 R5
>   （有阻塞读）与候选 w1（无阻塞读）：p50 `orch` `17279.28 → 4443.18 µs`
>   （**−12836 µs / −74.3%**），而 `device_wall` `17466.93 → 17910.32 µs`（**+443 µs**）
>   ⇒ **orchestrator 与 device 并发、两种情况下都提前完成，从不在关键路径上
>   ⇒ 去掉阻塞读的 ITL ROI = 0（微负）。** 原计划的「本地节流」候选与「判 ring 耗尽
>   机制」一并关闭。留下的可复用产物 = 两条设计规则 + 一个可量化否决门，见
>   [`design/performance/07-hardware-scheduler-performance.md`](design/performance/07-hardware-scheduler-performance.md) §9。
> - ★★ **同一份数据给出量级更大的新线索**：R5 每 invocation `simpler_run` p50
>   `26.45 ms`（与 ITL p50 `26.329 ms` 对得上），其中 **`bind.args` = `6.12 ms`
>   ≈ ITL 的 23%**，纯 host 侧参数绑定、与 `runner_run` 加性（对照臂 `5.87 ms` 同量级）
>   ⇒ **比 dispatch 域任何 small-op 融合大一到两个数量级**，建议作为下一条性能主线。
>
> 详见 §9 `ORCH-SCALAR-READ-VS-CROSSRANK-WAIT`、[`blockers.md`](blockers.md) 与
> [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md) 的 2026-08-21 条。

## 0. Agent 判定当前状态的强制顺序

1. 读取本文件和 [`planning/handoff.md`](planning/handoff.md) 的日期与状态。
2. 用 GitHub 远端 `refs/heads/stepfun/develop` 核对 commit；**不得用本地同名分支、
   worktree 名称或历史 N1 文档推断当前 tip**。
3. 区分“当前源码 tip”和“最新 release-qualified 镜像”。源码前进不代表新镜像已准出。
4. 镜像只认 manifest digest、明确 pin 和 immutable gate；禁止借用旧镜像数据。
5. `develop/N1/`、旧 phase、旧 benchmark 和 hang-debug case study 都是历史证据，
   不能作为当前 checkout、构建 pin 或发布状态。

## 1. 当前源码与 0162 本地集成

> **2026-08-12 当前源码 tip：`69ad31e4`，TP all-reduce small-message selector 已合入。**
> GitHub `csy0225/pypto-lib:stepfun/develop` 与 0162 指定 checkout 均为
> `69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d`（parent `9ca01d2`），clean。
> Main 在 rank-uniform `active_rows == 1` 时走静态 8 KiB 两波 one-shot mesh；
> 其他行数和 MTP 走完整静态三波 reduce-scatter + push all-gather fallback。
> landing tree=`e26d762cb8c4abd49a1546e7db2beddeb6480e14`，`decode_fwd.py`
> SHA256=`a17ae27440a4ff0e62f7fe8b6dc2d5548217ef617b0ddbccb927fda648600d01`。
> Whole BS1/ctx64K A/B/A 为 `31.065 / 29.912 / 30.999 ms`，candidate 相对
> baseline center `31.032 ms` 改善 `-1.120 ms / -3.609%`，三臂
> precision/per-iteration gate PASS。
> 这是固定 immutable 镜像上的 **source-overlay validation**；尚未构建包含
> `69ad31e4` 的新镜像，TP all-reduce 最新完整 release-qualified 回退基线仍是
> Wave5。

> **2026-08-12 Attention I7 历史快照（已由后续 `9ca01d2` 与 `69ad31e4`
> supersede）**：当时 GitHub 与 0162 checkout 均为
> `e5e26f9f5bf9184f97a4684ae7e865f1a8b0d228` 且 clean。该提交在
> `fa58b5cf` 上给 RMS producer 开 early resolve，并把非关键 head-gate 从
> speculative fanout 隔离，优先预置 QKV。五层 8-rank 中 QKV Worker gap p50
> 从 `+4.77 us` 变为 `-1.78 us`（setup 已在 RMS 完成前开始）；RMS raw-kernel
> end → QKV raw-kernel start residual p50/max 从 `5.00/5.48 us` 降到
> `2.64/3.16 us`，RMS Worker span p50/max=`4.35/4.78 us`。L3/L4 exact。
> 整网 BS1/ctx64K A/B/A p50=`30.992/30.997/31.136 ms`，三臂 hidden/token
> exact，candidate=`WITHIN_BASELINE_BRACKET`。这是相对 `fa58b5cf` 的 I7 GO，
> **不覆盖下述 I6 相对 `f9065261` 的整体 NO-GO**；仍未构建新镜像。

> **历史 2026-08-12 I6 结论：源码已集成，但当时 packed QKV 整体性能验收
> NO-GO。** 0162 候选
> worktree `/mnt/persist/chensiyu/workspace/develop-worktrees/qkv-prerope-mix`
> 的分支 `perf/qkv-prerope-mix-20260811` 已基于
> `f906526190dc2eca0d479f8e9fa9187ec6d31be9` 提交为
> **`fa58b5cffe41b30d3f8d94482230867ee34b9e84`**，并 fast-forward push 到
> 当时的 `stepfun/develop`。该 I6 landing 时远端、0162 main checkout 与
> candidate worktree 三者均指向 `fa58b5cf` 且 clean；该 Attention 快照随后由 I7 前进到
> `e5e26f9f`。实现把 Full 的 10 个、SWA 的 14 个 Q/K/V
> projection blocks 各自改为 packed QKV projection，并接一个
> `qkv_split_qknorm_rope` mixed kernel，设备图收敛为
> `qkv_proj → qkv_split_qknorm_rope → attn_mix`。
>
> 2026-08-12 对最终 clean commit 做了 fresh post-merge 验证。整网
> BS1/ctx64K A/B/A 的**精度通过**：三臂 hidden SHA256 全等
> `567b206b…`、finite、tail token `14371` exact；但 candidate ITL p50
> `33.194 ms`，相对 baseline center `31.846 ms` **回退
> `+1.348 ms / +4.233%`**，判定 `REGRESSION_BEYOND_BRACKET`。同日重新采集
> 8-rank×5-layer swimlane，strict `<46 us` 只有 **39/40**：rank7/L0 Full
> 为 **`54.54 us`**。异常来自约 `12 us` AICPU scheduler dispatch stall，
> QKV/fused kernel 本体时长正常；它仍属于权威端到端 span，不能剔除。
> 因而 2026-08-11 的 40/40、max `43.60 us` 只保留为历史单次 capture，
> 不再作为当前准出结论。
>
> 该结论仍是固定 K8 immutable 镜像上的 **source-overlay gate**：镜像内
> `pypto-lib` 仍为 `cb96747e`，没有构建包含本候选的新镜像。canonical analyzer
> 仍因零本地 routed-token early-dispatch 缺记录返回 `rc=1`；独立 Attention
> inventory/dependency audit 通过，但 timing gate 与整网性能门均失败。
> 该段只记录当时 `e5e26f9f` 的 Attention 判定；当前 all-reduce landing 与镜像边界
> 以上方 `69ad31e4` 状态为准。

> **2026-08-11 I6 landing 状态（历史）**：当时 GitHub 远端
> `stepfun/develop` fast-forward 到 `fa58b5cf`；该历史链随后由 I7 前进到
> `e5e26f9f`，当前远端以 §1 的 `69ad31e4` 为准。固定 K8 镜像仍只包含 `cb96747e`；下述新代码是在该 immutable 镜像上通过
> `/candidate` **source overlay** 编译和验证，不能写成镜像已包含新实现。

```text
image/source baseline  cb96747eb21f5f4932d6a24eddaa69c85d095ef6
  -> 21d928b9e257f14aeb4b151cdcea720083f460d0  Attention mixed kernels
  -> f906526190dc2eca0d479f8e9fa9187ec6d31be9  SWA RMSNorm multicore
  -> fa58b5cffe41b30d3f8d94482230867ee34b9e84  packed QKV + pre-RoPE epilogue
  -> 18d1b5197acf4829b171bfa144eb06e5b0cacfdf  RMS producer prestage
  -> e5e26f9f5bf9184f97a4684ae7e865f1a8b0d228  prioritize critical QKV prestage
  -> 9ca01d243e534949287fa769e5be35031ebc4be7  align Full QKV dispatch
  -> 69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d  single-row all-reduce selector

GitHub stepfun/develop  69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
0162 local develop     69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
remote push            DONE
new immutable image    NOT BUILT
```

两项集成均已完成：

- `swa_moe_chip_orch_swa_rmsnorm_zc`：`BATCH=16` 时按每 task 2 行切成 8 个
  storage-capacity-row-derived logical tasks（非 active-token-derived），设备上
  每 rank 映射到 8 个不同物理核；block
  max `4.46 us`、logical-stage span max `4.90 us`，两个 strict `<5 us` 门均 PASS，
  L3/L4 byte-exact；
- Attention：Full 的 QK→mask/softmax→SV→segment recurrence 合入
  `full_attn_mix`，SWA 每 active row 使用一个 `swa_attn_mix`；旧 split kernel
  family 在设备图中为 0。combined focused gate、focused mixed-kernel 8-rank
  DFX 与最终 BS1/ctx64K A/B/A 均 PASS；candidate p50 `31.790 ms`，相对 baseline center
  `32.276 ms` 为 `-0.486 ms / -1.506%`，三臂 hidden byte-exact、token `14371`。

完整证据：
[`benchmark/2026-08-11-step3p5-attention-mix-rmsnorm.md`](benchmark/2026-08-11-step3p5-attention-mix-rmsnorm.md)。

> **2026-08-11 K8 immutable-image baseline（镜像真相；源码 tip/source-overlay
> 状态以上文为准）**：模型侧 7 个
> control buffer 提到 window 最前面 + runtime 只清那 `47,616 B`。整网 A/B/A 双 bracket
> 一致：ITL p50 `33.84 → 32.08 ms`（**−1.7455 ms / −5.16%**，89.5× 检测地板），
> `hidden_sha256` `567b206b…` byte-exact、token `14371`。

| 仓库/组件 | 分支或 pin | 当前 commit | 状态 |
|---|---|---|---|
| pypto-lib（GitHub） | `csy0225/pypto-lib:stepfun/develop` | `69ad31e4` | 远端 tip；single-row all-reduce selector |
| pypto-lib（0162 local） | `stepfun/develop` | `69ad31e4` | 指定 checkout clean，与远端对齐 |
| pypto | `csy0225/pypto:stepfun/develop` | `1c048a74` | 远端 tip；= `8e92b468` + reset 仪表 + K8 选择性清零（`distributed_runner.py` +174/−22） |
| simpler | immutable pin | `e2efebcbd190302609c0775d2984f409f5f42c76` | 当前 canonical image pin |
| pto-isa | immutable pin | `ecb6c303f797749f811a494742c3c08156aacabb` | 当前 canonical image pin |
| PTOAS | immutable pin | `fc8c6caee561914b4fb991dfc8427bb63194269e` | 当前 canonical image pin |
| ptoas-bin | release | `v0.50` | 当前 canonical image pin |
| vLLM overlay | immutable pin | `1b3e538c35999e62b6d24e0651b3a85b7d16c826` | 当前 canonical image pin |

> **已发布的三个性能优化（按时间倒序，细节在 benchmark/，流水在 archive/）**：
> ① **K8 选择性清零**（`cb96747e`/`1c048a74`）bs=1 ctx=64k p50 `33.84 → 32.08 ms`
>   （−5.16%），byte-exact，见 [`design/performance/task-tracking.md`](design/performance/task-tracking.md)
>   2026-08-11 行；
> ② **P1a gate 解耦**（`d13b2ca6`）bs=1 `36.494 → 33.849 ms`（+7.25%）、
>   bs=8 `97.528 → 91.722 ms`（+5.95%），两档 byte-exact（bs=1 三臂 sha =
>   `567b206b…` = N256 发布 golden、tail token 14371），见
>   [`benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](benchmark/2026-08-10-step3p5-p1a-gate-decouple.md)；
> ③ **MoE BS1 N256**（`a31977fb`）bs=1 p50 `35.778 → 34.271 ms`（4.21%），
>   三臂 byte-exact，精度 replay `123/128`、TP spread=0，见
>   [`benchmark/2026-08-10-step3p5-moe-n256-final.md`](benchmark/2026-08-10-step3p5-moe-n256-final.md)。
>
> ⚠ bs=1 用 `blocks=512`、bs=8 用 `blocks=4096`，编译期容量不同，**绝对值不可横比**；
> bs=16 ctx=64k 物理不可行（16 GiB 单次 `rtMalloc` → `207001`）。
> **生产 baseline 权威 `hidden_sha256` = `567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e`、
> tail token `14371`**；任何整网 A/B 都以它为精度门。

默认 Main 仍为：

```text
models.step3p5.decode_fwd:whole_decode_step3p5
```

> **2026-08-12 tp-all-reduce 最终状态**：旧 `a791071` ring 实验经审计是
> standalone-builder A/A，未命中 production 或 two-layer collective，结论撤回且不得合入。
> 最终 `69ad31e4` 采用 HCCL small-message selector 思路：单行 8 KiB 走静态两波
> one-shot mesh，其他 Main 行数及 MTP 走静态三波 fallback；ownership 固定
> `HIDDEN // TP_WORLD_SIZE`，与 transfer chunk 解耦。unit=`365 passed, 7 skipped`；
> Main/MTP default+chunk256 compile、8 卡 rows `1/3/16` device gate 均 PASS。
> focused historical K6b-vs-smallmesh regular-call kernel-duration pooled mean
> 为 `38.325 → 22.667 µs/call`（-40.9%，不是 strict critical-tail，也不代表
> 完整最终 source tree）；Whole A/B/A
> `31.065 / 29.912 / 30.999 ms`，candidate=`-1.120 ms / -3.609%`，三臂
> precision/per-iteration gate PASS。详见
> [`design/performance/03-tp-allreduce-algorithm-comparison.md`](design/performance/03-tp-allreduce-algorithm-comparison.md#8-hccl-small-message-selector-思路迁移版2026-08-12)
> 与 [`design/vllm-pypto/04-tp-allreduce-ring-refactor.md`](design/vllm-pypto/04-tp-allreduce-ring-refactor.md)。


## 2. 镜像与验证状态

### TP all-reduce small-message selector：source-overlay GO

`69ad31e4` 的 landing tree 与最终候选一致。单行 selector、静态 fallback、MTP ABI、
ownership/transfer-grain 解耦均已通过 unit/compile/device gate；有效 Whole A/B/A 的
三臂精度和逐迭代门全部通过。device rows `1/3/16` 未持性能锁，只证明功能；
`-1.120 ms / -3.609%` 的性能结论来自持全机及两半机三个锁的 Whole A/B/A。
`dense_mlp_body_tp` 在 `mlp_layer_idx` 后新增 `num_tokens` 实参；仓内 Main/MTP
调用点已更新，仓外直接调用或 inline 该 body 的代码升级时也必须同步更新。
five-layer 只声明 L3/L4 exact、finite、TP spread=0；既有 zero-token canonical
structural fail-closed 未被覆盖。当前仍缺包含该提交的 immutable image 和镜像级
release/structural qualification。权威 `ABA_RESULT.json` SHA256 为
`383caa23124c7da42d676ef642bc8b488344349564fd4131efa560c6b5ea3757`。

### RMSNorm → QKV critical prestage：I7 source-overlay GO

`allow_early_resolve` 是 producer 属性：QKV producer 原有 flag 优化
`QKV → split/QKNorm/RoPE`，不能反向优化 `RMS → QKV`。最终
`fa58b5cf → 18d1b519 → e5e26f9f` 在 RMS producer 开 early resolve，并用
已存在的 `swa_attn_out_zero` TaskId 把非关键 head-gate 保持在 normal dispatch，
让 14-slice packed QKV 优先预驻留。

```text
QKV Worker gap p50                    +4.77 -> -1.78 us
QKV raw-kernel residual min/p50/max    2.08 / 2.64 / 3.16 us
baseline raw-kernel residual p50/max   5.00 / 5.48 us
candidate RMS Worker span min/p50/max  4.16 / 4.35 / 4.78 us
L3/L4                                  byte-exact, finite, TP spread 0
whole A/B/A p50                        30.992 / 30.997 / 31.136 ms
whole verdict                          WITHIN_BASELINE_BRACKET
```

五层 all-ranks swimlane bundle：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  rms-proj-critical-prestage-validation-20260812-r3/delivery/
  ALL_RANKS_swimlane_bundle.tar.gz
sha256
  2f58af78bc2f1d8121426aca1e531fb76ada88072263a1ed97d5dfb7936cf083
```

外层 five-layer runner 与 whole A/B/A 均 rc=0；canonical analyzer 仍因既有
zero-local-route missing-swim record fail-closed。当前只声明 I7 相对
`fa58b5cf` 无回退并解决单独约 `5 us` 的 Worker 调度等待，不声明新镜像或
production release，也不覆盖 I6 NO-GO。完整记录：
[`benchmark/2026-08-12-step3p5-rms-qkv-dispatch-gap.md`](benchmark/2026-08-12-step3p5-rms-qkv-dispatch-gap.md)。

### QKV projection + split/QKNorm/RoPE fusion：post-merge 最终验证

候选保持独立 `wq/wk/wv` 权重 ABI，不改 loader/holder 接口；projection 的 packed
activation 为 `[Q|K|V]`。unit、compile 与 focused correctness 均通过，但
2026-08-12 对最终 clean commit 的 fresh 整网/DFX 结果推翻了此前的性能准出：

```text
unit                    362 passed, 7 skipped
whole compile           PASS, rc=0
focused correctness     PASS
whole precision         PASS, hidden SHA exact, token 14371
whole ITL A1/A2         31.787 / 31.905 ms
whole ITL candidate     33.194 ms
whole ITL delta         +1.348 ms / +4.233%  FAIL
fresh five-layer gate   39/40  FAIL
fresh global max span   54.54 us at rank7/L0
```

严格门口径固定为 merged swimlane **Worker View**：每个 rank/layer 从最早
`*_qkv_proj ts` 到最晚 `*_qkv_split_qknorm_rope ts+dur`。fresh post-merge
分层范围：

```text
L0 Full      41.46–54.54 us  7/8
L1 SWA       38.90–43.14 us  8/8
L2 SWA       38.66–40.02 us  8/8
L3 SWA-MoE   39.16–41.72 us  8/8
L4 Full-MoE  39.38–41.50 us  8/8
```

图 inventory 独立门仍通过：L0/L4 各 10 个 packed projection + 1 个 fused
epilogue，L1/L2/L3 各 14 + 1；旧 Q/K/V projection、`qk_norm`、`rope_q`/
`rope_kv` family 均为 0。rank7/L0 超门限不是 kernel compute 变慢，而是约
`12 us` 的 AICPU scheduler dispatch stall；权威 stage span 必须包含该尾延迟。

```text
five-layer:
/mnt/persist/chensiyu/workspace/perf-2026q3/
  qkv-prerope-postmerge-validation-20260811-r1/five_layer/analysis_final/
attention_gate_report.json
  0b5cbe2064663d179a509739e8c6ccd89777c839fcaca1023c4d1403c3a025a1
attention_gate_report.md
  f00149e36403e264018abd55fee4531672535a4b517dd43e5a052a78715c582e

whole A/B/A:
/mnt/persist/chensiyu/workspace/perf-2026q3/
  attn-mix-device-gate-20260811/out/aba-bs1-ctx64k-20260812-102231/
ABA_RESULT.json
  065f67c889a5eb108c49770261ccadf4d8f2970882b657efafb205ee35d6510b
```

fresh DFX 外层 runner `rc=0`、candidate container `rc=1`；后者仍含已知零
routed-token record 缺失。独立 analyzer 因 39/40 timing failure 返回 `rc=1`，
inventory/dependency/legacy audit 单独为 PASS。当前只能声明实现与精度正确，
**性能集成 NO-GO**；更不能声明 canonical structural 或 immutable-image release
qualification。

### Attention mix + SWA RMSNorm multicore source-overlay gate

本轮没有制作新镜像。设备 substrate 固定为下节 K8 immutable digest，镜像内
`pypto-lib` 仍是 `cb96747e`；候选 `f9065261` 通过只读 `/candidate` overlay
进入 compile/runtime，`pypto` runtime 未 overlay。

```text
combined unit       357 passed, 7 skipped
whole compile       PASS, num_blocks=512
RMS strict timing   block max 4.46 us; logical span max 4.90 us
focused seal        PASS
precision           PASS, all A/B/A hidden SHA identical
A/B/A verdict       IMPROVEMENT_BEYOND_BRACKET (-1.506%)
```

权威 seal/hashes：

```text
RMS target_metrics.json  b3d07bfc119529b037a77be7b57334a8903046e97bb4b58aadbc5c2830264180
combined pytest.log      09d0e695b3448e65f7ce361bdee24a4c26bc0f683d4149bc212a013d502a1f18
combined compile.log     fad61e08f2640761b8182810f805903e9da583037459e202802b69cea13ec700
focused clean seal       ff8cd797a5eb4a7ff41731c48fe3f10fc58bb5cfb8f8743b218506f2609d721d
ABA_RESULT.json          7eca25b23d3d944a841433a43f65cd5a4c829b9341984b5d58410388f08c4c80
```

该状态是 **source integration GO**，不是 immutable-image release qualification。

### 最终 commit 前五层 DFX swimlane（limited delivery）

`f9065261` 在同一 immutable 镜像上完成 L0–L4、BS1、ctx64K、8-rank
source-overlay DFX capture：

```text
DFX_CAPTURE                  PASS
PRECISION_GATE               PASS
MIXED_ATTENTION_INVENTORY    PASS
SWA_RMSNORM_MULTICORE_LT_5US PASS
CANONICAL_STRUCTURAL_GATE    FAIL_CLOSED
DELIVERY_STATUS              LIMITED_NOT_RELEASE_QUALIFIED
candidate container rc       1 (postprocess analyzer fail-closed)
```

L3/L4 对 baseline byte-exact、finite、TP spread=0。LOW-WAIT 参考为 rank2，
makespan `2.124 ms`；L3 SWA RMSNorm 为 8 tasks / 8 distinct cores。全 rank
最坏 RMS slice `4.28 us`、stage span `4.30 us`。Full L0/L4 各 24 mixed blocks，
SWA L1/L2/L3 各 1 mixed block，forbidden split family=0。

canonical structural analyzer 按 fail-closed 规则拒绝 rank0/1/3/6：这些 rank
各有 5 个零本地 routed-token 的 early-dispatch task 没有 AICore swim record。
因此这里只声明 capture 可交付，**不声明 structural PASS、cross-rank release seal
或 production qualification**。

```text
campaign:
/mnt/persist/chensiyu/workspace/perf-2026q3/
five-layer-dfx-combined-f906526-20260811-final-v4/

all ranks bundle SHA256:
d6f689c73b7ecb19b7febbf019a99baea4f96d59a778b37bfecacadbdc00def5

LOW-WAIT rank2 bundle SHA256:
e0bb2cc2beaa196b52547a04019d69720c0cb410b57b1c64524a476d09cd6d9a

delivery seal SHA256:
088cf05ffbff717fd6da9fcf443122da88c4c9373c41276e4f5ae8dbfa51eb94

delivery report JSON SHA256:
7bc5811da7cf543d3ddf812ee90e8297c3238ce0e7b24160899c19584fc29688
```

### K8 immutable image（2026-08-11 构建 + 0162 验证）

```text
tag:
hub.i.basemind.com/stepcast/vllm-pypto:stepfun-develop-20260811-k8-selective
manifest: sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3
config:   sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
spec:     deployment/docker/builds/stepfun-develop-20260811-k8-selective.env
```

这是**第一个包含 K8 发布基线** pypto-lib=`cb96747e` / pypto=`1c048a74` 的
immutable image；它不包含 §1 当前 pypto-lib tip `69ad31e4`。构建在 devbox
（0162 无 buildkitd、github/proxy 均不通，结构上不能构建），
**全部验证在 0162、digest-only、无源码/runtime overlay**。

镜像内 audit + smoke 全 PASS：`IMAGE_IMMUTABLE_AUDIT` / `CANONICAL_ONLY_SYMBOL_AUDIT`
/ `K8_LANDING_PRESENT` / `[smoke] PASS`；五仓 pin clean、credential scrub PASS、
attention profile=`a2a3`、`l2_swimlane_reuse_dep_gen` required=1 且 constructed、
ptoas `0.50`。**落地件 == 被测件**：`distributed_runner.py` sha256
`fe50c11fb76ec77789636de05e7376711c731d2b00db5033f0564c07a739622e` 与 K8 落地件
权威 sha 一致，`decode_fwd.py` = `eb1f89bf7add419f2382836c1eab9a1c4b1f63f738923d47e771e4159f104fb5`。

**两条独立精度证据都过（cards 0-7 / 8-15，digest-only）**：

- **byte-exact**：ctx=65536 bs=1 的 `hidden_sha256` =
  `567b206bb03d89f84020e1dddd61098a8f79f32f81b8f4fcf56443113e27f03e`
  == 生产 baseline（`matches_production_baseline_sha=true`），tail token `14371`
  exact、`hidden_finite=true`、shape `[8,16,4096]`；
- **N=128 逐 token（预定义冻结 oracle，三轮）**：`123/128 = 96.09375%`、
  miss `[2,8,13,22,82]`、`tp_spread_max=0.0`、`finite=true`，三轮完全一致，
  且**与 Wave5 那轮逐位相同** —— K8 没有改动 token 轨迹。oracle 是冻结的
  vanilla vLLM W8A8 greedy 采集（seed 6127、sha256 `c9b2c721…dd947`，与
  Wave3/Wave5 同一份），全程离线、无 live server。

**性能（clean、非插桩）**：ctx=65536 / bs=1 / blocks=512 / warmup=10 / iters=100，
p50 **`32.14 ms`**（min `31.766` / mean `32.467` / p99 `37.644`）。对 pre-K8
`33.84 ms` = **−1.70 ms / −5.02%**；与 K8 source-overlay A/B/A 候选臂 `32.08 ms`
差 `+0.06 ms`，远小于 bs=1 检测地板 `0.634 ms` ⇒ 镜像**复现**了 K8 收益。

**K8 runtime 生效证据**：109/109 步 `k8_prefix_applied=true`、
`k8_control_bytes=47616`（`range_count=1` 单段连续）、`k8_full_window_bytes=32063232`；
`reset_body_us` p50 **`523.1 µs`**（K8 A/B/A 实测 518 µs）。

完整记录：[`benchmark/2026-08-11-k8-selective-window-zeroing-image.md`](benchmark/2026-08-11-k8-selective-window-zeroing-image.md)。

⚠ **本镜像尚未重跑的项**（不得由 §1 的 K8 结论或本节数据代替）：Main batch16、
MTP batch1/batch16、六档（BS 1/2/4/7/8/16）每请求独立 64K golden/A/B、formal
matched-source DFX。因此它**还不能标成完整 production release-qualified**；
完整矩阵的回退基线仍是 Wave5。

下方其余 digest 都是旧源码层级的 **pre-fix evidence**，不能标成当前源码
的最终发布镜像，也不能把其 golden、性能或 DFX 自动升级为当前 tip 的准出结论。

### 最近一次 Attention canonical image（pre-fix evidence）

```text
tag:
hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260806-attn-taskmajor-canonical
manifest: sha256:3eb694e0455749b370c2da441f04badb47f2752edb53f2cf4e6acb1fde125479
config:   sha256:a6095ba550aa8207e66a10ad2e8923d120af957c9e014349d26915d7ba33d216
```

该镜像绑定 pypto-lib=`c9af5790`，**不包含** §1 的 SWA mask 修复
`63814d4a`。其 credential、五仓 pin、clean tree、CANN 8.5.1 absence、
prepared-swimlane `RunConfig` 和 A2A3
QK/softmax/online blocks-per-task=`22/16/22` profile 审计均 PASS。
0162 digest-only、无源码/runtime overlay 验证：

- 整网 BS1、每请求64K、warmup=5、50 次：
  min/mean/p50/p99/max =
  `39.057/39.594/39.612/40.680/40.680 ms`；hidden finite、TP spread=0；
- 前两层 BS1×64K：p50 `3.6323 ms`，reference exact、TP spread=0；
  DFX `8/8` rank 完整，LOW-WAIT 为 `rank2/d0`。

完整记录：
[`benchmark/2026-08-06-attention-taskmajor-canonical.md`](benchmark/2026-08-06-attention-taskmajor-canonical.md)。

该镜像只完成 `c9af5790` 层级的 Attention/ITL/DFX gate；SWA mask 修复后，
这些性能与 DFX 只能作 pre-fix 对照。它也未重跑 Wave5 的 Main N=128×3、
Main batch16 和 MTP 全矩阵，不能自动继承完整 production release-qualified 标签。

### L0–L4 MoE formal image（pre-fix evidence）

```text
hub.i.basemind.com/stepcast/vllm-pypto@sha256:
  cab89668164cf85dc75e4f3ac53ef77ef4b8653767c7d147c5113cdee6a9d88c
```

该 digest 绑定 pypto-lib=`c9af5790`、pypto=`8e92b468`、attention profile=`a2a3`
和 prepared-swimlane reuse capability。0162 的 focused normal A/B 已完成
baseline/candidate × BS `1/2/4/7/8/16` × 3 轮，共 36/36 fresh-process run；
每条 sequence 独立 `context_len=65536`。六档 L3/L4 hidden 跨轮 hash exact，
性能均无回退。seal：

```text
/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-formal-c9af-20260806-v2/
  campaign/normal_seal_authority.json
SHA256 875804ddbb81b4f15a907e41e454ed3004aca3b56075063431edef5efc70c531
```

该 campaign 在 `c9af5790` 上已 seal，但 SWA mask 随 `63814d4a` 发生源码变化，
因此旧 L3/L4 golden 与性能数据不能自动升级为最终 release evidence。统一发布
commit 确定后，六档每请求独立 64K、双 hidden golden 和 A/B 必须在最终镜像上重跑。

### 最新完整 release-qualified 回退基线（Wave5）

```text
hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260803-attn-final-wave5
manifest: sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32
config:   sha256:4f2539c17fe60e61062bd27d96082a707e581b81fe716208c1bca4139dfd7394
```

Wave5 只对 0162 完整 release-qualified；其源码 pin 是 pypto `defa97c5`、pypto-lib
`7099476b`，**不是当前源码 tip**。64K p50 `49.796 ms`，Main N=128 三轮均
`123/128` 且 TP spread=0。

### 历史 2026-08-05 R1/R2（已 supersede）

- R1 已撤销；R2 从未发布，且其 pypto-lib `91c7f46e` 已被后续
  `491267c4`、`f9065261`、`fa58b5cf` 和当前 `69ad31e4` supersede。不得恢复
  R2 或用其状态覆盖当前源码。
- 历史记录：
[`benchmark/2026-08-05-attention-canonical-r1-r2.md`](benchmark/2026-08-05-attention-canonical-r1-r2.md)。

## 3. Attention 当前判断

- `63814d4a` 将 SWA tail-window mask 从 `pl.cmp` predicate 转换路径改为显式
  typed INT32 数值区间 mask，避免 predicate 数值转换破坏 sliding-window
  score mask。
- 0162 使用 `cab896…` substrate + `63814d4a` 精确 source overlay 的 N=128
  teacher-forced 回归为 `127/128=99.21875%`，唯一 miss 是
  `step94 expected=478 actual=320`，`hidden_tp_spread_max=0.0`，已通过
  `>=95%` source-level 精度门。证据：
  `/mnt/persist/chensiyu/workspace/moe-opt/tmp/moe-precision-fix-20260807-v2/runs/Lmask-v1-n128/summary.json`
  （SHA256 `7f91dcdb…`）。
- source-level PASS 不等于 immutable-image PASS；最终镜像上的精度、性能和 DFX
  仍需重跑，当前不能宣称 SWA 修复无性能回退。
- Full/SWA 核心计算中主要可避免的调度 bubble 已闭环；logical task 按 workload
  和 architecture profile 推导，不固定 24 个物理核。
- Full/SWA RoPE producer 已改为 workload-sized 单次 SPMD submit，QK 显式依赖
  两个 producer TaskId；A2A3 blocks-per-task profile 为 `22/16/22`、
  reduce fan-in=8。
- Full Pass-A 已并入 SV；只保留必要的 online-softmax reduce/finalize。
- Full/SWA out-proj cast 均融合。
- 已证伪或无稳定收益的 AR+residual、residual+RMS、RMS+projection 等方案不合入。
- pre-fix focused 两层矩阵已覆盖 bs1/2/4/8/16/7、每请求64K；`c9af5790` 镜像
  两层 BS1 p50 `3.6323 ms`，输出 exact，DFX task count 为 `24/32/24`。
- pre-fix immutable 整网 BS1×64K p50 `39.612 ms`，相对 Wave5 下降 `20.45%`；
  该比较跨越最新 MoE 等整栈改动，不能把全部收益归因于 Attention。
- 整网 bs16×每请求64K 在 prewarm 前约 `52,013 MiB/卡` 的基础上申请约
  16 GiB static arena，`rtMalloc 207001`；没有有效 bs16 ITL。
- 后续优先级是完整 production matrix 与 BS16 容量门禁，其次才是跨架构
  profile 校准和可证明的 collective overlap。

设计入口：
[`design/performance/04-attention-optimization.md`](design/performance/04-attention-optimization.md)。

## 4. MoE 当前判断

- 产品改动 `7928a275`、`cd19fe6b` active-route scheduling 和 `491267c4`
  route/precision release harness 均为当前远端 `stepfun/develop@69ad31e4` 的祖先。
- 当前 tip 的 `decode_fwd.py`
  SHA256=`a17ae27440a4ff0e62f7fe8b6dc2d5548217ef617b0ddbccb927fda648600d01`。旧正式 campaign 的 candidate
  SHA256=`7884da7c…`、baseline `56b3d477` SHA256=`3553664c…` 只绑定历史
  source policy。
- `c9af5790` pre-fix 六档 focused normal A/B 与 L3/L4 hidden golden 已通过；
  p50 改善分别为 `9.16/1.83/3.52/6.07/0.53/11.61%`，但最终镜像必须重跑。
- matched-source whole-net baseline/candidate 的 1-step×2、2-step×2 共 8/8 run
  均通过，输出分别为 `303` 和 `303,1207`；publication seal=`PASS`：
  `.../whole-net-matched-ab-20260807T024525Z/publication_seal_report.json`
  （SHA256 `c0a03127…`）。
- J1 保持 🟦/NO-GO：source-overlay N=128 已通过，但当前源码 tip 对应的 final
  immutable image 精度、六档 64K golden/A/B、formal matched-source DFX 12 runs
  和 route-aware reanalysis 尚未完成；本轮 L0–L4 all-rank swimlane 仅为
  structural `FAIL_CLOSED` 的 limited delivery。

设计入口：
[`design/performance/05-moe-optimization.md`](design/performance/05-moe-optimization.md)。

## 5. 当前下一步

**TP all-reduce 代码与 source-overlay gate 已完成；该专题只剩 immutable-image
qualification。** 构建后需按 release 合同重跑 audit/smoke、Main+MTP compile、
Main precision/ITL 与必要的 active-batch matrix。以下 Attention/MoE 项保持各自专题顺序，
不得用 source-overlay all-reduce 数据替代镜像级准出。

1. **基于 `pypto-lib@69ad31e4` 构建 immutable candidate image**，固定
   manifest/config 与所有组件 pin；不得把 source-overlay 数据当作新镜像数据。
2. 在新镜像上重跑同口径 Whole A/B/A、Main/MTP compile、Main N=128、多 batch
   与 canonical structural analyzer；zero-token raw-swim 限制未解决时继续
   fail-closed。
3. 保留 single-row/multi-row selector、ownership/transfer-grain 解耦与
   canonical/two-layer AST exact 合同；不再恢复 `a791071` Ring 或 K6b
   dynamic-valid-shape 产品路径。
4. 在最终镜像上重跑 BS
   `1/2/4/7/8/16`、每请求独立 64K、L3/L4 golden 与 counterbalanced A/B。
5. 为最终 image/source 重新生成 matched source policy：current candidate 必须绑定
   最终 commit 的源码哈希，
   baseline 从选定的 immutable control source 独立计算；完成 MoE
   formal all-rank DFX/swimlane 和 fail-closed 重分析。不得把历史
   `baseline=3553664c`、`candidate=7884da7c` policy 直接沿用为当前准出。
6. 用 `pypto-image-verify` 与 `pypto-perf-regression` 对最终 immutable image
   执行标准回归。
7. 若提升为完整 production release，按 Wave5 同口径补 Main N=128×3、
   Main batch16、MTP batch1/16 和 smoke/precision matrix。
8. BS16×每请求64K 必须先通过 runtime-memory 容量门禁；不能把 OOM 或两层数据
   写成整网性能。
9. 新架构重新 sweep workload task grain；不能把 A2A3 blocks-per-task
   `22/16/22` 或物理核心数当作跨架构常量。

## 6. 其它项目级 active work

真实 vLLM live front、paged-KV/dynamic batch、同代 Main→MTP absolute gate 和
3-way HBM 仍未闭环；这些属于 serving 集成，不改变本轮 attention/R2 的准出顺序。
旧 N1 standalone、0234 stall 和早期 pin 只保留为历史案例，不再列为当前源码状态。

## 7. 机器状态口径

0162 是本轮验证机（driver `25.5.2` / firmware `7.8.0.7.220` /
CANN `9.0.0-beta.1`）。ITL/DFX 完成后 container 已退出，16 张卡均无 NPU
process；后续作业前仍须重新检查卡占用，不能沿用旧 session 的空闲结论。

## 8. 组件 Pin Snapshot（降序，最新在最上）

| 日期 | 事件 | pypto | pypto-lib | pto-isa | PTOAS(src) | simpler | ptoas-bin |
|------|------|-------|-----------|---------|-----------|---------|-----------|
| 2026-08-12 | **TP all-reduce small-message selector 已合入**：`9ca01d2 → 69ad31e4`。Main 单行 8 KiB 走静态两波 one-shot mesh，其他行数与 MTP 保留静态三波 fallback；ownership 与 transfer chunk 解耦。unit `365 passed, 7 skipped`；Main/MTP default+chunk256 compile、8 卡 rows `1/3/16` PASS。focused 历史 regular-call kernel-duration pooled mean `38.325 → 22.667 µs/call`（-40.9%，非 strict critical-tail）；Whole A/B/A `31.065/29.912/30.999 ms`，delta `-1.120 ms/-3.609%`，precision/per-iteration PASS。landing tree `e26d762c…`。仅 source-overlay validation，未构建新 immutable image | `1c048a74` | **`69ad31e4`** | 未移动 | 未移动 | 未移动 | 未移动 |
| 2026-08-12 | **RMS→QKV critical prestage 已合入**：`fa58b5cf → 18d1b519 → e5e26f9f`。QKV Worker gap p50 `+4.77 → -1.78 us`（setup 与 RMS 重叠），raw-kernel residual p50/max `5.00/5.48 → 2.64/3.16 us`；RMS Worker span max `4.78 us`。五层 L3/L4 exact；整网 A/B/A `30.992/30.997/31.136 ms`、precision PASS、`WITHIN_BASELINE_BRACKET`。远端与 0162 指定 checkout clean@`e5e26f9f`；all-ranks bundle SHA `2f58af78…`。仅为 source-overlay I7 GO，I6 NO-GO 与无新镜像边界不变 | `1c048a74` | **`e5e26f9f`** | 未移动 | 未移动 | 未移动 | 未移动 |
| 2026-08-12 | **`fa58b5cf` post-merge 性能验收 NO-GO**：整网 BS1/ctx64K A/B/A 精度 byte-exact PASS，但 ITL p50 `31.846 → 33.194 ms`，回退 `+1.348 ms / +4.233%`；fresh 五层 DFX strict `<46 us` 为 39/40，rank7/L0=`54.54 us`。inventory/dependency PASS；异常点为约 12 us AICPU scheduler dispatch stall，但属于端到端门。暂不构建 release image，先拆分定位 packed projection 与 fused epilogue | `1c048a74` | **`fa58b5cf`** | 未移动 | 未移动 | 未移动 | 未移动 |
| 2026-08-11 | **packed QKV projection + pre-RoPE epilogue 源码集成**：`fa58b5cf`（parent `f9065261`）已 fast-forward push；origin、0162 main/candidate 三者同 commit 且 clean。独立五层 strict Attention 门 40/40 PASS，max `43.60 us`、margin `2.40 us`。验证仍是下行 K8 immutable 镜像上的 source overlay；未构建新镜像，canonical analyzer known `rc=1` 不构成 release qualification | `1c048a74` | **`fa58b5cf`** | 未移动 | 未移动 | 未移动 | 未移动 |
| 2026-08-11 | **K8 发布到 `csy0225/{pypto,pypto-lib}:stepfun/develop`：WholeDecode persistent window 每请求只清 control 前缀，整网 `−1.7455 ms/step`（`−5.16%`）byte-exact**。每请求原本清整个 `32,063,232 B` retained window，其中只有 `47,616 B`（7 个 signal/arrived counter）真需归零；9 个 data buffer 每请求先写后读、无跨请求残留依赖（codex window 审计 + 实测 byte-exact 双重确认）。**落地形态不动 simpler**：carve 严格顺序无 padding ⇒ 模型把 7 个 control buffer 声明到最前面即构成唯一连续前缀 `[0, 47616)`，一次 `memset_all` 即可。pypto-lib `cb96747e`（`decode_fwd.py` 只重排声明顺序，+11/−7，alloc 数量与大小不变）；pypto `1c048a74` = 两个 commit（`35027daf` 加 `PYPTO_PERSISTENT_RESET_TRACE` 埋点 + `1c048a74` K8 前缀清零，共 +174/−22，单文件）。**两仓落地文件与被测件逐字节相同**（`decode_fwd.py` sha `eb1f89bf…`、`distributed_runner.py` sha `fe50c11f…`）。0162 dev0-7 A/B/A（ctx=65536、warmup 10/iters 100）：`A1 33.842 / A2 33.803` ⇒ floor `0.0195`；`B 32.077` ⇒ **`−1.7455 ms`、89.5× floor**；reset host wall `2253 → 518 µs`。三臂 `hidden_sha256` 全等生产 baseline `567b206b…`、`token 14371`。同效应在另一独立 bracket 复现 `−1.7505 ms`（差 0.005 ms）。作用范围 fail-closed：只在 base-name 多重集精确匹配 WholeDecode 16 buffer 时启用，其它 persistent 程序（five-layer/two-layer/multi-program）保持全窗清零；control 出现在 data 之后 / 字节数偏离 pinned / carve 不符 ⇒ raise；trace `k8_prefix_applied` 供发布门识别静默回退。反向结论：先前拆成 6 次 `memset_all` 是 pessimization（`+1.886 ms`）—— 这条路径上多一次阻塞 broadcast 很贵、字节收益只有约 1:1。campaign `0162:.../k8-selective-20260811/v3-20260811-144622`，`K8_V3_RESULT.json` sha256 `7bb02263…`。详见 [`design/performance/task-tracking.md`](design/performance/task-tracking.md) | **`1c048a74`** | **`cb96747e`** | 未移动 | 未移动 | 未移动 | 未移动 |
| 2026-08-10 | **P1a gate 解耦发布到 `csy0225/pypto-lib:stepfun/develop`（`a31977fb` → `d13b2ca6`，单 commit FF）**：`gate_expert_fanout` 只存 raw FP32 logits，`inv_rms/sigmoid/bias` 尾巴搬进 `gate_topk`，解除对 `norm_quant_moe_input` 的串行。只改 `decode_fwd.py`（+63/-35），SHA `d392311c… -> 28080c53…`，与 A/B/A 的 candidate 臂逐字节相同故设备数据直接绑定发布代码。codegen 实证 `params_t70` 不再 `add_input(moe_inv_rms)`，task 数与 `block_num=9` 不变。0162 A/B/A：bs=1/64k/nb512 `36.494 -> 33.849 ms`（p50 +7.25%、min +4.87%）；bs=8/64k/nb4096 `97.528 -> 91.722 ms`（p50 +5.95%、min +6.19%）；bs=16/64k 物理不可行。**两个 batch 三臂 hidden 全 byte-exact**（bs=1 = N256 golden `567b206b…`、tail token 14371；bs=8 `1fcd4fcc…`）。5 层 swimlane（bs=1、发布代码）rank2 makespan `2.210 ms`、static CPM 81.7%、stall 19.5% 全 data-wait、`tp_all_reduce` 15.9%。NO-GO：`gate_up+act`（改判 ROI，非能力）、`act+h_quant`（grid 维度）、`tp_all_reduce` 降 ring step（前提未证实）。新硬约束：UB `188416 B` 是 per-kernel-per-core；`pl.pipeline(stage=2)` 需先缩 tile。见 [`benchmark/2026-08-10-step3p5-p1a-gate-decouple.md`](benchmark/2026-08-10-step3p5-p1a-gate-decouple.md) | 未移动 | **`d13b2ca6`** | 未移动 | 未移动 | 未移动 | 未移动 |
| 2026-08-10 | **MoE BS1 N256 发布到 `csy0225/pypto-lib:stepfun/develop`**：merge 远端 release harness `491267c4` 与已验证 candidate `7d3e02ae`；产品 `decode_fwd.py` SHA 保持 `d392311c…`。0162 BS1/64K A/B/A mean `36.354 -> 35.055 ms`（3.57%）、p50 `35.778 -> 34.271 ms`（4.21%）；precision replay `123/128`、128/128 spread=0；candidate compile-only PASS，merge tree pytest 30/30 + ruff PASS。`down24` NO-GO | 未移动 | `a31977fb` | 未移动 | 未移动 | 未移动 | 未移动 |
| 2026-08-03 | **Wave5 canonical release（0162）**：source partial 改 self-target TPUT，再走既有三波 reduce-scatter + push all-gather。manifest `sha256:4acc77cd…`、config `sha256:4f2539c1…`；immutable audit/smoke/Main+MTP compile、Main N=128×3、Main batch16、MTP batch1/batch16×2、64K/batch16 ITL/DFX 全 PASS。N=128 三轮均 `123/128` 且 spread=0；64K p50 `49.796 ms` | `defa97c5` | `7099476b` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | v0.50 |
| 2026-08-03 | **Wave4 historical immutable candidate**：canonical TP all-reduce 增第三 completion wave；two-layer harness 与 canonical AST 对齐。manifest `sha256:8125c678…`、config `sha256:c340001f…`；audit/smoke/compile/64K ITL/DFX PASS，p50 `50.204 ms`。N=128 Run1 `122/128` 但 step2 spread=`2.0`，Run2 `123/128` spread=0；raw token gate PASS，未通过当时 TP-spread stability gate，已由 Wave5 取代 | `defa97c5` | `d7e1381b` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | v0.50 |
| 2026-08-02 | **Attention/Vec 历史 clean candidate**：workload-derived tasks、Full/SWA cast 与 Vec 收口。manifest `sha256:64c573bc…`；64K p50 `50.563 ms`；N=128 三轮均 `121/128`，由后续 Wave3/4 lifetime 修复取代 | `defa97c5` | `76d96bdb` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | v0.50 |
| 2026-07-29 | **PERF-H1 自包含镜像 build + 回归**：`vllm-pypto:stepfun-develop-20260729-perf-h1`（registry digest `sha256:b4e8c8a457a5…`）。在 C4 发布镜像上前进 pypto→`1f704616`（gitlink→simpler `e2efebcb`，device-memset 清零）、pypto-lib→`4513007d`（`cfbdcce8` + ITL `--active-batch` + **MTP CI oracle-dir 可配置化**，去掉镜像外 username 硬路径）。0162 回归：smoke PASS；整网 CI `ok=true`（Main 8 步 token `303,1207,19384,872,428,6127,4231,2636` exact + MTP single/batch16 token `6178,410,303` exact，`hidden_tp_spread=0`）；**N=256 H1 vs C4 发布镜像 token 256/256 exact**（含 step127/128/255），全步 finite（raw-hidden run-to-run 抖动 ~34–44 = C4 push all-reduce 归约顺序，非 H1 回归，经 H1a-vs-H1b 复跑证实）。**ITL p50（`--num-blocks 512`）：1024 `50.9` / 8192 `52.0` / 32768 `58.0` / 65536 `64.1` ms —— 较 C4 同工作点降 23–27%**。MTP CI 挂掉真因=`_run_mtp` 用本次 Main hidden 当输入却比 0718 配对 golden（喂配对输入 pass_rate=1.0），wiring 修复 `0f3650c7`（test-only，mount 验证未 rebuild）。见 [`benchmark/2026-07-29-perf-h1-image-itl-dfx.md`](benchmark/2026-07-29-perf-h1-image-itl-dfx.md) | `1f704616` | `4513007d` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | v0.50 |
| 2026-07-29 | PERF-H1 host/device 分账 + retained window 清零改 device `aclrtMemset`。首次把 ITL 拆成 host vs device：85 ms 里只有 ~55 ms 是 device，`_reset_persistent_domains` 独占 21.5 ms（每步 248 次串行阻塞 mailbox 往返 + 244.7 MiB H2D）。改走 backend 给 fresh window 用的同一条 device memset（新增 `_CTRL_MEMSET`，`broadcast_control_all` 8 卡并行）：清零 `21.50→2.21 ms`、ITL p50 `85.02→65.55 ms`（−22.9%）、每步 H2D 归零；同镜像 A/B `main_hidden_only_report.json` 除 `run_sec` 外逐字段相同，单测 8/8。sim 平台仍走原 host 路径。⚠ live N=128 精度门未跑。证伪：`persistent=False` 更慢 3.25×（ITL 276.2 ms）。另立 H2（起跑阶梯 2.914 ms，v4-flash 同形状）/ H3（DFX 假长条，曾使 `tp_all_reduce` 被误判 74.1% wall）。见 [`benchmark/2026-07-29-host-window-memset.md`](benchmark/2026-07-29-host-window-memset.md) | `1f704616` | `cfbdcce8` | `ecb6c303` | `fc8c6cae` | `e2efebcb` | v0.50 |
| 2026-07-29 | PERF-C4 TP all-reduce → reduce-scatter + **push** all-gather 发布。根因=pull-after-remote-notify 跨方向握手无序（postmortems/13），修正 design/performance/03 §5。pypto-lib `cfbdcce8` 已推 `stepfun/develop`；simpler span-aware child provenance 入库为 `8459d60f`，并由 pypto `6933b1aa` 的 runtime gitlink 固定（postmortems/14）。镜像 `vllm-pypto:stepfun-develop-20260729-allreduce-push`（digest `sha256:7924925f…`）已推 registry：audit/smoke/整网 CI PASS，`hidden_tp_spread` 32 步全 `0.0`，ITL p50 65.942/66.455 ms | `6933b1aa` | `cfbdcce8` | `ecb6c303` | `fc8c6cae` | `8459d60f` | v0.50 |
| 2026-07-28 | C/D/G + BS1 收口；`stepfun/develop@563fe62a`；固定 expert physical lanes；最终自包含镜像 smoke/Main 8-step PASS，N=256 hidden finite/TP spread=0、token exact 241/256 | `ca21ab5f` | `563fe62a` | `ecb6c303` | `fc8c6cae` | `216e7632` | v0.50 |
| 2026-07-24 | PERF-A1 逐层 DFX 接线（holder N1_DFX→swim/pmu + harness `--dfx`）；在镜像 20260724(cards 8-15) 采集 → routed-expert 占 90.7% / PMU cube_int8 88.6%（benchmark/2026-07-24-perlayer-dfx） | `ca21ab5f` | `bc5eecb1`(+DFX over `fd26b1be`) | `ecb6c303` | `fc8c6cae` | `216e7632` | v0.50 |
| 2026-07-24 | 合并 origin/main 到 stepfun/develop（全 FF，保留 fork ITL harness `7cb2a6b3`）+ IPC 权重 interior 指针 provenance 修复（解 `submit_next_level child_memory` 卡点）；镜像 `vllm-pypto:stepfun-develop-20260724`(ptoas v0.50) 冒烟 PASS + 整网 8 步 `6127→303→1207→6127` 与 live vanilla 逐 token 一致 | `ca21ab5f` | `fd26b1be` | `ecb6c303` | `fc8c6cae` | `216e7632` | v0.50 |
| 2026-07-26 | B2 loop-form Main canonical-only 收口；`stepfun/develop@53eb7212`；删除兼容 package/alias 后与清理前镜像 N=256 token/hidden `256/256` exact | `ca21ab5f` | `53eb7212` | `ecb6c303` | `fc8c6cae` | `216e7632` | v0.50 |
| 2026-07-23 | decode-ITL profiling harness(hidden-only via holder;64k ≈654ms/step,含 host glue;权威基线见 benchmark/ device-KV 590ms raw rt.run;均计算受限、近平坦)| `8af501fc` | `7cb2a6b3`(+ITL mode over `4c48215b`) | `ecb6c303` | `72ada0a1` | `36957c6b` | v0.45 |
| 2026-07-23 | simpler develop 回退到可编译 36957c6b（c7fdc574 Phase-24 import_ipc 半成品编译不过, 存 tag backup/stepfun-develop-c7fdc574-20260723）+ pypto develop gitlink 同步 | `8af501fc`(9ec303f6+gitlink→36957c6b) | `4c48215b` | `ecb6c303` | `72ada0a1` | `36957c6b`(develop 回退; 0162 验证过的 .so 就是它) | v0.45 |
| 2026-07-23 | 五仓 stepfun/develop 对齐验证过的 N=1 pin（pto-isa/PTOAS FF-push 到 fork stepfun/develop）+ 可复现 Docker 镜像 | `9ec303f6` | `4c48215b` | `ecb6c303`(FF `e25732f0`→,+111) | `72ada0a1`(FF `da011a3d`→,+307) | `c7fdc574` | v0.45 |
| 2026-07-18 | N=1 single-submit 合入三仓 `stepfun/develop` + 干净回归 20/20 | `9ec303f6` | `e1513d22` | `ecb6c303` | `72ada0a1` | `c7fdc574` | v0.45(见 stable SSOT) |
| 2026-07-17 | N=1 stable env freeze（SSOT `develop/N1/N1-STABLE-ENV-0162-20260717.md`） | `n1fusion-base:e277de9f` | `feat/whole-net-n1-fusion:0e7a0fdd` | `ecb6c303` | `72ada0a1` | `n1fusion-base:36957c6b` | v0.45 |
| 2026-06-22 | Phase 2 设计落地；建项目跟踪仓 | `stepfun/develop:b00c8b23` | `stepfun/develop:b918e60` | `e25732f0` | `da011a3d` | `a6e06406` | v0.45 |

> 更早的 pin 历史在 [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)。

## 9. 当前 Blocker / Deferred（摘要，详见 [`blockers.md`](blockers.md)）

| # | Blocker | 严重度 | gate 什么 | 详情 |
|--:|---------|--------|-----------|------|
| ORCH-SCALAR-READ-VS-CROSSRANK-WAIT<br>（原 DISPATCH-FUSION-S1） | **已确立**：orchestration 级阻塞标量读（动态 spmd grid 定尺 `scatter_blocks = pl.read(local_route_count,[1])`，`decode_fwd.py:2611` → AICPU `get_tensor_data`）落在 `dispatch_meta` 的输出上，而 `dispatch_meta` 体内跨卡等 `meta_arrived` ⇒ **deadlock amplifier**。**❌ 已撤回**：「闭合死锁环」与「1 元凶 + 7 受害者」（提交序 627 < 933；8 个 AICPU pid 全部阻塞在同一 producer）。hang 是**概率性**的（R9 生产配置 3 次挂 2 次；R5 同配置 ITERS=1000 一轮长跑 PASS）⇒ 单次跑通不构成 liveness 门 | 🟡 已定案（负结论）；耦合在 R5 里仍潜伏但不可获利地修 | R6-R9 dispatch 融合线（**NO-GO**，MoE 生产继续 R5）；结构修复候选 device 门三臂全挂（那个阻塞读是承重 run-ahead 节流，删掉即 `orch_done=1` ⇒ ring 死锁；**不是容量账**）。⇒ **2026-08-21 收盘整条线关闭** —— 用已有 STRACE span 量化（不占卡）：p50 `orch` `17279 → 4443 µs`（−74.3%）而 `device_wall` `17467 → 17910`（+443）⇒ **orchestrator 从不在关键路径上，ROI = 0（微负）** ⇒「本地节流」与「判 ring 耗尽机制」一并关闭。留下 = 两条设计规则 + 一个可量化否决门。★★ 同数据新线索：`bind.args` `6.12 ms` ≈ **ITL 的 23%**（host 侧参数绑定），量级远大于本线 | [`blockers.md`](blockers.md) + `0162:…/dispatch-orch-decouple-20260821/FINDINGS.md` |
| UPSTREAM-NOTIFY-FENCE | pypto `MakeNotifyCodegenPTO` 把 `dcci`(invalidate-only) 排在 payload drain 之前；最小修复 = 一条 pre-CMO `pipe_barrier(PIPE_ALL)`（device 已证，消融矩阵闭合），Wave2 单点代价 `0.405 µs/call` | 🔴 Active / correctness | 一切「把 payload store 与它自己的 credit 拉近」的 AR 优化（删波次 / 合并波次 / 按 peer 融合） | [`blockers.md`](blockers.md) |
| N1-S-0234 | 0234 同步 pypto-lib 后 whole-net stall（完整对象未确认） | 🔴 Active / 未独立复核 | 取得 SSH 后核对三仓/runtime/环境重跑 canonical | [`blockers.md`](blockers.md) |
| N1-L | Phase 28 live：per-layer KV + 3-way HBM + live token-exact A/B | 🔴 Active | live single-handoff | [`planning/phases/28-live-integration.md`](planning/phases/28-live-integration.md) |
| 1 | Phase 20 production backend 未接入 | 🟡 功能 | 真实 vLLM 请求走 PyPTO runner | [`design/vllm-pypto/`](design/vllm-pypto/) |
| 2 | Prefill MoE L1 overflow（TASK-29） | 🟡 功能/性能 | 真实 PyPTO NPU prefill kernel | [`blockers.md`](blockers.md) |
| 3 | head_gate 语义（历史 ×1 旁路已由 on-device gate 取代） | 🟡 精度 | 在线 backend L1 parity | [`postmortems/09-attention-multiposition-corruption.md`](postmortems/09-attention-multiposition-corruption.md) |
| 5 | MTP 集成进 decode | 🟢 Deferred | speculative 吞吐 | [`blockers.md`](blockers.md) |
