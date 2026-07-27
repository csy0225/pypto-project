# 02 — Detailed Design (LLD)：各优化子任务

> 每个子任务一张卡片：问题（file:line 实证）/ **shape（step3p5 具体输入输出）** / **如何生效（用 shape 讲清搬了多少、算了多少、省了什么）** / 参考 / 改法 / 验证 / 落地边界。
> HLD 见 [`01-system-design.md`](01-system-design.md)，状态见 [`task-tracking.md`](task-tracking.md)。
>
> **2026-07-27 active release override**：`P/` 当前必须读
> `pypto-lib stepfun/develop@53eb7212`。唯一 Main 是
> `P/models/step3p5/decode_fwd.py:whole_decode_step3p5`。0724 unroll source、
> rollback selector、自定义 Main module/name 参数和旧 opt compatibility
> package/aliases 已删除。B2 已完成 N=256 replacement regression；
> 下方基于旧 unroll/generator 的内容只用于解释设计起点，不是可执行 base。
>
> 路径约定：`P/` = active `pypto-lib/`；`REF/` =
> `origin/main:models/deepseek/v4-flash/`（`git show REF/<f>` 读取）。
>
> 2026-07-24 的 unroll/generator file:line 已全部退休；当前实现位置只读
> `decode_fwd.py`、`dense_mlp.py`、`whole_decode_holder.py` 及对应合同测试。
> 对账：**A1/C2/B1/SwiGLU-per-layer/B2 已交付**，B/C 剩余 P0 是 **C1**。

---

## 0. step3p5 关键 shape 速查（TP=EP=8，per-rank）

| 量 | 全量 | per-rank（TP/EP 切分后） | 说明 |
|----|------|--------------------------|------|
| 残差 / hidden | `[BATCH=16, HIDDEN=4096]` BF16 | 同（TP 不切 batch/hidden） | decode step 的 residual stream；**仅 row0 有效，row1..15 padding** |
| full-attn q | `NUM_HEADS_FULL=64 → 8192` | `wq_full [4096, 1024]`，q `[16, 1024]` | 每 rank 8 头 (`NUM_HEADS_FULL_LOCAL=8`)，`HIDDEN_Q_FULL_LOCAL=1024` |
| swa-attn q | `NUM_HEADS_SWA=96 → 12288` | `wq_swa [4096, 1536]`，q `[16, 1536]` | 每 rank 12 头，`HIDDEN_Q_SWA_LOCAL=1536` |
| KV | `NUM_KV_HEADS=8 → 1024` | `wk/wv [4096, 128]`，k/v `[16, 128]` | 每 rank **1 KV 头**（`KV_HEADS_LOCAL=1`，`KV_HIDDEN_LOCAL=128`） |
| dense MLP | `INTERMEDIATE=11264` | `w_gate/up [4096, 1408]`，`w_down [1408, 4096]` | `INTERMEDIATE_LOCAL=1408`；3 个 dense 层 |
| MoE routed | `MOE_NUM_EXPERTS=288`，`MOE_INTERMEDIATE=1280` | 每 rank **36 专家**，每专家 `w1/w3 [4096,1280]`、`w2 [1280,4096]` | `MOE_NUM_EXPERTS_LOCAL=36`，`TOP_K=8`；42 个 MoE 层 |
| MoE shared | `SHARE_EXPERT_DIM=1280` | `w_gate/up_s [4096, 160]`，`w_down_s [160, 4096]` | `SHARE_EXPERT_DIM_LOCAL=160` |
| LM head | `VOCAB=128896` | `lm_head_weight [16112, 4096]`，logits shard `[16, 16112]` | `VOCAB_LOCAL=16112` |
| MoE comm 窗口 | — | `recv_x [LOCAL_RECV_MAX=1024, 4096]` BF16 ≈ 8MB，`routed_y [BATCH*TOP_K=128, 4096]` ≈ 1MB | 每层一套 ≈ 十几 MB；42 层 ≈ **766MB** |
| KV cache | — | `k_cache/v_cache [KV_CACHE_ROWS_DYN, 128]` BF16 | 45 层沿 leading 轴堆叠；paged `BLOCK_SIZE=128` |

派生：**MoE 权重/rank/层** BF16 = `36×4096×1280×2B ×3(gate/up/down) ≈ 1.13GB/层`；`×42 ≈ 47.6GB`（= 现状 IPC pool）。INT8 减半 → `≈24GB`。

## 0.1 PERF-B/C/G 落地前差异表（2026-07-26）

> 对照对象：current `models/step3p5/decode_fwd.py@53eb7212` 与
> `origin/main:models/deepseek/v4-flash/{decode_fwd.py,moe.py}`。本表是
> B3/C1/C3/G1 动手前的“差异 + 理由 + 改还是留”决策，避免机械照搬 v4-flash。

| 项 | v4-flash | step3p5 current | 差异理由 | 决策 |
|----|----------|-----------------|----------|------|
| B3 KV 所有权 | `CACHE_POOL_NAMES` + `RESIDENT_CACHE_OUTPUT_NAMES`，KV/state tensor 是 resident InOut | canonical `whole_chip_orch` / `host_orch` 已为 `pl.InOut`；holder 通过一次 `import_kv_all()` + `build_stacked_kv_pool()` 绑定 vLLM-owned IPC K/V section | step3p5 是 45 层 consolidated flat `[45*rows,128]`，v4-flash 是模型自己的多 pool ABI，不能只靠 reshape 互换 | **留 current ABI，补强合同与设备验证**：确认 prepare/import 只做一次、run 不 copy 整池、每层只按 `slot_mapping` 写一行 |
| C1 数据窗口 | host 只分配一套 `recv_* / arrived / data_arrived / routed_y / combine_arrived`；push 协议按 `moe_epoch` 单调累计 | 42 套 `pub_counts/send/recv/routed_src/routed_y`；pull dispatch/combine；signal 每层独立且固定 `expected=1` | current C2 的 pull 让 `send_*` / `routed_src` 由生产 rank 持有；若只做一波 `expected=epoch`，快 rank 可在慢 rank 完成 remote read 前覆盖下一层数据 | **改为单套 pull-safe 双波 epoch**：每层 ready 阈值 `2*epoch-1`，read-complete 阈值 `2*epoch`，均 `AtomicAdd + WaitCmp.Ge`；数据窗口只在第二波全 rank 完成后复用 |
| C1 TP all-reduce 窗口 | v4-flash 的 MoE 窗口不等于 step3p5 attention/shared TP all-reduce scratch | MoE 层同时带 attention/shared 的 `tmp+signal`，其 barrier 固定使用 `expected=1/2` | 这两类窗口不是 EP dispatch/combine 协议；直接压成一套会被上一层残留计数提前放行 | **留 per-layer**：本次只折叠 EP dispatch/combine 的 12 类窗口；attention/shared 的 4 类 scratch 继续逐层隔离 |
| C1 512B signal stride | DeepSeek control signal 仍是 `[N_RANKS,1] INT32` + `N_RANKS*4`；其大量 512B 用于 data tile/L2/MTE | step3p5 canonical 曾因 layer/slot stacked signal 邻接 cache-line 重叠发生 false sharing | compiler/runtime 可能只记录 compact byte-span/provenance，无法隔离相邻 layer/epoch slot | **本地限定约束**：仅 stacked/reused 且参与 notify/wait/AtomicAdd 的 control slot 用 512B physical stride，formal/window/slice `[128,1]`，逻辑 loop 只访问前 `n_ranks`；普通 data/MTP 独立 signal 不扩容 |
| C3 dispatch fan-out | dispatch payload 用 `pl.spmd(N_LOCAL)`，peer 元数据循环仍有顺序部分 | pull 的 `pub_counts` 快照、peer payload remote-load、barrier notify/wait 仍是 `pl.range(8)` | fixed-slot 地址虽然按 peer 分离，但当前 task ABI、signal ownership 和 join 尚未证明 non-aliasing | **暂留顺序**：先把 per-peer task 提升到 orchestration/SPMD 边界并建立 explicit join；禁止在 InCore 内直接改 `pl.parallel` |
| C3 combine fan-out | push combine 用 `pl.spmd(N_LOCAL)`，按 expert lane 并发 | current C2 为 inverse-map pull，主循环是 runtime `(token, topk)` | weighted gather 对共享 `moe_out[token]` 有累加语义，不能只凭 route 唯一就声明 write-disjoint | **暂留顺序**：设计独立 route-stage 输出与 join/reduction 后再并行；当前 C3 未完成 |
| G1 active-token ABI | host 传 `num_tokens_per_owner [N_RANKS]`，whole graph 取 `max`；MoE 的 route/dispatch/combine 使用 runtime `num_tokens` | holder 已知 `valid_tokens`，但 canonical program 只看固定 `BATCH=16`，padding rows 仍进入 gate/routing/固定槽 pull | 生产 sidecar 的 owner 数由 scheduler 决定；用 owner max 可保证 8 rank 动态边界一致，又不改变固定 storage tensor shape | **改**：canonical host ABI 增加 `[8] INT32` resident tensor；holder 每 step 写 `valid_tokens`；gate top-k、histogram、pack、inverse-map、fixed-slot pull、combine 仅处理 active rows |
| G1 调度轴 | gate/routed expert 以 experts/intermediate/feature 为主要 core fan-out；token 轴是 runtime sequential bound | routed expert 已 `pl.parallel(36)` + intermediate `pl.spmd`，但 gate 仍按静态 `[16,*]` 单块执行，combine 以 batch fan-out | decode 常见 T=1；专家/特征轴稳定且更宽，适合作为设备并行轴；动态 batch fan-out 曾触发 UB/lifetime 编译差异 | **改**：gate expert-column chunk 用 `pl.spmd(288/32=9)`；combine 保持 runtime token 顺序、TOPK=8 write-disjoint fan-out；保留 routed expert 36×feature fan-out |
| G1 attention storage/compute | v4-flash attention 的 token 循环是动态/顺序，模型 KV ABI不依赖 step3p5 的 15 个 padding reserve slot | step3p5 attention 用固定 BATCH=16 承载 compiler-validated QKV/FA tiles，并为 15 个 padding rows 写 allocator-owned reserve slots；历史 dynamic `pl.parallel(user_batch)` 有 `VEC UB not aligned` 风险 | 直接改成 dynamic parallel 会同时破坏固定 tile shape、KV reserve 隔离和已验证 UB lifetime，不可作为无证据替换 | **分层处理**：本批保留固定 16-row storage/KV write 契约；先交付 MoE active-token 和 expert/feature schedule。attention 的 score/context padding 消除仅在镜像 compile+batch 1/2/8/16 DFX 证明安全后并入，否则 G1 保持 🟦并单独收口 |

---

## Track A — 可观测性 & Baseline

### PERF-A1 · whole-net decode baseline + DFX 采集
- **问题**：whole-net 无 perf 数据 = 盲调。`docs/step3p5`、`docs/performance-tuning.md` 无延迟数据。
- **shape**：不改数据流；被测是 canonical `whole_decode_step3p5`（输入 hidden `[16,4096]`/rank → 输出 pre-final-norm hidden `[16,4096]`/rank）。
- **如何生效**：不优化，只建 baseline。采 `l2_swimlane`（每个 kernel 的起止 + 层间 gap）、`pmu.csv`（cube/vec/mte 利用率）、`perf_hints.log`（MTE 非 512B 对齐点）、`memory_after_AllocateMemoryAddr.txt`（各 buffer 占用峰值），把 45 层每层耗时拆出来 → 定位真正的热层/热 kernel，后续每项拿它回归。
- **改法**（不改模型）：`P/tools/step3p5/whole_decode_holder.py:280` 已有 `enable_scope_stats`；加 `--enable-l2-swimlane`(0/1/2)、`--enable-pmu` 透传 `rt.run`。参考 `P/docs/performance-tuning.md:12,139,247,263`。跑多步 decode，落 `docs/step3p5/perf-baseline.md`（新建）。
- **验证**：产出分层耗时表 + 4 个 DFX 工件路径。
- **边界**：零代码风险。**先做**——解锁 F2 与所有定量对比。

---

## Track B — Mega-kernel 结构

### PERF-B1 · resident 权重池 + leading-dim zero-copy view ABI ✅ 已交付
- **问题**：0724 baseline 的权重已经通过 consolidated IPC pool 常驻，并由
  `build_stacked_weight()` 跨 rank 绑定；current B1 真正缺口不是“消除每步
  全量 H2D”，而是 opt 需要从 canonical FULL/SWA 栈拿到连续的 MoE-only
  bucket，且不能为此复制一套设备权重。
- **shape**：MoE routed 权重 stacked 后 per-rank `moe_w_gate_r [8, 42, 36, 4096, 1280]`（`[N_RANKS, LAYER, N_LOCAL=36, HIDDEN, INTER]`）；attention `wq_full [8, 42, 4096, 1024]`；KV pool `k_cache [8, 45*KV_CACHE_ROWS, 128]`。层内取 `pl.slice(moe_w_gate_r[r], [36,4096,1280], [layer_idx*36, 0, 0])`。
- **如何生效**：沿用原有“一次 import、跨 step resident”的 pool；
  `Wsub()` 对每 rank 的 canonical `DeviceTensor` 做 outermost contiguous
  slice（FULL slots `1:11`、SWA slots `2:32`），再构造成
  `StackedDeviceTensor`。B2 的 dynamic `pl.slice(layer_idx)` 因而直接指向
  原 pool 地址，不创建 opt 专用权重副本。
- **参考**：`REF/decode_fwd.py:162-176`、`:1443-1450`、`REF/moe.py:928-945`。
- **改法**：`P/models/step3p5/weight_loader.py` 按层型堆 `[N_RANKS, L*dim, tail]`；每权重 spec + KV pool 打 `resident="stacked"`；层内消费改 `pl.slice`。
- **改造前 → 当前实现**：
  - 前：一次 IPC import、跨 step resident 已经存在；但只有 canonical
    FULL `[12,...]` / SWA `[33,...]` 桶，opt 的 10/30 层 MoE attention bucket
    没有可直接绑定的连续 ABI。
  - 后：`Wsub()` 复用 canonical pool，FULL 取 `1:11`、SWA `2:32`；
    每 rank 的子视图与跨 rank stacking 都是 zero-copy，dynamic `pl.slice`
    再按 `layer_idx` 取当前层。
- **收益**：为 B2 提供必要的 leading-dim ABI，同时避免单独 materialize
  10 个 FULL + 30 个 SWA attention bucket。按当前 BF16 shape
  （FULL 约 `18.125 MiB/layer`、SWA 约 `26.125 MiB/layer`）推导，
  避免约 `965 MiB≈0.94 GiB/rank` 的额外设备副本。该数字是 shape 容量
  对比，不是 H2D trace 或 latency A/B；0724 baseline 原本就没有
  `24 GiB/rank/step` 的重复权重 H2D，不能把它写成本次收益。
- **验证**：dynamic-offset `pl.slice` device probe 已通过；current N=256
  replacement token/hidden `256/256` exact，证明 layout/offset 改造未引入数值回归。
- **边界**：`resident=` IR 属性本身没有被当前 codegen 消费，收益来自 IPC
  resident + stacked sub-view，不应把属性名当成独立优化。

### PERF-B2 · 45 层 unroll → 单 `pl.range` 循环 ✅ 已交付
- **问题**：`P/tools/step3p5/_gen_faithful_real.py:1273-1326`（L0/L1/L2 直排）+ `:1337-1428`（42 MoE 逐层 emit）→ historical `decode_layer.py` 约 31,686 行、98 个 `*_chip_orch`。
- **shape**：循环体每层消费 hidden `[16,4096]` → 输出 hidden `[16,4096]`；层内从 stacked 权重（B1）`pl.slice` 出当层 shard（如 MoE `[36,4096,1280]`）；layer_idx 为 dynamic scalar。
- **如何生效**：unroll 在 Python/DSL 层重复描述各层的调用、切片和依赖；
  折成 `pl.range` 后由一个 loop body 加 dynamic `layer_idx` 表达重复层，
  hidden `[16,4096]` 的逐层串接和 per-layer communication stack 保持不变。
  这直接减少源码/IR 描述重复，但在没有同环境 compiler trace 前，不把它
  换算成“AICPU 调度边 ÷45”或编译耗时比例。
- **参考**：`REF/decode_fwd.py:404-565`（`pl.range(HCA_NUM_LAYERS)` 循环体）。
- **改法**：重写 `_gen_faithful_real.py::_host_orch`（`:1159`），按层型分桶（full-moe / swa-moe / dense）各一个 `pl.range`；首/尾特殊层保留显式。
- **改造前 → 当前实现**：
  - 前：历史 `3af13f4f` faithful whole-net `models/step3p5/decode_layer.py`
    为 `31,686` 行，45 层结构按 layer site 展开，MoE 主体重复 40 份。
  - 后：canonical `models/step3p5/decode_fwd.py` 为 `4,772` 行；L0 显式，L1/L2
    进入 `pl.range(2)`，L3-L42 进入 `pl.range(40)`，L43/L44 保留为必要的
    activation specialization；动态 scalar 通过 `pl.slice` 选择当前层权重。
- **收益**：主体源码体量约 **84.94% 减少**（`31,686→4,772`）；
  MoE 主体 specialization 从 `40→1` 个 runtime loop body，减少重复 IR/
  调度描述，并让编译器跨层复用 loop body。当前没有旧 31k implementation
  与 loop-form implementation 在同一环境重新编译的 wall-clock 对比，不能写成“编译耗时下降 X%”。
- **精度/稳定性验证**：0162 发布镜像内 N=256，canonical-only 清理前后
  token/hidden `256/256` exact，删除 package/alias 前后也为 `256/256` bit-exact，
  `max_abs_diff=0`，TP spread `0.0`；canonical
  step127/128/255 通过，0162 无 8-15 device residual process。
- **raw 边界**：canonical-only 发布镜像对同一 vanilla oracle 为
  `240/256=93.75%`，低于历史 raw `>=95%` gate；清理前 canonical 镜像完全复现，
  所以 raw 差异不是 B2 或兼容入口清理引入，但不能写成 vanilla raw PASS。
- **边界**：当前 B2 采用 per-layer window stack，**不包含 C1 单 window/
  `moe_epoch`**；C1 仍是独立后续优化。

### PERF-B3 · KV pool `resident` + in-place
- **问题**：KV 每 dispatch 可能重传（`P/models/step3p5/attention_full.py:183,211,218` consolidated multi-layer ABI）。
- **shape**：`k_cache/v_cache [KV_CACHE_ROWS_DYN, 128]` BF16 per rank（45 层堆叠）；每 step 只写当前 token 的 1 行 `[1,128]`（`slot_mapping` 定位），读窗口 `[ctx_len,128]`。
- **如何生效**：KV pool 大（数百 MB～GB/rank）。若每 step D2H/H2D 整池，带宽全浪费在没变的历史 KV 上。`resident` + InOut 让池**常驻、原地写**：每 step 仅 MTE 写 1 行 + 读有效窗口，省掉整池往返。
- **参考**：`REF/decode_fwd.py:151-176`（`CACHE_POOL_NAMES` + `RESIDENT_CACHE_OUTPUT_NAMES`）。
- **改法**：KV 归 `CACHE_POOL_NAMES`，`resident="stacked"` + InOut，kernel 原地写。
- **验证**：静态合同确认 canonical entry 是 `pl.InOut`、holder 的 KV IPC
  import/build 只发生在 `__enter__` 且 `run()` 无整池 `copy_`；设备多步 decode
  验证 KV 连续性 + L3 parity，并对相邻 step 的 IPC pool 做 row-diff，除
  `slot_mapping` 指向的每层一行外不得改写历史行。
- **边界**：随 B1；注意 vLLM per-layer pool vs step3p5 consolidated ABI 差异（memory `g5b_kv_bridge_not_pure_reshape`）。

---

## Track C — MoE 通信协议

### PERF-C1 · 单 window set + `moe_epoch` + `WaitCmp.Ge`（关键路径）🟦 落地中
- **问题**：`P/tools/step3p5/_gen_faithful_real.py:1342-1357` 每 MoE 层 16 个 `_L{pos}` 窗口 → 42 层 ≈ 672 窗口 / **≈766MB comm domain**。根因：RAW-only-v1 非别名（ADR-013），窗口无法跨层复用。
- **shape**：一套窗口 = `recv_meta [8,36]` + `recv_x [1024,4096]`(8MB) + `recv_aux [1024,AUX]` + `recv_route [1024,IDX]` + `arrived/data_arrived/combine_arrived [8,1]` + `routed_y [128,4096]`(1MB)。现状 42 套并存；目标 1 套。
- **如何生效**：现状把每层的 `recv_x [1024,4096]` 等都独立开一份 → 766MB 常驻、编译期窗口记账爆。current C2 已是 pull，故不能直接照搬 v4-flash 的单波 push epoch。改成**1 套 pull-safe 双波复用**：每层 MoE 调用传单调 `moe_epoch`（1→42）；生产端 pack/stage 完成后 `AtomicAdd`，ready wait 用 `WaitCmp.Ge(expected=2*epoch-1)`；所有 rank 完成 remote read 后再 `AtomicAdd`，completion wait 用 `WaitCmp.Ge(expected=2*epoch)`。第二波保证下一层覆盖 `send_* / routed_src` 前，上一层所有远端读都已结束。`recv_* / routed_y` 是本 rank 自有消费窗口，按本地程序序安全复用。
- **参考**：`REF/moe.py:120-175`（`dispatch_meta` notify `arrived`）、`:178-235`（`dispatch_push` notify `data_arrived`）、`:238-248`（anchored `dispatch_wait`）、`REF/decode_fwd.py:758-769`（一次性 8 窗口）、`:377,402,495,564,654`（`moe_epoch` 递增）。
- **改法**：host 侧把 EP dispatch/combine 的
  `pub_counts/count_done/recv_x/recv_scale/data_done/recv_route/send_x/send_scale/send_route/routed_y/combine_done/routed_src`
  折成一套；attention/shared TP all-reduce 的 `tmp+signal` 仍逐层保留。
  每次 MoE 传 `moe_epoch`；两波 wait 均用 `Ge`（禁 `Set/Eq`）；
  dispatch/combine wait 分别用 `expert_indices` / `inverse_map` read anchor。
- **512B signal stride 口径**：
  - DeepSeek v4 control signal 仍是 `[N_RANKS,1] INT32`，512B 主要对应
    data tile、L2 cache line 和 MTE 性能对齐，不能解释为通用 window ABI；
  - canonical 中 layer/slot stacked signal，以及跨 `moe_epoch` 复用的
    `count_done/combine_done` slot，使用
    `COMM_CONTROL_SIGNAL_BYTES=512`、`COMM_SIGNAL_STRIDE_I32=128`；
  - formal/window/slice shape 使用 `[128,1]`，通信 loop 只访问前
    `n_ranks` 行；
  - `recv_x/send_x/pub_counts/routed_src` 等普通 data window 按真实 payload
    分配；MTP 三个独立 signal 保持 `tp_size*4` compact allocation。
- **验证**：6 轮 RUN_CLEAN 稳定（liveness，`_probe_barrier_scale.py`）+ 多步 L3 精度不回退。
- **边界**：这是原目标架构中 B2 的前置假设；current B2 已通过继续使用
  per-layer communication stack 绕开该依赖。C1 现在是独立的窗口/HBM
  优化，不能因其未完成而回退 B2 的完成状态。

> **当前状态澄清**：上述 `766MB→十几 MB` 是 C1 设计目标，不是当前
> `53eb7212` release 的收益。当前 B2 仍使用 `NUM_MOE_LAYERS_TOTAL` leading
> offset 的 per-layer communication stacks。

### PERF-C2 · dispatch push → pull（fixed-slot）✅ 已交付
- **问题**：`P/models/step3p5/dispatch.py:252-310` push `remote_store` scatter = A2 随机 507018 stall（跨 die 写完成竞争，memory `n1_a2_primitive_exists_not_missing`）。
- **shape**：每 token 选 `TOP_K=8` 专家 → 每 rank 最多发 `BATCH*TOP_K=128` 条 `[1,4096]` 路由；接收端 `recv_x [1024,4096]`（`LOCAL_RECV_MAX=1024 = 8 rank × 128`）。
- **如何生效**：push 让每个源 rank 主动 `remote_store` 到目标 rank 的 `recv_x` 槽 → 跨 die 写完成顺序不确定，随机 stall。改 **fixed-slot pull**：目标 rank 按固定槽公式 `my_rank*MAX`/`peer*MAX` 主动拉，写完成由本地掌控 → 消除跨 die 写竞争。数据量不变（同样 128 条 `[1,4096]`/rank），只换发起方。
- **参考**：`REF/moe.py` dispatch + device-validated `P/models/step3p5/moe.py` 的 `ep_all_to_all` fixed-slot pull。
- **改法**（memory `n1_pull_dispatch_must_align_moepy_fixedslot`）：pack fixed-slot → AtomicAdd barrier → static `pl.range(T*TOPK)` a2a → LOCAL re-pack；**combine 保持 push**（combine push = jitter 非 stall）。
- **改造前 → 当前实现**：
  - 前：dispatch 由源 rank 对目标 rank 做 `remote_store`/push，写完成跨 die
    可见性和竞争顺序不稳定；combine 也存在 push 回写路径。
  - 后：dispatch 改成 fixed-slot peer-major layout，由目标 rank 按对称槽位
    `remote_load`/pull；combine 采用固定槽位 pull-back；本地 self-bucket
    与 peer pull 分开，完成顺序由 consumer 本地控制。
- **收益**：消除 push dispatch 的随机 stall/`507018` 风险路径，提升
  liveness 可重复性；通信字节数不变，不能宣称带宽或理论通信量下降。
- **验证**：当前 N=256 256-step regression 无 stall，hidden finite 全部为真，
  TP spread `0.0`；历史 device canonical pull path 已通过。精度与 B2
  replacement gate 同时保持 exact。
- **边界**：C2 不等于 C1；当前仍是 per-layer windows，C1 epoch/window
  reuse 需单独实现和回归。

### PERF-C3 · peer loop `pl.range(N_RANKS)` → `pl.spmd`/`pl.parallel` 🟦 落地中
- **问题**：`P/models/step3p5/dispatch.py:159,181,210,286,297,322` + `combine.py:182,209,216,225` 全顺序 barrier。
- **shape**：peer 循环 `pl.range(N_RANKS=8)` 逐个 rank 串行处理 `[1,4096]` 级搬运/notify。
- **如何生效**：目标是把无 carried-state、地址不别名的 peer task 提升到
  orchestration/SPMD fan-out；不能预先承诺通信段延迟 ÷8，实际收益由
  L2 swimlane 和带宽竞争决定。
- **参考**：`REF/moe.py:178`（`pl.spmd(N_LOCAL)`）、`REF/expert_routed.py:80`（`pl.parallel`）。
- **改法**：先为 peer snapshot/fixed-slot remote-load 建立 per-peer
  non-aliasing task ABI、独立 output token 和 explicit join，再在
  orchestration/SPMD 边界 fan-out。CSR prefix、expert-major compact 和
  weighted gather 等带 `running/cursor/accumulate` 的循环保留 `pl.range`。
  `pl.parallel` 仅允许 orchestration，不能放入 `FunctionType.InCore`。
- **验证**：L3 parity + L2 swimlane 显示 peer 并发。
- **边界**：C1 之后；当前实现尚未完成，合同测试只保证没有用非法 InCore
  `pl.parallel` 伪装完成。

---

## Track D — INT8-native W8A8 MoE（gap-5）

### PERF-D1 · gate deferred-norm + dispatch-side INT8 量化
- **问题**：step3p5 MoE 走 BF16-dequant（临时），gate 未做 dispatch-side 量化。
- **shape**：输入 x `[16,4096]` BF16；gate_w `[4096, N_EXPERTS=288]`；输出 `x_norm_i8 [16,4096]` INT8 + `x_norm_scale [16,1]` FP32 + router logits `[16,288]`。
- **如何生效**：现状 dispatch 发的是 BF16 `[16,4096]`（每 token 8KB）；且 RMSNorm 要单独一遍读写 x。改 deferred-norm：一遍算出 `sq_sum`/`amax` 并直接量化成 `x_norm_i8`（`inv_rms` 作标量随 `x_norm_scale` 下传、对称量化里抵消）→ 少一遍 x 全量 pass；dispatch 发 INT8 `[16,4096]`（每 token 4KB）**减半通信量**，为 D2 的 INT8 cube 备料。
- **参考**：`REF/gate.py:103-140`、`:152-170`。
- **改法**：`P/models/step3p5/gate.py` RMSNorm 融合 per-token INT8 量化，输出 `x_norm_i8` + `x_norm_scale`。
- **验证**：gate 输出 vs BF16 参考 `ratio_allclose`（单元级）。
- **边界**：独立数值 track，与结构线零耦合。

### PERF-D2 · routed expert INT8×INT8 + requant 链
- **问题**：expert 用 BF16 → `≈47.6GB/rank` IPC pool；cube 未吃 INT8。
- **shape**：per rank 36 专家，每专家 `w1/w3 [4096,1280]`、`w2 [1280,4096]`；输入 `recv_x_i8 [1024,4096]` INT8 + scale `[1024,1]`；中间 `h [·,1280]`；输出 `[·,4096]` BF16。
- **如何生效**：BF16 权重 `36×4096×1280×2B×3 ≈1.13GB/层`；INT8 减半 → `≈189MB×3=0.57GB/层`，42 层 `47.6GB→≈24GB/rank`（解共驻 OOM）。计算侧 `matmul(out_dtype=INT32)` 走 INT8 cube（吞吐 ~2× BF16）→ INT32 acc 用 `row_expand_mul(recv_scale)×col_expand_mul(w_scale)` dequant → SwiGLU → per-row requant 成 `h_i8 [·,1280]` → w2 INT8 cube → dequant(`h_scale×routing_weight×w2_scale`) → BF16。
- **参考**：`REF/expert_routed.py:88-158`（K_TILE=512 `pl.pipeline(stage=2)`）、`:160-175`（requant）。
- **改法**：`P/models/step3p5/expert_routed.py` 按上链路重写；IPC pool 改存 INT8。`QUANT_TILE=512` 命中 a2a3 L2 line。
- **验证**：多步 L3（N=128 ≥95% vs vanilla）+ 逐层 detail（当前 BF16-dequant 是待替换的临时路径）。
- **边界**：D1 之后；gap-5 正解（memory 多条 gap-5）。

---

## Track E — LM head

### PERF-E1 · LM head 4 段 decoupled + 复用 `recv_x_buf`
- **问题**：历史 unroll Main 曾把 `lm_head_orch` inline 进同一 program；当前 canonical Main 已收敛为 hidden-only，不再包含 LM head。
- **shape**：输入 hidden `[16,4096]`/rank；`lm_head_weight [16112,4096]`/rank（vocab 切 8）；输出 logits shard `[16,16112]`/rank → 全 vocab `[16,128896]`。
- **如何生效**：现状 LM head 与末层 MoE 串行、且单独占 buffer。拆 4 段 worker（publish→tp→route→finish）后：(a) publish 段（发 hidden `[16,4096]`）可与末层 MoE 的 combine 段**在调度上重叠**；(b) 4 段**复用 MoE 的 `recv_x [1024,4096]` 窗口**作 hidden/logits 中转（MoE 与 LM-head 时分不冲突）→ 省一份 `[16,16112]` 级 buffer。
- **参考**：`REF/lm_head.py:433-515` + `REF/decode_fwd.py:877-901`。
- **改法**：拆 `publish → tp → logits_route → finish` 四 worker，各 `device=r`；复用 `recv_x_buf`。
- **验证**：logits L3 parity + 端到端延迟下降。
- **边界**：该项只能落在 vLLM tail/backend seam，不得把 LM head 重新塞回 canonical hidden-only Main；是否继续实施需单独重审。

---

## Track F — intra-kernel L1/L0 微调

### PERF-F1 · attention `late_dep=task_dummy(deps)` + `allow_early_resolve`
- **shape**：full-attn 内 `qr_proj`（q `[16,1024]`）与 `kv_proj`（k/v `[16,128]`）+ 前置 RMSNorm（`[16,4096]`）。
- **如何生效**：现状 rms → qr_proj → kv_proj 依赖链偏串行。让 rms 返回 TaskId，kv_proj 挂 `task_dummy(deps=[rms_tid])` 落后 qr_proj 一拍 + scope `allow_early_resolve=True` → qr_proj 与 kv_proj 在 cube/mte 上重叠，藏住 kv_proj 延迟（k/v 只有 `[16,128]`，本就小）。
- **参考**：`REF/decode_attention_swa.py:144`、`REF/rmsnorm.py:35-56`、`REF/moe.py:133,182,241,251`。
- **改法**：`P/models/step3p5/attention_full.py` / `attention_swa.py` 加 tid 返回 + `task_dummy` deferral + `allow_early_resolve`。
- **验证**：L3 parity + L2 swimlane 显示 qr/kv 重叠。
- **边界**：A1 出数后评估收益；独立。

### PERF-F2 · matmul pipeline stage 调优 + MTE 512B 对齐
- **shape**：热点 matmul —— dense `[16,4096]×[4096,1408]`、MoE expert `[·,4096]×[4096,1280]`、LM head `[16,4096]×[4096,16112]`；K dim = 4096（大）。
- **如何生效**：K=4096 的 matmul 用 `pl.pipeline(stage=2)` 双缓 K-loop；最大 K（如 LM head/input-proj）升 `stage=4` 让 MTE 搬下一块与 cube 算当前块重叠。按 A1 `perf_hints.log` 把非 512B 对齐的 MTE 搬运补齐（INT8 行 512B / BF16 行 512B）→ 消 MTE 停顿。
- **参考**：`REF/expert_routed.py:97,110,167,185`、`P/docs/performance-tuning.md:220,296,263`。
- **改法**：依 A1 hints/PMU 调各 matmul `pl.pipeline(stage=)`；补 MTE 512B 对齐。
- **验证**：PMU cube 利用率↑ + L3 parity。
- **边界**：**需 A1**（照 hints 调，不盲调）。

### PERF-F3 · RMSNorm+quant fused deferred-norm（复用）
- **shape**：dense/attention 前的 RMSNorm 输入 `[16,4096]`。
- **如何生效**：把 D1 的 deferred-norm（一遍出 norm+scale，不 per-element 应用 `inv_rms`）套到 dense/attention 的 RMSNorm 路径 → 每处省一遍 `[16,4096]` 全量 pass。
- **参考**：`REF/rmsnorm.py`、`REF/gate.py`（deferred-norm 同源）。
- **改法**：复用 D1 的 fused norm。
- **验证**：L3 parity。
- **边界**：随 D1。

---

## Track G — 调度轴 / 动态 batch

### PERF-G1 · 调度轴 batch → experts/feature + dynamic active-token（对齐 DeepSeek）🟦 落地中
- **问题**：step3p5 把 **batch 轴当 core fan-out**（`P/models/step3p5/attention_full.py:309` `pl.spmd((BATCH//BATCH_TILE)*...)`、`:332`、`:353`、`:383`、`:438` `pl.parallel(BATCH)`、`:577` `pl.spmd(BATCH)`），`BATCH=16` 静态 padded（`config.py:280`，仅 row0 有效）。曾从 `pl.parallel(user_batch)`（动态）退回 `pl.parallel(BATCH)`（静态）——`attention_full.py:427-438` 注释。
- **shape**：现状 SPMD 在 `BATCH=16` 轴 fan-out（decode 常仅 1 行有效 → 16 核里 1 核干活、15 核算 padding）；专家/中间维（36 experts × `[4096,1280]`）却是循环内串行。DeepSeek 反过来：token 是 `pl.range(active_tokens=nt)`（`nt` runtime），核 fan-out 打在 `pl.spmd(N_LOCAL=36)`（专家）+ `pl.spmd(MOE_INTER//tile)`（中间维）。
- **如何生效**：decode batch 天生小（常=1）。拿 batch=16 做核调度 → 核占用被 batch 上限锁死、且大半在算 padding。迁到 experts(36)/intermediate(1280) 轴 fan-out → **无论 batch 多小核都吃满大维度**；再用 runtime `num_tokens` 让 MoE 只 route 真实 token（`pl.range(nt)`）→ padding 行根本不进 routing，通信/计算都省。对 step3p5 = 把"16 核处理 16 行(15 padding)"改成"36+ 核处理 36 专家 × 真实 nt 个 token"。
- **参考**：`REF/decode_fwd.py:278-280`（`nt=max(num_tokens_per_owner)`）、`REF/moe.py:104,122,133`（`for t in pl.range(active_tokens)`）、`:183,308`（`pl.spmd(N_LOCAL)`）、`REF/expert_routed.py:80,93,110`（`pl.parallel(N_LOCAL_EXPERTS)` + `pl.spmd(MOE_INTER//...)`）、`REF/decode_attention_swa.py:115`（`for b in pl.range(B)` 顺序）。
- **改法**：(a) canonical host ABI 新增
  `num_tokens_per_owner [N_RANKS=8] INT32`；holder 从 live
  `hidden.shape[0]` 每 step 更新，whole-chip 取 owner max 并 clamp 到
  `[0,BATCH]`；(b) gate top-k、dispatch histogram/pack、inverse-map、
  fixed-slot peer pull、combine pull/gather 只遍历 active rows；routed-src
  staging 用 `sum(local_expert_count)`；(c) gate 的 expert-column chunk 改
  `pl.spmd(N_EXPERTS/32)`，routed expert 继续沿 36 experts × intermediate/
  hidden feature fan-out；combine 的 token 轴保持 runtime sequential，
  write-disjoint TOPK 轴并行；(d) attention 暂保留固定 16-row storage 和
  allocator-owned KV padding reserve，避免恢复历史 dynamic parallel 的
  UB/lifetime 问题。若本次要把 G1 置 ✅，仍需镜像 batch=1/2/8/16 DFX
  证明 attention padding score/context 已安全消除；否则保持 🟦。
- **验证**：多步 L3（N=128 ≥95% vs vanilla）+ A1 baseline 显示核占用↑ / padding 计算消除。
- **边界**：与 **B2 协同**（mega-kernel 重写时一起改最省）；独立于 D 线。这是 memory `integration_churn_root_causes` 标记的"static-BATCH pad vs dynamic T"分歧正解——**动手前先补 step3p5-vs-v4-flash 差异表**（feedback `align_deepseek_architecture_first`）。

---

## 通用落地规范

1. **精度验收 = 多步 decode 逐 token** vs live vanilla vLLM W8A8 oracle，seed=6127 / N=128 →
   **≥95% ALIGNED**（`pypto-lib/tests/step3p5/ci/LIVE_PRECISION_AB.md`，`stepfun/develop`）。
   多步已含第一个 token，**不再单列单步/单 token 测试**。stall/hang 用 `_probe_barrier_scale.py`
   + `RUN_CLEAN`（liveness，独立于精度）。
2. **falsify-before-assert**：定位根因用可证伪的隔离实验，不写"假设即事实"（feedback `integration_churn_root_causes`）。
3. **对齐 DeepSeek**：动手前列 step3p5-vs-v4-flash 差异表（差异+理由+改/留），只有"性能更好"才留差异（feedback `align_deepseek_architecture_first`）。
4. **pin substrate**：落地前锁 5 仓 commit（CLAUDE.md 版本表）。
