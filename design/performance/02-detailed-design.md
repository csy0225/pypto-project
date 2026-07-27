# 02 — Detailed Design (LLD)：各优化子任务

> 每个子任务一张卡片：current-source问题实证 / producer-to-lifetime链路 / 五类差异归属 / capacity与layout / 如何生效 / 参考 / 改法 / 验证 / 落地边界。shape只用于容量和layout说明，不得单独用于架构判断。
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
## 0. 端到端同构审计合同

本LLD禁止局部shape/name比较。对每个B/C/G结论，先以current source为事实源，再沿以下链路核对V4-Flash与step3p5：

```text
producer → 数学变换/quant/route-map → transport/window
→ consumer → rounding/reduction/placement → lifetime/reuse/allocator
```

审计输出必须分别标注：

- 能力/算法差异；
- 数学语义差异；
- layout/shape差异；
- host/allocator集成差异；
- backend/profile workaround。

若V4-Flash已有同构能力，只有layout或shape不同，归为参数化/存储适配，不能写成架构差异。任何“step3p5独有/必须保留”必须由完整producer-to-lifetime证据支持。任务卡、旧设计和历史probe可能落后于current source；先核对当前source和工作树diff，再执行或更新任务描述。

本次反例仅用于说明通用判定方法：INT8必须连同scale、padding、dequant和rounding核对；owner max必须连owner-vector producer、跨rank聚合、各stage active bound和KV/通信consumer核对；`BATCH=16`必须区分capacity与runtime logical batch；route-weight placement必须连route/weight producer、transport、consumer和最终FP32 weighted reduction核对。它们不是固定shape合同。

> 2026-07-24 的 unroll/generator file:line 已全部退休；当前实现位置只读
> `decode_fwd.py`、`dense_mlp.py`、`whole_decode_holder.py` 及对应合同测试。
> 对账：**A1/B1/SwiGLU-per-layer/B2 已交付**；historical pull C2仅作回归基线。当前通信目标为 **C1 → C2 → C3**，G1目标为runtime dynamic active batch/token。

---

## 0.1 step3p5 关键 shape 速查（TP=EP=8，per-rank）

| 量 | 全量 | per-rank（TP/EP 切分后） | 说明 |
|----|------|--------------------------|------|
| 残差 / hidden | `[CAPACITY, HIDDEN=4096]` BF16 | 同（TP不切batch/hidden） | `CAPACITY`是frontend/部署可配置物理上界；本次逻辑batch/token由runtime active count决定，不能把默认16或row0-only写成产品合同 |
| full-attn q | `NUM_HEADS_FULL=64 → 8192` | `wq_full [4096, 1024]`，q `[CAPACITY, 1024]` | 每 rank 8 头 (`NUM_HEADS_FULL_LOCAL=8`)，`HIDDEN_Q_FULL_LOCAL=1024` |
| swa-attn q | `NUM_HEADS_SWA=96 → 12288` | `wq_swa [4096, 1536]`，q `[CAPACITY, 1536]` | 每 rank 12 头，`HIDDEN_Q_SWA_LOCAL=1536` |
| KV | `NUM_KV_HEADS=8 → 1024` | `wk/wv [4096, 128]`，k/v `[CAPACITY, 128]` | 每 rank **1 KV 头**（`KV_HEADS_LOCAL=1`，`KV_HIDDEN_LOCAL=128`） |
| dense MLP | `INTERMEDIATE=11264` | `w_gate/up [4096, 1408]`，`w_down [1408, 4096]` | `INTERMEDIATE_LOCAL=1408`；3 个 dense 层 |
| MoE routed | `MOE_NUM_EXPERTS=288`，`MOE_INTERMEDIATE=1280` | 每 rank **36 专家**，每专家 `w1/w3 [4096,1280]`、`w2 [1280,4096]` | `MOE_NUM_EXPERTS_LOCAL=36`，`TOP_K=8`；42 个 MoE 层 |
| MoE shared | `SHARE_EXPERT_DIM=1280` | `w_gate/up_s [4096, 160]`，`w_down_s [160, 4096]` | `SHARE_EXPERT_DIM_LOCAL=160` |
| LM head | `VOCAB=128896` | `lm_head_weight [16112, 4096]`，logits shard `[CAPACITY, 16112]` | `VOCAB_LOCAL=16112` |
| MoE comm 窗口 | — | 容量由`active_capacity × TOP_K`、local-expert lane ownership、dtype/alignment和consumer ABI推导 | 历史`1024/128`仅是某配置实例，不是迁移后hard shape；目标为42层共享一套满足capacity上界的window |
| KV cache | — | `k_cache/v_cache [KV_CACHE_ROWS_DYN, 128]` BF16 | 45 层沿 leading 轴堆叠；paged `BLOCK_SIZE=128` |

派生：**MoE 权重/rank/层** BF16 = `36×4096×1280×2B ×3(gate/up/down) ≈ 1.13GB/层`；`×42 ≈ 47.6GB`（= 现状 IPC pool）。INT8 减半 → `≈24GB`。

## 0.2 PERF-B/C/G 当前决策表（2026-07-27）

> 对照对象：current `models/step3p5/decode_fwd.py@53eb7212` 与
> `origin/main:models/deepseek/v4-flash/{decode_fwd.py,moe.py}`。本表是
> B3/C1/C2/C3/G1的当前目标合同。先核对current source与工作树diff，再用V4-Flash端到端数据流对账；旧任务描述/历史probe若冲突，以current source和最新合同为准。所有差异必须经过producer→数学变换→transport/window→consumer→rounding/reduction→lifetime核对。

| 项 | v4-flash | step3p5 current | 差异理由 | 决策 |
|----|----------|-----------------|----------|------|
| B3 KV 所有权 | `CACHE_POOL_NAMES` + `RESIDENT_CACHE_OUTPUT_NAMES`，KV/state tensor 是 resident InOut | canonical `whole_chip_orch` / `host_orch` 已为 `pl.InOut`；holder 通过一次 `import_kv_all()` + `build_stacked_kv_pool()` 绑定 vLLM-owned IPC K/V section | step3p5 是 45 层 consolidated flat `[45*rows,128]`，v4-flash 是模型自己的多 pool ABI，不能只靠 reshape 互换 | **留 current ABI，补强合同与设备验证**：确认 prepare/import 只做一次、run 不 copy 整池、每层只按 `slot_mapping` 写一行 |
| C1 数据窗口 | V4-Flash在多层decode中共享一套MoE windows，并用单调`moe_epoch`复用metadata/payload/combine arrival | current canonical仍体现pull时代的window与signal lineage | 目标数据流已改为V4-Flash expert-lane；C1只负责共享window、单调epoch和最终consumer前不得复用 | **改**：删除pull专属双波合同；迁移后按真实metadata/payload/combine arrival DAG决定signal lineage与epoch步长，统一使用`AtomicAdd + WaitCmp.Ge` |
| C1 TP all-reduce 窗口 | v4-flash 的 MoE 窗口不等于 step3p5 attention/shared TP all-reduce scratch | MoE 层同时带 attention/shared 的 `tmp+signal`，其 barrier 固定使用 `expected=1/2` | 这两类窗口不是 EP dispatch/combine 协议；直接压成一套会被上一层残留计数提前放行 | **留 per-layer**：本次只折叠 EP dispatch/combine 的 12 类窗口；attention/shared 的 4 类 scratch 继续逐层隔离 |
| C1 512B signal stride | V4-Flash 的 control signal 仍是紧凑 `[N_RANKS,1] INT32` + `N_RANKS*4`，且多层复用不等于 512B signal ABI | step3p5 当前 stacked/reused backing 在 backend span/provenance 下需要物理 slot 隔离 | 512B 是 step3p5 当前 backend/profile 适配，不是多层共享本身，也不是 DeepSeek 通用 ABI | **仅 stacked/reused 且参与 notify/wait/AtomicAdd 的 control slot 使用 512B physical stride**，formal/window/slice `[128,1]`，逻辑 loop 只访问前 `n_ranks`；普通 data/MTP 独立 signal 不扩容 |
| C2/C3 dispatch | expert-lane `pl.spmd(N_LOCAL)` payload push、独立payload arrival、expert-lane gather | current canonical仍是fixed-slot peer-major pull和顺序peer TGET | current实现是迁移前历史基线，不再是目标架构 | **直接迁移**V4-Flash数据流；shape由runtime capacity上界、route上界和consumer ABI推导，不采用probe peer-slab硬约束 |
| C2/C3 combine | expert-lane scatter/`tensor.put`回source-owned route buffer，独立arrival wait，token reducer按TOPK顺序做FP32累加 | current canonical是staged source + inverse-map TGET + 本地gather | current实现不再作为production目标 | **直接迁移**scatter/wait/token reduce；保留whole-net epoch、runtime active token与既定FP32顺序 |
| G1 runtime batch/token ABI | V4-Flash传runtime `num_tokens`，attention/MoE按active bound运行；formal storage可有capacity上界 | step3p5需要owner-vector汇总并贯穿whole-net、KV和通信 | holder/IPC接线是step3p5差异；默认`16`不是逻辑batch硬约束 | **改**：runtime active batch/token决定逻辑范围；若frontend需静态formal shape，只能使用可配置`CAPACITY`上界，并确保attention、MoE、combine和KV写入都屏蔽inactive rows |
| G1 调度轴 | gate/routed expert 以 experts/intermediate/feature 为主要 core fan-out；token 轴是 runtime sequential bound | routed expert 已 `pl.parallel(36)` + intermediate `pl.spmd`，但gate仍按静态capacity维单块执行，combine 以 batch fan-out | decode 常见 T=1；专家/特征轴稳定且更宽，适合作为设备并行轴；动态 batch fan-out 曾触发 UB/lifetime 编译差异 | **改**：gate expert-column chunk 用 `pl.spmd(288/32=9)`；combine 保持 runtime token 顺序、TOPK=8 write-disjoint fan-out；保留 routed expert 36×feature fan-out |
| G1 attention/KV | V4-Flash以runtime token bound驱动attention/KV逻辑访问 | current实现仍含以默认capacity实例生成的静态tile与padding/reserve路径 | 这些只能视为当前frontend实现限制，不能固化成产品语义 | **改**：attention和KV写入支持runtime active batch/token；若保留静态tile/formal shape，必须标为可配置capacity实现并用valid shape/predicate屏蔽inactive rows，禁止为padding token写永久reserve槽 |

---

## Track A — 可观测性 & Baseline

### PERF-A1 · whole-net decode baseline + DFX 采集
- **问题**：whole-net 无 perf 数据 = 盲调。`docs/step3p5`、`docs/performance-tuning.md` 无延迟数据。
- **shape**：不改数据流；被测是 canonical `whole_decode_step3p5`（输入 hidden `[CAPACITY,4096]`/rank → 输出 pre-final-norm hidden `[CAPACITY,4096]`/rank；runtime active count决定有效行）。
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
- **shape**：循环体每层消费 hidden `[CAPACITY,4096]` → 输出 hidden `[CAPACITY,4096]`，有效范围由runtime active count决定；层内从 stacked 权重（B1）`pl.slice` 出当层 shard（如 MoE `[36,4096,1280]`）；layer_idx 为 dynamic scalar。
- **如何生效**：unroll 在 Python/DSL 层重复描述各层的调用、切片和依赖；
  折成 `pl.range` 后由一个 loop body 加 dynamic `layer_idx` 表达重复层，
  hidden `[CAPACITY,4096]` 的逐层串接保持不变；`CAPACITY`是可配置上界，不是逻辑batch。
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

### PERF-C1 · shared window set + `moe_epoch` + `WaitCmp.Ge`（关键路径）🟦 重新实现
- **问题**：historical per-layer communication stacks 让42次MoE调用各自持有一套EP窗口，增加常驻HBM、编译期window记账和whole-net生命周期复杂度。
- **V4-Flash 基线**：`origin/main:models/deepseek/v4-flash/decode_fwd.py` 在整网decode中只分配一套MoE communication windows；`moe.py` 使用单调 `moe_epoch`、`AtomicAdd` 与 `WaitCmp.Ge` 区分跨层复用的 metadata arrival、payload arrival 和 combine arrival。
- **如何生效**：step3p5 host只保留一套EP dispatch/combine data/control windows，并让每次MoE调用携带单调epoch。每个signal lineage的notify必须位于其真实producer完成之后，wait必须覆盖远端arrival；window再次被写入前，上一epoch的最终semantic consumer必须结束。
- **协议边界**：C1不规定current pull专属的 `2*epoch-1` ready / `2*epoch` read-complete双波，也不规定必须把多个语义阶段压进同一counter。迁移后的signal数量、epoch步长和是否对阶段分窗，应从V4-Flash数据流、step3p5 whole-net复用关系及runtime DAG推导。
- **window/shape**：目标是“一套满足模型上界的共享window”，不是复制V4-Flash示例shape。payload、metadata、route和control容量必须由step3p5的可配置capacity/route上界、local-expert ownership、INT8 scale物理padding和consumer ABI推导；probe中的peer slab或常量shape不得进入产品合同。
- **512B signal stride 口径**：
  - V4-Flash control signal可保持逻辑 `[N_RANKS,1] INT32`；512B不是通用signal ABI；
  - step3p5仅对同backing中stacked/reused且参与`notify`/`wait`/`AtomicAdd`的control slot使用512B物理隔离；
  - 逻辑访问仍只覆盖实际rank行；普通data window、独立compact signal和probe tensor不得机械扩成512B。
- **保护区**：attention/shared TP all-reduce 的 `tmp+signal`、固定peer累加顺序、FP32 accumulator和最终一次BF16 store保持不变。
- **验证**：source ownership/epoch合同 → canonical compile/lowered DAG → 多epoch设备liveness → batch=1/2/8/16 active-route telemetry → live精度与KV row-diff。低层probe只能证明表达能力。

### PERF-C2 · 迁移 V4-Flash dispatch/combine 数据流 🟦 待代码落地
- **目标**：以 `origin/main:models/deepseek/v4-flash/moe.py` 为唯一算法基线，替换current fixed-slot pull目标架构。
- **dispatch**：metadata统计与发布；metadata arrival wait；`pl.spmd(N_LOCAL)` expert-lane payload push；payload arrival wait；`pl.spmd(N_LOCAL)` expert-lane gather。每个expert block拥有不重叠的lane或由已证明offset划分的输出区间。
- **combine**：expert-lane scatter把已完成的 routed-expert 输出写回 origin route；arrival wait 观察所有远端写入；token-level reducer 按既定 TOP-K 顺序在 FP32 中累加 routed rows 与 shared-expert结果。
- **weight placement**：按 V4-Flash，route weight 随 dispatch metadata/payload 进入 expert lane，并在 routed expert 的 W2 epilogue 中与 activation/weight scale 一起完成缩放；combine 不再重复乘 route weight。若 step3p5 的数值 oracle 证明必须保留其他 placement，必须明确记录 rounding 边界和等价性，不能只因当前代码方便而保留。
- **step3p5适配**：保留 runtime dynamic active batch/token、可配置 physical capacity、KV holder/IPC ownership、step3p5 router/expert shape 与 canonical whole-net 入口。INT8 activation + scale、route weight 随 payload 携带、expert-lane scatter/reduce 均直接沿用 V4-Flash；只有 scale padding、capacity、route/count/map 表示和下游 tensor ABI 做参数化适配。
- **不再保留的目标**：fixed-slot peer-major `remote_load` pull、combine pull-back及其专属ready/read-complete协议仅作为历史回归对照，不再是C2完成态或production目标。
- **shape/ABI边界**：不得把V4-Flash示例的 `RECV_MAX`、probe的peer-major slab、`core_num=N_RANKS`、`task_dummy`或任何固定二维shape写成硬约束。实际容量必须证明不会overflow且满足512B storage/alignment和下游consumer ABI。
- **验证**：route/count/map oracle、write-disjoint ownership、remote arrival DAG、active/inactive route数、combine FP32顺序、multi-rank multi-epoch liveness、live precision。

### PERF-C3 · expert-lane SPMD 与 whole-net 调度适配 🟦 待代码落地
- **目标**：在C2的V4-Flash数据流上完成可编译、可调度的expert-lane fan-out，而不是重新选择是否保留pull。
- **调度轴**：dispatch push/gather与combine scatter按local-expert lane并行；token reducer按token稳定轴并行或采用语义等价实现。worker只计算，不在InCore中递归submit；不得用`pl.parallel`伪装带状态通信并行。
- **真实依赖**：payload-arrival wait必须依赖完整scatter/push；gather依赖metadata与payload arrival；token reduce依赖combine arrival。local RAW不能替代远端arrival，fake scalar或`x*0`不能制造依赖。
- **whole-net适配**：与C1共享window/epoch、G1 active-token、42次MoE调用和现有expert/shared路径闭合；最终consumer完成前不得复用window。
- **非约束项**：`pl.spmd`与captured TaskId是V4-Flash参考表达；若当前frontend需要等价submit形式，只能做最小ABI适配。peer slab、`spmd_submit`、`task_dummy`、sole reducer及probe tensor shape都不是验收硬门槛。
- **保护区**：不改canonical入口、KV ownership、TP/all-reduce、top-k FP32累加顺序和最终BF16 store。
- **验证**：source/alias合同 → parser/compile → lowered TaskId/DAG → 2-rank/8-rank write-disjoint与多epochliveness → batch=1/2/8/16 DFX → live精度与性能。

---

## Track D — INT8-native W8A8 MoE（gap-5）

### PERF-D1 · gate deferred-norm + dispatch-side INT8 量化
- **问题**：当前 step3p5 的 gate/norm/quant 数据流尚未完全按 V4-Flash 的 deferred-norm 语义收敛，不能把“INT8 activation + scale”误写成 step3p5 新能力。
- **shape**：V4-Flash 基线输出 `x_norm_i8 [CAPACITY,4096]` INT8、`x_norm_scale [CAPACITY,1]` FP32 和 router logits；step3p5 只参数化 `CAPACITY`、router shape、bias 和现有 norm 数学，runtime active rows 有效。
- **如何生效**：沿 V4-Flash 同一条 producer-to-consumer 链，在一次 norm/amax 过程中形成可复用的 INT8 activation 与 dequant scale，并供 gate、dispatch、shared expert 使用；不得把已有的 INT8+scale transport 重新设计成 step3p5 独有 ABI。
- **参考**：`REF/gate.py:103-140`、`:152-170`。
- **改法**：对照 `REF/gate.py` 的 deferred-norm、per-token quant 和 scale 定义，收敛 step3p5 的 producer、payload 和 consumer；scale 统一按 V4-Flash 的 `[CAPACITY,1]` 逻辑语义，物理 padding 仅由 capacity/alignment 决定。
- **验证**：gate 输出 vs BF16 参考 `ratio_allclose`（单元级）。
- **边界**：独立数值 track，与结构线零耦合。

### PERF-D2 · routed expert INT8×INT8 + requant 链
- **问题**：step3p5 当前已经具备部分 INT8×INT8、scale 和 intermediate requant 路径；剩余工作是按 V4-Flash 统一 expert-lane、tile/pipeline、scale 和 W2 epilogue 语义，而不是从 BF16 重新发明 INT8 架构。
- **shape**：V4-Flash 基线是 per-rank local experts、INT8 activation + per-row dequant scale、INT8 routed weights、INT32 accumulation、intermediate per-row requant 和 BF16 output；step3p5 仅参数化 36 experts、`HIDDEN=4096`、`INTER=1280`、capacity 和下游 ABI。
- **如何生效**：直接对齐 V4-Flash 的 INT8 cube、pipeline、`QUANT_TILE=512` data-tile 以及 W2 epilogue：`activation_scale × intermediate_scale × route_weight × w2_scale` 在 expert lane 完成，之后 scatter BF16 routed output，combine 只做 FP32 token reduction。
- **参考**：`REF/expert_routed.py:88-158`（K_TILE=512 `pl.pipeline(stage=2)`）、`:160-175`（requant）。
- **改法**：以 `REF/expert_routed.py` 为模板收敛当前 routed expert 的 producer/consumer placement；只修改 step3p5 shape、weight layout、capacity 和 backend 必要适配。`QUANT_TILE=512` 仅属于 data tile/cache/MTE 性能对齐，不得与 control-signal 512B stride 混写。
- **验证**：多步 L3（N=128 ≥95% vs vanilla）+ 逐层 detail（当前 BF16-dequant 是待替换的临时路径）。
- **边界**：D1 之后；这是 V4-Flash 数据流迁移与 step3p5 shape 适配，不是新建独立 INT8 架构。

---

## Track E — LM head

### PERF-E1 · LM head 4 段 decoupled + 复用 `recv_x_buf`
- **问题**：历史 unroll Main 曾把 `lm_head_orch` inline 进同一 program；当前 canonical Main 已收敛为 hidden-only，不再包含 LM head。
- **shape**：输入hidden `[CAPACITY,4096]`/rank；`lm_head_weight [16112,4096]`/rank（vocab切8）；输出logits shard `[CAPACITY,16112]`/rank → 全vocab `[CAPACITY,128896]`，仅runtime active rows有效。
- **如何生效**：现状 LM head 与末层 MoE 串行、且单独占 buffer。拆 4 段 worker（publish→tp→route→finish）后：(a) publish段（仅发送runtime active hidden rows）可与末层 MoE 的 combine 段**在调度上重叠**；(b) 4 段**复用 MoE 的 `recv_x [1024,4096]` 窗口**作 hidden/logits 中转（MoE 与 LM-head 时分不冲突）→ 复用按capacity上界分配的logits中转buffer。
- **参考**：`REF/lm_head.py:433-515` + `REF/decode_fwd.py:877-901`。
- **改法**：拆 `publish → tp → logits_route → finish` 四 worker，各 `device=r`；复用 `recv_x_buf`。
- **验证**：logits L3 parity + 端到端延迟下降。
- **边界**：该项只能落在 vLLM tail/backend seam，不得把 LM head 重新塞回 canonical hidden-only Main；是否继续实施需单独重审。

---

## Track F — intra-kernel L1/L0 微调

### PERF-F1 · attention `late_dep=task_dummy(deps)` + `allow_early_resolve`
- **shape**：full-attn内`qr_proj`（q `[CAPACITY,1024]`）与`kv_proj`（k/v `[CAPACITY,128]`）+ 前置RMSNorm（`[CAPACITY,4096]`）；仅active rows参与逻辑计算。
- **如何生效**：现状 rms → qr_proj → kv_proj 依赖链偏串行。让 rms 返回 TaskId，kv_proj 挂 `task_dummy(deps=[rms_tid])` 落后 qr_proj 一拍 + scope `allow_early_resolve=True` → qr_proj 与 kv_proj 在 cube/mte 上重叠，藏住 kv_proj 延迟（k/v feature维较小，batch维仅是capacity上界）。
- **参考**：`REF/decode_attention_swa.py:144`、`REF/rmsnorm.py:35-56`、`REF/moe.py:133,182,241,251`。
- **改法**：`P/models/step3p5/attention_full.py` / `attention_swa.py` 加 tid 返回 + `task_dummy` deferral + `allow_early_resolve`。
- **验证**：L3 parity + L2 swimlane 显示 qr/kv 重叠。
- **边界**：A1 出数后评估收益；独立。

### PERF-F2 · matmul pipeline stage 调优 + MTE 512B 对齐
- **shape**：热点matmul——dense `[CAPACITY,4096]×[4096,1408]`、MoE expert `[active_rows,4096]×[4096,1280]`、LM head `[CAPACITY,4096]×[4096,16112]`；M维有效范围由runtime active count决定；K dim = 4096（大）。
- **如何生效**：K=4096 的 matmul 用 `pl.pipeline(stage=2)` 双缓 K-loop；最大 K（如 LM head/input-proj）升 `stage=4` 让 MTE 搬下一块与 cube 算当前块重叠。按 A1 `perf_hints.log` 把非 512B 对齐的 MTE 搬运补齐（INT8 行 512B / BF16 行 512B）→ 消 MTE 停顿。
- **参考**：`REF/expert_routed.py:97,110,167,185`、`P/docs/performance-tuning.md:220,296,263`。
- **改法**：依 A1 hints/PMU 调各 matmul `pl.pipeline(stage=)`；补 MTE 512B 对齐。
- **验证**：PMU cube 利用率↑ + L3 parity。
- **边界**：**需 A1**（照 hints 调，不盲调）。

### PERF-F3 · RMSNorm+quant fused deferred-norm（复用）
- **shape**：dense/attention前的RMSNorm输入 `[CAPACITY,4096]`，仅runtime active rows有效。
- **如何生效**：把 D1 的 deferred-norm（一遍出 norm+scale，不 per-element 应用 `inv_rms`）套到 dense/attention 的 RMSNorm 路径 → 每处省一遍capacity buffer上的全量pass，并避免处理inactive rows。
- **参考**：`REF/rmsnorm.py`、`REF/gate.py`（deferred-norm 同源）。
- **改法**：复用 D1 的 fused norm。
- **验证**：L3 parity。
- **边界**：随 D1。

---

## Track G — 调度轴 / 动态 batch

### PERF-G1 · experts/feature 调度轴 + runtime dynamic active batch/token（对齐DeepSeek）🟦 重新实现
- **问题**：current实现把默认capacity实例（历史配置为`BATCH=16`）同时当作formal shape、调度轴和逻辑batch，并在常见decode batch较小时计算或保留inactive padding rows。该数值是当前配置实例，不是step3p5产品硬约束。
- **目标语义**：每次调用由runtime active batch/token决定逻辑范围。owner-vector在whole-net入口汇总出一致active count，并传入attention、gate、dispatch、routed/shared expert、combine、residual和KV写入路径。
- **capacity合同**：若frontend要求静态formal shape，使用可配置`CAPACITY`作为physical storage上界，并满足shape/alignment要求；`CAPACITY`不得被解释为逻辑batch。所有kernel必须通过runtime loop bound、`valid_shape`、predicate或等价机制屏蔽`[active, capacity)`。
- **调度轴**：token轴是runtime bound；稳定并行轴优先使用local experts、expert columns、intermediate和hidden feature。不得为了静态编译继续把capacity轴当作唯一core fan-out。
- **attention/KV**：attention只计算active rows；KV只写active token对应的真实slot。历史固定16-row padding和allocator-owned padding reserve只作为current实现/编译风险记录，不是目标ABI，也不能作为保留inactive KV写入的理由。
- **通信**：C2/C3的V4-Flash expert-lane dispatch push/gather与combine scatter/wait/reduce只处理active routes/rows；window按capacity上界分配，但notify/count/route语义按本次active count运行。
- **参考**：`REF/decode_fwd.py`的owner-vector max，`REF/moe.py`的runtime `active_tokens`与expert-lane `pl.spmd`，以及V4-Flash attention中的runtime token循环。
- **验证**：至少覆盖runtime active batch/token `1/2/8/16`，并覆盖至少一组非16 physical capacity配置；检查active/inactive数值、route count、KV row-diff、window overflow、lowered predicate/valid bound、L3精度和设备DFX。默认16配置通过不能证明dynamic合同完成。
- **边界**：若某frontend版本不能表达所需dynamic bound，应记录当前ABI限制并使用可配置capacity适配；不得把该限制升级为模型固定batch或永久padding/reserve合同。

---

## 通用落地规范

1. **精度验收 = 多步 decode 逐 token** vs live vanilla vLLM W8A8 oracle，seed=6127 / N=128 →
   **≥95% ALIGNED**（`pypto-lib/tests/step3p5/ci/LIVE_PRECISION_AB.md`，`stepfun/develop`）。
   多步已含第一个 token，**不再单列单步/单 token 测试**。stall/hang 用 `_probe_barrier_scale.py`
   + `RUN_CLEAN`（liveness，独立于精度）。
2. **falsify-before-assert**：定位根因用可证伪的隔离实验，不写"假设即事实"（feedback `integration_churn_root_causes`）。
3. **对齐 DeepSeek**：动手前列 step3p5-vs-v4-flash 差异表（差异+理由+改/留），只有"性能更好"才留差异（feedback `align_deepseek_architecture_first`）。
4. **pin substrate**：落地前锁 5 仓 commit（CLAUDE.md 版本表）。
