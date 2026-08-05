# 04 · Attention 优化专项（step3p5 full / SWA flash decode）

> **最终实现覆盖说明（2026-08-03）**：本文已合并原
> `attention/attention-tiling-and-partitioning.md` 的最终 task/tile 设计。§0–§9
> 保留专项探索过程，包含早期 fixed-24 lane、四阶段 split、standalone Pass-A/B/C
> 与 cast 默认关闭等历史状态；**这些内容不得再作为当前实现说明。** 当前权威状态以
> [§12](#12-wave5-source-publication-稳定性收口2026-08-03) 和
> [§13](#13-当前最终实现task-切分与-tile-profile合并文档) 为准：
> logical task 数按 active workload 推导，runtime 再映射到物理 AIC/AIV；
> 5–10 us 仅是 task-grain 搜索起点；Full 的 SV 与 segment-local recurrence 已融合，
> 只保留必要的 `full_online_softmax_reduce/finalize`；Full/SWA out-proj cast 默认都融合。
> 当前源码为 `pypto-lib stepfun/develop@7099476b` 与
> `pypto stepfun/develop@defa97c5`。Wave5 以 self-target TPUT 显式发布 all-reduce
> source partial，已完成 immutable audit、Main/MTP compile、Main N=128×3、
> Main batch16、MTP batch1/batch16×2、64K/batch16 ITL/DFX，状态为
> **0162 release-qualified**；其它机器/架构未由本轮独立证明。
>
> **性质**：LLD 专项。聚焦 decode 阶段 flash-attention kernel 本体（`attention_full.py` /
> `attention_swa.py`）的重写路线，独立于 README 主表里的 Track A–H。收敛后其中的子项会以
> `PERF-*` ID 回填 [`task-tracking.md`](task-tracking.md)。
>
> **当前源码基线**：pypto-lib `stepfun/develop@7099476b7c4f13112b159e237e7a64344803caf0`，
> pypto `stepfun/develop@defa97c526fec7e8f032dbbfcc39c820add02bf7`；canonical Main =
> `models/step3p5/decode_fwd.py:whole_decode_step3p5`。正文中的旧 file:line 只对应当时快照，
> 不能覆盖当前源码。
>
> **审计口径**（沿用 [`README.md` 顶层审计方法](README.md)）：任何“step3p5 独有 / 必须保留”
> 判断都要沿 `producer → 数学变换 → transport → consumer → rounding/reduction → lifetime`
> 核对；V4-Flash 已有同构能力时，shape 不同只算参数化/存储适配，不升级为架构差异。

---

## 0. 一句话结论 + 边界

> **⭐ 2026-07-30 全专项实测收口（7 个 device 实验后的定论）**：**attention 不是 64k 延迟瓶颈，
> attention kernel 内已无可落地的延迟优化。** 证据：A.1 ≈0 / A.2 illegal / A.3b not-viable /
> B +7% 回退 / B'(8-row) illegal / B-nobias +5.2%(结构性) / **L1(o_proj cast 融合，纯减 GM 往返)
> = 中性(±1%)**。**去掉 attention 的 GM 往返(L1) 都不动 ITL** → decode 不受 attention MTE/compute
> 约束。perf-h1 实测：64k ITL = **~50 ms 固定 floor(context 无关, 80%) + ~12.7 ms context 部分**
> (1024→64k, = attention-over-KV, flash kernel 已近最优)。**~50ms floor(大头)不在 attention**——
> 是 per-layer 的 MoE 专家 matmul / projection / norm / tp_all_reduce / dense MLP / LM-head / host。
> **要降 64k 延迟必须打这个 floor（MoE/proj/norm/comm），不是 attention。** 下一步 = §5.6 定位
> floor 构成后精准打；attention 侧只保留 A.1（清理）+ L1（清理，去 FP32 scratch，token-exact）。

当前 full/SWA 的 **decode** flash 是 **4-stage split** 形态（QK / softmax / SV / online-softmax
各一个 `pl.spmd(BATCH)`，中间用 GM scratch 串联）。它能跑通、精度对齐（N=256 hidden exact），
但有**一个真实浪费 + 一个被证伪的假设**：GM scratch 按静态 capacity 与最大 context 预分配
（真实，见 §2.1）；SV stage 读 16 行只用 8（曾以为可砍半，但 device 证明 cube 16-fractal 下
不可 trim、也无 cube 收益，见 §2.2/§5.2）。V4-Flash（当前源码位于 `models/deepseek/v4/`）**是 MLA + sparse attention，
没有可直接照搬的 full-attention decode 结构**——能借的是它的 lowering 写法（additive-inf
bias、qk_pv 融合、FP32 直接 row_sum），不是它的 kernel 结构。它跑通的 shape 是
`H=64 / HEAD_DIM=512 / H_TILE=16`，与 step3p5 的 `H=8(或12) / HEAD_DIM=128 / pad=16(或24)`
差很远，**任何搬移都必须过 ST**。

**L1（o_proj FP32→BF16 cast 融合，vs qwen3）实测（2026-07-30，perf-h1，cards 8-15）**：
canonical CI PASS（token-exact，L1 数值等价）；ITL 1024 `50.73→51.31`、64k `63.44→64.15`——
**中性（±1%，噪声内）**。删掉 `partial_attn_proj_fp32` [16,4096] FP32 GM scratch + 整个
`full/swa_out_proj_cast` SPMD pass，token 不变——**纯清理/对齐 qwen3，但不动延迟**（再次证明
64k 不受 attention GM 往返约束）。patch `workspace/perf-patches/L1-oproj-cast-fuse.patch`
（branch `perf/step3p5-attn-l1`）。可作为无风险 cleanup 落地，但不宣称 perf 收益。



**本专项不承诺 latency 数字**。理由见 [§5.6](#56-64k-下-latency-收益暂无实测--主要理由是显存不是延迟)：
64k DFX 显示 attention 的 span 占比被插桩放大、不可当延迟占比。当前能站住的硬理由是
**显存 footprint**——64k 下每个 FULL attention invocation 的静态 scratch shape 约 161 MiB，
低 bs 时 90%+ 的 batch 分区不被访问（见 §2.1）。
注意 flash scratch 对 active batch **恒定**，所以它**不是** bs=16 撞 device HBM OOM 的增量来源
（那是 KV pool，见 [`task-tracking.md` 2026-07-29 行](task-tracking.md)）；scratch 收缩帮的是
低 bs footprint 与长 context 下的静态工作集，不直接解 bs=16 OOM。它是否形成实际 HBM 峰值，
取决于 compiler/runtime 对 inline call、loop body 和 transient tensor 的 allocation/liveness。

---

## 1. 现状 ground truth — 4-stage split flash

### 1.1 结构（`attention_full.py:574-738`）

注释自述（`:574-581`）：mirror qwen3/32b 的 split 形态，四个顺序 per-batch spmd
（`full_qk_matmul` / `full_softmax` / `full_sv_matmul` / `full_online_softmax`），
中间用 GM scratch 传 raw scores、softmax exp + mi/li、SV partials。选这条路是因为
`pl.slice(..., valid_shape=...)` 替代 `set_validshape + fillpad` 让 VEC lowering 走了一条
**proven-safe** 路径（Phase 15 单卡跑通的关键决策之一）。

| Stage | spmd 名 | 类型 | 行为 | file:line |
|-------|---------|------|------|-----------|
| 1 QK | `full_qk_matmul` | cube | `q_padded[16,128] @ k_tile[128,128]ᵀ → raw_scores[16,128]` 落 `all_raw_scores` | `:600-621` |
| 2 softmax | `full_softmax` | vec | 读 **8 行**（`valid_shape`）→ scale/row_max/exp → **double-cast** → `row_sum` → 落 exp(BF16) + mi/li | `:627-661` |
| 3 SV | `full_sv_matmul` | cube | 读 **16 行** exp → `@ v_tile[128,128] → oi[16,128]` 落 `all_oi_tmp` | `:664-684` |
| 4 online-softmax | `full_online_softmax` | vec | 只读 **前 8 行** oi/mi/li，跨 block 归并 → 归一化 → 写 `attn_out` | `:690-738` |

SWA 完全同构：`attention_swa.py:560` 起同样的 4-stage + scratch，只是
`Q_HEAD_PAD_SWA=24` / `Q_HEAD_BATCH_SWA=12` / `SWA_Q_PAD_ALIGNED=32`（`config.py:391-396`）。

### 1.2 decode / prefill 边界（重要，避免 decouple 返工）

canonical decode loop-form Main **不是**另抄一份，而是
`decode_fwd.py:219 attention_full_inline = pl.inline(attention_full._func)`，
在 `:436`（L0 full_dense）/`:1933`/`:2999` 三个 full call site inline 调用；SWA 也有对应
call site。**prefill 不使用这两个 decode 函数**，而是独立的
`prefill_attention_full.py` / `prefill_attention_swa.py`（`prefill_fwd.py:136-142,462,776`）。
因此改 decode kernel 不会自动传导到 prefill；若目标是统一 flash 数据流，prefill 必须作为
独立 patch scope 验证，不能把它写成同一 source 传导。

### 1.3 GM scratch 账本（`attention_full.py:582-597`）

每次 full-attn invocation 分配 5 个 transient GM tensor（`MAX_CTX_BLOCKS = MAX_SEQ // BLOCK_SIZE`）：

| tensor | shape | dtype | 64k(nb=512) 单层 | 备注 |
|--------|-------|-------|------------------|------|
| `all_raw_scores` | `[BATCH·nb·16, 128]` | FP32 | 64 MiB | Stage1→2 |
| `all_oi_tmp` | `[BATCH·nb·16, 128]` | FP32 | 64 MiB | Stage3→4，**用户漏算的另一半** |
| `all_exp_padded` | `[BATCH·nb·16, 128]` | BF16 | 32 MiB | Stage2→3 |
| `all_cur_mi`/`all_cur_li` | `[BATCH·nb, 16]` | FP32 | 1 MiB | Stage2→4 |
| **合计** | | | **≈161 MiB / FULL 层** | ✓ 对上用户数字 |

`BATCH = STORAGE_BATCH_CAPACITY`（`config.py:350`，capacity=16），**不是** active batch。

---

## 2. 两个新发现的问题（逐条 verified）

### 2.1 scratch 第一维用 capacity 而非 active batch，且随 context 膨胀

`pl.create_tensor([BATCH * MAX_CTX_BLOCKS * Q_HEAD_PAD_FULL, ...])` 的首维锁死 `BATCH=16`
capacity；spmd loop `for fa_b in pl.spmd(BATCH)` 用 `if fa_b < active_tokens` 跳过
inactive core（`:601`），但 **tensor 已按 16 全量分配**。

- **bs=1**：15/16 = **93.75% 是死内存**。
- **随 context 膨胀**：`MAX_CTX_BLOCKS = MAX_SEQ // 128`，64k → 512 blocks → 161 MiB/层。

| active batch | 实际使用 scratch | 物理分配 scratch/层 | inactive 行比例 |
|---|---|---|---|
| 1 | 1 × 512 blocks | 161 MiB | 93.8% |
| 8 | 8 × 512 blocks | 161 MiB | 50.0% |
| 16 | 16 × 512 blocks | 161 MiB | 0% |

当前 source 的 shape 是静态 `BATCH` capacity，不能把 active batch 再乘进单次 invocation 的
分配量。B2 后 **40** 个 MoE 层（L3..L42，`decode_fwd.py:194 NUM_MOE_LAYERS`）位于单个
`pl.range` loop body（`:3556`），loop 内 attention 每 iteration 走同一 inline body；L0 full /
L1-L2 swa / L43-L44 specialized（`NUM_MOE_LAYERS_TOTAL=42`）是循环外的独立 attention site，
各站点顺序执行。因此源码层面**存在复用 transient backing 的机会**；但 `pl.create_tensor` 最终
是否被 allocator 复用、inline 展开后是否共用同一 backing，不能只凭 Python 控制流下结论。故：

- `161 MiB` 是单个 FULL invocation 的静态 shape 账；
- `12 × 161 MiB ≈ 1.9 GiB` 只是完全不复用时的 source-shape 上界；
- 实际 peak 必须以 lowered IR / allocation plan / runtime ring-heap liveness 为准。

这修正了“bs=16 OOM 只归 KV pool 22.54 GiB”的旧结论：flash scratch 同时在涨，会对 HBM/ring-heap 有贡献，但不能仅凭该数字推断它是 OOM 或延迟的唯一来源；必须采集 allocation/liveness 峰值后再归因。

> ⚠ **补充修正见 [§5.5](#55-scratch-膨胀只发生在-full-attn-层swa-被-sliding-window-cap-住)**：这条只对 **12 个 FULL-attn 层**成立；33 个 SWA 层被
> `SLIDING_WINDOW=512` cap 住（≈4 blocks），scratch 不随 context 涨。

### 2.2 Stage3 读 16 行、Stage2 只写 8 行 → SV 白算一倍

- Stage2 softmax 只对 `Q_HEAD_BATCH_FULL=8` 行算 exp 并 `assemble` 进 `all_exp_padded`
  （`:637` slice 8 行，`:648` 写 8 行）；rows 8..15 在该 scratch_row 是陈旧/未初始化。
- Stage3 SV 读 **`Q_HEAD_PAD_FULL=16`** 行喂 matmul（`:680-683`），→ `oi_tmp[16,128]`
  一半算在陈旧数据上（`:684`）。
- Stage4 只读前 `Q_HEAD_BATCH_FULL=8` 行（`:696`/`:709`）。

**数值无害**（Stage4 丢弃后 8 行）。**但 A.2 device 证伪（2026-07-29，见 [§5.2](#52-a2-sv-slice-16-8-device-证伪cube-boxed-tile-行必须是-16-的倍数)）：这不是可省的浪费。**
cube 的 boxed tile 行必须是 `innerRows=16` 的倍数——把 SV matmul M 从 16 砍到 8 被 ptoas 直接
拒绝（`'pto.alloc_tile' op expects result boxed tile rows to be a multiple of innerRows (16),
but got 8`）；且 16 行是 cube fractal 的最小处理单位，M=8 与 M=16 cube 成本相同——**"白算一倍"
前提不成立**。真正的余量只有 exp_tile 从 GM 读 16 行的 MTE 带宽（cube FLOPs 省不掉）与读到
未初始化行的卫生问题（已被 Stage4 裁掉，无害）。SWA 同构：物理 `[32,128]` tile、
`valid_shape=[12]`、Stage3/4 按 32 行、裁出前 12——M=12 同样非 16 倍数，同样不可 trim。

---

## 3. V4-Flash 参考定位：借写法，不借结构

| 维度 | V4-Flash | step3p5 full / SWA | 能否照搬 |
|------|----------|--------------------|----------|
| 结构 | MLA + sparse attention（HCA/CSA/SWA/indexer/compressor） | 稠密 paged full-attn + 稠密 SWA | **不能**，无同构 full-attn decode |
| shape | `H=64 / HEAD_DIM=512 / H_TILE=16` | `H=8(12) / HEAD_DIM=128 / pad=16(24)` | lowering 不保证在 step3p5 shape 下安全 |
| mask | additive-inf bias（另一条 VEC lowering） | `pl.slice(valid_shape) + fillpad(min)` | 可借（=方案 C，B 的前提） |
| reduction | FP32 直接 `row_sum` | exp→BF16→FP32 double-cast 再 sum | 可借但**有耦合**，见 [§5.1](#51-double-cast-不是纯浪费是-liden-与-svnum-的一致性耦合) |
| pipeline | `pl.pipeline(编译期常量)` | `fa_ctx_blocks` 是 runtime | **未验证**，见 [§5.7](#57-方案-e-无-in-repo-证据现有-plpipeline-都是编译期常量-bound) |

结论：V4-Flash 提供的是**若干 lowering 惯用法**的证据，不是 kernel 骨架。凡搬一处，过一次 ST。

### 3.1 DeepSeek decode attention 的实际切分方式（2026-07-30 补充）

本轮重新核对了：

```text
models/deepseek/v4/decode_sparse_attn_swa.py
models/deepseek/v4/decode_sparse_attn.py
```

DeepSeek 并不是统一把 context block 铺满 24 核，而是根据单个 work item 的计算密度选择不同
切分方式。

#### 统一抽象：切分发生在哪个逻辑轴

Decode attention 的模型逻辑维度统一写为：

| 轴 | 上层含义 | Decode 场景 |
|---|---|---|
| `B` | batch/request 数 | active batch |
| `Sq` | query sequence length | 通常为 1 |
| `Skv` | KV/context sequence length | `ctx_len` 或 `WIN` |
| `Nq` | query head 数 | FULL=8，SWA=12（TP-local） |
| `Nkv` | KV head 数 | 当前 TP-local 为 1 |
| `D` | 单个 head 的向量维度 | `HEAD_DIM=128` |

模型张量通常写为：

```text
Q: [B, Sq, Nq, D]
K: [B, Skv, Nkv, D]
V: [B, Skv, Nkv, Dv]
```

GQA 下 `Nq/Nkv` 个 query heads 共享一个 KV head。固定一个 batch、query token 和 KV
group 后，QK/PV 的 GEMM 映射为：

```text
QK:
    Q      [GEMM M = query-head tile, GEMM K = D]
    K^T    [GEMM K = D,               GEMM N = Skv tile]

PV:
    P      [GEMM M = query-head tile, GEMM K = Skv tile]
    V      [GEMM K = Skv tile,         GEMM N = Dv]
```

因此本文准确描述为“沿 `Skv`/context 轴切分”：

- 在 QK GEMM 中对应 GEMM `N`；
- 在 PV GEMM 中对应 GEMM `K`；
- 不能笼统称为“沿矩阵 K 方向切分”。

工程实现还需区分以下层级：

1. **存储 block**：KV cache 的寻址粒度，例如 `BLOCK_SIZE=128`；
2. **计算 `Skv` tile**：一次 QK/PV matmul 实际消费的 KV token 数，例如
   `ATTN_K_TILE=128`（该变量名中的 K 指 key-token，而非 GEMM K）；
3. **work item**：可独立调度的逻辑工作，如 `(token, Skv tile)`；
4. **device task**：一次 dispatch，内部可以串行处理一个或多个 work items；
5. **lane/core 映射**：task 或 work item 如何分配到物理核心；
6. **核内 pipeline**：同一 task 内不同 M tile/`Skv` tile 如何交叠。

`BLOCK_SIZE` 不必等于 `ATTN_K_TILE`。当计算 tile 跨多个物理 cache blocks 时，通常需要先
gather/拼接到连续的 L1/L0 operand。反之，仅把两个 block 映射给同一个 task，并不意味着
已经形成一个 256-token matmul。

#### SWA：不切 `Skv` block，按 token/request 切 task，核内按 head tile 流水

DeepSeek V4 SWA 的关键静态配置为：

```python
WIN = 128
ATTN_K_TILE = 128
SPARSE_BLOCKS = 1
H_TILE = 16
QK_M_TILE = 32
```

并显式约束：

```python
assert WIN == ATTN_K_TILE
```

即整个 sliding window 作为一个 `Skv` tile，不再按 128-token block 继续拆 task。外层为：

```python
with pl.spmd(T, name_hint="qk_pv"):
    qk_t = pl.tile.get_block_idx()
```

每个 task 负责一个 token/request；task 内部通过：

```python
for qk_hb in pl.pipeline(H // QK_M_TILE, stage=2):
```

按 32 heads 一组执行 QK → softmax → PV。这样同一个 `[128, HEAD_DIM]` KV tile 可被多组
query heads 复用，同时避免把 QK、softmax、PV 拆成多个只有数微秒的独立 dispatch。

映射到统一抽象：

```text
work item      = t
device task    = (t, 一个完整 Skv tile)
Skv tile       = ATTN_K_TILE = 128 key tokens
QK GEMM        = [M=32, K=HEAD_DIM] × [K=HEAD_DIM, N=128]
PV GEMM        = [M=32, K=128] × [K=128, N=HEAD_DIM]
M compute tile = QK_M_TILE = 32 heads
M tile 数      = H / QK_M_TILE
qk_pv lane 数  = T
```

例如 `B=8, S=1, H=64` 时，`qk_pv` 有 8 个 task，每个 task 内有
`64 / 32 = 2` 个 M tile。单个 M tile 的主要矩阵形状为：

```text
Q tile:  [Nq_tile=32, D=HEAD_DIM]
KV tile: [Skv_tile=128, D=HEAD_DIM]
score:   [Nq_tile=32, Skv_tile=128]
```

所以 DeepSeek SWA 的跨核并行主要来自 `T`；`Skv` 轴保留在 task 内以复用 KV，M 轴则通过
`QK_M_TILE=32` 提高 cube 计算密度。

DeepSeek SWA 的 merge/normalize 再按 `(token, head_tile)` 切分：

```python
with pl.spmd(T * (H // H_TILE), name_hint="merge_norm"):
```

由于 `SPARSE_BLOCKS == 1`，SWA 实际没有跨 `Skv` block 的 online-softmax merge 循环。

#### CSA/HCA：多 block 时才使用 24 核，并先按有效负载重排

DeepSeek CSA 的 work item 是：

```text
(token, sparse_block)
QK_ITEMS = T * SPARSE_BLOCKS
```

它并非直接按静态 block index 轮转，而是先构造 `valid_block_mask` 和 `qk_order`：

1. 将非空/有效 sparse blocks 排到 `qk_order` 前部；
2. 将空 blocks 追加到队尾；
3. 再用 24 个 lane 做 stride 分发：

```python
with pl.spmd(NUM_QK_CORES, name_hint="qk_pv"):
    qk_flat = qk_core + qk_it * NUM_QK_CORES
    qk_item = qk_order[qk_flat]
```

这样有效重任务会优先一核一个，避免不同 token 的有效 block 数不同导致某些核心只处理空块、
另一些核心连续处理多个有效块。该策略适合 `SPARSE_BLOCKS > 1` 且 block 间负载不均的场景，
不适合计算量很小的 SWA window。

其参数映射为：

```text
work item      = (t, sparse_block)
Skv tile       = ATTN_K_TILE = 128 key tokens
QK GEMM        = [M=32, K=HEAD_DIM] × [K=HEAD_DIM, N=128]
PV GEMM        = [M=32, K=128] × [K=128, N=HEAD_DIM]
M compute tile = QK_M_TILE = 32 heads
lane 数        = NUM_QK_CORES = 24
lane 访问      = qk_order[core], qk_order[core + 24], ...
```

例如 `T=8、SPARSE_BLOCKS=4` 时共有 32 个 work items。24 个 lane 首轮最多各取一个，
剩余 8 个在第二轮处理。`qk_order` 只改变 work-item 调度顺序，不改变 QK/PV 的 tile shape。

另外，DeepSeek 中两个 head 参数职责不同：

```text
QK_M_TILE=32：cube QK/PV 的 M tile，偏计算资源利用；
H_TILE=16：partial 落盘和 merge_norm 粒度，偏存储布局与归并并行度。
```

#### 对 Step3.5 的含义

Step3.5 SWA 当前参数为：

```text
SLIDING_WINDOW = 512
BLOCK_SIZE = 128
SWA_WIN_BLOCKS = 4
Q_HEAD_BATCH_SWA = 12
```

2026-07-30 已实测过将 4 个 blocks 分散到 24 lanes 的版本
（`cd09f2bd perf(step3p5): stripe sliding-window context blocks`）。DFX 中单个 SWA kernel 的
执行时间约 `1.3–2.5 us`，而 dispatch-to-finish latency 约 `3.5–5.2 us`。结论是：

- block 并行确实消除了单核串行；
- 但 work item 太小，dispatch/同步开销超过 kernel 本体；
- SWA 不应以“铺满 24 核”为目标，而应优先提高每个 task 的计算密度。

因此当前对齐 DeepSeek 的低风险版本改为：

```text
branch: perf/attn
commit: 76bd04c4
策略：每个 batch 使用 2 个 group；每个 group 串行处理相邻 2 个 128-token blocks
```

统一表示为：

```text
work item       = (batch, block_group)
block_group     = 2 个连续 Skv blocks
Skv storage block = BLOCK_SIZE = 128
逻辑 Skv group  = 2 × 128 = 256 key tokens
M 有效行        = Q_HEAD_BATCH_SWA = 12
M storage pad   = SWA_Q_PAD_ALIGNED = 32
task 数         = BATCH × 2
active_batch=1 时有效 task 数 = 2
```

当前的 256-token group 只是**调度分组**，不是 256-token fused matmul。每个 task 内仍执行：

```text
block 0 -> [12,128] QK / softmax / PV
block 1 -> [12,128] QK / softmax / PV
```

它仍保留逐 block scratch 与 Stage4 online merge。真正的 `Skv tile=256` 需要把两个物理
cache blocks gather/拼接成连续 operand，并重新验证 cube boxing、UB 和 softmax lowering。

QK、softmax、SV 三个 stage 使用相同的 `(batch, block_group)` 映射，Stage4 保持原有确定性
online-softmax 归并顺序。该版本的目标是：

1. 将 active batch=1 时每阶段的有效 task 数从 24 降到 2；
2. 每个 task 承担两个 block，提升计算/dispatch 比；
3. 暂不引入 QK+softmax+PV mixed Cube/Vec 融合，避免重现历史 507018/boxed-tile 问题；
4. 保持 scratch 布局和 reduction/rounding 顺序不变，降低精度风险。

当前静态检查结果：

```text
attention FULL/SWA unit contracts: 7 passed
Python compile / git diff --check: passed
```

该 grouped 版本的无 DFX ITL、token-exact 精度和 DFX task 粒度仍需以 0162 实测结果为准，
在结果完成前不得宣称有性能收益。

#### FULL attention 与 DeepSeek 对齐的边界

Step3.5 FULL 64k 有 512 个 context blocks，属于多 work-item 场景。当前
`46662b20 perf(step3p5): stripe full attention context blocks across 24 lanes` 将 Stage1–3
按 24 lane 条带分配，512 blocks 时每核约 21/22 blocks，静态负载已经基本均匀；该版本曾测得
64k ITL `63.44 ms → 53.24 ms`（约 16.1%）。

FULL 后续若仍观察到动态不均衡，可借 DeepSeek CSA 的 `valid_block_mask + qk_order` 思路，
但必须先证明不同 batch/context row 的有效 block 数确实形成运行时偏斜。对于 active batch=1、
所有 512 blocks 均有效的典型 64k decode，增加 planning task 只会引入额外开销。当前更明确的
串行尾部是 Stage4 的 block merge，而不是 Stage1–3 的 21/22 block 分配差异。

FULL 的参数映射为：

```text
work item      = (batch, context_block)
Skv storage/tile = BLOCK_SIZE = 128
QK GEMM        = [M=8, K=HEAD_DIM] × [K=HEAD_DIM, N=128]
PV GEMM        = [M=8, K=128] × [K=128, N=HEAD_DIM]
M 有效行       = Q_HEAD_BATCH_FULL = 8
M storage pad  = Q_HEAD_PAD_FULL = 16
lane 数        = FULL_ATTN_CTX_LANES = 24
lane 映射      = lane + iteration × 24
```

64k、`active_batch=1` 时共有 512 个有效 `Skv` work items，每 lane 处理 21 或 22 个。
短 context（例如 1024）只有 8 个有效 blocks，此时最多只有 8 个有效 lane；继续增加 lane
不能增加实际并行度，应转而考虑单 task 的 M/K 计算密度。

---

## 4. 改写方案 A–E（按 风险/收益 排序，尚未动代码）

### A. 低风险速赢，不动结构（**建议先做**）

1. **改为 V4 数值顺序**：`row_sum` 直接吃 FP32 exp，PV 仍消费 BF16 exp——少两次 tile cast，
   但会改变当前分母的 rounding 语义，不能预称“精度更好”。
   ⚠ 前提见 [§5.1](#51-double-cast-不是纯浪费是-liden-与-svnum-的一致性耦合)：SV 数值来源与 li 归一化的一致性。
   ✅ **已落地并过 canonical CI（2026-07-29，image `…20260729-allreduce-push`，cards 0-7）**：
   `attention_full.py:645` + `attention_swa.py:624` 删 `exp_scores_fp32` 回读、`cur_li =
   row_sum(exp_scores)`。`main_hidden_8step rc=0 passed=True`（199s，8-step token 不变、
   `hidden_tp_spread=0`、无 stall）→ 分母 rounding 改动未扰动 8 步轨迹。commit `a71679ba` on
   `csy0225/pypto-lib:perf/step3p5-attn-a1`（off `cc850ee5`）；**formal N=128 vs live vanilla
   待补（需 oracle standup），过后 FF 入 `stepfun/develop`。**
2. ~~**Stage3 slice `Q_HEAD_PAD_FULL(16)`→`Q_HEAD_BATCH_FULL(8)`**~~ ❌ **NOT VIABLE（device 证伪
   2026-07-29）**：cube boxed tile 行必须是 `innerRows=16` 的倍数，M=8 被 ptoas 拒
   （3 个 `full_sv_matmul` 全 fail，见 [§5.2](#52-a2-sv-slice-16-8-device-证伪cube-boxed-tile-行必须是-16-的倍数)）；且 16 是 cube fractal 最小单位，
   trim 也省不了 cube FLOPs。SWA M=12 同样非 16 倍数。**A.2 撤回。**
3. **scratch 第一维 `BATCH` → runtime active capacity**——bs=1 省 93.8%。分两半：
   - **A.3a（只算 active）✅ 已是现状（2026-07-29 源码核实）**：FULL/SWA 各 4 个 flash stage
     全部 `if fa_b < active_tokens` guard（`attention_full.py:600-692` /
     `attention_swa.py:579-677`；`active_tokens = clamp(num_tokens, 0, BATCH)`），inactive
     core 跳过全部 QK/softmax/SV/online-softmax 计算。**无需改代码，关闭。**
   - **A.3b（分配按 active）❌ 不可行（standalone；2026-07-29 源码核实）**：`create_tensor`
     首维须静态（GM 分配）；改 runtime `active_tokens` 只能走 `pl.dynamic`，而 step3p5 全仓
     刻意把 model-bound 维静态化（`config.py:47-51`）正是为了绕开 [§5.3](#53-scratch-首维-batch-runtime--撞-pldynamic-首维的已知坑) 引的 §3（跨函数
     slice 丢 stride）/ §4（幻 int32 参数）/ host_orch NameError（`attention_swa.py:171`）。且
     `BATCH=16` 是产品 capacity 合同，不能静态缩小。→ scratch 显存只能靠结构性 **B**（消
     raw_scores+exp）/ **D**（消全部 scratch）拿，**A.3b 并入 B/D，不单独立项**。

### B. 主菜：融合 Stage 1+2+3（V4 的 qk_pv 形态），Stage4 保留归并

`all_raw_scores` + `all_exp_padded` 整个消失（nb=512 时省 ~96 MiB/FULL 层），两次 GM 往返消失。
依赖 C（✅ 已验证 additive-bias masking）。**风险**：这接近当年撞 507018 的形态（Phase 15 正是
为了绕开它才拆成 4-stage）。

✅ **FULL 已落地并过 canonical CI（2026-07-29，image `…20260729-perf-h1`，cards 8-15，
`--skip-mtp`）**：3 个 `pl.spmd` 合成一个 `full_qkv_fused`（QK→additive-bias softmax→PV 在
in-kernel tile 上完成，只把 `all_oi_tmp`/`all_cur_mi`/`all_cur_li` 写 GM），删掉
`all_raw_scores`+`all_exp_padded`。`SINGLE_CHIP_HIDDEN_CI=PASS`、`main_hidden_8step rc=0
passed=True`（204s）——**编译过（UB 压力 OK）、运行无 507018/stall、8-step token 不变**。
关键结论：**C 的 additive-bias 正是让融合绕开历史 507018 的原因**（valid_shape+fillpad 融合会撞，
additive-bias 不撞）。实现要点：PV matmul 保 M=`Q_HEAD_PAD_FULL`=16（cube 16-fractal）；
mi/li 存成 `[1,16]` 宽行（Stage4 读前 8），避开 `[N,1]` 列 slice；per-block partial 与 split
形态逐 bit 一致 → Stage4 未改。patch `workspace/perf-patches/B-fuse-full.patch`（branch
`perf/step3p5-attn-b`，未推）。
- **遗留**：SWA 仍是 split 形态（`SLIDING_WINDOW=512` window-capped，scratch 小、非峰值，
  故 SWA 融合的显存收益小）；FULL 融合已拿到峰值显存收益（峰值层 = 64k FULL 层）。SWA 融合
  作为一致性 cleanup 可后补。
- **待测（perf）**：64k 下 `AllocateMemoryAddr` scratch 实际下降 + ring-heap 峰值 + ITL delta
  vs perf-h1 baseline（64k p50 64.1 ms）——按 `--num-blocks 512` DFX 采集记入 perf-baseline。
- ⚠ **ITL 实测（2026-07-29，同环境 A/B，perf-h1 镜像，cards 8-15，`_stage_main_hidden_only
  --itl-iters 20`）：B 是延迟回退，不是提升**。ctx=1024 `50.73→52.21 ms`（+2.9%）、
  ctx=65536 `63.44→67.89 ms`（**+4.45 ms / +7.0%**）。同环境 baseline 63.44 ≈ 记录的未插桩
  64.1，故 STRACE 噪声仅 ~0.7 ms、可忽略，+7% 是真实回退。**根因假说**：additive-bias
  （arange/sub/max/min/mul/col_expand/add ~10 VEC op）**无条件**加在全部 512 blocks 上，但只有
  最后 1 个 partial block（`valid_len<128`）需要 mask——511 个 full block 的 bias 是纯浪费，
  盖过了省下的 GM 往返。**B 不按现状落地**（净负：+7% 延迟换 ~96 MiB 显存，且该显存不足以解
  bs=16 OOM）。
- ⚠⚠ **B' 已实测（2026-07-29，perf-h1）：8-row softmax 不可行——cube boxing**。想把融合里的
  softmax 从 16 行降到 8 行（`pl.slice(raw_scores, [8,128])`）→ ptoas 拒
  `'pto.alloc_tile' boxed tile rows must be multiple of innerRows(16), got 8`。根因：`raw_scores`
  是 QK matmul 输出 = **boxed（cube-fractal）tile**，切 8 行非法（同 A.2 §5.2）。
  **关键结构洞察**：split 形态能做 8-row softmax，正因为它先把 raw_scores 落 **GM**
  （`all_raw_scores`），从 GM 切 8 行合法——**split 的 GM 往返不是纯浪费，它把 cube 输出
  "de-box" 成普通 tensor，换来更便宜的 8-row softmax**。融合把它 re-box 回 16 行 → 强制
  16-row softmax（2× exp/row_max VEC），这正是 B +7% 的结构性来源，且**融合形态无法规避**
  （任何 per-block 融合都吃 cube 的 16-row 输出；D 同理）。
- **结论（data-backed）**：attention Stage1+2+3 融合对**延迟是负的且结构性受限**——省的 GM 往返
  比它强制的 16-row softmax 便宜。B 只剩 ~96 MiB 显存价值（不解 bs=16）。**当前 4-stage split
  实际上对 softmax 延迟已是较优形态**（GM de-box → 8-row softmax）。
- ⚠⚠⚠ **B-nobias 计时探针实测（2026-07-29，perf-h1）→ 融合方向判死**。为拆分 +7% 的构成，跑了
  一个去掉 bias 的计时变体（数值故意错、只计时）：64k p50 = **66.73 ms**（baseline 63.44）。
  即 +7% = **~+1.8% bias（可 guard）+ ~+5.2% 强制 16-row softmax（cube boxing，去不掉）**。
  **即便完美 B''（bias 全省）仍 +5.2% @64k，救不回来。** → **B / B'' / D 整个"per-block 融合"
  方向对延迟都是负的**（任何融合都吃 cube 的 16-row boxed 输出，逃不掉 16-row softmax）。
  attention kernel 本体**已无可落地的延迟优化**；64k 延迟瓶颈不在 attention（§5.6），下一步应回
  §5.6 的 ctx A/B 定位真瓶颈，不再在 attention 里抠。

### C. B 的解锁前提：`fillpad/valid_shape` → V4 的 additive-inf bias

走另一条 VEC lowering。✅ **已 device 验证（2026-07-29，image `…20260729-allreduce-push`，
cards 0-7，isolated FULL-Stage2 probe）**：把 `pl.slice(valid_shape=)+fillpad(min)` 换成 v4 式
`col_bias[1,128]`（`arange`→`sub(col_idx, valid_len)`→`neg/max(0)/min(1)`→`mul(·,1e20)` 得
0/−1e20）+ `col_expand` 广播到 8 行再 `add`——**在 step3p5 `[8,128]` shape 下能 lower，且
`main_hidden_8step rc=0 passed=True`（8-step token 不变、tp_spread=0、无 stall）**。即 additive
bias 与 fillpad 数值等价、lowering path 干净。
- **踩坑**：`valid_len` 是 INDEX，`pl.cast(valid_len, FP32)` 报 `Cast between float and index
  types is not supported`；须两步 `pl.cast(pl.cast(valid_len, INT32), FP32)`。
- **不单独落地**：C 在 split 形态下只是多几个 VEC op、无收益——**其价值是证明 B 的 masking
  机制可行**，随 B 一起进（B 用 additive bias 才能融合 Stage1+2+3）。probe patch:
  `workspace/perf-patches/C-additive-bias.patch`（branch `perf/step3p5-attn-c`，未推）。

### D. 终局：经典 flash-attention（UB 内单 `(m,l,o)` 累加器跨 KV-block 滚动）

per-block partial 一个都不物化，Stage4 连带消失，scratch 从 161 MiB 掉到 KB 级。
反正 block 循环本来就是单核串行，物化 partial 零收益——这才是这段代码本该长的样子。
**风险最高**（等于重写 kernel）。
- ⚠ **延迟同 B 判死**：D 仍是 per-block 融合，每 block 的 QK 出 cube 16-row boxed 输出 → 逃不掉
  16-row softmax（B-nobias 实测 +5.2% @64k 的结构性来源）。D 的价值只剩**显存**（scratch→KB），
  延迟同样是负的。仅当"显存"成为硬约束（如真要解 bs=16）时才值得，且要先确认省下的显存能跨过
  OOM 门槛（B 的 ~96 MiB 不够）。**不作为延迟优化立项。**

### E. `pl.range` + `pl.pipeline(stage=2)`

V4 pipeline 的是编译期常量 `h // QK_M_TILE`；step3p5 的 `fa_ctx_blocks` 是 runtime 值。
签名看似接受 `RangeArg`，但 **runtime bound 能否 pipeline 是未知项，不能当既定方案**。
见 [§5.7](#57-方案-e-无-in-repo-证据现有-plpipeline-都是编译期常量-bound)。

---

## 5. 补充（本轮 review 的增量，用户分析里没覆盖的点）

### 5.1 double-cast 不是纯浪费，是 li(den) 与 SV(num) 的一致性耦合

`:645-647`：`exp → cast BF16 (exp_scores_bf16) → cast FP32 → row_sum → cur_li`。
关键：**Stage3 SV 消费的正是 `exp_scores_bf16`**（`:648` 写、`:679` 读）。所以 li 是对
**BF16-rounded** 的 exp 求和——刻意让分母 li 与分子 oi(=exp_bf16 @ v) 用同一批 rounded 值。
若方案 A 直接 `row_sum(exp_scores)`(FP32) 但 SV 仍吃 BF16 exp，则分子用 rounded、分母用
unrounded → 归一化语义变化。当前 V4 源码已经明确采用这一顺序：
`qk_li = row_sum(qk_exp)`，随后 `qk_exp_bf16 = cast(qk_exp, BF16)` 再做 PV。
因此 open question 不是“V4 怎么做”，而是 **step3p5 是否接受从 rounded-denominator 改为
FP32-denominator**。A.1 应做成明确的数值变更实验，并以 attention ST + whole-net token/hidden
门判断，不能把它包装成无语义变化的删 cast。

### 5.2 A.2 SV slice 16→8 device 证伪：cube boxed tile 行必须是 16 的倍数

`:683 oi_tmp = pl.matmul(exp_tile, v_tile)`，`exp_tile` 首维就是 matmul 的 **M**。现在 M=16；
A.2 想砍成 8。**2026-07-29 device 实测（image `…20260729-allreduce-push`）：M=8 被 ptoas 拒绝**：

```
'pto.alloc_tile' op expects result boxed tile rows to be a multiple of innerRows (16), but got 8
```

3 个 FULL SV matmul 实例（`full_sv_matmul` / `full_moe_chip_orch_full_sv_matmul` /
`…swiglu7_swiglu16_full_sv_matmul`）全部 codegen fail。根因 = cube 的 16×16 fractal：matmul
输出 boxed tile 的行必须是 `innerRows=16` 的倍数，M=8（及 SWA 的 M=12）非法。

**更关键：即便合法也无收益。** 16 行是 cube fractal 的最小处理单位——M=8 与 M=16 的 cube
成本相同。所以 §2.2 "白算一倍" 的前提不成立：SV 的 16 行不是可删的多余 FLOPs，是硬件粒度。
唯一真实余量是 exp_tile 从 GM 读 16 行的 **MTE 带宽**（cube 算力省不掉）。**A.2 撤回**；若要省
那点 MTE，只能靠 B/D 的结构性重写（消掉 GM 往返），不是简单 slice。

### 5.3 scratch 首维 BATCH→runtime，会撞 `pl.dynamic` 首维的已知坑

`known-pypto-pitfalls.md §3`：`pl.dynamic` 首维 → 跨函数 slice 丢父 stride；§4：可能在
kernel 签名冒出幻 `int32_t` 参数。scratch 首维改 runtime 不是把常量换个变量那么简单，要么走
`active_tokens` 的 dynamic-dim 且验证所有下游 slice 的 stride，要么退一步：**保持 capacity 分配
但只对 `active_tokens` 段循环**（现状已如此，省的是**计算**不是**分配**）。真正省显存要动分配维，
需先过 dynamic-shape 编译验证。→ 方案 A.3 拆成两级：A.3a“只算 active”（低风险，可能已被 spmd
guard 覆盖）、A.3b“分配也按 active”（碰 dynamic 首维，另立子任务）。

### 5.4 `all_oi_tmp`（64 MiB FP32）是 161 MiB 的另一半，A/D 别只盯 raw_scores/exp

用户方案 B 只点了 `all_raw_scores + all_exp_padded`（96 MiB）消失，但 `all_oi_tmp` 还有 64 MiB
FP32 SV partials，且同样按 16 行过量写（`:684`）。方案 D（UB 滚动累加器）才真正消掉它；方案 B
之后 oi partial 仍要落 GM 给 Stage4 归并。→ 显存账要把 `all_oi_tmp` 算进去，B 只省 ~60%，D 才归零。

### 5.5 scratch 膨胀只发生在 FULL-attn 层，SWA 被 sliding-window cap 住

`SLIDING_WINDOW=512`（`config.py:173`）→ SWA 每步最多看 512 token = 4 blocks，**与 context 无关**。当前 SWA scratch 的物理行宽使用 `SWA_Q_PAD_ALIGNED=32`，不是有效布局行数 `Q_HEAD_PAD_SWA=24`；账本必须按 32 重算。
所以随最大 context 增长的 source-shape 只出现在 **12 个 FULL-attn 层**（1 full_dense +
10 full_moe + 1 full_moe_swiglu16）；**33 个 SWA 层**（2+30+1）scratch 是小且恒定的。
SWA 当前源码明确用 `SWA_WIN_BLOCKS=ceil(SLIDING_WINDOW/BLOCK_SIZE)=4` 分配并用同一上界迭代，
因此不会按 `MAX_SEQ` 分配；它仍有 capacity 方向的 inactive-row 浪费。
→ 显存优先级：FULL 是长 context 静态 shape 的重点；但在拿到 allocation/liveness 前，不能写成
bs=16 OOM 的大头。SWA 主要是 capacity/inactive 分区问题。方案 A.3 对两者都有效，但 B/D 的
省显存收益集中在 FULL 路径。而 §2.2 的 SV over-compute
对 **全部 42 attention 层**都成立（FULL 16→8，SWA 32→12；SWA 的物理 padding 比例更高）。

### 5.6 64k 下 latency 收益暂无实测，主要理由是显存不是延迟

[`task-tracking.md` A1 行 / 2026-07-29](task-tracking.md)：64k DFX 里 attention 的 97.9% 含
**插桩放大**（span 膨胀 5.21×），不可当延迟占比；`tp_all_reduce` 仅 1.84%，routed expert
busy 0.99%。ITL floor ≈70 ms 的构成被插桩掩盖，待“同镜像 ctx=1024 vs 65536 DFX 相减”拆开。
→ 本专项的**硬理由是显存 footprint**（方案 A.3 / B / D 减小 64k 固定 scratch + 低 bs 死内存）；
但 flash scratch 对 bs 恒定（§2.1），**不是** bs=16 OOM 的增量来源，别把这几个方案写成
「解锁 bs=16」——bs=16 OOM 的大头是 KV pool。latency 收益（省 GM 往返、SV 砍半）是**假说**，
落地后要用 DFX A/B 相减实测，不预先写成加速数字。

### 5.7 方案 E 无 in-repo 证据：现有 `pl.pipeline` 都是编译期常量 bound

`attention_full.py:267/281` 已有 `pl.pipeline(decode_scope1_hidden_blocks, stage=4)`，但
`decode_scope1_hidden_blocks = HIDDEN // INPUT_PROJ_K_CHUNK`（`:216`）是**编译期常量**。
`decode_compressor_ratio128.py:147` 的 `pl.pipeline(s_dim, stage=2)` 也要确认 `s_dim` 是否
常量（大概率是 compress 的静态维）。→ **没有任何 in-repo 证据证明 `pl.pipeline` 接受 runtime
bound**；方案 E 保持“未知项”，要落地必须先写最小 probe 验证 runtime-bound pipeline 能否编译。

### 5.8 目前没有 attention 专属 ST，"过 ST" 的 gate 要先立起来

`tests/step3p5/unit/` 已有两个 active-bound AST/source contract，但没有 attention/full 专属 device 数值 ST。当前已有的 attention 数值门仍是 whole-net
canonical N=128/256 逐 token（[`ci/LIVE_PRECISION_AB.md`](../../../workspace/pypto-lib/tests/step3p5/ci/)）。
→ 每个方案落地前，要么复用 whole-net 回归（慢、覆盖但难定位），要么按单卡 ST/UT 铁律
（`apply_perrank_patch`，TP=8 per-rank slice）新写一个 full/SWA attention kernel ST 做快速
per-kernel 数值 + liveness 定位。**建议先补 ST**，否则 A–E 每步都只能靠整网回归判对错。

---

## 6. 落地映射 + 建议顺序

### 收益量化（改哪里 / 好处 / 预计提升）

> **纪律**：显存收益可确切算（scratch shape 账）；**延迟收益一律不预报百分比**——64k 下
> attention 的 DFX span 份额被插桩放大 5.21×（§5.6），真实延迟贡献要 B 落地后 ctx A/B DFX
> 相减才知。下表 "预计提升" 分开写"显存（确切）"与"延迟（假说，待测）"。

| 点 | 改哪里 | 好处 | 显存（确切） | 延迟（待实测） | 状态 |
|----|--------|------|-------------|----------------|------|
| **A.1** | softmax 删 `exp→BF16→FP32` 回读（`attention_full.py:645`/`swa:624`） | 少 1 次全 tile cast + 分母精度对齐 v4 | 0 | ≈0（2 个小 tile cast） | ✅ 落地 |
| **A.2** | (Stage3 SV M 16→8) | — | 0 | 0 | ❌ cube 16-fractal 否决 |
| **A.3a** | 无（现状 spmd guard） | inactive 核已跳过 | 0（已有） | 0（已有） | ✅ 已实现 |
| **A.3b** | (scratch 首维 runtime) | — | 0 | 0 | ❌ 静态维约束，并入 B/D |
| **C** | masking → additive −inf bias | 无（standalone），为 B 铺路 | 0 | 0 | ✅ 验证，随 B 进 |
| **B** | 融合 Stage1+2+3（qk_pv），留 Stage4 | 消 `all_raw_scores`+`all_exp_padded` + 2 次 GM 往返 | **~96 MiB/FULL 层**（64k：161→65 MiB） | **实测 +7% @64k（回退非提升）**：additive-bias 无条件加全 512 block；B' guard partial-block 待做 | ⚠ 回归通过但**不按现状落地** |
| **D** | UB 滚动累加器，partial 不物化 | 连 `all_oi_tmp` 也消，Stage4 并入 | **~161 MiB/FULL 层 → KB 级**（最大显存收益） | **延迟同 B 判死**（per-block 融合逃不掉 16-row softmax，+5.2% 结构性） | ⚠ 仅显存价值，非延迟优化 |

> 显存收益是**每次 FULL-attn invocation 的静态 shape 账**；峰值随 loop liveness 折算（§2.1）。
> 12 个 FULL 层受 context 影响，33 个 SWA 层被 `SLIDING_WINDOW=512` cap（§5.5）。落地口径：
> B/D 的显存收益按 `AllocateMemoryAddr` + ring-heap 峰值实测记入 `perf-baseline.md`；延迟按
> 同镜像 ctx A/B DFX 相减记入，不空写 %。

| 本专项方案 | 建议 PERF ID | 层（README 第二维度） | 收益类型 | 依赖 | 风险 |
|-----------|-------------|----------------------|---------|------|------|
| A.1 V4 FP32-den/BF16-num 顺序 ✅ | PERF-F(new) | L1 kernel 数据流 | 少两次 cast；数值语义变化 | attention ST | 中 · **canonical CI passed，N=128 待补** |
| ~~A.2 SV slice 16→8~~ ❌ | — | L1 | 无（cube 16-fractal） | — | **NOT VIABLE（M 须 ×16，device 证伪 §5.2）** |
| A.3a 只算 active | 可能已被 spmd guard 覆盖 | L1 | — | 核对现状 | 低 |
| A.3b 分配按 active | PERF-G 关联 | 结构/codegen | 省显存 | §5.3 dynamic 首维 | 中 |
| B 融合 1+2+3 | PERF-新 | 结构 | 省 ~100 MiB + 2 次 GM 往返 | C | 高 |
| C additive-inf bias | B 的前置 | L1 lowering | 解锁 B | 复现验证 507018 | 中-高 |
| D UB 滚动累加器 | PERF-新 | 结构 | scratch→KB 级 | — | 最高 |
| E runtime pipeline | 未知项 | L0 流水 | 重叠 | probe 验证 | 未知 |
| ST 脚手架 | 前置 | 可观测性 | 定位能力 | — | 低 |

**建议推进顺序**（A-tier 已收口：A.1 ✅ 落地；A.2 ❌ device 证伪；A.3a ✅ 已实现关闭；A.3b ❌ 并入 B/D）：
`A.1 formal N=128（补 vanilla oracle）→ C additive-bias probe → B(qk_pv + 消 raw_scores/exp)
→ D(UB 滚动累加器，消全部 scratch) → allocation/liveness 实测`。E 单独 probe。原因是：

- A.1 已过 canonical 8-step token-exact + `tp_spread=0`，只差 formal N=128 vs live vanilla（需 oracle）；
- A.2 / A.3b 均已 device/源码证伪，不再是速赢——scratch 显存只能靠 B/D 结构性拿（§2.1/§5.2/§5.3）；
- C（additive-inf bias）是 B 的前置：先隔离验证它在 step3p5 shape 下能 lower 且绕开 507018；
- B 消掉 raw_scores/exp 两块 GM（~96 MB/FULL 层）+ 2 次 GM 往返，是 D 前的中间态；
- D 是终局（scratch→KB 级），风险最高，放最后。

**每步的 gate（沿用铁律 7 + [`pypto-perf-regression` skill](../../../workspace/pypto-lib/.claude/skills/)）**：
1. liveness：`RUN_CLEAN` + `_probe_barrier_scale.py`（stall/deadlock 独立判定）；
2. 精度：多步 decode 逐 token vs vanilla W8A8，seed=6127 / N=128 ≥95% ALIGNED，且 hidden
   finite + TP spread=0；
3. （结构性方案）显存：记录 ring-heap 峰值 + bs=1/8/16 是否解锁；
4. （若声称 latency）DFX A/B 相减，不空写加速。

---

## 7. 落地前要先解开的 open questions

1. ~~**§5.2**：cube matmul M=8/M=12 是否合法？~~ ✅ **已答（device 2026-07-29）：非法**——cube
   boxed tile 行须 ×16，M=8/12 被 ptoas 拒；且 16 是 fractal 最小单位，trim 无 cube 收益。A.2 撤回。
2. **§5.5**：SWA 当前已核实按 `SWA_WIN_BLOCKS=4` 分配；后续只需补 allocation/liveness 实测，确认 transient tensor 的生命周期和峰值。
3. **§5.1**：V4 已核实为 FP32 `row_sum` + BF16 PV；step3p5 接受该 rounding 变化的
   whole-net 精度已过 canonical 8-step token-exact（A.1），**formal N=128 vs live vanilla 待补**。
4. **§5.3**：`active_tokens` 能否安全做 `create_tensor` 首维（dynamic）？→ dynamic-shape 编译 probe。
5. **§5.7**：`pl.pipeline` 能否接受 runtime bound？→ 最小 runtime-bound pipeline probe。
6. **§5.6**：70 ms ITL floor 里 attention 真实占多少？→ ctx=1024 vs 65536 同镜像 DFX 相减
   （这本是 A1/H3 的待办，attention 优化的 latency 立项依赖它）。

---

## 8. 相关

- 现状 kernel：`workspace/pypto-lib/models/step3p5/attention_full.py` / `attention_swa.py`
- canonical Main inline 点：`decode_fwd.py:219`（`pl.inline`）+ `:436`/`:1933`/`:2999`
- 参考实现（当前 perf 分支工作树在 `models/deepseek/v4/`）：
  `decode_sparse_attn.py`、`decode_sparse_attn_hca.py`、`decode_sparse_attn_swa.py`、
  `decode_compressor_ratio128.py`（直接读工作树）。
  ⚠ 命名：本文沿用兄弟文档 01/02/03 的 “V4-Flash” 叫法；该 MLA+sparse 参考在 perf 分支路径是
  `v4/`，在 `origin/main` 上是 `v4-flash/`（同一参考、不同分支路径），引用前先确认自己的 checkout。
- 已知坑：`pypto-lib/docs/known-pypto-pitfalls.md`（§1 [N,1] UB align、§3 dynamic 首维、§4 幻参数）
- 调优 playbook：`pypto-lib/docs/performance-tuning.md` / `precision-tuning.md`（no-double-cast 条目）
- 回归 runbook：`.claude/skills/pypto-perf-regression/SKILL.md`
- 主表 / 分层：[`README.md`](README.md)；跟踪：[`task-tracking.md`](task-tracking.md)
- Phase 15 为何拆 4-stage：`workspace/pypto-lib/docs/step3p5/phases/15-singlerank-npu.md`

---

## 9. 复查落地：context-split 首个 device 结果（2026-07-30）

上一版结论“attention kernel 本体已无可落地的延迟优化”需要修正。B/D
per-block fusion 的回退，只能说明**在原 batch-only core mapping 下融合不合适**；
它没有否定 context 轴并行化。

### 9.1 实现

独立 worktree branch `perf/step3p5-attn-context-split`，commit `5ef8a517`：

- 只改 FULL decode attention；
- 保留原四阶段 split、GM scratch、BF16 PV 输入、Stage4 online-softmax 顺序；
- QK / softmax / SV 三个 stage 从 `pl.spmd(BATCH)` 改成固定
  `pl.spmd(FULL_ATTN_CTX_LANES=24)`；
- 每个 lane 对每个 active batch row 处理
  `sb = lane + n * 24`；
- 24 lane 不足当前 context 时通过静态 guard 跳过；
- 没有改变 KV 物理 block（仍为 128 token），也没有改变 cube 的 16-row
  boxed tile 合同；
- Stage4 仍按 context block 顺序串行 merge，因此该版本是**最小并行分解
  probe**，不是最终 partial-per-core 方案。

这个版本刻意没有引入新的 `(m,l,o)` per-core partial ABI，目的是先隔离
“context 轴没有并行化”这一变量。

### 9.2 验证

- Python syntax / AST contract：通过；
- `test_attention_full_runtime_active_bound.py`：`3 passed`；
- canonical whole-net hidden CI（8 cards，skip MTP）：`SINGLE_CHIP_HIDDEN_CI=PASS`；
- 无 507018、无 stall，token/hidden gate 通过。

同环境 perf-h1 A/B（cards 0-7，20 iters，warmup 3）：

| context | baseline p50 | context-split p50 | delta |
|---:|---:|---:|---:|
| 1,024 | 50.73 ms | 51.09 ms | **+0.36 ms / +0.7%** |
| 65,536 | 63.44 ms | 53.24 ms | **−10.20 ms / −16.1%** |

64k 结果表明：原实现确实存在 context 轴串行瓶颈；把长 context block
条带分配到 24 lane 后，虽然保留 GM scratch 和 Stage4 merge，仍获得了
明显收益。1k 的轻微回退说明在短 context 下 24-lane dispatch / 重复 runtime
循环的固定开销超过了并行收益，不能对所有 context 无条件启用。

### 9.3 当前结论和后续

新的性能结论应改为：

> **FULL decode 的首要延迟问题是 batch-only core mapping。**
> 在 bs=1、长 context 下，应优先做 context-split；Stage1+2+3 fusion
> 不是当前首选，因其会把 cube boxed 输出重新带入 16-row softmax。

建议下一步：

1. 增加按 context 的 dispatch threshold：短 context 保留旧 batch-only
   路径，长 context 使用 24-lane path；
2. 扫描 `NUM_CTX_LANES = 4/8/12/16/24`，确定 context 长度和 active batch
   的选择函数；
3. 在 24-lane 版本上把多个 block 的 `(m,l,o)` 收缩为 per-core partial，
   再做小型 merge，减少当前 per-block scratch 和 Stage4 GM 读取；
4. 最后再做 KV block grouping（建议 1/2/4 个物理 block/task）和
   MTE/cube/Vec 双缓冲流水；
5. SWA 暂不照搬：window 只有 4 blocks，context-split 的并行收益不足，
   仍应保留 batch-oriented path。

---

## 10. 历史 clean candidate 实现与发布状态（2026-08-02）

本节曾覆盖 §0–§9；当前发布状态再由 §12 的 2026-08-03 Wave5 结果覆盖。

### 10.1 当时实现

- 不再固定 24 个物理核心。各 stage 的 logical task 数由 active rows、每行真实
  `seq_len` 和 architecture profile grain 推导，runtime 再映射到 AIC/AIV wave。
- `5–10 us/task` 只是 sweep 起点。最终选择联合考虑 task duration、stage span、wave、
  packing、tail、dispatch、归约依赖链与 batch16。
- A2A3 默认：Full QK `22 blocks/task`、block-softmax `12 blocks/task`、
  SV+segment recurrence `16 blocks/task`、reduce fan-in `8`。
- Full 已把历史 Pass-A 合入 `full_sv_matmul`；该版本只保留跨 task 必需的
  `full_online_softmax_reduce` 与 per-row `full_online_softmax_finalize`。
  该版本无 `full_online_softmax_pass_a/pass_b/pass_c`。
- Full/SWA out-proj 各自保留独立 profile，当时默认均为
  `matmul N=64`、`tiles/task=3`、`vector N=128`、`cast fusion=1`；
  在默认 decode 配置下不会生成 standalone `full_out_proj_cast` /
  `swa_out_proj_cast`。源码仍保留 `FUSE_CAST=0` 的 fallback 分支；prefill
  路径也仍有独立的 `prefill_full_out_proj_cast` /
  `prefill_swa_out_proj_cast`，因此不能把“默认 decode graph 无该 kernel”
  扩写成“整个仓库没有这些符号”。
- TP all-reduce 保留 reduce-scatter + push all-gather，transfer chunk 为 512；
  residual epilogue 不复用通信粒度。dense RMSNorm direct BF16 reread 与 dense
  down-proj cast fusion 保留；AR+residual、residual+RMS stats、RMS+projection、
  gate/up+SiLU 等无稳定收益方案不合入。

当时源码：

```text
pypto-lib stepfun/develop
  76d96bdbeac280f12ecf626b1bbd722b9278719e

pypto stepfun/develop
  defa97c526fec7e8f032dbbfcc39c820add02bf7
```

后者修复 workload-derived 动态 SPMD launch bound 在 orchestration codegen 中的变量
重命名/声明问题。实现保持 PyPTO 分层：Orchestration 构建 logical task DAG，InCore
只执行自己的 tile/segment；没有新增 app-side persistent work-stealing loop。

### 10.2 batch16 与 Full/SWA 边界

`BATCH=16` 是 storage capacity，不是永久 logical batch。active-batch=16、ctx=1 已验证
所有 active hidden finite/nonzero 且 TP spread=0；异构 16-row context 已验证 task 数按
各行 workload 求和。uniform batch16/64K 的 online grain 单轮结果中，16 与 24 仅差
约 0.17%，不足以把 batch-aware 分支硬编码进数学语义。

Full 的长 context 需要 context split 和层次归约；SWA 最多 4 个 KV blocks，保持每个
active row 一个高密度 task，不机械复制 Full 的 reduction graph。

### 10.3 clean canonical candidate

```text
image:
  hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260802-attn-final-canonical

manifest:
  sha256:64c573bcf64497da6df0d3d28d7de85dfddde8e2a2a1b70e8bd5123edd51cb9d

config/image ID:
  sha256:c7f612a2562e932908d2a0d9ffadd1a1bd155c70bff0e82c24be32ef6b9f79ea
```

镜像演进：

1. `attn-final`：缺动态 SPMD codegen 修复，immutable 编译失败；
2. `attn-final-v2`：代码可执行，但 image config 仍含旧 CANN 8.5.1 字符串；
3. `attn-final-canonical`：显式清理 `PATH/PYTHONPATH/CMAKE_PREFIX_PATH`，镜像
   config 与内容 audit 全绿。

canonical 已通过：

```text
IMAGE_CONFIG_CANN_851_AUDIT=PASS
IMAGE_WORKTREE_CLEAN_AUDIT=PASS
IMAGE_GIT_CREDENTIAL_AUDIT=PASS
CANONICAL_ONLY_AUDIT=PASS
CANN_851_RUNTIME_AUDIT=PASS
EXPECTED_OPTIMIZATION_SYMBOL_AUDIT=PASS
PTOAS_LDD_AUDIT=PASS
smoke=PASS
```

验证为 immutable image：只挂 driver(ro)、checkpoint(ro)、output(rw)，没有挂载宿主源码。
本轮只使用 0162 cards `0–7`；未操作 cards `8–15` 或其 PID
`2045390–2045397`，测试结束后 cards `0–7` 无残留进程。

### 10.4 最终 ITL、DFX 与发布 blocker

64K、bs=1、512 blocks、warmup=3、20 measured iterations：

```text
min  = 49.213 ms
mean = 50.568 ms
p50  = 50.563 ms
p99  = 52.537 ms
max  = 52.537 ms
```

结果路径：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_canonical_20260802/itl64k/itl_report.json
```

DFX 必须使用 rank2 作为本轮 LOW-WAIT reference：

```text
0162:/mnt/persist/chensiyu/workspace/attn-opt/out/
  image_attn_final_canonical_20260802/itl64k/build_output/
  WholeDecodeStep3p5_20260802_162729/dfx_outputs/rank2/d0/
    critical_path_report.md
    merged_swimlane_20260802_162823.json
```

rank2 makespan 为 `38.924 ms`，其中 `tp_all_reduce` critical-path compute 为
`2.049 ms`。rank4–7 的约 `371–381 ms` makespan 主要是 collective 自旋等待被记为
compute；例如 rank5 的 `tp_all_reduce` critical-path compute 为 `344.553 ms`。
因此 rank5 可用于观察完整 collective span，但不能称为 LOW-WAIT reference。

该历史版本当时未通过发布门禁。同一 fresh oracle 上 canonical 三轮均为：

```text
121/128 = 94.53125% < 95%
```

三轮 miss 分别为：

```text
run1 [2,8,13,15,22,82,93]；TP spread=0
run2 [2,8,15,22,29,82,93]；step39 spread=0.953125
run3 [2,8,13,14,22,82,93]；step68/70 spread=1.1875/3.25
```

所有 hidden finite。不能借用 v2 的历史 `123/128` 宣称 clean canonical PASS，也不应
无限重跑直到偶然过线。该历史镜像当时是
**clean canonical candidate / release blocked**；源码合入、audit/smoke、ITL、DFX
已收尾，但当时正式发布仍需先闭环 raw precision/collective 非确定性。当前发布结论
以 §12 Wave5 为准。

完整记录见
[`../../benchmark/2026-08-02-step3p5-attention-final.md`](../../benchmark/2026-08-02-step3p5-attention-final.md)。


## 11. Wave3/Wave4 communication-window lifetime 收口（2026-08-03）

§10 的 `76d96bdb` clean candidate 作为历史基线保留。后续发现 Wave 2 只发布 push
完成，rank 在 final local copy 后立即返回时，peer 仍可能读取将被下一 collective 复用的
window。最小修复 `d58b6be7` 在 final copy 后增加 Wave 3 completion barrier；
`d7e1381b` 再把 two-layer harness 与 canonical `tp_all_reduce` AST 对齐。

Wave3 immutable 为 `124/128`、TP spread=0；Wave4 两轮为 `122/128`（step2
spread=`2.0`）和 `123/128`（spread=0）。Wave4 64K p50 `50.204 ms`，LOW-WAIT
rank2 makespan `38.504 ms`、TP AR compute `2.125 ms`。因此 attention/Vec 优化本体
当时已收尾，raw token gate 已过，但 Wave4 正式发布仍等待 TP-spread 稳定性；
该历史 blocker 后续由 §12 Wave5 关闭。


## 12. Wave5 source-publication 稳定性收口（2026-08-03）

§11 的 Wave4 仍有一轮 step2 TP spread=`2.0`，说明 final-read lifetime 闭合并不能
自动证明 Wave 1 前的 source payload 已在 notify 前可靠发布。Wave5
`7099476b7c4f13112b159e237e7a64344803caf0` 做最小修复：把普通 local source store
改为 self-target synchronous TPUT，再进入既有三波协议。

```text
local source
-> self-target drained TPUT
-> Wave 1 source publication
-> rank-owned reduce-scatter
-> push all-gather
-> Wave 2 result publication
-> final local copy
-> Wave 3 lifetime close
```

该变化同步覆盖 canonical Main、selected MTP 与 two-layer harness，并显式保留
MTP input projection 的 all-reduce 返回值 lineage。不改变 rank ownership、固定 peer
顺序、单 FP32 accumulator、最终一次 BF16 cast、host ABI 或 task graph；也不新增
orchestration kernel。

Wave5 immutable release：

```text
image:
  hub.i.basemind.com/stepcast/vllm-pypto:
  stepfun-develop-20260803-attn-final-wave5

manifest:
  sha256:4acc77cdce05c40fff7fdbcedb5612fa49c2edc847a534c218389ddc08667b32

config:
  sha256:4f2539c17fe60e61062bd27d96082a707e581b81fe716208c1bca4139dfd7394
```

0162 immutable gate：

- Main N=128 预定义三轮均 `123/128=96.09375%`，miss
  `[2,8,13,22,82]`，hidden finite，TP spread=0；
- Main active-batch=16 为 `8/8 exact`、finite、128 个 active rank rows、
  TP spread=0；
- MTP batch1/batch16 两轮 token `[6178,410,303]`、pass rate 1.0、
  max diff 0、TP spread=0；
- 64K ITL p50 `49.796 ms`；batch16/context1 p50 `112.827 ms`；
- DFX LOW-WAIT heuristic rank2：64K makespan `38.367 ms`、TP AR compute
  `2.437 ms`；batch16 makespan `107.076 ms`、TP AR compute `2.429 ms`。

critical-path 工具会把 collective 内部自旋等待计入 kernel compute，因此其它 rank
的长 span 不能解释为 all-reduce 算术耗时。

发布判断：

```text
Wave5 canonical release
machine scope = 0162 release-qualified
```

当前证据支持 source publication/lifetime ordering 是 0162 的关键边界；没有跨所有
硬件的 bit-level 证明，因此不能写成 self-target TPUT 是所有架构的唯一根因。attention
task grain、Full/SWA cast、online-softmax 和 Vec 结论不变：不固定 24 核，`5–10 us`
仅为 sweep 起点，`BATCH=16` 仍只是 storage capacity，无稳定收益的
AR+residual/RMS/projection 融合不合 canonical。

完整证据见
[`../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md`](../../benchmark/2026-08-03-step3p5-wave5-allreduce-stability.md)。

## 13. 当前最终实现：task 切分与 tile profile（合并文档）

本节合并原 `attention/attention-tiling-and-partitioning.md`，作为当前实现的单一
设计入口。§0–§9 的早期实验、fixed-lane probe 和失败方案仍保留用于解释为什么
最终没有选择那些路线，但不能覆盖本节。

### 13.1 核心抽象：workload-derived tasks，而不是固定核心数

```text
logical_tasks(row, stage)
  = ceil(actual_work(row, stage) / architecture_profile_grain(stage))

total_tasks(stage)
  = sum(logical_tasks(row, stage) for row in active_rows)
```

PyPTO Orchestration 只构建 logical task DAG；runtime 根据目标架构的 AIC/AIV
资源和当前可调度状态形成一个或多个 wave。由此可同时支持：

- 让任务数接近一个或若干完整 wave；
- 在不同架构上使用不同 grain；
- 按 active batch 和每行真实 `seq_len` 生成任务；
- 将静态 `BATCH=16` 作为 storage capacity，而不是永久 logical batch。

**5–10 us/task 只是 sweep 起点，不是目标函数。** 最终选择必须联合比较：

```text
per-task duration
+ stage span
+ AIC/AIV wave 与 packing/core-wait
+ dispatch overhead
+ reduction/finalize dependency tail
+ batch16 / heterogeneous-context behavior
```

A2A3 上已经出现过 task body 接近 10 us 但因多一 wave 而慢于约 15–20 us
单-wave候选的情况。

### 13.2 shape、storage 与逻辑工作量

| 参数 | Full | SWA |
|---|---:|---:|
| TP-local Q heads | 8 | 12 |
| Q physical pad | 16 | 24/32（阶段相关） |
| `HEAD_DIM` | 128 | 128 |
| KV cache block | 128 tokens | 128 tokens |
| 最大有效 KV blocks | context 决定；64K 为 512 | window 512 tokens，即最多 4 |
| storage batch capacity | 默认 16；编译时可配置，已验证 32 | 默认 16；编译时可配置，已验证 32 |

必须区分：

1. storage block/tile；
2. logical task grain；
3. physical AIC/AIV mapping；
4. active workload；
5. static capacity。

两个 128-token storage blocks 被同一 task 连续处理，只表示调度 grouping；除非
实现还完成 operand gather、合法的新 tile 和对应 lowering，否则不能称为
256-token fused matmul。

### 13.3 Full attention 最终任务图

```text
full_qk_matmul
-> full_softmax
-> full_sv_matmul                  # SV + segment-local recurrence
-> full_online_softmax_reduce      # write-disjoint group reduction
-> full_online_softmax_finalize    # per-row merge/normalize/BF16 store
-> full_out_proj_matmul_{aic,aiv}  # FP32 accumulator -> BF16 cast fused
```

默认 decode 配置不会生成：

```text
full_online_softmax_pass_a
full_online_softmax_pass_b
full_online_softmax_pass_c
full_out_proj_cast
```

`FUSE_CAST=0` fallback 与 prefill 的独立 cast 路径仍可存在；“默认 decode graph
没有 standalone cast”不能扩写成“仓库内删除了所有相关符号”。

#### 13.3.1 QK / block-softmax

task 映射到 `(batch_row, task_in_row, block_start)`。每个 active row 先由真实
`seq_len` 得到 `context_blocks`，再按 stage grain 计算 task 数；短 row 不按最大
context 补齐无效任务。

A2A3 默认：

```text
QK blocks/task       = 22
softmax blocks/task  = 12
```

64K、batch1 时：

```text
QK       ceil(512 / 22) = 24 logical tasks
softmax  ceil(512 / 12) = 43 logical tasks
```

这里的 `24` 是 workload 与 grain 的结果，不是源码固定使用 24 个物理核心。
QK 使用 AIC、softmax 使用 AIV，wave 必须按两类资源分别计算。

#### 13.3.2 SV 与 online-softmax

`full_sv_matmul` 同时完成：

1. 每个 KV block 的 `P @ V`；
2. 同一 task 所拥有 segment 内的 `(m,l,o)` recurrence；
3. 写出一个 segment partial。

因此历史 Pass-A 已消失。A2A3 profile：

```text
SV + segment recurrence blocks/task = 22
reduce fan-in                        = 8
```

这里的 22 是显式 `a2a3` profile；默认 `portable` fallback 仍为 16。两者都按
实际 workload 计算 task 数，不表示固定使用 22 或 16 个物理核心。

两个后继 kernel 仍需保留：

- `full_online_softmax_reduce`：不同 SV task 的 segment partial 需要跨 task
  合并；每个 reduce task 只写自己的 destination；
- `full_online_softmax_finalize`：每个 active row 合并 group outputs，
  normalize、flatten，并完成 FP32→BF16 最终 store。

若机械并入所有 SV task，要么多个 task 并发写同一 row，要么退回单 task 串行
消费整行。这两者都比保留明确的 RAW/liveness 边界更差。

### 13.4 SWA 保持不同结构

SWA 的有效 window 最多 4 个 KV blocks。当前每个 active row 是一个 logical task，
task 内顺序处理完整 window：

```text
swa_qk_matmul -> swa_softmax -> swa_sv_matmul -> swa_online_softmax
```

`swa_online_softmax` 的代表性执行约 `2.9–3.2 us`；进一步拆层次归约会增加
scratch、dispatch 和依赖。Full 与 SWA 不应机械共用 task graph：

- Full：长 context、多 work items，适合 context split 和层次归约；
- SWA：window 很短，优先保持每 active row 的任务密度。

### 13.5 out-proj 参数与 cast fusion

Full/SWA 保留独立 profile knob；当前 A2A3 默认恰好相同：

```text
matmul N tile        = 64
matmul tiles/task    = 3
vector N             = 128
cast fusion          = 1
```

`4096/64=64` 个合法 N tiles 按每 task 3 tiles 分成 22 个 logical tasks，
接近 A2A3 一个 AIC wave；这是校准结果，不是数学约束。

cast 融合后的数据流：

```text
FP32 matmul accumulator
-> same mixed task AIV cast
-> BF16 partial output
```

独立开关仍保留，便于新架构发现 mixed-kernel 不合适时回退。

### 13.6 active batch=1–32 与异构 context

默认 `BATCH=16` 只决定 tensor/ABI capacity。logical tasks 由 `active_tokens` 和
每行 `seq_lens` 推导；inactive rows 不参与 attention/KV metadata 工作。batch32
使用 compile-time capacity=32，不在 capacity=16 的二进制上越界运行。

除 active-batch=16、ctx=1 和异构 16-row `ceil` 求和外，最终显式 A2A3 profile
还完成了 **fixed-total-context=64K** 验证：

| active batch | per-row context | 两层 p50 |
|---:|---:|---:|
| 1 | 65536 | 3.7839 ms |
| 4 | 16384 | 3.7675 ms |
| 8 | 8192 | 3.7599 ms |
| 12 | 5504 / 5376 | 3.8710 ms |
| 16 | 4096 | 3.9480 ms |
| 32 | 2048 | 4.8368 ms |

所有点 exact、finite、TP spread=0，逐迭代输出 `unique_count=1`。batch16 的 200
轮稳定性结果为 p50 `3.9192 ms`、p99 `7.4785 ms`、max `11.7102 ms`。少量长尾
是系统/collective 到达抖动信号，不足以新增 batch-aware 数学路径。

### 13.7 all-reduce 与 Vec 邻接优化边界

最终 Wave5 all-reduce：

```text
self-target TPUT source publication
-> Wave 1 notify/wait
-> rank-owned reduce-scatter
-> push all-gather
-> Wave 2 notify/wait
-> final local copy
-> Wave 3 notify/wait
```

transfer grain=512 是通信 profile，不应机械继承给 residual Vec epilogue。已验证：

- producer 直接写 AR window：正确但无稳定收益，不合入；
- AR final copy + residual：512 grain 变慢，128 grain 仅噪声级，不合入；
- residual + RMS stats：多个粒度均变慢，不合入；
- dense RMSNorm direct BF16 reread：保留；
- dense down-proj cast fusion：保留；
- gate/up + SiLU、RMSNorm + projection：只保留 probe。

原则是：**能融合不等于应该融合**。必须同时满足 correctness、稳定收益、
资源映射、batch16 与最小改动。

### 13.8 PyPTO 架构边界

- Orchestration 由 runtime scalar 构建 logical task DAG 与 dependency；
- InCore task 只执行自己的 tile/segment；
- runtime 决定 logical tasks 到物理核和 wave 的映射；
- task-grain 参数属于 architecture profile，不进入模型数学语义；
- 当前没有 app-side persistent work-stealing loop。

“每个核拉取 5–10 us task”的核心效果已经由 logical task scheduler + runtime wave
dispatch 覆盖。真正的 persistent worker/device-side work queue 需要修改 runtime ABI；
当前证据不支持为了它替换现有模式。

### 13.9 portable / A2A3 profile 与跨架构校准

```text
                              portable(default)  a2a3(explicit)
Full QK blocks/task                    22              22
Full block-softmax blocks/task         12              12
Full SV+segment recurrence blocks/task 16              22
Full online reduce fan-in               8               8
four uniform O(1) mappings              0               1
```

两者当前共享的 out-proj/collective 默认值：

```text
Full/SWA out-proj matmul N             = 64
Full/SWA out-proj tiles/task           = 3
Full/SWA vector N                      = 128
Full/SWA out-proj cast fusion          = 1
TP all-reduce transfer chunk           = 512
```

`--platform a2a3` 不隐式选择 attention profile。0162 A2A3 运行必须显式设置：

```text
PYPTO_STEP3P5_ATTN_TASK_PROFILE=a2a3
```

八个 QK/softmax/online/reduce 单项 override 的优先级高于 profile；做可信 profile
A/B 前必须清除它们，避免得到未命名的混合配置。

新架构必须重新 sweep。建议目标：

```text
minimize:
  total critical-path stage span
  + extra-wave/core-wait/dispatch cost
  + reduction/finalize tail

subject to:
  logical-task counter limit
  legal cube/vector tile and UB/L1 budget
  active-batch/capacity correctness
  finite + TP consistency
  canonical precision gate
```

### 13.10 lowering、DFX 与负面候选门禁

权威源码：

```text
pypto-lib stepfun/develop
91c7f46ee949045e2fce807276412b48d8121763

pypto stepfun/develop
defa97c526fec7e8f032dbbfcc39c820add02bf7
```

最终源码验证为 `218 passed, 3 skipped`。compile-only 还会读取真实生成的
`orchestration/chip_orch.cpp`，检查 Full/SWA 各 stage 的动态 launch bound、
launch/scalar SSA 一致性、TaskId publication 和完整 dependency chain；checker
必须返回空列表，不能只用“编译成功”替代 lowering 合同。

最新 bs16/64K DFX 的 LOW-WAIT reference：

```text
/mnt/persist/chensiyu/workspace/attn-opt/out/
attn_a2a3_profile_64k_bs16_dfx_8_15_50x_20260805/
build_output/TwoLayerAttnPerf_20260805_072659/
dfx_outputs/rank2/d0/l2_swimlane_records.json
```

rank2 makespan 为 `1055.5 us`，四次 TP all-reduce 的 critical-path span 合计约
`185.24 us`。其它 rank 的数百毫秒 collective 主要是 peer-arrival spin wait 被
计入 kernel duration。

Full/SWA RoPE packed staging 保持 **NO-GO**：bs12 独立 40 轮出现
`unique_count=40` 和明显数值错误，且早期速度数据未包含最终 TaskId chain。
在最终图上完成隔离 A/B 和逐迭代稳定性证明前，不得恢复该候选。

### 13.11 Attention bubble 收尾判断与后续优化方向

#### 当前判断

针对当前 Step3p5、0162/A2A3 profile 和已验证的 batch/context 范围：

> **Full/SWA attention 核心计算中主要的、可避免的调度 bubble 已经闭环；
> 但不能把该结论扩写为“swimlane 上不应再有任何空白”或“所有架构均已最优”。**

最新 bs16/fixed-total-context=64K 的 rank2 LOW-WAIT DFX 中，Full 主链为：

| stage | stage span |
|---|---:|
| `full_qk_matmul` | `24.16 us` |
| `full_softmax` | `15.50 us` |
| `full_sv_matmul`（含 segment-local recurrence） | `37.28 us` |
| `full_online_softmax_reduce` | `4.02 us` |
| `full_online_softmax_finalize` | `3.72 us` |
| `full_out_proj_matmul`（cast fused） | `24.84 us` |

旧图中的 standalone `full_online_softmax` 百微秒级 span、Pass-A/B/C 和独立
decode out-proj cast 已不再存在。observed critical path 的 runtime stall 为
`233 us`，全部分类为 `data-wait`；没有 `core-wait` 或 `front-gap`。这说明此前由
错误 task mapping、stage chain 不完整或 dispatch packing 引起的主要调度空洞已消除。

#### 仍可见但不应直接视为缺陷的空白

1. **bs16 的不满尾 wave**：每行 32 个 KV blocks，在 grain=22 时 QK/SV 合计
   32 个 logical tasks，映射到 24 AIC 必然形成第二个不满 wave。rank2 的 QK/SV
   AIC packing efficiency 分别约 `54.7%` / `61.0%`。更粗 grain 的单-wave
   候选已经 sweep，未得到稳定 wall-time 收益；因此当前不增加 batch-specific
   数学或调度分支。
2. **reduce/finalize 的低占用**：两个 stage 均只有 16 个 per-row tasks，无法铺满
   48 AIV，但 span 仅约 `4.02 us` / `3.72 us`。它们承担跨 segment partial 的
   write-disjoint reduction 和最终 normalize/store，是必要 RAW/liveness 边界。
3. **stage 间 data-wait**：QK、softmax、SV、reduce、finalize 的显式 TaskId chain
   会在图上形成依赖等待。机械删除依赖会恢复并发写 race 或读取未完成 scratch；
   这种等待不能仅凭 swimlane 空白判定为可优化 bubble。

因此评审 bubble 时必须同时检查 task count、AIC/AIV wave、packing、stall kind 和
端到端 wall time；不能用“核心未全满”或“kernel 之间有空白”作为单独合入依据。

#### 后续优化优先级

1. **P0：immutable 镜像复核，而不是继续改 attention 数学。**
   最终 canonical 镜像必须以 digest-only 方式复采同一 workload 的 ITL/DFX，
   确认 source-mounted focused 结果没有被镜像环境、profile 或 override 改写。
2. **P1：RoPE + KV-cache staging。**
   当前两层 observed critical path 中，Full/SWA `rope_kv_cache` compute 分别约
   `128 us` / `120 us`，已比单个 QK/softmax/online stage 更突出。它是 attention
   邻接路径中最值得继续研究的方向；但现有 packed-staging 候选有 bs12
   `40/40` 轮输出不唯一和明显数值错误，必须先闭环 ownership、lifetime、逐迭代
   determinism 和 canonical precision，不能直接恢复。
3. **P2：按真实服务分布重新校准 architecture/workload profile。**
   只有当某个 batch/context 分布占主导，且交替多轮 A/B 显示收益稳定超过噪声和
   维护成本时，才考虑新增 profile；不要为了消除 bs16 的可视尾 wave 写入模型语义。
4. **P3：跨 stage producer-consumer pipeline。**
   理论上可让部分 QK 完成后提前启动对应 softmax/SV，但必须解决不同 grain 的
   dependency mapping、scratch lifetime 和 write ownership。历史 QK+softmax+PV
   融合受 cube boxed 16-row softmax 限制，曾回退约 `5%–7%`；后续方案不能简单
   恢复该融合形态。
5. **P4：attention 邻接 collective/overlap。**
   最新两层 LOW-WAIT DFX 中四次 TP all-reduce critical-path span 合计约
   `185 us`，已高于任一单独 attention core stage。若目标是整网 ITL，应优先评估
   通信到达和安全 overlap；但不得破坏现有 source-publication、三波 lifetime
   close、rank ownership 和固定 reduction order。

在新的可信 A/B 证据出现前，当前 canonical 建议为：**attention 核心图收尾，
RoPE/KV-cache 作为独立后续任务，整网继续优先看 collective 与可证明的 overlap。**
