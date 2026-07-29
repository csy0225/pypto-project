# 04 · Attention 优化专项（step3p5 full / SWA flash decode）

> **性质**：LLD 专项。聚焦 decode 阶段 flash-attention kernel 本体（`attention_full.py` /
> `attention_swa.py`）的重写路线，独立于 README 主表里的 Track A–H。收敛后其中的子项会以
> `PERF-*` ID 回填 [`task-tracking.md`](task-tracking.md)。
>
> **验证基线**：pypto-lib `perf/step3p5-bc-20260726 @ 4513007d`，canonical Main =
> `models/step3p5/decode_fwd.py:whole_decode_step3p5`（loop-form，B2 后）。
> 本文所有 file:line 均对当前 source 逐条核对（2026-07-29）。
>
> **审计口径**（沿用 [`README.md` 顶层审计方法](README.md)）：任何“step3p5 独有 / 必须保留”
> 判断都要沿 `producer → 数学变换 → transport → consumer → rounding/reduction → lifetime`
> 核对；V4-Flash 已有同构能力时，shape 不同只算参数化/存储适配，不升级为架构差异。

---

## 0. 一句话结论 + 边界

当前 full/SWA 的 **decode** flash 是 **4-stage split** 形态（QK / softmax / SV / online-softmax
各一个 `pl.spmd(BATCH)`，中间用 GM scratch 串联）。它能跑通、精度对齐（N=256 hidden exact），
但有**一个真实浪费 + 一个被证伪的假设**：GM scratch 按静态 capacity 与最大 context 预分配
（真实，见 §2.1）；SV stage 读 16 行只用 8（曾以为可砍半，但 device 证明 cube 16-fractal 下
不可 trim、也无 cube 收益，见 §2.2/§5.2）。V4-Flash（当前源码位于 `models/deepseek/v4/`）**是 MLA + sparse attention，
没有可直接照搬的 full-attention decode 结构**——能借的是它的 lowering 写法（additive-inf
bias、qk_pv 融合、FP32 直接 row_sum），不是它的 kernel 结构。它跑通的 shape 是
`H=64 / HEAD_DIM=512 / H_TILE=16`，与 step3p5 的 `H=8(或12) / HEAD_DIM=128 / pad=16(或24)`
差很远，**任何搬移都必须过 ST**。

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
  bs=16 OOM）。**修法 B'**：把 bias 构造/应用 guard 到 `valid_len < BLOCK_SIZE` 只在 partial
  block 执行 → 预期恢复延迟、保留显存收益（待验证 pypto 是否允许 per-block runtime `if`）。

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
| **D** | UB 滚动累加器，partial 不物化 | 连 `all_oi_tmp` 也消，Stage4 并入 | **~161 MiB/FULL 层 → KB 级**（最大显存收益） | 同 B，待测 | ⏸ 待做 |

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
