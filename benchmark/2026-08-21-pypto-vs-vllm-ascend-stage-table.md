# PyPTO vs vLLM-Ascend 逐阶段时间对比表（2026-08-21）

> **一句话**：PyPTO 单层 wall 与 vLLM-Ascend 同类层同量级（full-MoE 层 `458.82` vs
> `408.50 us`，**1.12×**）。差距**不在 expert 计算**（GMM1 `41.72` vs `42.52 us` 已经打平），
> 而集中在 **EP combine 通信**（独占 `55.95 us`，vLLM 无对应物）与 **full attention**
> （独占 `90.33 us`，占 PyPTO 整层 19.7%）。
>
> ⚠ **硬边界（先读，否则会误用本表）**：两侧**不是同一 workload** —— PyPTO 是
> BS1 / ctx 64K 的 5 层 synthetic harness，vLLM 是 BS4 / **ctx 不可知**的真实 service
> trace。**attention 与 AllReduce 两行的绝对比值不成立**，只有结构占比可读。详见 §6。

---

## 1. Provenance

| 项 | 值 |
|---|---|
| 执行主机 | `gpu-a910x-0162`（全部解析就地执行，未占卡、未重采） |
| 工作目录 | `/mnt/persist/chensiyu/workspace/perf-2026q3/vllm-pypto-stage-table-20260821/` |
| **PyPTO 输入** | R5 生产基线 packed-NZ 五层 DFX：`.../moe-routed-packed-fusion-20260815/dfx-packed-nz-architecture-20260817-213730/out/runtime/build_output/FiveLayerMoe_20260817_134105/dfx_outputs/rank{0..7}/d0/merged_swimlane_20260817_1342*.json` |
| PyPTO 配置 | `PYPTO_STEP3P5_MAX_SEQ=65536` `ROPE_SEQ=65536` `STORAGE_BATCH_CAPACITY=16`，TP8/EP8，**BS1**，8 rank 各 1 次 invocation |
| rank2 swimlane sha256 | `f9ef1dbe51d9867d9f981b6bf6da9b5b1d5446ca08fbdfcfccaa8c513efdf013` |
| **vLLM 输入** | `/mnt/persist/chensiyu/workspace/develop/trace_view (3).json`，sha256 `ddba08673aa3787147c261403223995b2b345219911317c9f50c05215d65abf2` |
| vLLM 规模 | 68 个 decode step × 45 层 Main（Model 87），TP8，**BS4**（由 `hcom count=16384 = 4×4096` 推出） |
| 分析脚本 | `pypto/pypto_v2.py` `822156f8…`、`pypto/excl.py`、`vllm/vllm_stages2.py` `eec7826e…`、`vllm/agg3.py` `82874455…` |
| 机读产物 | `pypto/pypto_v2_raw.json` `af5c9aa4…`、`vllm/vllm_stage_table.json` `a9c372a5…` |

### 层身份判定（两侧都是独立核实过的，不靠名字猜）

- **PyPTO**：层归属**只能靠 `rNtM` task-id 区间**，名字前缀不可靠（L1/L2 的 attention
  都叫 `swa_*` 无前缀，L4 的 MoE 也无前缀）。区间：ring2 `t0-21`=L0、`t22-39`=L1、
  `t40-58`=L2、`t59-94`=L3、`t95-127`=L4；ring3 每 3 个 task 一层（out_proj + AR）。
  五层 = L0 full+dense、L1/L2 SWA+dense、L3 SWA+MoE、L4 full+MoE。**全部 task 归桶，0 未分类。**
- **vLLM**：trace 内无 literal `decode` marker。用 replay 内**稳定的 `Task Id`**（每 replay
  重置，68 个 replay 每个恰好 1430 事件）+ 每层 2 个 `GemmaRmsNorm` 作层边界，
  阶段按 anchor 之间的 taskid 区间归桶（不靠 kernel 名，因为 `MatMulV2_…98513`
  同时被 QKV、head-gate、dense-MLP 复用）。
- **attention 变体自动分离出 12 full + 33 SWA**，与 step3p5 层表
  （`full_dense×1 + full_moe×10 + full_moe_swiglu7_swiglu16×1 = 12` full，
  `swa_dense×2 + swa_moe×30 + swa_moe_swiglu7_silu×1 = 33` SWA）**完全吻合** ——
  这是层识别正确性的独立佐证。

### 交叉验证（全部通过）

| 量 | 本次算出 | 既有报告 | 判定 |
|---|---:|---:|---|
| vLLM `GroupedMatmulSwigluQuant` p50 | `43.98`（regular 全体） | `43.98` | ✅ 逐位一致 |
| vLLM `GroupedMatmul`(down) p50 | `28.80` | `28.80` | ✅ 逐位一致 |
| vLLM MoE-输出 AR p50 / p95 | `23.08` / `50.84` | `22.281` / `50.682` | ✅ 一致（既有把 40+2 层混算） |
| vLLM `route_org` = init_routing+Cumsum | `40.16` | `11.52+28.19=39.71` | ✅ |
| PyPTO packed GMM1 span | `41.72`(L4) / `40.90`(L3) | `41.91` | ✅ |
| PyPTO packed down span | `13.36`(L4) / `14.53`(L3) | `14.30` | ✅ |
| PyPTO local MoE envelope | `249.06`(L4 router→residual) | `250.92` | ✅ |
| PyPTO shared 分支 envelope | `59.04` | `58.75` | ✅ |
| vLLM 45 层 span 之和 | `16.742 ms/step` | Main span `18.067 ms` | ✅ 差 `1.33 ms` = 层间 gap + embedding/final-norm/LM-head |

未对上的 1 项（已查明是口径差异，非 bug）：既有报告 "down end → combine-scatter start
`5.58 us`"，本次测到**负值**（`combine_scatter` 在 `routed_down` 结束前就起来了，两者重叠）。
既有那个数应为逐 expert 或单 rank 口径。

---

## 2. 两侧计量口径不同 —— 必须先理解，否则表会读错

| | PyPTO | vLLM-Ascend |
|---|---|---|
| 执行模型 | persistent `@pl.program` **DAG**，24 AIC + 48 AIV 核并行，shared/routed/comm 是可重叠 lane | 单 stream **串行**下发 kernel，AR 在独立 comm stream |
| 阶段度量 | ① `union_wall` = 该大项区间并集<br>② **`独占` = 该区间内没有任何其他阶段在跑的部分**<br>③ `busy` = 核占用（core-µs，跨核累加） | `sum` = 该阶段所有 kernel duration 之和（串行 ⇒ sum ≈ wall） |
| 百分比之和 | **不等于 100%**（阶段重叠）；`独占` 之和 < 100%，缺口 = 纯调度 gap | ≈ 96–98%，缺口 = kernel 间 gap |
| AllReduce 口径 | `tp_all_reduce` task 的 wall（含跨卡等待） | ① 计算流 `CAPTURE_WAIT` 停顿；② comm 流 `hcom_allReduce` duration |

> **可比的列**：PyPTO `独占` ↔ vLLM `sum`。两者都是"这个阶段在关键路径上花了多少"。
> **不可比的列**：PyPTO `busy`（core-µs）不能和 vLLM 的 wall 比 —— 它是 24+48 核的累加。

---

## 3. ★ 大项总览（一眼看对比）

### 3.1 full-attention + MoE 层 —— PyPTO **L4** vs vLLM **L4/8/…/40**（10 层）

PyPTO layer wall = **`458.82 us`**；vLLM layer wall = **`408.50 us`** ⇒ **PyPTO 1.12×**

| # | 大项 | PyPTO 独占 | PyPTO 占比 | vLLM sum | vLLM 占比 | 独占比值 | 读法 |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | pre-attn | `26.76` | 5.8% | `47.80` | 11.7% | **0.56×** | PyPTO 快（norm/QKV/RoPE 与 head-gate 并行掉了） |
| 2 | **attn** | **`90.33`** | **19.7%** | `76.52` | 18.7% | 1.18× | ⚠ 见 §6，batch/ctx 不同，比值不成立 |
| 3 | post-attn | `28.17` | 6.1% | `34.04` | 8.3% | 0.83× | PyPTO 快（head-gate 完全并行掉） |
| 4 | all-reduce | `42.03` | 9.2% | `43.97` | 10.8% | 0.96× | 打平 |
| 5 | pre-MoE | `17.07` | 3.7% | `41.00` | 10.0% | **0.42×** | PyPTO 显著快（router 与 norm 并行） |
| 6 | dispatch | **`0.00`** | **0.0%** | `42.79` | 10.5% | **0×** | ★ PyPTO 的 EP dispatch **完全被藏进 shared expert 之后**，零关键路径成本 |
| 7 | MoE experts | `51.22` | 11.2% | `94.84` | 23.2% | **0.54×** | PyPTO 快（shared 与 dispatch 重叠） |
| 8 | **combine** | **`72.33`** | **15.8%** | `12.80` | 3.1% | **5.65×** | ★★ **最大结构性劣势**，vLLM 本卡 unpermute，PyPTO 要跨卡 EP combine |
| — | 阶段独占合计 | `327.91` | 71.5% | `393.78` | 96.4% | | |
| — | 调度 gap | `130.91` | **28.5%** | `14.72` | 3.6% | | ★ PyPTO 有 `131 us` 谁都没在跑的空隙 |

**两条最大的可攻击项，按 ROI 排序**：

1. **调度 gap `130.91 us` = 整层 28.5%** —— 比任何单个 kernel 都大。是 19 个阶段边界的
   launch/依赖解析间隙累积（既有报告量化过的 "boundary tax"：dispatch-gather → GMM1
   `25.22 us`、GMM1 → down `6.57 us` 等）。vLLM 同口径只有 3.6%。
2. **EP combine `72.33 us` = 15.8%** —— 其中 `ep_combine_comm` 独占 `55.95 us`。
   这是 EP8 拓扑的**固有代价**，vLLM 的 local routing 根本不产生它；
   但 `63.05 us` 的 span 里只有 `49.93` core-µs 是真 busy ⇒ 大部分是跨卡等待，
   有 overlap 空间。

### 3.2 SWA + MoE 层 —— PyPTO **L3** vs vLLM **30 层 SWA MoE**

PyPTO layer wall = **`387.80 us`**；vLLM = **`371.00 us`** ⇒ **PyPTO 1.05×**

| # | 大项 | PyPTO 独占 | 占比 | vLLM sum | 占比 | 独占比值 |
|---|---|---:|---:|---:|---:|---:|
| 1 | pre-attn | `0.00` | 0.0% | `50.11` | 13.5% | **0×** |
| 2 | attn (SWA) | `0.00` | 0.0% | `30.28` | 8.2% | **0×** |
| 3 | post-attn | `47.11` | 12.1% | `40.90` | 11.0% | 1.15× |
| 4 | all-reduce | `15.09` | 3.9% | `39.79` | 10.7% | 0.38× |
| 5 | pre-MoE | `10.95` | 2.8% | `41.84` | 11.3% | 0.26× |
| 6 | dispatch | `0.00` | 0.0% | `39.79` | 10.7% | **0×** |
| 7 | MoE experts | `53.16` | 13.7% | `96.92` | 26.1% | 0.55× |
| 8 | **combine** | **`76.03`** | **19.6%** | `12.82` | 3.5% | **5.93×** |
| — | 独占合计 | `202.34` | 52.2% | `352.46` | 95.0% | |
| — | 调度 gap | `185.46` | **47.8%** | `18.54` | 5.0% | |

★ **SWA 层里 PyPTO 的 attention/pre-attn/dispatch 独占全部为 0** —— 整个 attention 前半段
被完全藏进其他 lane。代价是 **gap 涨到 47.8%**：SWA 层的计算量太小，DAG 的调度间隙
反而成了主导项。**这是 30 层（占模型 2/3）的形态，是全网最值得攻的地方。**

### 3.3 SWA + dense-MLP 层 —— PyPTO **L2** vs vLLM **L1/L2**

PyPTO layer wall = **`277.41 us`**；vLLM = **`194.75 us`** ⇒ **PyPTO 1.42×**（三类层里最差）

| # | 大项 | PyPTO 独占 | 占比 | vLLM sum | 占比 | 独占比值 |
|---|---|---:|---:|---:|---:|---:|
| 1 | pre-attn | `18.45` | 6.7% | `44.05` | 22.6% | 0.42× |
| 2 | attn (SWA) | `5.69` | 2.1% | `31.82` | 16.3% | 0.18× |
| 3 | post-attn | `36.75` | 13.2% | `37.15` | 19.1% | 0.99× |
| 4 | all-reduce | `47.83` | 17.2% | `25.81` | 13.3% | **1.85×** |
| 5 | pre-FFN norm | `18.53` | 6.7% | `4.00` | 2.1% | **4.63×** |
| 7 | dense MLP | `83.18` | 30.0% | `43.84` | 22.5% | **1.90×** |
| 8 | residual | `8.24` | 3.0% | `3.88` | 2.0% | 2.12× |
| — | 独占合计 | `218.67` | 78.8% | `190.56` | 97.8% | |
| — | 调度 gap | `58.74` | 21.2% | `4.19` | 2.2% | |

★ dense 层**没有 MoE 的可重叠 lane，所以一切都在关键路径上**：`dense_mlp` 1.90×、
两次 AR 1.85×、pre-FFN norm 4.63×。**只有 3 层，全网权重低**，但它是"PyPTO 在无 overlap
可用时的裸速度"的干净读数 —— 说明**当前优势主要来自 overlap，不是来自单 kernel 更快**。

---

## 4. ★ PyPTO L4 层内并行结构（缩进 = 同时在跑）

时间轴单位 µs，相对层起点；median of 8 ranks。`独占` = 该 task 区间内无其他阶段在跑。
（`start`/`end` 是逐 rank 取 median 后的值，所以 `end−start` 与 `union` 列可能差零点几 µs。）

```
时刻        阶段                          区间        span    独占     busy(core-µs)
──────────────────────────────────────────────────────────────────────────────────
  0.0  ┌─ ① pre-attn ∥ ③ head-gate  ── 并行块 wall 0.0→51.0 = 51.0 ────────────
       │  ①.input_norm               0.0→ 11.2    11.2    0.01        10.92
       │  ③.residual_hold            0.0→  2.8     2.7    0.00         2.38
       │  ①.qkv_proj                 2.2→ 41.9    39.6    4.28       275.75
       │  ③.head_gate               16.0→ 36.1    20.5    0.00 ←全被藏 105.99
       └  ①.rope + kv_cache         15.5→ 51.0    35.2    8.00         6.95
 49.7  ┌─ ② attention ── wall 49.7→141.3 (union 91.2) ─────────────────────────────
       │  ②.attn_core               49.7→125.6    75.7   70.90      4783.38
       └  ②.softmax_reduce         121.7→141.3    19.6   15.29        19.35
                                   〈gap 141.3→147.7 = 6.4〉
147.7     ③.o_proj                 147.7→175.5    28.2   28.17      1588.00
                                   〈gap 175.5→181.9 = 6.4〉
181.9     ④.AR(attn)               181.9→207.8    25.3   25.30        25.02
                                   〈gap 207.8→209.8 = 2.0〉
209.8  ┌─ ⑤ pre-MoE ∥ ⑦ shared ∥ ⑥ dispatch ∥ ④ AR ── 6 路并行 209.8→311.3 = 101.6 ─
       │  ⑤.ffn_norm              209.9→221.9    12.2    0.00 ←全被藏  11.96
       │  ⑤.router_topk           209.8→237.2    27.7    5.16       162.02
       │  ⑦.shared_expert         226.8→285.8    59.0    2.05 ←几乎全藏 212.98
       │  ⑥.route_org             239.4→267.6    28.1    0.00 ←全被藏  26.76
       │  ④.AR(shared/ffn)        244.0→311.3    67.1   16.61        23.63
       └  ⑥.ep_dispatch_comm      257.0→293.2    33.7    0.00 ←全被藏  57.51
                                   〈gap 311.3→318.5 = 7.2〉
318.5     ⑦.routed_gmm1           318.5→360.6    41.7   41.72      2364.35
                                   〈gap 360.6→367.2 = 6.6〉
367.2  ┌─ ⑦ down ∥ ⑧ combine 通信 ── 367.2→437.5 = 70.3 ──────────────────────
       │  ⑦.routed_down           367.2→380.9    13.4    6.52       518.10
       └  ⑧.ep_combine_comm       374.8→437.5    63.1   55.95        49.93
                                   〈gap 437.5→442.6 = 5.1〉
442.6     ⑧.merge + residual      442.6→458.8    16.3   16.34        45.14
──────────────────────────────────────────────────────────────────────────────────
层 wall = 458.82    阶段独占合计 = 327.91 (71.5%)    调度 gap 合计 = 130.91 (28.5%)
核占用合计 = 10 271.65 core-µs（24 AIC + 48 AIV；÷458.82 ⇒ 平均并行度 22.4 核）
```

**并行块读法**：`独占 = 0.00` 的阶段（head-gate、ffn_norm、route_org、ep_dispatch_comm）
**对 ITL 完全免费** —— 它们 100% 藏在别的阶段背后。优化它们的 ROI = 0。
真正付钱的是 `attn_core 70.90`、`ep_combine_comm 55.95`、`routed_gmm1 41.72`、
`o_proj 28.17`、`AR(attn) 25.30`、`merge 16.34`、`AR(ffn) 16.61`。

vLLM 侧对应结构是**一条直线**（单 stream 串行），唯一的并发是 AR：
计算流 `CAPTURE_WAIT` 停顿 `29.79 us` 而 comm 流 `hcom_allReduce` 是 `26.37 us`
⇒ AR 基本**没有**和计算重叠，是裸串行代价。

---

## 5. 小项级完整对照（full-MoE 层）

| 大项 | 小项 | PyPTO span | PyPTO 独占 | PyPTO busy | vLLM sum | vLLM 对应 kernel |
|---|---|---:|---:|---:|---:|---|
| ① pre-attn | input_norm | `11.22` | `0.01` | `10.92` | `10.36` | `GemmaRmsNorm` + Cast/Add |
| | qkv_proj | `39.64` | `4.28` | `275.75` | `16.86` | `MatMulV2_…98513` |
| | rope + kv_cache | `35.20` | `8.00` | `6.95` | `20.58` | `split_qkv_rmsnorm_rope` + `reshape_and_cache` |
| ② attn | attn_core | `75.66` | `70.90` | `4783.38` | `76.52` | `FusedInferAttentionScore`(5100…203) |
| | softmax_reduce | `19.60` | `15.29` | `19.35` | — | 已融进上面那个 kernel |
| ③ post-attn | head_gate | `20.52` | `0.00` | `105.99` | `16.64` | `MatMulV2` + `Sigmoid` + `Mul` |
| | o_proj | `28.17` | `28.17` | `1588.00` | `13.94` | `MatMulV2_…98499` |
| | residual | `2.66` | `0.00` | `2.38` | `3.46` | `Add`（PyPTO 的加法融进了 AR task） |
| ④ all-reduce | ar_attn | `25.30` | `25.30` | `25.02` | `14.18` / comm `13.23` | `CAPTURE_WAIT` / `hcom_allReduce` |
| | ar_ffn | `67.11` | `16.61` | `23.63` | `29.79` / comm `26.37` | 同上 |
| ⑤ pre-MoE | ffn_norm | `12.22` | `0.00` | `11.96` | `7.50` | `GemmaRmsNorm` + Cast |
| | router_topk | `27.70` | `5.16` | `162.02` | `33.50` | router FP32 MM + `MoeGatingTopK` + Index/NotEqual/Abs |
| ⑥ dispatch | route_org | `28.05` | `0.00` | `26.76` | `42.79` | `MoeInitRoutingCustom` + `Cumsum`(AI_CPU) |
| | **ep_dispatch_comm** | `33.72` | `0.00` | `57.51` | **n/a** | vLLM 无跨卡 EP，本卡 permutation |
| ⑦ experts | routed_gmm1 | `41.72` | `41.72` | `2364.35` | `42.52` | `GroupedMatmulSwigluQuant` |
| | routed_down | `13.36` | `6.52` | `518.10` | `28.52` | `GroupedMatmul` |
| | shared_expert | `59.04` | `2.05` | `212.98` | `23.80` | shared MM + `SwiGlu` + MM |
| ⑧ combine | **ep_combine_comm** | `63.05` | `55.95` | `49.93` | **n/a** | vLLM 无跨卡 EP |
| | unpermute | — | — | — | `5.08` | `MoeTokenUnpermute`（PyPTO 融进 combine_reduce） |
| | merge + residual | `16.34` | `16.34` | `45.14` | `7.72` | `Add` ×2 |

**几个反直觉但重要的读数**：

- **`routed_gmm1` 已经打平**（`41.72` vs `42.52`）：packed-NZ 融合把 PyPTO 的 GMM1
  做到了和 vLLM `GroupedMatmulSwigluQuant` 同一水平。**这条线没有剩余空间了。**
- **`routed_down` PyPTO 快 2.1×**（`13.36` vs `28.52`）。
- **`shared_expert` PyPTO 慢 2.5×**（span `59.04` vs `23.80`），但**独占只有 `2.05`**
  ⇒ 它躲在 dispatch 后面，**对 ITL 几乎不要钱**。既有报告的结论"不要先优化 shared
  expert"由此得到独立印证。
- **`o_proj` PyPTO 慢 2.0×**（`28.17` vs `13.94`）且**全额独占** ⇒ 这是一条被忽视的
  真实关键路径项，`1588 core-µs` busy 说明它是算力受限而非等待。
- **`qkv_proj` busy `275.75 core-µs` vs `o_proj` `1588`**：差 5.8×。o_proj 需要复核
  是否 tile/chunk 配置不佳。

---

## 6. ⚠ 有效性边界（不许跨过去用）

1. **workload 不同，attention 与 AllReduce 的绝对比值不成立。**
   PyPTO = BS1 / ctx 64K；vLLM = **BS4 / ctx 未知**。trace 里**没有 tensor shape**
   （arg key 只有 `Task Id`/`Model Id`/`Physic Stream Id`/`count`/`size(Byte)` 等），
   `FusedInferAttentionScore` 的 KV 长度**不可恢复**。attention 与 KV 相关的行只能
   读结构占比，不能读比值。
2. **PyPTO 是 5 层 synthetic harness，不是整网。** 层间流水、weight prefetch、
   LM head、MTP3 都不在里面。**不要把本表的层时间乘 45 当整网预测**（粗推
   ≈`17.6 ms` vs vLLM `16.72 ms`，但这只是量级校验，不是承诺）。整网口径是
   R5 的 ITL p50 `27.757 ms` @ITERS=100。
3. **PyPTO 每 rank 只有 1 次 invocation ⇒ 无法给 p50/p95**，本表所有 PyPTO 数字是
   **8 个 rank 的 median**，离散度用 min/max 看（层 wall 跨 rank 只差 ±1%，很稳）。
   vLLM 数字是跨 `68×层数` 个实例的 p50/p95，统计强度高得多。**两侧统计强度不对等。**
4. **PyPTO L0 不可用**：7/8 rank 在 L0 的 `tp_all_reduce_residual_bs1` 上阻塞
   `330–630 ms` 等最慢 rank 到齐（进程启动 skew 的**冷启动 barrier**，不是性能）。
   L0 已从所有对比中剔除，因此**没有 full-attention + dense 层的 PyPTO 读数**。
5. **`alloc` / `attn_out_zero` 被调度器提前吊起**（起点比所属层早 589 ms），
   已从 span/层窗口里剔除（busy 合计仅 `~3 core-µs`，剔除不影响结论）。
   若不剔除会把层 span 算成 589 ms —— 这是本次分析踩到并修掉的第一个陷阱。
6. **MoE 拓扑差异是真实的，不是可消除的对齐误差**：vLLM 本卡 local routing +
   `MoeInitRoutingCustom`/`Cumsum`/`MoeTokenUnpermute`；PyPTO EP8 跨卡
   dispatch/combine。`ep_dispatch_comm` / `ep_combine_comm` 两行**在 vLLM 侧不存在
   对应物**，标 `n/a` 而不是硬凑格子。
7. **L43/L44 special 层已单独测出**（vLLM `special_full 490.38` / `special_swa 385.25`，
   GMM1 走 `GroupedMatmul + Slice/Swish/Clip/Mul + DynamicQuant` = `96.42/98.14 us`），
   但 **PyPTO 五层 harness 里没有 special 层**，无法对比。
8. **MTP3（Model 86）3 层未做分解** —— 超出本次范围。

---

## 7. 结论与下一步候选（按 ROI 排序，均未立项）

| # | 候选 | 依据 | 天花板（单层） | 全网粗估 |
|---|---|---|---:|---|
| 1 | **收 SWA 层的调度 gap** | SWA MoE 层 gap = 整层 **47.8%**（`185.46 us`），vLLM 同口径 5.0% | ~`100+ us` | 30 层 ⇒ 最大的一块 |
| 2 | **收 full-MoE 层的调度 gap** | gap = 28.5%（`130.91 us`），19 个阶段边界累积 | ~`60 us` | 10 层 |
| 3 | **EP combine 通信 overlap** | `ep_combine_comm` 独占 `55.95`，但 busy 仅 `49.93 core-µs`，span `63.05` ⇒ 含跨卡等待 | ~`30 us` | 40 层 |
| 4 | **`o_proj` 复核** | 慢 vLLM 2.0× 且**全额独占**，`1588 core-µs` busy 偏高 | ~`14 us` | 45 层 |
| 5 | dense 层裸速度 | `dense_mlp` 1.90×、AR 1.85×、pre-FFN norm 4.63×，无 overlap 可藏 | — | 仅 3 层，权重低 |

**已确认没有空间的方向**（不要再立项）：`routed_gmm1`（已打平 `41.72` vs `42.52`）、
`shared_expert`（独占仅 `2.05 us`）、`head_gate` / `ffn_norm` / `route_org` /
`ep_dispatch_comm`（独占 = `0.00`，对 ITL 完全免费）。

> 任何候选立项前按 `.claude/skills/pypto-perf-regression/` 走：先算 ROI 天花板
> vs A/B/A 检测地板，再无卡 codegen 门。**本表只给天花板，不给裁决。**
