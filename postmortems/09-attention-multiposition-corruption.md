# 专项：step3p5 attention 多 position 乱码 — Q-side RoPE pack + head-gate `matmul_acc` N=16 codegen bugs（`rot_q_hi` band corrupt / `gate_logits` ~20× too small）

| 字段 | 值 |
|------|----|
| **子系统** | codegen / vllm-pypto |
| **error signature** | full-attn ctx>1 `bad_ratio≈0.90` / head-gate `gate_logits` ~20× 偏小 → `bad_ratio≈0.97` |
| **首次出现** | 2026-07-02（rope-q-pack）/ 2026-07-03（head-gate `matmul_acc` N=16） |
| **状态** | ✅ 已解（model-side workaround；上游 codegen bug 待提） |
| **相关 skill / doc** | `pypto-lib/docs/upstream-issues/step3p5-rope-q-pack-codegen.md`、`.../step3p5-head-gate-matmul-acc-n16-codegen.md`；复现器 `_stage_scope12_qk.py` / `_stage_attn_worker.py` |

## 1. 背景（Background）

事故发生在 live vLLM+PyPTO layer-0 attention 集成路径上。目标机 `gpu-a910x-0162`（Ascend 910B2C，CANN 9.0.0 non-GA，PTOAS v0.45，per-rank 配置 `apply_perrank_patch`：`NUM_HEADS_FULL_LOCAL=8` / `NUM_HEADS_FULL_LOCAL_PAD=16` / `KV_HEADS_LOCAL=1` / `HIDDEN_Q_FULL_LOCAL=1024` / `HEAD_DIM=128` / `BATCH=BATCH_TILE=16`）。想达成：把 step3p5 full-attention kernel 接到 live `vllm serve` 8001 上，layer-0 attention 输出与 vanilla 对齐、生成连贯文本。

两个 root cause 在连续两个 session 里被分别定位：

- **2026-07-02**：full-attn 多 position（ctx>1 / prefill）输出乱码，单 position（ctx=1）正确。复现器 `_stage_scope12_qk.py`（standalone L3 `@pl.program`，逐字复制 `attention_full.py` Scope 1 + Scope 2 + Stage-1 QK）。
- **2026-07-03**：layer-0 attention 与 vanilla `bad_ratio≈0.97`、live 生成乱码；根因在 head-gate 的 `gate_logits = normed_all @ w_g`（输出 N=16）matmul。

step3p5 head-gate 使用 **NORMED hidden**（不是 raw hidden）：`gate = sigmoid(input_layernorm(hidden) @ w_g)` per head，在 o_proj 前乘到 `attn_out` 上（`config.use_head_wise_attn_gate=True`）。这一点对理解 07-03 的根因至关重要。

## 2. 现象（Symptom）

### 2.1 rope-q-pack（2026-07-02）

`seq_lens = arange(BATCH)+1`（row *i* attends positions 0..i，"crossrow"）下，step3p5 full-attention decode 层与 torch golden 的对比：

```
row 0 (ctx_len=1)  : PASS
row 1..15 (ctx>1)  : bad_ratio ≈ 0.90   rows 1..15 bad
```

单 position decode（所有 `seq_lens=1`，即既有 `test_decode_layer_full_dense_st`）PASS。即既有 ST 完全掩盖了这个 bug。

复现器 `_stage_scope12_qk.py` 逐层 dump vs golden：

```
q_proj_norm   (Scope 1.b + 1.e)  bad_ratio 0.0000   ✅
k_proj_norm   (Scope 1.c + 1.e)  bad_ratio 0.0000   ✅
k_cache block (Scope 2 K write)  bad_ratio 0.0000   ✅
all_q_padded  (Scope 2 Q pack)   bad_ratio 0.0889   ❌ first diverge (row 0, col 32)
all_raw_scores (Stage-1 QK)      bad_ratio 0.2482   ❌ downstream of wrong Q
```

`REAL_ROPE=1` 时误差更大（`all_q_padded` 0.19、scores 0.90）。首错列 **32 = `ROTARY_HALF_FULL`**，即 `rot_q_hi` 段的起点；`rot_q_lo`（cols 0..31）正确。

### 2.2 head-gate `matmul_acc` N=16（2026-07-03）

live 8001 in-process A/B（`PYPTO_ATTN_AB=1`，同 live 输入跑 pypto + vanilla，log `bad_ratio`，return vanilla 维持服务连贯）：

```
pypto full layer-0 vs vanilla : bad_ratio ≈ 0.97   (garbage generation)
per-rank o_proj partial (hot rank) : ~40× vanilla
per-rank o_proj partial (7 non-hot ranks) : 0.4–15 (ungated)
```

dump kernel `gate_logits`（覆写 `resid1[:, :16]` 经 worker 读出）：

```
kernel  gate_logits[0,:4] = [-0.65, -0.06, -0.51, -1.09]
python  gate_logits[0,:4] = [-13.84, -2.21, -13.01, -15.09]   # normed_all @ w_g, FP32
ratios                    ≈ [21.4,  34,    25.6,  13.8]        # varying → partial K sum
```

`gate_logits` 系统性 ~20× 偏小、逐 head 比例不一 → `sigmoid(-0.6)≈0.35` 而非 `≈0` → gate 不压制 → "热"头切片直通 → 该 TP rank o_proj partial 爆 ~40× → 整层 `bad_ratio≈0.97`。

## 3. 根因（Root Cause）

### 3.1 rope-q-pack：Scope 2 Q partial-RoPE 打包的 codegen VALUE bug

`attention_full.py` Scope 2 原写法（有 bug）：

```python
q_block = pl.reshape(
    pl.slice(q_proj_norm, [1, Q_HEAD_BATCH_FULL * HEAD_DIM], [b, q_base * HEAD_DIM]),
    [Q_HEAD_BATCH_FULL, HEAD_DIM])                       # [8, 128]
q_lo = pl.slice(q_block, [Q_HEAD_BATCH_FULL, ROTARY_HALF_FULL], [0, 0])              # cols 0..31
q_hi = pl.slice(q_block, [Q_HEAD_BATCH_FULL, ROTARY_HALF_FULL], [0, ROTARY_HALF_FULL])  # cols 32..63
rot_q_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
rot_q_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
all_q_padded = pl.assemble(all_q_padded, cast(q_block),  [pad_row_base, 0])
all_q_padded = pl.assemble(all_q_padded, cast(rot_q_lo), [pad_row_base, 0])            # [8,32] @ col 0
all_q_padded = pl.assemble(all_q_padded, cast(rot_q_hi), [pad_row_base, ROTARY_HALF_FULL])  # [8,32] @ col 32  <-- corrupts
```

codegen miscompile 链路 = **"`reshape([N, HEAD_DIM])` of a `[1, N*HEAD_DIM]` slice → 后续 `[N, ROTARY_HALF]` sub-slice at col offset `ROTARY_HALF` → `[N, 32]`-at-col-offset-`ROTARY_HALF` `assemble`"**。精确定位在 `rot_q_hi` 写入区（cols `ROTARY_HALF_FULL..ROTARY_DIM`）。

证据链：

- 单行 K 路径（`rot_k_hi`，`[1,32]` slice of `k_proj_norm` assembled at col `ROTARY_HALF_FULL`）**正确**。
- 多行 Q 路径（`[8,32]` 来自 reshaped `q_block`、在 col 32 assemble）**错误**。
- 即 bug 特异于"多行 `[N, ROTARY_HALF]` tile sourced from a `reshape`d `q_block` 并 assemble 到 column offset `ROTARY_HALF_FULL`"。
- `ctx_len=1` 掩盖 bug 的原因：单元素 softmax 权重=1，输出恒=V₀，与 q·k 分数无关 → 错误的 q·k 值在 ctx=1 完全不可见，只在 ctx>1（按分数加权）暴露。

### 3.2 head-gate `matmul_acc` 小 N=16 K-accumulation 丢累加

`attention_full.py` Scope 1.f（on-device head-gate logit 投影，输出 N=`NUM_HEADS_FULL_LOCAL_PAD=16`）：

```python
gp_acc = pl.matmul(normed_all_chunk0, w_g_chunk0, out_dtype=pl.FP32)   # [BT,16]
for kb in pl.range(1, decode_scope1_hidden_blocks):   # accumulate over K=HIDDEN
    gp_acc = pl.matmul_acc(gp_acc, normed_all_chunk_kb, w_g_chunk_kb)
gate_logits = pl.assemble(gate_logits, gp_acc, [b0, 0])
```

`normed_all`（验证：与喂给 q_proj 的同一 tensor 完全相同）和 `w_g`（验证：bit-exact vs HF checkpoint slice）都正确。**但 compiled kernel 的 `gate_logits` 出来 ~20× 偏小、逐 head 比例不一**——一个 *partial* K-contraction，仿佛只有约第一块 K-block 生效。

对照证据：同循环模式在 **q_proj（输出 N=`Q_OUT_CHUNK`，大 N）** 下正确。唯一差别是输出宽度 **N=16**。

隔离实验（07-03 in-process A/B + targeted kernel probes）：

| probe | result | verdict |
|---|---|---|
| force `gate_sig = 0` in kernel | all ranks collapse to `hidden` | gate **apply** path OK |
| force sigmoid input `= -10`（bypass matmul） | all ranks collapse | **sigmoid** OK |
| dump kernel `gate_logits`（覆写 `resid1[:, :16]`，worker 读） | ~20× too small，varying | **matmul** wrong |
| 把 gate matmul 挪到 q/k/v 之前（fresh `normed_all`） | 无变化 | not staleness |
| 把 `matmul_acc` 替换成 `matmul` + `pl.add` | **编译失败** | `'pto.tmatmul' op expects dst to be in the acc address space` |

即：`matmul_acc` 是唯一 K-累加路径（standalone `matmul` dst 必须在 acc 地址空间），而小 N=16 下它丢累加 → on-device gate 不可用。

## 4. 如何解决（Fix）

两个 bug 都用 **model-side workaround** 绕过（上游 codegen bug 待提），都在 `pypto-lib/models/step3p5/`。

### 4.1 rope-q-pack 修复

把"reshape 成 `[8,128]` + col-32 子列切片 + `[8,32]@col-32` assemble"改成**逐 head 用 `[1, ROTARY_HALF_FULL]` 连续切片**（完全镜像已验证正确的 K 路径），逐 head assemble 进 `all_q_padded`：

```python
q_base = ki * Q_PER_KV_FULL
pad_row_base = b * KV_HEADS_LOCAL * (Q_PER_KV_FULL // Q_HEAD_BATCH_FULL) * Q_HEAD_PAD_FULL + ki * Q_HEAD_PAD_FULL
for qh in pl.range(Q_HEAD_BATCH_FULL):
    qh_col = (q_base + qh) * HEAD_DIM
    q_lo_h = pl.slice(q_proj_norm, [1, ROTARY_HALF_FULL], [b, qh_col])
    q_hi_h = pl.slice(q_proj_norm, [1, ROTARY_HALF_FULL], [b, qh_col + ROTARY_HALF_FULL])
    rot_q_lo_h = pl.sub(pl.col_expand_mul(q_lo_h, cos_lo), pl.col_expand_mul(q_hi_h, sin_lo))
    rot_q_hi_h = pl.add(pl.col_expand_mul(q_hi_h, cos_hi), pl.col_expand_mul(q_lo_h, sin_hi))
    q_row = pad_row_base + qh
    all_q_padded = pl.assemble(all_q_padded, cast(pl.slice(q_proj_norm, [1, HEAD_DIM], [b, qh_col])), [q_row, 0])
    all_q_padded = pl.assemble(all_q_padded, cast(rot_q_lo_h), [q_row, 0])
    all_q_padded = pl.assemble(all_q_padded, cast(rot_q_hi_h), [q_row, ROTARY_HALF_FULL])
```

应用到 `attention_full.py`（Scope 2）和 `attention_swa.py`（Scope 2；SWA 无 full-row assemble，保留结构，只改 rot_lo + rot_hi + zero-pad）。数学等价；差别是逐 head 连续切片替代 reshape + col-offset slice。

验证（0162 card 8）：

| Check | before | after |
|---|---|---|
| `_stage_scope12_qk.py` scores, identity rope | 0.2482 | **0.0018**（bf16 noise） |
| `_stage_scope12_qk.py` scores, `REAL_ROPE=1` | 0.8998 | **0.0000** |
| `_stage_attn_e2e.py ATTN_PERRANK=1` crossrow（全 decode 层 attn+MLP） | 0.8374 | **0.0000 PASS** |
| `test_decode_layer_full_dense_st -d 8`（单 position 回归） | PASS | **PASS 7.97s** |

### 4.2 head-gate `matmul_acc` N=16 修复

把 gate 计算移到 worker 端 python 预算，kernel 接收 finished 乘子：

- **Kernel `attention_full.py`**：删掉 on-device `gate_logits` matmul（Scope 1.f）+ sigmoid + block-diag `gate_r` expand。现有 `gate_r` 参数（`[NUM_HEADS_FULL_LOCAL_PAD, HIDDEN_Q_FULL_LOCAL]`）现在 **承载预算好的 per-feature 乘子 `gate_exp`** —— `BATCH == NUM_HEADS_FULL_LOCAL_PAD == 16`，所以 `[16, 1024]` slot **无签名/形状变化**地装下。o_proj 逐元素乘 `attn_out * gate_r`（cast bf16→fp32）。`w_g` 留在签名里（worker 仍发送）但 kernel 不再用。
- **Worker `_stage_attn_worker.py::_AttnService.attn()`**：每次调用前计算 `gate_exp = repeat_interleave(sigmoid(RMSNorm(current_hidden, input_rms, eps=1e-5) @ w_g_local[:, :NUM_HEADS_FULL_LOCAL]), HEAD_DIM)` → `[BATCH, HIDDEN_Q_FULL_LOCAL]`，复制进 `gate_r` buffer。即 vLLM 的 `sigmoid(g_proj(input_layernorm(hidden)))` per head。

验证（0162 card 8 / live 8001，W8A8 ckpt）：

| Check | before | after |
|---|---|---|
| in-process AB `bad_ratio`（pypto full layer-0 vs vanilla） | 0.97 | **0.0000–0.0002**（max\|d\| 0.007–0.375，bf16 noise） |
| per-rank o_proj partial（7 non-hot ranks） | 0.4–15（ungated） | **~0.06 = hidden**（suppressed，matches vanilla） |
| live generation（AB off，layer-0 returns pypto，`layer 0 SUCCESS`） | garbage | **coherent，matches vanilla**（CN + EN） |
| 8000 / 8001 health | 200 / 200 | 200 / 200 |

### 4.3 副作用与适用边界

- rope-q-pack workaround 不改变数学，只换切片结构；不适用场景：若未来 `KV_HEADS_LOCAL>1`（unsliced/`apply_tp1_patch` 路径）会触发另一个独立 bug（Stage-1 `q_padded_row = fa_b*Q_HEAD_PAD_FULL` 与 Scope-2 stride 不一致），生产 per-rank（`KV_HEADS_LOCAL=1`）不受影响。
- head-gate workaround 把 gate 计算搬到 host python，增加一次 host→device `gate_r` 传输（`[16,1024]` bf16 = 32KB / layer / step），性能损失小但 on-device gate 优势丧失。on-device gate 的恢复 gated on 上游 `matmul_acc` 小 N 修复。
- 两个 codegen bug 本身均未上游修复，workaround 是唯一工作路径。

## 5. 走过的弯路（Detours / What We Got Wrong）

### 5.1 rope-q-pack

- ❌ 假设"单 position ST PASS = attention 正确" → 证伪：`test_decode_layer_full_dense_st` 全程用 `seq_lens=1`，单元素 softmax 权重=1 与 q·k 无关，天然掩盖 Q-side bug。改 crossrow `seq_lens=arange+1` 立刻 `bad_ratio=0.84` 暴露。
- ❌ 假设"K 路径对、Q 路径只是多一行"→ 无效：单行 K `[1,32]` 正确不代表多行 Q `[8,32]` 正确；bug 特异于 reshape 后的 col-offset assemble。
- ❌ 尝试用 unconditional `assemble` 直接 dump `all_q_padded` → 崩 `fuse_create_assemble_to_slice` pass：`TypeError: Operator 'eq' does not accept bool dtype`（第二个独立 codegen 脆弱点）。改用 traced-conditional `if _DUMP==N:` assemble + `matmul(q_padded, eye)` 才绕过。
- ❌ 短暂怀疑 tp1/unsliced 路径 Stage-1 stride bug 是根因 → 证伪：仅当 `KV_HEADS_LOCAL>1` 触发，生产 per-rank（`KV_HEADS_LOCAL=1`）不触发；单独记为独立 bug。

### 5.2 head-gate `matmul_acc` N=16

- ❌ 假设"gate 输入 `normed_all` 被某处覆写 / stale" → 证伪：把 gate matmul 挪到 q/k/v 之前用 fresh `normed_all`，`bad_ratio` 无变化（表 3.2 probe 4）。
- ❌ 假设"sigmoid apply 路径错" → 证伪：force `gate_sig=0` / sigmoid input `=-10` 两条 probe 都让所有 rank collapse 到 `hidden`，apply 路径 OK。
- ❌ 假设"`w_g` 权重 slice 拿错" → 证伪：bit-exact 对比 HF checkpoint slice，一致。
- ❌ 尝试用 `matmul` + `pl.add` 替代 `matmul_acc` 绕过 → 编译失败：`'pto.tmatmul' op expects dst to be in the acc address space`。standalone `matmul` dst 必须在 acc 地址空间，`matmul_acc` 是唯一 K-累加路径。
- ❌ 假设"是 BF16 精度问题" → 证伪：matmul 已指定 `out_dtype=pl.FP32`，仍是 ~20× 偏小且逐 head 比例不一（精度问题不会 varying per column）。

## 6. 如何避免（Prevention）

- **单 position ST 不能作为 attention 正确性的 gate**。任何 attention kernel 的 ST 必须含 crossrow / multi-position 用例（`seq_lens=arange(BATCH)+1`）。早期识别信号：ST PASS 但 live/prefill 乱码 → 立刻怀疑 q·k 值错误被 ctx=1 softmax 掩盖。
- **小 N cube matmul 要单独 probe**。`matmul` + `matmul_acc` 累加循环在 N=16 这种小输出宽度下可能丢累加，与 N≥128 大 N 行为不同。早期识别信号：matmul 输出系统性偏小、逐 column 比例不一（非 uniform scale）→ 立刻 dump 输出 vs python FP32 ref 比 ratio。
- **head-gate 数值口径要写清**：`gate = sigmoid(input_layernorm(hidden) @ w_g)` per head，**NORMED hidden 不是 raw hidden**；`gate_r` slot 在 `BATCH==NH_PAD==16` 下可承载 `gate_exp`，作为 on-device gate 不可用时的 fallback。
- **dump create+assemble tensor 用 matmul-eye 技巧**：直接 assemble 崩 `fuse_create_assemble_to_slice`；用 `matmul(tensor, eye)` 打断 create+assemble lineage 后再 dump。
- **model-side workaround 上游 bug 时，保留原 buggy 写法的 reproducer**：rope 的 `_stage_scope12_qk.py`、head-gate 的 in-process A/B + `matmul(→N=16)` standalone repro，都是提上游 issue 时必须附的最小复现器。
- **相关约束落点**：`pypto-lib/docs/upstream-issues/step3p5-rope-q-pack-codegen.md`、`.../step3p5-head-gate-matmul-acc-n16-codegen.md`；head-gate on-device 版本恢复依赖上游修 `matmul_acc` 小 N；rope 上游修依赖确认 reshape / col-offset sub-slice / col-offset assemble 哪一步 miscompile。另见 `postmortems/04-tmov-vec-lhs-matmul.md`（同属 attention codegen 回归类）。
