# 02 — Detailed Design (LLD)：各优化子任务

> 每个子任务一张卡片：问题（file:line 实证）/ **shape（step3p5 具体输入输出）** / **如何生效（用 shape 讲清搬了多少、算了多少、省了什么）** / 参考 / 改法 / 验证 / 落地边界。
> HLD 见 [`01-system-design.md`](01-system-design.md)，状态见 [`task-tracking.md`](task-tracking.md)。
>
> 路径约定：`P/` = `pypto-lib/`（**最新 `stepfun/develop @ bc5eecb1`**，fork csy0225）；`REF/` = `origin/main:models/deepseek/v4-flash/`（`git show REF/<f>` 读取）。
>
> **⚠ 2026-07-24 base 校正（覆盖下方所有卡片的文件引用）**：LIVE 整网 = 手写维护的
> **`P/models/step3p5/decode_layer_single_chip_hidden.py`**（hidden-only，45× unroll `whole_chip_orch`）。
> commit `759c23e8` 已删 `decode_layer.py`、旧 generator（`_gen_faithful_real.py`/`_gen_single_chip_real.py`）、
> `decode_fwd/mtp/step3p5_decode`、canonical §5 round-trip。**下方卡片里出现的 `decode_layer.py` /
> `_gen_*` / `whole_decode_holder.py:183-228` / generator / round-trip / `3af13f4f` / `feat/whole-net-n1-fusion`
> 一律改读 `decode_layer_single_chip_hidden.py` 对应结构**（file:line 见 [`task-tracking.md`](task-tracking.md) 各行「阻塞/备注」列）：
> C1 窗口在 `whole_chip_orch` signature 4857-4914（16 MoE stack）+ per-layer slice 5066-5081…；
> wait 在 701/737/798/1323/2316/2385/2486。对账：**A1/C2/B1/SwiGLU-per-layer 已交付**，B/C 剩 **C1 + B2**。

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

---

## Track A — 可观测性 & Baseline

### PERF-A1 · whole-net decode baseline + DFX 采集
- **问题**：whole-net 无 perf 数据 = 盲调。`docs/step3p5`、`docs/performance-tuning.md` 无延迟数据。
- **shape**：不改数据流；被测是 `whole_decode_faithful_real`（输入 hidden `[16,4096]`/rank → 输出 logits shard `[16,16112]`/rank）。
- **如何生效**：不优化，只建 baseline。采 `l2_swimlane`（每个 kernel 的起止 + 层间 gap）、`pmu.csv`（cube/vec/mte 利用率）、`perf_hints.log`（MTE 非 512B 对齐点）、`memory_after_AllocateMemoryAddr.txt`（各 buffer 占用峰值），把 45 层每层耗时拆出来 → 定位真正的热层/热 kernel，后续每项拿它回归。
- **改法**（不改模型）：`P/tools/step3p5/whole_decode_holder.py:280` 已有 `enable_scope_stats`；加 `--enable-l2-swimlane`(0/1/2)、`--enable-pmu` 透传 `rt.run`。参考 `P/docs/performance-tuning.md:12,139,247,263`。跑多步 decode，落 `docs/step3p5/perf-baseline.md`（新建）。
- **验证**：产出分层耗时表 + 4 个 DFX 工件路径。
- **边界**：零代码风险。**先做**——解锁 F2 与所有定量对比。

---

## Track B — Mega-kernel 结构

### PERF-B1 · 权重 leading-dim stacking + `resident="stacked"`
- **问题**：权重已 IPC 常驻（`P/tools/step3p5/whole_decode_holder.py:183-228`），但布局非 `[N_RANKS, L*dim, ...]`、未打 `resident="stacked"`，无法被 B2 的 `pl.slice` 层循环消费。
- **shape**：MoE routed 权重 stacked 后 per-rank `moe_w_gate_r [8, 42, 36, 4096, 1280]`（`[N_RANKS, LAYER, N_LOCAL=36, HIDDEN, INTER]`）；attention `wq_full [8, 42, 4096, 1024]`；KV pool `k_cache [8, 45*KV_CACHE_ROWS, 128]`。层内取 `pl.slice(moe_w_gate_r[r], [36,4096,1280], [layer_idx*36, 0, 0])`。
- **如何生效**：现状每次 forward 把 `≈47.6GB/rank` 权重从 host 端搬上卡（或经 IPC 但按非 stacked 布局逐张管理）。改成 `[N_RANKS, L*dim, ...]` + `resident="stacked"` 后，**每个 shard 上传一次**、跨所有 decode step 复用，省掉每 step 的 H2D；`pl.slice` 用 dynamic scalar `layer_idx` 在常驻大 buffer 内零拷贝取当前层。
- **参考**：`REF/decode_fwd.py:162-176`、`:1443-1450`、`REF/moe.py:928-945`。
- **改法**：`P/models/step3p5/weight_loader.py` 按层型堆 `[N_RANKS, L*dim, tail]`；每权重 spec + KV pool 打 `resident="stacked"`；层内消费改 `pl.slice`。
- **验证**：多步 L3（N=128 ≥95% vs vanilla）parity（布局改，数值不变）。
- **边界**：可在现 unroll 结构下先落（B2 前独立价值）；`REF/decode_fwd.py:160-161` 警告 resident-stacked 目前 decode-world-only，需确认 runner 支持。

### PERF-B2 · 45 层 unroll → 单 `pl.range` 循环
- **问题**：`P/tools/step3p5/_gen_faithful_real.py:1273-1326`（L0/L1/L2 直排）+ `:1337-1428`（42 MoE 逐层 emit）→ `decode_layer.py` 31,636 行、98 个 `*_chip_orch`。
- **shape**：循环体每层消费 hidden `[16,4096]` → 输出 hidden `[16,4096]`；层内从 stacked 权重（B1）`pl.slice` 出当层 shard（如 MoE `[36,4096,1280]`）；layer_idx 为 dynamic scalar。
- **如何生效**：unroll 让编译器为 45 层各生成一份 kernel/依赖图 → IR 体量 ×45、AICPU 调度边 ×45。折成 `pl.range` 后**循环体只 codegen 一份**，layer_idx 动态切权重/窗口 → IR 崩塌到数百行、跨层复用 SSA buffer、调度边 ÷45。hidden `[16,4096]` 逐层串接不变，数值等价。
- **参考**：`REF/decode_fwd.py:404-565`（`pl.range(HCA_NUM_LAYERS)` 循环体）。
- **改法**：重写 `_gen_faithful_real.py::_host_orch`（`:1159`），按层型分桶（full-moe / swa-moe / dense）各一个 `pl.range`；首/尾特殊层保留显式。
- **验证**：多步 L3（N=128 ≥95%）+ 逐层 detail compare（`ratio_allclose atol=0.04`）。
- **边界**：**硬依赖 B1 + C1**；XL / 高风险；先 dense-only 循环验证再扩 MoE。

### PERF-B3 · KV pool `resident` + in-place
- **问题**：KV 每 dispatch 可能重传（`P/models/step3p5/attention_full.py:183,211,218` consolidated multi-layer ABI）。
- **shape**：`k_cache/v_cache [KV_CACHE_ROWS_DYN, 128]` BF16 per rank（45 层堆叠）；每 step 只写当前 token 的 1 行 `[1,128]`（`slot_mapping` 定位），读窗口 `[ctx_len,128]`。
- **如何生效**：KV pool 大（数百 MB～GB/rank）。若每 step D2H/H2D 整池，带宽全浪费在没变的历史 KV 上。`resident` + InOut 让池**常驻、原地写**：每 step 仅 MTE 写 1 行 + 读有效窗口，省掉整池往返。
- **参考**：`REF/decode_fwd.py:151-176`（`CACHE_POOL_NAMES` + `RESIDENT_CACHE_OUTPUT_NAMES`）。
- **改法**：KV 归 `CACHE_POOL_NAMES`，`resident="stacked"` + InOut，kernel 原地写。
- **验证**：多步 decode KV 连续性 + L3 parity。
- **边界**：随 B1；注意 vLLM per-layer pool vs step3p5 consolidated ABI 差异（memory `g5b_kv_bridge_not_pure_reshape`）。

---

## Track C — MoE 通信协议

### PERF-C1 · 单 window set + `moe_epoch` + `WaitCmp.Ge`（关键路径）
- **问题**：`P/tools/step3p5/_gen_faithful_real.py:1342-1357` 每 MoE 层 16 个 `_L{pos}` 窗口 → 42 层 ≈ 672 窗口 / **≈766MB comm domain**。根因：RAW-only-v1 非别名（ADR-013），窗口无法跨层复用。
- **shape**：一套窗口 = `recv_meta [8,36]` + `recv_x [1024,4096]`(8MB) + `recv_aux [1024,AUX]` + `recv_route [1024,IDX]` + `arrived/data_arrived/combine_arrived [8,1]` + `routed_y [128,4096]`(1MB)。现状 42 套并存；目标 1 套。
- **如何生效**：现状把每层的 `recv_x [1024,4096]` 等都独立开一份 → 766MB 常驻、编译期窗口记账爆。改成**1 套复用**：每层 MoE 调用传单调 `moe_epoch`（1→42），wait 用 `WaitCmp.Ge` 对 AtomicAdd 计数器（`arrived[src] >= moe_epoch`、`data_arrived[src] >= moe_epoch*36`）→ 上一层的 notify 只把计数器抬高，本层的 `Ge` 仍判 done，**天然跨层排序**、同一 `recv_x [1024,4096]` 安全复用 42 次。comm domain 从 766MB → 十几 MB。
- **参考**：`REF/moe.py:120-175`（`dispatch_meta` notify `arrived`）、`:178-235`（`dispatch_push` notify `data_arrived`）、`:238-248`（anchored `dispatch_wait`）、`REF/decode_fwd.py:758-769`（一次性 8 窗口）、`:377,402,495,564,654`（`moe_epoch` 递增）。
- **改法**：host 侧一次性 `pld.alloc_window_buffer` 8 个；每次 MoE 传 `moe_epoch`；wait 改 `Ge`（禁 `Set/Eq`）；`dispatch_wait` anchored（`_idx_anchor = pl.read(indices,[0,0])`）。
- **验证**：6 轮 RUN_CLEAN 稳定（liveness，`_probe_barrier_scale.py`）+ 多步 L3 精度不回退。
- **边界**：**B2 的硬前置**；修 whole-net no-drain stall（memory `n1_wholenet_stall_singleprogram_nodrain`）。

### PERF-C2 · dispatch push → pull（fixed-slot）
- **问题**：`P/models/step3p5/dispatch.py:252-310` push `remote_store` scatter = A2 随机 507018 stall（跨 die 写完成竞争，memory `n1_a2_primitive_exists_not_missing`）。
- **shape**：每 token 选 `TOP_K=8` 专家 → 每 rank 最多发 `BATCH*TOP_K=128` 条 `[1,4096]` 路由；接收端 `recv_x [1024,4096]`（`LOCAL_RECV_MAX=1024 = 8 rank × 128`）。
- **如何生效**：push 让每个源 rank 主动 `remote_store` 到目标 rank 的 `recv_x` 槽 → 跨 die 写完成顺序不确定，随机 stall。改 **fixed-slot pull**：目标 rank 按固定槽公式 `my_rank*MAX`/`peer*MAX` 主动拉，写完成由本地掌控 → 消除跨 die 写竞争。数据量不变（同样 128 条 `[1,4096]`/rank），只换发起方。
- **参考**：`REF/moe.py` dispatch + device-validated `P/models/step3p5/moe.py` 的 `ep_all_to_all` fixed-slot pull。
- **改法**（memory `n1_pull_dispatch_must_align_moepy_fixedslot`）：pack fixed-slot → AtomicAdd barrier → static `pl.range(T*TOPK)` a2a → LOCAL re-pack；**combine 保持 push**（combine push = jitter 非 stall）。
- **验证**：6 轮 0 stall（`RUN_CLEAN`/探针，liveness）+ 多步 L3 精度不回退。
- **边界**：C1 之后；不可只改 barrier 语义（Set-alone 不是 fix）。

### PERF-C3 · peer loop `pl.range(N_RANKS)` → `pl.spmd`/`pl.parallel`
- **问题**：`P/models/step3p5/dispatch.py:159,181,210,286,297,322` + `combine.py:182,209,216,225` 全顺序 barrier。
- **shape**：peer 循环 `pl.range(N_RANKS=8)` 逐个 rank 串行处理 `[1,4096]` 级搬运/notify。
- **如何生效**：8 个 peer 顺序做 = 8× 串行延迟。改 `pl.spmd(8)` 让 8 个 peer 的搬运/notify 在多核上 fan-out 并发 → 通信段延迟 ÷8（受核数/带宽约束）。
- **参考**：`REF/moe.py:178`（`pl.spmd(N_LOCAL)`）、`REF/expert_routed.py:80`（`pl.parallel`）。
- **改法**：per-peer barrier 循环改 SPMD fan-out。
- **验证**：L3 parity + L2 swimlane 显示 peer 并发。
- **边界**：C1 之后。

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
- **问题**：`P/models/step3p5/decode_layer.py:26871` `lm_head_orch` inline 进同一 program（可用但阻塞、不复用 buffer）。
- **shape**：输入 hidden `[16,4096]`/rank；`lm_head_weight [16112,4096]`/rank（vocab 切 8）；输出 logits shard `[16,16112]`/rank → 全 vocab `[16,128896]`。
- **如何生效**：现状 LM head 与末层 MoE 串行、且单独占 buffer。拆 4 段 worker（publish→tp→route→finish）后：(a) publish 段（发 hidden `[16,4096]`）可与末层 MoE 的 combine 段**在调度上重叠**；(b) 4 段**复用 MoE 的 `recv_x [1024,4096]` 窗口**作 hidden/logits 中转（MoE 与 LM-head 时分不冲突）→ 省一份 `[16,16112]` 级 buffer。
- **参考**：`REF/lm_head.py:433-515` + `REF/decode_fwd.py:877-901`。
- **改法**：拆 `publish → tp → logits_route → finish` 四 worker，各 `device=r`；复用 `recv_x_buf`。
- **验证**：logits L3 parity + 端到端延迟下降。
- **边界**：C1 之后（共享窗口约定）；当前 fused 可用，非阻塞。

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

### PERF-G1 · 调度轴 batch → experts/feature + dynamic active-token（对齐 DeepSeek）
- **问题**：step3p5 把 **batch 轴当 core fan-out**（`P/models/step3p5/attention_full.py:309` `pl.spmd((BATCH//BATCH_TILE)*...)`、`:332`、`:353`、`:383`、`:438` `pl.parallel(BATCH)`、`:577` `pl.spmd(BATCH)`），`BATCH=16` 静态 padded（`config.py:280`，仅 row0 有效）。曾从 `pl.parallel(user_batch)`（动态）退回 `pl.parallel(BATCH)`（静态）——`attention_full.py:427-438` 注释。
- **shape**：现状 SPMD 在 `BATCH=16` 轴 fan-out（decode 常仅 1 行有效 → 16 核里 1 核干活、15 核算 padding）；专家/中间维（36 experts × `[4096,1280]`）却是循环内串行。DeepSeek 反过来：token 是 `pl.range(active_tokens=nt)`（`nt` runtime），核 fan-out 打在 `pl.spmd(N_LOCAL=36)`（专家）+ `pl.spmd(MOE_INTER//tile)`（中间维）。
- **如何生效**：decode batch 天生小（常=1）。拿 batch=16 做核调度 → 核占用被 batch 上限锁死、且大半在算 padding。迁到 experts(36)/intermediate(1280) 轴 fan-out → **无论 batch 多小核都吃满大维度**；再用 runtime `num_tokens` 让 MoE 只 route 真实 token（`pl.range(nt)`）→ padding 行根本不进 routing，通信/计算都省。对 step3p5 = 把"16 核处理 16 行(15 padding)"改成"36+ 核处理 36 专家 × 真实 nt 个 token"。
- **参考**：`REF/decode_fwd.py:278-280`（`nt=max(num_tokens_per_owner)`）、`REF/moe.py:104,122,133`（`for t in pl.range(active_tokens)`）、`:183,308`（`pl.spmd(N_LOCAL)`）、`REF/expert_routed.py:80,93,110`（`pl.parallel(N_LOCAL_EXPERTS)` + `pl.spmd(MOE_INTER//...)`）、`REF/decode_attention_swa.py:115`（`for b in pl.range(B)` 顺序）。
- **改法**：(a) 引入 runtime `num_tokens` scalar，MoE gate/dispatch/combine 用 `pl.range(active_tokens)` 替 static `BATCH`；(b) attention/MoE core fan-out 从 batch 迁到 experts/intermediate，batch 退化为 sequential inner loop。
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
