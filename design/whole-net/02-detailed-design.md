# pypto 整网集成 · 详细设计（LLD）

> **层级**：Detailed / Low-Level Design。系统视角见
> [`01-system-design.md`](01-system-design.md)。本文给到 file:line、关键结构、
> 接口、算法与不变量，用于实现/审阅/调试。
>
> 代码位置：模型在 `pypto-lib/models/step3p5/`，工具/生成器/live 桥在
> `pypto-lib(-live)/tools/step3p5/`。行号锚定 2026-07 clean pin（见
> [`../../reference/canonical-test.md`](../../reference/canonical-test.md) 的 pin），
> 重构后如漂移以符号名为准。

## 1. 单 `@pl.program` 结构与生成器

**唯一生产入口**（精简瘦身后，`pypto-lib-live` @ `feat/whole-net-vllm-live`）：

| 元素 | 位置（`models/step3p5/decode_layer_single_chip_hidden.py`） |
|------|------|
| builder | `:523` `_build_whole_decode_faithful_real_single_chip_hidden_only_program` |
| `@pl.program` 类 | `:615` `WholeDecodeFaithfulRealSingleChipHiddenOnly` |
| 模块 binding | `:6472` `whole_decode_faithful_real_single_chip_hidden_only = _build_..._program()` |
| host_orch | `:6311`，输出 `next_hidden_out[tp,BATCH,HIDDEN]` BF16（**pre-final-norm**，无 lm_head） |
| 生成器 | `tools/step3p5/_gen_faithful_real.py`（文本生成，只保留 final single-submit 实现 + 共享 helper） |
| bisect 旋钮 | `_FAITHFUL_MOE_LAYERS = int(os.environ.get('P_FAITHFUL_MOE_LAYERS','42'))` |

- **strict raw-hidden 边界**：整网跑完 Main 45 层，输出 pre-final-norm hidden；
  **final RMSNorm + lm_head + sampling 全在下游**（standalone 走 host
  `tools/step3p5/final_logits_from_vllm.py`：final norm + lm_head → logits → argmax；
  live 走 vLLM）。**kernel 内无 lm_head、无 per-layer production dispatcher**。
- host_orch 是**源码 unroll 的 45 层链**（无 Python `for` over layers）；attention /
  dense MLP / MoE body 全部 inline 进整 program。
- MTP 有独立 hidden-only fwd `mtp_hidden_fwd.py`（同样 raw-hidden 边界）。

> **历史/命名**：瘦身前 live 线还并存过全 logits 变体
> `decode_layer_single_chip.py`（`whole_decode_faithful_real_single_chip`，host_orch 末尾
> 调 `lm_head_orch` 出 `logits_shard_out`）与更早的 `decode_layer.py:whole_decode_faithful_real`
> predecessor；`cleanup(step3p5): 收敛 single-submit 为唯一整网入口` 后**均已移除**。
> 下文 §2–§9 的层内算法/结构（MoE / window / tp_all_reduce / KV / weight）在各形态一致，
> 行号以本模块符号名为准。

## 2. host_orch single-submit 与 resident holder

- host_orch 签名/体：`decode_layer.py:27544`（`@pl.function(level=HOST, role=Orchestrator)`）。
- 每层段内 `for r in pl.range(pld.world_size())`（如 `:27692/:27709/:27727`）→ **每 rank 一次 `_submit_chip`**。TP=8 = 8 次 submission/step，全部出自这一个 host_orch。
- resident holder：`whole_decode_holder.py:42` `WholeDecodeHolder`；`build()`(`:139`) 编译；`__enter__`(`:211`) `compiled.prepare()` 常驻 + `import_weights_all` + `import_kv_all`；`run()`(`:283/:507`) 每 step 一次 `self.rt.run(self.compiled, *self._args_list)`。
- per-step 变的 host tensor：`current_hidden`、attn-meta（`seq_lens`/`block_table`/`slot_mapping`/`rope_*`）、KV（IPC `add_inout`）；resident 不变：weights、`gate_r_full/swa`（block-diag R 常量）、`final_norm`、`lm_head`。

## 3. 2-buffer 层间数据流（不变量）

- `A = h_mid_out`，`B = next_hidden_out`，逐层乒乓（见 HLD §5）。
- 每层 attention/MoE 的中间张量（`h_moe_L{pos}`、`h_mid` 等）**write-once per layer**，不跨层复用 SSA（复用会触发 [scheduler timeout](../../postmortems/07-whole-net-scheduler-timeout.md)）。
- 收尾：末层写 `B = next_hidden_out`（**pre-final-norm** BF16），host_orch 直接输出，
  **kernel 内无 lm_head / 无 final norm / 无 all-gather**。下游（standalone host /
  live vLLM）做 final RMSNorm + lm_head + argmax（standalone 见
  `tools/step3p5/final_logits_from_vllm.py`）。

## 4. MoE 路径：fixed-slot pull dispatch + pull combine

### 4.1 gate
`gate.py:112`：FP32 sigmoid + 加性 `router_bias` + flat top-8 + renorm ×
`MOE_ROUTER_SCALING_FACTOR=3.0`（`config.py:118` `NEED_FP32_GATE=True`）。
`expert_indices`/`expert_weights` **replicated across ranks**。

### 4.2 dispatch（fixed-slot pull）
`dispatch.py`：`histogram_and_prefix_sum`(140) / `pack_send_payload`(189) /
`build_inverse_map`(250) / `build_local_expert_csr`(309)。常量
`N_LOCAL_EXPERTS=36` / `LOCAL_RECV_MAX=1024` / `PER_RANK_BUCKETS=288`。

fixed-slot pull 由 `tools/step3p5/_patch_moepy_dispatch.py` 施加：recv_x
**peer-major**，静态 `pl.range(T*TOPK)` + compound-scalar 定槽
（`my_rank*MAX`/`peer*MAX`），AtomicAdd rendezvous barrier，**pull 循环内不读
runtime `pub_counts`**（避免运行时 loop-bound → 死锁；见
[`../../postmortems/06-gate-topk-deadlock.md`](../../postmortems/06-gate-topk-deadlock.md)）。

### 4.3 EP all-to-all
`collectives.py:451` `ep_all_to_all`：pull-side ring，`pld.tile.remote_load` +
`pld.system.notify/wait`；`pub_counts[N_RANKS*N_RANKS, N_LOCAL_EXPERTS]` INT32
是跨 rank 计数表。

### 4.4 combine（pull）
`combine.py`：`weighted_gather_and_add`(95) / `push_routed_y_to_sources`(155) /
`publish_src_route_table`(235)；`combine_done[N_RANKS,1]` INT32 单写者信号。
pull-combine 由 `_patch_combine_pull.py` 施加（`_pull_routed_y` compound-scalar
offset `my_rank*MAX+within`）。**组合 = pull-dispatch + pull-combine**。

### 4.5 experts
`expert_routed.py`（`select_expert_routed(layer_idx)` @`:278` 按逐层激活表选
SiLU / SwigluStep@7）、`expert_shared.py`。

## 5. W8A8 native INT8 routed MoE

| 环节 | 位置 / 要点 |
|------|------------|
| 权重签名 | host_orch `:27568-27573`：`moe_w_gate_r INT8[tp,42,n_local,HIDDEN,inter]` + `moe_w_gate_r_scale FP32[tp,42,n_local,inter]`（up/down 对称） |
| loader（INT8 保留） | `weight_loader.py:494` `_load_quantized_expert_projector_int8` → `(int8[out,in], fp32 scale[out])`，**不 dequant**；拒绝非零 `_offset`(`:519`) |
| loader（旧 BF16） | `_load_quantized_expert_projector`(480) → `_dequant_w8a8_dynamic_weight`(449) —— **禁用路径** |
| in-kernel dequant | `tools/step3p5/_a5_int8_transform.py`，range-scoped 施加到 inlined `_expert_routed` |

**5 步 dequant 链**（`_a5_int8_transform.py`）：
1. routed 输入 tile per-token INT8 量化（`routed_x_quant`，`x_scale_dq` SSA-carry）
2. gate/up INT8×INT8 → INT32 → dequant（row `x_scale_dq` × col `w_*_scale`，`pl.row_expand_mul`）
3. SwiGLU
4. `h_tile` per-token INT8 requant（`routed_h_quant`，`h_scale_dq`）
5. down INT8×INT8 → dequant

生成器把它拼进 builder：`_gen_faithful_real.py:496` `FRESH_QUANT_MOE_INPUT` 模板。
> 背景：早期"in-expert 量化"路径在 device 上 miscompile（gap-5），已切到 dispatch-side quant。见 [`../../postmortems/10-gap5-attention-quant-scope.md`](../../postmortems/10-gap5-attention-quant-scope.md)。

## 6. 通信、per-layer window、512B signal、两波 barrier

- **per-layer window**：每层新分配 `_L{pos}` 前缀 buffer（`attn_tmp_buf_L0` / `pub_counts_buf_L0` / `recv_x_buf_L0` / `combine_done_buf_L0` / `routed_y_window_buf_L0` …，`decode_layer.py:27761+`，`_L1`…`_L41` 重复）；dense 前缀层用 `l0_/l1_/l2_`。**死的旧 alloc 必须删**（否则 `MaterializeCommDomainScopes` 报错）。
- **512B signal**：`COMM_CONTROL_SIGNAL_BYTES=512`（`decode_layer.py:24895`），逻辑 `[tp,1]` INT32，物理独占 cache line。
- **`tp_all_reduce` 四相**（`decode_layer.py:24905-24980`）：① stage-in（`ar_chunk=HIDDEN//8`）② notify(AtomicAdd+1)→wait(Ge 1) ③ own-load + `remote_load` + FP32 tadd ④ 完成 barrier(AtomicAdd+1→wait Ge 2)。第 ④ 波由 `tools/step3p5/_add_allreduce_completion_wave.py` 加（单波在 ≥41 层挂）。
- collectives 全用 `pld.tile.remote_load` + `pld.system.notify/wait`（`NotifyOp.AtomicAdd`/`Set`，`WaitCmp.Ge`）。

## 7. KV 与权重（数据结构 + IPC）

**KV**（host_orch `:27577-27585`，`config.py:59-62`）：
- `seq_lens[tp,USER_BATCH_DYN] INT32`、`block_table[tp,512] INT32`（`MAX_BLOCKS_PER_SEQ 32 × BATCH 16`）、`slot_mapping[tp,USER_BATCH_DYN]`、`k/v_cache[tp,KV_CACHE_ROWS_DYN=4096,128] BF16`。
- KV-IPC：`tools/step3p5/pypto_kv_ipc.py:96` `import_kv_ipc_all` → `rt.import_ipc_all` → 每 rank `KvIpcMap`；holder `kv_ipc=True` 时绑 `k/v_cache`（`whole_decode_holder.py:202`）。

**权重**：
- `tools/step3p5/pypto_weight_ipc.py`：`import_weights_all`(508)；`WeightIpcMap`(454) `peer_base = rt.import_ipc(key, worker_id)`，按 byte offset 寻址。
- `StackedDeviceTensor`（`pypto.runtime.device_tensor`）：`build_stacked_weight(wmaps,key)` 组每 rank shard；`W_reshape`(holder `:197`) 自定义 per-rank shape。
- **三分类 slice**（`weight_loader.py`）：REPLICATED / TP-sliced / EP-sliced；`expected_shapes(tp)`(`:203`) 是 canonical shape 表；slice helper `_slice_q_proj`(539)/`_slice_kv_proj`(551)/`_slice_o_proj`(563)/`_slice_g_proj`(575)/`_slice_mlp_col`(601)/`_slice_mlp_row`(613)/`_slice_lm_head`(626)。

## 8. 关键 file:line 速查

| 主题 | 位置（`models/step3p5/decode_layer_single_chip_hidden.py` 除非注明） |
|------|------|
| builder / program 类 / binding | `:523` / `:615` / `:6472` |
| host_orch（出 pre-final-norm hidden） | `:6311` |
| bisect 旋钮 `P_FAITHFUL_MOE_LAYERS` | `:570` |
| 512B signal 常量 | `:609` `COMM_CONTROL_SIGNAL_BYTES=512` |
| tp_all_reduce / ep_all_to_all | `:619` / `:709` |
| gate / dispatch / combine | `gate.py:112` / `dispatch.py:140+` / `combine.py:95+` |
| INT8 transform | `tools/step3p5/_a5_int8_transform.py` |
| dispatch/combine pull patch | `tools/step3p5/_patch_moepy_dispatch.py` / `_patch_combine_pull.py` |
| 两波 barrier patch | `tools/step3p5/_add_allreduce_completion_wave.py` |
| weight/KV IPC | `tools/step3p5/pypto_weight_ipc.py` / `pypto_kv_ipc.py` |
| holder | `tools/step3p5/whole_decode_holder.py` |
| host 侧 final norm+lm_head（standalone argmax） | `tools/step3p5/final_logits_from_vllm.py` |
| 生成器 | `tools/step3p5/_gen_faithful_real.py` |

## 9. 不变量清单（改代码前对照）

1. 只有一个 whole-net `@pl.program`；per-layer 通信 buffer 用 `_L{pos}` 且层间不复用。
2. routed 权重 INT8 + FP32 scale；不引入 BF16-dequant。
3. dispatch/combine 都是 pull；pull 循环用静态 bound + compound-scalar 定槽，不读 runtime count。
4. `tp_all_reduce` 保留两波完成 barrier。
5. 生成器改动后必须 strip→regenerate→byte-compare（roundtrip gate）。
6. 单卡 ST/UT 用 `apply_perrank_patch()`（保 TP=8 per-rank slice 宽度），不用 unslice。

## 10. 相关文档

- 系统设计：[`01-system-design.md`](01-system-design.md) · 三轴与 program 墙：[`03-integration-axes.md`](03-integration-axes.md)
- 复盘：[`../../postmortems/07-whole-net-scheduler-timeout.md`](../../postmortems/07-whole-net-scheduler-timeout.md) · [`08`](../../postmortems/08-multiprogram-coprepare-deadlock.md) · [`10`](../../postmortems/10-gap5-attention-quant-scope.md)
- 强约束 skill：`.claude/skills/pypto-dev-constraints/` · hang 排查：`.claude/skills/pypto-whole-net-hang-debug/`
- kernel 硬限制：`pypto-lib/docs/known-pypto-pitfalls.md`
