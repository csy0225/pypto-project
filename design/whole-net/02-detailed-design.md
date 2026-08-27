# pypto 整网集成 · 详细设计（LLD）

> **层级**：Detailed / Low-Level Design。系统视角见
> [`01-system-design.md`](01-system-design.md)。本文给到 file:line、关键结构、
> 接口、算法与不变量，用于实现/审阅/调试。
>
> 代码位置：模型在 `pypto-lib/models/step3p5/`，工具/生成器/live 桥在
> `pypto-lib(-live)/tools/step3p5/`。行号锚定 2026-07 clean pin（见
> [`../../reference/canonical-test.md`](../../reference/canonical-test.md) 的 pin），
> 重构后如漂移以符号名为准。
>
> **2026-08-27 current override**：当前为 pypto `14de90fd` /
> pypto-lib `e6c7d8ec`，已烧入 r12。replicated-input local-owner MoE 与 prepared
> TaskArgs descriptor/signature cache 已落地；正式产品 `decode_fwd.py` 的 host rank
> loop 仍是 `pl.range`，生成代码仍为 serial 8-rank / 8 个独立 `_submit_chip`。
> 下方大部分行号仍锚定 2026-07 历史生成链；全局状态以
> [`../../STATUS.md`](../../STATUS.md) 为准。

## 1. 单 `@pl.program` canonical 结构

**唯一生产入口**（`pypto-lib` @ `stepfun/develop`）：

| 元素 | 位置 |
|------|------|
| `@pl.program` 类 | `models/step3p5/decode_fwd.py:WholeDecodeStep3p5` |
| 模块 binding | `whole_decode_step3p5 = WholeDecodeStep3p5` |
| host_orch | 输出 `next_hidden_out[tp,BATCH,HIDDEN]` BF16（**pre-final-norm**，无 lm_head） |
| 重复层结构 | L1/L2 与 L3-L42 使用 runtime `pl.range`；L0/L43/L44 显式 specialization |
| 共享 dense kernel | `models/step3p5/dense_mlp.py:dense_mlp_body_tp`，供 Main/MTP inline；`69ad31e4` 起在 `mlp_layer_idx` 后新增 `num_tokens` 实参 |
| 诊断 | 仅在 `tests/step3p5/probes/`，产品 program 不含截断/debug ABI |

- **strict raw-hidden 边界**：整网跑完 Main 45 层，输出 pre-final-norm hidden；
  **final RMSNorm + lm_head + sampling 全在下游**（standalone 走 host
  `tools/step3p5/final_logits_from_vllm.py`：final norm + lm_head → logits → argmax；
  live 走 vLLM）。**kernel 内无 lm_head、无 per-layer production dispatcher**。
- host_orch 是**源码 unroll 的 45 层链**（无 Python `for` over layers）；attention /
  dense MLP / MoE body 全部 inline 进整 program。
- MTP 有独立 hidden-only fwd `mtp_hidden_fwd.py`（同样 raw-hidden 边界）。

## 2. host_orch 逐 rank submit、TaskArgs cache 与 resident holder

- host_orch 签名/体：`decode_layer.py:27544`（`@pl.function(level=HOST, role=Orchestrator)`）。
- 当前产品源码在 `decode_fwd.py` 末端用 `for r in pl.range(pld.world_size())`；
  生成 `host_orch.py` 保留一个 Python `range`，**每 rank 一次 `_submit_chip`**。
  TP=8 = 8 次独立 submission/step；r12 未切 native group-submit，也未证明并行下发。
- pypto `14de90fd` 在 prepared boundary 为 public args 生成稳定 descriptor key /
  generation token；`tensor_arg.py` 用该 token 复用 TaskArgs signature/cache。
  未知/可变对象或 cache 异常 fail-open 回完整 signature，cache 与 memo 都有界。
- `free()` 与 submit validation/token publication/同步 graph construction 共用互斥，
  保证 cache hit 的 TaskArgs 在 proof window 内不会引用已释放 Buffer。
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

## 6. 通信、per-layer window 与 TP all-reduce selector

- **stacked all-reduce window**：dense attention/MLP 与 MoE attention/shared-expert
  各自分配 tmp/signal stack，再按 layer offset 切出 `[BATCH,HIDDEN]` tmp slice 和
  一个 signal slice；不是旧生成链的每层独立 `_L{pos}` alloc。
- **512B signal stride**：`COMM_CONTROL_SIGNAL_BYTES=512`，
  `COMM_SIGNAL_STRIDE_I32=128`；每个逻辑 signal slice 为 `[128,1]` INT32，
  物理独占 512 B。
- **Main 单行 selector**：所有 TP rank 一致的 `active_rows == 1` 时，静态
  `1×4096` self-TPUT → Wave 1 publication → 固定 rank `0..7` 顺序完整行
  remote-load + 单 FP32 accumulator → 一次 BF16 cast → Wave 2 completion。
- **静态 fallback**：Main `active_rows != 1` 与 MTP 都走静态三波
  reduce-scatter + push all-gather + final local copy。MTP 三个调用传静态 `BATCH`。
- **ownership 与 transfer grain 解耦**：
  `TP_ALL_REDUCE_OWNED_CHUNK = HIDDEN // TP_WORLD_SIZE = 512` 决定 rank ownership；
  `TP_ALL_REDUCE_CHUNK` 只决定 self-TPUT/final-copy 搬运粒度。
- **rank-uniform 合同**：同一次 collective 的所有 rank 必须得到相同
  `active_rows`，否则 selector 分叉会死锁。
- **dense 调用 ABI**：仓内 Main 给 `dense_mlp_body_tp` 传运行时 `num_tokens`，
  MTP 传静态 `BATCH`；仓外 direct/inline 调用方升级时必须同步补该实参。
- collective 数据面按 selector 分支使用 `pld.tensor.put`、`pld.tile.remote_load`
  与 `pld.tile.remote_store`，控制面使用 `pld.system.notify/wait`
  （`NotifyOp.AtomicAdd`/`Set`，`WaitCmp.Ge`）。

## 7. KV 与权重（数据结构 + IPC）

**KV**（host_orch `:27577-27585`，`config.py:59-62`）：
- `seq_lens[tp,USER_BATCH_DYN] INT32`、`block_table[tp,512] INT32`（`MAX_BLOCKS_PER_SEQ 32 × BATCH 16`）、`slot_mapping[tp,USER_BATCH_DYN]`、`k/v_cache[tp,KV_CACHE_ROWS_DYN=4096,128] BF16`。
- KV-IPC：`tools/step3p5/pypto_kv_ipc.py:96` `import_kv_ipc_all` → `rt.import_ipc_all` → 每 rank `KvIpcMap`；holder `kv_ipc=True` 时绑 `k/v_cache`（`whole_decode_holder.py:202`）。

**权重**：
- `tools/step3p5/pypto_weight_ipc.py`：`import_weights_all`(508)；`WeightIpcMap`(454) `peer_base = rt.import_ipc(key, worker_id)`，按 byte offset 寻址。
- `StackedDeviceTensor`（`pypto.runtime.device_tensor`）：`build_stacked_weight(wmaps,key)` 组每 rank shard；`W_reshape`(holder `:197`) 自定义 per-rank shape。
- **三分类 slice**（`weight_loader.py`）：REPLICATED / TP-sliced / EP-sliced；`expected_shapes(tp)`(`:203`) 是 canonical shape 表；slice helper `_slice_q_proj`(539)/`_slice_kv_proj`(551)/`_slice_o_proj`(563)/`_slice_g_proj`(575)/`_slice_mlp_col`(601)/`_slice_mlp_row`(613)/`_slice_lm_head`(626)。

## 8. 关键位置速查

| 主题 | 位置 |
|------|------|
| program 类 / binding | `decode_fwd.py:WholeDecodeStep3p5` / `whole_decode_step3p5` |
| host_orch（出 pre-final-norm hidden） | `decode_fwd.py:host_orch` |
| 512B stacked/reused control slot | `decode_fwd.py:COMM_CONTROL_SIGNAL_BYTES/COMM_SIGNAL_STRIDE_I32` |
| tp_all_reduce / dispatch / combine | `decode_fwd.py:tp_all_reduce/dispatch_step/combine_step` |
| gate / dispatch / combine | `gate.py:112` / `dispatch.py:140+` / `combine.py:95+` |
| INT8 transform | `tools/step3p5/_a5_int8_transform.py` |
| dispatch/combine pull patch | `tools/step3p5/_patch_moepy_dispatch.py` / `_patch_combine_pull.py` |
| TP all-reduce 当前算法与验证 | [`../performance/03-tp-allreduce-algorithm-comparison.md`](../performance/03-tp-allreduce-algorithm-comparison.md#8-hccl-small-message-selector-思路迁移版2026-08-12) |
| weight/KV IPC | `tools/step3p5/pypto_weight_ipc.py` / `pypto_kv_ipc.py` |
| holder | `tools/step3p5/whole_decode_holder.py` |
| host 侧 final norm+lm_head（standalone argmax） | `tools/step3p5/final_logits_from_vllm.py` |
| 生成器 | `tools/step3p5/_gen_faithful_real.py` |

## 9. 不变量清单（改代码前对照）

1. 只有一个 whole-net `@pl.program`；TP all-reduce tmp/signal 使用 dense/MoE
   family stack，并以 layer offset 切出互不别名的 per-layer slice。
2. routed 权重 INT8 + FP32 scale；不引入 BF16-dequant。
3. dispatch/combine 都是 pull；pull 循环用静态 bound + compound-scalar 定槽，不读 runtime count。
4. `tp_all_reduce` 保留 rank-uniform selector：单行静态两波；其他 Main/MTP
   静态三波；固定 peer 顺序、FP32 累加、ownership/transfer-grain 解耦不得改变。
5. 生成器改动后必须 strip→regenerate→byte-compare（roundtrip gate）。
6. 单卡 ST/UT 用 `apply_perrank_patch()`（保 TP=8 per-rank slice 宽度），不用 unslice。

## 10. 相关文档

- 系统设计：[`01-system-design.md`](01-system-design.md)
- 复盘：[`../../postmortems/07-whole-net-scheduler-timeout.md`](../../postmortems/07-whole-net-scheduler-timeout.md) · [`08`](../../postmortems/08-multiprogram-coprepare-deadlock.md) · [`10`](../../postmortems/10-gap5-attention-quant-scope.md)
- 强约束 skill：`.claude/skills/pypto-dev-constraints/` · hang 排查：`.claude/skills/pypto-whole-net-hang-debug/`
- kernel 硬限制：`pypto-lib/docs/known-pypto-pitfalls.md`
