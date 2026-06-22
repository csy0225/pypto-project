# vLLM ↔ pypto Operator Mapping

Operator-level correspondence between vLLM's `Step3p5Model` (torch
eager) and pypto's `decode_fwd` (compiled kernel). This doc is the
reference Phase 20's `weight_translate.py` and the whole-model
monkey-patch implementer needs.

## Vertical comparison

```
vLLM side                              pypto side
─────────────────────────────────      ──────────────────────────────────
Step3p5Model.forward(                  pypto.ir.compile(
  input_ids,                             Step3p5DecodeFwd,
  positions,                             distributed_config=...
  intermediate_tensors=None,           )
  inputs_embeds=None,                  ↓
)                                       compiled callable
  ↓                                       ↓
  hidden_states = embed_input_ids        (host fills inputs dict from
    OR inputs_embeds                      vLLM forward context)
  ↓                                       ↓
  for i in range(start, end):             decode_fwd executes all
    hidden_states = self.layers[i](       45 layers + lm_head as
      positions, hidden_states            one fused kernel
    )                                     ↓
                                          logits [B, VOCAB_LOCAL]
  ↓                                       (TP-sliced; need gather
  hidden_states                            after if vLLM sampler wants
                                           full VOCAB)
```

## Per-layer correspondence

For each `Step3p5DecoderLayer.forward(positions, hidden_states)` call,
the equivalent pypto compute is one slice of `decode_fwd`'s 45-layer
loop. Operation-by-operation:

| Step | vLLM Step3p5DecoderLayer | pypto decode_layer (dense or MoE variant) |
|------|--------------------------|-------------------------------------------|
| 1 | `input_layernorm(hidden_states)` | zero-centered RMSNorm (`_ops.py:_zero_centered_rmsnorm`) |
| 2 | `self_attn(positions, hidden_states)` — QKV proj + RoPE + paged KV cache update + flash attention + (head_gate? sigmoid) + out_proj | `attention_full.py` / `attention_swa.py` doing QKV proj + Q/K head-wise zc RMS norm + partial / full RoPE + KV cache slot write + online-softmax flash attention + head_gate **× 1 bypass** + out_proj |
| 3 | residual add | residual add |
| 4 | `post_attention_layernorm` | zero-centered RMSNorm |
| 5 | `mlp(hidden_states)` if dense | gate_up matmul → SiLU(gate) * up → down matmul (`decode_layer.py:_dense_mlp_body_tp`) |
| 5 | `moe(hidden_states)` if MoE | gate (top-k routing + bias) → dispatch (EP a2a) → routed experts MLP (36 per rank) → combine (weighted gather) + shared expert add (`moe.py` chip_orch) |
| 6 | `tp_all_reduce` (vLLM via HCCL) | `tp_all_reduce` (pypto via simpler shmem-IPC window) |
| 7 | residual add | residual add |

Plus at end of all layers:

| Step | vLLM Step3p5Model | pypto decode_fwd |
|------|-------------------|-------------------|
| 8 | final RMSNorm | final RMSNorm |
| 9 | `lm_head` (separate torch module on logit-processor side) | `rms_lm_head` per-rank VOCAB_LOCAL slice |

## Per-layer state mapping

| vLLM Step3p5DecoderLayer state | pypto kernel input(s) |
|---------------------------------|------------------------|
| `self_attn.qkv_proj.weight` | `wq` (Q part) + `wk` (K part) + `wv` (V part) — split from vLLM's concatenated [Q\|K\|V] |
| `self_attn.q_norm.weight` | `q_norm_weight` (per-layer `[HEAD_DIM]`) |
| `self_attn.k_norm.weight` | `k_norm_weight` (per-layer `[HEAD_DIM]`) |
| `self_attn.head_gate.weight` (FP32) | `w_g` (BF16, zero-padded to `NUM_HEADS_*_LOCAL_PAD=16`) — currently bypassed; see blocker §3 |
| `self_attn.o_proj.weight` | `wo` |
| `self_attn.rotary_emb.cos_sin_cache` | `rope_cos` + `rope_sin` (split from vLLM's joint cache) |
| `mlp.gate_up_proj.weight` | `dense_gate` + `dense_up` (split from vLLM's concatenated [gate\|up]) |
| `mlp.down_proj.weight` | `dense_down` |
| `moe.experts.w13_weight` (packed [w1\|w3]) | `w_gate_r` + `w_up_r` (split, per-expert) |
| `moe.experts.w2_weight` | `w_down_r` |
| `moe.shared_experts.gate_up_proj.weight` (packed) | `w_gate_s` + `w_up_s` (split) |
| `moe.shared_experts.down_proj.weight` | `w_down_s` |
| `moe.gate.weight` | `gate_w` |
| `moe.gate.bias` (if exists) | `router_bias` |
| `input_layernorm.weight` | `input_rms_weight` (per-layer `[HIDDEN]`) |
| `post_attention_layernorm.weight` | `post_attn_rms_weight` |

The weight loader (`pypto-lib/models/step3p5/weight_loader.py:197
expected_shapes`) returns a 30-key dict with the full layer-stacked
tensors that the pypto kernel expects. `weight_translate.py` (Phase 20
task 1.2) needs to walk vLLM `model.named_parameters()` and rebuild
this 30-key dict.

## Forward context state mapping

Per-decode-step state needed by both sides:

| vLLM AttentionMetadata field | pypto kernel input |
|-------------------------------|---------------------|
| `seq_lens` (per-batch context length) | `seq_lens` `[B]` INT32 |
| `block_tables` (per-batch, [B, MAX_BLOCKS_PER_SEQ]) | `block_table` flattened `[B * MAX_BLOCKS_PER_SEQ]` INT32 |
| `slot_mapping` (per-batch, where current position writes) | `slot_mapping` `[B]` INT32 |
| `positions` (per-batch position index) | derived from `seq_lens - 1` |
| `kv_cache[layer_idx]` `[num_blocks, block_size, num_kv_heads, head_dim]` | `k_cache` + `v_cache` flat views `[KV_CACHE_ROWS_DYN, HEAD_DIM]` |

These are bridged by Phase 20 tasks 1.3 (`kv_bridge.py`) and 1.4
(`attn_meta_bridge.py`).

## What's NOT mapped (vLLM-side, untouched by Phase 20)

These vLLM components run as-is — pypto kernel does not handle them:

- Tokenizer
- Sampler (top-k / top-p / temperature)
- Vocab parallel embedding (input side)
- Continuous batching scheduler
- Block manager (KV cache page allocation)
- Request lifecycle / sequence tracking

The monkey-patch surface is **only** `Step3p5Model.forward`. Everything
above and below stays in vLLM hands.

## Known semantic differences (vLLM eager vs pypto kernel)

| Difference | vLLM | pypto | Effect on precision parity |
|------------|------|-------|----------------------------|
| `head_gate` | applies `sigmoid(head_gate_logits)` per head | × 1 identity bypass | ~2× attention output magnitude per layer; cumulative across 45 layers makes hidden_states diverge meaningfully. **Blocker §3** — calibrate via Phase 21 §2.7. |
| `tp_all_reduce` backend | HCCL | simpler shmem-IPC | Numerical equivalence (both sum). No precision effect. |
| Numerical accumulator | torch eager FP32 accumulators in matmul/RMS | pypto FP32 accumulators (matches) | Negligible. Phase 21 L1 tolerance `ratio_allclose(atol=0.04, rtol=0.04)` accommodates bf16 rounding noise. |

## Related docs

- [`overview.md`](overview.md) — high-level system view
- [`../phases/20-vllm-backend-monkey-patch.md`](../phases/20-vllm-backend-monkey-patch.md)
  — implementation plan that consumes this mapping
- [`../phases/21-precision-validation.md`](../phases/21-precision-validation.md)
  — three-tier comparison harness
- vLLM source: `<vllm_repo>/vllm/model_executor/models/step3p5.py`
  HEAD `0e0901376`
- pypto source: `pypto-lib/models/step3p5/decode_fwd.py:198`
  (`_build_decode_fwd_program`) and `weight_loader.py:197`
  (`expected_shapes`)
