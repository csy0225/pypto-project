# vLLM ↔ pypto Operator 映射

vLLM `Step3p5Model`（torch eager）与 pypto `decode_fwd`（编译 kernel）
之间的 op 级对应。本 doc 是 Phase 20 `weight_translate.py` 实现者和
整模型 monkey-patch 实现者需要的参考。

## 纵向对比

```
vLLM 侧                                pypto 侧
─────────────────────────────────      ──────────────────────────────────
Step3p5Model.forward(                  pypto.ir.compile(
  input_ids,                             Step3p5DecodeFwd,
  positions,                             distributed_config=...
  intermediate_tensors=None,           )
  inputs_embeds=None,                  ↓
)                                       编译后 callable
  ↓                                       ↓
  hidden_states = embed_input_ids        （host 从 vLLM forward
    OR inputs_embeds                      context 填 inputs dict）
  ↓                                       ↓
  for i in range(start, end):             decode_fwd 一次跑完
    hidden_states = self.layers[i](       全部 45 层 + lm_head
      positions, hidden_states            ↓
    )                                     logits [B, VOCAB_LOCAL]
                                          （TP 切的；vLLM sampler 想要
  ↓                                        全 VOCAB 时之后要 gather）
  hidden_states
```

## 每层对应

每个 `Step3p5DecoderLayer.forward(positions, hidden_states)` call，
等价的 pypto 计算是 `decode_fwd` 45 层循环中的一片。Op-by-op：

| 步骤 | vLLM Step3p5DecoderLayer | pypto decode_layer（dense 或 MoE 变体） |
|------|--------------------------|----------------------------------------|
| 1 | `input_layernorm(hidden_states)` | zero-centered RMSNorm（`_ops.py:_zero_centered_rmsnorm`） |
| 2 | `self_attn(positions, hidden_states)` —— QKV proj + RoPE + paged KV cache update + flash attention + (head_gate? sigmoid) + out_proj | `attention_full.py` / `attention_swa.py` 做 QKV proj + Q/K head-wise zc RMS norm + partial / full RoPE + KV cache slot write + online-softmax flash attention + head_gate **× 1 旁路** + out_proj |
| 3 | residual add | residual add |
| 4 | `post_attention_layernorm` | zero-centered RMSNorm |
| 5 | 如果 dense 走 `mlp(hidden_states)` | gate_up matmul → SiLU(gate) * up → down matmul（`decode_layer.py:_dense_mlp_body_tp`） |
| 5 | 如果 MoE 走 `moe(hidden_states)` | gate（top-k routing + bias）→ dispatch（EP a2a）→ routed experts MLP（每 rank 36）→ combine（加权 gather）+ shared expert add（`moe.py` chip_orch） |
| 6 | `tp_all_reduce`（vLLM 走 HCCL） | `tp_all_reduce`（pypto 走 simpler shmem-IPC window） |
| 7 | residual add | residual add |

加所有层结束后：

| 步骤 | vLLM Step3p5Model | pypto decode_fwd |
|------|-------------------|-------------------|
| 8 | 最终 RMSNorm | 最终 RMSNorm |
| 9 | `lm_head`（logit-processor 侧独立 torch 模块） | `rms_lm_head` 每 rank VOCAB_LOCAL 切片 |

## 每层状态映射

| vLLM Step3p5DecoderLayer state | pypto kernel input(s) |
|--------------------------------|------------------------|
| `self_attn.qkv_proj.weight` | `wq`（Q 部分）+ `wk`（K 部分）+ `wv`（V 部分）—— 从 vLLM 拼接的 [Q\|K\|V] 切出来 |
| `self_attn.q_norm.weight` | `q_norm_weight`（每层 `[HEAD_DIM]`） |
| `self_attn.k_norm.weight` | `k_norm_weight`（每层 `[HEAD_DIM]`） |
| `self_attn.head_gate.weight`（FP32） | `w_g`（BF16，0-padded 到 `NUM_HEADS_*_LOCAL_PAD=16`）—— 当前旁路；见 blocker §3 |
| `self_attn.o_proj.weight` | `wo` |
| `self_attn.rotary_emb.cos_sin_cache` | `rope_cos` + `rope_sin`（从 vLLM joint cache 拆出） |
| `mlp.gate_up_proj.weight` | `dense_gate` + `dense_up`（从 vLLM 拼接的 [gate\|up] 拆出） |
| `mlp.down_proj.weight` | `dense_down` |
| `moe.experts.w13_weight`（packed [w1\|w3]） | `w_gate_r` + `w_up_r`（拆，per-expert） |
| `moe.experts.w2_weight` | `w_down_r` |
| `moe.shared_experts.gate_up_proj.weight`（packed） | `w_gate_s` + `w_up_s`（拆） |
| `moe.shared_experts.down_proj.weight` | `w_down_s` |
| `moe.gate.weight` | `gate_w` |
| `moe.gate.bias`（如存在） | `router_bias` |
| `input_layernorm.weight` | `input_rms_weight`（每层 `[HIDDEN]`） |
| `post_attention_layernorm.weight` | `post_attn_rms_weight` |

Weight loader（`pypto-lib/models/step3p5/weight_loader.py:197
expected_shapes`）返回 30-key dict，里面有 pypto kernel 期望的全 layer
堆叠 tensor。`weight_translate.py`（Phase 20 任务 1.2）需要走 vLLM
`model.named_parameters()` 重新搭这个 30-key dict。

## Forward context state 映射

每个 decode step 两边都要的 state：

| vLLM AttentionMetadata 字段 | pypto kernel input |
|------------------------------|---------------------|
| `seq_lens`（每 batch 上下文长度） | `seq_lens` `[B]` INT32 |
| `block_tables`（每 batch, [B, MAX_BLOCKS_PER_SEQ]） | `block_table` 展平 `[B * MAX_BLOCKS_PER_SEQ]` INT32 |
| `slot_mapping`（每 batch，当前 position 写哪） | `slot_mapping` `[B]` INT32 |
| `positions`（每 batch position index） | 从 `seq_lens - 1` 派生 |
| `kv_cache[layer_idx]` `[num_blocks, block_size, num_kv_heads, head_dim]` | `k_cache` + `v_cache` 平 view `[KV_CACHE_ROWS_DYN, HEAD_DIM]` |

这些由 Phase 20 任务 1.3（`kv_bridge.py`）和 1.4
（`attn_meta_bridge.py`）桥接。

## **不**映射的（vLLM 侧，Phase 20 不动）

下面这些 vLLM 组件按原样跑 —— pypto kernel 不管：

- Tokenizer
- Sampler（top-k / top-p / temperature）
- Vocab parallel embedding（输入侧）
- Continuous batching scheduler
- Block manager（KV cache 页分配）
- 请求生命周期 / 序列跟踪

Monkey-patch surface **只有** `Step3p5Model.forward`。上面下面都留给
vLLM。

## 已知语义差（vLLM eager vs pypto kernel）

| 差异 | vLLM | pypto | 对精度对齐的影响 |
|------|------|-------|------------------|
| `head_gate` | 每头 apply `sigmoid(head_gate_logits)` | × 1 identity 旁路 | 每层 attention 输出 ~2× 量级；45 层累积让 hidden_states 漂得明显。**Blocker §3** —— 通过 Phase 21 §2.7 标定。 |
| `tp_all_reduce` backend | HCCL | simpler shmem-IPC | 数值等价（都是 sum）。无精度影响。 |
| 数值 accumulator | torch eager matmul/RMS FP32 accumulator | pypto FP32 accumulator（匹配） | 可忽略。Phase 21 L1 容忍 `ratio_allclose(atol=0.04, rtol=0.04)` 容得下 bf16 rounding noise。 |

## 相关文档

- [`overview.md`](../00-context-and-goals.md) —— 高层系统视图
- [`../phases/20-vllm-backend-monkey-patch.md`](../../archive/completed-phases/20-vllm-backend-monkey-patch.md)
  —— 消费本映射的实现计划
- [`../phases/21-precision-validation.md`](../../archive/completed-phases/21-precision-validation.md)
  —— 三层对比 harness
- vLLM 源：`<vllm_repo>/vllm/model_executor/models/step3p5.py` HEAD
  `0e0901376`
- pypto 源：`pypto-lib/models/step3p5/decode_fwd.py:198`
  （`_build_decode_fwd_program`）和 `weight_loader.py:197`
  （`expected_shapes`）
