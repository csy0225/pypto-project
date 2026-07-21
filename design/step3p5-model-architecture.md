# step3p5 模型架构

> 本文只描述 **step3p5 模型本身**（config + 层结构 + 完整层数流程图），与"怎么在
> pypto 上实现/在 vLLM 里集成"解耦。实现见 [`whole-net/`](whole-net/)，集成见
> [`vllm-pypto/`](vllm-pypto/)。参数源：`pypto-lib/models/step3p5/config.py`
> （对齐 `step3p5_flash_release_hf_mtp3` checkpoint）。

## 1. Config 参数

### 1.1 全局

| 参数 | 值 | 说明 |
|------|----|------|
| `HIDDEN` | 4096 | hidden size |
| `VOCAB` | 128896 | 词表 |
| `NUM_HIDDEN_LAYERS` | 45 | 主 decoder 层数 |
| `NUM_NEXTN_PREDICT_LAYERS` | 3 | MTP（multi-token predict）层，ckpt 索引 45..47 |
| `NUM_TOTAL_LAYERS` | 48 | 45 + 3 |
| `MAX_POSITION_EMBEDDINGS` | 262144 | 最大位置 |
| `EPS` | 1e-5 | RMSNorm epsilon |
| `ZERO_CENTERED_NORM` | True | RMSNorm 有效 gamma = stored_gamma + 1.0 |

### 1.2 Attention（两种变体，逐层由 `LAYER_TYPES` 选）

| 参数 | full_attention | sliding_attention (SWA) |
|------|----------------|--------------------------|
| 每层出现规律 | 每 4 层一次（i % 4 == 0） | 其余 3/4 |
| `NUM_HEADS` | 64（q hidden 8192） | 96（q hidden 12288） |
| `NUM_KV_HEADS` | 8（KV hidden 1024，GQA） | 8 |
| `Q_PER_KV` | 8 | 12 |
| `HEAD_DIM` | 128 | 128 |
| `SLIDING_WINDOW` | —（全局） | 512 tokens |
| RoPE θ | 5e6（+ YaRN scaling） | 1e4 |
| partial rotary | 0.5（rotary_dim=64） | 1.0（rotary_dim=128） |
| `ATTN_SCALE` | 1/√128 | 1/√128 |

公共开关：`USE_QK_NORM=True`（per-head q_norm/k_norm）、
`USE_HEAD_WISE_ATTN_GATE=True`（per-head sigmoid gate `g_proj`，o_proj 前）。

### 1.3 FFN

| 类型 | 层 | 参数 |
|------|----|------|
| **dense MLP** | L0, L1, L2 | `INTERMEDIATE = 11264`（SwiGLU：gate/up/down） |
| **MoE** | L3 … L44（42 层） | 见下 |

**MoE**：`MOE_NUM_EXPERTS=288`、`MOE_TOP_K=8`、`MOE_INTERMEDIATE=1280`（routed
expert hidden）、`SHARE_EXPERT_DIM=1280`（shared expert）、routing =
`sigmoid` + 学习偏置（`USE_MOE_ROUTER_BIAS=True`）+ top-8 renorm
（`NORM_EXPERT_WEIGHT=True`）× `MOE_ROUTER_SCALING_FACTOR=3.0`。

### 1.4 层类型表（`LAYER_TYPES`，`[full, swa, swa, swa] × 12`）

```
idx : 0  1  2  3  4  5  6  7  8 ... 44 | 45 46 47 (MTP)
type: F  S  S  S  F  S  S  S  F ... F  | S  S  S
FFN : D  D  D  M  M  M  M  M  M ... M  | D  D  D
```

`F`=full_attention `S`=sliding_attention · `D`=dense MLP `M`=MoE。
主 45 层 = **12 full + 33 swa**（attention）× **3 dense + 42 MoE**（FFN）。
判定：`is_full_attention(li)` = `i % 4 == 0`；`is_moe_layer(li)` = `3 ≤ i < 45`。

## 2. 完整层数流程图

```mermaid
flowchart TD
    IN["input token id"] --> EMB["embed_tokens<br/>[·, HIDDEN=4096]"]

    EMB --> L0

    subgraph MAIN["主 decoder · 45 层"]
        direction TB
        L0["L0 · full-attn + dense MLP"]
        L123["L1, L2 · swa-attn + dense MLP"]
        LMOE["L3 … L44（42 层）· swa/full-attn + MoE<br/>(full 出现在 L4,L8,…,L44；其余 swa)"]
        L0 --> L123 --> LMOE
    end

    LMOE --> FN["final RMSNorm"]
    FN --> LM["lm_head [VOCAB=128896, HIDDEN]"]
    LM --> LOGITS["logits → argmax/sample → next token"]

    LMOE -. 末层 hidden .-> MTP
    subgraph MTP["MTP · 3 层（speculative, 可选）"]
        direction TB
        M["L45, L46, L47 · swa-attn + dense MLP<br/>(next-token predict)"]
    end
    MTP --> FN

    classDef a fill:#4C6EF5,stroke:#1E3A8A,color:#fff;
    classDef m fill:#12B886,stroke:#0B7285,color:#fff;
    classDef t fill:#BE4BDB,stroke:#6B2178,color:#fff;
    classDef x fill:#F59F00,stroke:#B36A00,color:#fff;
    class EMB,IN a; class L0,L123 a; class LMOE m; class FN,LM,LOGITS t; class M,MTP x;
```

## 3. 单层内部结构

### 3.1 attention 层（full / swa 同构，仅头数/窗口/RoPE 不同）

```mermaid
flowchart LR
    H["hidden"] --> RN1["input RMSNorm"]
    RN1 --> QKV["q/k/v proj (GQA: 64或96 q head, 8 kv head)"]
    QKV --> QN["q_norm / k_norm (per-head)"]
    QN --> ROPE["RoPE (full: rot64,θ5e6+YaRN / swa: rot128,θ1e4)"]
    ROPE --> FA["flash attention (swa: window 512) + paged KV"]
    FA --> GATE["head-wise sigmoid gate (g_proj)"]
    GATE --> OP["o_proj"]
    OP --> ADD1["+ residual"]
    classDef a fill:#4C6EF5,stroke:#1E3A8A,color:#fff;
    class H,RN1,QKV,QN,ROPE,FA,GATE,OP,ADD1 a;
```

### 3.2 MoE 层 FFN（L3..L44）

```mermaid
flowchart LR
    X["hidden"] --> RN2["post-attn RMSNorm"]
    RN2 --> GT["router: sigmoid + bias → top-8 renorm × 3.0"]
    RN2 --> SH["shared expert (dim 1280)"]
    GT --> DP["dispatch (288 experts, top-8)"]
    DP --> EX["routed expert SwiGLU (inter 1280)"]
    EX --> CB["combine (加权求和)"]
    SH --> SUM["+"]
    CB --> SUM --> ADD2["+ residual"]
    classDef m fill:#12B886,stroke:#0B7285,color:#fff;
    class X,RN2,GT,SH,DP,EX,CB,SUM,ADD2 m;
```

（dense MLP 层把 §3.2 的 router/dispatch/experts/combine 换成单个 SwiGLU，
`INTERMEDIATE=11264`。）

## 4. 相关文档

- 整网怎么在 pypto 上实现（TP=8/EP=8、单 `@pl.program`）：[`whole-net/01-system-design.md`](whole-net/01-system-design.md)
- 怎么接进 vLLM serving：[`vllm-pypto/01-system-design.md`](vllm-pypto/01-system-design.md)
- 参数权威源：`pypto-lib/models/step3p5/config.py`
