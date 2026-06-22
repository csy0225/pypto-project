# System Architecture Overview

The pypto step3p5 stack assembles 5 code repos + 1 integration target
into an end-to-end serving system. This doc shows what each piece does
and where data flows.

## Big picture

```
                          ┌─────────────────────────────┐
                          │  vLLM stepcast fork         │
                          │  (Phase 2 integration       │
                          │   target — gitlab-internal) │
                          │                              │
                          │  • LLM engine + scheduler    │
                          │  • Continuous batching       │
                          │  • Sampler + tokenizer       │
                          │  • Paged KV cache mgr        │
                          │  • Step3p5Model.forward      │
                          │                              │
                          │  patched via                 │
                          │  pypto.step3p5.vllm_backend  │
                          │  (in pypto-lib)              │
                          └──────────────┬──────────────┘
                                         │ calls
                                         ▼
                          ┌─────────────────────────────┐
   pypto-lib             │  decode_fwd                  │
   "model + kernels"     │  (one fused 45-layer kernel) │
                         └──────────────┬──────────────┘
                                        │ compiled by
                                        ▼
   pypto                 ┌─────────────────────────────┐
   "framework"           │  pypto.ir.compile            │
                         │  • multi-level IR            │
                         │  • codegen passes            │
                         │  • emits .so + .bin          │
                         └──────────────┬──────────────┘
                                        │ uses
                                  ┌─────┴─────┐
                                  ▼           ▼
   pto-isa                 ┌──────────┐  ┌──────────────────────┐
   "tile library"          │ pto-isa  │  │      PTOAS            │
                           │ virtual  │  │   bytecode assembler │
                           │ tile ISA │  │      (= ptoas-bin)   │
                           └──────────┘  └──────────────────────┘
                                        ▲
                                        │ produces bytecode for
                                        │
                          ┌─────────────────────────────┐
   simpler               │  PTO runtime                 │
   "execution layer"     │  • AICPU + AICore dispatcher │
   (submodule of pypto)  │  • inter-card IPC (shmem)    │
                         │  • collectives               │
                         └──────────────┬──────────────┘
                                        │ runs on
                                        ▼
                          ┌─────────────────────────────┐
                          │  Ascend 910B / A2A3          │
                          │  • driver 25.5.2             │
                          │  • firmware 7.8.0.7.220      │
                          │  • CANN 9.0.0-beta.1         │
                          └─────────────────────────────┘
```

## Repo roles

### pypto

Programming framework. Provides the Python DSL (`pypto.language`,
`pypto.language.distributed`), multi-level IR, and codegen passes.
Compiles a `@pl.program` to PTOAS bytecode + a host-side `.so` for
dispatch.

Runtime: `pypto/runtime/` is a git submodule that points to **simpler**.

### pypto-lib

Tensor-level kernel implementations and end-to-end LLM models. Hosts
the step3p5 family:

- `models/step3p5/decode_fwd.py` — 45-layer fused decode + lm_head
- `models/step3p5/decode_layer.py` — per-layer dispatcher (dense vs MoE)
- `models/step3p5/{attention_full,attention_swa}.py` — attention variants
- `models/step3p5/moe.py` + 5 MoE component files — MoE block
- `models/step3p5/weight_loader.py` — HF safetensors → per-rank bundle

Phase 2 integration code will live at
`models/step3p5/vllm_backend/` (Phase 20 task 1.1+).

### pto-isa

Tile-ISA virtual implementations. Defines tile operations (matmul,
reduce, broadcast, etc.) that pypto codegen lowers into. Hardware-
specific (Ascend 910B for our case).

### PTOAS

LLVM/MLIR-based bytecode assembler. Takes the MLIR pypto emits and
produces device bytecode + dispatch metadata.

Binary distribution: `ptoas-bin` (currently v0.45) — the assembler we
actually run; PTOAS source repo is for reference / building from
scratch.

### simpler (pypto submodule)

PTO runtime. Manages task dispatch across AICPU + AICore, inter-card
IPC via shmem windows, collective primitives. The most platform-touchy
component — Phase 16 three-pillars binding exists primarily to make
simpler work.

### vLLM stepcast fork (Phase 2 target)

Internal fork of vLLM with step3p5 model implementation
(`vllm/model_executor/models/step3p5.py`). Provides everything around
the decoder: tokenizer, sampler, KV cache manager, request scheduler,
continuous batching.

Phase 2 integration: monkey-patch `Step3p5Model.forward` to call
pypto-compiled `decode_fwd` instead of torch eager. See
[`vllm-step3p5-mapping.md`](vllm-step3p5-mapping.md).

## Data flow at decode time (after Phase 2 v0.1)

```
User prompt
    │
    ▼
vLLM tokenizer ─────────────► token_ids
    │
    ▼
vLLM scheduler ─────────────► batch (B requests)
    │
    ▼
vLLM Step3p5Model.forward (monkey-patched)
    │
    ▼
pypto decode_fwd (compiled .so)
    │
    ├─► 45 layers (dense / MoE mixed-mode)
    │       │
    │       ├─► attention: QKV + RMS norm + RoPE + paged KV cache update + flash
    │       └─► MoE or dense MLP
    │
    └─► lm_head + rms norm
    │
    ▼
Logits [B, VOCAB]
    │
    ▼
vLLM Sampler ─────────────► next_token_id per batch element
    │
    ▼
vLLM appends to seq_lens, KV cache slot_mapping advances, loops
```

KV cache lives in HBM, layout shared between vLLM-side allocation and
pypto kernel access via zero-copy view (see Phase 20 task 1.3
`kv_bridge.py`).

## Build dependency order

When rebuilding from source:

1. `pypto` (the framework — provides the Python DSL pypto-lib imports)
2. `pto-isa` (used at codegen time)
3. `PTOAS` (used at codegen time; usually replaced by `ptoas-bin`)
4. `simpler` (submodule under pypto/runtime)
5. `pypto-lib` (depends on all of the above)

The deploy host typically only needs:
- `pypto-lib` source
- `pypto` installed (`pip install -e <workspace>/pypto`)
- `simpler` built and installed (via pypto submodule build)
- `pto-isa` source (referenced by `$PTO_ISA_ROOT`)
- `ptoas-bin` (in `$PATH` and `$LD_LIBRARY_PATH`)

See [`../deployment/version-matrix.md`](../deployment/version-matrix.md)
for pinned versions.

## Related docs

- [`vllm-step3p5-mapping.md`](vllm-step3p5-mapping.md) — vLLM ↔ pypto
  operator mapping for Phase 20 monkey-patch
- [`../phases/20-vllm-backend-monkey-patch.md`](../phases/20-vllm-backend-monkey-patch.md)
  — Phase 20 design that consumes this overview
- [`../deployment/phase16-three-pillars.md`](../deployment/phase16-three-pillars.md)
  — hardware platform binding
