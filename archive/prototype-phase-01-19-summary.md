# PyPTO Kernel Prototype — Development Archive (Phase 01-19)

This document archives the pypto step3p5 kernel prototype development
journey from initial design (Phase 01) through MoE single-card ST
blocker identification (Phase 19), spanning May-June 2026.

Subsequent work (Phase 20+: vLLM Ascend backend integration via
monkey-patch) is tracked in:

- Live tracker: `<workspace>/pypto/CLAUDE.md` (outer, local-only)
- In-repo phase docs: `pypto-lib/docs/step3p5/phases/20+`
- Active blockers carried forward: [`known-blockers.md`](known-blockers.md)

## What was built

The pypto step3p5 kernel suite — a 48-layer (45 hidden + 3 MTP) decoder
implementation on Ascend NPUs (910B/C platform target `a2a3`) using
the pypto / pto-isa / simpler / PTOAS toolchain stack. Component
inventory:

| Component | Files | Role |
|-----------|-------|------|
| Attention (full / SWA) | `attention_full.py`, `attention_swa.py`, `prefill_attention_*.py`, `prefill_qkv_proj_rope.py` | QKV projection + RMS-norm + RoPE + paged KV-cache + flash-attention; full and SWA variants per layer-types table |
| MoE block | `gate.py`, `dispatch.py`, `expert_routed.py`, `expert_shared.py`, `combine.py`, `moe.py`, `prefill_moe.py` | Top-k routing + EP all-to-all + 36-expert-per-rank routed + shared expert + EP all-to-all back + weighted gather |
| Decode layer dispatcher | `decode_layer.py` | Per-layer dense vs MoE routing via `select_decode_layer(layer_idx)` |
| Decode forward composer | `decode_fwd.py` | 45-layer fused composition + final RMS + `rms_lm_head.py` lm-head per-rank slice |
| MTP | `mtp.py` | 3 next-token-predict layers (not yet wired into decode_fwd) |
| Prefill family | `prefill_fwd.py`, `prefill_*.py` | Per-layer + composer for initial prompt processing |
| Weight loader | `weight_loader.py` | HF safetensors → 30-key per-rank flat-tensor bundle (`expected_shapes()` for any TP world size) |
| Top-level entrypoint | `step3p5_decode.py`, `step3p5_prefill.py` | CLI: smoke (CPU torch reference) + `run_real_npu` |

All files live under `pypto-lib/models/step3p5/`.

## Phase timeline

| Phase | Title | Status | Completion |
|------:|-------|--------|------------|
| 01 | Config baseline + migration plan | ✅ | 2026-05 |
| 02 | Checkpoint shape verification (jfs ckpt vs config.py) | ✅ | 2026-05 |
| 03 | Single-layer decode drafts (full / SWA) | ✅ | 2026-05 |
| 04 | Parametric attention + decode_layer dispatcher | ✅ | 2026-05 |
| 05 | MoE block (single-card) | ✅ | 2026-05 |
| 06 | decode_fwd 45 layers + lm_head | ✅ | 2026-05 |
| 07 | MTP (3 layers) | ✅ | 2026-05 |
| 08 | Prefill (single-card) | ✅ | 2026-05 |
| 09 | E2E integration + smoke + weights | ✅ | 2026-05 |
| 10 | TP=8 + EP=8 refactor | ✅ | 2026-05 |
| 11 | Driver / install.md cleanup | ✅ | 2026-06-04 |
| 12 | Frontend bring-up rc=0 (10 X-phase sub-tasks) | ✅ | 2026-06-04 |
| 13 | Re-sync to latest commit + smoke reverify | ✅ | 2026-06-05 |
| 14 | Pypto codegen pass (IR → PTOAS bytecode) | ✅ | 2026-06-08 (14.C-14.G ✅; prefill single-layer deferred to Phase 17) |
| 15 | Single-rank NPU bring-up | ✅ | 2026-06-15 (rc=0; 20 tasks complete; head_gate ×1 bypass + TP=1 patch + LAYER_*_ROWS_DYN overrides) |
| 16 | Multi-rank NPU + real weight load + toolchain upgrade | ✅ | 2026-06-19 (driver 25.5.2 + firmware 7.8.0.7.220 + CANN 9.0.0-beta.1; simpler L3 allreduce double-card golden match) |
| 17 | 64K prefill + 16-step decode e2e | ⏸ | Blocked by prefill MoE L1 overflow (TASK-29) |
| 18 | Performance: l2_swimlane + PMU | ⏸ | Deferred to Phase 22 (vLLM integration phase) |
| 19 | MoE single-card ST + precision alignment | ⏸ | 2026-06-17 (Blocker 1/2/3/4 ✅, MoE device runtime 507018 ⏸; 6 variants smoke 6/6 PASS, dense ST device PASS) |

Phase docs for 01-19 live in the outer tracker
`<workspace>/pypto/docs/step3p5/phases/` (loose, unversioned). They
were never migrated into the pypto-lib git tree because that decision
was made starting Phase 20.

## Hardware platform validation

### Phase 16 multi-card deployment requirements (production-critical)

The minimum viable deployment for any production multi-rank step3p5 run
is the **three-pillars** binding documented in the live tracker. As of
2026-06-22 only `gpu-a910x-0162` has all three:

| Component | Required | Failure mode if older |
|-----------|----------|------------------------|
| Driver | `25.5.2` | older: `support_shmem_map_exbus=0`, `aclrtIpcMemImportByKey` returns 507899 |
| Firmware | `7.8.0.7.220` (chip flash, persists across reboot) | older: same cap gap |
| CANN | `9.0.0-beta.1` (NOT GA — GA fails simpler init with 507018 BootstrapDispatcher) | GA: AICPU `libaicpu_extend_kernels.so` not pushed by TDT |

For the deployment runbook see
`runtime/.claude/skills/ascend-phase16-deploy/SKILL.md` in the simpler
runtime fork.

### Single-card validation results (2026-06-22 reverified post-reboot)

| Test | Path | Status | Time |
|------|------|--------|------|
| Frontend smoke | `_smoke_program_build.py` | ✅ rc=0 | <2s |
| Phase 16 baseline | `simpler L3 allreduce_distributed -p a2a3 -d 0-1` | ✅ `max\|out-expected\|=0` | seconds |
| ST-1 dense full | `test_decode_layer_full_dense_st -p a2a3 -d 0` | ✅ ratio_allclose PASS | 7.93s |
| ST-2 dense swa | `test_decode_layer_swa_dense_st -p a2a3 -d 0` | ✅ ratio_allclose PASS | 14.85s |
| MoE 6 variants smoke | `test_decode_layer_moe_st --variant ... --smoke` | ✅ 6/6 compile clean | <1s each |
| MoE device runtime | `test_decode_layer_moe_st --variant full_silu_silu -d 0` | ⏸ 507018 within ~5s | (faults) |
| Single-card e2e (Phase 15) | `tools/p15_trace/run_with_trace.py` | ✅ rc=0, 20 tasks complete | 6.69s |

## Selected milestones (compressed from session logs)

### Phase 15 single-card e2e unblock (2026-06-15)

Three layered fixes required to reach rc=0:

1. `attention_full.py:658-690` — `full_head_gate` bypassed as identity
   (`attn_out_gated = attn_out`). Root cause: `pl.row_expand_mul` over
   a `[N, 1]` FP32 operand fails AIV 32-byte row alignment. No clean
   model-side workaround; proper fix is upstream pto-isa cube-matmul
   over a block-diagonal R matrix (TASK-L). Bypassing loses the sigmoid
   gate semantic but unblocks the e2e pipeline.
2. `tools/p15_trace/run_with_trace.py` — `--tp-world-size 1` triggers
   the `step3p5_decode.run_real_npu` monkey-patch path, which reloads
   `attention_full` / `decode_layer` modules with `TP_WORLD_SIZE=1` /
   `EP_WORLD_SIZE=1`. Codegen elides the `tp_all_reduce` kernel.
3. `step3p5_decode.py:391-414` — TP=1 patch sets `LAYER_INTER_ROWS_DYN`
   and `LAYER_QHIDDEN_ROWS_DYN` to TP=8-derived values, working around
   pypto upstream bugs #3/#4 (`pl.dynamic` first-dim slice loses parent
   stride / phantom int32 kernel param).

### Phase 16 multi-card resolved (2026-06-19)

The IPC `support_shmem_map_exbus=0` blocker (filed as simpler#1037)
was a **driver capability gap**, not a code bug. Resolution:

1. Upgrade driver 25.0.rc1.2 → 25.5.2.
2. Upgrade firmware 7.7.0.3.220 → 7.8.0.7.220 (writes to chip flash;
   persists across host reboot).
3. CANN must be `9.0.0-beta.1`, NOT GA (GA's TDT does not push the
   AICPU `libaicpu_extend_kernels.so`, breaking simpler init).
4. simpler `comm_hccl.cpp` add `__has_include`-guarded `*Inner` macro
   alias for CANN GA forward-compat (no-op under beta.1).

Verified end-to-end: `aclrtIpcMemImportByKey + ENABLE_PEER_ACCESS`
cross-card returns rc=0 with `peer_va == parent ptr`; simpler L3
`allreduce_distributed` produces `max|out-expected|=0` double-card
golden match.

### Phase 19 MoE blocker resolution (2026-06-17)

Six independent blockers were identified for the 6 MoE variants:

| Blocker | Description | Resolution |
|--------:|-------------|------------|
| 1 | PTOAS v0.44 `pto.tci ui32 {descending=false}` parser bug | ✅ Upgraded ptoas-bin to v0.45 (commit `caf57c50`, includes upstream fix `505abd64`) |
| 2 | sh_mlp / gate_matmul L1/UB overflow | ✅ Was a shape-choice artifact: `apply_tp1_patch` (full unsliced widths) vs canonical TP=8 per-rank widths (8/12/1/1408/160/36). Per-rank path clean. |
| 3 | dispatch.py 32B-align — `PER_RANK_BUCKETS` not pad-8 | ✅ Added `PER_RANK_BUCKETS = pad8(N_RANKS * N_LOCAL_EXPERTS)` + `N_RANKS_PAD = pad8(N_RANKS)` and mirrored across 5 files |
| 4 | CCEC bf16 type cast not supported in gate_topk / moe_combine | ✅ Switched `expert_weights` from BF16 to FP32 across the 6 emission sites; gate already FP32 internally |
| 5 | MoE device runtime 507018 | ⏸ **STILL OPEN — see [`known-blockers.md`](known-blockers.md)** |
| 6 | MTP wrapper integration | ❌ Deferred (no impact on dense or MoE ST) |

After Blockers 1-4: **6 MoE variants smoke compile PASS at canonical
TP=8 per-rank widths.** All 8 ST (2 dense + 6 MoE) build clean. Only
device runtime for MoE remains.

### 5-repo push to fork (2026-06-20)

Rebased pypto / pypto-lib / pto-isa / PTOAS / simpler all onto
`origin/main`, audited that 4 local simpler patches and 6 pypto-lib
step3p5 commits are still needed, dropped patches that upstream
subsumed (none affecting our work this cycle), pushed the 5 working
branches to `csy0225/<repo>` `stepfun/develop`. simpler submodule
landed at `a6e06406` with the four production-critical patches
(zero-size view + `--no-as-needed` libhcomm + IPC ENABLE_PEER_ACCESS
+ SDMA_OFF + llvm-strip).

## Phase 1 exit checklist

The criteria below define what "pypto kernel prototype done" means in
the context of unblocking Phase 2 (vLLM backend integration).

| Criterion | Required for Phase 2 v0.1 (single-card) | Required for Phase 2 v0.3 (multi-card) | Status |
|-----------|:---------------------------------------:|:--------------------------------------:|--------|
| Single-card dense decode_layer device runs | ✅ MUST | ✅ MUST | ✅ Phase 19 ST PASS |
| Single-card e2e (Phase 15) rc=0 | ✅ MUST | ✅ MUST | ✅ rc=0, 20 tasks complete |
| 48-layer frontend smoke rc=0 | ✅ MUST | ✅ MUST | ✅ all program builders + 8 layer-idx variants |
| Multi-card collective primitive | — | ✅ MUST | ✅ simpler L3 allreduce double-card golden |
| Multi-card decode_layer e2e | — | ✅ MUST | ⏸ barrier all_reduce gate ([`known-blockers.md`](known-blockers.md) §1) |
| MoE 6 variants compile clean | ✅ MUST | ✅ MUST | ✅ smoke 6/6 PASS @ canonical TP=8 widths |
| MoE device runtime green | — | (only Phase 2 v1.0) | ⏸ 507018 gate ([`known-blockers.md`](known-blockers.md) §2) |
| HF safetensors weight loader | ✅ MUST | ✅ MUST | ✅ `weight_loader.py:load_step3p5_weights_for_rank` + 30-key `expected_shapes()` |
| Phase 16 deployment three-pillars | ✅ MUST | ✅ MUST | ✅ verified on 0162; 0234 driver/firmware upgrade pending |
| 5 repos pushed to fork stepfun/develop | ✅ NICE | ✅ MUST | ✅ pypto / pypto-lib / pto-isa / PTOAS / simpler |

**Verdict**: Phase 1 deliverables are **sufficient for Phase 2 v0.1
and v0.2** (single-card flows). v0.3 multi-card and v1.0 full-pypto-MoE
remain gated on barrier all_reduce + MoE 507018 fixes; both fixes can
proceed in parallel with Phase 20/21 implementation work and are not
on the critical path until Phase 22 multi-card section opens.

## Lessons learned (cross-references)

The development surfaced multiple recurring pitfalls now captured in
permanent reference docs:

- [`known-pypto-pitfalls.md`](../known-pypto-pitfalls.md) — pypto /
  pto-isa / simpler hard limits at the kernel / codegen layer (8
  entries): `[N, 1]` intra-UB VEC tile / Vec row 32-byte alignment /
  `pl.dynamic` slice stride / phantom int32 param / AICPU stderr
  restriction / kernel-loop primitives / `pl.range(constant)` UB
  overflow
- [`dev-workflow-gotchas.md`](../dev-workflow-gotchas.md) — non-pypto
  workflow pitfalls (5 entries): stale pyc / triple-source activation
  / HTTP/2 timeout / SSH-on-netboot / gh CLI fallback to curl
- Single-card ST/UT shape iron rule (CLAUDE.md top, memorialised in
  Claude project memory `feedback_single_card_st_shape_iron_rule`)

## Open issues at archive cutoff (2026-06-22)

See [`known-blockers.md`](known-blockers.md) for active item details.
Quick summary:

- **Critical (gate Phase 22 multi-card)**: barrier all_reduce
  UB-friendly rewrite, MoE 507018 device runtime
- **Accuracy**: head_gate ×1 bypass / TASK-L cube-matmul fix
- **Infrastructure**: 0234 driver/firmware upgrade
- **Deferred**: Phase 17 prefill MoE L1 overflow, MTP integration into
  decode_fwd

## Repository state at archive cutoff

| Repo | Branch | Pin | What we hold beyond origin/main |
|------|--------|-----|----------------------------------|
| pypto | `stepfun/develop` | `b00c8b23` | DFX env hooks for `output_prefix` / `dep_gen` / `l2_swimlane` + 16 full_rope debug repros |
| pypto-lib | `stepfun/develop` | `69f22b1` | step3p5 model in tree + Phase 19 padding + ST scaffolds + dev-workflow gotchas + Phase 20-22 design |
| pto-isa | `stepfun/develop` | `e25732f0` | = origin/main (no local patches) |
| PTOAS | `stepfun/develop` | `da011a3d` | = origin/main; binary ptoas-bin at v0.45 |
| simpler (submodule) | — | `a6e06406` | zero-size view + libhcomm `--no-as-needed` + IPC ENABLE_PEER_ACCESS + SDMA_OFF + llvm-strip (4 production patches) |

## References

- vLLM stepcast fork (Phase 2 integration target):
  `<workspace>/pd_sep/update_019/ascend/vllm/vllm/model_executor/models/step3p5.py`
- Live tracker: `<workspace>/pypto/CLAUDE.md`
- In-repo phase docs (Phase 20+): [`phases/`](phases/)
