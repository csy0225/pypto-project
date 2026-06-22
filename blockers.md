# Active Blockers

Single source of truth for open issues that gate project progress.
Each entry: **symptom / root cause / current status / unblock criteria
/ links**.

When a blocker is resolved, **delete its section here** and append a
short post-mortem entry to [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)
under "Resolved blockers".

**Last reviewed**: 2026-06-22.

---

## 1. Barrier `tp_all_reduce` UB overflow

**Severity**: 🔴 Critical — gates Phase 22.3 (multi-card dense) and
all v0.3+ tiers.

**Symptom**: pypto compile fails at `AllocateMemoryAddr` pass:

```
Verification failed after 'AllocateMemoryAddr':
  Function 'tp_all_reduce': Vec buffer usage (655360 bytes)
  exceeds platform limit (188416 bytes)
  Location: <pypto-lib>/models/step3p5/decode_layer.py:487
```

**Root cause**: pypto compiler unrolls `for peer in pl.range(group_size=8)`
because `group_size` is a Python int from factory closure, then treats
each iteration's `recv` / `recv_fp32` / loop-carried `acc` as distinct
SSA values without UB reuse. UB cost = `7 × ~80 KB ≈ 560 KB`, blows
the 184 KB Vec UB limit on A2A3.

Categorical doc: `pypto-lib/docs/known-pypto-pitfalls.md` §7.

**Where the WIP sits**: `csy0225/pypto-lib` branch
`wip/step3p5-barrier-allreduce-20260622` HEAD `b5bb6ee`. The branch
contains the *intent* (replace ring all_reduce with barrier-style mirror
of `pypto/tests/st/distributed/test_l3_allreduce.py`) but trips the UB
overflow on dense ST device 0 compile.

**Unblock criteria** (any of):

A. Rewrite `acc` carry as in-place store/reload through `local`
   (per-iter UB ≈ 144 KB, fits 184 KB). Recipe in
   `pypto-lib/docs/known-pypto-pitfalls.md` §7 "avoidance recipe B".
B. Make peer loop bound runtime-dynamic via `pld.nranks(ctx)` (mirror
   the canonical test). Recipe (A) in same doc.
C. Combine A + B for safety.

**Estimate**: ~3-5 days of pypto-lib work + dense ST device 0
regression check. No upstream dependency.

**Owner**: unassigned.

---

## 2. MoE device runtime 507018

**Severity**: 🔴 Critical — gates Phase 22 v1.0 (full pypto MoE).
Phase 2 v0.1-v0.3 (mixed-mode MoE) does **not** depend on this.

**Symptom**: All 6 MoE variants compile clean (smoke 6/6 PASS at
canonical TP=8 per-rank widths) but device runtime fails within ~5s:

```
[ERROR] sync_run_streams: aclrtSynchronizeStreamWithTimeout (AICPU) failed: 507018
[ERROR] orch_error_code=2 sched_error_code=0 runtime_status=-2
RuntimeError: run_prepared failed with code 507018
```

The host plog (`~/ascend/log/run/plog/plog-*.log`) only shows clean
init then unrecoverable stream sync timeout. Device-side log
(`device-*_*.log`) only shows init phase. **No task_id / kernel_name /
fault address surfaces in host logs** (unlike Phase 15 dense which
exposed `tslot:6` + `errcode 0x800`).

**Root cause hypothesis**: MoE-specific path (gate_topk → dispatch
EP-a2a → routed expert MLP → combine EP-a2a → shared expert) — one
of these tasks triggers AICore/AICPU fault. CLAUDE.md memory previously
classified this "same family as simpler#1023 zero-shape view" but
that's wrong — dense ST passes, so simpler#1023 is fixed. Real cause
is in MoE-specific kernels.

**Reproducer**: `gpu-a910x-0162`, 2026-06-22:

```bash
cd <pypto-lib>
python -m tests.step3p5.test_decode_layer_moe_st \
    --variant full_silu_silu -p a2a3 -d 0
# Faults at runtime within ~5s.
```

**Unblock criteria**: dispatch-cut bisect tool to localise. Two paths:

A. **Add `P19_DISPATCH_LIMIT` env hook** mirroring Phase 15
   `P15_DISPATCH_LIMIT`. Allows running host_orch with first N tasks
   only. Binary search to find which task triggers the fault. Then
   inspect that task's IR / generated kernel / runtime trace to
   localise further.

B. **Enable DFX swimlane + dep-graph dump** during MoE run via
   `PYPTO_DISTRIBUTED_DEP_GEN=1` + `PYPTO_DISTRIBUTED_L2_SWIMLANE=1`
   (env hooks landed in pypto `03136bf6`). Look for the last
   completed task before the fault.

**Estimate**: 1-2 weeks (deep upstream-touching debug; may need a
simpler upstream issue filed if root cause is in runtime).

**Owner**: unassigned.

---

## 3. head_gate × 1 bypass — accuracy parity with upstream vLLM

**Severity**: 🟡 Accuracy — gates Phase 21 L1 (per-layer hidden_states)
exact parity. Does not gate v0.1 / v0.2 functional bring-up; gates the
"precision validation green" exit criterion.

**Symptom**: `attention_full.py:658-690` and `attention_swa.py` mirror
have `attn_out_gated = attn_out` (× 1 identity) instead of
`attn_out_gated = attn_out * sigmoid(head_gate_logits)`. Output of
each attention layer is roughly 2× the upstream-expected magnitude
(the average `sigmoid` output is ~0.5).

**Root cause**: pypto kernel cannot express the head_gate operation
without `pl.row_expand_mul([N, K], [N, 1])` over a 1-column FP32
operand, which fails AIV 32-byte row alignment. This is a hard
pto-isa limit — no model-side workaround.

Categorical doc: `pypto-lib/docs/known-pypto-pitfalls.md` §1.

**Tracking**: TASK-L (pto-isa upstream — cube-matmul over a
block-diagonal R matrix). Filed on backlog.

**Unblock criteria** (any of, in order of preference):

A. Upstream pto-isa lands the `[N, 1]` slice 32-byte alignment static
   reject (mentioned in §1 doc) AND we use cube-matmul × block-diag R
   construction in attention_full / attention_swa to express head_gate
   without intra-UB `[N, 1]` Vec tile.
B. Phase 21 §2.7 calibration — patch upstream vLLM
   `Step3p5Attention` to also × 1 (drop the gate semantically). Loses
   ~2× attention scaling vs production but allows L1 ratio_allclose
   to pass between two implementations of the same (degraded) model.
C. Widen Phase 21 L1 tolerance to absorb the ~50% magnitude difference
   on attention-output-only paths. Less rigorous; documents the gap.

**Estimate**:
- Path A: weeks (upstream-gated)
- Path B: 1-2 days (vLLM-side patch + rerun)
- Path C: 0.5 day (tolerance config change + re-baseline)

**Owner**: TASK-L upstream; project-side decision pending.

---

## 4. Prefill MoE L1 overflow (TASK-29)

**Severity**: 🟢 Deferred — gates Phase 17 (full prompt processing
e2e), which is **out of scope for Phase 22 decode-only perf**.

**Symptom**: `models/step3p5/prefill_moe.py` compile fails with L1
buffer overflow (~5 MB > limit) on the `moe_gate_up` MLP. Prefill MoE
layers can't compile.

**Root cause**: Prefill operates on much wider sequence dimension than
decode (e.g. SEQ=4096 vs BATCH=16), so the same MoE kernel structure
that fits decode UB blows L1 in prefill.

**Tracking**: TASK-29 in backlog.

**Unblock criteria**: redesign prefill_moe with multi-step gate_up
chunking. ~1-2 weeks of dedicated work.

**Workaround in Phase 22 decode-only perf**: pre-populate KV cache
with synthetic data sized to target input length, skip prefill,
measure decode-only TPS / ITL. Documented in
[`phases/22-perf-baseline.md`](phases/22-perf-baseline.md) "Prefill
workaround".

**Owner**: unassigned.

---

## 5. Machine 0234 driver+firmware upgrade

**Severity**: 🟢 Infrastructure — secondary deploy host. Does not
block Phase 2 work on `0162`.

**Symptom**: 0234 has driver `25.5.1` / firmware `7.8.0.6.201` /
CANN `9.0.0-beta.1`. The `support_shmem_map_exbus=0` cap is still set
because driver+firmware are below Phase 16 minimum (`25.5.2` /
`7.8.0.7.220`). Cross-card `aclrtIpcMemImportByKey` returns 507899.
Multi-card e2e on 0234 is not possible.

**Root cause**: standard Phase 16 deployment requirement not yet
applied to 0234.

**Unblock criteria**: run upgrade per
[`deployment/machine-recovery.md`](deployment/machine-recovery.md). The
two `.run` packages are staged on 0162 at
`/mnt/persist/ascend-staging/`:

```
Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run
Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run
```

scp to 0234, stop kubelet, run driver `--upgrade --quiet`, reboot, done.

CANN is already correct on 0234 — **must not** run cluster automation
that reverts to GA (back up the beta.1 install before any cluster
script touches `/usr/local/Ascend/`).

**Estimate**: ~2 hours wallclock including reboot.

**Owner**: unassigned.

---

## 6. (Deferred) MTP integration into decode_fwd

**Severity**: 🟢 Deferred — speculative-decoding throughput multiplier.
Not on Phase 2 critical path.

**Symptom**: 3 MTP layers exist as kernels (`models/step3p5/mtp.py`)
but are not wired into `decode_fwd`. vLLM's MTP path expects
1-main-token + N-speculative-tokens + verification accept/reject,
giving ~3× throughput when accept rate is high.

**Root cause**: never built; deferred during Phase 1 to focus on the
critical 45-layer dense+MoE path.

**Unblock criteria**: Phase 23 design (TBD) — wire MTP into
`decode_fwd`'s output stage; integrate with vLLM's speculative decoding
pipeline.

**Estimate**: 2-4 weeks once Phase 22 baseline is established.

**Owner**: unassigned, deferred.

---

## How to add a new blocker

1. Insert a new section above the most-deferred item; pick the right
   severity icon.
2. Number sections sequentially (don't reuse old numbers).
3. Link from the new section to wherever the symptom was first seen
   (a phase doc, a session log in `archive/milestones-2026-Q2.md`,
   etc.).
4. Add a row to [`STATUS.md`](STATUS.md) "Hard blockers" table.
5. If it gates a specific phase, link from that phase doc's "Risks"
   section.
