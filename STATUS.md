# STATUS

Live status board for the pypto step3p5 project. **Update with every
phase / sub-task / blocker state change.** For historical context see
[`archive/`](archive/).

**Last updated**: 2026-06-22

---

## Phase tracker

| Phase | Title | State | Detail |
|------:|-------|-------|--------|
| **1** | **pypto kernel prototype** | ✅ **COMPLETED** | [`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md) |
| **2** | **vLLM Ascend backend integration** | 🟡 **IN PROGRESS** (design landed) | this section |

### Phase 2 sub-phases

| Sub-phase | Scope | State | Doc | Estimate |
|-----------|-------|-------|-----|----------|
| **2.0 (Phase 20)** | vLLM monkey-patch e2e — whole-model patch on `Step3p5Model.forward`; single-card TP=1; mixed-mode MoE | 📐 Design landed; **tasks 1.1-1.9 NOT STARTED** | [`phases/20-vllm-backend-monkey-patch.md`](phases/20-vllm-backend-monkey-patch.md) | 3-4 weeks |
| **2.1 (Phase 21)** | Precision validation harness vs upstream vLLM; L1/L2/L3 three-tier | 📐 Design landed; gated on Phase 20 | [`phases/21-precision-validation.md`](phases/21-precision-validation.md) | 3-4 weeks |
| **2.2 (Phase 22)** | Perf baseline + 2 optimisation rounds; TP=8 multi-card | 📐 Design landed; gated on Phase 21 + 2 hard blockers | [`phases/22-perf-baseline.md`](phases/22-perf-baseline.md) | 6-8 weeks |

**Target total to v1.0 production decode**: ~12-16 weeks from
2026-06-22 (includes parallel work on hard gates).

---

## Phase 2 deliverable tiers (track which sub-version we're at)

| Tier | What works | Required parts of Phase 2 | Required blockers to be cleared |
|------|------------|---------------------------|----------------------------------|
| **v0.1** | Single-card dense path + mixed-mode MoE through vLLM | Phase 20 | none |
| **v0.2** | Single-card 45 layers mixed-mode (dense pypto, MoE vLLM eager) | Phase 20 | none |
| **v0.3** | TP=8 multi-card dense + mixed-mode MoE | Phase 20 + Phase 22.1-3 | barrier all_reduce UB fix |
| **v1.0** | TP=8 / EP=8 full pypto MoE + production decode perf published | Phase 20-22 complete | barrier all_reduce + MoE 507018 |

**Current**: ahead of v0.1 (Phase 1 PASS). v0.1 entry is **GATE-FREE**
— Phase 20 implementation can start immediately.

---

## Immediate next actions (in priority order)

1. **Phase 20.1**: `config_align.py` — assert vLLM `hf_config` matches
   pypto `config.py` constants. 1 day, no dependencies. Cheapest entry
   point.
2. **Phase 20.2**: `weight_translate.py` — vLLM `nn.Module` → pypto
   bundle dict. 5 days. Core engineering for Phase 20.
3. **In parallel — blocker gate 1**: write UB-friendly barrier
   all_reduce rewrite (`acc` carry → in-place store/reload via
   `local`). Targets Phase 22 multi-card entry. See
   [`blockers.md`](blockers.md) §1.
4. **In parallel — blocker gate 2**: write `P19_DISPATCH_LIMIT`
   dispatch-cut bisect tool for MoE 507018. Targets Phase 22 v1.0
   entry. See [`blockers.md`](blockers.md) §2.

---

## Pin snapshot (most recent)

| Date | Event | pypto | pypto-lib | pto-isa | PTOAS (src) | simpler (submodule) | ptoas-bin |
|------|-------|-------|-----------|---------|-------------|----------------------|-----------|
| 2026-06-22 | Phase 2 design landed; project tracker repo created | `stepfun/develop:b00c8b23` | `stepfun/develop:69f22b1` (pre-revert of misplaced docs) | `stepfun/develop:e25732f0` | `stepfun/develop:da011a3d` | `a6e06406` | `v0.45` |

Historical pin snapshots: [`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md).

---

## Hard blockers (gating Phase 22)

| # | Blocker | Severity | Gates | Owner | Detail |
|--:|---------|----------|-------|-------|--------|
| 1 | barrier `tp_all_reduce` UB overflow (`pl.range(constant)` unroll, 624KB > 184KB UB limit) | 🔴 Critical | Phase 22.3 multi-card dense, v0.3+ | unassigned | [`blockers.md`](blockers.md) §1 |
| 2 | MoE device runtime 507018 (kernel-internal AICPU/AICore fault, no host log) | 🔴 Critical | Phase 22 v1.0 full pypto MoE | unassigned | [`blockers.md`](blockers.md) §2 |
| 3 | head_gate × 1 bypass — semantic loss vs upstream (sigmoid gate replaced with identity) | 🟡 Accuracy | Phase 21 L1 layer-level parity | TASK-L (upstream pto-isa) | [`blockers.md`](blockers.md) §3 |
| 4 | Prefill MoE L1 overflow (TASK-29) | 🟢 Deferred | Phase 17 prefill e2e (out of scope for Phase 22 decode-only) | unassigned | [`blockers.md`](blockers.md) §4 |
| 5 | 0234 driver+firmware upgrade pending | 🟢 Infrastructure | secondary deploy host | unassigned | [`blockers.md`](blockers.md) §5 |

---

## Verified working state on `gpu-a910x-0162` (Phase 16 host)

| Component | Verified | Notes |
|-----------|----------|-------|
| driver 25.5.2 | ✅ 2026-06-22 | `npu-smi info -t board -i 0` reports |
| firmware 7.8.0.7.220 | ✅ (chip flash) | persists across host reboot |
| CANN 9.0.0-beta.1 | ✅ symlink at `/usr/local/Ascend/cann-9.0.0-beta.1` → NVMe | NOT GA — see [`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md) |
| simpler L3 allreduce_distributed -d 0-1 | ✅ 2026-06-22 | `max\|out-expected\|=0` double-card golden match |
| pypto-lib frontend smoke rc=0 | ✅ 2026-06-22 | 4 program builders + 8 layer-idx variants |
| Phase 19 ST-1 full dense @ device 0 | ✅ 7.93s (ratio_allclose PASS) | per-rank TP=8 slice widths preserved |
| Phase 19 ST-2 swa dense @ device 0 | ✅ 14.85s (ratio_allclose PASS) | same |
| Phase 19 MoE 6 variants smoke compile | ✅ 6/6 PASS | TP=8 per-rank slice path |
| Phase 19 MoE device runtime | ⏸ 507018 fault within ~5s | blocker §2 |
| Phase 15 single-card e2e | ✅ rc=0, 20 tasks complete | head_gate ×1 bypass + TP=1 patch path |

---

## Verified working state on `gpu-a910x-0234`

Not yet upgraded. Driver `25.5.1` / firmware `7.8.0.6.201` / CANN
`9.0.0-beta.1`. Multi-card e2e blocked by driver shmem-exbus gap until
driver+firmware upgrade. `.run` packages staged on 0162
`/mnt/persist/ascend-staging/` — see
[`deployment/machine-recovery.md`](deployment/machine-recovery.md) for
upgrade runbook.
