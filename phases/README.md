# Phases

Active phase tracking for the pypto step3p5 project. Each phase doc
describes a coherent unit of work with goal / scope / decisions /
tasks / exit criteria / risks / status.

Phase 01-19 (pypto kernel prototype, completed June 2026) summary is
in [`../archive/prototype-phase-01-19-summary.md`](../archive/prototype-phase-01-19-summary.md).
Phase 20+ documents the vLLM Ascend backend integration effort and is
tracked actively here.

## Index

| Phase | Title | Status | Doc |
|------:|-------|--------|-----|
| **20** | vLLM backend monkey-patch — e2e flow | 📐 Design landed 2026-06-22; tasks 1.1-1.9 NOT STARTED | [20-vllm-backend-monkey-patch.md](20-vllm-backend-monkey-patch.md) |
| **21** | Precision validation harness (vs upstream vLLM) | 📐 Design landed 2026-06-22; gated on Phase 20 | [21-precision-validation.md](21-precision-validation.md) |
| **22** | Perf baseline + tuning | 📐 Design landed 2026-06-22; gated on Phase 21 + 2 hard blockers | [22-perf-baseline.md](22-perf-baseline.md) |

For real-time state of each phase see [`../STATUS.md`](../STATUS.md).

## Update protocol per phase

When a phase node changes state:

1. Update the phase doc's `## Status` section.
2. Update the row in the index table above.
3. Update [`../STATUS.md`](../STATUS.md) "Phase 2 sub-phases" table.
4. If a new blocker emerges, add to [`../blockers.md`](../blockers.md).
5. Record the new pin snapshot at the top of the phase doc.

## Cross-references

- [`../STATUS.md`](../STATUS.md) — live status board
- [`../blockers.md`](../blockers.md) — active open issues
- [`../architecture/overview.md`](../architecture/overview.md) — system
  architecture across 5 repos + vLLM
- `pypto-lib/docs/known-pypto-pitfalls.md` — kernel / codegen hard
  limits (in pypto-lib repo)
- `pypto-lib/docs/dev-workflow-gotchas.md` — workflow pitfalls (in
  pypto-lib repo)
