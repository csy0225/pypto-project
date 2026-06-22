# PyPTO Step3p5 Project

End-to-end serving of the **step3p5** large language model on Ascend
NPUs, using the **pypto** programming framework for the decoder kernel
and **vLLM** for serving / scheduling / batching.

This repository is the **project-level tracker** that spans 5 code
repositories. The actual code lives elsewhere — this repo holds status,
phase tracking, blockers, deployment specs, and architectural notes.

## Repos in this project (where the actual code lives)

| Repo | Role | Origin | Our fork |
|------|------|--------|----------|
| `pypto` | Programming framework — multi-level IR + codegen | `hw-native-sys/pypto` | `csy0225/pypto` |
| `pypto-lib` | Tensor-level kernels + step3p5 model | `hw-native-sys/pypto-lib` | `csy0225/pypto-lib` |
| `pto-isa` | Tile-ISA virtual implementations | `hw-native-sys/pto-isa` | `csy0225/pto-isa` |
| `PTOAS` | LLVM/MLIR PTO bytecode assembler | `hw-native-sys/PTOAS` | `csy0225/PTOAS` |
| `simpler` | PTO runtime (AICPU+AICore dispatcher) | `hw-native-sys/simpler` (submodule of pypto) | `csy0225/simpler` |
| **(integration target)** vLLM stepcast fork | Serving / scheduling / sampler / tokenizer | internal stepcast fork | not forked |

All our forks live on the `stepfun/develop` branch. Pin snapshot lives
in [`STATUS.md`](STATUS.md).

## Project status at a glance

**Phase 1 — pypto kernel prototype**: ✅ **COMPLETED** (2026-06-22).
See [`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md).

**Phase 2 — vLLM Ascend backend integration**: 🟡 **IN PROGRESS**
(design landed, implementation NOT STARTED).
- Phase 20: vLLM monkey-patch e2e flow → [`phases/20-vllm-backend-monkey-patch.md`](phases/20-vllm-backend-monkey-patch.md)
- Phase 21: precision validation vs upstream vLLM → [`phases/21-precision-validation.md`](phases/21-precision-validation.md)
- Phase 22: perf baseline + tuning → [`phases/22-perf-baseline.md`](phases/22-perf-baseline.md)

**Active blockers** (carried forward): see [`blockers.md`](blockers.md).

**Production deployment**: see [`deployment/`](deployment/). The
Phase 16 three-pillars binding is a hard requirement for multi-card
deploys.

## Where to look (by question)

| Question | Path |
|----------|------|
| What is the current state of work? | [`STATUS.md`](STATUS.md) |
| What's the active phase I should pick up tasks from? | [`phases/`](phases/) |
| What's blocked / what should I help unblock? | [`blockers.md`](blockers.md) |
| How do I deploy this on a new machine? | [`deployment/`](deployment/) |
| How does it fit together architecturally? | [`architecture/`](architecture/) |
| What was the prototype journey (Phase 01-19)? | [`archive/prototype-phase-01-19-summary.md`](archive/prototype-phase-01-19-summary.md) |
| What pitfalls should I know writing pypto kernels? | `pypto-lib/docs/known-pypto-pitfalls.md` (in pypto-lib repo) |
| What pitfalls should I know in dev workflow (pyc, env, git)? | `pypto-lib/docs/dev-workflow-gotchas.md` (in pypto-lib repo) |

## Quick start (on a verified Phase 16 host, e.g. `gpu-a910x-0162`)

```bash
# 1. Source the three-pillars env (CANN beta.1, NOT GA)
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa

# 2. Verify pypto frontend
cd <workspace>/pypto-lib
python -m models.step3p5._smoke_program_build
# expected: === probe rc=0 ===

# 3. Verify multi-card collective baseline
cd <workspace>/pypto/runtime
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1
# expected: max |out - expected| = 0.000e+00

# 4. Verify single-card dense decode_layer ST
cd <workspace>/pypto-lib
python -m tests.step3p5.test_decode_layer_full_dense_st -p a2a3 -d 0
# expected: ratio_allclose PASS, ~8s
```

If any of these fails, consult [`blockers.md`](blockers.md) and the
pypto-lib reference docs.

## Update protocol

When a phase / sub-task / blocker state changes:

| Trigger | Update what |
|---------|-------------|
| sub-task complete | Status section of `phases/NN-*.md` |
| Phase entry / exit | `STATUS.md` current phase + `phases/README.md` |
| New blocker discovered | `blockers.md` |
| Blocker resolved | Remove from `blockers.md`, optionally add to `archive/` |
| Session-end summary | Append to `archive/milestones-2026-Q2.md` |
| Component pin moves | `STATUS.md` "Pin snapshot" |

`CLAUDE.md` in this repo is for Claude session bootstrap only — it
should stay short (~50 lines). Do **not** put status / history in it.

## Repo layout

```
pypto-project/
├── README.md                            # this file
├── CLAUDE.md                            # Claude session bootstrap (slim)
├── STATUS.md                            # live status board
├── blockers.md                          # active open issues (SSOT)
├── deployment/                          # production deployment specs
│   ├── README.md
│   ├── phase16-three-pillars.md         # driver + firmware + CANN binding
│   ├── machine-recovery.md              # 0162/0234 runbook
│   └── version-matrix.md                # 5-repo version compatibility
├── phases/                              # active phase tracking
│   ├── README.md
│   ├── 20-vllm-backend-monkey-patch.md
│   ├── 21-precision-validation.md
│   └── 22-perf-baseline.md
├── archive/                             # historical record
│   ├── README.md
│   ├── prototype-phase-01-19-summary.md
│   └── milestones-2026-Q2.md
└── architecture/                        # cross-repo design notes
    ├── README.md
    ├── overview.md
    └── vllm-step3p5-mapping.md
```
