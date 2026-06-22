# Claude Session Bootstrap

This file loads automatically when a Claude session opens this repo.
**Keep it short** — anything that's not iron-rule or routing-pointer
belongs in [`STATUS.md`](STATUS.md), [`blockers.md`](blockers.md), or a
phase doc.

## What this repo is

Project-level tracker for the pypto step3p5 effort. See
[`README.md`](README.md) for the full description. Five code repos live
elsewhere (`pypto`, `pypto-lib`, `pto-isa`, `PTOAS`, `simpler`); this
repo only tracks them.

## Where to look first

1. **Current state**: [`STATUS.md`](STATUS.md)
2. **What's blocked**: [`blockers.md`](blockers.md)
3. **Active phase tasks**: [`phases/`](phases/)
4. **Production deploy spec**: [`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md)

## Iron rules (apply to every session)

These are recurring mistakes from prior sessions. Re-violating them
costs hours.

### 1. Single-card ST/UT must keep TP=8 per-rank slice widths

When writing or running kernel-level ST/UT, use `apply_perrank_patch()`
not `apply_tp1_patch()`. The per-rank helper preserves canonical TP=8
slice widths (8/12/1/1408/160/36 etc.) while flipping
`TP_WORLD_SIZE`/`EP_WORLD_SIZE` to 1 so codegen elides collectives.
The unslice helper (full widths) only suits Phase 15 e2e and overflows
kernels whose chunk constants follow the slice (`sh_mlp`,
`gate_matmul`).

Detail: `pypto-lib/tests/step3p5/_perrank_setup.py` docstring;
`pypto-lib/docs/known-pypto-pitfalls.md` references this throughout.

### 2. Phase 16 three-pillars version binding

Any production multi-card deploy needs **all three** of:

| Component | Required | Failure if older |
|-----------|----------|------------------|
| Driver | 25.5.2 | `support_shmem_map_exbus=0`, IPC fails 507899 |
| Firmware | 7.8.0.7.220 (chip flash, persistent) | same cap gap |
| CANN | 9.0.0-beta.1 (NOT GA) | simpler init fails 507018 |

Full spec + failure analysis: [`deployment/phase16-three-pillars.md`](deployment/phase16-three-pillars.md).

### 3. Stale .pyc after monkey-patching module globals

Any test that runs `apply_perrank_patch` / `apply_tp1_patch` /
`cfg.X = Y` serializes the patched values into `__pycache__/*.pyc`.
The next fresh `python -m ...` reads those back. Before re-running:

```bash
find <pypto-lib>/models/step3p5 -name "*.py" -exec touch {} +
```

Detail: `pypto-lib/docs/dev-workflow-gotchas.md` §1.

### 4. Triple-source env activation

`activate.sh` only sets up the venv. Every fresh shell on the deploy
host needs three sources:

```bash
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa
```

Detail: `pypto-lib/docs/dev-workflow-gotchas.md` §2.

### 5. git push needs HTTP/1.1 on the deploy host network

```bash
git -c http.version=HTTP/1.1 push ...
```

Default HTTP/2 silently times out at 130s. Detail:
`pypto-lib/docs/dev-workflow-gotchas.md` §3.

## What NOT to put in this file

- Session-by-session milestones → `archive/milestones-2026-Q2.md`
- Phase task lists → `phases/NN-*.md`
- Open issues → `blockers.md`
- Pin snapshot history → `archive/milestones-2026-Q2.md`
- Deployment runbooks → `deployment/`

If you're tempted to write more than 50 lines here, write it elsewhere
and link to it.
