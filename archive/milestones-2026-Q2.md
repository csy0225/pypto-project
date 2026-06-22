# Milestones — 2026 Q2

Session-by-session milestone log, append-only, descending by date.
For the high-level Phase 01-19 summary see
[`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md).

## 2026-06-22 (evening) — Project tracker repo created ✅

Created `pypto-project` as a dedicated tracker repo at
`<dev-host>/data/chensiyu/hw_project/pypto/pypto-project/`. Pushes to
`csy0225/pypto-project` (private fork-style). Migration of scattered
docs:

- Moved Phase 20/21/22 docs + archive content out of
  `pypto-lib/docs/step3p5/` (wrong home — these are cross-repo
  concerns) into `pypto-project/phases/` + `archive/`.
- Wrote new top-level entry docs: README.md, STATUS.md, CLAUDE.md
  (slim), blockers.md.
- Outer tracker `<workspace>/pypto/CLAUDE.md` (the 594-line monolith)
  retired — superseded by this repo.

**Resolves**: scattered-docs concern raised by project owner. Single
source of truth for project status now lives in this repo.

## 2026-06-22 (afternoon) — WIP push split + dev-workflow docs + Phase 20-22 design ✅

### WIP push split

3 commits to fork csy0225:

- `csy0225/pypto-lib stepfun/develop`: `ffaf5d6 → 73dbd12`
  (tests/step3p5/ 12 ST/UT scaffolds + Chinese architecture guide, +3381 lines)
- `csy0225/pypto-lib wip/step3p5-barrier-allreduce-20260622`: NEW
  `b5bb6ee` (4 files -267/+181: barrier-style all_reduce + per_rank input broadcast)
- `csy0225/pypto stepfun/develop`: `03136bf6 → b00c8b23`
  (10 full_rope SSA/scheduling debug repros, +2199 lines)

**Key decision**: WIP barrier all_reduce did NOT go onto
`stepfun/develop` (would regress dense ST device 0 compile due to UB
overflow). Side branch preserves intent for follow-up.

### Dev workflow + pitfalls docs (push: `73dbd12 → a6b5faa`)

- New `pypto-lib/docs/known-pypto-pitfalls.md` §7:
  `pl.range(constant)` unrolls without SSA buffer reuse → UB overflow.
- New `pypto-lib/docs/dev-workflow-gotchas.md`: 5 entries cataloguing
  non-pypto workflow time-sinks (stale pyc / triple-source activation /
  HTTP/2 timeout / SSH on netboot / gh CLI absence).

### Phase 20-22 design landed (push: `a6b5faa → 69f22b1`)

Three phase docs written, each ~200-300 lines. These docs were later
moved to this `pypto-project` repo (see evening entry above).

## 2026-06-22 (morning) — 0162 reboot recovery + reverify + MoE 507018 reproduce ⏸

### Environment recovery post-reboot

`gpu-a910x-0162` was reboot'd; all three pillars survived (driver
25.5.2, firmware 7.8.0.7.220 chip flash, CANN 9.0.0-beta.1 via NVMe
symlink). All 4 git repos at expected HEADs, simpler submodule at
`a6e06406`.

### Smoke probe red herring (resolved)

First `python -m models.step3p5._smoke_program_build` returned rc=1
with `valid_cols (48) exceeds bound 16` at attention_swa.py:396.
**Root cause**: stale `__pycache__/config.cpython-311.pyc` from prior
session's `apply_perrank_patch(TP=2)`. Python pyc invalidation only
compares source mtime, not module dict values.

**Resolution**: `find models/step3p5 -name "*.py" -exec touch {} +`.
Memorialised as workflow gotcha §1.

### Verification baseline

- simpler L3 allreduce_distributed -d 0-1: `max|out-expected|=0` ✅
- Phase 19 ST-1 full dense: PASS 7.93s ✅
- Phase 19 ST-2 swa dense: PASS 14.85s ✅
- MoE 6 variants smoke: 6/6 PASS ✅
- MoE device runtime (full_silu_silu -d 0): 507018 fault within ~5s ⏸

Documented as blocker §2; needs `P19_DISPATCH_LIMIT` dispatch-cut tool.

## 2026-06-20 — 5-repo rebase to origin/main + fork push ✅

Rebased pypto / pypto-lib / pto-isa / PTOAS / simpler all onto
`origin/main`. Audited that 4 simpler local patches + 6 pypto-lib
step3p5 commits + 3 pypto commits are still needed (upstream did not
subsume any). Pushed to `csy0225/`:

- pypto: `926941e0 → 03136bf6`
- pypto-lib: `93826904 → ffaf5d69`
- pto-isa: `109c9f72 → e25732f0`
- simpler: `c66b4120 → a6e06406`

Verified on 0162: smoke probe rc=0, simpler L3 allreduce double-card
golden, ST-1 dense device PASS, MoE 6/6 smoke PASS.

**Rebuild trap**: `pip install -e .` first failed due to
`tensor.h:535 buffer_elems` `-Werror=unused-variable` (NDEBUG +
release flag). Fix: don't pass `CMAKE_BUILD_TYPE` (use dev default).

## 2026-06-19 — Phase 16 multi-card IPC blocker RESOLVED ✅

The `support_shmem_map_exbus=0` cap (simpler#1037) was a driver
capability gap. Resolution = three pillars together:

1. Driver `25.0.rc1.2 → 25.5.2`
2. Firmware `7.7.0.3.220 → 7.8.0.7.220` (chip flash, persistent)
3. CANN `9.0.0-beta.1` (NOT GA — GA's TDT fails to push AICPU
   `libaicpu_extend_kernels.so`, breaking simpler init with 507018)

Plus simpler `comm_hccl.cpp` patch (CANN GA forward-compat alias).

**Traps**:

- CANN GA vs beta.1: 3+ hours wasted on GA path before identifying.
- 0162 is netboot/tmpfs: `/usr/local/Ascend/`, `/etc/`, `~/.ssh/`
  vanish on reboot. Built `RECOVERY.sh` for idempotent restore;
  persistent state on NVMe at `/mnt/persist/`.
- Kubernetes DaemonSet (`device-plugin`, `npu-exporter`) blocks
  driver `.run --upgrade`. Must `systemctl stop kubelet` + manual kill.

**Validation**: `aclrtIpcMemImportByKey + ENABLE_PEER_ACCESS` cross-card
returns rc=0 with `peer_va == parent ptr`; simpler L3
`allreduce_distributed` produces `max|out-expected|=0` double-card
golden match.

**0234 path**: only needs driver+firmware upgrade (CANN already
correct). `.run` packages staged at 0162 `/mnt/persist/ascend-staging/`.
Tracked as blocker §5.

## 2026-06-17 — Phase 19 MoE blockers 1-4 cleared + dense ST device PASS ✅

Detail in [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
"Phase 19 MoE blocker resolution". MoE device runtime 507018 remains
(blocker §2). Dense ST device 0 passed (full 7.93s, swa 14.85s).

## 2026-06-15 — Phase 15 single-card e2e rc=0 ✅

Single-rank decode_layer end-to-end runs to completion on device 0,
20 dispatched tasks complete. Three layered fixes combined: head_gate
×1 bypass + `--tp-world-size 1` monkey-patch + `LAYER_*_ROWS_DYN`
overrides. Validated `next_hidden_out shape=[1, 16, 4096],
max|value|=0` (zero-weight expected zero output). Run time 6.69s.

---

## Pin snapshot history (descending)

| Date | Event | pypto | pypto-lib | pto-isa | PTOAS (src) | simpler | ptoas-bin |
|------|-------|-------|-----------|---------|-------------|---------|-----------|
| 2026-06-22 eve | pypto-project repo created | `develop:b00c8b23` | `develop:69f22b1` | `develop:e25732f0` | `develop:da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-22 aft | Phase 20-22 design + dev-workflow docs | `develop:b00c8b23` | `develop:69f22b1` | `develop:e25732f0` | `develop:da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-20 | 5-repo rebase + fork push | `develop:03136bf6` | `develop:ffaf5d6` | `develop:e25732f0` | `develop:da011a3d` | `a6e06406` | `v0.45` |
| 2026-06-19 | Phase 16 three-pillars validated | `main:a1b066df` | `main:9c5593fb` | `main:109c9f72` | `main:29a8af28` | `afb5c5a9` | `v0.44` |
| 2026-06-17 | Phase 19 blockers 1-4 cleared | `main:3f421313` | `main:08f71692` | `main:8e436661` | `main:a1efed75` | `6e84154d` | `v0.43` |
| 2026-06-15 | Phase 15 single-card e2e rc=0 | `main:3f421313` | `main:af4b2ed5` | `main:12e766d1` | `main:5392d5da` | `6e84154d` | `v0.43` |
| 2026-06-05 | Phase 13 re-sync + smoke green | `main:3f421313` | `main:08f71692` | `main:8e436661` | `main:a1efed75` | `6e84154d` | `v0.43` |

---

## Resolved blockers (post-mortems)

### 2026-06-22 — simpler#1018 libhcomm DT_NEEDED ✅

`comm_init` segfault — `hccl_comm.h` declares HCCL weak, x86 default
`--as-needed` drops `libhcomm.so` from `DT_NEEDED`. Fix in
simpler `a6e06406`: wrap `${HCCL_LINK_TARGETS}` with
`-Wl,--no-as-needed ... -Wl,--as-needed` in
`src/{a2a3,a5}/platform/onboard/host/CMakeLists.txt`.

### 2026-06-19 — simpler#1037 IPC support_shmem_map_exbus=0 ✅

Three-pillars fix (driver 25.5.2 + firmware 7.8.0.7.220 + CANN
beta.1). See 2026-06-19 milestone above.

### 2026-06-17 — Phase 19 blockers 1-4 ✅

1. PTOAS v0.44 `pto.tci ui32 {descending=false}` parser: upstream
   v0.45 fix `505abd64`.
2. sh_mlp / gate_matmul L1/UB overflow: was a shape-choice artifact
   (`apply_tp1_patch` wrong, `apply_perrank_patch` correct).
3. dispatch.py 32B alignment: `PER_RANK_BUCKETS = pad8(...)` mirrored
   across 5 files.
4. CCEC bf16 type cast: `expert_weights` BF16 → FP32 across 6 emission
   sites.

Detail: [`prototype-phase-01-19-summary.md`](prototype-phase-01-19-summary.md)
"Phase 19 MoE blocker resolution".
