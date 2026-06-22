# Version Matrix

Compatibility matrix for the 5 code repos + 3 toolchain pillars. A row
in the "Validated combinations" table is a state-set known to work
end-to-end. Mixing across rows is **not** supported without
re-verification.

## Validated combinations

### Production target (2026-06-22)

| Slot | Pin | Notes |
|------|-----|-------|
| Driver | `25.5.2` | Phase 16 minimum |
| Firmware | `7.8.0.7.220` | chip flash, persistent |
| CANN | `9.0.0-beta.1` | NOT GA |
| pypto | `csy0225/pypto stepfun/develop:b00c8b23` | + 3 commits beyond origin/main (DFX env hook + repros + simpler submodule pin) |
| pypto-lib | `csy0225/pypto-lib stepfun/develop:69f22b1` | + 9 commits beyond origin/main (step3p5 model + Phase 19 padding + ST scaffolds + dev-workflow docs + Phase 20-22 design [pre-revert]) |
| pto-isa | `csy0225/pto-isa stepfun/develop:e25732f0` | = origin/main (no local patches) |
| PTOAS | `csy0225/PTOAS stepfun/develop:da011a3d` | = origin/main; binary `ptoas-bin` at `v0.45` |
| simpler | `csy0225/simpler a6e06406` (pypto submodule) | + 4 patches beyond origin/main (zero-size view + `--no-as-needed` libhcomm + IPC ENABLE_PEER_ACCESS + SDMA_OFF + llvm-strip) |
| ptoas-bin | `v0.45` | binary release |
| Python | `3.11.14` | venv at `<workspace>/.venv311` |

Validation evidence: see
[`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md)
"2026-06-22 (morning) — Verification baseline".

## Compatibility rules

### Pypto / pto-isa / PTOAS / ptoas-bin

The pypto codegen emits MLIR consumed by PTOAS. Their wire format
changes occasionally; mismatched pypto + ptoas-bin throws parser
errors at compile time.

Known historical mismatch:
- pypto post-`505abd64` (TCIOp `hasCustomAssemblyFormat`) requires
  ptoas-bin ≥ `v0.45`. The Phase 19 blocker 1 was exactly this
  mismatch — pypto had moved on, ptoas-bin was still `v0.44`.

Rule: when bumping pypto across an upstream commit touching MLIR ops,
bump ptoas-bin alongside.

### pypto / simpler

simpler is a git submodule of pypto under `pypto/runtime/`. The pin
in the `pypto` repo dictates which simpler commit is built. When you
update simpler, you must `git submodule update` and re-commit the
pypto-side submodule pin.

Current simpler pin (a6e06406) carries 4 patches that the upstream
maintainers have not yet merged. Track these in
`<workspace>/pypto/runtime` working tree.

### CANN

CANN beta.1 is **required**. CANN GA breaks simpler init (see
[`phase16-three-pillars.md`](phase16-three-pillars.md) "CANN GA failure
mode"). Do NOT upgrade CANN unless Huawei releases a new beta or GA
that explicitly fixes the AICPU `libaicpu_extend_kernels.so` push
path.

### Driver + firmware

Always paired. Driver-only or firmware-only upgrade is not validated.
The cap `support_shmem_map_exbus` is gated by both.

## Upgrade order (when moving everything forward)

Recommended sequence:

1. Firmware (writes chip flash; do this first while the rest is still
   on old versions).
2. Driver (re-installs in host filesystem; requires daemonset drain).
3. Reboot host.
4. CANN (only if Huawei releases a beta/GA that's verified compatible).
5. simpler (submodule under pypto).
6. pypto + pto-isa + PTOAS + pypto-lib (in any order, but rebuild in
   the order pypto → pto-isa → PTOAS → pypto-lib if reinstalling).
7. ptoas-bin (binary drop-in, paired with PTOAS source pin).

Always run the smoke + simpler L3 allreduce verification after each
step.

## Repos not in the project but adjacent

| Repo | Role | Pin we track |
|------|------|--------------|
| `vLLM stepcast fork` | Phase 2 integration target | `0e0901376` on `develop` (gitlab.basemind.com/sys/stepcast/vllm) |
| `pypto-serving` | Earlier serving wrapper (predates this project) | not actively tracked; see `<workspace>/pypto-serving/` if needed |

## Related docs

- [`phase16-three-pillars.md`](phase16-three-pillars.md) — why the
  driver/firmware/CANN binding is hard
- [`machine-recovery.md`](machine-recovery.md) — how to install/upgrade
- [`../STATUS.md`](../STATUS.md) — most recent pin snapshot row
- [`../archive/milestones-2026-Q2.md`](../archive/milestones-2026-Q2.md) "Pin
  snapshot history" — past pins
