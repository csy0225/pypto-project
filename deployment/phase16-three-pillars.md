# Phase 16 Three-Pillars Binding

The minimum viable deployment for any production multi-rank step3p5
run on Ascend 910B / A2A3 platform. **All three components must be at
the listed version.** Mixing in older versions silently breaks
multi-card collectives.

## The binding

| Component | Required | Notes |
|-----------|----------|-------|
| Driver | `25.5.2` | Linux x86-64 .run package: `Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run` |
| Firmware | `7.8.0.7.220` | .run package: `Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run`. **Writes to chip flash; persists across host reboot.** |
| CANN | `9.0.0-beta.1` | NOT GA. Toolkit + nnal. Must NOT be replaced with `9.0.0` or later GA. |

Plus a small simpler-side patch (already in `csy0225/simpler` HEAD
`a6e06406`):
`comm_hccl.cpp` adds `__has_include`-guarded `*Inner` macro alias for
forward-compatibility with CANN GA's renamed HCCL entries. No-op under
beta.1.

## Why all three are required (failure modes)

### Older driver: `support_shmem_map_exbus = 0`

Drivers below 25.5.2 expose this device capability flag as 0. Any
attempt to do cross-card IPC via
`aclrtIpcMemImportByKey + ENABLE_PEER_ACCESS` returns 507899:

```
[ERROR] aclrtIpcMemImportByKey failed: 507899
```

This blocks every multi-card collective primitive. simpler L3
`allreduce_distributed` cannot make progress.

### Older firmware: same cap gap

Firmware and driver gate the cap together. Both must move.

### CANN GA (not beta.1): TDT does not push AICPU library

CANN 9.0.0 GA's TDT **does not** push
`Ascend-aicpu_extend_syskernels.tar.gz` (encrypted aa55aa55 format)
to the AICPU device-side at `/usr/lib64/aicpu_kernels/`. Without that
tarball, simpler's `BootstrapDispatcher` cannot find
`DynTileFwkKernelServerInit` and fails with:

```
[ERROR] Load so libaicpu_extend_kernels.so failed
[ERROR] BootstrapDispatcher: aclrtSynchronizeStream failed: 507018
```

This is independent of the driver/firmware cap fix. Both must be
correct simultaneously.

CANN beta.1's TDT, by contrast, does push the tarball at init. Hence
the binding to beta.1 specifically.

### When this binding may relax

- Upstream simpler rewrites `BootstrapDispatcher` to not depend on
  `DynTileFwkKernelServerInit` / `libaicpu_extend_kernels.so`. Then
  CANN constraint may relax to GA. PR `#1061` did NOT do this — that
  PR removed simpler's own `simpler_aicpu_init` monitor kernel, not
  the upstream-AICPU-library hardcode.
- Huawei's next CANN release (≥ 9.0.0 second beta or 9.1+) where
  TDT's tarball-push behaviour matches beta.1 will let us upgrade.

Until then: **bind tight to the three pillars above.**

## Verifying current state on a host

```bash
# Driver + firmware via npu-smi
npu-smi info -t board -i 0 | grep -E "Software|Firmware"
# Expected:
#   Software Version    : 25.5.2
#   Firmware Version    : 7.8.0.7.220

# CANN install path
ls -la /usr/local/Ascend/cann-9.0.0-beta.1
# Expected: directory or symlink to NVMe install

# CANN env script readable
test -f /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh && echo OK
```

If any check fails, the host is not Phase 16 compliant. See
[`machine-recovery.md`](machine-recovery.md) for the upgrade runbook.

## Validation reference (gpu-a910x-0162, 2026-06-22)

The reference machine validating this binding is `gpu-a910x-0162` in
the lab cluster. Validation evidence:

- `probe2.c` cross-card `aclrtIpcMemImportByKey + ENABLE_PEER_ACCESS`
  returns `rc=0` with `peer_va == parent ptr = 0x12c1c0000000` (i.e.
  same VA cross-card mapping established).
- simpler L3 `allreduce_distributed -p a2a3 -d 0-1` produces
  `max|out-expected|=0.000e+00` on both ranks (golden match).

Source code commit `csy0225/simpler@c66b4120` originally validated this
on `stepfun/develop`; current pin is `a6e06406`.

## Related docs

- [`machine-recovery.md`](machine-recovery.md) — how to install /
  recover this binding on a fresh or rebooted host
- [`version-matrix.md`](version-matrix.md) — full 5-repo + toolchain
  pin compatibility
- [`../blockers.md`](../blockers.md) §5 — 0234 driver+firmware upgrade
  pending
