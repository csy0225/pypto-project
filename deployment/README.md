# Deployment

Production deployment specs for the pypto step3p5 stack. Read these
before deploying to a new machine or upgrading existing ones.

## Contents

| Doc | Purpose |
|-----|---------|
| [`phase16-three-pillars.md`](phase16-three-pillars.md) | Driver + firmware + CANN hard binding for multi-card e2e. **Required** for any production deploy. |
| [`machine-recovery.md`](machine-recovery.md) | Per-host setup / recovery runbook. `gpu-a910x-0162` (verified) + `gpu-a910x-0234` (pending upgrade). |
| [`version-matrix.md`](version-matrix.md) | 5-repo + toolchain version compatibility table. What pins are validated together. |

## Pre-deploy checklist

1. ✅ Read [`phase16-three-pillars.md`](phase16-three-pillars.md) and
   verify all three components are at the required version.
2. ✅ Confirm CANN is **beta.1**, not GA. GA fails simpler init with
   507018.
3. ✅ Confirm host filesystem layout — if netboot/tmpfs, follow the
   NVMe-persistence pattern in
   [`machine-recovery.md`](machine-recovery.md).
4. ✅ Drain Kubernetes daemonsets (`device-plugin`, `npu-exporter`)
   before driver upgrade.
5. ✅ Back up any existing `/usr/local/Ascend/cann-9.0.0-beta.1/` to
   persistent storage **before** running any cluster automation —
   automation may revert to CANN GA and break simpler.

## Verification (post-deploy)

After all three pillars are in place, run these in order:

```bash
# 1. Driver / firmware via npu-smi
npu-smi info -t board -i 0 | grep -E "Software|Firmware"
# Expected: Software Version 25.5.2, Firmware Version 7.8.0.7.220

# 2. CANN symlink
ls -la /usr/local/Ascend/cann-9.0.0-beta.1
# Expected: points to NVMe install

# 3. simpler L3 allreduce (double-card)
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa
cd <workspace>/pypto/runtime
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1
# Expected: max |out - expected| = 0.000e+00 on both ranks
```

Any failure → consult [`machine-recovery.md`](machine-recovery.md)
"Common deploy failures".
