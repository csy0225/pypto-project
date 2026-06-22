# Machine Recovery Runbook

How to bring up a host to Phase 16 compliance, and how to recover
after a reboot on a netboot/tmpfs host.

## Host inventory

| Host | Type | Phase 16 compliant | Notes |
|------|------|---------------------|-------|
| `gpu-a910x-0162` | Netboot/tmpfs (8x 910B2C) | ✅ Yes (2026-06-22 reverified) | Reference machine. Persistent state on NVMe. |
| `gpu-a910x-0234` | TBD | ⏸ No (driver/firmware below min) | Upgrade pending. See [`../blockers.md`](../blockers.md) §5. |

## 0162 — fresh boot recovery

0162 is netboot/tmpfs. After a host reboot, the following are LOST:

- `/usr/local/Ascend/driver/` (driver kernel module + libdrv_*.so)
- `/etc/ascend_install.info` (driver install state)
- Most of `/etc/`, `~/.ssh/authorized_keys` (replaced by cluster provisioning)

The following SURVIVE (persistent NVMe via symlinks):

- `/mnt/persist/` (entire dir, contains CANN install + .run package staging + backups)
- `/data/chensiyu/` (entire dir, contains workspace + venv + git repos + probe2)
- Firmware (chip flash, board-level — `7.8.0.7.220` once written stays)

### Recovery steps

A one-shot recovery script is staged at `/mnt/persist/RECOVERY.sh` on
the host:

```bash
# As root
sudo bash /mnt/persist/RECOVERY.sh

# Verify
bash /mnt/persist/RECOVERY.sh --verify
```

What it does (idempotent):

1. Stops `kubelet` + `bip-agent` + DaemonSet processes holding
   `/dev/davinci*`.
2. Re-installs driver 25.5.2 from
   `/mnt/persist/ascend-staging/Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run`
   (firmware untouched — already in chip flash).
3. Ensures `/usr/local/Ascend/` symlinks point to NVMe-persistent
   CANN install at `/mnt/persist/Ascend/cann-9.0.0-beta.1`.
4. Restarts `kubelet`.

After driver re-install, the workspace activates normally — venv on
NVMe survives. User-side verification:

```bash
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa

cd <workspace>/pypto-lib
python -m models.step3p5._smoke_program_build       # expect: probe rc=0

cd <workspace>/pypto/runtime
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1
# expect: max |out - expected| = 0.000e+00 on both ranks
```

## 0234 — pending upgrade

Current state:

- Driver `25.5.1` (below required 25.5.2)
- Firmware `7.8.0.6.201` (below required 7.8.0.7.220)
- CANN `9.0.0-beta.1` ✅ (correct — DO NOT touch)

Multi-card e2e is blocked until driver + firmware are upgraded.

### Upgrade steps (minimal — only driver + firmware)

```bash
# 1. SCP .run packages from 0162 staging
scp infra@gpu-a910x-0162:/mnt/persist/ascend-staging/Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run /tmp/
scp infra@gpu-a910x-0162:/mnt/persist/ascend-staging/Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run /tmp/

# 2. Back up CANN beta.1 BEFORE any cluster automation runs
sudo cp -a /usr/local/Ascend/cann-9.0.0-beta.1 \
          /<persistent>/cann-9.0.0-beta.1.backup-$(date +%Y%m%d)
# This is critical — cluster automation may revert to CANN GA

# 3. Stop daemonsets holding /dev/davinci*
sudo systemctl stop kubelet
sudo systemctl stop bip-agent
sudo pkill -f device-plugin
sudo pkill -f npu-exporter

# 4. Run driver upgrade
sudo bash /tmp/Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run --upgrade --quiet

# 5. Run firmware upgrade (writes to chip flash)
sudo bash /tmp/Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run --upgrade --quiet

# 6. Reboot
sudo reboot

# 7. After reboot, verify
npu-smi info -t board -i 0 | grep -E "Software|Firmware"
# Expected: 25.5.2 + 7.8.0.7.220
```

Estimated wallclock: ~2 hours (including reboot).

## Common deploy failures

### `aclrtIpcMemImportByKey` returns 507899

Driver below 25.5.2 OR firmware below 7.8.0.7.220. Check both via
`npu-smi info -t board -i 0` and upgrade accordingly.

### simpler init fails with 507018 (BootstrapDispatcher)

CANN is GA, not beta.1. Restore the beta.1 symlink:

```bash
sudo ln -sfn /mnt/persist/Ascend/cann-9.0.0-beta.1 \
            /usr/local/Ascend/cann-9.0.0-beta.1
```

If beta.1 was deleted by cluster automation, restore from the backup
created in upgrade step 2 above.

### Driver upgrade fails: device busy

Kubernetes daemonsets hold `/dev/davinci*`. `kubectl drain` is not
enough — DaemonSets are pulled up by containerd directly.

```bash
sudo systemctl stop kubelet
sudo systemctl stop bip-agent
sudo pkill -f device-plugin
sudo pkill -f npu-exporter
sleep 2
# Re-try driver upgrade
```

### Driver install fails with "buffer_elems" `-Werror` (when re-building pypto)

This is not a driver failure — it's a `pip install -e .` of pypto
hitting `tensor.h:535 buffer_elems` `-Werror=unused-variable` under
`CMAKE_BUILD_TYPE=Release`. Fix: do not pass `CMAKE_BUILD_TYPE`, use
dev default. Also `rm -rf build/cp311-* build/cache build/lib` on
first build after rebase.

### `git push` over HTTPS times out at 130s

Use HTTP/1.1: `git -c http.version=HTTP/1.1 push ...`. See
`pypto-lib/docs/dev-workflow-gotchas.md` §3.

## Related docs

- [`phase16-three-pillars.md`](phase16-three-pillars.md) — what binding
  to deploy and why
- [`version-matrix.md`](version-matrix.md) — full version compatibility
  table
- [`../blockers.md`](../blockers.md) §5 — 0234 upgrade owner / status
