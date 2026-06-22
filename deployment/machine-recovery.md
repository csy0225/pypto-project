# 机器恢复 Runbook

把主机升到 Phase 16 合规，以及 netboot/tmpfs 主机重启后怎么恢复。

## 主机清单

| 主机 | 类型 | Phase 16 合规 | 备注 |
|------|------|---------------|------|
| `gpu-a910x-0162` | Netboot/tmpfs（8× 910B2C） | ✅ 是（2026-06-22 重验） | 参考机。持久 state 在 NVMe。 |
| `gpu-a910x-0234` | TBD | ⏸ 否（driver/firmware 低于 min） | 升级未做。详见 [`../blockers.md`](../blockers.md) §5。 |

## 0162 —— 重启后恢复

0162 是 netboot/tmpfs。主机重启后下面这些**丢失**：

- `/usr/local/Ascend/driver/`（driver kernel module + libdrv_*.so）
- `/etc/ascend_install.info`（driver 安装状态）
- 大部分 `/etc/`、`~/.ssh/authorized_keys`（被集群 provisioning 重写）

下面这些**保留**（NVMe 持久 + symlink）：

- `/mnt/persist/`（整个目录，含 CANN 安装 + .run 包 staging + 备份）
- `/data/chensiyu/`（整个目录，含 workspace + venv + git 仓库 + probe2）
- Firmware（chip flash 板级 —— `7.8.0.7.220` 一旦写入就留）

### 恢复步骤

一键脚本在主机 `/mnt/persist/RECOVERY.sh`：

```bash
# 以 root 运行
sudo bash /mnt/persist/RECOVERY.sh

# 验证
bash /mnt/persist/RECOVERY.sh --verify
```

脚本做什么（幂等）：

1. 停 `kubelet` + `bip-agent` + 占 `/dev/davinci*` 的 DaemonSet 进程。
2. 从 `/mnt/persist/ascend-staging/Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run`
   重装 driver 25.5.2（firmware 不动 —— chip flash 还在）。
3. 确保 `/usr/local/Ascend/` symlink 指向 NVMe 持久的 CANN
   安装 `/mnt/persist/Ascend/cann-9.0.0-beta.1`。
4. 重启 `kubelet`。

driver 重装完后，workspace 正常激活 —— NVMe 上的 venv 还在。用户侧验证：

```bash
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa

cd <workspace>/pypto-lib
python -m models.step3p5._smoke_program_build       # 期望: probe rc=0

cd <workspace>/pypto/runtime
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1
# 期望: max |out - expected| = 0.000e+00 两卡
```

## 0234 —— 升级未做

当前状态：

- Driver `25.5.1`（低于 required 25.5.2）
- Firmware `7.8.0.6.201`（低于 required 7.8.0.7.220）
- CANN `9.0.0-beta.1` ✅（**正确，别动**）

driver + firmware 升上去之前，多卡 e2e 被卡。

### 升级步骤（最小集 —— 只升 driver + firmware）

```bash
# 1. 从 0162 staging scp .run 包
scp infra@gpu-a910x-0162:/mnt/persist/ascend-staging/Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run /tmp/
scp infra@gpu-a910x-0162:/mnt/persist/ascend-staging/Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run /tmp/

# 2. 在任何集群自动化跑之前**先**备份 CANN beta.1
sudo cp -a /usr/local/Ascend/cann-9.0.0-beta.1 \
          /<persistent>/cann-9.0.0-beta.1.backup-$(date +%Y%m%d)
# 关键 —— 集群自动化可能 revert 到 CANN GA

# 3. 停占 /dev/davinci* 的 daemonset
sudo systemctl stop kubelet
sudo systemctl stop bip-agent
sudo pkill -f device-plugin
sudo pkill -f npu-exporter

# 4. 跑 driver 升级
sudo bash /tmp/Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run --upgrade --quiet

# 5. 跑 firmware 升级（写 chip flash）
sudo bash /tmp/Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run --upgrade --quiet

# 6. 重启
sudo reboot

# 7. 重启后验证
npu-smi info -t board -i 0 | grep -E "Software|Firmware"
# 期望: 25.5.2 + 7.8.0.7.220
```

估计 wallclock：~2 小时（含重启）。

## 常见部署失败

### `aclrtIpcMemImportByKey` 返回 507899

driver 低于 25.5.2 或 firmware 低于 7.8.0.7.220。`npu-smi info -t board
-i 0` 检查两者并相应升级。

### simpler init 失败 507018 (BootstrapDispatcher)

CANN 是 GA，不是 beta.1。恢复 beta.1 symlink：

```bash
sudo ln -sfn /mnt/persist/Ascend/cann-9.0.0-beta.1 \
            /usr/local/Ascend/cann-9.0.0-beta.1
```

如果 beta.1 被集群自动化删了，从升级步骤 2 的备份恢复。

### driver 升级失败：device busy

Kubernetes daemonset 占 `/dev/davinci*`。`kubectl drain` 还不够 ——
DaemonSet 由 containerd 直接拉起。

```bash
sudo systemctl stop kubelet
sudo systemctl stop bip-agent
sudo pkill -f device-plugin
sudo pkill -f npu-exporter
sleep 2
# 再试 driver 升级
```

### driver 安装失败 "buffer_elems" `-Werror`（rebuild pypto 时）

这不是 driver 失败 —— 是 `pip install -e .` of pypto 撞到
`tensor.h:535 buffer_elems` `-Werror=unused-variable` 在
`CMAKE_BUILD_TYPE=Release` 下。修法：别传 `CMAKE_BUILD_TYPE`，用 dev
default。另外 rebase 后第一次 build 要
`rm -rf build/cp311-* build/cache build/lib`。

### git push 走 HTTPS 130 秒超时

用 HTTP/1.1：`git -c http.version=HTTP/1.1 push ...`。详见
`pypto-lib/docs/dev-workflow-gotchas.md` §3。

## 相关文档

- [`phase16-three-pillars.md`](phase16-three-pillars.md) —— 部署什么绑定
  以及为什么
- [`version-matrix.md`](version-matrix.md) —— 完整版本兼容
- [`../blockers.md`](../blockers.md) §5 —— 0234 升级 owner / 状态
