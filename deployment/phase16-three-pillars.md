# Phase 16 三剑合璧绑定

Ascend 910B / A2A3 平台上任何生产多卡 step3p5 run 的最低部署要求。
**三个组件必须全在指定版本**。混入旧版本会静默破坏多卡 collective。

## 绑定

| 组件 | 必需版本 | 说明 |
|------|----------|------|
| Driver | `25.5.2` | Linux x86-64 .run 包：`Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run` |
| Firmware | `7.8.0.7.220` | .run 包：`Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run`。**写入 chip flash，跨主机重启持久。** |
| CANN | `9.0.0-beta.1` | NOT GA。Toolkit + nnal。**不能**被替换成 `9.0.0` 或更新的 GA 版本。 |

加 simpler 侧一个小 patch（已在 `csy0225/simpler` HEAD `a6e06406`）：
`comm_hccl.cpp` 加 `__has_include` 守护的 `*Inner` macro alias，对 CANN
GA 重命名 HCCL 入口的 forward-compatibility。beta.1 下守护不激活，
无副作用。

## 为什么三件都必需（失败模式）

### Driver 旧：`support_shmem_map_exbus = 0`

25.5.2 以下的 driver 把这个设备 capability flag 显示为 0。任何跨卡 IPC
通过 `aclrtIpcMemImportByKey + ENABLE_PEER_ACCESS` 都返回 507899：

```
[ERROR] aclrtIpcMemImportByKey failed: 507899
```

这会阻塞所有多卡 collective primitive。simpler L3
`allreduce_distributed` 无法 progress。

### Firmware 旧：同样 cap 缺口

Firmware 和 driver 一起 gate 这个 cap。必须一起升。

### CANN GA（不是 beta.1）：TDT 不推 AICPU 库

CANN 9.0.0 GA 的 TDT **不会** 把
`Ascend-aicpu_extend_syskernels.tar.gz`（aa55aa55 加密格式）推到 AICPU
设备端 `/usr/lib64/aicpu_kernels/`。没这个 tarball，simpler 的
`BootstrapDispatcher` 找不到 `DynTileFwkKernelServerInit`，失败信息：

```
[ERROR] Load so libaicpu_extend_kernels.so failed
[ERROR] BootstrapDispatcher: aclrtSynchronizeStream failed: 507018
```

这跟 driver/firmware cap 修复独立。两件都得对。

CANN beta.1 的 TDT 反过来，init 时会推这个 tarball。因此硬绑 beta.1。

### 这个绑定何时可能放宽

- 上游 simpler 改写 `BootstrapDispatcher`，不再依赖
  `DynTileFwkKernelServerInit` / `libaicpu_extend_kernels.so`。届时 CANN
  限制可放宽到 GA。PR `#1061` **没有** 做这件事 — 那个 PR 删的是 simpler
  自己的 `simpler_aicpu_init` 监控 kernel，**不是** 上游 AICPU 库的
  hardcoded 依赖。
- Huawei 下一个 CANN 版本（≥ 9.0.0 第二个 beta，或 9.1+）TDT 行为修复
  了 GA 那条 push 路径。届时可升级。

在那之前：**死绑上面三件**。

## 在主机上验证当前状态

```bash
# driver + firmware 通过 npu-smi
npu-smi info -t board -i 0 | grep -E "Software|Firmware"
# 期望:
#   Software Version    : 25.5.2
#   Firmware Version    : 7.8.0.7.220

# CANN 安装路径
ls -la /usr/local/Ascend/cann-9.0.0-beta.1
# 期望: 目录 或 symlink 指 NVMe install

# CANN env 脚本可读
test -f /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh && echo OK
```

任一项失败 → 主机非 Phase 16 合规。升级 runbook 见
[`machine-recovery.md`](machine-recovery.md)。

## 验证参考（gpu-a910x-0162，2026-06-22）

验证这个绑定的参考机是实验集群里的 `gpu-a910x-0162`。验证证据：

- `probe2.c` 跨卡 `aclrtIpcMemImportByKey + ENABLE_PEER_ACCESS` 返回
  `rc=0`，`peer_va == parent ptr = 0x12c1c0000000`（即跨卡 same VA
  mapping 建立）。
- simpler L3 `allreduce_distributed -p a2a3 -d 0-1` 两卡都给
  `max|out-expected|=0.000e+00`（golden match）。

源码 commit `csy0225/simpler@c66b4120` 原始验证；当前 pin `a6e06406`。

## 相关文档

- [`troubleshooting-multirank-507899.md`](troubleshooting-multirank-507899.md)
  —— **还在撞 `507899` / `507018`？** 自助诊断决策树（哪个错误码对应哪件
  没配齐）+ 排查清单 + issue 链
- [`machine-recovery.md`](machine-recovery.md) —— 怎么在新机器或重启后的
  主机上安装/恢复这个绑定
- [`version-matrix.md`](version-matrix.md) —— 完整 5 仓库 + 工具链 pin
  兼容
- [`../blockers.md`](../blockers.md) §5 —— 0234 driver+firmware 升级未做
