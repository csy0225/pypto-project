# 部署 Deployment

pypto step3p5 栈的生产部署 spec。**部署新机器或升级现有机器之前请读
本目录所有文档。**

## 内容

| 文档 | 用途 |
|------|------|
| [`phase16-three-pillars.md`](phase16-three-pillars.md) | Driver + firmware + CANN 多卡 e2e 的硬绑定。**任何生产部署都必读**。 |
| [`machine-recovery.md`](machine-recovery.md) | 各主机 setup / 重启恢复 runbook。`gpu-a910x-0162`（已验证）+ `gpu-a910x-0234`（待升级）。 |
| [`version-matrix.md`](version-matrix.md) | 5 仓库 + 工具链版本兼容表。哪些 pin 是一起验证过的。 |
| [`troubleshooting-multirank-507899.md`](troubleshooting-multirank-507899.md) | **撞到 `507899` / `507018` 多卡 IPC 报错怎么自助排查**。诊断决策树 + issue 链。 |
| [`troubleshooting-0234-l3-ipc-pid-validation.md`](troubleshooting-0234-l3-ipc-pid-validation.md) | **0234 PyPTO L3 allreduce 507899/207006**：`SetImportPid` 白名单路径问题、`DISABLE_PID_VALIDATION` 修复和验证记录。 |
| [`troubleshooting-8001-pypto-bridge.md`](troubleshooting-8001-pypto-bridge.md) | **live 8001（dense/shared/attention PyPTO bridge）运维排障**：恢复顺序铁律（先 8001 再 worker，否则 HCCL `rtBinaryGetFunction 107000`）、`pkill -f` 自匹配、exbus 句柄泄漏、安全停 8001。 |
| [`troubleshooting-mat-mat-tmov-vec-lhs-matmul.md`](troubleshooting-mat-mat-tmov-vec-lhs-matmul.md) | **升级栈（pypto `5e619dc7`）编译 step3p5 撞 `'pto.tmov' … supported tmov address-space pair`**：PR #1601 的 Vec-LHS staging × 910B GM-pipe 的 Mat→Mat copy-out（非法）；模型侧 `OUT_PROJ_N_CHUNK` 256→64 数据通路 reshape 修复 + stale-pyc 陷阱 + 上游 bug。 |

## 部署前 checklist

1. ✅ 读 [`phase16-three-pillars.md`](phase16-three-pillars.md) 确认三个组件
   都达到 required 版本。
2. ✅ 确认 CANN 是 **beta.1**，不是 GA。GA 会让 simpler init 507018 失败。
3. ✅ 确认主机文件系统布局 —— 如果是 netboot/tmpfs，按
   [`machine-recovery.md`](machine-recovery.md) 走 NVMe 持久化模式。
4. ✅ 在升级 driver 前 drain Kubernetes daemonset
   (`device-plugin`, `npu-exporter`)。
5. ✅ 跑任何集群自动化前**先**把 `/usr/local/Ascend/cann-9.0.0-beta.1/`
   备份到 persistent storage —— 自动化可能 revert 到 CANN GA 弄坏 simpler。

## 部署后验证

三剑合璧就位后，按顺序跑：

```bash
# 1. driver / firmware 通过 npu-smi
npu-smi info -t board -i 0 | grep -E "Software|Firmware"
# 期望: Software Version 25.5.2, Firmware Version 7.8.0.7.220

# 2. CANN symlink
ls -la /usr/local/Ascend/cann-9.0.0-beta.1
# 期望: 指向 NVMe install

# 3. simpler L3 allreduce（双卡）
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa
cd <workspace>/pypto/runtime
python examples/workers/l3/allreduce_distributed/main.py -p a2a3 -d 0-1
# 期望: max |out - expected| = 0.000e+00 两卡都通过
```

任一失败 → 查 [`machine-recovery.md`](machine-recovery.md)
"常见部署失败"。
