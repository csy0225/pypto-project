# 部署 Deployment

pypto step3p5 栈的**生产 runbook**。部署新机器或升级现有机器之前请读本目录所有文档。

> **从零装环境（拉了 GitHub 仓库后一步步装到跑通）**：见
> [`../.claude/skills/pypto-runtime-install/SKILL.md`](../.claude/skills/pypto-runtime-install/SKILL.md)
> —— 分步 + 每步踩坑规避。本目录的三篇是它引用的硬性依据。

## 内容

| 文档 | 用途 |
|------|------|
| [`phase16-three-pillars.md`](phase16-three-pillars.md) | Driver + firmware + CANN 三剑合璧硬绑定。**任何生产多卡部署必读**。 |
| [`machine-recovery.md`](machine-recovery.md) | 0162 / 0234 主机 setup + 重启恢复 runbook。 |
| [`version-matrix.md`](version-matrix.md) | 5 仓库 + 工具链版本兼容表（哪些 pin 是一起验证过的）。 |

## 排障去哪里

旧的 `troubleshooting-*.md` 和 cotenancy 文档已**移到** [`../postmortems/`](../postmortems/README.md)。
撞到运行时报错时：

| 症状 | 去向 |
|------|------|
| `507899` / `507018` 多卡 IPC | [`../postmortems/01-multirank-ipc-507899-507018.md`](../postmortems/01-multirank-ipc-507899-507018.md) |
| 0234 L3 allreduce `207006` / `507899` | [`../postmortems/02-0234-l3-ipc-pid-validation.md`](../postmortems/02-0234-l3-ipc-pid-validation.md) |
| vLLM + pypto 同卡 HCCL 冲突 | [`../postmortems/03-hccl-cotenancy.md`](../postmortems/03-hccl-cotenancy.md) |
| 8001 bridge live 运维（恢复顺序 / exbus 泄漏 / PID ns） | [`../postmortems/11-8001-bridge-live-ops.md`](../postmortems/11-8001-bridge-live-ops.md) |
| 其他已知工程问题 | [`../postmortems/README.md`](../postmortems/README.md) 索引 |
| 当前活跃 blocker | [`../blockers.md`](../blockers.md) |

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

任一失败 → 查 [`machine-recovery.md`](machine-recovery.md) "常见部署失败"，
或对照 [`../postmortems/`](../postmortems/README.md) 里的 error signature。

## 相关

- 事故复盘索引：[`../postmortems/`](../postmortems/README.md)
- 版本兼容背景：[`../STATUS.md`](../STATUS.md) Pin Snapshot
- canonical 验收：[`../reference/canonical-test.md`](../reference/canonical-test.md)
