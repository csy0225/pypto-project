# 执行主机契约 —— 0162 是唯一执行主机

> 立于 2026-08-10。适用于**所有** agent / session（Claude、codex、以及后续任何协作者）。
> 起因：codex 在本地机器上跑了 P3 swimlane 分析，产生了不可在目标机复现的 artifact。

---

## 1. 铁律

| 主机 | 允许 | 禁止 |
|------|------|------|
| **`gpu-a910x-0162`** | 编译、测试、回归、device 运行、**任何数据分析** | — |
| **本地 `/data/chensiyu`** | 代码同步、写文档、**创作**脚本、读 artifact | **执行**任何产生结论的代码 |

判定边界（这条最容易含糊，写清）：

- ✅ **本地写脚本 → 上传 0162 → 在 0162 执行**。这是正确模式。
- ❌ **在本地执行脚本产生结论**。即使脚本是纯 stdlib、即使结论"看起来一样"。

理由不是形式主义，是三个具体的漂移源：

1. **工具漂移**：把 0162 的工具（如 `simpler_setup/tools/swimlane_converter.py`）拷到本地
   vendored 使用，没有 commit 记录 → 上游一变，本地这份静默过期，而且没人知道。
2. **解释器漂移**：0162 是 python **3.11.14**；本地是 **3.10**。dict 序、float repr、
   stdlib 行为差异都可能改变派生指标。
3. **不可复现**：产出 artifact 的脚本不在目标机上，别人无法在目标机重算，
   评审时无法独立验证。

---

## 2. 分析类工作的落地规范

任何要写进报告/决策的分析，必须满足：

1. 脚本放在 `0162:/mnt/persist/chensiyu/workspace/perf-2026q3/<campaign>/`，
   与它消费的 artifact **同机**。
2. 输入指向 **0162 上的原始 artifact**，不指向任何副本。
3. 产出的 JSON/CSV 里带 provenance 字段（输入文件绝对路径 + sha256）。
4. 本地若需要副本用于阅读，放 provenance 标记文件说明它是快照、非事实源。

---

## 3. 双 agent 并发的资源管理

16 张卡（`dev0-15`），三个锁文件都在 `0162:/mnt/persist/chensiyu/workspace/`：

| 锁 | 覆盖 | 谁用 |
|---|---|---|
| `0162-cards0-7.lock` | dev0–7 | Claude |
| `0162-cards8-15.lock` | dev8–15 | codex |
| `0162-full-machine-perf.lock` | 全 16 卡 | **最终 A/B/A 发布门专用** |

格式同既有整机锁（`holder=none` 表示空闲）。

**方法论约束**：并发 8+8 会通过 host CPU / PCIe 竞争污染**绝对**计时。所以

- 诊断 / 相对分析 → 允许并发
- **A/B/A 发布门 → 必须持整机锁串行跑，两半都空**

### vanilla 精度 oracle

容器 `vllm-oracle-0724-n256`（8× `VLLMWorker_TP`，各 54.7 GiB，占 dev0–7）。
2026-08-10 已 `stop`（用户授权），状态 `Exited (0)`，可 `nerdctl start` 拉回。

只有当 candidate **不是 hidden byte-exact** 时才需要它（多步 decode 逐 token vs live
vanilla，N=128 ALIGNED ≥ 95%）。byte-exact 的改动用 hidden sha256 即可，不需要 oracle。

---

## 4. 无卡的 codegen 门

不占卡就能验证 codegen 是否通过（约 13.5s，NB=512）：

```bash
0162:/mnt/persist/chensiyu/workspace/compile_gate.sh <source-tree> <label> [NB]
```

必须带的 nerdctl flag（少一个就挂，都踩过）：

- `--security-opt apparmor=unconfined` → 否则 `apparmor_parser resolves to executable in current directory`
- `--net host` → 否则 `needs CNI plugin "bridge" ... /opt/cni/bin/bridge: no such file`

产出 `manifest.txt`（source sha256 / image digest / NB / rc）+ `compile.log`。
