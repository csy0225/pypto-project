# 专项：0234 L3 allreduce 跨卡 IPC `aclrtIpcMemSetImportPid 207006` / `ImportByKey 507899`

| 字段 | 值 |
|------|----|
| **子系统** | deployment / whole-net |
| **error signature** | `aclrtIpcMemSetImportPid -> 207006`、`domain_alloc_via_ipc: ImportByKey(...) -> 507899` |
| **首次出现** | 2026-06-29 |
| **状态** | ✅ 已解（本地 patch 落地，L3 allreduce 双卡 golden match） |
| **相关 skill / doc** | [`01-multirank-ipc-507899-507018.md`](01-multirank-ipc-507899-507018.md)、[`../deployment/phase16-three-pillars.md`](../deployment/phase16-three-pillars.md) |

---

## 1. 背景（Background）

- **机器 / 容器**：`gpu-a910x-0234.host.platform.shaipower.com`（`tmux attach -t pypto-ascend`）
- **日期**：2026-06-29
- **范围**：PyPTO / simpler 自己的多卡 L3 runtime 回归；**不是** vLLM 集成路径。
- **前置条件**：0234 已经升级到 Phase 16 所需的 driver / firmware：

```text
Software Version : 25.5.2
Firmware Version : 7.8.0.7.220
```

- **预期目标**：PyPTO L3 `allreduce_distributed` 在 0234 双卡（device 0,1）跑通 `max |out - expected| = 0.000e+00`。

---

## 2. 现象（Symptom）

### 现象 A：裸 ACL 跨卡 IPC probe 通过

```text
aclrtIpcMemGetExportKey(..., ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION)
aclrtIpcMemImportByKey(..., ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS)
=> import_rc=0, payload readback ok
```

裸 probe 在 `device 0 -> device 1` 上可以通过。

### 现象 B：PyPTO L3 allreduce 原路径仍失败

```text
[chip_process pid=594 dev=0] ready
[chip_process pid=596 dev=1] ready
[ERROR] domain_alloc_via_ipc: [comm_hccl.cpp:831]
[comm rank 0] alloc_domain: ImportByKey(peer_dr=1 pid=596) -> 507899
[ERROR] domain_alloc_via_ipc: [comm_hccl.cpp:831]
[comm rank 1] alloc_domain: ImportByKey(peer_dr=0 pid=594) -> 507899
RuntimeError: alloc_domain(allocation_id=0) failed on 2/2 chips
```

### 现象 C：只改 export flag 后错误变成 SetImportPid 207006

把两处 `aclrtIpcMemGetExportKey(..., 0)` 改为 `ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION` 重编后再跑：

```text
[ERROR] domain_alloc_via_ipc: [comm_hccl.cpp:802]
[comm rank 0] alloc_domain: SetImportPid -> 207006
[ERROR] domain_alloc_via_ipc: [comm_hccl.cpp:802]
[comm rank 1] alloc_domain: SetImportPid -> 207006
```

### 现象 D：跳过 SetImportPid 后 L3 allreduce 通过

```text
[chip_process pid=8098 dev=1] ready
[chip_process pid=8096 dev=0] ready
[allreduce] platform=a2a3 devices=[0, 1] nranks=2
[allreduce] running 2-chip allreduce DAG...
[allreduce] chip 0: rank=0/2 window=[0x12c0c01e4000 +4096B] scratch=0x12c0c01e4000
[allreduce] chip 1: rank=1/2 window=[0x12c0c01e4000 +4096B] scratch=0x12c0c01e4000
[allreduce] chip 0: max |out - expected| = 0.000e+00
[allreduce] chip 1: max |out - expected| = 0.000e+00
[allreduce] all ranks matched golden ✅
```

---

## 3. 根因（Root Cause）

0234 当前环境中的剩余问题不是跨卡 IPC capability 完全不通（裸 probe 通过证明 capability 已通），而是 **PyPTO / simpler 原来的 `export flags=0 + aclrtIpcMemSetImportPid + peer import` 白名单路径在该容器 / forked `chip_process` 场景下不可靠**。

### 证据链

仓库：`pypto/runtime` submodule（`simpler`），代码路径 `src/a2a3/platform/onboard/host/comm_hccl.cpp`。涉及两个 IPC allocation path：

1. `alloc_windows_via_ipc(...)`
2. `domain_alloc_via_ipc(...)`（本次 L3 allreduce 失败点）

**原始关键路径**：

```cpp
aclrtIpcMemGetExportKey(localBuf, win_size, myName, kIpcNameLen, 0);
aclrtIpcMemSetImportPid(myName, peerPids.data(), peerPids.size());
aclrtIpcMemImportByKey(&peerVa, peers[p].name,
                       ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
```

**关键实验**：

| 步骤 | export flag | SetImportPid | import flag | 结果 |
|------|-------------|--------------|-------------|------|
| 1. 裸 ACL probe | `DISABLE_PID_VALIDATION` | skip | `ENABLE_PEER_ACCESS` | ✅ `import_rc=0`，payload readback ok |
| 2. L3 allreduce 原路径 | `0`（DEFAULT） | call | `ENABLE_PEER_ACCESS` | ❌ `ImportByKey -> 507899` |
| 3. 只改 export flag | `DISABLE_PID_VALIDATION` | call | `ENABLE_PEER_ACCESS` | ❌ `SetImportPid -> 207006` |
| 4. 同时跳过 SetImportPid | `DISABLE_PID_VALIDATION` | skip | `ENABLE_PEER_ACCESS` | ✅ `[allreduce] all ranks matched golden` |

从步骤 2→3 的变化（`507899` → `207006`）证明：`DISABLE_PID_VALIDATION` 与 `SetImportPid` 白名单模式不应混用；既然 export 已经禁用 PID validation，就应跳过 `aclrtIpcMemSetImportPid`。步骤 3→4 证明：跳过 `SetImportPid` 后 L3 allreduce 稳定通过。

### 修复后关键路径

```cpp
aclrtIpcMemGetExportKey(localBuf, win_size, myName, kIpcNameLen,
                        ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION);
// skip aclrtIpcMemSetImportPid(...)
aclrtIpcMemImportByKey(&peerVa, peers[p].name,
                       ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS);
```

---

## 4. 如何解决（Fix）

将 a2a3 onboard IPC window export 统一改成：

```cpp
ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION
```

并在该模式下跳过：

```cpp
aclrtIpcMemSetImportPid(...)
```

import 侧继续使用：

```cpp
ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS
```

### 复现和验证过程

#### 1. 底层跨卡 ACL IPC probe 通过

命令：

```bash
/data/chensiyu/hw_project/pypto/workspace/run_p1_ipc_multicard_0234.sh
```

日志：

```text
/data/chensiyu/hw_project/pypto/workspace/p1_ipc_multicard_0234_20260629_073604.log
```

结果：

```text
EXPORTER device=0 dev_ptr=0x12c0c0013000 export_flag=0x1
IMPORTER device=1 import_flag=0x1 import_rc=0 imported=0x12c1c0000000
IMPORTER got_head=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
exp_head=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
nbytes=4096 ok=True
P1_MULTICARD_IPC_PROBE_PASS export_device=0 import_device=1 nbytes=4096
```

#### 2. PyPTO L3 allreduce 原路径失败

命令：

```bash
/data/chensiyu/hw_project/pypto/workspace/run_pypto_l3_allreduce_0234.sh
```

日志：

```text
/data/chensiyu/hw_project/pypto/workspace/pypto_l3_allreduce_0234_20260629_073654.log
```

#### 3. 只改 export flag 后错误变成 SetImportPid 207006

先将两处 `aclrtIpcMemGetExportKey(..., 0)` 改为 `ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION`，重编：

```bash
/data/chensiyu/hw_project/pypto/workspace/rebuild_runtime_0234.sh
```

重编日志：

```text
/data/chensiyu/hw_project/pypto/workspace/rebuild_runtime_0234_20260629_074551.log
```

再次跑 L3 allreduce：

```text
/data/chensiyu/hw_project/pypto/workspace/pypto_l3_allreduce_0234_20260629_074813.log
```

#### 4. 跳过 SetImportPid 后 L3 allreduce 通过

追加 patch：两处 `aclrtIpcMemSetImportPid(...)` 跳过，并保留同步 barrier。重编：

```bash
/data/chensiyu/hw_project/pypto/workspace/rebuild_runtime_0234.sh
```

重编日志：

```text
/data/chensiyu/hw_project/pypto/workspace/rebuild_runtime_0234_20260629_075011.log
```

最终验证：

```bash
/data/chensiyu/hw_project/pypto/workspace/run_pypto_l3_allreduce_0234.sh
```

日志：

```text
/data/chensiyu/hw_project/pypto/workspace/pypto_l3_allreduce_0234_20260629_075133.log
```

### 副作用与适用边界

- 禁用 PID validation 放宽了 ACL IPC key 的 PID 白名单限制，适合作为容器化 forked worker 环境下的 runtime 兼容路径；实际安全边界依赖作业容器、设备文件权限和集群调度隔离。
- commit message / issue 描述不要写成"fix PID namespace mismatch"，而应写成：

```text
avoid brittle aclrtIpcMemSetImportPid whitelist path in forked chip_process deployments
```

---

## 5. 走过的弯路（Detours / What We Got Wrong）

- ❌ 假设「0234 问题是容器 PID 和 host PID 不一致（PID namespace mismatch）」 → 证伪：不要把这个问题简单归因成"容器 PID 和 host PID 不一致"。如果 162 / 234 的容器 PID 和 host PID 一致，也仍可能遇到该问题。现有证据只能证明 234 当前环境下 `SetImportPid` 白名单模式不可靠；234 上底层 peer IPC capability 是通的（`DISABLE_PID_VALIDATION + ENABLE_PEER_ACCESS` 的裸 probe 和 L3 allreduce 均通过）。commit message 不应写"fix PID namespace mismatch"。
- ❌ 尝试「只改 export flag 为 `DISABLE_PID_VALIDATION`，保留 `aclrtIpcMemSetImportPid`」 → 无效：错误从 `ImportByKey 507899` 变成 `SetImportPid 207006`。`DISABLE_PID_VALIDATION` 与 `SetImportPid` 白名单模式不应混用；既然 export 已经禁用 PID validation，就应跳过 `aclrtIpcMemSetImportPid`。判据：步骤 3 的 `[comm rank 0] alloc_domain: SetImportPid -> 207006`。
- ❌ 假设「162 之前没遇到 = 162 的容器 PID 和 host PID 一致」 → 证伪：更准确的说法是 162 之前没遇到，可能是运行形态、容器镜像、device plugin / driver mount、代码路径或 simpler commit 组合不同，使 `SetImportPid` 路径没有失败；也可能是 162 的历史验证没有覆盖这次失败的 precise dynamic-domain path。不能简单归因于 PID 一致性。
- ❌ 假设「driver/firmware 不对导致 507899」 → 证伪（本场景）：0234 已升到 driver 25.5.2 + firmware 7.8.0.7.220，裸 ACL probe `import_rc=0` 通过。非三件套问题（参 [`01-multirank-ipc-507899-507018.md`](01-multirank-ipc-507899-507018.md)），是 simpler 白名单路径在 forked `chip_process` 场景下的可靠性问题。

---

## 6. 如何避免（Prevention）

### 铁律

- **容器化 forked worker 环境下统一用 `DISABLE_PID_VALIDATION + skip SetImportPid`**：a2a3 onboard IPC window export 统一改成 `ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION`，并在该模式下跳过 `aclrtIpcMemSetImportPid`。import 侧继续用 `ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS`。
- **不要把 507899 / 207006 一律归因为三件套**：先跑裸 ACL probe（`run_p1_ipc_multicard_0234.sh`）确认底层 capability 通不通。通 → 白名单路径问题（本文）；不通 → 三件套问题（[`01-multirank-ipc-507899-507018.md`](01-multirank-ipc-507899-507018.md)）。
- **commit message 不要写"fix PID namespace mismatch"**：应写 `avoid brittle aclrtIpcMemSetImportPid whitelist path in forked chip_process deployments`。

### 早期识别信号

- 任何 `domain_alloc_via_ipc: ImportByKey(...) -> 507899` 但裸 ACL probe 通过 → 立刻怀疑 `SetImportPid` 白名单路径。
- 任何改了 export flag 为 `DISABLE_PID_VALIDATION` 后出现 `SetImportPid -> 207006` → 立刻跳过 `aclrtIpcMemSetImportPid`（两种模式不应混用）。

### 后续建议

1. 在 162 上用相同 commit 跑同一条 L3 allreduce，确认无回归；
2. 如需更严谨，补裸 ACL matrix probe：
   - `export=DEFAULT + SetImportPid + import=ENABLE_PEER_ACCESS`；
   - `export=DEFAULT + no SetImportPid + import=ENABLE_PEER_ACCESS`；
   - `export=DISABLE_PID_VALIDATION + no SetImportPid + import=ENABLE_PEER_ACCESS`；
3. 将这组 probe 或 L3 allreduce 纳入 Phase 16 deployment smoke。

### 相关约束的落点

- [`01-multirank-ipc-507899-507018.md`](01-multirank-ipc-507899-507018.md)（三件套主问题，区分判据：裸 probe 是否通过）
- [`../deployment/phase16-three-pillars.md`](../deployment/phase16-three-pillars.md)（三件套版本绑定 spec）
