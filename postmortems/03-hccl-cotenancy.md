# 专项：pypto worker 与 vLLM 同卡共存时 HCCL control comm 冲突（`HcclCommInitRootInfo failed: 7`）

| 字段 | 值 |
|------|----|
| **子系统** | vllm-pypto / deployment |
| **error signature** | `HcclCommInitRootInfo failed: 7`（HCCL_E_UNAVAIL） |
| **首次出现** | 2026-07-11 |
| **状态** | ✅ 已解 |
| **相关 skill / doc** | memory `project_g4_cotenancy_hccl_conflict.md` / [`../blockers.md`](../blockers.md) §G4 / [`../design/vllm-pypto/`](../design/vllm-pypto/) / [`../deployment/phase16-three-pillars.md`](../deployment/phase16-three-pillars.md) |

## 1. 背景（Background）

场景：把一个 pypto `DistributedWorker`（fork 自己的 chip_process、做自己的 TP all_reduce / EP a2a collective，例如 whole-decode worker `_stage_whole_decode_run.py --worker`）与 vLLM **同卡**跑——即 vLLM 8001 已在 cards 0-7（或任意 8 卡）上建好 HCCL comm world 并在跑，pypto worker 想在同批 NPU 上拉起自己的 collective。

机器：0162（cards 8-15，driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1 三剑合璧已在册，见 [`../deployment/phase16-three-pillars.md`](../deployment/phase16-three-pillars.md)）。

命令：`_stage_whole_decode_run.py --worker`（或任何建自己 HCCL world 的 pypto worker），与 vLLM 8001 enforce-eager TP=8 同卡共存。

目标：让 pypto whole-decode worker 与 live vLLM 同卡共存（G4 co-tenancy），为后续 G5 live A/B 铺路。

## 2. 现象（Symptom）

worker 在第一个 dispatch 的 `orch.allocate_domain → _ensure_comm_base` 挂：

```
comm_hccl.cpp:301 [comm rank 0..7] HcclCommInitRootInfo failed: 7
worker.py:3620 _ensure_comm_base failed on 8/8 chips; control_comm_init failed / comm_init failed
```

- standalone（无 vLLM）同一 worker `comm_init` 成功、rc=0。
- 卡在 bootstrap：8 rank 全失败，error 7 = HCCL_E_UNAVAIL。
- 与 vLLM 8001 已建好的 HCCL world 同批 NPU 冲突。

## 3. 根因（Root Cause）

纯 **HCCL 共存冲突**：vLLM 已在这些卡建了一个 HCCL comm world，worker 的第二个 `HcclCommInitRootInfo`（不同进程、同一批 NPU）返回 error 7。

证据链：
1. standalone 同一 worker `comm_init` 成功 → 非 worker 代码本身的问题。
2. `distinct HCCL_IF_BASE_PORT` 无效 → device 级资源冲突，非 socket/port 问题。
3. 对比 per-op `_MlpService`（phase 24）能同卡共存，是因为它 **worker 侧无 collective**（per-rank partial，all_reduce 由 vLLM 做）→ 不建 HCCL world。做 collective 的 worker 才撞。

深挖 vestigial 性质：simpler 的 HCCL control comm 其实**只用** `HcclGetRootInfo` / `HcclCommInitRootInfo` / `HcclBarrier` / `HcclCommDestroy`，**没有** HcclAllReduce/Send/Recv。所有跨卡数据搬运 + comm-domain 建立**已经**走 `file_barrier`（文件标记 cross-rank barrier）+ IPC peer-access（`aclrtIpcMemImportByKey` + `aclrtDeviceEnablePeerAccess`）。唯一的 HCCL-comm 消费者 `comm_barrier` 在 dispatch 路径**无调用者**（生成的 `host_orch.py` 从不调）。

→ HCCL control comm 是历史遗留的 bootstrap/barrier 通道，与真实数据通路解耦，可以安全旁路。

## 4. 如何解决（Fix）

env-gated `SIMPLER_COMM_NO_HCCL=1`：`comm_init` 跳过 `HcclGetRootInfo` / `HcclCommInitRootInfo`（保留 run_token 文件写入 + `file_barrier("rootinfo_ready")` 做同步），`hccl_comm` 留 nullptr；`comm_alloc_domain_windows` 放宽 `hccl_comm==nullptr` 检查；`comm_barrier` 在 null comm 上 no-op。**flag 未设 = 原 HCCL 路径，字节级不变（安全）。**

patch：`runtime/src/a2a3/platform/onboard/host/comm_hccl.cpp`（5 处 env-gated anchors，备份 `.bak_nohccl`）。

重编：

```bash
python -m simpler_setup.build_runtimes --platforms a2a3
# 重出 comm_hccl.cpp.o + libhost_runtime.so
```

怎么用（另一个 thread 照做）：

1. 确保用的是**打了 patch 并重编过**的 a2a3 simpler runtime（0162 共享 workspace 已重编；若在别处，先 apply patch + rebuild）。
2. 给你的 pypto worker 进程设 env：
   ```bash
   export SIMPLER_COMM_NO_HCCL=1
   ```
3. 恢复顺序照旧（先起 vLLM 等 HCCL init 完成 / health=200，再起 pypto worker）——见 [`../deployment/troubleshooting-8001-pypto-bridge.md`](11-8001-bridge-live-ops.md) 症状 1。
4. 这不改数值路径（TP all_reduce / EP a2a 仍走 IPC，只是 bootstrap/barrier 不再用 HCCL）。

**适用范围**：任何 pypto+vLLM 同卡 co-tenancy，不限 whole-decode。设计上下文见 [`../design/vllm-pypto/`](../design/vllm-pypto/)。

### device 验证（0162 cards 8-15，2026-07-11）

- idle vLLM 8001（enforce-eager，TP=8，util 0.5，health=200）+ whole-decode worker `SIMPLER_COMM_NO_HCCL=1`（synthetic + real W8A8）→ **PREPARE OK、all steps dispatched、rc=0**、无 `HcclCommInitRootInfo failed`、无 507018、8001 health=200 前后不变。
- real-weight torch-ref：**L0 full_dense PASS 1.000**（max|diff|=0.75）——co-resident 下 TP all_reduce 数值精确。
- standalone HCCL vs NO_HCCL：L0 full_dense **bit-identical**（30.875）。
- ⚠ 边界：swa / MoE 层的全层数值 token-exactness 仍走 **live A/B（real KV）** 定论——offline synthetic + dummy-KV 会 residual 爆 nan（input-independent，HCCL 路径同样，非 NO_HCCL 引入），不是可信 oracle（项目约定）。

### ⚠ 共存 hazard：pypto force-reset 会连累 vLLM（G5 live 必须防）

`DeviceRunner::force_reset_device()`（`device_runner.cpp:609`，`aclrtResetDeviceForce` `:570/622`）是**卡级全局 reset，无 foreign-owner 守护**——pypto worker 一旦 AICore timeout / device poison（`run()` 拒绝 `:199`，finalize force-reset `:696`），会把同卡 vLLM 8001 的 ACL context 一起 nuke。（`AclInitGuard` 尊重外部 ACL owner，但 `aclrtResetDeviceForce` 没有对应守护。）

- **对比 `aclInit`**：process-wide、容忍 `ACL_ERROR_REPEAT_INITIALIZE`（`device_runner.cpp:128-150`），pypto 建自己的 context 与 vLLM 共存无冲突——只有 **force-reset** 是破坏性的。
- **G5 live 缓解**：(1) 确保 pypto dispatch 前 vLLM stream idle（减 halMemCtl 争用 + 507018）；(2) 避免 pypto AICore timeout（force-reset 触发器）；(3) 上游建议：给 reset 加 co-tenancy 守护（`--no-force-reset` / 检测 foreign owner 时跳过 `aclrtResetDeviceForce`），mirror `AclInitGuard`。
- 另：`code 13 = kHalMemCtlEacces`（`host_regs.cpp:114`）= 并发 chip_process bring-up 抢 halMemCtl 串行化窗口的信号（已 retry 3×/50ms）；co-tenancy 下若复现，是 vLLM 争用该窗口，先起 8001 等 idle 可减。

## 5. 走过的弯路（Detours / What We Got Wrong）

- ❌ 假设：用 `distinct HCCL_IF_BASE_PORT` 把 worker 的 HCCL comm world 隔离到不同端口 → 证伪：**device 级资源冲突，非 socket/port**，改端口对 error 7 无影响。
- ❌ 误判：曾以为 simpler 的 HCCL comm 是数据通路必需 → 证伪：grep `comm_barrier` 在生成的 `host_orch.py` 中**无调用者**；数据通路早已走 `file_barrier` + IPC peer-access，HCCL 仅 bootstrap/barrier 用。
- ❌ 误判方向：曾考虑 Phase-22-option-A 式重构（per-op service 化、让 worker 不建 collective）→ 无效/过重：per-op `_MlpService` 能共存恰恰因为不做 collective，但 whole-decode worker 必须做 collective，per-op 化等于推翻设计；env-gated 旁路即可，不必重构。
- ⚠ 早期研究失误：曾基于 stale local `feat/whole-net-n1-fusion` checkout 断言"simpler 无 env escape hatch" → 证伪：authoritative 0162 stepfun/develop 上 flag 存在；**device evidence > stale-branch code reading**。

## 6. 如何避免（Prevention）

- **铁律**：任何 pypto+vLLM 同卡 co-tenancy 线程，worker 进程必须 `export SIMPLER_COMM_NO_HCCL=1`；启动顺序先 vLLM（等 health=200）再 pypto worker。
- **早期识别信号**：worker `comm_init` 阶段报 `HcclCommInitRootInfo failed: 7` + standalone 同 worker rc=0 → 立刻怀疑 HCCL world 冲突，不要再调 port / 重装 CANN。
- **不要把 stale branch 的 code reading 当事实**：研究 simpler 是否有 escape hatch 时，以 0162 stepfun/develop（authoritative）为准，本地 feature 分支可能落后。
- **落点**：`SIMPLER_COMM_NO_HCCL=1` 已写入 co-tenancy runbook（[`../deployment/`](../deployment/)）与 [`../design/vllm-pypto/`](../design/vllm-pypto/)；force-reset 的 co-tenancy 守护待上游（建议 mirror `AclInitGuard`）。
