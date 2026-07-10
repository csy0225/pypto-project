# Co-tenancy: run a pypto DistributedWorker on the same NPUs as vLLM (`SIMPLER_COMM_NO_HCCL`)

> **给另一个 thread / 任何要让 pypto worker 与 vLLM 同卡共存的场景。**
> 2026-07-11 定位 + device 验证于 0162 cards 8-15。详细根因见
> memory `project_g4_cotenancy_hccl_conflict.md` + [`../blockers.md`](../blockers.md) §G4。

## 问题

一个 pypto `DistributedWorker`（fork 自己的 chip_process、做自己的 TP all_reduce / EP a2a
collective，例如 whole-decode worker `_stage_whole_decode_run.py --worker`）与 vLLM **同卡**
运行时，在第一个 dispatch 的 `orch.allocate_domain → _ensure_comm_base` 挂：

```
comm_hccl.cpp:301 [comm rank 0..7] HcclCommInitRootInfo failed: 7
worker.py:3620 _ensure_comm_base failed on 8/8 chips; control_comm_init failed / comm_init failed
```

- standalone（无 vLLM）同一 worker `comm_init` 成功、rc=0。
- 纯 **HCCL 共存冲突**：vLLM 已在这些卡建了一个 HCCL comm world，worker 的第二个
  `HcclCommInitRootInfo`（不同进程、同一批 NPU）返回 error 7（HCCL_E_UNAVAIL）。
- distinct `HCCL_IF_BASE_PORT` **无效**（device 级资源冲突，非 socket/port）。
- 对比：per-op `_MlpService`（phase 24）能同卡共存，是因为它 **worker 侧无 collective**
  （per-rank partial，all_reduce 由 vLLM 做）→ 不建 HCCL world。做 collective 的 worker 才撞。

## 修法（root cause，非绕过）

simpler 的 HCCL control comm 其实是 **vestigial**：只用 `HcclGetRootInfo` /
`HcclCommInitRootInfo` / `HcclBarrier` / `HcclCommDestroy`，**没有** HcclAllReduce/Send/Recv。
所有跨卡数据搬运 + comm-domain 建立**已经**走 `file_barrier`（文件标记 cross-rank barrier）+
IPC peer-access（`aclrtIpcMemImportByKey` + `aclrtDeviceEnablePeerAccess`）。唯一的 HCCL-comm
消费者 `comm_barrier` 在 dispatch 路径**无调用者**（生成的 `host_orch.py` 从不调）。

→ env-gated `SIMPLER_COMM_NO_HCCL=1`：`comm_init` 跳过 HcclGetRootInfo / HcclCommInitRootInfo
（保留 run_token 文件写入 + `file_barrier("rootinfo_ready")` 做同步），`hccl_comm` 留 nullptr；
`comm_alloc_domain_windows` 放宽 `hccl_comm==nullptr` 检查；`comm_barrier` 在 null comm 上 no-op。
**flag 未设 = 原 HCCL 路径，字节级不变（安全）。**

- patch：`runtime/src/a2a3/platform/onboard/host/comm_hccl.cpp`（5 处 env-gated anchors，
  备份 `.bak_nohccl`）。
- 重编：`python -m simpler_setup.build_runtimes --platforms a2a3`（重出 `comm_hccl.cpp.o` +
  `libhost_runtime.so`）。

## 怎么用（另一个 thread 照做）

1. 确保用的是**打了 patch 并重编过**的 a2a3 simpler runtime（0162 共享 workspace 已重编；
   若在别处，先 apply patch + rebuild）。
2. 给你的 pypto worker 进程设 env：
   ```bash
   export SIMPLER_COMM_NO_HCCL=1
   ```
3. 恢复顺序照旧（先起 vLLM 等 HCCL init 完成 / health=200，再起 pypto worker）——见
   [`troubleshooting-8001-pypto-bridge.md`](troubleshooting-8001-pypto-bridge.md) 症状 1。
4. 这不改数值路径（TP all_reduce / EP a2a 仍走 IPC，只是 bootstrap/barrier 不再用 HCCL）。

**适用范围**：任何 pypto+vLLM 同卡 co-tenancy，不限 whole-decode。

## device 验证（0162 cards 8-15，2026-07-11）

- idle vLLM 8001（enforce-eager，TP=8，util 0.5，health=200）+ whole-decode worker
  `SIMPLER_COMM_NO_HCCL=1`（synthetic + real W8A8）→ **PREPARE OK、all steps dispatched、rc=0**、
  无 `HcclCommInitRootInfo failed`、无 507018、8001 health=200 前后不变。
- real-weight torch-ref：**L0 full_dense PASS 1.000**（max|diff|=0.75）——co-resident 下
  TP all_reduce 数值精确。
- standalone HCCL vs NO_HCCL：L0 full_dense **bit-identical**（30.875）。
- ⚠ 边界：swa / MoE 层的全层数值 token-exactness 仍走 **live A/B（real KV）** 定论——
  offline synthetic + dummy-KV 会 residual 爆 nan（input-independent，HCCL 路径同样，
  非 NO_HCCL 引入），不是可信 oracle（项目约定）。
