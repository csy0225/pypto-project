# Phase 23 —— 零拷贝 KV-IPC 集成:step 1-5 验证 + 重制定 plan

> **组件 pin snapshot（验证时）**：2026-07-03，0162 card 8。CANN `9.0.0` non-GA、
> ptoas-bin `v0.45`、pypto/pypto-lib/simpler = 当前 0162 workspace HEAD。
> 验证脚本为 staging 探针（项目根 `/data/chensiyu/hw_project/pypto/_stage_*.py`，
> 未入 sub-repo），落地时再固化。
>
> **本 phase 是什么**：把「PyPTO runtime 通过 device-IPC **零拷贝**接管 vLLM 的
> KV 计算」这条路线,从 step 1 到 step 5 在 device 上逐步验证,给出可行性结论,
> 并据此**重制定** step 6/7 及与 Phase 20-22 的关系。
>
> 关联:[`22-device-shared-inprocess.md`](22-device-shared-inprocess.md)（P1-P8 前置轨）、
> [`../architecture/aclgraph-vs-pypto-execution-model.md`](../architecture/aclgraph-vs-pypto-execution-model.md)、
> [`../deployment/troubleshooting-8001-pypto-bridge.md`](../deployment/troubleshooting-8001-pypto-bridge.md)。

---

## 1. 为什么做这件事（纠偏）

此前进展偏成了「**算子桥接**」：每 rank 一个独立 PyPTO worker 进程,算单算子
per-rank partial,通过 **Unix socket / device-IPC** 桥回 vLLM TP worker。它
(a) 丢了 PyPTO 整栈的**跨层融合**收益,(b) 引入 vLLM 没有的 host round-trip
开销（实测 ~2.6 tps vs baseline ~4.9-9.4）,(c) 一堆 IPC 卡点。

目标口径回正：**PyPTO runtime 接管计算,vLLM 保留 API / 调度 / paged-KV 显存**。
专家给的 7 步路线（零拷贝 IPC）正是达成这一点的正确路径。本 phase 验证它。

---

## 2. Step 1-5 验证结论（全部 0162 device 实测,2026-07-03）

| 步 | 结论 | 证据 / 脚本 |
|---|---|---|
| **1** torch_npu IPC tensor | ✅ torch_npu 有 torch.cuda 级 IPC（`rebuild_npu_tensor` / `storage._share_npu_` / `torch_npu.multiprocessing` / `NPUIPCTypes.cpp`）+ 裸 ACL 双通路;device tensor 导出 rc=0 | `_stage_va_ipc_probe.py`;复用 P2 |
| **1'** 跨进程 VA | ✅ **测量：跨进程 import 的 VA 不同**（exporter `0x12c041…` → importer `0x12c1c0…`）**但 offset 保留**（base+off 读回正确） | `_stage_va_ipc_probe.py` `VA-VERDICT DIFFERENT` |
| **2** 跨进程零拷贝喂 kernel | ✅ import 的 IPC 指针 → `DeviceTensor` → 真 kernel `bad_ratio=0` | 复用 P4/P7 + 今日底座重确认 |
| **3** VA-map + 自动 offset | ✅ 一 key + `DeviceTensor(peer_base, pool_shape)[block]` 自动 offset;多块 kernel 读取全 `bad_ratio=0` | `_stage_vamap_multiblock.py` `VAMAP_MULTIBLOCK_PASS` |
| **4** 全量 kv_cache 映射 | ✅ 45 层 KV 合一 buffer → **1 个 key** → 1 次 import → **90 条 offset map** → **无 per-tensor MemPool → 无 OOM** | `_stage_kvpool_pageattn.py` |
| **5** page_attention 零拷贝 | ✅ 嵌套 offset（层 map + block_table 分页索引）零拷贝喂 kernel,跨层 0/22/44 × 块 0/3/7,K/V 全 `bad_ratio=0` | `_stage_kvpool_pageattn.py` `KVPOOL_PAGEATTN_PASS` |

**关键机制事实（后续所有 KV 共享都靠这条）**：跨进程 `aclrtIpcMemImportByKey`
返回的 VA 与 exporter 不同,但**块内 offset 保留**。所以 KV 池映射 = **一张
per-block base map**:`pypto_ptr = peer_base + offset`,用 `DeviceTensor.__getitem__`
自动算 `data_ptr + offset*elem`,`device_tensor_to_continuous` 包成
`ContinuousTensor(child_memory=True)` 喂 kernel（无 H2D/D2H）。**全程无需改
pypto-core C++**（沿用 P4/P7 的 `DistributedWorker.import_ipc` + `DeviceTensor`）。

---

## 3. 技术解决：IPC 主卡点根因 + 「一 key map」正解

**根因**：此前用「每 tensor 一个 `torch.npu.MemPool`」让 KV 落到 block base
（满足 `aclrtIpcMemGetExportKey` 只在块基址成功的约束,否则 507899）。但每个
MemPool 预留一段 NPU 虚地址,45 层 × K/V = **90 个 pool → `rtReserveMemAddress
out of memory 207001`**,只能撑到 4 层。

**正解（专家路线 step 4）**：vLLM-Ascend 真实 KV 分配点在
`model_runner_v1._allocate_kv_cache_tensors`（per-layer `torch.zeros(k_size,int8)` /
`torch.zeros(v_size,int8)`）。把所有层的 K/V **合并进一个 backing buffer**
（一个块基址 → **一个 export key**）,记录 `map[(layer,K|V)] = 字节 offset`。
PyPTO 侧**一次 import** 整池 → `peer_base` → 用 map 定位任意层任意块。

实测：45 层 45 MiB 池、**1 个 key**、**90 条 map**、**零 OOM**、5 个采样点
（跨层/K/V/块）分页读取全 `bad_ratio=0`。**IPC 主卡点（507899 + 207001）解除。**

---

## 4. 剩余技术问题（指导后续,勿踩重）

| # | 问题 | 性质 | 状态 |
|---|---|---|---|
| A | **live 接线（非可行性）**：把 step-4 patch 接进 live 8001 `_allocate_kv_cache_tensors`（产出「一 buffer + 一 key + map」）+ page_attention 挂进 forward | 工程 | 机制已验证;此前 socket-bridge 已在 live 导出真实 KV + attention decode 层 0-3 `bad_ratio=0`（用会 OOM 的 per-tensor MemPool,换一 key map 即扩到 45 层） |
| B | **head-gate matmul_acc N=16 codegen bug** | pypto codegen | 已本地绕过（worker 端预算 gate,复用 gate_r 槽）;上游诉求见 `pypto-lib/docs/upstream-issues/step3p5-head-gate-matmul-acc-n16-codegen.md` |
| C | **rope-q-pack codegen bug**（ctx>1 乱码） | pypto codegen | 已本地修（逐 head 连续切片重写）;`…/step3p5-rope-q-pack-codegen.md` |
| D | **prefill 用 decode kernel → 空行 NaN** | model 接线 | backend 侧 `seq_len==0` 检测路由 vanilla;真正 prefill kernel 接线是后续 |
| E | **MoE 8 卡 507018 / prefill MoE L1 overflow** | pypto | 见 `blockers.md`;gate 全 PyPTO MoE,dense 层先行不受阻 |
| F | **force_reset_device 需在 hosted 模式关掉** | runtime | simpler `aclrtResetDeviceForce` 会拆 vLLM 的 device context;in-process 路径要注意（当前 out-of-process worker 不受影响） |

---

## 5. 重制定 plan（基于 step 1-5 结论）

**范式确定**：out-of-process PyPTO worker + **device-IPC 零拷贝**（一 key 整池
map）。socket round-trip 版本（算子桥接）**降级为精度 oracle,不再作为生产
路径**。tail / layer_ref monkey-patch 保留作回归基线。

### Phase 24 —— step 6：整层 live 替换（via 一 key map）

| 任务 | 内容 | 准出 |
|---|---|---|
| 24.1 | patch live `_allocate_kv_cache_tensors`：45 层 KV 合一 buffer → 一 key + emit map 表（取代 per-tensor MemPool） | restart 8001 起得来、**无 OOM**、map 文件产出 |
| 24.2 | worker/backend：一次 import 整池 + 建 VA-map；page_attention 走 map + block_table（接 `attention_full`,decode 已验证 `bad_ratio=0`） | 单层 live decode A/B `bad_ratio=0` |
| 24.3 | 扩到**全 45 层 attention**（此前受 OOM 限到 4 层） | 45 层 decode A/B `bad_ratio=0`,零拷贝无 socket round-trip |
| 24.4 | 整层：attention + MLP/MoE 一整个 decode 层走 PyPTO（dense 层先；MoE gate 507018） | 整层 live decode 对齐 vanilla |

估时 ~3-4 周;gate = blocker E（MoE）只影响 MoE 层,dense 层不阻塞。

### Phase 25 —— step 7：真 module 全网 + whole-model orchestration

| 任务 | 内容 |
|---|---|
| 25.1 | in-memory `nn.Module` → PyPTO bundle 权重翻译（落地 `weight_translate` transform plan） |
| 25.2 | wire `Step3p5DecodeFwd.host_orch` **48 层融合**（Wave-3；host 出 loop）—— 这是整栈融合收益的**前提**（见 aclgraph-vs-pypto doc §4） |
| 25.3 | 全网走 PyPTO；精度 L1/L2/L3 gate（复用 Phase 21 harness + dump oracle） |

### Phase 26 —— perf（原 Phase 22）

零拷贝已消除 host round-trip,此时才能测真实 fusion 收益。按 aclgraph-vs-pypto
doc 的 roofline：**甜区 = prefill + 中大 batch decode（flash + MoE comm-compute
重叠）**;纯小 batch decode 收益小（诚实结论,不要拿它当卖点）。

### 与旧 Phase 20-22 的关系

- 旧 **Phase 20**（tail/layer_ref/full monkey-patch）：tail/layer_ref 保留作
  precision oracle;`full` 的 whole-model runner = 本 plan Phase 25.2。
- 旧 **Phase 21**（精度 harness）：不变,作为 Phase 24/25 的在线 gate。
- 旧 **Phase 22**（perf）：= 本 plan Phase 26。

---

## 6. Status

- **step 1-5 ✅ 已验证**（device,2026-07-03）。IPC 主卡点（507899 / 207001 OOM）
  经「一 key 整池 map」正解**解除**。
- **step 6/7 ⏸ 待做** = Phase 24 / 25（上表）。
- 产出脚本：`_stage_va_ipc_probe.py`、`_stage_vamap_multiblock.py`、
  `_stage_kvpool_pageattn.py`（均 0162 `PASS`）。
- 边界：step 4/5 验证的是**真实 KV 布局/规模下的机制**;接进 live 8001 服务
  loop 是 Phase 24 的工程（此前 socket-bridge 已部分打通真实 KV 导出 + decode
  attention `bad_ratio=0`）。
