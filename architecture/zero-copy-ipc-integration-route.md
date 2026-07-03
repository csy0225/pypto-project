# 零拷贝 IPC 集成路线（专家推荐方向）

> **本文档是什么**：技术专家提出的「PyPTO runtime 通过 device-IPC **零拷贝**接管
> vLLM KV 计算」的 7 步集成方向 —— 作为 pypto step3p5 vLLM 集成的**指导路线**。
>
> - **验证记录**（step 1-5 device 实测证据）：[`../phases/23-zero-copy-kv-ipc-validation.md`](../phases/23-zero-copy-kv-ipc-validation.md)
> - **执行 plan**（step 6/7 落地）：Phase 24 / 25 / 26（见 phases/23 §5、[`../phases/README.md`](../phases/README.md)）
> - **执行模型上限对比**：[`aclgraph-vs-pypto-execution-model.md`](aclgraph-vs-pypto-execution-model.md)

---

## 0. 为什么是这条路线

目标：**PyPTO runtime 接管模型计算，vLLM 保留 API 层、请求调度、paged-KV 显存管理、
采样**。此前进展偏成了「算子桥接」（每 rank 独立 PyPTO worker 进程，通过 Unix
socket 桥单算子 partial），丢了整栈融合收益且引入 host round-trip 开销（实测
~2.6 tps vs baseline ~4.9-9.4）。

这条路线用「**vLLM 的 KV 显存 IPC 零拷贝映射进 PyPTO 进程 + kernel launch 时自动
offset**」达成真正的 runtime 接管：PyPTO kernel 直接在 vLLM 的 device KV 张量上
原地读写，无数据拷贝；vLLM 的 paged-attention 调度/显存管理不变。

---

## 1. 专家路线原文（7 步）

> 1. 参考 torch.gpu 的 IPC tensor 能力，验证 torch_npu 的 IPC tensor 能力
> 2. 基于 1，做跨进程传达 tensor 可以无拷贝复用做 kernel 执行的效果
> 3. 基于 2，当前 IPC tensor 在不同进程里的 VA 不一样，做 VA 地址的 map 管理，
>    在 kernel_launch 算子执行时做自动的 offset 偏移转换
> 4. 基于 3，找到 vllm 里所有的 kv_cache table 申请的地方，默认全部 IPC 映射到
>    PyPTO 进程里并建立好 map 表
> 5. 基于 4，做一个 page_attention 算子示例，达成对 KV_cache table 的 tensor 做
>    自动 offset 偏移后可以无冗余 copy 达成 kernel 执行效果
> 6. 基于 5，做整层 layer 替换
> 7. 基于 6，做真 module 模型替换

---

## 2. 每步含义 + 当前状态

| 步 | 含义 | 状态（2026-07-03） |
|---|------|------|
| **1** | torch_npu 是否具备（类 torch.cuda 的）跨进程 IPC tensor 能力 | ✅ **验证通过**：torch_npu 有高层 IPC（`rebuild_npu_tensor` / `storage._share_npu_` / `torch_npu.multiprocessing`）+ 裸 ACL 双通路 |
| **2** | 跨进程传来的 tensor 能无拷贝直接喂 kernel 执行 | ✅ **验证通过**：IPC 指针 → `DeviceTensor` → 真 kernel `bad_ratio=0` |
| **3** | 跨进程 VA 不同 → 建 VA-map，kernel launch 时自动 offset 翻译 | ✅ **验证通过**：跨进程 VA 不同但 **offset 保留**，所以 map = per-block base（`pypto_ptr = peer_base + offset`），`DeviceTensor.__getitem__` 自动算偏移 |
| **4** | 找到 vLLM 所有 kv_cache 申请点，全部 IPC 映射进 PyPTO 并建 map 表 | ✅ **机制验证通过**：真实分配点 = `vllm-ascend model_runner_v1._allocate_kv_cache_tensors`；45 层合一 buffer → **1 key** → 90 条 map → **无 OOM**（取代会 OOM 的 per-tensor MemPool）。live 接线 = Phase 24 |
| **5** | page_attention 算子对 KV_cache table 自动 offset 后无冗余 copy 执行 | ✅ **机制验证通过**：嵌套 offset（层 map + block_table 分页）零拷贝喂 kernel，跨层/K/V/块全 `bad_ratio=0`。完整 attention 数学此前已单独验证（P8 + live decode 层 0-3 `bad_ratio=0`） |
| **6** | 整层 layer 替换 | ⏸ **Phase 24**：把 step-4/5 机制接进 live 8001，整个 decode 层走 PyPTO，扩全 45 层 |
| **7** | 真 module 模型替换 | ⏸ **Phase 25**：真 nn.Module 权重翻译 + `Step3p5DecodeFwd.host_orch` 48 层融合（Wave-3，host 出 loop） |

**结论**：step 1-5 在 0162 device 上逐步验证通过，**IPC 主卡点（子指针导出 507899 +
MemPool VA 预留 OOM 207001）经「一 key 整池 map」正解解除**。这条路线**可行且方向
正确**，step 6/7 是工程落地（非可行性问题）。

---

## 3. 关键机制（一句话）

跨进程 `aclrtIpcMemImportByKey` 返回的 VA 与 exporter 不同，但**块内 offset 保留**。
于是把整个 KV 池一次导出（1 个 key）、一次 import（得 `peer_base`），用
`map[(layer, K|V)] = 字节 offset` + block_table 分页索引组合成
`DeviceTensor(peer_base + layer_off)[block]`，`child_memory=True` 喂 kernel ——
**无 H2D/D2H 拷贝，无需改 pypto-core C++**。

---

## 4. 产出脚本（0162 staging，逐步验证）

| 脚本 | 验证 | 结果 |
|---|---|---|
| `_stage_va_ipc_probe.py` | step 1/3：torch_npu IPC API + 跨进程 VA/offset 测量 | `VA_IPC_PROBE_PASS`（VA 不同、offset 保留） |
| `_stage_vamap_multiblock.py` | step 3：一 key + VA-map 多块自动 offset + kernel | `VAMAP_MULTIBLOCK_PASS` |
| `_stage_kvpool_pageattn.py` | step 4/5：45 层一 key 整池 map + block_table 分页零拷贝 | `KVPOOL_PAGEATTN_PASS` |

> 脚本目前在 pypto workspace root（`/data/chensiyu/hw_project/pypto/_stage_*.py`），
> Phase 24 落地时固化进 sub-repo。
