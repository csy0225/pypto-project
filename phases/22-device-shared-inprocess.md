# Phase 22 — 设备共享 / in-process PyPTO 集成（device-shared path）

> **状态**：设计中（2026-06-29 起）。这是通向"全模型 PyPTO"的主线——一次性
> 解锁 attention、MoE-routed，并顺带修掉 tail 的 warm-ChipWorker bug。
> 前置工作（已完成）：dense MLP(0-2) + MoE 共享专家(3-44) 已用 host-worker 桥
> 在线上跑真实 kernel（见 STATUS.md / PYPTO_DENSE_MLP_E2E_REPORT.md）。

## 1. 目标

让 PyPTO 真实 @pl kernel **直接在 vLLM 的 device 张量上原地计算**（共享设备指针，
无 host round-trip）。这解决三件当前 host-worker 桥做不到的事：

| 层 | host-worker 桥为何不行 | 设备共享路径如何解决 |
|---|---|---|
| **attention** | 读写 vLLM paged KV cache（torch_npu device mem），独立进程无法寻址 | kernel 直接拿 KV cache 的 device 指针读写 |
| **MoE routed** | 需要跨 rank EP all-to-all，per-rank host worker 做不到 | 复用 vLLM tp/ep_group 通信，或 pypto 在共享 device 上做 collective |
| **tail（已知 warm bug）** | warm ChipWorker 多 kernel 复用 buffer sizing 错 | 直接 device 指针 + 正确 context，绕开 warm 复用缺陷 |
| **性能（所有层）** | 每 token d2h→UDS→h2d，~2.6 tps | 零拷贝，device 上直接算 |

## 2. 核心技术障碍

PyPTO runtime 用 **fork 的 `chip_process` 子进程**执行 kernel，每个子进程
`aclrtSetDevice` 建立**自己的 ACL context**。vLLM 的张量（KV cache、hidden）在
**主进程 torch_npu 的 context** 里分配。**ACL device 指针是 context-specific 的**
——torch_npu context 的指针在 pypto chip_process 的 context 里直接用是非法的。

所以"设备共享"的本质 = **跨 context（且很可能跨进程）共享 device 内存**。

## 3. 三个架构选项

### 选项 A：in-process 同 context kernel launch（最干净，最难）
pypto kernel 在 **vLLM worker 的同一进程、同一 ACL context** 内执行（不 fork
chip_process），直接用 torch_npu 张量的 `data_ptr()`。
- 需要：(a) pypto 可在容器内 import + 运行（见 §5 build 前置）；(b) pypto runtime
  提供"在当前 context 同步执行已编译 program"的模式——但 pypto 程序是
  `host_orch`（AICPU 编排）+ 多 AICore kernel，不是单个 `aclrtLaunchKernel`，
  当前 runtime 总是 fork chip_process。需要 pypto core 支持 in-context 执行。
- 评估：最理想（真正 in-process），但要 pypto runtime 改造，工作量最大。

### 选项 B：device-IPC 句柄交换（复用 Phase 16 机制）★ 推荐起点
vLLM 把 KV cache / hidden 的 device buffer 通过 **`aclrtIpcMemGetExportKey`** 导出
句柄 → 传给 pypto worker → chip_process `aclrtIpcMemImportByKey` 导入 → 在子进程
context 里得到合法指针 → kernel 直接读写。**0162 已具备前提**：driver 25.5.2 +
firmware 7.8.0.7.220 把 `support_shmem_map_exbus` cap 打开（Phase 16 为跨卡 IPC 做
的，同卡 IPC 同样适用）。
- 需要：torch_npu 侧导出 IPC handle（ctypes 调 libascendcl），pypto runtime 侧
  接受"外部 device 指针"作为 kernel 参数（DeviceTensor 已有 child_memory 概念，
  可能可复用——见 `pypto/runtime/device_tensor.py`）。
- 评估：进程模型不变（pypto 仍 fork chip_process），只加一层 device-mem IPC；
  与现有 host-worker 架构最兼容；**作为第一步验证最划算**。

### 选项 C：同卡 aclrtIpcMem，保持现有 worker 进程
就是选项 B 的"保持独立 host worker"变体——worker 仍是独立进程，但 hidden/KV/输出
走 device IPC 而非 host round-trip。
- 评估：增量最小（在现有 host-worker 桥上把数据通路从 host 换成 device-IPC）。
  但 attention 的 KV 在容器 torch_npu context，worker 在 host context，仍需 §4 探针。

## 4. 决定性可行性探针（按顺序，每步 gate 下一步）

1. **P1 — device-IPC 同卡可行性**：两个进程都 `aclrtSetDevice(8)`，进程 A
   `aclrtMalloc` + 写已知值 + `aclrtIpcMemGetExportKey`；进程 B `aclrtIpcMemImportByKey`
   + d2h 读回，验证值一致。用 ctypes 直调 `libascendcl`。**这是整条路的地基**——
   不通则选项 A/B/C 全黄，只能继续 host-worker。
   - ✅ **PASS（2026-06-29）**：ctypes 探针 `/tmp/p1_ipc_probe.py` 在 card 8（与 live
     vLLM rank0 + pypto worker 同卡共存）跑通：export 用 `EXPORT_FLAG_DISABLE_PID_VALIDATION`
     (0x1) 免 PID 白名单；import 用 `IMPORT_FLAG_DEFAULT`（同卡无需 ENABLE_PEER_ACCESS）；
     跨进程跨 context 读回 `got==exp`（[0,1,2,3,4...]）完全一致。**设备共享地基成立**。
     ACL IPC API：`aclrtIpcMemGetExportKey(devPtr,size,key,len,flags)` /
     `aclrtIpcMemImportByKey(&devPtr,key,flags)` / `aclrtIpcMemSetImportPid` /
     `aclrtIpcMemClose`（见 `acl/acl_rt.h`，cann-9.0.0 non-GA）。
     本次 `pypto-ascend` / 0234 容器的实际环境变量快照见
     [`22-device-shared-inprocess-p1-env-0234.md`](22-device-shared-inprocess-p1-env-0234.md)。

2. **P2 — torch_npu 张量导出**：在容器内对一个 `torch.ones(...,device=npu)` 张量，
   用 ctypes 拿 `data_ptr()` + `aclrtIpcMemGetExportKey` 导出；外部进程导入读回。
   验证 vLLM 的 device 张量能被外部 context 共享。
   - ✅ **PASS（2026-06-29）**：容器内 torch_npu `torch.arange(...,device=npu:0)`
     （`ASCEND_RT_VISIBLE_DEVICES=8`→物理卡 8）`data_ptr` 直接 `aclrtIpcMemGetExportKey`
     **rc=0**（torch_npu caching-allocator 的内存可 IPC 导出）；host 独立进程
     `aclrtSetDevice(8)` + `ImportByKey` + D2H 读回 `got==exp` 一致。**含义：vLLM 真实
     device 张量（含 paged KV cache）可零拷贝共享给 pypto**——option B 地基（含 attention
     KV 可达）成立。probe：`/logs/p2_probe.py`（共享 mount，container export / host import）。
3. **P3 — pypto 容器内运行**（§5）：host-built pypto（CANN non-GA）能否在容器
   CANN beta.1 下 import + 跑通 hello_world kernel？不行则需在容器内重编 pypto。
4. **P4 — kernel 吃外部 device 指针**：pypto kernel 以导入的 device 指针作为输入/
   输出运行，对比 golden。验证 DeviceTensor/child_memory 路径能接 IPC 指针。
   - 🔧 **P4 = option B 的第一个实现步，非纯探针（2026-06-29 查清）**。关键约束：pypto
     kernel 跑在 fork 的 `chip_process`（独立 context），**父进程 import 的 IPC 指针在子
     进程 context 里非法**——IPC import 必须发生在 chip_process 内。现状：
     - ✅ 现成：`DeviceTensor(data_ptr, shape, dtype)` 公开构造器 + runner 的
       `device_tensor_to_continuous`（child_memory，device-resident，不 h2d）→ "kernel
       吃 device 指针"这半边已具备（qwen serving runner 已在用 `worker.malloc`+WorkerTensor）。
     - ❌ **唯一缺口**：ChipWorker/simpler 有 `malloc/free/copy_to/copy_from`，**没有
       `import_ipc`**——没有在 chip_process context 里 `aclrtIpcMemImportByKey` 的钩子。
     - **要加的接口**：`ChipWorker.import_ipc(key) -> dev_ptr`（与 `malloc` 同层，转发到
       forked chip_process 的 simpler worker / host_runtime.so，在子进程 context 内
       import）；返回的 ptr 即可包成 `DeviceTensor` 作 kernel 参数。属 pypto-core
       （C++/simpler）改动，多日任务，是 option B 桥落地的起点。
   - ✅ **PASS（2026-06-29）—— 且无需 C++ 改动！** 实测推翻了"需改 simpler C++"的预判：
     **simpler L2 不 fork**（`run()` 直接 `self._chip_worker._run_slot(...)`，fork 只在
     L3+ 的 `_start_hierarchical`），所以 **L2 kernel 在调用进程的 context 内执行**。因此
     在 worker 进程里用**纯 ctypes** `aclrtIpcMemImportByKey` import 得到的 ptr 对 kernel
     **直接合法**。P4 probe（`/tmp/p4_probe.py`）：export 进程把 hidden[16,4096] bf16 写进
     device buffer 导出 key；worker 进程 ctypes import（rc=0）→ `DeviceTensor(ptr,[16,4096],
     bf16)` → 跑**真实 `dense_swiglu_perrank` kernel**（hidden=IPC 指针，weights=torch）→
     输出对 golden **`ok=True bad_ratio=0.0000`**。
     **含义**：`import_ipc` 只是一个**薄 Python helper**（ctypes import + `DeviceTensor`），
     **不需要动 pypto-core C++**（至少 L2 路径）。**option B 端到端成立**（P1+P2+P4 全绿）：
     vLLM 导出 device 张量（含 KV cache）→ pypto kernel 零拷贝 import + 计算正确。

5. **P5 — 零拷贝 dense（in+out 都走 device IPC）+ 两条硬约束（2026-06-29）**：
   - ⚠️ **约束 A：ACL 一个内存块只能 export 一次**。实测 diag：torch_npu allocator 把两个
     同尺寸 bf16 张量打进**同一块**（`a.ptr=...200000`, `b.ptr=...220200`, delta=131584），
     `export(a)`=rc0、`export(b)`=**507899**（同块二次导出失败）。**修法**：每块只导一个
     base key，子张量用 **offset** 寻址（`DeviceTensor(base+off, shape, dtype)`）。**KV cache
     天然是一整块** → 一个 key + page offset 即可，正合 attention。
   - ⚠️ **约束 B：IPC handle 生命周期**。探针从不调 `aclrtIpcMemClose`，反复跑（~10 次）
     在 card 8 累积泄漏 exbus handle，导致后续 export 间歇性 507899。**生产实现必须**
     export 后 `aclrtIpcMemClose`（或复用长生命周期 key）。这是探针卫生问题，非机制缺陷。
   - **结论**：零拷贝 dense 机制由 P1/P2/P4 证明可行；落地需 (1) 每块一 key + offset，
     (2) 正确的 handle close 生命周期。P5 一次性 demo 受上面两条 + 探针泄漏影响呈间歇性，
     不再纠缠；生产 patch 在 vLLM 干净状态下按 (1)(2) 实现即可。

P1+P2 通过即证明"设备共享"地基成立；P3 决定 in-process(A) 还是跨进程 device-IPC(B/C)；
P4 是端到端 kernel 验证。

6. **P6 — attention 的 fork 边界 + fork-child IPC import 可行性（2026-06-29）**：
   - **静态发现（先于写代码）**：dense/shared 能用纯 Python ctypes-IPC，是因为它们是
     **L2 单芯片 `@pl.jit`**（不 fork，kernel 在调用进程 ACL context 跑，父进程 import 的
     IPC 指针合法）。但 **`attention_full` 是 Wave-2 三层 L3 HOST-Orchestrator**
     （`host_orch` @ `attention_full.py:909`），runner 走 **fork `chip_process`** 派发
     （`golden/runner.py` `_try_l3_dispatch`，非 in-process `execute_compiled`）。ACL IPC
     映射 per-context → **父进程 import 的 KV 指针在 fork 出的子进程里非法**。L2 那个轻量
     helper trick 对 attention 是死路。
   - **KV 布局本身 OK**：每 rank `KV_HEADS_LOCAL=1`，vLLM `(2,num_blocks,block_size,1,128)`
     与 pypto `k_cache/v_cache [4096,128]` 天然一致（head 维 singleton），不需转置。唯一拦路
     是 fork 边界。
   - ✅ **PASS（2026-06-29）—— fork-child IPC import 经验证可行**：探针
     `/logs/p6_fork_ipc.py` 在 card 8 复刻 chip_process 模型：parent `setDevice+malloc+写
     [0..N)+ExportKey`；`os.fork()` 出 child，child `aclrtSetDevice(8)` 建自己的 context →
     (a) 直接用 parent 的 device ptr `aclrtMemcpy` D2H **rc=0 但读出全 0**（fork COW 继承了
     VA range 但指向 child context 里的错/零物理内存——正是 context-specific 指针的坑）；
     (b) `aclrtIpcMemImportByKey(key)` **rc=0** → D2H 读回 `[0,1,2,3,4]` **完全一致**。
   - **含义**：**option B 经验证成立**——只要在 simpler 的 `chip_process` 里加
     `import_ipc`（在子进程 context 内 `aclrtIpcMemImportByKey`），forked child 就能拿到
     vLLM 导出的 KV cache 的合法指针；且**必须**在 child 内 import（直接传父指针读到垃圾）。
     这是 attention 设备共享落地的去风险结论：`ChipWorker.import_ipc`-in-chip_process 钩子
     （C++/simpler 层，多日工作）是下一步。

7. **import_ipc 实现 + P7 真机验证 ✅（2026-06-29）—— option B 主干打通**：
   - **实现（跨层，已编译 + 安装）**：simpler runtime 加 `import_ipc` 全链路——
     `worker_manager.h/.cpp`（opcode `CTRL_IMPORT_IPC=12` + `control_import_ipc`：key_len→ARG0、
     key bytes→`MAILBOX_OFF_ARGS`、forked child 在 `_CTRL_IMPORT_IPC` 分支用 ctypes
     `aclrtIpcMemImportByKey` import）、`orchestrator.h/.cpp`（`Orchestrator::import_ipc`）、
     `worker_bind.h`（nanobind `_Orchestrator.import_ipc`）、Python `worker.py`/`orchestrator.py`/
     `distributed_runner.py`（`DistributedWorker.import_ipc`）。`pip install -e .` 编过，live
     8001 worker 不受影响（持已加载 .so）。**纯 Python child 侧 import（不需改 host_runtime.so）**。
   - ✅ **P7 PASS**：`_stage_p7_import_ipc_validate.py` 在 card 8（窗口内停 8001 腾卡）跑通：
     exporter ctypes 写 `[128,128]` fp32 pattern(`i%97`) + export key；worker 编译 trivial L3
     decode kernel → `DistributedWorker([decode])`（fork chip child pid 2461531 dev8）→
     `rt.import_ipc(key)` → `ptr=0x12c1c0000000` → `DeviceTensor`(child_memory) → 跑 decode →
     `ok=True bad_ratio=0.0000`，sample `[0,1,2,3,4]` 完全一致。
   - **含义**：`import_ipc` C++ 全链路在设备上**端到端验证通过**（DistributedWorker→Orchestrator→
     C++ mailbox→forked child ctypes import→合法 ptr→真 kernel 经 child_memory 读对）。
     option B 主干（device-shared 原语）落地。**child_memory** 是现成 arg 路径，KV 跨 dispatch
     复用也现成（`multi_program_kv_cache` 例）——attention 只差 wiring：vLLM 侧
     `export_kv_block_key`（一 block 一 key + offset 寻址 K/V，见 `_stage_attention_wiring.py`）
     + worker 侧 `import_kv_cache` + `attention_full` 20-arg 组装 + socket `attn` op。下一步在
     新窗口接一个 attention 层端到端对 golden。

8. **一层 attention 设备共享 e2e ✅ PASS（2026-06-29）—— option B 端到端落地。**
   - `_stage_attn_e2e.py`：exporter ctypes 零初始化 `(2,4096,128)` bf16 KV 块 + export key；worker
     `apply_tp1_patch` → `select_decode_layer(0)`（full_dense = attention_full + dense MLP，L3 HOST-orch
     fork chip child）→ `ir.compile(..., distributed_config=DistributedConfig(device_ids=[8]))` →
     `DistributedWorker([compiled])` → `rt.import_ipc(key)` → `k_dt/v_dt = DeviceTensor(base / base+plane,
     (1,4096,128), bf16)` → 按 `compiled._get_metadata()` 参数序组装（KV=DeviceTensor，其余 shared-mem host
     tensor，layer_idx scalar）→ `rt.run(compiled, *ordered)`。**`ok=True bad_ratio=0.0000`** vs golden
     `_torch_attn_no_gate(slim TP1_KV=1/NH=Q_PER_KV_FULL) + _torch_dense_mlp`。card 8（窗口内停 8001，
     baseline 8000 全程健康）。
   - **唯一缺口 = `DeviceTensor.__getitem__`**：生成的 L3 `host_orch.py` 对每个输入做 per-rank Python 下标
     `tensors["k_cache__ssa_v0"][r_idx, 0:4096, 0:128]`，再 `make_tensor_arg`→`device_tensor_to_continuous`
     （后者早已支持 DeviceTensor）。P7 的 trivial host_orch 不切片所以没暴露。新增 `__getitem__` 返回连续子视图
     DeviceTensor（row-major offset ptr + drop int 维 + resize slice 维；非连续内层 slice 报错）。纯 Python，
     pypto -e 安装无需重编，rsync 到 0162。
   - **含义**：option B（device-mem IPC）**attention 端到端成立**——vLLM 导出的 KV 块零拷贝喂给 forked chip
     child 的真实 attention kernel，原地读写 + 对 golden。下一步 = 接入 live vLLM（worker `attn` op 持驻留 KV
     DeviceTensor 跨 decode step，vLLM 侧 patch 导出 paged-KV block key 并路由 attention_full）。

## 5. Build 前置：pypto 能否在容器内运行

- 容器：CANN 9.0.0-**beta.1** + torch_npu 2.9，无 pypto。
- host：pypto 全栈，.so 编译于 CANN 9.0.0-**non-GA**。
- 探针 P3 决定：
  - 若 non-GA .so 能在 beta.1 容器加载运行 → 选项 A（in-process）可行，mount + PYTHONPATH/LD_LIBRARY_PATH 即可。
  - 若 ABI 不兼容 → 需在容器内按 beta.1 重编 pypto/simpler/PTOAS/pto-isa（一次性，重）。
- ✅ **P3 step1 PASS（2026-06-29）—— 链接层 ABI 兼容**：把 host onboard
  `libhost_runtime.so`（编译于 `/mnt/persist/Ascend/cann-9.0.0/cann-9.0.0` non-GA）拷进
  `/logs`，在 w8a8 容器内 `ldd` → **全部 CANN 依赖（libascendcl / libruntime / libhcomm /
  libhccl_* / libplatform / liberror_manager …）由容器 CANN 9.0.0-beta.1 干净解析，0 个
  "not found"**。DT_NEEDED = libruntime / libascendcl / libhcomm / libstdc++ / libgcc_s / libc。
  **含义**：non-GA 与 beta.1 的 CANN ABI 链接层兼容 → **option A 的"pypto 跑进容器"链接层不需
  重编**，只需 mount host workspace + PYTHONPATH/LD_LIBRARY_PATH。剩余 P3 风险只是**行为层**
  （kernel 真跑对不对），需腾卡/窗口跑 hello_world + 一个真 kernel 确认；step1 已是最大的 gate。

## 6. 各层落地（地基通过后）

- **tail**：选项 A/B 下用正确 context + device 指针执行 rms_lm_head，绕开 warm
  ChipWorker buffer-sizing bug；输出 per-rank vocab shard，vLLM all_gather（已写好）。
- **attention**：从 vLLM-Ascend attention backend（`vllm_ascend/attention/attention_v1.py`）
  + `model_runner_v1.py` 的 `_get_block_table_and_slot_mapping` 拿 block_table /
  slot_mapping / KV cache device 指针，喂给 pypto attention kernel（`attention_full.py`/
  `attention_swa.py`）。KV 原地读写（device IPC 指针）。这是最大的一块。
- **MoE routed**：pypto MoE kernel（`decode_layer.py` 的 DecodeLayerMoE，507018 已修）
  做 gate→dispatch→expert→combine；EP 通信用 vLLM 的 ep_group 或 pypto 自带 simpler
  comm（Phase 16 windows）。需多卡 DistributedConfig EP group，非 per-rank worker。

## 7. 工作量与边界

- P1-P4 探针：~3-5 天（含腾卡/维护窗口协调）。
- 选项 B device-IPC 桥（tail + dense/shared 零拷贝化）：~2-3 周。
- attention 设备共享：~4-6 周（KV ABI 最复杂）。
- MoE routed EP：~3-4 周（多卡 EP runner）。
- 选项 A 真 in-process（pypto runtime 改造）：额外，需 pypto core 投入。

**第一步行动**：在一张可用卡（或维护窗口）跑 P1（device-IPC 同卡 ctypes 探针）。
通过则按选项 B 做 tail 零拷贝化作为端到端验证，再推进 attention。

## 8. 方案 A vs B 判定（2026-06-29，团队讨论后）

团队提出把选项 A 具体化为 **"share external context" 接口**：让 pypto 直接复用
vLLM/torch_npu 已建好的 ACL context。判定如下：

- **ACL context 是进程内私有**，无法跨进程共享。所以 "share external context" ⟹
  pypto kernel **必须 in-process（与 vLLM worker 同进程同 context）执行**。一旦同
  context，torch_npu 的 `data_ptr()` 天然合法，**无需 device-IPC**——这是最干净的终态。
- **障碍在 pypto/simpler runtime**：simpler 是**强 fork 架构**（`worker.py`：`init()`
  "only forks chip children"；chip_process "Runs in forked child process, loads
  host_runtime.so in own address space"，各自 `aclrtSetDevice` 建独立 context；
  HeapRing fork 前 mmap、scheduler/WorkerThreads fork 后在子进程起）。要复用外部
  context 就要新增**"非 fork、attach 外部 context、in-thread 执行"模式**——属 simpler
  runtime 较大改造，**不是薄接口**。vLLM 侧几乎为 0（context 已建好，只是别 fork）。
  （`worker.chip_contexts` 是 L3 给 sub-worker 传 context，仍在 fork 模型内，不可直接用。）

| | 共享什么 | 改哪里 | 代价 | 定位 |
|---|---|---|---|---|
| **A：share external context** | context（指针天然合法，真 in-process 零拷贝） | pypto/simpler 大改（non-fork in-thread + 注入 ctx + kernel 吃外部指针） | 高，需 pypto core | **终态目标** |
| **B：device-mem IPC** | 仅内存（aclrtIpcMem，沿用 fork 模型） | 加一层 device-mem 句柄交换，runtime 不重写 | 中 | **先行验证** |

**决策**：B 先行——不碰 fork 根基，更快拿到"零拷贝 + kernel 直接读写 vLLM device
张量（含 attention KV）"的端到端验证；同时并行让 pypto core 评估 A 的 non-fork
in-thread 模式。A 通了即切换，B 的 IPC 契约正好是 A 的过渡缝。先跑 P1 验证 B 地基。
