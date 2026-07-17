# Phase 27 —— N=1 整网融合（单 @pl.program 全 45 层 + tail）

> **独立攻关线**，与 Phase 25（Option-C 多程序 whole-model orchestration）并行、互不干扰。
> 分支：pypto-lib `feat/whole-net-n1-fusion`。
> 最终 release 验证机：`gpu-a910x-0162`，devices `8..15`。
> 本文较早的 0234 环境重建与历史 bisect 章节保留作历史证据；当前 release
> 状态以本文顶部与 `N1-CANONICAL-TEST.md` 为准。
> 0162 clean 环境的唯一 stable 版本记录见
> [`develop/N1/N1-STABLE-ENV-0162-20260717.md`](../develop/N1/N1-STABLE-ENV-0162-20260717.md)。

## Pin snapshot（升级栈，2026-07-09 从 0162 只读 bundle 导入 + 重建）

| 组件 | pin | 来源 |
|------|-----|------|
| pypto | `e277de9f` | final clean pin；包含 stacked IPC weight views、`import_ipc_all` 与 runtime pin |
| simpler | `36957c6b` | final clean pin；包含 child-process ACL IPC import 与 co-tenancy diagnostics |
| pypto-lib | `feat/whole-net-n1-fusion` @ `0e7a0fdd` | final standalone P42 release：native W8A8/KV-IPC，dispatch-pull + combine-pull，512B signal isolation |
| pto-isa | `ecb6c303` | origin/main |
| PTOAS | `72ada0a1` | origin/main |
| ptoas-bin | **self-report 0.45** (`fe7949daa62c…`) | 0162 stable binary；历史 `v0.49` 标签不用于当前环境重建 |

## Goal

把 step3p5 整个 decode（45 层 + 末尾 RMSNorm + LM-head）融进 **一个 `@pl.program`**，
vLLM 每 decode step 只做一次 handoff（vLLM idle 期间 pypto 自调度）。这是 program 个数
N 三档里的 **N=1 档**（perf 最优：零 host residual handoff、一次 launch），用户明确选定的攻关方向。

## 关键架构决策（团队交叉验证 2026-07-09）

**"一个 program" ≠ "一个 dispatch fuse attn+MoE"。** hw-analyst 定位：pypto 编译器 pass-37
`materialize_comm_domain_scopes_pass.cpp:426-473` 在**同一 dispatch 同时携带 TP window
(tp_all_reduce) + EP window (all_to_all)** 时，按 device-set 相等把 11 个 window 塌成一个
comm-domain → codegen comm handle 与 kernel notify/wait 不匹配 → host_orch 产 0 chip task →
`TaskMapSize=0` → 507018。**这是上游编译器限制，model 侧改不了**（Wall-2）。

**破局结构**：ONE `@pl.program`，内部按 **Mixed2Method**（`patch_mixed_2method.py` 已验证机制）
把 attention 做成 **TP-only orchestration method**（自己的 pass/dispatch → 自己的 TP comm-domain），
MoE 做成**独立 EP-only method**（自己的 pass → 自己的 EP comm-domain）。pass-37 于是把每种协议
聚成各自 domain（standalone TP-attn / standalone moe_block 各自都能 dispatch）。

- 仍是**一个 program**（一 chip_process、一 prepare）→ 满足 N=1。
- 无 co-prepare → 躲开 blocker-1 的 N≥6 dispatch wedge。
- per-protocol 分离 dispatch → 躲开 Wall-2 塌缩。
- 与 **DeepSeek** 原则一致（V4 从不把 attn+MoE 塞一个 dispatch，用独立 program）。

本质 = 把已验证的 `_build_fused_dense_lmhead_program`（1 dense 层 + tail 的两-method 两-pass
host_orch，`decode_layer.py:903-1140`）**泛化到 45 层 + tail**（~90 sequential host_orch pass，
`hidden` 经 resident `pl.Out` GM buffer 串接）。

## 两堵墙（team root-cause，upstream 均未修）

- **Wall-1（编译）swa_moe const-fold 级联**：`attention_swa.py:479` in-`pl.full` config 算术 +
  dynamic `valid_shape` 在 EP lowering 下 symbolize 成 free Var → "must be ConstInt"。tile-space
  softmax 重写（literal SWA_Q_PAD + `pl.tile.load`/store + 加性掩码）已 COMPILE 过全 7 类，
  **数值未验证**。但注意：attn 若作为 **TP-only method**（非 fused swa_moe），standalone TP-attn
  本就编译过 —— Wall-1 大概率只在真 fused 时触发；需实测确认 mixed program 是否 EP-lower attn method。
- **Wall-2（device dispatch）fused TP+EP → TaskMapSize=0**：见上（pass-37）。per-protocol 分离
  method 是否在一个 program 内真机躲开 —— **Phase 3 决定性实验**。

## Phasing

| Phase | 内容 | 门槛 |
|------:|------|------|
| **0** | 环境重建（0234 升级栈，clean 重编，MoE-block 编译 parity PASS）+ tmov 编译回归修复（`OUT_PROJ_N_CHUNK` 256→64，见 deployment troubleshooting doc） | ✅ 2026-07-10 |
| **1** | swa attention tile-space 重写**数值验证**（走 swa_dense known-good device 路径，无 oracle 风险） | gate device |
| **2** | 建 whole-decode @pl.program 骨架：**3 dense 层 + tail**（`_build_whole_decode_dense_prefix_program` = `WholeDecodeDensePrefix`；full/swa 隔离 method + per-layer 权重 slab @layer_idx=0 + resident `pl.Out` handoff + 2-pass host_orch）。**compile ✅ 2026-07-10**（8 卡 device pending） | compile ✅ |
| **3** | dense-attn method + 完整 EP-MoE method-set + tail 在**一个 program** 共存 + 链接（`_build_mixed_moe_tail_program` = `MixedMoeTail` = Mixed2Method + `lm_head_orch` + pass-3 tail）。**compile ✅ 2026-07-10 rc=0**（`build_output/MixedMoeTail_20260710_025646`）。per-protocol 分离躲 `TaskMapSize=0` 的 8 卡 device 决定性实验 pending | compile ✅ |
| **4** | 全 45 层：`_build_whole_decode_all_program`=`WholeDecodeNetwork`，ONE `@pl.program` 跑全 45 层（每层 attn-only method → EP-MoE method，per-protocol 分离）+ tail。**compile ✅ 2026-07-10 rc=0**（`build_output/WholeDecodeNetwork_20260710_043504`；N=4 先导→N=45 全深度，源码级显式展开——前端禁 `@pl.function` 体内 Python `for`）。**遗留**：8 卡 device dispatch（Wall-2 决定性实验）+ 权重索引 3-class 45 层解耦（当前复用单 slab / dense-prefix L0-L2 近似）+ 47GiB 单 key 权重 IPC + `_pypto_full_forward` single-handoff + live A/B 8001-vs-8000 | compile ✅ / device+runtime 进行中 |

## 权重索引（3-scalar split，`weight_loader.py:755-789`）

- norm 栈 `[45]` 按 **绝对 layer_idx**；attn 栈 type-local `[12 full]/[33 swa]`；dense-MLP `[3]` dense-order；
  MoE 专家 `[42]` pos=layer-3。融合程序每层需 **3 个 scalar**（layer_idx / attn_type_idx / dense-or-moe-pos），
  在 Python prepare 期预算，逐 pass 传入。

## Exit criteria

- 一个 `@pl.program` 编译通过全 45 层 + tail（Wall-1 clear）。
- 8 卡 device dispatch 无 `TaskMapSize=0`/507018（Wall-2 clear）。
- 逐层数值对齐 vLLM dump（L1 ratio_allclose atol=0.04）；final logits argmax 一致。
- live single-handoff A/B：8001(pypto 整网) vs 8000(vanilla) token-exact。

## Risks

- **Wall-2 可能在一个 program 内 42 个 MoE EP-pass 仍触发**（Phase 3 验）；若是上游 pass-37 硬限制，
  N=1 gate 在上游修 or 回退 Option-C（Phase 25 线）。
- swa attention 重写数值正确性未证（Phase 1 先验）。
- gap-5 INT8-native 仍 gated OFF（BF16-dequant 为工作路径）。

## Status

✅ **2026-07-16 standalone canonical closure**：Phase 27 的发布门已经完成，而不是
仍停留在历史的 random-hang 或 push-mode 状态。固定对象为
`whole_decode_faithful_real`、`P_FAITHFUL_MOE_LAYERS=42`、token `6127`、
gpu-a910x-0162 devices `8..15`、native W8A8 IPC weights、KV IPC、
**dispatch fixed-slot pull + combine pull**。release commit `0e7a0fdd`
exact-source 在 fresh exporter pool 上连续 **20/20** 通过，每次
`argmax=303`，runtime `2.50/2.5605/2.62s`（min/mean/max）；final smoke 也通过
（`2.57s`, `FINAL_SMOKE=PASS`）。最终代码为 pypto-lib
`0e7a0fddc90c4f2348f1d59e015fb817a0877a02`，已推送到 fork。

最终保留的实现边界包括：dispatch 生成 `recv_counts` 与
dispatch-produced `inverse_map`，combine 直接消费该 inverse map；self
走 local load、peer 走 remote load；per-layer distinct communication
buffers；signed tile remainder；native INT8 W8A8 routed expert；以及
logical `[8,1] INT32` signal 的 physical 512B isolation。生成器真实
strip/regenerate byte round-trip 通过（`PRECOMMIT_ROUNDTRIP=PASS`,
`ROUNDTRIP_CMP_RC=0`）。

**Phase 27 当前结论**：standalone canonical stall gate 已关闭；历史 push
mode 的随机 stall、旧的中间层数 gate、以及把 PUSH/TPUT 或某个 signal bit
写成唯一硬件根因的过强结论，不再是当前状态。下一阶段转入 Phase 28 的
per-layer KV bridge、live HBM/重复权重消除和 live token-exact A/B。

🟢 **历史整网编译里程碑（2026-07-10）**：`WholeDecodeNetwork` —— ONE `@pl.program` 跑**全 45 层** decode（每层 attn-only method → EP-MoE method，per-protocol 分离）+ tail —— **编译通过 rc=0** on 0234（当时升级栈 pypto `5e619dc7` + ptoas v0.45 + tmov 修复；`build_output/WholeDecodeNetwork_20260710_043504`）。这只是历史 compile 证据，不等同于当前 clean-pin device release。

- `WholeDecodeDensePrefix`（3 dense 层 + tail，compile ✅）—— 证明多 dense 层 + tail 经 resident `pl.Out` handoff + 多-pass host_orch 在一个 program 内链接。
- `Mixed2Method` full/swa（compile ✅）—— 证明 per-protocol 分离（attn method + MoE method）编译；**swa_moe 分离 compile 过（fused swa_moe const-fold 失败）**，确认 per-protocol 分离是必需且正确的结构。
- `MixedMoeTail`（dense-attn method + 完整 EP-MoE method-set + tail，compile ✅ rc=0）—— 证明三者在**一个 program** 共存 + 链接（45 层程序所需的确切结构）。
- `MoeLayerReal`（真实 MoE 层：`swa_attn_only_orch` 纯 attention→resid1 → MoE `chip_orch` → residual → tail，compile ✅ rc=0）—— 证明真实 MoE 层结构（独立 attention 喂 MoE block）。
- `WholeDecodeProgram`（1 dense + 1 MoE mixed-min，compile ✅ rc=0）+ `WholeDecodeMixedMin`（忠实 dense-prefix + 真实 MoE 层，compile ✅ rc=0）+ **`WholeDecodeNetwork`（全 45 层 all-MoE 近似，compile ✅ rc=0）** —— N=1 整网编译达成。
- **`WholeDecodeFaithful`（忠实全 45 层：L0 full-dense + L1/L2 swa-dense + L3-44 42× 真实 per-protocol MoE + tail，88 个源码级展开 pass block，compile ✅ rc=0，`build_output/WholeDecodeFaithful_20260710_051822`）** —— **真实 step3p5 层表**的整网编译完成（非 all-MoE 近似）。

**遗留（Phase 4 device+runtime = 下一步）**：(1) 8 卡 device dispatch 决定性实验（per-protocol 分离是否躲开 Wall-2 `TaskMapSize=0`）。(2) 权重索引 3-class 在 45 层规模的解耦（当前编译用复用单 slab / all-MoE-layer 近似；真实需 host-side stack 切片 + dense-prefix L0-L2 + full-attn L0）。(3) 47GiB 单 key 权重 IPC + `_pypto_full_forward` single-handoff + live A/B。

### Phase 4 device 尝试（2026-07-10，历史中间误判）—— 曾将 507899 归因于 0234 节点级 IPC poison

> **后续已证伪**：紧接着的 clean rebuild 实验表明，单卡 507018 来自
> stale/mismatched runtime `.so`，多卡 507899 来自
> `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE` force-on；不是节点级硬件 poison。

- **新数据点：canonical TP=8 编译 ✅ rc=0**。此前编译里程碑走 `apply_perrank_patch`（TP=1 单卡）；本次用 `tests/step3p5/_probe_whole_faithful_canonical.py`（**无 patch**，canonical TP=8/EP=8 + `DistributedConfig(device_ids=[0..7])`）编译 `whole_decode_faithful` → `COMPILE OK`（`build_output/WholeDecodeFaithful_20260710_053948`）。说明 pass-37 comm-domain materialization 在真 TP=8/EP=8 下不报编译错，host_orch 生成真实 per-rank task loop（`for r in range(world_size): self.full_chip_orch(...)` 等）。
- **8 卡 device dispatch → 507899 `ImportByKey`（不是 `TaskMapSize=0`）**。用 `tests/step3p5/_stage_whole_faithful_device.py`（80 个 dummy-zero 输入匹配 host_orch signature，`compiled(*inputs)` 就地 `pl.Out`）在 device_ids=[0..7] 跑，第一处 `orch.allocate_domain("comm_d0", workers=[0..7])` 即失败：`domain_alloc_via_ipc: ImportByKey -> 507899`（`comm_hccl.cpp:833`）→ `comm_alloc_domain_windows failed with code -1` 全 8 chip。
- **当时的隔离线索**：已知良好 baseline `allreduce_distributed -p a2a3 -d 0-7` 与 `-d 0-1` 也出现 507899，因此暂时排除了 WholeDecodeFaithful 自身，并误判为节点级 IPC poison。后续 reboot/clean rebuild 证据推翻了该归因。
- **驱动 cap 在位**：0234 driver `25.5.2` / firmware `7.8.0.7.220`（STATUS.md 旧记录 25.5.1 已过时），所以 Phase 16 cap 缺口当时已可排除。
- **历史恢复尝试**：容器内 reset、`aclrtResetDeviceForce`、清理 `/dev/shm`/`/tmp` 均未改变现象；这些负结果后来应解释为 runtime build/config 未修复，而不是硬件 poison 的证明。
- **历史错误计划**：当时曾要求 host reset/reboot；实际 reboot 未解决，真正恢复来自下一节的 clean runtime rebuild + SDMA workspace OFF。
- **本次新增测试脚手架**（pypto-lib `feat/whole-net-n1-fusion`）：`tests/step3p5/_probe_whole_faithful_canonical.py`（canonical TP=8 编译探针）、`tests/step3p5/_stage_whole_faithful_device.py`（8 卡 device dispatch harness，80 输入）。

### Phase 4 device 续（2026-07-10 晚）—— device 解封 ✅ + Wall-2 决定性答案 = 分离躲开 dispatch fault ✅ + 新 blocker: scheduler timeout ⏸

**reboot 没有解决 507899**（新 pod 上单卡 `hello_world -d 0` 也挂 507018）→ 排除硬件 poison，定位为 **runtime 构建问题**（升级栈 device 路径从没验过）。两个独立修复：

1. **单卡 AICPU 507018 = stale/mismatched `.so`** → `build_runtimes --platforms a2a3` clean 重编 → 单卡 hello PASS。
2. **多卡 IPC 507899 = `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE` force-ON**（升级栈 `71e39623` 丢了 Phase-16 SDMA-OFF patch）。其 `SdmaWorkspaceManager::Init()` 在 `domain_alloc_via_ipc` 发 AICPU `aclnnShmemSdmaStarsQuery`，fault 后毒化紧跟的 `aclrtIpcMemImportByKey`。**`set(...OFF)`（host/CMakeLists.txt:42）+ reconfigure + rebuild** → `allreduce -d 0-7` 全 8 卡 `max|out-expected|=0.000e+00` ✅。详见 [`deployment/troubleshooting-multirank-507899.md`](../deployment/troubleshooting-multirank-507899.md) 新增段。fix 已 commit simpler `98ce22a6`。

**Wall-2 决定性答案（本 phase 的核心问题）= ✅ 分离躲开 dispatch fault**：device 解封后跑 `_stage_whole_faithful_device -d 0,1,2,3,4,5,6,7`（canonical TP=8 + dummy 权重）——WholeDecodeFaithful **编译 rc=0 + 8 卡 comm domain 分配成功（无 507899）+ host_orch 真实 dispatch 并执行了 ~88 个 pass-block scope（strace `device_wall.orch/sched` 多 inv）**。**没有 `TaskMapSize=0`、没有 dispatch-time fault**。→ **per-protocol 分离（TP-attn method + EP-MoE method 各自 dispatch pass）在真机确实躲开了 fused attn+MoE 的 Wall-2**。这是 N=1 的 **device dispatch 里程碑**。

**新 blocker（Phase 4 遗留）= scheduler timeout（`sched_error_code=100` / 507018）mid whole-net 执行**：整网跑到执行中途某个 task 前向不进 → scheduler 超时。**不是数据退化**（random gate_w 分散路由后同样）、**不是 ring pool 不够**（`PTO2_RING_HEAP=4G/TASK_WINDOW=131072/DEP_POOL=131072` 后同样）。是 88-pass-block 顺序链里某个 scope 的真实 forward-progress 停滞。下一步 = dispatch-cut bisect（截断到第 N 个 pass-block 看在哪 stall），定位是某具体层 / tail / 还是深度/资源上限。**注意：这与 Wall-2 正交**——dispatch 是干净的，卡在执行。

### Phase 4 scheduler-timeout 二分定位（2026-07-10）—— 根因 = MoE 层间 comm-window 复用违反 RAW-only-v1 non-aliasing ✅ 定位

> **作用域**：本节只解释 reuse-one-slab 生成版本中 K=2 可重复出现的
> comm-window alias stall。它是一个独立且已修复的 blocker，不是后续
> 0162 random pull+pull stall、512B signal 问题或 0234 待复核记录的统一根因。

**前置修复**：`b90e82e` 的 3-scalar split 只改了 inline `_func` 签名（`layer_idx`→`norm_layer_idx`+`attn/mlp_layer_idx`），但 N=1 whole-net 的 **29 处 inline 调用点**仍传单 `layer_idx` → 全部 whole-net program 编不过。reuse-one-slab 修复：29 处 `layer_idx` 复制成两份（norm=attn/mlp=layer_idx，字节等价，编译近似不变）。`smoke rc=0`。

**dispatch-cut bisect**（新增 env 门控 `P_FAITHFUL_MOE_LAYERS`，默认 42 = full；诊断脚手架，不进产品路径）：

| 程序 | dense-prefix | MoE 层 | 窗口 | 结果 |
|------|------|------|------|------|
| `whole_decode_program`（新 harness `_stage_whole_program_device.py`） | L0 full | 1（full-attn） | 独立 l0_*/l3_* | **RUN_CLEAN** 24s |
| faithful `P_FAITHFUL_MOE_LAYERS=1` | L0+L1/L2 | 1（swa） | 用 1 次 | **RUN_CLEAN** 40s |
| faithful `P_FAITHFUL_MOE_LAYERS=2` | L0+L1/L2 | 2（swa） | **复用 1 套** | **STALL 507018** |

**根因（代码级证据）**：K=2 的生成 `orchestration/host_orch.py` 里，layer-3 (`_ta_4`) 与 layer-4 (`_ta_6`) 的 `chip_orch` 把**同一个** `combine_done_buf__ssa_v0`（以及 recv_x/pub_counts/routed_y/data_done 等）当输入 —— **同一 SSA 版本 = 别名**。scheduler timeout 明细 `completed=177/180 running=1 waiting=2 stuck_task_id=2^32+5(TaskKey scope1/task5)` = 卡在 layer-4 chip_orch / tail。整网 42 个 MoE 层复用一套 comm/scratch window，违反 pypto **RAW-only-v1 的 non-aliasing 前提**（设计 P3 / ADR-013：`IMemoryManager` 须保证 intermediate memory non-aliasing）。dense-prefix 用 per-layer 窗口（`l0_*/l1_*` 各一套）正是为规避此问题；MoE 层却共用一套。

**与 DeepSeek 差异**：DeepSeek V4 用 **multi-program**（每层独立 dispatch/scope，窗口天然新分配 = 非别名）从根本规避。N=1 把 42 层塞进一个 host_orch 复用窗口，正撞此墙。

**下一步（fix）**：(1) 验证 per-layer（非别名）窗口修复 K=2；(2) 评估 42 层 per-layer 窗口 comm-domain 内存（recv_x+send_buf 各 8MB → ~17MB/层 × 42 ≈ 735MB/rank，可能超）；(3) 若超，抉择「小窗口池 + 层间显式 fence（让 scheduler 认非别名）」vs「对齐 DeepSeek multi-program（Option-C resident-DeviceTensor，constraint H 的既定 whole-decode 路线）」。

### Phase 4 scheduler-timeout 修复（2026-07-10）—— per-layer comm 窗口，N=1 整网 8 卡 device 跑通 ✅ SOLVED

**用户约束**：只做 N=1 单 program 集成，**multi-program 不考虑**。所以 fix 必须在一个 `@pl.program` 内解决 comm-window 别名。

**fix = per-layer（非别名）comm 窗口**：把 faithful host_orch 里 42 个 MoE 层复用的 13 个 comm/scratch window（ad_attn_tmp/attn_tmp/pub_counts/count_done/recv_x/recv_r_route/data_done/sh_tmp/sh_sig/routed_y/combine_done + 2 sig）从「顶部分配一套、全层复用」改成**每层各自 `pld.alloc_window_buffer`（`_L{pos}` 后缀，distinct SSA）**。每个 window 只被 1 个 submission 写 → 满足 RAW-only-v1 non-aliasing。dead 的旧共享 alloc 必须删（`MaterializeCommDomainScopes` pass 会因 dead window alloc 报错，反证 pypto 不为 dead alloc 占内存）。

**device 验证（canonical TP=8 + dummy 权重，8 卡）**：

| 变体 | 窗口 | 结果 |
|------|------|------|
| shared K=2 | 复用 1 套 | STALL 507018 |
| **per-layer K=2** | 2 套 distinct | **RUN_CLEAN 40s** |
| **per-layer K=42（全 45 层）** | 42 套 distinct | **RUN_CLEAN 45.89s，无 507018 / 无 OOM** |

→ **N=1 整网（L0 full-dense + L1/L2 swa-dense + 42 真实 per-protocol MoE + tail，88 pass-block）在 8 卡 device 上编译 + dispatch + 执行全程无 stall**。42 套 window fits comm domain（无需 `pl.free`）。scheduler-timeout blocker 清除。

**代码**：`decode_layer.py` faithful builder 内 per-layer 窗口 + 诊断用 env 门控 `P_FAITHFUL_MOE_LAYERS`（默认 42 = 全量；bisect 时截断 N 层）。新 harness `tests/step3p5/_stage_whole_program_device.py`（1 dense+1 MoE endpoint）。前置：29 处 3-scalar inline 调用点 reuse-one-slab 修复（smoke rc=0）。

**边界**：dummy 权重 → 输出 0，只证 forward-progress（无 stall）；**逐层数值对齐 vLLM 仍是 task #2/Step-2**（真实 per-layer 权重 + L1 ratio_allclose）。

### Step-2 refined 蓝图（2026-07-10，code-explorer 精确定位）—— faithful reuse-one-slab → 真实 per-layer 权重

**逐层 3-class 索引映射（绝对层 L=0..44）**：
- **norm_layer_idx = L**（绝对；入 45 行 norm 栈：input_rms/post_rms/q_norm/k_norm；KV-cache base = L×per_layer_rows）
- **attn_layer_idx = type-local**：full 层用 `L//4`（0..11，入 12 行 `KEY_W*_FULL`）；swa 层用 `L-(L//4+1)`（0..32，入 33 行 `KEY_W*_SWA`）。例：L0 full→0；L1 swa→0；L2 swa→1；L3 swa→2；L4 full→1。
- **mlp_layer_idx = L**（L0/1/2 → 0/1/2，入 3 行 `KEY_DENSE_*`）
- **moe pos = L-3**（L3..44 → 0..41，入 42 行 `KEY_MOE_*`）

**两处必改（gap）**：
1. **faithful 内联 orchestrator**（`full_chip_orch`/`swa_chip_orch`/`swa_attn_only_orch`/`chip_orch`/`attn_dense_orch`）当前收单 `layer_idx` 复制成 `(layer_idx,layer_idx)` 传 inline。→ 改成收 3 个真实 scalar，host_orch 逐 pass-block 传 `(norm=L, attn=type_local, mlp/moe_pos)`。inline `_func` 签名**已就位**（norm+attn / norm+mlp），无需再改 kernel base 公式。
2. **MoE `chip_orch` 无 layer 维**：`gate_w`/`router_bias`/`w_{gate,up,down}_{r,s}` 直接传子步骤，无索引（仅 `post_rms` 按 `layer_idx` 切）。→ 加 `moe_pos` scalar + 给这些权重加前导 `[42]` 维在 body 内切片，**或**（推荐，对齐 `decode_layer.py:19142` 的 host-side slicing 延期note）host_orch 侧按 pos 切 slab 再传单层。

**两处 pre-existing 坑（真实权重才暴露）**：
- **swa `LAYER_HIDDEN_ROWS_DYN=49152`（=12×HIDDEN，full 高度）被 attention_swa 复用于 swa wq/wk/wv**，但 swa 有 33 层 → 真实 swa 栈应 `33×4096=135168`。reuse-one-slab（idx=0）掩盖了此不一致。喂真实 swa 权重前必须核对 loader `KEY_WQ_SWA[33,HIDDEN,...]` flatten 后的行数 + 调 config 常量。
- **L1/L2 各有独立 `l1_*/l2_*` slab 但都传 idx=0**；真实权重下 L1 用 swa-local 0、L2 用 swa-local 1 → 可合并成一份 swa 栈 + 逐层 idx。

**weight_loader 已就绪**（`weight_loader.py:747-864`，无需改）：bundle 已按 3 类栈（norm[45]abs / attn full[12]+swa[33] type-local / dense[3] / MoE[42]pos）+ 每 rank EP/TP 切好；W8A8 index + dequant 已支持。真实权重路径 `/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp/`。

**device harness `_stage_whole_faithful_device.py`**：单-slab 输入 → 全栈（`m_gate_w`→`[tp,42,HIDDEN,N_EXPERTS]`、`m_w_gate_r`→`[tp,42,36,HIDDEN,1280]` 等加前导 42；ma_*/m_* swa attn → `[tp,33,...]`；norm 已 `[tp,45,...]`）。

**验证**：compile rc=0（host）→ 8 卡 device 逐层 hidden vs vLLM eager dump（L1 ratio_allclose atol=0.04）；final logits argmax 一致。建议先加 flag/新 builder，保当前 per-layer-window reuse-one-slab 干净基线不回归。

### Step-2 增量①（2026-07-10，pushed `7b9693b`）—— dense-prefix 真实 per-layer 索引 ✅

`full_chip_orch`/`swa_chip_orch` 单 `layer_idx`→3 scalar；host_orch dense 调用传 L0=(0,0,0)/L1=(1,0,1)/L2=(2,1,2)（norm=abs/attn=type-local/mlp=abs）。dense 权重栈已 stack-sized，索引 in-bounds，无 shape 改。smoke rc=0；全 45 层 8 卡 device DISPATCH_CLEAN（dummy，无回归）。

### Step-2 策略决定（下个增量用）：attn+MoE 权重走 **host-side slicing**，norm 走 full-stack

两条路线权衡后选 **host-slice**（更干净，避开两处 friction）：
- **NORM**（input_rms/post_rms/q_norm/k_norm）：**full [45] stack + 真实 `norm_layer_idx=L`**（kernel 索引；KV-cache base = `norm*cache_rows` 必须用绝对 L，不能 host-slice）。
- **ATTN**（wq/wk/wv/wo/w_g/gate_r）+ **MoE**（gate_w/router_bias/experts）：host_orch 侧按层 `stack[pos]` 切成**单层 slab** 传进 orchestrator，orchestrator/kernel 的 weight-index 传 **0**（收到的已是该层 slab）。→ **避开 swa 33-stack resize**（不再需要 kernel 索引 33 层）**+ 避开 chip_orch [42]-dim body 改写**（MoE 权重预切，gate/dispatch/expert 步骤不动）。
- **增量① 的 dense 用了 kernel-index（真实 attn idx）**：dense 栈小无 friction，可保留；但 MoE-层 attn + experts 统一走 host-slice。或把 dense 也改 host-slice 以统一（attn/mlp idx=0）。**下个增量二选一并统一。**
- 待改：`swa_attn_only_orch`(norm=L,attn=0) + `chip_orch`(norm=L, MoE 权重预切) 签名/调用；host_orch MoE 段按 `pos=L-3` / `attn=swa-local` slice [42]/[33] stack；harness 传 [45]/[42]/[33] full stack（由 weight_loader 真实权重填充）；device 逐层 vs vLLM。

### Step 2 实施蓝图（3-class 权重解耦；compile 可先做，device 数值验证 gate 在 reboot 后）

> N=1 分支（base `94aa015`）**没有** `8b4bf3fa`（stepfun/develop）的 3-scalar split —— grep 零 `norm_layer_idx`/`attn_layer_idx`/`mlp_layer_idx`；`WholeDecodeFaithful` 全部方法只吃单个 `layer_idx: pl.Scalar[INT32]`（`decode_layer.py` full_chip_orch@20727 / swa_chip_orch@20784 / chip_orch@20502 / swa_attn_only_orch@20842 / attn_dense_orch@20454）。

**现状（reuse-one-slab）**：`full_chip_orch` 把一个 `layer_idx` 同时传给 `attention_full_inline` + `dense_mlp_inline`，两者内部各自 `layer_idx*HIDDEN`(attn/mlp base) / `[layer_idx,:]`(norm) / `layer_idx*INTER_LOCAL`(w_down) 取行。42 个 MoE 层共用 `ma_*`/`m_*` slab + `layer_idx=0` → 全读 layer-0 权重（编译近似）。

**目标**：每层读真实权重，三类栈索引解耦：
- norm（`KEY_INPUT_RMS`/`KEY_POST_ATTN_RMS`/`KEY_Q_NORM`/`KEY_K_NORM`）`[45]` = **absolute** `layer_idx`
- attn（`KEY_WQ_FULL[12]`/`KEY_WQ_SWA[33]` + wk/wv/wo/w_g）= **type-local** idx（full→0..11 / swa→0..32）
- dense-MLP（`KEY_DENSE_GATE/UP/DOWN[3]`）= dense-order idx（L0/1/2 → 0/1/2）
- MoE experts（`KEY_MOE_GATE_W`/`ROUTER_BIAS`/`W_*_R`/`W_*_S`）`[42]` = **pos = layer_idx − 3**

**改动清单**（execution，reboot 后一次跑通 + device vs vLLM 验证）：
1. **inline kernel** `attention_full._func` / `attention_swa._func` / `_dense_mlp_body_tp._func` / `moe` 各体：单 `layer_idx` → `norm_layer_idx` + `attn_layer_idx`（+ `mlp_layer_idx`/`moe_pos`），每类 base 用各自 scalar。（可 cherry-pick `8b4bf3fa` 的 74-edit split，注意 N=1 splice 自旧签名，需对齐。）
2. **5 个 orch 方法**签名：单 `layer_idx` → 3 scalar，转传给 inline。
3. **host_orch**：`ma_*`/`m_*` 单-slab 入参 → **全栈**（`ma_input_rms [tp,45,HIDDEN]`、`m_w_gate_r [tp,42,n_local_experts,HIDDEN,inter]` …）；88 pass-block 每层传 `(norm=abs_li, attn=type_local_li, moe_pos=li-3)`。
4. **device harness** `_stage_whole_faithful_device.py`：dummy 全栈 shape 对齐（first-dim 45/42/type-local）+ 逐层 index。真权重经 `weight_loader.py:747-810`（已按 3 类 stack，无需改 loader）。
5. **验证**：compile rc=0（host，可先做）→ reboot 后 8 卡 device 逐层 hidden vs vLLM dump（L1 ratio_allclose atol=0.04）。

**注意**：本蓝图**不动**当前 compile-passing 的 reuse-one-slab `_build_whole_decode_faithful_program`（保留 device dispatch 决定性实验的干净基线）；3-class 变体建议加 flag 或新 builder，避免回归编译。

### Step-2 增量②（2026-07-11）—— 真实 per-layer 权重 + full+swa 完整路由（新 builder）✅ compile + device dispatch

**用户拍板两决策**：(1) 11 个 full-attn MoE 层（L4,8,…,44）本 session 直接做 **full+swa 完整路由**（不留 swa 近似）；(2) 用**新 builder + 新 program**（`whole_decode_faithful_real`），保护 Task#1 的 reuse-one-slab device 干净基线不回归。

**decisive 探针（`tests/step3p5/_probe_single_layer_inline.py`）**：确认 `pl.inline(attention_swa._func)` **接受实参 leading dim 比标注小**（单层 `[HIDDEN,…]` 喂进标注 `[LAYER_HIDDEN_ROWS_DYN=12*HIDDEN,…]` 的 inline，`attn_layer_idx=0` 读 `[0:HIDDEN]`）→ `RESULT=SINGLE_LAYER_INLINE_OK`。因此 swa-MoE attn 走**干净 host-slice 单层**，不需 33-stack resize / sub-stack 回退。

**统一设计**（`weight_loader.expected_shapes()` 1:1 对上 host_orch）：
- norm（input_rms/post_rms/q_norm/k_norm）`[45]` full stack，kernel-index 绝对 `norm_layer_idx=L`（KV base = `norm*rows` 需绝对 L）。
- attn（wq/wk/wv/wo/w_g/gate_r）+ dense-MLP（w_gate/w_up/w_down）：**host-slice 单层 slab**，kernel weight-index=0。full attn 走 `full_attn_only_orch`（新增），swa 走 `swa_attn_only_orch`（重写单层）。
- MoE experts（gate_w/router_bias/w_*_{r,s}）`[42]` stack，host-slice by `pos=L-3`；`chip_orch` 仅 `layer_idx→norm_layer_idx` rename（post_rms），expert 参数不变。

**codegen 落地**：一次性生成器 `tools/step3p5/_gen_faithful_real.py` 从 compile+device 验证过的 `_build_whole_decode_faithful_program` 派生 `_build_whole_decode_faithful_real_program`（`WholeDecodeFaithfulReal`）：method-set 逐字复用，chip_orch rename，重写 full/swa_chip_orch + swa_attn_only_orch 单层，新增 full_attn_only_orch，重写 host_orch（统一栈签名 + 42 层 full/swa 路由 + 真实索引/host-slice）。N_FULL=12 / N_SWA=33 / N_DENSE=3 / N_MOE=42。

**验证**：
- smoke rc=0（import 构造 `whole_decode_faithful_real`）。
- **compile rc=0**（canonical TP=8，`build_output/WholeDecodeFaithfulReal_20260710_164924`，harness `tests/step3p5/_stage_whole_faithful_real_device.py --compile-only`）。
- **8 卡 device DISPATCH_CLEAN 285.36s**（dummy 权重，88 pass-block 全跑，无 507018 / 无 scheduler stall；`_stage_whole_faithful_real_device -d 0..7`）。
- 真实 W8A8 权重 device run harness `tests/step3p5/_stage_whole_faithful_real_weights.py`（`weight_loader` 逐 rank bundle stack；ckpt `/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`）—— 运行验证中。

**遗留（Task#4 逐层对齐 vLLM 的真正 gate）**：vLLM eager dump 是 **18-token PREFILL**，decode kernel 是 **BATCH=16 单 token**——shape 不兼容，逐层 kernel-vs-dump 无法直接比。真正 token-exact 对齐需 **decode-step golden** 或 **live single-handoff A/B（Task#3）**。`current_hidden` + KV cache 也需真实 decode 上下文（当前 harness 用 synthetic）。

### 历史

🟡 **Phase 0（2026-07-09）**：0234 五仓 + ptoas-bin 切升级栈；clean 重编 pypto+simpler；团队 3 份调查报告交付（Wall-1/Wall-2 root-cause + N=1 结构建议）。

### Step-3 kickoff（2026-07-11）—— live vLLM-IPC 集成：复用 co-tenancy 修复 + 迁移到 0162 ✅ 基础就绪

> NEXT-SESSION-N-1.md 把 Task①（HCCL 同卡共存）当成核心 gate；实际上它**已在另一线程解决**
> （`deployment/cotenancy-simpler-no-hccl.md` + memory `project_g4_cotenancy_hccl_conflict`）。本 session
> 复用该修复并把 N=1 整网线迁到唯一可达的 device 机 0162。

**机器现状变更（重要）**：0234（原 N=1 dev 机）**不可达**（DNS→timeout）。0162 可达且是 co-tenancy 修复的验证机，
故本 session 把 N=1 线**consolidate 到 0162**（此前 phase27 写的"不碰 0162"因 0234 宕机作废）。0162 = **完全一致的升级栈**
（pypto `5e619dc7` / pto-isa `ecb6c303` / PTOAS `72ada0a1` / simpler `878f3742`=71e39623+co-tenancy patch），
与 N=1 program 验证栈相同。CANN 走 non-GA symlink `/usr/local/Ascend/cann`（非 beta，见 memory
`env_0162_cann_symlink_not_beta`）。

**Task① 复用确认（HCCL 层）✅**：`SIMPLER_COMM_NO_HCCL=1` 跳过 `HcclCommInitRootInfo`。device 实测：不带 flag →
`HcclCommInitRootInfo failed:7`（8/8 rank）；带 flag → error 7 消失。修复在 simpler runtime 层、program-agnostic，
对 `whole_decode_faithful_real` worker 同样生效。comm_hccl.cpp 已 patch+重编（07-11）。

**机器拓扑发现**：一个 **vanilla vLLM 8000（step3p5 W8A8 oracle）正跑在 cards 0-7 @ util 0.96**（58GB/card，health=200）
—— 这是 A/B 的对照 oracle。**cards 8-15 空闲**。（npu-smi HBM 列读 0/0 是 idle 遥测假象，真实占用看 proc-mem。）
最初 allreduce 在 0-7 失败（error 7 + rtMalloc 207001）其实是**撞上 vLLM 8000 的同卡共存**，非节点 poison。

**0162 基础验证 ✅**：
- allreduce twophase `-d 8-15`（空闲卡，normal HCCL）= `max|out-expected|=0.000e+00` 8/8 golden（env 铁律 baseline 满足；SDMA-ON 在 0162 无 IPC poison）。
- `whole_decode_faithful_real` compile + **8 卡 device DISPATCH_CLEAN 158.22s**（cards 8-15，dummy 权重，无 507018/stall，rc=0）—— N=1 program 在 0162 升级栈上工作，与 0234 一致。
- 非破坏性 `git worktree`（`pypto-lib-n1` @ branch `n1-live`，源 d9b7dc6）避免扰动 0162 stepfun/develop 的 gap-5 未提交工作。

**Task② vLLM-IPC 权重 handoff —— 蓝图落定 + card-free 验证**：
- `pypto_weight_ipc.py`（WeightIpcExporter/WeightIpcMap）从 stepfun/develop 移植进 N=1 分支（commit `d3f155b`，push `feat/whole-net-n1-fusion`）。
- 权重桥：`weight_loader.KEY_*` → `whole_decode_faithful_real.host_orch` 位置参数 **1:1**（顺序见 `_stage_whole_faithful_real_device.py:104-157` 与作废的 `_weights.py:121-176`）。`gate_r`（full+swa）不在 checkpoint bundle → 合成 zeros/ones（head-gate ×1 bypass）。
- card-free layout smoke PASS：45 keys、pool **47.46 GiB/rank**、fp32_keys 正确、map round-trip OK。
- **⚠ live A/B 的 HBM gate = gap-5**：47.46 GiB/rank 是 dequant-BF16 pool，与 vLLM 常驻 W8A8（~24GB）在 64GB 卡上共存 → OOM。co-resident live A/B 需 **in-kernel W8A8 dequant（保 INT8 ~24GB 共享）**，属 perf 阶段的 net-new kernel 工作（task6 memory `project_task6_live_wiring_plan` gap-5）。bring-up 可先走 pypto 独占空闲 8 卡 + 47GB checkpoint-H2D pool（fits，验证 IPC 机制 + real-weight dispatch），但非 co-resident。

**Task③ wire `_pypto_full_forward` —— 蓝图落定（未实现）**：
- 目标 `tools/step3p5/vllm_monkey_patch.py:233`（fail-closed stub）；`install(mode=full)` 已就位（换 `Step3p5Model.forward` + tail）。
- net-new：(1) resident `DistributedWorker` holder（install() 建一次，fork 8 chip on 8-15 带 `SIMPLER_COMM_NO_HCCL=1`）；(2) 从 `ir.compile+compiled(*inputs)` auto-shard 改成 `DistributedWorker + import_ipc` per-rank 权重 DeviceTensor（Task②）；(3) resident hidden DeviceTensor handoff；(4) live `forward_context` KV/slot_mapping/block_table（见 `_maybe_dump_forward_context`）。`enforce_eager` 必需。
- live A/B：可复用 `/mnt/nvme1/chensiyu/logs/step3p5_910b_w8a8_v001/start_8001_cotenancy.sh`（8-15 @ util 0.5）pypto vs 8000 vanilla oracle。这就是 Task#4 逐层对齐的真正落点。

**下个 session 的 punch-list**：(1) resident-worker + weight-IPC device run（Task② flavor-a，pypto 独占 8-15 + 47GB checkpoint-H2D，验证 IPC 机制）；(2) `_pypto_full_forward` body；(3) gap-5 in-kernel W8A8（co-resident HBM gate）；(4) live A/B。

### Step-3 更新（2026-07-11 续）—— 0234 经 tmux 可达 + 真实权重 self-load OOM（确认 Task② 架构）

**历史机器记录（2026-07-11）**：当时 `ssh 0234` 裸主机名不解析（→0.0.0.156），但
0234 可经 tmux `pypto-ascend-0:0` 使用，并与编辑机共享 NFS。该记录只解释早期
开发环境；2026-07-16 最终 release 已转到 0162 devices 8..15 完成 20/20，不得再据此
把“不要使用 0162”当作当前约束。

**关键发现：真实权重 self-load = 死路（OOM）**。在 0234 cards 0-7 跑 `_stage_whole_faithful_real_weights.py --ckpt <real> -d 0..7`：**compile OK**（`WholeDecodeFaithfulReal_20260711_034124`，45 perf hints），但 `compiled(*inputs)` dispatch 阶段被 **OOM-killer 杀（exit 137）**——harness 在单 driver 进程里 stack 全 8 rank bundle + `[tp,...]` 副本 ≈ 752GB > 607GB free。**这确认了文档 Task② 的架构决定**：真实权重必须**逐 rank**（绝不 8× stack 在一个 host），只能走 (a) 每 forked chip 内逐 rank load（`step3p5_decode.py:275-293` / `_stage_whole_decode_run.py --worker` 模式）或 (b) vLLM-IPC（`WeightIpcExporter.export_from_checkpoint` 逐 rank 1 bundle/进程 mem-safe → 1 key/rank → worker `import_ipc` child ctx → 逐 rank DeviceTensor）。两者都需 **DistributedWorker**（非 `ir.compile+compiled(*inputs)` auto-shard）。

**Task② 下一步（net-new）**：为 N=1 program 建 DistributedWorker + 逐 rank 权重（forked-load 或 IPC）harness。权重桥 `weight_loader.KEY_*` → host_orch 位置参数 1:1 已 mapped，card-free layout smoke PASS（45 keys / 47.46 GiB/rank）。⚠ co-resident live A/B 仍 gate 在 gap-5（47GB BF16 池 + vLLM 24GB > 64GB）。

**G-series（0162）现状复用价值**：`_stage_whole_decode_run.py --worker/--serve` 是成熟 Option-C sidecar（逐 rank forked-load 真实 W8A8、AF_UNIX socket、SIMPLER_COMM_NO_HCCL co-tenancy）+ 容器 backend `pypto_whole_decode_backend.py`（sitecustomize autoload，rank0 embed→socket→sidecar→broadcast，compute_logits 留 vLLM）。G5a 已 device-verified live plumbing（当前 4 层 + synthetic + dummy-KV → token garbage）。gate = G3 真实 KV-IPC + 全 45 层 + 真实权重（HBM/gap-5）。**N=1 live sidecar 可复用此 socket 协议/backend 模式。**

### Step-3 决策（2026-07-11 再续）—— N=1 + 借用 import_ipc（不照搬 G-series Option-C）

**用户拍板**：只**借用 0162 G-series 的 `import_ipc` feature**（纯 Python `_CTRL_IMPORT_IPC=16`，
KV/weight 零拷贝导入），**不照搬其 Option-C 架构**；**继续支持 N=1 program `whole_decode_faithful_real`**。

**为何 N=1 + import_ipc 是更优路径**：
1. **N=1 已编过全 45 层**（含 swa-MoE，DISPATCH_CLEAN 285s dummy）——**绕开 G-series 最硬 gate
   swa_moe const-fold 级联**（`attention_swa.py:480 Sub→:573 Var` 多站点，卡 30/45 层，多 session 未解）。
   N=1 用 host-slice 单层 attn（`full/swa_attn_only_orch`），不触发那条 EP-lowering const-fold。
2. **N=1 是单 program、一次 dispatch、45 层权重一次性喂进**——G-series 靠"逐层 host 权重 streaming"
   避 OOM 的办法对 N=1 不适用（无法 per-layer stream）。**N=1 唯一不 OOM 的权重路 = import_ipc 零拷贝**
   （每 rank 导入自己的 47GB 池，不 8× host stack；实测 self-load 8× stack = 752GB OOM exit137）。

**借用的机制（mirror KV 路，apply 到 weight）**：
- `import_ipc_all({dev:256B key})->{dev:peer_va}`（`distributed_runner.py:1086`，纯 Python broadcast，无需 C++ facade）。
- 用法：per-rank key → `_dworker.import_ipc_all({dev_offset+r:key_r})` → per rank `WeightIpcMap(peer_base=va,map).device_tensor(KEY_*)` → `DeviceTensor(va+off,shape,dtype)` → `StackedDeviceTensor([rank shards],full_shape,ids)` 对上 host_orch `[tp,...]` 签名。（`pypto_kv_ipc.py` KvIpcMap/build_stacked_kv 是模板；`rt.run` 已证收 DeviceTensor。）

**N=1 weight-IPC turnkey build（下一步，全在 0162 pypto-lib-n1 worktree + import_ipc-patched 运行时）**：
1. **Exporter（per-rank，避 OOM）**：8 进程各 `WeightIpcExporter.export_from_checkpoint(ckpt, rank=r, out_dir, dev=8+r)`——每进程只 load rank-r 的 1 个 bundle（47GB host transient，H2D 后释放）+ malloc 47GB device 池（card 8+r）+ export 1 key，进程常驻持池。
2. **改 `pypto_weight_ipc.py`**：`WeightIpcMap` 加 `import_ipc_all` 批量路径（replace 孤儿 `rt.import_ipc`）+ `build_stacked_weights(maps)->{host_orch_key: StackedDeviceTensor}`（KEY_*→位置参数 1:1，gate_r 合成 zeros/ones）。
3. **N=1 worker harness**（新 `_stage_whole_faithful_real_ipc.py`）：compile `whole_decode_faithful_real` → `compiled.prepare()` → `rt`（DistributedWorker）→ `rt.import_ipc_all(weight_keys)` → 建 weight StackedDeviceTensor → `rt.run(compiled, *ordered)`（weight=DeviceTensor + KV/hidden 先 dummy host）→ 验 DISPATCH_CLEAN + finite logits（**证 import_ipc 解 N=1 real-weight OOM**）。
4. 之后：KV import（同 import_ipc_all，真 KV）+ socket serve（复用 G5a 协议 + `pypto_whole_decode_backend.py`）+ live A/B vs 8000。

**运行机**：**0162**（import_ipc 运行时改动在 0162 pypto 工作树未提交；建议 commit 进 csy0225/pypto stepfun/develop）。N=1 用 worktree `$WS/pypto-lib-n1`（branch n1-live），`PYTHONPATH=$WS/pypto/python:$WS/pypto-lib-n1`。co-tenancy 时 sidecar 设 `SIMPLER_COMM_NO_HCCL=1 WD_RING_HEAP=1073741824`。⚠ 0162 有 G5b 平行 thread 在 stepfun/develop 主 checkout 工作——用独立 worktree 不冲突，device 跑避免同卡并发。

### Step-3 device 落地（2026-07-11 三续）—— N=1 real-weight import_ipc DEVICE-VALIDATED（核心复用坐实）

**做了什么**：`tests/step3p5/_stage_whole_faithful_real_ipc.py`（pypto-lib commit `6d09e47`）+
`pypto_weight_ipc.import_weights_all/build_stacked_weight`（`b2cf225`）—— N=1 `whole_decode_faithful_real`
真实 W8A8 权重经**借用的 import_ipc**（8 exporter 各 hold 一 rank 47GB 池 → worker `compiled.prepare()`
→ `import_ipc_all` → `build_stacked_weight` 零拷贝喂）。在 **0234 cards 0-7（8 卡空闲）** 跑。

**device 验证（run #1）—— 核心复用全部坐实 ✅**：
- 8 exporter 全部 export 真实 47.46 GiB 池 + key（`pool_base=0x12c1c0000000`，与 phase-16 probe2 VA 一致）。
- `whole_decode_faithful_real` compile OK（`WholeDecodeFaithfulReal_20260711_142935`）。
- **`import_ipc_all` 返回 8 个 real-weight peer_bases**（`0x12c1c0000000 ×8`）—— **N=1 真实权重零拷贝 IPC 导入 device 打通**（解 self-load 752GB OOM）。
- built 42 args（权重全 DeviceTensor via IPC）。
- ⛔ `rt.run` 报 `TypeError: host torch.Tensor 必须 .share_memory_() 且在 prepare() 前分配`（DistributedWorker 契约，G-series `sh[...].share_memory_()` 同款）——**小契约错，非机制错**。

**修复**：dummy host 张量（current_hidden/KV/rope/gate_r/outputs/logits）改为 **prepare() 前 `.share_memory_()` 分配**；VOCAB_LOCAL 从 map json 读（prepare 前）。run #2 验证中。

**意义**：借用 import_ipc（不照搬 Option-C）让 N=1 单 program 真实权重 device 可跑——Task② 的 device 机制坐实。剩：run #2 出 finite logits → 真 KV import（同 import_ipc_all）+ socket serve（复用 G5a backend）→ live A/B。co-resident live A/B 仍 gate 在 gap-5 HBM（0234 空闲卡跑 standalone 无此限）。

### Step-3 run#3（2026-07-11 四续）—— import_ipc 全链通到 rt.run；剩 StackedDeviceTensor `[r,k]` 结构 gate

run#3（share_memory + norm FP32 两 fix 后）在 0234 cards 0-7 走到**最远**：8 exporter export 真实
47.46GiB 池 → `import_ipc_all` 8 peer_bases → compile OK → 42 args built → 8 chip_process ready →
进 `rt.run`。**所有 arg-marshalling 契约全过**（compile / import_ipc / share_memory / dtype）。

**剩余唯一结构 gate**：`rt.run` 报 `ValueError: StackedDeviceTensor only supports whole-shard slicing
on the leading dim; got partial slice 0 on axis 1 (shard shape (12, 4096, 1024))`。根因：N=1 host_orch
把权重按 `full_wq[r, k]` 索引（rank r + 层 k 在 axis 1），但 `StackedDeviceTensor.__getitem__`
（`pypto/python/pypto/runtime/device_tensor.py:261`）只支持 leading（rank）维整 shard 切片，不支持 axis-1
partial 切。dummy `compiled(*inputs)` 用 host torch 张量支持 `[r,k]`，故没暴露；IPC StackedDeviceTensor 暴露。

**下 session 解法候选**：(a) 增强 `StackedDeviceTensor.__getitem__` 支持 `[r,k]` → 返回
`DeviceTensor(shard_r.base + k*layer_stride, [4096,1024])`（权重池 per-rank full_wq 是 [12,4096,1024]
连续，axis-1 slice 是合法 sub-view，最干净，改 device_tensor.py runtime）；(b) build per-(protocol,layer)
StackedDeviceTensor + 改 host_orch/harness 只索引 `[r]`（需改 _gen_faithful_real，程序签名变大）；(c) 每 rank
单 DeviceTensor 直传不 stack（看 runtime arg-bind 是否支持）。推荐 (a)。

**结论**：N=1 real-weight import_ipc 距 finite logits 只差这一个 runtime slicing 增强；Task② device 落地
在下 session 一步之遥。co-resident live A/B 仍另 gate 在 gap-5 HBM。

### Step-3 run#4/#5/#6（2026-07-11 五续）—— getitem gate 解决；连暴两个 device blocker（arena-OOM → collective-stall）

**① `StackedDeviceTensor.__getitem__` `[r,k]` gate 已修（推荐方案 a）✅**：`pypto/python/pypto/runtime/device_tensor.py`
`__getitem__` 现对 partial trailing 选择 delegate 给 shard 自己的 `DeviceTensor.__getitem__`（连续 sub-view，
算 `base + k*layer_stride*elem`，非连续则 loud `NotImplementedError`）。whole-shard 形式（`x[r]`/`x[r,...]`/
`x[r,0:N,0:M]`）行为不变仍返回整 shard。对齐 host_orch 全部索引形式（4D 权重 `[r,k,0:N,0:M]`、MoE 5D
`[r,k,0:36,0:1280,0:4096]`、router_bias 2D `[r,k,0:288]`、whole-shard norm `[r,0:45,0:4096]`）。CPU 逻辑
校验全过（offset/shape/identity/非连续 reject）。UT `test_device_tensor.py` 的 `test_partial_tail_slice_rejected`
（断言旧 loud-reject 契约）已按新契约替换为 per-layer-plane + contiguous-partial + noncontiguous-reject 三例。
**run#4 device 验证：getitem gate 通过**——8 exporter export 真 47.46GiB 池 → compile OK → 8 chip ready →
`import_ipc_all` 8 peer_bases → 42 args → 进 host_orch，**不再报 whole-shard ValueError**，走到真正的 device
dispatch。Task① 完成。

**② 新 device blocker A — 静态 arena OOM（`rtMalloc 207001 size≈16GB`）**：run#4 越过 getitem 后在
`ensure_static_arenas` 报 `rtMalloc failed: 207001 (size=17179870207)` / `Failed to setup pooled static arena`。
根因：静态 arena `total_heap = Σ heap_sizes[r]`（`runtime_maker.cpp:501 derive_arena_static_sizes`），
`PTO2_MAX_RING_DEPTH=4`（`pto_runtime2_types.h:70`）× `PTO2_RING_HEAP=4GB` = **16GB arena**。单卡预算：
3.34GiB baseline + 47.46GiB 权重池 + 16GiB arena ≈ 66.9GiB > 64GiB（65536 MiB）→ OOM。**非泄漏**（kill 后
各卡 3413-3416 MiB baseline，~61GiB free）。这也修正了 memory `n1_live_integration` 中「standalone 47GB 池
fits」的乐观估计——漏算了 16GB runtime arena。**缓解**：降 `PTO2_RING_HEAP` 缩 arena（现有 env 旋钮，非新
gate）。max-fit = 3GB（arena 12GiB，总 ~64.3GiB，margin ~0.86GiB）。

**③ 新 device blocker B — 首个 collective running-stalled（507018 / sched=100）**：run#5（`PTO2_RING_HEAP=2GB`，
arena 8GiB，越过 OOM）allocate + **实际执行 45.4s** 后报 `orch_error_code=8 sched_error_code=100
S1:running-stalled completed=4/32 running=1 waiting=3 stuck_task_id=0x100000003 stuck_core=24/26/28`。
**8 卡对称**卡在同一 task（scope-level 1, task 3）——典型**首个 tp_all_reduce/EP barrier collective 不完成**。
非 heap-alloc deadlock（无 `Task Allocator Deadlock`），非 OOM。已知 `whole_decode_faithful_real` 曾
dummy-weight DISPATCH_CLEAN（collective 本身能通），故 delta = (a) heap 降到 2GB 或 (b) 真 47GB IPC 权重池与
comm-domain IPC window 共存干扰。**run#6（`PTO2_RING_HEAP=3GB` = max-fit）验证中**：若清除 = heap-starvation
（2GB 太小饿死 comm path）；若同样 4/32 stall = IPC 权重池干扰 collective，需查 comm-domain window 与 47GB 池
共存（deeper）。

**当前净结论**：Task① getitem 已 device 落地；Task② finite-logits 被两个新 device blocker 阻（arena-OOM 已解，
collective-stall 待 run#6 判定）。Task③（真 KV + live A/B）仍 gate 在 gap-5 HBM（需 live 8001 co-resident，
0234 standalone 无 8001）。

### Step-3 run#7 + 隔离实验（2026-07-11 六续，历史中间假设）—— 曾将 stall 归因于 IPC child_memory 权重 × collective

**run#6（heap=3GB）**：不 fit——12GiB heap-arena 分配后，另一个 **~2.5GiB 固定 arena 组件**（tensormap/sm，
`rtMalloc size=2571110271`）顶爆。故 `PTO2_RING_HEAP` fit 天花板 ≈ 2.6GB（arena = 4×heap + ~2.5GiB 固定）。

**run#7（heap=2GB + `ASCEND_PROCESS_LOG_PATH` capture）**：与 run#5 **逐字复现** stall（8 卡对称
`completed=4/32 stuck_task_id=0x100000003`），确定性、非 flaky。默认 log level 不落 AICPU per-core stall dump
（只有 host summary）。

**决定性隔离 —— dummy 权重 + heap=2GB → `DISPATCH_CLEAN` 269s ✅**：`_stage_whole_faithful_real_device.py`
跑**同一 program `whole_decode_faithful_real`**、**同样 full-size 权重 shape（zeros）**、**同 heap=2GB**，但权重经
`compiled(*inputs)` 正常 H2D-staged device 内存（非 IPC）→ **全 45 层跑完无 stall**。证明：(a) heap=2GB 不是 stall
根因；(b) 47GB/卡 device 权重 + collective 本身能通。

**deconfound（读码，非跑机）**：`compiled(*inputs)`（device.py 干净路径）与 `prepare()+rt.run()`（IPC stall 路径）
**共用同一 `_dispatch`**（`distributed_runner.py:611`；`execute_distributed@742` 与 `_run_compiled@1395` 都调它），
prepared-vs-oneshot 只差 dep_gen 开销、不差 execution。故 runtime-path 非 confound。

**历史中间结论（后续 exact TASK/kernel 映射和修复实验已推翻其根因归因，保留仅作排查过程）**：同 program、同 `_dispatch`、同 heap、同（零）activation——当时记录的 execution-relevant 差异 =
**权重内存种类**：normal H2D-staged device（干净）vs **IPC `child_memory` peer-mapped 池**（stall layer-0 task 3）。
当时的候选解释是 whole-decode layer-0 task3 与 IPC child-memory 共存相关。offset 已验证
（pool 是 `[L,N,M]` layer-major 连续，getitem `base+k*layer_stride` 正确），故非 wrong-address。G5b 曾 device 读
IPC KV 成功（小池、0162），故非「AICore 完全不能读 IPC」——差异在 47GB 权重池规模 / peer-access flag 是否覆盖
AICore MTE / 与 comm-domain window 的 peer-access 地址空间共存。

后续 exact TASK/kernel 证据把 task3 映射到 `gate_topk`，修复 merge-sort 级联
后该 deterministic blocker 消失。因此 IPC child-memory 不是已证实根因；更后续
0162 残余随机 stall 又由 signal-layout A/B 单独处理。

**下 session 攻坚方向**（不再盲跑 12min run）：(1) dispatch-cut bisect（加 task-limit / `P_FAITHFUL_MOE_LAYERS`）
精确 pin task 3-4 是 matmul 还是 collective；(2) 核对 weight-IPC import flag = ENABLE_PEER_ACCESS 是否让 AICore MTE
可达该池（对比能通的 KV 池 import 路径）；(3) 查 comm-domain window 与 47GB 权重池 peer-access 地址空间是否冲突。

**历史净结论（不再作为当前 active root cause）**：Task① getitem **DONE + device-proven**；Task② finite-logits
当时被记录为 **Blocker B（IPC 权重× collective 共存）**。后续 0162 设备实验已证明该归因不是当前 release 的
最终根因定案。Task③（真 KV + live A/B）仍 gate 在 gap-5 HBM。arena-OOM（Blocker A）已解
（`PTO2_RING_HEAP≤2.6GB` fit）。

### Step-3 run#8 —— 历史 IPC 路径隔离记录（曾将 Blocker B 定案为 MoE `ep_all_to_all`）

**run#8（`P_FAITHFUL_MOE_LAYERS=0`，IPC 真权重，heap=2GB）→ `RESULT=REAL_WEIGHT_IPC_RUN_CLEAN` 2.17s ✅**：
MoE body 全跳过时，**N=1 IPC 全链（真 W8A8 权重 import_ipc + getitem fix + rt.run + attention/dense 的
tp_all_reduce collective）attention+dense 端到端跑通**（非只过 gate，是完整 RUN 到 RESULT=CLEAN）。

**历史中间假设（后续 exact TASK/kernel 证据已证伪其根因归因）**：`P_FAITHFUL_MOE_LAYERS=0` clean / ≥1 stall（MoE
`chip_orch` task3）→ 当时将 **Blocker B = MoE block `ep_all_to_all`（首个 all-to-all）× IPC 权重池共存**，与
attention/dense 的 tp_all_reduce（all-reduce，
已证 IPC-兼容）不同。当时只能确认问题进入 MoE task 范围，不能确认 gap 纯在 EP all-to-all。
`_ta_4`=MoE chip_orch 内序 gate→shared+tp_all_reduce→dispatch(EP a2a≈task3)→routed matmul(task4+)，stall 在
routed matmul **之前**，于是形成了“大概率 a2a”的候选。后续 task3=`gate_topk`
的证据推翻了该映射。

**历史下 session 起点（已完成，不再是当前 release 计划）**：(1) `P_FAITHFUL_MOE_LAYERS=1` 最小 MoE repro + MoE chip_orch 内 a2a-vs-matmul bisect；
(2) 查 `ep_all_to_all` comm-domain window / peer-access 与 47GB(→INT8 24GB) IPC 池共存冲突；(3) INT8-native
（routed MoE→INT8，scope=仅 3 个 matmul=整池，gap-5）同时解 Blocker A + heap 压缩，且腾 HBM 可能顺带解 Blocker B
的 a2a comm-window 挤压。Task③ 仍 gate 在 gap-5 HBM + live 8001 co-resident。
