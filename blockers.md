# 活跃 Blocker

阻塞项目进展的 open issue 的 SSOT。每条：**症状 / 根因 / 当前状态 /
解除条件 / 链接**。

Blocker 解决时，**删掉本文件里这一节**，到
[`archive/milestones-2026-Q2.md`](archive/milestones-2026-Q2.md)
"Resolved blockers" 段补一条 post-mortem。

**最后检视**：2026-07-10。

---

## ⛔ NEW 2026-07-10 — 0234 节点级跨卡 HCCL IPC poison（N=1 device dispatch + 所有多卡 e2e 硬 blocker）

**严重度**：🔴 阻塞 0234 上一切多卡 device 运行（N=1 device dispatch / Option-C 整网链 / MoE 8 卡精度）。**需要 host 级介入，容器内不可解。**

**症状**：任何多卡（2..8 卡）跨卡 comm-domain 分配在第一处 `orch.allocate_domain(...)` 即失败：
```
domain_alloc_via_ipc: [comm_hccl.cpp:833] alloc_domain: ImportByKey(peer_dr=0 pid=...) -> 507899
RuntimeError: comm_alloc_domain_windows failed with code -1  (全 N chip)
```
随后 `destroy_comm_stream ... 507018`（stream teardown 尾声，非根因）。

**决定性隔离**：已知良好 baseline `allreduce_distributed -p a2a3 -d 0-7`（以及 `-d 0-1`）**现在同样 507899** —— 该 baseline 2026-06-29 在 0234 跑过 `max|out-expected|=0`。→ **是 0234 节点级 IPC 状态卡死，不是任何一个 pypto 程序，也不是 N=1 Wall-2**。

**根因**：driver cap 在位（0234 = driver `25.5.2` / firmware `7.8.0.7.220`，非 Phase 16 cap 缺口），但运行期 driver 跨卡 IPC（`aclrtIpcMemImportByKey`）状态被某次崩溃的多卡 run 卡死，survive fresh 进程。对齐 `pypto/runtime/conftest.py:1039`："poison survives close()+device-reset on shared box；只有 fresh worker process 拿到 clean device" —— 但此处 fresh 进程也失败，poison 在 process/ACL-reset 之下（driver 级）。

**容器内恢复全部无效（已逐一验证）**：
- `npu-smi set -t reset -i N -c 0`（out-of-band）+ `-m 1`（in-band）→ `rc=214 "This command cannot be executed on a common container"`。
- `aclrtResetDeviceForce`（ctypes libascendcl，`workspace/reset_cards_acl.py`）→ 8 卡全 rc=0 **但 poison 未清**（baseline 复跑仍 507899）。
- `/dev/shm`（空）+ `/tmp`（无 hccl/rootinfo/sock）无残留可清；无 `/sys` reset 节点；`CapEff=00000000a80465fb`（无 CAP_SYS_ADMIN，非特权容器）；仅 8 卡无备用组。
- `npu-smi info` AICore(%) 85-93% + 0 HBM + 无进程 = 这批 910B2C idle telemetry 噪声（reset 后不变），非卡死 kernel。

**解除条件**：**0234 host 侧 `npu-smi set -t reset`（宿主机，非容器）或整机 reboot**。之后先用 `allreduce_distributed -d 0-7` 复验 `max|out-expected|=0`，再重跑 N=1 device dispatch（`tests/step3p5/_stage_whole_faithful_device.py`）。

**链接**：`phases/27-n1-whole-net-fusion.md` §Phase 4 device 尝试（2026-07-10）。

---

## ⭐ NEW 2026-07-09 — 融合 dense 程序 layer_idx 复用 = 整网多层 index-class bug（Track A whole-network 硬 blocker）

**严重度**：🔴 阻塞整网离线精度对齐 —— dense 层（L0/L1/L2）在多层串联时拿错权重。

> **2026-07-10 更新 — kernel 修复已 committed（pypto-lib `8b4bf3fa`），降级 🟡**：3-scalar split
> （`norm_layer_idx` abs / `attn_layer_idx` type-local / `mlp_layer_idx` dense-order）已落地，共 74 内核
> edit + callers + dense ST arity（原子 patch，reverse-review 语义 GO：index-class/arity/dispatch 全对）。
> `_smoke_program_build` rc=0。含 `layer_cache_base`（KV cache [45]abs）修正。单层行为不变（L0 三者相等）。
> **剩余 = device/多层正确性验证**：经 Option-C 整网链 vs vLLM（Item 2），非单层 ST（单层自洽、不触发碰撞；
> 且 dense ST 因 pre-existing `moe.py:208` apply_tp1_patch assert 本树无法 run）。MoE ST ScalarSpec redesign
> （w_gate_d 12-vs-3 层 OOB）deferred。

**症状**：`_stage_whole_decode_run.py` 跑 real-weights 多层链时，dense 层（L1/L2 swa_dense）会拿到错误的 attn 或 norm/MLP 权重。单层 ST（L0 / L3 / L4 …）从不暴露——因为单层 absolute-li == type-local-idx，一个 layer_idx 恰好两边都对。

**根因（本会话逐行读 `decode_layer.py` + `weight_loader.py` 定位，布局已完全确认）**：融合 dense 程序 `_build_decode_layer_dense_program`（`decode_layer.py:553`）只接 **一个** `layer_idx: pl.Scalar[INT32]`（`:688`），却要同时索引 **三类布局不同**的权重栈：
- **norm 栈**（`input_rms_weight` / `post_rms_weight` / `q_norm_weight` / `k_norm_weight`）`[LAYER_DYN=45, ...]`（`config.py:63` 编译常量）= **45 行 absolute-li**（loader `weight_loader.py:794` `for li in range(45)` 全层 append）。
- **attn 栈**（`wq`/`wk`/`wv`/`wo`/`w_g`）`[LAYER_HIDDEN_ROWS_DYN, ...]`（动态）= **type-local**（loader `KEY_WQ_FULL=[NUM_FULL_LAYERS]` / `KEY_WQ_SWA=[NUM_SWA_LAYERS]`，`:809` 按 `is_full_attention(li)` 分桶）；attention_inline 用 `layer_idx*HIDDEN` 作 base。
- **dense-MLP 栈**（`w_gate`/`w_up`/`w_down`）`[LAYER_HIDDEN_ROWS_DYN/LAYER_INTER_ROWS_DYN, ...]`（动态）= **`[NUM_DENSE_LAYERS=3]` dense-order**（loader `KEY_DENSE_GATE=[3]`，`:862` 只 append dense 层）；`_dense_mlp_body_tp` 用 `layer_hidden_base=layer_idx*HIDDEN`。

**三类 index 只有 L0 重合**（absolute=full-type=dense-order=0）。L1（abs=1/swa-type=0/dense-order=1）、L2（abs=2/swa-type=1/dense-order=2）三向发散 → 一个 `layer_idx` 无解。

**为何 per-layer 全 PASS**：dense 单层 ST 只跑一层、三 index 都=0（或匹配），从未触发；MoE 层 ST（§7.1）根本不走 dense 程序。**只有整网多层 dense 串联暴露**。

**当前状态**：bug + 精确布局已完全确认（sw-analyst 逐行验证）。**范围比初判更广**：norm 45-row 栈被 type-local `rlidx` 索引的问题**不止 dense L1/L2，还命中每个 MoE 层的 attn step**（whole-decode harness 的 standalone TP-attention 同样吃 45-row norm 栈 + type-local 索引；abs-li≠type-local 的层全错）。per-layer ST 全 PASS 是因为走 per-layer 单层（非 stacked）权重，从不触发 stacked-index。

**❌ harness-only 修法不可行（已证伪）**：三类 order（attn type-local `[12]/[33]` / dense-MLP dense-order `[3]` / norm abs-li `[45]`）互不相容，**不存在**单个 scalar 同时满足；也不能传 1-row pre-sliced（`LAYER_DYN=45` 是编译常量）。必须改程序签名。

**解除条件（唯一 root-cause 修法 = 3-scalar split，改生产 kernel）**：把融合程序的单个 `layer_idx` 拆成三个 scalar：
- `attn_layer_idx`（type-local）→ 索引 `wq/wk/wv/wo/w_g`（`attention_full.py:214/276`、`attention_swa.py` 镜像）；
- `norm_layer_idx`（abs-li）→ 索引 `input_rms/q_norm/k_norm`（`attention_full.py:258/349/361`）+ `post_rms`（`decode_layer.py:307`）；
- `mlp_layer_idx`（dense-order）→ 索引 `w_gate/w_up/w_down`（`decode_layer.py:271-272/332/380`）。
穿过 `attention_full/swa` 签名 + `_dense_mlp_body_tp` + `DecodeLayerDense.chip_orch/host_orch`（`:546/605`）+ harness `_stage_whole_decode_run.py`（dense step 传 `attn=rlidx / norm=li / mlp=dense_ctr`；MoE attn step 传 `attn=rlidx / norm=li`）。**机械改动，但改 `attention_full/swa`（所有层共用）→ 必须重跑全部 per-layer ST 回归确认无退化**。

**链接**：memory `feedback_step3p5_weight_stack_index_class.md`（本 bug 是该陷阱实例）；task #14（sw-analyst 已交付完整 3-class 定位 + 3-scalar split spec）；task #15（Track A 修 + 验证）。

**Owner**：team-lead（sw-analyst 出 3-scalar split diff → review → apply → device 验证 L0-3 vs vLLM → 全 per-layer ST 回归）。

---

## ⭐ NEW 2026-07-08 — 多程序 DistributedWorker N=8 co-prepare 死锁（whole-decode 整网 dispatch 硬 blocker）

**严重度**：🔴 阻塞整网 live single-handoff —— whole-decode worker 需把全部层程序 co-prepare 到一个 worker，N=8 时死锁。

**症状**：whole-decode worker（`_stage_whole_decode_run.py --worker`，多程序 DistributedWorker #1706）：
- prepare **3 程序**（dense L0/L1/L2）✅、**5 程序**（dense + 1 MoE Option-C）✅ 均 rc=0，residual 串接正确（30.4→53.5→64.0）。
- prepare **8 程序**（全 45 层 deduped：dense_full/swa + attn_full/swa + 4 个 MoE 变体）→ **第一次 rt.run dispatch（L0 full_dense，已验证程序）即死锁**。3 次 device 尝试：
  1. 默认 ring → `sched_error_code=100`（SCHEDULER_TIMEOUT，dev8）。
  2. `PTO2_RING_*` env raise（16GB/524288）→ `code -1`（SCOPE_DEADLOCK，dev14）。
  3. `RunConfig(ring_task_window=2^20, ring_heap=16GB, ring_dep_pool=2^20)` per-dispatch → `code -1`（SCOPE_DEADLOCK，dev8）。

**根因（wiki 定位，https://github.com/hw-native-sys/simpler/wiki/Device-Error-Codes_zh）**：
- `code -1` = **SCOPE_DEADLOCK**（编排码 1）：单个 scope 内任务数达 task_window 上限、slot 到 scope_end 才释放。`sched=100` = SCHEDULER_TIMEOUT。
- L0 dense 单独 + N≤5 均 PASS → **不是 L0 kernel bug**，是 **N=8 co-prepare 把共享 worker ring 在 prepare/init（COW pre-fork）阶段顶满**。
- **关键**：per-dispatch `RunConfig(ring_*)` **不解决** → 耗尽的是 prepare-time 的**共享 worker ring**，不是 per-dispatch ring。（`PTO2_RING_*` env 本 build 疑似不读取；RunConfig 是 per-dispatch，也没覆盖共享 ring。）

**当前状态**：机制在 N≤5 验证通过（dense + MoE Option-C dispatch、residual 串接）；N=8（全 45 层所需）死锁。cards 已全恢复（16 OK，8000 up）。**停止盲目 device 重试**（3 次失败，避免 reset 退化卡）。

**解除条件（待 team 根因，rule-5 root-cause 非绕过）**：
1. **prepare-time ring sizing**：DistributedWorker init/prepare 是否有 ring 配置入口（非 per-dispatch RunConfig）？→ hw-analyst 查 distributed_runner。
2. **上游 N-limit 修复**：#1706 是否有 co-prepare 程序数上限 / prepare-time ring 分配修复？TestMultiProgram 是否测过大 N？→ upstream-scout。
3. 若上游无解，候选（按 rule-5 优先 root-cause）：(a) prepare-time ring 配置；(b) scope 拆分（kernel 侧）；(c) 单融合程序（Phase 25 DenseChainN，perf 路线 + NaN 风险 + swa_moe 编译级联）；(d) 分批 co-prepare（≤5/批，host 侧跨批串 residual）= 明确的 work-around，仅 root-cause 不可行时用。

**链接**：memory `project_whole_model_pypto_design.md` 2026-07-08 段；harness `_stage_whole_decode_run.py --worker`（backups `/tmp/_stage_whole_decode_run.py.bak_worker{,2,3}`）。

**Owner**：未指派（team `vllm-pypto-e2e` hw-analyst + upstream-scout 调查中）。

### 2026-07-08 续 — root cause 定位 + 两条路 + path-1 fix 在验证

**N 阈值实测**：N=5 co-prepared 通、**N=6 起就 SCOPE_DEADLOCK**（`--layers 0,1,2,3,4`）。2-批 prepare **对 serving 不可行**（prepare 是 load 时一次性 + chip 常驻;不能 per-token 重 prepare,也不能在 8 卡上跑两套 8-chip worker）→ per-layer 路线**必须**把全部 8 个程序 co-prepare 到一个 worker → 必须解 N 天花板。

**为什么 `PTO2_RING_TASK_WINDOW` env 没生效（源码定位）**：`runtime/src/a2a3/runtime/tensormap_and_ringbuffer/host/runtime_maker.cpp:261`：`task_window_size = ring_task_window ? ring_task_window : parse_env("PTO2_RING_TASK_WINDOW")`。**call_config 的 `ring_task_window` 优先于 env**;多程序 worker 的 INIT call_config（`distributed_runner.py:709` `_make_call_config(dc)` 无 run_config → `CallConfig()` 默认非零 ring）压过了我设的 env。且 per-dispatch RunConfig（line 930）太晚——共享 ring 在 `_w.init()` 时已按 line-709 call_config 分配。默认 `PTO2_TASK_WINDOW_SIZE=16384`;官方 sizing 指南（`tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md:243-245`）：task_window ≥ 最大并行 scope 任务数,生产建议 65536。

**两条路（按用户「先 env,没有则改源码」）**：
- **路径1（per-layer,进行中）**：改 pypto 源码 `distributed_runner.py:709`,给 INIT call_config 注入大 ring（helper `_wd_init_ring_config()`,env `PYPTO_WD_INIT_RING=1` 默认开,ring_task_window=2^20 / heap=16GB / dep_pool=2^20）。backup `/tmp/distributed_runner.py.bak_initring`。**正在 device 验证 N=8（全 45 层）**。成 → per-layer whole-decode 解锁,复用全部已验证程序。
- **路径2（融合,备选）**：swa_moe const-fold 级联（`attention_swa.py:479` Sub + 后续 Var）→ 改 pypto EP-lowering pass 源码 const-fold + 重编。model 侧三试全死(int()/module-global/hardcode-cascade)。

**Owner**：team-lead（path-1 init-ring fix 验证中）。

### 2026-07-08 续2 — path-1 拆成两阶段：PREPARE ✅ 解决（ring sizing），DISPATCH ❌ 新 blocker（host 侧多程序派发 wedge）

**device 实测（cards 8-15，全 45 层 N=8 co-prepare，synthetic weights）把 blocker 精确拆成两个正交阶段**：

**阶段 1 — PREPARE 死锁 = ring task_window sizing。✅ 已解决。**
- `task_window=16384`（默认）**太小** → N≥6 `SCOPE_DEADLOCK`（`scope_task_count >= task_window_size`，upstream-scout 定位 `pto_orchestrator.cpp:328-353`）。
- `task_window=2^20`（init-ring 初版）**太大** → `rtMalloc failed: 207001 (size=68719477759)` ≈ **64 GiB** "pooled static arena" OOM（arena ≈ task_window × ~65536 B/task；2^20×64KB≈64GB 顶爆 64GB 卡）。
- **`task_window=65536`（2^16，生产推荐值，arena ≈ 4GB）→ `[worker] PREPARE OK`**。N=8 全 45 层 8 个 distinct 程序 co-prepare 成功。
- 修法：`distributed_runner.py:724` INIT call_config 注入 `RunConfig(ring_task_window=2^16, ring_heap=2^32, ring_dep_pool=2^16)`（env `PTO2_WD_INIT_RING_TASK_WINDOW=65536` 可调）+ harness per-dispatch `_rc` 同步降到 2^16。这是 root-cause sizing（把 ring 配到实际 scope 任务数），非绕过。

**阶段 2 — DISPATCH 派发 fault（NEW blocker）。❌ 未解决。**
- PREPARE OK 后，**第一个 dispatch step（L0 full_dense，standalone 已验证程序）即 fault**：device AICPU scheduler **空转 28s**（device log `HandleTaskTimeout ... Split kernel TaskMapSize=0`，event id[0..63] 全 value 0），派发的 L0 任务**从未到达 device AICPU** → `taskTimeout=28s` → AICore `507018` / `sched_error_code=100` / `runtime_status=-100` → 8 卡 poison + `aclrtResetDeviceForce`。
- **不是 kernel bug**（L0 standalone device PASS）、**不是 ring sizing**（PREPARE 已过、arena 已 4GB）、**不是编译**。是 **host 侧多程序 worker 在 N≥6 时派发路由 wedge**：任务没被推到 device。N≤5 dispatch 正常（residual 串接验证过）→ 纯 co-prepare **程序数** 效应。
- upstream-scout：#1706 无 N-ceiling / 派发修复可 cherry-pick；per-program state 按 `id(program)` keyed（`distributed_runner.py:667`），func_id 上限 1024（8 程序远不到）。→ 疑似 `_run_compiled`/`_dispatch` 在多程序下 per-program stream/state 选择或 orch_fn 路由的微妙 bug（`self._states[id(program)]`）。

**解除条件（更新）**：
1. **（user-preferred）融合单程序路径 — 2026-07-08 深挖后确认是架构级 blocker**：需让 fused swa_moe（`select_decode_layer(3)`）在 EP-distributed lowering 下编译过 → 单 @pl.program 45 层 pypto 自调度，绕开多程序-worker 类 bug。**根因（本会话逐点验证）**：
   - **Step A ✅**：`attention_swa.py:479` 的 config-arith `pl.full([SWA_Q_PAD_ALIGNED - Q_HEAD_BATCH_SWA, HEAD_DIM])` Sub 被 EP lowering symbolize → 改字面量 `[20, HEAD_DIM]`（32−12），保留 pad-assemble 结构，**TP attn_swa standalone 每次都编译过（TP-safe）**。
   - **Step B ❌ 架构级**：softmax scoring loop 的 dynamic `valid_shape=[Q_HEAD_BATCH_SWA, valid_len]`（`valid_len` 运行时标量）在 EP lowering 下把 `valid_len` symbolize 成 free Var → `tile.create shape element must be ConstInt, got Var`（`memory.cpp:360` at `attention_swa.py:572` row_max）。**四条 model 侧修法全部撞墙**：
     - `set_validshape(const-slice, SWA_Q_PAD_ALIGNED, valid_len)`（DSV4 `decode_sparse_attn.py:161` 模式）→ Var 仍经 fillpad→row_max 传播到 tile.create（EP 下 valid_shape **metadata** 里的 Var 也 fault；DSV4 只是没被 EP 这样 symbolize）。
     - computed column-mask（`pl.tile.ci` 索引 + `cmps(GE)` + `sel` + `col_expand` + `add`）→ 撞**根本 impedance**：step3p5 SWA scores 存在 **GM tensor 空间**（`all_raw_scores = pl.create_tensor()` at `:500`，整条 slice/mul/row_max/fillpad 都是 tensor 级），而 mask ops（`tile.ci`/`cmps`/`sel`）是 **UB tile 空间** → `col_expand: cannot mix Tensor and Tile arguments`。tile-space mask 无法 attach 到 tensor-space scores。
   - **真正解除条件**：(a) 把 SWA scoring 从 GM-tensor 重构成 **UB-tile 空间**（DSV4 式，把 QK matmul 结果留在 tile 上做 softmax）——较大 rewrite；或 (b) 上游 pypto EP-lowering 修复：不要把 tensor slice 的 runtime `valid_shape` symbolize 进 tile.create；或 (c) 上游提供 tensor-space 的 mask ops（ci/cmps/sel 的 tensor 版）。
   - WIP（Step A + 90% computed-mask，仅差 tensor/tile impedance）存 `0162:/tmp/attention_swa.py.wip_computed_mask_stepAB`；pristine 已 revert（regression 保持干净）。
   - **⭐ 2026-07-08 doc-backed 方向判断（推荐，待用户拍板）**：查阅权威文档后确认这是**正解方向**且**局部可行**（~80-120 行 model 侧 rewrite，无上游依赖）：
     - **API 文档一直存在**（本该先读）：`pypto/docs/en/user/01-language_guide.md`（三级 dispatch：`pl.*` unified / `pl.tensor.*` DDR / `pl.tile.*` on-chip；跨界 `pl.tile.load`/`store`）+ `02-operation_reference.md`。namespace 陷阱：`pl.arange`/`pl.full` = **Tensor** 版，tile 要 `pl.tile.ci`/`pl.tile.full`；`cmps`/`sel`/`col_expand` 要求全 Tile。
     - **`pypto_top_level_documents/tensor_valid_shape.md`**：`shape`(存储) 必须 512B+ConstInt，`valid_shape`(逻辑范围) **可运行时 Expr** 且自动传播（`row_max→[src.vs[0],1]`）。我们的 `tile.create got Var` = runtime `valid_len` / config 派生 `Q_HEAD_BATCH_SWA` 在 EP lowering 下泄漏进 **storage shape**（非 valid_shape）。
     - **正解 = 对齐 DeepSeek `models/deepseek/v4/decode_sparse_attn.py:139-172` 的 `qk_pv` 融合**：把 step3p5 SWA 现有 3 个 GM-scratch spmd stage（qk_matmul/softmax/sv_matmul，经 `all_raw_scores=pl.create_tensor()` at :500 中转）合并成**一个 tile-space fused spmd scope**——QK matmul 结果留 tile-local SSA（不落 GM），`set_validshape(tile, 常量 rows, 运行时 cols)`+`fillpad(PadValue.min)` 做掩码（storage shape 全 const，valid_shape 携带 runtime 列范围），只把 reduce 后的 mi/li/oi 落 GM 供 Stage-4 online-merge（Stage-4 不动）。DeepSeek 在同栈已证编译+运行通过。
     - **其余相关顶层 doc（blocker 1 dispatch wedge 参考）**：`machine_hierarchy_and_function_hierarchy.md`（sub-worker/function 层级）、`multi_level_runtime_ring_and_pypto_free_api.md`（ring）、`simpler_distributed_runtime_design.md`（多程序 runtime）、`HL_new_feature_Expand_Mixed_Kernel_and_call_spmd.md`（mixed kernel/call_spmd）。
   - **⭐⭐ 2026-07-08 KEY CRACKED — `pl.tile.load` 解掉困扰 5+ 次尝试的 EP `got Var`**：诊断确认（bias 完全 bypass、`scores=scores_scaled` 仍 fault）→ **根因是 `row_max` 作用在 GM-tensor 空间的 scores（`all_raw_scores=pl.create_tensor([BATCH*...])`）上，EP-distributed lowering 把 row 维 symbolize 成 free Var**（`SWA_Q_PAD_ALIGNED=32`/`BLOCK_SIZE=128` 都是字面量也没用——是 GM-tensor provenance 的问题，非常量性问题）。**把 score slice `pl.tile.load` 进 UB tile 后，`row_max` 输出 tile.create 变 const → EP Var fault 消失**（TP + fused swa_moe 都过了该点）。这坐实了 doc/DeepSeek 的结论：SWA scoring 必须在 **UB tile 空间**做，不能在 GM。
   - **剩余 = 纯机械 tile-op plumbing（限 Stage-2 `swa_softmax`，~20-30 行）**：`scores_tile=pl.tile.load(all_raw_scores,[scratch_row,0],[SWA_Q_PAD_ALIGNED,BLOCK_SIZE])` → `mul(scale)` → tile-space 加性掩码（`pl.tile.ci(valid_len_i,[1,BLOCK],INT32,descending=True)`→cast FP32→clamp→`*3e38`→`col_expand`+`add`；valid_len 只进 ci 的 start）→ `rmax_tmp=pl.create_tile([SWA_Q_PAD_ALIGNED,1],FP32,Vec)`；`row_max(scores,rmax_tmp)`（tile row_max 需 tmp_tile，见 `deepseek/v4/hc_pre.py:177`）→ `exp`/`row_expand_sub` → `row_sum(_, rsum_tmp)` → `pl.tile.store(exp_bf16,[scratch_row,0],all_exp_padded)` + store mi/li。Stage 1/3/4 不动。WIP（tile.load + DIAG）存 `0162:/tmp/attention_swa.py.wip_tileload_breakthrough`；pristine 已 revert。
   - **✅✅ 2026-07-08 COMPILE SOLVED — fused swa_moe + 全 7 层类都编译过**：完整修法落地并验证编译：(1) Step A(:479 Sub→字面量20)；(2) tile-space Stage-2 softmax（`pl.tile.load`→tile ops→`pl.tile.store`，加性掩码用 `pl.tile.ci` descending + `create_tile` tmp for row_max/row_sum）；(3) **`SWA_Q_PAD_ALIGNED`→字面量 32 全量 inline**（config-local var 在 EP lowering 下 tile-dim 处 symbolize 成 Var；`create_tile`/matmul-output/InitMemRef 需 ConstInt/static；`pl.tile.load`/`pl.slice` 容忍但 `create_tile`/`InitMemRef` 不容忍）。**验证：`select_decode_layer` 的 L0 full_dense / L1,L2 swa_dense / L3 swa_moe_silu / L4 full_moe_silu / L43 swa_moe_swiglu7 / L44 full_moe_swiglu16 全部 COMPILE OK**（TP attn_swa 也过，无 TP 回归）。→ **融合整网路径 compile 层解锁**（45 层都归这 7 类）。改动 `attention_swa.py` +64/-49 行，**WIP 未提交**（保 `0162:/tmp/attention_swa.py.FUSED_COMPILE_OK_20260708`）。
   - **⚠ 诚实边界 + 下一步（必做）**：仅验证 **COMPILE**，tile-space softmax + 加性掩码替换 valid_shape+fillpad 的**数值正确性未验证**。下一步：device 数值验证——`test_decode_layer_swa_dense_st`（SWA attention vs torch ref）+ swa_moe 层 vs vLLM dump（`_stage_moe_block_precision`/`decode_acceptance`）。数值过后才提交 attention_swa + 构建融合整网单程序（45 层 inline，无多程序 co-prepare → 绕开 blocker 1 dispatch wedge）。
   - **⭐ 2026-07-08 续 — 数值验证卡在 harness 设备选择 gap（非 kernel bug，未证对也未证错）**。最终重写版 = **matmul-SSA + set_validshape+fillpad**（合并 Stage1+2，scores 走 matmul-SSA → row_max/row_sum 无显式 `[32,1]` tmp，规避 hw-analyst §1/§2 对齐陷阱；用 `pl.assemble` 重赋值非 `tile.store`，规避 reverse-review C1 SSA 断链）。4-agent team 确认这是 DeepSeek-v4 `decode_sparse_attn.py:161` 规范范式。**编译过**（全 7 类）。**运行时验证两次尝试都被 harness 混淆**：(1) 原 moe_st ws=1 空 compile_cfg → **ChipWorker fork card 0**（oracle 争用 → 507018，无效测试）；(2) 加 `DistributedConfig(device_ids=[8])` 定向 card 8 → **DistributedWorker-1-rank AICPU-idle dispatch**（`TaskMapSize=0` 60s → 507018，与 blocker-1 N=8 wedge 同签名）。**控制实验**（pristine attention 走同 DistributedConfig-1-rank 路径）→ **编译就失败**（`convert_tensor_to_tile_ops_pass`，正是 valid_shape EP 问题）→ 反证**我的 compile 修复真实且必要**（pristine 走不通这条 distributed 路径，只有重写能）。**下一步（下会话，无 oracle 风险）**：修单卡验证路径——(a) 让 ChipWorker 认 `device_id`→fork card 8，或 (b) 修 DistributedWorker-1-rank AICPU-idle dispatch（关联 blocker-1）；然后数值验证重写 → 提交 → 建融合整网单程序 → 接 vLLM。WIP 存 `0162:/tmp/attention_swa.py.FUSED_COMPILE_OK_20260708`（working tree 已 revert pristine 保回归干净）；moe_st 设备定向改动存 `/tmp/moe_st.py.bak_devtarget`。
   - **⭐⭐ 2026-07-08 关键 reframe — fused swa_moe 运行时 fault = host→AICPU dispatch 问题（同 blocker-1），非我的 attention kernel**。3 次 device 验证（ChipWorker card0 / DistributedConfig-1-rank card8 / 8-card cards8-15）全部 507018，**V0 device log 全是 `HandleTaskTimeout Split kernel TaskMapSize=0`**（AICPU 空转 60s，任务从未到达 device AICPU）——**不是 kernel/对齐 fault（无 0x800/errcode）**，是 host_orch 没把 task 推到 AICPU。与 blocker-1（N=8 co-prepare dispatch wedge）**同签名同 family**。**因为 pristine attention 连这条 EP/fused 路径都编译不过（control 已证），所以 fused swa_moe 从来没在真机 dispatch 过——dispatch gate 一直存在，只是被我刚移除的 compile blocker 挡着**。含 MoE 的 fused 层 host_orch 在真机不 dispatch。→ **整网集成的真正 gate = device dispatch（host_orch→AICPU task 提交），不是 compile、不是 attention kernel**。我的 SWA 重写的**数值正确性无法验证，直到 dispatch 通**。
   - **修正后的下一步**：(a) 根因 device dispatch（为何 MoE-containing / fused host_orch 在真机 `TaskMapSize=0`）——大概率上游 simpler runtime `host_orch`→AICPU 提交路径，关联 blocker-1；先 isolate 纯 swa_dense（无 MoE）fused 层能否 dispatch（需先修单卡 device 定向让它落 8-15）。(b) 若 dispatch 通 → 数值验证 attention 重写 → 提交 → 建融合整网单程序。(c) team（reverse/hw/sw/upstream）已确认 attention 重写方法学正确（DeepSeek 范式），问题不在 kernel 内容。
   - **✅✅ 2026-07-08 定位确认 — 我的 attention 重写 device 8卡 RUNS，fault = MoE-fusion dispatch（非 kernel）**。`_stage_whole_decode_run.py -p a2a3 --tp 8 --dev-offset 8 --layers 1`（swa_dense = attention + dense MLP，**无 MoE**，我的重写）→ **`layer 1 kind=swa_dense DEVICE: PASS (21.99s) rc=0`** on cards 8-15。**证明**：(1) 我的 attention 重写 device dispatch-correct（8卡跑通，与 original swa_dense 同 known-good 路径）；(2) fused swa_moe 的 507018 **不是 attention kernel**——是 **MoE-fusion host_orch dispatch**（EP dispatch/combine 机器 + attention 合进一个 program）。moe_block 单独 dispatch ✓（memory），swa_dense 单独 dispatch ✓（现证），但 attention+MoE fused ✗。→ **whole-model 真正 gate = MoE dispatch（host→AICPU），与 blocker-1 收敛为同一根因**。**Option-C 解耦**（TP-attention 程序 + moe_block，两者都 dispatch）是可行的 whole-model 结构，gate 在 blocker-1 co-prepare dispatch。**注**：swa_dense DEVICE PASS = dispatch/runtime 跑通（rc=0，else 分支无 golden）；attention 数值 golden 对比仍需 perrank golden 路径。attention 重写 WIP 未提交（`/tmp/attention_swa.py.FUSED_COMPILE_OK_20260708`；working tree pristine）——Option-C 不需要它（original attention 走 Option-C 即可），fused 需要它但 fused 不 dispatch，故暂不入库。
2. **（per-layer 路径）修 host 侧多程序派发 wedge**：需 V0 stuck-task 深挖 + 大概率上游 simpler runtime `_dispatch`/worker task-submission 在 N≥6 的修复。
3. **（validation-only 权宜）** per-layer 独立 prepare+run+release（非 co-prepare，`--chain` 已验证 dense 前缀 residual 串接）：但 MoE 层 `select_decode_layer` 返回 fused swa_moe（编译不过）→ 仍需 (1) 的 swa 重写，或用 Option-C 每层独立 prepare 两程序（attn + moe_block）。

**Owner**：team-lead（决策 fused vs per-layer；sw-analyst 查 swa 重写、hw-analyst 查 arena/HBM budget、upstream-scout 已交付）。

### 2026-07-08 续3 — ⭐ 决定性答案：DeepSeek 不 fuse attention+MoE，用 separate programs（fused 是错的结构）

**用户问「DeepSeek 不是这样实现的么？他们没这个问题？区别是什么」→ agent 深挖纠正了前提**：
- **DeepSeek V4 根本不 fuse attention+MoE**。`models/deepseek/v4/moe_ep.py:175-234` 的 decode host_orch **只有 MoE**（hc_pre→gate→shared→dispatch→routed→combine→hc_post，无 attention）；attention 是 `@pl.jit.inline` sub-kernel（`decode_attention_swa.py`），**无 host_orch**；V3_2 把 decode 拆成**独立的 `decode_front`(attention) + `decode_back`(MoE) 两个 program**。**DeepSeek 把 attention 和 MoE 当两个独立 program 跑——所以它没这个问题。**
- **区别**：step3p5 的 fault 来自把 **TP-attention + EP-MoE fuse 进一个 `chip_orch`**（`decode_layer.py:2629-2860`，11 个 TP+EP 混合 comm window），DeepSeek 从不这么做。TP attention 需要 `tp_all_reduce` collective + EP MoE 需要 a2a——两种 collective 协议塞进一个 program 的 host_orch，pass37 comm-domain + inline-attention 交互 → host_orch 产 0 个 chip task → `TaskMapSize=0` / 507018。

**device 证据（8卡 cards 8-15，two-method）**：`full_silu_silu --two-method` **`next_hidden_out` PASS**（ratio_allclose atol=0.04；`h_mid_out FAIL` 是 Phase 25 已记录的 benign readback artifact，非 bug）；`swa_silu_silu --two-method` **507018**（swa-specific dispatch fault，dispatched 8 chip 但 runtime fault）。→ two-method（一个 program 两 method）对 full 通、对 swa 挂；但**真正对的结构是 DeepSeek 的 separate programs**。

**⭐ 修正后的 whole-model 路径（mirror DeepSeek，收敛所有发现）**：
- **用 Option-C 解耦**（step3p5 已有）：`_build_tp_attention_{swa,full}_program`（独立 TP-attention program）→ resid1 经 GM → `select_moe_block`（独立 moe_block program）。**两个独立 program**，正好 mirror DeepSeek 的 decode_front/decode_back。
- **两部分都已单独 dispatch 通**：swa_dense/TP-attention 8卡 ✓（本会话证），moe_block 8卡 ✓（memory）。**且用 original attention 即可**——`_build_tp_attention_swa_program` 用 original attention_swa 早就 COMPILE PASS（2026-07-07 Option-C 45/45）；valid_shape EP symbolize 只在 **fused** swa_moe 触发，standalone TP-attention 不触发。**→ 我的 fused-swa-moe 重写不是整网路径所需**（fused 是错结构；重写是探索 fused 时的产物，保留在 /tmp，可留作 fused 若未来需要）。
- **whole-model 真正 blocker = Option-C 45 层的 co-prepare dispatch = blocker-1**（N=8 co-prepare PREPARE 已解 ring-sizing，DISPATCH 在 N≥6 wedge）。整网集成收敛到**单一 blocker：Option-C 多程序 co-prepare 的 host→AICPU dispatch（blocker-1）**。

**下一步**：(a) 根因 blocker-1 co-prepare N≥6 dispatch（host→AICPU，同 `TaskMapSize=0` family）——或 (b) 分批 co-prepare（≤5/批，host 侧跨批串 residual，作为 validation-first 权宜）跑通 Option-C 45 层整网 → 数值对齐 vs vLLM → 接 single-handoff。attention 用 original（不需 fused 重写）。

### 2026-07-08 续4 — ⭐ 真根因轴 = distinct-program 个数 N（用户调研 + 定位），Option-C dispatch 验证 + 数值 handoff 要求

**用户调研纠正根因轴**：不是"per-layer vs 融合"，而是 **distinct program 个数 N**。两种 multi-program：
- **A 复用型（DeepSeek）**：只编几个 distinct block-type program（v3.2 front+back=2；v4 attn/moe_ep/lm_head≈4-6），同一 compiled program 按层反复 `rt.run`，**per-layer 权重走 runtime tensor 参数（不 bake 进编译）** → N=block-type 数=2~6，永远 <6 墙。
- **B 一层一程序（step3p5 初版）**：每层 co-prepare distinct program（45×(attn+moe)≈87）→ N 线性爆 → N≥6 死锁。

**代码确认**：step3p5 activation 是**编译期 baked**（`decode_layer.py:1918-1946` 模块级 `_routed/_shared_swiglu_step` 布尔；注释：inline `pl.if_` 在 @pl.program body 不支持）→ 4 MoE 变体=4 distinct program。dedup 后 N≈7-8（full_dense/swa_dense/attn_full/attn_swa + 3 activation MoE block），仍 >6。

**device 验证（8卡 cards 8-15，`_stage_whole_decode_run --worker`）**：Option-C 链 **N=4（L0,1,3）PREPARE OK + all 4 steps dispatched + rc=0** ✅（synthetic）；N=5 验证中。→ Option-C 多层链在 N<6 时 dispatch 通。

**⭐ reverse-review：Option-C MoE handoff 数值要求（现 worker 缺，synthetic 0.0 没暴露）**：
- **C1 CRITICAL**：standalone `select_moe_block`(EpTpMoE) 收**已 norm 的 x**，但 standalone attention program 输出**未 norm 的 resid1** → worker 直接喂 resid1 **跳过 post-attention RMSNorm** → router/expert 输入差 RMS 因子。修：worker 在 attn→moe 间加 post_rms（或扩 EpTpMoE 自带 norm，像 `_dense_mlp_body_tp`）。
- **C2 CRITICAL**：standalone moe_block 返回 `moe_out`（routed+shared）**不含 residual** → worker 须 `next_hidden = resid1 + moe_out`（**FP32** add，对齐 fused 的 FP32 accumulate）。
- **H2**：standalone moe_block **有 W8A8 routed-input INT8 quant 修复**（commit 3b236e6），fused 版没有 → W8A8 下 Option-C(standalone) 反而更对。dense 层 handoff 正确（自带 norm+residual）；shape/dtype/align 干净；跨批 BF16 residual copy bit-exact。

**完整路径（全 mapped）**：(1) **dispatch**：N<6（Option-C reuse；silu 层 N=5，swiglu 2 层分批或收 activation-runtime）；(2) **numerical**：Option-C MoE handoff 加 post_rms + FP32 residual（C1/C2）；(3) 真权重 golden vs vLLM；(4) vLLM single-handoff。attention 用 original。

**下一步**：worker Option-C MoE step 补 post_rms + FP32 residual（C1/C2）→ N≤5 跑 silu 层子集真权重数值对齐 → 补 swiglu 层 → 全 45 层 → vLLM handoff。或先根因 N≥6 墙（sw-analyst 查阈值可调性 + 分批 create/close 可行性）。

### 2026-07-08 续5 — dispatch 阈值实测 + sw-analyst 根因 + 完整 batched 计划

**device 实测（8卡 cards 8-15，131072 ring tier）**：
- **N=6（6 distinct, 7 steps, L0-4）→ dispatch OK rc=0** ✅
- **N=8（45 层, 8 distinct, 87 steps）→ PREPARE OK（131072 修好 prepare OOM）但 DISPATCH 首步 507018** ❌
- → **dispatch 墙在 6~8 之间**（N=6 通，N=8 挂）；ring tier 修好 prepare 但**修不了 dispatch**。

**sw-analyst 根因（`worker.py:1971-2121`）**：N≥threshold dispatch wedge = **co-prepare fork-then-prewarm 协议的结构性 race**（非可调 limit；MAX_REGISTERED_CALLABLE_IDS=64 远够）。首次 `run()` 后 N 个程序的 pre-warm `_CTRL_PREPARE` fan-out 与 dispatch 的 `TASK_READY` race → chip child 没推进 → task 不到 AICPU → `TaskMapSize=0` → 60s timeout → 507018。**根 fix = pre-warm loop 加 per-chip prepare-ack barrier + timeout**（上游 simpler runtime）。
**batched co-prepare VIABLE（sw-analyst 代码确认）**：`create→prepare≤5→run→close→recreate` 可行（chip child 在 close 时 reaped，per-process statics 随 child 死，无全局阻塞）；坑：批间 `close()+del+gc.collect()` 先 reap 再 fork（memory `feedback_verify_processes_killed_before_launch`）。

**⭐ 完整 whole-decode 计划（全 root-caused，无 research 剩余，纯实现）**：
1. **dispatch**：batched Option-C，每批 distinct ≤6。真模型 activation 分布：silu 主导（~40 层）+ swiglu7_silu(L43) + swiglu7_swiglu16(L44)。批1 = {full_dense, swa_dense, attn_full, attn_swa, moe_silu}=5 distinct 跑 L0-42；批2 = {attn+moe_swiglu 变体} 跑 L43,L44。host 跨批串 residual（BF16 bit-exact）。
2. **numerical handoff（reverse-review C1/C2）**：Option-C MoE step 补 post-attention RMSNorm（moe_block 收 normed x，attn 出 un-normed resid1）+ FP32 residual add（`next_hidden=resid1+moe_out`）。清洁做法 = 扩 EpTpMoE 自带 norm+residual（mirror fused DecodeLayerMoE post-attn 段 `decode_layer.py:2726/2844` + `_dense_mlp_body_tp`），不加 distinct program。
3. 真权重 golden vs vLLM（silu 子集先，再全）。
4. vLLM single-handoff + live A/B。
attention 用 original（fused 重写非所需，存 /tmp 备 fused 若未来修 fork-prewarm race）。

### 2026-07-08 续6 — ⭐ 定位收敛：N≥6「墙」大部分是 over-counting，真实 N=7 device 跑通

**用户要求「先把 block 到底是什么限制定死再进下一阶段」。4-agent 交叉验证 + team-lead 追踪，结论如下：**

**(1) 之前的 distinct 程序数记错了。** `select_moe_block(li)` 是 **attention-agnostic**：对 L3(swa_moe_silu)/L4(full_moe_silu)/L5/L8 返回**同一个 program 对象**（`id()` 实测相同）——silu moe_block 是**一个共享程序**，swa/full 通用。真实整网 Option-C distinct = **7**（`full_dense`, `swa_dense`, `attn_full`, `attn_swa`, `moe_silu`(L3-42), `moe_swiglu7_silu`(L43), `moe_swiglu16`(L44)），**不是 8**。文档/harness 记成 8，是因为 `_stage_whole_decode_run.py` 的 `moeblk_cache` 按 `kind`（含 swa/full）做 key，把同一个 silu 程序编译两遍 → 多算 1 个。修法：按 `id(select_moe_block(li))` 去重。

**(2) 真实 N=7 device 跑通。** 探针 `pypto-lib/_probe_Nsweep_v0.py`（cards 8-15, TP=8, V0, synthetic）：编译 7 个真实 distinct 程序 → `c0.prepare(extra_compiled=[其余6])` → 派发 program-0(full_dense)。结果 **`PREPARE OK N=7 -> DISPATCH OK N=7 rc=0 -> SUCCESS`，clean finalize，无 507018、无 reset**。log `/tmp/probe_n7.log`。-> **真实 7 程序整网装得下单 worker，live-serving 可行，无需分批、无需改 runtime、无需 per-token re-prepare**。实测墙 = N=6 通 / N=8 挂 / **N=7 通**（之前从没测过 7）。

**(3) 已 file:line 否定的 N-scaling 候选**（勿再查）：comm-domain window（同名 `comm_d0` + dup-live-reject + 每次 alloc `aclrtMemset` 清零，per-dispatch time-multiplexed）、tensormap/ring/arena（per-Worker，无 program-id 字段，xMAX_RING_DEPTH 非 xN）、fork-prewarm（阻塞 ack-barrier）、per-program state select（对象 key dict）。**用户的「signal window + GM 通信缓冲区 O(N) 池撑爆」假设被代码否定——没有这样一个固定池。**

**(4) 诚实边界 / 遗留**：探针只派发了 program-0 一次；还需验证**完整 7 程序链**（各层类型顺序派发 + residual 串接）。「为什么 N=8 挂」的确切设备侧天花板（hw: IPC export-key/handle 表；sw: AICPU prepared-identity/slot 表）**未 micro-pin**，需 device bisect——只在将来 co-prepare >=8 程序时才要紧（tail TpRmsLmHead 作第 8 个会撞，但 tail 走独立 live compute_logits 路径，故整网层保持 7）。

**链接**：memory `blocker1_coprepare_wall_overcounting_N7.md`。**Owner**：team-lead（进入下一阶段：完整 7 程序链 + 数值 handoff C1/C2 + 真权重 + tail + live A/B）。

---

## 0. Phase 20 production backend 未接入

**严重度**：🟡 功能 —— dump-based 精度闭环已完成，但真实 vLLM 请求还没有走 PyPTO NPU full runner。

**症状**：当前 BF16/W8A8 decode 与 W8A8 prefill 的结论来自 vLLM eager detail dump + PyPTO reference/detail/final-logits replay；这证明数值路径与权重翻译口径可对齐，但还不是 production backend。

**根因**：`Step3p5DecodeFwd` / prefill runner、vLLM `Step3p5Model.forward` monkey-patch、runtime weight bundle 注入、KV cache / block table / slot mapping ABI 尚未接入成一条在线请求路径。

**解除条件**：Phase 20 落地：

1. `config_align.py` 校验 vLLM `hf_config` 与 PyPTO constants；
2. `weight_translate.py` 支持 vLLM module → PyPTO bundle；
3. runner 接入 vLLM 请求路径，至少 decode-only 能返回 token；
4. Phase 21 在线 L1/L2/L3 precision gate 通过。

**Owner**：未指派。

---

## 1. head_gate × 1 旁路 —— 跟 vLLM 原生精度对齐

**严重度**：🟡 精度 —— gate Phase 21 L1（per-layer hidden_states）严格
对齐。**不**阻塞 v0.1 / v0.2 功能 bring-up；只是"精度验证全绿"准出条件
的一部分。

**症状**：`attention_full.py:658-690` 和 `attention_swa.py` mirror 用
`attn_out_gated = attn_out`（× 1 identity），不是
`attn_out_gated = attn_out * sigmoid(head_gate_logits)`。每层 attention
输出大致是上游期望值的 2 倍（`sigmoid` 平均输出 ~0.5）。

**根因**：pypto kernel 没法表达 head_gate 操作而不触发
`pl.row_expand_mul([N, K], [N, 1])` 在 1 列 FP32 操作数上 — 这会撞 AIV
32-byte 行对齐限制。这是 pto-isa 硬限制，model 侧无干净绕路。

分类参考：`pypto-lib/docs/known-pypto-pitfalls.md` §1。

**跟踪**：TASK-L（pto-isa 上游 — 用 cube-matmul 配 block-diag R 矩阵
构造）。在 backlog 里跟踪。

**解除条件**（任一，按优先级）：

A. 上游 pto-isa 落 `[N, 1]` slice 32-byte 静态对齐 reject（§1 doc 提到）
   **同时**我们在 attention_full / attention_swa 用 cube-matmul × block-
   diag R 构造表达 head_gate，避免 intra-UB `[N, 1]` Vec tile。
B. Phase 21 §2.7 标定 —— patch 上游 vLLM `Step3p5Attention` 也走 × 1
   identity（语义上丢掉 gate）。失去 ~2× attention scaling 对生产意义不
   利，但允许 L1 ratio_allclose 在两个（同样降级的）实现之间通过。
C. 拓宽 Phase 21 L1 容忍区间，吸收 attention-output-only 路径 ~50%
   magnitude 差。less rigorous；记录差距即可。

**估时**：
- 路径 A：周（上游 gate）
- 路径 B：1-2 天（vLLM 侧 patch + 重跑）
- 路径 C：0.5 天（tolerance 配置改 + 重 baseline）

**Owner**：TASK-L 上游；项目侧决策待定。

---

## 2. Prefill MoE L1 overflow (TASK-29)

**严重度**：🟢 Deferred —— gate Phase 17（完整 prompt processing
e2e），**Phase 22 decode-only perf 不需要**。

**症状**：`models/step3p5/prefill_moe.py` 编译时 L1 buffer overflow
（~5 MB > 限）在 `moe_gate_up` MLP。Prefill MoE 层编译不过。

**根因**：Prefill 在很宽的 SEQ 维度上跑（如 SEQ=4096 vs decode BATCH=16），
decode UB 装得下的 MoE kernel 结构到 prefill 会爆 L1。

**跟踪**：TASK-29 in backlog。

**解除条件**：重设计 prefill_moe，加 multi-step gate_up chunking。~1-2
周专门工作。

**Phase 22 decode-only perf 的绕路**：用合成数据预填 KV cache 到目标 input
length，跳 prefill，测 decode-only TPS / ITL。详见
[`phases/22-perf-baseline.md`](phases/22-perf-baseline.md) "Prefill
workaround"。

**Owner**：未指派。

---

## 7. out_proj Vec-LHS → 910B 非法 Mat→Mat tmov（decode 已绕过；prefill perf 待根因）

**严重度**：🟢 已绕过（decode 不受影响）；根因修复 gate prefill 性能。

**症状**：升级到 stepfun/develop latest（pypto `5e619dc7`）后，step3p5
`full_out_proj_matmul` 编译报 `'pto.tmov' op expects a supported tmov
address-space pair for this target`。

**根因（team `vllm-pypto-e2e` 4-agent 定位，2026-07-10）**：out_proj 的
LHS `attn_out` 是 UB/Vec-resident；tiler 走 #1601 `stage_lhs_to_mat`
（`auto_tile_matmul_l0_pass.cpp:694`）→ Vec→Mat staging → 非法 Mat→Mat
`tmov`。910B **无 L1→L1 DMA**（`#1960` 只检测不修复），是 ISA 硬限制。
N=256 时 cube RHS `[256,256]` BF16 = 128KB 超 L0B 64KB，tiling 是必需的，
staging 因此被迫触发。

**已绕过（decode 生产路径 OK）**：`OUT_PROJ_N_CHUNK 256→64`
（`config.py:294`，commit pypto-lib `d3075ac9`）——RHS 降到 32KB 原生放得下
L0B，不再 tile-staging，无 tmov。数值 parity-safe（K-reduction 不变）。
MoE per-rank compile rc=0。

**根因修复（deferred → Phase 17/22 prefill perf）**：chunk=64 相比 Qwen3
canonical N=256 损失 L1 利用率，prefill（大 batch）代价更明显。两条候选，
**arch-gate 单独不行**（hw-analyst：跳过 Vec-LHS staging 会重新触发 L0B 溢出，
三条出口全堵）：
- (a) 对齐 **Qwen3-14B** `decode_layer.py:880-904` 的 out_proj = split-K×split-N
  **atomic-add** 结构（N=256/512，`OUT_TN=512`），从根上不走 Vec-LHS staging；
- (b) 上游 `ExpandMixedKernel` 把 #1601 GM-pipe Mat→Mat copy-out 换成合法路径
  （defer-tfree / per-chunk-pop）——多 session codegen 工作。

**解除条件**：prefill kernel 落地时（TASK-29）选 (a) 或 (b)，把
`OUT_PROJ_N_CHUNK` 恢复到 256 且编译通过。

**链接**：`deployment/troubleshooting-mat-mat-tmov-vec-lhs-matmul.md`（前会话
"试过但不行" arch-gate 记录）；memory `feedback_codegen_issue_platform_gate.md`。

**Owner**：未指派（deferred）。

---

## 5. 机器 0234 driver+firmware 升级

**严重度**：🟢 基础设施 —— 备用部署机。**不**阻塞 0162 上的 Phase 2 工作。

**症状**：0234 driver `25.5.1` / firmware `7.8.0.6.201` /
CANN `9.0.0-beta.1`。`support_shmem_map_exbus=0` cap 还在因为
driver+firmware 都低于 Phase 16 minimum（`25.5.2` / `7.8.0.7.220`）。
跨卡 `aclrtIpcMemImportByKey` 返回 507899。0234 上跑多卡 e2e 不可能。

**根因**：标准 Phase 16 部署需求还没在 0234 上应用。

**解除条件**：按 [`deployment/machine-recovery.md`](deployment/machine-recovery.md)
跑升级。两个 `.run` 包已 stage 在 0162 `/mnt/persist/ascend-staging/`：

```
Ascend-hdk-910b-npu-driver_25.5.2_linux-x86-64.run
Ascend-hdk-910b-npu-firmware_7.8.0.7.220.run
```

scp 到 0234，停 kubelet，跑 driver `--upgrade --quiet`，重启，搞定。

CANN 在 0234 上**已经是正确版本** —— **千万不要**跑可能 revert 到 GA
的集群自动化（升级前把 beta.1 install 备份到 persistent storage）。

**估时**：~2 小时 wallclock（含重启）。

**Owner**：未指派。

---

## 6. (Deferred) MTP 集成进 decode_fwd

**严重度**：🟢 Deferred —— speculative decoding 吞吐倍率。**不在** Phase 2
关键路径上。

**症状**：3 个 MTP 层有 kernel（`models/step3p5/mtp.py`）但没拼进
`decode_fwd`。vLLM 的 MTP 路径期望 1 main token + N speculative tokens
+ verification accept/reject，accept rate 高时给 ~3× 吞吐。

**根因**：没建过；Phase 1 期间为了聚焦关键的 45-layer dense+MoE 路径
deferred 掉。

**解除条件**：Phase 23 设计（TBD）—— 把 MTP 拼到 `decode_fwd` 输出阶段；
跟 vLLM speculative decoding pipeline 集成。

**估时**：Phase 22 baseline 出来后 2-4 周。

**Owner**：未指派，deferred。

---



## 怎么加新 blocker

1. 在最 deferred 项的位置之前插一节，选对严重度图标。
2. 顺序编号（不复用老编号）。
3. 从新节链回去症状第一次出现的地方（某 phase doc / `archive/milestones-
   2026-Q2.md` 里的某次 session log 等）。
4. 在 [`STATUS.md`](STATUS.md) "硬 Blocker" 表加一行。
5. 如果 gate 某个具体 phase，从那个 phase doc 的 "Risks" 段链过来。

## 4. Final e2e precision prerequisites

**严重度**：🔴 Critical —— gate 最终验收“端到端精度正确且无阻塞”。

**当前预检命令**：

```bash
cd <pypto-lib>
python tools/step3p5/e2e_precision_readiness.py --batch 2
```

**2026-06-24 结果**：host 级 smoke 全绿，但最终 e2e 精度仍被以下前置条件阻塞：

1. 0162 未挂载默认真实权重目录 `/mnt/chensiyu-jfs/multi-hardware/models/step3p5_flash_release_hf_mtp3_bf16`。
2. 当前环境未发现 vLLM / stepcast 原生 Step3p5 模型代码或 Python package。
3. `Step3p5DecodeFwd.host_orch` 仍是 final RMS + LM head skeleton，尚未 wire 45 层 per-layer program。
4. head_gate 当前在 PyPTO 侧是 ×1 bypass；vLLM parity 需要同策略 patch 或明确接受 L1 差异。

**解除条件**：真实权重 + vLLM oracle 可见；`decode_fwd` 45 层接线完成；能导出同一 decode step 的 hidden/KV/cache/slot 输入；8 rank logits shard concat 后与 vLLM logits/top-k 对齐。

