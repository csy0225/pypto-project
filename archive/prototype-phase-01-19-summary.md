# PyPTO Kernel 原型 —— 开发归档（Phase 01-19）

本文档归档 pypto step3p5 kernel 原型开发之旅，从初始设计（Phase 01）到
MoE 单卡 ST blocker 识别（Phase 19），时间跨度 2026 年 5-6 月。

后续工作（Phase 20+：vLLM Ascend 后端集成 via monkey-patch）跟踪在：

- 实时跟踪器：`<workspace>/pypto/CLAUDE.md`（外部，本机本地）
- 仓内 phase docs：`pypto-project/phases/20+`
- 跨阶段遗留 blocker：[`../blockers.md`](../blockers.md)

## 建了什么

pypto step3p5 kernel 套件 —— 在 Ascend NPU（910B/C 平台 target `a2a3`）
上、基于 pypto / pto-isa / simpler / PTOAS 工具链栈实现的 48 层
（45 hidden + 3 MTP）decoder。组件清单：

| 组件 | 文件 | 角色 |
|------|------|------|
| Attention（full / SWA） | `attention_full.py`, `attention_swa.py`, `prefill_attention_*.py`, `prefill_qkv_proj_rope.py` | QKV proj + RMS-norm + RoPE + paged KV-cache + flash-attention；按 layer-types 表分 full / SWA |
| MoE block | `gate.py`, `dispatch.py`, `expert_routed.py`, `expert_shared.py`, `combine.py`, `moe.py`, `prefill_moe.py` | Top-k routing + EP all-to-all + 每 rank 36 expert routed + shared expert + EP all-to-all 回 + 加权 gather |
| Decode layer dispatcher | `decode_layer.py` | 每层 dense vs MoE 路由，via `select_decode_layer(layer_idx)` |
| Decode forward composer | `decode_fwd.py` | 融合 45 层 + 最终 RMS + `rms_lm_head.py` 每 rank lm-head 切片 |
| MTP | `mtp.py` | 3 个 next-token-predict 层（未拼进 decode_fwd） |
| Prefill 家族 | `prefill_fwd.py`, `prefill_*.py` | per-layer + composer，处理初始 prompt |
| Weight loader | `weight_loader.py` | HF safetensors → 30-key 每 rank flat-tensor bundle（`expected_shapes()` 支持任意 TP world size） |
| 顶层入口 | `step3p5_decode.py`, `step3p5_prefill.py` | CLI：smoke（CPU torch reference）+ `run_real_npu` |

所有文件在 `pypto-lib/models/step3p5/` 下。

## Phase 时间线

| Phase | 标题 | 状态 | 完成 |
|------:|------|------|------|
| 01 | Config baseline + 迁移计划 | ✅ | 2026-05 |
| 02 | Checkpoint shape 验证（jfs ckpt vs config.py） | ✅ | 2026-05 |
| 03 | 单层 decode 草稿（full / SWA） | ✅ | 2026-05 |
| 04 | Parametric attention + decode_layer dispatcher | ✅ | 2026-05 |
| 05 | MoE block（单卡） | ✅ | 2026-05 |
| 06 | decode_fwd 45 层 + lm_head | ✅ | 2026-05 |
| 07 | MTP（3 层） | ✅ | 2026-05 |
| 08 | Prefill（单卡） | ✅ | 2026-05 |
| 09 | E2E 集成 + smoke + weights | ✅ | 2026-05 |
| 10 | TP=8 + EP=8 重构 | ✅ | 2026-05 |
| 11 | Driver / install.md 清理 | ✅ | 2026-06-04 |
| 12 | 前端 bring-up rc=0（10 X-phase 子任务） | ✅ | 2026-06-04 |
| 13 | Re-sync 到最新 commit + smoke 重验 | ✅ | 2026-06-05 |
| 14 | Pypto codegen pass（IR → PTOAS bytecode） | ✅ | 2026-06-08（14.C-14.G ✅；prefill 单层 deferred 到 Phase 17） |
| 15 | 单 rank NPU bring-up | ✅ | 2026-06-15（rc=0；20 task complete；head_gate ×1 旁路 + TP=1 patch + LAYER_*_ROWS_DYN override） |
| 16 | 多 rank NPU + 真权重 load + 工具链升级 | ✅ | 2026-06-19（driver 25.5.2 + firmware 7.8.0.7.220 + CANN 9.0.0-beta.1；simpler L3 allreduce 双卡 golden match） |
| 17 | 64K prefill + 16 步 decode e2e | ⏸ | 被 prefill MoE L1 overflow 卡（TASK-29） |
| 18 | Performance: l2_swimlane + PMU | ⏸ | Deferred 到 Phase 22（vLLM 集成阶段） |
| 19 | MoE 单卡 ST + 精度对齐 | ⏸ | 2026-06-17（Blocker 1/2/3/4 ✅，MoE device runtime 507018 ⏸；6 variants smoke 6/6 PASS，dense ST device PASS） |

Phase 01-19 的 phase docs 在外部 tracker
`<workspace>/pypto/docs/step3p5/phases/`（loose，未版本化）。从 Phase 20
开始决定 phase docs 也版本化，但 01-19 没有迁过来。

## 硬件平台验证

### Phase 16 多卡部署要求（生产关键）

任何生产多卡 step3p5 run 的最低部署是实时 tracker 文档的 **三剑合璧**
绑定。2026-06-22 起 `gpu-a910x-0162` 是唯一三件全装的机器：

| 组件 | 必需 | 旧版本失败模式 |
|------|------|---------------|
| Driver | `25.5.2` | 旧：`support_shmem_map_exbus=0`，`aclrtIpcMemImportByKey` 返回 507899 |
| Firmware | `7.8.0.7.220`（chip flash，跨重启持久） | 同 cap 缺口 |
| CANN | `9.0.0-beta.1`（NOT GA — GA 让 simpler init 507018 BootstrapDispatcher 失败） | GA：TDT 不推 AICPU `libaicpu_extend_kernels.so` |

部署 runbook 见 simpler 仓 fork 的
`runtime/.claude/skills/ascend-phase16-deploy/SKILL.md`。

### 单卡验证结果（2026-06-22 重启后重验）

| 测试 | 路径 | 状态 | 时间 |
|------|------|------|------|
| 前端 smoke | `_smoke_program_build.py` | ✅ rc=0 | <2s |
| Phase 16 baseline | `simpler L3 allreduce_distributed -p a2a3 -d 0-1` | ✅ `max\|out-expected\|=0` | 秒 |
| ST-1 dense full | `test_decode_layer_full_dense_st -p a2a3 -d 0` | ✅ ratio_allclose PASS | 7.93s |
| ST-2 dense swa | `test_decode_layer_swa_dense_st -p a2a3 -d 0` | ✅ ratio_allclose PASS | 14.85s |
| MoE 6 variants smoke | `test_decode_layer_moe_st --variant ... --smoke` | ✅ 6/6 编译干净 | 各 <1s |
| MoE device runtime | `test_decode_layer_moe_st --variant full_silu_silu -d 0` | ⏸ 5s 内 507018 | (faults) |
| 单卡 e2e（Phase 15） | `tools/p15_trace/run_with_trace.py` | ✅ rc=0，20 task complete | 6.69s |

## 选 milestone（session log 压缩版）

### Phase 15 单卡 e2e 解锁（2026-06-15）

到 rc=0 需要的三个层叠修复：

1. `attention_full.py:658-690` —— `full_head_gate` 旁路成 identity
   （`attn_out_gated = attn_out`）。根因：`pl.row_expand_mul` 在 `[N, 1]`
   FP32 操作数上撞 AIV 32-byte 行对齐。Model 侧无干净绕路；正解走上游
   pto-isa 用 cube-matmul 配 block-diagonal R 矩阵（TASK-L）。旁路丢
   sigmoid gate 语义但解锁 e2e 管道。
2. `tools/p15_trace/run_with_trace.py` —— 加 `--tp-world-size 1` 触发
   `step3p5_decode.run_real_npu` monkey-patch 路径：reload
   `attention_full` / `decode_layer` 模块 with `TP_WORLD_SIZE=1` /
   `EP_WORLD_SIZE=1`。Codegen 消掉 `tp_all_reduce` kernel。
3. `step3p5_decode.py:391-414` —— TP=1 patch 把 `LAYER_INTER_ROWS_DYN`
   和 `LAYER_QHIDDEN_ROWS_DYN` 设到 TP=8-derived 值，绕开 pypto 上游
   bug #3/#4（`pl.dynamic` 首维 slice 丢父 stride / 幻 int32 kernel
   参数）。

### Phase 16 多卡解决（2026-06-19）

IPC `support_shmem_map_exbus=0` blocker（filed as simpler#1037）是
**driver 能力缺口**，不是代码 bug。解决：

1. Driver 25.0.rc1.2 → 25.5.2 升级。
2. Firmware 7.7.0.3.220 → 7.8.0.7.220 升级（写 chip flash，跨主机重启
   持久）。
3. CANN 必须是 `9.0.0-beta.1`，NOT GA（GA 的 TDT 不推 AICPU
   `libaicpu_extend_kernels.so`，会让 simpler init 崩）。
4. simpler `comm_hccl.cpp` 加 `__has_include` 守护的 `*Inner` macro
   alias，对 CANN GA forward-compat（beta.1 下 no-op）。

端到端验证：`aclrtIpcMemImportByKey + ENABLE_PEER_ACCESS` 跨卡 rc=0、
`peer_va == parent ptr`；simpler L3 `allreduce_distributed` 双卡
`max|out-expected|=0` golden match。

### Phase 19 MoE blocker 解决（2026-06-17）

6 MoE variant 识别出 6 个独立 blocker：

| Blocker | 描述 | 解决 |
|--------:|------|------|
| 1 | PTOAS v0.44 `pto.tci ui32 {descending=false}` parser bug | ✅ 升级 ptoas-bin v0.45（commit `caf57c50`，含上游 fix `505abd64`） |
| 2 | sh_mlp / gate_matmul L1/UB overflow | ✅ 是 shape-choice artifact：`apply_tp1_patch`（全宽 unsliced）vs canonical TP=8 per-rank（8/12/1/1408/160/36）。per-rank 路径干净。 |
| 3 | dispatch.py 32B 对齐 —— `PER_RANK_BUCKETS` 不 pad-8 | ✅ 加 `PER_RANK_BUCKETS = pad8(N_RANKS * N_LOCAL_EXPERTS)` + `N_RANKS_PAD = pad8(N_RANKS)`，跨 5 文件 mirror |
| 4 | gate_topk / moe_combine 不支持 CCEC bf16 类型转换 | ✅ 把 `expert_weights` 从 BF16 切 FP32，跨 6 个 emission 点；gate 本来内部就 FP32 |
| 5 | MoE device runtime 507018 | ⏸ **仍 open —— 见 [`../blockers.md`](../blockers.md)** |
| 6 | MTP wrapper 集成 | ❌ Deferred（对 dense 和 MoE ST 无影响） |

Blocker 1-4 后：**6 MoE variant smoke 编译 PASS at canonical TP=8
per-rank widths**。所有 8 ST（2 dense + 6 MoE）都编译干净。只剩 MoE
device runtime。

### 5 仓 push 到 fork（2026-06-20）

把 pypto / pypto-lib / pto-isa / PTOAS / simpler 全 rebase 到
`origin/main`，audit 出 4 个 simpler 本地 patch 和 6 个 pypto-lib
step3p5 commit 仍要保（上游本周期没 subsume 任何一个），把 5 个工作
分支推到 `csy0225/<repo>` `stepfun/develop`。simpler submodule 落在
`a6e06406` 配 4 个生产关键 patch（zero-size view + `--no-as-needed`
libhcomm + IPC ENABLE_PEER_ACCESS + SDMA_OFF + llvm-strip）。

## Phase 1 准出 checklist

下面这些 criterion 定义什么叫"pypto kernel 原型完成"——是 Phase 2
（vLLM backend 集成）能起步的前置。

| Criterion | Phase 2 v0.1（单卡）需要 | Phase 2 v0.3（多卡）需要 | 状态 |
|-----------|:------------------------:|:------------------------:|------|
| 单卡 dense decode_layer device 跑通 | ✅ MUST | ✅ MUST | ✅ Phase 19 ST PASS |
| 单卡 e2e（Phase 15）rc=0 | ✅ MUST | ✅ MUST | ✅ rc=0，20 task complete |
| 48 层前端 smoke rc=0 | ✅ MUST | ✅ MUST | ✅ 所有 program builder + 8 layer-idx variant |
| 多卡 collective primitive | — | ✅ MUST | ✅ simpler L3 allreduce 双卡 golden |
| 多卡 decode_layer e2e | — | ✅ MUST | ⏸ barrier all_reduce gate（[`../blockers.md`](../blockers.md) §1） |
| MoE 6 variants 编译干净 | ✅ MUST | ✅ MUST | ✅ smoke 6/6 PASS at canonical TP=8 |
| MoE device runtime 绿 | — | （只 Phase 2 v1.0） | ⏸ 507018 gate（[`../blockers.md`](../blockers.md) §2） |
| HF safetensors weight loader | ✅ MUST | ✅ MUST | ✅ `weight_loader.py:load_step3p5_weights_for_rank` + 30-key `expected_shapes()` |
| Phase 16 三剑合璧部署 | ✅ MUST | ✅ MUST | ✅ 0162 验证；0234 driver/firmware 升级未做 |
| 5 仓 push 到 fork stepfun/develop | ✅ NICE | ✅ MUST | ✅ pypto / pypto-lib / pto-isa / PTOAS / simpler |

**结论**：Phase 1 交付物**对 Phase 2 v0.1 和 v0.2 足够**（单卡 flow）。
v0.3 多卡和 v1.0 全 pypto MoE 还 gate 在 barrier all_reduce + MoE
507018 修复；两个修复可以跟 Phase 20/21 实现并行做，Phase 22 多卡段
打开前不在关键路径上。

## 经验 lesson（交叉引用）

开发过程暴露了多个重复出现的坑，永久 reference doc 已捕获：

- `pypto-lib/docs/known-pypto-pitfalls.md` —— pypto / pto-isa / simpler
  kernel / codegen 层的硬限制和 bug（8 条）：
  1. `[N, 1]` intra-UB VEC tile（Phase 15 head_gate 根因）
  2. Vec / none_box tile 行 32-byte 对齐
  3. `pl.dynamic` 首维丢跨函数 slice stride
  4. `pl.dynamic` 加幻 unreferenced int32 kernel 参数
  5. AICPU `aicpu_orchestration_entry` 不能 `fprintf(stderr)`
  6. Kernel body 必须用 `pl.range/parallel/unroll/...` 不是裸 `for`
  7. `pl.range(constant)` 展开后不复用 SSA buffer → UB overflow
- `pypto-lib/docs/dev-workflow-gotchas.md` —— 非 pypto workflow 坑（5
  条）：stale pyc / 三件套 activation / HTTP/2 timeout / netboot SSH /
  gh CLI fallback to curl
- 单卡 ST/UT shape 铁律（CLAUDE.md 顶部，Claude project memory
  `feedback_single_card_st_shape_iron_rule` 永久 memo）

## 归档截止时（2026-06-22）的 open issue

详情见 [`../blockers.md`](../blockers.md)。简单总结：

- **Critical（gate Phase 22 多卡）**：barrier all_reduce UB-friendly
  重写、MoE 507018 device runtime
- **精度**：head_gate ×1 旁路 / TASK-L cube-matmul fix
- **基础设施**：0234 driver/firmware 升级
- **Deferred**：Phase 17 prefill MoE L1 overflow，MTP 拼进 decode_fwd

## 归档截止时的仓库状态

| 仓库 | 分支 | Pin | 超过 origin/main 的内容 |
|------|------|-----|------------------------|
| pypto | `stepfun/develop` | `b00c8b23` | DFX env hook（`output_prefix` / `dep_gen` / `l2_swimlane`）+ 16 个 full_rope debug repros |
| pypto-lib | `stepfun/develop` | `9c4773f` | step3p5 模型在树 + Phase 19 padding + ST 脚手架 + dev-workflow gotchas（误置的 Phase 20-22 design 已撤回） |
| pto-isa | `stepfun/develop` | `e25732f0` | = origin/main（无本地 patch） |
| PTOAS | `stepfun/develop` | `da011a3d` | = origin/main；binary ptoas-bin v0.45 |
| simpler（submodule） | — | `a6e06406` | zero-size view + libhcomm `--no-as-needed` + IPC ENABLE_PEER_ACCESS + SDMA_OFF + llvm-strip（4 个生产 patch） |

## References

- vLLM stepcast fork（Phase 2 集成目标）：
  `<workspace>/pd_sep/update_019/ascend/vllm/vllm/model_executor/models/step3p5.py`
- 实时跟踪器：`<workspace>/pypto/CLAUDE.md`
- 仓内 phase docs（Phase 20+）：[`../phases/`](../phases/)
