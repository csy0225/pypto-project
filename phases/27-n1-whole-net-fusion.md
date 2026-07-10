# Phase 27 —— N=1 整网融合（单 @pl.program 全 45 层 + tail）

> **独立攻关线**，与 Phase 25（Option-C 多程序 whole-model orchestration）并行、互不干扰。
> 分支：pypto-lib `feat/whole-net-n1-fusion`（其余仓 `n1fusion-base`）。
> 执行机：`gpu-a910x-0234`（tmux `pypto-ascend`）。**不碰 0162**。

## Pin snapshot（升级栈，2026-07-09 从 0162 只读 bundle 导入 + 重建）

| 组件 | pin | 来源 |
|------|-----|------|
| pypto | `5e619dc7` | 0162 bundle（rebased origin/main + step3p5 glue；#1828 在内） |
| simpler | `71e39623` | 0162 bundle（pypto 5e619dc7 submodule 精确锁此） |
| pypto-lib | `feat/whole-net-n1-fusion` @ `94aa015c` | fork stepfun/develop `b511da0e`(SplitIncoreOrch 修复) + gap-5 docs |
| pto-isa | `ecb6c303` | origin/main |
| PTOAS | `72ada0a1` | origin/main |
| ptoas-bin | **v0.49** | PTOAS release `ptoas-bin-x86_64.tar.gz` |

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

🟢 **N=1 整网编译里程碑达成（2026-07-10）**：`WholeDecodeNetwork` —— ONE `@pl.program` 跑**全 45 层** decode（每层 attn-only method → EP-MoE method，per-protocol 分离）+ tail —— **编译通过 rc=0** on 0234（升级栈 pypto `5e619dc7` + ptoas v0.45 + tmov 修复；`build_output/WholeDecodeNetwork_20260710_043504`）。逐级验证链全绿：

- `WholeDecodeDensePrefix`（3 dense 层 + tail，compile ✅）—— 证明多 dense 层 + tail 经 resident `pl.Out` handoff + 多-pass host_orch 在一个 program 内链接。
- `Mixed2Method` full/swa（compile ✅）—— 证明 per-protocol 分离（attn method + MoE method）编译；**swa_moe 分离 compile 过（fused swa_moe const-fold 失败）**，确认 per-protocol 分离是必需且正确的结构。
- `MixedMoeTail`（dense-attn method + 完整 EP-MoE method-set + tail，compile ✅ rc=0）—— 证明三者在**一个 program** 共存 + 链接（45 层程序所需的确切结构）。
- `MoeLayerReal`（真实 MoE 层：`swa_attn_only_orch` 纯 attention→resid1 → MoE `chip_orch` → residual → tail，compile ✅ rc=0）—— 证明真实 MoE 层结构（独立 attention 喂 MoE block）。
- `WholeDecodeProgram`（1 dense + 1 MoE mixed-min，compile ✅ rc=0）+ `WholeDecodeMixedMin`（忠实 dense-prefix + 真实 MoE 层，compile ✅ rc=0）+ **`WholeDecodeNetwork`（全 45 层 all-MoE 近似，compile ✅ rc=0）** —— N=1 整网编译达成。
- **`WholeDecodeFaithful`（忠实全 45 层：L0 full-dense + L1/L2 swa-dense + L3-44 42× 真实 per-protocol MoE + tail，88 个源码级展开 pass block，compile ✅ rc=0，`build_output/WholeDecodeFaithful_20260710_051822`）** —— **真实 step3p5 层表**的整网编译完成（非 all-MoE 近似）。

**遗留（Phase 4 device+runtime = 下一步）**：(1) 8 卡 device dispatch 决定性实验（per-protocol 分离是否躲开 Wall-2 `TaskMapSize=0`）。(2) 权重索引 3-class 在 45 层规模的解耦（当前编译用复用单 slab / all-MoE-layer 近似；真实需 host-side stack 切片 + dense-prefix L0-L2 + full-attn L0）。(3) 47GiB 单 key 权重 IPC + `_pypto_full_forward` single-handoff + live A/B。

### Phase 4 device 尝试（2026-07-10）—— 编译再确认 ✅ + device 被 0234 节点级 IPC poison 卡住 ⛔（非 Wall-2、非本程序）

- **新数据点：canonical TP=8 编译 ✅ rc=0**。此前编译里程碑走 `apply_perrank_patch`（TP=1 单卡）；本次用 `tests/step3p5/_probe_whole_faithful_canonical.py`（**无 patch**，canonical TP=8/EP=8 + `DistributedConfig(device_ids=[0..7])`）编译 `whole_decode_faithful` → `COMPILE OK`（`build_output/WholeDecodeFaithful_20260710_053948`）。说明 pass-37 comm-domain materialization 在真 TP=8/EP=8 下不报编译错，host_orch 生成真实 per-rank task loop（`for r in range(world_size): self.full_chip_orch(...)` 等）。
- **8 卡 device dispatch → 507899 `ImportByKey`（不是 `TaskMapSize=0`）**。用 `tests/step3p5/_stage_whole_faithful_device.py`（80 个 dummy-zero 输入匹配 host_orch signature，`compiled(*inputs)` 就地 `pl.Out`）在 device_ids=[0..7] 跑，第一处 `orch.allocate_domain("comm_d0", workers=[0..7])` 即失败：`domain_alloc_via_ipc: ImportByKey -> 507899`（`comm_hccl.cpp:833`）→ `comm_alloc_domain_windows failed with code -1` 全 8 chip。
- **决定性隔离：已知良好 baseline 复现同样的 507899**。`allreduce_distributed -p a2a3 -d 0-7`（Phase 16 canonical 多卡健康检查，0234 曾 2026-06-29 跑 `max|out-expected|=0`）**现在也 507899**；`-d 0-1` 2 卡同样 507899。→ **是 0234 节点级跨卡 HCCL IPC poison，不是 WholeDecodeFaithful 程序，也不是 Wall-2**。Wall-2 决定性实验因此**无法在当前 0234 状态回答**。
- **驱动 cap 在位**：0234 driver `25.5.2` / firmware `7.8.0.7.220`（STATUS.md 旧记录 25.5.1 已过时）。所以不是 Phase 16 cap 缺口，是运行期 driver IPC 状态卡死（曾工作 → fresh 进程仍失败）。
- **容器内恢复手段全部无效**：`npu-smi set -t reset`（out-of-band + in-band `-m 1`）→ `rc=214 "cannot be executed on a common container"`；`aclrtResetDeviceForce`（ctypes libascendcl，`reset_cards_acl.py`）→ 8 卡全 rc=0 **但 poison 未清**（对齐 `pypto/runtime/conftest.py:1039` "poison survives close()+device-reset on shared box"）；`/dev/shm`+`/tmp` 无残留；无 sysfs reset 节点；`CapEff=00000000a80465fb`（无 CAP_SYS_ADMIN，非特权容器）；只有 8 卡无备用组。**AICore(%) 85-93% + 0 HBM + 无进程 是这批 910B2C 的 idle telemetry 噪声，非卡死 kernel**（reset 后不变）。
- **需要 host 级介入**：0234 需 **host 侧 `npu-smi set -t reset` 或整机 reboot** 才能清 IPC poison；容器内不可达。在此之前 N=1 device dispatch（step 1）+ 权重解耦验证（step 2）+ IPC/live A/B（step 3）全部 gate 住。
- **本次新增测试脚手架**（pypto-lib `feat/whole-net-n1-fusion`）：`tests/step3p5/_probe_whole_faithful_canonical.py`（canonical TP=8 编译探针）、`tests/step3p5/_stage_whole_faithful_device.py`（8 卡 device dispatch harness，80 输入）。

### 历史

🟡 **Phase 0（2026-07-09）**：0234 五仓 + ptoas-bin 切升级栈；clean 重编 pypto+simpler；团队 3 份调查报告交付（Wall-1/Wall-2 root-cause + N=1 结构建议）。
