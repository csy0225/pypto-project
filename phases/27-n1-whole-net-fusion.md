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
| **4** | 扩到全 45 层（full/swa dense + swa/full-attn-only method + MoE method-set + tail，46-pass host_orch）+ 权重索引 3-class 解耦（45 层 slab 不能逐层显式传参，需 host-side layer 切片 or inline body 拆 norm/attn/kv/mlp idx）+ 47GiB 单 key 权重 IPC + wire `_pypto_full_forward` single-handoff + live A/B 8001-vs-8000 | 进行中（下一步） |

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

🟢 **compile 基础全部打通（2026-07-10）**：N=1 所需的 per-protocol 结构积木**全部 compile-verified** on 0234（升级栈 pypto `5e619dc7` + ptoas v0.45 + tmov 修复）：

- `WholeDecodeDensePrefix`（3 dense 层 + tail，compile ✅）—— 证明多 dense 层 + tail 经 resident `pl.Out` handoff + 多-pass host_orch 在一个 program 内链接。
- `Mixed2Method` full/swa（compile ✅）—— 证明 per-protocol 分离（attn method + MoE method）编译；**swa_moe 分离 compile 过（fused swa_moe const-fold 失败）**，确认 per-protocol 分离是必需且正确的结构。
- `MixedMoeTail`（dense-attn method + 完整 EP-MoE method-set + tail，compile ✅ rc=0）—— 证明三者在**一个 program** 共存 + 链接（45 层程序所需的确切结构）。

**遗留（Phase 4 = 下一步）**：(1) 全 45 层单 program 组装 —— 核心难点是权重索引 3-class 在 45 层规模的解耦（per-layer slab 显式传参在 3 层可行，45 层不可行；需 host-side layer 切片或把 inline body 的单一 `layer_idx` 拆成 norm/attn/kv/mlp 四个 idx）。(2) 8 卡 device dispatch 决定性实验（per-protocol 分离是否躲开 Wall-2 `TaskMapSize=0`）。(3) 47GiB 单 key 权重 IPC + `_pypto_full_forward` single-handoff + live A/B。

### 历史

🟡 **Phase 0（2026-07-09）**：0234 五仓 + ptoas-bin 切升级栈；clean 重编 pypto+simpler；团队 3 份调查报告交付（Wall-1/Wall-2 root-cause + N=1 结构建议）。
