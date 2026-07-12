# NEXT SESSION — N=1 整网 decode 集成 + 端到端精度对齐 vs vLLM

> 直接把最底部 code block 当第一条消息粘贴。自包含。更新于 2026-07-12（本 session 尾）。
> **运行环境：0234 机器，通过本地 tmux `pypto-ascend-0:0` 登陆**（8 卡 0-7；781GB RAM；driver 25.5.2 / firmware 7.8.0.7.220 / CANN 9.0.0-beta.1）。
> 编辑机 `b-csy-develop`（**无 python，有 npu-smi，能直连 github**；NFS 与 0234 共享，编辑即时可见）。分支 `pypto-lib feat/whole-net-n1-fusion`。

---

## 🎯 总目标（north star，勿只盯单步）

**完成 step3p5 整网 decode 阶段的集成 + 端到端精度对齐 vs vLLM**（真 W8A8 + IPC，token-exact）。
本 track = N=1 整网融合（offline）：`whole_decode_faithful_real` —— **全 45 层（42 MoE + 3 dense/swa）内联进一个 `@pl.program`**，真 W8A8 权重+KV 经 **IPC** 加载，harness `tests/step3p5/_stage_whole_faithful_real_ipc.py`，分支 `feat/whole-net-n1-fusion`。
> ⚠ **别和另一个 track 搞混**：`NEXT-SESSION.md` 是 **G5b / per-layer 逐层 golden + live vLLM** track（harness `_stage_whole_decode_run.py`，本分支无此文件）。目标/harness/分支都不同，但 L2/live 阶段可复用其 KV-bridge / co-tenancy 机器。

### 里程碑路线图（当前进度）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 单算子 probe | matmul_acc N=16 / head-gate / gate_topk vs torch | ✅ PASS |
| M1 功能 bring-up | 42 MoE 真 W8A8 + 权重+KV 双 IPC 8 卡 dispatch-clean | ✅（Blocker B 解） |
| M2 per-layer gate_r | monolithic 整网自算逐层 head-gate（on-device，token-exact-capable） | ✅（路径 a，本 session） |
| **M3 单层 MoE 数值正确** | **NaN 修掉 → finite** | **✅（2026-07-13：根因=attn_only_orch 的 Out 未写回，非 MoE/INT8）** |
| **M3b 单层 MoE 幅值正确** | **next_hidden 1e12 → 合理幅值**（chip_orch 读到被别名的 huge h_mid） | **⛔ 当前卡这里** |
| M4 L1 ctx=1 token-exact | 全 42 层放开，`--hidden-token 6127` → **argmax=303** vs vLLM | ⏸ gated on M3 |
| M5 L2 多 token / decode-step | vLLM→whole-net KV bridge 或 live A/B（8001 vs 8000），多 token token-exact | ⏸（需 port G5b 机器） |
| M6 整网 decode 集成落地 | 接入 serving 路径（live single-handoff），端到端精度双过准出 | ⏸ |

**判据**：L1 per-layer hidden `ratio_allclose(atol=0.04)` / L2 logits cos≥0.999+topK overlap≥4/5 / L3 greedy top-1≥95%。**oracle = vLLM eager dump，synthetic golden 会 stale。**

## ⭐⭐ 本 session（2026-07-13）— M3 NaN 根因找到并修复；前 8 次 UPDATE 的 MoE-INT8 判断是错的

**M3 NaN 真根因 = `attn_only_orch` 的 Out 未写回（不是 MoE/INT8/gap-5）。**
- `full_attn_only_orch`/`swa_attn_only_orch`（generator `_gen_faithful_real.py` FRESH_*_ATTN_ONLY）里 `resid3_out = pl.create_tensor(...)` **遮蔽了 `resid3_out` pl.Out 参数** → attention 写进新建的局部 tensor，`h_mid_out[rd]` **从没被写** → base `chip_orch` 读到未初始化的 h_mid → NaN。dummy-0 权重把它掩盖了（所以 dummy 跑 DISPATCH_CLEAN；真权重才暴露）。
- **修复**（已落 real-builder 两个 orch 定义 + generator）：把 attention 结果 `pl.assemble` 进 `resid3_out` Out（`with pl.at(CORE_GROUP): for _co: resid3_out = pl.assemble(resid3_out, slice(resid_a,...))`）。**device 验证 P=1：NaN→finite，`h_mid 0.0→448`，`next_hidden nan→1.05e12`，logits/argmax 有限。**
- **定位手法**（有用，记下来）：harness 打印 `max|h_mid|`（attn_only 输出，chip_orch 不覆写它）+ 编译期 `P_MOE_PASSTHROUGH` 旋钮（chip_orch 直接返回它的 `resid1`=h_mid 输入）。`h_mid`(harness)=448 但 `PASSTHROUGH`(chip_orch 读到的)=6.31e11 → **chip_orch 读到的 h_mid ≠ attn_only 写的**。

**M3b 新阻塞 = MoE 输出幅值 1e11–1e12（不是 INT8 数学问题）。**
- NaN 修完后 P=1 `next_hidden`≈1e11–1e12（有限但巨大），argmax≠303。
- bisect 旋钮（都是 device-if，已证会 runtime-select）：`P_MOE_PASSTHROUGH`(返回 resid1)=6.31e11；`P_MOE_NORM_ONLY`(返回 post_norm)=4.23e11；`P_MOE_SHARED_ONLY`(resid+shared)=2.86e11。→ chip_orch 读到 huge h_mid 输入 → 经残差加流进 next_hidden + 压垮 rmsnorm。幅值爆炸对 shared(BF16)+routed(INT8) **共同** → 是被坏输入带的，**不是 MoE compute**。
- **把 attn_only+chip_orch 的两个 per-rank 循环合成一个循环（pos=0）→ 幅值仍 1.99e11 未修** → 是 **buffer 别名**，不是循环顺序。`h_mid_out`/`next_hidden_out` 两 buffer 乒乓被全部 45 层的 orch 复用 → chip_orch 读到被别名的 huge 值。这正是 memory `n1_whole_net_scheduler_timeout_fixed_perlayer_windows`（per-layer distinct buffers）的领域；文档之前"aliasing REFUTED"只覆盖了 3 层 dense P=0 乒乓，**从没测过 MoE 层内 attn_only→chip_orch 这段 handoff**。
- **下 session M3b 修法**：给 attn_only→chip_orch 的 handoff **一个专用 buffer**（不复用共享 h_mid_out；per-MoE-layer distinct，或至少破掉别名）让 chip_orch 读到 attn_only 的真实输出。可能需要 DCE 的 `CommDomainScopeStmt` 修复（见 memory）。修完 P=1 幅值→合理 → 放开 42 层 → L1 argmax→303。

**代码状态**（pypto-lib `feat/whole-net-n1-fusion` 工作区，**未 commit**，在 NFS 0234）：
- **保留**（真修复）：attn_only writeback（两个 orch 定义 + generator FRESH_*_ATTN_ONLY）；in-expert INT8 quant padding-mask（`_expert_routed` amax/quant 循环把未初始化的 RECV_TILE padding 行 mask 到 tile_valid，两份 copy —— 正确且对 partial tile 有用，但**不是** NaN 根因）。harness `_stage_whole_faithful_real_ipc.py` +1 行 `max|h_mid|` 打印（保留）。
- **commit 前要清**（诊断脚手架，默认 0 inert）：3 个 module global + `if _MOE_{SHARED_ONLY,PASSTHROUGH,NORM_ONLY} > 0:` gated 提前 return（在 10 份 chip_orch copy 里各一个）；host_orch pos=0 的 merged-loop（中性，没修好）。
- 8 个 hold-mode exporter 之前在 0234 上跑（真 W8A8 + KV IPC）。

### M3b 深入（续）— 三个 ordering 修法全失败 → 是 pypto 跨-orch 依赖墙；正解 = FUSE

- 追加验证：**chip_orch 在 attn_only 写 h_mid_out 之前就读它**（读到未初始化内存 → 跨 run 不确定：NaN 或 ~1e11-1e12；h_mid 也在 448-504 跳）。三个 ordering 修法**都没让 chip_orch 读到 attn_only 的真实输出**：
  1. 合并两个 `for r` 循环成一个 → 仍 1.99e11。
  2. 捕获 attn_only 返回值传给 chip_orch 当 current_hidden（强制 data-dep）→ PASSTHROUGH 仍 3.05e11（harness h_mid=394）。
  3. attn_only 改成把 Out 直接传给 inline attention（对齐 dense 的 `dense_mlp_inline(h0_out)` 写法，"C1"）→ 回退成 `next_hidden=nan`。
- **对比**：dense 层内 attention+MLP 在**一个** `swa/full_chip_orch` 里做（无层内 split），所以 dense L1→L2 的 h_mid handoff 正常（P=0→502）。MoE 层被拆成 attn_only+chip_orch 两个 orch，跨-orch 依赖 pypto 没排上序。**这正是 SKILL H 说的 N=1 inline 墙**（"whole-decode worker 用 multi-program + resident DeviceTensor，不 inline 45 层 body"）。
- **正解（下 session）= FUSE：把 attention 塞进 MoE chip_orch**（chip_orch 对 current_hidden 调 attention_inline → resid1 局部 → post_norm+MoE；镜像 dense full_chip_orch），消掉层内 handoff。需两个 fused 变体（full/swa，因 11 full + 31 swa MoE 层）+ generator 重写 + regen；有编译墙风险（attention+全 MoE 一个 orch 可能过大）。**或**转 SKILL 推荐的 multi-program resident-DeviceTensor（非本 N=1 track）。
- **当前最佳工作态 = attn_only assemble-writeback 修复（finite ~1e12，NaN 已消）；C1 已回退。** decode_layer.py 工作区仍带诊断旋钮 + merged-loop/capture-pass pos=0，commit 前清。



## ⛔ 用户硬约束（不可违背，勿走弯路）

- **必须用 IPC 共享显存机制**做端到端，**KV cache 和权重都走 IPC**。**不许 H2D 绕路**、不许换非-IPC 方案。（权重+KV IPC 已 device 跑通。）
- **必须用真实权重加载**（真 W8A8 checkpoint，非 dummy）。**真权重调试，不走其他弯路。**
- 遇到问题只能**解决它**，不能绕开（work-around）。诊断脚手架只能定位、不能进产品路径。
- **correctness 和 speed 都要**：既要跑出正确结果、也要推进到底完成目标；别用"correctness"当借口停在半路，也别为"快"造出错误的精度数字。
- **不能只盯一个子目标**——总目标是整网 decode 集成 + 端到端精度对齐；单步（如 MoE NaN）修完要立刻推进下一里程碑。
- **对齐 DeepSeek/Qwen**：遇问题先看 DeepSeek v4/Qwen 实现 + 历史开发文档，尽量对齐；step3p5-vs-DeepSeek 差异必须论证（只在"性能更好"时保留）。
- **架构优先**：coding 前先系统分析 + 整体设计。**严格遵守 SKILL.md**（`pypto-project/.claude/skills/pypto-dev-constraints/SKILL.md`）；不满足约束可能是设计不合理需重设计。
- **⚠ 历史文档可能 stale，先核对当前代码再下结论**。

## ✅ 已解决并推送（pypto-lib `f07da3b`）

- **M1 Blocker B（gate_topk mrgsort）+ 权重+KV 双 IPC 8 卡 device-clean**（`4bede85`/`c61046b`/`b92031f`）。gate_topk 修复对齐 DeepSeek format1 链，device 数值验证 PASS（vs torch.topk）。
- **M2 整网 per-layer gate_r 结构性阻塞已解 —— on-device head-gate（路径 a）**：
  - `matmul_acc N=16` 丢 K 累加的 codegen bug（当年把 head-gate 移 worker 的原因）**现栈已修**（`_probe_matmul_acc_n16` + full-chain `_probe_head_gate_full` device PASS）。
  - `attention_full.py`/`attention_swa.py` **Scope 1.f 恢复 on-device gate**：`gate_logits = normed_all @ w_g`（K-chunk matmul_acc N=16）→ `sigmoid` → `gate_exp = gate_score @ R`（N-chunk）；Scope 3.a o_proj 乘 `gate_exp`（2 scope 分离控 UB）。
  - `gate_r` 槽改承载 **block-diag R 常量**（`R[h,h*HEAD_DIM+d]=1` 实头；**layer-independent → 喂一次全 45 层通用**），harness 填 R（实头=HQ//HEAD_DIM：full=8/swa=12）。→ monolithic 整网每层从自己 `normed_all` 自算 gate → **token-exact-capable**（不再需 per-layer dispatch / resident-DeviceTensor）。
  - 对齐 vLLM `modeling_step3p5` L489（g_proj on **post-input-layernorm** hidden，非 raw）+ L527-531。`whole_decode_faithful_real` **TP=8 COMPILE OK**（attention inline 从 `._func` 重导，无需 regen）。
  - memory `n1_head_gate_ondevice_restored_l1_nan` / `step3p5_head_gate_uses_normed_hidden`。

## ⛔ M3 当前阻塞：单层 INT8 W8A8 routed-MoE 的**有效行**计算产生 NaN

**L1 ctx=1 A/B 现状**：pypto worker `--hidden-token 6127 --kv-ipc` **RUN_CLEAN ~3.6s 但 `next_hidden=nan / logits=nan / argmax=0`**（vLLM golden：tid=6127「北京」→ next=**303**「，」）。

**已完成的 bisect（旋钮 `P_FAITHFUL_MOE_LAYERS`，`decode_layer.py:19182`，默认 42，emit N 层 MoE）**：
- `P_FAITHFUL_MOE_LAYERS=0`（仅 3 dense/swa attention 层）→ **FINITE**（`next_hidden=502.0 / logits=9.03 / argmax=27527`）→ **attention 路径 + on-device head-gate 确认干净**。
- `P_FAITHFUL_MOE_LAYERS=1`（3 attention + **1 MoE**）→ **NaN** → **单个 INT8 MoE 层就复现**，非跨层累积。MoE 输入 post_attention_layernorm 归一化（O(1)，非 502），不是输入幅度问题。

**已排除**：
- **不是 gate/head-gate**（bisect MoE=0 干净）。
- **不是 A-operand fractal-32 padding**（gap-5 经典坑）：试过把 `routed_x_quant`/`routed_h_quant` 的 `x_i8`/`h_i8` padding 行 [tile_valid:RECV_TILE] 用 `fillpad(set_validshape(cast_tile,tile_valid,K), PadValue.zero)` 置零（对 cast **TILE**，非 tensor —— set_validshape 只吃 TileType）→ 编译+跑通**仍 NaN** → 已 revert（tree clean f07da3b）。amax 是 per-row（`row_max` over K → `[1,RECV_TILE]`），有效行 scale 干净，padding 置零证实无效。
- **不是 stale 权重格式**：exporter `int8_routed=True` → INT8 pool 25.35GiB；`w_g` padding zero-pad（`weight_loader._slice_g_proj` L594）；KV pool `torch.zeros`（`pypto_weight_ipc.export_from_checkpoint` L394）。

**关键结构事实**：whole-net 内联 MoE 是 **INT8-native in-expert quant**（`decode_layer.py` `routed_x_quant`/`routed_h_quant` scope：`xe_amax→scale→cast INT8 trunc→matmul out_dtype=INT32→dequant col/row_expand_mul`），`_quant_moe_input` 次数 **decode_layer.py=0 vs moe.py=2** —— 即 whole-net 内联 MoE 与 **standalone-validated `moe.py` INT8 kernel（dispatch-side quant, Option A）DECOUPLED**（旧 in-expert 路子）。只有 **2** 处 `routed_x_quant`（base + real builder），routed expert body 是共享 inline（**非 42 份**）→ 定位/修改是**单点**。

**剩余嫌疑（有效行计算，按可疑度）**：
1. **routed INT8 gate/up/down dequant**：`gate_2d = col_expand_mul(row_expand_mul(cast(gate_acc,FP32), x_scale_dq), wg_scale_row)`（`decode_layer.py:~20113`）—— 查 IPC 里 `moe_w_gate_r_scale`/`moe_w_up_r_scale`/`moe_w_down_r_scale` 是否含 0/inf/NaN（weight_loader dequant / exporter INT8 scale 逻辑）。
2. **combine** routing weight × routed_y + shared。
3. **shared expert**（BF16 swiglu；layer-0=full_moe_silu_silu 用 silu）。
4. **dispatch a2a**（comm windows / CSR）。

**M3 攻克手法（下 session，用户已批准集中攻）**：
- **手法 A（推荐，最快隔离 权重 vs 代码）**：把真 IPC INT8 routed 权重+scale 喂进 **standalone `moe.py` MoE-block harness**（已 device-validated 的 kernel）跑单 MoE block。standalone **finite** → whole-net 内联副本 bug（→ 手法 C）；standalone **也 NaN** → 权重/scale bug（→ 查 weight_loader/exporter INT8 scale dequant）。
- **手法 B（per-op dump 仪表化）**：real builder routed expert 加中间 Out（gate_2d / h_bf16 / routed_y / shared_out）经 host_orch + harness 拉回，逐 stage 看谁先 NaN。monolithic 无中间输出，必须显式加 Out。
- **手法 C（根治，A5 大改）**：把 `moe.py` 的 `_quant_moe_input` + dispatch-side INT8 recv（Option A，已 validated）经 `tools/step3p5/_gen_faithful_real.py` regen 应用到 whole-net 内联 MoE（替换 decoupled 旧副本）。工作量大但根治。
- **收尾即推进 M4**：修完 `P_FAITHFUL_MOE_LAYERS=1` finite → 逐步放开 2/4/…/42 → 全量 L1 A/B（tid 6127 期望 **argmax=303**）→ 进 M5（多 token / KV bridge / live A/B）。

### ⭐ 本 session 更新（2026-07-12 续）— NaN 可靠定位到 MoE，attention 干净；两个 confound 已记

- **数据全排除**：36288 个 routed INT8 `*_scale` 全有限（`tests/step3p5/_diag_check_w8a8_scales.py`，min1.6e-4/max1.5e-2）；shared expert 权重确为 BF16（ckpt 无 `_scale`）且有限；layer-3 attn q/k/v/o_proj 也是 BF16 有限（与有限的 layer-1 同构）。
- **决定性编译期测试**：新增 host-orch 编译期开关 `P_L3_ATTN_ONLY`（跳过 L3 的 MoE `chip_orch`，把 attention 输出 `h_mid` 直接喂 lm_head）→ `logits=0.0000 argmax=0` **有限、无 NaN** → **attention 输出有限，NaN 由 MoE(chip_orch) 引入**（重新确认原 INT8-MoE 定位）。
- **⚠ 两个 confound（勿再踩）**：
  1. pypto 在 `@pl.function` 体内对 module-int 的 `if` 会变成 **device-if（两分支都 trace）** → combine 旁路开关 `P_MOE_ROUTED_OFF`/`P_MOE_SHARED_OFF`/`P_MOE_OUT_OFF` **全不可靠**（都误报 NaN）。**可靠 gating = host-orch 层比较式 `if`**（像 `_FAITHFUL_MOE_LAYERS` 的 `if X>0:`），且**不能用三元 IfExp**（被 reject）。
  2. `chip_orch` 读且**覆写 `h_mid_out`** → MoE 一跑就读不到干净 attention 输出，必须编译期跳过 MoE 才能读。`h_mid` 确实 copy-back（P=0 显示 294）。
- **次要异常（可能是触发点）**：L3 `h_mid≈0.0000`（本应 ~502，attention_swa L802-808 确有残差加法），且"L2 输出"幅值跨 run 不一致（P=0→448 vs _L3_ATTN_ONLY→502）→ 疑 **整网 buffer 数据流/aliasing** 把退化的 ~0 输入喂给 MoE，INT8 MoE 在其上出 NaN。
- **代码状态**：诊断脚手架（decode_layer.py 616 行 device-if 开关）**已 revert 回 f07da3b clean**；保留 `_diag_check_w8a8_scales.py` + harness 1 行 `h_mid` 上报。
- **下一步（修复，不再重新定位）**：① 可靠子阶段隔离——把 `sh_y`/`routed_y` 作 **copied-back Out** 串出 chip_orch→host_orch→harness（**不用 device-if**）确认 shared vs routed；或 ② 直接上 gap-5 根治方案 **dispatch-side INT8 quant**（对齐 moe.py `_quant_moe_input`）；③ 并查 h_mid≈0 / buffer 幅值异常（可能才是真触发点）。重建 `P_L3_ATTN_ONLY` 记法见 memory `n1_head_gate_ondevice_restored_l1_nan` UPDATE5。

### ⭐ 本 session 收尾（2026-07-12 续2）— 根因假设=hidden-state 乒乓 SSA 别名；per-layer 修复撞 pypto DCE 墙

- **根因假设（对齐先例 `n1_whole_net_scheduler_timeout_fixed_perlayer_windows`）**：整网 `whole_decode_faithful_real` 把 hidden state 在**仅 2 个 `pl.Out`**（`next_hidden_out`/`h_mid_out`）之间乒乓、跨 45 层各写 ~44 次 → 与 comm-window 别名同一类 P3/ADR-013 违反，但发生在 hidden state。真权重下 L3-attn 读到 stale SSA 版本 → `h_mid≈0` → MoE 退化输入 → NaN；dummy 全零权重时不可见（所以之前 DISPATCH_CLEAN 没发现）。
- **修复尝试 + 受阻**：改 `_gen_faithful_real.py` `_host_orch`，给每个 sub-orch 输出独立 slot（先 4D `hchain[tp,87,B,H]`，后 3D-flattened `hchain[tp*87,B,H]` 用 `hchain[r*87+slot]`）。**两种都编译失败** `pypto.InternalError: Unhandled ScopeStmt subtype in DCE: CommDomainScopeStmt`（P=1 和 P=42 都失败）。根因：**单个 pl.Out 参数在 RUNTIME 偏移 `r*87+slot` 处写入** → pypto DCE 无法静态匹配写/读偏移 → 认定带 comm-domain 的写是 dead → DCE 其 `CommDomainScopeStmt`（不支持）。旧乒乓能编只因偏移是裸 `[r]`（可静态匹配）。
- **唯一 DCE-safe 的 per-layer 形式** = **N 个独立 3D `pl.Out` 参数，各在 `[r]` 处写一次**（像旧的 2 个但要 ~87 个）——大改、有风险（87 参数签名 + harness alloc + lm_head 的 slot 依 `_FAITHFUL` 需编译期 if/elif，因为把 Out 参数别名到局部 `_lm_in` 会被 SSA reject）。
- **代码状态**：generator + decode_layer + harness **全 revert 回 clean f07da3b**（`git diff` 空）。**别名假设仍未证实**（所有单参数修复都编不过）。
- **下 session（三选一，建议先做最便宜的确认）**：
  - **(最便宜确认)** 只给 **L2→L3-attn 这一处**加 1 个独立 3D 参数 `hs_test`（L2 写 hs_test、L3-attn 读 hs_test），P=1 跑；若 `next_hidden` 变 finite → 别名确认 → 再做 87-参数全量版。（DCE-safe。⚠ 直接改生成后的 decode_layer.py 有大量重复签名/调用，建议改 generator 再 regen。）
  - **(全量根治)** N 个独立 3D 参数（generator 生成 87 个 + harness）。
  - **(上游)** 修 pypto DCE pass 让它处理 `CommDomainScopeStmt`（pypto core，较深）。
  - 若最终证实**不是**别名 → 转查 attention-compute 或 KV 喂给 MoE 的路径。
- 详见 memory `n1_head_gate_ondevice_restored_l1_nan` UPDATE6/UPDATE7 + `feedback_no_over_localize_just_fix`。

### ⭐ 本 session 收尾（2026-07-12 续3）— 两假设 REFUTED（gate + aliasing）；pypto DCE bug 已修；嫌疑收窄

用户建议下用两个 agent（DCE 排查 + 反向质疑）从别的角度看，结果：
- **GATE REFUTED**：加编译期 `P_GATE_BYPASS`（whole-net `gate_step` 写死路由 experts0-7/均权，绕过真 `_gate` sort/mrgsort），device P=1 **仍 `logits=nan`** → 真 gate 不是 NaN 源（反向 agent 的头号嫌疑被否）。（编译期 `if X>0:` device-if 确实 runtime-select 正常，gate bypass 跑了。）
- **ALIASING REFUTED**（反向 agent）：P=0 用**同一个 2-buffer 乒乓** L0→L1→L2 却 finite（argmax 27527）；若乒乓别名 dense 层就会暴露。`h_mid=0` 也不是别名的症状（别名给 stale ~502）。**放弃 87-参数 buffer 重写方案。**
- **pypto DCE bug 已定位+修复（独立价值）**：`pypto/src/ir/transforms/utils/dead_code_elimination.cpp` 两处 ScopeStmt-rebuild dispatch 缺 `CommDomainScopeStmt` case（~L290 + ~L462）→ `Unhandled ScopeStmt subtype in DCE`。已补（仿 RuntimeScopeStmt）。**未 commit、未 rebuild（inert）**，值得 commit+上游（解锁任何 computed-offset Out 写进 comm-domain orch）。
- **NaN 是真的**（logits=nan），在 `chip_orch` 内、gate 下游 → **dispatch / routed-INT8 experts / shared-BF16 / combine**。standalone moe.py MoE-block 曾 device-验证 PASS 但用 `--bypass-gate` + 合成输入 + H2D 权重；whole-net 差在真 attention-输出输入 + IPC INT8 权重。
- **待破的矛盾**：`P_L3_ATTN_ONLY`（跳 chip_orch、lm_head 读 attn 输出）→ `logits=0.0000`（finite 且为 0）→ attention 输出 ≈ 0（device-可靠，经 lm_head 非 host readback）。但解析上 MoE(0-输入)=0（routed amax floor 1e-4 / shared BF16 / combine 求和）应 finite 0 而非 NaN。⇒ 要么 (a) attention 确实输出 ~0（zeroed-KV ctx=1；dense L1/L2 非零只因 dense-MLP 加幅值，attention 部分本就 ~0），MoE(~0) 撞真 codegen/div NaN（rsqrt(0) in norm，或 in-kernel INT8 quant partial-tile gap-5）；要么 (b) attention 输出是非零 garbage 而 h_mid=0 仍误导。
- **下一步**：gate bypass 下（已证 device-if 可靠）重跑 routed-off / shared-off 干净拆分 shared-BF16 vs routed-INT8；若 routed → 攻 gap-5 in-kernel INT8 quant（`routed_x_quant`/`routed_h_quant`）。detail: memory `n1_head_gate_ondevice_restored_l1_nan` UPDATE8。

## 🎯 精度对齐三档（对应 M1/M4/M5）

- **L0 单算子 probe（M0 ✅）**：`_probe_matmul_acc_n16`、`_probe_head_gate_full`、`_probe_gate_sort` 全 PASS。范式见 🐞。
- **L1 ctx=1 单 token A/B（M4，脚手架已建可跑）**：`_stage_whole_faithful_real_ipc.py --hidden-token <id>` 把 `embed(token)` 灌进 `current_hidden` row0 + pos-0 identity rope（cos=1/sin=0），seq_lens=ones（16 行全 ctx=1，避开 seq_len=0 NaN）。vLLM 1-token prompt 首 token = argmax(logits(pos0))，等价 ctx=1 self-attn，**不需 KV bridge**。**当前卡 M3 MoE NaN**，修完即通。
- **L2 整网多 token / decode-step golden（M5，终极）**：多 token 需 vLLM→whole-net KV bridge（分页池→整网 flat KV，memory `g5b_kv_bridge_not_pure_reshape` / `g5b_kv_is_bf16_not_int8` / `g5b_import_ipc_facade_missing`）或 live A/B（8001 pypto vs 8000 vanilla，co-tenancy `SIMPLER_COMM_NO_HCCL=1`，memory `project_g4_cotenancy_hccl_conflict`）。**这套是 G5b track 机器（0162 working tree），本分支需 port。**

## 🖥 环境 / vLLM oracle 启动（本 session 验证可用）

- **三件套激活**（每 fresh shell，`activate.sh` 不带 CANN env）：
  `source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh && export PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib`（`WS=/data/chensiyu/hw_project/pypto/workspace`）。
- **vLLM W8A8 oracle（0234 可跑）**：**⚠ 先 `unset PYTHONPATH` 再 `source cann/set_env.sh`**（顺序关键——先 source 再 unset 会把 acl 抹掉 → `ModuleNotFoundError: acl`），**不 export pypto PYTHONPATH**，再：
  `vllm serve <W8A8ckpt> --served-model-name step3p5 --trust-remote-code --quantization ascend --tensor-parallel-size 8 --enable-expert-parallel --enforce-eager --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85`
  → 占 8 卡 0-7，~5-6min load。
  - **⚠⚠ vLLM 跑在 0234，查询必须在 tmux 0234 里 `curl localhost:8000`**。**从 b-csy-develop 的 Bash `curl localhost:8000` 命中的是它自己的 nginx → 404**（本 session 踩过，别误判 vLLM 挂了）。
  - 1-token golden：`_l1_ab_vllm.py --word 北京`（取首 token 作 tid，查 vLLM greedy next-token id+text）。本 session：tid=6127→303。
  - vLLM 与 pypto **同卡** → 先 vLLM 出 golden 再 kill 腾卡跑 pypto（offline A/B）。kill：`pkill -f "[v]llm serve"; pkill -9 -f "[E]ngineCore"`。
- **W8A8 ckpt** = `/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp`（arch `Step3p5ForCausalLM`，45 层，embed 在 shard 00048 非量化）。
- **8 卡 pypto env**：`PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
- **exporters（IPC 权重/KV）—— bisect 高效设置**：默认 worker mode 自己 launch 8 exporter + 跑完写 STOP（每次 15min 冷载）。**多次 bisect 用 hold-mode**：先手动起 8 个常驻 `--export-rank r --dev r --kv-ipc`（一次 15min 冷载，写 `ready.rank{r}`+`pypto_weight.key.rank{r}` 后 hold 等 STOP），再多次 `--reuse-exporters` worker 秒级 attach（每次只 compile+run ~5min）。收尾 `touch /tmp/n1_weight_ipc/STOP` 或 `pkill -f export-rank` 释放卡。
  - **⚠ `/tmp` 每机独立**：exporter 写 0234 的 `/tmp/n1_weight_ipc/`，**从 b-csy-develop `ls /tmp/...` 看不到**（本 session 误以为 0 ready）。查 ready keys 必须在 **tmux 0234**：`ls /tmp/n1_weight_ipc/ready.rank* | wc -l`。

## 🐞 Debug 方式（累积经验）

- **数字 device error 先查 [wiki Device-Error-Codes_zh](https://github.com/hw-native-sys/simpler/wiki/Device-Error-Codes_zh)**。`507018` 是泛化 host 码，看 `orch_error_code`/`sched_error_code`/`sub_class` 定真因。
- **device stall 快照**（定位卡住 kernel/核，本 session 定位 gate_topk 用过）：harness `logging.getLogger("simpler").setLevel(15)` + `export ASCEND_GLOBAL_LOG_LEVEL=1 ASCEND_PROCESS_LOG_PATH=<预建目录>` → 读 `<dir>/debug/device-*/device-*.log` 的 `log_stall_diagnostics`（`TASK state=RUNNING kernels=[...] running_on=[core=X]`）。`orch_error=8`=TENSOR_WAIT_TIMEOUT，`sched=100`=running-stalled。root-owned，tmux(root) grep。
- **NaN/精度 bisect（本 session 定位 MoE 用的）**：`P_FAITHFUL_MOE_LAYERS=N` 层数二分（0=纯 attention / 1=单 MoE / …/42）隔离 attention-vs-MoE、单层-vs-累积。monolithic 无中间输出 → 要 op 级须显式加 Out dump（手法 B）。
- **单算子精度 probe 范式**（隔离验证）：module-level `@pl.jit` + `from golden import TensorSpec, run_jit, ratio_allclose` + torch golden_fn（见 `_probe_head_gate_full.py`/`_probe_matmul_acc_n16.py`）。⚠ `@pl.jit` 的 shape/const 必须 module-level；per-K-chunk cast 避免 `[T,HIDDEN]` FP32 UB overflow（256KB>184KB）；2-scope 分离让 K-loop buffer 先释放再做 N-chunk expand；小 N=16 输出 matmul_acc 现栈已正确累加 K。
- **launch 前**：`pkill -f '[_]stage_whole'` + `rm -f /tmp/n1_weight_ipc/{STOP,ready.rank*,pypto_weight.*}`（在 0234）；**禁 `-9` 强杀 device 进程 / `npu-smi set -t reset`**（netboot 机锁死全卡）；`npu-smi info -t usages -i <c>` 确认 HBM<10%。stale pyc：改 models 后 `find models/step3p5 -name '*.py' -exec touch {} +`。
- **每次 device run 慢**：exporter 全 ckpt load ~15min（用 hold+reuse 省重复冷载）；compile 42 层 ~4min，少层更快。

## ⭐⭐ 铁律（勿再踩）

1. **单卡 ST/UT shape**：`apply_perrank_patch()`（保 TP=8 per-rank slice），不用 `apply_tp1_patch()`。gate_matmul 单卡 unsliced 会 Mat/Vec 溢出——验 gate 用隔离 probe。
2. **gap-5 坑**：in-kernel `pl.cast(bf16/fp32,INT8)` 喂 cube 可能静默错；照抄 DeepSeek cast 链 + create_tensor 位置 + scope。**但本 track 的 MoE NaN 已证不是 A-operand padding**（经典 gap-5 fix 试过无效），是有效行计算。
3. **ccec ND2ND**：scale slice 必须 contiguous row-slice + reshape；a2a3sim compile 过 ≠ device ccec-clean（必须真 device 跑）。
4. **push**：PAT `/data/chensiyu/secrets/github.env` + `git -c http.version=HTTP/1.1`，输出屏蔽 token。**github 从 0234 连不通**（直连 130s 超时；proxy `deploy.i.shaipower.com/httpproxy` 返回的 proxy 需 auth → 407）；**从 b-csy-develop 的 Bash 能直连 github**（NFS 共享同一 repo；commit 在 tmux(root) 做完后 `git push` 从 b-csy-develop 跑）。pypto-lib `.git/objects` root-owned → commit 走 tmux(root)，push 可从 b-csy-develop（读 objects OK）。pypto-project chensiyu-owned 可直接。跨仓 push 同步 STATUS pin。
5. **文档 stale 风险**：核对当前代码再下结论（head-gate 就栽在 stale 注释）。
6. **别只盯单步**：M3 修完立刻推进 M4→M5→M6，总目标是整网 decode 集成 + 端到端精度对齐。

## 本 session commits

pypto-lib `feat/whole-net-n1-fusion`：**无新 commit**（本 session 所有修复尝试 hchain/87-param/gate-bypass 均 revert 回 clean `f07da3b`；`git diff` 空）。
pypto (`src/ir/transforms/utils/dead_code_elimination.cpp`)：**未 commit 的真 bug 修复**（补 `CommDomainScopeStmt` case 两处；inert 未 rebuild）—— 值得单独 commit + 上游。
pypto-project `main`：`e75d59e`/`ee510d7`/`a94e968`（M3 定位到 MoE→gate/aliasing/rmsnorm 三假设 REFUTED + DCE bug + 嫌疑收窄）。

---

```
继续 step3p5 **N=1 整网 decode 集成 + 端到端精度对齐 vs vLLM**（总目标；不是只修一个 bug）。当前挡路 = M3
单层 MoE(chip_orch) 数值 NaN；修完立刻推进 M4 全 42 层 L1 token-exact（tid 6127 → argmax=303）→ M5 多 token
（KV bridge / live A/B）→ M6 整网 decode 集成落地。本 track = whole_decode_faithful_real（45 层内联一个
@pl.program，真 W8A8 权重+KV 经 IPC，harness tests/step3p5/_stage_whole_faithful_real_ipc.py，分支
feat/whole-net-n1-fusion，pypto-lib HEAD f07da3b clean）。⚠ 别和 NEXT-SESSION.md 的 G5b/per-layer track 混。

【硬约束】IPC(KV+权重)+真实权重、遇问题只解不绕(诊断脚手架不进产品路径)、correctness+speed、不只盯单步、
对齐 DeepSeek/Qwen、架构优先、严格遵守 pypto-dev-constraints SKILL、历史文档可能 stale 先核对代码、
**不要走弯路/不需要定位的别定位**（本 session 教训：过度 device 定位烧了太多 cycle）、遇难点可开 agent 从别角度看。

【里程碑】M0 单算子 probe ✅ / M1 双 IPC 8 卡 dispatch-clean ✅ / M2 on-device head-gate ✅ /
**M3 单层 MoE NaN ⛔当前** / M4 L1 token-exact ⏸ / M5 L2 多 token ⏸ / M6 集成落地 ⏸。

⭐【M3 本 session 已验证 / 已排除 —— 别重复】
- NaN 是真的（logits=nan 是最终输出、确 copy-back），在 chip_orch 内、gate 下游。
- bisect P_FAITHFUL_MOE_LAYERS=0（仅 3 attention 层）→ FINITE(argmax 27527)；=1（+1 MoE 层）→ NaN。
- **数据全 finite**：36288 routed INT8 scale（_diag_check_w8a8_scales.py，min1.6e-4/max1.5e-2）；shared BF16 finite；
  layer-3 attn q/k/v/o_proj BF16 finite（与 finite 的 layer-1 同构）。
- **GATE REFUTED**：P_GATE_BYPASS 写死路由(experts0-7/均权)绕过真 gate sort/mrgsort → device P=1 仍 logits=nan。
- **ALIASING(hidden 乒乓) REFUTED**：P=0 用同一 2-buffer 乒乓却 finite；h_mid=0 非别名症状(别名给 stale~502)。
  ⇒ **放弃 87-参数/hchain buffer 重写**（且 hchain 单参数 runtime-offset 写会撞 pypto DCE 墙，见下）。
- **rmsnorm-eps REFUTED**：post_norm rmsnorm 确加 EPS=1e-5（decode_layer.py:25237）→ rmsnorm(0)=0 finite。
- **A-operand fractal-32 padding fix**（zero x_i8/h_i8 quant 之后）：UPDATE4 试过无效已 revert。
- **附带修了真 pypto bug**（DCE 缺 CommDomainScopeStmt case，dead_code_elimination.cpp ~L290+~L462，已补未commit/未rebuild）。

【M3 未破的矛盾 —— 下 session 要害】P_L3_ATTN_ONLY（编译期跳 chip_orch、lm_head 读 attn 输出）→ logits=0.0000
（finite 且为 0）→ attention 输出 ≈ 0（device 可靠，经 lm_head 非 host readback）。但解析上 MoE(0-输入)=0 应 finite，
不该 NaN。⇒ 要么 (a) attention 确输出~0（zeroed-KV ctx=1；dense L1/L2 非零只因 dense-MLP 加幅值）而 MoE(~0)
撞 codegen/未初始化-读 NaN；要么 (b) h_mid=0 仍误导、真 attn 输出是非零 garbage。**先破这个矛盾再谈修。**

【下一阶段任务（按优先级）】
1. **[最优先] 试 gap-5 partial-tile 正解**：ctx=1 下每 expert <32 token（tile_valid<RECV_TILE=32），
   routed_x_quant 的 amax/matmul 读**未初始化的 local_routed_x padding 行[tile_valid:32]**（GM 垃圾/NaN）。
   修法 = **zero-init `local_routed_x` recv buffer**（或 fillpad padding 行）让 amax 读 0 非垃圾
   （区别于 UPDATE4 只 zero 了 quant 之后的 x_i8/h_i8）。改 decode_layer.py real-builder `_dispatch_stage`/
   `local_routed_x` create_tensor 或 `_expert_routed` 入口 fillpad。P=1 跑，看 logits 是否 finite。
2. **[干净拆分] gate-bypass 下重跑 routed-off vs shared-off**（device-if 已证 runtime-select 可靠）：
   加 P_MOE_ROUTED_OFF（combine 跳 routed 读，moe_out=sh_y）/ P_MOE_SHARED_OFF（moe_out=routed，acc 从
   routed_y[0]×0 起，避 pl.full Tensor-vs-Tile 冲突）；配 P_GATE_BYPASS=1 排除 gate 干扰。定位 shared vs routed。
3. **[破矛盾] 确认 attention 是否真输出 0**：swa_attn_only_orch(layer3) vs swa_chip_orch(L1/L2) 都调同一
   attention_swa_inline；查 L3 attn 输入 current_hidden（=L2 输出，应 ~502）是否被正确读到、残差是否保留。
4. 修完 P=1 finite → 放开 P_FAITHFUL_MOE_LAYERS=42 全量 → L1 A/B（tid 6127 期望 argmax=303）→ 进 M5。

【debug 方法 / confounds（本 session 血泪）】
- **可靠 gating = HOST-orch 层 或 编译期 `if X > 0:`**（像 _FAITHFUL_MOE_LAYERS）；`@pl.function` 体内对 module-int
  的裸 `if X:` 会被当 device-if(INDEX)拒；三元 IfExp 被拒；`_lm_in = out_param` 别名被 SSA 拒。
- **device-if（两分支都 trace）会 runtime-select**（gate-bypass 已证），所以 combine 的 routed-off/shared-off 结果可信。
- **chip_orch 读且覆写 h_mid_out** → MoE 一跑读不到干净 attn 输出，必须编译期跳 MoE(P_L3_ATTN_ONLY)才能读。
- **单参数 runtime-offset Out 写（hchain[r*N+slot]）撞 pypto DCE**（Unhandled CommDomainScopeStmt）；
  DCE-safe 只有裸 `[r]` 索引的独立参数（已补 pypto DCE case，未 rebuild）。
- 数字 device error 先查 wiki Device-Error-Codes_zh；stall 快照 setLevel(15)+ASCEND_GLOBAL_LOG_LEVEL=1；
  NaN bisect 用 P_FAITHFUL_MOE_LAYERS 层数二分；禁 -9 / npu-smi reset。

【环境】0234 tmux pypto-ascend-0:0（8 卡）；三件套 `source /usr/local/Ascend/cann/set_env.sh && source $WS/activate.sh
&& export PTO_ISA_ROOT=$WS/pto-isa && export PYTHONPATH=$WS/pypto/python:$WS/pypto-lib:$PYTHONPATH`（PYTHONPATH 要
**append** 否则丢 acl）。8 卡 env `PTO2_RING_HEAP=4294967296 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072`。
W8A8 ckpt=/mnt/hw910test-jfs/models/step3p5_flash_release_hf_mtp3_w8a8_0328-copy-mtp。
bisect 高效：8 hold-mode exporter（--export-rank r --dev r --kv-ipc，一次~15min 冷载 或秒级若已在）+ 多次
`--reuse-exporters` worker（改 models 后先 `find models/step3p5 -name '*.pyc' -delete`；每次 compile+run ~2-5min）。
⚠ /tmp 每机独立（查 ready keys 在 tmux 0234）。push 走 b-csy-develop（0234 连不通 github）+ HTTP/1.1 + PAT
`/data/chensiyu/secrets/github.env`；pypto/pypto-lib .git objects root-owned → commit 走 tmux(root)。
全部细节读上面 🎯/⛔/✅ 段 + memory n1_head_gate_ondevice_restored_l1_nan（UPDATE5-8）。
```
