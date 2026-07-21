# 专项：gap-5 — in-kernel `cast(→INT8)` 喂 cube A-operand 误编译（`infer_tile_memory_space_pass` 未推 INT8 cube fractal，~98% wrong，no fault）

| 字段 | 值 |
|------|----|
| **子系统** | codegen |
| **error signature** | INT8-native routed MoE `bad_ratio≈0.9847` / `max diff ~254` / **no device fault** |
| **首次出现** | 2026-07-09（IR dump 实证定位）；现象首见更早 session |
| **状态** | 🟡 缓解（model-side 切 dispatch-side quant；上游 IR-level fix 未落） |
| **相关 skill / doc** | `pypto-lib/docs/upstream-issues/gap5-cast-int8-cube-codegen.md`、`postmortems/09-attention-multiposition-corruption.md`、`design/whole-net/02-detailed-design.md` §5 |

## 1. 背景（Background）

事故发生在 step3p5 **INT8-native routed MoE** 路径上（BF16-dequant 之外的真 W8A8 target）。目标机 `gpu-a910x-0162`（Ascend 910B2C，CANN 9.0.0 non-GA，PTOAS v0.45→v0.49，pypto `5e619dc7`，pto-isa `ecb6c303`，simpler `71e39623`，per-rank 配置同 09）。想达成：把 routed MoE 从 BF16-dequant（存 47GB/rank）改成真 W8A8 INT8-native（存 ~24GB INT8 + 片上 W8A8，数学照抄 DeepSeek v4 `expert_routed.py`、与 vLLM W8A8 一致），使 HBM 占用从 OOM 边缘降到可共存。

整网 e2e 长期走 BF16-dequant moe.py（`0.9995` vs vLLM），但**真正的出口是 INT8-native**。gap-5 就是 INT8-native 路径上的 device 误编译 gap。用户在 2026-07-09 goal session 拍板：**完全对齐 DeepSeek**（on-the-fly 分歧在正确性 + 性能上都更差），**不改 compiler，走 model-side staging**。

本事故的"走过的弯路"是核心价值：**3 次被 refuted 的假设**，每次都有 falsifying experiment，最后才在 IR dump 实证下定位到 `infer_tile_memory_space_pass.cpp:55-56`。

## 2. 现象（Symptom）

最小复现（单 device，Ascend 910B，per-rank 配置）：

```
PASS control: tests/step3p5/_probe_p2_downmirror.py
        INT8 A-operand from GM (pre-quantized in separate program)
        down-tile T=32 / K=1280 / N=128         → ratio_allclose PASS

FAIL:       tests/step3p5/_probe_fixb_onthefly.py
        same shape, A-operand from in-kernel pl.cast(bf16, int8)
                                                → ~98.4% outputs wrong
                                                  max diff ~254
                                                  NO device fault
```

设备侧 INT8-native routed MoE（restored stash moe.py，cards 8-15，`--layer 3 --w8a8-native --target out --bypass-gate` vs vLLM `out.pt`）：

```
next_hidden_out bad_ratio = 24.31%  (127458/524288，was 84%)
clean 10s，no 507018
```

FIXB 探针（in-kernel cast→cube，v0.49 全栈）：

```
bad_ratio = 98.47%   (FAIL)
P2 控制组（INT8 copy from GM）: PASS
```

即：**任何 in-kernel `pl.cast(<bf16/fp32>, pl.INT8)` 结果喂 `tile.matmul_mx*` cube A-operand 都静默 miscompile**；同样形状的 INT8 operand 从 GM 拷进来（预 fractal 化）就 PASS。无设备 fault，纯 codegen VALUE bug。

## 3. 根因（Root Cause）

### 3.1 IR-level root cause（upstream-scout, 2026-07-09）

- **`pypto/src/ir/transforms/infer_tile_memory_space_pass.cpp:55-56`** — primary：`tile.matmul_mx*` 在 `kUnregisteredCubeOps` 里 → INT8 cube A-operand 的 fractal/layout 从不被推导。
  - INT8 cube fractal = **32 rows**（BF16 = 16）。
  - `pto.tcvt`（`tile.cast` 的 lowering）输出保持 **plain Vec layout** → cube 读 garbage rows。
  - GM-copied INT8 在 authoring time 已预 fractalize → 正确。
- `src/backend/common/pto_ops_common.cpp:3382-3390`（`tile.cast`→`pto.tcvt` via `MakeTileCvtCodegenPTO`），`:2307-2312`（`tile.matmul_mx*`）；`src/ir/op/tile_ops/unary.cpp:112-152`（`DeduceTileCastType` — INT8 cast 输出无 cube/fractal-layout 约束）。

### 3.2 历史定位纠正（2026-07-09 晚，推翻 matmul_mx 定位）

dump FAIL 探针 `_probe_fixb_onthefly.py --smoke` 的 passes_dump：

```
registered tile.matmul      99×
registered matmul_acc       99×
registered tile.matmul_mx   0×   ← 该 op 从不 emit（A5-only，910B 无 codegen）
```

真因 = in-kernel `cast(→INT8)` 结果喂 cube Left 操作数时，pypto 给它默认 `fractal=512`（`GetImplicitTileView` 只对 Acc 推 fractal，`ComputeRewrittenType:439` 复制该错值）→ INT8 cube 读错行。经 materialized tensor 的 `tile.load` 会被 DMA 正确 fractalize，故 **GM/staged INT8 对、in-kernel cast 错**。

这条 IR-level 定位推翻了早期"matmul_mx op 本身有问题"的猜测（该 op 从不 emit）。

### 3.3 上游无修复

upstream-scout 搜 `git log --all` for `int8`/`quant`/`tcvt`/`cast`/`cube`/`w8a8`/`matmul_mx` —— **无 commit 修此 bug**。五组件全升级（含 ptoas-bin v0.49，LLVM21）**仍未修复**。DeepSeek-v4 W8A8 在 separate program 里量化、从 GM 读 int8，**从不走 in-kernel cast→cube**——这条路径在 production/CI 都未被 exercise，这就是上游 CI 漏检的原因。

## 4. 如何解决（Fix）

### 4.1 修复方向：dispatch-side quant（model-side staging）

不改 compiler。对齐 DeepSeek v4 `expert_routed.py` + vLLM W8A8：

- **Option A（dispatch-side quant，DeepSeek 精确对齐 + a2a 半字节 + recv_x 预 fractal 化）**：routed 输入在 dispatch 侧 quant 成 INT8，a2a 传 INT8（半字节带宽），recv_x 进 expert 前已预 fractal 化 → cube A-operand 来自 materialized tensor，避开 in-kernel cast。
- moe.py 已 device-validated（`moe.py cd3ef0d` pushed）：真 INT8×INT8 W8A8 routed MoE（DeepSeek v4 + vLLM-aligned），`_quant_moe_input` = scheduled `InCore(pl.range)` 2-output kernel。
- 用户决策（STATUS.md 续⁸）：确认走 dispatch-side 量化（Option A）。

### 4.2 当前 production 路径：BF16-dequant

INT8-native gated OFF（`select_moe_block(..., w8a8_native=False)`）。BF16-dequant 是工作路径（`0.9995` vs vLLM）。

⚠ **BF16-dequant 路径现已禁用作为 target**：47GB/rank pool + vLLM 24GB → 64GB OOM 边缘，且不是真正的 W8A8 出口。INT8-native 是 target，gap-5 是其唯一阻塞。

### 4.3 修复进度

- device 里程碑（cards 8-15，restored INT8-native moe.py，`--layer 3 --w8a8-native --target out --bypass-gate` vs vLLM `out.pt`）：**next_hidden_out 24.31%**（127458/524288，was 84%），干净 10s 无 507018。materialization 已解 cast→cube codegen bug（84%→24%）。
- 剩余 24% = device 侧 INT8 精度残差（**非 codegen garbage**）：误差 ~0.05 刚过 atol=0.04；rounding 正确（INT32-rint→INT8-trunc）；CPU 复算同方案 0.9998 cos。嫌疑 = down-leg `[RECV_TILE,1]` h_scale_tile 重量化 bridge / dequant fp32-vs-bf16 / partial-tile。下一步 `--dump-stages` 定位 dispatch vs expert-compute（Phase-21 式精度对齐）。

### 4.4 上游 issue 草案

hw-analyst 起草：`GetImplicitTileView` 对 Left/Mat 不推 cube fractal（只 Acc 推）+ P2(PASS)/FIXB(FAIL) 最小复现 + v0.49 仍 FAIL 证据，待提 `hw-native-sys/pypto`。建议上游修：把 `tile.matmul_mx*` 加进 memory-space inference，或强制对"target dtype=INT8 且 sole consumer=cube op"的 `tile.cast` 输出 re-fractalize/pad。

## 5. 走过的弯路（Detours / What We Got Wrong）

本事故被 refuted **3 次**才定位到 IR-level 真因。每次都有 falsifying experiment。

### 5.1 ❌ 假设 1："`tile.matmul_mx*` op 本身 lowering 有 bug"

- 依据：FIXB 探针 ~98% wrong，`matmul_mx` 是 INT8 cube 的自然嫌疑 op。
- 证伪实验：dump FAIL 探针 `_probe_fixb_onthefly.py --smoke` 的 passes_dump → 只有 registered `tile.matmul`（99×）+ `matmul_acc`（99×），**0 个 `tile.matmul_mx`**。该 op 从不 emit（A5-only，910B 无 codegen）。
- 结论：bug 不在 `matmul_mx` lowering，在它的 A-operand layout 推导。真因 = `GetImplicitTileView` 只对 Acc 推 fractal，Left/Mat 用默认 `fractal=512`，INT8 cube fractal=32 → 读错行。

### 5.2 ❌ 假设 2："in-expert INT8 量化路径"（in-expert quant）

- 依据：whole-net inlined MoE 是 INT8-native（in-expert quant），`_quant_moe_input` count `decode_layer.py=0` vs `moe.py=2`，与 standalone-validated moe.py INT8 kernel **decoupled**。单层 MoE（`P_FAITHFUL_MOE_LAYERS=1`）即 NaN（`next_hidden=nan / argmax=0`），看似 in-expert quant 本身有问题。
- 证伪实验 1：A-operand padding 假设 DISPROVEN —— 按 gap-5 fix 把 `routed_x_quant`/`routed_h_quant` 的 `x_i8`/`h_i8` padding 行 `[tile_valid:RECV_TILE]` 用 `fillpad(set_validshape(cast_tile,...),zero)` 置零 → 编译+跑通但**仍 NaN**。即 NaN **不在** fractal-32 A-operand padding，在有效行计算里。
- 证伪实验 2：standalone moe.py MoE-block harness 喂同一真 IPC 权重 → 不 NaN。即问题不在 in-expert quant 数学，在 whole-net inlined copy 与 moe.py 的 decoupling（inlined copy 有 11+ OWN inlined MoE copies，ZERO `_quant_moe_input`）。
- 结论：in-expert quant 本身对（moe.py validated），whole-net inlined copy 没接上。真因仍回到 cast→cube codegen。

### 5.3 ❌ 假设 3："two-class quant"（gap-5 WIP stash 之一）

- 依据：`gap5-wip+splitincore-20260709` stash 含 two-class + INT8-native + resid1 harness 三版实验，试图通过分两类量化路径绕过。
- 证伪实验：两版 stash moe.py 同样 43/45 层编译（与 committed `94aa015c` 持平），但 device 精度仍 84%（未解 cast→cube）。materialization 解法（dispatch-side quant + recv_x 预 fractal 化）才把 84%→24%。
- 结论：two-class 是症状层 patch，没触 IR-level 根因；materialization（让 INT8 operand 从 GM 进而非 in-kernel cast）才是正解。
- 处置：gap-5 WIP（two-class + INT8-native + resid1 harness）已 `git stash`，不进 clean base。

### 5.4 ❌ 附：早期"BF16-dequant 是可接受 target"的假设

- 依据：BF16-dequant `0.9995` vs vLLM，整网 e2e 跑通。
- 证伪：47GB/rank pool + vLLM 24GB → 64GB OOM 边缘（G3 HBM 共存 gate）；且 BF16-dequant 不是真正的 W8A8 出口，验收口径必须是 live token-exact A/B（INT8-native）。
- 结论：BF16-dequant 是**临时工作路径**，现在是**禁用 target**。INT8-native 是唯一 target，gap-5 是其唯一阻塞。

## 6. 如何避免（Prevention）

- **任何 in-kernel `pl.cast(→INT8)` 喂 cube A-operand 都是 suspect**。早期识别信号：INT8 cube matmul 输出 ~98% wrong、`max diff ~254`、**no device fault** → 立刻怀疑 `infer_tile_memory_space_pass` 未推 INT8 cube fractal。控制组：同 shape INT8 operand 从 GM 拷进来（预 fractal 化）→ PASS 即确认。
- **对齐 DeepSeek/Qwen FIRST，不要自创 on-the-fly 量化路径**。DeepSeek-v4 W8A8 在 separate program 里量化、从 GM 读 int8，从不走 in-kernel cast→cube——这条路径在 production/CI 都未被 exercise，这就是上游 CI 漏检的原因。自创 on-the-fly 分歧在正确性 + 性能上都更差（多一次每-expert 重量化 + BF16 a2a 翻倍）。
- **`matmul_mx` 不 emit 不代表 INT8 cube 路径没问题**。passes_dump 数 op 出现次数是必要排查；0× `matmul_mx` + 仍 FAIL → 真因在 A-operand layout，不在 op lowering。
- **whole-net inlined copy 要与 standalone kernel 保持 decouple 监控**。`_quant_moe_input` count `decode_layer.py=0` vs `moe.py=2` 是 inlined copy 与 validated kernel decoupled 的直接证据；whole-net inlined MoE 有 11+ OWN copies，任何 kernel 修复都要同步到 inlined copy 或 rebuild from generator（参 `postmortems/05-splitincoreorch-swiglu-l43-l44.md` 同类问题）。
- **gap-5 的 3 次 refuted 假设都是"假设写在事实前面"**。铁律：falsify-before-assert；每次假设必须配 falsifying experiment（见 `feedback_align_deepseek_architecture_first` memory：先产出 step3p5-vs-DeepSeek 对齐表，不要 deep-dive codegen）。
- **相关约束落点**：`pypto-lib/docs/upstream-issues/gap5-cast-int8-cube-codegen.md`（file-ready 上游报告）；`design/whole-net/02-detailed-design.md` §5（W8A8 native INT8 routed MoE 设计，5 步 dequant 链，dispatch-side quant Option A）；`postmortems/09-attention-multiposition-corruption.md`（同属 attention/MoE codegen 回归类）；`pypto-lib/docs/known-pypto-pitfalls.md`（pypto kernel 编码坑汇总）。
