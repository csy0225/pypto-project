# 专项：swiglu MoE 层 (L43/L44) SplitIncoreOrch 编译 precondition 失败（`InCore ScopeStmt found in non-InCore function`）

| 字段 | 值 |
|------|----|
| **子系统** | codegen |
| **error signature** | `SplitIncoreOrch: InCore ScopeStmt found in non-InCore function (should have been outlined)` |
| **首次出现** | 2026-07-10 |
| **状态** | 🟡 缓解（定位到 pass-level；根因 pass 未最终锁定） |
| **相关 skill / doc** | PR #1828 / commit `b511da0e` / `pypto-lib/docs/known-pypto-pitfalls.md` |

## 1. 背景（Background）

升级栈（pypto `5e619dc7` / origin-main-rebase，含 #1828 precondition safety-net；`be90f992` 无此提交）上编译 step3p5 **swiglu 变体 MoE 层**（L43 `swa_moe_swiglu7_silu`、L44 `full_moe_swiglu7_swiglu16`）时，chip_orch codegen precondition 失败。

场景：承接 tmov fix 之后的整网 Option-C 编译，定位于 2026-07-10。

命令：Option-C 整网编译 sweep（`/tmp/wholenet_optc_compile.py`，clean pypto `5e619dc7` + tmov model-fix + ptoas v0.45）。

目标：整网 45/45 层 COMPILE OK，打通 device chain → live A/B。

## 2. 现象（Symptom）

```
=== Option-C whole-network compile: 43/45 layers PASS ===
=== RESULT: FAIL layers [43, 44] ===
```

L0（dense_full）/ L1,L2（dense_swa）/ L3–L42（silu MoE，Option-C）全部 **COMPILE OK**；唯 **L43/L44（swiglu 变体 MoE）** 失败：

```
Failed to generate orchestration 'chip_orch': Verification failed after
'GenerateOrchestration preconditions' for properties {SplitIncoreOrch, ...}:
[1] ERROR - SplitIncoreOrch
  Message: InCore ScopeStmt found in non-InCore function (should have been outlined)
  Location: models/step3p5/moe.py:1813  (_quant_moe_input 的 `for qtg in pl.spmd(...)`)
```

- 43/45 层 COMPILE OK；仅 L43（1 层）+ L44（1 层）swiglu 变体阻塞。
- L43/L44 是全网**仅有的 2 个非-silu MoE 层**（L43 swiglu7_silu / L44 swiglu7_swiglu16）。

## 3. 根因（Root Cause）

- `_quant_moe_input`（`@pl.function(type=InCore)`，`moe.py:1800`）体内的 `pl.spmd`（`moe_input_quant`）是合法的 InCore scope（routed 输入 per-token INT8 dynamic-quant，两条路径都用）。
- #1828 的 `GenerateOrchestration` precondition（origin-main-rebase 引入，`be90f992` 无）要求：inline 进 **非 InCore 的 chip_orch（Orchestration）** 的 InCore scope 必须先被 `outline_incore_scopes_pass` outline 成独立 InCore 函数。
- **silu 的 chip_orch**：`_quant_moe_input` 的 spmd 被正确 outline → PASS（L3–L42）。
- **swiglu 的 chip_orch**：结构不同（多了 swiglu_limit clamp 段），导致 outline pass **漏 outline** `_quant_moe_input` 的 spmd → precondition 报错。
- 即：**#1828 precondition × swiglu chip_orch 结构 = _quant_moe_input spmd 未 outline**。与 tmov（#1601）、silu-SplitIncoreOrch（`b511da0e` 已修 `_serialize_after_shared` / `_zero_routed_y_buf` 的冗余 `pl.at`）同属「升级栈 codegen 回归」类，但**这是新实例**：`b511da0e` 只覆盖 silu；swiglu 的 `_quant_moe_input` spmd 未覆盖。

### 深挖更新（2026-07-10，实测排除法）

- **已 dump pass IR 逐 pass 排查**：`09_after_OutlineIncoreScopes.py` 显示 `_quant_moe_input` **已被正确 outline 成独立 `@pl.function(InCore)`（call 在 chip_orch）**；`expert_gate_up` / `expert_down` / `routed_dyn_quant` 也都是 `with pl.spmd(...): self.<method>(...)` 形式。即在 pass 09 时并无未 outline 的 InCore scope。但 precondition verifier 在**所有 pass 之后**（`codegen_preconditions.cpp:71`，pass 43 之后）才报错 → **某个 09 之后的 pass 重新引入/未能重新 outline 了 InCore scope**。traceback 含 **`split_vector_kernel_pass.cpp:277`（SplitVectorKernel，pass #22）**——疑似该 pass 对 swiglu（更多 vec 算子/clamp）产出的结构与 silu 不同，留下未 outline 的 InCore scope。

## 4. 如何解决（Fix）

修复方向（待定，二选一）：

1. **pypto codegen（outline pass）**：让 `outline_incore_scopes_pass` 对 swiglu chip_orch 也 outline `_quant_moe_input` 的 spmd（对齐 silu 路径）。根因在 outline pass 对 swiglu 结构的处理，或 #1828 precondition 过严。
2. **模型侧（mirror tmov fix 思路）**：调整 swiglu chip_orch 结构，使 `_quant_moe_input` 的 InCore scope 与 silu 路径一致地被 outline（例如把 clamp 段与 quant 段的 scope 组织对齐 silu）。需读 `_build_decode_layer_moe_program` / EpTpMoE chip_orch 的 swiglu 分支 inline 顺序。

**下一步定位法**：dump silu(L3) 与 swiglu(L44) 的**每一个** pass IR（09→43），在 `_quant_moe_input`/其它 InCore scope 上逐 pass diff，找到 **swiglu 侧从"已 outline"变回"未 outline"或新增未 outline InCore scope 的那个 pass**（首要嫌疑 SplitVectorKernel #22）。需 pass-level 调试，非表层结构猜测（本轮两个结构猜测均被实测证伪）。

### 已验证 / 排除

- 非 tmov（tmov model-fix 已生效，dense 层编过）。
- 非 stash-vs-committed（committed `94aa015c` 与 `gap5-wip` stash 两版 `moe.py` 同样 43/45，L43/L44 同错）。
- `_serialize_after_shared` / `_zero_routed_y_buf` 已是干净无冗余 `pl.at`（`b511da0e` 生效）；当前失败点是 `_quant_moe_input` 的 spmd（1813），不同 helper。
- silu MoE（L3–L42）全过 → 问题限于 swiglu chip_orch 的 outline。

## 5. 走过的弯路（Detours / What We Got Wrong）

- ❌ 假设：根因是 `routed_dyn_quant` 中间重量化 spmd（`moe.py:1118` 的 `if _routed_swiglu_step:` 分支）→ 证伪：把 `if _routed_swiglu_step:` 改 `if False:` 后 **L44 仍在 `moe.py:1813` 同样 SplitIncoreOrch FAIL**。`routed_dyn_quant` **不是**根因。
- ❌ 假设：根因是 `_quant_moe_input` 本身结构有问题 → 证伪：silu（L3–L42）用**同一** `_quant_moe_input` 却编过 → 根因是 swiglu 结构（clamp 等）在 **09 之后的某 pass** 上与 silu 分叉，非某单一 helper。
- ❌ 假设：可能是 `matmul_mx` 相关 gap5 问题 → 排除：见 gap5 记忆；与本 precondition 无关。
- ❌ 表层结构猜测（2 次均被实测证伪）：未做 pass-level diff 前凭 chip_orch 结构猜改动点 → 均无效。结论：必须 pass-level IR diff 定位，非表层结构猜测。
- ❌ 误判 outline pass 的责任边界：pass 09（`OutlineIncoreScopes`）已正确 outline，但 pass 22（`SplitVectorKernel`）疑似重新引入未 outline InCore scope → 早期只盯 pass 09 浪费了时间。

## 6. 如何避免（Prevention）

- **早期识别信号**：升级栈（含 #1828）编译 MoE 层报 `InCore ScopeStmt found in non-InCore function` + silu 变体过 / swiglu 变体挂 → 立刻怀疑 outline pass × swiglu chip_orch 结构交互，不要只盯报错点那个 helper。
- **pass-level diff 先于结构猜测**：两个相似层（一个 PASS 一个 FAIL）的 codegen 分叉，先 dump 每一个 pass 的 IR diff，定位"从 PASS 状态变回 FAIL 状态"的那个 pass，再读该 pass 源码；凭 chip_orch 表层结构猜改动点在本类问题上命中率低。
- **升级栈 codegen 回归需逐变体覆盖**：`b511da0e` 只覆盖 silu 路径；新增 codegen precondition 后，每个 model 变体（silu / swiglu7 / swiglu16 / swiglu7_swiglu16 等）都要单独 compile-gate，不能靠"silu 过等于 MoE 全过"。
- **落点**：`pypto-lib/docs/known-pypto-pitfalls.md`（#1828 precondition × swiglu 变体）；pass-level diff 方法论可进 `pypto-lib/docs/dev-workflow-gotchas.md`。
