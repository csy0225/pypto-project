# Troubleshooting —— swiglu MoE 层 (L43/L44) SplitIncoreOrch 编译失败（升级栈）

> 适用：升级栈（pypto `5e619dc7` / origin-main-rebase，含 #1828 precondition safety-net）
> 上编译 step3p5 **swiglu 变体 MoE 层**（L43 `swa_moe_swiglu7_silu`、L44
> `full_moe_swiglu7_swiglu16`）时，chip_orch codegen precondition 失败。
> 定位于 2026-07-10（承接 tmov fix 之后的整网 Option-C 编译）。

## 症状

Option-C 整网编译 sweep（`/tmp/wholenet_optc_compile.py`，clean pypto 5e619dc7 +
tmov model-fix + ptoas v0.45）：

```
=== Option-C whole-network compile: 43/45 layers PASS ===
=== RESULT: FAIL layers [43, 44] ===
```

L0（dense_full）/ L1,L2（dense_swa）/ L3–L42（silu MoE，Option-C）全部 **COMPILE OK**；
唯 **L43/L44（swiglu 变体 MoE）** 失败：

```
Failed to generate orchestration 'chip_orch': Verification failed after
'GenerateOrchestration preconditions' for properties {SplitIncoreOrch, ...}:
[1] ERROR - SplitIncoreOrch
  Message: InCore ScopeStmt found in non-InCore function (should have been outlined)
  Location: models/step3p5/moe.py:1813  (_quant_moe_input 的 `for qtg in pl.spmd(...)`)
```

## 根因

- `_quant_moe_input`（`@pl.function(type=InCore)`，moe.py:1800）体内的 `pl.spmd`
  (`moe_input_quant`) 是合法的 InCore scope（routed 输入 per-token INT8 dynamic-quant，
  两条路径都用）。
- #1828 的 `GenerateOrchestration` precondition（origin-main-rebase 引入，`be90f992` 无）
  要求：inline 进 **非 InCore 的 chip_orch（Orchestration）** 的 InCore scope 必须先被
  `outline_incore_scopes_pass` outline 成独立 InCore 函数。
- **silu 的 chip_orch**：`_quant_moe_input` 的 spmd 被正确 outline → PASS（L3–L42）。
- **swiglu 的 chip_orch**：结构不同（多了 swiglu_limit clamp 段），导致 outline pass
  **漏 outline** `_quant_moe_input` 的 spmd → precondition 报错。
- 即：**#1828 precondition × swiglu chip_orch 结构 = _quant_moe_input spmd 未 outline**。
  与 tmov（#1601）、silu-SplitIncoreOrch（b511da0e 已修 `_serialize_after_shared`/
  `_zero_routed_y_buf` 的冗余 pl.at）同属「升级栈 codegen 回归」类，但**这是新实例**：
  b511da0e 只覆盖 silu；swiglu 的 `_quant_moe_input` spmd 未覆盖。

## 已验证 / 排除

- 非 tmov（tmov model-fix 已生效，dense 层编过）。
- 非 stash-vs-committed（committed 94aa015c 与 gap5-wip stash 两版 moe.py 同样 43/45，
  L43/L44 同错）。
- `_serialize_after_shared`/`_zero_routed_y_buf` 已是干净无冗余 pl.at（b511da0e 生效）；
  当前失败点是 `_quant_moe_input` 的 spmd（1813），不同 helper。
- silu MoE（L3–L42）全过 → 问题限于 swiglu chip_orch 的 outline。

## 修复方向（待定，二选一）

1. **pypto codegen（outline pass）**：让 `outline_incore_scopes_pass` 对 swiglu chip_orch
   也 outline `_quant_moe_input` 的 spmd（对齐 silu 路径）。根因在 outline pass 对
   swiglu 结构的处理，或 #1828 precondition 过严。
2. **模型侧（mirror tmov fix 思路）**：调整 swiglu chip_orch 结构，使 `_quant_moe_input`
   的 InCore scope 与 silu 路径一致地被 outline（例如把 clamp 段与 quant 段的 scope
   组织对齐 silu）。需读 `_build_decode_layer_moe_program` / EpTpMoE chip_orch 的
   swiglu 分支 inline 顺序。

## 边界

- 整网 **43/45 层 COMPILE OK**；仅 L43（1 层）+ L44（1 层）swiglu 变体阻塞。
- L43/L44 是全网**仅有的 2 个非-silu MoE 层**（L43 swiglu7_silu / L44 swiglu7_swiglu16）。
- 阻塞**整网 45/45 编译 → device chain → live A/B**；dense + 40 silu MoE 层不受影响。
