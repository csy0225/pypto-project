# Troubleshooting —— Vec-LHS 矩阵乘在 910B 上的 Mat→Mat `pto.tmov` 编译失败

> 适用：升级栈（pypto `5e619dc7` / origin-main-rebase，含 PR #1601）上编译 step3p5
> **含 attention out_proj 的层**（decode_layer MoE / dense、attention_full/swa）时，
> ptoas 报 `'pto.tmov' op expects a supported tmov address-space pair for this target`。
> 定位于 2026-07-09/10（Phase 27 N=1 融合环境）。

## 症状

```
full_out_proj_matmul.pto:34:3: error: 'pto.tmov' op expects a supported tmov
address-space pair for this target
```
`_probe_moe_compile_perrank.py` / 任何编译 `DecodeLayerMoE` 或 attention 层的路径，
在 ptoas codegen 阶段失败（frontend/IR pass 全过，卡在把最终 IR 交给 ptoas）。

## 根因（pass-dump 逐 pass IR 确认）

1. step3p5 的 out_proj 矩阵乘左操作数 = `attn_out * gate_exp` 内联算在 **Vec**
   （`attention_full.py` Scope 3.a）→ **Vec-resident LHS**。
2. out_proj 的 Right 操作数（Wo chunk）= `[K_CHUNK=256, OUT_PROJ_N_CHUNK=256]` bf16
   = **128KB > 64KB L0B**，逼 `AutoTileMatmulL0`（pass #14）tiling。
3. **PR #1601（`0d3993b1`，origin/main rebase 引入，`be90f992` 无此提交）** 给
   Vec-LHS 矩阵乘加了 `stage_lhs_to_mat`：用 `BuildMoveToMat` 把 Vec LHS 经
   `tile.move(Vec→Mat)` staging，`ExpandMixedKernel`（pass #19）在 910B GM-pipe
   路径把它降为 `tpop_from_aiv(→Mat pipe slot)` + **把 slot copy-out 到持久 Mat tile**。
4. RHS 操作数的 copy-out 是 `Mat→Right`（合法 cube load）；**唯独 LHS staging 的
   copy-out 是 `Mat→Mat`**（`o_acc__tile_l0_lmat_mat → o_acc__tile_l0_lmat`，同 shape
   同 layout 纯定址搬运）。**pto-isa 的 `TMOV` 只支持 `Mat↔Vec`，不支持 `Mat→Mat`（无
   L1→L1 move 引擎）；且 AIC 侧无 Vec 可中转** → ptoas 拒绝。

一句话：**#1601 的 Vec-LHS staging × 910B GM-pipe 边界的 Mat→Mat copy-out = 非法。**

## 试过但不行（记录以免重复）

| 方案 | 结果 |
|------|------|
| 平台门控：910B 跳过 Vec-LHS tiling（`return nullopt`）| tmov 消失，但**重新触发 L0B 溢出**（un-tiled Right 128KB）——该 tiling 对 L0B 是**必需**的，不能禁 |
| `stage_lhs_to_mat=false`（从 Vec 直接 extract）| #1601 记载的 **dangling cross-boundary free variable**（非法） |
| 编译器合法中转 `Mat→Vec→Mat` | **AIC 侧无 Vec**，且 copy-out 在 GM-pipe 路径（非 `NeedsPostTpopMove` 分支）——需 defer-tfree/逐-chunk-pop 的深度 `ExpandMixedKernel` 重构（多 session） |
| 单 pass 回退到 #1601 之前 | = 跳过 Vec-LHS tiling = 同 L0B 溢出 |

## 修复（模型侧数据通路 reshape，pypto C++ 不动）

**思路**：让 out_proj 矩阵乘小到 **L0-sized**，则 `AutoTileMatmulL0` 直接放行（不 tiling
→ 不 staging → 无 Mat→Mat tmov，也无溢出）。

`pypto-lib`（fusion 分支）3 处、5/5 行：

1. `models/step3p5/config.py`：`OUT_PROJ_N_CHUNK` **256 → 64**
   （out_proj 矩阵乘 `[16,256]@[256,64]`：Right 32KB / Left 8KB / Acc 16KB 全落 L0）。
2. `models/step3p5/attention_full.py`：out_proj cast 的 `fp32_chunk` → `oproj_fp32_chunk`。
3. `models/step3p5/decode_layer.py`：dense down-cast 的 `fp32_chunk` → `dense_fp32_chunk`
   （chunk 变小后二者 shape 不再相等，内联进同一 chip_orch scope 会同名冲突
   `Cannot reassign 'fp32_chunk' with a different type: [16,64] vs [16,256]`）。

**验证**：`_probe_moe_compile_perrank.py` → `COMPILE OK rc=0`（DecodeLayerMoE，升级栈
pypto `5e619dc7` + ptoas v0.45）。

### ⚠ 关键陷阱：stale `.pyc`

改完 `attention_full.py` 第一次重编仍报同一 `fp32_chunk` 冲突——是 **stale 字节码**：
`apply_perrank_patch()` monkey-patch 了 config 全局，`__pycache__/*.pyc` 序列化了旧值。
**每次改模型后必须**：

```bash
find <pypto-lib>/models/step3p5 -name "*.py" -exec touch {} +
```

（触碰源码 mtime > pyc mtime 即失效，无需删 pyc；见铁律 §3 / `dev-workflow-gotchas.md`。）

## 诚实边界 + 后续

- **仅 COMPILE 验证**；device 运行 + 数值正确性待复核（chunk 变细是同一数学的等价
  tiling，风险低，但需 device gate）。
- **perf**：out_proj spmd 迭代 ×4（256→64）。可后续按平台门控 `OUT_PROJ_N_CHUNK`
  或等上游修编译器后调回。
- **上游 bug**（应提 issue）：`ExpandMixedKernel` 在 910B GM-pipe 路径对 **Mat-dest**
  的 CV-boundary copy-out 产 `Mat→Mat` `pto.tmov`——任何**需 tiling 的 Vec-LHS 矩阵乘**
  都会中招。根因方向：让 tpop 直接落到持久 Mat（extract 读 slot + tfree 延后），
  或逐-chunk-pop（如 RHS 的 `Mat→Right`）。

## 经验法则（记忆）

编译器/codegen 失败先判断是否**架构相关**：
- 若某 pass **不必需**（禁用不影响功能）→ 平台门控（`BackendHandler` 谓词）或取消注册。
- 若该 pass **必需**（如本例 tiling 对 L0B 必需）→ **改数据通路让它不触发问题路径**
  （本例：缩 chunk 让矩阵乘 L0-sized），而非禁用。
