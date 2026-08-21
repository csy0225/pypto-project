# vLLM-Ascend decode/MoE Trace 对齐与 PyPTO 整改进展（2026-08-12）

> **状态：已在 0162 按顺序实施、验证并回滚失败候选；三方复核后确认完整
> MoE `<200 us` 尚未达到，阶段/整网 parity 仍为 fail-closed。**
>
> 所有源码、脚本、编译、设备测试和证据均位于 0162。本地项目只更新本文。
> 当前保留的正向方案是 **pair-grid combine + adaptive routed grids +
> 4096-wide combine-reduce/residual fusion**；产品代码未 commit/push。
>
> **口径纠正：** `97.01 us` 只是 routed expert 子路径，不是完整 MoE；
> `29.767 ms` 是 synthetic Main-only hidden-forward wall time，不是与 vLLM
> Main+MTP3 或 service ITL 对齐后的结果。

## 1. 当前结论

1. 0162 clean main 已同步并再次核验为
   `stepfun/develop@69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d`；2026-08-12
   的 `git ls-remote` 结果仍指向同一 SHA。
2. 固定验证镜像为
   `sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3`，
   config 为
   `sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39`。
   镜像内 `pypto-lib@cb96747e`，当前 source overlay 为 `69ad31e4`，前进 7 个
   commit；不能写成“源码与镜像完全同 commit”。
3. vLLM Trace 中没有 literal `decode` marker。依据 graph replay 结构定位到
   68 个 decode step：每步 45 层 Main，随后 MTP3；Main 的 L3–L44 共 42 个
   MoE。
4. 捕获到的 vLLM MoE 是 **local routing + grouped experts + standalone TP
   AllReduce**；PyPTO 是 **EP8 dispatch/combine + shared TP AllReduce**。
   两边可以按语义阶段比较，但不能按 kernel 名称或 kernel 数量一比一替换。
5. 已完成的正式 whole-network A/B/A：
   - 两个 GMM1 fusion 候选均回退，**NO-GO**；
   - pair-grid combine 稳定正收益，**GO**；
   - source-grid combine 整网回退，**NO-GO**；
   - pair + adaptive routed grids 相对冻结 pair 基线再取得正收益，**GO**。
6. 当前保留代码的正式增量 A/B/A 为：

   ```text
   pair A1 / pair+adaptive B / pair A2
   31.130 / 29.767 / 29.898 ms
   baseline midpoint  30.514 ms
   floor              0.616 ms
   delta             -0.747 ms / -2.448%
   delta / floor      1.213×
   verdict            IMPROVEMENT_BEYOND_BRACKET
   precision          PASS
   ```

   A1/A2 漂移较大，因此不能把 `-0.747 ms` 当作绝对加速承诺；但 B 仍比两次
   A 中更快的一次低 `0.131 ms`，并按预先固定的 half-range 规则击穿 bracket。
7. 第二项低风险优化 **combine-reduce + residual fusion** 已完成并集成到
   0162 integration tree。旧/融合4096/旧的 all-rank median physical final
   envelope 为：
   - L3：`15.16 / 5.58 / 15.04 us`；
   - L4：`15.56 / 4.61 / 15.40 us`。
   4096 版本 **GO**；2048 版本在 4096/2048/4096 A/B/A 中稳定回退，
   **NO-GO**；不再测试 1024，也不使用 Cube kernel。
8. 后续 pure-AIV `act+quant` 融合已按门禁停止：R4 因 16B FP32 Vec row
   不满足 32B 合同而静态 NO-GO；R8 三次固定镜像 compile 均失败，未上设备。
   该失败候选已回退，不在当前 integration tree 中。
9. 当前不存在 authority-complete、同 workload 的 vLLM/PyPTO parity harness。
   Trace 与 PyPTO harness 在输入、真实 KV、Main/MTP3 计时边界和拓扑上均未
   对齐，所以本文只证明 PyPTO 内部候选的相对收益，不证明 vLLM parity。

## 2. 源码、镜像和集成基线

### 2.1 clean main 与 integration tree

```text
clean main
  /mnt/persist/chensiyu/workspace/develop/pypto-lib
  branch  stepfun/develop
  HEAD    69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
  status  clean

integration
  /mnt/persist/chensiyu/workspace/develop-worktrees/vllm-moe-opt-20260812
  branch  perf/vllm-moe-opt-20260812
  HEAD    69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
```

最终 integration tree 仅两个文件 dirty：

```text
models/step3p5/decode_fwd.py
tests/step3p5/unit/test_performance_bc_contract.py
```

最终权威 SHA：

```text
decode  de5712383775823eef82ce6b082d04ac54cf232f93452a1d9090550ac797d936
test    d394ea4ce5aeba23a07e8f0f0a1f5be0c22d3b3d56c38412b2a00e1047fa2038
```

`git diff --check` 和 Ruff PASS；无 Torch 的 host 静态合同为
`31 passed, 5 deselected`，固定镜像直接执行 5 个数学合同全部 PASS。这只
表示目标合同集通过，不应扩写为全仓 unit suite PASS。

### 2.2 固定镜像

```text
image
  hub.i.basemind.com/stepcast/vllm-pypto@
  sha256:076af8a167405d5d0831e234cd16521c77d8bfdd173eff063d820802057c47f3

config
  sha256:a9d111880883cea0b02e425fdfeaccc2b14bb1d1174c0b73488d8ee6d8004d39
```

镜像 pins：

```text
pypto       1c048a744d5f63a8bce1ddb45dac8d1b7f458bb0
pypto-lib   cb96747eb21f5f4932d6a24eddaa69c85d095ef6
pto-isa     ecb6c303f797749f811a494742c3c08156aacabb
PTOAS       fc8c6caee561914b4fb991dfc8427bb63194269e
simpler     e2efebcbd190302609c0775d2984f409f5f42c76
vllm-patch  1b3e538c35999e62b6d24e0651b3a85b7d16c826
```

验证形态是 immutable image substrate + read-only `pypto-lib@69ad31e4`
source overlay。当前 source 相比镜像 `pypto-lib` 多 7 个提交，其中最新一项是
single-row TP AllReduce 优化。

## 3. vLLM Trace 中 decode 和 MoE 的定位

Trace：

```text
/mnt/persist/chensiyu/workspace/develop/trace_view (3).json
SHA256 ddba08673aa3787147c261403223995b2b345219911317c9f50c05215d65abf2
```

JSON 不含 literal `decode` range；定位依据是固定 replay 拓扑：

- Model 87 重放 68 次；每次 45 个 Main attention、42 个 MoE；
- 每个 Model 87 后紧随一个 3 层 MTP3 Model 86；
- 68 次 replay 分成两段各 34 步；
- CPU build 与 device graph 的 attention/MoE 计数一致。

因此以下结果是“按 graph replay 结构识别的 decode”，不是依赖字符串 marker。

| Trace 指标 | 结果 |
|---|---:|
| Main Model 87 span p50 / p95 | `18.067 / 21.585 ms` |
| Main + MTP3 span p50 / p95 | `27.138 / 30.676 ms` |
| start-to-start period p50 / p95 | `31.345 / 34.935 ms` |
| Main 层数 | `45` |
| MoE 层 | L3–L44，共 `42` |
| 全 Trace MoE core 次数 | `68 × 42 = 2856` |
| 单 MoE core p50 / p95 | `229.75 / 250.50 us` |
| 42 层 MoE core span 合计 p50 | `9.689 ms/step` |

代表性 regular L3–L42 路径：

```text
RMSNorm / Cast
→ router FP32 MatMul
→ MoeGatingTopK 与索引处理
→ MoeInitRoutingCustom / Cumsum
→ GroupedMatmulSwigluQuant
→ GroupedMatmul (down)
→ MoeTokenUnpermute
→ shared MatMul / SwiGLU / MatMul
→ routed + shared Add
→ standalone HCCL AllReduce
→ residual Add
```

关键 p50：

| vLLM kernel | p50 |
|---|---:|
| router FP32 MatMul | `12.68 us` |
| `MoeGatingTopK` | `4.84 us` |
| `MoeInitRoutingCustom` | `11.52 us` |
| `Cumsum` | `28.19 us` |
| `GroupedMatmulSwigluQuant` | `43.98 us` |
| routed down `GroupedMatmul` | `28.80 us` |
| `MoeTokenUnpermute` | `3.68 us` |
| MoE-output HCCL AllReduce | `22.28 us` |

通信计数显示 Main 每步 91 次 AllReduce，AllGather/AllToAll/MC2/FUSED_MC2
均为 0。HCCL event 与同时间的 AIV row 是同一通信的两种展示，不能重复计时。

L43/L44 是特殊路径：普通 GMM1 后执行 Slice/Swish/Clip/Mul、DynamicQuant，
再执行 GMM2；它们不使用 regular `GroupedMatmulSwigluQuant`。PyPTO 也保留
独立 `swiglu7` specialization，本轮从未把 special 层强制并入 regular fusion。

## 4. PyPTO 当前路径

最终保留的 regular MoE 路径：

```text
attention
→ deferred norm + BF16/INT8/scale producers
├─ gate_expert_fanout → gate_topk
├─ shared: sh_gate_up_act → sh_down → TP AllReduce
└─ routed:
   dispatch_count_publish / dispatch_meta
   → dispatch_push / wait / gather
   → expert_gate_up
   → expert_gate_up_act
   → routed_h_quant
   → expert_down
   → pair-grid combine_scatter
   → combine_wait
   → combine_reduce_residual
```

whole decode 是 persistent `@pl.program`；shared 与 routed 是可重叠 DAG lane，
源码书写顺序不等于设备串行顺序。

当前两个已落地优化：

1. **pair-grid combine**：logical work item 从 active expert 改为
   `(active expert, source rank)`，物理 grid cap 为 36；每项写区间互斥。
   scatter 结束后显式 fence，单 control task 对每个 peer 聚合
   `AtomicAdd(36)`，wait threshold 仍为 `moe_epoch × 36`。
2. **adaptive routed grids**：gate-up 保持固定 `23/22` workers，以保留 early
   submit；读取 active-expert count 后只收缩下游无实际工作的 grid：

   ```text
   act    min(max(active_experts × 20, 1), routed_workers)
   quant  min(max(active_experts,      1), routed_workers)
   down   min(max(active_experts × 16, 1), routed_workers)
   ```

`expert_down` 的 scratch 与 `down_workers` 同变量缩放；L43/L44 仍用静态 special
grid。

3. **4096-wide combine-reduce/residual fusion**：删除 `moe_out` GM 中间张量
   和独立 `moe_residual_add` task，由 16 个 AIV block 各负责一个
   `[1, 4096]` token row。保持
   `FP32 TOPK reduce → BF16(rint) → FP32 + residual → BF16`，并覆盖
   regular/SWA/L43/L44 四条调用路径。

## 5. vLLM 阶段如何对应到 PyPTO

对齐单位是“输入状态、ownership 和输出状态的语义转换”，而不是同名 kernel：

```text
FFN input
→ norm/input preparation
→ router/TopK
→ route organization
→ expert input ready
→ GMM1 + activation + requant
→ down projection
→ token-order restore / TOPK reduce
→ shared/global merge
→ residual output
```

关键拓扑差异：

- vLLM Trace：本卡 permutation/unpermute，merge 后做 TP AllReduce；
- PyPTO：EP8 dispatch 到 36 个 local experts，再以 EP combine 返回；shared 分支
  单独做 TP AllReduce。

因此 vLLM 的 `MoeInitRoutingCustom/Cumsum/MoeTokenUnpermute` 只能对应 PyPTO
的 route organization / return 语义阶段，不能直接替换 EP 协议。

统一阶段 endpoint：

| 阶段 | vLLM endpoint | PyPTO endpoint |
|---|---|---|
| input ready | attention 输出进入 RMSNorm 前 | attention producer / `resid_hold` 完成 |
| norm/quant done | norm/Cast 结果可供 router/expert | `norm_quant_moe_input` 所有 outputs ready |
| router done | TopK/route metadata ready | `gate_topk` 完成 |
| route plan done | Cumsum/local permutation plan ready | `dispatch_meta` 完成 |
| expert input ready | grouped expert input ready | `dispatch_gather` 完成 |
| routed act/quant done | `GroupedMatmulSwigluQuant` 完成 | `routed_h_quant` 完成 |
| routed down done | down GMM 与 routed postprocess 完成 | `expert_down` 完成 |
| combine/global ready | unpermute、shared merge、TP AR 完成 | EP wait、stable TOPK reduce、shared merge 完成 |
| output ready | residual Add 完成 | 当前为 `combine_reduce_residual` 完成；旧版本为 `moe_residual_add` 完成 |

这些 span 是并行 DAG 的 wall span，不能简单相加。正式 DFX 必须对每层、每 rank
取阶段全部 physical tasks 的 min start / max end，不能取第一个同名 task。

## 6. 三列对比：vLLM、PyPTO 前、PyPTO 后

| vLLM | PyPTO 前（本轮优化前） | PyPTO 后（当前保留态） |
|---|---|---|
| **输入准备：** RMSNorm/Cast 后供 router 与 expert 使用。 | deferred norm 同时生产 BF16、INT8、scale，多 consumer DAG。 | 保持现有融合和 early resolve；不为减少 kernel 数重新串行化。 |
| **Router：** FP32 MM → `MoeGatingTopK` → 索引处理。 | `gate_expert_fanout → gate_topk`。 | 保持阶段边界；当前证据不支持强融 router。 |
| **Route organization：** `MoeInitRoutingCustom → Cumsum`，本卡 permutation。 | EP8 count/meta/push/wait/gather，完成跨 rank ownership 转换。 | 保留 EP8 语义；不能用 local permutation primitive 删除真实通信。 |
| **GMM1+activation+requant：** regular L3–L42 为单个 `GroupedMatmulSwigluQuant`。 | `expert_gate_up → expert_gate_up_act → routed_h_quant`，三个 task 均使用固定 routed grid，并物化 INT32/BF16 中间结果。 | 仍保留三个 task；两次 GMM fusion 和 pure-AIV act+quant 均 NO-GO。gate-up 固定早提交，act/quant 使用 adaptive grid。真正对齐需要 compiler primitive，而不是模型 DSL 拼接。 |
| **Down：** 独立 `GroupedMatmul`。 | 独立 `expert_down`，route weight 在 epilogue；固定 worker/scratch。 | 保持独立；仅按 active expert 收缩 worker 和 scratch，不改 route-weight/精度语义。 |
| **Restore：** local `MoeTokenUnpermute`。 | EP `combine_scatter → wait → reduce`；每个 expert 串行遍历 source ranks。 | **已 GO：** `(active expert, source rank)` write-disjoint work item，grid cap 36；scatter 后 fence，control task 聚合固定 36 credits。 |
| **Shared/global merge：** shared MLP、routed/shared Add、最终 standalone TP AllReduce。 | shared 是独立 DAG lane并先做 TP AR；`combine_reduce` 合并 shared 与 stable TOPK routed reduce。 | 保持 PyPTO 拓扑和 overlap；`combine_reduce_residual` 只融合稳定 reduce、双舍入和 residual，不把 `combine_wait` 塞入计算 kernel，也不做无证据 MC2。 |
| **Special/Residual：** L43/L44 独立非 fused expert 路径，最后 residual Add。 | L43/L44 `swiglu7` specialization，residual 独立。 | expert special 路径仍独立，但其最终 reduce/residual writer 也复用同一融合 task；四条调用路径均已静态和 generated audit 覆盖。 |
| **性能口径：** Trace Main p50 `18.067 ms`、Main+MTP3 `27.138 ms`、MoE core `229.75 us`。 | PyPTO harness 使用 synthetic BS1 ctx64K metadata、45 层 Main-only source overlay；不是同一输入/计时边界。 | 五层 DFX 的 scheduler-local `E0→E7` median 为 L3 `254.83 us`、L4 `260.52 us`；它仍只是描述性 L3/L4 截面，不是同 workload parity 或整网 ITL。 |

## 7. 按执行顺序的候选裁决

每轮都使用该轮 A1/A2 的 half-range 作为 floor，不复用历史固定阈值。

| 顺序 | 候选 | A1 / B / A2 p50 | 相对 A midpoint | floor | 裁决 |
|---:|---|---|---:|---:|---|
| 1 | GMM N128/NONE/slot2 | `29.717 / 30.386 / 29.859 ms` | `+0.598 ms / +2.008%` | `0.071 ms` | **NO-GO** |
| 2 | GMM N256/UP_DOWN/slot1 | `29.921 / 30.209 / 29.974 ms` | `+0.2615 ms / +0.873%` | `0.0265 ms` | **NO-GO**；另有 dynamic valid-shape 风险 |
| 3 | pair-grid combine | `29.902 / 29.836 / 29.998 ms` | `-0.114 ms / -0.381%` | `0.048 ms` | **GO**，`2.375× floor` |
| 4 | source-grid combine | `29.906 / 31.529 / 29.927 ms` | `+1.6125 ms / +5.390%` | `0.0105 ms` | **NO-GO** |
| 5 | pair + adaptive routed grids | `31.130 / 29.767 / 29.898 ms` | `-0.747 ms / -2.448%` | `0.616 ms` | **GO**，`1.213× floor` |
| 6 | pure-AIV act+quant R4 | 未进入 compile | N/A | N/A | **STATIC NO-GO**：FP32 `[1,4]` 只有 16B，违反 32B Vec row 合同 |
| 7 | pure-AIV act+quant R8 | 三次 fixed-image compile `rc=1` | N/A | N/A | **COMPILE NO-GO**；未上设备，已恢复 adaptive |

已完成 whole A/B/A 的五个候选均通过 hidden SHA、token、finite 精度门。source-grid
即使 focused DFX join 比 pair 快约 `11.41 us mean`，也不能推翻整网 `+5.39%`
回退。

## 8. adaptive routed-grid 的验证

### 8.1 compile 与 generated contract

固定镜像 whole compile：

```text
COMPILE_OK 74.377 s
decode SHA a94b43f0e5a7d4ea39891e217a6c686b0c32d7d6bd961989931047b90308556d
```

generated-contract audit：`11/11 PASS`。确认：

- gate task submit 在 active-count control read 之前；
- act/quant/down 的 block num 与 kernel stride 使用同一动态 scalar；
- down scratch 随 `down_workers` 缩放；
- deps、predicate、early resolve 均保持；
- pair combine 的 work/credit/fence 合同不变；
- L43/L44 special AST 和 generated kernels 不变。

### 8.2 five-layer 精度与 DFX

L3/L4 对 golden 和冻结 pair 均 byte-exact，8-rank DFX/dep-gen 完整。
canonical analyzer 对 zero-token task 的 `missing_on_swim` 仍返回 `rc=1`，但 pair
与 adaptive 的异常 JSON 完全相同；裁决为
`PASS_WITH_KNOWN_STRUCTURAL_ANALYZER_LIMITATION`，不是隐去 analyzer 失败。

| 指标 | Pair | Adaptive | 变化 |
|---|---:|---:|---:|
| routed logical blocks | `1472` | `848` | `-42.39%` |
| observed physical slices | `2208` | `1708` | `-22.64%` |
| routed envelope p50 | `103.62 us` | `97.01 us` | `-6.61 us` |
| data-wait p50 | `0.3685 ms` | `0.3585 ms` | `-0.0100 ms` |
| shared/routed envelope overlap p50 | `61.22 us` | `60.18 us` | `-1.04 us` |
| busy-union overlap | `45.71 us` | `47.65 us` | `+1.94 us` |
| five-layer harness p50 | `14.2732 ms` | `14.1480 ms` | `-0.1252 ms` |

five-layer 只有 3 次计时且包含 outlier，不能替代 whole A/B/A；它只负责准入和
解释 block/overlap 变化。

## 9. GMM/act+quant fusion 为何未保留，combine/residual 为何保留

### 9.1 直接 GMM1 fusion

N128 和 N256 两个候选都通过精度，但 whole A/B/A 分别回退 `+2.01%` 和
`+0.87%`。N256 还存在动态 valid-shape 正确性风险，因此不能因为 kernel 数
减少就默认启用。

### 9.2 pure-AIV `expert_gate_up_act_quant`

目标是删除 graph-wide BF16 bridge，同时保留：

```text
activation FP32 → BF16 round-trip → FP32 amax
N256 rowwise amax
mul(recip(amax), 127)
FP32 → INT32 rint → FP16 round → INT8 trunc
```

R4 在 compile 前被拒绝：FP32 `[1,4]` 仅 16B，低于 A2/A3 Vec/none-box 的
32B row 合同。

R8 解决对齐后仍三次 fixed-image compile 失败：

1. fused-local Tile 与 down Tensor 同名导致 SSA type conflict；
2. `tile.assemble` source/result pad mode 不一致；
3. 最终 PTOAS 的 non-mat `tmov` source 为 `v_col=?`，destination subview 为
   `v_col=64`，descriptor 不匹配。

按 fail-closed 门停止，没有通过 GM BF16 fallback、syncall、cross-core max 或
额外 wait 绕过。即使 descriptor 问题修复，R8 还会把 BS1 单 expert 的 activation
并行 work item 从约 20 降到 2，性能先验不佳。因此未进入设备门，并恢复
adaptive authority。

### 9.3 `combine_reduce + residual`

这个融合与 GMM1/act+quant 不同：它不跨 Cube/Vector 资源边界，不改变 EP
协议，也不删除模型要求的 BF16 物化语义，只把原来的 GM bridge 改成 kernel
内部显式 BF16 round-trip。旧/融合/旧重复显示 L3/L4 final envelope 分别稳定
回收约 `9.5/10.9 us`，因此 4096 版本保留。2048 把 block 从 16 增至 32，
setup、barrier 和 GM/MTE 竞争上升，已被 A/B/A 否决。

## 10. 后续整改方案

下一步仍以 vLLM 的**整体阶段粒度**为参考，但不要求 kernel 完全同名：

1. **冻结当前正向基线**：pair-grid + adaptive +
   4096-wide combine-reduce/residual fusion，保持现有 SHA、镜像和全部
   A/B/A/DFX 证据；任何新候选都以该 frozen source 做 A。
2. **不再做模型 DSL 的机械拼接**：regular GMM1+SwiGLU+requant 若继续推进，
   应新增 compiler primitive/lowering，满足：
   - 不物化完整 BF16 GM bridge；
   - 保留 BF16 round-trip 和原 requant cast chain；
   - 保留 expert×N 或等价的 AIV 并行度，不能退化为 1–2 个 row owner；
   - 原生支持动态 valid rows、32B Vec row、local staging assemble 和 full-row
     reduction descriptor；
   - down、route-weight、L43/L44 保持独立。
3. **先解决 primitive/codegen feasibility**：compile/PTO signature 必须证明
   BF16 仅在 Vec/local memory；UB `<188416 B`，建议 `<160 KiB`；失败立即
   NO-GO，不上设备。
4. **验证顺序固定**：静态合同 → fixed-image whole compile → intermediate
   `h_i8/scale` byte-exact probe → five-layer L3/L4 + special guard → DFX critical
   envelope → whole A/B/A。task 数减少不能替代 stage wall span 和 whole 结果。
5. **继续保留下游自适应粒度**：若新 primitive 未成熟，优先从 active-expert
   worker/scratch、route skew、combine tail 做可独立回滚的小改动；每项必须
   击穿自己的 A1/A2 bracket。
6. **不启动无证据 MC2**：本 Trace 没有 MC2/FUSED_MC2；只有新 profile 证明
   collective 成为当前关键路径且多仓能力齐备时，才进入通信计算融合。
7. **单独建立 parity campaign**：冻结同一 checkpoint 全分片 manifest、显式
   token IDs、真实 65536 KV、同卡 TP8、同 Main-only 或 Main+MTP3 边界、相同
   warmup/iters/seed，并保存两边输出 hash 和原始 latency。完成前不写“达到
   vLLM-Ascend”。

## 11. 为什么当前不能计算 vLLM/PyPTO 加速比

现有证据存在以下不可消除差异：

- Trace 没有启动命令、精确镜像/源码 commit、完整 checkpoint manifest、prompt
  token IDs 和 TP rank authority；
- vLLM Trace 的 `27.138 ms` 是 Main+MTP3，当前 PyPTO A/B/A harness 是 45 层
  Main-only；
- PyPTO 使用 synthetic BS1 ctx64K metadata，并非同一次真实 prefill KV；
- vLLM MoE 是 local grouped experts + TP AR，PyPTO 是 EP8 dispatch/combine；
- 现有 vanilla oracle 不含等价 MTP3/acceptance 边界。

所以不能把 PyPTO candidate `29.767 ms` 直接除以 vLLM `27.138 ms` 或
`18.067 ms`。当前目标仍是达到/超过 vLLM-Ascend，但状态只能写成：

```text
PyPTO 内部正向优化已验证；跨框架 parity 尚未建立。
```

## 12. 权威证据

```text
Trace 分析
  /mnt/persist/chensiyu/workspace/perf-2026q3/
    vllm-trace-moe-gap-20260812/TRACE_FINDINGS.md
  report SHA 89e1352e8d1dbb04b7a9ddb1e240945ab521912b1fbfe2280e17c9ce26e1ef2e

GMM N128 A/B/A
  .../vllm-moe-opt-validation-20260812/whole-aba-n128-slot2-20260812-194045
  ABA_RESULT SHA 90195d11b743a63e64be768ef011f9a53b9785943436a00376d63392547403ef

GMM N256 A/B/A
  .../whole-aba-n256-ud-slot1-20260812-202404
  ABA_RESULT SHA 9b75593d3bb5e0d0217e9a7d94510791579b1574464fdee14117a66018c81df1

pair-grid A/B/A
  .../whole-aba-combine-only-20260812-205654
  ABA_RESULT SHA cee57cf64604a6401bb61e4fc446357273cf62f08a89fa9c33d24e4660495530

source-grid A/B/A
  .../whole-aba-source-grid-20260812-212545
  ABA_RESULT SHA 37f23b7ddb01e2a820d13bd8766feff7fb8a278f60cd9c5d9813b80e1a7c3e64

adaptive generated audit
  .../pair-grid-adaptive-routed-grid-20260812-220659
  audit JSON SHA 76af3705b8b64f45ea03e12c9eb005b883a4b967e58eb3c1eb1c8078fde738c9

adaptive five-layer
  .../pair-grid-adaptive-routed-grid-five-layer-20260812-221141
  DFX summary SHA f0e0b23d488ef9e0649191f6d0782894a26dc4f87e79122b70d5f9aed81cc24c

adaptive whole A/B/A
  .../whole-aba-pair-adaptive-grid-20260812-221943/out/
    smallmesh-aba-pair-adaptive-grid-bs1-ctx64k-20260812-222139
  ABA_RESULT SHA 5528e1320766d7f7baae5f6a907975c40b71674164b7aac8b4b75b7fb6b9158e
  postflight SHA 1a6f2a1a1059f5bd8ea3cbd5dd29a65dacc098508eeb443bef68ec6afa531185

pure-AIV R4 static NO-GO
  .../pure-aiv-r4-static-nogo-20260812-224654/R4_STATIC_NO_GO.md
  SHA 9482cefd4b1b4767e2003246a7e087d9ca4667b776841cf9d09134369dc175b3

pure-AIV R8 compile NO-GO
  .../pure-aiv-r8-act-quant-20260812-224842/R8_COMPILE_NO_GO.md
  SHA fcf1d50c103c12c5a7d5c1694a10c949461c1a9737a619ee0a252737396adff9
```

## 13. 三方复核后的可比阶段合同与 `<200 us` 目标

本节是两名独立 Reviewer 和一名反向/对抗 Reviewer 的收口结论。三方一致认为：

- kernel 名称和 TP/EP 拓扑不必相同，但完整语义输入、输出和所有通信/等待必须
  落在同一计时闭包内；
- 当前 `97.01 us`、shared envelope 和二者 overlap 都不能拼成完整 MoE；
- 单次 five-layer DFX 可以用于定位阶段缺口，但不能作为 release authority；
- 必须先冻结完整 MoE endpoint，再继续以 `<200 us` 为硬目标优化。

### 13.1 完整 MoE 的统一 endpoint

| Endpoint | 语义 | vLLM | PyPTO |
|---|---|---|---|
| E0 | 当前 FFN 语义输入 ready，进入当前 FFN 数据流闭包 | FFN RMSNorm 首个硬件事件；若工作被 hoist，边界仍从 FFN 输入 ready 计 | 当前 FFN 输入 ready 后，`norm_quant_moe_input` / `gate_expert_fanout` 等首个真实 consumer |
| E1 | norm/input preparation ready | RMSNorm、input Cast 完成 | BF16/INT8/scale 等当前 FFN 输入 outputs ready |
| E2 | route decision ready | TopK、Index、mask、route weight metadata 完成 | `gate_topk` 及 ownership/weight metadata 完成 |
| E3 | expert input globally ready | InitRouting、Cumsum、local permutation 完成 | 所有真实 expert input data-visible；zero-route rank 也完成 ack |
| E4 | GMM1+activation+requant ready | fused GMM1-SwiGLU-quant 或 special 分解路径完成 | `expert_gate_up → expert_gate_up_act → routed_h_quant` 完成 |
| E5 | routed down ready | down GMM 与 routed postprocess 完成 | `expert_down` 全部 physical slices/zero-work ack 完成 |
| E6 | 原 token ownership 上最终 FFN delta ready | unpermute、shared、merge、TP AR 和 fence 完成 | EP return/combine、shared TP AR、stable reduce、merge 和 fence 完成 |
| E7 | residual 输出可被下一层消费 | residual Add 完成，且无阻塞复用的遗留工作 | 当前 `combine_reduce_residual` 完成；旧版本 `moe_residual_add` 完成；两者都要求无异步 DMA/credit/AR 尾巴 |

完整 MoE 的唯一正式 headline 是：

```text
T_moe(step, layer) = E7 - E0
```

E1–E6 用于解释 DAG，不允许通过逐段 p50 相加构造总时延。early dispatch 下
“首个 expert start”只能作为早启诊断点；它不是所有 rank 都成立的全局 cut。

如果某项融合把 norm/router hoist 到 attention，或把 residual sink 到下一层，
仍必须按数据流闭包把这部分工作计回当前 FFN。无法无扰切分时，只允许报告
layer-ready → next-layer-ready 或整网结果，不能用缩短后的 kernel 名称边界声称
`<200 us`。

### 13.2 跨 rank 与 42 层聚合规则

跨 rank 时钟已校准时：

```text
Local(step, layer, rank) = E7 - E0
Global(step, layer)      = max_rank(E7) - min_rank(E0)
X(step, layer)           = max(Global, max_rank(Local))
Aggregate(step)          = sum(layer=3..44, X(step, layer))
```

不能只计算 `max_rank(E7-E0)`：E0 和 E7 的关键 rank 可能不同。若 device clock
不能校准到同一时间轴，则必须使用 common causal epoch 和 all-rank completion
ack；否则 `Global` 不可用，结果保持 fail-closed。

聚合还必须满足：

1. 逐 `(step, layer)` 先形成 global critical span，再跨 step 求分位数；
2. 直接保存每步 42 层 aggregate，禁止 `42 × 单层 p50`；
3. 如存在跨层重叠，使用区间 union/critical path，禁止重复累计；
4. L3–L42 regular 与 L43/L44 special 分别报告，并共同进入总分布；
5. HCCL event 与描述同一通信的 AIV event 按 correlation 去重。

### 13.3 当前结果重新定性

0162 已补充一个只读、可复现的 local snapshot analyzer：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  vllm-pypto-stage-contract-review-20260812/
    recompute_local_moe_snapshot.py
    local_moe_snapshot.json
    REVIEW_SUMMARY.md
```

SHA：

```text
analyzer  8fb0b6941f357c25e503141690c2742e45cd677d0c1306774a1966971dcbbaf9
snapshot  b8b84a9b113c7e02491df36995f95926d7025b1487542bd8797e9eefbd200eb9
review    f16f5fd2a59b105b8553c463a2072d4d54a1cea7041d00d0e1739d3f41c6757d
```

同一 five-layer workload 的单次 L3/L4 × 8 rank worker-local 描述性结果：

| 版本 | local snapshot median |
|---|---:|
| pair-grid | `276.54 us` |
| pair + adaptive | `270.70 us` |
| 描述性变化 | `-5.84 us / -2.11%` |

该结果只有 16 个 rank-layer 截面，不是 16 次独立 decode sample，也没有全局
rank clock/remote-ready authority；因此只能用于估算缺口，不能称为“完整 MoE
正式 p50”。

按相同 local snapshot 估算，距离 `<200 us` 仍差：

```text
270.70 → 200 us
至少 -70.70 us / -26.12%
```

旧 vLLM Trace 的完整 MoE core 为 p50 `229.75 us`、p95 `250.50 us`；
42 层 aggregate p50 `9.689 ms`。`<200 us/层` 对应 42 层 `<8.4 ms`，目标本身
要求相对该旧 Trace aggregate 再快约 `13.3%`。由于旧 Trace 只有 NPU0 且缺少
同 workload authority，这些差值只能用于定方向，不能作为正式 PyPTO/vLLM
比值。

### 13.4 `<200 us` 的验收门

最终允许写“完整 MoE `<200 us`”必须同时满足：

| Gate | 要求 |
|---|---:|
| layer-call p50 | 单侧 95% CI 上界 `<200 us` |
| layer-call p95 | 单侧 95% CI 上界 `<200 us` |
| slow-rank | worst-rank local p95 `<200 us` |
| worst-layer | L3–L44 每层 p95 均 `<200 us` |
| 42 层 aggregate p50 | `<8.4 ms` |
| 42 层 aggregate p95 | `<8.4 ms` |

开发中间门可以暂设 p50 `<200 us`、p95 `<220 us`，但不能据此宣称目标完成。

authority campaign 至少需要：

- vLLM/PyPTO 使用相同 checkpoint logical tensor manifest、token IDs、真实 64K
  prefill KV、decode 序列、8 卡资源、频率和输出边界；
- 全 8 rank、全 42 MoE 层、每个 measured step 保存 E0–E7 原始数据；
- 至少 5 次独立重启、累计不少于 500 个 steady decode step；
- device-cycle 到同一 monotonic clock 的校准残差 p95 `≤0.5 us`、max `≤1 us`；
- Main-only、Main+MTP3、client token emission 三个 ITL 边界分别统计；
- randomized ABBA/BABA，直接比较 clean baseline 与最终方案，保存原始 samples
  并计算 bootstrap CI；不能再跨 campaign 累加百分比。

### 13.5 `<200 us` 的阶段预算和整改顺序

以下为 critical-path 增量预算，不是可重叠 kernel envelope 的简单求和：

| 关键阶段 | 当前诊断 median | 目标预算 | PyPTO 整改 |
|---|---:|---:|---|
| input/router join | `~49.8 us` | `≤40 us` | 减少 norm 重算、router task/GM fragmentation，研究 block-local partial TopK/merge |
| route organization | `~19.7 us` | `≤15 us` | TopK→count/slot/meta route-plan primitive，减少控制 task 与 GM |
| expert-ready EP path | `~26.4 us` | `≤20 us` | metadata/payload pipeline、coalesced put/notify、按 source 提前 gather |
| GMM1+activation+requant | `~77.3 us` | `≤52 us` | **最高优先级** compiler primitive；保留 BF16 round-trip、动态 valid row、expert×N 并行度 |
| down | `~26.7 us` | `≤22 us` | adaptive worker/scratch，消除 active-expert tail |
| EP return/restore | `~46.6 us` | `≤30 us` | streaming scatter、按 source/chunk credit、提前 reduce，避免全局等待 36 credits |
| global join | `~10.4 us` | `≤8 us` | 消除 wait→reduce gap；需要时引入保持 rounding 的 join primitive |
| residual | `~11.3 us` | `≤8 us` | 消除前置 gap；允许与 join 融合但 E7 仍取 next-layer consumer-visible ready |
| **总预算** | `~268 us` | **`195 us`** | 留 `5 us` 波动余量 |

shared lane 还需独立要求 `shared_ready ≤175 us`，建议 `≤170 us`；否则 routed
优化后 shared TP AR 会成为新的 E6 临界路径。

执行顺序调整为：

1. 先落地 E0–E7 全 rank instrumentation 和 authority harness；
2. GMM1+activation+requant compiler primitive；
3. EP return/restore，再优化 dispatch/expert-ready；
4. shared lane 及 down→TP AR 流水；
5. norm/router route-plan；
6. global join/residual；
7. 每项都用 direct clean-baseline ABBA/BABA 与 `<200 us` 全部门禁裁决。

在第 1 步完成前，当前状态只能写成：

```text
pair+adaptive 有描述性局部收益；
完整 MoE <200 us、42 层 aggregate 和同 workload ITL 均尚未证明。
```

## 14. 第二轮独立复核：统一阶段时延与 v7 裁决

本轮再次启用两名独立 Reviewer 和一名反向 Reviewer。权威验证日期仍为
**2026-08-12**；0162 部分目录中的 `20260813` 是机器未来时钟产生的原始标签，
不代表验证发生在 2026-08-13。

### 14.1 粗粒度阶段的可比时延

为避免 kernel 名称和相邻工作放置不同造成错配，本轮只比较以下完整语义闭包：

| 可比阶段 | 统一边界 |
|---|---|
| input + router | `E0→E2` |
| route organization / dispatch | `E2→E3` |
| GMM1 + activation + requant | `E3→E4` |
| routed down | `E4→E5` |
| return / restore / global merge | `E5→E6` |
| residual | `E6→E7` |
| 完整 MoE | `E0→E7` |

0162 新增的只读分析证据：

```text
/mnt/persist/chensiyu/workspace/perf-2026q3/
  vllm-pypto-stage-contract-review-20260812/second_round/
    analyze_vllm_semantic_stages.py
    vllm_semantic_stage_stats.json
    pypto_v7_semantic_stage_stats.json
    comparable_phase_table.json
```

SHA256：

```text
analyzer       8484ef86c5eb1a3d3ba22825f2aae88d999ec00cf1263e593b66825b00e98b1b
vLLM stats     5160b115d27c7cef6186d1333e7cdf5fa8ed7b5680cf5ac28fef794bd3ae03fe
PyPTO stats    744112e9368588d7065b3abc408ad6c5632b4462821aa197272dd7c34732c010
phase table    dcd8abc7bf621790ccdfe440629cf822e73a865ca797ca3d7f597fce7f5b3b55
```

描述性 p50：

| 语义阶段 | vLLM regular L3–L42 | PyPTO v7 N128/E1 | 差值 | PyPTO v7 N256/E3 | 差值 |
|---|---:|---:|---:|---:|---:|
| input + router `E0→E2` | `42.00 us` | `56.10 us` | `+14.10 us` | `57.14 us` | `+15.14 us` |
| route organization `E2→E3` | `42.50 us` | `46.38 us` | `+3.88 us` | `45.98 us` | `+3.48 us` |
| GMM1 + activation + requant `E3→E4` | `45.25 us` | `61.17 us` | `+15.92 us` | `79.83 us` | `+34.58 us` |
| down `E4→E5` | `34.50 us` | `26.43 us` | `-8.07 us` | `50.73 us` | `+16.23 us` |
| return / merge `E5→E6` | `57.50 us` | `68.99 us` | `+11.49 us` | `27.87 us` | `-29.63 us` |
| residual `E6→E7` | `4.50 us` | `11.24 us` | `+6.74 us` | `10.55 us` | `+6.05 us` |
| **完整 MoE `E0→E7`** | **`229.25 us`** | **`269.33 us`** | **`+40.08 us`** | **`272.10 us`** | **`+42.85 us`** |

阶段 p50 不能相加重建完整 MoE p50。以上也不是正式 parity：

- vLLM 是 NPU0、68 step、regular L3–L42 的 `2720` 个调用；
- PyPTO 是单次 five-layer DFX，N128/E1 只有 `10` 个、N256/E3 只有 `2`
  个 rank-layer 截面；
- 两边 workload、rank authority、TP/EP 拓扑和 route 分布仍未冻结为同一输入。

因此可以写“统一阶段定义下，当前描述性缺口约 `40–43 us/层`”，不能写
“PyPTO 已正式比 vLLM 慢 17–19%”。同理，距离 `<200 us` 的描述性缺口约为
`69–72 us`，仍不是 release 统计量。

N256 的结果还说明阶段必须整体看：E3 rank 的 producer/down 变长，但后续等待
缩短，完整 `E0→E7` 并未随某个子阶段同比变化。不能用单段 `E3→E4` 代替完整
MoE 裁决。

### 14.2 v7 `manual_dep` 的第二轮裁决

v7 相对 v6 的产品源码差异只是在互斥 producer 共写的 `h_bf16` 和
`h_partial_amax` 上设置 `manual_dep=True`。五层运行：

- L3/L4 对 golden byte-exact；
- `runner_rc=0`；
- `container.rc=1` 仍来自已知的 false-predicate `missing_on_swim` structural
  analyzer 限制，不能冒充完整 structural PASS。

关键 pre/post：

| 指标 | v6 | v7 | 结论 |
|---|---:|---:|---|
| class1 false retire − true producer end | `+5.53 us` | `-45.43 us` | false WAW marker 已提前 |
| class1 true producer → requant start | `8.99 us` | `8.54 us` | 只改善约 `0.45 us` |
| class1 `E3→E4` | `63.88 us` | `61.17 us` | 跨 run，不能归因 |
| class1 `E0→E7` | `271.51 us` | `269.33 us` | class0/class2 同步漂移，不能归因 |

预先设定的 `producer→requant ≤5 us` 和 `16/16 E0→E7 no-regression` 两项门禁
均失败。结论：

```text
manual_dep correctness experiment: dependency shape verified
manual_dep performance integration: NO-GO
```

它不应作为性能优化合入。若保留实验分支，最小证伪实验必须在同一容器执行
`A(v6)-B(v7)-B(v7)-A(v6)`，每 arm 至少 10 次，并以 class0 作为漂移负对照。

### 14.3 N128/N256 条件方案与下一步

两名正向 Reviewer 均认为 N128/E1、N256/E3 的 tile/fusion 形态本身没有被
明显低质量 codegen 误判：五层 byte-exact，N128/N256 physical slice 互斥，
N256 的 AIV/Right 资源虽紧但无 spill/fallback。反向 Reviewer 同意 N256/E3
相对全 N128 的 `E3→E4` 有约 `16 us` 方向性改善，但指出：

- 当前 class2 只有 2 个样本；
- E4 及以上 active-expert cardinality 尚未覆盖；
- class0 无 routed producer 时完整 MoE 仍约 `270.87 us`，证明只改 producer
  不可能把完整 MoE 压到 `<200 us`。

统一裁决：

```text
N128/E1 + N256/E3 conditional kernel: experimental integration GO
whole/release/performance claim: NO-GO，等待 42 层 route census 和 whole ABBA
```

Reviewer 对下一单变量的分歧也已 fail-closed 处理：

1. `dispatch_meta` 直接生成 compact nonzero `(expert, src)` pair plan 暂不实施。
   当前 E3 的 24 个 scatter physical slices 均为真实工作，主要 gap 是调度排队；
   新 ABI/control plan 尚无证据能减少 grid，不能重复 source-grid 的失败。
2. E3 专用 `combine_scatter cap=16` 也暂不启用。当前 24 个 write-disjoint
   pair 正好形成一波真实工作；cap16 会让 8 个 block 串行处理第二个 pair，
   但不会删除 orchestration TaskId。除非隔离 micro-A/B 证明 launch contention
   大于新增的 per-block 尾部，否则默认为 NO-GO。
3. `combine_reduce + residual` 已完成：删除 `moe_out` GM 中间结果和独立
   residual task，保留 FP32→BF16→FP32 双舍入，并覆盖
   regular/SWA/L43/L44。4096 版本 GO；2048/1024/Cube 方案 NO-GO，详见
   第 15 节。
4. 大杠杆仍是 compiler primitive 级
   `GMM1 + activation + requant`，以及有 causal DFX 证据后再推进的 streaming
   EP return。两者均不得用 kernel 数替代 `E0→E7` 和 whole-net ABBA。

截至第 15 节收口时，仍没有同 session whole-net ABBA/BABA，也没有同
workload ITL。因此当前不能新增任何整网或 ITL 收益数字。

## 15. 第二项落地：4096-wide combine-reduce/residual fusion

本节权威验证日期为 **2026-08-12**。0162 原始路径中的 `20260813` 是机器
未来时钟产生的标签，不表示验证日期变成 2026-08-13。本轮只看前五层，重点
L3/L4。

### 15.1 阶段映射和最终实现

| vLLM | PyPTO 前 | PyPTO 后 |
|---|---|---|
| `MoeTokenUnpermute` 把 routed 输出恢复到 token ownership。 | EP `combine_scatter → combine_wait` 返回原 rank，随后独立 `combine_reduce`。 | EP 返回协议不变；不把 `combine_wait` 融入计算 kernel。 |
| routed/shared merge 后形成最终 FFN delta。 | `combine_reduce` 按 TOPK 顺序做 FP32 累加，写 `moe_out` BF16 GM。 | `combine_reduce_residual` 在同一 4096-wide AIV block 中完成 TOPK reduce，并显式执行 FP32→BF16(rint) 边界。 |
| residual Add 产生下一层可见 hidden。 | 独立 `moe_residual_add` 再从 `moe_out` 读回，做 BF16→FP32、加 residual、写 BF16。 | 同一 task 继续 BF16→FP32、加 `resid_hold`、写 `next_hidden_out`；删除 `moe_out` 和独立 residual task。 |
| regular 与 L43/L44 special expert kernel 可以不同，但输出阶段语义相同。 | regular/SWA/L43/L44 四条路径都调用独立 reduce 和 residual。 | 四条路径均调用同一融合 writer；expert special kernel 本身不变。 |

数值顺序固定为：

```text
shared BF16 → FP32
→ 依 route 0..7 原序累加 routed BF16
→ BF16(rint)
→ FP32 + residual FP32
→ BF16 output
```

最终选择 4096 tile、16 blocks，每个 block 独占一个 token 的完整
`[1, 4096]` row。没有矩阵运算，因此不写独立 Cube 程序。

### 15.2 多 Agent 语义、SPMD、融合粒度和 Cube 复核

两名正向 Reviewer 和一名反向性能 Reviewer 的最终结论一致：

- **4096 fusion GO**：四路径、TOPK 顺序、双 BF16 边界、TensorMap 和 DAG
  均无 blocker；
- **2048 semantics GO / performance NO-GO**：32-owner 覆盖和写区间互斥
  正确，但性能稳定回退；
- **1024 NO-GO**：64 blocks 会进一步增加启动 cohort、barrier 和调度压力，
  无需继续上卡；
- **Cube NO-GO**：该阶段只有 cast、向量累加、residual add 和 GM 搬运，
  Cube 会新增重排和启动开销；
- 4096 的 Vec 占用约 `48 KiB / 184 KiB`，2048 为 `24 KiB`，两者都无
  spill；UB 减半没有形成 occupancy 收益；
- 2048 的 32 blocks 实际出现 `16+16` 两个启动 cohort，而 4096 的 16 blocks
  为单 cohort。每 block 的 8 次 route load/cast/add 和约 21 次 barrier 没有
  随 tile 减半，因此 block setup、barrier、GM/MTE 竞争成为回退主因；
- `/2`、`%2` 每 block 只执行一次，不是 2048 回退的主因；
- 不应继续把 `combine_wait` 融进计算 kernel，否则会复制通信等待或破坏单点
  credit 聚合。

### 15.3 集成态和验证

0162 integration tree：

```text
/mnt/persist/chensiyu/workspace/develop-worktrees/vllm-moe-opt-20260812
HEAD    69ad31e4fd6e40b30e43c2566ce8f8ebd0b2427d
decode  de5712383775823eef82ce6b082d04ac54cf232f93452a1d9090550ac797d936
test    d394ea4ce5aeba23a07e8f0f0a1f5be0c22d3b3d56c38412b2a00e1047fa2038
```

验证结果：

- `git diff --check` PASS；
- Ruff PASS；
- host 静态合同：`31 passed, 5 deselected`；
- 固定镜像直接数学合同：`5/5 PASS`，新增覆盖
  `active_tokens ∈ {0,1,15,16}` 的 active/inactive row；
- byte-identical `decode_fwd` 的 fixed-image whole compile PASS；
- PTOAS 四个融合 kernel 均生成，唯一 GM store，无 spill；
- 五层 L3/L4 八卡输出 byte-exact，`max_abs=0`、`bad_ratio=0`；
- DFX runner 完成；`container.rc=1` 只来自已知的
  `missing_on_swim` structural analyzer 限制，不能写成完整 structural PASS。

### 15.4 4096/2048/4096 切分 A/B/A

Rank0：

| 层/口径 | 4096 A1 | 2048 B | 4096 A2 | 裁决 |
|---|---:|---:|---:|---|
| L3 scheduler task envelope | `8.42 us` | `11.08 us` | `7.74 us` | 2048 回退 |
| L4 scheduler task envelope | `8.00 us` | `10.14 us` | `8.00 us` | 2048 回退 |
| L3 physical core envelope | `5.22 us` | `8.80 us` | `5.44 us` | 2048 回退 |
| L4 physical core envelope | `4.32 us` | `6.30 us` | `4.82 us` | 2048 回退 |

all-rank median：

| 层/口径 | 4096 A1 | 2048 B | 4096 A2 |
|---|---:|---:|---:|
| L3 final physical envelope | `5.28 us` | `8.59 us` | `5.58 us` |
| L4 final physical envelope | `4.27 us` | `6.99 us` | `4.61 us` |
| L3 `combine_wait→final` | `10.47 us` | `13.78 us` | `10.66 us` |
| L4 `combine_wait→final` | `9.60 us` | `12.06 us` | `9.84 us` |

2048 对两侧 4096 均稳定变差，最终保留 4096。

### 15.5 旧/融合/旧收益和当前 MoE

all-rank median physical wall：

| 层/口径 | 旧 A1 | 融合4096 | 旧 A2 | 相对旧 midpoint |
|---|---:|---:|---:|---:|
| L3 final envelope | `15.16 us` | `5.58 us` | `15.04 us` | `-9.52 us / -63.05%` |
| L4 final envelope | `15.56 us` | `4.61 us` | `15.40 us` | `-10.87 us / -70.22%` |
| L3 `combine_wait→final` | `20.32 us` | `10.66 us` | `20.28 us` | `-9.64 us / -47.49%` |
| L4 `combine_wait→final` | `20.70 us` | `9.84 us` | `20.47 us` | `-10.75 us / -52.20%` |
| L3 physical `E0→E7` | `266.28 us` | `252.10 us` | `267.65 us` | `-14.87 us / -5.57%` |
| L4 physical `E0→E7` | `265.59 us` | `256.95 us` | `266.36 us` | `-9.03 us / -3.39%` |

用与前文 local snapshot 相同的 merged-swimlane endpoint 重新计算：

| 层 | worker-local median | scheduler-local median | captured max |
|---|---:|---:|---:|
| L3 | `252.36 us` | `254.83 us` | scheduler `259.30 us` |
| L4 | `257.29 us` | `260.52 us` | scheduler `261.70 us` |

scheduler-local 旧 A midpoint 到融合版本的描述性变化为：

```text
L3  268.22 → 254.83 us   -13.39 us / -4.99%
L4  267.10 → 260.52 us    -6.58 us / -2.46%
```

其中 final tail 的 `~9.6–10.7 us` 回收有旧/融合/旧直接证据；完整
`E0→E7` 还包含上游单次 capture 漂移，因此只作描述性结果。

相对 `<200 us` 目标，当前 scheduler-local median 仍差：

```text
L3  54.83 us，仍需约 21.5% 降幅
L4  60.52 us，仍需约 23.2% 降幅
```

所以本轮**没有达到 `<200 us`**。旧 vLLM 描述性 regular MoE 为
`229.25 us`；当前 PyPTO L3/L4 仍分别高约 `25.6/31.3 us`，但两边不是同
workload authority，不能写成正式性能比。

五层 DFX harness p50 为 `13.226 ms`；旧 A1/A2 为
`13.458/15.214 ms`。该结果只有 3 次计时、A1/A2 漂移大且仅覆盖五层，因此
不能当作整网收益或 ITL。**同 workload whole-network ITL 仍为未知。**

### 15.6 后续只做有门槛的优化

1. 冻结当前 4096 fusion；不合入 2048，不继续 1024，不写 Cube kernel。
2. 如果继续压 final tail，只允许单变量 prototype 4096 AIV ping-pong：
   MTE2 预取下一 route，与当前 route cast/add overlap，TOPK 顺序和双 BF16
   边界不变。
3. 该 prototype 的进入门槛：
   - L3 physical envelope `≤4.0 us`；
   - L4 physical envelope `≤3.5 us`；
   - L3/L4 `wait→final` 各改善 `≥1 us`；
   - `8/8` rank 不回退且 byte-exact。
4. 即使上述 prototype 成功也只能再贡献约 `1–2 us`。达到 `<200 us` 的主要
   工作仍在更大的 `E0→E1`、`E3→E4` 和 `E5→E6` 阶段。
5. 本轮按用户要求止于前五层 L3/L4；在另行授权 whole-network ABBA/BABA
   前，整网 ITL 保持未知。

### 15.7 本轮权威证据

```text
4096 A1
  .../combine-reduce-residual-fusion-five-layer-20260812-20260813-044246
2048 B
  .../combine-reduce-residual-tile2048-five-layer-20260812-20260813-045226
4096 A2
  .../combine-reduce-residual-fusion4096-repeat-five-layer-20260812-20260813-045432

旧 A1
  .../conditional-n128-n256-v6-eq3-five-layer-dfx-20260812
旧 A2
  .../conditional-n128-n256-v6-eq3-repeat-five-layer-20260812-20260813-045512

physical-wall analyzer
  .../analyze_combine_residual_aba.py
  SHA bc4c2b8ada2bda404823b9898627f8a0eb1c5378b377731777e28bb1593d574c
old/fused/old JSON
  SHA 3f47d7dbcd1c75e3548a16d5f639c447cd906e1a2840d2a147c599b89e6d3451
tile A/B/A JSON
  SHA 308974c79e298b9936bf4c42419f190b41c600666827ddd1d1939919c40d759d

merged endpoint analyzer
  .../analyze_local_moe_endpoint_aba.py
  SHA f9a243ebe2f654f0454f4581fd92337997c13901b94ddf91c20724ad8d39fecc
old/fused/old endpoint JSON
  SHA 7c5e23c187ffb5cf6c9dc3094a718aa8c26e3b405512b260f4c0e80787b94b28

integration static/fixed-image math
  .../combine-reduce-residual-fusion-integration-final-20260812
```
