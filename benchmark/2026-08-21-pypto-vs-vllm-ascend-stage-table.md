# PyPTO vs vLLM-Ascend 逐阶段时间对比表（2026-08-24 r9 overlay；保留 2026-08-21 历史对照）

> **历史基线一句话（2026-08-21 R5）**：PyPTO 单层 wall 与 vLLM-Ascend 同类层同量级（full-MoE `458.82` vs
> `408.50 us`，**1.12×**）。差距**不在 expert 计算**（`routed_gmm1` `41.72` vs `42.52 us`
> 已打平），而集中在 **EP combine 跨卡通信**（独占 `55.95 us`，vLLM 无对应物）与
> **每 task 边界约 4 us 的固定调度开销**（MoE 层 32 task ⇒ `~125 us`，占整层 27%）。
>
> ⚠ **本版对 v1 表格的分类做了 7 处修正 + 1 处口径修正，见 §2。v1 的三个结论已被推翻**：
> ① SWA 层 gap 不是 47.8% 而是 **32.5%**；② SWA 层 pre-attn/attn 独占**不是 0**；
> ③ dense_mlp 不是慢 1.90× 而是 **1.57×**。引用 v1 数字前请先读 §2。
>
> ⚠ **硬边界**：两侧**不是同一 workload** —— PyPTO 是 BS1 / ctx 64K 的 5 层 synthetic
> harness，vLLM 是 BS4 / **ctx 不可知**的真实 service trace。**attention 与 AllReduce 两行
> 的绝对比值不成立**，只有结构占比可读。详见 §8。

---

## 0. 2026-08-24 current r9 overlay（**不是**旧表的同口径替换）

本节把当前 immutable r9 的前五层 observed critical path 叠加到本表，供
MoE 阶段定位和 decoder 逻辑阶段对齐使用；`§1–§9` 仍保留 2026-08-21 的
**R5 PyPTO / vLLM-Ascend trace** 历史 kernel/major-stage 审计。新主表按
E0–E7 语义 endpoint 对齐，不要求 kernel 名称一致；不要把两种统计口径混算。

### 0.1 Current r9 provenance and run contract

| 项 | current r9 |
|---|---|
| 镜像 | `hub.i.basemind.com/stepcast/vllm-pypto@sha256:b637f00c66d4dc976c053c617d2e19e6d6d66f68f4bef30250984da7a71690f6` |
| Config | `sha256:f6c8f72eecad0a9d40d0c4ea55afaab09dd4e2f5fe54d6a091e332465e421dae` |
| 主机 / 输入 | `gpu-a910x-0162`；`FiveLayerMoe_20260824_071710`，8 ranks，`chip_swimlane_records.json` |
| workload | BS1，ctx `65536`，active batch `1`，`num_blocks=512`，TP8/EP8 |
| H4 合同 | `PYPTO_H4_RESIDENT=all`（与发布性能合同一致；镜像 Config 未 bake） |
| LOW-WAIT 参考 | `rank2/d0`：makespan `1.867 ms`；static CPM / observed compute `1.544 ms`；data-wait `0.323 ms`；80 observed-path tasks |
| DFX 报告 | `.../outer-swimlane-r9-h4-20260824-151416/runtime/dfx_analysis/moe_critical_path_report.md`（sha256 `22e76217…f21f18af`） |
| rank2 报告 | `.../runtime/build_output/FiveLayerMoe_20260824_071710/dfx_outputs/rank2/d0/critical_path_report.md`（sha256 `f856c196…4c31f9ac`） |
| merged swimlane | `.../runtime/build_output/FiveLayerMoe_20260824_071710/dfx_outputs/rank2/d0/merged_swimlane_20260824_071753.json`（sha256 `dcce013d…190f9794`） |

发布与 H4 双合同的完整 provenance 见
[`2026-08-24-upgrade-r9-release.md`](2026-08-24-upgrade-r9-release.md)。

### 0.2 Decoder 逻辑阶段对齐（E0–E7；本节为主比较）

本节按 **decoder dataflow endpoint** 对齐，不要求两侧 kernel 名称或 kernel
数量一致。统一边界沿用 MoE 阶段合同：

| 逻辑阶段 | 语义边界 | r9 对应工作 | vLLM 对应工作 |
|---|---|---|---|
| `E0→E2` input + router | FFN 输入 ready → route decision ready | `norm_quant_moe_input` + `gate_topk`（`gate_expert_fanout` 作为 TopK endpoint 的前置依赖纳入，不另拆阶段） | RMSNorm/Cast + TopK/Index/route metadata |
| `E2→E3` route organization / dispatch | route decision → expert input globally ready | ownership metadata、EP push/wait/gather | local permutation + Cumsum/route organization |
| `E3→E4` GMM1 + activation + requant | expert input ready → routed activation ready | `expert_gate_up` + activation + `routed_h_quant` | `GroupedMatmulSwigluQuant` 或等价 special path |
| `E4→E5` routed down | routed activation ready → routed down ready | `expert_down` physical slices / zero-work ack | down projection + routed postprocess |
| `E5→E6` return / restore / global merge | routed down ready → token-ownership FFN delta ready | 完整 endpoint 应含 EP combine、shared branch、stable reduce、TP AR/fence；当前观测只捕获 routed combine 子路径，未形成统一 E6 endpoint | unpermute、shared/global merge、TP AR/fence |
| `E6→E7` residual | FFN delta ready → 下一层可消费 | residual writer 完成且无尾部工作 | residual Add 完成 |
| `E0→E7` complete MoE | 完整 FFN 语义闭包 | 当前报告未保存全局 E0/E7 endpoint（这是观测缺口，不是运行失败） | 完整 semantic-stage reference |

下面的数值是 **phase-aligned descriptive comparison**。主比较列使用
`rank2/d0` 每个逻辑阶段全部 physical tasks 的
`min(start)→max(end)` **semantic envelope**；另列 observed critical-path
贡献作诊断。vLLM 列为历史 E0–E7 semantic-stage p50（regular L3–L42，阶段合同与数值见
[`2026-08-12-vllm-ascend-decode-moe-trace-gap.md`](2026-08-12-vllm-ascend-decode-moe-trace-gap.md)
§14.1；对应 `vllm_semantic_stage_stats.json` / `comparable_phase_table.json`
证据）。这些是 semantic-endpoint 统计，不是旧 §4 的 kernel-major sum。
因此阶段语义可对齐，但仍不能把它们当成同 workload 的 release ratio。

| 逻辑阶段 | r9 semantic envelope（L3 / L4） | r9 observed-CP 诊断贡献（L3 / L4） | 历史 vLLM semantic p50 | 描述性差值（非同 workload） | 读法 |
|---|---:|---:|---:|---:|---|
| `E0→E2` input + router | `43.54 / 44.72 us` | `44.3 / 44.5 us` | `42.00 us` | `+1.54 / +2.72 us` | 逻辑输入/路由阶段接近；CP 列含 data-wait |
| `E2→E3` route organization / dispatch | `49.40 / 48.66 us` | `52.0 / 50.5 us` | `42.50 us` | `+6.90 / +6.16 us` | EP ownership transfer 相对 local route plan 多出跨卡等待 |
| `E3→E4` GMM1 + activation + requant | `85.42 / 79.34 us` | `87.8 / 82.0 us` | `45.25 us` | `+40.17 / +34.09 us` | **这是逻辑阶段差异，不是 kernel-name 差异**；r9 三个 task 聚合后与 vLLM 一个语义阶段对齐 |
| `E4→E5` routed down | `35.48 / 25.14 us` | `40.1 / 26.9 us` | `34.50 us` | `+0.98 / −9.36 us` | L3 接近、L4 方向偏快；仍受 BS/context 与 route 分布影响 |
| `E5→E6` return / restore / merge | `n/a / n/a` | `59.30* / 52.66* us` | `57.50 us` | n/a | 完整 r9 endpoint 的观测缺失；`*` 仅为 routed-combine partial diagnostic，不与 vLLM 完整阶段直比 |
| `E6→E7` residual | `15.72 / 14.64 us` | `10.0 / 11.3 us` | `4.50 us` | `+11.22 / +10.14 us` | residual/output-ready 方向上 r9 仍有余量 |
| **`E0→E7` complete MoE** | **n/a / n/a** | **n/a** | **`229.25 us`** | **n/a** | 当前 r9 报告没有形成包含 shared + TP AR + global fence 的统一 E0/E7 endpoint |

并行 shared branch 的 envelope 为 L3 `48.4 us`、L4 `47.6 us`；它属于
`E5→E6` 的并行输入，**不能再加到上表阶段值或构造完整 E0→E7**。
当前 r9 全 rank 的 combine tail 最大为 `145.2 us`（L3）/
`148.5 us`（L4），这是 remote-producer completion tail，不是单独的
restore 算术时间。

> semantic envelope 的阶段之间存在 overlap，不能逐行相加重建 E0→E7。
> r9 的 observed-CP 贡献 `266.0 us (L3) / 258.8 us (L4)` 仍保留为诊断值；
> 它们不是完整层 wall，也不能替代上表的 endpoint envelope。
> 作为附加诊断，rank2 routed-subgraph 的 `norm_quant→moe_residual_add`
> local envelope 为 `263.84 us (L3) / 258.26 us (L4)`；这两个数不包含
> 完整 E5→E6 的 shared/TP-AR/global completion，不能与 vLLM `E0→E7`
> 直接作差。

> **术语澄清**：r9 不是“完全没有 fusion”，而是采用
> `staged_fused_gate_up`：AIC 侧保留 gate/up 阶段融合，AIV 侧的
> `expert_gate_up_act` 与 `routed_h_quant` 仍分开。由于本节按
> `E3→E4` 逻辑阶段对齐，这三个 task 仍应合并与 vLLM 的
> `GMM1 + activation + requant` 阶段比较。镜像没有采用 vLLM 式单 kernel；
> 完整 fusion 候选在 R5/R8 门禁中分别因 whole-net A/B/A 回退、32B Vec row
> 合同及 fixed-image compile/descriptor 问题被 NO-GO，未进入 r9。

### 0.3 当前 r9 与 vLLM 的阶段差异

1. **当前观测中最大的候选差异是 `E3→E4` 逻辑阶段，而不是某个同名 kernel。**
   r9 L3/L4 的阶段贡献为 `87.8/82.0 us`，历史 vLLM semantic p50
   为 `45.25 us`；这是按 decoder 逻辑闭包聚合后的描述性差异，是否为当前
   第一优先级仍需同 workload paired measurement。
2. **`E2→E3` 的差异主要来自 ownership topology。**
   vLLM 是 local route organization，PyPTO 是 EP8 dispatch；因此应优化
   dispatch 的 route/overlap/remote completion，而不是机械寻找同名 vLLM kernel。
3. **`E5→E6` 当前只能报告 routed combine 子路径。**
   `combine_scatter → combine_wait → combine_reduce` 已在 observed path，
   但 shared branch、TP AR 和 global fence 没有在当前 rank2 报告中形成统一
   E6 endpoint；完整阶段应记 `n/a`，不能从 partial sum 推导全阶段比值。
4. **R5 的 `~4 us/task` gap 仍只属于历史 stage-exclusive capture。**
   当前 r9 LOW-WAIT rank2 的 stall `0.323 ms` 全部是 data-wait，没有
   `core-wait` 或 `front-gap`，不能把历史 gap 直接搬到 r9。
5. **H4 的 whole-net 收益与阶段表分开。**
   `PYPTO_H4_RESIDENT=all` 的 64K/1000 p50 为 `22.253 ms`，默认 `none`
   为 `27.812 ms`；它不是 E0–E7 stage measurement。

### 0.4 本轮可以下的结论 / 不能下的结论

| 结论 | 状态 |
|---|---|
| kernel 不同但 decoder 逻辑阶段可按 E0–E7 对齐 | ✅ 本节已统一边界 |
| r9 的 `E3→E4` 应与 vLLM 的 GMM1+activation+requant 作为同一逻辑阶段比较 | ✅ 可做方向性阶段比较 |
| r9 当前完整 `E0→E7` 已经比 vLLM 快/慢多少 | ❌ 当前报告没有全局 E0/E7 endpoint，且 workload/statistics 不同 |
| EP combine 是 `E5→E6` 的主要结构性差异 | ✅ 方向性支持；完整 E6 仍需全 rank endpoint |
| 旧表 `routed_gmm1` kernel parity 是否等于 r9 phase parity | ❌ 两者不是同一层次；以本节 E3→E4 为准 |

---

## 1. Provenance

> **历史附录边界**：从本节开始的 `§1–§9` 保留 2026-08-21 R5
> stage-exclusive / major-stage sum / kernel audit 原表。它们与上方
> `§0.2` 的 current-r9 E0–E7 descriptive table **不可混用**；旧表中的
> `1.05×/1.12×` 仍是历史对照，不是 current r9 release ratio。

| 项 | 值 |
|---|---|
| 执行主机 | `gpu-a910x-0162`（全部解析就地执行，未占卡、未重采） |
| 工作目录 | `/mnt/persist/chensiyu/workspace/perf-2026q3/vllm-pypto-stage-table-20260821/` |
| **PyPTO 输入** | 2026-08-21 R5 **历史 source-overlay** packed-NZ 五层 DFX（升级前 MoE 基线）：`.../moe-routed-packed-fusion-20260815/dfx-packed-nz-architecture-20260817-213730/out/runtime/build_output/FiveLayerMoe_20260817_134105/dfx_outputs/rank{0..7}/d0/merged_swimlane_20260817_1342*.json` |
| PyPTO 配置 | `PYPTO_STEP3P5_MAX_SEQ=65536` `ROPE_SEQ=65536` `STORAGE_BATCH_CAPACITY=16`，TP8/EP8，**BS1**，8 rank 各 1 次 invocation |
| rank2 swimlane sha256 | `f9ef1dbe51d9867d9f981b6bf6da9b5b1d5446ca08fbdfcfccaa8c513efdf013` |
| **vLLM 输入** | `/mnt/persist/chensiyu/workspace/develop/trace_view (3).json`，sha256 `ddba08673aa3787147c261403223995b2b345219911317c9f50c05215d65abf2` |
| vLLM 规模 | 68 个 decode step × 45 层 Main（Model 87），TP8，**BS4**（由 `hcom count=16384 = 4×4096` 推出） |
| **分析脚本（v2 分类）** | `pypto/pypto_v3.py` `569b1ef7…`（分类规则）、`pypto/pypto_v4.py` `26b4b47b…`（major/sub 双层口径 + 全部表格数字）、`pypto/gapcheck.py`（仅供 §4.4 的 task 计数）、`vllm/vllm_v3.py` `7c2aa732…` |
| 机读产物 | `pypto/pypto_v4_raw.json` `f97b89aa…`、`vllm/vllm_rows3.pkl` `5ce05629…` |
| 审计脚本（本次分类 review 用） | `pypto/audit2.py`（逐 task 时序 dump）、`vllm/audit_vllm.py`（逐 kernel taskid dump） |
| v1 遗留脚本（**已废弃，勿引用**） | `pypto/pypto_v2.py`、`pypto/excl.py`、`vllm/vllm_stages2.py`、`vllm/agg3.py` |

### 层身份判定（两侧都独立核实过，不靠名字猜）

- **PyPTO**：层归属**只能靠 `rNtM` task-id 区间**，名字前缀不可靠（L1/L2 的 attention
  都叫 `swa_*` 无前缀，L4 的 MoE 也无前缀）。区间：ring2 `t0-21`=L0、`t22-39`=L1、
  `t40-58`=L2、`t59-94`=L3、`t95-127`=L4；ring3 每 3 个 task 一层（out_proj + AR）。
  五层 = L0 full+dense、L1/L2 SWA+dense、L3 SWA+MoE、L4 full+MoE。**全部 task 归桶，0 未分类。**
- **vLLM**：trace 内无 literal `decode` marker。用 replay 内**稳定的 `Task Id`**（每 replay
  重置，68 个 replay 每个恰好 1430 事件）+ 每层 2 个 `GemmaRmsNorm` 作层边界，
  阶段按 anchor 之间的 taskid 区间归桶（不靠 kernel 名，因为 `MatMulV2_…98513`
  同时被 QKV、head-gate、dense-MLP 复用）。
- **attention 变体自动分离出 12 full + 33 SWA**，与 step3p5 层表
  （`full_dense×1 + full_moe×10 + full_moe_swiglu7_swiglu16×1 = 12` full，
  `swa_dense×2 + swa_moe×30 + swa_moe_swiglu7_silu×1 = 33` SWA）**完全吻合** ——
  这是层识别正确性的独立佐证。

### 交叉验证（全部通过）

| 量 | 本次算出 | 既有报告 | 判定 |
|---|---:|---:|---|
| vLLM `GroupedMatmulSwigluQuant` p50 | `43.98`（regular 全体） | `43.98` | ✅ 逐位一致 |
| vLLM `GroupedMatmul`(down) p50 | `28.80` | `28.80` | ✅ 逐位一致 |
| vLLM MoE-输出 AR p50 / p95 | `23.08` / `50.84` | `22.281` / `50.682` | ✅ 一致（既有把 40+2 层混算） |
| PyPTO packed GMM1 span | `41.72`(L4) / `40.90`(L3) | `41.91` | ✅ |
| PyPTO packed down span | `13.36`(L4) / `14.53`(L3) | `14.30` | ✅ |
| PyPTO shared 分支 span | `59.04` | `58.75` | ✅ |
| vLLM 45 层 wall 之和 | `16.723 ms/step` | Main span `18.067 ms` | ✅ 差 `1.34 ms` = 层间 gap + embedding/final-norm/LM-head |

---

## 2. ★ 分类审计（v1 → v2 的 7 处分类修正 + 1 处口径修正）

用户给的 8 阶段（pre-attn / attn / post-attn / all-reduce / pre-moe / moe / dispatch /
combine）是**参考**。把它当规范直接套，会踩下面 8 个坑。修正依据是**逐 task 时序 dump**
（`audit2.py` / `audit_vllm.py`），不是猜。

### 2.0 先确立一条原则：大项按 **dataflow 角色** 分，不按 **执行时刻** 分

PyPTO 是 DAG，**执行时刻 ≠ 数据流位置**。例：L4 的 `head_gate` 实际在 `16.0→36.1 us`
（attention 之前！）跑完，但它在数据流上属于 attention 输出的门控路径。

- 若按**执行时刻**分桶 ⇒ PyPTO 的 head_gate 进 pre-attn、vLLM 的进 post-attn ⇒ **两侧不可比**。
- 若按**dataflow 角色**分桶 ⇒ 两侧都在 post-attn ⇒ 可比，且 "PyPTO 独占 = 0" 恰好
  量化出 "PyPTO 把它调度到前面藏起来了" 这件事。

**所以：大项固定按数据流（§4 的表），执行时刻单独用时间轴呈现（§5）。** PyPTO 的全部
优势正来自这两者的错位，把它们混在一个维度里就看不见了。

### 2.1 七处分类错误（v1 有，v2 已修）

| # | v1 的错 | 证据 | v2 改成 | 影响 |
|---|---|---|---|---|
| 1 | `attn_residual_hold` 和 `resid_snapshot` 都塞进 `post_attn/residual_hold` | L4 `attn_residual_hold` 在 `t=0.28`（层首）；L3 `resid_snapshot` 在 `t=130.84`（AR 之后）。**是两个不同的东西** | 前者 → `pre_attn/residual_snapshot`（快照层输入）；后者 → `pre_ffn/residual_snapshot`（快照 FFN 残差） | ★★★ 这是 v1 最大的错。L3 的 post_attn union 被撑成 `0.02→136.54`，把中间所有阶段都"遮住" ⇒ **v1 报的 "SWA 层 pre-attn / attn 独占 = 0.00" 是假的**，真值 `27.24` / `5.58`；**gap 也从假的 47.8% 回到 32.5%** |
| 2 | 残差 add 归到 `8_combine` | dense 层根本没有 combine，但 `dense_residual_add_tp` 却被算进 "combine" | 新增 `9_epilogue/merge_residual`，收 PyPTO `moe_residual_add`/`dense_residual_add_tp` 与 vLLM 尾部 `Add` | combine 大项从 `72.33` → `57.44`（L4），变成纯跨卡 EP + unpermute；dense 层不再出现空的 combine 桶 |
| 3 | `7_ffn` 把 shared 与 routed 混成一桶 | shared 独占 `8.10`、routed `48.21`，ROI 差 6× 却看不出来 | 拆 `7a_shared` / `7b_routed`（dense 层保留 `7_ffn/dense_mlp`） | 直接暴露 "shared 免费、routed_gmm1 才是钱" |
| 4 | dispatch 子项两侧**功能错位** | vLLM `MoeInitRoutingCustom`(10.80) 是本卡 permute、`Cumsum`(31.32) 是算 offset；PyPTO 的 permute 是 `dispatch_gather_spmd`，却被归进 `ep_dispatch_comm` | 按功能对齐三子项：`route_offsets`（vLLM Cumsum ↔ PyPTO count_publish+meta）、`permute`（vLLM MoeInitRouting ↔ PyPTO gather）、`ep_transfer`（PyPTO push/wait，vLLM n/a） | dispatch 内部终于可比：PyPTO permute `7.02` vs vLLM `11.02`，PyPTO offset `28.05` vs vLLM Cumsum `31.60`（且 vLLM 那个跑在 **AI_CPU** 上） |
| 5 | vLLM `qkv_proj` 后的 `Cast`/`Add`×2 被算进 `input_norm` | taskid 173-176 夹在 QKV matmul(172) 与 rope(177) 之间，是 QKV 的 epilogue | 新增 `pre_attn/qkv_epilogue` | vLLM `input_norm` 从虚高的 `10.36` 回到真值 **`4.02`** ⇒ PyPTO input_norm `11.22` 其实**慢 2.8×**，v1 报的 "两侧打平" 是错的 |
| 6 | vLLM attention 前的 `NOP`/`EVENT_WAIT`（taskid 179-188）漏进 `5_pre_ffn/ffn_norm` | 分类链缺一条分支，落到 MoE 兜底 | 新增 `_sync/pre_attn_sync` 桶并从大项合计中排除 | 只有 `0.18 us`，数值无关，但是逻辑洞 |
| 7 | `attn_out_zero` 丢进匿名 `0_hoisted` 看不见 | 它被调度器提前 174-820 us 吊起，但语义上是 attention 的 buffer 初始化 | 改标 `2_attn/attn_out_zero_HOISTED`，仍排除出层窗口但可见 | busy 仅 `~1 core-us`，纯可读性 |

### 2.2 一处口径错误：独占值在细分桶下不可加

**v1 的 bug**：`独占` 是 "此区间内无其他阶段在跑" ⇒ 当两个子项**互相**重叠时，
合成一桶算独占会把它们的重叠算成"独占于本大项"，拆成两桶则互相抵消。
⇒ **`独占` 之和随分桶粒度变化，gap 也跟着变。** v1 报的 gap 与 v2 更细的分桶不可直接比。

**v2 的修法（两层口径）**：

- **大项独占** = union(该大项全部子项) 对 union(其他所有大项) 求独占。**粒度稳定、可加**，
  `Σ大项独占 + gap = layer wall` 恒等。← §4 用这个，也只有这个能对 vLLM 的 kernel sum。
- **子项独占** = 该子项对 **其余一切** 求独占 = "这个子项消失能省多少 ITL" = **ROI 天花板**。
  **不可加**（重叠子项会互相抵消）。← §6 用这个，只读单行，不读列和。

> **gap 的定义（口径要钉死）**：`gap = median_8rank(wall) − Σ_大项 median_8rank(独占)`，
> 即**先对每个大项跨 rank 取 median 再相减**。若反过来（每 rank 先相减再取 median）
> 会得到略不同的值（L1 `69.31` vs `67.67`），且**表格列和不再等于 wall**。
> 本表统一用前者，所以 §4 / §7.3 的每张表都能逐列加回 wall。

v1 的 §3.3 dense 表还混用了 `span`（含内部空隙）和 `独占` ⇒ `dense_mlp 83.18` 是 span，
真独占是 `68.77`。所以 **v1 "dense_mlp 慢 1.90×" 应修正为 1.57×**。

### 2.3 保留 v1 的三个判断（审计后确认没错）

| 判断 | 为什么保留 |
|---|---|
| `head_gate` 留在 `3_post_attn` | 见 §2.0。它是 schedule-mobile op，按 dataflow 归桶才可比 |
| vLLM `Abs`(taskid 214) 留在 `8_combine` | 时序 dump 显示它紧贴 `MoeTokenUnpermute`(215) 之前，是 combine 权重的预处理，位置与语义**一致** |
| PyPTO-only `2_attn/softmax_reduce` | vLLM 融进 `FusedInferAttentionScore` 无对应物，但大项 `2_attn` 层面两侧仍然可比，所以保留子项 + 标 "已融合" |

### 2.4 大项定稿（10 个，v1 是 8 个）

`1_pre_attn` `2_attn` `3_post_attn` `4_all_reduce` `5_pre_ffn`（原 "pre-moe"，
改名以容纳 dense 层的 pre-MLP norm）`6_dispatch` `7_ffn`/`7a_shared`+`7b_routed`
`8_combine` `9_epilogue`（新）。`_sync` / `_infra` / `*_HOISTED` 不计入大项合计。

---

## 3. 两侧计量口径不同 —— 必须先理解，否则表会读错

| | PyPTO | vLLM-Ascend |
|---|---|---|
| 执行模型 | persistent `@pl.program` **DAG**，24 AIC + 48 AIV 核并行，shared/routed/comm 是可重叠 lane | 单 stream **串行**下发 kernel，AR 在独立 comm stream |
| 阶段度量 | ① **大项独占**（可加，见 §2.2）② 子项独占（ROI 天花板，不可加）③ `union`（区间并集）④ `busy`（core-µs，跨核累加） | `sum` = 该阶段所有 kernel duration 之和（串行 ⇒ sum ≈ wall） |
| 百分比之和 | 大项独占之和 = 100% − gap | ≈ 95–98%，缺口 = kernel 间 gap |
| AllReduce 口径 | `tp_all_reduce*` task 的 wall（含跨卡等待） | ① 计算流 `CAPTURE_WAIT` 停顿；② comm 流 `hcom_allReduce` duration |

> **可比的列**：PyPTO **大项独占** ↔ vLLM `sum`。两者都是"这个阶段在关键路径上花了多少"。
> **不可比的列**：PyPTO `busy`（core-µs）不能和 vLLM 的 wall 比 —— 它是 24+48 核的累加。

---

## 4. ★ 大项总览（一眼看对比）

### 4.1 full-attention + MoE 层 —— PyPTO **L4** vs vLLM **10 层 moe_full**

PyPTO layer wall = **`458.82 us`**；vLLM = **`408.50 us`** ⇒ **PyPTO 1.12×**

| # | 大项 | PyPTO 独占 | PyPTO 占比 | vLLM sum | vLLM 占比 | 比值 | 读法 |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | pre-attn | `31.46` | 6.9% | `47.76` | 11.7% | **0.66×** | PyPTO 快（norm/QKV/RoPE 与 head-gate 并行掉） |
| 2 | **attn** | **`88.20`** | **19.2%** | `76.52` | 18.7% | 1.15× | ⚠ 见 §8，batch/ctx 不同，比值不成立 |
| 3 | post-attn | `28.17` | 6.1% | `34.04` | 8.3% | 0.83× | PyPTO 快（head-gate 完全并行掉） |
| 4 | all-reduce | `42.03` | 9.2% | `43.97` | 10.8% | 0.96× | 打平 |
| 5 | pre-FFN | `17.07` | 3.7% | `40.80` | 10.0% | **0.42×** | PyPTO 显著快（router 与 norm 并行） |
| 6 | dispatch | **`0.82`** | **0.2%** | `42.62` | 10.4% | **0.02×** | ★ PyPTO 的 EP dispatch **几乎完全藏进 shared expert 之后** |
| 7a | shared expert | `8.10` | 1.8% | `23.80` | 5.8% | 0.34× | 藏在 dispatch 后面，几乎不要钱 |
| 7b | routed expert | `48.21` | 10.5% | `71.04` | 17.4% | 0.68× | PyPTO 快（down 快 2.1×，gmm1 打平） |
| 8 | **combine** | **`57.44`** | **12.5%** | `5.08` | 1.2% | **11.3×** | ★★ **最大结构性劣势**，vLLM 本卡 unpermute，PyPTO 要跨卡 EP combine |
| 9 | epilogue | `12.55` | 2.7% | `7.72` | 1.9% | 1.63× | 残差 merge |
| — | 大项独占合计 | `334.05` | 72.8% | `393.37` | 96.3% | | |
| — | **调度 gap** | **`124.77`** | **27.2%** | `15.13` | 3.7% | | ★ 32 个 task 边界 × ~3.9 us |

### 4.2 SWA + MoE 层 —— PyPTO **L3** vs vLLM **30 层 moe_swa**

PyPTO layer wall = **`387.80 us`**；vLLM = **`371.00 us`** ⇒ **PyPTO 1.05×**

| # | 大项 | PyPTO 独占 | 占比 | vLLM sum | 占比 | 比值 |
|---|---|---:|---:|---:|---:|---:|
| 1 | pre-attn | `27.24` | 7.0% | `50.08` | 13.5% | 0.54× |
| 2 | attn (SWA) | `5.58` | 1.4% | `30.28` | 8.2% | **0.18×** |
| 3 | post-attn | `34.74` | 9.0% | `40.90` | 11.0% | 0.85× |
| 4 | all-reduce | `42.37` | 10.9% | `39.79` | 10.7% | 1.06× |
| 5 | pre-FFN | `17.31` | 4.5% | `41.64` | 11.2% | 0.42× |
| 6 | dispatch | **`0.00`** | **0.0%** | `39.41` | 10.6% | **0×** |
| 7a | shared expert | `9.25` | 2.4% | `23.72` | 6.4% | 0.39× |
| 7b | routed expert | `53.16` | 13.7% | `73.20` | 19.7% | 0.73× |
| 8 | **combine** | **`61.01`** | **15.7%** | `5.06` | 1.4% | **12.1×** |
| 9 | epilogue | `11.19` | 2.9% | `7.76` | 2.1% | 1.44× |
| — | 大项独占合计 | `261.85` | 67.5% | `351.85` | 94.8% | |
| — | **调度 gap** | **`125.95`** | **32.5%** | `19.15` | 5.2% | |

★ **v1 修正**：v1 报 "SWA 层 pre-attn / attn / dispatch 独占全部为 0，gap 47.8%"。
真相是**只有 dispatch 是 0**；pre-attn `27.24`、attn `5.58` 都是实的，gap 是 **32.5%**。
v1 那三个 0 是 §2.1#1 的 `residual_hold` 错分造成的遮蔽假象。

### 4.3 SWA + dense-MLP 层 —— PyPTO **L2** vs vLLM **2 层 dense_swa**

PyPTO layer wall = **`277.41 us`**；vLLM = **`194.75 us`** ⇒ **PyPTO 1.42×**（三类层里最差）

| # | 大项 | PyPTO 独占 | 占比 | vLLM sum | 占比 | 比值 |
|---|---|---:|---:|---:|---:|---:|
| 1 | pre-attn | `24.38` | 8.8% | `44.03` | 22.6% | 0.55× |
| 2 | attn (SWA) | `5.69` | 2.1% | `31.82` | 16.3% | **0.18×** |
| 3 | post-attn | `35.88` | 12.9% | `37.15` | 19.1% | 0.97× |
| 4 | all-reduce | `47.83` | 17.2% | `25.81` | 13.3% | **1.85×** |
| 5 | pre-FFN norm | `18.53` | 6.7% | `3.80` | 2.0% | **4.88×** |
| 7 | dense MLP | `68.77` | 24.8% | `43.84` | 22.5% | **1.57×** |
| 9 | epilogue | `8.24` | 3.0% | `3.88` | 2.0% | 2.12× |
| — | 大项独占合计 | `209.32` | 75.5% | `190.34` | 97.7% | |
| — | 调度 gap | `68.09` | 24.5% | `4.41` | 2.3% | |

★ dense 层**没有 MoE 的可重叠 lane，所以一切都在关键路径上**：`dense_mlp` 1.57×
（**v1 报 1.90× 是把 span 当独占**，见 §2.2）、两次 AR 1.85×、pre-FFN norm 4.88×。
**只有 3 层，全网权重低（4%）**，但它是"PyPTO 在无 overlap 可用时的裸速度"的干净读数
—— 说明**该 R5 artifact 的优势主要来自 overlap，不是来自单 kernel 更快**。

### 4.4 ★ 调度 gap 的真实性质：每 task 边界约 4 us 的固定开销

| PyPTO 层 | wall | 计入大项的 task 数 | gap | **gap / task** | gap 占比 |
|---|---:|---:|---:|---:|---:|
| L1 swa_dense | `279.96` | 16 | `67.67` | **`4.23`** | 24.2% |
| L2 swa_dense | `277.41` | 16 | `68.09` | **`4.26`** | 24.5% |
| L3 swa_moe | `387.80` | 32 | `125.95` | **`3.94`** | 32.5% |
| L4 full_moe | `458.82` | 32 | `124.77` | **`3.90`** | 27.2% |

**gap/task 在四层里稳定在 `3.90–4.26 us`，跨 2× 的 task 数、1.65× 的 wall 都不变。**

⇒ gap **不是**"SWA 层计算量太小所以调度间隙成主导"（v1 的解释，错），
而是**每个 task 边界固定约 4 us 的 launch + 依赖解析开销**。
两条直接推论：

1. **减少 task 数就是减少 gap**，且收益可预测 ≈ `4 us × 消除的 task 数`。
   MoE 层 32 task ⇒ 融合掉 8 个 task ≈ 省 `32 us/层`。
2. **优化单个 task 的时长不减 gap**。所有"把某 kernel 做快"的候选，天花板都不含这 `~4 us`。

---

## 5. ★ PyPTO L4 层内并行结构（缩进 = 同时在跑）

时间轴单位 µs，相对层起点；median of 8 ranks。`独占` = 子项 ROI 天花板（§2.2）。
（`start`/`end` 逐 rank 取 median，所以 `end−start` 与 `union` 可能差零点几 µs。）

```
时刻        阶段                          区间           span   独占   busy(core-µs)
──────────────────────────────────────────────────────────────────────────────────────
  0.0  ┌─ ① pre-attn ∥ ③ head-gate ── 并行块 wall 0.0→51.0 = 51.0 ─────────────────
       │  ①.input_norm                 0.0→ 11.2    11.2   0.01        10.92
       │  ①.residual_snapshot          0.0→  2.8     2.7   0.00 ←全被藏  2.38
       │  ①.qkv_proj                   2.2→ 41.9    39.6   4.28       275.75
       │  ③.head_gate                 16.0→ 36.1    20.5   0.00 ←全被藏 105.99
       └  ①.rope + kv_cache           15.5→ 51.0    35.2   8.00         6.95
 49.7  ┌─ ② attention ── wall 49.7→141.3 (union 89.2) ──────────────────────────────
       │  ②.attn_core                 49.7→125.6    75.7  70.90      4763.40
       └  ②.softmax_reduce           121.7→141.3    19.6  13.19        19.35
                                     〈gap 141.3→147.7 = 6.4〉
147.7     ③.o_proj                   147.7→175.5    28.2  28.17      1588.00
                                     〈gap 175.5→181.9 = 6.4〉
181.9     ④.AR(attn) + resid_add     181.9→207.8    25.3  25.30        25.02
                                     〈gap 207.8→209.8 = 2.0〉
209.8  ┌─ ⑤ pre-FFN ∥ ⑦a shared ∥ ⑥ dispatch ∥ ④ AR ── 6 路并行 209.8→311.3 ────────
       │  ⑤.ffn_norm                 209.9→221.9    12.2   2.88        11.96
       │  ⑤.router_topk              209.8→237.2    27.7   5.16       162.02
       │  ⑦a.shared_expert           226.8→285.8    59.0   8.10 ←几乎全藏 212.98
       │  ⑥.route_offsets            239.4→267.6    28.1   0.82 ←几乎全藏  26.76
       │  ④.AR(ffn/shared)           244.0→311.3    67.1  16.78        23.63
       │  ⑥.ep_transfer(dispatch)    257.0→282.1    25.1   0.00 ←全被藏  21.33
       └  ⑥.permute(gather)          286.0→293.2     7.2   0.00 ←全被藏  36.64
                                     〈gap 311.3→318.5 = 7.2〉
318.5     ⑦b.routed_gmm1            318.5→360.6    41.7  41.72      2364.35
                                     〈gap 360.6→367.2 = 6.6〉
367.2  ┌─ ⑦b down ∥ ⑧ combine 跨卡 ── 367.2→437.5 = 70.3 ─────────────────────────
       │  ⑦b.routed_down             367.2→380.9    13.4   6.52       518.10
       └  ⑧.ep_transfer(combine)     374.8→437.5    62.7  55.95        49.93
                                     〈gap 437.5→442.6 = 5.1〉
442.6     ⑧.unpermute_reduce         442.6→446.3     3.7   1.26        36.26
443.9     ⑨.epilogue merge_residual  443.9→458.8    14.9  12.55         8.91
──────────────────────────────────────────────────────────────────────────────────────
层 wall = 458.82   大项独占合计 = 334.05 (72.8%)   调度 gap = 124.77 (27.2%, 32 task)
核占用合计 = 10 251 core-µs（24 AIC + 48 AIV；÷458.82 ⇒ 平均并行度 22.3 核）
```

**并行块读法**：`独占 ≈ 0` 的子项（residual_snapshot、head_gate、dispatch 的 ep_transfer
与 permute、route_offsets）**对 ITL 基本免费** —— 它们几乎 100% 藏在别的阶段背后。
优化它们的 ROI ≈ 0。真正付钱的是 `attn_core 70.90`、`ep_combine 55.95`、
`routed_gmm1 41.72`、`o_proj 28.17`、`AR(attn) 25.30`、`AR(ffn) 16.78`、
`softmax_reduce 13.19`、`epilogue 12.55`，外加**不属于任何阶段的 `gap 124.77`**。

vLLM 侧对应结构是**一条直线**（单 stream 串行），唯一的并发是 AR：
计算流 `CAPTURE_WAIT` 停顿 `29.79 us` 而 comm 流 `hcom_allReduce` 是 `26.37 us`
⇒ AR 基本**没有**和计算重叠，是裸串行代价。

---

## 6. 小项级完整对照（full-MoE 层）

PyPTO 列为**子项独占 = ROI 天花板**（§2.2，**不可加**，只读单行）。

| 大项 | 小项 | PyPTO union | PyPTO 独占 | PyPTO busy | vLLM sum | vLLM 对应 kernel |
|---|---|---:|---:|---:|---:|---|
| ① pre-attn | input_norm | `11.22` | `0.01` | `10.92` | **`4.02`** | `GemmaRmsNorm` |
| | qkv_proj | `39.64` | `4.28` | `275.75` | `16.86` | `MatMulV2_…98513` |
| | **qkv_epilogue** | — | — | — | **`6.30`** | `Cast`+`Add` ×2（PyPTO 已融合） |
| | rope + kv_cache | `35.20` | `8.00` | `6.95` | `20.58` | `split_qkv_rmsnorm_rope` + `reshape_and_cache` |
| | residual_snapshot | `2.66` | `0.00` | `2.38` | — | vLLM 无（隐式保留在寄存器/显存） |
| ② attn | attn_core | `75.66` | `70.90` | `4763.40` | `76.52` | `FusedInferAttentionScore`(5100…203) |
| | softmax_reduce | `17.40` | `13.19` | `19.35` | — | 已融进上面那个 kernel |
| ③ post-attn | head_gate | `18.51` | `0.00` | `105.99` | `16.64` | `MatMulV2` + `Sigmoid` + `Mul` |
| | o_proj | `28.17` | `28.17` | `1588.00` | `13.94` | `MatMulV2_…98499` |
| | residual_attn | — | — | — | `3.46` | `Add`（PyPTO 融进 AR task） |
| ④ all-reduce | ar_attn(+resid) | `25.30` | `25.30` | `25.02` | `14.18` / comm `13.23` | `CAPTURE_WAIT` / `hcom_allReduce` |
| | ar_ffn | `67.11` | `16.78` | `23.63` | `29.79` / comm `26.37` | 同上 |
| ⑤ pre-FFN | ffn_norm | `12.22` | `2.88` | `11.96` | `7.30` | `GemmaRmsNorm` + `Cast` |
| | router_topk | `18.44` | `5.16` | `162.02` | `33.50` | router FP32 MM + `MoeGatingTopK` + `Index`/`NotEqual`/`Mul` |
| ⑥ dispatch | route_offsets | `28.05` | `0.82` | `26.76` | `31.60` | **`Cumsum`（跑在 AI_CPU）** |
| | permute | `7.02` | `0.00` | `36.64` | `11.02` | `MoeInitRoutingCustom` |
| | **ep_transfer** | `21.94` | `0.00` | `21.33` | **n/a** | vLLM 无跨卡 EP |
| ⑦a shared | shared_expert | `58.02` | `8.10` | `212.98` | `23.80` | shared MM + `SwiGlu` + MM |
| ⑦b routed | routed_gmm1 | `41.72` | `41.72` | `2364.35` | `42.52` | `GroupedMatmulSwigluQuant` |
| | routed_down | `13.36` | `6.52` | `518.10` | `28.52` | `GroupedMatmul` |
| ⑧ combine | **ep_transfer** | `63.05` | `55.95` | `49.93` | **n/a** | vLLM 无跨卡 EP |
| | unpermute_reduce | `3.80` | `1.26` | `36.26` | `5.08` | `Abs` + `MoeTokenUnpermute` |
| ⑨ epilogue | merge_residual | `14.89` | `12.55` | `8.91` | `7.72` | `Add` ×2 |

**几个反直觉但重要的读数**：

- **`routed_gmm1` 已经打平**（`41.72` vs `42.52`）：packed-NZ 融合把 PyPTO 的 GMM1
  做到了和 vLLM `GroupedMatmulSwigluQuant` 同一水平。**这条线没有剩余空间了。**
- **`routed_down` PyPTO 快 2.1×**（`13.36` vs `28.52`）。
- **`input_norm` PyPTO 慢 2.8×**（`11.22` vs `4.02`）—— v1 因为把 vLLM 的 `qkv_epilogue`
  错算进 input_norm（`10.36`）而报成"打平"。但独占仅 `0.01` ⇒ **免费，不值得动**。
- **`shared_expert` PyPTO 慢 2.4×**（union `58.02` vs `23.80`），但**独占只有 `8.10`**
  ⇒ 它躲在 dispatch 后面，**对 ITL 几乎不要钱**。"不要先优化 shared expert" 得到独立印证。
- **`o_proj` PyPTO 慢 2.0×**（`28.17` vs `13.94`）且**全额独占** ⇒ 这是一条被忽视的
  真实关键路径项，`1588 core-µs` busy 说明它是算力受限而非等待。
- **`qkv_proj` busy `275.75` vs `o_proj` `1588` core-µs**：差 5.8×，两者 FLOPs 同量级
  ⇒ `o_proj` 的 tile/chunk 配置需要复核。
- **vLLM 的 `Cumsum` 花 `31.60 us` 且跑在 AI_CPU 上** —— 这是 vLLM 侧一个明显的结构缺陷，
  PyPTO 同功能的 `route_offsets` union `28.05` 但独占仅 `0.82`（藏起来了）。

---

## 7. ★ 逐层统计

### 7.1 vLLM 45 层完整分布（68 个 decode step，p50）

| 层类 | 层数 | wall p50 | 层号 | 主要差异 |
|---|---:|---:|---|---|
| `dense_full` | 1 | `242.88` | L0 | full attn `77.38`，无 MoE |
| `dense_swa` | 2 | `194.75` | L1, L2 | SWA attn `31.82`，最快的层 |
| `moe_swa` | **30** | `371.00` | L3…L42 中的 SWA | **全网主力形态** |
| `moe_full` | 10 | `408.50` | L4/8/…/40 | attn `76.52` 比 SWA 多 `46 us` |
| `special_swa` | 1 | `385.25` | L43 | `routed_gmm1` `98.14`（走 `GroupedMatmul`+`Slice/Swish/Clip/Mul`+`DynamicQuant`） |
| `special_full` | 1 | `490.38` | L44 | `routed_gmm1` `96.42` + epilogue `57.90`（含 LM-head 前置） |

**类内离散度**（说明层识别与聚合是干净的）：
`moe_swa` n=2040 实例，min `358.62` / p50 `371.00` / max `382.88` ⇒ 极差 **6.5%**；
`moe_full` n=680，min `403.12` / p50 `407.75` / max `421.00` ⇒ 极差 **4.4%**。

### 7.2 每类层对一个 decode step 的贡献

| 层类 | 层数 × wall | 合计 (µs) | 占 step |
|---|---|---:|---:|
| `moe_swa` | 30 × `371.00` | **`11 130.00`** | **66.6%** |
| `moe_full` | 10 × `408.50` | `4 085.00` | 24.4% |
| `special_full` | 1 × `490.38` | `490.38` | 2.9% |
| `dense_swa` | 2 × `194.75` | `389.50` | 2.3% |
| `special_swa` | 1 × `385.25` | `385.25` | 2.3% |
| `dense_full` | 1 × `242.88` | `242.88` | 1.5% |
| **45 层合计** | | **`16 723.01`** | 100% |

> Main span 实测 `18.067 ms` ⇒ 层外开销 `1.34 ms`（层间 gap + embedding + final norm + LM head）。
> **优化排序的第一性依据：`moe_swa` 一类占 2/3。任何只作用于 full 层的优化，
> 全网权重上限就是 24.4%；只作用于 dense 层的上限是 3.8%。**

### 7.3 PyPTO 五层（median of 8 ranks，大项独占口径）

| 大项 | L1 swa_dense | L2 swa_dense | L3 swa_moe | L4 full_moe |
|---|---:|---:|---:|---:|
| 1 pre-attn | `25.34` | `24.38` | `27.24` | `31.46` |
| 2 attn | `5.80` | `5.69` | `5.58` | **`88.20`** |
| 3 post-attn | `36.74` | `35.88` | `34.74` | `28.17` |
| 4 all-reduce | `47.89` | `47.83` | `42.37` | `42.03` |
| 5 pre-FFN | `18.59` | `18.53` | `17.31` | `17.07` |
| 6 dispatch | — | — | `0.00` | `0.82` |
| 7 dense MLP | `69.76` | `68.77` | — | — |
| 7a shared | — | — | `9.25` | `8.10` |
| 7b routed | — | — | `53.16` | `48.21` |
| 8 combine | — | — | **`61.01`** | **`57.44`** |
| 9 epilogue | `8.17` | `8.24` | `11.19` | `12.55` |
| **大项合计** | `212.29` | `209.32` | `261.85` | `334.05` |
| **调度 gap** | `67.67` (24.2%) | `68.09` (24.5%) | `125.95` (32.5%) | `124.77` (27.2%) |
| **layer wall** | **`279.96`** | **`277.41`** | **`387.80`** | **`458.82`** |

**跨层稳定性**：L1 与 L2 是同构层（都 swa_dense），逐大项差异均 ≤ `1.0 us`
（最大 `dense_mlp 0.99`，占 wall 的 0.36%）⇒ **测量重复性 < 0.5%**，可以放心做单层对比。
层 wall 跨 8 rank 只差 ±1%。

**L0 不可用**：7/8 rank 在 L0 的 `tp_all_reduce_residual_bs1` 上阻塞 `330–630 ms`
等最慢 rank 到齐（进程启动 skew 的**冷启动 barrier**，不是性能），已从所有对比剔除。
因此**没有 full-attention + dense 层的 PyPTO 读数**。

### 7.4 同类层跨栈对照

| 层类 | PyPTO wall | vLLM wall | 比值 | 全网权重 (vLLM) |
|---|---:|---:|---:|---:|
| swa + dense | `277.41` (L2) | `194.75` | **1.42×** | 2.3% |
| swa + MoE | `387.80` (L3) | `371.00` | **1.05×** | **66.6%** |
| full + MoE | `458.82` (L4) | `408.50` | **1.12×** | 24.4% |
| full + dense | — (L0 冷启动污染) | `242.88` | n/a | 1.5% |
| special ×2 | — (harness 无 special 层) | `385.25`/`490.38` | n/a | 5.2% |

**加权粗估**（**只做量级校验，不是承诺**，见 §8）：把 PyPTO 三类比值按 vLLM 权重加权，
覆盖到的 93.3% 层里平均 `1.08×` ⇒ 整网 `~18.1 ms` vs vLLM `16.72 ms`。

---

## 8. ⚠ 有效性边界（不许跨过去用）

1. **workload 不同，attention 与 AllReduce 的绝对比值不成立。**
   PyPTO = BS1 / ctx 64K；vLLM = **BS4 / ctx 未知**。trace 里**没有 tensor shape**
   （arg key 只有 `Task Id`/`Model Id`/`Physic Stream Id`/`count`/`size(Byte)` 等），
   `FusedInferAttentionScore` 的 KV 长度**不可恢复**。attention 与 KV 相关的行只能
   读结构占比，不能读比值。
2. **PyPTO 是 5 层 synthetic harness，不是整网。** 层间流水、weight prefetch、
   LM head、MTP3 都不在里面。**不要把 §7.4 的加权推估当整网预测。**
   历史 R5 source-overlay 的整网旁证是 ITL p50 `27.757 ms` @ITERS=100；
   它不替代 current r9 的默认 `none=27.812 ms` 或发布合同 `all=22.253 ms`。
3. **PyPTO 每 rank 只有 1 次 invocation ⇒ 无法给 p50/p95**，本表 PyPTO 数字是
   **8 个 rank 的 median**，离散度用 min/max 看（层 wall 跨 rank 只差 ±1%，很稳；
   §7.3 的 L1/L2 同构层复现性 < 0.5%）。vLLM 数字是跨 `68×层数` 个实例的 p50。
   **两侧统计强度不对等。**
4. **PyPTO 大项独占之和可加，子项独占之和不可加**（§2.2）。**§6 的 PyPTO 独占列
   不要求和**，只能逐行读作 ROI 天花板。
5. **`alloc` / `attn_out_zero` 被调度器提前吊起**（L3 的 alloc span 达 `589 ms`），
   已从 span/层窗口剔除（busy 合计仅 `~3 core-µs`）。若不剔除会把层 span 算成 589 ms。
6. **MoE 拓扑差异是真实的，不是可消除的对齐误差**：vLLM 本卡 local routing；
   PyPTO EP8 跨卡 dispatch/combine。`6_dispatch/ep_transfer` 与 `8_combine/ep_transfer`
   两行**在 vLLM 侧不存在对应物**，标 `n/a` 而不是硬凑格子。
7. **L43/L44 special 层 PyPTO 侧缺失**，无法对比（§7.1 只给 vLLM 侧读数）。
8. **MTP3（Model 86）3 层未做分解** —— 已探到结构（`ConcatD` + 两个大 MatMul
   `63.52`/`98.80` + `GatherV2`/`MaskedFill` + `ArgMaxV2` `17.48` + `Pack` + AR 停顿三连），
   但未归桶聚合，超出本次范围。
9. **vLLM 的 `qkv_epilogue`（`Cast`+`Add`×2）功能归属是推断**，依据是它夹在 QKV matmul
   与 rope 之间。trace 无 shape 无法证实。若它其实是别的东西，`pre_attn` 大项总数不变，
   只是子项归属变。

---

## 9. 2026-08-21 历史结论与下一步候选（当时均未立项）

> 本节只描述 R5/source-overlay + 2026-08-21 vLLM trace 的历史候选。
> 当前 r9 的 observed-path 证据不能由本节的 gap/task 或 ROI 天花板直接推出；
> 当前 roadmap 以 `design/performance/09-swimlane-derived-next-optimizations.md`
> 和 `design/performance/task-tracking.md` 为准。

| # | 候选 | 依据 | 天花板（单层） | 全网权重 |
|---|---|---|---:|---|
| 1 | **减少 MoE 层 task 数** | **历史 R5** gap = `4 us × task 数`（§4.4，四层实测 `3.90–4.26`，稳定）。MoE 层 32 task ⇒ `~125 us` gap | `~4 us × 消除的 task 数` | **40 层 (91%)** |
| 2 | **EP combine 通信 overlap** | `8_combine/ep_transfer` 独占 `55.95`(L4)/`59.79`(L3)，但 busy 仅 `49.93 core-µs`、span `63.05` ⇒ 含跨卡等待 | `~30 us` | 40 层 |
| 3 | **`o_proj` 复核** | 慢 vLLM 2.0× 且**全额独占**；busy `1588 core-µs` vs 同量级 FLOPs 的 `qkv_proj` `276` ⇒ 差 5.8×，疑 tile/chunk 配置 | `~14 us` | 45 层 |
| 4 | **`AR(attn)` overlap** | 独占 `25.30` 全额在关键路径，且 `busy` 仅 `25.02 core-µs` ⇒ 几乎纯等待 | `~15 us` | 45 层 |
| 5 | **`epilogue merge_residual`** | 独占 `12.55`(L4)/`11.19`(L3)，慢 vLLM 1.6×，busy 仅 `8.91` ⇒ 不是算力受限 | `~8 us` | 45 层 |
| 6 | `softmax_reduce` 融进 attn_core | 独占 `13.19`，vLLM 已融合 ⇒ 存在性证明 | `~13 us` | **仅 12 full 层** |
| 7 | dense 层裸速度 | `dense_mlp` 1.57×、AR 1.85×、pre-FFN norm 4.88×，无 overlap 可藏 | — | 仅 3 层 (3.8%) |

**在该历史 R5 对照中已确认没有空间的方向**（不要据此覆盖 r9 结论）：
`routed_gmm1`（已打平 `41.72` vs `42.52`）、`shared_expert`（独占 `8.10`，藏在 dispatch 后）、
`head_gate` / `residual_snapshot` / `dispatch` 全部子项（独占 ≈ `0.00`，对 ITL 免费）、
`input_norm`（虽慢 2.8× 但独占 `0.01`）。

**历史优化排序的第一性原则**（§7.2）：`moe_swa` 一类占 decode step 的 **66.6%**。
候选 1/2 直接命中它；候选 6 只作用于 12 个 full 层，全网权重上限 24.4%；
候选 7 上限 3.8%。

> 任何候选立项前按 `.claude/skills/pypto-perf-regression/` 走：先算 ROI 天花板
> vs A/B/A 检测地板，再无卡 codegen 门。**本表只给天花板，不给裁决。**
