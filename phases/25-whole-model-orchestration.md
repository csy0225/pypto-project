# Phase 25 —— 整网对齐 / whole-model orchestration（Wave-3）

> **状态**：设计 kickoff 2026-07-03。这是项目的终局架构 —— **整个 decode forward
> 编译成一个 pypto program（一个 chip_process，host 出 loop）**，也是你要的「PyPTO
> 整栈接管」。**它同时消灭 Phase 24.4 的「双 chip_process 同卡 co-tenancy」阻塞**
> （whole-model = 一个 program = 一个 chip_process/卡，不存在第二个）。
>
> 承接 [`24-live-layer-replacement.md`](24-live-layer-replacement.md)。多周级工程,分阶段。

---

## 1. 为什么（与 24.4 的关系）

Phase 24 用「每 op 一个出进程 worker + device-IPC」把 attention 全 45 层零拷贝接进
live vLLM（24.1-24.3 ✅）。但 24.4 整层撞到:**两个 pypto chip_process 同卡不能共存**
（各自 AICPU orchestration / device-side singleton 冲突 → `run_prepared code 13`；mem
config 改不动,已证伪）。

**根因是「多 worker 出进程」路线本身**:每加一类 op（MLP/MoE）就多一个 chip_process。
**Phase 25 的 whole-model program 从架构上终结它** —— 整网（attn+MLP+MoE+lm_head 48 层）
编译进**一个** `@pl.program`,一个 chip_process 跑完整 forward,host 一次 upload+kick+sync
出 loop（见 [`../architecture/aclgraph-vs-pypto-execution-model.md`](../architecture/aclgraph-vs-pypto-execution-model.md) §4）。这才是「整栈融合收益」的载体。

## 2. 当前 gap（Wave-3 未做）

`models/step3p5/decode_fwd.py` 的 `Step3p5DecodeFwd.host_orch`（L356-370）**是
placeholder**:只跑最后的 RMSNorm + lm_head;48 层 dispatch 被 staged 在 program 外
（每个 per-layer program 各自带 host_orch —— 这正是 24.x 逐层 out-of-process 的根源）。

`select_decode_layer(li)` + 2 个 per-layer builder（`_build_decode_layer_dense_program`
/ `_build_decode_layer_moe_program`）已存在且各自能跑（Phase 19 ST）。Phase 25 = 把它们
**融进顶层那一个 chip_orch**。

## 3. 硬约束：§10 no-nested-program → body-copy 内联

pypto **不能**在 `@pl.program` body 里实例化另一个 `@pl.program`（known-pitfalls §10）。
所以 48 层不能「调用 48 个 per-layer program」,必须把每层 body **逐字内联**成
`@pl.function(InCore/Inline)` 进顶层 `chip_orch` → 生成**一个** orchestration `.so` →
48 层几百个 task 进同一张 AICPU task graph → 一次 kick、host 出 loop。

- full/swa、dense/MoE 是**不同 code shape**（编译期定死 shape）,即使融进一个 program
  仍是 8 种内联特化出现在各自层位（§5）。
- 层间 `current_hidden` 串接,被 `tp_all_reduce` reduce 过。

## 4. 每步 ABI（从 vLLM 取，Phase 24 已探明）

whole-model runner 每步需:hidden、positions、seq_lens、**block_table、slot_mapping、
KV cache view**、rope tables、全部 per-layer 权重。**block_table/slot_mapping/KV 在
vLLM-Ascend eager path 的 `ForwardContext` 里拿不到**（实测 `attn_metadata=None` /
`slot_mapping={}` / `kv_cache.shape=[0]`）→ 必须从 `model_runner.input_batch` +
attention backend 内部取（可能要 vLLM-Ascend 上游改动）。KV 用 Phase 24 的
**一 key 整池 map**（in-process 后甚至可直接 data_ptr,无需 IPC）。

## 5. 权重翻译

in-memory `nn.Module` → PyPTO bundle:`weight_translate.py` 已验证 live 参数 metadata
contract（744 params）+ `--emit-vllm-transform-plan`（qkv split / gate_up split / MoE
dequant）。**待落地真实 in-memory tensor extraction**（当前只到 plan + disk loader）。

## 6. 阶段里程碑

| 里程碑 | 内容 | gate |
|---|---|---|
| **25.1** | 单层 fused `decode_fwd`（1 层 dense 内联进 host_orch）compile + run，对 golden | §10 内联骨架跑通 |
| **25.2** | dense-only 3 层（0-2）fused，host 出 loop，对 baseline | 25.1 |
| **25.3** | 全 45 层 mixed（dense + MoE）fused compile | MoE 段 gate 507018 |
| **25.4** | wire vLLM `full` mode（`vllm_monkey_patch.py` full 目前 fail-closed）+ in-memory 权重翻译 + block_table/slot_mapping ABI；真实请求走 whole-model runner | 25.3 + ABI（可能上游 vLLM-Ascend patch） |
| **25.5** | L1/L2/L3 在线精度 gate（复用 Phase 21 harness + dump oracle） | 25.4 |

## 7. 依赖 / 阻塞

- **MoE 段 507018**:25.3 的 MoE 内联段在真机跑仍会撞 507018（同 24.4-MoE）。dense 路径
  （25.1/25.2）不受阻,可先行。
- **vLLM-Ascend ABI**:block_table/slot_mapping 可达性可能要上游改动（25.4 最大未知）。
- **§10 内联 + buffer 上限**:48 层融一个 program,L0/L1/UB 分配 + task DAG 规模要实测
  （可能撞 buffer 上限,需分段 chunk）。

## 8. Status

- **设计 kickoff（2026-07-03）**:架构 + gap + §10 约束 + ABI + 里程碑已定。
- **未实现**:25.1 起。这是**多周级工程**;dense 路径先行,MoE 段 gate 507018,vLLM full-mode
  wire gate ABI。
- **与 24.4 的关系**:whole-model program（一个 chip_process）**取代** 24.4 的单-worker 合并
  中间态 —— 不必再做 24.4 的 attn+MLP worker 合并,直接在 Phase 25 收敛。
