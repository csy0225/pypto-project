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

## 9. 进展 — 核心可行性 de-risk（2026-07-04）

两个最大未知已用最小 device probe 验证清除:

- **§7 融合规模 — 45 层链 COMPILE OK**:`_stage_chain_scale.py`（0162,host-only compile,
  TP=8 DistributedConfig）用 `pl.unroll(N)` 把自包含的 `_dense_mlp_body_tp` 在**一个**
  `@pl.program` chip_orch 里串 N 次(复用一对 window,只探 compile 规模)。**N=1 / 8 / 45 全部
  `COMPILE OK`**(完整 frontend→IR→ptoas→distributed codegen)。→ 整网融合在 dense 路径上
  **编译规模可行到 45 层**;§7 的 task-DAG / L0/L1/UB buffer 规模不再是未知。
- **head-gate on-device — matmul_acc N=16 不是 blocker**:`_stage_gate_matmul.py`(0162 card 8)
  证明 `normed[16,4096] @ w_g[4096,16]` 经 K-chunk matmul_acc(真实 KC=256)**PASS**——历史
  记载的 "N=16 matmul_acc 丢 K 累加" 是 context-specific(stale/reused normed_all),非小-N
  matmul 本质缺陷。N=8 触发 `alloc_tile` 列须 mult-of-16(所以真实 gate pad 到 16)。→ whole-model
  的 on-device gate 可行:gate_logits matmul(N=16) → sigmoid → block-diag expand(常量 gate_r
  [16,1024],large-N 安全)→ 乘 attn_out。免去 live worker 的 Python per-step gate 预算。
- **§10 many-body inline 已被 DecodeLayerMoE 证明**(整套 EpTpMoE + attention 内联进一个
  chip_orch,8 卡 runtime PASS);chain probe 又证明 DEPTH(45)也能编。

frontend 关键规则(写真实融合 chip_orch 必看):(1) 层循环用 `pl.unroll(N)` 不能用裸 `range()`;
(2) `pl.inline(fn._func)` 的 body free-var 按**调用方 module globals** 解析 → whole-model builder
必须 import 所有被内联 body 引用的常量(K_CHUNK/EPS/LAYER_*_DYN/MLP_OUT_CHUNK/SWIGLU_LIMITS…);
(3) `_dense_mlp_body_tp` 调 `self.tp_all_reduce(...)` → 外层 program 必须定义 barrier
`tp_all_reduce` @pl.function(InCore)。注意 attention 的 tmp_window 是 `[BATCH, HIDDEN//TP]`
(只有 TP=1 时才 == `[BATCH,HIDDEN]`)。

### 剩余工作(after de-risk;仍多 session)
1. 真实融合 chip_orch:逐层 {attention_full|attention_swa} + {dense_mlp|MoE} + 末尾 rms_lm_head,
   `pl.unroll` 45 层,layer-indexed 权重切片,per-call-site 新 window(数量爆炸 = 设计项)。
2. on-device gate 落 attention_full(已 de-risk,待 in-context 验证)。
3. RUN 融合 program 对 golden(8 张空闲卡 + 真实权重 bundle + vLLM dump 的真实 decode 输入;
   oracle = 8000 vanilla logits)。MoE 段在融合上下文可能撞 507018(单层 MoE 8 卡已过,融合-45 未测)。
4. 25.4 live:in-memory 权重翻译 + block_table/slot_mapping/KV ABI(eager ForwardContext
   `attn_metadata=None` → 大概率要 vLLM-Ascend 上游 patch)+ 真实 decode 请求走 whole-model runner。

probe 脚本(0162 `/data/chensiyu/hw_project/pypto/`):`_stage_chain_scale.py`(CHAIN_N 环境)、
`_stage_gate_matmul.py`(GATE_KC/GATE_N)。
