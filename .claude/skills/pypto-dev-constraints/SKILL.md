---
name: pypto-dev-constraints
description: >
  PyPTO / step3p5 开发约束与审计分层指南。用于修改或审查 kernel、@pl.program、
  MoE/attention/KV、通信、依赖、编译、真机部署和 vLLM 集成。先按 DeepSeek
  基线判断语义与数据流；只有明确的 step3p5 差异或当前 backend ABI 才增加适配和 probe。
---

# PyPTO / step3p5 约束分层指南

本文件不是历史 workaround、某次 probe 的源码模板，也不是把当前产品选择永久化的“设计宪法”。
它规定如何判断一条规则是否能约束产品代码。

## 0. 总原则：DeepSeek-first，证据分层

### 0.1 基线优先级

实现新路径时按以下顺序判断：

1. **DeepSeek V4-Flash 已有同构实现**：以 `origin/main:models/deepseek/v4-flash/` 为首要源码基线，优先沿用其语义、数据所有权、通信方向、累加顺序和生命周期；不因为 PyPTO probe 的写法不同而另造架构。
2. **DeepSeek 同构、仅参数/shape 不同**：保留同一算法和数据流，只参数化适配。
3. **step3p5 有明确独有差异**：例如 45-layer whole-net 生命周期、active-token/KV storage ABI、stacked/reused signal slot、现有 vLLM/IPC 集成。先写差异和影响，再实现最小适配。
4. **仅因“可能编译不过”而提出的替代结构**：先查当前 PyPTO/PTOAS/simpler 能力和上游状态；不能让 synthetic probe 反向规定产品架构。

### 0.2 端到端同构审计法（顶层方法论）

禁止用局部 `shape`、函数名、变量名、单个 primitive 或单段调用链做架构判断。任何“step3p5 独有”“必须保留”“必须回退”“与 DeepSeek 不同”的结论，必须沿完整数据路径核对：

```text
producer
  → 数学变换 / quant / route-map
  → transport / communication window
  → consumer
  → rounding / reduction / placement
  → lifetime / reuse / allocator ownership
```

逐项审计至少回答：

1. producer 产生什么数学对象、有效范围和metadata；
2. 中间是否发生quant、scale、route、weight或layout变换；
3. transport/window 的写入方向、可见性、ownership和capacity是什么；
4. consumer实际读取什么、如何解释metadata和有效范围；
5. rounding、FP32/BF16、top-k顺序、weight placement和最终store是否等价；
6. 最终consumer何时完成，buffer何时复用/回收，host/allocator/IPC谁拥有生命周期。

完成端到端核对后，必须把差异归入以下类别，不能把它们混写：

- **能力/算法差异**：是否缺少或新增了真实执行能力、并行轴、通信方向或阶段；
- **数学语义差异**：量化、scale、route、weight、rounding、reduction顺序或有效范围改变；
- **layout/shape差异**：同一数学对象的存储布局、padding、formal shape、capacity或tile不同；若V4-Flash已有同构能力，单独的layout/shape不同不能称为架构差异；
- **host/allocator集成差异**：入口、resident/IPC、KV ownership、window materialization、allocator或whole-net调用生命周期不同；
- **backend/profile workaround**：特定frontend/backend/镜像/设备版本的编译、span、alignment、liveness规避；不能升级为模型语义或永久产品约束。

任务描述、旧设计和历史状态可能落后于代码。执行前必须先核对当前source、当前调用链和当前工作树diff，再与 `origin/main:models/deepseek/v4-flash/` 对账；文档任务卡不能覆盖current source事实。若current source与任务描述冲突，先记录冲突和证据等级，再按用户最新决定更新合同或实现。

本次形成通用规则的反例包括：

- **INT8 + scale**：只看INT8 activation shape会漏掉per-token scale、scale padding、dequant位置和consumer读取方式；必须沿quant→scale transport→expert dequant→rounding全链核对；
- **owner max**：只看`num_tokens` scalar名称会漏掉owner-vector producer、跨rank max、active范围下传和KV/通信consumer；必须核对owner publish→max→各stage bound→lifetime；
- **BATCH=16**：formal `[16,...]`可能只是capacity上界，不能据此断言逻辑batch或永久padding/reserve；必须核对runtime active batch/token、valid范围和KV写入；
- **route-weight placement**：只看route index或weight tensor shape会漏掉route与weight的producer/transport/consumer配对、top-k位置和FP32 weighted reduction；必须核对route/weight placement到最终reduction。

这些反例不是新的固定shape或实现模板，而是说明为什么必须使用端到端方法。

### 0.3 规则的四种状态

每条规则必须标注以下一种状态，不能混写：

- **语义硬约束**：违反会改变模型数学语义、内存安全、真实依赖正确性或产品入口。
- **当前 ABI 约束**：只对明确版本/backend/profile 有效，必须带适用范围和验证方式。
- **step3p5 产品 profile**：当前交付选择或资源配置，可演进，不是通用 PyPTO 禁令。
- **历史/诊断经验**：用于定位问题，不能单独阻止语义等价的产品方案。

### 0.4 证据不能越级

报告必须区分：

```text
source contract
compile
lowered IR/codegen
synthetic
semantic invariant
runtime DAG/liveness
real device
performance
```

低等级证据不能宣称高等级完成。standalone probe 能编译，只证明该 probe 可编译；不能证明 canonical 产品架构、DeepSeek 语义、runtime liveness 或设备行为。

---

## 1. 可以作为硬约束的内容

本章所有条目状态均为 **[语义硬约束]**；若某条依赖当前 runtime ABI，会在条目内单独注明。

### 1.1 语义、所有权和生命周期

以下是产品正确性约束，不绑定某一种实现形式：

- runtime logical batch/token 必须由每次调用的 active count 决定；若frontend要求静态formal shape，`shape`只能表示可配置的physical capacity上界，`valid_shape`/loop bound表示本次逻辑有效范围。某个默认capacity（包括16）不得升级为产品逻辑batch硬约束。
- `Out`/`InOut`/返回值的物理 alias 和 ownership 必须明确；多个并发 writer 的物理写区间必须不重叠。
- buffer 只能在所有真实 consumer 完成、scope/token 生命周期闭合且 fanout 已满足后回收或复用。
- 跨 epoch 复用的 data/control buffer 必须等待前一 epoch 的最终 semantic consumer；不能只等待 remote transfer 发起或某个 peer 返回。
- 不得用变量名、提交顺序、地址偶然性或 `x * 0` 等 fake dependency 代替真实 DAG 边。
- capacity中未激活的rows、物理padding、KV slot和route row必须被runtime active bound正确屏蔽；不得把inactive capacity rows当作逻辑batch成员，也不能依赖未初始化内存。

具体是 peer slab、fixed-slot、push、pull、`spmd_submit`、builtin collective 还是其他合法实现，属于实现选择；硬约束是 ownership、alias、lifetime 和结果语义。

### 1.2 执行模型和依赖

在当前 PyPTO frontend/runtime 中：

- Orchestration 负责建 DAG/submit；InCore/worker 负责计算，不能在 worker 内递归 submit。
- `pl.parallel` 不得被当作带 carried state 的 InCore task 并行化工具；`pl.spmd` 的层级和语义必须按当前 frontend 规则使用。
- `manual_scope` 关闭或限制自动依赖时，必须显式补齐真实 producer-consumer 边。
- 当前 runtime 若采用 RAW-only external dependency ABI，必须保证真实 RAW 路径可追踪；不得把该 ABI误写成所有未来 backend 的普遍定律。
- completion notify 必须发生在最终数据消费/输出写入之后；reuse wait 必须与同一 signal lineage 和 epoch 协议闭合。

### 1.3 DeepSeek/step3p5 数学语义

以下只约束对应模型路径，不是所有 PyPTO reduction 的通用模板：

- `tp_all_reduce` 保持固定 peer 顺序 `0..tp_size-1`。
- reduction 使用单一 FP32 accumulator：self 加 own tile，其他 peer 加 remote load；peer loop 内不得 BF16 中间写回 reduction target；最终只做一次 BF16 cast/store。
- 不得用 rank-dependent 的 local 初值、跳过 self、逐轮 BF16 写回等形式改变加法顺序和 rounding。
- W8A8/int8 quant、router bias、shared expert、EPS、top-k weighted gather 等必须以 DeepSeek/vLLM 数学语义和明确 oracle 为准，不能用 silent fallback 掩盖错误。
- KV cache 的 layer/slot/owner/address 规则必须与实际 holder、runtime 和模型调用契约一致。

若怀疑 compiler/lowering 破坏了上述语义，应单独记录 compiler issue；不能为取得 compile PASS 而改变产品数学语义。

---

## 2. step3p5 的明确产品差异

本章条目状态为 **[step3p5 产品 profile]**，除非条目明确标为语义硬约束。它们不是 DeepSeek 通用 ABI，只在 step3p5 当前路径适用，并且应在代码/文档中注明来源。

### 2.1 Canonical 入口

**状态：[step3p5 产品 profile]。**

唯一产品入口为：

```python
models.step3p5.decode_fwd:whole_decode_step3p5
```

retired `decode_layer_single_chip_hidden.py` 不得作为默认入口；历史文件可删除，不再为其维护 release 约束。

### 2.2 Whole-net 与 active-token/KV ABI

**状态：[V4-Flash 已有的通用模型模式 + step3p5 产品 profile 的参数化差异]。**

V4-Flash 已经在整网 decode 中共享一套 MoE communication windows，并在多次 MoE 调用之间使用单调 `moe_epoch`；因此“多层共享 window”与“跨调用 epoch”不是 step3p5 独有，不能作为独有架构理由重复发明。

step3p5 的实际差异是：

- canonical 是 45-layer whole-net graph，window 类型和层型组合不同；
- C2/C3 直接迁移 V4-Flash expert-lane dispatch/combine 数据流，并在 whole-net 中适配共享 window、单调 epoch、active-token、INT8 scale/route metadata 与现有 expert ABI；
- consolidated KV pool、vLLM IPC ownership与allocator-owned KV capacity是step3p5集成profile；capacity是可配置物理上界，不定义本次逻辑batch/token；
- owner-vector到whole-net的runtime active batch/token接线是当前step3p5 host/runtime ABI。

C1只规定共享communication window、单调epoch、`AtomicAdd`/`WaitCmp.Ge`和真实arrival/completion生命周期；不得把current pull的ready/read-complete双波阈值写成迁移后协议的硬约束。epoch的具体计数步长由V4-Flash数据流及step3p5 whole-net复用关系确定，并用runtime DAG/liveness证明。

physical capacity与runtime active batch/token分离本身也不是step3p5独有；V4-Flash同样使用容量上界与runtime `num_tokens`。step3p5独有的是holder/IPC/KV地址与capacity配置方式，不是`BATCH=16`、固定padding行或固定reserve行数。

### 2.3 Stacked/reused control signal 的 512B stride

**状态：[当前 backend/profile 约束]，不是永久模型语义；backend span/provenance 行为或上游实现变化后必须重新验证。**

DeepSeek 的 control signal 仍可以是：

```text
[N_RANKS, 1] INT32
alloc_window_buffer(N_RANKS * 4)
```

DeepSeek 中大量 512B 主要服务于 data tile、L2 cache line、MTE 和性能对齐，不是通用 control-signal window ABI。

V4-Flash 已经复用这些 compact control windows；step3p5 仅因当前 stacked/reused backing 的 backend span/provenance 风险，在以下条件同时满足时使用本地 512B 隔离：

- signal 被 `notify` / `wait` / `AtomicAdd` 使用；
- 多个 layer/slot 在同一个 backing buffer 中 stacked，或跨 epoch reused；
- 需要保证相邻物理 slot 隔离，并规避当前 backend 对 byte-span/window provenance 处理可能造成的邻接 slot 重叠或 false sharing。

此时推荐：

```python
COMM_CONTROL_SIGNAL_BYTES = 512
COMM_SIGNAL_STRIDE_I32 = 128
formal/window shape = [COMM_SIGNAL_STRIDE_I32, 1]
```

逻辑循环仍只访问前 `n_ranks` 行。普通 data window、独立非堆叠 signal、MTP compact signal、rollback signal 不得机械扩成 512B。

### 2.4 DeepSeek V4-Flash 同构的 dispatch/combine

**状态：[DeepSeek V4-Flash 算法基线]。**

V4-Flash 已有的 expert-lane SPMD dispatch push、metadata/payload arrival、expert-lane gather、combine scatter、arrival wait 和 token reduction，是 C2/C3 的产品算法基线。step3p5 必须直接迁移这一数据流；current fixed-slot pull 只保留为迁移前历史基线和回归对照，不再是目标架构，也不再等待“先量化再决定是否迁移”。

迁移允许参数化适配step3p5的runtime active batch/token、可配置capacity上界、`TOPK/HIDDEN/N_LOCAL`、INT8 scale物理padding、route/count/map表示、whole-net shared-window/epoch和下游expert tensor ABI，但不得改变以下语义：

1. dispatch 以 local-expert lane 划分 write-disjoint ownership；
2. metadata arrival 与 payload arrival 有真实依赖和可复用 epoch；
3. expert-lane gather 只写本 expert 的最终或明确分区输出；
4. combine 使用 expert-lane scatter、arrival wait 和 token-level reducer；
5. top-k weighted reduction 保持既定 FP32 累加顺序与最终 store 语义；
6. completion/reuse 不得早于最终 semantic consumer；
7. host/window ABI 的容量和 shape 由模型上界、对齐和现有 consumer ABI推导。

V4-Flash 的示例常量、某次 probe 的 peer-major slab、`spmd_submit`、`task_dummy`、sole reducer 或具体 tensor shape 都不是产品硬约束。probe 只能验证当前 backend 对某个适配点的表达能力，不能定义产品数据流或窗口容量。

---

## 3. 当前 ABI/profile：可以用，但不能冒充通用定律

本章条目状态均为 **[当前 ABI 约束]** 或 **[当前部署 profile]**，不得升级为通用硬约束。

下列规则只在实际使用的 PyPTO/PTOAS/simpler 版本、Ascend backend 或 step3p5 release profile 中有效。修改前必须确认当前代码和版本仍适用：

- `DistributedTensor`/CommCtx 的 materialization、remote op 和 submit return lineage。
- `submit`/`spmd_submit` 的 parser 形态、TaskId 可见范围和 `manual_scope` 依赖规则。
- storage shape 的 backend alignment、tile layout、DMA/GM 对齐。
- `pl.range(constant)` 的 unroll、UB/SSA pressure、dynamic leading-dim、small-N matmul 等 compiler risk。
- ring heap、task-id pool、dependency pool、CANN/PTOAS/simpler 版本组合、driver/firmware 和 device placement。
- TP=8/EP=8、cards 8–15、0726 image、`enforce_eager`、IPC import location 等当前部署 profile。
- `pl.parallel`/`pl.spmd` 的具体 parser 限制和 codegen 形态。

这些条目必须写成：

```text
适用版本/backend/profile
已观察现象
最小复现或证据路径
若上游修复/代码变化，重新评估
```

不能写成无条件的“所有代码必须如此”。

---

## 4. Probe、contract 和 release gate 的边界

### 4.1 先有差异，再有 probe

建立 probe 前必须回答：

- DeepSeek 基线是什么？
- step3p5 与它的差异是什么？
- 差异为何无法直接沿用 DeepSeek？
- probe 验证的是哪一个适配点？
- probe 失败是否意味着语义不成立，还是当前 backend/版本不支持？

如果没有明确差异，默认直接沿用 DeepSeek，不新增 probe。

### 4.2 Probe 不能反向规定架构

以下证据不足以规定 canonical 产品结构：

- standalone minimal program compile PASS；
- 源码出现 `pl.spmd_submit`；
- `core_num=n_ranks`；
- 某种 peer-major slab shape；
- 一个 `task_dummy` 存在；
- synthetic 单 epoch/固定 active token 数值通过。

probe 可以报告当前表达能力或版本限制，但产品代码首先服从 DeepSeek 语义和 step3p5 明确差异。

### 4.3 发布门

验收应按以下层级记录：

1. source/AST contract；
2. compile；
3. lowered IR/codegen；
4. semantic invariant；
5. runtime DAG/liveness；
6. real device；
7. performance。

任何低层 PASS 都不能把 C1/C3/B3/G1 直接写成 device DONE。

当前 release-validation profile 指定在 0162 的 0726 镜像/设备中产出真实 device 结论，devbox 只能产出本地 source/contract 等级证据。未来若项目更新指定验证机或镜像，应以新的 profile 替换本条，而不是把 0162 固化为产品语义。

---

## 5. 历史经验的使用方式

本章条目状态为 **[历史/诊断经验]**。

历史事故、旧版 compiler workaround、某次 507018/507899、某个 stale artifact、某个 synthetic probe 结果，默认是**风险提示**，不是永久约束。

只有满足以下条件时，历史经验才可升级为当前约束：

1. 仍能在当前版本复现；
2. 有明确根因或稳定的语义/资源不变量；
3. 适用范围明确；
4. 有正例/负例或 lowered/device 证据；
5. 不与 DeepSeek 基线冲突。

“为让 artifact 看起来有依赖”“为绕过某个编译错误而改变数据流”“把旧文件当默认入口”“把所有 window 扩成 512B”等做法不能作为产品 workaround。

---

## 6. 修改前后检查清单

### 修改前

- [ ] 写出 DeepSeek 基线和 step3p5 明确差异。
- [ ] 标注本次规则属于语义硬约束、当前 ABI、产品 profile 还是历史经验。
- [ ] 明确保护区：canonical 入口、`tp_all_reduce`、KV ownership、signal epoch、host/window ABI。
- [ ] 判断是否真的需要 probe；没有差异就不增加 probe。
- [ ] 记录 source/compile/lowered/device 证据等级。

### 修改后

- [ ] C2/C3 已按 V4-Flash expert-lane push/gather + combine scatter/wait/reduce 数据流实现；任何 step3p5 适配均有明确差异依据。
- [ ] DeepSeek 的数据流、通信方向、route/count/map 语义和数值顺序未被无依据重构。
- [ ] runtime active batch/token贯穿attention、MoE、combine与KV写入；静态formal shape仅作为可配置capacity上界，inactive/padding不进入逻辑计算。
- [ ] alias、真实 RAW、completion/reuse 和 inactive/padding 语义闭合。
- [ ] 512B 只用于 stacked/reused notify/wait/AtomicAdd control slot。
- [ ] 没有把 probe ABI、历史 workaround 或版本 bug 写进产品语义。
- [ ] `tp_all_reduce` 保持固定 peer 顺序、FP32 accumulator、最终一次 BF16 store。
- [ ] retired 入口没有恢复为默认产品入口。
- [ ] 按当前 release-validation profile 在指定镜像/设备验证，并 graceful 释放所用设备；当前 profile 为 0162 cards 8–15。
- [ ] 遵守当前工作流要求：`git diff --check`、中文注释/commit；是否 push 以本次用户指令为准。

---

## 7. 参考资料

- DeepSeek V4-Flash 源码基线：`origin/main:models/deepseek/v4-flash/`，重点为 `moe.py`、`decode_fwd.py` 和 `config.py`；用 `git show` 只读对照，源码优先于旧设计文档和 probe。
- 工作树中的 `models/deepseek/v4/` 不是本项目指定的 V4-Flash 基线，除非任务另有明确说明，不得替代上述路径。
- step3p5 对照/适配实现：`pypto-lib/models/step3p5/moe.py`、`dispatch.py`、`combine.py`。
- canonical 产品入口：`pypto-lib/models/step3p5/decode_fwd.py`。
- PyPTO 编程与 ABI：`pypto-lib/docs/{known-pypto-pitfalls,pypto-coding-style,compile-runtime-workflow,debugging}.md`。
- step3p5 项目设计与交付：`pypto-project/design/performance/`、`deployment/`、`STATUS.md`。
- 当前版本/设备证据：0162 0726 镜像验证记录。

若参考资料与当前源代码、上游 DeepSeek 实现或当前 backend 行为冲突，先记录冲突并重新确认适用范围；不能只因为旧文档写成“不可违反”就继续沿用。
