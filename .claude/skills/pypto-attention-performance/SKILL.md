---
name: pypto-attention-performance
description: >
  PyPTO attention 性能分析与优化方法。用于分析 Full/SWA decode attention、
  online softmax、out-proj、task grain、SPMD logical-task 切分、AIC/AIV wave、
  GM/UB/L1 数据流、cast/算子融合以及 attention 关键路径上的 collective/Vec 开销。
  当任务要求找瓶颈、设计跨架构任务粒度、评审融合方案、执行最小改动 A/B 或判断优化
  是否收尾时使用；本 skill 只教授性能分析与优化，不负责环境安装、镜像发布或通用故障排查。
---

# PyPTO Attention 性能分析与优化

## 范围

本 skill 的目标是把 attention 优化从“看 kernel 名猜性能”变成可复核的工程流程：

```text
固定被测对象
-> 建 workload / dataflow / resource model
-> 建可信 baseline
-> 用 DFX 定位关键路径
-> 提出单变量假设
-> sweep task grain 或融合边界
-> correctness + performance 双门验证
-> 只保留有稳定收益的最小改动
```

只讨论性能分析、优化设计、实现评审和验证方法。遇到以下任务时转用相邻 skill：

- 产品语义、ownership、lifetime、PyPTO 分层约束：
  [`../pypto-dev-constraints/SKILL.md`](../pypto-dev-constraints/SKILL.md)
- 完成优化后的整网回归：
  [`../pypto-perf-regression/SKILL.md`](../pypto-perf-regression/SKILL.md)
- hang、507018、随机 stall、跨 rank 死锁：
  [`../pypto-whole-net-hang-debug/SKILL.md`](../pypto-whole-net-hang-debug/SKILL.md)

Step3p5/0162 的实例、实测和负面结果见
[reference](references/step3p5-attention-lessons.md)。只有处理该模型或复用本轮经验时
才加载；不要把其中的 A2A3 参数当成跨架构常量。

## 不可妥协的原则

1. **先冻结对象，再比较性能。** source、依赖 pin、设备、输入、batch、context、
   oracle、采集方式不同，不得写成同一组 A/B。
2. **先证明瓶颈，再改代码。** kernel 数多、span 长、核利用率低都只是线索，
   不是单独的根因证明。
3. **logical task 不等于物理核心。** 模型代码描述工作量；runtime 将 logical tasks
   映射到物理资源和 wave。不要把 24、48 或其它 core count 写进模型语义。
4. **5–10 us/task 只是 sweep 起点。** 最优点由 stage span、wave、packing、tail、
   dispatch、依赖尾部和端到端延迟共同决定。
5. **Full 与 SWA 分开建模。** 长 context 的 Full 可以需要 context split 和层次归约；
   很短的 SWA window 常常更适合每 active row 一个高密度 task。
6. **少 kernel 不等于更快。** 融合必须同时改善数据移动或关键路径，并保持合法 tile、
   资源映射、数值顺序、ownership 和 batch 边界。
7. **性能结论必须有适用范围。** 写清 machine、architecture profile、workload 和
   confidence；单机结果不能升级为硬件定律。

## 工作流

### 1. 定义问题与停止条件

先把问题改写成可测目标：

```text
目标 workload:
  decode/prefill, Full/SWA, batch, context distribution, active rows

目标指标:
  ITL / device makespan / target stage span / memory peak / throughput

约束:
  precision, TP consistency, memory, legal tile, task-count limit, minimal diff

停止条件:
  候选均无稳定收益，或 attention 已不在关键路径，或收益小于噪声/维护成本
```

不要用“把 attention 优化到极致”替代明确的 workload 和指标。

### 2. 冻结被测对象

每轮记录：

```text
repo / branch / commit / dirty diff
pypto / runtime / pto-isa / PTOAS / compiler pins
machine / device ids / driver / firmware / CANN
checkpoint / input / seed / active batch / per-row context
compile flags / architecture profile / environment overrides
oracle / precision threshold
warmup / measured iterations / DFX mode / artifact path
```

禁止混用：

- compile-only 与真实 device run；
- standalone probe 与 canonical whole-net；
- cold DFX 与 warm ITL；
- host wall-clock 与 device makespan；
- 历史镜像与当前源码挂载；
- 一次偶然 PASS 与预定义轮次稳定 PASS。

### 3. 建立 workload、数据流和资源模型

先画完整链路：

```text
Q/K/V producer
-> QK
-> masking / softmax
-> PV / online recurrence
-> cross-task reduction
-> normalize / cast
-> out-proj
-> residual / collective / next consumer
```

对每一阶段写清：

| 项 | 必答问题 |
|---|---|
| 逻辑工作 | 沿 batch、head、context block、N tile 还是 reduction partial 切分？ |
| 有效范围 | active rows 和每行真实 context 如何决定工作量？ |
| 存储 | capacity、padding、scratch、GM traffic 和 lifetime 是什么？ |
| 计算 | cube/vector 合法 tile、boxed/fractal 限制、UB/L1 预算是什么？ |
| 资源 | 使用 AIC、AIV 还是 mixed task？每类物理资源有多少？ |
| 依赖 | 哪些边是必要 RAW/liveness 边界，哪些只是历史拆分？ |
| 数值 | FP32/BF16 cast、reduction order、最终 store 在哪里发生？ |

统一使用 workload-derived task 公式：

```text
logical_tasks(row, stage)
  = ceil(actual_work(row, stage) / architecture_profile_grain(stage))

total_tasks(stage)
  = sum(logical_tasks(row, stage) for row in active_rows)
```

`BATCH` 等静态维通常是 storage capacity，不应自动变成永久 logical workload。

### 4. 建立两层测量面

优先同时保留：

1. **faithful focused harness**：只保留目标 attention 层及必要前后继，源码通过
   `inline` 或同一函数复用，避免 probe 漂移；用于快速 compile、DFX 和参数 sweep。
2. **canonical whole-net**：用于确认真实依赖、精度、跨 rank 行为和端到端收益。

focused harness 必须与 canonical 对齐：

- 同一 kernel body、config 和 runtime scalar；
- 同一 active-batch/context 语义；
- 同一 collective 协议；
- 对关键 helper 建 source/AST contract，防止两条路径悄悄分叉。

### 5. 正确读取 DFX

至少同时观察：

```text
host wall-clock
device makespan
target stage span
per-task p50/p99
logical task count
AIC/AIV wave count
peak/average concurrency
packing efficiency
dependency tail
all-rank long tail
GM/UB/L1 与 perf hints
```

DFX 纪律：

- 先 warmup，再采 measured iteration；
- dependency generation 与 swimlane 分开采；
- collective 内 spin wait 可能被记为 kernel compute；
- 最短 makespan rank 只能作为 `LOW-WAIT REFERENCE heuristic`；
- 仍需交叉检查所有 rank，不能把一个 rank 的 span 直接解释为算术时间；
- DFX 插桩可能放大短 kernel，不用插桩占比直接推断真实 ITL 占比。

### 6. 形成可证伪的性能假设

一个合格候选应写成：

```text
现象:
  哪个 stage / dependency / memory round-trip 在关键路径上？

机制:
  为什么当前 mapping、tile、wave 或数据流造成该成本？

最小改动:
  只改变哪个变量？

预期信号:
  task count、wave、stage span、GM traffic 或 critical-path task 应如何变化？

失败信号:
  哪个结果会否定该假设？
```

优先级通常为：

1. 消除错误的串行轴或负载不均；
2. 校准 logical task grain；
3. 消除确认存在的 GM round-trip；
4. 缩短层次归约的依赖尾部；
5. 最后才考虑扩大 mixed-kernel 或通算融合边界。

### 7. 校准 task grain

按架构 profile sweep，而不是选择一个“看起来合理”的常量。

每个候选至少记录：

```text
grain
logical tasks
resource type
waves
task p50/p99
stage span
downstream reduce/finalize tail
wall p50
LOW-WAIT reference makespan
batch16 / heterogeneous-context result
```

搜索目标可写成：

```text
minimize:
  target_stage_span
  + extra_wave_cost
  + core_wait_and_packing_cost
  + dispatch_cost
  + reduction_finalize_tail

subject to:
  legal tile and UB/L1 budget
  logical-task counter limit
  active-row and tail correctness
  finite + TP consistency
  canonical precision gate
```

执行原则：

- 先做粗扫，再围绕候选做至少 3 轮交替 A/B；
- 以 median 和置信区间判断，不挑单次最低值；
- AIC 与 AIV 分开计算 wave；
- 同时覆盖短/长 context、batch1/batch16、均匀/异构 context；
- 若差异落入噪声，不增加 workload-specific 分支。

### 8. 评审层次归约

对于跨 context task 的 online softmax，先区分：

```text
segment-local recurrence
-> write-disjoint group reduction
-> per-row final merge / normalize / store
```

可以融合的通常是同一 task 私有的 segment-local 工作。跨 task reduction 只有在
ownership 明确且无 concurrent writer 时才能继续融合。

不要为了删除 `reduce/finalize`：

- 让多个 SV task 写同一个 row；
- 退化成单 task 串行消费整行；
- 依赖 fake dependency；
- 改变 `(m,l,o)` reduction order 却不重新做精度验证。

### 9. 评审 cast 与算子融合

按以下顺序评审：

1. standalone kernel 是否真的在关键路径；
2. 是否存在可删除的 GM materialize/readback；
3. producer accumulator 与 consumer dtype 是否允许同 task epilogue；
4. 融合后是否仍保持 write-disjoint tile；
5. mixed AIC/AIV 是否改变 split、wave、UB/L1；
6. batch16 是否仍放得下；
7. whole-net 是否有稳定收益。

融合准入表：

| 结果 | 决策 |
|---|---|
| 正确、减少 GM、关键路径稳定下降、资源映射不退化 | 合入 |
| 正确但只减少 kernel 数，性能在噪声内 | 默认不合入 |
| focused probe 变快但 whole-net 不变/变慢 | 保留 probe，不合产品 |
| 改变 ownership、lifetime 或 reduction order | 先建立新协议/数值合同 |
| 需要破坏合法 split 或显著增加 mixed-kernel 风险 | 除非收益足够大，否则拒绝 |

### 10. 评审 collective 邻接优化

先优化 collective 自身，再考虑通算融合。画出：

```text
producer
-> payload publication
-> notify
-> wait
-> remote/local consumer
-> result publication
-> final consumer
-> buffer reuse
```

保持独立合同：

- rank/chunk ownership；
- peer 顺序与 reduction order；
- accumulator dtype 和最终 cast；
- data publication 与 control publication；
- 最后一个 semantic consumer 到 reuse 的 lifetime；
- communication chunk 与 Vec epilogue grain 不必相同。

如果 fused epilogue 继承通信 chunk 后降低 Vec 并行度，即使少一个 copy 也可能更慢。

### 11. 最小改动实现

优先：

- 将 grain 放入 architecture profile，而不是模型语义；
- 复用现有 orchestration DAG 和 InCore kernel；
- 只改变一个 stage、一个 ownership 或一个 epilogue；
- 保留可回退开关用于新架构校准；
- 对 task mapping、generated symbols、consumer lineage 建 contract。

避免：

- app-side persistent worker / work-stealing，除非已有数据证明必须改 runtime ABI；
- worker 内递归 submit；
- 为性能 probe 改变产品数学语义却不标注；
- 同一 patch 同时改 grain、fusion、collective 和 harness。

### 12. 验证阶梯

按顺序执行：

```text
1. source / AST contract
2. py_compile / lint / diff check
3. compile-only
4. lowered IR / generated symbol / memory audit
5. faithful focused probe
6. canonical real-device smoke
7. multi-step precision
8. repeated stability
9. active-batch / heterogeneous-context / MTP
10. immutable-object audit
11. DFX + performance comparison
```

精度至少分开报告：

```text
token alignment
hidden finite
hidden TP spread
replacement equivalence
```

token gate 通过但 TP spread 非零，不能发布性能结论；性能变快但 correctness 失败，
候选同样失败。

## 快速决策矩阵

| 观测 | 优先假设 | 下一步 |
|---|---|---|
| task 很短且进入多 wave | grain 太小/dispatch 过多 | 增大 grain，比较 stage span 与 tail |
| task 较长但单 wave 尾部明显 | grain 太大/负载不均 | 减小 grain或按实际 workload 重排 |
| Full 长 context 慢、SWA 正常 | context 轴未并行或归约尾长 | Full 专用 context split/层次归约 |
| SWA 只有数微秒 | 过度拆分风险 | 保持 row-oriented 高密度 task |
| standalone cast 明显 | GM round-trip 候选 | 尝试 producer epilogue，审计 mixed resource |
| kernel 数减少但 wall 不动 | 非关键路径或融合开销抵消 | 回退，不以图更短作为收益 |
| all-reduce span 跨 rank 差异巨大 | 可能是 spin wait/到达抖动 | 全 rank 对比并单独做 collective probe |
| batch1 好、batch16 退化 | task 爆炸、memory 或 packing 问题 | 重新看 active-row 总 task 与容量 |

## 结果记录模板

```text
问题与 workload:
冻结对象:
baseline:
关键路径证据:
性能假设:
最小改动:
task/resource/wave 变化:
correctness:
performance A/B:
负面结果:
适用范围:
保留/回退决定:
下一候选或收尾理由:
```

## 常见反模式

- 固定 24 核或把“任务数接近核心数”写成算法合同；
- 把 AIC 与 AIV 合并成同质“总核数”；
- 强制每个 task 落在 5–10 us；
- 只看 kernel duration，不看 stage span 和依赖尾；
- 只看一个 LOW-WAIT rank；
- 用单轮最低值选默认参数；
- 用 kernel 数量判断融合成功；
- 把 standalone PASS 写成 whole-net PASS；
- 用 stale oracle 或不同镜像解释当前 A/B；
- 无限重跑直到偶然过线；
- 把当前 A2A3 profile 外推到其它架构。
