# N1 整网随机 Stall 定位案例

> 本文是 `pypto-whole-net-hang-debug` 的实践 example。它记录 2026-07-15
> 至 2026-07-16 在 0162 上对 N=1 whole-net 随机 hang 的证据链、方向纠正、
> 最终修复和设计阶段可预防项。本文不是新的 active prompt；当前准出事实仍以
> `N1-CANONICAL-TEST.md` 和最终 handoff 为准。

## 目录

1. [案例摘要](#1-案例摘要)
2. [固定被测对象](#2-固定被测对象)
3. [症状与第一性分类](#3-症状与第一性分类)
4. [从 task 到 exact kernel](#4-从-task-到-exact-kernel)
5. [跨 rank 还原通信边界](#5-跨-rank-还原通信边界)
6. [历史方向纠正](#6-历史方向纠正)
7. [设计和代码审计](#7-设计和代码审计)
8. [最终最小 A/B](#8-最终最小-ab)
9. [完整准出验证](#9-完整准出验证)
10. [保留项与因果边界](#10-保留项与因果边界)
11. [设计阶段如何提前避免](#11-设计阶段如何提前避免)
12. [可复用排查清单](#12-可复用排查清单)

## 1. 案例摘要

问题表现为：

```text
whole_decode_faithful_real
完整 42 个 MoE layer
8 卡
真 W8A8 权重 + 真 KV IPC
有时 2~3 秒完成并得到 argmax=303
有时约 timeout 后报 507018 / S1 running-stalled
```

最终发布组合：

```text
machine = gpu-a910x-0162
devices = 8..15
program = whole_decode_faithful_real
P_FAITHFUL_MOE_LAYERS = 42
token = 6127
weights = native W8A8 IPC
KV = IPC
dispatch = fixed-slot pull
combine = pull
golden argmax = 303
```

最终最小 layout A/B：

```text
logical signal view = [8,1] INT32 = 32B
physical allocation = 512B
COMM_CONTROL_SIGNAL_BYTES = 512
216/216 signal buffers nbytes=512
all relative offsets %512=0
whole comm window size %512=0
```

候选 512B 版本先在 fresh exporter pool 上完成 canonical P42 连续 20 次；
证据审计发现该日志与整理后的 release commit source SHA 不同后，又对
`0e7a0fdd` exact-source 重新执行完整 20-run，20 次均 `argmax=303`。
这一步把“候选版本 A/B 证据”和“release 准出证据”重新分开并补齐。

结论边界：

- 在 0162 上，512B physical signal isolation 与随机 stall 消失具有强关联，
  并由 release exact-source 20/20 支持；
- 现有记录没有提供完整 matched A/B 表，不能把它升级成严格的跨机器
  “强单变量因果证明”；
- 项目后续记录称 0234 在仅确认 pypto-lib 三个 release 文件与
  `0e7a0fdd` byte-match 后 fresh canonical 3/3 stall；由于完整
  pypto/simpler/runtime binary/environment 未绑定，本次审计不能把它称为
  同一 release object 的跨机器复现；本次审计也因 SSH permission denied
  未能独立复核 0234；
- 没有 bit-level trace 证明某一个 signal bit 丢失；
- 没有真实 AICore PC 证明某条 TPUT/TGET/WAIT 指令是唯一根因；
- 不能把历史某次停留的 kernel 名当成所有失败的唯一位置。

## 2. 固定被测对象

### 2.1 为什么先冻结 canonical

历史上曾出现：

- push+push、pull+push、pull+pull 交替；
- P1、P20、P42 交替；
- 只检查 `RUN_CLEAN`；
- clean run 得到 303，但另一轮 stall；
- generator 与 active builder 不一致；
- exporter、设备残留、日志目录混用。

如果不冻结对象，任何“修好了”都可能只是换了问题。

本案例最终把以下口径焊死：

```text
真实 token 6127
BATCH=16，只有 row0 有效，row1..15 为 padding
真 native W8A8 权重，不允许 BF16-dequant fallback
真 KV IPC
完整 P42
argmax(full_logits) == 303
同一份 final source 连续 20 次
```

### 2.2 “老基线 303”到底表示什么

历史 clean run 的确多次得到 `argmax=303`。这证明：

```text
在某些时序下，数学路径、权重和主要边界可以得到正确 token
```

它不证明：

```text
该版本稳定
该通信协议无 race
某个历史补丁已经修复 stall
```

因此 303 是精度 golden，不是“概率 clean 版本”的稳定性基线。

## 3. 症状与第一性分类

### 3.1 历史跨机器症状线索（非 exact-object reproduction）

历史记录先在 0234 的 push+push 路径观察到随机 stall，之后在 0162 的后续
固定对象上复现同类症状。由于两轮协议、源码/build 和 runtime manifest
没有完整绑定，这不是严格的同一测试对象跨机器复现。
0162 的六次历史运行中：

```text
2 次 clean
4 次 stall
clean run argmax=303
```

这些记录只能排除“症状一定只在 0234 单机硬件上发生”的过早判断，并建立：

```text
精度可以正确
稳定性仍不正确
```

### 3.2 不能从 507018 直接定性

典型 host 行：

```text
PTO2 scheduler timeout sub_class=S1:running-stalled
completed=39/109
running=1 ready=0 waiting=1
orch_done=0
stuck_task_id=4294967319
stuck_core=26
```

`507018` 只是外层错误。真正有用的是：

```text
sched_error_code=100
sub_class=S1
RUNNING=1
READY=0
WAITING=1
```

因此分类为：

```text
无进展快照中至少存在一个已分配到 core 的 RUNNING task；
该 task 在采样时尚未完成。
```

不是：

```text
READY 无法派发（S3）
只有 dependency WAIT（S4）
orchestrator 不提交（S5）
```

`WAITING=1` 只证明快照中存在 WAIT task；它是否直接依赖该 RUNNING task，
必须由同轮 dependency dump 证明。

历史上调大 ring heap、task window、dep pool 没有建立修复，且没有对应容量 detector
作为直接证据，所以“容量耗尽”被降级。

### 3.3 timeout 只用于区分 slow 与 hang

曾把 scheduler/op/stream timeout 抬高约 10 倍。失败运行仍耗完整 op 预算后不完成。

该实验支持：

```text
不是多等几分钟就能完成的普通慢 kernel
```

但它不提供修复，也不告诉我们卡在哪个子操作。

### 3.4 dmesg 使用 before/after

每轮保存：

```bash
dmesg -T > dmesg.before.txt
# run
dmesg -T > dmesg.after.txt
diff -u dmesg.before.txt dmesg.after.txt
```

重点检查：

```text
devmm/page fault
illegal VA/instruction
DMA/UB fault
507018
running-stalled
stranded CQE
```

案例中的一个重要经验：

- 旧的 `devmm_ioctl_ipc_mem_query` 行在 before/after 中都存在；
- 它不是当前 run 新增证据；
- release exact-source 20-run 的 20 个逐 worker-run 窗口与 smoke worker-run
  窗口没有新增相关 fault；
- 20-run 完成后关闭 fresh exporter pool，outer 窗口新增 2 条
  `stranded cqe`，发生在 dev8/dev11 exporter teardown，不在任何 worker-run
  窗口内。

只 grep 全局 dmesg 会把旧错误误归因给当前运行。

## 4. 从 task 到 exact kernel

### 4.1 第一个关键纠错：task id 不是 func id

典型：

```text
stuck_task_id=4294967319
```

解码：

```text
4294967319 = 0x100000017
ring_id = 1
local_task_id = 23
```

早期曾把 local task 23 直接映射成 kernel 23。这是错误的。

历史 handoff 曾记录同轮 device `TASK`：

```text
TASK ... task_id=4294967319 state=RUNNING ...
kernels=[aic:-1 aiv0:28 aiv1:-1]
```

真正需要查的是 `func_id=28`。但本次 Skill 审计没有重新取得该次失败 run 的
完整 TASK source、exact build hash 与 config hash，所以该映射在本文只作为
历史定位线索；新案例必须按 run/rank/snapshot 表格重新绑定，不能直接复用。

### 4.2 第二个关键纠错：不能按完成比例猜 kernel

早期一次 build 被清理后，只看到：

```text
completed=78/81
```

曾据此猜“已经接近末尾，所以 func36 应是 combine push”。后续保留 exact build，
读取：

```text
build_output/<exact-build>/next_levels/full_moe_chip_orch/kernel_config.py
```

历史记录给出的映射为：

```text
func 28 = _dispatch_pull
func 29 = _dispatch_stage
func 36 = _stage_routed_src
func 37 = _pull_routed_y
func 38 = moe_combine / weighted gather
func 39 = moe_residual_add
```

这推翻了按“位置接近末尾”做的映射。由于本次审计未重新绑定到原始 exact
build，后续 agent 应在新失败 run 中重新读取同轮 `kernel_config.py`，不要把
此表视为当前 0234 stall 的已验证映射。

### 4.3 从 kernel_config 继续到生成源码

映射链：

```text
TASK kernels=[aiv0:37]
-> kernel_config.py func_id 37
-> _pull_routed_y
-> kernels/aiv/_pull_routed_y.cpp
-> orchestration/full_moe_chip_orch.cpp 中对应 task
-> dependency dump 中 producer/consumer
```

保留的生成物包括：

```text
orchestration/host_orch.py
next_levels/<orch>/kernel_config.py
next_levels/<orch>/orchestration/<orch>.cpp
next_levels/<orch>/kernels/{aic,aiv}/*.cpp
passes_dump/36_after_AutoDeriveTaskDependencies.py
report/memory_after_AllocateMemoryAddr.txt
report/perf_hints.log
```

### 4.4 本案例的 PC 边界

普通 scheduler stall 日志可能提供：

```text
task / kernel / core
COND=ack 或 fin
```

但保存的 N1 失败日志没有可用的：

```text
AICore current_pc
error_pc
kernel_start_pc
```

保存的 N1 失败材料足以支持的方法边界是：

```text
exact kernel + 跨 rank 通信边界
```

没有声称定位到某条机器指令。

如果当时存在真实 PC，正确下一步应为：

```text
pc_offset = error_pc - kernel_start_pc
```

再用同一 binary 的 CCE/PTOAS map 定位，并检查该指令之前的 publish/notify/wait
配对操作。没有 PC 时只能通过 phase marker 或 kernel split 继续二分。

## 5. 跨 rank 还原通信边界

### 5.1 不同 rank 停在不同深度

历史 handoff 曾报告：

```text
rank 8–14: _pull_routed_y
rank 15:   _dispatch_pull
```

其他 run 中也报告过 28/29/37/38/39 的组合。本次审计没有重新取得逐 rank
原始 TASK 行和 exact build 绑定，所以这些记录用于说明“跨 rank 反推最早
边界”的方法，不作为当前 0234 stall 的固定挂点。

这说明：

- 不能写成“所有 rank 都固定挂在 routed_h_quant”；
- 下游 combine/residual 可能只是处于更深的流水位置；
- 应寻找最早未完成的跨 rank generation，而不是多数 rank 的最后一个函数。

### 5.2 dispatch 边界审计

最终边界：

```text
pack_publish
-> dispatch_pull
-> dispatch_stage
```

审计内容：

- source 按固定 slot 发布 payload；
- receiver 拉完整 count snapshot；
- receiver 生成 `recv_counts` 和 local expert offset/count；
- 同一 dispatch 边界生成 source-local `inverse_map`；
- self payload 用 local load；
- peer payload 用 remote load；
- `count_done` / `data_done` generation、初始化和 expected 一致。

### 5.3 combine 边界审计

最终边界：

```text
stage_routed_src
-> pull_routed_y(inverse_map)
-> weighted_gather
-> residual
```

关键约束：

- combine 直接消费 dispatch 产生的 `inverse_map`；
- 不在 combine 重新读取 distributed count matrix 构建另一份映射；
- self routed row 使用 local load；
- peer routed row 使用 remote load；
- 每层拥有 distinct routed/signal buffer。

### 5.4 为什么要看“上一对操作”

若快照停在 wait 或 remote load，真正错误可能在前一对 producer 操作：

```text
producer 没写完 payload
producer fence 未覆盖对应 pipe
producer notify generation 错
consumer expected 继承旧值
consumer offset 来自另一份 count snapshot
```

因此 kernel 名只是入口，边界上的 producer/consumer 配对才是检查单元。

## 6. 历史方向纠正

### 6.1 push/TPUT 不是最终唯一根因结论

历史上 push+push 的挂率较高，dispatch pull 一度降低挂率，因此
“跨 die push 脆弱”有实验线索。但后续：

- pull+pull 也会 stall；
- 停留位置跨 dispatch/combine 漂移；
- 最终最小 A/B 是 signal physical isolation；
- 没有 PC 或 bit-level trace 证明某个 TPUT 唯一失败。

所以最终文档只保留：

```text
push/pull 改造是边界设计和实验过程的一部分；
不能写成 PUSH/TPUT 已被硬件层证明为唯一根因。
```

### 6.2 P1/P20 不能替代 P42

P1、P20 曾用于快速判断：

- 单层代码是否必现；
- 深度是否影响概率；
- 某个边界是否能跑。

但 N1 发布问题是完整 P42。中间层数可能改变：

- 通信 generation 数量；
- buffer 布局；
- allocator offset；
- 调度重叠；
- stall 概率。

最终直接使用 P42 做决定性 A/B 和 20-run release。任何 P20 “wrong/clean”
都不再承担最终结论。

### 6.3 completion-wave、串行调度和增大 timeout

这些实验能排除或分类部分假设，但都没有成为最终最小修复：

- completion-wave：可修独立协议缺陷，但不能解释所有残余随机 stall；
- serial orchestrator gate：device 仍可 stall；
- 增大 timeout：只证明非普通慢；
- retry：只能掩盖概率问题。

### 6.4 exporter/compile 失败不是 kernel stall

曾出现 checkpoint 路径、stale exporter、COMMINIT 等问题。只有到达：

```text
built args
rt.run
device scheduler snapshot
```

后，才能归入本案例的 kernel stall。阶段必须在日志中明确标记。

## 7. 设计和代码审计

### 7.1 generator 与 active builder 不一致

历史 generator 曾把：

```text
完整 count snapshot
inverse_map reconstruction
```

移动到 combine，和正在 device 验证的 active boundary 不一致。

修复要求：

```text
strip active real builder
-> run generator
-> byte compare regenerated source
```

只验证“generator 拒绝覆盖已有 builder”不算 round-trip。

这个问题不是最终 512B A/B 变量，但必须保留修复，否则下一次 regenerate
会重新引入未验证边界。

### 7.2 per-layer distinct buffer

PyPTO 当前依赖模型是 RAW-only v1，依赖 non-aliasing intermediate memory。
本案例审计每层：

```text
attn_sig_buf_Ln
count_done_buf_Ln
data_done_buf_Ln
sh_sig_buf_Ln
combine_done_buf_Ln
send/recv/routed windows
```

要求层后缀唯一，生命周期不重叠复用。该结构被保留，避免把远端尚在访问的窗口
提前复用给下一层。

### 7.3 对齐规则不能混

本案例复核四种不同约束：

```text
storage shape 512B：设计/correctness 不变量
UB Vec row 32B：静态或运行时 correctness 约束
GM<->UB tile 512B：DMA/静态约束
L2 cache line 512B：性能提示，不是 correctness 定律
```

它们作用对象和失败形式不同。

特别是 signal：

```text
logical [8,1] INT32 = 32B
```

逻辑 shape 没错，但物理上 32B signal 可能与相邻 signal/payload 共用 512B line。

### 7.4 dtype、tail、padding 和初始化

保留的正确性约束：

- routed input 在 dispatch 前动态量化为 INT8 + FP32 per-token scale；
- clamp 后中间激活进行第二次 per-token INT8 requant；
- gate/up/down 使用 native INT8 matmul；
- shared expert 保持 BF16；
- `router_bias` BF16-round、EPS=`1e-5`，使用 layer-specific swiglu clamp；
- 不回退 BF16-dequant weights；
- signed `tile_rem` 在转 index 前处理空 tail；
- BATCH=16 中 row1..15 padding 行有定义的初始化和屏蔽；
- communication window 在第一次 notify/wait 前 zero-init；
- routed/gather destination 按协议显式初始化。

这些项不都被证明是 stall 唯一根因，但违反任何一项都可能产生：

```text
错误 offset
提前 wait 通过
永久等待
NaN/越界
507018
数值污染
```

因此不能为了“最小 diff”删除。

### 7.5 片上 memory report

最终 retained build 的报告显示示例：

```text
_dispatch_pull Vec 11.1KB / 184KB
_pull_routed_y Vec 8KB / 184KB
_stage_routed_src Vec 64KB / 184KB
```

这用于排除明显 UB 容量超限，并提供 buffer address/live range。
它不能替代 GM communication window 的 offset/cache-line 审计。

## 8. 最终最小 A/B

### 8.1 A 组风险

逻辑 signal：

```text
[8,1] INT32
8 * 4 = 32B
```

如果物理 allocation 也是 32B，多个 signal 或 signal 与 payload 会紧密排列。
在 512B L2/cache-line 粒度下，它们可能共享同一物理 line。

潜在影响：

- 不同协议的 AtomicAdd/Set 热点共线；
- control plane 与 payload 读写共线；
- 某一 generation 的 wait/notify 受相邻对象影响；
- 问题随层数、allocator offset、调度时序概率出现。

这仍是候选机制，不是自动证明。

### 8.2 B 组只改物理隔离

改动：

```python
COMM_CONTROL_SIGNAL_BYTES = 512
```

保持不变：

```text
logical shape [8,1]
dtype INT32
dispatch/combine 数学
native W8A8
真实 token
P42
同步 API 语义
```

覆盖：

```text
dense prefix attention/MLP signals
每个 MoE layer 的 attn_sig
count_done
data_done
sh_sig
combine_done
```

生成物审计：

```text
CommBufferSpec = 684
signal/done = 216
216/216 nbytes = 512
all relative offsets %512 = 0
window size = 766525440
window size %512 = 0
```

注意：

```text
size=512B 不够；
(actual_window_base + offset) 也必须 %512==0。
```

本次静态审计证明了 relative offset 和 window size 的 512B 对齐；若 allocator
ABI 没有正式声明 base alignment，还需要逐 rank 记录 actual base。不能用
`window_size%512==0` 单独替代实际地址证明。

### 8.3 为什么该 A/B 有意义

这是最终收敛前最小的 layout 变量：

- 没有换模型；
- 没有缩层；
- 没有换输入；
- 没有回退 dtype；
- 没有重写数学；
- 没有靠 timeout/retry；
- 没有混入另一套 dispatch/combine 组合。

候选 512B 版本应用后，0162 历史随机 stall 收敛为 fresh pool 20/20，这是
强关联证据。发现候选 20-run 与 release smoke source SHA 不同后，继续对
release commit `0e7a0fdd` 重跑 exact-source 20/20，补齐了同一最终源码的
0162 发布证明。由于 matched 32B A 组的 source/build/run 表没有完整归档，
且 0234 项目记录仍有同 commit stall，本文不再称其为严格跨机器单变量证明。

## 9. 完整准出验证

### 9.1 最终代码

```text
repo = csy0225/pypto-lib
branch = feat/whole-net-n1-fusion
commit = 0e7a0fddc90c4f2348f1d59e015fb817a0877a02
```

pypto-lib 发布文件：

```text
models/step3p5/decode_layer.py
models/step3p5/moe.py
tools/step3p5/_gen_faithful_real.py
```

后续审计确认 canonical IPC worker 还实际依赖：

```text
pypto n1fusion-base e277de9f
simpler n1fusion-base 36957c6b
```

其中 pypto 提供 stacked weight per-layer sub-view 和
`DistributedWorker.import_ipc_all`，simpler 在 forked chip child ACL context
中执行 IPC import。只拉 pypto-lib 不构成完整复现。

历史 exact-source 20-run 的模型文件来自 `pypto-lib 0e7a0fdd`，但当时
pypto/simpler 的相关 runtime 支持仍是旧 HEAD 上的 dirty source：

```text
pypto HEAD 5e619dc7 + dirty runtime support
simpler HEAD 98ce22a6 + dirty child IPC import
```

因此不能声称 20-run 直接运行在新的 clean pins 上。`e277de9f/36957c6b`
formalize 了相关运行支持，并由下面的 clean-pin smoke 单独验证；该 smoke
不把新提交追溯成旧 20-run 的 byte-identical runtime。

### 9.2 release exact-source 20-run

日志：

```text
/data/chensiyu/hw_project/pypto/workspace/logs_n1/signal512/
signal512_p42_20_20260717_001135
```

结果：

```text
pass=20/20
each rc=0
each argmax=303
TOP5=[303, 9592, 768, 1043, 410]
runtime min/mean/max=2.50/2.5605/2.62s
```

20 次数值指纹一致：

```text
max|next_hidden|=264192.0000
row0|next_hidden|=588.0000
max|h_mid|=294.0000
max|logits|=14.0506
```

源码 SHA 与 release commit 一致：

```text
decode_layer.py          9b6c83ca915ca9fcb5b02223e1a733c1c28fabca45dec6019b3b41a5f3fd7d5d
moe.py                   8a3670a047aff5b5af5d352446d8a35c866708f0eccba2b70904ad18896d5a2a
_gen_faithful_real.py    bf65295b2167bd96516e8ef2cebd97b69ebc7d46a86e13d304180ebf6a514010
```

### 9.3 整理后 smoke

```text
signal512_final_smoke_20260716_230225
release commit = 0e7a0fddc90c4f2348f1d59e015fb817a0877a02
source SHA（日志中的三文件）：
decode_layer.py  9b6c83ca...
moe.py           8a3670a04...
_gen_faithful_real.py  bf65295b...
CANONICAL_RC=0
RUN done 2.57s
argmax=303
FINAL_SMOKE=PASS
```

20 个逐 worker-run 窗口和 smoke worker-run 窗口均无新增：

```text
devmm/page fault
illegal VA/instruction
DMA/UB fault
507018
running-stalled
stranded CQE
```

20-run 的 outer dmesg 窗口包含 exporter pool teardown；该阶段新增 2 条
`stranded cqe`。它们不在任一 worker-run 窗口内，所以不能归因于 whole-net
worker kernel，也不能被删除或笼统写成“整个生命周期 dmesg 完全 clean”。

### 9.4 最终三仓 clean-pin smoke

```text
log =
  /data/chensiyu/hw_project/pypto/workspace/logs_n1/release_manifest/
  final_stack_smoke_20260717_015635

pypto-lib = 0e7a0fddc90c4f2348f1d59e015fb817a0877a02
pypto     = e277de9f2a55a686956d66933301204520bd7374
simpler   = 36957c6b56700ecba3aeb8dbbedd6240594e01de
```

目录时间戳来自 0162 机器时钟。结果：

```text
P42 / pull+pull / token 6127 / native W8A8 IPC / KV IPC
rc=0
RUN done 2.58s
argmax=303
TOP5=[303, 9592, 768, 1043, 410]
worker-window added relevant dmesg=0
```

outer 窗口在 exporter teardown 后新增 1 条 dev14 `stranded cqe`，不在
worker 执行窗口内。相关 runtime focused tests 为 `127 passed`。

runtime binary SHA256：

```text
libhost_runtime.so
  7b29004b9d047d550ee6689120be83e650a3bcf39b196fd0ea112a3c6271891a
libaicpu_kernel.so
  62b8c2430abc9cafe257b758148c22fc1ab6da1085b0a103ae7bc465c57ca390
libsimpler_aicpu_dispatcher.so
  1b4b8467f0c899af64ebcd2f0a98e83b89160dca32177d0baecebddd3be4f973
_task_interface.cpython-311-x86_64-linux-gnu.so
  318510dfc2a55b27749609fd56850657b77691bc4078d6a7064f6451076f2c53
```

## 10. 保留项与因果边界

### 10.1 应保留的结构性修改

以下修改继续保留：

- fixed-slot pull dispatch；
- dispatch 边界生成 `recv_counts + inverse_map`；
- combine 直接消费该 `inverse_map`；
- self local load、peer remote load；
- per-layer distinct communication buffers；
- signal whole-window zero-init；
- signed tail；
- native INT8 gate/up/down；
- generator 真实 round-trip；
- control signals 512B aligned physical isolation。

原因分三类：

1. **0162 上与 stall 消失强关联的变量**：512B signal isolation；
2. **架构/边界正确性**：inverse_map 所属边界、self/peer 路径、buffer lifetime；
3. **精度/可生成性正确性**：native W8A8、signed tail、generator 一致。

不能把第 2/3 类都包装成“stall root cause”，但也不能删除它们。

### 10.2 可以写和不能写

可以写：

> 在 0162 上，512B signal isolation 是最终最小 layout 变量，并与随机 stall
> 消失强关联；release commit `0e7a0fdd` 已完成 exact-source 20/20。
> worker-run dmesg 窗口与 exporter teardown outer 窗口分别归档。
> 现有材料未证明其为跨机器充分条件或唯一根因。

不能写：

> 已在硬件层证明某个具体 signal bit 丢失。

不能写：

> PUSH/TPUT 或某一个 PC 指令是唯一根因。

不能写：

> 因为某次 rank 停在 `_pull_routed_y`，所有失败都固定发生在那里。

## 11. 设计阶段如何提前避免

### 11.1 在设计评审中提交 buffer ledger

每个 layer/collective 在编码前写清：

```text
logical shape / valid_shape / dtype
physical bytes / alignment
producer / consumer
self/peer access
notify/wait generation
initial value
first use / last use
是否与相邻 payload 共 cache line
是否跨层复用
```

如果当初明确区分：

```text
logical signal bytes != physical isolation bytes
```

32B control signal 共线风险可以在编码前暴露。

### 11.2 把 control plane 和 data plane 分开

设计上要求：

- signal 不与大 payload 共 512B line；
- 不同协议的 atomic counter 不共 line；
- 每层 signal 独立；
- generation 明确，不依赖历史残值；
- whole-window init 和 logical init 均有责任方。

### 11.3 把 layer boundary 当作接口

以 dispatch 为例，接口输出应明确：

```text
recv_x
recv_scale
recv_counts
local expert offsets
inverse_map
completion generation
```

combine 只能消费这些接口输出，不应重新读取分布式状态推导另一套映射。

### 11.4 设计时同时覆盖 batch/pad/dtype

至少列出：

```text
single active row + padded rows
multi-batch
empty tail
full tile / partial tile
self route / peer route
INT8 payload + FP32 scale
padding 行是否进入 route/reduction
```

若未定义，运行时可能表现为数值错、NaN、越界或 stall，且容易被误归因给 scheduler。

### 11.5 让生成物成为评审对象

源码正确不等于 codegen 后正确。设计 gate 应包含：

- exact `kernel_config.py`；
- orchestration task 顺序；
- dependency dump；
- memory report；
- physical window offset；
- generator round-trip；
- final kernel source 中真实 fence/notify/load/store。

## 12. 可复用排查清单

遇到新的整网 hang，按顺序回答：

1. 失败是否发生在 `rt.run` 内？
2. exact source、build、环境、输入是否冻结？
3. `507018` 下的真实 orch/sched code 是什么？
4. 属于 S1/S3/S4/S5 哪类？
5. `TASK` 中的 kernel id 是什么？
6. 是否使用同轮 exact `kernel_config.py`？
7. `COND=ack` 还是 `fin ANOMALY`？
8. 是否存在真实 AICore PC？若无，是否诚实停在 kernel 级？
9. 所有 rank 的最早阻塞边界是什么？
10. 上一个 publish/fence/notify 与当前 wait/load 是否配对？
11. control signal 的 physical isolation 是否满足平台 allocator/coherency ABI？
    N1 当前保留值为 512B；actual `(base+offset)` 是否对齐？
12. storage/UB/GM-L2 四类对齐是否分别检查？
13. buffer 是否跨层别名或提前复用？
14. dtype、tail、padding、batch、初始化是否有定义？
15. P1/P20 是否只作为诊断，而最终回到完整 canonical？
16. A/B 是否只改一个变量？
17. 最终是否同时通过稳定性、精度、dmesg delta 和 generator gate？
