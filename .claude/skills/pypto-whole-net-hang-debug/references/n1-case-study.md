# N1 整网随机 Stall 定位案例

> 本文是 `pypto-whole-net-hang-debug` 的实践 example。它以 2026-07-10
> 至 2026-07-17 的 N=1 whole-net 主线为核心，并向前追溯到 2026-06-15
> 至 2026-07-09 的多卡环境、MoE、co-tenancy、dispatch 和 W8A8 前史。
> 目标不是只记录最终 512B 修改，而是解释：为什么外层长期都表现为
> `507018`，实际却是多个不同 blocker 依次叠加；为什么一些“已经修好”
> 的结论随后又被更强的 device/canonical 验证推翻；以及最终如何把
> task、kernel、跨 rank 边界、buffer layout、精度和 release manifest
> 串成可复核证据链。
>
> 本文不是新的 active prompt。当前准出事实仍以 `N1-CANONICAL-TEST.md`、
> `develop/N1/N1-STABLE-ENV-0162-20260717.md` 和最终 handoff 为准。

## 目录

1. [案例摘要](#1-案例摘要)
2. [证据标签与案例边界](#2-证据标签与案例边界)
3. [为什么这个问题持续很久](#3-为什么这个问题持续很久)
4. [完整时间线](#4-完整时间线)
5. [固定被测对象](#5-固定被测对象)
6. [症状与第一性分类](#6-症状与第一性分类)
7. [从 task 到 exact kernel](#7-从-task-到-exact-kernel)
8. [跨 rank 还原通信边界](#8-跨-rank-还原通信边界)
9. [历史方向纠正](#9-历史方向纠正)
10. [设计和代码审计](#10-设计和代码审计)
11. [最终最小 A/B](#11-最终最小-ab)
12. [完整准出验证](#12-完整准出验证)
13. [保留项与因果边界](#13-保留项与因果边界)
14. [设计阶段如何提前避免](#14-设计阶段如何提前避免)
15. [可复用排查清单](#15-可复用排查清单)
16. [原始证据索引](#16-原始证据索引)

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

## 2. 证据标签与案例边界

### 2.1 本文如何标注历史结论

这次事件跨越多个 session，历史文档中存在“当时合理、后来被更强证据推翻”的
判断。本文使用以下标签，避免把过程性判断重新传播成事实：

| 标签 | 含义 |
|---|---|
| **[直接证实]** | 同轮 device 日志、exact build、寄存器/TASK、数值输出或发布测试直接支持 |
| **[强关联]** | 单变量或近似单变量 A/B 与结果高度相关，但仍缺 matched 对照、PC 或跨机器充分性证明 |
| **[当时假设]** | 当时用于决定下一实验的候选解释，尚未完成排他证明 |
| **[后续证伪]** | 后续 device、exact mapping、控制实验或稳定性复验推翻 |
| **[历史记录，未独立复核]** | 来自 Git 历史或旧机器记录，本次文档重构没有重新运行原始对象 |

历史文档中的“root cause”“SOLVED”“device-verified”不能脱离它当时的
测试对象和验证 bar 阅读。例如：

```text
P1 clean
```

只能说明 P1 对象 clean，不能自动升级成 P42 clean；同理：

```text
argmax=303 once
```

只能说明一次精度路径正确，不能升级成稳定 release。

### 2.2 本文回顾的不是一个根因

长期事件的共同外观是 `507018`、timeout、hang 或“运行不前进”，但至少经历了
以下不同故障族：

```text
环境和 stale runtime / pyc
跨卡 IPC capability / SDMA workspace
host -> AICPU dispatch TaskMapSize=0
跨层 communication-window alias
host RAM / device arena OOM
gate_topk mrgsort 状态机不终止
pl.Out / generator / layer-index 边界错误
native W8A8 routed expert 数值错误
概率性 publish/fence/notify/wait stall
最终 control-signal physical layout 风险
```

因此本文最重要的边界是：

> **相同外层错误码不等于相同故障；相同“看起来卡住”也不等于同一 kernel、
> 同一通信原语或同一层。**

### 2.3 六个不能互相替代的验证 bar

本案例反复被推翻，核心原因之一是把不同强度的 bar 混写成同一个“通过”：

```text
compile clean
    != prepare clean
    != dispatch clean
    != rt.run clean
    != numerical correct
    != repeated canonical release
```

对应到 N1：

```text
P1/P20 clean != P42 release
RUN_CLEAN != argmax 303
一次 303 != 20/20
model source exact != runtime manifest exact
clean-pin smoke != clean-pin 20-run
```

### 2.4 机器和对象作用域

历史过程先后使用 0162 和 0234，也曾在 live vLLM co-tenancy、standalone
dummy-weight、standalone real-weight IPC 等不同对象上运行。本文只把
2026-07-16 至 2026-07-17 的下列对象称为 release-qualified：

```text
machine   = gpu-a910x-0162
devices   = 8..15
program   = whole_decode_faithful_real
P42       = 42 MoE layers
dispatch  = fixed-slot pull
combine   = pull
weights   = native W8A8 IPC
KV        = IPC
token     = 6127
golden    = argmax 303
batch     = 16, row0 valid, row1..15 padding
```

0234 的历史结果、push-mode 结果、P1/P20 结果和 live vLLM 结果都是排障证据，
不是这个 release object 的同义词。

## 3. 为什么这个问题持续很久

### 3.1 多个 blocker 共用同一个外层错误

`507018` 在这个项目中曾表示：

- CANN/AICPU bootstrap 缺库；
- stale runtime `.so`；
- host 到 AICPU 没有真正派发 task；
- AICore kernel 已 RUNNING 但不完成；
- 越界、未初始化或错误状态机引发的 kernel hang；
- timeout 包裹的通信等待；
- cleanup/恢复阶段的次生错误。

如果只按错误码搜索，很容易把一个阶段的修复错误套到另一个阶段。
正确做法是先记录失败阶段：

```text
exporter
compile
prepare
IPC import
dispatch
rt.run
worker teardown
```

再读取对应阶段的 TASK/CLUSTER/COND、kernel 和 dmesg。

### 3.2 每移除一个 blocker，下一层才暴露

这次长期过程具有明显的“洋葱结构”：

1. 编译没通过时，看不到 device dispatch；
2. device runtime 不一致时，看不到 whole-net scheduler；
3. shared comm window alias 时，看不到 full P42 real-weight 路径；
4. host/device OOM 时，看不到首个真实 MoE kernel；
5. `gate_topk` deterministic hang 修复后，才看到精度/NaN；
6. 精度路径修到 `argmax=303` 后，才有资格讨论随机 stall；
7. 随机 stall 消失后，才发现候选 20-run 与 release source SHA 不完全一致；
8. exact-model-source 20/20 补齐后，才进一步 formalize 三仓 clean pins。

因此“问题持续很久”不意味着团队一直在重复修同一行代码；更多时候是旧 blocker
挡住了后面的真实故障。

### 3.3 诊断对象持续漂移

历史上先后出现：

```text
单卡 / 双卡 / 8 卡
dummy 权重 / BF16-dequant / native W8A8
H2D 权重 / IPC 权重 / KV IPC
P0 / P1 / P20 / P31 / P41 / P42
push+push / pull+push / pull+pull
0234 / 0162
不同 generator 输出 / 不同 dirty runtime
```

如果没有每轮保存 manifest，一次 clean 和下一次 stall 可能根本不是同一对象。
这也是后来必须建立 `N1-CANONICAL-TEST.md` 和 stable environment 文档的原因。

### 3.4 诊断工具本身也曾制造错误结论

本案例出现过三类典型诊断污染：

1. **stale `__pycache__`**：源码已改，运行却仍读取旧配置；
2. **generator substring 截断**：调试生成器命中错误的 `return`，把 MoE body
   截断，导致一批 bisect 实际运行的是坏程序；
3. **orchestration 内 early-return 无法裁掉后续 DAG**：以为只跑到某 stage，
   实际后续 InCore kernel 仍 materialize。

所以任何诊断旋钮都必须通过 exact generated source 和 build artifact 证明它
真的改变了被测路径。

### 3.5 概率问题会制造“过早胜利”

最终随机 stall 的 clean 概率一度约为三分之一。这样的概率足以制造：

- 连续 3 次 clean；
- logging 打开后 clean；
- 某个 patch 后暂时 clean；
- 换机器后偶发 clean；
- 某个错误修改仍得到 `argmax=303`。

如果验证只做 1 到 3 次，极易把时序扰动误判为修复。2026-07-14 的
completion-wave“已修”结论随后被 clean tree 的 `STALL/CLEAN/STALL` 复验推翻，
正是这个问题。

### 3.6 精度和 forward progress 是两个独立 gate

历史 clean run 能得到 `argmax=303`，说明正确数学路径存在；它不说明通信稳定。
反过来，dummy 权重或错误 gate 可能让程序稳定完成，却完全没有证明真实精度。

最终要求始终是：

```text
stability == PASS
AND
argmax == 303
AND
native W8A8
AND
full P42
AND
exact source/manifest
```

任何通过回退 BF16-dequant、缩到 P1、使用 dummy gate 或仅检查 rc=0 得到的“稳定”
都不构成解决。

## 4. 完整时间线

### 4.1 一页总览

| 日期 | 被测对象 | 主要现象 | 当时判断/动作 | 后续结论 | 是否保留 |
|---|---|---|---|---|---|
| 06-15～06-24 | 单卡/多卡基础 | 507899、Bootstrap 507018、MoE runtime 507018 | 先补 driver/firmware/CANN；清 stale pyc；对空尾做 dispatch-cut | 环境问题、空 tail 逻辑和后续 gate stall 是不同故障 | 保留环境/shape 检查 |
| 07-04～07-05 | routed kernel + live co-tenancy | 独立 worker 与 vLLM 同卡 507018；16GB arena OOM | 调 ring/HBM、换进程组织 | 属 live co-tenancy 前史，不是 standalone P42 根因 | 保留为分类案例 |
| 07-06～07-07 | EpTpMoE 单块 | `gate_topk` deterministic RUNNING hang | V0 TASK→func 映射，修 mrgsort | 真实状态机 bug，后在 N1 内联副本再次出现 | 修复保留 |
| 07-08～07-09 | multi-program/融合探索 | co-prepare、TaskMapSize=0、compile/device 混淆 | ring sizing、distinct-program sweep、底座升级 | 是架构探索和底座漂移前史；N1 最终只允许单 program | 经验保留 |
| 07-10 | N1 45 层 | compile clean 后 507899/507018；随后 K=2 stall | clean runtime + SDMA OFF；dispatch-cut | stale runtime/SDMA 解环境；per-layer comm window 解 deterministic alias | 两类修复保留 |
| 07-11 | real-weight IPC | host OOM、arena OOM、S1 task3 | 建 exporter/import_ipc、改 slicing、降 arena | IPC/VA/a2a 归因后来被 exact TASK 推翻 | runtime 支持保留 |
| 07-12 | P42 native W8A8 | `task_id=3`, `aiv0:3` RUNNING | exact build 映射到 `gate_topk` | mrgsort format2 前置条件错误 | 修复保留 |
| 07-12～07-13 | 精度/NaN | NaN、1e11 幅值、错误 argmax | 层数 bisect、Out 修复、FUSE、op dump | 多个边界 bug；最终 routed expert 漂移由可靠 dump 定位 | 结构修复保留 |
| 07-13～07-14 | P42 精度 + stall | P31 clean/P42 stall；首次 303 | 猜 pool bytes、L2 index、completion wave | pool/alias等多次被推翻；L2 index 是真精度 bug；stall 未关闭 | 分类保留 |
| 07-14 晚 | clean tree 复验 | STALL/CLEAN(303)/STALL | 推翻“A2 已修” | 概率通过不能宣称完成 | 作为核心教训 |
| 07-15 | push/pull 深挖 | S1 RUNNING func28；pull 也会 stall | exact TASK→kernel、超时 A/B、push/pull 重写 | 某轮 kernel 位置不是统一根因；协议/layout 需整体审计 | fixed-slot pull结构保留 |
| 07-15 | 0162 复现 | push+push 6 次 2 clean/4 stall | 排除“只在 0234” | 不是严格 exact-manifest 跨机 A/B | 历史线索 |
| 07-16 | final layout A/B | pull+pull 仍随机停在不同 kernel | 32B signal physical allocation→512B | 0162 fresh pool 收敛；最小 layout 变量 | 512B 保留 |
| 07-17 | release 审计 | 候选 20-run 与 release source SHA 不同 | exact-source 重跑 20/20；三仓 clean-pin smoke | 补齐模型源码 release 证据，区分 old dirty runtime 与 clean pins | release 事实 |

### 4.2 2026-06-15～06-24：先学会区分环境失败、shape bug 和 kernel stall

#### 2026-06-15：单卡 e2e 能跑，但仍有临时语义

**[历史记录，未独立复核]** Phase 15 单卡 e2e 通过依赖：

- head gate 暂时 identity bypass；
- TP=1 monkey patch；
- dynamic shape/stride workaround。

这证明单卡数据流能前进，不证明完整模型语义、TP=8 或 MoE runtime 已就绪。
后续真实 head gate 恢复后 NaN 才流出，说明“为了 bring-up 屏蔽输出”会推迟
问题暴露。

#### 2026-06-19：跨卡 IPC capability 是独立基础设施 blocker

**[直接证实于历史部署记录]** 旧 driver/firmware 下
`support_shmem_map_exbus=0`，`aclrtIpcMemImportByKey` 返回 507899。升级到：

```text
driver   25.5.2
firmware 7.8.0.7.220
CANN     9.0.0-beta.1
```

后，跨卡 same-VA IPC 和 simpler L3 allreduce 才能通过。CANN GA 又会因
AICPU 扩展库未下发，在 BootstrapDispatcher 阶段表现为 507018。

经验不是“507018 都是 CANN”，而是：

```text
如果在 Bootstrap/comm init 前失败，先查环境；
只有进入 rt.run 并拿到 TASK/CLUSTER，才讨论 kernel stall。
```

#### 2026-06-22：重启后两个同名故障，只有一个是 kernel

**[直接证实于历史记录]**

- 第一次 frontend smoke 失败来自 stale `__pycache__/config...pyc`；
- 清理/刷新 Python source mtime 后 smoke 恢复；
- 同时 MoE device runtime 仍在 5 秒内 507018。

这一天已经暴露出后来反复出现的方法论问题：

> 先证明正在运行的源码和环境，再解释 device 行为。

#### 2026-06-24：第一个 MoE runtime 507018 来自 empty-tail 提交

**[历史记录，未独立复核]** 8 卡 `DecodeLayerMoE` 在 runtime 继续 507018。
dispatch-cut 显示 dispatch-only 能过、dispatch+routed 失败；最终定位到 routed
expert 对 `tile_valid <= 0` 的空尾 tile 仍提交 kernel。补：

```text
if tile_valid > 0:
    submit expert tile
```

后 8 卡 runtime 通过。

这与 07-06 的 `gate_topk` mrgsort hang、07-10 的跨层 alias 和 07-16 的随机
signal-layout stall 都不是同一个问题。它提供的长期经验是：

- signed tail 和 empty tile 必须在 cast/index/submit 前处理；
- `507018` 可以来自确定性的局部 shape 逻辑；
- 一次 MoE runtime pass 仍不包含 golden 精度，也不代表整网稳定。

### 4.3 2026-07-04～07-05：live co-tenancy 的 507018 不是后来 standalone P42 的根因

#### routed expert 单独运行先通过

**[历史记录，未独立复核]** routed-expert per-rank kernel 用真实 W8A8
权重完成 device 精度验证。这里复用了 06-24 已补齐的 empty-tail guard；
本阶段主要证明单独 routed 计算和真实权重数学可以正确。

这说明一部分 507018 可以来自非常局部的 shape/tail 逻辑，但不能据此推导
整网后来也挂在同一位置。

#### 与 active vLLM 同卡后出现另一类 507018

live 单层 MoE 的独立 ChipWorker 与 vLLM Worker_TP 共卡时：

```text
默认配置 -> routed 507018
ring heap=4GB -> 16GB static arena OOM 207001
降低 vLLM gpu memory -> OOM 消失，但 507018 仍在部分 rank 出现
```

**[历史判断]** 当时把它归为独立 runtime/device-context co-tenancy blocker。
该对象是 live 双 runtime 共卡，不是之后的 standalone single program P42。

保留的经验：

- 207001 与 507018 要分开；
- 调内存能消掉 OOM，不代表消掉 kernel/runtime hang；
- 不允许把 live co-tenancy 的结论直接套到 standalone。

### 4.4 2026-07-06～07-07：第一次用 TASK→kernel 真正抓到 `gate_topk`

EpTpMoE 单块 8 卡真实 W8A8 出现：

```text
sched_error_code=100
task state=RUNNING
kernels=[aiv0:2]
```

使用该次 build 的 `kernel_config.py` 映射：

```text
func_id 2 -> gate_topk
```

**[直接证实]** 根因是 sort pipeline：

```text
sort32
-> mrgsort(block_len=64)
-> format2 merge(left_half, right_half)
```

format2 要求左右输入各自已经是完整有序序列，但当时每个半块仍包含多个
64-run。输入不满足状态机前置条件，分散分数下 kernel 不终止。

修为 DeepSeek 风格的 format1 渐进链后，gate 运行和 top-k 数值都通过。

这个修复后来极其关键：N1 whole-net 的 inlined generator 又保留或重新引入了
同类错误，2026-07-12 的 `task_id=3` 最终再次映射到 `gate_topk`。这不是
“历史 bug 不可能再出现”，而是说明：

```text
standalone validated source
!= generator 内联副本
```

### 4.5 2026-07-08～07-09：compile、prepare、dispatch 和底座漂移交织

这两天主要是 whole-model 组织方式探索，虽然不是最终随机 stall 根因，但解释了
后续为什么必须固定架构和 manifest。

#### multi-program co-prepare 与 `TaskMapSize=0`

历史多程序路径出现：

- ring init 太小，prepare 失败；
- 设到 `2^20` 又引发约 64GiB arena OOM；
- 65536/131072 一类配置能让 prepare 前进；
- prepare clean 后，首个 dispatch 仍可能 `TaskMapSize=0`，task 根本没有到
  device AICPU。

这个签名与 S1 RUNNING kernel 完全不同：

```text
TaskMapSize=0 / AICPU idle
```

说明 host→AICPU dispatch 没建立；不能去调某个 AICore kernel。

#### fused attention+MoE 的 compile/device 混淆

一度 attention 重写 compile clean，但 device 验证仍被同一个
`TaskMapSize=0` 路径挡住。后来 separate program/Option-C 路径能运行，
说明“编译成功”只清除了前端 blocker。

#### 07-09 升级栈带来新的漂移

全栈升级引入：

- ptoas/bin 版本差异；
- SplitIncoreOrch 新检查；
- Phase-16 的 SDMA-OFF patch 丢失；
- 一些历史精度 workaround 与 native W8A8 分支分叉。

**[直接经验]** 从这一阶段开始，任何“same code”都必须同时写五仓/工具链/
runtime binary，而不能只写 pypto-lib 分支。

### 4.6 2026-07-10：N1 编译成功后，连续解开环境和 deterministic alias 两个 blocker

#### 45 层单 program compile milestone

`WholeDecodeNetwork`、`WholeDecodeFaithful` 等全 45 层结构先后 compile clean。
这证明前端、链接和大规模 host orchestration 可生成，不证明 device 已运行。

#### 507899 曾被误判为 0234 node poison

首次 canonical TP=8 device run 在 comm-domain allocation 报 507899。由于连
known-good allreduce 也失败，当时一度判断节点 IPC poison，并尝试 reboot。

**[后续证伪]** reboot 无效。真正是两个独立问题：

1. 单卡 hello 507018：stale/mismatched runtime `.so`；
2. 多卡 507899：升级栈把 `SIMPLER_ENABLE_PTO_SDMA_WORKSPACE` force-on。

clean rebuild runtime，并关闭 SDMA workspace 后：

```text
single-card hello PASS
multi-card allreduce PASS
whole-net comm domain allocation PASS
```

这一步证明了为什么“重启后仍失败”不能自动叫硬件坏。

#### dispatch clean 后出现 mid-run scheduler timeout

device 真正执行后，P1 clean、P2 stall。dispatch-cut 和 exact generated
`host_orch.py` 显示：

```text
layer 3 chip_orch
layer 4 chip_orch
```

复用了同一个 `combine_done_buf`、recv/pub/routed/signal SSA。N1 单 program
把多个 layer 放在同一 orchestration 中，而依赖模型是 RAW-only v1；远端访问
尚未结束时复用同一个 communication window，违反 non-aliasing。

**[直接证实]**

```text
shared window K=2 -> deterministic STALL
per-layer distinct K=2 -> RUN_CLEAN
per-layer distinct K=42 -> RUN_CLEAN
```

因此：

```text
07-10 comm-window alias
```

是已解决的 deterministic architecture bug，但它不等于 07-16 的 probabilistic
signal-layout stall。

#### 同期确认三类 layer index 不能混用

模型里至少有三种索引空间：

```text
norm absolute layer index
attention type-local index
dense/MoE local weight index
```

把它们压成一个 `layer_idx` 会导致合法 shape 下读错层。后来的 dense L2 精度
问题正是这一类边界错误的具体实例。

### 4.7 2026-07-11：真实权重 IPC bring-up 逐层暴露 host、runtime 和 device blocker

#### self-load 先在 host 侧 OOM

把 8 rank 全部权重 stack 到一个 driver 进程，host 瞬时占用约 752GiB，进程
exit 137。这个失败发生在 device kernel 之前。

架构因此改为：

```text
8 per-rank exporter
-> each holds one rank device pool
-> import_ipc_all
-> StackedDeviceTensor
-> N1 single-program rt.run
```

#### IPC import 成功后，先修 runtime contract

依次出现：

- host tensor 未在 prepare 前 `.share_memory_()`；
- `StackedDeviceTensor[r,k]` 不支持 trailing contiguous sub-view。

增强 `StackedDeviceTensor` 后，权重可以按层产生正确 device sub-view。这个
runtime 支持后来必须提交到 pypto clean pin；只拉 pypto-lib 无法复现。

#### 进入 device 后先撞 arena OOM

真实 BF16 pool 约 47GiB，再加 4×4GiB ring static arena 和固定组件，超过
64GiB 卡容量，出现 207001。降低 ring heap 到可 fit 后才进入真正的 S1：

```text
completed=4/32
task=0x100000003
running=1
waiting=3
```

#### IPC/VA/a2a 的中间归因

同一 program、同 heap：

```text
dummy H2D weights -> clean
real IPC weights -> stall
P=0 -> clean
P>=1 -> stall
```

于是历史上一度把根因写成：

```text
large child-memory IPC pool / VA interaction
或首个 ep_all_to_all
```

**[后续证伪]** 当时没有 exact TASK→func 映射，只是按 MoE 内部顺序猜 task3。
第二天 exact device log 证明 task3 实际是 `gate_topk`。

### 4.8 2026-07-12：exact TASK 映射推翻 IPC/VA 假设

隔离日志使用：

```text
ASCEND_GLOBAL_LOG_LEVEL=1
ASCEND_PROCESS_LOG_PATH=<fresh isolated dir>
```

同轮 device 快照：

```text
TASK task_id=3 state=RUNNING
kernels=[aiv0:3]
core=28
fanin=3/3
```

使用该 exact build 的 `kernel_config.py`：

```text
func_id 3 -> gate_topk
```

**[直接证实]** N1 inlined gate 使用：

```text
SCORE_PAD=512
sort32
-> 16 x 64-run
-> mrgsort(block_len=64)
-> 4 x 256-run
-> format2 two-way merge of two half-blocks
```

最后两个半块各自仍含两个有序段，不满足 format2 输入契约。修为完整的
format1 渐进 merge chain 后：

- P42 真 native W8A8 IPC 约 3.48s clean；
- sort-only probe 与 torch top-k 对齐；
- 权重 IPC + KV IPC 双路径进入 clean device run。

这一步同时推翻：

```text
IPC child-memory 是唯一根因
task3 一定是 ep_all_to_all
P0/P1 边界足以映射具体 kernel
```

### 4.9 2026-07-12～07-13：从“能跑”进入精度和 layer boundary 排障

#### 恢复真实 head gate 后 NaN 流出

旧 dummy `gate_r=0` 把 attention 输出乘成 0，掩盖了后续数值问题。恢复
on-device head gate 后：

```text
RUN_CLEAN
next_hidden=NaN
logits=NaN
argmax=0
```

说明 forward progress 与 numerical correctness 已分离。

#### P0 finite、P1 NaN 的第一次解释仍然错了

层数 bisect：

```text
P0 -> finite
P1 -> NaN
```

当时自然怀疑第一层 INT8 MoE。A-operand padding mask 实验仍 NaN，进一步把
嫌疑放到 routed expert、dispatch、combine 和 shared。

但随后发现真正的第一层边界 bug：

```text
attn_only_orch 的 resid3_out 局部 tensor
遮蔽了 pl.Out 参数
-> attention 结果没有写到 h_mid_out
-> chip_orch 读取未初始化 handoff
```

修复 Out writeback 后 NaN 消失，但仍出现 1e11～1e12 幅值。

#### 三个 ordering/FUSE 判断又被诊断工具污染

先后尝试：

- 合并两个 per-rank loop；
- 捕获 attn_only 返回值建立 data dependency；
- 直接把 Out 传给 inline attention；
- 把 attention fuse 进 MoE orchestration；
- orchestration 内 early return 做 stage bisect。

期间 generator `_be` 用 substring 搜索 `return next_hidden_out`，命中了错误的
调试分支，把 MoE body 截断在 norm-only。于是若干“某 stage 正常/异常”的数字
实际来自 truncated program。

同时，orchestration 内 Python early return 并不保证后续 InCore kernels 不进入
完整 DAG，导致“只跑 attention”“只跑 norm”的假隔离。

这批结论随后全部降级。可靠的诊断必须使用：

```text
独立 pl.Out dbg_out
host 可见的 op-level dump
exact generated source
```

#### 可靠 op-level dump 首次把错误钉到 routed expert

P1 的独立输出：

| stage | max abs | 结论 |
|---|---:|---|
| post_norm | 1.45 | 正常 |
| local_routed_x | 1.45 | dispatch 正常 |
| shared expert output | 1.62 | 正常 |
| local_routed_y | 3.99e11 | 首个异常 |
| moe_out | 1.95e11 | 继承 routed garbage |

**[直接证实]** 错误来自 whole-net inlined `_expert_routed`，不是 FUSE、
collective 或输入幅值。inlined 版本与 standalone validated `moe.py` 漂移：

```text
旧 in-expert INT8 quant/cast/cube
vs
validated dispatch-side quant + materialized INT8
```

修复方向因此明确为原生 W8A8 对齐，而不是回退 BF16-dequant。

### 4.10 2026-07-13：native W8A8 修复正确，但 P42 stall 的解释继续变化

对齐 validated routed expert 后：

```text
local_routed_y: 3.99e11 -> 1.41
P1/P20/P31 -> clean
P42 -> stall
```

一度根据 comm-window 字节：

```text
P20 ~186MB clean
P31 ~290MB clean
P42 ~391MB stall
```

以及人为膨胀 P20 窗口后 stall，判断存在约 290～390MB 字节上限，并以
dispatch-side INT8 缩小 `recv_x` 为“确定修法”。

**[后续证伪]** standalone allreduce 在更大窗口和不同权重共存条件下仍能通过，
说明不存在这样一个简单的固定 byte cap。缩小 footprint 是正确的 native W8A8
设计，也改变了布局和概率，但不能被写成 P42 stall 的唯一根因。

这类修改应该按两类记录：

- **保留**：因为 native W8A8、带宽、内存和已验证 `moe.py` 边界正确；
- **撤回根因措辞**：不能声称它独立证明了“comm pool byte limit”。

### 4.11 2026-07-14：首次得到 token 303，但发生两次过早闭环

#### dense L2 `attn_layer_idx` 是真实精度 bug

用 ctx=1 dense torch golden 和逐层 row0 cosine 对拍：

```text
L0/L1 -> cos=1.0
L2 -> cos=0.931
```

L2 权重已在 call site 预切到单层，却仍传 `attn_layer_idx=1`，内部再次按
layer offset 读取，越界到相邻层权重。修为 0 后：

```text
L2 cos -> 0.999999
```

**[直接证实]** 这是独立的精度根因，应保留。

随后在 logging 时序扰动下 P42 跑通一次：

```text
token 6127 -> argmax=303
```

这证明精度路径已经可以正确，但 confirm run 又 stall，所以稳定性仍未关闭。

#### completion-wave 与 combine race 一度被宣布为最终修复

历史阶段加入：

- `tp_all_reduce` completion wave；
- combine zero-done/publication barrier；

并记录：

```text
P42 7 次 RUN_CLEAN
argmax 303 3/3
```

随后 vector diff 又把残余数值抖动指向 combine/routed gather，并尝试
`_serialize_after_shared`；该串行化在 P42 又造成 507018。

#### 同日晚 clean-tree 复验推翻“A2 已修”

对包含上述修改的 clean tree、不打开 logging、同 P42 真输入复验：

```text
run1 STALL
run2 CLEAN argmax=303
run3 STALL
```

**[直接证实]** completion-wave 没有可靠关闭随机 stall。历史 7 次 clean 是概率
样本或环境/对象差异，不能作为 release。

这一晚形成最重要的流程规则：

> commit message、handoff 或 agent 报告里的“device verified”必须在当前 exact
> 对象上独立多次复验；概率问题不能由短连续 clean 宣布结束。

### 4.12 2026-07-15：从现象推论转向 exact kernel 位置

#### canonical 口径先被冻结

明确唯一精度/稳定性对象：

```text
whole_decode_faithful_real
P_FAITHFUL_MOE_LAYERS=42
token 6127
native W8A8 weights IPC
KV IPC
argmax must be 303
```

一个重要陷阱是 shell 中遗留：

```text
P_FAITHFUL_MOE_LAYERS=1
```

会产生 2.7s 左右的假 clean 和 P0/P1 诊断 argmax，而不是完整 P42。之后每轮
日志必须打印 P42。

#### push baseline 的 exact TASK 证据

push+push baseline 同时能观察到：

```text
clean -> argmax=303
stall -> 507018 / S1
```

同轮失败日志：

```text
TASK ...319 state=RUNNING fanin=6/6 kernels=[aiv0:28] core=26
TASK ...320 state=WAIT    kernels=[aiv0:29] missing_deps=1
completed=39/137
```

exact build 映射：

```text
func28 = _dispatch_push
func29 = _dispatch_stage
```

**[直接证实]** 输入依赖已经满足，`_dispatch_push` 已在 core 上 RUNNING 但不完成；
下游 `_dispatch_stage` 只是等待它。它不是 S4 fanin dependency deadlock。

#### 有 kernel 位置，不等于有子指令根因

当时先后怀疑：

- 跨 die bulk remote_store/TPUT 完成；
- count_done/data_done barrier；
- 单波 completion protocol；
- 跨 rank dispatch order。

实验包括：

- 增大 scheduler/op/stream timeout 10 倍，仍耗满约 450s 不完成；
- TPut 后补 fence；
- pre-push rendezvous；
- 把 remote write 改成 local write；
- count/data done 加 two-wave；
- AtomicAdd/Set/read-back 等变体。

这些实验会改变挂率或挂点，但没有给出稳定修复。特别是：

- local-write 后仍可挂，削弱“bulk TPUT 是唯一根因”；
- two-wave 后仍约三分之二 stall，推翻“单波就是最终原因”；
- 没有真实 AICore PC，不能声称卡在具体 `wait_flag` 或某条 ISA。

#### push→pull 重写既有价值，也产生过新 bug

先改 dispatch pull，后改 combine pull。过程中曾按 `completed=78/81`
猜 hang 已移到 combine；下一轮保留 exact build 后发现：

```text
func28 = _dispatch_pull
```

从而推翻按完成比例猜 kernel 的结论。

早期 full-pull 实现 P1 就能 stall，证明当时 pull 实现自身存在 rendezvous/
offset/handoff 问题，不能以“pull 原语在别处 clean”替代整网验证。修订后：

- dispatch pull + combine push 的 clean 概率提高，但仍有 residual stall；
- combine pull P1 clean，P42 仍随机；
- 最终所有发布组合仍必须直接跑 P42。

应保留的不是“pull 天生不会挂”，而是最终 fixed-slot pull 的清晰边界、
self local-load/peer remote-load 和 dispatch-produced inverse map。

#### 0162 二次复现

在 0162 devices 8..15 上，同一历史 push+push 对象 6 次：

```text
2 clean
4 stall
clean runs argmax=303
```

**[历史记录，未作为 final exact-manifest A/B]** 这排除了“症状必然只在
0234 某个坏芯片”的过早结论，但两台机器当时的完整 runtime dirty state 没有
严格绑定，所以不能升级为完全 matched 的跨机因果实验。

### 4.13 2026-07-16：kernel 位置漂移迫使排查回到跨 rank 边界和物理布局

在后续 pull+pull 失败 build 中，历史记录显示：

```text
rank 8-14 -> _pull_routed_y
rank 15   -> _dispatch_pull
```

这推翻了：

```text
所有 rank 都挂 routed_h_quant
所有 failure 都固定在一个 kernel
```

正确解释是：不同 rank 处于同一个跨 rank protocol 的不同深度。需要从最早
未完成的 publish/fence/notify/wait/load generation 反推，而不是选择出现次数
最多的 kernel。

#### 回到 buffer ledger 后发现 control signal 物理共线风险

逻辑 signal：

```text
[8,1] INT32 = 32B
```

此前物理 allocation 也只有 32B。多个 signal 以及 signal/payload 在一个巨大
comm window 中紧密排列。静态布局审计显示 control-plane hotspot 可能共享
512B line。

最终只改物理分配：

```text
32B -> 512B
COMM_CONTROL_SIGNAL_BYTES = 512
```

保持：

- logical shape/dtype；
- rank indexing；
- native W8A8 数学；
- P42；
- token 6127；
- dispatch pull + combine pull；
- AtomicAdd/Ge 协议。

生成物审计：

```text
216/216 signals nbytes=512
all relative offsets %512=0
window size 766525440 %512=0
```

**[强关联]** 应用该最小 layout 变量后，0162 fresh exporter pool 的随机 stall
收敛为连续 20 次 clean 且每次 `argmax=303`。

#### 最终 layer boundary 被固定

```text
dispatch:
  pack_publish
  -> dispatch_pull
  -> dispatch_stage
  -> recv_counts + inverse_map

combine:
  stage_routed_src
  -> pull_routed_y(dispatch-produced inverse_map)
  -> weighted_gather
```

self 使用 local load，peer 使用 remote load；每层 communication window distinct；
整窗 zero-init；routed expert 保持 native W8A8；tail 先 signed 判断再 cast。

这些修改中只有 512B physical isolation 是最终最小 layout A/B 变量；其他项因
架构、精度或生成器一致性保留，不能全部包装成 stall 唯一根因。

### 4.14 2026-07-17：release 证据审计又发现一次“对象不完全相同”

初始候选 20-run：

```text
signal512_p42_20_20260716_220004
```

完成 20/20 后，证据审计发现候选日志绑定的 model source SHA 与整理后的 release
smoke 不完全相同。于是没有直接把候选 20-run 追溯成 release 证明，而是对
release commit：

```text
pypto-lib 0e7a0fddc90c4f2348f1d59e015fb817a0877a02
```

重新执行 exact-model-source 20-run：

```text
signal512_p42_20_20260717_001135
20/20
each rc=0
each argmax=303
runtime 2.50/2.5605/2.62s
```

随后把当时 20-run 依赖的 pypto/simpler dirty runtime 支持 formalize 为 clean
commits：

```text
pypto   e277de9f
simpler 36957c6b
```

并执行 clean-pin P42 smoke：

```text
final_stack_smoke_20260717_015635
rc=0
2.58s
argmax=303
worker-window relevant dmesg=0
```

结论必须分开：

- exact-model-source 在当时实际 runtime 组合上有 20/20；
- 三仓 clean pins 有一次独立 P42 smoke；
- 不能把一次 clean-pin smoke 写成 clean-pin 20-run。

### 4.15 blocker 演化矩阵：哪些是同一外观下的不同问题

| 阶段 | 表面 | 决定性证据 | 真正处理 | 与最终随机 stall 的关系 |
|---|---|---|---|---|
| Phase 16 环境 | 507899/507018 | capability、Bootstrap 日志 | driver/firmware/CANN | 基础前置 |
| 07-10 runtime | 单卡 507018、多卡 507899 | hello/allreduce 对照 | clean `.so` + SDMA OFF | 基础前置 |
| 07-08 dispatch | 507018 timeout | `TaskMapSize=0`, AICPU idle | program/prepare/dispatch 组织 | 不同机制 |
| 07-10 alias | mid-run 507018 | K1 clean/K2 shared stall/per-layer clean | per-layer distinct windows | 已解决 deterministic bug |
| 07-11 OOM | exit137/207001 | host RSS、rtMalloc size | exporter IPC、arena sizing | 不属于 kernel stall |
| 07-12 gate | S1 RUNNING | TASK aiv0:3→`gate_topk` | mrgsort format1 chain | 已解决 deterministic kernel bug |
| 07-12～14 精度 | NaN/wrong argmax | Out dump、layer golden | Out/index/native W8A8 修复 | 与稳定性正交 |
| 07-14～16 随机 stall | clean 303 或 S1 | multi-run、exact TASK、跨-rank漂移 | 边界审计 + signal layout A/B | 最终 0162 release blocker |

### 4.16 被推翻的核心判断清单

| 历史判断 | 为什么当时看似合理 | 什么证据推翻 |
|---|---|---|
| 0234 节点 IPC poison | known-good allreduce 也失败 | reboot 无效；clean runtime + SDMA OFF 恢复 |
| real-weight IPC/VA 导致 task3 stall | H2D clean、IPC stall | exact task3→`gate_topk` |
| task3 是首个 ep_all_to_all | 按 MoE 源码顺序猜 | exact `kernel_config.py` 映射 |
| P0 finite/P1 NaN 证明 INT8 MoE 内核错 | 首个 MoE 层加入即坏 | `pl.Out` 被遮蔽，handoff 未写 |
| FUSE/跨-orch handoff 是 1e11 唯一根因 | 调试旋钮数字支持 | generator 截断 + reliable op-level Out dump |
| comm window 有固定 290～390MB cap | P31/P42 和膨胀实验相关 | 更大 standalone window clean，条件不匹配 |
| completion-wave 已修 A2 | 短期 7 clean、303 3/3 | clean tree STALL/CLEAN/STALL |
| task local id 23 是 func23/tp_all_reduce | 把 raw task 低位直接当 func | TASK `kernels=[aiv0:28]` |
| completed=78/81 表示 combine | 接近 orchestration 尾部 | exact build func28=`_dispatch_pull` |
| PUSH/TPUT 是最终唯一根因 | push 探针和 func28 线索 | pull+pull 也 stall，kernel 跨 rank 漂移，无 PC |
| P20 clean 可以代替 P42 | 迭代快、概率低 | 完整深度改变 layout/generation/stall 概率 |
| 一次 argmax303 表示 ready | 数学路径正确 | 同版本下一轮仍 stall |

## 5. 固定被测对象

### 5.1 为什么先冻结 canonical

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

### 5.2 “老基线 303”到底表示什么

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

## 6. 症状与第一性分类

### 6.1 历史跨机器症状线索（非 exact-object reproduction）

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

### 6.2 不能从 507018 直接定性

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

### 6.3 timeout 只用于区分 slow 与 hang

曾把 scheduler/op/stream timeout 抬高约 10 倍。失败运行仍耗完整 op 预算后不完成。

该实验支持：

```text
不是多等几分钟就能完成的普通慢 kernel
```

但它不提供修复，也不告诉我们卡在哪个子操作。

### 6.4 dmesg 使用 before/after

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

## 7. 从 task 到 exact kernel

### 7.1 第一个关键纠错：task id 不是 func id

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

### 7.2 第二个关键纠错：不能按完成比例猜 kernel

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

### 7.3 从 kernel_config 继续到生成源码

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

### 7.4 本案例的 PC 边界

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

## 8. 跨 rank 还原通信边界

### 8.1 不同 rank 停在不同深度

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

### 8.2 dispatch 边界审计

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

### 8.3 combine 边界审计

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

### 8.4 为什么要看“上一对操作”

若快照停在 wait 或 remote load，真正错误可能在前一对 producer 操作：

```text
producer 没写完 payload
producer fence 未覆盖对应 pipe
producer notify generation 错
consumer expected 继承旧值
consumer offset 来自另一份 count snapshot
```

因此 kernel 名只是入口，边界上的 producer/consumer 配对才是检查单元。

## 9. 历史方向纠正

### 9.1 push/TPUT 不是最终唯一根因结论

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

### 9.2 P1/P20 不能替代 P42

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

### 9.3 completion-wave、串行调度和增大 timeout

这些实验能排除或分类部分假设，但都没有成为最终最小修复：

- completion-wave：可修独立协议缺陷，但不能解释所有残余随机 stall；
- serial orchestrator gate：device 仍可 stall；
- 增大 timeout：只证明非普通慢；
- retry：只能掩盖概率问题。

### 9.4 exporter/compile 失败不是 kernel stall

曾出现 checkpoint 路径、stale exporter、COMMINIT 等问题。只有到达：

```text
built args
rt.run
device scheduler snapshot
```

后，才能归入本案例的 kernel stall。阶段必须在日志中明确标记。

## 10. 设计和代码审计

### 10.1 generator 与 active builder 不一致

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

### 10.2 per-layer distinct buffer

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

### 10.3 对齐规则不能混

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

### 10.4 dtype、tail、padding 和初始化

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

### 10.5 片上 memory report

最终 retained build 的报告显示示例：

```text
_dispatch_pull Vec 11.1KB / 184KB
_pull_routed_y Vec 8KB / 184KB
_stage_routed_src Vec 64KB / 184KB
```

这用于排除明显 UB 容量超限，并提供 buffer address/live range。
它不能替代 GM communication window 的 offset/cache-line 审计。

## 11. 最终最小 A/B

### 11.1 A 组风险

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

### 11.2 B 组只改物理隔离

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

### 11.3 为什么该 A/B 有意义

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

## 12. 完整准出验证

### 12.1 最终代码

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

### 12.2 release exact-source 20-run

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

### 12.3 整理后 smoke

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

### 12.4 最终三仓 clean-pin smoke

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

## 13. 保留项与因果边界

### 13.1 应保留的结构性修改

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

### 13.2 可以写和不能写

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

## 14. 设计阶段如何提前避免

### 14.1 在设计评审中提交 buffer ledger

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

### 14.2 把 control plane 和 data plane 分开

设计上要求：

- signal 不与大 payload 共 512B line；
- 不同协议的 atomic counter 不共 line；
- 每层 signal 独立；
- generation 明确，不依赖历史残值；
- whole-window init 和 logical init 均有责任方。

### 14.3 把 layer boundary 当作接口

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

### 14.4 设计时同时覆盖 batch/pad/dtype

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

### 14.5 让生成物成为评审对象

源码正确不等于 codegen 后正确。设计 gate 应包含：

- exact `kernel_config.py`；
- orchestration task 顺序；
- dependency dump；
- memory report；
- physical window offset；
- generator round-trip；
- final kernel source 中真实 fence/notify/load/store。

## 15. 可复用排查清单

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


## 16. 原始证据索引

以下索引按“先看当前事实、再看过程、最后看架构约束”的顺序排列。旧的
`NEXT-SESSION-N-1.md` 已经清理了 active prompt；需要恢复被删除的阶段记录时，
应从 Git history 按提交读取，而不是把历史快照当作当前状态。

### 16.1 当前 stable / canonical

- [`N1-CANONICAL-TEST.md`](../../../../N1-CANONICAL-TEST.md)：唯一 standalone canonical 命令、P42、
  fresh exporter、20-run gate、dmesg 窗口和三仓复现边界。
- [`N1-STABLE-ENV-0162-20260717.md`](../../../../develop/N1/N1-STABLE-ENV-0162-20260717.md)：0162 stable
  SSOT，源码/runtime pin、clean-pin smoke、实际 layout 和未覆盖范围。
- [`N1-NEXT-SESSION-HANDOFF-20260715.md`](../../../../develop/N1/N1-NEXT-SESSION-HANDOFF-20260715.md)：
  final pull+pull boundary、512B isolation、20/20 与因果措辞。
- [`N1-W8A8-ROUTED-BOUNDARY-DESIGN-20260716.md`](../../../../develop/N1/N1-W8A8-ROUTED-BOUNDARY-DESIGN-20260716.md)：
  attention→dispatch→expert→combine 的 native W8A8 layer contract。

### 16.2 主过程记录

- [`phases/27-n1-whole-net-fusion.md`](../../../../phases/27-n1-whole-net-fusion.md)：从全网 compile、
  runtime/SDMA、comm-window alias、IPC bring-up 到最终 release 的主 phase 记录。
- [`STATUS.md`](../../../../STATUS.md)：跨 phase 的状态变化、0162/0234 scope 和 live blocker。
- [`blockers.md`](../../../../blockers.md)：N1-S-0162、N1-S-0234、Phase 28 live blocker 及历史纠正。
- [`notes/08-integration-churn-postmortem.md`](../../../../notes/08-integration-churn-postmortem.md)：
  为什么“ready”会被更强验证推翻，以及如何建立证伪优先的流程。
- [`notes/09-cache-line-and-signal-isolation.md`](../../../../notes/09-cache-line-and-signal-isolation.md)：
  signal 物理隔离、512B line、base/offset 约束的设计背景。

### 16.3 早期 blocker 和部署背景

- [`archive/prototype-phase-01-19-summary.md`](../../../../archive/prototype-phase-01-19-summary.md)：Phase 15/16/19 的
  单卡、多卡 capability、MoE 早期 507018 和 stale-pyc 经验。
- [`archive/milestones-2026-Q2.md`](../../../../archive/milestones-2026-Q2.md)：按日期压缩的项目主时间线，
  包含 07-04～07-17 的关键里程碑。
- [`deployment/phase16-three-pillars.md`](../../../../deployment/phase16-three-pillars.md)：driver/firmware/CANN
  三件套和 IPC capability 约束。
- [`deployment/troubleshooting-multirank-507899.md`](../../../../deployment/troubleshooting-multirank-507899.md)：
  507899、507018、stale runtime 和 SDMA workspace 的区分。
- [`deployment/troubleshooting-moe-block-8card-gate-topk.md`](../../../../deployment/troubleshooting-moe-block-8card-gate-topk.md)：
  gate_topk 的 V0 TASK→kernel→mrgsort 修复链。
- [`deployment/moe-block-nextwork-and-constraints.md`](../../../../deployment/moe-block-nextwork-and-constraints.md)：
  native W8A8、DeepSeek 对齐、不得绕过 gate 和测试纪律。

### 16.4 Git 历史恢复点

以下提交是恢复 07-12～07-16 过程记录的主要锚点；读取时使用：

```bash
git show <commit>:NEXT-SESSION-N-1.md
git show <commit> -- NEXT-SESSION-N-1.md STATUS.md phases/27-n1-whole-net-fusion.md
```

| 提交 | 过程节点 | 本案例使用的证据 |
|---|---|---|
| `6f5256d` | M3 Out writeback 与初始幅值误诊 | 未初始化 handoff、NaN→finite 的边界 |
| `c28b7ac` | 跨 orchestration / FUSE 假设 | 三个 ordering 修法和错误方向 |
| `3f63429` | FUSE 复诊与 generator truncation | 诊断旋钮不可直接信任 |
| `7f7a144` | op-level dump | `local_routed_y` 首先爆炸，定位 native W8A8 漂移 |
| `d628b89` | A1/A2 中间判断 | pool-byte 假设及后续证伪入口 |
| `af794a6` | L2 index 精度 bug | dense golden L2 cos=0.931→0.999999 |
| `b20cda2` | 首次 token 303 | 一次正确不等于稳定 |
| `653bb0f` | completion-wave 过早闭环 | 3/3 与 7-run 结论的历史状态 |
| `b08f6ae` / `28a6ef3` | combine/vector diff | 残余数值抖动的历史定位 |
| `18ce42b` | clean-tree 复验 | STALL/CLEAN/STALL 推翻 A2 已修 |
| `4d2fc42` | push/pull exact TASK 线 | 10× timeout、pull wiring、kernel mapping 纠正 |
| `80a8ce3` | 0162 二次复现 | push+push 6 次 2 clean/4 stall |
| `2e46485` | 512B isolation 与 release | final pull+pull、20/20、因果边界 |
| `8bbb9cc` / `0ef0109` | stable manifest formalize | release manifest 和 clean environment scope |

### 16.5 顶层设计约束

设计或复查 buffer、通信和层间边界时，优先读取：

```text
/data/chensiyu/hw_project/pypto/pypto_top_level_documents/
  pypto-runtime-arch-docs/02-logical-view/04-memory.md
  pypto-runtime-arch-docs/02-logical-view/05-machine-level-registry.md
  pypto-runtime-arch-docs/02-logical-view/07-task-model.md
  pypto-runtime-arch-docs/02-logical-view/08-communication.md
  pypto-runtime-arch-docs/02-logical-view/11-machine-memory-model.md
  pypto-runtime-arch-docs/02-logical-view/12-dependency-model.md
  pypto-runtime-arch-docs/08-design-decisions.md
  tensor_layout.md
  tensor_valid_shape.md
  tpush_tpop_isa_design_v3.md
```

本案例文档总结的是项目实际排障证据；若与顶层约束发生冲突，应先停止局部
修补，回到顶层设计文档确认语义和适用范围。
