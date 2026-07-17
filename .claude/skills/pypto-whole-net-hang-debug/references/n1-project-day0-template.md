# 整网项目 Day 0 模板与正式编码条件

> 复制本模板建立新的 `<PROJECT>-DESIGN-AND-ADMISSION.md`。模板只记录项目已经
> 批准的架构，不允许 Agent 借模板重新提出被排除的 program 形态。

## Day 0 交付物

| 文档 | Day 0 是否创建 | 内容 |
|---|---|---|
| DESIGN-AND-ADMISSION | 是 | topology、layer contract、buffer、protocol、Gate 状态 |
| CANONICAL-TEST | 是 | 输入、golden、命令、重复次数、dmesg 判据 |
| STABLE-ENV | release qualification 前 | 多仓、binary、工具链、checkpoint 和设备 |

## 15.16 新项目 Day 0（项目启动日）可直接复制的设计模板

后续 Agent 可以把下面模板放入 `<PROJECT>-DESIGN-AND-ADMISSION.md`：

```text
# <Project> Whole-Net Design and Admission

## A. Canonical object
program:
model topology:
total layers:
parallel topology:
machine/devices:
input/context:
batch/valid rows/padding:
weights dtype/source:
KV source:
dispatch/combine:
numerical golden:
stability gate:
not covered:

## B. Top-level constraints reviewed
runtime architecture docs:
memory docs:
task/dependency docs:
communication docs:
shape/layout/sharding docs:
applicable ADRs:
known deviations:

## C. Layer and index table
absolute layer | type | attention index | dense index | MoE pos |
norm index | KV offset | weight keys | program/method | buffer suffix

## D. Layer contracts
layer/phase | input | output | logical shape | valid shape | dtype |
scale | producer | consumer | self/peer | initialization | last use

## E. Buffer ledger
layer/phase | buffer | logical | physical bytes | alignment source |
relative offset | actual base check | owner | first write | last read |
init | protocol/generation | alias/reuse

## F. Communication state machines
collective:
producer:
payload:
fence:
notify:
wait:
generation:
self path:
peer path:
completion/recycle:

## G. Kernel design cards
kernel | role | input/output | tile | on-chip budget | loop/pipeline |
tail/pad | synchronization | probe golden

## H. Numerical contract
stage | input dtype | accumulation | output dtype | scale/bias |
quant boundary | golden | tolerance

## I. Observability
isolated run dir:
all-rank logs:
TASK/kernel mapper:
dmesg before/after:
phase markers:
debug Out:
actual address dump:
numeric fingerprints:

## J. Validation ladder
Level 0 static:
Level 1 compile/round-trip:
Level 2 prepare/dispatch:
Level 3 op probes:
Level 4 one layer:
Level 5 two layers:
Level 6 intermediate:
Level 7 full depth once:
Level 8 full depth repeated:
Level 9 clean manifest repeated:

## K. Repository and release manifest
repo commits/dirty:
submodules:
runtime binary hashes:
toolchain:
checkpoint hashes:
environment:
canonical command:
evidence paths:

## L. Gate status
Gate 0 project definition:
Gate 1 architecture/index:
Gate 2 layer contract:
Gate 3 buffer/memory:
Gate 4 communication:
Gate 5 numerical/dtype:
Gate 6 kernel/on-chip:
Gate 7 generator:
Gate 8 observability:
Gate 9 validation:
Gate 10 release manifest:

## M. Decisions and refutations
date | object | hypothesis | decisive experiment | result |
evidence level | keep/revert/diagnostic | scope
```
## 15.17 从本案例反推：这些 Gate 能提前避免什么

| 本案例暴露的问题 | 如果新项目按本标准启动，最早在哪个 Gate 暴露 |
|---|---|
| stale `pyc`、runtime `.so`、SDMA 配置漂移 | Gate 0/8/10：环境与 binary manifest、独立日志 |
| `TaskMapSize=0` 被当成 AICore kernel 错误 | Gate 8/9：先标失败阶段，再进入 kernel 定位 |
| 两个 MoE 层复用 communication window | Gate 2/3/4：layer contract、lifetime ledger、两层协议测试 |
| `gate_topk` standalone 已修、whole-net 内联副本仍旧 | Gate 5/7：source-of-truth 和 generator round-trip |
| task3 被猜成 all-to-all | Gate 8：同轮 TASK + exact `kernel_config.py` |
| `pl.Out` 被局部 tensor 遮蔽 | Gate 2：唯一 producer、真实 writeback 和 debug Out |
| `local_routed_y` 首先出现 1e11 | Gate 5：逐算子 numerical contract 和独立输出 |
| dense L2 使用错误 `attn_layer_idx` | Gate 1：完整 layer/index namespace 表 |
| 只执行 20/31 层通过，被误作完整深度结论 | Gate 0/9：canonical object 和验证梯度 |
| completion-wave 短样本通过后过早宣布修复 | Gate 9：matched、无额外 logging、完整深度重复测试 |
| 32B signal 与相邻对象共线风险 | Gate 3/4：logical/physical 分离、平台 cache-line descriptor |
| 只拉 pypto-lib 无法在另一机器复现 | Gate 10：多仓 commit、runtime binary、工具链和 checkpoint manifest |
| 全局 dmesg 旧错误被归因当前 run | Gate 8：before/after 和 worker/teardown 窗口分离 |
## 15.18 最终启动判定：什么时候 Agent 才可以进入正式编码

新整网项目只有满足以下条件，才可以从设计阶段进入正式实现：

```text
Gate 0 项目定义 PASS
AND
Gate 1 整网架构和索引表 PASS
AND
目标 slice 的 Gate 2 layer contract PASS
AND
目标 slice 的 Gate 3 buffer ledger / memory budget PASS
AND
目标 collective 的 Gate 4 communication state machine PASS
AND
Gate 5 numerical/dtype contract READY
AND
目标 kernel 的 Gate 6 design card PASS
AND
Gate 7 source-of-truth / generator round-trip plan READY
AND
Gate 8 observability plan READY
AND
canonical test 文档已创建
```

开始编码后，也不是所有代码一次性铺开。推荐顺序：

```text
实现并验证单算子
-> 实现单层
-> 验证两个连续层和跨层 lifetime
-> 扩展 generator
-> 生成物 round-trip
-> 扩展到完整深度
-> 完整对象单次精度
-> 完整对象重复稳定性
-> clean manifest 重复准出
```

一句话总结：

> **整网开发不是“把单层 kernel 拼起来再调”，而是先冻结整网合同、层边界、
> 内存生命周期、通信状态机、数值语义和准出对象，再让每一个 kernel 成为这些
> 合同的局部实现。**
