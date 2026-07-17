# 整网项目实现与发布准入：Gate 6～10

> 设计 Gate 通过后，再用本文约束 kernel 边界、片上预算、生成器、可观测性、
> 验证梯度和 release manifest。一次运行完成或一次得到正确 token 都不构成准出。

## 实现到发布路径

```mermaid
flowchart LR
    K[Kernel design card] --> G[Generator round-trip]
    G --> O[Observability bundle]
    O --> V[Level 0～9 validation]
    V --> R[Release manifest]
    R --> Q[完整深度 + 精度 + 重复稳定]
```

## 15.9 Gate 6：kernel 边界与片上 buffer 准入

**目的：** 先设计每个 kernel 做什么、用多少片上空间，再开始写 kernel body。

每个 kernel 需要一张设计卡：

```text
kernel name
所属 layer/phase
输入/输出 tensor
logical/physical shape
tile shape / valid shape
UB/L1/L0A/L0B/L0C 预算
循环轴和 pipeline stage
跨核 split 策略
load/compute/store 顺序
tail/padding 行为
是否含 wait/fence/notify
预期运行时间和拆分依据
```

设计原则：

- kernel 边界应与数据所有权和 lifetime 边界一致；
- 不为了“少一个 task”把互不相关的 TP/EP、control/data 或不同 lifetime 强行
  fuse；
- 也不把一个有明确局部复用的算子拆成大量微小 kernel；
- 片上 buffer 必须按实际 live range 预算，不能只把每个 tensor 大小单独相加；
- partial tile、空尾和多 batch 必须在 kernel 设计卡中有独立 case；
- 所有 wait/fence/notify 都属于通信协议 Gate，而不是随手写进计算 kernel。

**PASS 条件：**

- memory report 预算与设计卡一致；
- 每个片上 buffer 有明确创建/释放 scope；
- 不存在未定义的 padded row；
- split/fuse 决定有架构和 memory 依据；
- kernel probe 可以独立验证其输入契约。

**NO-GO：**

- 编译到 UB overflow 后才决定 tile；
- 为了规避一个错误随机拆 kernel；
- 把所有逻辑塞入一个超大 orchestration 后依靠 early return 调试；
- kernel 中的 signal wait 没有对应 producer 设计；
- 只验证 full tile，不验证 partial/empty tail。
## 15.10 Gate 7：生成器和生成物准入

**目的：** 保证“审阅的源码”就是“设备实际运行的源码”。

Agent 必须指定唯一 source of truth：

```text
generator
-> active builder
-> generated host orchestration
-> kernel_config.py
-> generated kernel source
-> compiled binary
```

提交前必须执行真实 round-trip：

```text
剥离/删除 active generated block
-> 运行 generator
-> 与提交前 active block 做 byte compare
-> 结果必须一致
```

还必须审阅：

- `host_orch.py` 中实际 layer 顺序、buffer SSA 和参数；
- `kernel_config.py` 中 func id；
- orchestration C++ 中 task 顺序；
- dependency dump 中 fanin/fanout；
- memory report 中地址、offset 和 live range；
- final kernel 中真实 load/store/fence/notify/wait。

**PASS 条件：**

- generator round-trip 完全一致；
- 没有 standalone 和 whole-net 两份长期漂移的实现；
- 所有 debug knob 默认关闭且不会改变生成边界；
- build directory、generator hash 和 source hash 被记录。

**NO-GO：**

- 只验证 generator “拒绝覆盖已有代码”；
- 手改生成文件但不改 generator；
- 用 substring 搜索脆弱地截取生成 body；
- build 结束后删除唯一失败生成物；
- 按源码函数顺序猜 func id。
## 15.11 Gate 8：可观测性准入

**目的：** 第一次真机运行前就具备定位能力，而不是卡死后才补日志。

每个 run 必须自动产生独立 evidence bundle：

```text
run id / timestamp
source commits / dirty diff / source hash
generator hash / build directory / binary hash
machine / device / runtime / toolchain
canonical or diagnostic object
stdout / stderr / return code / runtime
all-rank device logs
TASK / CLUSTER / dependency snapshot
dmesg before / after / delta
numeric fingerprints / argmax / intermediate debug outputs
actual communication window base and relevant offsets
exporter process ids and lifecycle
```

在第一个 device run 前预留：

- 不改变主数学路径的 phase marker；
- 可选独立 `pl.Out` debug output；
- task 到 kernel 的 exact mapping 工具；
- all-rank 日志收集，而不是只看 rank0；
- worker 执行窗口和 exporter teardown 窗口分离；
- 自动打印最终执行层数和通信组合，防止环境变量残留。

**PASS 条件：**

- 任意失败都能回答发生在 export/compile/prepare/import/dispatch/run/teardown；
- 任意 S1 都能绑定同轮 TASK 和 exact build；
- dmesg 只归因当前 before/after 增量；
- 日志中明确写出完整或缩减测试对象。

**NO-GO：**

- 共享全局日志目录；
- 只保存 host 的 507018；
- 失败 build 被下一轮覆盖；
- 只看完成比例猜 kernel；
- 用旧 dmesg 或旧 device 日志补当前证据。
## 15.12 Gate 9：验证梯度准入

**目的：** 每一级测试只回答它能回答的问题，不能相互替代。

| Level | 测试对象 | 主要回答 | 不能证明 |
|---:|---|---|---|
| 0 | 静态设计审计 | shape、dtype、index、buffer、protocol 是否定义完整 | device 可运行 |
| 1 | compile / generator round-trip | 前端、生成器和代码生成可成立 | task 已派发 |
| 2 | prepare / dispatch smoke | runtime 资源和 task mapping 建立 | kernel 完成、数值正确 |
| 3 | 单算子真实 device probe | kernel 输入合同和局部数值 | layer handoff、跨层 lifetime |
| 4 | 单层真实权重 | 一层数学和通信基本成立 | 第二层复用安全、完整深度稳定 |
| 5 | 两个连续层 | layer handoff、generation、跨层 alias | 完整深度、概率稳定 |
| 6 | 中间深度 | 深度趋势、内存预算、诊断候选 | 正式 release |
| 7 | 完整深度单次 | 正式对象有机会完成并得到 golden | 概率问题已关闭 |
| 8 | 完整深度重复正式测试 | 稳定性和精度同时满足 | 另一机器/另一 manifest |
| 9 | 最终 clean manifest 重复测试 | 发布对象可复现 | 未覆盖的 live/其他 batch 场景 |

新的整网项目至少要求：

```text
单算子 golden
-> 单层真实权重
-> 两个连续层
-> 完整深度单次且数值正确
-> 最终 clean manifest 上完整深度重复测试
```

重复次数由 canonical 文档定义；对于本类概率性卡死，工程上默认建议连续 20 次
作为发布稳定性门槛，但这不是框架常量。项目可以根据失败概率和置信要求定义
更严格的次数，不能由 Agent 为了尽快通过而临时降低。

每次升级到下一 Level 前必须记录：

```text
本级对象
实际结果
能证明什么
不能证明什么
下一阶段新增了哪些变量
```

**NO-GO：**

- 用单层或中间深度替代完整深度；
- 一次正确 token 就宣布稳定；
- 20 次使用的 source 与 release source 不同；
- exact model source 重复测试与 clean runtime smoke 混写；
- logging 开启和关闭的不同对象混成同一组样本。
## 15.13 Gate 10：release manifest 准入

**目的：** 确保另一台机器拉取的不只是模型仓库，而是完整运行对象。

release manifest 至少冻结：

```text
所有 Git 仓库 commit / branch / clean status / submodule
模型源码与 generator SHA256
runtime binary 路径与 SHA256
driver / firmware / CANN
PTOAS / ptoas binary / pto-isa
Python / package environment
checkpoint 路径与 hash
machine / devices / topology
全部环境变量
canonical command
fresh exporter command and lifecycle
正式日志路径
dmesg worker-window 与 teardown-window 结果
golden 与数值指纹
未覆盖范围
```

未来项目的发布门槛应为：

```text
all repos clean and pinned
AND
runtime binaries hashed
AND
generator round-trip pass
AND
complete canonical repeated pass
AND
numerical golden pass
AND
worker-window dmesg clean
```

如果 clean commit 是在历史 dirty source 通过后才 formalize，必须在 clean
manifest 上重新执行完整重复测试，不能只做一次 smoke 就追溯成旧 20-run。
## 15.14 Agent 在每个阶段必须输出什么

Agent 不应只回复“正在定位”或“已经修复”。每个阶段的更新必须包含：

```text
1. 当前 exact object
2. 当前通过的 Gate
3. 当前 blocker 属于哪个阶段
4. 直接观测
5. 为什么提出当前假设
6. 下一决定性实验
7. 本轮唯一变量
8. 实际结果
9. 结论证据等级
10. 保留/撤回/仅诊断的修改
11. canonical 是否已恢复
12. 文档和 commit 是否已同步
```

编码前的实现计划还必须列出：

- 将修改的文件和 generator；
- 每个 layer/kernel 的边界；
- 新增/复用的 buffer 及 ledger 行；
- 地址对齐和内存预算；
- dtype、padding、tail、single/multi-batch 行为；
- 测试梯度和停止条件；
- 回滚不等于修复，禁止把旧概率版本当基线。
## 15.15 Agent 必须停止并重新设计的 NO-GO 条件

遇到以下任一情况，Agent 应停止继续堆局部 patch，回到设计 Gate：

- 顶层设计文档与当前方案冲突；
- layer boundary 无法说明唯一 producer/consumer；
- buffer lifetime 或 actual address 无法证明；
- 一个 `layer_idx` 承担多个索引空间；
- signal generation、initial value 或 writer 数量不明确；
- native W8A8 只能靠 BF16 fallback 才能通过；
- 单层 source 与 whole-net 生成副本不同；
- generator 不能 round-trip；
- 只能通过 retry、timeout、logging 或缩层提高通过率；
- 一次修改同时改变协议、数学、layout 和测试输入，无法形成 A/B；
- 当前日志不能绑定 exact build；
- 完整正式对象没有可复现命令；
- 多仓/runtime dirty 状态未记录；
- 文档中的“已修复”已被新结果推翻但未撤回。
