# N=1 案例：整网 Hang 排障实践导航

> 本文只保留执行路径和分流入口。定位 task/kernel/通信边界时读“stall 定位”；
> 审计 buffer、做 A/B、验证精度与发布时读“设计审计与准出”。

## 排障链条总图

```mermaid
flowchart TB
    A[固定 canonical object] --> B[隔离 run 目录与 dmesg before/after]
    B --> C{失败发生在哪个阶段?}
    C -->|export / compile / prepare| D[环境、版本、生成物、内存预算]
    C -->|rt.run| E[orch/sched code 与 S1/S3/S4/S5]
    E --> F[TASK + CLUSTER + dependency snapshot]
    F --> G[同轮 kernel_config.py / func_id / source]
    G --> H{是否有真实 PC?}
    H -->|否| I[停在 kernel/阶段级结论]
    H -->|是| J[PC → 同 image ISA map]
    I --> K[all-rank 最早阻塞边界]
    J --> K
    K --> L[publish → fence → notify → wait → load]
    L --> M[buffer / alignment / dtype / init]
    M --> N[单变量 A/B]
    N --> O[完整深度精度 + 重复稳定性]
```

## 按需阅读

| 当前问题 | 文档 |
|---|---|
| 失败阶段、507018 分类、TASK/CLUSTER、task→kernel、PC、跨 rank 边界 | [stall 定位入口](n1-stall-localization.md) |
| generator、buffer、对齐、dtype 和最小 A/B | [设计审计与最小 A/B](n1-design-audit.md) |
| exact-source 20-run、clean-pin、因果边界和准出清单 | [发布验证](n1-release-validation.md) |
| 需要知道这些步骤在本案例中如何逐日演化 | [时间线导航](n1-timeline.md) |

## 通信边界状态图

```mermaid
stateDiagram-v2
    [*] --> Compute
    Compute --> PayloadWritten: producer writes
    PayloadWritten --> Visible: fence / publish
    Visible --> Notified: generation update
    Notified --> Waiting: wait expected generation
    Waiting --> Loaded: local or peer load
    Loaded --> Consumed: consumer computes
    Consumed --> Recyclable: last reader + scope released
    Recyclable --> [*]
    Waiting --> Stall: no producer / stale generation / alias
    PayloadWritten --> Stall: notify before visibility
```
